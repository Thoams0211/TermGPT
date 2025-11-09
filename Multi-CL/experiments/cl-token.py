import argparse
import logging
from dataclasses import dataclass, field
import os
import matplotlib.pyplot as plt
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
from tqdm import tqdm
import json
import random

import torch
from torch import nn
import torch.utils
from torch.utils.data import Dataset, DataLoader, SequentialSampler, DistributedSampler
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, NLLLoss
from torch.optim import AdamW
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger

import transformers
from transformers import (
    AutoConfig,
    PreTrainedModel,
    PretrainedConfig,
    AutoModelForCausalLM,
    MODEL_FOR_MASKED_LM_MAPPING,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    LlamaConfig,
    MistralConfig,
    GemmaConfig,
    Qwen2Config,
    set_seed,
    AutoTokenizer,
)
from transformers.trainer_utils import seed_worker
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.utils import send_example_telemetry
from transformers.utils.versions import require_version

from peft import LoraConfig, get_peft_model, PeftModel

current_dir = os.path.dirname(__file__)
subfolder_path = os.path.join(current_dir, '..')
sys.path.append(subfolder_path)

from model import LLM2Vec
from model.loss.utils import load_loss
from model.experiment_utils import generate_experiment_id
from model.dataset import E5Data, Wiki1M
from model.dataset.dataset import DataSample



class Dataset_MixCL(Dataset):
    def __init__(self, context: str, response: list[list], tokenizer, sign=None, context_len=256, response_len=128, neg_num=None, **kwargs):
        super(Dataset, self).__init__()

        # mixCL Project -- class CLData
        self.context = context
        self.data_size = len(context)
        self.tokenizer = tokenizer
        self.context_len = context_len
        self.response_len = response_len
        self.sign = sign
        self.response = response

        # mixCL Project -- class BatchData
        self.neg_num = neg_num

    def ent_mix(self, text1, ent1, ent2):
        # Replace all ent1 with ent2 in text1
        replaced_text = text1.replace(ent1, ent2)
        
        # Split replaced_text by ent2
        segs = replaced_text.split(ent2)
        
        # Initialize new_segs and signs
        new_segs = []
        signs = []
        
        # Iterate over the segments
        for i, seg in enumerate(segs):
            if seg:  # If the segment is not empty
                new_segs.append(seg)
                signs.append(1)  # The part from text1 is marked as 1
            
            # If it is not the last segment, add ent2
            if i != len(segs) - 1:
                new_segs.append(ent2)
                signs.append(-1)  # ent2 is marked as -1
        
        return new_segs, signs


    def __getitem__(self, idx):

        r"""
        
        In this function, we override Dataset class to construct MixCL dataset
        
        """

        batch_question = []
        batch_neg_response = []
        batch_neg_sign = []
        batch_pos_response = []
        batch_pos_sign = []

        # our context/response only contains one type of elements, so we can index them directly
        question = self.context[idx]
        response = self.response[idx]

        pos_text = response[0][0]
        pos_sub = response[0][1]

        positive_segs, positive_signs = self.ent_mix(pos_text, pos_sub, pos_sub)
        batch_pos_response.append(positive_segs)
        batch_pos_sign.append(positive_signs)

        for n in range(1, len(response)):
            neg_sub = response[n][1]
            negative_segs, negative_signs = self.ent_mix(pos_text, pos_sub, neg_sub)
            batch_neg_response.append(negative_segs)
            batch_neg_sign.append(negative_signs)

        batch_question.append(question)

        return batch_question, batch_pos_response, batch_pos_sign, batch_neg_response, batch_neg_sign
    
    def __len__(self):
        return self.data_size
    
    @staticmethod
    def collate_fn(data):

        batch_question, batch_pos_response, batch_pos_sign, batch_neg_response, batch_neg_sign = zip(*data)

        return {
            'question_list': batch_question,
            'positive_segs_list': batch_pos_response,
            'positive_signs_list': batch_pos_sign,
            'negative_segs_list': batch_neg_response,
            'negative_signs_list': batch_neg_sign
        }


class LLM2VecMixcl(LLM2Vec):

    def __init__(self, *args, **kwargs):
        super(LLM2VecMixcl, self).__init__(*args, **kwargs)

        # TODO: Qwen
        self.the_classifier = nn.Linear(3584, self.model.config.vocab_size).to(
            device=self.model.device,
        )

        # # TODO: Llama
        # self.the_classifier = nn.Linear(4096, self.model.config.vocab_size).to(
        #     device=self.model.device,
        # )

    
    @classmethod
    def from_pretrained(cls, base_model_name_or_path, peft_model_name_or_path=None, merge_peft=False, enable_bidirectional=True, **kwargs):
        cls.base_model_path = base_model_name_or_path
        # pop out encoder args
        keys = ["pooling_mode", "max_length", "doc_max_length", "skip_instruction"]
        encoder_args = {
            key: kwargs.pop(key, None) for key in keys if kwargs.get(key) is not None
        }

        tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        _model = AutoModelForCausalLM.from_pretrained(base_model_name_or_path)
        _model_config = _model.config.to_dict()
        config = AutoConfig.for_model(**_model_config)
        config_class_name = config.__class__.__name__

        model_class = cls._get_model_class(
            config_class_name, enable_bidirectional=enable_bidirectional
        )

        model = model_class.from_pretrained(base_model_name_or_path, **kwargs)

        if os.path.isdir(base_model_name_or_path) and os.path.exists(
            f"{base_model_name_or_path}/config.json"
        ):
            with open(f"{base_model_name_or_path}/config.json", "r") as fIn:
                config_dict = json.load(fIn)
            config = PretrainedConfig.from_dict(config_dict)
            model.config._name_or_path = config._name_or_path

        # For special case where config.json and adapter weights are in the same directory
        if hasattr(model, "peft_config"):
            model = PeftModel.from_pretrained(
                model,
                base_model_name_or_path,
            )
            model = model.merge_and_unload()

        if peft_model_name_or_path is not None:
            model = PeftModel.from_pretrained(
                model,
                peft_model_name_or_path,
            )
            if merge_peft:
                model = model.merge_and_unload()

        config = {}
        config_addr = (
            peft_model_name_or_path
            if peft_model_name_or_path is not None
            else base_model_name_or_path
        )
        if os.path.exists(f"{config_addr}/llm2vec_config.json"):
            with open(f"{config_addr}/llm2vec_config.json", "r") as fIn:
                llm2vec_config = json.load(fIn)
            config.update(llm2vec_config)

        for key, value in encoder_args.items():
            config[key] = value

        del _model
        torch.cuda.empty_cache()

        return cls(model=model, tokenizer=tokenizer, **config)


    def forward(
        self, 
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
    ):

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        outputs = self.model(**inputs)
        hidden_states = outputs[0]
        hidden_states = hidden_states.to(torch.bfloat16)

        logits = self.the_classifier(hidden_states)

        return logits
    


def load_dataset_mixcl(filePath: str):

    # ensure the dataset is .json file
    assert filePath.split('.')[-1] == "jsonl"

    # initial dataset
    context = []
    response = []

    # reading jsonl file
    with open(filePath, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            _context = data['question']
            _response = []

            # add positive sample
            posTuple = (data['positive_sentence'], data['anchor_word'])
            _response.append(posTuple)
            
            # add negative samples
            for neg in data['sim_words']:
                negTuple = ("_", neg)
                _response.append(negTuple)

            context.append(_context)
            response.append(_response)

    return context, response




transformers.logging.set_verbosity_error()

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = get_logger(__name__, log_level="INFO")
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def initialize_peft(
    model,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_modules: Optional[List[str]] = None,
):
    if lora_modules is None and model.config.__class__.__name__ in [
        "LlamaConfig",
        "MistralConfig",
        "GemmaConfig",
        "Qwen2Config",
    ]:
        lora_modules = [
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    elif lora_modules is None:
        raise ValueError("lora_modules must be specified for this model.")

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=None,
    )

    model = get_peft_model(model, config)
    print(f"Model's Lora trainable parameters:")
    model.print_trainable_parameters()
    return model


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The base model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    peft_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": ("The PEFT model checkpoint to add on top of base model.")},
    )
    bidirectional: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Whether to enable bidirectional attention in the model. If set to False, the model will use unidirectional attention."
            )
        },
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    attn_implementation: Optional[str] = field(
        default="sdpa",
        metadata={
            "help": ("The attention implementation to use in the model."),
            "choices": ["eager", "sdpa", "flash_attention_2"],
        },
    )
    pooling_mode: Optional[str] = field(
        default="mean",
        metadata={
            "help": ("The pooling mode to use in the model."),
            "choices": ["mean", "weighted_mean", "eos_token"],
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use. Options: E5"},
    )
    dataset_file_path: Optional[str] = field(
        default=None, metadata={"help": "The input training data file or folder."}
    )
    
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )


@dataclass
class CustomTrainingArguments(TrainingArguments):
    optimizer: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The optimizer to use for training. If not specified, the default optimizer will be used."
            ),
            "choices": ["AdamW", "AdamP", "LAMB", "SGD"],
        },
    )

    deepspeed: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The Deepspeed Path."
            ),
        },
    )



@dataclass
class CustomArguments:
    """
    Custom arguments for the script
    """

    lora_dropout: float = field(
        default=0.05, metadata={"help": "The dropout rate for lora"}
    )

    lora_r: int = field(default=8, metadata={"help": "The r value for lora"})

    stop_after_n_steps: int = field(
        default=10000, metadata={"help": "Stop training after n steps"}
    )

    experiment_id: Optional[str] = field(
        default=None, metadata={"help": "The experiment id"}
    )

    loss_class: Optional[str] = field(
        default="HardNegativeNLLLoss",
        metadata={
            "help": "The loss class to use for training. Options: HardNegativeNLLLoss"
        },
    )

    loss_scale: float = field(
        default=50.0, metadata={"help": "The loss scale for the loss function"}
    )



class StopTrainingCallback(TrainerCallback):
    def __init__(self, stop_after_n_steps: int):
        self.stop_after_n_steps = stop_after_n_steps

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= self.stop_after_n_steps:
            control.should_training_stop = True


class MixCLTrainer(Trainer):
    def __init__(self, *args, save_per_step=100, ckpt_name='model', **kwargs):
        super().__init__(*args, **kwargs)
        self.save_per_step = save_per_step
        self.ckpt_name = ckpt_name
        self.steps = 0


    def compute_loss(self, model, inputs, return_outputs=False):

        def tokenize_batch_with_signs(batch_clauses, batch_signs, tokenizer, max_length):
            batch_input_ids = []
            batch_sign_maps = []

            for clauses, signs in zip(batch_clauses, batch_signs):
                encoded_clauses = tokenizer(clauses, add_special_tokens=False)
                
                # Merge input_ids and signs
                input_ids = [id for clause_ids, sign in zip(encoded_clauses['input_ids'], signs) 
                        for id in clause_ids]
                sign_map = [sign for clause_ids, sign in zip(encoded_clauses['input_ids'], signs)
                        for _ in clause_ids]
                
                batch_input_ids.append(input_ids)
                batch_sign_maps.append(sign_map)

            # Pad the input_ids and sign_map
            tokenizer.padding_side = "right"
            padded_inputs = tokenizer.pad(
                {"input_ids": batch_input_ids},
                padding="max_length",
                max_length=max_length,
                return_tensors="pt"
            )

            # Pad the sign_map
            padded_sign_maps = torch.zeros(len(batch_sign_maps), max_length, dtype=torch.long)
            for i, sign_map in enumerate(batch_sign_maps):
                length = min(len(sign_map), max_length)
                padded_sign_maps[i, :length] = torch.tensor(sign_map[:length])

            return padded_inputs, padded_sign_maps

        # set max padding length
        max_length = 80
        
        _tokenizer = AutoTokenizer.from_pretrained(self.model.base_model_path)

        question_list = inputs['question_list']
        positive_segs_list = inputs['positive_segs_list']
        positive_signs_list = inputs['positive_signs_list']
        negative_segs_list = inputs['negative_segs_list']
        negative_signs_list = inputs['negative_signs_list']


        # Create a loss function instance, and ignore the padding and special tokens
        cross_entropy_loss_fct = CrossEntropyLoss(ignore_index=_tokenizer.pad_token_id).to(model.device)
        nll_loss_fct = NLLLoss(ignore_index=_tokenizer.pad_token_id).to(model.device)

        # Batch process the questions, using batch_encode_plus
        # batch_encode_plus will return input_ids and attention_mask, for input model
        questions = [q[0] for q in question_list]
        encoded_inputs = _tokenizer.batch_encode_plus(
            questions,  # question list
            padding="max_length",  # pad to make all sequences have the same length
            truncation=True,  # truncate the input if it is longer than max_length
            max_length=max_length,  # set max length
            return_tensors="pt",  # return PyTorch tensors
        ).to(model.device)

        # Extract input_ids and attention_mask from the returned dictionary
        input_ids = encoded_inputs["input_ids"]
        attention_mask = encoded_inputs["attention_mask"]

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        outputs = self.model(**encoded_inputs)
        
        total_loss = 0
        for output, negative_segs, negative_signs in zip(outputs, negative_segs_list, negative_signs_list):
            logits = output
            losses = []
            # print(f"logits shape: {logits.shape}")
            for negative_seg_tmp, negative_sign_tmp in zip(negative_segs, negative_signs):

                negative_seg = [negative_seg_tmp]
                negative_sign = [negative_sign_tmp]

                labels, signs_map = tokenize_batch_with_signs(negative_seg, negative_sign, _tokenizer, max_length)
                signs_map_tensor = signs_map.to(self.model.model.device)
                labels = labels['input_ids'].to(self.model.model.device)
                shift_labels = labels.contiguous().view(-1)
                shift_sign_map = signs_map_tensor.contiguous().view(-1).to(self.model.model.device)
                shift_logits = logits.contiguous().view(-1, logits.size(-1))

                pos_tok_num = max(((shift_sign_map == 1) & (shift_labels != _tokenizer.pad_token_id)).float().sum(), 1)
                neg_tok_num = max(((shift_sign_map == -1) & (shift_labels != _tokenizer.pad_token_id)).float().sum(), 1)

                # Process sign as -1, calculate cross entropy loss
                pos_loss = None
                neg_loss = None
                valid_indices_ce = (shift_sign_map == -1) & (shift_labels != self.model.tokenizer.pad_token_id)

                if valid_indices_ce.any():
                    valid_shift_logits_ce = shift_logits[valid_indices_ce]
                    valid_shift_labels_ce = shift_labels[valid_indices_ce]
                    neg_loss = cross_entropy_loss_fct(valid_shift_logits_ce, valid_shift_labels_ce)


                # Process sign as 1, calculate negative log likelihood loss
                valid_indices_nll = (shift_sign_map == 1) & (shift_labels != self.model.tokenizer.pad_token_id)
                if valid_indices_nll.any():
                    valid_shift_logits_nll = shift_logits[valid_indices_nll]
                    valid_shift_labels_nll = shift_labels[valid_indices_nll]
                    log_probs = torch.log_softmax(valid_shift_logits_nll, dim=-1)
                    pos_loss = nll_loss_fct(log_probs, valid_shift_labels_nll)    

                # we set loss as special value if no valid token
                assert pos_tok_num > 0 and neg_tok_num > 0, "No valid token for loss calculation"
                assert pos_tok_num is not None and neg_tok_num is not None, "None token for loss calculation"
                if pos_loss is None:
                    pos_loss = 0
                if neg_loss is None:
                    neg_loss = 0
                loss = (pos_loss / int(pos_tok_num) + neg_loss / int(neg_tok_num))
                losses.append(loss)
            
            total_loss += sum(losses) 

        total_loss /= len(negative_segs_list)
        torch.cuda.empty_cache()

        total_loss = total_loss.to(self.model.model.device)

        return total_loss
    
    
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        self.model.save(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))


    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training `~torch.utils.data.DataLoader`.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        dataset = self.train_dataset

        # Check if it is distributed training
        if self.args.local_rank != -1:  
            train_sampler = DistributedSampler(
                dataset, 
                num_replicas=self.args.world_size,  
                rank=self.args.local_rank,  
                shuffle=True  
            )
            shuffle = False  
        else:
            train_sampler = None
            shuffle = True  

        dataloader = DataLoader(
            dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=train_sampler,
            shuffle=shuffle,  
            collate_fn=Dataset_MixCL.collate_fn,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

        return dataloader
    


class LossLoggingCallback(TrainerCallback):
    def __init__(self, warmup_steps=0):
        super().__init__()
        self.warmup_steps = warmup_steps
        self.losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        print(f"Global step: {state.global_step}", end=", ")
        if logs and 'loss' in logs and state.global_step > self.warmup_steps:
            print(f"Loss: {logs['loss']}")
            self.losses.append(logs['loss'])


class GradientMonitorCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs['model']
        print("=" * 80)
        for name, param in model.named_parameters():
            if param.grad is not None:
                print(f"Layer: {name}, Grad Norm: {param.grad.norm().item()}")
        print("=" * 80)



def main_mixCL():

    # Loading arguments
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, CustomTrainingArguments, CustomArguments)
    )
    if len(sys.argv) == 3 and sys.argv[2].endswith(".json"):
        model_args, data_args, training_args, custom_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[2])
        )
    else:
        (
            model_args,
            data_args,
            training_args,
            custom_args,
        ) = parser.parse_args_into_dataclasses()

    # Initialize accelerator & ddp
    if training_args.ddp_find_unused_parameters:
        kwargs = [
            DistributedDataParallelKwargs(
                dim=0,
                broadcast_buffers=True,
                bucket_cap_mb=25,
                find_unused_parameters=True,
                check_reduction=False,
                gradient_as_bucket_view=False,
            )
        ]
    else:
        kwargs = []


    set_seed(training_args.seed)

    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    if custom_args.experiment_id is not None:
        experiment_id = custom_args.experiment_id
    else:
        experiment_id = generate_experiment_id(
            name=data_args.dataset_name,
            split="train",
            model_name=(
                model_args.model_name_or_path
                if "/" not in model_args.model_name_or_path
                else model_args.model_name_or_path.split("/")[-1]
            ),
            pooling_mode=model_args.pooling_mode,
            train_batch_size=training_args.per_device_train_batch_size,
            * training_args.gradient_accumulation_steps,
            max_seq_length=model_args.max_seq_length,
            bidirectional=model_args.bidirectional,
            epochs=training_args.num_train_epochs,
            seed=training_args.seed,
            warmup_steps=training_args.warmup_steps,
            lr=training_args.learning_rate,
            lora_r=custom_args.lora_r,
        )

    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )

    model = LLM2VecMixcl.from_pretrained(
        base_model_name_or_path=model_args.model_name_or_path,
        enable_bidirectional=model_args.bidirectional,
        peft_model_name_or_path=model_args.peft_model_name_or_path,
        merge_peft=True,
        pooling_mode=model_args.pooling_mode,
        max_length=model_args.max_seq_length,
        torch_dtype=torch_dtype,
        attn_implementation=model_args.attn_implementation,
    )

    model.model = initialize_peft(
        model.model,
        lora_r=custom_args.lora_r,
        lora_alpha=2 * custom_args.lora_r,
        lora_dropout=custom_args.lora_dropout,
    )

    training_args.output_dir = f"{training_args.output_dir}/{experiment_id}"

    file_path = data_args.mixcl_path
    context, response = load_dataset_mixcl(file_path)
    train_dataset = Dataset_MixCL(
        context=context,
        response=response,
        tokenizer=model.tokenizer,
        context_len=128,
        response_len=64
    )

    lossCallBack = LossLoggingCallback(warmup_steps=training_args.warmup_steps)
    gradienCallBack = GradientMonitorCallback()

    trainer = MixCLTrainer(
        model=model,
        args=training_args,
        callbacks=[lossCallBack],
        train_dataset=train_dataset,
    )


    trainer.train()

    # plot loss
    losses = lossCallBack.losses
    plt.plot(losses)
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig(f'./figures/loss_token.png')

    


if __name__ == "__main__":
    os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

    main_mixCL()

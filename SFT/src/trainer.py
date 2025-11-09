import bitsandbytes
import json
import time
import torch
import os
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
    TrainerCallback,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint

from .data import QCADataset, RulesDataset


class LossLoggingCallback(TrainerCallback):
    def __init__(self, warmup_steps=0):
        super().__init__()
        self.warmup_steps = warmup_steps
        self.losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and 'loss' in logs and state.global_step > self.warmup_steps:
            self.losses.append(logs['loss'])


class CustomTrainer:
    def __init__(
        self,
        model_name_or_path: str,
        mode: str,
        dataset_path: str,
        deepspeed_config_path: str,
        lora_config_path: str,
        output_dir: str = "./results",
        max_input_length: int = 512,
        max_output_length: int = 128,
        per_device_train_batch_size: int = 4,
        per_device_eval_batch_size: int = 4,
        num_train_epochs: int = 3,
        learning_rate: float = 5e-5,
        logging_dir: str = "./logs",
        logging_steps: int = 10,
        save_steps: int = 500,
        eval_steps: int = 500,
        save_total_limit: int = 2,
        fp16: bool = True,
        lr_scheduler_type: str = "linear",
        warmup_steps: int = 100,
        weight_decay: float = 0.01,
        gradient_accumulation_steps: int = 4,
        max_grad_norm: float = 1.0,
        optimizer: str = "AdamW",
        seed: int = 42,
        early_stopping_patience: int = 3,
    ):
        
        """Initialize CustomTrainer
        
        Args:
            model_name_or_path (str): The path or name of the pre-trained model
            mode (str): The mode of training (e.g., "QCA")
            dataset_path (str): The path of the dataset
            deepspeed_config_path (str): The path of the DeepSpeed configuration file
            lora_config_path (str): The path of the LoRA configuration file
            output_dir (str): The output directory
            max_input_length (int): The maximum length of the input
            max_output_length (int): The maximum length of the output
            per_device_train_batch_size (int): The training batch size per device
            per_device_eval_batch_size (int): The evaluation batch size per device
            num_train_epochs (int): The number of training epochs
            learning_rate (float): The learning rate
            logging_dir (str): The logging directory
            logging_steps (int): The logging steps interval
            save_steps (int): The saving steps interval
            eval_steps (int): The evaluation steps interval
            save_total_limit (int): The total limit of saved models
            fp16 (bool): Whether to use mixed precision training
            lr_scheduler_type (str): The type of learning rate scheduler
            warmup_steps (int): The number of warmup steps for learning rate scheduler
            weight_decay (float): The weight decay coefficient
            gradient_accumulation_steps (int): The number of gradient accumulation steps
            max_grad_norm (float): The maximum gradient norm for gradient clipping
            optimizer (str): The type of optimizer
            seed (int): The random seed
            early_stopping_patience (int): The patience for early stopping
        """

        self.model_name_or_path = model_name_or_path
        self.mode = mode
        self.dataset_path = dataset_path
        self.deepspeed_config_path = deepspeed_config_path
        self.lora_config_path = lora_config_path
        self.output_dir = output_dir
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.per_device_train_batch_size = per_device_train_batch_size
        self.per_device_eval_batch_size = per_device_eval_batch_size
        self.learning_rate = learning_rate
        self.num_train_epochs = num_train_epochs
        self.logging_dir = os.path.join("./logs/", time.strftime("%Y%m%d-%H%M%S", time.localtime()))
        self.logging_steps = logging_steps
        self.save_steps = save_steps
        self.eval_steps = eval_steps
        self.save_total_limit = save_total_limit
        self.fp16 = fp16
        self.lr_scheduler_type = lr_scheduler_type
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.optimizer = optimizer
        self.seed = seed
        self.early_stopping_patience = early_stopping_patience
        self.lossCallBack = LossLoggingCallback(warmup_steps=self.warmup_steps)
        self.callbacks = [self.lossCallBack]

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.dataset = self._load_dataset(self.mode)
        self.model = self._load_model()
        self.deepspeed_config = self._load_deepspeed_config()
        self.training_args = self._setup_training_args()
        self.trainer = self._setup_trainer()

    @classmethod
    def from_config_file(cls, config_path: str) -> "CustomTrainer":
        
        """Initialize CustomTrainer from a JSON configuration file.
        
        Args:
            config_path (str): The path of the configuration file.

        Returns:
            CustomTrainer: The CustomTrainer instance.
        
        """
        
        with open(config_path, "r") as f:
            config = json.load(f)

        return cls(**config)


    def _load_dataset(self, mode):
        """Loading dataset"""
        if mode == "QCA":
            return QCADataset(
                self.dataset_path,
                self.tokenizer,
                max_input_length=self.max_input_length,
                max_output_length=self.max_output_length,
            )
        elif mode == "Rules":
            return RulesDataset(
                self.dataset_path,
                self.tokenizer,
                max_input_length=self.max_input_length,
                max_output_length=self.max_output_length,
            )
        
        else:
            raise ValueError(f"Invalid mode: {mode}. Mode should be 'QCA' or 'Rules'.")
            

    def _load_model(self):
        """Loading model with Lora"""

        model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path)

        # load LoRA from peft
        with open(self.lora_config_path, "r") as f:
            lora_config_dict = json.load(f)
        lora_config = LoraConfig(**lora_config_dict)

        return get_peft_model(model, lora_config)

    def _load_deepspeed_config(self):
        """Loading Deepspeed configs"""
        with open(self.deepspeed_config_path, "r") as f:
            return json.load(f)

    def _setup_training_args(self):
        """Setup training configs"""
        return TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=self.per_device_train_batch_size,
            per_device_eval_batch_size=self.per_device_eval_batch_size,
            num_train_epochs=self.num_train_epochs,
            learning_rate=self.learning_rate,
            logging_dir=self.logging_dir,
            logging_steps=self.logging_steps,
            save_steps=self.save_steps,
            save_total_limit=self.save_total_limit,
            fp16=self.fp16,
            lr_scheduler_type=self.lr_scheduler_type,
            warmup_steps=self.warmup_steps,
            weight_decay=self.weight_decay,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            max_grad_norm=self.max_grad_norm,
            optim=self.optimizer,
            seed=self.seed,
            deepspeed=self.deepspeed_config,
        )
    

    def _setup_trainer(self):
        """Setup trainer"""
        return Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.dataset,
            eval_dataset=None,
            tokenizer=self.tokenizer,
            data_collator=default_data_collator,
            callbacks=self.callbacks,
        )

    def train(self):
        self.trainer.train()

    def save_model(self, output_dir: str = None):
        """Save the model"""
        if output_dir is None:
            output_dir = self.output_dir
        self.trainer.save_model(output_dir)



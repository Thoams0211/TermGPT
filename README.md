<h1 align="center">TermGPT: Multi-Level Contrastive Fine-Tuning for Terminology Adaptation in Legal and Financial Domains </h1>

This repository contains the code for the TermGPT model, which is a framework for terminology finetuning. 

![model](asset/model.png)

## Environment
Before using the code, please create a conda environment with the following command:
```bash
conda env create -f envs/termgpt.yml
conda env create -f envs/multicl.yml
```

## Sentence Graph based Data Augmentation
First, you should perform data augmentation, and activate the following conda environment before data augmentation:
```bash
conda activate termgpt
```

To construct the sentence graph, you should following the following command:
```bash
cd GraphConstruct

python graph.py \
    --rulePath  <rule_path> \
    --schemaPath <schema_path> \
    --erPath <er_path> \
    --embedPath <embed_path> \
```

To perform data augmentation, you should following the following command:
```bash
cd QCAconstruct

python construct.py \
    --datasetDir <your_dataset_path>  \
    --senMessage <your_sentence_message_path> \
    --tokMessage <your_token_message_path> \
    --mixclDir <your_output_mixcl_data_path> \
    --resultQCA <your_result_path> \
```

You can also follow the specific [instructions of GraphConstruct](GraphConstruct/README.md) to construct the sentence graph, and [instructions of QCAconstruct](QCAconstruct/README.md) to perform data augmentation.

## Multi-level Contrastive Learning
Second, you can perform multi-level contrastive learning and activate the following conda environment before multi-level contrastive learning:
```bash
conda activate multicl
cd Multi-CL
```

To perform sentence-level contrastive learning, you should following the following command:
```bash
# Start MNTP training before CL
DS_SKIP_CUDA_CHCK=1 deepspeed --include localhost:0,1 experiments/run_mntp.py -c <data_path>

# Data Preprocess
python scripts/wash.py \
    --rawPath <raw_path> \
    --outputPath <output_path> \

# Train the model
torchrun --nproc_per_node=<num_gpus> experiments/cl-sentence.py <data_path>
```

To perform token-level contrastive learning, you should following the following command:
```bash
# Data Preprocess
python scripts/mixup.py \
    -r <raw_data_path> \
    -m <mix_data_path> \

# Training the model
deepspeed --include localhost:0,1 \
    experiments/cl-token.py <mix_data_path>
```

You can also follow the specific [instructions of Multi-CL](Multi-CL/README.md) to perform multi-level contrastive learning.

## Validation
If you want to validate the model, you should activate the following conda environment:
```bash
conda activate termgpt
```

To validate the model in QA format, you should following the following command:
```bash
cd validation_qa

python eval.py \
    --model_name <model_name> \
    --output_path <output_path> \
    --data_path <data_path> \
    --bert_path <bert_path> \
    --batch_size <batch_size> \
    --max_length <max_length> \
    --top_p <top_p> \
    --top_k <top_k> \
    --temperature <temperature> \
    --api_key <api_key> \
```

To validate the model in QCA format, you should following the following command:
```bash
cd validation_choice

python eval.py \
    --model_name <model_name> \
    --output_path <output_path> \
    --data_path <data_path> \
    --batch_size <batch_size> \
    --max_length <max_length> \
    --top_p <top_p> \
    --top_k <top_k> \
    --temperature <temperature> \
```

You can also follow the specific [instructions of QA format validation](validation_qa/README.md) or [instructions of QCA format validation](validation_choice/README.md) to validate the model.
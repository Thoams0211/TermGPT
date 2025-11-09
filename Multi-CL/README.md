# Multi-level Contrastive Learning

This directory contains the code for Multi-level Contrastive Learning, which consists of **sentence-level** and **token-level** Contrastive Learning.

### Todo
```bash
mkdir cache
mkdir output && cd output
mkdir mntp && cd mntp && mkdir finance jecqa && cd ..
mkdir mntp-supervised && cd mntp-supervised && mkdir finance jecqa && cd ..
```
<!-- 
### Cache
- `cache`: This folder contains dataset used for training and testing the model. 
- `figures`: This folder contains the figures generated during the training of the model.

### Scripts
- `experiments`: This folder contains the results of the experiments conducted with the model.
- `scripts`: This folder contains the scripts used for training and testing the model. -->

### Setup
`train_configs`: This folder contains the configuration files for training the model.

### Start
To start the process of sentence-level Contrastive Learning, you can run the following command:
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

To start the process of token-level Contrastive Learning, you can run the following command:
```bash
# Data Preprocess
python scripts/mixup.py \
    -r <raw_data_path> \
    -m <mix_data_path> \

# Training the model
deepspeed --include localhost:0,1 \
    experiments/cl-token.py <mix_data_path>
```
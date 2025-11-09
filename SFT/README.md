# Supervised Fine-Tuning (SFT) 

This directory contains the code for supervised fine-tuning (SFT) of the model.

### Todo
```bash
mkdir figures logs saves
cd figures && mkdir finance jecqa && cd ..
cd saves && mkdir finance jecqa && cd ..
```
<!-- 
### Cache
- `figures`: Folder that saves the figures of loss in SFT process.
- `logs`: Folder that saves the logs of the SFT process.
- `saves`: Folder that saves the checkpoints of the SFT process. -->

### Setup
- `setup/config_QCA.json`: Configuration files for basic arguments for SFT in QCA dataset.
- `setup/config_Rules.json`: Configuration files for basic arguments for SFT in Rules dataset.
- `setup/deepspeed_config.json`: Configuration file for DeepSpeed.
- `setup/lora_config.json`: Configuration file for LoRA.

<!-- ### Scripts
- `src/configer.py`: Contains the config management class.
- `src/dataset.py`: Contains the dataset class for loading and processing data.
- `src/merge.py`: Contains the script for merging checkpoints and the foundation model.
- `src/trainer.py`: Contains the training class for fine-tuning the model.

### Main Scripts
- `run.py`: The main script that orchestrates the SFT process.
- `run.sh`: A shell script to run the SFT process. -->

### Start
To start the SFT process, you can run the following command with QCA dataset:
```bash
deepspeed --include localhost:0,1 run.py --mode QCA
```
or with raw rule dataset:
```bash
deepspeed --include localhost:0,1 run.py --mode Rules
```

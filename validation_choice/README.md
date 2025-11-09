# Validation in choice format

### Todo
```bash
mkdir logs output
cd output && mkdir ablation finance jecqa
cd finance && mkdir baselines local && cd ..
cd jecqa && mkdir baselines local && cd ..
cd ..
```
<!-- 
### Cache
- `logs`: Folder that saves the logs of the validation process.
- `messages`: Folder that saves the messages of calling batch API of Qwen or Deepseek.
- `output/ablation`: Folder that saves the output of the validation process of ablation study.
- `output/finance`: Folder that saves the output of the validation process on the finance dataset.
- `output/jecqa`: Folder that saves the output of the validation process on the JECQA dataset.

### Scripts
- `src/batch_script.py`: Saving methods that call batch API of Qwen or Deepseek.
- `src/inference.py`: The main script that orchestrates the validation process.

### Main Scripts
- `eval_baseline.py`: Script to evaluate the baselines(Qwen, Deepseek) on the test set.
- `eval_baseline.sh`: Shell script to run the evaluation of baselines.
- `eval.py`: Script to evaluate the fine-tune model on the test set.
- `eval_lawyer.py`: Script to evaluate the LawyerLlama model on the test set.
- `eval.sh`: Shell script to run the evaluation of fine-tune model. -->

### Start
To start the validation process of TermGPT, you can run the following command:
```bash
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

When you want to evaluate the baseline models, you can run the following command:
```bash
python eval_baseline.py \
    --model_name <model_name> \
    --output_path <output_path> \
    --rawQCA_path <rawQCA_path> \
    --bert_path <bert_path> \
    --batch_size <batch_size> \
    --api_key <api_key> \
    --top_p <top_p> \
    --top_k <top_k> \
    --temperature <temperature> \
```
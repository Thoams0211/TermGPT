# QCA Construct
This directory contains the code for constructing QCA. Please make sure you have activated the `termgpt` conda environment before running the code.

### Todo
```bash
mkdir finance jecqa logs
cd finance && mkdir dataset_output dataset_sentence dataset_token && cd ..
cd jecqa && mkdir dataset_output dataset_sentence dataset_token && cd ..
```
<!-- 
### Cache
- `finance/dataset_output`: Folder that saves the output of the finance QCA.
- `finance/dataset_sentence`: Folder that saves the messages of calling batch API to generate sentence-level QCA.
- `finance/dataset_token`: Folder that saves the messages of calling batch API to generate token-level QCA.
- `jecqa/dataset_output`: Folder that saves the output of the JECQA QCA.
- `jecqa/dataset_sentence`: Folder that saves the messages of calling batch API to generate sentence-level QCA.
- `jecqa/dataset_token`: Folder that saves the messages of calling batch API to generate token-level QCA.
- `logs`: Folder that saves the logs of the QCA construction process.

### Scripts
- `utils/generation.py`: Contains the methods for generating QCA.
- `utils/logTools.py`: Contains the methods for logging the QCA construction process.
- `utils/sentence_batch.py`: Contains the methods for calling the batch API to generate sentence-level QCA.
- `utils/token_batch.py`: Contains the methods for calling the batch API to generate token-level QCA.
- `utils/similarity.py`: Contains the methods for calculating the similarity between sentences.

### Main Scripts
- `construct_finance.sh`: A shell script to run the QCA construction process on the finance dataset.
- `construct_jecqa.sh`: A shell script to run the QCA construction process on the JECQA dataset.
- `construct.py`: The main script that orchestrates the QCA construction process.
- `id2type_sentence.json`: A JSON file that maps the IDs to the types (e.g. "民法", "商法") of sentences in sentence-level QCA.
- `id2type_token.json`: A JSON file that maps the IDs to the types of tokens in token-level QCA. -->

### Start
**🚨 Before running the code, you should make sure that you have constructed the sentence graph.**

To start the QCA construction process, you can run the following command:
```bash
python construct.py \
    --datasetDir <your_dataset_path>  \
    --senMessage <your_sentence_message_path> \
    --tokMessage <your_token_message_path> \
    --mixclDir <your_output_mixcl_data_path> \
    --resultQCA <your_result_path> \
```
`config.yaml` is the configuration file for the QCA construction process.

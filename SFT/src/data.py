import json
import os
import random
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

class QCADataset(Dataset):
    def __init__(self, folder_path, tokenizer, max_input_length, max_output_length):
        """Initialize the dataset
        
        Args:
            folder_path (str): The folder path of the data file
            tokenizer (transformers.PreTrainedTokenizer): The tokenizer
            max_input_length (int): The maximum length of the input
            max_output_length (int): The maximum length of the output
        
        """
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"文件夹路径不存在: {folder_path}")

        with open(os.path.join(folder_path, "sentenceQA.json"), "r", encoding="utf-8") as f:
            data = json.load(f)
        
        with open(os.path.join(folder_path, "tokenQA.json"), "r", encoding="utf-8") as f:
            data.extend(json.load(f))

        self.data = data
            

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        # reconstruct the data: turn json into text
        item = self.data[idx]
        question = item["question"]
        choices = []
        cnt = 1
        while True:
            if f"choice_{cnt}" in item.keys():
                choices.append(item[f"choice_{cnt}"])
                cnt += 1
            else:
                break
        random.shuffle(choices)
        input_text = f"问题: {question}\n"
        for i, choice in enumerate(choices):
            input_text += f"选项_{i + 1}: {choice}\n"

        answer = "答案: " + item["answer"]
        # tokenize the input and the answer
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        labels = self.tokenizer(
            answer,
            max_length=self.max_output_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )["input_ids"]

        # return the input and the answer
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
        }
    

class RulesDataset(Dataset):
    def __init__(self, dataPath_root, tokenizer, max_input_length, max_output_length):
        """Initialize the dataset
        
        Args:
            data_path (str): The path of the data file
            tokenizer (transformers.PreTrainedTokenizer): The tokenizer
            max_input_length (int): The maximum length of the input
            max_output_length (int): The maximum length of the output
        
        """
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

        # Process each subdirectory
        data = []
        for root, dirs, files in os.walk(dataPath_root):
            # Process relevant JSON files
            for filename in files:
                if filename == 'train.json':
                    with open(os.path.join(root, filename), "r", encoding="utf-8") as f:
                        data.extend(json.load(f))

        self.data = data
            

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        # reconstruct the data: turn json into text
        item = self.data[idx]
        input_text = item["text"]

        # tokenize the input and the answer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        labels = inputs["input_ids"].clone()

        # return the input and the answer
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
        }


if __name__ == "__main__":
    pass
import argparse
import json
import random
from tqdm import tqdm


parser = argparse.ArgumentParser(description='Wash data for MultiCL')
parser.add_argument("--rawPath", type=str, required=True, help="Rule Path")
parser.add_argument("--outputPath", type=str, required=True, help="ER result Path")
args = parser.parse_args()
assert args.rawPath.split('.')[-1] == 'json', "Please input a json file"
assert args.outputPath.split('.')[-1] == 'jsonl', "Please output a jsonl file"


with open(args.rawPath) as f:
    data = json.load(f)
with open(args.outputPath, "w", encoding='utf-8') as f:
    f.write('')

resList = []
for data_point in tqdm(data, desc="Writing to file"):
    if data_point is None:
        raise ValueError("Data point is None")
    choices = [data_point['choice_1'], data_point['choice_2'], data_point['choice_3'], data_point['choice_4']]
    negatives = [x for x in choices if x != data_point['answer']]
    for negative in negatives:
        temp = {
            'query': data_point['question'],
            'positive': data_point['answer'],
            'negative': negative
        }
        resList.append(temp)

# shuffle the data
random.shuffle(resList)

# write to file
with open(args.outputPath, "a", encoding='utf-8') as f:
    for item in resList:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

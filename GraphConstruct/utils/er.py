# /-*- coding: utf-8 -*/
import ast
import json
from openai import OpenAI
import os
import pandas as pd
import re
from tqdm import tqdm


def extract_fields(text):
    pattern = r"行为规范:\s*(.*?);\s*违规后果:\s*(.*?);\s*合规依据:\s*(.*?);"
    match = re.search(pattern, text)

    if match:
        return {
            "行为规范": match.group(1).strip(),
            "违规后果": match.group(2).strip(),
            "合规依据": match.group(3).strip(),
        }
    else:
        return None

def subwordCheck(subwords: list, oriRule: str) -> list:

    masks = [ 1 for subword in subwords ]

    # check if subwords is in oriRule
    for s in range(len(subwords)):
        subword = subwords[s]
        if subword not in oriRule:
            # print(f"word {subword} not in rule: {oriRule}")
            masks[s] = 0

    resLis = []
    for s in range(len(subwords)):
        subword = subwords[s]
        if masks[s] != 0:
            resLis.append(subword)

    return resLis



def erRules(rules: list, schema:str, apiKey: str) -> any:
    
    r"""
    #### This function is used to extract entities from the rules.
    
    ## Args:
        dataPath (str):
            The path of the rules JSON file.
        erPath (str):
            The path of the entity recognition JSON file.
        apiKey (str):
            The API key of Qwen.
            
            
    ## Returns:
        None
    
    """
        
    client = OpenAI(
        api_key=apiKey, 
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    
    dumpList = []
    
    for r in tqdm(range(len(rules)), desc="NER Process", leave=False):
        ruleDic = rules[r]
        rawRule = ruleDic['text']

        # TODO: this lines are adapted to the my dataset
        rule = ';'.join(rawRule.split(';', 1)[1:])
        
        content = f"""
        
            请根据以下指令帮助我从提供的文本中提取术语，并严格按照指定的格式输出结果。

            Schema (上位词：示例):

                {schema}


            任务说明与回答格式：

            1. 文本分析：仔细阅读下面给出的文本，并识别出文本中所有的术语名词和偏正短语。
            2. 术语归类：将识别到的术语名词和偏正短语归类到以上schema类别中。
            3. 格式化输出：以JSON格式返回结果，确保每个类别下的术语都用数组形式列出，即使该类别下只有一个术语。如果某个类别没有找到匹配的术语，则返回空数组。无需回复解释、理由等其他信息。
            4. 精确性：确保所有提取的术语准确无误且与文本内容相关，且必须是原文本的子字符串; 并且不得抽取任何和数字、日期相关的词语（包括阿拉伯数字和汉语数字）
            5. 原子性：确保抽取的是单个名词，而非动宾短语或偏正短语。

            待分析文本：
            {rule}

        """

        content += """
            \n预期回答格式（JSON）：
            {
                "类别1": ["词语1", ...],
                "类别2": ["词语2", "词语3", ...],
                "类别3": ["词语4", ...],
                ... ...
            }
        
        """ 
        
        max_retries = 5
        for cnt in range(max_retries):
            try:
                completion = client.chat.completions.create(
                    model="qwen-plus",
                    messages=[
                        {'role': 'system', 'content': '你是一名信息抽取专家'},
                        {'role': 'user', 'content': content}],
                    temperature=0.5
                )

            except:
                print(f"{r} Dic Error")
                raise Exception("API Error")
                
            responseDic = json.loads(completion.model_dump_json())
            text = responseDic['choices'][0]['message']['content']

            try:
                resDict = json.loads(text)
                break
            except:
                if cnt == max_retries - 1:
                    print(f"{r} Error")
                continue
            
        
        entities = []
        for _, values in resDict.items():
            for value in values:
                if value not in entities:
                    entities.append(value)

        # check entity if the entity is a subword of original rule
        entities = subwordCheck(entities, rule)
        dumpList.append({'rule': rawRule, 'entities': entities})

    return dumpList


def process_single_file(input_file: str, output_file: str, schema: str, apiKey: str, process_func: callable) -> None:
    """Processes a single JSON file with error handling and validation.

    Args:
        input_file: Full path to input JSON file
        output_file: Full path for output JSON file
        schema: Schema for the data processing
        apiKey: API key for external services (if needed)
        process_func: Data processing function

    Raises:
        json.JSONDecodeError: If input file contains invalid JSON
        IOError: If file cannot be read/written
    """
    print(f"Processing: {input_file} -> {output_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    processed_data = process_func(data, schema, apiKey)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    


def er(input_root: str, output_root: str, schemaPath: str, apiKey: str, process_func: callable) -> None:
    """Processes legal text JSON files while preserving directory structure.

    The function automatically maps input paths to corresponding output paths in the cache
    directory, maintaining the same relative structure.

    Args:
        input_root: Absolute path to the raw rules directory (e.g., '/home/.../rawRules')
        output_root: Absolute path to the cache directory (e.g., '/home/.../cache')
        schemaPath: Path to the schema file used for data processing
        apiKey: API key for external services (if needed)
        process_func: Data transformation function that processes list of dictionaries

    Raises:
        ValueError: If input_root doesn't exist or isn't a directory
        OSError: For filesystem-related errors during processing
    """
    # Validate input directory
    if not os.path.isdir(input_root):
        raise ValueError(f"Input directory does not exist: {input_root}")
    
    # Loading schema
    with open(schemaPath, 'r', encoding='utf-8') as f:
        schema = f.read()

    # Process each subdirectory
    for root, dirs, files in os.walk(input_root):
        # Calculate corresponding output directory
        relative_path = os.path.relpath(root, input_root)
        output_dir = os.path.join(output_root, relative_path)
        
        # Create output directory if needed
        os.makedirs(output_dir, exist_ok=True)

        # Process relevant JSON files
        for filename in files:
            if filename in ('train.json', 'test.json'):
                process_single_file(
                    input_file=os.path.join(root, filename),
                    output_file=os.path.join(output_dir, filename),
                    schema=schema,
                    apiKey=apiKey,
                    process_func=process_func
                )
        
        

if __name__ == '__main__':
    pass

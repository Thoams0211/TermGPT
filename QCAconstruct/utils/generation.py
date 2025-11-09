# -*- coding: utf-8 -*-
import os
import re
from openai import OpenAI
import time
from tqdm import tqdm
import string
import json
from math import ceil
from pathlib import Path
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from .logTools import logger_process, logger_fileID


# File lock in case of race condition
file_lock = threading.Lock()
file_create_lock = threading.Lock()

# Config API url
API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def parseSentenceResponse(response:str, the_type:str) -> dict:

    """This function is used to parse the sentence-level response from OpenAI API.
    
    Args:
        response (str):
            The response from OpenAI API.
    
    Returns:
        dict: The parsed response.
    
    """
    
    pattern = re.compile(
        r'(?:<问题>:|问题:)\s*(.*?)\s*' 
        r'(?:<选项A>:|选项A:)\s*(.*?)\s*' 
        r'(?:<选项B>:|选项B:)\s*(.*?)\s*' 
        r'(?:<选项C>:|选项C:)\s*(.*?)\s*' 
        r'(?:<选项D>:|选项D:)\s*(.*?)\s*' 
        r'(?:<正确答案>:|正确答案:)\s*(.*)', 
        re.DOTALL 
    )


    match = re.search(pattern, response)

    
    if match:
        return {
            "question": match.group(1).strip(),
            "choice_1": match.group(2).strip(),
            "choice_2": match.group(3).strip(),
            "choice_3": match.group(4).strip(),
            "choice_4": match.group(5).strip(),
            "answer": match.group(6).strip(),
            "type": the_type,
        }
    else:
        with open("./logs/waste.txt", 'a', encoding='utf-8') as file:
            file.write(f"{response}\n")
            file.write(f"=====================\n")
        return None
    


def parseTokenResponse(response:str, the_type:str) -> dict:

    """This function is used to parse the token-level response from OpenAI API.
    
    Args:
        response: str
            The response from OpenAI API.

    Returns:
        dict: The parsed response.

    """

    # extract the question
    question_pattern = r"<问题>:\s*(.+?)(?=\n|$)"
    question_match = re.search(question_pattern, response, re.DOTALL)
    question = question_match.group(1).strip() if question_match else None
    if question is None:
        question_pattern = r"问题:\s*(.+?)(?=\n|$)"
        question_match = re.search(question_pattern, response, re.DOTALL)
        question = question_match.group(1).strip() if question_match else None

    # extract the correct answer
    correct_answer_pattern = r"<正确答案>:\s*(.+?)(?=\n|$)"
    correct_answer_match = re.search(correct_answer_pattern, response, re.DOTALL)
    correct_answer = correct_answer_match.group(1).strip() if correct_answer_match else None
    if correct_answer is None:
        correct_answer_pattern = r"正确答案:\s*(.+?)(?=\n|$)"
        correct_answer_match = re.search(correct_answer_pattern, response, re.DOTALL)
        correct_answer = correct_answer_match.group(1).strip() if correct_answer_match else None

    # extract the rewritten sentence
    rewritten_sentence_pattern = r"<改写句子>:\s*(.+)"
    rewritten_sentence_match = re.search(rewritten_sentence_pattern, response, re.DOTALL)
    rewritten_sentence = rewritten_sentence_match.group(1).strip() if rewritten_sentence_match else None
    if rewritten_sentence is None:
        rewritten_sentence_pattern = r"改写句子:\s*(.+)"
        rewritten_sentence_match = re.search(rewritten_sentence_pattern, response, re.DOTALL)
        rewritten_sentence = rewritten_sentence_match.group(1).strip() if rewritten_sentence_match else None
    if "\n\n" in rewritten_sentence:
        rewritten_sentence = rewritten_sentence.split("\n\n")[0]

    # add question, answer
    res = {
        "question": question,
    }
    res['answer'] = correct_answer.strip()
    res['choice_1'] = correct_answer.strip()

    # index similar entities
    with open("similarDict.pkl", 'rb') as file:
        similarDict = pickle.load(file)
    simEntities = list(similarDict[res["answer"]])

    # add similar entities
    for i in range(len(simEntities)):
        res[f"choice_{i+2}"] = simEntities[i]

    # record the type of the question
    res['type'] = the_type
    res["rewritten_sentence"] = rewritten_sentence.strip()


    return res
    



def deleteFile(fileID:str, apiKey:str) -> None:

    """This function is used to delete the file & batch task in Aliyun-Bailian cloud server, in case of the file is not needed anymore or any exceptions.
    
    Args:
        fileID (str): 
            The file ID in Aliyun-Bailian cloud server.
        apiKey (str):
            The API key of OpenAI API.

    Returns:
        None
    
    """


    client = OpenAI(
        api_key=apiKey,
        base_url=API_URL,
    )

    file_object = client.files.delete(fileID)
    file_id_dict = json.loads(file_object.model_dump_json())
    file_id = file_id_dict.get('id')  
    logger_fileID.info(f"[INFO] DELETE: File ID: {file_id}")



def process_generation(filePaths, outputDir, apiKey, groupNum, mode) -> None:

    """This function is the single thread function for data agumentation procession. It will call OpenAI API to generate the QCA dataset.

    Args:
        filePaths (list): 
            The list of file paths.
        outputDir (str): 
            The output directory.
        apiKey (str): 
            The API key of OpenAI API.
        groupNum (int): 
            The group number.
        mode (str): 
            The mode of the data agumentation, either 'sentence' or 'token'.

    Returns:
        None    
    
    """

    assert mode in ["sentence", "token"], "The mode should be either 'sentence' or 'token'."

    # set the parseResponse function to parse the openai response
    if mode == "sentence":
        parseResponse = parseSentenceResponse
    else:
        parseResponse = parseTokenResponse

    # create the OpenAI client
    client = OpenAI(
        api_key=apiKey,
        base_url=API_URL,
    )

    logger_process.info(f"[INFO] START Processing group {groupNum}")

    # record waste count
    waste = 0

    # iterate over the batch files
    for j in range(len(filePaths)):

        # get the file path
        filePath = filePaths[j]

        try:
            # upload the file to the cloud server
            with file_create_lock:
                file_object = client.files.create(file=Path(filePath), purpose="batch")
                file_id = file_object.id
                logger_fileID.info(f"[INFO] UPLOADING: {j+1} / {len(filePaths)} in Group {groupNum}, File: {filePath}, FileID: {file_id}")

            # create a batch task
            batch_id = client.batches.create(
                input_file_id=file_id,
                endpoint="/v1/chat/completions",
                completion_window="24h"
            ).id

            # save the file_id and batch_id to the log file
            with file_lock:
                with open("./logs/file_ids.txt", 'a', encoding='utf-8') as file:
                    file.write(f"{file_id}\n")
                with open("./logs/batch_ids.txt", 'a', encoding='utf-8') as file:
                    file.write(f"{batch_id}\n")

            # wait for the batch to complete
            cnt = 0
            while True:
                # API calling
                batch = client.batches.retrieve(batch_id)
                output_file_id = batch.output_file_id
                if output_file_id:
                    break
                else:
                    time.sleep(60)
                    logger_process.info(f"[INFO] {cnt} min waiting for the batch to complete {j+1} / {len(filePaths)} of Group: {groupNum}, batchID: {batch_id}")
                    cnt += 1
            
            # fetch the content of response
            content = client.files.content(file_id=output_file_id)
            data = content.text.split("\n")


        except Exception as e:
            # catch any exception and log the error
            logger_process.error(f"[ERROR] Processing file: {filePath} failed. Group: {groupNum}. Error: {e}")
            client.batches.cancel(batch_id)
            raise e

        finally:
            # delete the file in Aliyun-Bailian cloud server
            deleteFile(
                fileID=file_id,
                apiKey=apiKey
            )

        # save the data to the json file
        restLis = []
        
        # Load required_id to type
        with open(f"id2type_{mode}.json", 'r', encoding='utf-8') as file:
            id2type = json.load(file)

        with file_lock:
            with open(f"{outputDir}/{mode}QA.json", 'r', encoding='utf-8') as file:
                existLis = json.load(file)
            for r in data:
                try:
                    response = json.loads(r)
                except:
                    restResponse = r.split("\n")
                    restLis.extend(restResponse)
                    continue

                try:
                    content = response['response']['body']['choices'][0]['message']['content']
                    content = parseResponse(content, id2type[response['custom_id']])
                except:
                    with open("./logs/waste.txt", 'a', encoding='utf-8') as file:
                        file.write(f"{content}\n")
                        file.write(f"=====================\n")
                    waste += 1
                    continue
                if not content:
                    waste += 1
                    continue
                existLis.append(content)

            for r in restLis:
                if r == "":
                    with open("./logs/waste.txt", 'a', encoding='utf-8') as file:
                        file.write(f"content empty\n")
                        file.write(f"=====================\n")
                    waste += 1
                    continue
                response = json.loads(r)
                content = response['response']['body']['choices'][0]['message']['content']
                content = parseResponse(content, id2type[response['custom_id']])
                existLis.append(content)


            with open(f"{outputDir}/{mode}QA.json", 'w', encoding='utf-8') as file:
                json.dump(existLis, file, ensure_ascii=False, indent=4)

        logger_process.info(f"[INFO] FINISH {j+1} / {len(filePaths)} of Group: {groupNum}")
        
    logger_process.info(f"[INFO] WASTE: {waste}")


def generation(fileDir:str, outputDir:str, apiKey:str, mode:str):

    """This function isthe core function of data agumentation procession. It will utilize multi-threading and calling API in batches.

    Args:
        fileDir (str): 
            The directory of the input files.
        outputDir (str): 
            The directory of the output files.
        apiKey (str): 
            The API key of OpenAI API.
        mode (str): 
            The mode of the data agumentation, either 'sentence' or 'token'.
    
    """

    assert mode in ["sentence", "token"], "The mode should be either 'sentence' or 'token'."

    # check if has the dataset directory
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    # Check if has the output file. If exists, clean the file
    outputFile = f"{outputDir}/{mode}QA.json"
    with open(outputFile, 'w', encoding='utf-8') as file:
        json.dump([], file, ensure_ascii=False, indent=4)

    # set the number of files per group
    files_per_group = 1

    # get all jsonl files in the directory
    files = []
    for root, directories, file_names in os.walk(fileDir):
        for filename in file_names:
            files.append(os.path.join(root, filename))


    # split the files into groups
    num_groups = ceil(len(files) / files_per_group)
    groups = []
    for i in range(num_groups):
        start_index = i * files_per_group
        end_index = min((i + 1) * files_per_group, len(files))
        group = files[start_index:end_index]
        groups.append(group)


    # create the thread pool
    logger_process.info(f"================================== START GENERATION ==============================")
    with ThreadPoolExecutor(max_workers=num_groups) as executor:
        futures = [executor.submit(process_generation, filePaths, outputDir, apiKey, i+1, mode) for i, filePaths in enumerate(groups)]
        for future in as_completed(futures):
            future.result()
    logger_process.info(f"================================== END GENERATION ==============================")


    # delete similar entity dictionary
    if os.path.exists("similar_entity.pkl"):
        os.remove("similar_entity.pkl")



if __name__ == "__main__":
    pass
    
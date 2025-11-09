import argparse
from datetime import datetime
import json
import random
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from peft import PeftModel, PeftConfig
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from src.inference import batch_inference



def shuffle_choices(data):
    """Shuffle the choices in the data.

    Args:
        data (list): The input data.

    Returns:
        list: The shuffled data.
    """

    for d in data:
        try:
            answer = d['choice_1']
            choices = [d['choice_1'], d['choice_2'], d['choice_3'], d['choice_4']]
            random.shuffle(choices)
            for c in range(4):
                choice = choices[c]
                d['choice_' + str(c+1)] = choice
                if choice == answer:
                    d['answer'] = f"choice_{c+1}"
        except:
            continue
    return data



def load_model_and_tokenizer(model_name, lora_weights=None):
    """Loading the pre-trained model and tokenizer.

    Args:
        model_name (str): The name or path of the pre-trained model.
        lora_weights (str, optional): The path of the LoRA weights. Default is None.

    Returns:
        model: The loaded model.
        tokenizer: The loaded tokenizer.
    """

    # load the model
    model1 = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")

    # load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # load the LoRA weights
    if lora_weights:
        print("Loading LoRA weights from:", lora_weights)
        model2 = PeftModel.from_pretrained(model1, lora_weights)
        return model2, tokenizer
    else:
        print("Loading model without LoRA weights.")
        return model1, tokenizer
    

def inference_process(model, tokenizer, data_batches, output_path, errorPath, args):

    truths_process = []
    answers_process = []
    types_process = []
    error_list_total_process = []

    for data_batch in tqdm(data_batches, desc="Batch Inference", unit="batch"):
        # Perform batch inference
        generated_texts, error_list = batch_inference(
            model=model,
            tokenizer=tokenizer,
            questions=data_batch,
            max_length=args.max_length,
            do_sample=True,
            top_p=args.top_p,
            temperature=args.temperature,
            top_k=args.top_k,
        )

        # Record the error list
        error_list_total_process.extend(error_list)

        # Print the generated texts
        qatDictList = []
        for i, text in enumerate(generated_texts):
            if text:
                qatDict = {"question": data_batch[i], "answer": text, "truth": data_batch[i]['answer'], "type": data_batch[i]['type']}
                qatDictList.append(qatDict)

        with open(output_path, "r", encoding='utf-8') as f:
            oriDictList = json.load(f)
        
        oriDictList.extend(qatDictList)

        with open(output_path, "w", encoding='utf-8') as f:
            json.dump(oriDictList, f, ensure_ascii=False, indent=4)
        
        # Calculate the metrics
        for qat in tqdm(qatDictList, desc=f"Evaluating", unit="item"):
            truths_process.append(qat["truth"])
            answers_process.append(qat["answer"])
            types_process.append(qat["type"])

            # insert into error file
            if qat["answer"] != qat["truth"]:
                with open(errorPath, "r", encoding='utf-8') as f:
                    oriErrorList = json.load(f)
                oriErrorList.append(qat)
                with open(errorPath, "w", encoding='utf-8') as f:
                    json.dump(oriErrorList, f, ensure_ascii=False, indent=4)

    return truths_process, answers_process, types_process, error_list_total_process



def calculate_metrics(predicted_choices, actual_choices):
    # Map the choices to integers
    choice_to_int = {'choice_1': 0, 'choice_2': 1, 'choice_3': 2, 'choice_4': 3}
    
    # Convert the predicted and actual choices to integers
    predicted = [choice_to_int[choice] for choice in predicted_choices]
    actual = [choice_to_int[choice] for choice in actual_choices]
    
    # Calculate Precision, Recall, F1, Accuracy
    precision = precision_score(actual, predicted, average='macro')
    recall = recall_score(actual, predicted, average='macro')
    f1 = f1_score(actual, predicted, average='macro')
    accuracy = accuracy_score(actual, predicted)
    
    return accuracy, precision, recall, f1


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, help="The name or path of the pre-trained model.")
    parser.add_argument("--lora_weights", type=str, help="The path of the LoRA weights.")
    parser.add_argument("--data_path", type=str, help="The path of the input data.")
    parser.add_argument("--bert_path", type=str, help="The path of the BERT model.")
    parser.add_argument("--output_path", type=str, help="The path of the output file.")
    parser.add_argument("--batch_size", type=int, default=8, help="The batch size for inference.")
    parser.add_argument("--max_length", type=int, default=50, help="The maximum length of the generated text.")
    parser.add_argument("--top_p", type=float, default=0.9, help="The top-p value for sampling.")
    parser.add_argument("--temperature", type=float, default=0.2, help="The temperature value for sampling.")
    parser.add_argument("--top_k", type=int, default=10, help="The top-k value for sampling.")
    args = parser.parse_args()

    model_name = args.model_name
    lora_weights = args.lora_weights
    data_path = args.data_path
    bert_path = args.bert_path
    output_path = args.output_path
    assert data_path.endswith(".json"), "The input data must be in JSON format."


    # Load the pre-trained model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name, lora_weights)
    tokenizer.pad_token = tokenizer.eos_token

    # Load the input data
    with open(data_path, "r") as f:
        data = json.load(f)

    # shunffle sentence-level QCA & token-level QCA
    random.shuffle(data)

    # initialize the output file
    with open(output_path, "w", encoding='utf-8') as f:
        json.dump([], f, ensure_ascii=False, indent=4)

    # Shuffle the choices
    shuffle_choices(data)

    # Split the data into batches
    data_batches = [data[i : i + args.batch_size] for i in range(0, len(data), args.batch_size)]

    # initialize the scores
    answers = []
    truths = []
    types = []
    error_list_total = []

    # construct the error file
    errorPath = f"logs/timestamp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(errorPath, "w", encoding='utf-8') as f:
        json.dump([], f, ensure_ascii=False, indent=4)

    # Perform inference
    truths_tmp, answers_tmp, types_tmp, error_list_total_tmp = inference_process(
        model=model,
        tokenizer=tokenizer,
        data_batches=data_batches,
        output_path=output_path,
        errorPath=errorPath,
        args=args
    )
    answers.extend(answers_tmp)
    truths.extend(truths_tmp)
    error_list_total.extend(error_list_total_tmp)
    types.extend(types_tmp)

    while error_list_total != []:
        # Perform inference on the error list
        data_batches = [error_list_total[i : i + args.batch_size] for i in range(0, len(error_list_total), args.batch_size)]
        truths_tmp, answers_tmp, types_tmp, error_list_total_tmp = inference_process(
            model=model,
            tokenizer=tokenizer,
            data_batches=data_batches,
            output_path=output_path,
            errorPath=errorPath,
            args=args
        )
        answers.extend(answers_tmp)
        truths.extend(truths_tmp)
        types.extend(types_tmp)
        error_list_total = error_list_total_tmp
        

    # Calculate Total Precision, Recall, and F1
    acc, precision, recall, f1 = calculate_metrics(answers, truths)
    

    # Calculate Type-wise Precision, Recall, and F1
    type_wise_metrics = {}
    unique_types = set(types)
    for t in unique_types:
        type_indices = [i for i, x in enumerate(types) if x == t]
        type_answers = [answers[i] for i in type_indices]
        type_truths = [truths[i] for i in type_indices]
        acc_type, precision_type, recall_type, f1_type = calculate_metrics(type_answers, type_truths)
        type_wise_metrics[t] = {
            "accuracy": acc_type,
            "precision": precision_type,
            "recall": recall_type,
            "f1": f1_type
        }
        print("=" * 100)
        print(f"Type {t} metrics:")
        print(f"Accuracy: {acc_type}, Precision: {precision_type}, Recall: {recall_type}, F1: {f1_type}")
        print("=" * 100)

    print("=" * 100)
    print("Total metrics:")
    print(f"Accuracy: {acc}, Precision: {precision}, Recall: {recall}, F1: {f1}")
    print("=" * 100)



if __name__ == "__main__":
    main()
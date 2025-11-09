# -*- coding: utf-8 -*-
import argparse
import deepspeed
import json
import torch
import random
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from peft import PeftModel

from src.util import calculate_bleu, calculate_rouge, calculate_bertscore, calculate_llm, calculate_meteor_chinese
from src.inference import batch_inference



def load_model_and_tokenizer(model_name, lora_weights=None):
    """Loading the pre-trained model and tokenizer.

    Args:
        model_name (str): The name or path of the pre-trained model.
        lora_weights (str, optional): The path of the LoRA weights. Default is None.

    Returns:
        model: The loaded model.
        tokenizer: The loaded tokenizer.
    """

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Load the model
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")

    # Load the LoRA weights
    if lora_weights:
        model = PeftModel.from_pretrained(model, lora_weights)

    return model, tokenizer


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, help="The name or path of the pre-trained model.")
    parser.add_argument("--lora_weights", type=str, help="The path of the LoRA weights.")
    parser.add_argument("--data_path", type=str, help="The path of the input data.")
    parser.add_argument("--bert_path", type=str, help="The path of the BERT model.")
    parser.add_argument("--output_path", type=str, help="The path of the output file.")
    parser.add_argument("--batch_size", type=int, default=8, help="The batch size for inference.")
    parser.add_argument("--api_key", type=str, help="The API key for the model.")
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
    api_key = args.api_key
    assert data_path.endswith(".json"), "The input data must be in JSON format."


    # Load the pre-trained model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name, lora_weights)
    tokenizer.pad_token = tokenizer.eos_token

    # Load the input data
    with open(data_path, "r") as f:
        data = json.load(f)

    # initialize the output file
    with open(output_path, "w", encoding='utf-8') as f:
        json.dump([], f, ensure_ascii=False, indent=4)

    # Split the data into batches
    data_batches = [data[i : i + args.batch_size] for i in range(0, len(data), args.batch_size)]

    # Perform batch inference
    answers = []
    truths = []
    types = []
    questions = []
    for data_batch in tqdm(data_batches, desc="Batch Inference", unit="batch"):

        questions_tmp = [d["question"] for d in data_batch]
        answers_tmp = [d["answer"] for d in data_batch]
        types_tmp = [d['type'] for d in data_batch]

        # Perform batch inference
        generated_texts = batch_inference(
            model=model,
            tokenizer=tokenizer,
            questions=questions_tmp,
            max_length=200,
            do_sample=True,
            top_p=args.top_p,
            temperature=args.temperature,
            top_k=args.top_k,
        )

        # Print the generated texts
        qatDictList = []
        waste = 0
        for i, text in enumerate(generated_texts):
            if text:
                qatDict = {"question": questions_tmp[i], "answer": text, "truth": answers_tmp[i], "type": types_tmp[i]}
                qatDictList.append(qatDict)
            if text is None:
                waste += 1
        print(f"Waste: {waste}")

        # Save the generated texts to the output file
        with open(output_path, "r", encoding='utf-8') as f:
            oriDictList = json.load(f)
        oriDictList.extend(qatDictList)
        with open(output_path, "w", encoding='utf-8') as f:
            json.dump(oriDictList, f, ensure_ascii=False, indent=4)

        # Load the output
        answers.extend([qat["answer"] for qat in qatDictList])
        truths.extend([qat["truth"] for qat in qatDictList])
        questions.extend([qat["question"] for qat in qatDictList])
        types.extend([qat["type"] for qat in qatDictList])


    # Calculate total average scores
    bleu_score_total = calculate_bleu(truths, answers, tokenizer)
    rouge_score_total = calculate_rouge(truths, answers, tokenizer)
    bert_score_total = calculate_bertscore(truths, answers, model_path=args.bert_path, device="cuda")
    meteor_score_total = calculate_meteor_chinese(truths, answers)
    llm_score = calculate_llm(questions, truths, answers, api_key)

    # Calculate the average scores by the type
    type_scores = {}
    unique_types = set(types)
    for t in unique_types:
        type_indices = [i for i, x in enumerate(types) if x == t]
        type_answers = [answers[i] for i in type_indices]
        type_truths = [truths[i] for i in type_indices]
        type_questions = [questions[i] for i in type_indices]
        bleu_score = calculate_bleu(type_truths, type_answers, tokenizer)
        rouge_score = calculate_rouge(type_truths, type_answers, tokenizer)
        bert_score = calculate_bertscore(type_truths, type_answers, model_path=args.bert_path, device="cuda")
        meteor_score = calculate_meteor_chinese(type_truths, type_answers)
        llm_score = calculate_llm(type_questions, type_truths, type_answers, api_key)

        type_scores[t] = {
            "BLEU Score": bleu_score,
            "ROUGE Score": rouge_score,
            "BERTScore": bert_score,
            "Meteor Score": meteor_score,
            "LLM Score": llm_score
        }

    
    # Print scores by type
    for t, scores in type_scores.items():
        print(f"Type: {t}")
        print(f"BLEU Score: {scores['BLEU Score']}, ROUGE Score: {scores['ROUGE Score']}, BERTScore: {scores['BERTScore']}, Meteor Score: {scores['Meteor Score']}, LLM Score: {scores['LLM Score']}")
        print("=" * 50)

    print("Overall Scores:")
    print(f"BLEU Score: {bleu_score_total}, ROUGE Score: {rouge_score_total}, BERTScore: {bert_score_total}, Meteor Score: {meteor_score_total}")

if __name__ == "__main__":
    main()
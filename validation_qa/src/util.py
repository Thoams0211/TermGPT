# /-*- coding: utf-8 -*/
import jieba
import nltk
import torch
import re
import json
from rouge_chinese import Rouge
from bert_score import score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from transformers import AutoTokenizer
from openai import OpenAI


PROMPT = """
你是一个智能评估系统，负责评估用户提供的答案与标准答案之间的相似性或距离。请根据以下输入，从五个维度对用户答案进行评分和解释。

### 任务描述
1. 比较用户答案（Candidate Answer）与标准答案（Reference Answer）的语义与内容。
2. 分别从以下五个维度打分，每个维度的分值范围为 0 到 1：
   准确性（Accuracy）:事实是否与标准答案一致，有无错误陈述。
   覆盖度（Coverage）:是否包含标准答案中的核心信息点。
   相关性（Relevance）:是否紧扣问题，避免冗余或跑题内容。
   清晰度（Clarity）:表达是否流畅、通顺，语义是否清晰。
   法律专业性（Legal Appropriateness）:是否正确使用法律术语，是否存在术语误用或不当概括。
3. 计算并给出一个综合评分（可以是五个维度的平均值）。
4. 严格按照格式输出，禁止输出其他内容。

### 输出格式
[准确性]: x.xx
[覆盖度]: x.xx
[相关性]: x.xx
[清晰度]: x.xx
[法律专业性]: x.xx
[解释]: {{对每个评分维度进行简要解释，指出优点和不足}}

### 示例
问题：被宣告死亡人生还后，通过什么方式可以依法恢复其民事权利能力和主体资格？
标准答案：被宣告死亡人生还，并由本人或利害关系人申请，法院撤销死亡宣告。
用户答案：被宣告死亡人生还后，可以通过向人民法院申请撤销原死亡宣告的方式依法恢复其民事权利能力和主体资格。

输出：
[准确性]: 1.00  
[覆盖度]: 0.85  
[相关性]: 1.00  
[清晰度]: 1.00  
[法律专业性]: 0.95  
[解释]: 用户答案与标准答案在事实表述上高度一致，准确性满分。未明确提及“本人或利害关系人申请”，因此覆盖度略有不足。回答紧扣问题，语言表达清晰，术语“撤销死亡宣告”运用得当，具备良好的法律专业性。

### 开始评估
问题：{question}
标准答案：{reference}
用户答案：{answer}

最终输出: 
"""


def calculate_bleu(references, candidates, tokenizer) -> dict:
    """Calculates average BLEU-1 and BLEU-4 scores between a list of reference answers and a list of candidate answers.

    Args:
        references (list of str): The list of reference answers.
        candidates (list of str): The list of generated answers.
        tokenizer: The tokenizer to use for tokenizing the strings.

    Returns:
        dict: A dictionary with the average BLEU-1 and BLEU-4 scores.
    """
    # Tokenize all references and candidates
    references_processed = [[tokenizer.tokenize(ref)] for ref in references]
    candidates_processed = [tokenizer.tokenize(cand) for cand in candidates]

    # Initialize the smoothing function (helps with low counts in BLEU calculation)
    smoothing_function = SmoothingFunction().method4

    # Calculate BLEU-1 and BLEU-4 scores for all pairs
    bleu_1_scores = []
    bleu_4_scores = []

    for ref_tokens, cand_tokens in zip(references_processed, candidates_processed):
        bleu_1 = sentence_bleu(ref_tokens, cand_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothing_function)  # BLEU-1
        bleu_4 = sentence_bleu(ref_tokens, cand_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing_function)  # BLEU-4
        
        bleu_1_scores.append(bleu_1)
        bleu_4_scores.append(bleu_4)

    # Compute average BLEU-1 and BLEU-4 scores
    avg_bleu_1 = sum(bleu_1_scores) / len(bleu_1_scores)
    avg_bleu_4 = sum(bleu_4_scores) / len(bleu_4_scores)

    # Return the average BLEU scores in a dictionary
    return {
        "BLEU-1": avg_bleu_1,
        "BLEU-4": avg_bleu_4
    }



def calculate_rouge(references, candidates, tokenizer, rouge_types=['rouge-1', 'rouge-l']) -> dict:
    """Calculates the ROUGE-1 and ROUGE-L F1 scores using rouge_chinese between a list of reference answers and a list of candidate answers.

    Args:
        references (list of str): The list of reference answers.
        candidates (list of str): The list of generated answers.
        tokenizer: The tokenizer to use for tokenizing the strings.
        rouge_types (list): A list of ROUGE types to calculate (default includes 'rouge-1' and 'rouge-l').

    Returns:
        dict: A dictionary containing average ROUGE-1 and ROUGE-L F1 scores.
    """
    # Tokenize all references and candidates at once
    references_processed = [" ".join(tokenizer.tokenize(ref)) for ref in references]
    candidates_processed = [" ".join(tokenizer.tokenize(cand)) for cand in candidates]

    # Initialize ROUGE evaluation using rouge_chinese
    rouge = Rouge()

    # Calculate ROUGE scores for all pairs
    scores = rouge.get_scores(candidates_processed, references_processed)

    # Initialize lists to store the ROUGE scores
    rouge_1_f1_scores = []
    rouge_l_f1_scores = []

    # Extract the desired ROUGE scores for each sentence pair
    for score in scores:
        if 'rouge-1' in rouge_types:
            rouge_1_f1_scores.append(score['rouge-1']['f'])
        if 'rouge-l' in rouge_types:
            rouge_l_f1_scores.append(score['rouge-l']['f'])

    # Calculate the average ROUGE scores
    avg_rouge_1_f1 = sum(rouge_1_f1_scores) / len(rouge_1_f1_scores) if rouge_1_f1_scores else 0
    avg_rouge_l_f1 = sum(rouge_l_f1_scores) / len(rouge_l_f1_scores) if rouge_l_f1_scores else 0

    # Return the average ROUGE scores in a dictionary
    return {
        'ROUGE-1': avg_rouge_1_f1,
        'ROUGE-L': avg_rouge_l_f1
    }



def calculate_bertscore(references, candidates, model_path=None, lang="zh", device=None) -> list:
    """Calculates the BERTScore between a list of reference answers and a list of candidate answers.

    Args:
        references (list of str): The list of reference answers.
        candidates (list of str): The list of generated answers.
        model_path (str, optional): Path to the local BERT model. If None, uses the default model.
        lang (str, optional): Language code (e.g., "en" for English). Defaults to "zh".
        device (str, optional): Device to use for computation (e.g., "cuda" or "cpu"). If None, auto-detects.

    Returns:
        list of float: A list of BERTScore F1 scores, each ranging from 0 to 1.
    """
    # Truncate inputs to 512 tokens (BERT's maximum input length)
    references = [ref[:512] for ref in references]
    candidates = [cand[:512] for cand in candidates]

    # Initialize the BERTScorer with the specified model path or default model
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Calculate BERTScore for all pairs
    P, R, F1 = score(candidates, references, model_type=model_path, lang=lang, device=device)

    # Convert F1 scores to a list of floats
    f1_scores = F1.tolist()
    avg_f1_score = sum(f1_scores) / len(f1_scores) if f1_scores else 0  # Compute average

    return avg_f1_score



def calculate_meteor_chinese(references, candidates, alpha=0.9, beta=3.0, gamma=0.5) -> dict:
    """Calculates METEOR score for Chinese text between a list of reference answers and a list of candidate answers.

    Args:
        references (list of str): The list of reference answers.
        candidates (list of str): The list of generated answers.
        alpha (float): Weight of the synonym matching (default=0.9).
        beta (float): Weight of stemming matching (default=3.0).
        gamma (float): Weight of word order matching (default=0.5).

    Returns:
        dict: A dictionary containing the average METEOR score for Chinese text.
    """
    # Tokenize all references and candidates using jieba
    # references_processed = [" ".join(jieba.cut(ref)) for ref in references]
    # candidates_processed = [" ".join(jieba.cut(cand)) for cand in candidates]
    references_processed = [list(jieba.cut(ref)) for ref in references]
    candidates_processed = [list(jieba.cut(cand)) for cand in candidates]

    # Calculate METEOR score for all pairs
    meteor_scores = []
    for ref, cand in zip(references_processed, candidates_processed):
        ref = set(ref)
        score = meteor_score([ref], cand)
        meteor_scores.append(score)

    # Calculate the average METEOR score
    avg_meteor_score = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0

    return {
        'METEOR': avg_meteor_score
    }


def calculate_llm(question, reference, candidate, apiKey) -> float:
    """Calculates the LLM score between a reference and a candidate answer.

    Args:
        question (str): The question.
        reference (str): The reference answer.
        candidate (str): The generated answer.
        apiKey (str): The API key for the model.

    Returns:
        float: The LLM score, ranging from 0 to 1.
    """

    apiKey = apiKey
    client = OpenAI(
        api_key=apiKey, 
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    content = PROMPT.format(question=question, reference=reference, answer=candidate)
    message = [
        {'role': 'system', 'content': '你是一个法考智能评估系统'},
        {'role': 'user', 'content': content}
    ]

    completion = client.chat.completions.create(
        model="qwen-plus",
        messages=message,
        temperature=0.2
    )

    responseDic = json.loads(completion.model_dump_json())
    text = responseDic['choices'][0]['message']['content']

    output = text.split("最终回答: ")[-1]

    pattern = r"\[准确性\][:：]\s([0-9.]+).*?" \
            r"\[覆盖度\]: ([0-9.]+).*?" \
            r"\[相关性\]: ([0-9.]+).*?" \
            r"\[清晰度\]: ([0-9.]+).*?" \
            r"\[法律专业性\]: ([0-9.]+).*?" \
            r"\[解释\]: (.+)"

    match = re.search(pattern, output, re.DOTALL)
    if match:
        result = {
            "准确性": float(match.group(1)),
            "覆盖度": float(match.group(2)),
            "相关性": float(match.group(3)),
            "清晰度": float(match.group(4)),
            "法律专业性": float(match.group(5)),
        }
        print(result)
    else:
        print("未匹配到数据")
        result = None

    return result




if __name__ == "__main__":
    pass


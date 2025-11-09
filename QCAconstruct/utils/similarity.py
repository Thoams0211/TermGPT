# -*- coding: utf-8 -*-
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

BATCH_SIZE = 16
THRESHOLD = 0.9

def get_sentence_embedding(sentences, tokenizer, model, device):
    """ Generate embeddings for a list of sentences using the provided tokenizer and model.
    Args:
        sentences (list): List of sentences to generate embeddings for.
        tokenizer (AutoTokenizer): Tokenizer for the model.
        model (AutoModel): Pre-trained model to generate embeddings.
        device (torch.device): Device to run the model on (CPU or GPU).
    Returns:
        dict: Dictionary mapping sentences to their embeddings.
    """

    embeddings_dict = {}
    for i in tqdm(range(0, len(sentences), BATCH_SIZE), desc="Generating Embeddings"):
        # Generate embeddings for a batch of sentences
        batch_sentences = sentences[i:i + BATCH_SIZE]
        inputs = tokenizer(batch_sentences, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)

        # Handle possible tuple returned by DataParallel
        if isinstance(outputs, (list, tuple)):
            batch_embeddings = outputs[0].last_hidden_state.mean(dim=1).cpu().numpy()
        else:
            batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

        # Update the embeddings dictionary
        for j, sentence in enumerate(batch_sentences):
            embeddings_dict[sentence] = batch_embeddings[j]
        
    return embeddings_dict


def similarityCheck(posNegPairs: list, modelPath: str) -> list:
    """This function is used to check the similarity between the positive and negative sentences.
    
    Args:
        posNegPairs (list): List of tuples containing positive and negative sentence pairs.
        modelPath (str): Path to the pre-trained model.
    
    Returns:
        list: Filtered list of tuples with similarity score above the threshold.
    """

    # Ensure CUDA is available
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    device = torch.device("cuda")

    # Get the number of GPUs
    gpuNum = torch.cuda.device_count()
    print(f"Number of GPUs: {gpuNum}")

    # Load the model
    tokenizer = AutoTokenizer.from_pretrained(modelPath, max_length=512, truncation=True)
    model = AutoModel.from_pretrained(modelPath, torch_dtype=torch.float16).to(device)

    # If multiple GPUs are available, use DataParallel to wrap the model
    if gpuNum > 1:
        model = torch.nn.DataParallel(model)

    # Combine positive and negative samples into one set to avoid duplicate embedding generation
    all_samples = list(set([sample[0] for sample in posNegPairs] + [sample[1] for sample in posNegPairs]))
    print(f"Total number of unique samples: {len(all_samples)}")

    # Generating embeddings once for all samples
    embeddings_dict = get_sentence_embedding(all_samples, tokenizer, model, device)

    # mapping positive and negative samples
    posNegDict = {}
    for posNegPair in posNegPairs:
        if posNegPair[0] not in posNegDict:
            posNegDict[posNegPair[0]] = []
        posNegDict[posNegPair[0]].append(posNegPair[1])


    # Similarity calculation & checking
    filtered_pairs = []
    for pos in tqdm(posNegDict.keys(), desc="Checking Similarity"):
        for neg in posNegDict[pos]:
            similarity = cosine_similarity([embeddings_dict[pos]], [embeddings_dict[neg]])[0][0]
            if similarity >= THRESHOLD:
                filtered_pairs.append((pos, neg))

    # Destroy the process group if distributed training was used
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

    return filtered_pairs
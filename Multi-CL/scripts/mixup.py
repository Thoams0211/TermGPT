import argparse
import json
import random
import string


def mixup(question: str, anchorSentence: str, anchorWord: str, sims: list):
    positive = anchorSentence
    negatives = []
    for sim in sims:
        negative = positive.replace(anchorWord, sim)
        negatives.append(negative)

    return {
        "question": question,
        "positive_sentence": positive,
        "negatives_sentence": negatives,
        "anchor_word": anchorWord,
        "sim_words": sims
    }


def main(rawPath: str, outputPath: str):

    assert rawPath.endswith('.json'), 'The rawPath must be a json file'
    assert outputPath.endswith('.jsonl'), 'The outputPath must be a jsonl file'

    with open(rawPath, 'r', encoding='utf-8') as file:
        dataset = json.load(file)

    # check if exist output file
    with open(outputPath, 'w') as file:
        file.write('')
        file.close

    for data in dataset:
        # parse similar words
        choiceKeys = [f'choice_{i}' for i in range(1,100)]
        
        for choiceKey in choiceKeys:
            sims = []
            if choiceKey in data.keys():
                if data[choiceKey] != data['answer']:
                    sims.append(data[choiceKey])
                else:
                    continue
            else:
                break

            # mixup anchor sentence & similar words
            resDict = mixup(
                question = data['question'],
                anchorSentence=data['rewritten_sentence'],
                anchorWord=data['answer'],
                sims=sims
            )

            with open(outputPath, 'a') as file:
                file.write(json.dumps(resDict, ensure_ascii=False) + '\n')
                file.close()

    with open(outputPath, 'r') as file:
        lines = file.readlines()
    
    random.shuffle(lines)

    with open(outputPath, 'w') as file:
        for line in lines:
            file.write(line)

    
    print(f"Mixup dataset has been saved in {outputPath}")

        


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("-r", "--rawPath", type=str, required=True, help="The path saving raw token-level dataset")
    parser.add_argument("-m", "--mixPath", type=str, required=True, help="The path saving mixup token-level dataset")
    args = parser.parse_args()

    rawPath = args.rawPath
    mixPath = args.mixPath

    main(
        rawPath=rawPath,
        outputPath=mixPath
    )  

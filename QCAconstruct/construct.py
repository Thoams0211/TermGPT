# -*- coding: utf-8 -*-
import argparse
import yaml

from utils.token_batch import tokenLevelConstruct
from utils.sentence_batch import sentenceLevelConstruct
from utils.generation import generation


def main():

    # Parsing the arguments
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("--datasetDir", type=str, required=True, help="The path of the raw dataset")
    parser.add_argument("--senMessage", type=str, required=True, help="The path saving sentence-level QCA candidates")
    parser.add_argument("--tokMessage", type=str, required=True, help="The path saving token-level QCA candidates")
    parser.add_argument("-m", "--mixclDir", type=str, required=True, help="The path saving the mixcl file")
    parser.add_argument("-r", "--resultQCA", type=str, required=True, help="The path saving sentence-level & token-level QCA")
    args = parser.parse_args()


    # load the configurations
    with open("../config.yaml", 'r') as file:
        configs = yaml.safe_load(file)

        neo4j_user = configs['neo4j_user']
        neo4j_psw = configs['neo4j_psw']
        neo4j_url = configs['neo4j_url']

        apiKey = configs['API_KEY']
        modelPath = configs['model_path']

        # Saving paths
        savingPath = {
            "fin_train": configs['fin_train'],
            "fin_test": configs['fin_test'],
            "jec_train": configs['jec_train'],
            "jec_test": configs['jec_test'],
        }

    
    # setence-level candidate set construction
    sentenceLevelConstruct(
        neo4j_url=neo4j_url,
        neo4j_user=neo4j_user,
        neo4j_psw=neo4j_psw,
        datasetDir=args.datasetDir,
        messageDir=args.senMessage,
        modelPath=modelPath,
        mode='test'
    )

    # token-level candidate set construction
    tokenLevelConstruct(
        neo4j_url=neo4j_url,
        neo4j_user=neo4j_user,
        neo4j_psw=neo4j_psw,
        datasetDir=args.datasetDir,
        messageDir=args.tokMessage,
        mixclDir=args.mixclDir,
        mode='test'
    )

    # data augmenrtation for sentence-level QCA
    generation(
        fileDir=args.senMessage,
        outputDir=savingPath['fin_test'],
        apiKey=apiKey,
        mode='sentence'
    )

    # data augmenrtation for token-level QCA
    generation(
        fileDir=args.tokMessage,
        outputDir=savingPath['fin_test'],
        apiKey=apiKey,
        mode='token'
    )




if __name__ == "__main__":
    main()
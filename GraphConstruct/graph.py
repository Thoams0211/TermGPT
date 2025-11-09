import argparse
import yaml

from utils.er import er
from utils.insertDB import insertDB


def main():
    parser = argparse.ArgumentParser(description='Entity Recognization & Graph construction')
    parser.add_argument("--rulePath", type=str, required=True, help="Rule Path")
    parser.add_argument("--schemaPath", type=str, required=True, help="schema.txt file Path")
    parser.add_argument("--erPath", type=str, required=True, help="ER result Path")
    parser.add_argument("--embedPath", type=str, required=True, help="Embedding Path")
    args = parser.parse_args()
    
    # Load custom configurations
    rulePath = args.rulePath
    schemaPath = args.schemaPath
    erPath = args.erPath
    embedPath = args.embedPath

    # Load basic configurations
    with open("../config.yaml", 'r') as file:
        configs = yaml.safe_load(file)
    neo4j_user = configs['neo4j_user']
    neo4j_psw = configs['neo4j_psw']
    neo4j_url = configs['neo4j_url']
    apiKey = configs['API_KEY']
    
    # Entity Recognition and save as JSON
    er(rulePath, schemaPath, erPath, apiKey)
    
    # Insert data into Neo4j
    insertDB(neo4j_url, neo4j_user, neo4j_psw, erPath, embedPath)
    
    

if __name__ == "__main__":
    main()
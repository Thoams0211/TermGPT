# -*- coding: utf-8 -*-
import json
import os
import pickle
from py2neo import Graph
from tqdm import tqdm


TOK_SUBQUERY = """
MATCH (n)-[r:SIM_ENTITY]-(connected_node)
WHERE id(n) = $node_id AND r.entity1 IS NOT NULL AND r.entity2 IS NOT NULL
RETURN connected_node, r.entity1, r.entity2
"""

TOK_TEMPLATE = """

    请根据提供的包含指定词语的句子以及其相似词语，按照"问题-答案"格式生成一道题。具体要求如下：

    1. 问题应当由你生成，内容应该基于提供的背景信息，并且围绕指定词语{anchor}进行提问，不可照抄背景信息。
    2. {anchor}应当为正确答案
    3. 最后仅根据生成的问题，将正确答案和问题改写为一句完整的陈述句，不得添加任何题目外的补充内容，不得包含背景信息中的额外内容
    4. 直接给我回复需要的内容即可，无需添加任何解释性文字。
    5. 禁止出现"哪些"、"下列哪些"、"以下...是/不是"等指代不明的设问表述。
    6. 必须按照指定格式回复，不要添加任何额外的内容，否则无法解析

    背景信息（其中包含指定词语）：
    {anchor_sentence}

    指定词语（正确答案）：
    {anchor}

    请按照以上要求，围绕并确保词语“{anchor}”为正确答案，基于提供的句子生成题目。必须遵照以下格式：
    <问题>: ... ...
    <正确答案>: {anchor}
    <改写句子>: ... ...

"""

BATCH_SIZE = 28000

def searchCandidate(graph:Graph, node: any, testSet: list, mode: str) -> dict:

    """This function is used to search the candidate nodes from the Neo4j database.
    
    Args:
        graph (Graph): The Neo4j database graph.
        node (any): The node of the anchor rule.
        testSet (list): The testing set.
        mode (str): The mode of the dataset.
    
    Returns:
        dict: The dictionary of the candidate nodes.
    """
    
    # The node set of the rules that can be used as the condidate rules
    candidateDict = {}
    
    # subquery
    subquery = TOK_SUBQUERY
    
    # search the node set of the rules share similar entity with anchor rule
    for subrecord in graph.run(subquery, node_id=node.identity):

        connected_node = subrecord["connected_node"]["content"]

        # Check if the connected node is in the training set or testing set
        if connected_node in testSet and mode == 'train':
            continue

        # if the node dont have any similar entity, skip the current rule
        if subrecord["r.entity1"] is None or subrecord["r.entity2"] is None:
            raise Exception(f"Node {node.identity} has no similar entity but has similar relationship")
        
        # determine the anchor and sim
        if subrecord["r.entity1"] not in node['content']:
            anchor = subrecord["r.entity2"]
            sim = subrecord["r.entity1"]
        else:
            anchor = subrecord["r.entity1"]
            sim = subrecord["r.entity2"]

        # make sure the anchor is in the node
        if subrecord["r.entity1"] not in node['content'] and subrecord["r.entity2"] not in node['content']:
            print(node["content"], node.identity)
            print(subrecord["r.entity1"], subrecord["r.entity2"])
            raise Exception(f"Node {node.identity} has no similar entity but has similar relationship")
        
        anchor = anchor.replace(" ", "").replace("\n", "")
        sim = sim.replace(" ", "").replace("\n", "")

        # add sim to the candidate set
        if anchor not in candidateDict:
            candidateDict[anchor] = set()
        candidateDict[anchor].add(sim)  
        
    return candidateDict


def tokenLevelConstruct(neo4j_url:str, neo4j_user:str, neo4j_psw:str, datasetDir:str, messageDir:str, mixclDir: str, mode: str) -> list[dict]:
    
    """This function is used to construct the token-level QCA dataset.

    Args:
        neo4j_url (str): Neo4j database URL.
        neo4j_user (str): Neo4j database username.
        neo4j_psw (str): Neo4j database password.
        datasetDir (str): The path of the raw dataset.
        messageDir (str): The path saving sentence-level QCA candidates.
        mixclDir (str): The path saving the mixcl file.
        mode (str): The mode of the dataset.

    Returns:
        list[dict]: The token-level QCA dataset.
    
    """

    # record training set & testing set
    trainingSet = []
    testingSet = []
    token2type = {}

    # check if has the dataset directory
    if not os.path.exists(messageDir):
        os.makedirs(messageDir)
    else:
        for file in os.listdir(messageDir):
            os.remove(os.path.join(messageDir, file))

    # Process each subdirectory
    for root, dirs, files in os.walk(datasetDir):
        # Process relevant JSON files
        for filename in files:
            if filename == 'train.json':
                with open(os.path.join(root, filename), 'r', encoding='utf-8') as file:
                    _data = json.load(file)
                    trainingSet.extend([_d['text'] for _d in _data])
            elif filename == 'test.json':
                with open(os.path.join(root, filename), 'r', encoding='utf-8') as file:
                    _data = json.load(file)
                    testingSet.extend([_d['text'] for _d in _data])

                    the_type = root.split('/')[-1]
                    for _d in _data:
                        token2type[_d['text']] = the_type
            else:
                continue
    
    # result set
    result = []
    
    # Connect to the Neo4j database server
    graph = Graph(neo4j_url, auth=(neo4j_user, neo4j_psw))

    # Query the database
    query = """
    MATCH (n)
    RETURN n
    """

    # iterate over the results
    records = list(graph.run(query))
    messages = []
    similarDict = {}
    id2type = {}
    idx = 0
    for record in tqdm(records, desc="Token Level Generation", leave=False):
        node = record["n"]
        rule = node.get("content")

        # check if the rule is in the training set or testing set
        if mode == 'train' and rule not in trainingSet:
            continue
        if mode == 'test' and rule not in testingSet:
            continue
        
        # search candidate nodes
        candidateSet = searchCandidate(graph, node, testingSet, mode)

        # add the candidate set to the similar dict
        for anchor, sims in candidateSet.items():
            if anchor not in similarDict:
                similarDict[anchor] = sims

        # if the candidate set is None, skip the current rule
        if candidateSet is None:
            continue
        
        # generate the QCA
        for anchor, sims in candidateSet.items():
            messages.append(TOK_TEMPLATE.format(anchor_sentence=rule, anchor=anchor))
            id2type[f'request-{idx}'] = token2type[rule]
            idx += 1

            # saving the raw data for mixCL (the next step)
            if mode == 'train':
                with open(mixclDir, 'a', encoding='utf-8') as file:
                    res = {
                        "anchor_sentence": rule,
                        "anchor_word": anchor,
                        "sims": list(sims)
                    }
                    file.write(json.dumps(res, ensure_ascii=False) + '\n')

    # save the similar dict
    with open("similarDict.pkl", 'wb') as file:
        pickle.dump(similarDict, file)

    # save the id2type dictionary
    with open(f"id2type_token.json", 'w', encoding='utf-8') as file:
        json.dump(id2type, file, ensure_ascii=False, indent=4)
        
    # create jsonl file and split the messages in batches
    print(f"Token-Level Message length: {len(messages)}")  
    print(f"Token-Level Batch number: {len(messages) // BATCH_SIZE + 1}")
    outfile = None
    
    for i in range(0, len(messages), BATCH_SIZE):
        batchMessages = messages[i:i+BATCH_SIZE]
        with open(f"{messageDir}/messages_{i // BATCH_SIZE + 1}.jsonl", 'a', encoding='utf-8') as file:
            for cnt in range(len(batchMessages)):
                prompt = batchMessages[cnt]
                singleDict = {
                        "custom_id": f"request-{cnt}", 
                        "method": "POST", 
                        "url": "/v1/chat/completions", 
                        "body": {
                            "model": "qwen-plus", 
                            "messages": [
                                {'role': 'system', 'content': '你是一个法考出题人'},
                                {'role': 'user', 'content': prompt}
                            ]
                        }
                    }
                
                file.write(json.dumps(singleDict, ensure_ascii=False) + '\n')

        
    if outfile:
        outfile.close()
            
    return result

if __name__ == "__main__":
    neo4j_url = ""
    neo4j_user = ""
    neo4j_psw = ""
    apiKey = ""
    tokenLevelConstruct(
        neo4j_url=neo4j_url,
        neo4j_user=neo4j_user,
        neo4j_psw=neo4j_psw
    )
# -*- coding: utf-8 -*-
import json
import os
from py2neo import Graph
from tqdm import tqdm
from .similarity import similarityCheck


SEN_SUBQUREY = """
MATCH (n)-[:SHARES_ENTITY]-(connected_node)
WHERE id(n) = $node_id
RETURN connected_node
"""

SEN_TEMPLATE = """

    请根据以下提供的锚样本和负样本，按照Question-choice-answer (QCA)格式生成一道选择题。要求如下：

    1. 选择题包含一个问题（Question），四个选项（Choice），以及正确答案（Answer）。
    2. 四个选项中，选项A是正确选项，来自锚样本；选项B是错误选项，来自负样本；另外两个错误选项C和D由你生成。
    3. 保证四个选项应当为完整句子，而不是单词或短语。
    4. 选项B应当是根据负样本回答的问题，但是回答错误的选项。
    5. 选项C和D应当是选项A的难负样本，且必须包含负样本（错误信息来源）中的元素，并确保其语义符合正常逻辑
    6. 请直接给出正确答案的文字内容，而不是使用ABCD等字母标记。
    7. 问题中禁止出现"哪些"、"下列哪些"、"以下...是/不是"等指代不明的设问表述；答案中禁止出现"根据...法规"、"根据...规定"等模表述；
    8. 请直接生成我需要的格式（注意不要漏掉尖括号），不要包含解释、注解等多余的信息。

    锚样本（正确信息来源）：
    {anchor}

    负样本（错误信息来源）：
    {negative}

    请你按照以下格式生成题目：
    <问题>: ... ...
    <选项A>: ... ...
    <选项B>: ... ...
    <选项C>: ... ...
    <选项D>: ... ...
    <正确答案>: ... ...

"""

BATCH_SIZE = 28000
MAX_FILE_SIZE_BYTES = 1024 * 1024 * 95


def searchCandidateSentences(graph:Graph, node: any, ruleType: str) -> set:

    """This function is used to search the candidate sentences from the Neo4j database.
    
    Args:
        graph (Graph): The Neo4j database graph.
        node (any): The node of the anchor rule.
        ruleType (str): The type of the anchor rule.
    
    Returns:
        set: The set of the candidate sentences.
    """
    
    # The node set of the rules that can be used as the condidate rules
    candidateSet = set()
    
    # subquery
    subquery = SEN_SUBQUREY
    
    # search the node set of the rules share sanme entity with anchor rule
    for subrecord in graph.run(subquery, node_id=node.identity):
        connected_node = subrecord["connected_node"]["content"]
        candidateSet.add(connected_node)
        
    # search the node set of the rules as the same type as the anchor rule
    subquery = """
    MATCH (n1:Rule {rule_type: $rule_type}) 
    WHERE id(n1) <> $node_id
    RETURN n1
    """
    for subrecord in graph.run(subquery, rule_type=ruleType, node_id=node.identity):
        similar_node = subrecord["n1"]["content"]
        candidateSet.add(similar_node)
        
    return candidateSet


def sentenceLevelConstruct(neo4j_url:str, neo4j_user:str, neo4j_psw:str, datasetDir:str, messageDir:str, modelPath: str, mode: str):
    
    """This function is used to construct the sentence-level QCA dataset.

    Args:
        neo4j_url (str): Neo4j database URL.
        neo4j_user (str): Neo4j username.
        neo4j_psw (str): Neo4j password.
        datasetDir (str): Directory to save the dataset.
        messageDir (str): Directory to save API calling messages.
        modelPath (str): Path to the model for similarity check.
        mode (str): Mode of operation ('train' or 'test').

    Returns:
        None
    
    """

    # record training set & testing set
    trainingSet = []
    testingSet = []

    # sentence to type
    sentence2type = {}

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
                        sentence2type[_d['text']] = the_type
            else:
                continue
    
    # Connect to the Neo4j database server
    graph = Graph(neo4j_url, auth=(neo4j_user, neo4j_psw))

    # Query the database
    query = """
    MATCH (n)
    RETURN n
    """

    # iterate over the results
    records = list(graph.run(query))
    posNegPairs = []
    for record in tqdm(records, desc="Sentence Level Generation", leave=False):
        node = record["n"]
        rule = node.get("content")
        ruleType = node.get("rule_type")

        # check if the rule is in the training set or testing set
        if mode == 'train' and rule not in trainingSet:
            continue
        if mode == 'test' and rule not in testingSet:
            continue
        
        # search candidate nodes
        candidateSet = searchCandidateSentences(graph, node, ruleType)

        # In traing mode, check if the candidate rules are in the testing set
        if mode == 'train':
            candidateSet = [candidate for candidate in candidateSet if candidate not in testingSet]

        # add into the positive-negative pairs and the first element is the anchor rule, the second element is the negative rule
        for candidate in candidateSet:
            posNegPairs.append((rule, candidate))

        
    # check the similarity of the positive-negative pairs
    posNegPairs = similarityCheck(
        posNegPairs=posNegPairs,
        modelPath=modelPath
    )

    # add filtered positive-negative pairs into the messages
    messages = []
    id2type = {}
    idx = 0
    for posNegPair in posNegPairs:
        messages.append(SEN_TEMPLATE.format(anchor=posNegPair[0], negative=posNegPair[1]))
        id2type[f"request-{idx}"] = sentence2type[posNegPair[0]]
        idx += 1

    # save the id2type dictionary
    with open(f"id2type_sentence.json", 'w', encoding='utf-8') as file:
        json.dump(id2type, file, ensure_ascii=False, indent=4)
    
        
    # create jsonl file and split the messages in batches
    print(f"Sentence-Level Message length: {len(messages)}")  
    print(f"Sentence-Level Batch number: {len(messages) // BATCH_SIZE + 1}")
    outfile = None
    
    # construct query set and save to jsonl file
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
            

if __name__ == "__main__":
    neo4j_url = ""
    neo4j_user = ""
    neo4j_psw = ""
    apiKey = ""
    sentenceLevelConstruct(
        neo4j_url=neo4j_url,
        neo4j_user=neo4j_user,
        neo4j_psw=neo4j_psw,
        apiKey=apiKey
    )
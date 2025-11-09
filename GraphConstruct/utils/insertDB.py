from concurrent.futures import ThreadPoolExecutor
import json
import os
import pickle
from py2neo import Graph, Node, Relationship, Subgraph
from transformers import AutoTokenizer, AutoModel
import time
import threading
import torch
from torch import nn
from tqdm import tqdm


SIM_THRESHOLD_MIN = 0.75    # the similarity threshold for two entities
SIM_THRESHOLD_MAX = 0.9     # the similarity threshold for two entities
SIM_BATCH_SIZE = 102400     # The batch size that calculating similarity between the two words
THREAD_NUM = 20             # the number of threads for parallel processing
EDGE_BATCH_SIZE = 512       # The batch size that inserting edges into Neo4J

    

def process_entity(entity: str, model: AutoModel, tokenizer: AutoTokenizer, device, results, idx):
    """The process function for parallel processing, which is used to extract the embedding of the target word.
    
    Args:
        entity (str): The target word to be embedded.
        model (AutoModel): The pre-trained embedding model.
        tokenizer (AutoTokenizer): The tokenizer for the pre-trained model.
        device: The device to run the model on (CPU or GPU).
        results (list): The list to store the embeddings.
        idx (int): The index of the entity in the list.
    
    Returns:
        None: The function modifies the results list in place.

    """   

    # Tokenize and move data to device
    tokens = tokenizer(entity, return_tensors="pt", add_special_tokens=True).to(device)
    with torch.no_grad():
        outputs = model(**tokens)
        embeddings = outputs.last_hidden_state
    results[idx] = embeddings[0, 0, :].cpu()  # detach and move to CPU

    
def process_simCalculation(quadruapleBatch:list[tuple], embedMap:dict) -> any:
    """The function to calculate the similarity of the quadruaple batch.

    Args:
        quadruapleBatch (list[tuple]): A batch of quadruaples, where each quadruaple is a tuple of (rule1, entity1, rule2, entity2).
        embedMap (dict): A dictionary mapping rules to their corresponding entity embeddings.

    Returns:
        results (list[dict]): A list of dictionaries containing the quadruaples that meet the similarity criteria.
    
    """
    
    # return None
    tensors1 = []
    tensors2 = []
    results = []
    
    # Collect tensors for the two sets of entities
    for rule1, entity1, rule2, entity2 in quadruapleBatch:
        try:
            tensors1.append(embedMap[rule1][entity1])
            tensors2.append(embedMap[rule2][entity2])
        except:
            print(f"Rule1: {rule1}")
            print(f"Entity1: {entity1}")
            print(f"Rule2: {rule2}")
            print(f"Entity2: {entity2}")
            for key in embedMap[rule1].keys():
                print(key)
            print("=" * 80)
            for key in embedMap[rule2].keys():
                print(key)
            raise ValueError("The entity is not in the embedding map.")

    # Transform the lists of tensors into a single tensor for batch processing
    tensors1 = torch.stack(tensors1).to("cuda:0")  # assuming CUDA is available
    tensors2 = torch.stack(tensors2).to("cuda:0")  
    
    # Calculate cosine similarity
    sim = nn.CosineSimilarity(dim=1)(tensors1, tensors2)
    
    # Filter the results based on the similarity threshold
    mask = ((sim > SIM_THRESHOLD_MIN) & (sim < SIM_THRESHOLD_MAX)) | (sim == 1)  # create a mask for the similarity condition
    selected_indices = torch.nonzero(mask).squeeze().tolist()  # get the indices of the selected quadruaples
    
    for idx in selected_indices:
        results.append({
            "rule1": quadruapleBatch[idx][0],  
            "entity1": quadruapleBatch[idx][1],
            "rule2": quadruapleBatch[idx][2],
            "entity2": quadruapleBatch[idx][3]
        })
    
    return results


def entityEmbeddings(entities: list, model_path: str, num_threads=4) -> list:
    """The function for parallel processing, which is used to extract the embedding of the target words.
    
    Args:
        entities (list): A list of unique entities to be embedded.
        model_path (str): The path to the pre-trained embedding model.
        num_threads (int): The number of threads to use for parallel processing.

    Returns:
        results (list): A list of embeddings for the input entities, where each embedding corresponds to the entity at the same index in the input list.
    
    """

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(device)
    
    # result saving
    results = [None] * len(entities)
    threads = []
    
    # cerate threads
    for idx, entity in enumerate(entities):
        thread = threading.Thread(
            target=process_entity, 
            args=(entity, model, tokenizer, device, results, idx)
        )
        threads.append(thread)
        thread.start()
        
        if len(threads) >= num_threads:
            for t in threads:
                t.join()  # wait for threads to finish
            threads = []
    
    # wait for all threads to finish
    for t in threads:
        t.join()
    
    return results


@torch.no_grad()
def batchEntityEmbeddings(entity_list: list, model_path: str, batch_size=64) -> dict:
    """
    Get embeddings for a list of unique entities in batch.

    Args:
        entity_list (list): A list of unique entities to be embedded.
        model_path (str): The path to the pre-trained embedding model.
        batch_size (int): The number of entities to process in each batch.
    
    Returns:
        A dictionary mapping each entity to its embedding.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True, device_map="auto")
    model.eval()

    entity_to_embedding = {}
    all_entities = list(set(entity_list))  
    
    for i in range(0, len(all_entities), batch_size):
        batch = all_entities[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]  

        for ent, emb in zip(batch, embeddings):
            entity_to_embedding[ent] = emb.cpu()

    return entity_to_embedding
    

def parseRE(ruleEntity_data: list, embedModelPath: str) -> list[dict]:
    """This function is used to parse the rule-entity JSON file and match the target word in each rule.

    Args:
        ruleEntity_data (list): A list of dictionaries, where each dictionary contains a rule and its associated entities.
        embedModelPath (str): The path to the pre-trained embedding model.

    Returns:
        results (list[dict]): A list of dictionaries, where each dictionary contains a quadruaple of (rule1, entity1, rule2, entity2) that meet the similarity criteria.
    
    """
    
    entityEmbedMap = {}     # the map of entity and its embedding
    results = []            # the result of quadruaple
    
    # load dataset
    data = ruleEntity_data

    # delete old embedding map
    if os.path.exists("utils/embedMap.pkl"):
        os.remove("utils/embedMap.pkl")
        

    # Step 1: Collect all entities from the data
    all_entities = []
    for item in data:
        all_entities.extend(item["entities"])

    # Step 2: Batch process the entity embeddings
    entity_to_embedding = batchEntityEmbeddings(all_entities, embedModelPath, batch_size=64)

    # Step 3: Create the entity embedding map
    entityEmbedMap = {}
    for item in tqdm(data, desc="Assigning entity embeddings to each rule"):
        rule = item["rule"]
        entities = item["entities"]
        entityEmbedMap[rule] = {entity: entity_to_embedding[entity] for entity in entities}
    
    # save the embedding map         
    with open("utils/embedMap.pkl", "wb") as file:
        pickle.dump(entityEmbedMap, file)
    
    # read the embedding map, when you wanna use former embedding map. you can comment the above code(182-197).
    with open("utils/embedMap.pkl", "rb") as file:
        entityEmbedMap = pickle.load(file)
             
    # iterate anchor rule-entities pairs
    quadruaples = []
    for i in tqdm(range(len(data)), desc="Parsing rule-entity pairs"):
        rule1 = data[i]["rule"]
        entities1 = data[i]["entities"]
        
        for j in range(i+1, len(data)):
            rule2 = data[j]["rule"]
            entities2 = data[j]["entities"]
            
            # iterate every entity in rule1 and rule2
            for entity1 in entities1:
                for entity2 in entities2:
                    quadruaples.append((rule1, entity1, rule2, entity2))
    
    # calculate the similarity of each quadruaple
    for i in tqdm(range(0, len(quadruaples), SIM_BATCH_SIZE), desc="Calculating the similarity"):
        batch = quadruaples[i:i+SIM_BATCH_SIZE]
        batch_results = process_simCalculation(batch, entityEmbedMap)
        results.extend(batch_results)
    
    
    return results

    
def insertDB(neo4j_url: str, neo4j_user: str, neo4j_psw: str, ruleEntity_root: str, embedPath: str) -> None:
        
    """This function is used to insert the rule-entity pairs into the Neo4j database.

    Args:
        neo4j_url (str): The URL of the Neo4j database.
        neo4j_user (str): The username for the Neo4j database.
        neo4j_psw (str): The password for the Neo4j database.
        ruleEntity_root (str): The path to the rule-entity JSON file.
        embedPath (str): The path to the pre-trained embedding model.

    Returns:
        None: The function modifies the Neo4j database in place.
    
    """
    
    # connect to the Neo4j database
    graph = Graph(neo4j_url, auth=(neo4j_user, neo4j_psw))
    print("Connected to the Neo4j database!", flush=True)
    
    # clear the database
    graph.run("MATCH (n) DETACH DELETE n")
    print("Database has been cleared!", flush=True)

    # Process each subdirectory
    rawData = []
    for root, dirs, files in os.walk(ruleEntity_root):
        # Process relevant JSON files
        for filename in files:
            if filename in ('train.json', 'test.json'):
                with open(os.path.join(root, filename), "r", encoding='utf-8') as file:
                    _data = json.load(file)
                    rawData.extend(_data)

    # filter the replicated data
    seen = set()
    data = []
    for item in rawData:
        rule = item['rule']
        if rule not in seen:
            seen.add(rule)
            data.append(item)
    
    # construct and save the rule node
    rules = []  
    for item in tqdm(data, desc="Inserting nodes into Neo4j database"):
        rule_name = item["rule"]
        rule_type = rule_name.split(";")[0].split("管理事项: ")[-1]
        rules.append(rule_name)
        rule_node = Node("Rule", content=rule_name, rule_type=rule_type)
        graph.merge(rule_node, "Rule", "content")
        
    # construct the edges
    edges = parseRE(data, embedPath)

    print(f"edge Nums: {len(edges)}")
    
    """
    
    The following code is used to insert the relationships into the Neo4j database using transaction machanism.
    The BATCH_SIZE is set to control the number of relationships inserted in each transaction.
    If the batch size is too large, the memory may be exhausted.
    
    What's more, we ensure the transaction is closed and rollback if any exception occurs.
    
    """
    
    # insert the edges
    all_rules = {record["content"]: record["n"] for record in graph.run("MATCH (n:Rule) RETURN n, n.content AS content").data()}
    def create_relationships_in_batch(tx, relationships):
        for rel in relationships:
            tx.create(rel)
    for rule in tqdm(rules, desc="Inserting edges into the Neo4j database"):
        rels = [d for d in edges if d.get('rule1') == rule]
        batch = []
        try:
            while rels:
                tx = graph.begin()
                committed = False
                for _ in range(EDGE_BATCH_SIZE):
                    if not rels:
                        break
                    rel = rels.pop(0)
                    rule1 = rel["rule1"]
                    entity1 = rel["entity1"]
                    rule2 = rel["rule2"]
                    entity2 = rel["entity2"]
                    node1 = all_rules.get(rule1)
                    node2 = all_rules.get(rule2)
                    if node1 and node2:
                        if entity1 == entity2:
                            relationship = Relationship(node1, "SHARES_ENTITY", node2, entity=entity1)
                        else:
                            relationship = Relationship(node1, "SIM_ENTITY", node2, entity1=entity1, entity2=entity2)
                        batch.append(relationship)
                if batch:
                    create_relationships_in_batch(tx, batch)
                    tx.commit()
                    committed = True
                    batch = []
            if batch:
                tx = graph.begin()
                create_relationships_in_batch(tx, batch)
                tx.commit()
                committed = True
        except Exception as e:
            # rollback the transaction if any exception occurs
            if 'tx' in locals() and not committed:
                tx.rollback()
            raise e 
        finally:
            # ensure the transaction is closed
            if 'tx' in locals() and not committed:
                tx.rollback()

    print("All relationships have been inserted.")
            

if __name__ == "__main__":
    pass
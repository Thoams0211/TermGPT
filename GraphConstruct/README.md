# Sentence Graph Construction

This module is designed to construct a sentence graph from a given text file. 
<!-- 
### Scripts
- `utils/batch_script.py`: Saving methods that call batch API of Qwen
- `utils/er.py`: Calling API to extract entities from text
- `utils/insertDB.py`: Inserting data into the graph database
- `graph.py`: The main script that orchestrates the graph construction process
- `graph_finance.sh`: A shell script to run the graph construction process on finance dataset

### Cache
- `schema`: Folder that saves the schema file for Entity Recognition process. -->

### Start
**🚨 Before running the code, you should make sure that you have deployed the Neo4j database.**

To start the graph construction process, you can run the following command:

```bash
python graph.py \
    --rulePath  <rule_path> \
    --schemaPath <schema_path> \
    --erPath <er_path> \
    --embedPath <embed_path> \
```
where 
- `rule_path`: The path to the rule file.
- `schema_path`: The path to the schema file.
- `er_path`: The path to the Entity Recognition cache folder.
- `embed_path`: The path to the embedding model.
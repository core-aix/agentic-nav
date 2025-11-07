# llm-agents


## Installation
Install Ollama and pull `nomic-embed-text` model for embedding computation used in RAG.
```commandline
ollama pull nomic-embed-text
```

To use local model, also pull 
```commandline
ollama pull gpt-oss:20b
```

The chat model can be configured to use `gpt-oss:120b-cloud`. To use it, get an Ollama API key and pass it in the command line args (see below).



## Agent for NeurIPS 2025 papers

### Setup
Download https://neurips.cc/static/virtual/data/neurips-2025-orals-posters.json and put the file in the `./data` folder.
```commandline
wget -O data/neurips-2025-orals-posters.json https://neurips.cc/static/virtual/data/neurips-2025-orals-posters.json
```

#### Regular RAG
Run `uv run build_json_rag.py` to build the vector database for RAG. 

#### GraphRAG
Setting up a GraphRAG system requires a graph database. Inside `tools/knowledge_graph`, we provide a docker-compose file 
for setting up a local neo4j database. 
Run `docker compose -f tools/knowledge_graph/docker-compose.yaml up -d` first. 





### Agent interaction
Run the following command.

For local model:
```commandline
uv run main_ui.py \
    -a http://localhost:11434 \
    -m ollama_chat/gpt-oss:20b \
    -t 0.4 \
    --max-tokens 6000 \
    -c 131072 \
    --max-num-papers 10
```

For Ollama cloud model:
```commandline
uv run main_ui.py \
    -a https://ollama.com \
    -k $OLLAMA_API_KEY \
    -m ollama_chat/gpt-oss:120b-cloud \
    -t 0.4 \
    --max-tokens 6000 \
    -c 131072 \
    --max-num-papers 10
```

The backend uses LiteLLM, which allows you to use a variety of LLM inference endpoints. 
Find details on the various providers [here](https://docs.litellm.ai/docs/providers).
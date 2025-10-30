# llm-agents


## Installation
Install Ollama and pull `nomic-embed-text` and `gpt-oss:20b` models (gpt-oss seems to work the best based on experience form manual runs)
```commandline
ollama pull nomic-embed-text
ollama pull gpt-oss:20b
```
The chat model can be configured to gpt-oss:120b-cloud. To use it, get an Ollama API key and write it into the environment variable `OLLAMA_API_KEY`.
Call `ollama pull gpt-oss:120b-cloud` instead of `ollama pull gpt-oss:20b`.



## Agent for NeurIPS 2025 papers

### Setup
Download https://neurips.cc/static/virtual/data/neurips-2025-orals-posters.json and put the file in the `./data` folder.
```commandline
wget -O data/neurips-2025-orals-posters.json https://neurips.cc/static/virtual/data/neurips-2025-orals-posters.json
```
Run `uv run build_json_rag.py` to build the vector database for RAG. 

### Agent interaction
Run the following command: 
```commandline
uv run main_ui.py \
    -a http://localhost:11434 \
    -m ollama_chat/gpt-oss:20b \
    -t 0.4 \
    --max-tokens 6000 \
    -c 131072 \
    --max-num-papers 10
```
The backend uses LiteLLM, which allows you to use a variety of LLM inference endpoints. 
Find details on the various providers [here](https://docs.litellm.ai/docs/providers).
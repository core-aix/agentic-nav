# llm-agents

Install Ollama and pull `nomic-embed-text` and `gpt-oss:20b` models (gpt-oss seems to work the best based on experience form manual runs)
```commandline
ollama pull nomic-embed-text
ollama pull gpt-oss:20b
```

### Agent for NeurIPS 2025 papers

Download https://neurips.cc/static/virtual/data/neurips-2025-orals-posters.json and put the file in the `./data` folder.
```commandline
wget -O data/neurips-2025-orals-posters.json https://neurips.cc/static/virtual/data/neurips-2025-orals-posters.json
```

Run `uv run build_json_rag.py` to build the vector database for RAG. 

Then run using `uv run main_ui.py -a http://localhost:11434 -m ollama_chat/gpt-oss:20b -t 0.4 --max-tokens 6000 -c 131072 --max-num-papers 10`. 
The api-base expects an OpenAI-compatible inference endpoint.
# llm-agents

Install Ollama and pull `nomic-embed-text` and `llama3.1` models

### Agent for NeurIPS 2025 papers

Download https://neurips.cc/static/virtual/data/neurips-2025-orals-posters.json and put the file in the `./data` folder.

Run `uv run build_json_rag.py` to build the vector database for RAG. 

Then run using `uv run main.py`.
# llm-agents

Install Ollama and pull `nomic-embed-text` model for embedding computation.

The chat model is currently configured to gpt-oss:120b-cloud. To use it, get an Ollama API key and write it into the environment variable `OLLAMA_API_KEY`.

### Agent for NeurIPS 2025 papers

Download https://neurips.cc/static/virtual/data/neurips-2025-orals-posters.json and put the file in the `./data` folder.

Run `uv run build_json_rag.py` to build the vector database for RAG. 

Then run using `uv run main_ui.py`.
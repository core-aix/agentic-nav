# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is LLMAgents, a Python package for AI research analysis agents. The system helps researchers browse papers, find similar papers, write summaries, and plan conference schedules using Neo4j knowledge graphs and LLM agents.

## Key Commands

### Environment Setup
```bash
# Install dependencies
uv sync

# Setup environment variables (required before running)
export $(grep -v '^#' .env | xargs)

# Prepare gradio from source (for Python 3.14 compatibility)
bash scripts/prepare_gradio.sh
```

### Running the Application
```bash
# CLI interface
uv run agentic-nav-cli -t 0.4 --max-tokens 6000 -c 131072 --max-num-papers 10

# Web interface
agentic-nav-web
```

### Database and Knowledge Graph
```bash
# Start required services
docker compose up neo4j_db ollama_embed ollama_agent -d

# Build knowledge graph from NeurIPS 2025 data
uv run agentic_nav/tools/knowledge_graph/graph_generator.py \
    --input-json-file data/neurips-2025-orals-posters.json \
    --embedding-model $EMBEDDING_MODEL_NAME \
    --ollama-server-url $EMBEDDING_MODEL_API_BASE \
    --embedding-gen-batch-size 32 \
    --max-parallel-workers 28 \
    --similarity-threshold 0.8 \
    --output-file graphs/knowledge_graph.pkl

# Import knowledge graph to Neo4j
uv run agentic_nav/tools/knowledge_graph/neo4j_db_importer.py \
    --graph-path graphs/knowledge_graph.pkl \
    --neo4j-uri bolt://localhost:7687 \
    --batch-size 100 \
    --embedding-dimension 768
```

### Testing
```bash
# Run all tests (recommended - avoids gradio conflicts)
uv run pytest tests/

# Run tests with coverage report
uv run pytest tests/ --cov=agentic_nav --cov-report=term-missing

# Alternative: Use the custom test runner
python run_tests.py

# Run specific test categories
uv run pytest tests/ -m unit          # Unit tests only
uv run pytest tests/ -m integration   # Integration tests only
uv run pytest tests/ -m "not slow"    # Skip slow tests

# Run tests for specific module
uv run pytest tests/agents/
uv run pytest tests/tools/
uv run pytest tests/utils/
uv run pytest tests/frontend/

# Run single test file
uv run pytest tests/agents/test_base.py

# Run with verbose output
uv run pytest tests/ -v

# Generate HTML coverage report
uv run pytest tests/ --cov=agentic_nav --cov-report=html
# View coverage report at htmlcov/index.html

# Note: Always specify tests/ directory to avoid conflicts with gradio workspace
```

### Development
```bash
# Run full system with Docker
docker compose up --build -d

# Import pre-generated NeurIPS 2025 knowledge graph
bash scripts/import_neurips2025_kg.sh
```

## Architecture

### Core Components

1. **Agent System (`agentic_nav/agents/`)**
   - `base.py`: Core LLMAgent class with streaming support and tool execution
   - `neurips2025_conference.py`: Specialized agent for NeurIPS 2025 conference data
   - Uses LiteLLM for model abstraction, supports Ollama models

2. **Tools System (`agentic_nav/tools/`)**
   - Knowledge graph tools: `search_similar_papers`, `find_neighboring_papers`, `traverse_graph`
   - Graph traversal strategies: breadth-first, depth-first, neo4j builtin
   - Tool registry automatically discovers callable functions

3. **Frontend (`agentic_nav/frontend/`)**
   - `cli.py`: Rich terminal interface with streaming, command history, auto-completion
   - `browser_ui.py`: Gradio web interface for browser-based interactions
   - Both interfaces support the same agent functionality

4. **Knowledge Graph (`agentic_nav/tools/knowledge_graph/`)**
   - Neo4j-based paper similarity and relationship storage
   - Embedding-based vector search for paper discovery
   - Support for graph traversal algorithms

### Key Data Flow

1. User input → Frontend (CLI/Web)
2. Frontend → Agent (stateless interaction with streaming)
3. Agent → LLM (via LiteLLM) + Tools (knowledge graph queries)
4. Tools → Neo4j database for paper retrieval
5. Results streamed back to user with live markdown rendering

## Configuration

### Required Environment Variables
```bash
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=<password>
EMBEDDING_MODEL_NAME=nomic-embed-text
EMBEDDING_MODEL_API_BASE=http://localhost:11435
AGENT_MODEL_NAME=gpt-oss:20b
AGENT_MODEL_API_BASE=http://localhost:11436
OLLAMA_API_KEY=<optional>
POPULATE_DATABASE_NIPS2025=false
AGENTIC_NAV_LOG_LEVEL=INFO
```

### Model Support
- Primary: Ollama models (local and remote)
- Remote Ollama models via https://ollama.com with API key
- Uses LiteLLM for provider abstraction

## Dependencies

- **Python**: 3.14+ required
- **uv**: For dependency management
- **Neo4j**: Graph database for knowledge storage
- **Ollama**: LLM inference (supports GPU acceleration with Nvidia Container Toolkit)
- **Gradio**: Built from source for Python 3.14 compatibility

## Development Notes

- The system is designed for multi-user sessions via stateless agent interactions
- Streaming responses are supported in both CLI and web interfaces
- Tool calls are automatically executed and results fed back to the LLM
- Chat history can be saved/loaded in JSON format
- Logging is configured per environment with structured output to `logs/` directory
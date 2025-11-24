---
title: AgenticNav
emoji: 📚
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.50.0
app_file: agentic_nav/frontend/browser_ui.py
python_version: 3.13
pinned: true
hf_oauth: true
hf_oauth_scopes:
  - inference-api
license: apache-2.0
short_description: Agent for NeurIPS paper discovery and visit schedule builder
---

# AgenticNAV - Your AI conference companion

[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
![Coverage](https://github.com/core-aix/agentic-nav/workflows/Tests/badge.svg)

This repository contains code for an agent that can help you do related work for your next research project. 
Given the sheer amount of new publications that are being published at major machine learning conferences, this agent
can help browse papers, find similar papers, and help you write summaries to quickly get an overview of what is currently
going on. 

The agent can also support you in planning your next conference trip by providing a schedule around one or more topics 
that you are interested in.

## Installation & usage of the web-based interface
The agent is conveniently packaged as a docker image. You can spin up the entire system by using the commands below. 
Make sure to have the [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installation) installed.
At the moment we only support `ollama` models.

Instead of a local agent model, you can also make use of remote ollama models. A full list is available here: 
https://docs.ollama.com/cloud.
To make use of these large models, set `AGENT_MODEL_NAME=<your model of choice>` and 
`AGENT_MODEL_API_BASE=https://ollama.com`. 
Don't forget to set your `OLLAMA_API_KEY` either directly via the environment or in the browser.

**Important note:** The ollama docker containers cannot use GPU acceleration on MacOS. If you want to use your Mac's GPU, 
you need to run ollama without containerization (i.e., you need to manually spin up the ollama server). 
With `NEO4J_DB_NODE_RETURN_LIMIT`, we set a strict return limit of 200 nodes per query to avoid overstraining the database.
You can set it as needed for your use case.

```commandline
# Database config 
echo "NEO4J_USERNAME=neo4j" >> .env
echo "NEO4J_PASSWORD=<a password of your choice>" >> .env
echo "NEO4J_DB_URI=bolt://neo4j_db:7687" >> .env
echo "NEO4J_DB_NODE_RETURN_LIMIT=200" >> .env

echo "EMBEDDING_MODEL_NAME=nomic-embed-text" >> .env
echo "EMBEDDING_MODEL_API_BASE=http://ollama_agent:11434" >> .env

echo "AGENT_MODEL_NAME=gpt-oss:20b" >> .env
echo "AGENT_MODEL_API_BASE=http://ollama_agent:11434" >> .env

# Optional: set your OLLAMA_API_KEY when using remote models
echo "OLLAMA_API_KEY=<your key here>" >> .env

# Set the following to true if you would like to import our pre-generated knowledge graph for the NeurIPS 2025 conference
# Warning (!): Setting the parameter below to 'true' will clear any existing data inside the docker-based neo4j database
echo "POPULATE_DATABASE_NIPS2025=false" >> .env

git clone https://github.com/core-aix/agentic-nav.git
cd agentic-nav 
docker compose up --build -d
```

This will launch the agent and its web interface, available via `http://localhost:7860`, along with a neo4j database 
(community edition). 
**It will also populate the database with all accepted papers of the NeurIPS 2025 machine learning conference (if you set `POPULATE_DATABASE_NIPS2025=true`).** 
We include pair-wise similarity scores to enable graph traversals and the search for broadly related papers.

After the docker containers are up and running, you can interact with the agent. Have fun!


## Development & contributing to the agent
If you are interested in understanding the system in detail, you may want to run all setup steps manually and avoid a 
containerized runtime. Run the following steps to setup a development environment. 

We use [uv](https://docs.astral.sh/uv/) for dependency management.
Our docker containers for serving LLMs use Ollama and GPU acceleration. For that, you need the [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installation)
Make sure to have both installed before you proceed. 

### Installation
After you cloned the repository, you need to setup the database. We use neo4j to manage the knowledge graph data we need 
for the agent to work properly.
**Note:** We are using gradio built from source as the latest release (as of Nov. 12, 2025) does not yet support python 3.14.

First, export all necessary environment variables: 
```commandline
echo "NEO4J_USERNAME=neo4j" >> .env
echo "NEO4J_PASSWORD=<a password of your choice>" >> .env
echo "NEO4J_DB_URI=bolt://localhost:7687" >> .env
echo "NEO4J_DB_NODE_RETURN_LIMIT=200" >> .env

echo "EMBEDDING_MODEL_NAME=ollama/nomic-embed-text" >> .env
echo "EMBEDDING_MODEL_API_BASE=http://localhost:11435" >> .env

echo "AGENT_MODEL_NAME=ollama_chat/gpt-oss:20b" >> .env
echo "AGENT_MODEL_API_BASE=http://localhost:11436" >> .env

# Optional: set your OLLAMA_API_KEY when using remote models
echo "OLLAMA_API_KEY=<your key here>" >> .env

# Set the following to true if you would like to import our pre-generated knowledge graph for the NeurIPS 2025 conference
# Warning (!): Setting the parameter below to 'true' will clear any existing data inside the docker-based neo4j database
echo "POPULATE_DATABASE_NIPS2025=false" >> .env

# Make sure you also have those values in your commandline environment
export $(grep -v '^#' .env | xargs)
```

Then get the project files:
```commandline
git clone https://github.com/core-aix/agentic-nav.git
cd agentic-nav
docker compose up neo4j_db ollama_embed ollama_agent -d

# The following command is only needed if you'd like to use the gradio-based GUI
# This will eventually go away once gradio bumps their release version to support python 3.14
bash scripts/prepare_gradio.sh

uv sync 
```

### Building the NeurIPS 2025 knowledge graph locally
You can also build the knowledge graph yourself and, for example, swap the embedding model we use by default.
Follow the steps below to do so. Note, that you still need to setup the project as described in the `Installation` 
subsection above. Make sure to set `POPULATE_DATABASE_NIPS2025=false` in your .env file.

#### Get the data 
Download https://neurips.cc/static/virtual/data/neurips-2025-orals-posters.json and put the file in the `./data` folder.
```commandline
wget -O data/neurips-2025-orals-posters.json https://neurips.cc/static/virtual/data/neurips-2025-orals-posters.json
```

#### Build the knowledge graph
You can build the knowledge graph per your needs by running the following script: 
```commandline
uv run llm_agents/tools/knowledge_graph/graph_generator.py \
    --input-json-file data/neurips-2025-orals-posters.json \
    --embedding-model $EMBEDDING_MODEL_NAME \
    --ollama-server-url $EMBEDDING_MODEL_API_BASE \
    --embedding-gen-batch-size 32 \
    --max-parallel-workers 28 \
    --similarity-threshold 0.6 \
    --output-file graphs/knowledge_graph.pkl \
    # --limit-num-papers  # Optional 
```
**Important note:** Generating the full graph for over 6k papers can take more than 1 hour. You can find a set of pre-generated 
knowledge graphs here (the "thresh" in the file name indicates the `similarity-threshold` for which we create a `similar_to` relationship between papers): [LRZ Sync+Share](https://syncandshare.lrz.de/getlink/fiFMhMLLH7FaQ3Jipqqsye/)


#### Importing the knowledge graph to a neo4j database
We provide an importer to move the knowledge graph into a graph database that supports vector-based similarity search. 
```commandline
uv run llm_agents/tools/knowledge_graph/neo4j_db_importer.py \
    --graph-path graphs/knowledge_graph.pkl \
    --neo4j-uri $NEO4J_DB_URI \
    --batch-size 100 \
    --embedding-dimension 768 # This must match the vector dims generated by the embedding model.
```
**Note:** Depending on what your graph looks like this can also take a while (> 20min for 6K papers). Also, beware that
running this script will first clear any existing entries before the new graph is written to the database.


### Agent interactions
We offer two ways of interacting with agents, via the command line and via the browser.
The backend uses LiteLLM, which allows you to use a variety of LLM inference endpoints. 
Find details on the various providers [here](https://docs.litellm.ai/docs/providers).

#### Commandline interface
The agent can also be used via a versatile CLI.
Below are two examples how to run a local and a remote model. 
We are using LiteLLM to abstract away from individual inference API providers.
Note, that we currently only test with Ollama models.
```commandline
uv run agentic-nav-cli \
    -t 0.4 \
    --max-tokens 6000 \
    -c 131072 \
    --max-num-papers 10
```

#### Web-based interface (beginner friendly)
We use gradio to provide a chat interface with the same functionalities as the commandline-based interface. 
You can launch the web app by running: 
```commandline
agentic-nav-web
```
All the hyperparameters you need to set can be configured in the web interface and will be used in you individual session.
Once you close the browser window, your session will terminate and all custom configuration will be removed.
At the moment, the web UI only supports Ollama models.

### Debugging agent interactions
The agent involves a set of asynchronous operations. We provide a built-in logging instance to capture all relevant logs 
for debugging. To set the right debugging level for your application, you can use the environment variable `AGENTIC_NAV_LOG_LEVEL`. 
By default, it is set to `INFO`.

#### Running tests
We try to cover all tools and agent functionalities in thorough unit tests. 
You can run them via: 
```commandline
uv run pytest tests/
```

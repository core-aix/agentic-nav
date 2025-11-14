#!/bin/bash

# Download the pre-built knowledge graph
wget -O graphs/neurips2025_knowledge_graph.pkl https://syncandshare.lrz.de/dl/fiERzDh4Kze641JA1Dk8Kv/knowledge_graph_thresh_0.6.pkl

# Import the knowledge graph to the database
uv run llm_agents/tools/knowledge_graph/neo4j_db_importer.py \
    --graph-path graphs/neurips2025_knowledge_graph.pkl \
    --neo4j-uri bolt://neo4j_db:7687 \
    --neo4j-username $NEO4J_USERNAME \
    --neo4j-password $NEO4J_PASSWORD \
    --batch-size 100 \
    --embedding-dimension 768

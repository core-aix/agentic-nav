"""
TBD
"""
import networkx as nx
import json

from litellm import completion

try:
    from .schema import DynamicSchemaGenerator
except ImportError:
    from schema import DynamicSchemaGenerator

from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple


class HybridKGBuilder:
    def __init__(
        self,
        paper: Dict[str, List],
        rag_system=None,
        llm_name: str = "ollama_chat/gpt-oss:20b",
        api_base: str = "http://localhost:11434",
        llm_default_params={}
    ):
        print(f"Found {len(paper["results"])} papers.")
        results = paper["results"]

        self.papers = {p['uid']: p for p in results}
        self.rag = rag_system
        self.graph = nx.MultiDiGraph()
        self.schema_generator = DynamicSchemaGenerator(
            llm_name=llm_name,
            api_base=api_base,
            llm_default_params=llm_default_params
        )

    def build_kg_for_query(
            self,
            user_query: str,
            paper_uids: Optional[List[str]] = None,
            max_papers: int = 100
    ) -> Tuple[nx.MultiDiGraph, Dict]:
        """Main method to build knowledge graph for a user query"""

        # Step 1: Generate schema
        print("Generating schema...")
        sample_papers = self._get_sample_papers(paper_uids, n=5)
        schema = self.schema_generator.generate_schema(user_query, sample_papers)
        print(f"Schema generated: {json.dumps(schema, indent=2)}")

        # Step 2: Filter relevant papers
        print("Filtering papers...")
        if paper_uids is None:
            paper_uids = self._filter_papers_for_query(user_query, schema, max_papers)
        print(f"Processing {len(paper_uids)} papers")

        # Step 3: Build graph
        print("Building knowledge graph...")
        for i, uid in enumerate(paper_uids):
            if i % 10 == 0:
                print(f"  Processed {i}/{len(paper_uids)} papers")

            paper = self.papers.get(uid)
            if not paper:
                continue

            # Extract metadata entities (fast)
            self._extract_metadata_entities(paper, schema)

            # Extract text-based entities (slower)
            if schema.get('extracted_entities'):
                self._extract_text_entities(paper, schema)

            # Create relationships
            self._extract_relationships(paper, schema)

        print(f"Graph built: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        return self.graph, schema

    def _get_sample_papers(self, paper_uids: Optional[List[str]], n: int = 5) -> List[Dict]:
        """Get sample papers for schema generation"""
        if paper_uids:
            sample_uids = paper_uids[:n]
        else:
            sample_uids = list(self.papers.keys())[:n]

        return [self.papers[uid] for uid in sample_uids if uid in self.papers]

    def _filter_papers_for_query(
            self,
            user_query: str,
            schema: Dict,
            max_papers: int
    ) -> List[str]:
        """Filter papers based on query and schema"""

        scored_papers = []
        filters = schema.get('filters', {})

        for uid, paper in self.papers.items():
            score = 0

            # Topic matching
            if filters.get('topic_keywords'):
                topic = paper.get('topic', '').lower() if paper.get('topic', '') is not None else ""

                for keyword in filters['topic_keywords']:
                    if keyword.lower() in topic:
                        score += 2

            # Abstract keyword matching
            if filters.get('abstract_keywords'):
                abstract = paper.get('abstract', '').lower() if paper.get('abstract', '') is not None else ""
                for keyword in filters['abstract_keywords']:
                    if keyword.lower() in abstract:
                        score += 1

            # Basic query matching
            query_lower = user_query.lower()
            abstract_lower = paper.get('abstract', '').lower() if paper.get('abstract', '') is not None else ""

            # Simple keyword overlap
            query_words = set(query_lower.split())
            abstract_words = set(abstract_lower.split())
            overlap = len(query_words & abstract_words)
            score += overlap * 0.1

            if score > 0.5:  # Threshold
                scored_papers.append((uid, score))

        # Sort by score and return top papers
        scored_papers.sort(key=lambda x: x[1], reverse=True)
        return [uid for uid, _ in scored_papers[:max_papers]]

    def _extract_metadata_entities(self, paper: Dict, schema: Dict):
        """Extract entities directly from structured metadata"""

        paper_node = paper['uid']
        self.graph.add_node(
            paper_node,
            type='Paper',
            name=paper.get('name', ''),
            abstract=paper.get('abstract', '')[:500],  # Truncate for storage
            topic=paper.get('topic'),
            session=paper.get('session')
        )

        # Authors
        if 'Author' in schema.get('metadata_entities', {}):
            for author in paper.get('authors', []):
                author_id = f"author_{author['id']}"

                if not self.graph.has_node(author_id):
                    self.graph.add_node(
                        author_id,
                        type='Author',
                        fullname=author.get('fullname', ''),
                        institution=author.get('institution', '')
                    )

                self.graph.add_edge(
                    paper_node,
                    author_id,
                    relation='authored_by',
                    confidence=1.0
                )

        # Topics
        if 'Topic' in schema.get('metadata_entities', {}) and paper.get('topic'):
            topic_node = f"topic_{paper['topic'].replace(' ', '_')}"

            if not self.graph.has_node(topic_node):
                self.graph.add_node(
                    topic_node,
                    type='Topic',
                    name=paper['topic']
                )

            self.graph.add_edge(
                paper_node,
                topic_node,
                relation='belongs_to_topic',
                confidence=1.0
            )

        # Sessions
        if 'Session' in schema.get('metadata_entities', {}) and paper.get('session'):
            session_node = f"session_{paper['session'].replace(' ', '_')}"

            if not self.graph.has_node(session_node):
                self.graph.add_node(
                    session_node,
                    type='Session',
                    name=paper['session'],
                    room=paper.get('room_name', '')
                )

            self.graph.add_edge(
                paper_node,
                session_node,
                relation='presented_at',
                confidence=1.0
            )

    def _extract_text_entities(self, paper: Dict, schema: Dict):
        """Extract entities from paper text using LLM"""

        extracted_entities = schema.get('extracted_entities', {})
        if not extracted_entities:
            return

        abstract = paper.get('abstract', '')
        if not abstract:
            return

        # Build extraction prompt
        entity_descriptions = []
        for entity_type, entity_def in extracted_entities.items():
            entity_descriptions.append(
                f"- {entity_type}: {entity_def.get('description', '')}"
            )

        prompt = f"""Extract entities from this paper abstract.
        
Entity types to extract:
{chr(10).join(entity_descriptions)}

Abstract:
{abstract}

Return ONLY valid JSON array (no markdown):
[
  {{
    "type": "EntityType",
    "name": "entity name",
    "attributes": {{}},
    "confidence": 0.9
  }}
]

Be selective - only extract clearly mentioned entities."""

        try:
            entities = self.schema_generator._call_llm(prompt, response_format="json")

            # Handle if wrapped in object
            if isinstance(entities, dict) and 'entities' in entities:
                entities = entities['entities']

            for entity in entities:
                entity_type = entity.get('type', 'Unknown')
                entity_name = entity.get('name', '').strip()

                if not entity_name:
                    continue

                # Create unique node ID
                entity_id = f"{entity_type.lower()}_{entity_name.replace(' ', '_')}"

                if not self.graph.has_node(entity_id):
                    self.graph.add_node(
                        entity_id,
                        type=entity_type,
                        name=entity_name,
                        **entity.get('attributes', {})
                    )

                self.graph.add_edge(
                    paper['uid'],
                    entity_id,
                    relation='mentions',
                    confidence=entity.get('confidence', 0.7)
                )

        except Exception as e:
            print(f"Error extracting entities from paper {paper['uid']}: {e}")

    def _extract_relationships(self, paper: Dict, schema: Dict):
        """Extract relationships based on schema"""

        relationships = schema.get('relationships', {})

        for rel_name, rel_def in relationships.items():
            extraction_type = rel_def.get('extraction', '')

            # Skip metadata relationships (already handled)
            if 'metadata' in extraction_type:
                continue

            # Derived relationships (e.g., co-authorship)
            if 'derived' in extraction_type or rel_name == 'co_authors':
                self._create_derived_relationships(paper, rel_name, rel_def)

            # LLM-based relationship extraction
            elif 'LLM' in extraction_type or 'abstract' in extraction_type:
                self._llm_extract_relations(paper, rel_name, rel_def)

    def _create_derived_relationships(self, paper: Dict, rel_name: str, rel_def: Dict):
        """Create derived relationships like co-authorship"""

        if rel_name == 'co_authors' or (
            rel_def.get('source') == 'Author' and rel_def.get('target') == 'Author'
        ):
            authors = [f"author_{a['id']}" for a in paper.get('authors', [])]

            for i, auth1 in enumerate(authors):
                for auth2 in authors[i+1:]:
                    if self.graph.has_node(auth1) and self.graph.has_node(auth2):
                        self.graph.add_edge(
                            auth1,
                            auth2,
                            relation='co_authors',
                            paper_uid=paper['uid'],
                            confidence=1.0
                        )

    def _llm_extract_relations(self, paper: Dict, rel_name: str, rel_def: Dict):
        """Extract relationships from text using LLM"""

        abstract = paper.get('abstract', '')
        if not abstract:
            return

        # Get existing entities from this paper
        paper_entities = []
        for node in self.graph.successors(paper['uid']):
            node_data = self.graph.nodes[node]
            paper_entities.append({
                'id': node,
                'type': node_data.get('type'),
                'name': node_data.get('name', node)
            })

        if len(paper_entities) < 2:
            return  # Need at least 2 entities for relationships

        prompt = f"""Find relationships in this abstract.

Relationship: {rel_name}
Description: {rel_def.get('description', '')}
Source type: {rel_def.get('source')}
Target type: {rel_def.get('target')}
Example phrases: {rel_def.get('examples', [])}

Known entities:
{json.dumps(paper_entities, indent=2)}

Abstract:
{abstract}

Return ONLY valid JSON array (no markdown):
[
  {{
    "source_id": "entity_id",
    "target_id": "entity_id",
    "evidence": "text snippet",
    "confidence": 0.8
  }}
]

Only extract clearly stated or strongly implied relationships."""

        try:
            relations = self.schema_generator._call_llm(prompt, response_format="json")

            # Handle if wrapped in object
            if isinstance(relations, dict) and 'relations' in relations:
                relations = relations['relations']

            for rel in relations:
                source = rel.get('source_id')
                target = rel.get('target_id')

                if source and target and self.graph.has_node(source) and self.graph.has_node(target):
                    self.graph.add_edge(
                        source,
                        target,
                        relation=rel_name,
                        evidence=rel.get('evidence', ''),
                        confidence=rel.get('confidence', 0.7)
                    )

        except Exception as e:
            print(f"Error extracting relations from paper {paper['uid']}: {e}")

    def query_graph(self, query: str, schema: Dict) -> Dict:
        """Query the built knowledge graph"""

        results = {
            'nodes': self.graph.number_of_nodes(),
            'edges': self.graph.number_of_edges(),
            'node_types': defaultdict(int),
            'relation_types': defaultdict(int)
        }

        # Count node types
        for node, data in self.graph.nodes(data=True):
            node_type = data.get('type', 'Unknown')
            results['node_types'][node_type] += 1

        # Count relation types
        for _, _, data in self.graph.edges(data=True):
            rel_type = data.get('relation', 'Unknown')
            results['relation_types'][rel_type] += 1

        return results

    def export_graph(self, filepath: str, format: str = 'gexf'):
        """Export graph to file"""
        if format == 'gexf':
            nx.write_gexf(self.graph, filepath)
        elif format == 'json':
            data = nx.node_link_data(self.graph)
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        elif format == 'graphml':
            nx.write_graphml(self.graph, filepath)



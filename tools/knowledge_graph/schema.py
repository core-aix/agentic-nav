"""
TBD
"""
import json

from litellm import completion
from pathlib import Path

from typing import List, Dict


class DynamicSchemaGenerator:
    def __init__(self, llm_name: str = "ollama_chat/gpt-oss:20b", llm_default_params: dict = {}, api_base: str = "http://localhost:11434"):
        self.llm = llm_name
        self.api_base = api_base
        self.llm_default_params: dict = llm_default_params

    def generate_schema(self, user_query: str, sample_papers: List[Dict]) -> Dict:
        """Generate extraction schema based on user query and sample papers"""

        metadata_summary = self._analyze_metadata(sample_papers)
        sample_abstracts = self._format_sample_abstracts(sample_papers[:3])

        prompt = f"""Given this user query: "{user_query}"

Available paper metadata includes:
- Authors: name, institution, unique IDs
- Topics: {metadata_summary['topics']}
- Has keywords: {metadata_summary['has_keywords']}
- Session/Event information
- Abstract text

Sample abstracts:
{sample_abstracts}

Generate a knowledge graph schema that:
1. Leverages structured metadata when possible (more reliable)
2. Extracts additional entities from abstracts when needed
3. Creates relationships between metadata and extracted entities

Return ONLY valid JSON (no markdown, no explanations) in this exact format:
{{
    "query_intent": "brief analysis of user's goal",
    "metadata_entities": {{
        "Author": {{
            "source": "authors field",
            "attributes": ["fullname", "institution"],
            "use_for": "collaboration networks"
        }},
        "Topic": {{
            "source": "topic field",
            "attributes": ["category"],
            "use_for": "categorization"
        }}
    }},
    "extracted_entities": {{
        "Method": {{
            "source": "abstract text",
            "description": "machine learning methods or algorithms",
            "attributes": ["name", "type"],
            "extraction_instruction": "look for specific algorithm names"
        }}
    }},
    "relationships": {{
        "authored_by": {{
            "source": "Paper",
            "target": "Author",
            "extraction": "direct from metadata"
        }},
        "uses_method": {{
            "source": "Paper",
            "target": "Method",
            "extraction": "LLM from abstract",
            "examples": ["proposes", "uses", "implements"]
        }}
    }},
    "filters": {{
        "topic_keywords": ["relevant", "keywords"],
        "abstract_keywords": ["key", "terms"]
    }}
}}"""

        schema = self._call_llm(prompt, response_format="json")
        return self._validate_schema(schema)

    @staticmethod
    def _analyze_metadata(papers: List[Dict]) -> Dict:
        """Analyze available metadata across papers"""
        topics = list(set(p.get("topic", "") for p in papers if p.get("topic")))
        keywords_exist = any(p.get("keywords") for p in papers)

        return {
            "topics": topics[:10],  # Sample of topics
            "has_keywords": keywords_exist,
            "has_authors": all("authors" in p for p in papers),
            "has_sessions": all("session" in p for p in papers),
            "num_papers": len(papers)
        }

    @staticmethod
    def _format_sample_abstracts(papers: List[Dict]) -> str:
        """Format sample abstracts for prompt"""
        formatted = []
        for i, paper in enumerate(papers, 1):
            abstract = paper.get('abstract', '')[:300]  # Truncate
            formatted.append(f"Paper {i}: {abstract}...")
        return "\n\n".join(formatted)


    def _call_llm(self, prompt: str, response_format: str = "json_object") -> Dict:
        """Call LLM API - adapt this to your LLM provider"""
        response = completion(
            model=self.llm,
            messages=[
                {"role": "system", "content": "You are an expert is extracting schemas for dynamic knowledge graphs."},
                {"role": "user", "content": prompt}
            ],
            api_base=self.api_base,
            stream=False,
            response_format={"type": response_format},
            **self.llm_default_params
        )

        schema_str = response.choices[0].message.content
        return json.loads(schema_str)

    @staticmethod
    def _validate_schema(schema: Dict) -> Dict:
        """Validate and fix schema structure"""
        required_keys = ["query_intent", "metadata_entities", "extracted_entities", "relationships"]
        for key in required_keys:
            if key not in schema:
                schema[key] = {}

        if "filters" not in schema:
            schema["filters"] = {}

        return schema


if __name__ == "__main__":
    project_root_path = Path(__file__).parent.parent.parent

    with open(f"{project_root_path}/data/neurips-2025-orals-posters.json", "r") as f:
        papers = json.load(fp=f)
        f.close()

    sample_papers = papers["results"][:3]
    gen = DynamicSchemaGenerator()
    schema = gen.generate_schema(
        user_query="What deep learning methods are used for dimension reduction?",
        sample_papers=sample_papers
    )

    print(schema)

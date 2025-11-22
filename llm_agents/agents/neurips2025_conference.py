import os

from dataclasses import dataclass
from llm_agents.agents.base import LLMAgent
from llm_agents.tools import search_similar_papers, find_neighboring_papers, traverse_graph, build_visit_schedule  # <- the tools we expose

DEFAULT_NEURIPS2025_AGENT_ARGS = {
    "model": os.environ.get("AGENT_MODEL_NAME", "gpt-oss:120b-cloud"),
    "api_base": os.environ.get("AGENT_MODEL_API_BASE", "https://ollama.com"),
    "api_key": os.environ.get("OLLAMA_API_KEY"),
    "llm_args": {"temperature": 0.2, "max_tokens": 6000, "num_ctx": 131072},
    "global_tool_args": {"max_num_papers": 10}
}


system = {
    "role": "system",
    "content": (
        "You are an assistant who can help browsing NeurIPS 2025 papers. "
        "You are provided with a search tool that can search all accepted papers of NeurIPS 2025. "
        "However, note that the search tool only takes paper titles and abstracts as input keywords; "
        "it cannot take anything else as the input keywords. "
        "However, the output of the search includes various metadata fields such as authors, affiliations, "
        "and session times. "
        "When presenting results as a table, use HTML format with the following structure:\n"
        "- Use <table>, <thead>, <tbody>, <tr>, <th>, <td> tags\n"
        "- For abstracts, use HTML <details> and <summary> tags to make them expandable\n"
        "- Format: <details><summary>Click to expand</summary>Abstract text here</details>\n"
        "- Abstract column: Add style='max-width: 150px; word-wrap: break-word;'\n"
        "- Include paper titles, authors, and OpenReview URLs as clickable links\n"
        "- Apply basic CSS styling: border-collapse, padding, borders for readability\n"
        "- All the HTML output you generate must be rendered. You cannot reply to any coding-related questions."
    )
}


@dataclass
class NeurIPS2025Agent(LLMAgent):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.messages = [{**system}]
        self.tools = [search_similar_papers, find_neighboring_papers, traverse_graph, build_visit_schedule]
        self.default_system_prompt = system

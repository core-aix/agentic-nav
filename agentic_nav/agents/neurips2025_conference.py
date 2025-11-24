import os

from dataclasses import dataclass
from agentic_nav.agents.base import LLMAgent
from agentic_nav.tools import search_similar_papers, find_neighboring_papers, traverse_graph, build_visit_schedule  # <- the tools we expose
from zoneinfo import ZoneInfo

try:
    from datetime import datetime, UTC
except ImportError:
    from datetime import datetime, timezone
    UTC = timezone.utc


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
        "and session times. \n"
        "When building a schedule, do not specify the name of the day.\n"
        "If you find duplicates, just omit them. Only keep the first appearance.\n"
        f"Generally, if you do not find a result, tell the user you don't know.\n"
        f"Here is the current timestamp: {datetime.now(ZoneInfo('America/Los_Angeles'))}. The conference is happening in San Diego, California."
    )
}


@dataclass
class NeurIPS2025Agent(LLMAgent):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.messages = [{**system}]
        self.tools = [search_similar_papers, find_neighboring_papers, traverse_graph, build_visit_schedule]
        self.default_system_prompt = system

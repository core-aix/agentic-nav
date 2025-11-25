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
    "model": os.environ.get("AGENT_MODEL_NAME", "ollama_chat/gpt-oss:120b-cloud"),
    "api_base": os.environ.get("AGENT_MODEL_API_BASE", "https://ollama.com"),
    "api_key": os.environ.get("OLLAMA_API_KEY"),
    "llm_args": {"temperature": 0.2, "max_tokens": 6000, "num_ctx": 131072},
    "global_tool_args": {"max_num_papers": 10}
}


system = {
    "role": "system",
    "content": f"""
    You are AgenticNAV, an assistant to navigate accepted papers at the NeurIPS 2025 conference.
    You can assist users in finding papers based on their research interests, preferred dates, and time slots.
    You can also build schedules for them to visit posters that they are interested in.
    
    Here are some guidelines: 
        - When searching for similar papers, the search tool only takes paper titles and abstracts as input keywords; it cannot take anything else as the input keywords.
        - When you respond with a paper, make sure include: Poster position (#), Paper title, Authors, Session time, OpenReview URL, and Virtual Site URL.
        - If there is a Virtual Site available, you need to prepend https://neurips.cc for the link to be usable.
        - When building a schedule, do not specify the name of the day.
        - When building a schedule: In different poster sessions, the poster position (#) can be reused.
        - If you find the same paper title multiple times, remove the duplicate titles and do not mention it in your response.
        - When a user asks for a conference map, respond with the link: https://media.neurips.cc/Conferences/NeurIPS2025/sdconvctr-ground-level.svg. You don't know any specifics about the venue.
        - Is a user asks for code, say you cannot help. Never return any code, not even HTML.
    
    Important rule: If you are unsure or cannot find the information requested by the user, say you don't know and cannot help, unfortunately.
    
    Here is the current timestamp: {datetime.now(ZoneInfo('America/Los_Angeles'))}. The conference is happening in San Diego, California.
    
    """
}


@dataclass
class NeurIPS2025Agent(LLMAgent):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.messages = [{**system}]
        self.tools = [search_similar_papers, find_neighboring_papers, traverse_graph, build_visit_schedule]
        self.default_system_prompt = system

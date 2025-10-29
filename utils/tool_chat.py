from __future__ import annotations
from typing import Any, Dict, List, Callable, get_args, get_origin, Literal, Annotated, Union
from dataclasses import dataclass, field
import json, inspect
from litellm import completion

from tools.rag_search import search_papers  # <- the tool we expose

# ---- Minimal auto-schema from function signature ----
def _json_type(t: Any) -> Dict[str, Any]:
    origin, args = get_origin(t), get_args(t)
    if origin is Literal:
        return {"type": "string", "enum": list(args)}
    if origin in (list, List):
        return {"type": "array", "items": {"type": "string"}}
    if t in (str,): return {"type": "string"}
    if t in (int,): return {"type": "integer"}
    if t in (float,): return {"type": "number"}
    if t in (bool,): return {"type": "boolean"}
    return {"type": "string"}

def infer_tool(func: Callable[..., Any]) -> Dict[str, Any]:
    sig = inspect.signature(func)
    hints = getattr(func, "__annotations__", {})
    props, required = {}, []
    for name, p in sig.parameters.items():
        if name in ("self", "cls"): continue
        schema = _json_type(hints.get(name, str))
        if p.default is inspect._empty: required.append(name)
        props[name] = schema
    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": (inspect.getdoc(func) or f"Call {func.__name__}"),
            "parameters": {"type": "object", "properties": props, "required": required},
        },
    }

# ---- LiteLLM chat wrapper with tool loop ----
@dataclass
class ToolChat:
    model: str = "ollama_chat/llama3.1"
    # model: str = "ollama_chat/ibm/granite4:350m"
    api_base: str = "http://localhost:11434"
    default_params: Dict[str, Any] = field(default_factory=lambda: {"temperature": 0.2, "max_tokens": 2000})

    def tool_loop(self, messages: List[Dict[str, Any]], registry: Dict[str, Callable[..., Any]], max_rounds: int = 3) -> Dict[str, Any]:
        tools = [infer_tool(fn) for fn in registry.values()]
        msgs = list(messages)
        for _ in range(max_rounds):
            resp = completion(model=self.model, messages=msgs, tools=tools, tool_choice="auto", api_base=self.api_base, **self.default_params)

            msg = resp["choices"][0].get("message", {})
            calls = msg.get("tool_calls") or []
            print("tool_calls:", calls)

            if not calls:
                return resp

            # execute tools and append results
            for call in calls:
                name = call["function"]["name"]
                args = call["function"].get("arguments", "{}")
                try:
                    parsed = json.loads(args) if isinstance(args, str) else (args or {})
                except json.JSONDecodeError:
                    parsed = {}
                out = registry[name](**parsed)
                msgs.append({
                    "role": "tool",
                    "tool_call_id": call.get("id"),
                    "name": name,
                    "content": json.dumps(out, ensure_ascii=False),
                })
        return completion(model=self.model, messages=msgs, api_base=self.api_base, **self.default_params)

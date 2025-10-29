from __future__ import annotations
from typing import Any, Dict, List, Callable, get_args, get_origin, Literal, Annotated, Union
from dataclasses import dataclass, field
import json, inspect
from litellm import completion
import sys

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
    # model: str = "ollama_chat/phi4-mini"
    # model: str = "ollama_chat/qwen3"
    # model: str = "ollama_chat/llama3.1"
    # model: str = "ollama_chat/ibm/granite4:350m"
    model: str = "ollama_chat/gpt-oss:20b"
    api_base: str = "http://localhost:11434"
    default_params: Dict[str, Any] = field(default_factory=lambda: {"temperature": 0.2, "max_tokens": 6000})

    def tool_loop(self, messages: List[Dict[str, Any]], tool_funcs: List[Callable[..., Any]], max_rounds: int = 3) -> Dict[str, Any]:
        registry = {fn.__name__: fn for fn in tool_funcs}
        tools = [infer_tool(fn) for fn in registry.values()]
        msgs = list(messages)
        for _ in range(max_rounds):
            stream_iter = completion(
                model=self.model,
                messages=msgs,
                tools=tools,
                tool_choice="auto",
                api_base=self.api_base,
                stream=True,
                **self.default_params,
            )

            collected = ""
            calls = []
            # try:
            for chunk in stream_iter:
                choices = chunk.get("choices", []) or []
                if not choices:
                    continue
                choice = choices[0]

                # try several places where partial content may appear
                content = None
                delta = choice.get("delta")

                if "content" in delta:
                    content = delta["content"]
                elif "message" in delta and isinstance(delta["message"], dict):
                    content = delta["message"].get("content")

                if "tool_calls" in delta:
                    calls.extend(delta["tool_calls"] or [])
                
                if content is None:
                    msg = choice.get("message")
                    if isinstance(msg, dict):
                        content = msg.get("content")

                if content is None:
                    content = choice.get("text")

                if content:
                    if not isinstance(content, str):
                        try:
                            content = json.dumps(content, ensure_ascii=False)
                        except Exception:
                            content = str(content)
                    sys.stdout.write(content)
                    sys.stdout.flush()

                    collected += content
            # append the assembled assistant message so tool execution sees the assistant's follow-up
            msgs.append({"role": "assistant", "content": collected})
            print("\n", flush=True)

            if not calls:
                return msgs
            else:
                print("tool_calls:", calls, flush=True)

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
        return msgs

from __future__ import annotations
from typing import Any, Dict, List, Callable, get_args, get_origin, Literal, Annotated, Union
from dataclasses import dataclass, field
import json, inspect
import litellm
import os
from rich.console import Console
from rich.live import Live
from rich.text import Text


# litellm._turn_on_debug()


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

def infer_tool(func: Callable[..., Any], tool_args: Dict[Any, Any]) -> Dict[str, Any]:
    sig = inspect.signature(func)
    hints = getattr(func, "__annotations__", {})
    props, required = {}, []
    for name, p in sig.parameters.items():
        if name in ("self", "cls"): continue
        schema = _json_type(hints.get(name, str))
        if p.default is inspect._empty: required.append(name)
        props[name] = schema

    parameter_values = {}
    for arg_name, arg_val in tool_args.items():
        if arg_name in props.keys():
            parameter_values[arg_name] = arg_val

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": (inspect.getdoc(func) or f"Call {func.__name__}"),
            "parameters": {"type": "object", "properties": props, "required": required},
        },
        "parameter_properties_values": parameter_values
    }

# ---- LiteLLM chat wrapper with tool loop ----
@dataclass
class ToolChat:
    model: str
    api_base: str
    api_key: str
    default_params: Dict[str, Any] = field(default_factory=lambda: {"temperature": 0.2, "max_tokens": 6000, "num_ctx": 131072})

    def tool_loop(
        self,
        messages: List[Dict[str, Any]],
        tool_funcs: List[Callable[..., Any]],
        tool_args: Dict[Any, Any],
        max_rounds: int = 10
    ) -> Dict[str, Any]:
        registry = {fn.__name__: fn for fn in tool_funcs}
        tools = [infer_tool(fn, tool_args=tool_args) for fn in registry.values()]
        msgs = list(messages)

        console = Console()

        print(self.api_key)
        with console.screen():
            console.print("\n[bold green]Assistant is working...[/bold green]\n")
            
            for _ in range(max_rounds):
                stream_iter = litellm.completion(
                    model=self.model,
                    messages=msgs,
                    tools=tools,
                    tool_choice="auto",
                    api_base=self.api_base,
                    api_key=self.api_key,
                    stream=True,
                    **self.default_params,
                )

                collected = ""
                calls = []

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
                        console.print(content, end="", style="cyan")

                        collected += content
                # append the assembled assistant message so tool execution sees the assistant's follow-up
                msgs.append({"role": "assistant", "content": collected})

                if not calls:
                    return msgs
                else:
                    # calls_tmp = []
                    # for c in calls:
                    #     c_tmp = dict(c)
                    #     if "function" in c_tmp:
                    #         c_tmp["function"] = dict(c_tmp["function"])
                    #     calls_tmp.append(c_tmp)
                    # msgs[-1]["tool_calls"] = calls_tmp
                    msgs[-1]["tool_calls"] = calls
                    console.print(f"tool_calls: {calls}", style="cyan")

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

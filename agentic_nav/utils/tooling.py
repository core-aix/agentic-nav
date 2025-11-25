from __future__ import annotations

import inspect

from typing import Any, Dict, List, Callable, get_args, get_origin, Literal


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

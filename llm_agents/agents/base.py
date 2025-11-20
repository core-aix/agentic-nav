import json

import litellm
import logging

from datetime import datetime, UTC
from dataclasses import dataclass, field
from typing import List, Dict

from llm_agents.tools import get_all_tools
from llm_agents.utils.tooling import infer_tool


LOGGER = logging.getLogger(__name__)


@dataclass
class LLMAgent:
    model: str = "ollama_chat/gpt-oss:20b"
    api_base: str = "http://localhost:11434"
    api_key: str = None
    llm_args: dict = field(default_factory=lambda: {"temperature": 0.2, "max_tokens": 6000, "num_ctx": 131072})
    tools: List[callable] = field(default_factory=lambda: get_all_tools())
    global_tool_args: dict = field(default_factory=lambda: {"max_num_papers": 10})
    max_interaction_rounds: int = 10
    messages: List[Dict] = field(default_factory=lambda: [])
    tool_registry: Dict = None
    tool_descriptions: List = None
    default_system_prompt: Dict[str, str] = None

    def test_llm_connection(self):
        try:
            response = litellm.completion(
                model=self.model,
                messages=[{"role": "user", "content": "test", "_ts": str(datetime.now(UTC))}],
                tool_choice="auto",
                api_base=self.api_base,
                api_key=self.api_key,
                stream=True,
                **self.llm_args,
            )

            LOGGER.info(f"Model is available! Response: {response.choices[0].message.content}")
        except Exception as e:
            LOGGER.error(f"Model not available or connection failed: {str(e)}")

    def setup_session(self, tool_funcs: List[callable] = None):
        self.tool_registry = {fn.__name__: fn for fn in self.tools} if tool_funcs is None else {fn.__name__: fn for fn in tool_funcs}
        self.tool_descriptions = [infer_tool(fn, tool_args=self.global_tool_args) for fn in self.tool_registry.values()]
        LOGGER.info(f"Agent setup and tools ready to use.")
        LOGGER.debug(f"Available tools: {self.tools}")

    def remove_session(self):
        """De-registers tools and resets messages to the initial state."""
        self.tool_registry = None
        self.tool_descriptions = None
        self.messages = [self.default_system_prompt if not None else {"role": "system", "content": "You are a helpful assistant."}]

    def interact(self, message: Dict):
        assert self.tool_registry is not None, "Make sure to call 'setup_session()' before the first interaction."
        assert self.tool_descriptions is not None, "Make sure to call 'setup_session()' before the first interaction."

        assert type(message) == dict, "Make sure to pass a dictionary as next message for the agent."
        assert "role" in message.keys(), "The message must contain a 'role' key."
        assert "content" in message.keys(), "The message must contain a 'content' key."

        if "_ts" not in message.keys():
            message["_ts"] = str(datetime.now(UTC))

        self.messages.append(message)
        for _ in range(self.max_interaction_rounds):
            collected, calls = self._send_to_llm(
                messages=self.messages,
                model=self.model,
                api_base=self.api_base,
                api_key=self.api_key
            )
            # append the assembled assistant message so tool execution sees the assistant's follow-up
            self.messages.append({"role": "assistant", "content": collected, "_ts": str(datetime.now(UTC))})
            LOGGER.debug(f"Agent response: {collected}")

            if not calls:
                return self.messages
            else:
                self.messages[-1]["tool_calls"] = calls
                LOGGER.debug(f"Agent requested tool calls: {calls}")

            # execute tools and append results
            for call in calls:
                self.messages.append(
                    self.call_tool(
                        tool_call=call
                    )
                )

        print(f"MESSAGES: {self.messages}")
        return self.messages

    def interact_stateless(
        self,
        messages: List[Dict],
        model: str,
        api_base: str,
        api_key: str,
        llm_args: Dict = None
    ):
        """
        This method is designed to support multi-user sessions and requires state management outside the agent class.
        """
        assert self.tool_registry is not None, "Make sure to call 'setup_session()' before the first interaction."
        assert self.tool_descriptions is not None, "Make sure to call 'setup_session()' before the first interaction."

        # Sanity check for all messages
        for message in messages:
            if "_ts" not in message.keys():
                message["_ts"] = str(datetime.now(UTC))

        for round_num in range(self.max_interaction_rounds):
            # Stream the LLM response
            collected = ""
            calls = []

            # Create initial assistant message
            assistant_msg_idx = len(messages)
            messages.append({"role": "assistant", "content": "", "_ts": str(datetime.now(UTC))})

            stream_iter = litellm.completion(
                model=model if model is not None else self.model,
                messages=messages[:assistant_msg_idx],  # Don't include the empty assistant message
                tools=self.tool_descriptions,
                tool_choice="auto",
                api_base=api_base if api_base is not None else self.api_base,
                api_key=api_key if api_key is not None else self.api_key,
                stream=True,
                **llm_args if llm_args is not None else self.llm_args,
            )

            for chunk in stream_iter:
                choices = chunk.get("choices", []) or []
                if not choices:
                    continue
                choice = choices[0]

                # Extract content from chunk
                content = None
                delta = choice.get("delta")

                if delta and "content" in delta:
                    content = delta["content"]
                elif delta and "message" in delta and isinstance(delta["message"], dict):
                    content = delta["message"].get("content")

                if delta and "tool_calls" in delta:
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

                    collected += content
                    # Update the assistant message with accumulated content
                    messages[assistant_msg_idx]["content"] = collected

                    # Yield the updated messages for streaming display
                    yield messages.copy()

            # After streaming is complete, update with final content
            messages[assistant_msg_idx]["content"] = collected
            LOGGER.debug(f"Agent response: {collected}")

            if not calls:
                yield messages
                return
            else:
                messages[assistant_msg_idx]["tool_calls"] = calls
                LOGGER.debug(f"Agent requested tool calls: {calls}")
                yield messages.copy()

            # Execute tools and append results
            for call in calls:
                messages.append(self.call_tool(tool_call=call))
                yield messages.copy()

        yield messages

    def _send_to_llm(
        self,
        messages: List[Dict],
        model: str,
        api_base: str,
        api_key: str,
        llm_args: Dict = None
    ):
        stream_iter = litellm.completion(
            model=model if model is not None else self.model,
            messages=messages,
            tools=self.tool_descriptions,
            tool_choice="auto",
            api_base=api_base if api_base is not None else self.api_base,
            api_key=api_key if api_key is not None else self.api_key,
            stream=True,
            **llm_args if llm_args is not None else self.llm_args,
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
                    except Exception as e:
                        LOGGER.error(f"JSON encoding error encountered. {e}. Treating agent response as regular text.")
                        content = str(content)

                collected += content

        return collected, calls

    def call_tool(self, tool_call: Dict):
        name = tool_call["function"]["name"]
        args = tool_call["function"].get("arguments", "{}")
        LOGGER.debug(f"Preparing tool call: {name}")
        LOGGER.debug(f"Expected tool arguments: {args}")
        try:
            parsed = json.loads(args) if isinstance(args, str) else (args or {})
            LOGGER.debug(f"Parsed tool call arguments: {parsed}")
        except json.JSONDecodeError:
            parsed = {}
            LOGGER.warning(f"Tool call arguments: COULD NOT BE PARSED")
        out = self.tool_registry[name](**parsed)
        LOGGER.debug(f"Tool call output: {parsed}")

        return {
            "role": "tool",
            "tool_call_id": tool_call.get("id"),
            "name": name,
            "content": json.dumps(out, ensure_ascii=False),
            "_ts": str(datetime.now(UTC))
        }

    def set_history(self, messages):
        self.messages = messages
        LOGGER.info(f"Set new message history.")

    def get_history(self):
        return self.messages

    @staticmethod
    def set_system_prompt(new_system_prompt: str, messages: List[Dict]):
        messages = [m for m in messages if m.get("role") != "system"]
        messages.insert(0, {
            "role": "system",
            "content": new_system_prompt,
            "_ts": str(datetime.now(UTC))
        })
        LOGGER.info(f"New system prompt set and configured.")
        LOGGER.debug(f"New system prompt: {new_system_prompt}")
        return messages

    def get_system_prompt(self):
        for message in self.messages:
            if "role" in message.keys() and message["role"] == "system":
                return message

        return None

    def get_most_recent_assistant_message(self):
        for message in reversed(self.messages):
            if message.get("role") == "assistant":
                return message
        return None

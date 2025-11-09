"""
Lightweight terminal chat UI that interacts with an agent implementation.

Contract with main.py:
- main.py should export a function with this signature:
    def agent_respond(messages: list[dict]) -> dict:
        # messages is a list of {"role": "system|user|assistant", "content": str}
        # returns a single assistant message dict: {"role": "assistant", "content": str}
- If your main.py uses a different API, adapt the import below or provide a thin shim.

Features:
- Renders assistant messages as Markdown (uses rich if available, falls back to plain text).
- Multi-line input via $EDITOR using the "/edit" command.
- Commands: /help, /exit, /system, /edit, /history, /save <path>
"""

import os
import tempfile
import json
import datetime

import click
from rich.console import Console
from rich.markdown import Markdown

from pathlib import Path

from llm_agents.utils.logging import setup_logging

# Try to import agent_respond from main.py
try:
    from llm_agents.agent_neurips_2025 import agent_respond, msgs  # main.py should define this
except Exception as e:
    raise RuntimeError(
        "Could not import agent_respond from main.py. "
        "Please implement def agent_respond(messages: list[dict]) -> dict in main.py"
    )


PROJECT_ROOT = Path(__file__).parent

console = Console(soft_wrap=True)
def render_markdown(text):
    console.print(Markdown(text))

PROMPT = "You> "

def open_editor(initial_text=""):
    editor = os.environ.get("EDITOR")
    if not editor:
        # Minimal sensible defaults
        if os.name == "nt":
            editor = "notepad"
        else:
            editor = "nano"
    with tempfile.NamedTemporaryFile(suffix=".md", delete=False, mode="w+", encoding="utf-8") as tf:
        path = tf.name
        tf.write(initial_text)
        tf.flush()
    try:
        # Open editor and wait
        rc = os.system(f'{editor} "{path}"')
        if rc != 0:
            print(f"(editor exit code {rc})")
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
    finally:
        try:
            os.unlink(path)
        except Exception:
            pass
    return content.strip()

def print_help():
    help_text = """
Commands:
  /help           Show this help
  /exit           Exit the chat
  /system         Set or replace system prompt (multi-line via $EDITOR)
  /edit           Compose multi-line user message via $EDITOR
  /history        Show conversation history (JSON)
  /save <path>    Save conversation history to a file (JSON)
Typing anything else will send it as a user message.
"""
    print(help_text)

def show_history(messages):
    for i, m in enumerate(messages):
        ts = m.get("_ts", "")
        role = m.get("role", "")
        content = m.get("content", "")
        header = f"[{i}] {role} {ts}"
        print(header)
        print("-" * len(header))
        print(content)
        print()

def save_history(messages, path):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(messages, f, indent=2, ensure_ascii=False)
        print(f"Saved to {path}")
    except Exception as e:
        print("Save failed:", e)

@click.command()
@click.option("-a", "--api-base", default="http://localhost:11434", type=str, help="Base URL to access the inference endpoint.")
@click.option("-k", "--api-key", default=None, type=str, help="API key for authentication.")
@click.option("-m", "--model", default="ollama_chat/gpt-oss:20b", type=str, help="Specify the model you want to use in LiteLLM format.")
@click.option("-t", "--temperature", default=0.2, type=float, help="Specify the model temperature.")
@click.option("--max-tokens", default=6000, type=int, help="Specify the max. number of model output tokens.")
@click.option("-c", "--num-ctx", default=131072, type=int, help="Specify the model context window.")
@click.option("-l", "--max-num-papers", default=50, type=int, help="Specify the maximum number of papers to retrieve.")
def main(api_base, api_key, model, temperature, max_tokens, num_ctx, max_num_papers):

    # Setup logging
    setup_logging(
        log_dir=f"logs", # This directory will spawn in the directory wherever main() is called from.
        level=os.environ.get("LLM_AGENTS_LOG_LEVEL", "DEBUG")
    )

    messages = msgs.copy()
    # Optionally seed with a system message; leave empty by default
    # messages.append({"role": "system", "content": "You are a helpful assistant.", "_ts": str(datetime.utcnow())})
    print("Lightweight Chat UI. Type /help for commands.")

    # Config for the LLM messages
    llm_config = {
        "temperature": temperature,
        "max_tokens": max_tokens,
        "num_ctx": num_ctx
    }

    # Parameters to limit the tool calling scope
    tool_args = {
        "num_records": max_num_papers
    }

    while True:
        try:
            line = input(PROMPT).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue

        if line.startswith("/"):
            parts = line.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""
            if cmd == "/help":
                print_help()
                continue
            elif cmd == "/exit":
                print("Goodbye.")
                break
            elif cmd == "/edit":
                content = open_editor()
                if content:
                    msg = {"role": "user", "content": content, "_ts": str(datetime.datetime.now(datetime.UTC))}
                    messages.append(msg)
                else:
                    print("(no content)")
                    continue
            elif cmd == "/system":
                content = open_editor()
                if content:
                    # Replace or append system message as first element
                    sys_msg = {"role": "system", "content": content, "_ts": str(datetime.datetime.now(datetime.UTC))}
                    # Remove existing system messages
                    messages = [m for m in messages if m.get("role") != "system"]
                    messages.insert(0, sys_msg)
                    print("System prompt set.")
                else:
                    print("(no content)")
                    continue
            elif cmd == "/history":
                show_history(messages)
                continue
            elif cmd == "/save":
                path = arg.strip() or "chat_history.json"
                save_history(messages, path)
                continue
            else:
                print("Unknown command. Type /help.")
                continue
        else:
            # Regular single-line user message
            msg = {"role": "user", "content": line, "_ts": str(datetime.datetime.now(datetime.UTC))}
            messages.append(msg)

        # Call the agent
        try:
            # Provide agent_respond with a copy without timestamps
            msgs_for_agent = [{"role": m["role"], "content": m["content"]} for m in messages]
            updated_msgs = agent_respond(
                msgs_for_agent,
                model_name=model,
                api_base=api_base,
                api_key=api_key,
                llm_args=llm_config,
                tool_args=tool_args
            )
            for msg in updated_msgs:
                if "_ts" not in msg and "role" in msg and msg.get("role") == "assistant":
                    msg["_ts"] = str(datetime.datetime.now(datetime.UTC))
            # messages = updated_msgs  # TODO: Currently not using this due to issue with tool call mismatch error
            assistant_msg = updated_msgs[-1]
            messages.append(assistant_msg)

            # Render assistant message as Markdown
            if console:
                render_markdown(assistant_msg["content"])
            else:
                print("\nAssistant (Markdown):\n")
                render_markdown(assistant_msg["content"])
                print()
        except KeyboardInterrupt:
            print("\n(interrupted)")
            continue
        except Exception as e:
            print("Error from agent:", e)
            # keep going

if __name__ == "__main__":
    main()
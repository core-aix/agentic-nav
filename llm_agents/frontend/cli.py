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
import click
import os
import datetime
import logging

from rich.console import Console
from rich.markdown import Markdown

from pathlib import Path

# TODO: Build an agent factory for the user to select which agent to use. Relevant down the road as we build more.
from llm_agents.agents import NeurIPS2025Agent
from llm_agents.utils.logging import setup_logging
from llm_agents.utils.file_handlers import save_chat_history
from llm_agents.utils.cli import open_editor, show_history, print_help


PROJECT_ROOT = Path(__file__).parent
LOGGER = logging.getLogger(__name__)

console = Console(soft_wrap=True)
def render_markdown(text):
    console.print(Markdown(text))

PROMPT = "You> "


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
        log_dir=f"logs", # This subdirectory will spawn in the directory wherever main() is called from.
        level=os.environ.get("LLM_AGENTS_LOG_LEVEL", "INFO")
    )

    print("Lightweight Chat UI. Type /help for commands.")
    LOGGER.info(f"Agent runtime started")

    # Config for the LLM messages
    llm_config = {
        "temperature": temperature,
        "max_tokens": max_tokens,
        "num_ctx": num_ctx
    }
    LOGGER.info(f"LLM configuration: {llm_config}")

    # Parameters to limit the tool calling scope
    tool_args = {
        "num_records": max_num_papers
    }
    LOGGER.info(f"Global tool arguments: {tool_args}")

    agent = NeurIPS2025Agent(
        model=model,
        api_base=api_base,
        api_key=api_key,
        llm_args=llm_config,
        global_tool_args=tool_args,
    )

    agent.setup_session()

    while True:
        try:
            line = input(PROMPT).strip()
            LOGGER.debug(f"USER PROMPT: {line}")
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
                LOGGER.info("Goodbye!")
                break

            elif cmd == "/edit":
                content = open_editor()
                if content:
                    next_message = {"role": "user", "content": content, "_ts": str(datetime.datetime.now(datetime.UTC))}
                else:
                    LOGGER.warning("No content found. Continuing...")
                    continue

            elif cmd == "/system":
                content = open_editor()
                if content:
                    messages = agent.set_system_prompt(
                        messages=agent.get_history(),
                        new_system_prompt=content
                    )

                    agent.set_history(messages=messages)

                    continue

                else:
                    LOGGER.warning("No content found. Continuing...")
                    continue

            elif cmd == "/history":
                show_history(
                    agent.get_history()
                )
                continue

            elif cmd == "/save":
                Path(f"chat_histories/").mkdir(exist_ok=True, parents=True)
                time_now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
                path = arg.strip() or f"chat_histories/{time_now}_chat_history.json"
                save_chat_history(
                    agent.get_history(),
                    path
                )
                continue
            else:
                LOGGER.warning("Unknown command. Type /help.")
                continue
        else:
            # Regular single-line user message
            next_message = {"role": "user", "content": line, "_ts": str(datetime.datetime.now(datetime.UTC))}

        # Call the agent
        try:
            agent.interact(
                message=next_message
            )
            # TODO: WE NEED TO ADD TOOL OUTPUTS TO THE SEQUENCE, ESPECIALLY WHEN A USER HITS "/save".
            assistant_message = agent.get_most_recent_assistant_message()

            # Render assistant message as Markdown
            if console:
                render_markdown(assistant_message["content"])
            else:
                print("\nAssistant (Markdown):\n")
                render_markdown(assistant_message["content"])
                print()
        except KeyboardInterrupt:
            LOGGER.info(f"Agent interrupted by user command")
            print("\n(interrupted)")
            continue
        except Exception as e:
            LOGGER.error(f"Agent encountered an error: {e}")
            print("Error from agent:", e)
            # keep going


if __name__ == "__main__":
    main()

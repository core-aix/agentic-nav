"""
Enhanced terminal chat UI with async streaming and full terminal functionality.

Features:
- Async streaming output as LLM generates tokens
- Rich prompt with command history and auto-completion
- Live markdown rendering during streaming
- Multi-line input via Ctrl+O or /edit command
- Commands: /help, /exit, /system, /edit, /history, /save <path>, /clear
- Keyboard shortcuts: Ctrl+C to cancel, Ctrl+D to exit
"""
import asyncio
import click
import os
import datetime
import logging
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.formatted_text import HTML

from llm_agents.agents import NeurIPS2025Agent
from llm_agents.utils.logging import setup_logging
from llm_agents.utils.file_handlers import save_chat_history
from llm_agents.utils.cli import open_editor, show_history, print_help


PROJECT_ROOT = Path(__file__).parent
LOGGER = logging.getLogger(__name__)

EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "nomic-embed-text")
EMBEDDING_MODEL_API_BASE = os.environ.get("EMBEDDING_MODEL_API_BASE", "http://localhost:11435")

AGENT_MODEL_NAME = os.environ.get("AGENT_MODEL_NAME", "gpt-oss:20b")
AGENT_MODEL_API_BASE = os.environ.get("AGENT_MODEL_API_BASE", "http://localhost:11436")
OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY")

console = Console(soft_wrap=True)

# Command completer for auto-complete
command_completer = WordCompleter(
    ['/help', '/exit', '/system', '/edit', '/history', '/save', '/clear'],
    ignore_case=True,
    sentence=True
)

bindings = KeyBindings()

@bindings.add('c-o')
def _(event):
    """Multi-line input with Ctrl+O"""
    event.current_buffer.insert_text('\n')


def create_prompt_session():
    """Create a prompt_toolkit session with history and auto-suggest"""
    history_file = Path.home() / ".llm_agents_history"

    return PromptSession(
        history=FileHistory(str(history_file)),
        auto_suggest=AutoSuggestFromHistory(),
        completer=command_completer,
        complete_while_typing=True,
        key_bindings=bindings,
        enable_open_in_editor=True,
        multiline=False,
    )


def render_markdown(text: str, title: Optional[str] = None):
    """Render markdown with optional panel title"""
    if title:
        console.print(Panel(Markdown(text), title=title, border_style="blue"))
    else:
        console.print(Markdown(text))


def stream_agent_response_sync(agent, message: dict):
    """
    Stream agent response with live markdown rendering.
    Uses the agent's interact_stateless method for streaming.
    """
    # Get current history and add the new message
    messages = agent.get_history().copy()
    messages.append(message)

    accumulated_text = ""
    tool_calls_made = []
    final_messages = None

    with Live(console=console, refresh_per_second=10) as live:
        try:
            # Use interact_stateless for streaming (it's a generator)
            for updated_messages in agent.interact_stateless(
                messages=messages,
                model=agent.model,
                api_base=agent.api_base,
                api_key=agent.api_key,
                llm_args=agent.llm_args
            ):
                final_messages = updated_messages

                # Extract the last assistant message
                for msg in reversed(updated_messages):
                    if msg.get("role") == "assistant":
                        content = msg.get("content", "")
                        if content != accumulated_text:
                            accumulated_text = content

                            # Show streaming content
                            if accumulated_text:
                                live.update(Markdown(accumulated_text))

                        # Check for tool calls
                        if "tool_calls" in msg and msg["tool_calls"] != tool_calls_made:
                            tool_calls_made = msg["tool_calls"]
                            # Show tool execution
                            tool_names = [tc["function"]["name"] for tc in tool_calls_made]
                            tool_info = Text(f"\n🔧 Executing tools: {', '.join(tool_names)}", style="yellow")
                            live.update(tool_info)
                        break

            # Update agent's history with final messages
            if final_messages:
                agent.set_history(final_messages)

        except KeyboardInterrupt:
            live.stop()
            console.print("\n[yellow]⚠ Response cancelled by user[/yellow]")
            raise
        except Exception as e:
            live.stop()
            console.print(f"\n[red]❌ Error: {e}[/red]")
            LOGGER.error(f"Streaming error: {e}", exc_info=True)
            raise


async def async_interact(agent, message: dict):
    """Async wrapper for agent interaction with streaming"""
    try:
        # Run the synchronous streaming function in a thread pool
        await asyncio.to_thread(stream_agent_response_sync, agent, message)
    except KeyboardInterrupt:
        LOGGER.info("Agent interaction cancelled by user")
    except Exception as e:
        LOGGER.error(f"Agent interaction failed: {e}")
        console.print(f"[red]Error: {e}[/red]")


def print_welcome():
    """Print welcome message"""
    welcome = Text()
    welcome.append("╔═══════════════════════════════════════╗\n", style="bold blue")
    welcome.append("║   ", style="bold blue")
    welcome.append("LLM Agent Chat Interface", style="bold white")
    welcome.append("     ║\n", style="bold blue")
    welcome.append("╚═══════════════════════════════════════╝\n", style="bold blue")
    welcome.append("\nCommands:\n", style="bold yellow")
    welcome.append("  /help     - Show help\n", style="cyan")
    welcome.append("  /edit     - Multi-line input\n", style="cyan")
    welcome.append("  /history  - Show conversation history\n", style="cyan")
    welcome.append("  /system   - Set system prompt\n", style="cyan")
    welcome.append("  /save     - Save conversation\n", style="cyan")
    welcome.append("  /clear    - Clear screen\n", style="cyan")
    welcome.append("  /exit     - Exit (or Ctrl+D)\n", style="cyan")
    welcome.append("\nShortcuts:\n", style="bold yellow")
    welcome.append("  Ctrl+O    - New line in input\n", style="cyan")
    welcome.append("  Ctrl+C    - Cancel current response\n", style="cyan")
    welcome.append("  Ctrl+D    - Exit\n", style="cyan")
    welcome.append("  ↑/↓       - Navigate history\n", style="cyan")
    welcome.append("  Tab       - Auto-complete commands\n", style="cyan")

    console.print(welcome)


@click.command()
@click.option("-t", "--temperature", default=0.2, type=float,
              help="Specify the model temperature.")
@click.option("--max-tokens", default=6000, type=int,
              help="Specify the max. number of model output tokens.")
@click.option("-c", "--num-ctx", default=131072, type=int,
              help="Specify the model context window.")
@click.option("-l", "--max-num-papers", default=50, type=int,
              help="Specify the maximum number of papers to retrieve.")
def main(temperature, max_tokens, num_ctx, max_num_papers):
    """Enhanced LLM Agent CLI with async streaming and rich terminal features"""

    # Setup logging
    setup_logging(
        log_dir="logs",
        level=os.environ.get("LLM_AGENTS_LOG_LEVEL", "INFO")
    )

    print_welcome()
    LOGGER.info("Agent runtime started")

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

    # Initialize agent
    agent = NeurIPS2025Agent(
        model=f"ollama_chat/{AGENT_MODEL_NAME}",
        api_base=AGENT_MODEL_API_BASE,
        api_key=OLLAMA_API_KEY,
        llm_args=llm_config,
        global_tool_args=tool_args,
    )

    agent.setup_session()
    console.print("[green]✓ Agent initialized successfully[/green]\n")

    # Create prompt session
    session = create_prompt_session()

    # Main interaction loop
    while True:
        try:
            # Get user input with rich prompt
            line = session.prompt(
                HTML('<ansiyellow><b>You></b></ansiyellow> '),
                multiline=False,
            ).strip()

            LOGGER.debug(f"USER PROMPT: {line}")

        except (EOFError, KeyboardInterrupt):
            console.print("\n[yellow]Goodbye! 👋[/yellow]")
            LOGGER.info("User exited")
            break

        if not line:
            continue

        # Handle commands
        if line.startswith("/"):
            parts = line.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            if cmd == "/help":
                print_help()
                continue

            elif cmd == "/exit":
                console.print("[yellow]Goodbye! 👋[/yellow]")
                LOGGER.info("User exited via /exit command")
                break

            elif cmd == "/clear":
                console.clear()
                print_welcome()
                continue

            elif cmd == "/edit":
                content = open_editor()
                if content:
                    next_message = {
                        "role": "user",
                        "content": content,
                        "_ts": str(datetime.datetime.now(datetime.UTC))
                    }
                else:
                    console.print("[yellow]⚠ No content provided[/yellow]")
                    continue

            elif cmd == "/system":
                content = open_editor()
                if content:
                    messages = agent.set_system_prompt(
                        messages=agent.get_history(),
                        new_system_prompt=content
                    )
                    agent.set_history(messages=messages)
                    console.print("[green]✓ System prompt updated[/green]")
                    continue
                else:
                    console.print("[yellow]⚠ No content provided[/yellow]")
                    continue

            elif cmd == "/history":
                show_history(agent.get_history())
                continue

            elif cmd == "/save":
                Path("chat_histories/").mkdir(exist_ok=True, parents=True)
                time_now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
                path = arg.strip() or f"chat_histories/{time_now}_chat_history.json"

                try:
                    save_chat_history(agent.get_history(), path)
                    console.print(f"[green]✓ Chat saved to {path}[/green]")
                except Exception as e:
                    console.print(f"[red]❌ Failed to save: {e}[/red]")
                    LOGGER.error(f"Save failed: {e}")
                continue

            else:
                console.print(f"[red]❌ Unknown command: {cmd}[/red]")
                console.print("[yellow]Type /help for available commands[/yellow]")
                continue
        else:
            # Regular single-line user message
            next_message = {
                "role": "user",
                "content": line,
                "_ts": str(datetime.datetime.now(datetime.UTC))
            }

        try:
            console.print()
            asyncio.run(async_interact(agent, next_message))
            console.print()

        except KeyboardInterrupt:
            console.print("\n[yellow]⚠ Interrupted[/yellow]")
            continue
        except Exception as e:
            console.print(f"\n[red]❌ Error: {e}[/red]")
            LOGGER.error(f"Interaction error: {e}", exc_info=True)
            continue


if __name__ == "__main__":
    main()

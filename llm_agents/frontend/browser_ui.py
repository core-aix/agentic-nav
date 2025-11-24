"""
Gradio web UI that interacts with an agent implementation.

Features matching terminal UI:
- Multi-turn chat with Markdown rendering
- System prompt editing
- View conversation history
- Save chat history to file
- All model configuration options
- Clear chat functionality
- **Per-user conversation state management with stateless agent**
"""
from venv import logger

import gradio as gr
import os
import datetime
import logging
import json

from functools import partial
from pathlib import Path
from typing import List, Tuple, Optional, Dict

from llm_agents.agents import NeurIPS2025Agent, DEFAULT_NEURIPS2025_AGENT_ARGS
from llm_agents.utils.logger import setup_logging
from llm_agents.utils.file_handlers import save_chat_history


LOGGER = logging.getLogger(__name__)

EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "nomic-embed-text")
EMBEDDING_MODEL_API_BASE = os.environ.get("EMBEDDING_MODEL_API_BASE", "http://localhost:11435")

AGENT_MODEL_NAME = os.environ.get("AGENT_MODEL_NAME", "gpt-oss:20b")
AGENT_MODEL_API_BASE = os.environ.get("AGENT_MODEL_API_BASE", "http://localhost:11436")
OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY", DEFAULT_NEURIPS2025_AGENT_ARGS["api_key"])


def initialize_agent():
    """Initialize the AGENT instance."""
    agent = NeurIPS2025Agent(
        model=f"ollama_chat/{AGENT_MODEL_NAME}",
        api_base=AGENT_MODEL_API_BASE,
        api_key=OLLAMA_API_KEY,
        llm_args=DEFAULT_NEURIPS2025_AGENT_ARGS["llm_args"],
        global_tool_args=DEFAULT_NEURIPS2025_AGENT_ARGS["global_tool_args"],
    )
    agent.setup_session()
    return agent


def configure_agent(
        api_base: str,
        api_key: str,
        model: str,
        temperature: float,
        max_tokens: int,
        num_ctx: int,
        max_num_papers: int,
        current_config: Dict
):
    """Initialize the agent with given configuration."""
    LOGGER.info(f"Agent runtime started via Gradio UI for session")
    current_config.update({
        "model": model,
        "api_base": api_base,
        "api_key": api_key,
        "llm_args": {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "num_ctx": num_ctx
        },
        "global_tool_args": {"max_num_papers": max_num_papers}
    })

    current_config_to_print = current_config.copy()
    if "api_key" in current_config_to_print:
        del current_config_to_print["api_key"]
    LOGGER.info(f"User-defined configuration saved. Config: {current_config_to_print}")

    return current_config, "✓ Agent initialized successfully!"


def chat_fn(
        new_message: str,
        history: List[Dict],
        config: Optional[Dict],
        messages: Optional[List[Dict]],
        agent: NeurIPS2025Agent,
) -> Tuple[List[Dict], Optional[List[Dict]]]:
    """
    Handle chat interaction using stateless agent.

    Args:
        new_message: User's input message
        history: Chat history as list of message dictionaries with role/content
        config: Configuration dict with model, api_base, api_key, llm_args
        messages: Current conversation messages list
        agent: Agent instance

    Returns:
        Tuple of (updated_history, messages)
    """
    if not new_message.strip():
        yield history, messages
        return

    LOGGER.debug(f"USER PROMPT: {new_message}")

    # Safety check: ensure messages is a list
    if messages is None or not isinstance(messages, list):
        LOGGER.warning("Messages state was not properly initialized, resetting...")
        messages = [agent.get_system_prompt()]

    # Create a copy of history and messages to avoid mutation issues
    history = history.copy() if history else []
    messages = messages.copy()

    # Add user message to history immediately with empty assistant response
    user_msg_dict = {"role": "user", "content": new_message}
    assistant_msg_dict = {"role": "assistant", "content": ""}
    history.extend([user_msg_dict, assistant_msg_dict])

    try:
        # Create user message with timestamp
        user_message = {
            "role": "user",
            "content": new_message,
            "_ts": str(datetime.datetime.now(datetime.timezone.utc))
        }

        # Add user message to conversation
        messages.append(user_message)

        # Stream the response
        accumulated_response = ""
        for partial_messages in agent.interact_stateless(
                messages=messages,
                model=config["model"],
                api_base=config["api_base"],
                api_key=config["api_key"],
                llm_args=config["llm_args"]
        ):
            # Get the latest assistant message content
            for msg in reversed(partial_messages):
                if msg.get("role") == "assistant":
                    accumulated_response = msg["content"]
                    break

            # Update the last assistant message in history with accumulated response
            history[-1]["content"] = accumulated_response
            yield history, partial_messages

        # Final update with complete messages
        messages = partial_messages
        LOGGER.info("Agent response generated successfully")

    except Exception as e:
        LOGGER.error(f"Agent encountered an error: {e}", exc_info=True)
        error_msg = f"❌ Error: {str(e)}"
        history[-1]["content"] = error_msg
        yield history, messages


def update_system_prompt(
    new_prompt: str,
    messages: Optional[List[Dict]],
    agent: NeurIPS2025Agent
) -> Tuple[str, Optional[List[Dict]]]:
    """Update the system prompt in the message history.

    Args:
        new_prompt: New system prompt
        messages: Current message history
        agent: Agent instance

    Returns:
        Tuple of (status_message, agent_instance, config, updated_messages)
    """
    if not new_prompt.strip():
        return "System prompt cannot be empty.", messages

    try:
        # Initialize messages if None
        if messages is None:
            messages = []

        # Use the static method to update system prompt
        messages = agent.set_system_prompt(new_system_prompt=new_prompt, messages=messages)

        LOGGER.info("System prompt updated")
        LOGGER.info(f"New system prompt: {messages[0]}")
        return "✓ System prompt updated successfully!", messages
    except Exception as e:
        LOGGER.error(f"Error updating system prompt: {e}")
        return f"Error: {str(e)}", messages


def view_history(messages: Optional[List[Dict]]) -> str:
    """View the full conversation history in JSON format.

    Args:
        messages: Current message history

    Returns:
        JSON formatted history string
    """
    if messages is None:
        return "⚠️ No conversation history yet."

    try:
        # Format as pretty JSON
        return json.dumps(messages, indent=2, ensure_ascii=False)
    except Exception as e:
        LOGGER.error(f"Error viewing history: {e}")
        return f"❌ Error: {str(e)}"


def save_history(filename: str, messages: Optional[List[Dict]]) -> str:
    """Save chat history to a JSON file.

    Args:
        filename: Optional filename
        messages: Current message history

    Returns:
        Status message
    """
    if messages is None or len(messages) == 0:
        return "⚠️ No conversation history to save."

    try:
        # Create directory if it doesn't exist
        Path("chat_histories/").mkdir(exist_ok=True, parents=True)

        # Generate filename if not provided
        if not filename.strip():
            time_now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            # Add session identifier to prevent conflicts
            import uuid
            session_id = str(uuid.uuid4())[:8]
            filename = f"chat_histories/{time_now}_session_{session_id}_chat_history.json"
        else:
            filename = filename.strip()
            # Ensure it's in chat_histories directory
            if not filename.startswith("chat_histories/"):
                filename = f"chat_histories/{filename}"
            if not filename.endswith(".json"):
                filename += ".json"

        # Save the history
        save_chat_history(messages, filename)

        LOGGER.info(f"Chat history saved to {filename}")
        return f"✓ Chat history saved to: {filename}"

    except Exception as e:
        LOGGER.error(f"Error saving history: {e}")
        return f"❌ Error: {str(e)}"


def clear_chat(
        config: Optional[Dict],
        messages: Optional[List[Dict]],
        agent: NeurIPS2025Agent
) -> Tuple[str, List, Optional[List[Dict]]]:
    """Clear the chat history in the UI and reset message list.

    Args:
        config: Current configuration
        messages: Current message history
        agent: Agent instance

    Returns:
        Tuple of (status_message, empty_history, reset_messages)
    """
    system_prompt = agent.get_system_prompt()
    if isinstance(system_prompt, dict):
        reset_messages = [system_prompt]
    else:
        reset_messages = []

    LOGGER.info("Chat cleared and reset")
    return "✓ Chat cleared!", [], reset_messages


def submit_message(message, history, config, messages, agent):
    """Wrapper to clear input and process message"""
    yield from chat_fn(message, history, config, messages, agent)


def main():

    # Setup the agent instance
    agent = initialize_agent()

    with gr.Blocks(
        title="SciAgent For NeurIPS 2025",
        theme=gr.themes.Default(
            spacing_size=gr.themes.sizes.spacing_sm,
            radius_size=gr.themes.sizes.radius_none
        )
    ) as webapp:

        gr.Markdown("# 🤖 SciAgent For NeurIPS 2025")
        gr.Markdown("Initialize the agent with your settings, then start chatting!")
        gr.Markdown("*Each user session has its own independent conversation state. Enjoy!*")
        gr.Markdown("Please note that this tool is experimental and the agent may not be able to return all papers that match your query.")

        # Session state for agent instance, config, and messages
        config_state = gr.State(value=DEFAULT_NEURIPS2025_AGENT_ARGS)
        messages_state = gr.State(value=[agent.get_system_prompt()])

        with gr.Row():
            with gr.Column():
                # Main chat interface
                chatbot = gr.Chatbot(
                    label="Conversation Trail",
                    height=500,
                    type="messages",
                    show_copy_button=True,
                )

                with gr.Row():
                    msg_input = gr.Textbox(
                        label="Your message",
                        placeholder="Type your message here...",
                        lines=3,
                        scale=4
                    )
                    submit_btn = gr.Button("Send", variant="primary", scale=1)

                with gr.Row():
                    clear_btn = gr.Button("🗑️ Clear Chat", size="sm")
                    save_btn = gr.Button("💾 Save History", size="sm")

                with gr.Row():
                    # Help text at bottom
                    gr.Markdown("""
                        ### 📖 Usage Guide

                        1. **Initialize**: Configure settings and click "Initialize Agent"
                        2. **Chat**: Type messages and press Enter or click Send
                        3. **System Prompt**: Customize the agent's behavior via System Prompt panel
                        4. **History**: View or save your conversation using the History & Save panel
                        5. **Clear**: Start a fresh conversation with the Clear Chat button

                        ### Note on Ollama API Keys
                        In case you are experiencing an error calling the agent model (usually indicated by a message 
                        containing the word "unauthorized"), you may go to https://ollama.com and generate your own key. 
                        You can provide it in the configuration below. It will not be stored on our system and gets deleted 
                        when you end session (i.e., close your browser window).

                        **Note**: Each browser session maintains its own independent conversation state.
                        Uses stateless agent interaction for better concurrency support.
                        """
                    )

        with gr.Row():
            with gr.Column():
                # Settings panel
                gr.Markdown("### ⚙️ Agent Settings")

                with gr.Accordion("Configuration", open=True):
                    api_base_input = gr.Textbox(
                        label="API Base URL",
                        value=AGENT_MODEL_API_BASE,
                        placeholder="http://localhost:11434"
                    )

                    api_key_input = gr.Textbox(
                        label="API Key (only needed for remote models)",
                        value="",
                        type="password",
                        placeholder="Leave empty if not needed"
                    )

                    model_input = gr.Textbox(
                        label="Model",
                        value=f"ollama_chat/{AGENT_MODEL_NAME}" if "ollama_chat" not in AGENT_MODEL_NAME else AGENT_MODEL_NAME,
                        placeholder="ollama_chat/gpt-oss:20b"
                    )

                    temperature_input = gr.Slider(
                        label="Temperature",
                        minimum=0.0,
                        maximum=1.0,
                        value=0.2,
                        step=0.1
                    )

                    max_tokens_input = gr.Slider(
                        label="Max Tokens",
                        minimum=100,
                        maximum=8192,
                        value=6000,
                        step=10
                    )

                    num_ctx_input = gr.Number(
                        label="Context Window",
                        value=131072,
                        precision=0
                    )

                    max_papers_input = gr.Slider(
                        label="Max Papers to Retrieve",
                        minimum=0,
                        maximum=100,
                        value=50,
                        step=1
                    )

                    init_btn = gr.Button("Update Config", variant="primary")
                    init_status = gr.Textbox(label="Status", interactive=False)

                with gr.Accordion("System Prompt", open=False):
                    system_prompt_input = gr.Textbox(
                        label="System Prompt",
                        value=agent.get_system_prompt()["content"] if type(agent.get_system_prompt()) is dict else None,
                        placeholder="Enter custom system prompt here...",
                        lines=12
                    )
                    update_system_btn = gr.Button("Update System Prompt")
                    system_status = gr.Textbox(label="Status", interactive=False)

                with gr.Accordion("History & Save", open=False):
                    view_history_btn = gr.Button("📜 View Full History")
                    history_output = gr.Code(
                        label="Conversation History (JSON)",
                        language="json",
                        lines=10
                    )

                    save_filename_input = gr.Textbox(
                        label="Filename (optional)",
                        placeholder="Leave empty for auto-generated name",
                        value=""
                    )
                    save_status = gr.Textbox(label="Save Status", interactive=False)

        # Event handlers
        init_btn.click(
            fn=configure_agent,
            inputs=[
                api_base_input,
                api_key_input,
                model_input,
                temperature_input,
                max_tokens_input,
                num_ctx_input,
                max_papers_input,
                config_state
            ],
            outputs=[config_state, init_status]
        )

        # Chat submission
        submit_btn.click(
            fn=lambda msg_input, chatbot, config_state, messages_state: (yield from submit_message(msg_input, chatbot, config_state, messages_state, agent)),
            inputs=[msg_input, chatbot, config_state, messages_state],
            outputs=[chatbot, messages_state]
        ).then(
            fn=lambda: "",
            inputs=None,
            outputs=msg_input
        )

        msg_input.submit(
            fn=lambda msg_input, chatbot, config_state, messages_state: (yield from submit_message(msg_input, chatbot, config_state, messages_state, agent)),
            inputs=[msg_input, chatbot, config_state, messages_state],
            outputs=[chatbot, messages_state]
        ).then(
            fn=lambda: "",
            inputs=None,
            outputs=msg_input
        )

        # System prompt update
        update_system_btn.click(
            fn=lambda system_prompt_input, messages_state: update_system_prompt(system_prompt_input, messages_state, agent),
            inputs=[system_prompt_input, messages_state],
            outputs=[system_status, messages_state]
        )

        # History viewing
        view_history_btn.click(
            fn=view_history,
            inputs=messages_state,
            outputs=history_output
        )

        # Save history
        save_btn.click(
            fn=save_history,
            inputs=[save_filename_input, messages_state],
            outputs=save_status
        )

        # Clear chat
        clear_btn.click(
            fn=lambda config_state, messages_state: clear_chat(config_state, messages_state, agent),
            inputs=[config_state, messages_state],
            outputs=[save_status, chatbot, messages_state]
        )

    webapp.launch(
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860,  # Default Gradio port
        share=False,  # Set to True to create a public link
        show_error=True,
        debug=True,
    )


if __name__ == "__main__":
    # Setup logging (only needs to be done once globally)
    setup_logging(
        log_dir="logs",
        level=os.environ.get("LLM_AGENTS_LOG_LEVEL", "INFO")
    )

    main()

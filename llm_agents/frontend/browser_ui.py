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
from pathlib import Path
from typing import List, Tuple, Optional, Dict

from llm_agents.agents import NeurIPS2025Agent, DEFAULT_NEURIPS2025_AGENT_ARGS
from llm_agents.utils.logging import setup_logging
from llm_agents.utils.file_handlers import save_chat_history

LOGGER = logging.getLogger(__name__)

# Setup logging (only needs to be done once globally)
setup_logging(
    log_dir="logs",
    level=os.environ.get("LLM_AGENTS_LOG_LEVEL", "DEBUG")
)

AGENT = NeurIPS2025Agent(
    model=DEFAULT_NEURIPS2025_AGENT_ARGS["model"],
    api_base=DEFAULT_NEURIPS2025_AGENT_ARGS["api_base"],
    api_key=DEFAULT_NEURIPS2025_AGENT_ARGS["api_key"],
    llm_args=DEFAULT_NEURIPS2025_AGENT_ARGS["llm_args"],
    global_tool_args=DEFAULT_NEURIPS2025_AGENT_ARGS["global_tool_args"],
)
# This is needed to configure the tools
AGENT.setup_session()


def initialize_agent(
        api_base: str,
        api_key: str,
        model: str,
        temperature: float,
        max_tokens: int,
        num_ctx: int,
        max_num_papers: int,
        current_config: Dict
):
    """Initialize the agent with given configuration.

    Args:
        api_base: API base URL
        api_key: API key (optional)
        model: Model name
        temperature: Temperature parameter
        max_tokens: Max tokens to generate
        num_ctx: Context window size
        max_num_papers: Max papers to retrieve
        current_config: Current configuration dict
        messages: List of chat messages

    Returns:
        Tuple of (agent_instance, config_dict, status_message)
    """
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
    del current_config_to_print["api_key"]
    LOGGER.info(f"User-defined configuration saved. Config: {current_config_to_print}")

    return current_config, "✓ Agent initialized successfully!"


def chat_fn(
        message: str,
        history: List[Tuple[str, str]],
        config: Optional[Dict],
        messages: Optional[List[Dict]]
) -> Tuple[List[Tuple[str, str]], Optional[List[Dict]]]:
    """
    Handle chat interaction using stateless agent.

    Args:
        message: User's input message
        history: Chat history as list of (user_msg, assistant_msg) tuples
        config: Configuration dict with model, api_base, api_key, llm_args
        messages: Current conversation messages list

    Returns:
        Tuple of (updated_history, messages)
    """
    if not message.strip():
        return history, messages

    LOGGER.debug(f"USER PROMPT: {message}")
    assert messages is not None, "Make sure to properly initialize the agent by using the 'Model Config' on the right."

    try:

        # Create user message with timestamp
        user_message = {
            "role": "user",
            "content": message,
            "_ts": str(datetime.datetime.now(datetime.timezone.utc))
        }

        # Add user message to conversation
        messages.append(user_message)

        # Call the agent using stateless method
        messages = AGENT.interact_stateless(
            messages=messages,
            model=config["model"],
            api_base=config["api_base"],
            api_key=config["api_key"],
            llm_args=config["llm_args"]
        )

        # Get assistant response (last assistant message)
        assistant_content = None
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                assistant_content = msg["content"]
                break


        if assistant_content is None:
            assistant_content = "No response from agent."

        # Append to history
        history.append((message, assistant_content))

        LOGGER.info("Agent response generated successfully")
        return history, messages

    except Exception as e:
        LOGGER.error(f"Agent encountered an error: {e}")
        error_msg = f"❌ Error: {str(e)}"
        history.append((message, error_msg))
        return history, messages


def update_system_prompt(
    new_prompt: str,
    messages: Optional[List[Dict]]
) -> Tuple[str, Optional[List[Dict]]]:
    """Update the system prompt in the message history.

    Args:
        new_prompt: New system prompt
        config: Current configuration
        messages: Current message history

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
        messages = AGENT.set_system_prompt(new_system_prompt=new_prompt, messages=messages)

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
        messages: Optional[List[Dict]]
) -> Tuple[str, List, Optional[List[Dict]]]:
    """Clear the chat history in the UI and reset message list.

    Args:
        config: Current configuration
        messages: Current message history

    Returns:
        Tuple of (empty_history, status_message, agent_instance, config, empty_messages)
    """
    return "✓ Chat cleared!", [], [AGENT.get_system_prompt()]


def main():
    with gr.Blocks(title="SciAgent For NeurIPS 2025", theme=gr.themes.Soft()) as webapp:
        gr.Markdown("# 🤖 SciAgent For NeurIPS 2025")
        gr.Markdown("Initialize the agent with your settings, then start chatting!")
        gr.Markdown("*Each user session has its own independent conversation state. Enjoy!*")

        # Session state for agent instance, config, and messages
        config_state = gr.State(value=DEFAULT_NEURIPS2025_AGENT_ARGS)
        messages_state = gr.State(value=[AGENT.get_system_prompt()])

        with gr.Row():
            with gr.Column(scale=1):
                # Settings panel
                gr.Markdown("### ⚙️ Agent Settings")

                with gr.Accordion("Model Configuration", open=True):
                    api_base_input = gr.Textbox(
                        label="API Base URL",
                        value="http://localhost:11434",
                        placeholder="http://localhost:11434"
                    )

                    api_key_input = gr.Textbox(
                        label="API Key (optional)",
                        value="",
                        type="password",
                        placeholder="Leave empty if not needed"
                    )

                    model_input = gr.Textbox(
                        label="Model",
                        value="ollama_chat/gpt-oss:20b",
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

                    init_btn = gr.Button("🚀 Initialize Agent", variant="primary")
                    init_status = gr.Textbox(label="Status", interactive=False)

                with gr.Accordion("System Prompt", open=False):
                    system_prompt_input = gr.Textbox(
                        label="System Prompt",
                        value=AGENT.get_system_prompt()["content"] if type(AGENT.get_system_prompt()) is dict else None,
                        placeholder="Enter custom system prompt here...",
                        lines=8
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

            with gr.Column(scale=2):
                # Main chat interface
                chatbot = gr.Chatbot(
                    label="Conversation Trail",
                    height=500,
                    type="tuples",
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

        # Event handlers
        init_btn.click(
            fn=initialize_agent,
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
            fn=chat_fn,
            inputs=[msg_input, chatbot, config_state, messages_state],
            outputs=[chatbot, messages_state]
        )

        msg_input.submit(
            fn=chat_fn, # lambda msg_input, chatbot, config_state, message_state: chat_fn(msg_input, clean_history(chatbot), config_state, messages_state),
            inputs=[msg_input, chatbot, config_state, messages_state],
            outputs=[chatbot, messages_state]
        )

        # System prompt update
        update_system_btn.click(
            fn=update_system_prompt,
            inputs=[
                system_prompt_input,
                messages_state
            ],
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
            fn=clear_chat,
            inputs=[config_state, messages_state],
            outputs=[save_status, chatbot, messages_state]
        )

        # Help text at bottom
        gr.Markdown("""
        ### 📖 Usage Guide

        1. **Initialize**: Configure settings and click "Initialize Agent"
        2. **Chat**: Type messages and press Enter or click Send
        3. **System Prompt**: Customize the agent's behavior via System Prompt panel
        4. **History**: View or save your conversation using the History & Save panel
        5. **Clear**: Start a fresh conversation with the Clear Chat button

        All features from the terminal UI are available here!

        **Note**: Each browser session maintains its own independent conversation state.
        Uses stateless agent interaction for better concurrency support.
        """)

    webapp.launch(
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860,  # Default Gradio port
        share=False,  # Set to True to create a public link
        show_error=True,
        debug=True,
    )


if __name__ == "__main__":
    main()

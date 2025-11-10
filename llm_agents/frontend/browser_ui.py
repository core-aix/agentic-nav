"""
Gradio web UI that interacts with an agent implementation.

Features matching terminal UI:
- Multi-turn chat with Markdown rendering
- System prompt editing
- View conversation history
- Save chat history to file
- All model configuration options
- Clear chat functionality
- **Per-user agent instances using session state**
"""
import gradio as gr
import os
import datetime
import logging
import json
from pathlib import Path
from typing import List, Tuple, Optional

from llm_agents.agents import NeurIPS2025Agent
from llm_agents.utils.logging import setup_logging
from llm_agents.utils.file_handlers import save_chat_history

LOGGER = logging.getLogger(__name__)


def initialize_agent(
    api_base: str,
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
    num_ctx: int,
    max_num_papers: int,
    current_agent: Optional[NeurIPS2025Agent]
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
        current_agent: Current agent instance (if any)

    Returns:
        Tuple of (agent_instance, status_message)
    """
    # Setup logging (only needs to be done once globally)
    setup_logging(
        log_dir="logs",
        level=os.environ.get("LLM_AGENTS_LOG_LEVEL", "INFO")
    )

    LOGGER.info(f"Agent runtime started via Gradio UI for session")

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

    # Create new agent instance
    agent = NeurIPS2025Agent(
        model=model,
        api_base=api_base,
        api_key=api_key,
        llm_args=llm_config,
        global_tool_args=tool_args,
    )

    agent.setup_session()
    LOGGER.info("Agent session initialized for user")

    return agent, "✓ Agent initialized successfully!"


def chat_fn(message: str, history: List[Tuple[str, str]], agent: Optional[NeurIPS2025Agent]) -> Tuple[List[Tuple[str, str]], str, Optional[NeurIPS2025Agent]]:
    """
    Handle chat interaction.

    Args:
        message: User's input message
        history: Chat history as list of (user_msg, assistant_msg) tuples
        agent: Current agent instance

    Returns:
        Tuple of (updated_history, empty_string_to_clear_input, agent_instance)
    """
    if agent is None:
        return history + [(message, "⚠️ Please initialize the agent first using the settings panel.")], "", agent

    if not message.strip():
        return history, "", agent

    LOGGER.debug(f"USER PROMPT: {message}")

    try:
        # Create message with timestamp
        next_message = {
            "role": "user",
            "content": message,
            "_ts": str(datetime.datetime.now(datetime.UTC))
        }

        # Call the agent
        agent.interact(message=next_message)

        # Get assistant response
        assistant_message = agent.get_most_recent_assistant_message()
        assistant_content = assistant_message["content"]

        # Append to history
        history.append((message, assistant_content))

        LOGGER.info("Agent response generated successfully")
        return history, "", agent

    except Exception as e:
        LOGGER.error(f"Agent encountered an error: {e}")
        error_msg = f"❌ Error: {str(e)}"
        history.append((message, error_msg))
        return history, "", agent


def update_system_prompt(new_prompt: str, agent: Optional[NeurIPS2025Agent]) -> Tuple[str, Optional[NeurIPS2025Agent]]:
    """Update the agent's system prompt.

    Args:
        new_prompt: New system prompt
        agent: Current agent instance

    Returns:
        Tuple of (status_message, agent_instance)
    """
    if agent is None:
        return "⚠️ Please initialize the agent first.", agent

    if not new_prompt.strip():
        return "⚠️ System prompt cannot be empty.", agent

    try:
        agent.set_system_prompt(new_system_prompt=new_prompt)
        LOGGER.info("System prompt updated")
        return "✓ System prompt updated successfully!", agent
    except Exception as e:
        LOGGER.error(f"Error updating system prompt: {e}")
        return f"❌ Error: {str(e)}", agent


def view_history(agent: Optional[NeurIPS2025Agent]) -> str:
    """View the full conversation history in JSON format.

    Args:
        agent: Current agent instance

    Returns:
        JSON formatted history string
    """
    if agent is None:
        return "⚠️ Please initialize the agent first."

    try:
        history = agent.get_history()
        # Format as pretty JSON
        return json.dumps(history, indent=2, ensure_ascii=False)
    except Exception as e:
        LOGGER.error(f"Error viewing history: {e}")
        return f"❌ Error: {str(e)}"


def save_history(filename: str, agent: Optional[NeurIPS2025Agent]) -> str:
    """Save chat history to a JSON file.

    Args:
        filename: Optional filename
        agent: Current agent instance

    Returns:
        Status message
    """
    if agent is None:
        return "⚠️ Please initialize the agent first."

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
        save_chat_history(agent.get_history(), filename)

        LOGGER.info(f"Chat history saved to {filename}")
        return f"✓ Chat history saved to: {filename}"

    except Exception as e:
        LOGGER.error(f"Error saving history: {e}")
        return f"❌ Error: {str(e)}"


def clear_chat(agent: Optional[NeurIPS2025Agent]) -> Tuple[List, str, Optional[NeurIPS2025Agent]]:
    """Clear the chat history in the UI and reinitialize agent session.

    Args:
        agent: Current agent instance

    Returns:
        Tuple of (empty_history, status_message, agent_instance)
    """
    if agent is not None:
        try:
            agent.setup_session()  # Reinitialize session
            LOGGER.info("Chat cleared and session reinitialized")
            return [], "✓ Chat cleared!", agent
        except Exception as e:
            LOGGER.error(f"Error clearing chat: {e}")
            return [], f"❌ Error: {str(e)}", agent

    return [], "⚠️ Agent not initialized.", agent


def main():
    with gr.Blocks(title="SciAgent For NeurIPS 2025", theme=gr.themes.Soft()) as webapp:
        gr.Markdown("# 🤖 SciAgent For NeurIPS 2025")
        gr.Markdown("Initialize the agent with your settings, then start chatting!")
        gr.Markdown("*Each user session has its own independent agent instance. Enjoy!*")

        # Session state for agent instance
        agent_state = gr.State(value=None)

        with gr.Row():
            with gr.Column(scale=2):
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

            with gr.Column(scale=1):
                # Settings panel
                gr.Markdown("### ⚙️ Agent Settings")

                with gr.Accordion("Model Configuration", open=False):
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
                        maximum=2.0,
                        value=0.2,
                        step=0.1
                    )

                    max_tokens_input = gr.Slider(
                        label="Max Tokens",
                        minimum=100,
                        maximum=32000,
                        value=6000,
                        step=100
                    )

                    num_ctx_input = gr.Number(
                        label="Context Window",
                        value=131072,
                        precision=0
                    )

                    max_papers_input = gr.Slider(
                        label="Max Papers to Retrieve",
                        minimum=1,
                        maximum=200,
                        value=50,
                        step=1
                    )

                    init_btn = gr.Button("🚀 Initialize Agent", variant="primary")
                    init_status = gr.Textbox(label="Status", interactive=False)

                with gr.Accordion("System Prompt", open=False):
                    system_prompt_input = gr.Textbox(
                        label="System Prompt",
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

        # Event handlers
        init_btn.click(
            fn=initialize_agent,
            inputs=[
                api_base_input, api_key_input, model_input,
                temperature_input, max_tokens_input, num_ctx_input,
                max_papers_input, agent_state
            ],
            outputs=[agent_state, init_status]
        )

        # Chat submission
        submit_btn.click(
            fn=chat_fn,
            inputs=[msg_input, chatbot, agent_state],
            outputs=[chatbot, msg_input, agent_state]
        )

        msg_input.submit(
            fn=chat_fn,
            inputs=[msg_input, chatbot, agent_state],
            outputs=[chatbot, msg_input, agent_state]
        )

        # System prompt update
        update_system_btn.click(
            fn=update_system_prompt,
            inputs=[system_prompt_input, agent_state],
            outputs=[system_status, agent_state]
        )

        # History viewing
        view_history_btn.click(
            fn=view_history,
            inputs=agent_state,
            outputs=history_output
        )

        # Save history
        save_btn.click(
            fn=save_history,
            inputs=[save_filename_input, agent_state],
            outputs=save_status
        )

        # Clear chat
        clear_btn.click(
            fn=clear_chat,
            inputs=agent_state,
            outputs=[chatbot, save_status, agent_state]
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
        
        **Note**: Each browser session maintains its own independent agent instance.
        """)

    webapp.launch(
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860,  # Default Gradio port
        share=False,  # Set to True to create a public link
        show_error=True,
        debug=True
    )

if __name__ == "__main__":
    main()

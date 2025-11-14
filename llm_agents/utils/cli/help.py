

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
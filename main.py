from tools.rag_search import search_paper_titles_and_abstracts  # <- the tool we expose
from utils.tool_chat import ToolChat
import sys

if __name__ == "__main__":
    chat = ToolChat()

    system = {"role": "system", "content": "You are an assistant who can help browsing NeurIPS 2025 papers. "
    "You are provided with a search tool that can search all accepted papers of NeurIPS 2025. "
    "However, note that the search tool can only search for paper titles and abstracts; it cannot search for anything else. "
    "So, there can be cases where you DO NOT want to use the tool. "
    "If your answer includes a list of papers, cite titles, abstracts, and OpenReview URLs in your answers."}

    msgs = [system]
    
    while True:
        user = {"role": "user", "content": None}

        if len(sys.argv) > 1:
            user["content"] = " ".join(sys.argv[1:])
            multiturn = False
        else:
            user["content"] = input("Enter user prompt: ")
            multiturn = True

        msgs.append(user)

        msgs = chat.tool_loop(msgs, tool_funcs=[search_paper_titles_and_abstracts])
    
        if not multiturn:
            break
    


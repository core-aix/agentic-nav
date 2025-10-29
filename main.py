from tools.rag_search import search_papers  # <- the tool we expose
from utils.tool_chat import ToolChat
import sys

if __name__ == "__main__":
    chat = ToolChat()
    system = {"role": "system", "content": "You are a research assistant. Use the search tool when helpful, but you may want to search for a larger number of relevant papers, then select from those that are most relevant to the user's query by further examining the titles and abstracts of the papers found. Cite titles, abstracts, and OpenReview URLs in your answers."}
    user = {"role": "user", "content": "Find papers on large language models. Give me the title, abstract, and OpenReview URLs."}

    if len(sys.argv) > 1:
        user["content"] = " ".join(sys.argv[1:])
    else:
        user["content"] = input("Enter user prompt: ")

    resp = chat.tool_loop([system, user], registry={"search_papers": search_papers})
    print(resp["choices"][0]["message"]["content"])

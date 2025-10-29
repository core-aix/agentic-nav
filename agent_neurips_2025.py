from tools.rag_search import search_papers  # <- the tool we expose
from utils.tool_chat import ToolChat
import sys

system = {"role": "system", "content": "You are an assistant who can help browsing NeurIPS 2025 papers. "
"You are provided with a search tool that can search all accepted papers of NeurIPS 2025. "
"However, note that the search tool only takes paper titles and abstracts as input keywords; it cannot take anything else as the input keywords. "
"However, the output of the search includes various metadata fields such as authors, affiliations, and session times. "
"If your answer includes a list of papers, cite titles, abstracts, and OpenReview URLs in your answers."}

msgs = [system]

def agent_respond(messages):
    chat = ToolChat()
    return chat.tool_loop(messages, tool_funcs=[search_papers])[-1]

# if __name__ == "__main__":
#     chat = ToolChat()
    
#     while True:
#         user = {"role": "user", "content": None}

#         if len(sys.argv) > 1:
#             user["content"] = " ".join(sys.argv[1:])
#             multiturn = False
#         else:
#             user["content"] = input("User (please enter): ")
#             multiturn = True

#         msgs.append(user)

#         msgs = chat.tool_loop(msgs, tool_funcs=[search_paper_titles_and_abstracts])
    
#         if not multiturn:
#             break
    


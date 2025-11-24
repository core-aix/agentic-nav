import json


def save_chat_history(messages, path):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(messages, f, indent=2, ensure_ascii=False)
        print(f"Saved to {path}")
    except Exception as e:
        print("Save failed:", e)
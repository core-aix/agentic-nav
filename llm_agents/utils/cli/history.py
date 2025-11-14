
def show_history(messages):
    for i, m in enumerate(messages):
        ts = m.get("_ts", "")
        role = m.get("role", "")
        content = m.get("content", "")
        header = f"[{i}] {role} {ts}"
        print(header)
        print("-" * len(header))
        print(content)
        print()
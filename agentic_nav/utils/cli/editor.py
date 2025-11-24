import os
import tempfile


def open_editor(initial_text=""):
    editor = os.environ.get("EDITOR")
    if not editor:
        # Minimal sensible defaults
        if os.name == "nt":
            editor = "notepad"
        else:
            editor = "nano"
    with tempfile.NamedTemporaryFile(suffix=".md", delete=False, mode="w+", encoding="utf-8") as tf:
        path = tf.name
        tf.write(initial_text)
        tf.flush()
    try:
        # Open editor and wait
        rc = os.system(f'{editor} "{path}"')
        if rc != 0:
            print(f"(editor exit code {rc})")
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
    finally:
        try:
            os.unlink(path)
        except Exception:
            pass
    return content.strip()
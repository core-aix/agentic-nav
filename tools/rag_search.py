from __future__ import annotations
from typing import List, Dict, Any
import json
from pathlib import Path
import numpy as np
import faiss
from litellm import embedding
import time

API_BASE = "http://localhost:11434"
EMBED_MODEL = "ollama/nomic-embed-text"
INDEX_DIR = "rag_index_json"

# TODO: Make this configurable

# Load once at import (fast)
_FAISS = faiss.read_index(str(Path(INDEX_DIR) / "faiss.index"))
_META = json.loads(Path(INDEX_DIR, "meta.json").read_text())

def _embed_query(q: str) -> np.ndarray:
    resp = embedding(model=EMBED_MODEL, input=[q], api_base=API_BASE)
    v = np.array(resp["data"][0]["embedding"], dtype="float32")
    v = v / (np.linalg.norm(v) + 1e-12)
    return v.reshape(1, -1)

def _snippet(text: str, max_chars: int = 240) -> str:
    text = " ".join(text.split())
    return text if len(text) <= max_chars else text[:max_chars-1] + "…"

def search_papers(keywords_to_search: str, num_records: int = 20) -> Dict[str, Any]:
    """
    Search most relevant papers over title+abstract. The number of records returned can be controlled via `num_records`.
    Returns JSON with 'keywords_to_search' and 'results' (the results include metadata of the paper).
    """
    try:
        num_records = int(num_records)
    except Exception:
        num_records = 20
    
    if len(keywords_to_search) == 0 or num_records <= 0:
        return {"query": keywords_to_search, "results": []}

    # start_time = time.time()
    qv = _embed_query(keywords_to_search)
    scores, idxs = _FAISS.search(qv, num_records)
    scores, idxs = scores[0].tolist(), idxs[0].tolist()
    # print(f"Search for '{keywords_to_search}' took {time.time() - start_time:.3f}s")

    results = []
    for score, i in zip(scores, idxs):
        if i == -1:
            continue
        m = _META[i]
        text = f"{m['name']}\n\n{m['abstract']}"
        results.append(m)
    return {"query": keywords_to_search, "results": results}

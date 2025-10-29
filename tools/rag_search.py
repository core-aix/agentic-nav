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

def search_papers(query: str, k: int = 10) -> Dict[str, Any]:
    """
    Search top-k relevant papers over title+abstract. You need to specify k as the total number of results that you would like to get.
    Returns JSON with 'query' and 'results' (the results include metadata of the paper).
    """
    try:
        k = int(k)
    except Exception:
        k = 5
    
    start_time = time.time()
    qv = _embed_query(query)
    scores, idxs = _FAISS.search(qv, k)
    scores, idxs = scores[0].tolist(), idxs[0].tolist()
    print(f"Search for '{query}' took {time.time() - start_time:.3f}s")

    results = []
    for score, i in zip(scores, idxs):
        if i == -1:
            continue
        m = _META[i]
        text = f"{m['name']}\n\n{m['abstract']}"
        results.append(m)
    return {"query": query, "results": results}

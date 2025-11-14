# rag_build_json.py
from __future__ import annotations
import time
from typing import List
import json
import os
from pathlib import Path

import numpy as np
import faiss
from litellm import embedding
from tqdm import tqdm

from llm_agents.utils import batch_embed_documents

OLLAMA_LOCAL_EMBEDDING_MODEL_NAME = os.environ.get("OLLAMA_LOCAL_EMBEDDING_MODEL", "nomic-embed-text")
OLLAMA_LOCAL_EMBEDDING_MODEL_API_BASE = os.environ.get("OLLAMA_LOCAL_EMBEDDING_MODEL_API_BASE", "http://localhost:11434")
INDEX_DIR = "../rag_index_json"


def _load_papers(json_path: str) -> List:
    data = json.loads(Path(json_path).read_text())
    papers = data["results"]
    return papers

def _embed(texts: List[str], batch: int = 1) -> np.ndarray:
    vecs: List[List[float]] = []
    for i in tqdm(range(0, len(texts), batch)):
        chunk = texts[i:i+batch]
        retry = 0
        max_retries = 100
        while retry < max_retries:
            try:
                resp = embedding(
                    model=OLLAMA_LOCAL_EMBEDDING_MODEL_NAME,
                    input=chunk,
                    api_base=OLLAMA_LOCAL_EMBEDDING_MODEL_API_BASE
                )
                break
            except Exception as e:
                print(f"Error during embedding batch {i}-{i+batch}: {e}")
                # print(chunk)
                retry += 1
                if retry >= max_retries:
                    raise Exception("Max retries reached for embedding.")
                print(f"Retrying {retry} ...")
                time.sleep(0.1)

        vecs.extend([d["embedding"] for d in resp["data"]])
        # time.sleep(5)  # avoid rate limit
    arr = np.array(vecs, dtype="float32")
    # cosine similarity: normalize to unit length and use IndexFlatIP
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    return arr / norms

def build_index(json_path: str, out_dir: str = INDEX_DIR) -> None:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    papers = _load_papers(json_path)

    texts = [f"{p['name']}\n\n{p['abstract']}" if "name" in p and "abstract" in p else "" for p in papers]
    X = batch_embed_documents(
        texts,
        embedding_model=OLLAMA_LOCAL_EMBEDDING_MODEL_NAME,
        api_base=OLLAMA_LOCAL_EMBEDDING_MODEL_API_BASE,
        batch_size=32
    )
    dim = X.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(X)

    faiss.write_index(index, str(out / "faiss.index"))
    # metadata alongside embeddings
    (out / "meta.json").write_text(json.dumps(papers, ensure_ascii=False, indent=4, sort_keys=True))
    print(f"Indexed {len(papers)} papers → {out.resolve()}")

if __name__ == "__main__":
    # build_index("data/papers_test.json")
    build_index("../data/neurips-2025-orals-posters.json")

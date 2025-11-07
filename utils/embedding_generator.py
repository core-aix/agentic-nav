import litellm
import numpy as np
import time

from litellm import embedding
from tqdm import tqdm

from typing import List


def batch_embed_documents(
    texts: List[str],
    batch_size: int = 1,
    embedding_model: str = "ollama/nomic-embed-text",
    api_base: str = "http://localhost:11434"
) -> np.ndarray:

    if None in texts:
        print(f"WARNING: Detected documents with 'None' values. Replacing 'None' with an empty string...")
        texts = ['' if doc is None else doc for doc in texts]

    vecs: List[List[float]] = []
    for i in tqdm(range(0, len(texts), batch_size)):
        chunk = texts[i:i + batch_size]
        try:
            resp = embedding(
                model=embedding_model,
                input=chunk,
                api_base=api_base,
                **{"num_ctx": 2048}
            )
        except Exception as e:
            print(f"Error during embedding batch {i}-{i + batch_size}: {e}. Falling back to single sample processing")
            resp = []
            ctr = i
            for sample in chunk:
                try:
                    resp.append(
                        embedding(
                            model=embedding_model,
                            input=sample,
                            api_base=api_base,
                            **{"num_ctx": 2048}
                        )
                    )
                except litellm.BadRequestError:
                    print(f"Encountered error processing paper #{ctr}. Please inspect and retry afterwards.")

            print(f"RESP: {resp}")

        vecs.extend([d["embedding"] for d in resp["data"]])
        # time.sleep(5)  # avoid rate limit
    arr = np.array(vecs, dtype="float32")
    # cosine similarity: normalize to unit length and use IndexFlatIP
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    return arr / norms


if __name__ == "__main__":
    res = batch_embed_documents(
        texts=[
            "test1",
            "test2",
            "test3",
            "test4",
            "test5"
        ],
        batch_size=1
    )
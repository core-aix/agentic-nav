import logging

import litellm
import numpy as np
import spaces

from litellm import embedding
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from typing import List


LOGGER = logging.getLogger(__name__)
local_embedding_model = None


class EmbeddingResponse:
    def __init__(self, embeddings):
        self.data = [
            type('obj', (), {
                'embedding': emb.tolist(),
                'index': idx
            })()
            for idx, emb in enumerate(embeddings)
        ]

@spaces.GPU
def embed_hf_spaces(model, input, embedding_model_name: str = "nomic-ai/nomic-embed-text-v1.5", api_base=None, **kwargs):
    """
    Drop-in replacement for litellm.embedding()

    Args:
        model: Model name (ignored since we use the loaded model)
        input: Single string or list of strings to embed
        api_base: Ignored for local embedding
        **kwargs: Additional args like num_ctx (ignored for local)

    Returns:
        Object with same structure as LiteLLM response
    """
    global local_embedding_model
    local_embedding_model = SentenceTransformer(
        embedding_model_name,
        trust_remote_code=True
    )

    texts = [input] if isinstance(input, str) else input
    embeddings = model.encode(
        texts,
        convert_to_tensor=True,
        show_progress_bar=False,
        normalize_embeddings=True
    )

    embeddings_np = embeddings.cpu().numpy()

    return EmbeddingResponse(embeddings_np)


def embedding_fn(model, input, api_base, **kwargs):
    if api_base == "hf_spaces_local":
        embed_hf_spaces(input=input, embedding_model_name=model, api_base=api_base, **kwargs)
    elif api_base == "http://localhost:11435" or api_base == "https://ollama.com":
        return embed_hf_spaces(input=input, embedding_model_name=model, api_base=api_base, **kwargs)
    else:
        raise NotImplementedError("Unknown api_base for provider {api_base}. Available options: hf_spaces_local, ollama local (http://localhost:11435), ollama cloud (https://ollama.com)")


def batch_embed_documents(
    texts: List[str],
    batch_size: int = 1,
    embedding_model: str = f"ollama/nomic-embed-text",
    api_base: str = "http://localhost:11435",
) -> np.ndarray:

    if not texts:
        return np.array([], dtype="float32").reshape(0, 0)

    if None in texts:
        LOGGER.warning(f"WARNING: Detected documents with 'None' values. Replacing 'None' with an empty string...")
        texts = ['' if doc is None else doc for doc in texts]

    vecs: List[List[float]] = []
    for i in tqdm(range(0, len(texts), batch_size)):
        chunk = texts[i:i + batch_size]
        try:
            resp = embedding_fn(
                model=embedding_model,
                input=chunk,
                api_base=api_base,
                **{"num_ctx": 2048}
            )
        except Exception as e:
            LOGGER.error(f"Error during embedding batch {i}-{i + batch_size}: {e}. Falling back to single sample processing")
            individual_responses = []
            ctr = i
            for sample in chunk:
                try:
                    individual_responses.append(
                        embedding(
                            model=embedding_model,
                            input=sample,
                            api_base=api_base,
                            **{"num_ctx": 2048}
                        )
                    )
                except litellm.BadRequestError:
                    LOGGER.error(f"Encountered error processing paper #{ctr}. Please inspect and retry afterwards.")
                ctr += 1

            LOGGER.debug(f"Single sample response from embedding model: {individual_responses}")
            
            # Extract embeddings from individual responses
            for individual_resp in individual_responses:
                vecs.extend([d["embedding"] for d in individual_resp["data"]])
        else:
            # Normal batch processing
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
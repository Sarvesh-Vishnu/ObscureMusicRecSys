"""Sentence-transformer embeddings + FAISS index over the catalog."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from . import config
from .catalog import Track

_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_model = None


def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def encode(texts: Sequence[str]) -> np.ndarray:
    model = get_model()
    vecs = model.encode(list(texts), normalize_embeddings=True, convert_to_numpy=True)
    return vecs.astype("float32")


def build_index(tracks: Sequence[Track],
                index_path: Path = config.INDEX_PATH,
                vec_path: Path = config.EMBEDDINGS_PATH) -> None:
    import faiss
    vecs = encode([t.tag_text() for t in tracks])
    index = faiss.IndexFlatIP(vecs.shape[1])  # inner product on normalized vectors == cosine
    index.add(vecs)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    np.save(vec_path, vecs)


def load_index(index_path: Path = config.INDEX_PATH):
    import faiss
    return faiss.read_index(str(index_path))


def query(text: str, top_k: int = 25) -> tuple[np.ndarray, np.ndarray]:
    """Return (scores, indices) for nearest catalog neighbors of free-text query."""
    index = load_index()
    qvec = encode([text])
    scores, idx = index.search(qvec, top_k)
    return scores[0], idx[0]

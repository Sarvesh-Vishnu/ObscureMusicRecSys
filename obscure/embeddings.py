"""Sentence-transformer embeddings + FAISS index over the catalog.

Two index modes:
- Flat (IndexFlatIP): exact cosine; right choice for <10K tracks.
- Product Quantized (IndexPQ): ~24-64x smaller, <1% recall loss; right for
  catalogs >10K. Requires >=256 vectors to train.

The catalog id mapping (row_id -> track_id) is stored alongside the index so
we can resolve hits back to Track objects without depending on insertion order.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

from . import config
from .catalog import Track

_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_model = None

ID_MAP_PATH = config.DATA_DIR / "catalog.ids.json"


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
                vec_path: Path = config.EMBEDDINGS_PATH,
                use_pq: bool = False) -> None:
    import faiss
    vecs = encode([t.tag_text() for t in tracks])
    d = vecs.shape[1]

    if use_pq:
        # m = subquantizers (must divide d); nbits = log2 of centroids per subq
        m = 48 if d % 48 == 0 else (d // 8)
        index = faiss.IndexPQ(d, m, 8, faiss.METRIC_INNER_PRODUCT)
        index.train(vecs)
    else:
        index = faiss.IndexFlatIP(d)

    index.add(vecs)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    np.save(vec_path, vecs)

    ID_MAP_PATH.write_text(json.dumps([t.track_id for t in tracks]))


def load_index(index_path: Path = config.INDEX_PATH):
    import faiss
    return faiss.read_index(str(index_path))


def load_id_map() -> list[str]:
    if not ID_MAP_PATH.exists():
        return []
    return json.loads(ID_MAP_PATH.read_text())


def query(text: str, top_k: int = 25) -> tuple[np.ndarray, np.ndarray]:
    """Return (scores, indices) for nearest catalog neighbors of free-text query.

    Indices are *row positions* in the index — caller resolves to track_id via
    `load_id_map()`.
    """
    index = load_index()
    qvec = encode([text])
    scores, idx = index.search(qvec, top_k)
    return scores[0], idx[0]


def add_to_index(tracks: Sequence[Track]) -> None:
    """Incrementally append tracks to the existing index. Used by warm cache.

    Note: IndexPQ does NOT support incremental adds after training without a
    retrain. For now, incremental adds require a flat index. If we want PQ at
    large scale with a live cache, we'd swap to IndexIVFPQ which supports it.
    """
    import faiss
    index = load_index()
    vecs = encode([t.tag_text() for t in tracks])
    index.add(vecs)
    faiss.write_index(index, str(config.INDEX_PATH))

    ids = load_id_map() + [t.track_id for t in tracks]
    ID_MAP_PATH.write_text(json.dumps(ids))

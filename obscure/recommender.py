"""Recommendation: blend semantic similarity with an obscurity prior, then
optionally re-rank with a LinUCB contextual bandit learned from user feedback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import catalog as cat
from . import embeddings as emb
from . import features as feats


@dataclass
class Recommendation:
    track: cat.Track
    similarity: float       # cosine similarity to seed (0..1)
    obscurity: float        # 0..1, higher = more obscure
    score: float            # blended ranking score
    features: np.ndarray    # per-candidate feature vector (used for bandit update)
    bandit_bonus: float = 0.0  # LinUCB contribution (0 if no bandit applied)


def recommend(seed_text: str,
              tracks: list[cat.Track],
              k: int = 10,
              obscurity_weight: float = 0.4,
              exclude_titles: Optional[set[str]] = None,
              bandit=None,
              bandit_weight: float = 0.5,
              language_pivot: Optional[set[str]] = None) -> list[Recommendation]:
    """Return top-k recommendations.

    Ranking is:
        score = (1-w_obs)·similarity + w_obs·obscurity + w_bandit·bandit_score

    `tracks` should be the SAME ordered list used to build the FAISS index.
    The caller (app) hydrates it from `Catalog.all_tracks()`.
    """
    exclude = exclude_titles or set()

    # Resolve FAISS row indices to tracks via the persisted id-map. This is
    # robust to ordering changes between index build and now.
    id_map = emb.load_id_map()
    by_id = {t.track_id: t for t in tracks}

    scores, idx = emb.query(seed_text, top_k=max(k * 8, 80))

    candidates: list[Recommendation] = []
    for sim, i in zip(scores, idx):
        if i < 0:
            continue
        # row -> track_id -> Track
        if i >= len(id_map):
            continue
        t = by_id.get(id_map[i])
        if t is None:
            continue
        if t.title.lower() in exclude:
            continue
        if language_pivot:
            langs = feats.detect_languages(t)
            if not langs or (langs & language_pivot):
                continue

        sim_f = float(max(0.0, min(1.0, (sim + 1) / 2)))
        obs = cat.obscurity_score(t)
        blended = (1 - obscurity_weight) * sim_f + obscurity_weight * obs
        fv = feats.feature_vector(t, sim_f)
        candidates.append(Recommendation(
            track=t, similarity=sim_f, obscurity=obs, score=blended,
            features=fv,
        ))

    if not candidates:
        return []

    if bandit is not None and bandit_weight > 0:
        X = np.stack([c.features for c in candidates])
        ucb = bandit.score(X)
        ucb_min, ucb_max = float(ucb.min()), float(ucb.max())
        denom = (ucb_max - ucb_min) or 1.0
        ucb_norm = (ucb - ucb_min) / denom
        for c, u_n in zip(candidates, ucb_norm):
            c.bandit_bonus = float(u_n)
            c.score = c.score + bandit_weight * float(u_n)

    candidates.sort(key=lambda r: r.score, reverse=True)
    return candidates[:k]

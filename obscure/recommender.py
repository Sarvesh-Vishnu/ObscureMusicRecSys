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
              exclude_titles: set[str] | None = None,
              bandit=None,
              bandit_weight: float = 0.5,
              language_pivot: set[str] | None = None) -> list[Recommendation]:
    """Return top-k recommendations.

    Ranking is:
        score = (1-w_obs)·similarity + w_obs·obscurity + w_bandit·bandit_score

    `bandit`           Optional LinUCBReranker; if provided, its UCB scores are
                       added to the blended score with weight `bandit_weight`.
    `language_pivot`   If provided, exclude candidates whose tags match ANY of
                       these language buckets — implements "find me something in
                       a different language". Pass `feats.detect_languages(seed)`
                       to surface cross-language neighbors.
    """
    exclude = exclude_titles or set()
    scores, idx = emb.query(seed_text, top_k=max(k * 8, 80))

    candidates: list[Recommendation] = []
    for sim, i in zip(scores, idx):
        if i < 0 or i >= len(tracks):
            continue
        t = tracks[i]
        if t.title.lower() in exclude:
            continue
        if language_pivot:
            langs = feats.detect_languages(t)
            # require at least one language tag AND no overlap with seed's languages
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
        # normalize bandit scores to [0,1] within this candidate set so weight
        # is meaningful regardless of bandit calibration
        ucb_min, ucb_max = float(ucb.min()), float(ucb.max())
        denom = (ucb_max - ucb_min) or 1.0
        ucb_norm = (ucb - ucb_min) / denom
        for c, u, u_n in zip(candidates, ucb, ucb_norm):
            c.bandit_bonus = float(u_n)
            c.score = c.score + bandit_weight * float(u_n)

    candidates.sort(key=lambda r: r.score, reverse=True)
    return candidates[:k]

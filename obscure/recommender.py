"""Recommendation: blend semantic similarity with an obscurity prior."""

from __future__ import annotations

from dataclasses import dataclass

from . import catalog as cat
from . import embeddings as emb


@dataclass
class Recommendation:
    track: cat.Track
    similarity: float       # cosine similarity to seed (0..1)
    obscurity: float        # 0..1, higher = more obscure
    score: float            # blended ranking score


def recommend(seed_text: str,
              tracks: list[cat.Track],
              k: int = 10,
              obscurity_weight: float = 0.4,
              exclude_titles: set[str] | None = None) -> list[Recommendation]:
    """Return top-k blended recommendations.

    `obscurity_weight` of 0 ranks purely by similarity (a normal recommender);
    1.0 ranks purely by obscurity; the default 0.4 leans toward "relevant but
    long-tail" — the actual product hypothesis of this app.
    """
    exclude = exclude_titles or set()
    scores, idx = emb.query(seed_text, top_k=max(k * 5, 50))

    candidates: list[Recommendation] = []
    for sim, i in zip(scores, idx):
        if i < 0 or i >= len(tracks):
            continue
        t = tracks[i]
        if t.title.lower() in exclude:
            continue
        sim_f = float(max(0.0, min(1.0, (sim + 1) / 2)))  # [-1,1] -> [0,1]
        obs = cat.obscurity_score(t)
        blended = (1 - obscurity_weight) * sim_f + obscurity_weight * obs
        candidates.append(Recommendation(track=t, similarity=sim_f,
                                         obscurity=obs, score=blended))

    candidates.sort(key=lambda r: r.score, reverse=True)
    return candidates[:k]

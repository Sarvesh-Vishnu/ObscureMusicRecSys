"""Recommendation: blend semantic similarity with an obscurity prior, then
optionally re-rank with a LinUCB contextual bandit learned from user feedback.

Ranking runs in three stages, each a guard against a failure the v2 app had:

1. **Relevance pool.** Only the `pool_size` most similar candidates are eligible.
   Obscurity can reorder relevant tracks; it can no longer pull in irrelevant
   ones (v2 surfaced a cumbia band for "Carnatic violin").
2. **Common scale.** Similarity and obscurity are min-max scaled *within the
   pool* before blending. Raw cosine similarity spans ~0.15 across the top 80
   while obscurity spans ~0.5, so a raw blend let obscurity dominate ~2:1 at
   the default weight.
3. **Confidence-gated bandit.** The LinUCB term is off until the bandit has seen
   `BANDIT_MIN_EVENTS` ratings and ramps to full weight by `BANDIT_FULL_EVENTS`.
   With zero data its UCB bonus is just the norm of the feature vector, which
   reshuffled the list arbitrarily.

Finally, at most `max_per_artist` tracks per artist are kept.

`rank_candidates` is pure (no FAISS, no model) so it can be unit-tested and
reused by the offline evaluation harness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from . import catalog as cat
from . import features as feats

BANDIT_MIN_EVENTS = 10
BANDIT_FULL_EVENTS = 30


@dataclass
class Recommendation:
    track: cat.Track
    similarity: float       # cosine similarity to seed, mapped to 0..1
    obscurity: float        # 0..1, higher = more obscure
    score: float            # blended ranking score
    features: np.ndarray    # per-candidate feature vector (used for bandit update)
    bandit_bonus: float = 0.0  # LinUCB contribution actually added to score


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-9:
        return np.full_like(x, 0.5, dtype="float64")
    return (x - lo) / (hi - lo)


def bandit_ramp(n_events: int) -> float:
    """0 below BANDIT_MIN_EVENTS, linear to 1 at BANDIT_FULL_EVENTS."""
    if n_events < BANDIT_MIN_EVENTS:
        return 0.0
    span = BANDIT_FULL_EVENTS - BANDIT_MIN_EVENTS
    return min(1.0, (n_events - BANDIT_MIN_EVENTS + 1) / span)


def rank_candidates(pairs: Sequence[tuple[cat.Track, float]],
                    k: int = 10,
                    obscurity_weight: float = 0.4,
                    pool_size: Optional[int] = None,
                    max_per_artist: int = 1,
                    bandit=None,
                    bandit_weight: float = 0.5) -> list[Recommendation]:
    """Rank (track, similarity) pairs. Similarity is expected in 0..1.

    score = (1-w)·sim_scaled + w·obs_scaled + ramp(n)·w_bandit·ucb_scaled
    """
    if not pairs:
        return []
    pool_size = pool_size or max(3 * k, 30)

    # 1. relevance pool: keep the most similar candidates only
    pool = sorted(pairs, key=lambda p: p[1], reverse=True)[:pool_size]

    sims = np.array([s for _, s in pool], dtype="float64")
    obs = np.array([cat.obscurity_score(t) for t, _ in pool], dtype="float64")

    # 2. common scale
    blended = ((1 - obscurity_weight) * _minmax(sims)
               + obscurity_weight * _minmax(obs))

    recs = [
        Recommendation(track=t, similarity=float(s), obscurity=float(o),
                       score=float(b), features=feats.feature_vector(t, float(s)))
        for (t, s), o, b in zip(pool, obs, blended)
    ]

    # 3. confidence-gated bandit
    if bandit is not None and bandit_weight > 0:
        ramp = bandit_ramp(getattr(bandit, "n_events", 0))
        if ramp > 0:
            ucb = bandit.score(np.stack([r.features for r in recs]))
            bonus = ramp * bandit_weight * _minmax(ucb)
            for r, b in zip(recs, bonus):
                r.bandit_bonus = float(b)
                r.score += float(b)

    recs.sort(key=lambda r: r.score, reverse=True)

    # 4. artist diversity
    out: list[Recommendation] = []
    per_artist: dict[str, int] = {}
    for r in recs:
        key = r.track.artist.strip().lower()
        if per_artist.get(key, 0) >= max_per_artist:
            continue
        per_artist[key] = per_artist.get(key, 0) + 1
        out.append(r)
        if len(out) == k:
            break
    return out


def recommend(seed_text: str,
              tracks: list[cat.Track],
              k: int = 10,
              obscurity_weight: float = 0.4,
              exclude_titles: Optional[set[str]] = None,
              bandit=None,
              bandit_weight: float = 0.5,
              language_pivot: Optional[set[str]] = None,
              max_per_artist: int = 1) -> list[Recommendation]:
    """Retrieve neighbours of `seed_text` from FAISS, then `rank_candidates`."""
    from . import embeddings as emb

    exclude = exclude_titles or set()
    id_map = emb.load_id_map()
    by_id = {t.track_id: t for t in tracks}

    scores, idx = emb.query(seed_text, top_k=max(k * 8, 80))

    pairs: list[tuple[cat.Track, float]] = []
    for sim, i in zip(scores, idx):
        if i < 0 or i >= len(id_map):
            continue
        t = by_id.get(id_map[i])
        if t is None or t.title.lower() in exclude:
            continue
        if language_pivot:
            langs = feats.detect_languages(t)
            if not langs or (langs & language_pivot):
                continue
        pairs.append((t, float(max(0.0, min(1.0, (sim + 1) / 2)))))

    return rank_candidates(pairs, k=k, obscurity_weight=obscurity_weight,
                           max_per_artist=max_per_artist,
                           bandit=bandit, bandit_weight=bandit_weight)

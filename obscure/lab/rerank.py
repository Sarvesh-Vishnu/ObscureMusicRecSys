"""Re-rankers: turn an accurate list into a discovery list.

All three take a base model's candidates (top-C items per user, C > k) and
choose the final k. Each has one knob, swept to trace an accuracy-vs-discovery
frontier rather than reported at a single hand-picked point.

  PopPenalty    score / popularity^beta (score-level popularity debiasing;
                Steck, "Item popularity and recommendation accuracy", RecSys 2011)
  Calibrated    greedy: maximise (1-λ)·relevance - λ·KL(p_u || q(list)) over
                head / mid / tail popularity groups, so each list's mix matches
                the listener's own history (Steck, "Calibrated recommendations",
                RecSys 2018; popularity-group variant: Abdollahpouri et al. 2021)
  MMR           greedy: (1-λ)·relevance - λ·max similarity to items already
                chosen (Carbonell & Goldstein 1998), similarity = ItemKNN cosine
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .models import Recommender, top_k


def candidates(model: Recommender, X: sp.csr_matrix, users: np.ndarray, c: int
               ) -> tuple[np.ndarray, np.ndarray]:
    """Top-c unseen items and their base scores, min-max scaled per user to 0..1."""
    items = top_k(model, X, users, c)
    scores = np.empty(items.shape, dtype=np.float32)
    batch = max(16, int(2e7 // X.shape[1]))
    for s in range(0, len(users), batch):
        S = model.score(users[s:s + batch], X)
        scores[s:s + batch] = np.take_along_axis(S, items[s:s + batch], axis=1)
    lo = scores.min(axis=1, keepdims=True)
    hi = scores.max(axis=1, keepdims=True)
    rel = (scores - lo) / np.maximum(hi - lo, 1e-9)
    return items, np.nan_to_num(rel, nan=0.0, posinf=1.0, neginf=0.0)


class PopPenalty(Recommender):
    """Wrap a non-negative scorer: score(u, i) / popularity_i ** beta."""

    def __init__(self, base: Recommender, popularity: np.ndarray, beta: float):
        self.base, self.beta = base, beta
        self.w = (np.maximum(popularity, 1).astype(np.float32)) ** (-beta)
        self.name = f"{base.name} + PopPenalty(β={beta:g})"

    def fit(self, X):
        return self

    def score(self, users, X):
        return self.base.score(users, X) * self.w

    def params(self):
        return {"beta": self.beta, **self.base.params()}


def popularity_groups(popularity: np.ndarray, head_share: float = 0.5,
                      tail_share: float = 0.8) -> np.ndarray:
    """0 = head (items holding the top `head_share` of pairs), 1 = mid
    (up to `tail_share`), 2 = tail."""
    order = np.argsort(-popularity, kind="stable")
    cum = np.cumsum(popularity[order]) / popularity.sum()
    g = np.full(len(popularity), 2, dtype=np.int8)
    g[order[cum <= tail_share]] = 1
    g[order[cum <= head_share]] = 0
    g[order[0]] = 0
    return g


def calibrated(items: np.ndarray, rel: np.ndarray, X: sp.csr_matrix, users: np.ndarray,
               groups: np.ndarray, lam: float, k: int, alpha: float = 0.01) -> np.ndarray:
    n_groups = int(groups.max()) + 1
    out = np.empty((len(users), k), dtype=np.int64)
    for row, u in enumerate(users):
        hist = groups[X[u].indices]
        p = np.bincount(hist, minlength=n_groups) / max(len(hist), 1)
        cand, r = items[row], rel[row]
        g = groups[cand]
        counts = np.zeros(n_groups)
        chosen = np.zeros(len(cand), dtype=bool)
        total_rel = 0.0
        for step in range(k):
            # q after adding each candidate (vectorised over candidates)
            q = (counts[None, :] + np.eye(n_groups)[g]) / (step + 1)
            q = (1 - alpha) * q + alpha * p[None, :]
            with np.errstate(divide="ignore", invalid="ignore"):
                kl = np.where(p > 0, p * np.log(p / q), 0.0).sum(axis=1)
            obj = (1 - lam) * (total_rel + r) - lam * kl
            obj[chosen] = -np.inf
            j = int(np.argmax(obj))
            chosen[j] = True
            counts[g[j]] += 1
            total_rel += r[j]
            out[row, step] = cand[j]
    return out


def mmr(items: np.ndarray, rel: np.ndarray, W: sp.csr_matrix, lam: float, k: int) -> np.ndarray:
    Ws = W.maximum(W.T).tocsr()  # symmetric similarity
    out = np.empty((items.shape[0], k), dtype=np.int64)
    for row in range(items.shape[0]):
        cand, r = items[row], rel[row]
        S = Ws[cand][:, cand].toarray()
        max_sim = np.zeros(len(cand))
        chosen = np.zeros(len(cand), dtype=bool)
        for step in range(k):
            obj = (1 - lam) * r - lam * max_sim
            obj[chosen] = -np.inf
            j = int(np.argmax(obj))
            chosen[j] = True
            max_sim = np.maximum(max_sim, S[j])
            out[row, step] = cand[j]
    return out

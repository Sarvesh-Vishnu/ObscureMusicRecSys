"""Discovery metrics, computed per user and summarised with bootstrap CIs.

Accuracy (did we predict what they discovered?)
  recall@k        hits / min(|targets|, k)
  ndcg@k          binary-relevance NDCG, ideal DCG over min(|targets|, k)
  tail_recall@k   recall restricted to long-tail targets (users with >= 1)

Beyond accuracy (what kind of list is it?)
  novelty@k       mean self-information -log2(listeners_i / n_users) of the list
  tail_share@k    fraction of recommended items in the long tail
  artist_div@k    distinct artists / list length
  coverage@k      distinct items recommended across all users / n_items (global)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

PER_USER = ("recall", "ndcg", "tail_recall", "novelty", "tail_share", "artist_div")


@dataclass
class Result:
    name: str
    k: int
    per_user: dict[str, np.ndarray]
    coverage: float
    fit_seconds: float = 0.0
    params: dict = field(default_factory=dict)

    def mean(self, metric: str) -> float:
        v = self.per_user[metric]
        v = v[~np.isnan(v)]
        return float(v.mean()) if len(v) else float("nan")


def evaluate_lists(recs: np.ndarray, targets: list[np.ndarray], *,
                   popularity: np.ndarray, n_users: int, tail: np.ndarray,
                   artist: np.ndarray, k: int) -> tuple[dict[str, np.ndarray], float]:
    """recs: (n_eval_users, k) item indices, row-aligned with `targets`."""
    n = len(targets)
    out = {m: np.full(n, np.nan) for m in PER_USER}
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    self_info = -np.log2(np.maximum(popularity, 1) / n_users)

    for u in range(n):
        r = recs[u, :k]
        t = targets[u]
        hit = np.isin(r, t)
        denom = min(len(t), k)
        out["recall"][u] = hit.sum() / denom
        out["ndcg"][u] = (hit * discounts).sum() / discounts[:denom].sum()
        t_tail = t[tail[t]]
        if len(t_tail):
            out["tail_recall"][u] = np.isin(r, t_tail).sum() / min(len(t_tail), k)
        out["novelty"][u] = self_info[r].mean()
        out["tail_share"][u] = tail[r].mean()
        out["artist_div"][u] = len(np.unique(artist[r])) / len(r)

    coverage = len(np.unique(recs[:, :k])) / len(popularity)
    return out, coverage


def bootstrap_ci(values: np.ndarray, n_boot: int = 1000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float, float]:
    """Mean and percentile CI over users (NaNs dropped)."""
    v = values[~np.isnan(values)]
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    means = v[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(v.mean()), float(lo), float(hi)


def paired_diff_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 1000,
                   seed: int = 0) -> tuple[float, float, float]:
    """Mean of (a - b) over users where both are defined, with a bootstrap CI.
    Paired, so it is far tighter than comparing two marginal CIs."""
    ok = ~(np.isnan(a) | np.isnan(b))
    return bootstrap_ci(a[ok] - b[ok], n_boot=n_boot, seed=seed)

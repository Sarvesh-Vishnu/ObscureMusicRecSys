"""Shared benchmark plumbing: user sampling, scoring a set of lists, tables."""

from __future__ import annotations

import numpy as np

from . import data as D
from . import metrics as M


class Bench:
    """Fixed evaluation users + targets for one dataset."""

    def __init__(self, ds: D.Dataset, n_users: int, seed: int = 0, k: int = 20):
        self.ds, self.k = ds, k
        rng = np.random.default_rng(seed)
        n = min(n_users, len(ds.eval_users))
        pick = np.sort(rng.choice(len(ds.eval_users), size=n, replace=False))
        self.users = ds.eval_users[pick]
        self.targets = [ds.targets[i] for i in pick]
        self.tail = ~D.head_mask(ds.popularity, head_share=0.5)
        self.seed = seed

    def score(self, name: str, recs: np.ndarray, fit_seconds: float = 0.0,
              params: dict | None = None) -> M.Result:
        per_user, cov = M.evaluate_lists(recs, self.targets, popularity=self.ds.popularity,
                                         n_users=self.ds.n_users, tail=self.tail,
                                         artist=self.ds.artist, k=self.k)
        return M.Result(name=name, k=self.k, per_user=per_user, coverage=cov,
                        fit_seconds=fit_seconds, params=params or {})


def summarise(r: M.Result, seed: int = 0) -> dict:
    row = {"name": r.name, "params": r.params, "fit_seconds": round(r.fit_seconds, 1),
           "coverage": r.coverage}
    for m in M.PER_USER:
        row[m] = dict(zip(("mean", "lo", "hi"), M.bootstrap_ci(r.per_user[m], seed=seed)))
    return row


def pct(d: dict) -> str:
    if np.isnan(d["mean"]):
        return "n/a"
    return f"{100 * d['mean']:.2f} [{100 * d['lo']:.2f}, {100 * d['hi']:.2f}]"


def table(rows: list[dict], k: int) -> list[str]:
    out = [f"| Model | Recall@{k} | NDCG@{k} | Tail recall@{k} | Novelty | Tail share | "
           f"Artist div. | Coverage |",
           "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['name']} | {pct(r['recall'])} | {pct(r['ndcg'])} | "
                   f"{pct(r['tail_recall'])} | {r['novelty']['mean']:.2f} | "
                   f"{100 * r['tail_share']['mean']:.0f}% | "
                   f"{100 * r['artist_div']['mean']:.0f}% | {100 * r['coverage']:.1f}% |")
    return out

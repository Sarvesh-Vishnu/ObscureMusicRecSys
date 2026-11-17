"""Bandits and off-policy evaluation on full-feedback data.

ListenBrainz has no logs of a recommender showing songs, so there is nothing to
evaluate off-policy *for real*. The standard way to study the methods anyway is the
supervised-to-bandit conversion (Dudik, Langford & Li, "Doubly robust policy
evaluation and learning", ICML 2011): take data where the reward of *every* action
is known, let a logging policy pick one action per context and reveal only that
reward, then check how well each estimator recovers a target policy's true value,
which is computable exactly.

Here
  context    a test listener
  actions    their K candidate recordings (ItemKNN's top K, nothing already heard)
  reward     1 if the candidate is one of their discoveries in the test window
  features   x(u, a): a few numbers any production ranker would have (scores,
             popularity, long-tail flag, whether the artist is already known)

Estimators of V(pi) = E_u sum_a pi(a|u) r(u,a) from logs (u, a ~ pi0, p0(a|u), r):
  DM     sum_a pi(a|u) rhat(u,a), rhat a logistic model fitted on the logs
  IPS    w r,  w = pi(a|u) / p0(a|u)
  SNIPS  sum w r / sum w
  DR     DM + w (r - rhat(u,a))
The reward model is cross-fitted (two folds), so DR's correction term is unbiased
given rhat (Chernozhukov et al. 2018).

Online learners pick one candidate per round and see only its reward:
  epsilon-greedy, LinUCB (Li et al., WWW 2010; shared parameters) and linear
  Thompson sampling (Agrawal & Goyal, ICML 2013), all on the same features.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FEATURES = ["bias", "itemknn score", "itemknn rank", "als score", "log popularity", "long tail", "known artist"]


@dataclass
class Contexts:
    users: np.ndarray        # (n,) user rows
    cand: np.ndarray         # (n, K) candidate item ids, ItemKNN order
    knn: np.ndarray          # (n, K) ItemKNN score / user max (0..1)
    als: np.ndarray          # (n, K) ALS score / user max
    X: np.ndarray            # (n, K, d) features
    R: np.ndarray            # (n, K) rewards (0/1)
    tail: np.ndarray         # (n, K) long-tail flag
    logpop: np.ndarray       # (n, K) log(1 + listeners)

    @property
    def n(self) -> int:
        return self.cand.shape[0]

    @property
    def K(self) -> int:
        return self.cand.shape[1]

    def subset(self, rows: np.ndarray) -> "Contexts":
        return Contexts(*(getattr(self, f)[rows] for f in ("users", "cand", "knn", "als", "X", "R", "tail", "logpop")))


def features(knn: np.ndarray, als: np.ndarray, logpop: np.ndarray, tail: np.ndarray,
             known_artist: np.ndarray, pop_mu: float, pop_sd: float) -> np.ndarray:
    n, K = knn.shape
    rank = np.broadcast_to(np.log1p(np.arange(K)) / np.log(K), (n, K))
    return np.stack([np.ones((n, K)), knn, rank, als, (logpop - pop_mu) / pop_sd,
                     tail.astype(float), known_artist.astype(float)], axis=-1).astype(np.float32)


# ---------------------------------------------------------------- policies as (n, K) distributions

def greedy(scores: np.ndarray) -> np.ndarray:
    """Deterministic: all mass on the best-scoring candidate (first on ties)."""
    P = np.zeros_like(scores, dtype=np.float64)
    P[np.arange(len(scores)), np.argmax(scores, axis=1)] = 1.0
    return P


def softmax(scores: np.ndarray, tau: float, eps: float = 0.0) -> np.ndarray:
    z = (scores - scores.max(axis=1, keepdims=True)) / tau
    P = np.exp(z)
    P /= P.sum(axis=1, keepdims=True)
    return (1 - eps) * P + eps / scores.shape[1]


def uniform(n: int, K: int) -> np.ndarray:
    return np.full((n, K), 1.0 / K)


def true_value(P: np.ndarray, R: np.ndarray) -> float:
    return float((P * R).sum(axis=1).mean())


# ---------------------------------------------------------------- off-policy estimators

def fit_reward_model(X: np.ndarray, r: np.ndarray):
    from sklearn.linear_model import LogisticRegression
    if r.min() == r.max():                      # degenerate log: constant prediction
        c = float(r.mean())
        return lambda Z: np.full(Z.shape[:-1], c)
    m = LogisticRegression(C=1.0, max_iter=500).fit(X[:, 1:], r)   # sklearn adds its own bias
    return lambda Z: m.predict_proba(Z.reshape(-1, Z.shape[-1])[:, 1:])[:, 1].reshape(Z.shape[:-1])


def estimate(ctx: Contexts, rows: np.ndarray, a: np.ndarray, p0: np.ndarray, r: np.ndarray,
             targets: dict[str, np.ndarray], rng: np.random.Generator) -> dict[str, dict[str, float]]:
    """Value estimates for each target policy (rows index ctx; targets are (ctx.n, K))."""
    n = len(rows)
    fold = rng.permutation(n) % 2
    rhat_all = np.zeros((n, ctx.K))             # cross-fitted rhat(u, .) for every candidate
    for f in (0, 1):
        tr, te = fold != f, fold == f
        model = fit_reward_model(ctx.X[rows[tr], a[tr]], r[tr])
        rhat_all[te] = model(ctx.X[rows[te]])
    rhat_a = rhat_all[np.arange(n), a]
    out = {}
    for name, P in targets.items():
        Pu = P[rows]
        pi_a = Pu[np.arange(n), a]
        w = pi_a / p0
        dm = (Pu * rhat_all).sum(axis=1)
        out[name] = {
            "DM": float(dm.mean()),
            "IPS": float((w * r).mean()),
            "SNIPS": float((w * r).sum() / w.sum()) if w.sum() > 0 else 0.0,
            "DR": float((dm + w * (r - rhat_a)).mean()),
            "ess": float(w.sum() ** 2 / max((w ** 2).sum(), 1e-12)),
        }
    return out


def log_actions(P0: np.ndarray, rows: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Sample one action per logged context from the logging policy."""
    Pr = P0[rows]
    u = rng.random(len(rows))[:, None]
    a = (Pr.cumsum(axis=1) < u).sum(axis=1)
    a = np.minimum(a, Pr.shape[1] - 1)
    return a, Pr[np.arange(len(rows)), a]


# ---------------------------------------------------------------- online learners

class Learner:
    name = "learner"

    def choose(self, x: np.ndarray, rng: np.random.Generator) -> int:   # x: (K, d)
        raise NotImplementedError

    def update(self, x: np.ndarray, r: float) -> None:                   # x: (d,)
        pass


class Fixed(Learner):
    """A policy that never learns: always the candidate with the best given score."""

    def __init__(self, column: int, name: str):
        self.column, self.name = column, name

    def choose(self, x, rng):
        return int(np.argmax(x[:, self.column]))


class Random(Learner):
    name = "Random"

    def choose(self, x, rng):
        return int(rng.integers(len(x)))


class _Ridge(Learner):
    """Shared ridge regression of reward on features (Sherman-Morrison updates)."""

    def __init__(self, d: int, lam: float = 1.0):
        self.Ainv = np.eye(d) / lam
        self.b = np.zeros(d)

    @property
    def theta(self) -> np.ndarray:
        return self.Ainv @ self.b

    def update(self, x, r):
        Ax = self.Ainv @ x
        self.Ainv -= np.outer(Ax, Ax) / (1.0 + x @ Ax)
        self.b += r * x


class EpsGreedy(_Ridge):
    def __init__(self, d: int, eps: float, lam: float = 1.0):
        super().__init__(d, lam)
        self.eps, self.name = eps, f"ε-greedy (ε = {eps:g})"

    def choose(self, x, rng):
        if rng.random() < self.eps:
            return int(rng.integers(len(x)))
        return int(np.argmax(x @ self.theta))


class LinUCB(_Ridge):
    def __init__(self, d: int, alpha: float, lam: float = 1.0):
        super().__init__(d, lam)
        self.alpha, self.name = alpha, f"LinUCB (α = {alpha:g})"

    def choose(self, x, rng):
        width = np.sqrt(np.einsum("kd,de,ke->k", x, self.Ainv, x))
        return int(np.argmax(x @ self.theta + self.alpha * width))


class LinTS(_Ridge):
    def __init__(self, d: int, v: float, lam: float = 1.0):
        super().__init__(d, lam)
        self.v, self.name = v, f"Thompson (v = {v:g})"

    def choose(self, x, rng):
        cov = self.v ** 2 * self.Ainv
        L = np.linalg.cholesky(cov + 1e-12 * np.eye(len(cov)))
        th = self.theta + L @ rng.standard_normal(len(cov))
        return int(np.argmax(x @ th))


def run_online(learner: Learner, ctx: Contexts, order: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Play one round per entry of `order` (context rows); returns the reward per round."""
    rewards = np.zeros(len(order), dtype=np.float32)
    for t, i in enumerate(order):
        x = ctx.X[i].astype(np.float64)
        a = learner.choose(x, rng)
        r = float(ctx.R[i, a])
        learner.update(x[a], r)
        rewards[t] = r
    return rewards

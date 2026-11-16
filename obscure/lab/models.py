"""Recommenders for the offline harness. Every model sees only the training
matrix X (users x items, binary) and returns a score per (user, item); the
harness masks items the user already played and takes the top k.

Ladder (simplest first, each a standard reference point):
  RandomTail   v1's idea: random picks from the long tail
  Popularity   most training listeners
  ItemKNN      cosine item-item neighbours with shrinkage (Sarwar et al. 2001)
  EASE         closed-form linear autoencoder (Steck, WWW 2019)
  ALS          implicit-feedback matrix factorisation (Hu, Koren & Volinsky 2008)
  BPR          pairwise ranking factorisation (Rendle et al. 2009)
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


class Recommender:
    name = "base"

    def fit(self, X: sp.csr_matrix) -> "Recommender":
        raise NotImplementedError

    def score(self, users: np.ndarray, X: sp.csr_matrix) -> np.ndarray:
        """Dense (len(users), n_items) float32 scores. Higher is better."""
        raise NotImplementedError

    def params(self) -> dict:
        return {}


def top_k(model: Recommender, X: sp.csr_matrix, users: np.ndarray, k: int,
          batch: int | None = None) -> np.ndarray:
    """Top-k unseen items for each user, in order. Batches are sized so the
    dense score block stays around 80 MB whatever the catalog size."""
    batch = batch or max(16, int(2e7 // X.shape[1]))
    out = np.empty((len(users), k), dtype=np.int64)
    for s in range(0, len(users), batch):
        u = users[s:s + batch]
        S = model.score(u, X).astype(np.float32, copy=False)
        seen = X[u]
        S[np.repeat(np.arange(len(u)), np.diff(seen.indptr)), seen.indices] = -np.inf
        part = np.argpartition(-S, k, axis=1)[:, :k]
        # order by score, ties broken by item index, so lists of different
        # lengths agree on their common prefix
        order = np.lexsort((part, -np.take_along_axis(S, part, axis=1)), axis=1)
        out[s:s + len(u)] = np.take_along_axis(part, order, axis=1)
    return out


class Popularity(Recommender):
    name = "Popularity"

    def fit(self, X):
        self.pop = np.asarray(X.sum(axis=0)).ravel().astype(np.float32)
        return self

    def score(self, users, X):
        return np.broadcast_to(self.pop, (len(users), len(self.pop))).copy()


class RandomTail(Recommender):
    """Uniform random scores over long-tail items; head items rank last.
    The honest version of v1, which asked Spotify for a random popularity."""
    name = "RandomTail (v1 idea)"

    def __init__(self, tail: np.ndarray, seed: int = 0):
        self.tail = tail
        self.rng = np.random.default_rng(seed)

    def fit(self, X):
        return self

    def score(self, users, X):
        S = self.rng.random((len(users), len(self.tail)), dtype=np.float32)
        S[:, ~self.tail] -= 2.0
        return S


def cap_history(plays: sp.csr_matrix, cap: int | None) -> sp.csr_matrix:
    """Binary matrix where users with more than `cap` items keep only their
    `cap` most-played ones (ties by item index). Used only to compute item-item
    similarity: co-occurrence cost grows with |history|², and in this data the
    top 1% of users (bulk-imported libraries of up to ~32k recordings) account
    for 95% of it while linking everything to everything."""
    P = plays.tocsr()
    if cap is None:
        X = P.copy()
        X.data[:] = 1.0
        return X
    lens = np.diff(P.indptr)
    rows = np.repeat(np.arange(P.shape[0]), lens)
    order = np.lexsort((P.indices, -P.data, rows))
    r_sorted = rows[order]
    rank = np.arange(len(order)) - np.searchsorted(r_sorted, r_sorted, side="left")
    keep = order[rank < cap]
    return sp.csr_matrix((np.ones(len(keep), np.float32), (rows[keep], P.indices[keep])),
                         shape=P.shape)


def itemknn_neighbours(X: sp.csr_matrix, configs: list[tuple[int, float]],
                       max_nnz: int = 30_000_000) -> dict[tuple[int, float], sp.csr_matrix]:
    """Top-k shrunk-cosine neighbour matrices for several (k, shrink) configs in
    one pass over the co-occurrence matrix.

    Works on sparse co-occurrence rows only: each item's row holds just the
    items it co-occurs with, so nothing proportional to (chunk x n_items) is
    ever allocated. Chunks are sized by co-occurrence nnz, not row count.
    """
    Xc = X.tocsc().astype(np.float32)
    n = X.shape[1]
    norms = np.sqrt(np.asarray(Xc.sum(axis=0)).ravel())   # binary X: |U_i|
    XT = Xc.T.tocsr()
    # co-occurrence nnz per item row is bounded by sum over its users of their
    # history length; use that bound to size chunks.
    user_len = np.diff(X.indptr).astype(np.int64)
    bound = np.asarray(XT @ user_len.astype(np.float32)).ravel()
    parts = {c: ([], [], []) for c in configs}

    s = 0
    while s < n:
        cum = np.cumsum(bound[s:])
        e = s + max(1, int(np.searchsorted(cum, max_nnz)))
        co = (XT[s:e] @ Xc).tocsr()
        rows = np.repeat(np.arange(s, e), np.diff(co.indptr))
        cols, raw = co.indices, co.data
        keep = cols != rows
        rows, cols, raw = rows[keep], cols[keep], raw[keep]
        denom_base = norms[rows] * norms[cols]
        for (k, shrink) in configs:
            vals = raw / (denom_base + shrink)
            order = np.lexsort((-vals, rows))
            r_sorted = rows[order]
            first = np.searchsorted(r_sorted, r_sorted, side="left")
            rank = np.arange(len(order)) - first
            top = order[rank < k]
            R, C, V = parts[(k, shrink)]
            R.append(rows[top]); C.append(cols[top]); V.append(vals[top])
        s = e

    out = {}
    for c, (R, C, V) in parts.items():
        out[c] = sp.csr_matrix((np.concatenate(V), (np.concatenate(R), np.concatenate(C))),
                               shape=(n, n), dtype=np.float32)
    return out


def truncate_neighbours(W: sp.csr_matrix, k: int) -> sp.csr_matrix:
    """Keep each row's k largest entries. Exact: the top-k of a top-K list (K >= k)
    is the top-k of the full row, so one wide fit serves every smaller k."""
    W = W.tocsr()
    rows = np.repeat(np.arange(W.shape[0]), np.diff(W.indptr))
    order = np.lexsort((-W.data, rows))
    r_sorted = rows[order]
    rank = np.arange(len(order)) - np.searchsorted(r_sorted, r_sorted, side="left")
    top = order[rank < k]
    return sp.csr_matrix((W.data[top], (rows[top], W.indices[top])), shape=W.shape)


class ItemKNN(Recommender):
    """score(u, j) = sum_{i in u} sim(i, j), keeping each item's top `k` neighbours.
    sim(i, j) = |U_i ∩ U_j| / (sqrt(|U_i| |U_j|) + shrink)."""
    name = "ItemKNN"

    def __init__(self, k: int = 100, shrink: float = 10.0, W: sp.csr_matrix | None = None):
        self.k, self.shrink, self.W = k, shrink, W

    def params(self):
        return {"k": self.k, "shrink": self.shrink}

    def fit(self, X):
        if self.W is None:
            self.W = itemknn_neighbours(X, [(self.k, self.shrink)])[(self.k, self.shrink)]
        return self

    def score(self, users, X):
        return (X[users] @ self.W).toarray()


class EASE(Recommender):
    """B = I - P / diag(P), P = (XᵀX + λI)⁻¹, diag(B) = 0. Needs an n x n dense
    inverse, so it runs on the `max_items` most popular items; the rest score -inf."""
    name = "EASE"

    def __init__(self, lam: float = 500.0, max_items: int = 15000):
        self.lam, self.max_items = lam, max_items

    def params(self):
        return {"lambda": self.lam, "max_items": self.max_items}

    @staticmethod
    def gram(X, max_items: int) -> tuple[np.ndarray, np.ndarray]:
        """(columns kept, XᵀX over them). Reusable across λ values."""
        pop = np.asarray(X.sum(axis=0)).ravel()
        cols = np.sort(np.argsort(-pop, kind="stable")[:max_items])
        Xs = X[:, cols].astype(np.float32)
        return cols, (Xs.T @ Xs).toarray()

    def fit(self, X, gram: tuple[np.ndarray, np.ndarray] | None = None):
        self.cols, G = gram if gram is not None else self.gram(X, self.max_items)
        G = G.copy() if gram is not None else G
        G[np.diag_indices_from(G)] += self.lam
        P = np.linalg.inv(G)
        del G
        B = P / (-np.diag(P))
        B[np.diag_indices_from(B)] = 0.0
        self.B = B.astype(np.float32)
        self.n = X.shape[1]
        return self

    def score(self, users, X):
        S = np.full((len(users), self.n), -np.inf, dtype=np.float32)
        S[:, self.cols] = X[users][:, self.cols] @ self.B
        return S


class _Implicit(Recommender):
    def score(self, users, X):
        U = self.model.user_factors[users]
        return U @ self.model.item_factors.T

    def params(self):
        return dict(self._params)


class ALS(_Implicit):
    name = "ALS"

    def __init__(self, factors: int = 128, reg: float = 0.05, alpha: float = 10.0,
                 iterations: int = 15, seed: int = 0):
        self._params = {"factors": factors, "reg": reg, "alpha": alpha, "iterations": iterations}
        self.alpha, self.seed = alpha, seed

    def fit(self, X):
        from implicit.cpu.als import AlternatingLeastSquares
        p = self._params
        self.model = AlternatingLeastSquares(factors=p["factors"], regularization=p["reg"],
                                             alpha=p["alpha"], iterations=p["iterations"],
                                             random_state=self.seed)
        self.model.fit(X.astype(np.float32), show_progress=False)
        return self


class BPR(_Implicit):
    name = "BPR"

    def __init__(self, factors: int = 128, lr: float = 0.05, reg: float = 0.001,
                 iterations: int = 100, seed: int = 0):
        self._params = {"factors": factors, "lr": lr, "reg": reg, "iterations": iterations}
        self.seed = seed

    def fit(self, X):
        from implicit.cpu.bpr import BayesianPersonalizedRanking
        p = self._params
        self.model = BayesianPersonalizedRanking(factors=p["factors"], learning_rate=p["lr"],
                                                 regularization=p["reg"],
                                                 iterations=p["iterations"],
                                                 random_state=self.seed)
        self.model.fit(X.astype(np.float32), show_progress=False)
        return self

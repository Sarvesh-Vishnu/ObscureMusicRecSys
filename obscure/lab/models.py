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
        order = np.argsort(-np.take_along_axis(S, part, axis=1), axis=1)
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


class ItemKNN(Recommender):
    """score(u, j) = sum_{i in u} sim(i, j), keeping each item's top `k` neighbours.
    sim(i, j) = |U_i ∩ U_j| / (sqrt(|U_i| |U_j|) + shrink)."""
    name = "ItemKNN"

    def __init__(self, k: int = 100, shrink: float = 10.0, chunk: int | None = None):
        self.k, self.shrink, self.chunk = k, shrink, chunk

    def params(self):
        return {"k": self.k, "shrink": self.shrink}

    def fit(self, X):
        Xc = X.tocsc().astype(np.float32)
        n = X.shape[1]
        norms = np.sqrt(np.asarray(Xc.power(2).sum(axis=0)).ravel())
        XT = Xc.T.tocsr()
        chunk = self.chunk or max(64, int(5e7 // n))  # ~200 MB dense block
        rows, cols, vals = [], [], []
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            co = (XT[s:e] @ Xc).toarray()                     # (chunk, n)
            co /= np.outer(norms[s:e], norms) + self.shrink
            co[np.arange(e - s), np.arange(s, e)] = 0.0        # no self-similarity
            kk = min(self.k, n - 1)
            idx = np.argpartition(-co, kk, axis=1)[:, :kk]
            v = np.take_along_axis(co, idx, axis=1)
            keep = v > 0
            rows.append(np.repeat(np.arange(s, e), kk)[keep.ravel()])
            cols.append(idx[keep])
            vals.append(v[keep])
        # W[i, j] = sim(i, j) for j among i's neighbours
        self.W = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                               shape=(n, n), dtype=np.float32)
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

    def fit(self, X):
        pop = np.asarray(X.sum(axis=0)).ravel()
        self.cols = np.sort(np.argsort(-pop, kind="stable")[: self.max_items])
        Xs = X[:, self.cols].astype(np.float32)
        G = (Xs.T @ Xs).toarray()
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

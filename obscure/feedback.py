"""LinUCB contextual bandit + SQLite feedback store.

The bandit re-ranks the FAISS candidate set using per-track features. It learns
online from thumbs-up / thumbs-down / skip events. Algorithm: Li, Chu, Langford,
Schapire — "A Contextual-Bandit Approach to Personalized News Article
Recommendation" (WWW 2010), `α√(xᵀA⁻¹x)` upper-confidence bound.

State is small enough (~30×30 matrix + 30-vector) that we recompute it from the
event log at session start. No pickling of model state required — the log is
the source of truth.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from . import config
from .features import FEATURE_DIM

FEEDBACK_DB = config.DATA_DIR / "feedback.db"

# reward map
REWARDS = {"up": 1.0, "down": -1.0, "skip": -0.2}


# ---------------------------------------------------------------------- store

@dataclass
class FeedbackEvent:
    ts: float
    session_id: str
    seed_text: str
    candidate_id: str
    reward: float
    features: np.ndarray  # shape (FEATURE_DIM,)


class FeedbackStore:
    """SQLite-backed log of feedback events. Single writer, append-only."""

    def __init__(self, path: Path = FEEDBACK_DB):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with self._conn() as c:
            c.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    ts          REAL NOT NULL,
                    session_id  TEXT NOT NULL,
                    seed_text   TEXT NOT NULL,
                    candidate_id TEXT NOT NULL,
                    reward      REAL NOT NULL,
                    features    TEXT NOT NULL
                )
            """)
            c.execute("CREATE INDEX IF NOT EXISTS ix_session ON feedback(session_id)")

    def _conn(self):
        return sqlite3.connect(self.path)

    def record(self, session_id: str, seed_text: str, candidate_id: str,
               reward: float, features: np.ndarray) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO feedback VALUES (?, ?, ?, ?, ?, ?)",
                (time.time(), session_id, seed_text, candidate_id, reward,
                 json.dumps(features.tolist())),
            )

    def load(self, session_id: Optional[str] = None) -> list[FeedbackEvent]:
        sql = "SELECT ts, session_id, seed_text, candidate_id, reward, features FROM feedback"
        params: tuple = ()
        if session_id is not None:
            sql += " WHERE session_id = ?"
            params = (session_id,)
        sql += " ORDER BY ts"
        with self._conn() as c:
            rows = list(c.execute(sql, params))
        return [FeedbackEvent(ts, sid, seed, cid, r,
                              np.asarray(json.loads(feat), dtype="float32"))
                for (ts, sid, seed, cid, r, feat) in rows]

    def count(self, session_id: Optional[str] = None) -> int:
        sql = "SELECT COUNT(*) FROM feedback"
        params: tuple = ()
        if session_id is not None:
            sql += " WHERE session_id = ?"
            params = (session_id,)
        with self._conn() as c:
            return c.execute(sql, params).fetchone()[0]


# ---------------------------------------------------------------------- LinUCB

class LinUCBReranker:
    """Disjoint LinUCB over a fixed-dimension feature vector.

    Same model scores every candidate (we're not learning per-arm weights —
    candidates are not stable across sessions). The bandit is *per user*.
    """

    def __init__(self, d: int = FEATURE_DIM, alpha: float = 1.0,
                 ridge: float = 1.0):
        self.d = d
        self.alpha = alpha
        self.A = ridge * np.eye(d, dtype="float64")
        self.b = np.zeros(d, dtype="float64")

    def update(self, features: np.ndarray, reward: float) -> None:
        x = features.astype("float64")
        self.A += np.outer(x, x)
        self.b += reward * x

    def update_batch(self, events: Iterable[FeedbackEvent]) -> int:
        n = 0
        for ev in events:
            self.update(ev.features, ev.reward)
            n += 1
        return n

    def score(self, feature_matrix: np.ndarray) -> np.ndarray:
        """Compute UCB scores for a matrix of candidate features.

        Args:
            feature_matrix: shape (n_candidates, d).
        Returns:
            shape (n_candidates,) — higher is better.
        """
        X = feature_matrix.astype("float64")
        A_inv = np.linalg.inv(self.A)
        theta = A_inv @ self.b
        mean = X @ theta
        # quadratic form per row: sqrt(x A^-1 x)
        uncertainty = np.sqrt(np.einsum("ij,jk,ik->i", X, A_inv, X))
        return mean + self.alpha * uncertainty

    def theta(self) -> np.ndarray:
        """Current point estimate of feature weights (for introspection)."""
        return np.linalg.solve(self.A, self.b)


# ---------------------------------------------------------------------- glue

def hydrate_from_store(store: FeedbackStore, session_id: Optional[str] = None,
                       alpha: float = 1.0) -> LinUCBReranker:
    """Build a fresh bandit and replay every event for this user/session."""
    bandit = LinUCBReranker(alpha=alpha)
    bandit.update_batch(store.load(session_id=session_id))
    return bandit

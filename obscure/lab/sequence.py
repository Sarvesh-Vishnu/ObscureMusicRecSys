"""Listening sessions: what people actually play next.

Every ListenBrainz listen has a timestamp, so a listener's history is a
sequence. Two consecutive listens by the same person, less than
`gap_minutes` apart, form a **transition** a -> b (repeats of the same
recording are skipped). Transitions are counted on the warm catalog after
canonical merging; a non-catalog listen in between breaks nothing, it is simply
not an endpoint (a, X, b with X outside the catalog gives no transition).

Training transitions use only listens played *and* submitted before the split;
test transitions only those played and submitted inside the test window, the
same rule as data.build.

Next-song model ("Transition"):
    score(b | a) = c(a, b) / c_in(b)^gamma  +  eps * cosine_itemknn(a, b)
c_in(b) is b's total incoming transitions; gamma > 0 discounts recordings
that follow everything (a first-order Markov chain with popularity
debiasing and an ItemKNN back-off for rare rows). Simple neighbourhood
models like this are strong next-item baselines (Ludewig & Jannach,
"Evaluation of session-based recommendation algorithms", UMUAI 2018).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import data as D

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Transitions:
    T: sp.csr_matrix           # (n_items, n_items) training transition counts a -> b
    test: pd.DataFrame         # held-out transitions: user (row or -1), a, b, new_to_user
    meta: dict


def _con() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    spill = ROOT / "data" / "raw" / "duckdb_tmp"
    spill.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET memory_limit='3GB'; SET threads=4; SET preserve_insertion_order=false; "
                f"SET temp_directory='{spill}'")
    return con


def _listens(con, files: list[str], lo: str | None, hi: str, mapped: str, join: str
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(user_id, epoch seconds, warm index or -1) for every listen in the window,
    one file at a time so memory stays flat."""
    where = f"t.listened_at < TIMESTAMP '{hi}'" + (f" AND t.listened_at >= TIMESTAMP '{lo}'" if lo else "")
    us, ts, ix = [], [], []
    for f in files:
        df = con.execute(f"""
            SELECT t.user_id, epoch(t.listened_at)::BIGINT AS ts, coalesce(w.idx, -1)::INT AS idx
            FROM read_parquet('{f}') t {join}
            LEFT JOIN warm w ON w.mbid = {mapped}
            WHERE {where}
        """).df()
        us.append(df["user_id"].to_numpy(np.int64))
        ts.append(df["ts"].to_numpy(np.int64))
        ix.append(df["idx"].to_numpy(np.int32))
    return np.concatenate(us), np.concatenate(ts), np.concatenate(ix)


def _consecutive(user: np.ndarray, ts: np.ndarray, idx: np.ndarray, gap_minutes: int
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sort each listener's listens by time and return transitions (user, a, b):
    consecutive listens <= gap apart, both in the warm catalog, a != b."""
    order = np.lexsort((idx, ts, user))
    user, ts, idx = user[order], ts[order], idx[order]
    dup = np.zeros(len(user), dtype=bool)               # exact duplicate listens
    dup[1:] = (user[1:] == user[:-1]) & (ts[1:] == ts[:-1]) & (idx[1:] == idx[:-1])
    user, ts, idx = user[~dup], ts[~dup], idx[~dup]
    same = user[1:] == user[:-1]
    close = (ts[1:] - ts[:-1]) <= gap_minutes * 60
    a, b = idx[:-1], idx[1:]
    keep = same & close & (a >= 0) & (b >= 0) & (a != b)
    return user[1:][keep], a[keep], b[keep]


def build_transitions(ds: D.Dataset, listens_dir: Path = D.LISTENS,
                      redirect: Path | None = D.REDIRECT, gap_minutes: int = 30) -> Transitions:
    split, end = ds.meta["split"], ds.meta["end"]
    files = sorted(listens_dir.glob("*.parquet"))
    train_files = [str(p) for p in files if p.stem < split]
    test_files = [str(p) for p in files if split <= p.stem < end]
    canonical = redirect is not None and Path(redirect).exists()
    mapped = "coalesce(r.canonical_recording_mbid, t.recording_mbid)" if canonical else "t.recording_mbid"
    join = f"LEFT JOIN read_parquet('{redirect}') r USING (recording_mbid)" if canonical else ""

    con = _con()
    con.register("warm", pd.DataFrame({"mbid": ds.items["mbid"].to_numpy(),
                                       "idx": np.arange(ds.n_items, dtype=np.int32)}))
    _, a, b = _consecutive(*_listens(con, train_files, None, split, mapped, join), gap_minutes)
    te_user, te_a, te_b = _consecutive(*_listens(con, test_files, split, end, mapped, join), gap_minutes)
    con.close()

    n = ds.n_items
    T = sp.csr_matrix((np.ones(len(a), np.float32), (a, b)), shape=(n, n))
    T.sum_duplicates()
    te = pd.DataFrame({"user_id": te_user, "a": te_a, "b": te_b})
    user_index = pd.Series(np.arange(ds.n_users), index=ds.user_ids)
    te["user"] = user_index.reindex(te["user_id"]).fillna(-1).astype(np.int64).to_numpy()
    te = te.drop(columns="user_id")
    known = te["user"].to_numpy() >= 0
    new = np.zeros(len(te), dtype=bool)
    new[known] = np.asarray(ds.X[te["user"].to_numpy()[known], te["b"].to_numpy()[known]]).ravel() == 0
    te["new_to_user"] = new
    meta = {"gap_minutes": gap_minutes, "train_transitions": int(T.sum()), "train_pairs": int(T.nnz),
            "items_with_out_edges": int((np.diff(T.indptr) > 0).sum()),
            "test_transitions": int(len(te)), "test_by_warm_users": int(known.sum()),
            "test_new_to_user": int(new.sum())}
    return Transitions(T=T, test=te, meta=meta)


class NextSong:
    """score(b | a) for a batch of current recordings a. Rows are dense (n_items)."""

    def __init__(self, T: sp.csr_matrix | None, W: sp.csr_matrix | None, gamma: float = 0.0,
                 eps: float = 1e-3, pop: np.ndarray | None = None, name: str = "Transition"):
        self.name = name
        self.T, self.W, self.eps = T, W, eps
        if T is not None:
            c_in = np.asarray(T.sum(axis=0)).ravel()
            self.T = (T @ sp.diags((np.maximum(c_in, 1) ** -gamma).astype(np.float32))).tocsr()
            # scale so the largest transition score in each row is 1, keeping the
            # ItemKNN back-off strictly below any observed transition
            rmax = self.T.max(axis=1).toarray().ravel()
            self.T = (sp.diags(1.0 / np.maximum(rmax, 1e-12)) @ self.T).tocsr()
        self.pop = None if pop is None else (pop / pop.max()).astype(np.float32)
        self.gamma = gamma

    def score(self, a: np.ndarray) -> np.ndarray:
        n = (self.T if self.T is not None else self.W).shape[1]
        S = np.zeros((len(a), n), dtype=np.float32)
        if self.T is not None:
            S += self.T[a].toarray()
        if self.W is not None:
            S += (self.eps if self.T is not None else 1.0) * self.W[a].toarray()
        if self.pop is not None:
            S += 1e-6 * self.pop
        S[np.arange(len(a)), a] = -np.inf
        return S


def next_song_eval(model: NextSong, a: np.ndarray, b: np.ndarray, k: int = 20,
                   batch: int = 256) -> dict[str, np.ndarray]:
    """Per-transition hit@k and reciprocal rank@k of the true next recording b."""
    hit = np.zeros(len(a), dtype=bool)
    rr = np.zeros(len(a), dtype=np.float32)
    for s in range(0, len(a), batch):
        S = model.score(a[s:s + batch])
        target = S[np.arange(len(S)), b[s:s + batch]]
        # 0-based rank; ties broken uniformly at random in expectation
        rank = (S > target[:, None]).sum(axis=1) + ((S == target[:, None]).sum(axis=1) - 1) / 2
        ok = rank < k
        hit[s:s + batch] = ok
        rr[s:s + batch] = np.where(ok, 1.0 / (rank + 1), 0.0)
    return {"hit": hit.astype(np.float64), "rr": rr.astype(np.float64)}

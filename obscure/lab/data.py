"""Build the offline evaluation dataset from ListenBrainz daily dumps.

Protocol (global temporal split, no leakage):

* Each file in data/listens/ holds the listens *submitted* on that date
  (fetch_listenbrainz.py names files by the dump's window start).
* TRAIN = everything submitted before `split` (any listened_at earlier than
  `split`). This includes bulk-imported history, so user profiles are rich,
  but nothing submitted after the cutoff is visible to a model.
* TEST = listens submitted on/after `split` whose listened_at is also inside
  [split, end): real-time listening during the test window.
* A user's DISCOVERIES are test items absent from that user's training
  history. These are the targets: a discovery recommender should rank them.

Items are MusicBrainz recording ids. Training is filtered once: items need
>= `min_item_users` distinct training listeners, then users need >=
`min_user_items` distinct kept items. Eval users must also have >= 1
discovery inside the kept item universe.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[2]
LISTENS = ROOT / "data" / "listens"
CACHE = ROOT / "data" / "eval"


@dataclass
class Dataset:
    X: sp.csr_matrix              # users x items, binary training interactions
    plays: sp.csr_matrix          # users x items, training play counts
    items: pd.DataFrame           # row i = item i: mbid, artist_credit_id, artist, title, listeners
    user_ids: np.ndarray          # row u = ListenBrainz user_id
    eval_users: np.ndarray        # user rows that have >= 1 discovery
    targets: list[np.ndarray]     # aligned with eval_users: discovered item indices
    meta: dict

    @property
    def n_users(self) -> int:
        return self.X.shape[0]

    @property
    def n_items(self) -> int:
        return self.X.shape[1]

    @property
    def popularity(self) -> np.ndarray:
        """Distinct training listeners per item."""
        return self.items["listeners"].to_numpy()

    @property
    def artist(self) -> np.ndarray:
        return self.items["artist_key"].to_numpy()


def head_mask(popularity: np.ndarray, head_share: float = 0.5) -> np.ndarray:
    """Head = the most-listened items that together hold `head_share` of all
    training (user, item) pairs. Everything else is the long tail."""
    order = np.argsort(-popularity, kind="stable")
    cum = np.cumsum(popularity[order]) / popularity.sum()
    n_head = int(np.searchsorted(cum, head_share) + 1)
    mask = np.zeros(len(popularity), dtype=bool)
    mask[order[:n_head]] = True
    return mask


def build(split: str = "2026-11-12", end: str = "2026-11-16",
          min_item_users: int = 10, min_user_items: int = 10,
          listens_dir: Path = LISTENS) -> Dataset:
    files = sorted(p for p in listens_dir.glob("*.parquet"))
    train_files = [str(p) for p in files if p.stem < split]
    test_files = [str(p) for p in files if split <= p.stem < end]
    if not train_files or not test_files:
        raise RuntimeError(f"need files before and after {split} in {listens_dir}")

    con = duckdb.connect()
    con.execute("SET memory_limit='3GB'")
    con.execute(f"""
        CREATE TEMP TABLE train AS
        SELECT user_id, recording_mbid AS mbid, count(*)::INT AS plays,
               any_value(artist_credit_id) AS artist_credit_id,
               any_value(artist_name) AS artist, any_value(recording_name) AS title
        FROM read_parquet({train_files})
        WHERE listened_at < TIMESTAMP '{split}'
        GROUP BY 1, 2
    """)
    con.execute(f"""
        CREATE TEMP TABLE items AS
        SELECT mbid, count(*)::INT AS listeners,
               any_value(artist_credit_id) AS artist_credit_id,
               any_value(artist) AS artist, any_value(title) AS title
        FROM train GROUP BY 1 HAVING count(*) >= {min_item_users}
    """)
    con.execute(f"""
        CREATE TEMP TABLE users AS
        SELECT t.user_id FROM train t JOIN items i USING (mbid)
        GROUP BY 1 HAVING count(*) >= {min_user_items}
    """)
    items = con.execute("""
        SELECT mbid, listeners, artist_credit_id, artist, title
        FROM items ORDER BY listeners DESC, mbid
    """).df()
    user_ids = con.execute("SELECT user_id FROM users ORDER BY user_id").df()["user_id"].to_numpy()

    item_index = pd.Series(np.arange(len(items)), index=items["mbid"])
    user_index = pd.Series(np.arange(len(user_ids)), index=user_ids)

    pairs = con.execute("""
        SELECT t.user_id, t.mbid, t.plays FROM train t
        JOIN items USING (mbid) JOIN users USING (user_id)
    """).df()
    rows = user_index[pairs["user_id"]].to_numpy()
    cols = item_index[pairs["mbid"]].to_numpy()
    shape = (len(user_ids), len(items))
    plays = sp.csr_matrix((pairs["plays"].to_numpy(np.float32), (rows, cols)), shape=shape)
    X = plays.copy()
    X.data[:] = 1.0
    # listeners after the user filter (what models actually see)
    items["listeners"] = np.asarray(X.sum(axis=0)).ravel().astype(int)

    test = con.execute(f"""
        SELECT DISTINCT user_id, recording_mbid AS mbid
        FROM read_parquet({test_files})
        WHERE listened_at >= TIMESTAMP '{split}' AND listened_at < TIMESTAMP '{end}'
    """).df()
    test = test[test["user_id"].isin(user_index.index) & test["mbid"].isin(item_index.index)]
    t_rows = user_index[test["user_id"]].to_numpy()
    t_cols = item_index[test["mbid"]].to_numpy()
    seen = np.asarray(X[t_rows, t_cols]).ravel() > 0
    t_rows, t_cols = t_rows[~seen], t_cols[~seen]   # discoveries only

    order = np.argsort(t_rows, kind="stable")
    t_rows, t_cols = t_rows[order], t_cols[order]
    eval_users, starts = np.unique(t_rows, return_index=True)
    targets = np.split(t_cols, starts[1:])

    # artist key for diversity: credit id when present, else lower-cased name
    items["artist_key"] = pd.factorize(
        items["artist_credit_id"].fillna(-1).astype("int64").astype(str)
        .where(items["artist_credit_id"].notna(), "n:" + items["artist"].str.lower())
    )[0]

    meta = {
        "split": split, "end": end,
        "train_files": [Path(f).stem for f in train_files],
        "test_files": [Path(f).stem for f in test_files],
        "min_item_users": min_item_users, "min_user_items": min_user_items,
        "n_users": int(shape[0]), "n_items": int(shape[1]),
        "train_pairs": int(X.nnz),
        "eval_users": int(len(eval_users)),
        "discoveries": int(len(t_cols)),
    }
    return Dataset(X=X, plays=plays, items=items, user_ids=user_ids,
                   eval_users=eval_users, targets=targets, meta=meta)


def save(ds: Dataset, path: Path = CACHE) -> None:
    path.mkdir(parents=True, exist_ok=True)
    sp.save_npz(path / "plays.npz", ds.plays)
    ds.items.to_parquet(path / "items.parquet")
    lengths = np.array([len(t) for t in ds.targets])
    np.savez(path / "eval.npz", user_ids=ds.user_ids, eval_users=ds.eval_users,
             target_lengths=lengths,
             targets=np.concatenate(ds.targets) if ds.targets else np.array([], int))
    (path / "meta.json").write_text(json.dumps(ds.meta, indent=2))


def load(path: Path = CACHE) -> Dataset:
    plays = sp.load_npz(path / "plays.npz").tocsr()
    X = plays.copy()
    X.data[:] = 1.0
    z = np.load(path / "eval.npz")
    targets = np.split(z["targets"], np.cumsum(z["target_lengths"])[:-1])
    return Dataset(X=X, plays=plays, items=pd.read_parquet(path / "items.parquet"),
                   user_ids=z["user_ids"], eval_users=z["eval_users"],
                   targets=targets, meta=json.loads((path / "meta.json").read_text()))

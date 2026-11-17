"""Cold-start benchmark: recommending recordings collaborative filtering can't see.

The warm catalog (data.build) keeps recordings with >= 10 training listeners.
Everything else is invisible to ItemKNN, EASE and ALS: they have no row for it.
Most obscure music lives there.

Protocol
  cold catalog   recordings outside the warm catalog that have AcousticBrainz
                 features and are played by someone in the test window. (A
                 recommender knows a new recording exists when it is added to
                 the catalog; it doesn't know who will play it.)
  targets        each warm user's test-window plays of cold recordings they had
                 not played in training
  slices         "known artist": the recording's artist credit has warm
                 recordings; "new artist": it has none, so only the audio is left

Scorers rank the whole cold catalog for each user. All of them see only
training data plus the recordings' content (audio descriptors, artist credit).
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
FEATURES = ROOT / "data" / "acousticbrainz.parquet"

NUMERIC = ["average_loudness", "dynamic_complexity", "mfcc_zero_mean", "bpm",
           "bpm_histogram_first_peak_bpm_mean", "bpm_histogram_second_peak_bpm_mean",
           "danceability", "onset_rate", "tuning_frequency", "tuning_equal_tempered_deviation"]
LOG = {"bpm", "bpm_histogram_first_peak_bpm_mean", "bpm_histogram_second_peak_bpm_mean"}
# circle of fifths position, so neighbouring keys are neighbours in feature space
FIFTHS = {k: i for i, k in enumerate(["C", "G", "D", "A", "E", "B", "F#", "C#", "G#", "D#", "A#", "F"])}
ENHARMONIC = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#"}


# ---------------------------------------------------------------- features

class AudioEncoder:
    """Raw AcousticBrainz descriptors -> 13 standardised numbers.

    Numeric descriptors are clipped to their 1st-99th percentile and z-scored
    (statistics fitted on warm recordings only); tempo is logged; key becomes
    (sin, cos) on the circle of fifths and scale becomes +/-1."""

    def fit(self, df: pd.DataFrame) -> "AudioEncoder":
        raw = self._numeric(df)
        self.lo, self.hi = np.nanquantile(raw, [0.01, 0.99], axis=0)
        clipped = np.clip(raw, self.lo, self.hi)
        self.mu, self.sd = np.nanmean(clipped, axis=0), np.nanstd(clipped, axis=0) + 1e-9
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        z = (np.clip(self._numeric(df), self.lo, self.hi) - self.mu) / self.sd
        z = np.nan_to_num(z, nan=0.0)
        key = df["key_key"].map(lambda k: FIFTHS.get(ENHARMONIC.get(k, k), np.nan)).to_numpy(float)
        ang = 2 * np.pi * key / 12
        sin, cos = np.nan_to_num(np.sin(ang)), np.nan_to_num(np.cos(ang))
        scale = np.where(df["key_scale"].to_numpy() == "major", 1.0, -1.0)
        return np.column_stack([z, sin, cos, scale]).astype(np.float32)

    @staticmethod
    def _numeric(df: pd.DataFrame) -> np.ndarray:
        cols = [np.log(np.maximum(df[c].to_numpy(float), 1.0)) if c in LOG else df[c].to_numpy(float)
                for c in NUMERIC]
        return np.column_stack(cols)


# ---------------------------------------------------------------- dataset

@dataclass
class ColdSet:
    items: pd.DataFrame            # cold item j: mbid, artist_credit_id, train_listeners, known_artist
    audio_cold: np.ndarray         # (n_cold, d) encoded features
    audio_warm: np.ndarray         # (n_warm, d), zeros where missing
    warm_has_audio: np.ndarray     # (n_warm,) bool
    seen: sp.csr_matrix            # (n_users, n_cold) training plays of cold items (excluded)
    users: np.ndarray              # warm user rows with >= 1 cold discovery
    targets: list[np.ndarray]      # aligned with users: cold item indices
    meta: dict

    @property
    def n_items(self) -> int:
        return len(self.items)


def build_cold(ds: D.Dataset, features: Path = FEATURES, listens_dir: Path = D.LISTENS,
               redirect: Path | None = D.REDIRECT) -> ColdSet:
    split, end = ds.meta["split"], ds.meta["end"]
    files = sorted(listens_dir.glob("*.parquet"))
    train_files = [str(p) for p in files if p.stem < split]
    test_files = [str(p) for p in files if split <= p.stem < end]
    canonical = redirect is not None and Path(redirect).exists()
    mapped = "coalesce(r.canonical_recording_mbid, t.recording_mbid)" if canonical else "t.recording_mbid"
    join = f"LEFT JOIN read_parquet('{redirect}') r USING (recording_mbid)" if canonical else ""

    con = duckdb.connect()
    spill = ROOT / "data" / "raw" / "duckdb_tmp"
    spill.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET memory_limit='3GB'; SET threads=4; SET preserve_insertion_order=false; "
                f"SET temp_directory='{spill}'")
    con.register("warm_items", ds.items[["mbid"]])
    con.register("warm_users", pd.DataFrame({"user_id": ds.user_ids}))
    con.execute(f"CREATE TEMP TABLE feat AS SELECT * FROM read_parquet('{features}')")
    con.execute(f"""
        CREATE TEMP TABLE test AS
        WITH t AS (SELECT DISTINCT user_id, recording_mbid, artist_credit_id FROM read_parquet({test_files})
                   WHERE listened_at >= TIMESTAMP '{split}' AND listened_at < TIMESTAMP '{end}')
        SELECT t.user_id, {mapped} AS mbid, any_value(t.artist_credit_id) AS artist_credit_id
        FROM t {join} GROUP BY 1, 2
    """)
    # cold catalog: not warm, has audio, played by anyone in the test window
    con.execute("""
        CREATE TEMP TABLE cold AS
        SELECT mbid, any_value(artist_credit_id) AS artist_credit_id
        FROM test SEMI JOIN feat USING (mbid) ANTI JOIN warm_items USING (mbid)
        GROUP BY 1
    """)
    con.execute(f"""
        CREATE TEMP TABLE cold_train AS
        WITH t AS (SELECT DISTINCT user_id, recording_mbid FROM read_parquet({train_files})
                   WHERE listened_at < TIMESTAMP '{split}')
        SELECT DISTINCT t.user_id, {mapped} AS mbid FROM t {join}
        SEMI JOIN cold ON cold.mbid = {mapped}
    """)
    items = con.execute("""
        SELECT c.mbid, c.artist_credit_id, coalesce(n.listeners, 0)::INT AS train_listeners
        FROM cold c LEFT JOIN (SELECT mbid, count(*) AS listeners FROM cold_train GROUP BY 1) n USING (mbid)
        ORDER BY c.mbid
    """).df()
    feats = con.execute("SELECT * FROM feat").df().set_index("mbid")
    seen_pairs = con.execute("SELECT * FROM cold_train SEMI JOIN warm_users USING (user_id)").df()
    test = con.execute("""
        SELECT t.user_id, t.mbid FROM test t SEMI JOIN cold USING (mbid) SEMI JOIN warm_users USING (user_id)
    """).df()
    n_warm_test = con.execute("""
        SELECT count(*) FROM test t SEMI JOIN warm_users USING (user_id)
        ANTI JOIN warm_items USING (mbid) ANTI JOIN feat USING (mbid)
    """).fetchone()[0]
    con.close()

    warm_artists = set(ds.items["artist_credit_id"].dropna().astype("int64"))
    items["known_artist"] = items["artist_credit_id"].map(
        lambda a: pd.notna(a) and int(a) in warm_artists).astype(bool)

    enc = AudioEncoder()
    warm_feat = feats.reindex(ds.items["mbid"])
    has = warm_feat["average_loudness"].notna().to_numpy()
    enc.fit(warm_feat[has])
    audio_warm = np.zeros((ds.n_items, 13), dtype=np.float32)
    audio_warm[has] = enc.transform(warm_feat[has])
    audio_cold = enc.transform(feats.loc[items["mbid"]])

    user_index = pd.Series(np.arange(ds.n_users), index=ds.user_ids)
    item_index = pd.Series(np.arange(len(items)), index=items["mbid"])
    seen = sp.csr_matrix((np.ones(len(seen_pairs), np.float32),
                          (user_index[seen_pairs["user_id"]].to_numpy(),
                           item_index[seen_pairs["mbid"]].to_numpy())),
                         shape=(ds.n_users, len(items)))
    rows = user_index[test["user_id"]].to_numpy()
    cols = item_index[test["mbid"]].to_numpy()
    keep = np.asarray(seen[rows, cols]).ravel() == 0
    rows, cols = rows[keep], cols[keep]
    order = np.lexsort((cols, rows))
    rows, cols = rows[order], cols[order]
    users, starts = np.unique(rows, return_index=True)
    targets = np.split(cols, starts[1:])

    meta = {"split": split, "end": end, "n_cold_items": int(len(items)),
            "known_artist_items": float(items["known_artist"].mean()),
            "users": int(len(users)), "discoveries": int(len(cols)),
            "warm_items_with_audio": float(has.mean()),
            "cold_test_pairs_without_audio": int(n_warm_test),
            "train_listeners_hist": np.bincount(items["train_listeners"], minlength=10).tolist()}
    return ColdSet(items=items, audio_cold=audio_cold, audio_warm=audio_warm,
                   warm_has_audio=has, seen=seen, users=users, targets=targets, meta=meta)


# ---------------------------------------------------------------- scorers

class ColdScorer:
    name = "base"

    def score(self, users: np.ndarray) -> np.ndarray:  # (len(users), n_cold)
        raise NotImplementedError


def cold_top_k(scorer: ColdScorer, cs: ColdSet, users: np.ndarray, k: int) -> np.ndarray:
    out = np.empty((len(users), k), dtype=np.int64)
    batch = max(16, int(2e7 // cs.n_items))
    for s in range(0, len(users), batch):
        u = users[s:s + batch]
        S = scorer.score(u).astype(np.float32, copy=True)
        sub = cs.seen[u]
        S[np.repeat(np.arange(len(u)), np.diff(sub.indptr)), sub.indices] = -np.inf
        part = np.argpartition(-S, k, axis=1)[:, :k]
        ps = np.take_along_axis(S, part, axis=1)
        o = np.lexsort((part, -ps), axis=1)
        out[s:s + batch] = np.take_along_axis(part, o, axis=1)
    return out


class Random(ColdScorer):
    name = "Random"

    def __init__(self, cs: ColdSet, seed: int = 0):
        self.n, self.rng = cs.n_items, np.random.default_rng(seed)

    def score(self, users):
        return self.rng.random((len(users), self.n), dtype=np.float32)


class ColdPopularity(ColdScorer):
    """Rank by training listeners (0-9): the only CF signal a cold item has."""
    name = "Cold popularity"

    def __init__(self, cs: ColdSet):
        self.s = cs.items["train_listeners"].to_numpy(np.float32)

    def score(self, users):
        return np.broadcast_to(self.s, (len(users), len(self.s)))


def _artist_codes(ds: D.Dataset, cs: ColdSet) -> tuple[np.ndarray, np.ndarray, int]:
    """Shared integer codes for artist credits across warm and cold items (-1 = none)."""
    warm = ds.items["artist_credit_id"]
    cold = cs.items["artist_credit_id"]
    allc = pd.concat([warm, cold]).astype("Int64")
    codes, uniq = pd.factorize(allc, use_na_sentinel=True)
    return codes[:len(warm)], codes[len(warm):], len(uniq)


class ArtistAffinity(ColdScorer):
    """Share of the user's training plays that went to the item's artist credit,
    tie-broken by cold popularity. Recommends only artists you already play."""
    name = "Artist affinity"

    def __init__(self, ds: D.Dataset, cs: ColdSet):
        wc, cc, n = _artist_codes(ds, cs)
        P = ds.plays.tocoo()
        ok = wc[P.col] >= 0
        A = sp.csr_matrix((P.data[ok], (P.row[ok], wc[P.col[ok]])), shape=(ds.n_users, n))
        tot = np.asarray(ds.plays.sum(axis=1)).ravel()
        self.A = sp.diags(1.0 / np.maximum(tot, 1)) @ A
        self.cc = cc
        self.tie = 1e-4 * cs.items["train_listeners"].to_numpy(np.float32) / 10

    def score(self, users):
        A = self.A[users].toarray()
        out = np.where(self.cc >= 0, A[:, np.maximum(self.cc, 0)], 0.0)
        return out + self.tie


class AudioProfile(ColdScorer):
    """Cosine between the item's audio vector and the mean audio vector of the
    user's (warm) history: "more music that sounds like what you play"."""
    name = "Audio profile"

    def __init__(self, ds: D.Dataset, cs: ColdSet):
        X = ds.X.multiply(cs.warm_has_audio[None, :]).tocsr()
        n = np.maximum(np.asarray(X.sum(axis=1)).ravel(), 1)
        prof = sp.diags(1.0 / n) @ X @ cs.audio_warm
        self.prof = prof / (np.linalg.norm(prof, axis=1, keepdims=True) + 1e-9)
        V = cs.audio_cold
        self.V = (V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)).T

    def score(self, users):
        return self.prof[users] @ self.V


class FactorScorer(ColdScorer):
    """score(u, j) = user factor . predicted item factor (cold items)."""

    def __init__(self, name: str, user_factors: np.ndarray, item_vectors: np.ndarray):
        self.name, self.U, self.V = name, user_factors, item_vectors.T.astype(np.float32)

    def score(self, users):
        return self.U[users] @ self.V


def artist_centroids(item_factors: np.ndarray, warm_codes: np.ndarray, n_codes: int,
                     leave_one_out: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Mean CF factor of each artist's warm recordings. With leave_one_out, row i
    of the result is item i's artist centroid *excluding item i* (for training
    a mapping without letting an item see itself); otherwise one row per code.
    Returns (vectors, has_vector)."""
    ok = warm_codes >= 0
    sums = np.zeros((n_codes, item_factors.shape[1]), dtype=np.float64)
    np.add.at(sums, warm_codes[ok], item_factors[ok])
    counts = np.bincount(warm_codes[ok], minlength=n_codes).astype(np.float64)
    if not leave_one_out:
        has = counts > 0
        return (sums / np.maximum(counts, 1)[:, None]).astype(np.float32), has
    c = np.where(ok, warm_codes, 0)
    n_other = np.where(ok, counts[c] - 1, 0)
    other = sums[c] - item_factors
    has = n_other > 0
    vec = np.where(has[:, None], other / np.maximum(n_other, 1)[:, None], 0.0)
    return vec.astype(np.float32), has


class ContentToFactor:
    """Regress CF item factors from content, then use the prediction for cold
    items (van den Oord, Dieleman & Schrauwen, "Deep content-based music
    recommendation", NeurIPS 2013, with audio descriptors in place of spectrograms).

    Input = [audio (13), artist centroid (F), has-centroid flag]; either part can
    be switched off. Trained on warm items with audio, MSE on factors, small MLP.
    """

    def __init__(self, use_audio: bool = True, use_artist: bool = True, hidden: int = 512,
                 epochs: int = 30, weight_decay: float = 1e-4, lr: float = 1e-3, seed: int = 0):
        self.use_audio, self.use_artist = use_audio, use_artist
        self.hidden, self.epochs, self.wd, self.lr, self.seed = hidden, epochs, weight_decay, lr, seed

    def params(self) -> dict:
        return {"audio": self.use_audio, "artist": self.use_artist, "hidden": self.hidden,
                "epochs": self.epochs, "weight_decay": self.wd}

    def _inputs(self, audio, centroid, has):
        parts = []
        if self.use_audio:
            parts.append(audio)
        if self.use_artist:
            parts += [centroid * self.scale, has[:, None].astype(np.float32)]
        return np.hstack(parts).astype(np.float32)

    def fit(self, audio, centroid, has, target, val_frac: float = 0.05):
        import torch
        torch.manual_seed(self.seed)
        self.scale = 1.0 / (np.abs(target).std() + 1e-9)
        self.tscale = self.scale
        Xin = self._inputs(audio, centroid, has)
        Y = (target * self.tscale).astype(np.float32)
        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(len(Xin))
        n_val = int(len(Xin) * val_frac)
        va, tr = perm[:n_val], perm[n_val:]
        d_in, d_out = Xin.shape[1], Y.shape[1]
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_in, self.hidden), torch.nn.ReLU(),
            torch.nn.Linear(self.hidden, self.hidden), torch.nn.ReLU(),
            torch.nn.Linear(self.hidden, d_out))
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        Xt, Yt = torch.from_numpy(Xin), torch.from_numpy(Y)
        best, best_state, self.curve = np.inf, None, []
        for ep in range(self.epochs):
            self.net.train()
            for b in np.array_split(rng.permutation(tr), max(1, len(tr) // 1024)):
                opt.zero_grad()
                loss = torch.nn.functional.mse_loss(self.net(Xt[b]), Yt[b])
                loss.backward()
                opt.step()
            self.net.eval()
            with torch.no_grad():
                v = float(torch.nn.functional.mse_loss(self.net(Xt[va]), Yt[va]))
            self.curve.append(v)
            if v < best:
                best, best_state = v, {k: t.clone() for k, t in self.net.state_dict().items()}
        self.net.load_state_dict(best_state)
        base = float(((Y[va] - Y[tr].mean(axis=0)) ** 2).mean())
        self.val_r2 = 1 - best / base
        return self

    def predict(self, audio, centroid, has) -> np.ndarray:
        import torch
        with torch.no_grad():
            out = self.net(torch.from_numpy(self._inputs(audio, centroid, has))).numpy()
        return out / self.tscale

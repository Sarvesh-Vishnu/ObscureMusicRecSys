from datetime import datetime

import numpy as np
import pandas as pd

from obscure.lab import coldstart as C
from obscure.lab import data as D


def _feat_rows(mbids):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"mbid": mbids, "submissions": 1})
    for c in C.NUMERIC:
        df[c] = rng.random(len(mbids)) + 100.0
    df["key_key"] = "C"
    df["key_scale"] = "major"
    return df


def _day(path, rows):
    df = pd.DataFrame(rows, columns=["user_id", "recording_mbid", "artist_credit_id", "listened_at"])
    df["artist_name"] = "A"
    df["recording_name"] = df["recording_mbid"]
    df["release_name"] = "R"
    df["listened_at"] = pd.to_datetime(df["listened_at"])
    df.to_parquet(path)


def _cold_fixture(tmp_path):
    t = lambda d: datetime(2026, 11, d, 12)  # noqa: E731
    listens = tmp_path / "listens"
    listens.mkdir()
    train = [(u, w, 10, t(1)) for u in (1, 2, 3) for w in ("w0", "w1")]
    train += [(1, "c0", 10, t(1))]                       # c0: one training listener -> cold
    _day(listens / "2026-11-01.parquet", train)
    _day(listens / "2026-11-03.parquet", [
        (2, "c0", 10, t(3)),   # cold, new to user 2, artist has warm recordings
        (1, "c0", 10, t(3)),   # user 1 played it in training: not a target
        (3, "c1", 20, t(3)),   # cold, zero training listeners, new artist
        (3, "c2", 20, t(3)),   # no audio features: outside the cold catalog
        (2, "w0", 10, t(3)),   # warm: not part of the cold benchmark
    ])
    feats = tmp_path / "ab.parquet"
    _feat_rows(["w0", "w1", "c0", "c1"]).to_parquet(feats)
    ds = D.build(split="2026-11-03", end="2026-11-05", min_item_users=2, min_user_items=1,
                 listens_dir=listens, redirect=None)
    return ds, C.build_cold(ds, features=feats, listens_dir=listens, redirect=None)


def test_cold_catalog_and_targets(tmp_path):
    ds, cs = _cold_fixture(tmp_path)
    assert sorted(ds.items["mbid"]) == ["w0", "w1"]
    items = cs.items.set_index("mbid")
    assert sorted(items.index) == ["c0", "c1"]               # c2 has no audio
    assert items.loc["c0", "train_listeners"] == 1 and items.loc["c1", "train_listeners"] == 0
    assert bool(items.loc["c0", "known_artist"]) and not bool(items.loc["c1", "known_artist"])
    mb = cs.items["mbid"].tolist()
    got = {int(ds.user_ids[u]): [mb[j] for j in t] for u, t in zip(cs.users, cs.targets)}
    assert got == {2: ["c0"], 3: ["c1"]}
    assert cs.warm_has_audio.all()


def test_cold_top_k_excludes_training_plays(tmp_path):
    ds, cs = _cold_fixture(tmp_path)
    u1 = int(np.flatnonzero(ds.user_ids == 1)[0])
    recs = C.cold_top_k(C.ColdPopularity(cs), cs, np.array([u1]), 1)
    assert cs.items["mbid"][recs[0, 0]] == "c1"              # c0 is more popular but already heard


def test_artist_affinity_prefers_played_artist(tmp_path):
    ds, cs = _cold_fixture(tmp_path)
    u3 = int(np.flatnonzero(ds.user_ids == 3)[0])
    s = C.ArtistAffinity(ds, cs).score(np.array([u3]))[0]
    j = dict(zip(cs.items["mbid"], range(cs.n_items)))
    assert s[j["c0"]] > s[j["c1"]]


def test_audio_encoder_key_circle():
    df = _feat_rows(["a", "b", "c"])
    df["key_key"] = ["C", "G", "F#"]
    z = C.AudioEncoder().fit(df).transform(df)
    k = z[:, -3:-1]                                          # (sin, cos) on the circle of fifths
    assert np.linalg.norm(k[0] - k[1]) < np.linalg.norm(k[0] - k[2])
    assert z.shape == (3, 13)


def test_artist_centroids_leave_one_out():
    V = np.array([[1.0], [3.0], [5.0], [7.0]], dtype=np.float32)
    codes = np.array([0, 0, 1, -1])
    loo, has = C.artist_centroids(V, codes, 2, leave_one_out=True)
    assert has.tolist() == [True, True, False, False]
    np.testing.assert_allclose(loo[:2, 0], [3.0, 1.0])
    full, has_code = C.artist_centroids(V, codes, 2)
    np.testing.assert_allclose(full[:, 0], [2.0, 5.0])
    assert has_code.tolist() == [True, True]


def test_content_to_factor_learns_a_simple_map():
    rng = np.random.default_rng(1)
    audio = rng.normal(size=(4000, 13)).astype(np.float32)
    W = rng.normal(size=(13, 8)).astype(np.float32)
    target = audio @ W
    cent = np.zeros((4000, 8), np.float32)
    has = np.zeros(4000, bool)
    m = C.ContentToFactor(use_artist=False, hidden=64, epochs=150).fit(audio, cent, has, target)
    assert m.val_r2 > 0.9
    assert m.predict(audio[:5], cent[:5], has[:5]).shape == (5, 8)

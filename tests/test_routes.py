from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import scipy.sparse as sp

from obscure.lab import data as D
from obscure.lab import routes as RT
from obscure.lab import sequence as S


def _day(path, rows):
    df = pd.DataFrame(rows, columns=["user_id", "recording_mbid", "listened_at"])
    df["artist_credit_id"] = 1
    df["artist_name"] = "A"
    df["recording_name"] = df["recording_mbid"]
    df["release_name"] = "R"
    df.to_parquet(path)


def test_transitions_respect_gap_split_and_catalog(tmp_path):
    t0 = datetime(2026, 11, 1, 12)
    m = lambda x: t0 + timedelta(minutes=x)  # noqa: E731
    rows = []
    for u in (1, 2):                                   # warm catalog: i0, i1, i2 (2 listeners each)
        rows += [(u, "i0", m(0)), (u, "i1", m(5)), (u, "i2", m(10))]
    rows += [(1, "x", m(12)), (1, "i0", m(14))]        # x not in catalog: no i2->x or x->i0 ...
    rows += [(2, "i0", m(200))]                        # ... and a 190-minute gap is not a transition
    rows += [(2, "i0", m(201))]                        # a repeat is not a transition
    _day(tmp_path / "2026-11-01.parquet", rows)
    _day(tmp_path / "2026-11-03.parquet", [(1, "i2", datetime(2026, 11, 3, 9)),
                                           (1, "i1", datetime(2026, 11, 3, 9, 3)),
                                           (2, "i1", datetime(2026, 11, 2, 9))])   # played before split: ignored
    ds = D.build(split="2026-11-03", end="2026-11-05", min_item_users=2, min_user_items=1,
                 listens_dir=tmp_path, redirect=None)
    tr = S.build_transitions(ds, listens_dir=tmp_path, redirect=None)
    idx = {mb: i for i, mb in enumerate(ds.items["mbid"])}
    T = tr.T.toarray()
    assert T[idx["i0"], idx["i1"]] == 2 and T[idx["i1"], idx["i2"]] == 2
    assert T.sum() == 4                                 # nothing through x, across the gap, or repeated
    assert len(tr.test) == 1
    row = tr.test.iloc[0]
    assert (row["a"], row["b"]) == (idx["i2"], idx["i1"]) and not row["new_to_user"]


def test_next_song_eval_counts_hits():
    T = sp.csr_matrix(np.array([[0, 5, 1], [1, 0, 0], [0, 0, 0]], dtype=np.float32))
    m = S.NextSong(T, None, pop=np.array([1.0, 1.0, 1.0], np.float32))
    r = S.next_song_eval(m, np.array([0, 0]), np.array([1, 2]), k=1)
    assert r["hit"].tolist() == [1.0, 0.0]
    assert r["rr"][0] == 1.0


def _line_graph():
    # 0 -> 1 -> 2 -> 3 and a tempting shortcut 0 -> 4 -> 5
    n = 6
    T = np.zeros((n, n), np.float32)
    T[0, 1], T[1, 2], T[2, 3] = 3, 3, 3
    T[0, 4], T[4, 5] = 5, 5
    model = S.NextSong(sp.csr_matrix(T), None, pop=np.ones(n, np.float32))
    audio = np.zeros((n, 13), np.float32)
    audio[4:, 0] = 10.0                                  # 4 and 5 sound far from 0
    g = RT.build_graph(model, n, audio, np.ones(n, bool), M=3)
    return g, n


def test_route_obeys_blocks_artists_and_tail():
    g, n = _line_graph()
    artist = np.arange(n)
    tail = np.zeros(n, bool)
    tail[[3, 5]] = True
    free = np.zeros(n, bool)
    assert RT.find_route(g, 0, 2, blocked=free, artist=artist, tail=tail) == [4, 5]
    blocked = free.copy()
    blocked[4] = True
    assert RT.find_route(g, 0, 3, blocked=blocked, artist=artist, tail=tail) == [1, 2, 3]
    same_artist = artist.copy()
    same_artist[4] = same_artist[0]
    assert RT.find_route(g, 0, 3, blocked=free, artist=same_artist, tail=tail) == [1, 2, 3]


def test_audio_penalty_prefers_smooth_route():
    g, n = _line_graph()
    artist = np.arange(n)
    tail = np.ones(n, bool)
    free = np.zeros(n, bool)
    assert RT.find_route(g, 0, 2, blocked=free, artist=artist, tail=tail, lam=0.0)[0] == 4
    assert RT.find_route(g, 0, 2, blocked=free, artist=artist, tail=tail, lam=5.0)[0] == 1


def test_step_metrics():
    keys = RT.pair_keys(np.array([0, 1]), np.array([1, 2]), 10)
    audio = np.zeros((10, 13), np.float32)
    audio[2, 0] = 2.0
    m = RT.step_metrics([[0, 1, 2, 3]], keys, 10, audio, np.ones(10, bool))
    assert m["realism"] == 2 / 3
    assert np.isclose(m["audio_step"], (0 + 2 + 2) / 3)

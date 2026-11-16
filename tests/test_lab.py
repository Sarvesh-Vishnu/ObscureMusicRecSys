"""The harness: metrics by hand, models against brute force, protocol vs leakage."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from obscure.lab import data as D
from obscure.lab import metrics as M
from obscure.lab import models as R


# ------------------------------------------------------------------ metrics

def test_recall_and_ndcg_by_hand():
    recs = np.array([[5, 1, 9]])
    targets = [np.array([1, 7])]
    pop = np.ones(10)
    out, cov = M.evaluate_lists(recs, targets, popularity=pop, n_users=10,
                                tail=np.zeros(10, bool), artist=np.arange(10), k=3)
    assert out["recall"][0] == pytest.approx(1 / 2)
    # hit at rank 2: DCG = 1/log2(3); ideal over min(2,3)=2 hits = 1 + 1/log2(3)
    assert out["ndcg"][0] == pytest.approx((1 / np.log2(3)) / (1 + 1 / np.log2(3)))
    assert np.isnan(out["tail_recall"][0])  # no tail targets -> undefined, not 0
    assert cov == pytest.approx(3 / 10)


def test_beyond_accuracy_metrics():
    recs = np.array([[0, 1, 2, 3]])
    tail = np.array([False, False, True, True])
    artist = np.array([0, 0, 1, 2])
    pop = np.array([8, 4, 2, 1])
    out, _ = M.evaluate_lists(recs, [np.array([3])], popularity=pop, n_users=8,
                              tail=tail, artist=artist, k=4)
    assert out["tail_share"][0] == 0.5
    assert out["artist_div"][0] == 0.75
    assert out["novelty"][0] == pytest.approx(np.mean([0, 1, 2, 3]))
    assert out["tail_recall"][0] == 1.0


def test_head_mask_holds_half_the_pairs():
    pop = np.array([50, 30, 10, 5, 5])
    head = D.head_mask(pop, 0.5)
    assert head.tolist() == [True, False, False, False, False]


def test_bootstrap_ci_brackets_mean():
    v = np.random.default_rng(1).normal(0.3, 0.1, 2000)
    m, lo, hi = M.bootstrap_ci(v)
    assert lo < m < hi and hi - lo < 0.02


# ------------------------------------------------------------------ models

@pytest.fixture
def X():
    rng = np.random.default_rng(0)
    return sp.csr_matrix((rng.random((40, 25)) < 0.25).astype(np.float32))


def test_top_k_never_returns_seen_items(X):
    users = np.arange(X.shape[0])
    recs = R.top_k(R.Popularity().fit(X), X, users, k=5)
    for u in users:
        assert not set(recs[u]) & set(X[u].indices)


def test_itemknn_matches_brute_force_cosine(X):
    m = R.ItemKNN(k=X.shape[1] - 1, shrink=0.0).fit(X)
    A = X.toarray()
    co = A.T @ A
    n = np.sqrt(np.diag(co))
    S = co / np.outer(n, n)
    np.fill_diagonal(S, 0)
    np.testing.assert_allclose(m.W.toarray(), S, rtol=1e-5, atol=1e-6)


def test_ease_closed_form(X):
    m = R.EASE(lam=10.0, max_items=X.shape[1]).fit(X)
    assert np.allclose(np.diag(m.B), 0)
    A = X.toarray().astype(np.float64)
    P = np.linalg.inv(A.T @ A + 10 * np.eye(A.shape[1]))
    B = -P / np.diag(P)
    np.fill_diagonal(B, 0)
    np.testing.assert_allclose(m.B, B, rtol=1e-3, atol=1e-5)


def test_ease_caps_to_popular_items(X):
    m = R.EASE(lam=10.0, max_items=10).fit(X)
    S = m.score(np.arange(3), X)
    assert np.isinf(S).sum(axis=1).tolist() == [X.shape[1] - 10] * 3


# ------------------------------------------------------------------ protocol

def _write_day(path, rows):
    df = pd.DataFrame(rows, columns=["user_id", "recording_mbid", "listened_at"])
    df["artist_credit_id"] = 1
    df["artist_name"] = "A"
    df["recording_name"] = df["recording_mbid"]
    df["release_name"] = "R"
    df["listened_at"] = pd.to_datetime(df["listened_at"])
    df.to_parquet(path)


def test_split_has_no_leakage(tmp_path):
    t = lambda d: datetime(2026, 11, d, 12)  # noqa: E731
    train = [(u, f"i{i}", t(1)) for u in (1, 2, 3) for i in range(3)]   # everyone: i0..i2
    train += [(1, "i4", t(1)), (2, "i4", t(1))]                          # only users 1, 2: i4
    _write_day(tmp_path / "2026-11-01.parquet", train)
    _write_day(tmp_path / "2026-11-03.parquet", [
        (1, "i0", t(3)),   # already heard -> not a discovery
        (3, "i4", t(3)),   # new to user 3, in the universe -> DISCOVERY
        (3, "i1", t(2)),   # submitted after the split, played before it -> ignored
        (2, "i4", t(2)),   # same: must not double-count into training
    ])
    _write_day(tmp_path / "2026-11-04.parquet", [(2, "i9", t(4))])  # i9 never in train

    ds = D.build(split="2026-11-03", end="2026-11-05", min_item_users=1, min_user_items=1,
                 listens_dir=tmp_path, redirect=None)
    mbid = ds.items["mbid"].tolist()
    by_user = {int(ds.user_ids[u]): sorted(mbid[i] for i in tg)
               for u, tg in zip(ds.eval_users, ds.targets)}
    assert by_user == {3: ["i4"]}
    assert ds.X.nnz == 11                                     # 9 + 2, nothing from after the split
    assert ds.plays.sum() == 11                               # late-submitted i4 not counted
    assert dict(zip(mbid, ds.popularity))["i4"] == 2


def test_canonical_redirect_merges_versions(tmp_path):
    t = lambda d: datetime(2026, 11, d, 12)  # noqa: E731
    listens = tmp_path / "listens"
    listens.mkdir()
    # users 1-2 played the album version, user 3 the single; both are one song
    _write_day(listens / "2026-11-01.parquet",
               [(1, "album", t(1)), (2, "album", t(1)), (3, "single", t(1)), (3, "other", t(1))])
    # user 3 later plays the album version: a re-listen, not a discovery
    _write_day(listens / "2026-11-03.parquet", [(3, "album", t(3))])
    red = tmp_path / "redirect.parquet"
    pd.DataFrame({"recording_mbid": ["single"], "canonical_recording_mbid": ["album"]}).to_parquet(red)

    ds = D.build(split="2026-11-03", end="2026-11-05", min_item_users=1, min_user_items=1,
                 listens_dir=listens, redirect=red)
    assert dict(zip(ds.items["mbid"], ds.popularity)) == {"album": 3, "other": 1}
    assert len(ds.eval_users) == 0
    assert ds.meta["canonical_recordings"] is True


def test_itemknn_top_k_keeps_largest_similarities(X):
    k, shrink = 3, 2.0
    W = R.itemknn_neighbours(X, [(k, shrink), (5, 0.0)])[(k, shrink)].toarray()
    A = X.toarray()
    co = A.T @ A
    n = np.sqrt(np.diag(co))
    S = co / (np.outer(n, n) + shrink)
    np.fill_diagonal(S, 0)
    for i in range(S.shape[0]):
        expect = np.sort(S[i][S[i] > 0])[::-1][:k]
        got = np.sort(W[i][W[i] > 0])[::-1]
        np.testing.assert_allclose(got, expect, rtol=1e-5)


def test_truncate_matches_direct_fit(X):
    wide = R.itemknn_neighbours(X, [(12, 3.0)])[(12, 3.0)]
    direct = R.itemknn_neighbours(X, [(4, 3.0)])[(4, 3.0)].toarray()
    cut = R.truncate_neighbours(wide, 4).toarray()
    for i in range(X.shape[1]):
        np.testing.assert_allclose(np.sort(cut[i]), np.sort(direct[i]), rtol=1e-6)


def test_ease_gram_reuse(X):
    g = R.EASE.gram(X, X.shape[1])
    a = R.EASE(lam=7.0, max_items=X.shape[1]).fit(X, gram=g).B
    b = R.EASE(lam=7.0, max_items=X.shape[1]).fit(X).B
    np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(R.EASE(lam=7.0, max_items=X.shape[1]).fit(X, gram=g).B, a)  # gram untouched


def test_cap_history_keeps_most_played():
    plays = sp.csr_matrix(np.array([[5, 1, 3, 0], [1, 0, 0, 2]], dtype=np.float32))
    capped = R.cap_history(plays, 2).toarray()
    assert capped.tolist() == [[1, 0, 1, 0], [1, 0, 0, 1]]
    assert R.cap_history(plays, None).toarray().tolist() == [[1, 1, 1, 0], [1, 0, 0, 1]]

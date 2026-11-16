import numpy as np
import pytest
import scipy.sparse as sp

from obscure.lab import models as R
from obscure.lab import rerank as RR


@pytest.fixture
def setup():
    rng = np.random.default_rng(3)
    X = sp.csr_matrix((rng.random((30, 60)) < 0.2).astype(np.float32))
    knn = R.ItemKNN(k=20, shrink=1.0).fit(X)
    pop = np.asarray(X.sum(axis=0)).ravel()
    users = np.arange(30)
    items, rel = RR.candidates(knn, X, users, c=25)
    return X, knn, pop, users, items, rel


def test_candidates_are_base_top_k_with_scaled_scores(setup):
    X, knn, _, users, items, rel = setup
    np.testing.assert_array_equal(items, R.top_k(knn, X, users, 25))
    assert rel.min() >= 0 and rel.max() <= 1
    assert np.all(np.diff(rel, axis=1) <= 1e-6)  # still in descending order


def test_strength_zero_reproduces_base(setup):
    X, knn, pop, users, items, rel = setup
    base = items[:, :10]
    groups = RR.popularity_groups(pop)
    np.testing.assert_array_equal(RR.calibrated(items, rel, X, users, groups, 0.0, 10), base)
    np.testing.assert_array_equal(RR.mmr(items, rel, knn.W, 0.0, 10), base)
    np.testing.assert_array_equal(R.top_k(RR.PopPenalty(knn, pop, 0.0), X, users, 10), base)


def test_pop_penalty_lowers_popularity(setup):
    X, knn, pop, users, *_ = setup
    plain = R.top_k(knn, X, users, 10)
    pen = R.top_k(RR.PopPenalty(knn, pop, 1.0), X, users, 10)
    assert pop[pen].mean() < pop[plain].mean()


def test_full_calibration_matches_history_mix():
    # one user: history 50% group 0, 50% group 2; candidates mostly group 0
    X = sp.csr_matrix(np.array([[1, 1, 0, 0, 0, 0, 0, 0, 1, 1]], dtype=np.float32))
    groups = np.array([0, 0, 0, 0, 0, 0, 0, 2, 2, 2], dtype=np.int8)
    items = np.array([[2, 3, 4, 5, 6, 7]])
    rel = np.array([[1.0, 0.9, 0.8, 0.7, 0.6, 0.1]])
    out = RR.calibrated(items, rel, X, np.array([0]), groups, lam=0.99, k=2)
    assert sorted(groups[out[0]].tolist()) == [0, 2]   # pulls in the low-relevance tail item


def test_mmr_avoids_near_duplicates():
    W = sp.csr_matrix(np.array([[0, .99, 0], [.99, 0, 0], [0, 0, 0]], dtype=np.float32))
    items = np.array([[0, 1, 2]])
    rel = np.array([[1.0, 0.95, 0.5]])
    assert RR.mmr(items, rel, W, 0.0, 2)[0].tolist() == [0, 1]
    assert RR.mmr(items, rel, W, 0.5, 2)[0].tolist() == [0, 2]

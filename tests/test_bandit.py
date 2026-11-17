"""Bandit policies, off-policy estimators and online learners on synthetic contexts."""

import numpy as np

from obscure.lab import bandit as BD


def synthetic(n=4000, K=20, seed=0):
    rng = np.random.default_rng(seed)
    knn = rng.random((n, K)).astype(np.float32)
    als = rng.random((n, K)).astype(np.float32)
    logpop = rng.normal(3, 1, (n, K)).astype(np.float32)
    tail = rng.random((n, K)) < 0.4
    known = rng.random((n, K)) < 0.2
    X = BD.features(knn, als, logpop, tail, known, 3.0, 1.0)
    # reward depends on the itemknn score and known artist: a linear-logistic truth
    p = 1 / (1 + np.exp(-(-4 + 3 * knn + 1.5 * known)))
    R = (rng.random((n, K)) < p).astype(np.float64)
    return BD.Contexts(users=np.arange(n), cand=np.zeros((n, K), int), knn=knn, als=als, X=X, R=R, tail=tail, logpop=logpop)


def test_policies_are_distributions():
    ctx = synthetic(200)
    for P in (BD.greedy(ctx.knn), BD.softmax(ctx.knn, 0.1, 0.1), BD.uniform(ctx.n, ctx.K)):
        assert np.allclose(P.sum(axis=1), 1.0)
        assert (P >= 0).all()
    assert (BD.softmax(ctx.knn, 0.1, 0.1) >= 0.1 / ctx.K - 1e-12).all()   # full support for IPS


def test_true_value_is_the_expected_hit_rate():
    R = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert BD.true_value(np.array([[1.0, 0.0], [1.0, 0.0]]), R) == 0.5
    assert BD.true_value(BD.uniform(2, 2), R) == 0.5


def test_logging_samples_follow_the_policy():
    rng = np.random.default_rng(0)
    P = np.tile([0.7, 0.2, 0.1], (50_000, 1))
    a, p = BD.log_actions(P, np.arange(50_000), rng)
    assert np.allclose(np.bincount(a, minlength=3) / 50_000, [0.7, 0.2, 0.1], atol=0.01)
    assert np.allclose(p, P[0][a])


def test_ips_and_dr_are_unbiased_on_average():
    ctx = synthetic()
    P0 = BD.softmax(ctx.knn, 0.3, 0.1)
    targets = {"greedy": BD.greedy(ctx.knn), "uniform": BD.uniform(ctx.n, ctx.K)}
    truth = {k: BD.true_value(P, ctx.R) for k, P in targets.items()}
    rng = np.random.default_rng(1)
    acc = {k: {e: [] for e in ("IPS", "DR", "SNIPS")} for k in targets}
    for _ in range(40):
        rows = rng.integers(ctx.n, size=2000)
        a, p0 = BD.log_actions(P0, rows, rng)
        out = BD.estimate(ctx, rows, a, p0, ctx.R[rows, a], targets, rng)
        for k in targets:
            for e in acc[k]:
                acc[k][e].append(out[k][e])
    for k in targets:
        for e in ("IPS", "DR"):
            v = np.array(acc[k][e])
            # mean within 3 standard errors of the truth
            assert abs(v.mean() - truth[k]) < 3 * v.std() / np.sqrt(len(v)) + 1e-3, (k, e, v.mean(), truth[k])
    # DR should not be noisier than IPS for a policy far from logging
    assert np.std(acc["greedy"]["DR"]) <= np.std(acc["greedy"]["IPS"]) * 1.05


def test_effective_sample_size_falls_as_policies_diverge():
    ctx = synthetic()
    P0 = BD.softmax(ctx.knn, 0.3, 0.1)
    rng = np.random.default_rng(2)
    rows = rng.integers(ctx.n, size=3000)
    a, p0 = BD.log_actions(P0, rows, rng)
    out = BD.estimate(ctx, rows, a, p0, ctx.R[rows, a], {"same": P0, "greedy": BD.greedy(ctx.knn)}, rng)
    assert out["same"]["ess"] == 3000 * 1.0 or abs(out["same"]["ess"] - 3000) < 1
    assert out["greedy"]["ess"] < 0.5 * out["same"]["ess"]


def test_learners_beat_random_and_learn_the_signal():
    ctx = synthetic(K=20)
    rng = np.random.default_rng(3)
    order = rng.integers(ctx.n, size=6000)
    d = ctx.X.shape[-1]
    rand = BD.run_online(BD.Random(), ctx, order, np.random.default_rng(4)).mean()
    for learner in (BD.LinUCB(d, 0.2), BD.LinTS(d, 0.2), BD.EpsGreedy(d, 0.05)):
        got = BD.run_online(learner, ctx, order, np.random.default_rng(4))
        assert got[-2000:].mean() > rand + 0.05, learner.name
    lin = BD.LinUCB(d, 0.2)
    BD.run_online(lin, ctx, order, np.random.default_rng(5))
    th = lin.theta
    i_knn, i_known, i_tail = BD.FEATURES.index("itemknn score"), BD.FEATURES.index("known artist"), BD.FEATURES.index("long tail")
    assert th[i_knn] > 0 and th[i_known] > 0 and abs(th[i_tail]) < th[i_known]

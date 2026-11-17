"""Phase 6: off-policy evaluation and contextual bandits (supervised-to-bandit).

    python scripts/run_phase6.py --stage prep     # candidates, features, rewards (~10 min, fits ALS)
    python scripts/run_phase6.py --stage ope      # estimators vs the exact truth
    python scripts/run_phase6.py --stage online   # tune on half the users, report on the other half
    python scripts/run_phase6.py --stage all

Protocol in obscure/lab/bandit.py and docs/PROTOCOL.md. Every choice that is not
tuned (K, the logging policy, the log sizes) is fixed here before any result is seen.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from obscure.lab import bandit as BD          # noqa: E402
from obscure.lab import bench as B            # noqa: E402
from obscure.lab import data as D             # noqa: E402
from obscure.lab import metrics as M          # noqa: E402
from obscure.lab import models as R           # noqa: E402

RESULTS = ROOT / "results"
CACHE = D.CACHE / "bandit_k100.npz"
K = 100                                         # candidates per listener: ItemKNN's top 100
USERS = 10_000                                  # the phase-2 test listeners (same seed)
KNN = (15, 20.0)                                # phase 2's tuned ItemKNN
ALS_PARAMS = {"factors": 512, "alpha": 40, "reg": 0.05, "iterations": 15}
LOG_TAU, LOG_EPS = 0.1, 0.1                     # logging: softmax(ItemKNN / 0.1) mixed with 10% uniform
LOG_SIZES = [1_000, 3_000, 10_000, 30_000]
EXPLORE_EPS = 0.5                               # follow-up, added after the first result: a log that explores more
REPS = 100
ROUNDS, SEEDS = 50_000, 10
GRID = {"eps": [0.01, 0.05, 0.1], "alpha": [0.05, 0.2, 0.5, 1.0], "v": [0.05, 0.2, 0.5]}


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------- prep

def stage_prep(args) -> None:
    ds = D.load(D.CACHE)
    bench = B.Bench(ds, n_users=USERS, seed=args.seed)
    W = sp.load_npz(D.CACHE / f"knn_k{KNN[0]}_s{KNN[1]:g}.npz").tocsr()
    knn = R.ItemKNN(k=KNN[0], shrink=KNN[1], W=W)
    t0 = time.time()
    als = R.ALS(**ALS_PARAMS).fit(ds.X)
    log(f"ALS fit in {time.time() - t0:.0f}s")
    Uf, Vf = als.model.user_factors, als.model.item_factors
    pop = ds.popularity
    tail = ~D.head_mask(pop)
    artist = ds.artist
    n = len(bench.users)
    cand = np.zeros((n, K), np.int32)
    ks = np.zeros((n, K), np.float32)
    al = np.zeros((n, K), np.float32)
    known = np.zeros((n, K), bool)
    Rw = np.zeros((n, K), bool)
    for s in range(0, n, 200):
        users = bench.users[s:s + 200]
        S = knn.score(users, ds.X)
        heard = ds.X[users]
        S[np.repeat(np.arange(len(users)), np.diff(heard.indptr)), heard.indices] = -np.inf
        top = np.argpartition(-S, K, axis=1)[:, :K]
        o = np.argsort(-np.take_along_axis(S, top, axis=1), axis=1, kind="stable")
        top = np.take_along_axis(top, o, axis=1)
        for j, u in enumerate(users):
            i = s + j
            c = top[j]
            cand[i] = c
            ks[i] = S[j, c]
            al[i] = Uf[u] @ Vf[c].T
            known[i] = np.isin(artist[c], artist[heard[j].indices])
            Rw[i] = np.isin(c, bench.targets[i])
    knn_s = ks / np.maximum(ks.max(axis=1, keepdims=True), 1e-9)
    als_s = al / np.maximum(np.abs(al).max(axis=1, keepdims=True), 1e-9)
    logpop = np.log1p(pop[cand]).astype(np.float32)
    np.savez_compressed(CACHE, users=bench.users, cand=cand, knn=knn_s, als=als_s, known=known, R=Rw,
                        tail=tail[cand], logpop=logpop)
    log(f"prep: {n} listeners × {K} candidates; mean hits per listener {Rw.sum(axis=1).mean():.2f}; "
        f"listeners with ≥1 hit {(Rw.sum(axis=1) > 0).mean():.1%}; in {time.time() - t0:.0f}s")


def load_contexts() -> BD.Contexts:
    z = np.load(CACHE)
    lp = z["logpop"]
    X = BD.features(z["knn"], z["als"], lp, z["tail"], z["known"], float(lp.mean()), float(lp.std()))
    return BD.Contexts(users=z["users"], cand=z["cand"], knn=z["knn"], als=z["als"], X=X,
                       R=z["R"].astype(np.float64), tail=z["tail"], logpop=lp)


def target_policies(ctx: BD.Contexts) -> dict[str, np.ndarray]:
    pop = np.expm1(ctx.logpop)
    return {
        "Logging policy (softmax ItemKNN)": BD.softmax(ctx.knn, LOG_TAU, LOG_EPS),
        "ItemKNN, top pick": BD.greedy(ctx.knn),
        "ALS, top pick": BD.greedy(ctx.als),
        "PopPenalty β=0.5, top pick": BD.greedy(ctx.knn * np.maximum(pop, 1) ** -0.5),
        "Most popular candidate": BD.greedy(ctx.logpop),
        "Long tail first": BD.greedy(ctx.knn + 10.0 * ctx.tail),
        "Uniform over candidates": BD.uniform(ctx.n, ctx.K),
    }


# ---------------------------------------------------------------- off-policy evaluation

def stage_ope(args, eps: float = LOG_EPS, sizes: list[int] = LOG_SIZES) -> dict:
    ctx = load_contexts()
    P0 = BD.softmax(ctx.knn, LOG_TAU, eps)
    targets = target_policies(ctx)
    truth = {k: BD.true_value(P, ctx.R) for k, P in targets.items()}
    tail_truth = {k: BD.true_value(P, ctx.R * ctx.tail) for k, P in targets.items()}
    for k, v in truth.items():
        log(f"  true value {k:34s} {v:.4f}   long-tail part {tail_truth[k]:.4f}")
    rng = np.random.default_rng(args.seed)
    est = {n: {k: {e: [] for e in ("DM", "IPS", "SNIPS", "DR", "ess")} for k in targets} for n in sizes}
    t0 = time.time()
    for n in sizes:
        for rep in range(REPS):
            rows = rng.integers(ctx.n, size=n)            # a fresh log: listeners drawn with replacement
            a, p0 = BD.log_actions(P0, rows, rng)
            r = ctx.R[rows, a]
            out = BD.estimate(ctx, rows, a, p0, r, targets, rng)
            for k, d in out.items():
                for e, v in d.items():
                    est[n][k][e].append(v)
        log(f"  log size {n:6d}: {REPS} logs in {time.time() - t0:.0f}s")
    table = []
    for n in sizes:
        for k in targets:
            row = {"log_size": n, "policy": k, "truth": truth[k],
                   "ess": float(np.mean(est[n][k]["ess"]))}
            for e in ("DM", "IPS", "SNIPS", "DR"):
                v = np.array(est[n][k][e])
                row[e] = {"mean": float(v.mean()), "bias": float(v.mean() - truth[k]), "sd": float(v.std()),
                          "rmse": float(np.sqrt(((v - truth[k]) ** 2).mean())),
                          "q05": float(np.quantile(v, 0.05)), "q95": float(np.quantile(v, 0.95))}
            table.append(row)
    return {"logging_eps": eps, "logging_value": BD.true_value(P0, ctx.R), "truth": truth, "tail_truth": tail_truth, "table": table,
            "hits_per_listener": float(ctx.R.sum(axis=1).mean()),
            "listeners_with_hit": float((ctx.R.sum(axis=1) > 0).mean())}


# ---------------------------------------------------------------- online learning

def learners(d: int, eps: float, alpha: float, v: float) -> list[BD.Learner]:
    return [BD.Random(), BD.Fixed(1, "ItemKNN top pick (no learning)"), BD.EpsGreedy(d, 0.0),
            BD.EpsGreedy(d, eps), BD.LinUCB(d, alpha), BD.LinTS(d, v)]


def play(ctx: BD.Contexts, make, seeds: list[int], rounds: int) -> np.ndarray:
    out = []
    for s in seeds:
        rng = np.random.default_rng(1000 + s)
        order = rng.integers(ctx.n, size=rounds)
        out.append(BD.run_online(make(), ctx, order, rng))
    return np.stack(out)          # (seeds, rounds)


def stage_online(args) -> dict:
    ctx = load_contexts()
    d = ctx.X.shape[-1]
    perm = np.random.default_rng(args.seed).permutation(ctx.n)
    tune, report = ctx.subset(np.sort(perm[: ctx.n // 2])), ctx.subset(np.sort(perm[ctx.n // 2:]))
    t0 = time.time()
    best = {}
    for key, cls in (("eps", lambda x: BD.EpsGreedy(d, x)), ("alpha", lambda x: BD.LinUCB(d, x)),
                     ("v", lambda x: BD.LinTS(d, x))):
        scores = {x: float(play(tune, lambda x=x: cls(x), [0, 1, 2], args.rounds).mean()) for x in GRID[key]}
        best[key] = max(scores, key=scores.get)
        log(f"  tune {key}: " + ", ".join(f"{x:g} → {v:.4f}" for x, v in scores.items()) + f"  → {best[key]:g}")
    log(f"  tuning in {time.time() - t0:.0f}s")

    # references on the report half: the learners' own model (ridge regression of reward on the
    # features) fitted on all of the report half's feedback at once, and an oracle
    Xf = report.X.reshape(-1, d).astype(np.float64)
    theta_h = np.linalg.solve(Xf.T @ Xf + np.eye(d), Xf.T @ report.R.ravel())
    v_hind = BD.true_value(BD.greedy(report.X @ theta_h), report.R)
    v_oracle = float(report.R.max(axis=1).mean())

    curves, finals, names = {}, {}, []
    seeds = list(range(SEEDS))
    for proto in learners(d, best["eps"], best["alpha"], best["v"]):
        name = proto.name if not (isinstance(proto, BD.EpsGreedy) and proto.eps == 0) else "Greedy (learns, never explores)"
        def make(p=proto):
            if isinstance(p, BD.LinUCB):
                return BD.LinUCB(d, p.alpha)
            if isinstance(p, BD.LinTS):
                return BD.LinTS(d, p.v)
            if isinstance(p, BD.EpsGreedy):
                return BD.EpsGreedy(d, p.eps)
            return p
        rw = play(report, make, seeds, args.rounds)
        cum = rw.cumsum(axis=1) / np.arange(1, args.rounds + 1)
        marks = np.unique(np.geomspace(100, args.rounds, 40).astype(int)) - 1
        curves[name] = {"t": (marks + 1).tolist(), "mean": cum[:, marks].mean(axis=0).tolist(),
                        "lo": np.quantile(cum[:, marks], 0.1, axis=0).tolist(),
                        "hi": np.quantile(cum[:, marks], 0.9, axis=0).tolist()}
        per_seed = rw.mean(axis=1)
        last = rw[:, -args.rounds // 5:].mean(axis=1)       # the last fifth: what it has learned
        finals[name] = {"mean": float(per_seed.mean()), "sd": float(per_seed.std()),
                        "last_fifth": float(last.mean()), "last_fifth_sd": float(last.std()),
                        "gap_to_hindsight": float(args.rounds * v_hind - rw.sum(axis=1).mean())}
        names.append(name)
        log(f"  {name:34s} reward/round {per_seed.mean():.4f} ± {per_seed.std():.4f}; last fifth {last.mean():.4f}")
    final_theta = BD.LinUCB(d, best["alpha"])
    rng = np.random.default_rng(7)
    BD.run_online(final_theta, report, rng.integers(report.n, size=args.rounds), rng)
    return {"best": best, "grid": GRID, "rounds": args.rounds, "seeds": SEEDS,
            "tune_listeners": tune.n, "report_listeners": report.n,
            "hindsight_value": v_hind, "oracle_value": v_oracle,
            "hindsight_theta": dict(zip(BD.FEATURES, theta_h.round(4).tolist())),
            "linucb_theta": dict(zip(BD.FEATURES, final_theta.theta.round(4).tolist())),
            "curves": curves, "finals": finals, "order": names}


# ---------------------------------------------------------------- write

def pc(x: float, d: int = 2) -> str:
    return f"{100 * x:.{d}f}"


def write(ope: dict, onl: dict, explore: dict) -> None:
    meta = {"K": K, "users": USERS, "logging": {"tau": LOG_TAU, "eps": LOG_EPS, "base": "ItemKNN"},
            "log_sizes": LOG_SIZES, "reps": REPS, "knn": list(KNN), "als": ALS_PARAMS}
    (RESULTS / "phase6.json").write_text(json.dumps({"meta": meta, "ope": ope, "explore": explore, "online": onl}, indent=2))
    L = ["# Off-policy evaluation and bandits: phase 6", "",
         f"Supervised-to-bandit conversion (Dudík et al. 2011) on the phase-2 test listeners. Each of {USERS:,} "
         f"listeners has {K} candidates (ItemKNN's top {K}, nothing already heard); reward 1 if the candidate is one of "
         f"their discoveries in the test window. Mean hits per listener: {ope['hits_per_listener']:.2f}; "
         f"{ope['listeners_with_hit']:.0%} of listeners have at least one. Because every candidate's reward is known, "
         "each policy's true value is exact.", "",
         f"Logging policy: softmax over ItemKNN scores (τ = {LOG_TAU}) mixed with {LOG_EPS:.0%} uniform. "
         f"Each log draws listeners with replacement and one action each; {REPS} logs per size. "
         "Values are hit rates in %; RMSE in percentage points.", "",
         "## Off-policy estimates at 10,000 logged rounds", "",
         "| Target policy | True value | DM | IPS | SNIPS | DR | Effective sample size |",
         "|---|---|---|---|---|---|---|"]
    for row in (r for r in ope["table"] if r["log_size"] == 10_000):
        cell = lambda e: f"{pc(row[e]['mean'])} (RMSE {pc(row[e]['rmse'])})"   # noqa: E731
        L.append(f"| {row['policy']} | {pc(row['truth'])} | {cell('DM')} | {cell('IPS')} | {cell('SNIPS')} | "
                 f"{cell('DR')} | {row['ess']:,.0f} |")
    L += ["", "## RMSE by log size (percentage points)", "",
          "| Target policy | Estimator | " + " | ".join(f"{n:,}" for n in LOG_SIZES) + " |",
          "|---|---|" + "---|" * len(LOG_SIZES)]
    for pol in ope["truth"]:
        for e in ("DM", "IPS", "SNIPS", "DR"):
            vals = [next(r for r in ope["table"] if r["log_size"] == n and r["policy"] == pol)[e]["rmse"] for n in LOG_SIZES]
            L.append(f"| {pol} | {e} | " + " | ".join(pc(v) for v in vals) + " |")
    L += ["", f"## Follow-up: a logging policy that explores more (added after the first result)", "",
          f"Same softmax, but mixed with {explore['logging_eps']:.0%} uniform instead of {ope['logging_eps']:.0%}. "
          f"Logging hit rate falls from {pc(ope['logging_value'])}% to {pc(explore['logging_value'])}%. "
          "10,000 logged rounds, 100 logs.", "",
          "| Target policy | True value | DR RMSE, 10% explore | DR RMSE, 50% explore | ESS, 10% | ESS, 50% |", "|---|---|---|---|---|---|"]
    for pol in ope["truth"]:
        a = next(r for r in ope["table"] if r["log_size"] == 10_000 and r["policy"] == pol)
        b = next(r for r in explore["table"] if r["log_size"] == 10_000 and r["policy"] == pol)
        L.append(f"| {pol} | {pc(a['truth'])} | {pc(a['DR']['rmse'])} | {pc(b['DR']['rmse'])} | {a['ess']:,.0f} | {b['ess']:,.0f} |")
    L += ["", "## Online learning", "",
          f"Hyperparameters tuned on {onl['tune_listeners']:,} listeners, reported on the other {onl['report_listeners']:,}: "
          f"ε = {onl['best']['eps']:g}, α = {onl['best']['alpha']:g}, v = {onl['best']['v']:g}. "
          f"{onl['rounds']:,} rounds × {onl['seeds']} seeds; each round a random listener from the report half. "
          f"References: the learners' own linear model fitted on all the feedback at once, played greedily, {pc(onl['hindsight_value'])}%; "
          f"an oracle that always finds a hit when one exists {pc(onl['oracle_value'])}%.", "",
          "| Learner | Hit rate, all rounds | Hit rate, last fifth | Hits short of the hindsight fit |", "|---|---|---|---|"]
    for n in onl["order"]:
        f = onl["finals"][n]
        L.append(f"| {n} | {pc(f['mean'])} ± {pc(f['sd'])} | {pc(f['last_fifth'])} ± {pc(f['last_fifth_sd'])} | "
                 f"{f['gap_to_hindsight']:,.0f} |")
    L += ["", "Features: " + ", ".join(BD.FEATURES[1:]) + ".",
          "Hindsight ridge coefficients: " + ", ".join(f"{k} {v:+.3f}" for k, v in onl["hindsight_theta"].items()) + ".",
          "LinUCB's coefficients after one run: " + ", ".join(f"{k} {v:+.3f}" for k, v in onl["linucb_theta"].items()) + "."]
    (RESULTS / "phase6.md").write_text("\n".join(L) + "\n")
    log("wrote results/phase6.md and .json")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["prep", "ope", "online", "all"], default="all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=ROUNDS)
    args = ap.parse_args()
    if args.stage in ("prep", "all") and (args.stage == "prep" or not CACHE.exists()):
        stage_prep(args)
    if args.stage == "ope":
        json.dump(stage_ope(args), open(RESULTS / "phase6_ope.json", "w"), indent=2)
    if args.stage == "online":
        json.dump(stage_online(args), open(RESULTS / "phase6_online.json", "w"), indent=2)
    if args.stage == "all":
        write(stage_ope(args), stage_online(args), stage_ope(args, eps=EXPLORE_EPS, sizes=[10_000]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

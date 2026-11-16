"""Phase 2: tune on a validation split, then test tuned models + re-rankers.

    python scripts/run_phase2.py --stage tune   # validation split inside the training window
    python scripts/run_phase2.py --stage test   # refit tuned configs on full train, score test

Tuning never touches the test window. Validation = train files before
2026-11-08, targets from 2026-11-08 to 2026-11-11, exactly the Phase 1
protocol shifted back four days. Configs are selected on NDCG@20.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from obscure.lab import bench as B           # noqa: E402
from obscure.lab import data as D            # noqa: E402
from obscure.lab import metrics as M         # noqa: E402
from obscure.lab import models as R          # noqa: E402
from obscure.lab import rerank as RR         # noqa: E402

RESULTS = ROOT / "results"
VAL_CACHE = ROOT / "data" / "eval_val"
TUNED = RESULTS / "phase2_tuning.json"
K = 20

GRID = {
    "ItemKNN": {"shrink": [0, 10, 100], "k": [25, 50, 100, 200, 400], "max_user_items": 2000},
    "EASE": {"lambda": [100, 300, 1000, 3000]},
    "ALS": [{"factors": f, "alpha": a, "reg": 0.05, "iterations": 15}
            for f in (128, 256) for a in (2, 10, 40)],
    "BPR": [{"factors": 128, "lr": 0.01, "reg": 0.001, "iterations": 100},
            {"factors": 128, "lr": 0.05, "reg": 0.001, "iterations": 100},
            {"factors": 256, "lr": 0.05, "reg": 0.001, "iterations": 100}],
}

# Second round, run with --refine: the first round's best ItemKNN (k=25) and
# ALS (256 factors, alpha=40) sat on the edge of their grids.
REFINE = {
    "ItemKNN": {"shrink": [2, 5, 10, 20], "k": [5, 10, 15, 25], "max_user_items": 2000},
    "ALS": [{"factors": f, "alpha": a, "reg": 0.05, "iterations": 15}
            for f, a in ((256, 80), (256, 160), (512, 40), (512, 80), (512, 160))],
    "BPR": [{"factors": 128, "lr": 0.005, "reg": 0.001, "iterations": 100}],
}


def log(msg: str) -> None:
    print(msg, flush=True)


def candidates_for(ds: D.Dataset, family: str):
    """Yield (params, fitted model, fit seconds) over the grid for one family."""
    X = ds.X
    if family == "ItemKNN":
        kmax = max(GRID["ItemKNN"]["k"])
        cap = GRID["ItemKNN"]["max_user_items"]
        Xsim = R.cap_history(ds.plays, cap)
        for shrink in GRID["ItemKNN"]["shrink"]:
            t0 = time.time()
            W = R.itemknn_neighbours(Xsim, [(kmax, float(shrink))])[(kmax, float(shrink))]
            fit_s = time.time() - t0
            for k in GRID["ItemKNN"]["k"]:
                Wk = W if k == kmax else R.truncate_neighbours(W, k)
                p = {"k": k, "shrink": shrink, "max_user_items": cap}
                yield p, R.ItemKNN(k=k, shrink=shrink, W=Wk), fit_s
            del W
            gc.collect()
    elif family == "EASE":
        gram = R.EASE.gram(X, 15000)
        for lam in GRID["EASE"]["lambda"]:
            t0 = time.time()
            m = R.EASE(lam=lam, max_items=15000).fit(X, gram=gram)
            yield {"lambda": lam, "max_items": 15000}, m, time.time() - t0
            del m
        del gram
    elif family == "ALS":
        for p in GRID["ALS"]:
            t0 = time.time()
            m = R.ALS(**p).fit(X)
            yield p, m, time.time() - t0
    elif family == "BPR":
        for p in GRID["BPR"]:
            t0 = time.time()
            m = R.BPR(**p).fit(X)
            yield p, m, time.time() - t0


def build_model(family: str, params: dict, ds: D.Dataset):
    X = ds.X
    if family == "ItemKNN":
        k, shrink = params["k"], float(params["shrink"])
        W = R.itemknn_neighbours(R.cap_history(ds.plays, params.get("max_user_items")),
                                 [(k, shrink)])[(k, shrink)]
        return R.ItemKNN(k=k, shrink=shrink, W=W)
    if family == "EASE":
        return R.EASE(lam=params["lambda"], max_items=params["max_items"]).fit(X)
    if family == "ALS":
        return R.ALS(**params).fit(X)
    if family == "BPR":
        return R.BPR(**params).fit(X)
    raise KeyError(family)


def stage_tune(args) -> None:
    if (VAL_CACHE / "meta.json").exists():
        val = D.load(VAL_CACHE)
    else:
        t0 = time.time()
        val = D.build(split="2026-11-08", end="2026-11-12")
        D.save(val, VAL_CACHE)
        log(f"built validation dataset in {time.time() - t0:.0f}s")
    log(f"validation: {val.meta}")
    bench = B.Bench(val, n_users=args.tune_users, seed=args.seed, k=K)

    trials, best = [], {}
    families = [f for f in args.families.split(",") if f]
    prior = json.loads(TUNED.read_text()) if TUNED.exists() else {}
    for fam in families:
        for params, model, fit_s in candidates_for(val, fam):
            recs = R.top_k(model, val.X, bench.users, K)
            res = bench.score(fam, recs, fit_s, params)
            row = {"family": fam, "params": params, "fit_seconds": round(fit_s, 1),
                   **{m: res.mean(m) for m in ("recall", "ndcg", "tail_recall", "novelty")},
                   "coverage": res.coverage}
            trials.append(row)
            log(f"  {fam:8s} {json.dumps(params):70s} ndcg {row['ndcg']:.4f} "
                f"recall {row['recall']:.4f} tail {row['tail_recall']:.4f}")
            if fam not in best or row["ndcg"] > best[fam]["ndcg"]:
                best[fam] = row
            del model
            gc.collect()
        log(f"best {fam} this round: {best[fam]['params']}")

    # pool every round's trials; best = max NDCG over all trials of a family
    all_trials = prior.get("trials", []) + trials
    out = {}
    for t in all_trials:
        if t["family"] not in out or t["ndcg"] > out[t["family"]]["ndcg"]:
            out[t["family"]] = t
    for fam in families:
        log(f"best {fam} over all rounds: {out[fam]['params']}")
    TUNED.write_text(json.dumps({"validation": val.meta, "tune_users": len(bench.users),
                                 "selection_metric": "ndcg@20", "best": out,
                                 "rounds": prior.get("rounds", 1) + (1 if prior else 0),
                                 "trials": all_trials}, indent=2))
    log(f"wrote {TUNED.relative_to(ROOT)}")


def stage_capcheck(args) -> None:
    """Validation only: tuned ItemKNN with vs without the history cap."""
    tuning = json.loads(TUNED.read_text())
    p = tuning["best"]["ItemKNN"]["params"]
    val = D.load(VAL_CACHE)
    bench = B.Bench(val, n_users=args.tune_users, seed=args.seed, k=K)
    rows = []
    for cap in (p["max_user_items"], None):
        t0 = time.time()
        m = build_model("ItemKNN", {**p, "max_user_items": cap}, val)
        fit_s = time.time() - t0
        res = bench.score("ItemKNN", R.top_k(m, val.X, bench.users, K), fit_s)
        rows.append({"max_user_items": cap, "fit_seconds": round(fit_s, 1), "per_user": res.per_user,
                     **{mm: res.mean(mm) for mm in ("recall", "ndcg", "tail_recall", "novelty")}})
        log(f"cap {cap}: fit {fit_s:.0f}s ndcg {rows[-1]['ndcg']:.4f} recall {rows[-1]['recall']:.4f} "
            f"tail {rows[-1]['tail_recall']:.4f}")
        del m
        gc.collect()
    diff = {mm: dict(zip(("mean", "lo", "hi"), M.paired_diff_ci(
        rows[0]["per_user"][mm], rows[1]["per_user"][mm], seed=args.seed)))
        for mm in ("recall", "ndcg", "tail_recall")}
    for row in rows:
        row.pop("per_user")
    tuning["cap_check"] = {"params": p, "runs": rows, "capped_minus_uncapped": diff}
    TUNED.write_text(json.dumps(tuning, indent=2))
    log(f"capped - uncapped: {diff}")


def stage_test(args) -> None:
    tuned = json.loads(TUNED.read_text())["best"]
    ds = D.load()
    log(f"test dataset: {ds.meta}")
    bench = B.Bench(ds, n_users=args.test_users, seed=args.seed, k=K)
    pop = ds.popularity

    results: list[M.Result] = []
    pop_model = R.Popularity().fit(ds.X)
    results.append(bench.score("Popularity", R.top_k(pop_model, ds.X, bench.users, K)))

    knn = None
    for fam in ("ItemKNN", "EASE", "ALS", "BPR"):
        if fam not in tuned:
            continue
        params = tuned[fam]["params"]
        t0 = time.time()
        m = build_model(fam, params, ds)
        fit_s = time.time() - t0
        results.append(bench.score(f"{fam} (tuned)", R.top_k(m, ds.X, bench.users, K), fit_s, params))
        log(f"{fam} tuned {params}: recall {results[-1].mean('recall'):.4f} "
            f"tail {results[-1].mean('tail_recall'):.4f} fit {fit_s:.0f}s")
        if fam == "ItemKNN":
            knn = m
        else:
            del m
            gc.collect()

    frontier: list[M.Result] = []
    if knn is not None:
        items, rel = RR.candidates(knn, ds.X, bench.users, c=args.candidates)
        groups = RR.popularity_groups(pop)
        for beta in (0.25, 0.5, 0.75, 1.0):
            r = R.top_k(RR.PopPenalty(knn, pop, beta), ds.X, bench.users, K)
            frontier.append(bench.score(f"PopPenalty β={beta:g}", r, params={"beta": beta}))
        for lam in (0.2, 0.5, 0.8, 0.95):
            t0 = time.time()
            r = RR.calibrated(items, rel, ds.X, bench.users, groups, lam, K)
            frontier.append(bench.score(f"Calibrated λ={lam:g}", r, time.time() - t0, {"lambda": lam}))
        for lam in (0.1, 0.3, 0.5):
            t0 = time.time()
            r = RR.mmr(items, rel, knn.W, lam, K)
            frontier.append(bench.score(f"MMR λ={lam:g}", r, time.time() - t0, {"lambda": lam}))
        for f in frontier:
            log(f"  {f.name:22s} recall {f.mean('recall'):.4f} tail {f.mean('tail_recall'):.4f} "
                f"novelty {f.mean('novelty'):.2f}")

    write(ds, bench, tuned, results, frontier, args)


def write(ds, bench, tuned, results, frontier, args) -> None:
    rows = [B.summarise(r, args.seed) for r in results]
    frows = [B.summarise(r, args.seed) for r in frontier]
    ref = next((r for r in results if r.name == "ItemKNN (tuned)"), None)
    diffs = []
    if ref is not None:
        for r in results + frontier:
            if r is ref:
                continue
            d = {"name": r.name}
            for m in ("recall", "tail_recall", "novelty"):
                d[m] = dict(zip(("mean", "lo", "hi"),
                                M.paired_diff_ci(r.per_user[m], ref.per_user[m], seed=args.seed)))
            diffs.append(d)

    (RESULTS / "phase2.json").write_text(json.dumps(
        {"dataset": ds.meta, "k": K, "eval_users": len(bench.users), "tuned": tuned,
         "models": rows, "rerankers_on_itemknn": frows, "paired_vs_itemknn": diffs}, indent=2))

    meta = ds.meta
    n_t = sum(len(t) for t in bench.targets)
    n_tail = sum(int(bench.tail[t].sum()) for t in bench.targets)
    L = [
        "# Discovery benchmark: phase 2 (tuned models + re-rankers)",
        "",
        f"Same protocol as [phase 1](phase1.md) (see [docs/PROTOCOL.md](../docs/PROTOCOL.md)), with recordings merged "
        f"to MusicBrainz canonical recordings. {meta['n_users']:,} users × {meta['n_items']:,} recordings, "
        f"{meta['train_pairs']:,} training pairs; {len(bench.users):,} evaluated users, {n_t:,} discoveries "
        f"({n_tail / n_t:.0%} long-tail).",
        "",
        "Hyperparameters were chosen on a **validation split inside the training window** (train before "
        "2026-11-08, discoveries 2026-11-08 to 11-11) by NDCG@20, then refit on the full training window. "
        "A second, finer round was run for ItemKNN, ALS and BPR because round one's best settings sat on "
        "the edge of their grids. "
        "The test window was never used for selection. Grid and every trial: [phase2_tuning.json](phase2_tuning.json).",
        "",
        "## Tuned models",
        "",
        *B.table(rows, K),
        "",
        "Selected: " + "; ".join(f"{f} {v['params']}" for f, v in tuned.items()),
        "",
        "## Re-rankers on tuned ItemKNN",
        "",
        f"Each re-ranker chooses 20 of ItemKNN's top {args.candidates} candidates. One knob each, swept.",
        "",
        *B.table(frows, K),
        "",
        "## Paired differences vs tuned ItemKNN",
        "",
        "Same users, 95% bootstrap CI. Recall and tail recall in percentage points, novelty in bits.",
        "",
        "| | Δ Recall | Δ Tail recall | Δ Novelty |",
        "|---|---|---|---|",
    ]
    for d in diffs:
        n = d["novelty"]
        L.append(f"| {d['name']} | {B.pct(d['recall'])} | {B.pct(d['tail_recall'])} | "
                 f"{n['mean']:+.2f} [{n['lo']:+.2f}, {n['hi']:+.2f}] |")
    (RESULTS / "phase2.md").write_text("\n".join(L) + "\n")
    log("wrote results/phase2.md and .json")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["tune", "capcheck", "test", "all"], default="all")
    ap.add_argument("--families", default="ItemKNN,EASE,ALS,BPR")
    ap.add_argument("--refine", action="store_true", help="use the second-round grid")
    ap.add_argument("--tune-users", type=int, default=5000)
    ap.add_argument("--test-users", type=int, default=10000)
    ap.add_argument("--candidates", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    if args.refine:
        GRID.update(REFINE)
    if args.stage in ("tune", "all"):
        stage_tune(args)
    if args.stage == "capcheck":
        stage_capcheck(args)
    if args.stage in ("test", "all"):
        stage_test(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

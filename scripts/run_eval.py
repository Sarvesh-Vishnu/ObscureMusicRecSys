"""Run the offline discovery benchmark and write results/.

    python scripts/run_eval.py --rebuild          # build dataset from data/listens, then run
    python scripts/run_eval.py --models Popularity,ItemKNN
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from obscure.lab import data as D            # noqa: E402
from obscure.lab import metrics as M         # noqa: E402
from obscure.lab import models as R          # noqa: E402

RESULTS = ROOT / "results"


def make_models(tail: np.ndarray, names: list[str] | None):
    factories = {
        "RandomTail": lambda: R.RandomTail(tail),
        "Popularity": lambda: R.Popularity(),
        "ItemKNN": lambda: R.ItemKNN(k=100, shrink=10),
        "EASE": lambda: R.EASE(lam=500, max_items=15000),
        "ALS": lambda: R.ALS(factors=128, reg=0.05, alpha=10, iterations=15),
        "BPR": lambda: R.BPR(factors=128, lr=0.05, reg=0.001, iterations=100),
    }
    for n in names or list(factories):
        yield n, factories[n]


def fmt(m: float, lo: float, hi: float, pct: bool = True) -> str:
    if np.isnan(m):
        return "n/a"
    if pct:
        return f"{100 * m:.2f} [{100 * lo:.2f}, {100 * hi:.2f}]"
    return f"{m:.2f} [{lo:.2f}, {hi:.2f}]"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--models", default="")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--max-eval-users", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="phase1")
    args = ap.parse_args()

    if args.rebuild or not (D.CACHE / "meta.json").exists():
        t0 = time.time()
        ds = D.build()
        D.save(ds)
        print(f"built dataset in {time.time() - t0:.0f}s: {ds.meta}", flush=True)
    else:
        ds = D.load()
        print(f"loaded dataset: {ds.meta}", flush=True)

    pop = ds.popularity
    tail = ~D.head_mask(pop, head_share=0.5)
    rng = np.random.default_rng(args.seed)
    n_eval = min(args.max_eval_users, len(ds.eval_users))
    pick = np.sort(rng.choice(len(ds.eval_users), size=n_eval, replace=False))
    users = ds.eval_users[pick]
    targets = [ds.targets[i] for i in pick]
    tail_targets = sum(int(tail[t].sum()) for t in targets)
    n_targets = sum(len(t) for t in targets)
    print(f"eval users {n_eval:,}; discoveries {n_targets:,} ({tail_targets / n_targets:.0%} long-tail)",
          flush=True)

    names = [n for n in args.models.split(",") if n] or None
    results: list[M.Result] = []
    for name, factory in make_models(tail, names):
        model = factory()
        t0 = time.time()
        model.fit(ds.X)
        fit_s = time.time() - t0
        recs = R.top_k(model, ds.X, users, args.k)
        per_user, cov = M.evaluate_lists(recs, targets, popularity=pop, n_users=ds.n_users,
                                         tail=tail, artist=ds.artist, k=args.k)
        res = M.Result(name=model.name, k=args.k, per_user=per_user, coverage=cov,
                       fit_seconds=fit_s, params=model.params())
        results.append(res)
        print(f"{model.name:22s} recall {res.mean('recall'):.4f}  tail_recall "
              f"{res.mean('tail_recall'):.4f}  novelty {res.mean('novelty'):.2f}  "
              f"cov {cov:.3f}  fit {fit_s:.0f}s", flush=True)
        del model
        gc.collect()

    write(results, ds, tail, n_eval, n_targets, tail_targets, args)
    return 0


def write(results, ds, tail, n_eval, n_targets, tail_targets, args) -> None:
    RESULTS.mkdir(exist_ok=True)
    k = args.k
    base = next((r for r in results if r.name == "Popularity"), None)

    summary = {"dataset": ds.meta, "k": k, "eval_users_sampled": n_eval,
               "discoveries": n_targets, "tail_discoveries": tail_targets,
               "tail_items": int(tail.sum()), "head_items": int((~tail).sum()),
               "machine": f"{platform.machine()} / {platform.python_version()}",
               "models": []}
    for r in results:
        row = {"name": r.name, "params": r.params, "fit_seconds": round(r.fit_seconds, 1),
               "coverage": r.coverage}
        for m in M.PER_USER:
            row[m] = dict(zip(("mean", "lo", "hi"), M.bootstrap_ci(r.per_user[m], seed=args.seed)))
        if base is not None and r is not base:
            for m in ("recall", "tail_recall"):
                row[f"{m}_vs_popularity"] = dict(zip(("mean", "lo", "hi"), M.paired_diff_ci(
                    r.per_user[m], base.per_user[m], seed=args.seed)))
        summary["models"].append(row)
    (RESULTS / f"{args.tag}.json").write_text(json.dumps(summary, indent=2))

    meta = ds.meta
    lines = [
        f"# Discovery benchmark: {args.tag}",
        "",
        f"ListenBrainz listens submitted {meta['train_files'][0]} to {meta['train_files'][-1]} (train) "
        f"and {meta['test_files'][0]} to {meta['test_files'][-1]} (test). "
        f"{meta['n_users']:,} users × {meta['n_items']:,} recordings, {meta['train_pairs']:,} training pairs. "
        f"Targets are **discoveries**: test-window listens of recordings the user never played in training. "
        f"{n_eval:,} sampled users, {n_targets:,} discoveries, {tail_targets / n_targets:.0%} of them long-tail. "
        f"Long tail = recordings outside the most-listened set that holds half of all training pairs "
        f"({int(tail.sum()):,} of {len(tail):,} items).",
        "",
        f"Means with 95% bootstrap CIs over users. Recall/NDCG/tail figures in %. k = {k}.",
        "",
        f"| Model | Recall@{k} | NDCG@{k} | Tail recall@{k} | Novelty (bits) | Tail share | Artist diversity | Coverage | Fit |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r, row in zip(results, summary["models"]):
        c = lambda m, pct=True: fmt(row[m]["mean"], row[m]["lo"], row[m]["hi"], pct)  # noqa: E731
        lines.append(f"| {r.name} | {c('recall')} | {c('ndcg')} | {c('tail_recall')} | "
                     f"{c('novelty', False)} | {100 * row['tail_share']['mean']:.0f}% | "
                     f"{100 * row['artist_div']['mean']:.0f}% | {100 * r.coverage:.1f}% | "
                     f"{r.fit_seconds:.0f}s |")
    if base is not None:
        lines += ["", "Paired difference vs Popularity (percentage points, 95% CI):", "",
                  "| Model | Δ Recall | Δ Tail recall |", "|---|---|---|"]
        for row in summary["models"]:
            if "recall_vs_popularity" in row:
                a, b = row["recall_vs_popularity"], row["tail_recall_vs_popularity"]
                lines.append(f"| {row['name']} | {fmt(a['mean'], a['lo'], a['hi'])} | "
                             f"{fmt(b['mean'], b['lo'], b['hi'])} |")
    lines += ["", "Hyperparameters: " + "; ".join(
        f"{r.name} {r.params}" for r in results if r.params)]
    (RESULTS / f"{args.tag}.md").write_text("\n".join(lines) + "\n")
    print(f"wrote results/{args.tag}.md and .json", flush=True)


if __name__ == "__main__":
    sys.exit(main())

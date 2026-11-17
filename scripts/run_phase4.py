"""Phase 4: listening sessions, next-song prediction, and routes.

    python scripts/run_phase4.py --stage prep   # transitions + ItemKNN neighbours, cached
    python scripts/run_phase4.py --stage tune   # next-song popularity discount gamma, on validation
    python scripts/run_phase4.py --stage test   # next-song table + route benchmark on test

Needs the Phase 2 datasets (data/eval, data/eval_val) and the Phase 3 cold sets
(for the warm recordings' audio vectors).
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from obscure.lab import bench as B           # noqa: E402
from obscure.lab import data as D            # noqa: E402
from obscure.lab import metrics as M         # noqa: E402
from obscure.lab import models as R          # noqa: E402
from obscure.lab import rerank as RR         # noqa: E402
from obscure.lab import routes as RT         # noqa: E402
from obscure.lab import sequence as S        # noqa: E402

RESULTS = ROOT / "results"
VAL_CACHE = ROOT / "data" / "eval_val"
TUNED = RESULTS / "phase4_tuning.json"
K = 20
KNN_USER = (15, 20.0)      # Phase 2's tuned ItemKNN, for personal relevance
KNN_ITEM = (100, 20.0)     # wider neighbourhood for item -> item next-song back-off
GAMMAS = [0.0, 0.25, 0.5, 0.75, 1.0]
SEEDS, STEPS = 4, 5        # 4 routes x 5 new recordings = 20, the same budget as a top-20 list
ROUTES = {                 # name -> (beam, gamma, lam)
    "Greedy walk": (1, 0.0, 0.0),
    "Route: most probable": (10, 0.0, 0.0),
    "Route: smooth audio (λ=1)": (10, 0.0, 1.0),
    "Route: personal (γ=1)": (10, 1.0, 0.0),
    "Route: personal + smooth": (10, 1.0, 1.0),
    "Route: personal γ=3 + smooth": (10, 3.0, 1.0),
    "Route: personal γ=10 + smooth": (10, 10.0, 1.0),
}


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------- caches

def load_transitions(ds_dir: Path, ds: D.Dataset) -> S.Transitions:
    d = ds_dir / "transitions"
    if (d / "meta.json").exists():
        return S.Transitions(T=sp.load_npz(d / "T.npz").tocsr(), test=pd.read_parquet(d / "test.parquet"),
                             meta=json.loads((d / "meta.json").read_text()))
    t0 = time.time()
    tr = S.build_transitions(ds)
    d.mkdir(exist_ok=True)
    sp.save_npz(d / "T.npz", tr.T)
    tr.test.to_parquet(d / "test.parquet")
    (d / "meta.json").write_text(json.dumps(tr.meta, indent=2))
    log(f"built transitions for {ds_dir.name} in {time.time() - t0:.0f}s: {tr.meta}")
    return tr


def load_knn(ds_dir: Path, ds: D.Dataset, configs: list[tuple[int, float]]) -> dict:
    out, todo = {}, []
    for c in configs:
        p = ds_dir / f"knn_k{c[0]}_s{c[1]:g}.npz"
        if p.exists():
            out[c] = sp.load_npz(p).tocsr()
        else:
            todo.append(c)
    if todo:
        t0 = time.time()
        Ws = R.itemknn_neighbours(R.cap_history(ds.plays, 2000), todo)
        for c, W in Ws.items():
            sp.save_npz(ds_dir / f"knn_k{c[0]}_s{c[1]:g}.npz", W)
            out[c] = W
        log(f"ItemKNN neighbours {todo} for {ds_dir.name} in {time.time() - t0:.0f}s")
    return out


def load_audio(ds_dir: Path):
    cs = pickle.loads((ds_dir / "cold.pkl").read_bytes())
    return cs.audio_warm, cs.warm_has_audio


# ---------------------------------------------------------------- next song

def sample_transitions(tr: S.Transitions, n: int, seed: int) -> pd.DataFrame:
    te = tr.test
    te = te[te["user"] >= 0]
    rng = np.random.default_rng(seed)
    return te.iloc[np.sort(rng.choice(len(te), size=min(n, len(te)), replace=False))].reset_index(drop=True)


def next_song_models(ds, tr, W, gammas):
    pop = ds.popularity.astype(np.float32)
    yield "Popularity", _Pop(pop)
    yield "ItemKNN neighbours", S.NextSong(None, W, pop=pop, name="ItemKNN neighbours")
    for g in gammas:
        yield f"Transition γ={g:g}", S.NextSong(tr.T, W, gamma=g, pop=pop, name=f"Transition γ={g:g}")


class _Pop:
    name = "Popularity"

    def __init__(self, pop):
        self.p = (pop / pop.max()).astype(np.float32)

    def score(self, a):
        S_ = np.broadcast_to(self.p, (len(a), len(self.p))).copy()
        S_[np.arange(len(a)), a] = -np.inf
        return S_


def eval_next(model, sample: pd.DataFrame, tail: np.ndarray) -> dict[str, np.ndarray]:
    r = S.next_song_eval(model, sample["a"].to_numpy(), sample["b"].to_numpy(), k=K)
    new = sample["new_to_user"].to_numpy()
    is_tail = tail[sample["b"].to_numpy()]
    nan = lambda v, m: np.where(m, v, np.nan)  # noqa: E731
    return {"hit": r["hit"], "mrr": r["rr"], "hit_new": nan(r["hit"], new), "hit_tail": nan(r["hit"], is_tail)}


def stage_prep(args) -> None:
    for d, configs in ((VAL_CACHE, [KNN_ITEM]), (D.CACHE, [KNN_USER, KNN_ITEM])):
        ds = D.load(d)
        tr = load_transitions(d, ds)
        log(f"{d.name}: {tr.meta}")
        load_knn(d, ds, configs)


def stage_tune(args) -> None:
    ds = D.load(VAL_CACHE)
    tr = load_transitions(VAL_CACHE, ds)
    W = load_knn(VAL_CACHE, ds, [KNN_ITEM])[KNN_ITEM]
    sample = sample_transitions(tr, args.tune_transitions, args.seed)
    tail = ~D.head_mask(ds.popularity)
    trials = []
    for name, m in next_song_models(ds, tr, W, GAMMAS):
        if not name.startswith("Transition"):
            continue
        r = eval_next(m, sample, tail)
        t = {"gamma": m.gamma, **{k: float(np.nanmean(v)) for k, v in r.items()}}
        trials.append(t)
        log(f"  {name:18s} " + " ".join(f"{k} {v:.4f}" for k, v in t.items() if k != "gamma"))
    best = max(trials, key=lambda t: t["hit_new"])
    TUNED.write_text(json.dumps({"validation": tr.meta, "transitions_sampled": len(sample),
                                 "selection": "hit@20 on transitions to recordings new to the listener",
                                 "trials": trials, "best_gamma": best["gamma"]}, indent=2))
    log(f"best gamma {best['gamma']} → wrote {TUNED.relative_to(ROOT)}")


# ---------------------------------------------------------------- routes

def ragged_metrics(lists: list[list[int]], targets: list[np.ndarray], pop, n_users, tail, artist):
    n = len(lists)
    out = {m: np.full(n, np.nan) for m in ("recall", "tail_recall", "novelty", "tail_share", "artist_div",
                                           "length")}
    si = -np.log2(np.maximum(pop, 1) / n_users)
    for u, (r, t) in enumerate(zip(lists, targets)):
        r = np.asarray(r, dtype=np.int64)
        out["length"][u] = len(r)
        if len(r) == 0:
            out["recall"][u] = 0.0
            continue
        out["recall"][u] = np.isin(r, t).sum() / min(len(t), K)
        tt = t[tail[t]]
        if len(tt):
            out["tail_recall"][u] = np.isin(r, tt).sum() / min(len(tt), K)
        out["novelty"][u] = si[r].mean()
        out["tail_share"][u] = tail[r].mean()
        out["artist_div"][u] = len(np.unique(artist[r])) / len(r)
    return out


def stage_test(args) -> None:
    gamma = json.loads(TUNED.read_text())["best_gamma"]
    ds = D.load(D.CACHE)
    tr = load_transitions(D.CACHE, ds)
    Ws = load_knn(D.CACHE, ds, [KNN_USER, KNN_ITEM])
    audio, has_audio = load_audio(D.CACHE)
    pop = ds.popularity
    tail = ~D.head_mask(pop)
    log(f"transitions: {tr.meta}")

    # ---- next-song table
    sample = sample_transitions(tr, args.test_transitions, args.seed)
    nrows = []
    for name, m in next_song_models(ds, tr, Ws[KNN_ITEM], sorted({0.0, gamma})):
        r = eval_next(m, sample, tail)
        row = {"name": name}
        for k, v in r.items():
            row[k] = dict(zip(("mean", "lo", "hi"), M.bootstrap_ci(v, seed=args.seed)))
        nrows.append(row)
        log(f"  {name:20s} hit {row['hit']['mean']:.4f} mrr {row['mrr']['mean']:.4f} "
            f"new {row['hit_new']['mean']:.4f} tail {row['hit_tail']['mean']:.4f}")

    # ---- routes
    model = S.NextSong(tr.T, Ws[KNN_ITEM], gamma=gamma, pop=pop.astype(np.float32))
    gp = D.CACHE / f"route_graph_g{gamma:g}_m{args.fanout}.npz"
    if gp.exists():
        z = np.load(gp)
        g = RT.RouteGraph(nbr=z["nbr"], logp=z["logp"], audio=z["audio"], audio_median=float(z["audio_median"]))
    else:
        t0 = time.time()
        g = RT.build_graph(model, ds.n_items, audio, has_audio, M=args.fanout)
        np.savez(gp, nbr=g.nbr, logp=g.logp, audio=g.audio, audio_median=g.audio_median)
        log(f"route graph (fan-out {args.fanout}) in {time.time() - t0:.0f}s")
    log(f"median edge audio distance {g.audio_median:.2f}")
    test_keys = RT.pair_keys(tr.test["a"].to_numpy(), tr.test["b"].to_numpy(), ds.n_items)
    bench = B.Bench(ds, n_users=args.route_users, seed=args.seed, k=K)
    knn = R.ItemKNN(k=KNN_USER[0], shrink=KNN_USER[1], W=Ws[KNN_USER])
    artist = ds.artist
    rng = np.random.default_rng(args.seed)

    names = ["Top-20: ItemKNN", "Top-20: ItemKNN, one per artist", "Top-20: PopPenalty β=0.5", "One jump from each seed", "Random walk",
             *ROUTES]
    lists = {n: [] for n in names}
    steps = {n: [] for n in names}
    t0 = time.time()
    batch = 100
    for s in range(0, len(bench.users), batch):
        users = bench.users[s:s + batch]
        Su = knn.score(users, ds.X)
        heard = ds.X[users]
        Su[np.repeat(np.arange(len(users)), np.diff(heard.indptr)), heard.indices] = -np.inf
        top200 = np.argsort(-Su, axis=1)[:, :200]
        top = top200[:, :K]
        pen = Su * (np.maximum(pop, 1).astype(np.float32) ** -0.5)
        top_pen = np.argsort(-pen, axis=1)[:, :K]
        for row, u in enumerate(users):
            for n, lst in (("Top-20: ItemKNN", top[row]), ("Top-20: PopPenalty β=0.5", top_pen[row])):
                lists[n].append(lst.tolist())
                steps[n].append([lst.tolist()])          # played in order, as a playlist
            _, first = np.unique(artist[top200[row]], return_index=True)
            lists["Top-20: ItemKNN, one per artist"].append(top200[row][np.sort(first)][:K].tolist())
            steps["Top-20: ItemKNN, one per artist"].append([lists["Top-20: ItemKNN, one per artist"][-1]])
            seeds = RT.user_seeds(ds.plays, int(u), artist, SEEDS)
            sc = Su[row]
            rel = np.where(np.isfinite(sc), sc, 0.0) / max(float(sc[np.isfinite(sc)].max(initial=0)), 1e-9)
            base_block = np.zeros(ds.n_items, dtype=bool)
            base_block[heard[row].indices] = True
            # one jump: each seed's best legal next songs
            blk, items, paths = base_block.copy(), [], []
            for sd in seeds:
                c = [int(b) for b in g.nbr[sd] if b >= 0 and not blk[b] and artist[b] != artist[sd]][:STEPS]
                blk[c] = True
                items += c
                paths += [[sd, b] for b in c]
            lists["One jump from each seed"].append(items)
            steps["One jump from each seed"].append(paths)
            variants = [("Random walk", (1, 0.0, 0.0), rng)] + [(n, p, None) for n, p in ROUTES.items()]
            for n, (beam, gm, lam), r_ in variants:
                blk, items, paths = base_block.copy(), [], []
                for sd in seeds:
                    path = RT.find_route(g, sd, STEPS, blocked=blk, artist=artist, tail=tail, rel=rel,
                                         gamma=gm, lam=lam, beam=beam, rng=r_)
                    blk[path] = True
                    items += path
                    paths.append([sd] + path)
                lists[n].append(items)
                steps[n].append(paths)
        if s == 0:
            log(f"first {len(users)} users in {time.time() - t0:.0f}s")
    log(f"routes for {len(bench.users):,} users in {time.time() - t0:.0f}s")

    results = {}
    for n in names:
        per = ragged_metrics(lists[n], bench.targets, pop, ds.n_users, tail, artist)
        sm = [RT.step_metrics(p, test_keys, ds.n_items, audio, has_audio) for p in steps[n]]
        per["realism"] = np.array([x["realism"] for x in sm])
        per["audio_step"] = np.array([x["audio_step"] for x in sm])
        results[n] = per
        log(f"  {n:28s} " + " ".join(f"{k} {np.nanmean(v):.4f}" for k, v in per.items()))
    write(ds, tr, nrows, results, gamma, g, bench, args)


ROUTE_METRICS = ("recall", "tail_recall", "novelty", "tail_share", "artist_div", "realism", "audio_step", "length")


def write(ds, tr, nrows, results, gamma, g, bench, args) -> None:
    rows = []
    for n, per in results.items():
        row = {"name": n}
        for m in ROUTE_METRICS:
            row[m] = dict(zip(("mean", "lo", "hi"), M.bootstrap_ci(per[m], seed=args.seed)))
        rows.append(row)
    ref = "Top-20: PopPenalty β=0.5"
    diffs = []
    for n, per in results.items():
        if n == ref:
            continue
        d = {"name": n}
        for m in ("recall", "tail_recall", "realism", "audio_step"):
            d[m] = dict(zip(("mean", "lo", "hi"), M.paired_diff_ci(per[m], results[ref][m], seed=args.seed)))
        diffs.append(d)
    (RESULTS / "phase4.json").write_text(json.dumps(
        {"transitions": tr.meta, "k": K, "gamma": gamma, "fanout": args.fanout, "seeds": SEEDS, "steps": STEPS,
         "routes": {n: list(p) for n, p in ROUTES.items()}, "route_users": len(bench.users),
         "audio_median_edge": g.audio_median, "next_song": nrows, "route_table": rows, "paired_vs": ref,
         "paired": diffs}, indent=2))

    def pct(d):
        return B.pct(d)

    tm = tr.meta
    L = ["# Sessions and routes: phase 4", "",
         f"Consecutive listens by the same person less than {tm['gap_minutes']} minutes apart are transitions. "
         f"Training: {tm['train_transitions']:,} transitions over {tm['train_pairs']:,} distinct pairs, both ends in "
         f"the warm catalog. Test window: {tm['test_transitions']:,} transitions, "
         f"{tm['test_new_to_user']:,} of them to a recording new to the listener.", "",
         "## Next song", "",
         f"{args.test_transitions:,} held-out transitions by warm listeners. Given the current recording, rank every "
         "other recording; a hit is the true next one in the top 20. The popularity discount γ was picked on the "
         f"validation split ([phase4_tuning.json](phase4_tuning.json)): γ = {gamma:g}.", "",
         "| Model | Hit@20 | MRR@20 | Hit@20, new to listener | Hit@20, long-tail next song |",
         "|---|---|---|---|---|"]
    for r in nrows:
        L.append(f"| {r['name']} | {pct(r['hit'])} | {pct(r['mrr'])} | {pct(r['hit_new'])} | {pct(r['hit_tail'])} |")
    L += ["", "## Routes", "",
          f"{len(bench.users):,} test listeners. Each gets {SEEDS} routes of {STEPS} steps from their {SEEDS} most-played "
          f"recordings (one per artist): {SEEDS * STEPS} new recordings, the same budget as a top-{K} list. A route may "
          "not revisit a recording, reuse an artist or land on anything the listener has heard, and must end in the "
          f"long tail. Next-song candidates: the {args.fanout} best per recording under the tuned transition model.", "",
          "- **Realism**: share of steps that some listener actually made in the held-out window. For a top-20 list "
          "the steps are consecutive items, played as a playlist; for 'one jump' they are seed → item.",
          f"- **Audio step**: mean z-scored audio distance per step (lower is smoother; median over graph edges "
          f"{g.audio_median:.2f}).", "",
          "| Model | Recall@20 | Tail recall | Novelty | Tail share | Artist div. | Realism | Audio step | Length |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['name']} | {pct(r['recall'])} | {pct(r['tail_recall'])} | {r['novelty']['mean']:.2f} | "
                 f"{100 * r['tail_share']['mean']:.0f}% | {100 * r['artist_div']['mean']:.0f}% | "
                 f"{pct(r['realism'])} | {r['audio_step']['mean']:.2f} | {r['length']['mean']:.1f} |")
    L += ["", f"## Paired differences vs {ref}", "",
          "Same listeners, 95% bootstrap CI. Recall, tail recall and realism in percentage points; audio step in "
          "z-units (negative = smoother).", "",
          "| | Δ Recall | Δ Tail recall | Δ Realism | Δ Audio step |", "|---|---|---|---|---|"]
    for d in diffs:
        a = d["audio_step"]
        L.append(f"| {d['name']} | {pct(d['recall'])} | {pct(d['tail_recall'])} | {pct(d['realism'])} | "
                 f"{a['mean']:+.2f} [{a['lo']:+.2f}, {a['hi']:+.2f}] |")
    (RESULTS / "phase4.md").write_text("\n".join(L) + "\n")
    log("wrote results/phase4.md and .json")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["prep", "tune", "test", "all"], default="all")
    ap.add_argument("--tune-transitions", type=int, default=50000)
    ap.add_argument("--test-transitions", type=int, default=100000)
    ap.add_argument("--route-users", type=int, default=10000)
    ap.add_argument("--fanout", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    if args.stage in ("prep", "all"):
        stage_prep(args)
    if args.stage in ("tune", "all"):
        stage_tune(args)
    if args.stage in ("test", "all"):
        stage_test(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

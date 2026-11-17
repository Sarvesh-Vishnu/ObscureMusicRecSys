"""Phase 3: cold-start, recommending recordings with fewer than 10 listeners.

    python scripts/run_phase3.py --stage prep   # build cold sets (validation + test), cached
    python scripts/run_phase3.py --stage tune   # pick content-model settings on validation
    python scripts/run_phase3.py --stage test   # score every cold scorer once on test

Needs data/acousticbrainz.parquet (scripts/fetch_acousticbrainz.py) and the
Phase 2 datasets in data/eval and data/eval_val. Protocol in
obscure/lab/coldstart.py and docs/PROTOCOL.md.
"""

from __future__ import annotations

import argparse
import gc
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from obscure.lab import bench as B           # noqa: E402
from obscure.lab import coldstart as C       # noqa: E402
from obscure.lab import data as D            # noqa: E402
from obscure.lab import metrics as M         # noqa: E402
from obscure.lab import models as R          # noqa: E402

RESULTS = ROOT / "results"
VAL_CACHE = ROOT / "data" / "eval_val"
TUNED = RESULTS / "phase3_tuning.json"
ALS_PARAMS = {"factors": 512, "alpha": 40, "reg": 0.05, "iterations": 15}   # Phase 2's tuned ALS
K = 20
GRID = [{"hidden": h, "weight_decay": wd} for h in (256, 1024) for wd in (1e-4, 1e-2)]
VARIANTS = {"Audio → factors": (True, False), "Artist → factors": (False, True),
            "Audio + artist → factors": (True, True)}
METRICS = ("recall", "ndcg", "new_artist_recall", "known_artist_recall", "artist_div")


def log(msg: str) -> None:
    print(msg, flush=True)


def cold_path(ds_dir: Path) -> Path:
    return ds_dir / "cold.pkl"


def load_pair(ds_dir: Path) -> tuple[D.Dataset, C.ColdSet]:
    ds = D.load(ds_dir)
    p = cold_path(ds_dir)
    if p.exists():
        cs = pickle.loads(p.read_bytes())
    else:
        t0 = time.time()
        cs = C.build_cold(ds)
        p.write_bytes(pickle.dumps(cs, protocol=4))
        log(f"built cold set for {ds_dir.name} in {time.time() - t0:.0f}s")
    return ds, cs


class ColdBench:
    def __init__(self, ds: D.Dataset, cs: C.ColdSet, n_users: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        n = min(n_users, len(cs.users))
        pick = np.sort(rng.choice(len(cs.users), size=n, replace=False))
        self.ds, self.cs = ds, cs
        self.users = cs.users[pick]
        self.targets = [cs.targets[i] for i in pick]
        self.known = cs.items["known_artist"].to_numpy()
        _, cc, _ = C._artist_codes(ds, cs)
        self.artist = np.where(cc >= 0, cc, -(np.arange(len(cc)) + 1))
        self.pop = cs.items["train_listeners"].to_numpy() + 1

    def score(self, name: str, scorer: C.ColdScorer, fit_seconds: float = 0.0,
              params: dict | None = None) -> M.Result:
        recs = C.cold_top_k(scorer, self.cs, self.users, K)
        kw = dict(popularity=self.pop, n_users=self.ds.n_users, artist=self.artist, k=K)
        a, cov = M.evaluate_lists(recs, self.targets, tail=~self.known, **kw)
        b, _ = M.evaluate_lists(recs, self.targets, tail=self.known, **kw)
        per = {"recall": a["recall"], "ndcg": a["ndcg"], "new_artist_recall": a["tail_recall"],
               "known_artist_recall": b["tail_recall"], "artist_div": a["artist_div"]}
        return M.Result(name=name, k=K, per_user=per, coverage=cov,
                        fit_seconds=fit_seconds, params=params or {})


def fit_als(ds: D.Dataset):
    t0 = time.time()
    m = R.ALS(**ALS_PARAMS).fit(ds.X)
    log(f"ALS {ALS_PARAMS} fit in {time.time() - t0:.0f}s")
    return np.asarray(m.model.user_factors), np.asarray(m.model.item_factors)


def content_scorers(ds, cs, U, V, configs: dict[str, dict]):
    """Yield (name, scorer, fit_seconds, params) for the CF-mapped content scorers."""
    wc, cc, n_codes = C._artist_codes(ds, cs)
    cent_loo, has_loo = C.artist_centroids(V, wc, n_codes, leave_one_out=True)
    cent, has_code = C.artist_centroids(V, wc, n_codes)
    cold_has = (cc >= 0) & has_code[np.maximum(cc, 0)]
    cold_cent = np.where(cold_has[:, None], cent[np.maximum(cc, 0)], 0.0).astype(np.float32)

    # no learning: an unseen recording gets its artist's mean factor (or the
    # catalog mean for a new artist)
    mean = V.mean(axis=0)
    yield ("Artist centroid", C.FactorScorer("Artist centroid", U, np.where(cold_has[:, None], cold_cent, mean)),
           0.0, {})

    train = cs.warm_has_audio
    for name, (use_audio, use_artist) in VARIANTS.items():
        p = configs[name]
        t0 = time.time()
        m = C.ContentToFactor(use_audio=use_audio, use_artist=use_artist, **p).fit(
            cs.audio_warm[train], cent_loo[train], has_loo[train], V[train])
        pred = m.predict(cs.audio_cold, cold_cent, cold_has)
        fit_s = time.time() - t0
        log(f"  {name:26s} {p} held-out factor R² {m.val_r2:.3f} ({len(m.curve)} epochs) fit {fit_s:.0f}s")
        yield name, C.FactorScorer(name, U, pred), fit_s, {**m.params(), "factor_r2": round(m.val_r2, 4)}


def stage_prep(args) -> None:
    for d in (VAL_CACHE, D.CACHE):
        _, cs = load_pair(d)
        log(f"{d.name}: {json.dumps(cs.meta)}")


def stage_tune(args) -> None:
    ds, cs = load_pair(VAL_CACHE)
    bench = ColdBench(ds, cs, args.tune_users, args.seed)
    U, V = fit_als(ds)
    trials = []
    for p in GRID:
        cfg = {name: dict(p, epochs=args.epochs) for name in VARIANTS}
        for name, scorer, fit_s, params in content_scorers(ds, cs, U, V, cfg):
            if name == "Artist centroid":
                continue
            r = bench.score(name, scorer, fit_s, params)
            trials.append({"name": name, "params": params, "ndcg": r.mean("ndcg"), "recall": r.mean("recall"),
                           "new_artist_recall": r.mean("new_artist_recall")})
            log(f"    → ndcg {trials[-1]['ndcg']:.4f} recall {trials[-1]['recall']:.4f} "
                f"new-artist {trials[-1]['new_artist_recall']:.4f}")
    best = {}
    for name in VARIANTS:
        t = max((t for t in trials if t["name"] == name), key=lambda t: t["ndcg"])
        best[name] = {"hidden": t["params"]["hidden"], "weight_decay": t["params"]["weight_decay"],
                      "epochs": args.epochs}
        log(f"best {name}: {best[name]}")
    TUNED.write_text(json.dumps({"validation": cs.meta, "tune_users": len(bench.users), "als": ALS_PARAMS,
                                 "grid": GRID, "trials": trials, "best": best}, indent=2))
    log(f"wrote {TUNED.relative_to(ROOT)}")


def stage_test(args) -> None:
    best = json.loads(TUNED.read_text())["best"]
    ds, cs = load_pair(D.CACHE)
    log(f"cold test set: {json.dumps(cs.meta)}")
    bench = ColdBench(ds, cs, args.test_users, args.seed)
    results = [bench.score("Random", C.Random(cs, args.seed)),
               bench.score("Cold popularity", C.ColdPopularity(cs)),
               bench.score("Artist affinity", C.ArtistAffinity(ds, cs)),
               bench.score("Audio profile", C.AudioProfile(ds, cs))]
    U, V = fit_als(ds)
    for name, scorer, fit_s, params in content_scorers(ds, cs, U, V, best):
        results.append(bench.score(name, scorer, fit_s, params))
    del U, V
    gc.collect()
    for r in results:
        log(f"  {r.name:26s} " + " ".join(f"{m} {r.mean(m):.4f}" for m in METRICS))
    write(cs, bench, results, best, args)


def _row(r: M.Result, seed: int) -> dict:
    row = {"name": r.name, "params": r.params, "fit_seconds": round(r.fit_seconds, 1), "coverage": r.coverage}
    for m in METRICS:
        row[m] = dict(zip(("mean", "lo", "hi"), M.bootstrap_ci(r.per_user[m], seed=seed)))
    return row


def write(cs, bench, results, best, args) -> None:
    rows = [_row(r, args.seed) for r in results]
    by = {r.name: r for r in results}
    pairs = [("Audio + artist → factors", "Artist → factors", "what audio adds on top of the artist"),
             ("Audio → factors", "Cold popularity", "audio alone vs the cold-popularity baseline"),
             ("Audio → factors", "Random", "audio alone vs random"),
             ("Audio + artist → factors", "Artist affinity", "best content model vs 'more from artists you play'"),
             ("Artist → factors", "Artist centroid", "learned vs averaged artist embedding")]
    diffs = []
    for a, b, why in pairs:
        d = {"a": a, "b": b, "why": why}
        for m in ("recall", "new_artist_recall", "known_artist_recall"):
            d[m] = dict(zip(("mean", "lo", "hi"), M.paired_diff_ci(by[a].per_user[m], by[b].per_user[m],
                                                                    seed=args.seed)))
        diffs.append(d)
    (RESULTS / "phase3.json").write_text(json.dumps(
        {"cold_set": cs.meta, "k": K, "eval_users": len(bench.users), "als": ALS_PARAMS, "tuned": best,
         "models": rows, "paired": diffs}, indent=2))

    meta = cs.meta
    n_t = sum(len(t) for t in bench.targets)
    n_new = sum(int((~bench.known[t]).sum()) for t in bench.targets)
    hist = meta["train_listeners_hist"]
    L = [
        "# Cold-start benchmark: phase 3",
        "",
        "Can content recommend recordings collaborative filtering can't see? The cold catalog is every recording "
        "**outside** the warm catalog (fewer than 10 training listeners) that has AcousticBrainz audio "
        "descriptors and was played by someone in the test window: "
        f"{meta['n_cold_items']:,} recordings, {hist[0] / meta['n_cold_items']:.0%} of them with zero training "
        f"listeners. Each model ranks all of them for {len(bench.users):,} users; targets are the {n_t:,} cold "
        f"recordings those users went on to play ({n_new / n_t:.0%} by artists with no warm recording at all).",
        "",
        "Protocol and limitations: [docs/PROTOCOL.md](../docs/PROTOCOL.md). Content-model settings were picked on "
        "the validation split ([phase3_tuning.json](phase3_tuning.json)).",
        "",
        "| Model | Recall@20 | NDCG@20 | New-artist recall | Known-artist recall | Artist div. | Coverage |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        L.append(f"| {r['name']} | {B.pct(r['recall'])} | {B.pct(r['ndcg'])} | {B.pct(r['new_artist_recall'])} | "
                 f"{B.pct(r['known_artist_recall'])} | {100 * r['artist_div']['mean']:.0f}% | "
                 f"{100 * r['coverage']:.1f}% |")
    L += ["", "Held-out R² of the content → ALS-factor mapping: " + "; ".join(
        f"{r['name']} {r['params']['factor_r2']:.3f}" for r in rows if "factor_r2" in r["params"]),
          "", "## Paired differences", "",
          "Same users, 95% bootstrap CI, percentage points.", "",
          "| A − B | Question | Δ Recall | Δ New-artist recall | Δ Known-artist recall |",
          "|---|---|---|---|---|"]
    for d in diffs:
        L.append(f"| {d['a']} − {d['b']} | {d['why']} | {B.pct(d['recall'])} | {B.pct(d['new_artist_recall'])} | "
                 f"{B.pct(d['known_artist_recall'])} |")
    (RESULTS / "phase3.md").write_text("\n".join(L) + "\n")
    log("wrote results/phase3.md and .json")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["prep", "tune", "test", "all"], default="all")
    ap.add_argument("--tune-users", type=int, default=5000)
    ap.add_argument("--test-users", type=int, default=10000)
    ap.add_argument("--epochs", type=int, default=30)
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

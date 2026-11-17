"""Export the static data the web app reads (web/public/data/).

    python scripts/export_web.py            # routes.json, tracks.json, lab.json
    python scripts/fetch_previews.py        # then add 30-second previews (slow, cached)

Every route in the app is computed here by the same code the benchmark measured
(obscure.lab.routes.find_route on the phase-4 next-song graph). The browser does
no modelling: the dials step through a precomputed grid.

The app has no listener, so the seed song stands in for one. "Pull" (gamma) favours
songs that share listeners with the seed: ItemKNN similarity to the seed over two
hops, scaled to 0..1. Nothing is blocked except the seed itself; the route still may
not revisit a recording or reuse a lead artist, and prefers to end in the long tail
(exactly as in the benchmark: with strong pull it often can't). A seed is kept if
every setting gives a full route and the default setting ends in the long tail.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from obscure.lab import data as D            # noqa: E402
from obscure.lab import routes as RT         # noqa: E402

OUT = ROOT / "web" / "public" / "data"
RESULTS = ROOT / "results"
STEPS = 5
LAMS = [0.0, 1.0, 2.0]                 # smoothness
GAMMAS = [0.0, 1.0, 2.0]               # pull towards the seed's listeners
DEFAULT = (1.0, 0.0)                   # (lam, gamma) the app opens on: smooth, no pull
N_SEEDS = 120
# Indian artists present in the warm catalog (checked by hand against ds.items);
# ListenBrainz has few Indian listeners, so these routes are thinner.
INDIA = ["Hanumankind", "Local Train", "Arijit Singh", "Prateek Kuhad", "Diljit Dosanjh", "AP Dhillon",
         "A.R. Rahman", "Anirudh Ravichander", "Shreya Ghoshal", "Ritviz", "Badshah", "Pritam",
         "Kishore Kumar", "Indian Ocean", "The Yellow Diary", "Nucleya", "Santhosh Narayanan"]
DESCRIPTORS = ["bpm", "key_key", "key_scale", "danceability", "average_loudness", "onset_rate"]


def log(msg: str) -> None:
    print(msg, flush=True)


def load():
    ds = D.load(D.CACHE)
    T = sp.load_npz(D.CACHE / "transitions" / "T.npz").tocsr()
    test = pd.read_parquet(D.CACHE / "transitions" / "test.parquet")
    W15 = sp.load_npz(D.CACHE / "knn_k15_s20.npz").tocsr()
    W100 = sp.load_npz(D.CACHE / "knn_k100_s20.npz").tocsr()
    z = np.load(D.CACHE / "route_graph_g0_m50.npz")
    g = RT.RouteGraph(nbr=z["nbr"], logp=z["logp"], audio=z["audio"], audio_median=float(z["audio_median"]))
    cs = pickle.loads((D.CACHE / "cold.pkl").read_bytes())
    return ds, T, test, W15, W100, g, cs.warm_has_audio


# punctuation splits anywhere; words split only with whitespace on both sides, so
# "Daft Punk" and "Taylor Swift" are not cut at "ft" and "AC/DC" stays whole
LEAD_SPLIT = re.compile(r"\s*[,&+×•;]\s*|\s+(?:feat\.?|ft\.?|featuring|with|w/|x|vs\.?|versus|and|presents|pres\.|meets|"
                        r"reinterpreted by|remixed by|conducted by|starring|/)\s+", re.I)


def lead_artist(name: str) -> str:
    """'Korn feat. Skrillex' -> 'korn'. Mirrored by leadArtist() in web/src/data.ts."""
    lead = LEAD_SPLIT.split(name, maxsplit=1)[0].strip()
    return (lead or name).casefold()


def seed_rel(W: sp.csr_matrix, seed: int) -> np.ndarray:
    w1 = W[seed]
    r = (w1 + w1 @ W).toarray().ravel()
    r[seed] = 0.0
    return r / max(float(r.max()), 1e-9)


def pick_seeds(ds, g, tail, has_audio, artist) -> tuple[list[int], list[int]]:
    items = ds.items
    order = np.argsort(-ds.popularity, kind="stable")
    seen, top = set(), []
    for i in order:
        a = artist[i]
        if a in seen or not has_audio[i] or (g.nbr[i] >= 0).sum() < 10:
            continue
        seen.add(a)
        top.append(int(i))
        if len(top) == N_SEEDS * 2:                    # spare: some lose a full route below
            break
    india = []
    for name in INDIA:
        m = items[items["artist"].str.fullmatch(name, case=False, na=False)]
        m = m[[(g.nbr[i] >= 0).sum() >= 5 for i in m.index]]
        if len(m):
            india.append(int(m["listeners"].idxmax()))
    return top, india


def _ci(d: dict) -> list[float]:
    return [round(d["mean"], 5), round(d["lo"], 5), round(d["hi"], 5)]


def lab_data() -> dict:
    """Only what the Lab figures plot, read from results/phase*.json (never retyped)."""
    p1, p2, p3, p4 = (json.loads((RESULTS / f"phase{i}.json").read_text()) for i in (1, 2, 3, 4))
    ds = p2["dataset"]
    base = next(m for m in p2["models"] if m["name"].startswith("ItemKNN"))
    return {
        "dataset": {"users": ds["n_users"], "items": ds["n_items"], "trainPairs": ds["train_pairs"],
                    "evalUsers": p2["eval_users"], "split": ds["split"], "end": ds["end"],
                    "trainDays": len(ds["train_files"]), "testDays": len(ds["test_files"]),
                    "tailItems": p1["tail_items"], "headItems": p1["head_items"], "k": p2["k"]},
        "models": [{"name": m["name"].replace(" (tuned)", ""), "params": m["params"], "recall": _ci(m["recall"]),
                    "ndcg": _ci(m["ndcg"]), "tail": _ci(m["tail_recall"]), "novelty": _ci(m["novelty"]),
                    "coverage": round(m["coverage"], 4)} for m in p2["models"]],
        "rerankers": [{"name": "ItemKNN", "recall": _ci(base["recall"]), "tail": _ci(base["tail_recall"]),
                       "artistDiv": _ci(base["artist_div"])}] +
                     [{"name": m["name"], "recall": _ci(m["recall"]), "tail": _ci(m["tail_recall"]),
                       "artistDiv": _ci(m["artist_div"])} for m in p2["rerankers_on_itemknn"]],
        "paired": {x["name"].replace(" (tuned)", ""): {"recall": _ci(x["recall"]), "tail": _ci(x["tail_recall"])}
                   for x in p2["paired_vs_itemknn"]},
        "cold": {"set": {k: p3["cold_set"][k] for k in ("n_cold_items", "known_artist_items", "users",
                                                       "discoveries", "cold_test_pairs_without_audio")},
                 "models": [{"name": m["name"], "r2": m["params"].get("factor_r2"), "recall": _ci(m["recall"]), "newArtist": _ci(m["new_artist_recall"]),
                             "knownArtist": _ci(m["known_artist_recall"])} for m in p3["models"]]},
        "transitions": p4["transitions"],
        "next": [{"name": m["name"], **{k: _ci(m[k]) for k in ("hit", "mrr", "hit_new", "hit_tail")}}
                 for m in p4["next_song"]],
        "routes": [{"name": m["name"], **{k: (_ci(m[k]) if isinstance(m[k], dict) else m[k])
                                          for k in ("recall", "tail_recall", "novelty", "tail_share",
                                                    "artist_div", "realism", "audio_step", "length")}}
                   for m in p4["route_table"]],
        "routeUsers": p4["route_users"], "audioMedian": p4["audio_median_edge"],
        "routeSeeds": p4["seeds"], "routeSteps": p4["steps"],
        **({"bandit": bandit_data(json.loads((RESULTS / "phase6.json").read_text()))}
           if (RESULTS / "phase6.json").exists() else {}),
    }


def bandit_data(p6: dict) -> dict:
    ope, onl = p6["ope"], p6["online"]
    return {
        "meta": p6["meta"], "hitsPerListener": ope["hits_per_listener"], "listenersWithHit": ope["listeners_with_hit"],
        "truth": ope["truth"], "tailTruth": ope["tail_truth"],
        "ope": [{"n": r["log_size"], "policy": r["policy"], "truth": r["truth"], "ess": r["ess"],
                 **{e: {k: r[e][k] for k in ("mean", "bias", "rmse", "q05", "q95")} for e in ("DM", "IPS", "SNIPS", "DR")}}
                for r in ope["table"]],
        "explore": {"eps": p6["explore"]["logging_eps"], "loggingValue": p6["explore"]["logging_value"],
                    "baseLoggingValue": ope["logging_value"],
                    "rows": [{"policy": r["policy"], "ess": r["ess"], "rmse": r["DR"]["rmse"]}
                             for r in p6["explore"]["table"] if r["log_size"] == 10_000]},
        "online": {k: onl[k] for k in ("best", "grid", "rounds", "seeds", "tune_listeners", "report_listeners",
                                       "hindsight_value", "oracle_value", "hindsight_theta", "curves", "finals", "order")},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    args = ap.parse_args()

    ds, T, test, W15, W100, g, has_audio = load()
    pop = ds.popularity
    tail = ~D.head_mask(pop)
    # the benchmark keys artists by credit, so "Korn" and "Korn feat. Skrillex" differ;
    # a visitor sees the same artist, so the app's routes use the lead artist
    artist = pd.factorize(ds.items["artist"].fillna("").map(lead_artist))[0]
    out_total = np.asarray(T.sum(axis=1)).ravel()
    test_keys = RT.pair_keys(test["a"].to_numpy(), test["b"].to_numpy(), ds.n_items)
    pop_rank = (np.argsort(np.argsort(-pop, kind="stable")) + 1)      # 1 = most played

    def held_out(a: int, b: int) -> bool:
        k = a * ds.n_items + b
        p = np.searchsorted(test_keys, k)
        return bool(p < len(test_keys) and test_keys[p] == k)

    def edge(a: int, b: int) -> dict:
        j = int(np.flatnonzero(g.nbr[a] == b)[0])
        return {"to": b, "count": int(T[a, b]), "out": int(out_total[a]), "rank": j + 1,
                "p": round(float(np.exp(g.logp[a, j])), 4), "audio": round(float(g.audio[a, j]), 3),
                "heldOut": held_out(a, b)}

    top, india = pick_seeds(ds, g, tail, has_audio, artist)
    routes, tracks = [], set()
    blocked = np.zeros(ds.n_items, dtype=bool)
    for group, pool, want in (("top", top, args.seeds), ("india", india, len(india))):
        kept = 0
        for sd in pool:
            if kept == want:
                break
            rel = seed_rel(W100, sd)
            blocked[sd] = True
            grid, ok = {}, True
            for lam in LAMS:
                for gm in GAMMAS:
                    path = RT.find_route(g, sd, STEPS, blocked=blocked, artist=artist, tail=tail, rel=rel,
                                         gamma=gm, lam=lam, beam=10)
                    if len(path) < STEPS or ((lam, gm) == DEFAULT and not tail[path[-1]]):
                        ok = False
                        break
                    full = [sd] + path
                    steps = []
                    for a, b in zip(full[:-1], full[1:]):
                        e = edge(a, b)
                        legal = [int(c) for c in g.nbr[a] if c >= 0 and c != b and c not in full
                                 and artist[c] not in {artist[x] for x in full[:full.index(b)]}][:3]
                        e["alts"] = [{"id": c, "count": int(T[a, c])} for c in legal]
                        e["rel"] = round(float(rel[b]), 3)
                        steps.append(e)
                        tracks.update([a, b, *legal])
                    grid[f"{lam:g}|{gm:g}"] = steps
                if not ok:
                    break
            blocked[sd] = False
            if not ok:
                continue
            w = W15[sd]
            o = w.indices[np.argsort(-w.data)][:5]
            tracks.update(o.tolist() + [sd])
            routes.append({"seed": sd, "group": group, "obvious": [int(i) for i in o], "grid": grid})
            kept += 1
        log(f"{group}: {kept} seeds with a full route at every setting")

    # ---- tracks
    ids = sorted(tracks)
    it = ds.items.iloc[ids]
    ab = pd.read_parquet(ROOT / "data" / "acousticbrainz.parquet", columns=["mbid", *DESCRIPTORS])
    ab = ab.set_index("mbid")
    tr_out = {}
    for i, row in zip(ids, it.itertuples()):
        t = {"title": row.title, "artist": row.artist, "mbid": row.mbid, "listeners": int(row.listeners),
             "rank": int(pop_rank[i]), "tail": bool(tail[i])}
        if row.mbid in ab.index:
            a = ab.loc[row.mbid]
            t["audio"] = {"bpm": round(float(a["bpm"])), "key": f"{a['key_key']} {a['key_scale']}",
                          "dance": round(float(a["danceability"]), 2), "loud": round(float(a["average_loudness"]), 2),
                          "onsets": round(float(a["onset_rate"]), 2)}
        tr_out[str(i)] = t

    meta = {"steps": STEPS, "lams": LAMS, "gammas": GAMMAS, "default": list(DEFAULT), "fanout": g.M, "beam": 10,
            "audioMedian": round(g.audio_median, 3), "nItems": ds.n_items,
            "headItems": int((~tail).sum()), "headMinListeners": int(pop[~tail].min()), "maxListeners": int(pop.max()),
            "minListeners": int(pop.min()), "split": ds.meta["split"], "end": ds.meta["end"],
            "trainTransitions": int(T.sum())}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "routes.json").write_text(json.dumps({"meta": meta, "routes": routes}, separators=(",", ":")))
    old = OUT / "tracks.json"
    if old.exists():                       # keep previews already matched
        prev = json.loads(old.read_text())
        for k, t in tr_out.items():
            if k in prev and "preview" in prev[k]:
                t["preview"] = prev[k]["preview"]
    (OUT / "tracks.json").write_text(json.dumps(tr_out, separators=(",", ":"), ensure_ascii=False))
    (OUT / "lab.json").write_text(json.dumps(lab_data(), separators=(",", ":"), ensure_ascii=False))
    log(f"wrote {len(routes)} seeds, {len(tr_out)} tracks → {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Routes: a path of small steps from a song you love to somewhere you've never been.

A route starts at a seed the listener already plays and takes L steps. Every
step must be a move real listeners make (an edge of the next-song graph), may
not revisit a recording or reuse an artist, and may not land on anything the
listener has already heard. The last stop must be in the long tail.

Among all such paths, beam search maximises
    sum over steps of  log p(b | a)  +  gamma * rel_u(b)  -  lam * audio(a, b)
  p(b | a)    next-song probability, renormalised over the M best next songs
  rel_u(b)    the listener's ItemKNN score for b, scaled to 0..1 (personal pull)
  audio(a,b)  z-scored AcousticBrainz distance between consecutive songs, divided
              by its median over graph edges (so lam = 1 costs one "typical
              step" of audio change); a step with unknown audio costs the median

gamma = lam = 0 gives the most *probable* route; lam > 0 the smoothest;
gamma > 0 bends it towards what this listener would like.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from .sequence import NextSong


@dataclass
class RouteGraph:
    nbr: np.ndarray        # (n_items, M) next-song candidates, best first (-1 = none)
    logp: np.ndarray       # (n_items, M) log p(b | a) over those candidates
    audio: np.ndarray      # (n_items, M) normalised audio distance a -> b
    audio_median: float

    @property
    def M(self) -> int:
        return self.nbr.shape[1]


def build_graph(model: NextSong, n_items: int, audio: np.ndarray, has_audio: np.ndarray,
                M: int = 50, batch: int = 256) -> RouteGraph:
    nbr = np.full((n_items, M), -1, dtype=np.int32)
    logp = np.full((n_items, M), -np.inf, dtype=np.float32)
    for s in range(0, n_items, batch):
        a = np.arange(s, min(s + batch, n_items))
        S = model.score(a)
        part = np.argpartition(-S, M, axis=1)[:, :M]
        ps = np.take_along_axis(S, part, axis=1)
        o = np.lexsort((part, -ps), axis=1)
        part, ps = np.take_along_axis(part, o, axis=1), np.take_along_axis(ps, o, axis=1)
        ok = ps > 0
        w = np.where(ok, ps, 0.0)
        p = w / np.maximum(w.sum(axis=1, keepdims=True), 1e-12)
        nbr[a] = np.where(ok, part, -1)
        logp[a] = np.where(ok, np.log(np.maximum(p, 1e-12)), -np.inf)
    # audio distance along edges
    src = np.repeat(np.arange(n_items), M)
    dst = nbr.ravel()
    valid = (dst >= 0) & has_audio[src] & has_audio[np.maximum(dst, 0)]
    d = np.full(len(dst), np.nan, dtype=np.float32)
    d[valid] = np.linalg.norm(audio[src[valid]] - audio[dst[valid]], axis=1)
    med = float(np.nanmedian(d)) if valid.any() else 1.0
    d = np.where(np.isnan(d), med, d) / med
    return RouteGraph(nbr=nbr, logp=logp, audio=d.reshape(n_items, M).astype(np.float32), audio_median=med)


def find_route(g: RouteGraph, seed: int, length: int, *, blocked: np.ndarray, artist: np.ndarray,
               tail: np.ndarray, rel: np.ndarray | None = None, gamma: float = 0.0, lam: float = 0.0,
               beam: int = 10, rng: np.random.Generator | None = None) -> list[int]:
    """Best route of `length` steps after `seed` (seed excluded from the result).

    blocked: bool (n_items,) recordings the route may not visit (already heard,
    already used by an earlier route). With `rng`, steps are *sampled* in
    proportion to p(b | a) instead (a random walk; beam must be 1).
    Returns fewer than `length` items if the graph runs out of legal moves."""
    beams = [(0.0, [seed], {artist[seed]})]
    for _ in range(length):
        nxt = []
        for score, path, arts in beams:
            a = path[-1]
            cand, lp, ad = g.nbr[a], g.logp[a], g.audio[a]
            ok = cand >= 0
            c0 = np.maximum(cand, 0)
            ok &= ~blocked[c0]
            ok &= ~np.isin(cand, path)
            ok &= ~np.isin(artist[c0], list(arts))
            if not ok.any():
                continue
            step = lp + (gamma * rel[c0] if rel is not None and gamma else 0.0) - lam * ad
            step = np.where(ok, step, -np.inf)
            if rng is not None:
                p = np.exp(np.where(ok, lp, -np.inf))
                j = int(rng.choice(len(cand), p=p / p.sum()))
                nxt.append((score + step[j], path + [int(cand[j])], arts | {artist[cand[j]]}))
                continue
            for j in np.argsort(-step)[:beam]:
                if np.isfinite(step[j]):
                    nxt.append((score + float(step[j]), path + [int(cand[j])], arts | {artist[cand[j]]}))
        if not nxt:
            break
        nxt.sort(key=lambda t: -t[0])
        beams = nxt[:beam]
    full = [b for b in beams if len(b[1]) == length + 1]
    pool = full or beams
    ends_tail = [b for b in pool if tail[b[1][-1]]]
    best = (ends_tail or pool)[0]
    return best[1][1:]


def user_seeds(plays: sp.csr_matrix, u: int, artist: np.ndarray, n: int) -> list[int]:
    """The user's n most-played recordings, one per artist."""
    row = plays[u]
    order = row.indices[np.argsort(-row.data, kind="stable")]
    seeds, seen = [], set()
    for i in order:
        if artist[i] not in seen:
            seeds.append(int(i))
            seen.add(artist[i])
        if len(seeds) == n:
            break
    return seeds


def pair_keys(a: np.ndarray, b: np.ndarray, n_items: int) -> np.ndarray:
    return np.unique(a.astype(np.int64) * n_items + b.astype(np.int64))


def step_metrics(paths: list[list[int]], test_keys: np.ndarray, n_items: int, audio: np.ndarray,
                 has_audio: np.ndarray) -> dict[str, float]:
    """For one listener's routes or playlist (each path includes its starting point):
    realism    share of steps that some listener made in the held-out window
               (test_keys from pair_keys over held-out transitions)
    audio_step mean z-scored audio distance per step (steps with audio on both ends)."""
    steps = np.array([(a, b) for p in paths for a, b in zip(p[:-1], p[1:])], dtype=np.int64)
    if len(steps) == 0:
        return {"realism": np.nan, "audio_step": np.nan}
    keys = steps[:, 0] * n_items + steps[:, 1]
    pos = np.searchsorted(test_keys, keys)
    real = (pos < len(test_keys)) & (test_keys[np.minimum(pos, len(test_keys) - 1)] == keys)
    both = has_audio[steps[:, 0]] & has_audio[steps[:, 1]]
    d = np.linalg.norm(audio[steps[both, 0]] - audio[steps[both, 1]], axis=1)
    return {"realism": float(real.mean()), "audio_step": float(d.mean()) if len(d) else np.nan}

import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { featured, leadArtist, median, parseHash, reachY, routeFor, routeHash, searchSeeds, stops, summarise, type Lab, type RoutesFile, type Tracks } from "./data";
import { fold, pct, stepEvidence, testWindow, windowDays } from "./format";

const read = <T,>(f: string): T => JSON.parse(readFileSync(new URL(`../public/data/${f}`, import.meta.url), "utf8")) as T;
const data = read<RoutesFile>("routes.json");
const tracks = read<Tracks>("tracks.json");
const lab = read<Lab>("lab.json");
const { meta } = data;

describe("exported routes keep the benchmark's rules", () => {
  it("has a route for every seed at every dial setting", () => {
    expect(data.routes.length).toBeGreaterThan(100);
    for (const r of data.routes)
      for (const l of meta.lams)
        for (const g of meta.gammas) expect(routeFor(r, l, g)).toHaveLength(meta.steps);
  });

  it("never revisits a song or reuses an artist, seed included", () => {
    for (const r of data.routes)
      for (const steps of Object.values(r.grid)) {
        const ids = stops(r, steps);
        expect(new Set(ids).size).toBe(ids.length);
        const artists = ids.map((i) => leadArtist(tracks[i].artist));
        expect(new Set(artists).size).toBe(artists.length);
      }
  });

  it("ends in the long tail at the default setting", () => {
    const [l, g] = meta.default;
    for (const r of data.routes) {
      const s = routeFor(r, l, g);
      expect(tracks[s[s.length - 1].to].tail).toBe(true);
    }
  });

  it("carries consistent step evidence", () => {
    for (const r of data.routes)
      for (const steps of Object.values(r.grid))
        for (const s of steps) {
          expect(s.count).toBeLessThanOrEqual(s.out);
          expect(s.rank).toBeGreaterThanOrEqual(1);
          expect(s.rank).toBeLessThanOrEqual(meta.fanout);
          expect(s.p).toBeGreaterThan(0);
          expect(s.p).toBeLessThanOrEqual(1);
          expect(s.rel).toBeGreaterThanOrEqual(0);
          expect(s.alts.every((a) => tracks[a.id] && a.id !== s.to)).toBe(true);
        }
  });

  it("labels the tail exactly by the head threshold", () => {
    for (const t of Object.values(tracks)) if (t.listeners >= meta.headMinListeners + 1) expect(t.tail).toBe(false);
  });

  it("stronger pull keeps routes nearer the seed's listeners on average", () => {
    const [l] = meta.default;
    const avg = (g: number) => median(data.routes.map((r) => median(routeFor(r, l, g).map((s) => tracks[s.to].listeners))));
    expect(avg(meta.gammas[meta.gammas.length - 1])).toBeGreaterThan(avg(meta.gammas[0]));
  });
});

describe("helpers", () => {
  it("parses and builds hashes", () => {
    expect(parseHash("#/lab/cold")).toEqual({ name: "lab", section: "cold" });
    expect(parseHash("#/route/42?l=2&g=1")).toEqual({ name: "route", seed: 42, lam: 2, gamma: 1 });
    expect(parseHash("")).toEqual({ name: "route", seed: null, lam: null, gamma: null });
    expect(routeHash(42, meta.default[0], meta.default[1], meta)).toBe("#/route/42");
    expect(routeHash(42, 2, 1, meta)).toBe("#/route/42?l=2&g=1");
  });

  it("reduces a credit to its lead artist", () => {
    expect(leadArtist("Korn feat. Skrillex")).toBe("korn");
    expect(leadArtist("Clean Bandit & Jess Glynne")).toBe("clean bandit");
    expect(leadArtist("Earth, Wind & Fire")).toBe("earth");
    expect(leadArtist("Simon & Garfunkel")).toBe("simon");
    expect(leadArtist("Radiohead")).toBe("radiohead");
    for (const [name, lead] of [["Linkin Park reinterpreted by Jay Gordon", "linkin park"], ["Coldplay × BTS", "coldplay"],
      ["24kGoldn • iann dior", "24kgoldn"], ["Apparat w/ Soap&Skin", "apparat"], ["Daft Punk", "daft punk"],
      ["Taylor Swift", "taylor swift"], ["AC/DC", "ac/dc"], ["Andrew Bird", "andrew bird"]]) expect(leadArtist(name)).toBe(lead);
  });

  it("maps listeners to height on a log scale", () => {
    expect(reachY(meta.maxListeners, meta)).toBe(0);
    expect(reachY(meta.minListeners, meta)).toBe(1);
    expect(reachY(0, meta)).toBe(1);
    expect(reachY(100, meta)).toBeLessThan(reachY(30, meta));
  });

  it("searches seeds ignoring case and accents", () => {
    const first = tracks[data.routes[0].seed];
    const hit = searchSeeds(data.routes, tracks, first.title.toUpperCase());
    expect(hit.map((r) => r.seed)).toContain(data.routes[0].seed);
    expect(searchSeeds(data.routes, tracks, "")).toHaveLength(data.routes.length);
    expect(searchSeeds(data.routes, tracks, "zzqqxx-no-such-song")).toHaveLength(0);
  });

  it("features a well-known seed whose default route reaches the tail", () => {
    const f = featured(data.routes, tracks, meta);
    const s = summarise(f, routeFor(f, ...meta.default), tracks);
    expect(f.group).toBe("top");
    expect(s.endsInTail).toBe(true);
    expect(tracks[f.seed].listeners).toBeGreaterThanOrEqual(500);
  });

  it("summarises a route from its data", () => {
    const r = data.routes[0];
    const s = summarise(r, routeFor(r, ...meta.default), tracks);
    expect(s.fold).toBeCloseTo(s.seedListeners / s.endListeners);
    expect(s.heldOut).toBeLessThanOrEqual(meta.steps);
  });

  it("formats evidence sentences", () => {
    const from = { title: "A", artist: "x", mbid: "", listeners: 10, rank: 1, tail: true };
    const base = { to: 1, rank: 1, p: 0.5, audio: 1, heldOut: false, rel: 0, alts: [] };
    expect(stepEvidence({ ...base, count: 48, out: 6090 }, from)).toBe("48 of 6,090 plays that followed “A” went here next (0.8%).");
    expect(stepEvidence({ ...base, count: 1, out: 6090 }, from)).toContain("(under 0.1%)");
    expect(stepEvidence({ ...base, count: 30, out: 100 }, from)).toContain("(30%)");
    expect(stepEvidence({ ...base, count: 0, out: 100 }, from)).toContain("back-off");
    expect(pct(0.0687)).toBe("6.87%");
    expect(pct(0)).toBe("0%");
    expect(fold(59)).toBe("59× fewer listeners than the seed");
    expect(fold(1)).toBe("about as many listeners as the seed");
    expect(testWindow({ split: "2026-11-12", end: "2026-11-16" })).toBe("12–15 Nov");
    expect(windowDays({ split: "2026-11-12", end: "2026-11-16" })).toBe(4);
  });
});

describe("lab data matches the benchmark results", () => {
  it("has every result the Lab quotes", () => {
    for (const n of ["ItemKNN", "ALS", "EASE", "BPR", "Popularity"]) expect(lab.models.some((m) => m.name === n)).toBe(true);
    for (const n of ["ItemKNN", "PopPenalty β=0.25", "PopPenalty β=1", "Calibrated λ=0.2", "MMR λ=0.5"]) expect(lab.rerankers.some((m) => m.name === n)).toBe(true);
    for (const n of ["Random", "Audio → factors", "Artist centroid"]) expect(lab.cold.models.some((m) => m.name === n)).toBe(true);
    for (const n of ["Top-20: PopPenalty β=0.5", "Route: most probable", "Route: personal γ=10 + smooth"]) expect(lab.routes.some((m) => m.name === n)).toBe(true);
  });

  it("has the bandit results the Lab quotes", () => {
    const b = lab.bandit!;
    expect(b).toBeDefined();
    for (const p of ["ItemKNN, top pick", "Long tail first", "PopPenalty β=0.5, top pick", "ALS, top pick"])
      expect(b.ope.some((r) => r.n === 10_000 && r.policy === p)).toBe(true);
    expect(b.explore.rows.some((r) => r.policy === "Long tail first")).toBe(true);
    for (const k of ["ItemKNN top pick (no learning)", "Greedy (learns, never explores)"]) expect(b.online.finals[k]).toBeDefined();
    expect(b.online.order.some((k) => k.startsWith("LinUCB"))).toBe(true);
    for (const r of b.ope) for (const e of [r.DR, r.DM]) expect(e.q05).toBeLessThanOrEqual(e.q95);
  });

  it("intervals contain their means", () => {
    const cis = [...lab.models.flatMap((m) => [m.recall, m.tail]), ...lab.routes.flatMap((m) => [m.recall, m.realism])];
    for (const [m, lo, hi] of cis) { expect(lo).toBeLessThanOrEqual(m + 1e-12); expect(hi).toBeGreaterThanOrEqual(m - 1e-12); }
  });
});

describe("axis scale", () => {
  it("lands ticks on round values", async () => {
    const { niceScale } = await import("./charts");
    expect(niceScale(0.0291).ticks).toEqual([0, 0.01, 0.02, 0.03]);
    expect(niceScale(0.075).ticks).toEqual([0, 0.02, 0.04, 0.06, 0.08]);
    expect(niceScale(0.7).ticks).toEqual([0, 0.2, 0.4, 0.6, 0.8]);
  });
});

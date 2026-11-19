// Types and pure helpers for the exported data (scripts/export_web.py).
// Nothing here models anything: every route was computed offline by the
// benchmarked code; the app only reads and explains it.

export interface Preview {
  url: string;
  art: string;
  link: string;
  exact: boolean;
  as: string;
}

export interface Track {
  title: string;
  artist: string;
  mbid: string;
  listeners: number;
  rank: number;
  tail: boolean;
  audio?: { bpm: number; key: string; dance: number; loud: number; onsets: number };
  preview?: Preview;
}

export interface Step {
  to: number;
  count: number;   // training transitions previous -> to
  out: number;     // all training transitions out of previous
  rank: number;    // position among previous's 50 likeliest next songs
  p: number;       // next-song probability over those 50
  audio: number;   // audio distance / median edge distance
  heldOut: boolean;
  rel: number;     // shared-listener closeness to the seed, 0..1
  alts: { id: number; count: number }[];
}

export interface SeedRoutes {
  seed: number;
  group: "top" | "india";
  obvious: number[];
  grid: Record<string, Step[]>;
}

export interface RouteMeta {
  steps: number;
  lams: number[];
  gammas: number[];
  default: [number, number];
  fanout: number;
  beam: number;
  audioMedian: number;
  nItems: number;
  headItems: number;
  headMinListeners: number;
  maxListeners: number;
  minListeners: number;
  split: string;
  end: string;
  trainTransitions: number;
}

export interface RoutesFile {
  meta: RouteMeta;
  routes: SeedRoutes[];
}

export type Tracks = Record<string, Track>;
export type CI = [number, number, number]; // mean, lo, hi

export interface Lab {
  dataset: {
    users: number; items: number; trainPairs: number; evalUsers: number; split: string; end: string;
    trainDays: number; testDays: number; tailItems: number; headItems: number; k: number;
  };
  models: { name: string; params: Record<string, unknown>; recall: CI; ndcg: CI; tail: CI; novelty: CI; coverage: number }[];
  rerankers: { name: string; recall: CI; tail: CI; artistDiv: CI }[];
  paired: Record<string, { recall: CI; tail: CI }>;   // model minus ItemKNN, same listeners
  cold: {
    set: { n_cold_items: number; known_artist_items: number; users: number; discoveries: number; cold_test_pairs_without_audio: number };
    models: { name: string; r2: number | null; recall: CI; newArtist: CI; knownArtist: CI }[];
  };
  transitions: { gap_minutes: number; train_transitions: number; train_pairs: number; test_transitions: number; test_new_to_user: number };
  next: { name: string; hit: CI; mrr: CI; hit_new: CI; hit_tail: CI }[];
  routes: {
    name: string; recall: CI; tail_recall: CI; novelty: CI; tail_share: CI; artist_div: CI; realism: CI; audio_step: CI; length: CI;
  }[];
  routeUsers: number;
  audioMedian: number;
  routeSeeds: number;
  routeSteps: number;
  bandit?: Bandit;
}

export interface Est { mean: number; bias: number; rmse: number; q05: number; q95: number }

export interface Bandit {
  meta: { K: number; users: number; logging: { tau: number; eps: number }; log_sizes: number[]; reps: number };
  hitsPerListener: number;
  listenersWithHit: number;
  truth: Record<string, number>;
  tailTruth: Record<string, number>;
  ope: { n: number; policy: string; truth: number; ess: number; DM: Est; IPS: Est; SNIPS: Est; DR: Est }[];
  explore: { eps: number; loggingValue: number; baseLoggingValue: number; rows: { policy: string; ess: number; rmse: number }[] };
  online: {
    best: { eps: number; alpha: number; v: number };
    grid: { eps: number[]; alpha: number[]; v: number[] };
    rounds: number; seeds: number; tune_listeners: number; report_listeners: number;
    hindsight_value: number; oracle_value: number; hindsight_theta: Record<string, number>;
    curves: Record<string, { t: number[]; mean: number[]; lo: number[]; hi: number[] }>;
    finals: Record<string, { mean: number; sd: number; last_fifth: number; last_fifth_sd: number; gap_to_hindsight: number }>;
    order: string[];
  };
}

// punctuation splits anywhere; words split only with whitespace on both sides
const LEAD_SPLIT = /\s*[,&+×•;]\s*|\s+(?:feat\.?|ft\.?|featuring|with|w\/|x|vs\.?|versus|and|presents|pres\.|meets|reinterpreted by|remixed by|conducted by|starring|\/)\s+/i;

/** "Korn feat. Skrillex" -> "korn". Mirrors lead_artist() in scripts/export_web.py. */
export function leadArtist(name: string): string {
  const lead = name.split(LEAD_SPLIT)[0].trim();
  return (lead || name).toLowerCase();
}

/** A paired difference whose 95% interval includes zero. */
export const spansZero = (c: CI) => c[1] <= 0 && c[2] >= 0;

export const gridKey = (lam: number, gamma: number) => `${lam}|${gamma}`;

export function routeFor(r: SeedRoutes, lam: number, gamma: number): Step[] {
  const steps = r.grid[gridKey(lam, gamma)];
  if (!steps) throw new Error(`no route for ${gridKey(lam, gamma)}`);
  return steps;
}

/** Track ids along a route, seed first. */
export const stops = (r: SeedRoutes, steps: Step[]) => [r.seed, ...steps.map((s) => s.to)];

export function median(xs: number[]): number {
  const s = [...xs].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

/** Vertical position for a listener count: log scale, 0 = most played, 1 = least. */
export function reachY(listeners: number, meta: Pick<RouteMeta, "maxListeners" | "minListeners">): number {
  const hi = Math.log(meta.maxListeners);
  const lo = Math.log(Math.max(1, meta.minListeners));
  const v = Math.log(Math.max(listeners, meta.minListeners));
  return Math.min(1, Math.max(0, (hi - v) / (hi - lo)));
}

/** Case- and accent-insensitive search over seeds by title or artist. */
export function searchSeeds(routes: SeedRoutes[], tracks: Tracks, q: string): SeedRoutes[] {
  const fold = (s: string) => s.normalize("NFKD").replace(/[̀-ͯ]/g, "").toLowerCase();
  const terms = fold(q).split(/\s+/).filter(Boolean);
  if (!terms.length) return routes;
  return routes.filter((r) => {
    const t = tracks[r.seed];
    const hay = fold(`${t.title} ${t.artist}`);
    return terms.every((w) => hay.includes(w));
  });
}

export interface RouteSummary {
  endListeners: number;
  seedListeners: number;
  fold: number;          // seed listeners / end listeners
  endsInTail: boolean;
  heldOut: number;       // steps also made by a listener after the cutoff
  observed: number;      // steps with at least one training transition
  tailStops: number;
  medianListeners: number;
}

export function summarise(r: SeedRoutes, steps: Step[], tracks: Tracks): RouteSummary {
  const seed = tracks[r.seed];
  const end = tracks[steps[steps.length - 1].to];
  return {
    endListeners: end.listeners,
    seedListeners: seed.listeners,
    fold: seed.listeners / end.listeners,
    endsInTail: end.tail,
    heldOut: steps.filter((s) => s.heldOut).length,
    observed: steps.filter((s) => s.count > 0).length,
    tailStops: steps.filter((s) => tracks[s.to].tail).length,
    medianListeners: median(steps.map((s) => tracks[s.to].listeners)),
  };
}

/** The landing seed: among well-known songs (500+ listeners) whose default route ends
 *  in the long tail, the one that descends most steadily (steps that lose listeners),
 *  then has the most steps repeated after the cutoff, then the most steps seen in training. */
export function featured(routes: SeedRoutes[], tracks: Tracks, meta: RouteMeta): SeedRoutes {
  const score = (r: SeedRoutes) => {
    const steps = routeFor(r, ...meta.default);
    const ls = stops(r, steps).map((i) => tracks[i].listeners);
    const down = ls.slice(1).filter((l, i) => l < ls[i]).length;
    const s = summarise(r, steps, tracks);
    const heard = stops(r, steps).every((i) => tracks[i].preview);   // a landing route you can hear end to end
    return (s.endsInTail ? 1000 : 0) + (heard ? 500 : 0) + down * 10 + s.heldOut * 3 + s.observed;
  };
  const known = routes.filter((r) => r.group === "top" && tracks[r.seed].listeners >= 500);
  return (known.length ? known : routes).reduce((best, r) => (score(r) > score(best) ? r : best));
}

/** Routes the hash encodes: #/route/<seed>?l=<lam>&g=<gamma>, #/lab, #/lab/<section>. */
export type View =
  | { name: "route"; seed: number | null; lam: number | null; gamma: number | null }
  | { name: "lab"; section: string | null };

export function parseHash(hash: string): View {
  const h = hash.replace(/^#\/?/, "");
  const [path, query = ""] = h.split("?");
  const parts = path.split("/").filter(Boolean);
  if (parts[0] === "lab") return { name: "lab", section: parts[1] ?? null };
  const q = new URLSearchParams(query);
  const num = (v: string | null | undefined) => (v != null && v !== "" && Number.isFinite(Number(v)) ? Number(v) : null);
  return { name: "route", seed: num(parts[0] === "route" ? parts[1] : null), lam: num(q.get("l")), gamma: num(q.get("g")) };
}

export function routeHash(seed: number, lam: number, gamma: number, meta: RouteMeta): string {
  const [dl, dg] = meta.default;
  const q = lam === dl && gamma === dg ? "" : `?l=${lam}&g=${gamma}`;
  return `#/route/${seed}${q}`;
}

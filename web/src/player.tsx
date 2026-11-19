import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import type { Tracks } from "./data";
import { Icon } from "./icons";

// Safari and iOS only let a page start audio from inside the user's own tap or click.
// So every control below sets the source and calls play() synchronously in its handler;
// nothing waits for a React render to start playback. Once an element has played from a
// gesture, the browser lets it continue on its own.
//
// Crossfades: two elements take turns. Before the end of a sample the next one starts on the
// other element and the two cross over with raised-cosine curves (slow start, slow finish, so
// neither arrival nor departure is sudden). A skip crosses both ways in SKIP seconds. Where the
// page cannot set volume (iOS Safari ignores it) there is one element and samples cut.
//
// Sample length: the listener picks how long each song plays (Apple's previews are 30 s).

export const SAMPLES = [10, 15, 30] as const;
export type Sample = (typeof SAMPLES)[number];
const fadeFor = (len: number) => (len >= 30 ? 4 : len >= 15 ? 3 : 2);   // seconds of overlap
const SKIP = 0.7;     // seconds to cross when the listener skips or picks another song
const STEP = 40;      // ms between volume steps
const KEY = "obscure.sample";
const SILENT = "data:audio/wav;base64,UklGRnQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YVAAAACAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgA==";

interface PlayerState {
  queue: number[];          // track ids with a preview, in play order
  index: number;
  playing: boolean;
  current: number | null;
  failed: number | null;    // the last track whose preview would not load
  sample: Sample;
  setSample: (s: Sample) => void;
  /** Play ids from ids[start]. If that song is already playing, it carries on and only the queue changes. */
  playQueue: (ids: number[], start?: number) => void;
  toggle: (id?: number) => void;
  next: () => void;
  prev: () => void;
}

const Ctx = createContext<PlayerState | null>(null);
export const usePlayer = () => {
  const c = useContext(Ctx);
  if (!c) throw new Error("usePlayer outside PlayerProvider");
  return c;
};

function volumeWorks(): boolean {
  if (typeof Audio === "undefined") return false;
  const t = new Audio();
  t.volume = 0.5;
  return t.volume === 0.5;
}

function savedSample(): Sample {
  try {
    const v = Number(localStorage.getItem(KEY));
    return (SAMPLES as readonly number[]).includes(v) ? (v as Sample) : 30;
  } catch { return 30; }
}

export function PlayerProvider({ tracks, children }: { tracks: Tracks; children: ReactNode }) {
  const els = useRef<HTMLAudioElement[]>([]);
  const cur = useRef(0);                                   // which element is the one we hear
  const blessed = useRef([false, false]);                  // played once from a gesture
  const canFade = useRef<boolean | null>(null);
  const fade = useRef<number | null>(null);                // running volume interval
  const handing = useRef(false);                           // this sample has started handing on
  const q = useRef<{ ids: number[]; i: number }>({ ids: [], i: 0 });
  const [queue, setQueue] = useState<number[]>([]);
  const [index, setIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [progress, setProgress] = useState(0);
  const [failed, setFailed] = useState<number | null>(null);
  const [sample, setSampleState] = useState<Sample>(savedSample);
  const len = useRef<Sample>(sample);
  const current = queue.length ? queue[index] : null;

  const setSample = useCallback((s: Sample) => {
    len.current = s;
    setSampleState(s);
    try { localStorage.setItem(KEY, String(s)); } catch { /* private mode: keep it for this visit */ }
  }, []);

  const pair = () => {
    if (!els.current.length) {
      canFade.current = volumeWorks();
      els.current = [0, 1].map(() => { const a = new Audio(); a.preload = "none"; return a; });
    }
    return els.current;
  };
  const el = () => pair()[cur.current];
  /** Where this sample ends: the listener's length, or the preview's end if shorter. */
  const endOf = (a: HTMLAudioElement) => Math.min(a.duration || Infinity, len.current);

  /** Stop any crossover now: the element we hear at full volume, the other silent. */
  const settle = () => {
    if (fade.current != null) { clearInterval(fade.current); fade.current = null; }
    const [a, b] = [el(), pair()[1 - cur.current]];
    if (canFade.current) a.volume = 1;
    if (!b.paused) b.pause();
  };

  /** Fade `out` down and `inn` up over `secs` with raised-cosine curves; `inn` may be null. */
  const cross = (out: HTMLAudioElement, inn: HTMLAudioElement | null, secs: number) => {
    if (fade.current != null) clearInterval(fade.current);
    const v0 = out.volume, t0 = performance.now();
    fade.current = window.setInterval(() => {
      const k = Math.min(1, (performance.now() - t0) / (secs * 1000));
      const up = (1 - Math.cos(k * Math.PI)) / 2;          // 0 → 1, flat at both ends
      out.volume = v0 * (1 - up);
      if (inn) inn.volume = up;
      if (k >= 1) {
        clearInterval(fade.current!); fade.current = null;
        out.pause();
      }
    }, STEP);
  };

  /** Load track i of ids and play it. Safe to call from a gesture handler. */
  const start = useCallback((ids: number[], i: number, how: "skip" | "hand-on" = "skip") => {
    const id = ids[i];
    const url = tracks[id]?.preview?.url;
    if (!url) return;
    const els2 = pair();
    const before = q.current;
    q.current = { ids, i };
    setQueue(ids);
    setIndex(i);
    setFailed(null);
    setProgress(0);
    handing.current = false;

    if (!canFade.current) {                                 // one element, cut from one sample to the next
      const a = els2[0];
      a.src = url;
      a.play().catch((e: DOMException) => {
        setPlaying(false);
        if (e?.name !== "AbortError") setFailed(id);       // AbortError = another track was chosen first
      });
      return;
    }

    const from = cur.current, to = 1 - from;
    const a = els2[from], b = els2[to];
    if (fade.current != null) { clearInterval(fade.current); fade.current = null; }
    const crossing = !a.paused && blessed.current[from];
    b.src = url;
    b.volume = crossing ? 0 : 1;
    cur.current = to;
    const p = b.play();
    blessed.current[to] = true;
    if (!blessed.current[from] && how === "skip") {         // first gesture: let the idle element play later too
      a.src = SILENT;
      a.play().then(() => a.pause()).catch(() => {});
      blessed.current[from] = true;
    } else if (crossing) {
      cross(a, b, how === "hand-on" ? fadeFor(len.current) : SKIP);
    }
    p.catch((e: DOMException) => {
      if (e?.name === "AbortError") return;
      if (how === "hand-on" && e?.name === "NotAllowedError") {
        // this browser won't start the second element by itself: go back to one element and cuts
        settleBack(from, before);
        return;
      }
      setPlaying(false);
      setFailed(id);
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tracks]);

  /** A hand-on was refused: keep hearing the old element and let its end start the next sample. */
  const settleBack = (from: number, before: { ids: number[]; i: number }) => {
    if (fade.current != null) { clearInterval(fade.current); fade.current = null; }
    canFade.current = false;
    const [keep, drop] = [els.current[from], els.current[1 - from]];
    keep.volume = 1;
    drop.removeAttribute("src");
    els.current = [keep, drop];                             // the one-element path plays on els[0]
    cur.current = 0;
    q.current = before;
    setIndex(before.i);
    handing.current = true;                                 // the "ended" event takes it from here
  };

  // element events: progress, state, the next sample, failures. Only the element we hear counts.
  useEffect(() => {
    const [e0, e1] = pair();
    const mine = (e: Event) => e.target === el();
    const onTime = (e: Event) => {
      if (!mine(e)) return;
      const a = el();
      const end = endOf(a);
      setProgress(Number.isFinite(end) && end > 0 ? Math.min(1, a.currentTime / end) : 0);
      if (handing.current || !Number.isFinite(end)) return;
      const { ids, i } = q.current;
      const more = i + 1 < ids.length;
      if (canFade.current) {
        if (end - a.currentTime > fadeFor(len.current)) return;
        handing.current = true;
        if (more) start(ids, i + 1, "hand-on");
        else if (end < a.duration) cross(a, null, fadeFor(len.current));   // last song: fade out at the sample's end
      } else if (end < a.duration && a.currentTime >= end) {               // no fades: cut at the sample's end
        handing.current = true;
        if (more) start(ids, i + 1);
        else a.pause();
      }
    };
    const onPlay = (e: Event) => { if (mine(e)) setPlaying(true); };
    const onPause = (e: Event) => { if (mine(e)) setPlaying(false); };
    const onEnd = (e: Event) => {
      if (!mine(e)) return;
      const { ids, i } = q.current;
      if (i + 1 < ids.length) start(ids, i + 1);
    };
    const onError = (e: Event) => {
      if (!mine(e) || !el().getAttribute("src")) return;
      const { ids, i } = q.current;
      setFailed(ids[i]);
      setPlaying(false);
      if (i + 1 < ids.length) start(ids, i + 1);           // skip a preview that will not load
    };
    const on: [string, (e: Event) => void][] = [["timeupdate", onTime], ["play", onPlay], ["pause", onPause], ["ended", onEnd], ["error", onError]];
    for (const a of [e0, e1]) for (const [k, f] of on) a.addEventListener(k, f);
    return () => { for (const a of [e0, e1]) for (const [k, f] of on) a.removeEventListener(k, f); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [start]);

  useEffect(() => () => { for (const a of els.current) a.pause(); }, []);

  const playQueue = useCallback((ids: number[], from = 0) => {
    const has = (id: number) => !!tracks[id]?.preview?.url;
    const playable = ids.filter(has);
    if (!playable.length) return;
    const first = ids.slice(from).find(has) ?? playable[0];   // a stop with no preview plays the next one that has one
    const k = playable.indexOf(first);
    const { ids: qi, i } = q.current;
    if (playable[k] === qi[i] && els.current.length && !el().paused) {   // already playing it: just follow the new route
      q.current = { ids: playable, i: k };
      setQueue(playable);
      setIndex(k);
      return;
    }
    start(playable, k);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [start, tracks]);

  const toggle = useCallback((id?: number) => {
    if (id != null && id !== current) { start([id], 0); return; }
    if (current == null) return;
    settle();
    const a = el();
    if (a.paused) {
      if (a.currentTime >= endOf(a) - 0.2) { a.currentTime = 0; handing.current = false; }   // sample finished: from the top
      a.play().catch(() => setPlaying(false));
    } else a.pause();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [current, start]);

  const next = useCallback(() => {
    const { ids, i } = q.current;
    if (i + 1 < ids.length) start(ids, i + 1);
  }, [start]);

  const prev = useCallback(() => {
    const a = el();
    const { ids, i } = q.current;
    if (a.currentTime > 3 || i === 0) { settle(); a.currentTime = 0; handing.current = false; return; }
    start(ids, i - 1);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [start]);

  // the operating system's media controls: lock screen, headset and keyboard media keys
  useEffect(() => {
    const ms = typeof navigator !== "undefined" ? navigator.mediaSession : undefined;
    if (!ms) return;
    const t = current != null ? tracks[current] : null;
    ms.metadata = t && typeof MediaMetadata !== "undefined"
      ? new MediaMetadata({ title: t.title, artist: t.artist, album: "Obscure · preview",
          artwork: t.preview?.art ? [{ src: t.preview.art, sizes: "300x300", type: "image/jpeg" }] : [] })
      : null;
    const set = (k: MediaSessionAction, f: (() => void) | null) => { try { ms.setActionHandler(k, f); } catch { /* unsupported */ } };
    set("play", current != null ? () => toggle() : null);
    set("pause", current != null ? () => toggle() : null);
    set("nexttrack", queue.length > 1 ? next : null);
    set("previoustrack", queue.length > 1 ? prev : null);
  }, [current, queue.length, tracks, toggle, next, prev]);

  useEffect(() => {
    document.body.classList.toggle("has-bar", current != null);
  }, [current]);

  const value = useMemo(() => ({ queue, index, playing, current, failed, sample, setSample, playQueue, toggle, next, prev }),
    [queue, index, playing, current, failed, sample, setSample, playQueue, toggle, next, prev]);

  return (
    <Ctx.Provider value={value}>
      {children}
      {current != null && <PlayerBar tracks={tracks} progress={progress} />}
    </Ctx.Provider>
  );
}

function PlayerBar({ tracks, progress }: { tracks: Tracks; progress: number }) {
  const { current, playing, toggle, next, prev, queue, index, failed } = usePlayer();
  if (current == null) return null;
  const t = tracks[current];
  const p = t.preview!;
  return (
    <div className="bar" role="region" aria-label="Player">
      <div className="progress" style={{ width: `${progress * 100}%` }} aria-hidden="true" />
      <div className="wrap">
        {p.art ? <img className="art" src={p.art} alt="" width={48} height={48} /> : <span className="art" />}
        <div className="now">
          <span className="t">{t.title}</span>
          <span className="a">
            {failed === current ? "This preview didn’t load" : t.artist}
            {queue.length > 1 ? ` · ${index + 1} of ${queue.length}` : ""}
          </span>
        </div>
        <div className="ctl">
          <a className="credit" href={p.link} target="_blank" rel="noreferrer">Preview · Apple Music ↗</a>
          {queue.length > 1 && (
            <button className="icon-btn" onClick={prev} aria-label="Previous">
              <Icon name="prev" />
            </button>
          )}
          <button className="play" onClick={() => toggle()} aria-label={playing ? `Pause ${t.title}` : `Play ${t.title}`}>
            <Icon name={playing ? "pause" : "play"} />
          </button>
          {queue.length > 1 && (
            <button className="icon-btn" onClick={next} aria-label="Next" disabled={index >= queue.length - 1}>
              <Icon name="next" />
            </button>
          )}
        </div>
      </div>
      <p className="sr-only" aria-live="polite">{playing ? `Playing ${t.title} by ${t.artist}` : ""}</p>
    </div>
  );
}

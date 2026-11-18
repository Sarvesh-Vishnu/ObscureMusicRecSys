import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import type { Tracks } from "./data";
import { Icon } from "./icons";

// Safari and iOS only let a page start audio from inside the user's own tap or click.
// So every control below sets the source and calls play() synchronously in its handler;
// nothing waits for a React render to start playback. Once an element has played from a
// gesture, the browser lets it continue on its own (the next preview when one ends).

interface PlayerState {
  queue: number[];          // track ids with a preview, in play order
  index: number;
  playing: boolean;
  current: number | null;
  failed: number | null;    // the last track whose preview would not load
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

export function PlayerProvider({ tracks, children }: { tracks: Tracks; children: ReactNode }) {
  const audio = useRef<HTMLAudioElement | null>(null);
  const q = useRef<{ ids: number[]; i: number }>({ ids: [], i: 0 });
  const [queue, setQueue] = useState<number[]>([]);
  const [index, setIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [progress, setProgress] = useState(0);
  const [failed, setFailed] = useState<number | null>(null);
  const current = queue.length ? queue[index] : null;

  const el = () => {
    if (!audio.current) {
      const a = new Audio();
      a.preload = "none";
      audio.current = a;
    }
    return audio.current;
  };

  /** Load track i of ids and play it. Safe to call from a gesture handler. */
  const start = useCallback((ids: number[], i: number) => {
    const id = ids[i];
    const url = tracks[id]?.preview?.url;
    if (!url) return;
    q.current = { ids, i };
    setQueue(ids);
    setIndex(i);
    setFailed(null);
    setProgress(0);
    const a = el();
    a.src = url;
    a.play().catch((e: DOMException) => {
      setPlaying(false);
      if (e?.name !== "AbortError") setFailed(id);   // AbortError = another track was chosen first
    });
  }, [tracks]);

  // element events: progress, state, the next preview, failures
  useEffect(() => {
    const a = el();
    const onTime = () => setProgress(a.duration ? a.currentTime / a.duration : 0);
    const onPlay = () => setPlaying(true);
    const onPause = () => setPlaying(false);
    const onEnd = () => {
      const { ids, i } = q.current;
      if (i + 1 < ids.length) start(ids, i + 1);
    };
    const onError = () => {
      const { ids, i } = q.current;
      setFailed(ids[i]);
      setPlaying(false);
      if (i + 1 < ids.length) start(ids, i + 1);           // skip a preview that will not load
    };
    a.addEventListener("timeupdate", onTime);
    a.addEventListener("play", onPlay);
    a.addEventListener("pause", onPause);
    a.addEventListener("ended", onEnd);
    a.addEventListener("error", onError);
    return () => {
      a.removeEventListener("timeupdate", onTime);
      a.removeEventListener("play", onPlay);
      a.removeEventListener("pause", onPause);
      a.removeEventListener("ended", onEnd);
      a.removeEventListener("error", onError);
    };
  }, [start]);

  useEffect(() => () => audio.current?.pause(), []);

  const playQueue = useCallback((ids: number[], from = 0) => {
    const playable = ids.filter((id) => tracks[id]?.preview?.url);
    if (!playable.length) return;
    start(playable, Math.max(0, playable.indexOf(ids[from])));
  }, [start, tracks]);

  const toggle = useCallback((id?: number) => {
    const a = el();
    if (id != null && id !== current) { start([id], 0); return; }
    if (current == null) return;
    if (a.paused) a.play().catch(() => setPlaying(false));
    else a.pause();
  }, [current, start]);

  const next = useCallback(() => {
    const { ids, i } = q.current;
    if (i + 1 < ids.length) start(ids, i + 1);
  }, [start]);

  const prev = useCallback(() => {
    const a = el();
    const { ids, i } = q.current;
    if (a.currentTime > 3 || i === 0) { a.currentTime = 0; return; }
    start(ids, i - 1);
  }, [start]);

  // the operating system's media controls: lock screen, headset and keyboard media keys
  useEffect(() => {
    const ms = typeof navigator !== "undefined" ? navigator.mediaSession : undefined;
    if (!ms) return;
    const t = current != null ? tracks[current] : null;
    ms.metadata = t && typeof MediaMetadata !== "undefined"
      ? new MediaMetadata({ title: t.title, artist: t.artist, album: "Obscure · 30-second preview",
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

  const value = useMemo(() => ({ queue, index, playing, current, failed, playQueue, toggle, next, prev }),
    [queue, index, playing, current, failed, playQueue, toggle, next, prev]);

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
          <a className="credit" href={p.link} target="_blank" rel="noreferrer">30-second preview · Apple Music ↗</a>
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

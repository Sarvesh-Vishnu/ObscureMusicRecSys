import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import type { Tracks } from "./data";
import { Icon } from "./icons";

interface PlayerState {
  queue: number[];          // track ids with a preview, in play order
  index: number;
  playing: boolean;
  current: number | null;
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
  const [queue, setQueue] = useState<number[]>([]);
  const [index, setIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [progress, setProgress] = useState(0);
  const current = queue.length ? queue[index] : null;

  useEffect(() => {
    const a = new Audio();
    a.preload = "none";
    audio.current = a;
    const onTime = () => setProgress(a.duration ? a.currentTime / a.duration : 0);
    const onPlay = () => setPlaying(true);
    const onPause = () => setPlaying(false);
    a.addEventListener("timeupdate", onTime);
    a.addEventListener("play", onPlay);
    a.addEventListener("pause", onPause);
    return () => { a.pause(); a.removeEventListener("timeupdate", onTime); a.removeEventListener("play", onPlay); a.removeEventListener("pause", onPause); };
  }, []);

  // advance through the queue when a preview ends
  useEffect(() => {
    const a = audio.current;
    if (!a) return;
    const onEnd = () => setIndex((i) => (i + 1 < queue.length ? i + 1 : i));
    a.addEventListener("ended", onEnd);
    return () => a.removeEventListener("ended", onEnd);
  }, [queue.length]);

  // load and play whenever the current track changes
  const lastLoaded = useRef<number | null>(null);
  useEffect(() => {
    const a = audio.current;
    if (!a || current == null) return;
    if (lastLoaded.current === current && a.src) return;
    const url = tracks[current]?.preview?.url;
    if (!url) return;
    lastLoaded.current = current;
    a.src = url;
    setProgress(0);
    a.play().catch(() => setPlaying(false));
  }, [current, index, tracks]);

  const playable = useCallback((ids: number[]) => ids.filter((id) => tracks[id]?.preview?.url), [tracks]);

  const playQueue = useCallback((ids: number[], start = 0) => {
    const q = playable(ids);
    if (!q.length) return;
    const startId = ids[start];
    const at = Math.max(0, q.indexOf(startId));
    lastLoaded.current = null;
    setQueue(q);
    setIndex(at);
  }, [playable]);

  const toggle = useCallback((id?: number) => {
    const a = audio.current;
    if (!a) return;
    if (id != null && id !== current) {
      if (!tracks[id]?.preview?.url) return;
      lastLoaded.current = null;
      setQueue([id]);
      setIndex(0);
      return;
    }
    if (a.paused) a.play().catch(() => setPlaying(false));
    else a.pause();
  }, [current, tracks]);

  const next = useCallback(() => setIndex((i) => Math.min(i + 1, queue.length - 1)), [queue.length]);
  const prev = useCallback(() => {
    const a = audio.current;
    if (a && a.currentTime > 3) { a.currentTime = 0; return; }
    setIndex((i) => Math.max(i - 1, 0));
  }, []);

  useEffect(() => {
    document.body.classList.toggle("has-bar", current != null);
  }, [current]);

  const value = useMemo(() => ({ queue, index, playing, current, playQueue, toggle, next, prev }),
    [queue, index, playing, current, playQueue, toggle, next, prev]);

  return (
    <Ctx.Provider value={value}>
      {children}
      {current != null && <PlayerBar tracks={tracks} progress={progress} />}
    </Ctx.Provider>
  );
}

function PlayerBar({ tracks, progress }: { tracks: Tracks; progress: number }) {
  const { current, playing, toggle, next, prev, queue, index } = usePlayer();
  if (current == null) return null;
  const t = tracks[current];
  const p = t.preview!;
  return (
    <div className="bar" role="region" aria-label="Player">
      <div className="progress" style={{ width: `${progress * 100}%` }} aria-hidden="true" />
      <div className="wrap">
        {p.art ? <img className="art" src={p.art} alt="" width={48} height={48} /> : <span className="art" />}
        <div className="now" aria-live="polite">
          <span className="t">{t.title}</span>
          <span className="a">{t.artist}{queue.length > 1 ? ` · ${index + 1} of ${queue.length}` : ""}</span>
        </div>
        <div className="ctl">
          <a className="credit" href={p.link} target="_blank" rel="noreferrer">30-second preview · Apple Music ↗</a>
          <button className="icon-btn" onClick={prev} aria-label="Previous" disabled={queue.length < 2 && index === 0}>
            <Icon name="prev" />
          </button>
          <button className="play" onClick={() => toggle()} aria-label={playing ? "Pause" : "Play"}>
            <Icon name={playing ? "pause" : "play"} />
          </button>
          <button className="icon-btn" onClick={next} aria-label="Next" disabled={index >= queue.length - 1}>
            <Icon name="next" />
          </button>
        </div>
      </div>
    </div>
  );
}

import { useEffect, useRef, useState } from "react";
import { reachY, type RouteMeta, type Track } from "./data";
import { int } from "./format";
import { useWidth } from "./useWidth";

const L = 50, R = 16, T = 30, B = 34;

function useTween(target: number[], ms = 320): number[] {
  const [v, setV] = useState(target);
  const from = useRef(target);
  const key = target.join(",");
  useEffect(() => {
    const reduce = typeof matchMedia === "function" && matchMedia("(prefers-reduced-motion: reduce)").matches;
    const start = from.current;
    if (reduce || start.length !== target.length) { from.current = target; setV(target); return; }
    let raf = 0;
    const t0 = performance.now();
    const tick = (now: number) => {
      const k = Math.min(1, (now - t0) / ms);
      const e = 1 - Math.pow(1 - k, 3);
      const cur = target.map((x, i) => start[i] + (x - start[i]) * e);
      from.current = cur;
      setV(cur);
      if (k < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);
  return v;
}

const TICKS = [1000, 300, 100, 30, 10];

/**
 * now: the stop whose preview is loaded in the player (null if none). travelling: the
 * route itself is the queue, so the stretch not yet heard is drawn faint.
 * obvious: the standard recommender's picks, drawn in grey in a column of their own past
 * the last step, so where they sit compares directly with where the route ends.
 */
export function RouteFigure({ stops, meta, selected, onSelect, label, now, travelling, playing = false, obvious }: {
  stops: Track[]; meta: RouteMeta; selected: number; onSelect: (i: number) => void; label: string;
  now: number | null; travelling: boolean; playing?: boolean; obvious: Track[];
}) {
  const [box, W] = useWidth<HTMLDivElement>(720);
  const H = W < 520 ? 250 : 300;
  const OB = obvious.length ? (W < 520 ? 60 : 92) : 0;     // width of the standard-picks column
  const IW = W - L - R - OB, IH = H - T - B;
  const ys = useTween(stops.map((s) => T + reachY(s.listeners, meta) * IH));
  const x = (i: number) => L + (i * IW) / (stops.length - 1);
  const yThr = T + reachY(meta.headMinListeners, meta) * IH;
  const path = (from: number, to: number) => ys.slice(from, to + 1)
    .map((y, k) => `${k ? "L" : "M"}${x(from + k).toFixed(1)},${y.toFixed(1)}`).join(" ");
  const cut = travelling && now != null ? now : stops.length - 1;
  const shown = now ?? selected;
  const sel = stops[shown];
  const sx = x(shown), sy = ys[shown];
  const anchor = shown === 0 ? "start" : shown === stops.length - 1 ? "end" : "middle";
  const obX = L + IW + OB * 0.62;
  const obY = obvious.map((t) => T + reachY(t.listeners, meta) * IH);
  const obTop = Math.min(...obY), obBot = Math.max(...obY);
  const maxTag = W < 520 ? 18 : 34;
  const tagY = sy - 16 < T - 8 ? sy + 26 : sy - 16;

  return (
    <div ref={box}>
    <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} role="group" aria-label={label}>
      <rect className="tailzone" x={L} y={yThr} width={W - R - L} height={T + IH - yThr} />
      {TICKS.map((t) => {
        const y = T + reachY(t, meta) * IH;
        return (
          <g key={t}>
            <line className="grid" x1={L} x2={W - R} y1={y} y2={y} />
            <text className="axis" x={L - 10} y={y + 4} textAnchor="end">{int(t)}</text>
          </g>
        );
      })}
      <line className="threshold" x1={L} x2={W - R} y1={yThr} y2={yThr} />
      {OB > 0 && <line className="ob-sep" x1={L + IW + OB * 0.24} x2={L + IW + OB * 0.24} y1={T - 6} y2={T + IH} />}
      <text className="axis zone" x={L + 8} y={yThr - 7}>head ↑</text>
      <text className="axis zone" x={L + 8} y={yThr + 15}>long tail ↓ under {meta.headMinListeners} listeners</text>
      <text className="axis" x={0} y={T - 14}>listeners</text>
      {stops.map((_, i) => (
        <text key={i} className="axis" x={x(i)} y={H - 8} textAnchor="middle">{i === 0 ? "seed" : i}</text>
      ))}
      {obvious.length > 0 && (
        <g className="ob">
          <line className="ob-range" x1={obX} x2={obX} y1={obTop} y2={obBot} />
          {obvious.map((t, i) => (
            <circle key={i} className="ob-dot" cx={obX + (i - (obvious.length - 1) / 2) * 6} cy={obY[i]} r={3.5}>
              <title>{`Standard recommender: ${t.title} — ${t.artist}, ${int(t.listeners)} listeners`}</title>
            </circle>
          ))}
          <text className="axis" x={Math.min(obX, W - 2 - 30)} y={T - 14} textAnchor="middle">standard</text>
        </g>
      )}
      <path className="line" d={path(0, cut)} />
      {cut < stops.length - 1 && <path className="line ahead" d={path(cut, stops.length - 1)} />}
      {stops.map((s, i) => (
        <g key={i} className="node" role="button" tabIndex={0} onClick={() => onSelect(i)}
          onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onSelect(i); } }}
          aria-label={`${i === now && travelling && playing ? "Pause" : "Play from"} ${i === 0 ? "the seed" : `step ${i}`}: ${s.title} by ${s.artist}, ${int(s.listeners)} listeners`}>
          <circle className="hit" cx={x(i)} cy={ys[i]} r={20}>
            <title>{`${s.title} — ${s.artist}: ${int(s.listeners)} listeners. Click to play from here.`}</title>
          </circle>
          <circle className="focus" cx={x(i)} cy={ys[i]} r={15} />
          {i === selected && i !== now && <circle className="halo" cx={x(i)} cy={ys[i]} r={11} />}
          {i === now && <circle className="now" cx={x(i)} cy={ys[i]} r={11} />}
          <circle className={`stop${i === 0 ? " seed" : ""}${i === stops.length - 1 ? " end" : ""}`} cx={x(i)} cy={ys[i]} r={i === stops.length - 1 ? 6.5 : 5.5} />
        </g>
      ))}
      <text className="tag" x={sx} y={tagY} textAnchor={anchor}>
        {sel.artist.length > maxTag ? `${sel.artist.slice(0, maxTag - 1)}…` : sel.artist}
        <tspan className="tag-sub" dx={8}>{int(sel.listeners)}</tspan>
      </text>
    </svg>
    </div>
  );
}

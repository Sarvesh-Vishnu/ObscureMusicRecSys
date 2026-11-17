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

export function RouteFigure({ stops, meta, selected, onSelect, label }: {
  stops: Track[]; meta: RouteMeta; selected: number; onSelect: (i: number) => void; label: string;
}) {
  const [box, W] = useWidth<HTMLDivElement>(720);
  const H = W < 520 ? 250 : 300;
  const IW = W - L - R, IH = H - T - B;
  const ys = useTween(stops.map((s) => T + reachY(s.listeners, meta) * IH));
  const x = (i: number) => L + (i * IW) / (stops.length - 1);
  const yThr = T + reachY(meta.headMinListeners, meta) * IH;
  const d = ys.map((y, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y.toFixed(1)}`).join(" ");
  const sel = stops[selected];
  const sx = x(selected), sy = ys[selected];
  const anchor = selected === 0 ? "start" : selected === stops.length - 1 ? "end" : "middle";
  const maxTag = W < 520 ? 18 : 34;
  const tagY = sy - 16 < T - 8 ? sy + 26 : sy - 16;

  return (
    <div ref={box}>
    <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} role="img" aria-label={label}>
      <rect className="tailzone" x={L} y={yThr} width={IW} height={T + IH - yThr} />
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
      <text className="axis zone" x={L + 8} y={yThr - 7}>head ↑</text>
      <text className="axis zone" x={L + 8} y={yThr + 15}>long tail ↓ under {meta.headMinListeners} listeners</text>
      <text className="axis" x={0} y={T - 14}>listeners</text>
      {stops.map((_, i) => (
        <text key={i} className="axis" x={x(i)} y={H - 8} textAnchor="middle">{i === 0 ? "seed" : i}</text>
      ))}
      <path className="line" d={d} />
      {stops.map((s, i) => (
        <g key={i} onClick={() => onSelect(i)}>
          <circle className="hit" cx={x(i)} cy={ys[i]} r={20}>
            <title>{`${s.title} — ${s.artist}: ${int(s.listeners)} listeners`}</title>
          </circle>
          {i === selected && <circle className="halo" cx={x(i)} cy={ys[i]} r={11} />}
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

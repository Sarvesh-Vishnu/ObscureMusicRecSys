// Hand-drawn SVG figures for the Lab. One accent (the thing the figure is about),
// neutrals for everything else, direct labels, a hover/focus tooltip on every mark
// and a table view beside each figure.
import { useRef, useState, type ReactNode } from "react";
import type { CI } from "./data";
import { useWidth } from "./useWidth";

type Fmt = (v: number) => string;

export interface Pt {
  id: string;
  label: string;
  x: CI;
  y: CI;
  on?: boolean;
  show?: "r" | "l" | "t" | "b" | null;   // direct label position; null = tooltip/table only
  tip: string[];
}

/** Axis bounds and ticks on a 1-2-2.5-5 step, so labels land on round values. */
export function niceScale(hi: number, lo = 0, target = 4): { min: number; max: number; ticks: number[] } {
  const raw = (hi - lo) / target;
  const p = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * p).find((s) => s >= raw) ?? 10 * p;
  const min = Math.floor(lo / step + 1e-9) * step;
  const max = Math.ceil(hi / step - 1e-9) * step;
  const ticks = Array.from({ length: Math.round((max - min) / step) + 1 }, (_, i) => +(min + i * step).toPrecision(10));
  return { min, max, ticks };
}

function useTip() {
  const box = useRef<HTMLDivElement>(null);
  const [tip, setTip] = useState<{ x: number; y: number; lines: string[]; title: string } | null>(null);
  const show = (e: { currentTarget: Element }, title: string, lines: string[]) => {
    const b = box.current?.getBoundingClientRect();
    const r = e.currentTarget.getBoundingClientRect();
    if (!b) return;
    const x = r.left + r.width / 2 - b.left;
    setTip({ x: Math.min(Math.max(x, 8), b.width - 200), y: r.top - b.top, title, lines });
  };
  const node = tip && (
    <div className="tip" style={{ left: tip.x, top: tip.y, transform: "translate(-10%, calc(-100% - 10px))" }} role="status">
      <b>{tip.title}</b>
      {tip.lines.map((l) => <span key={l} className="mono">{l}</span>)}
    </div>
  );
  return { box, show, hide: () => setTip(null), node };
}

export function Scatter({ points, paths = [], xLabel, yLabel, xFmt, yFmt, xMax, yMax, xMin = 0, yMin = 0, desc, height = 360 }: {
  points: Pt[]; paths?: { ids: string[]; on?: boolean }[]; xLabel: string; yLabel: string;
  xFmt: Fmt; yFmt: Fmt; xMax?: number; yMax?: number; xMin?: number; yMin?: number; desc: string; height?: number;
}) {
  const t = useTip();
  const [box, W] = useWidth<HTMLDivElement>(640);
  const H = W < 480 ? Math.round(height * 0.85) : height, L = 48, R = 16, T = 20, B = 44;
  const xs = niceScale(xMax ?? Math.max(...points.map((p) => p.x[2])) * 1.05, xMin);
  const ys = niceScale(yMax ?? Math.max(...points.map((p) => p.y[2])) * 1.05, yMin);
  const sx = (v: number) => L + ((v - xs.min) / (xs.max - xs.min)) * (W - L - R);
  const sy = (v: number) => T + (1 - (v - ys.min) / (ys.max - ys.min)) * (H - T - B);
  const byId = new Map(points.map((p) => [p.id, p]));
  return (
    <div className="chart-wrap" ref={(el) => { (t.box as React.MutableRefObject<HTMLDivElement | null>).current = el; (box as React.MutableRefObject<HTMLDivElement | null>).current = el; }}>
      <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} role="img" aria-label={desc}>
        {ys.ticks.map((v) => (
          <g key={`y${v}`}>
            <line className={v === ys.min ? "base" : "grid"} x1={L} x2={W - R} y1={sy(v)} y2={sy(v)} />
            <text className="axis" x={L - 8} y={sy(v) + 4} textAnchor="end">{yFmt(v)}</text>
          </g>
        ))}
        {xs.ticks.map((v) => (
          <text key={`x${v}`} className="axis" x={sx(v)} y={H - B + 18} textAnchor="middle">{xFmt(v)}</text>
        ))}
        <line className="base" x1={L} x2={L} y1={T} y2={H - B} />
        <text className="axis-title" x={W - R} y={H - 4} textAnchor="end">{xLabel} →</text>
        <text className="axis-title" x={L} y={T - 8} textAnchor="start">↑ {yLabel}</text>
        {paths.map((pa, i) => (
          <path key={i} className={`path${pa.on ? " on" : ""}`}
            d={pa.ids.map((id, j) => `${j ? "L" : "M"}${sx(byId.get(id)!.x[0])},${sy(byId.get(id)!.y[0])}`).join(" ")} />
        ))}
        {points.map((p) => (
          <g key={p.id}>
            <line className={`ci${p.on ? " on" : ""}`} x1={sx(p.x[1])} x2={sx(p.x[2])} y1={sy(p.y[0])} y2={sy(p.y[0])} />
            <line className={`ci${p.on ? " on" : ""}`} x1={sx(p.x[0])} x2={sx(p.x[0])} y1={sy(p.y[1])} y2={sy(p.y[2])} />
          </g>
        ))}
        {points.map((p) => {
          const cx = sx(p.x[0]), cy = sy(p.y[0]);
          // flip a side label that would run past the plot edge (≈7px per character)
          const room = (p.label.length * 7) + 12;
          const pos = p.show === "r" && cx + room > W - R ? "l" : p.show === "l" && cx - room < 0 ? "r" : p.show;
          const lx = pos === "r" ? cx + 10 : pos === "l" ? cx - 10 : cx;
          const ly = pos === "t" ? cy - 12 : pos === "b" ? cy + 20 : cy + 4;
          const anchor = pos === "r" ? "start" : pos === "l" ? "end" : "middle";
          return (
            <g key={p.id}>
              <circle className={`dot${p.on ? " on" : ""}`} cx={cx} cy={cy} r={5} />
              {pos && <text className={`lbl${p.on ? " on" : ""}`} x={lx} y={ly} textAnchor={anchor}>{p.label}</text>}
              <circle className="hit" cx={cx} cy={cy} r={14} role="img" tabIndex={0} aria-label={`${p.label}: ${p.tip.join(", ")}`}
                onMouseEnter={(e) => t.show(e, p.label, p.tip)} onMouseLeave={t.hide}
                onFocus={(e) => t.show(e, p.label, p.tip)} onBlur={t.hide} />
            </g>
          );
        })}
      </svg>
      {t.node}
    </div>
  );
}

export interface BarRow { id: string; label: string; v: CI; on?: boolean; tip: string[] }

/** Horizontal bars with 95% intervals: one row per model. */
export function Bars({ rows, fmt, max, desc, axis }: { rows: BarRow[]; fmt: Fmt; max?: number; desc: string; axis: string }) {
  const t = useTip();
  const [box, W] = useWidth<HTMLDivElement>(640);
  const row = 34, L = W < 480 ? 150 : 190, R = 56, T = 8, B = 34;
  const H = T + rows.length * row + B;
  const sc = niceScale(max ?? Math.max(...rows.map((r) => r.v[2])) * 1.02);
  const m = sc.max;
  const sx = (v: number) => L + (v / m) * (W - L - R);
  return (
    <div className="chart-wrap" ref={(el) => { (t.box as React.MutableRefObject<HTMLDivElement | null>).current = el; (box as React.MutableRefObject<HTMLDivElement | null>).current = el; }}>
      <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} role="img" aria-label={desc}>
        {sc.ticks.map((v) => (
          <g key={v}>
            <line className={v === 0 ? "base" : "grid"} x1={sx(v)} x2={sx(v)} y1={T} y2={H - B} />
            <text className="axis" x={sx(v)} y={H - B + 18} textAnchor="middle">{fmt(v)}</text>
          </g>
        ))}
        <text className="axis-title" x={W - R} y={H - 2} textAnchor="end">{axis} →</text>
        {rows.map((r, i) => {
          const y = T + i * row + row / 2;
          const w = Math.max(sx(r.v[0]) - L, 2);
          return (
            <g key={r.id}>
              <text className={`lbl${r.on ? " on" : ""}`} x={L - 12} y={y + 4} textAnchor="end">{r.label}</text>
              <rect className={`bar-rect${r.on ? " on" : ""}`} x={L} y={y - 7} width={w} height={14} rx={2} />
              <line className="ci" x1={sx(r.v[1])} x2={sx(r.v[2])} y1={y} y2={y} />
              <text className="axis" x={sx(Math.max(r.v[2], r.v[0])) + 8} y={y + 4}>{fmt(r.v[0])}</text>
              <rect className="hit" x={0} y={y - row / 2} width={W} height={row} role="img" tabIndex={0} aria-label={`${r.label}: ${r.tip.join(", ")}`}
                onMouseEnter={(e) => t.show(e, r.label, r.tip)} onMouseLeave={t.hide}
                onFocus={(e) => t.show(e, r.label, r.tip)} onBlur={t.hide} />
            </g>
          );
        })}
      </svg>
      {t.node}
    </div>
  );
}

export function DataTable({ head, rows, caption }: { head: string[]; rows: { on?: boolean; cells: ReactNode[] }[]; caption: string }) {
  return (
    <details className="table">
      <summary>Show as table</summary>
      <div className="tbl-scroll" tabIndex={0} role="region" aria-label={caption}>
        <table className="data">
          <caption className="sr-only">{caption}</caption>
          <thead><tr>{head.map((h) => <th key={h} scope="col">{h}</th>)}</tr></thead>
          <tbody>
            {rows.map((r, i) => <tr key={i} className={r.on ? "on" : ""}>{r.cells.map((c, j) => <td key={j}>{c}</td>)}</tr>)}
          </tbody>
        </table>
      </div>
    </details>
  );
}

export interface Band { lo: number; hi: number; mid: number; on?: boolean; name: string }
export interface IntervalRow { id: string; label: string; truth: number; bands: Band[]; note: string; tip: string[] }

/** One row per policy: its true value (ink tick) and each estimator's 5–95% range across logs. */
export function Intervals({ rows, fmt, desc, axis, max }: { rows: IntervalRow[]; fmt: Fmt; desc: string; axis: string; max?: number }) {
  const t = useTip();
  const [box, W] = useWidth<HTMLDivElement>(640);
  const nb = Math.max(...rows.map((r) => r.bands.length));
  const row = 18 + nb * 12, L = W < 480 ? 128 : 190, R = W < 480 ? 44 : 64, T = 8, B = 34;
  const H = T + rows.length * row + B;
  const hiV = max ?? Math.max(...rows.flatMap((r) => [r.truth, ...r.bands.map((b) => b.hi)]));
  const sc = niceScale(hiV, 0);
  const sx = (v: number) => L + (Math.max(v, 0) / sc.max) * (W - L - R);
  return (
    <div className="chart-wrap" ref={(el) => { (t.box as React.MutableRefObject<HTMLDivElement | null>).current = el; (box as React.MutableRefObject<HTMLDivElement | null>).current = el; }}>
      <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} role="img" aria-label={desc}>
        {sc.ticks.map((v) => (
          <g key={v}>
            <line className={v === 0 ? "base" : "grid"} x1={sx(v)} x2={sx(v)} y1={T} y2={H - B} />
            <text className="axis" x={sx(v)} y={H - B + 18} textAnchor="middle">{fmt(v)}</text>
          </g>
        ))}
        <text className="axis-title" x={W - R} y={H - 2} textAnchor="end">{axis} →</text>
        {rows.map((r, i) => {
          const y0 = T + i * row;
          const yc = y0 + row / 2;
          return (
            <g key={r.id}>
              <text className="lbl" x={L - 12} y={yc + 4} textAnchor="end">{r.label}</text>
              {r.bands.map((b, j) => {
                const y = y0 + 9 + j * 12 + (row - 18 - nb * 12) / 2 + 6;
                return (
                  <g key={b.name}>
                    <line className={`ci${b.on ? " on" : ""}`} strokeWidth={b.on ? 3 : 2} x1={sx(b.lo)} x2={sx(b.hi)} y1={y} y2={y} />
                    <circle className={`dot${b.on ? " on" : ""}`} cx={sx(b.mid)} cy={y} r={3.5} />
                  </g>
                );
              })}
              <line className="truth" x1={sx(r.truth)} x2={sx(r.truth)} y1={y0 + 3} y2={y0 + row - 3} />
              <text className="axis" x={W - R + 8} y={yc + 4}>{r.note}</text>
              <rect className="hit" x={0} y={y0} width={W} height={row} role="img" tabIndex={0} aria-label={`${r.label}: ${r.tip.join(", ")}`}
                onMouseEnter={(e) => t.show(e, r.label, r.tip)} onMouseLeave={t.hide}
                onFocus={(e) => t.show(e, r.label, r.tip)} onBlur={t.hide} />
            </g>
          );
        })}
      </svg>
      {t.node}
    </div>
  );
}

export interface Series { id: string; label: string; x: number[]; y: number[]; on?: boolean; dashed?: boolean }

/** Lines over a log-scaled x axis (rounds), labelled at their right end. */
export function Lines({ series, yFmt, xLabel, yLabel, desc, refs = [] }: {
  series: Series[]; yFmt: Fmt; xLabel: string; yLabel: string; desc: string; refs?: { label: string; y: number }[];
}) {
  const [box, W] = useWidth<HTMLDivElement>(640);
  const H = W < 480 ? 300 : 340, L = 48, R = W < 480 ? 96 : 150, T = 20, B = 44;
  const xs = series.flatMap((s) => s.x);
  const x0 = Math.log10(Math.min(...xs)), x1 = Math.log10(Math.max(...xs));
  const ys = niceScale(Math.max(...series.flatMap((s) => s.y), ...refs.map((r) => r.y)) * 1.05, 0);
  const sx = (v: number) => L + ((Math.log10(v) - x0) / (x1 - x0)) * (W - L - R);
  const sy = (v: number) => T + (1 - v / ys.max) * (H - T - B);
  const decades = [];
  for (let e = Math.ceil(x0); e <= Math.floor(x1); e++) decades.push(10 ** e);
  // nudge end labels apart so they never overlap
  const ends = series.map((s) => ({ id: s.id, y: sy(s.y[s.y.length - 1]) })).sort((a, b) => a.y - b.y);
  for (let i = 1; i < ends.length; i++) if (ends[i].y - ends[i - 1].y < 14) ends[i].y = ends[i - 1].y + 14;
  const endY = new Map(ends.map((e) => [e.id, e.y]));
  return (
    <div className="chart-wrap" ref={box}>
      <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} role="img" aria-label={desc}>
        {ys.ticks.map((v) => (
          <g key={v}>
            <line className={v === 0 ? "base" : "grid"} x1={L} x2={W - R} y1={sy(v)} y2={sy(v)} />
            <text className="axis" x={L - 8} y={sy(v) + 4} textAnchor="end">{yFmt(v)}</text>
          </g>
        ))}
        {decades.map((v) => <text key={v} className="axis" x={sx(v)} y={H - B + 18} textAnchor="middle">{v.toLocaleString("en-GB")}</text>)}
        <text className="axis-title" x={W - R} y={H - 4} textAnchor="end">{xLabel} →</text>
        <text className="axis-title" x={L} y={T - 8}>↑ {yLabel}</text>
        {refs.map((r) => (
          <g key={r.label}>
            <line className="threshold" x1={L} x2={W - R} y1={sy(r.y)} y2={sy(r.y)} />
          </g>
        ))}
        {[...series].sort((a, b) => Number(!!a.on) - Number(!!b.on)).map((s) => (
          <g key={s.id}>
            <path className={`path${s.on ? " on" : ""}`} strokeDasharray={s.dashed ? "4 4" : undefined}
              d={s.x.map((x, i) => `${i ? "L" : "M"}${sx(x).toFixed(1)},${sy(s.y[i]).toFixed(1)}`).join(" ")} />
            <text className={`lbl${s.on ? " on" : ""}`} x={W - R + 8} y={endY.get(s.id)! + 4}>{s.label}</text>
          </g>
        ))}
      </svg>
    </div>
  );
}

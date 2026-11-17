// Every sentence that quotes a number is built here from the data, never typed.
import type { CI, RouteMeta, Step, Track } from "./data";

const nf = new Intl.NumberFormat("en-GB");
export const int = (n: number) => nf.format(Math.round(n));

/** 0.0687 -> "6.87%"; digits chosen so small values stay readable. */
export function pct(x: number, digits?: number): string {
  const v = x * 100;
  const d = digits ?? (Math.abs(v) >= 10 ? 1 : Math.abs(v) >= 1 ? 2 : v === 0 ? 0 : 2);
  return `${v.toFixed(d)}%`;
}

export const ciText = (c: CI, digits?: number) => `${pct(c[0], digits)} (95% CI ${pct(c[1], digits)}–${pct(c[2], digits)})`;

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
export function day(iso: string, offset = 0): string {
  const d = new Date(`${iso}T00:00:00Z`);
  d.setUTCDate(d.getUTCDate() + offset);
  return `${d.getUTCDate()} ${MONTHS[d.getUTCMonth()]}`;
}

export const windowDays = (m: Pick<RouteMeta, "split" | "end">) =>
  Math.round((Date.parse(`${m.end}T00:00:00Z`) - Date.parse(`${m.split}T00:00:00Z`)) / 86400000);

const WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"];
export const spell = (n: number) => WORDS[n] ?? int(n);

/** "12–15 Nov": the test window (end is exclusive). */
export const testWindow = (m: Pick<RouteMeta, "split" | "end">) => `${day(m.split).split(" ")[0]}–${day(m.end, -1)}`;

export function stepEvidence(step: Step, from: Track): string {
  if (step.count === 0) {
    return `No one went straight here from “${from.title}” in training. The step comes from listeners the two songs share (the ItemKNN back-off).`;
  }
  const share = step.count / step.out;
  const sh = share >= 0.001 ? pct(share, share >= 0.1 ? 0 : 1) : "under 0.1%";
  return `${int(step.count)} of ${int(step.out)} plays that followed “${from.title}” went here next (${sh}).`;
}

export function soundChange(step: Step): string {
  const a = step.audio;
  if (a < 0.75) return `${a.toFixed(2)}× a typical step: close`;
  if (a <= 1.25) return `${a.toFixed(2)}× a typical step`;
  return `${a.toFixed(2)}× a typical step: a jump`;
}

export function reach(t: Track, meta: Pick<RouteMeta, "nItems">): string {
  return `${int(t.listeners)} listener${t.listeners === 1 ? "" : "s"} · #${int(t.rank)} of ${int(meta.nItems)}`;
}

export function fold(x: number): string {
  if (x >= 1.5) return `${x >= 10 ? int(x) : x.toFixed(1)}× fewer listeners than the seed`;
  if (x <= 1 / 1.5) return `${(1 / x).toFixed(1)}× more listeners than the seed`;
  return "about as many listeners as the seed";
}

export const plural = (n: number, one: string, many = `${one}s`) => `${n} ${n === 1 ? one : many}`;

import { readFileSync } from "node:fs";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type { Lab, RoutesFile, Tracks } from "./data";
import { LabView } from "./LabView";

const read = <T,>(f: string): T => JSON.parse(readFileSync(new URL(`../public/data/${f}`, import.meta.url), "utf8")) as T;
const lab = read<Lab>("lab.json");
const routes = read<RoutesFile>("routes.json");
const tracks = read<Tracks>("tracks.json");

describe("the Lab renders from the exported results", () => {
  const html = renderToStaticMarkup(<LabView lab={lab} routes={routes} tracks={tracks} section={null} />);
  const doc = new DOMParserLite(html);

  it("opens with one finding per chapter, each linking to a chapter on the page", () => {
    const links = doc.findingLinks();
    expect(links.length).toBe(lab.bandit ? 7 : 5);
    for (const id of links) expect(html).toContain(`id="${id}"`);
  });

  it("states a number in every finding", () => {
    for (const p of doc.findingSentences()) expect(p).toMatch(/\d/);
  });

  it("never prints a missing value", () => {
    expect(html).not.toMatch(/NaN|undefined|Infinity/);
  });
});

/** Just enough parsing for the findings list, without a DOM in the test environment. */
class DOMParserLite {
  private block: string;
  constructor(html: string) {
    this.block = html.match(/<section class="findings"[\s\S]*?<\/section>/)?.[0] ?? "";
  }
  findingLinks() {
    return [...this.block.matchAll(/<li>[\s\S]*?href="#\/lab\/([a-z]+)"/g)].map((m) => m[1]);
  }
  findingSentences() {
    return [...this.block.matchAll(/<li>[\s\S]*?<p>([\s\S]*?)<\/p>/g)].map((m) => m[1]);
  }
}

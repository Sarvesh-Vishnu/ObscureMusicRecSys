import { useEffect } from "react";
import { featured, median, routeFor, summarise, type Lab, type RoutesFile, type Tracks } from "./data";
import { int, pct } from "./format";

// Draft copy (Claude's wording, for the author to edit). Numbers are computed from the data.

export function AboutView({ routes, tracks, lab }: { routes: RoutesFile; tracks: Tracks; lab: Lab }) {
  useEffect(() => { document.title = "About — Obscure"; window.scrollTo({ top: 0 }); }, []);
  const m = routes.meta;
  const tailItems = m.nItems - m.headItems;
  const ends = routes.routes.map((r) => summarise(r, routeFor(r, ...m.default), tracks));
  const inTail = ends.filter((s) => s.endsInTail).length;
  const fold = median(ends.map((s) => s.fold));
  const obMedian = median(routes.routes.flatMap((r) => r.obvious.map((i) => tracks[i].listeners)));
  const f = featured(routes.routes, tracks, m);
  const knn = lab.models.find((x) => x.name === "ItemKNN");

  return (
    <main className="wrap about">
      <header>
        <p className="label">About</p>
        <h1>A way out of the songs everyone plays</h1>
        <p className="lede">
          Most recommenders play you more of what you already know. Obscure starts from a song you know and walks
          five steps to one almost nobody plays, so you hear where your taste could go, not where it already is.
        </p>
      </header>

      <section aria-labelledby="ab-1">
        <h2 id="ab-1">Not a sound-alike</h2>
        <p>
          The point is not a song that sounds like the one you started from. It is a route into the long tail: of
          the {int(m.nItems)} recordings in this data, {int(tailItems)} ({pct(tailItems / m.nItems, 0)}) have fewer
          than {int(m.headMinListeners)} listeners. A standard recommender rarely goes there; across the seeds here its
          picks have a median of {int(obMedian)} listeners. {inTail === routes.routes.length
            ? <>At the default setting every route here ends in the long tail (songs without such a route were left out),</>
            : <>At the default setting {inTail} of {routes.routes.length} routes end in the long tail,</>} on a song with a
          median {fold.toFixed(0)}× fewer listeners than where it began.
        </p>
      </section>

      <section aria-labelledby="ab-2">
        <h2 id="ab-2">Every step is one people took</h2>
        <p>
          Each step is a move real ListenBrainz listeners made, from one song straight to the next. Open any stop to see
          how many made it, how much the sound changed, and whether someone made it again after the model was frozen.
          Two dials shape the route: <b>smoothness</b> keeps each change in sound small, and <b>pull</b> keeps the route
          near songs that share listeners with where you started.
        </p>
      </section>

      <section aria-labelledby="ab-3">
        <h2 id="ab-3">It can miss</h2>
        <p>
          Going somewhere new means some routes end on a song you won’t like. Obscure doesn’t hide that; it makes a miss
          cheap and gives you somewhere else to go:
        </p>
        <ul>
          <li><b>Another route.</b> Each song has other routes that share no stop with the first. If one misses, try the next.</li>
          <li><b>Short samples.</b> Listen to each song for 10 seconds to scan a route quickly, or 30 to settle in.</li>
          <li><b>The dials.</b> More smoothness for gentler steps; more pull to stay closer to home (strong pull often never reaches the long tail).</li>
          <li><b>The evidence.</b> Every stop says why it’s there, so a miss is a step you can see, not a black box.</li>
        </ul>
        <p>
          What the data can’t say is whether you enjoy where a route ends. That takes listeners, and it is the test this
          project would run next.
        </p>
      </section>

      <section aria-labelledby="ab-4">
        <h2 id="ab-4">How it’s built</h2>
        <p>
          The routes come from a recommender benchmark on {int(lab.dataset.users)} ListenBrainz listeners: classic
          collaborative filtering{knn ? ` (ItemKNN finds ${pct(knn.recall[0])} of a listener’s next discoveries in its top 20)` : ""},
          re-ranking for reach, cold start from sound, next-song prediction, and off-policy evaluation. The <a href="#/lab">Lab</a> shows
          every result. Nothing runs in your browser except the player: no account, no cookies, no tracking.
        </p>
        <p className="cta">
          <a className="btn primary" href={`#/route/${f.seed}`}>Try a route</a>
          <a className="btn" href="https://github.com/Sarvesh-Vishnu/ObscureMusicRecSys">Read the code ↗</a>
        </p>
      </section>
    </main>
  );
}

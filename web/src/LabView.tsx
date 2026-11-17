import { useEffect } from "react";
import { Bars, DataTable, Intervals, Lines, Scatter, type Pt } from "./charts";
import { spansZero, type Bandit, type CI, type Lab, type RoutesFile, type Tracks } from "./data";
import { day, int, pct, spell } from "./format";

const REPO = "https://github.com/Sarvesh-Vishnu/ObscureMusicRecSys";
const pp = (a: number, b: number) => `${Math.abs((a - b) * 100).toFixed(2)} points`;
const ratio = (a: number, b: number) => (b > 0 ? `${(a / b).toFixed(1)}×` : "—");
const sp = (v: number) => `${v > 0 ? "+" : v < 0 ? "−" : ""}${Math.abs(v * 100).toFixed(2)}`;
const ci = (c: CI) => `${pct(c[0])} [${pct(c[1])}, ${pct(c[2])}]`;

function by<T extends { name: string }>(xs: T[], name: string): T {
  const x = xs.find((m) => m.name === name);
  if (!x) throw new Error(`missing result: ${name}`);
  return x;
}

export function LabView({ lab, routes, tracks, section }: { lab: Lab; routes: RoutesFile; tracks: Tracks; section: string | null }) {
  useEffect(() => { document.title = "Lab — Obscure"; }, []);
  useEffect(() => {
    if (section) document.getElementById(section)?.scrollIntoView();
    else window.scrollTo({ top: 0 });
  }, [section]);
  const d = lab.dataset;

  return (
    <main className="wrap lab">
      <header className="lab-head">
        <p className="label">Lab</p>
        <h1>The benchmark behind the routes</h1>
        <p>
          Each model was trained on ListenBrainz listens submitted before {day(d.split)} and scored on what the same people
          played for the first time over the next {spell(d.testDays)} days. The numbers on this page are read from the
          result files the benchmark wrote; none are typed by hand.
        </p>
        <dl className="facts">
          <div><dt>Listeners</dt><dd>{int(d.users)}</dd></div>
          <div><dt>Recordings</dt><dd>{int(d.items)}</dd></div>
          <div><dt>Listener–song pairs</dt><dd>{int(d.trainPairs)}</dd></div>
          <div><dt>Listeners scored</dt><dd>{int(d.evalUsers)}</dd></div>
        </dl>
        <ul className="toc" aria-label="Sections">
          {[["models", "1 Recommenders"], ["rerank", "2 Re-ranking"], ["cold", "3 Cold start"], ["next", "4 Next song"], ["routes", "5 Routes"], ...(lab.bandit ? [["ope", "6 Off-policy"], ["online", "7 Bandits"]] : []), ["limits", lab.bandit ? "8 Limits" : "6 Limits"]]
            .map(([id, t]) => <li key={id}><a href={`#/lab/${id}`}>{t}</a></li>)}
        </ul>
      </header>
      <Models lab={lab} />
      <Rerank lab={lab} />
      <Cold lab={lab} />
      <Next lab={lab} />
      <Routes lab={lab} />
      {lab.bandit && <Ope b={lab.bandit} />}
      {lab.bandit && <Online b={lab.bandit} />}
      <Limits lab={lab} routes={routes} tracks={tracks} />
    </main>
  );
}

function Chapter({ id, n, title, prose, children }: { id: string; n: string; title: string; prose: React.ReactNode; children: React.ReactNode }) {
  return (
    <section className="chapter" id={id} aria-labelledby={`${id}-h`}>
      <div className="prose">
        <p className="n">{n}</p>
        <h2 id={`${id}-h`}>{title}</h2>
        {prose}
      </div>
      <div>{children}</div>
    </section>
  );
}

function Models({ lab }: { lab: Lab }) {
  const m = lab.models;
  const knn = by(m, "ItemKNN"), als = by(m, "ALS"), ease = by(m, "EASE");
  const alsP = lab.paired["ALS"];
  const tip = (x: Lab["models"][number]) => [`recall@20 ${ci(x.recall)}`, `long-tail recall@20 ${ci(x.tail)}`, `catalog coverage ${pct(x.coverage)}`];
  const place: Record<string, Pt["show"]> = { ItemKNN: "t", ALS: "r", EASE: "t", BPR: "r", Popularity: "r" };
  const pts: Pt[] = m.map((x) => ({ id: x.name, label: x.name, x: x.recall, y: x.tail, on: x.name === "ItemKNN", show: place[x.name] ?? "r", tip: tip(x) }));
  return (
    <Chapter id="models" n="1 · Recommenders" title="Similar recall, very different reach" prose={<>
      <p>
        Recall@20 asks how many of a listener’s next discoveries appear in their top 20. On that, <b>ItemKNN</b> and ALS
        {spansZero(alsP.recall)
          ? <> are tied: {pct(knn.recall[0])} and {pct(als.recall[0])}, and the paired difference over the same listeners
            runs from {sp(alsP.recall[1])} to {sp(alsP.recall[2])} points.</>
          : <> differ by {pp(als.recall[0], knn.recall[0])}: {pct(knn.recall[0])} and {pct(als.recall[0])}.</>}{" "}Restrict the question to
        discoveries in the long tail and ItemKNN finds {ratio(knn.tail[0], als.tail[0])} as many as ALS. EASE, a strong
        linear model, finds {ease.tail[0] === 0 ? "none" : pct(ease.tail[0])}.
      </p>
      <p>
        So a single accuracy number hides the property this project cares about. Every later chapter reports both.
      </p>
      <p className="method">
        Tuned on a separate validation split (NDCG@20). {int(lab.dataset.evalUsers)} listeners, 95% bootstrap intervals.
        Long tail: recordings outside the head that holds half of all listening pairs.
      </p>
    </>}>
      <figure className="chart">
        <Scatter points={pts} xLabel="recall@20" yLabel="long-tail recall@20" xFmt={(v) => pct(v, 0)} yFmt={(v) => pct(v, 1)}
          desc="Scatter of five recommenders: recall at 20 against long-tail recall at 20." />
        <figcaption>Each dot is a model; the lines are 95% intervals. Up means more long-tail discoveries found.</figcaption>
        <DataTable caption="Recommenders on the test split" head={["Model", "Recall@20", "NDCG@20", "Long-tail recall@20", "Coverage"]}
          rows={m.map((x) => ({ on: x.name === "ItemKNN", cells: [x.name, ci(x.recall), ci(x.ndcg), ci(x.tail), pct(x.coverage)] }))} />
      </figure>
    </Chapter>
  );
}

function Rerank({ lab }: { lab: Lab }) {
  const rr = lab.rerankers;
  const base = by(rr, "ItemKNN"), p25 = by(rr, "PopPenalty β=0.25"), p1 = by(rr, "PopPenalty β=1");
  const mmr = by(rr, "MMR λ=0.5"), calHi = by(rr, "Calibrated λ=0.95");
  const calP = lab.paired["Calibrated λ=0.2"], mmrP = lab.paired["MMR λ=0.5"];
  const fam = (n: string) => rr.filter((x) => x.name.startsWith(n)).map((x) => x.name);
  const show: Record<string, Pt["show"]> = { ItemKNN: "b", "PopPenalty β=0.25": "r", "PopPenalty β=1": "t", "MMR λ=0.5": "b", "Calibrated λ=0.95": "l" };
  const pts: Pt[] = rr.map((x) => ({
    id: x.name, label: x.name, x: x.recall, y: x.tail, on: x.name.startsWith("PopPenalty"), show: show[x.name] ?? null,
    tip: [`recall@20 ${ci(x.recall)}`, `long-tail recall@20 ${ci(x.tail)}`, `artist diversity ${pct(x.artistDiv[0], 0)}`],
  }));
  return (
    <Chapter id="rerank" n="2 · Re-ranking" title="Buying reach with recall" prose={<>
      <p>
        Three re-rankers reorder ItemKNN’s top 200. <b>PopPenalty</b> divides each score by popularity<sup>β</sup>: at β = 0.25
        recall falls {pp(p25.recall[0], base.recall[0])} and long-tail recall rises {pp(p25.tail[0], base.tail[0])},
        {" "}{ratio(p25.tail[0], base.tail[0])} the base. At β = 1 the trade is steeper: {pp(p1.recall[0], base.recall[0])} of
        recall for {pp(p1.tail[0], base.tail[0])} of reach.
      </p>
      <p>
        Calibration (Steck 2018) at λ = 0.2 {spansZero(calP.recall)
          ? <>costs no measurable recall (paired change {sp(calP.recall[1])} to {sp(calP.recall[2])} points)</>
          : <>changes recall by {sp(calP.recall[0])} points</>} and adds {sp(calP.tail[0])} points of long-tail recall.
        MMR raises artist diversity from {pct(base.artistDiv[0], 0)} to {pct(mmr.artistDiv[0], 0)} at λ = 0.5,
        {spansZero(mmrP.tail) ? " but its change in long-tail recall is indistinguishable from zero." : ` and moves long-tail recall by ${sp(mmrP.tail[0])} points.`}
        {" "}Different artists are not the same as less-played songs. Pushed to λ = 0.95, calibration lands where the mildest
        penalty starts ({pct(calHi.recall[0])} and {pct(calHi.tail[0])} against {pct(p25.recall[0])} and {pct(p25.tail[0])}).
      </p>
    </>}>
      <figure className="chart">
        <Scatter points={pts} paths={[{ ids: ["ItemKNN", ...fam("PopPenalty")], on: true }, { ids: ["ItemKNN", ...fam("Calibrated")] }, { ids: ["ItemKNN", ...fam("MMR")] }]}
          xLabel="recall@20" yLabel="long-tail recall@20" xFmt={(v) => pct(v, 1)} yFmt={(v) => pct(v, 0)} xMin={0.045} yMin={0.02}
          desc="Re-rankers on ItemKNN: each family traces a path from the base model as its strength grows." />
        <figcaption>Each line starts at plain ItemKNN and follows one re-ranker as its strength grows; the axes are zoomed. Hover or focus a dot for its setting.</figcaption>
        <DataTable caption="Re-rankers on ItemKNN" head={["Re-ranker", "Recall@20", "Long-tail recall@20", "Artist diversity"]}
          rows={rr.map((x) => ({ on: x.name.startsWith("PopPenalty"), cells: [x.name, ci(x.recall), ci(x.tail), pct(x.artistDiv[0], 0)] }))} />
      </figure>
    </Chapter>
  );
}

function Cold({ lab }: { lab: Lab }) {
  const c = lab.cold;
  const rnd = by(c.models, "Random"), af = by(c.models, "Audio → factors"), cen = by(c.models, "Artist centroid");
  const audioOnly = new Set(["Audio profile", "Audio → factors"]);
  const order = [...c.models].sort((a, b) => b.recall[0] - a.recall[0]);
  return (
    <Chapter id="cold" n="3 · Cold start" title="Sound alone barely finds anything" prose={<>
      <p>
        Most obscure music has too few listeners for collaborative filtering to see it. The cold catalog here is
        {" "}{int(c.set.n_cold_items)} recordings outside the model’s vocabulary, each with AcousticBrainz audio analysis.
      </p>
      <p>
        A network that learns ALS taste vectors from <b>sound alone</b> reaches {pct(af.recall[0])} recall@20,
        {" "}{ratio(af.recall[0], rnd.recall[0])} random, and explains {af.r2 != null ? pct(af.r2, 1) : "—"} of the variance in
        held-out taste vectors. The average taste vector of the artist’s other songs reaches {pct(cen.recall[0])}.
      </p>
      <p>
        Sound describes how a song sounds, not who will like it. That is why the routes use it only to keep each step
        smooth, never as the reason for a step.
      </p>
      <p className="method">Content→factor model after van den Oord et al. (2013), with a leave-one-out artist centroid. Artist models score zero on new artists by construction.</p>
    </>}>
      <figure className="chart">
        <Bars rows={order.map((m) => ({
          id: m.name, label: m.name, v: m.recall, on: audioOnly.has(m.name),
          tip: [`recall@20 ${ci(m.recall)}`, `new artists ${ci(m.newArtist)}`, `known artists ${ci(m.knownArtist)}`],
        }))} fmt={(v) => pct(v, 1)} axis="recall@20 on the cold catalog" desc="Cold-start recall at 20 by model; audio-only models are highlighted." />
        <figcaption>Highlighted: the two models that see only the audio. Bars are means; the lines are 95% intervals.</figcaption>
        <DataTable caption="Cold-start models" head={["Model", "Recall@20", "New artists", "Known artists", "Factor R²"]}
          rows={c.models.map((m) => ({ on: audioOnly.has(m.name), cells: [m.name, ci(m.recall), ci(m.newArtist), ci(m.knownArtist), m.r2 != null ? m.r2.toFixed(3) : "—"] }))} />
      </figure>
    </Chapter>
  );
}

function Next({ lab }: { lab: Lab }) {
  const knn = by(lab.next, "ItemKNN neighbours");
  const tr = lab.next.find((x) => x.name.startsWith("Transition"))!;
  const t = lab.transitions;
  return (
    <Chapter id="next" n="4 · Next song" title="What people play next is its own signal" prose={<>
      <p>
        Two listens by the same person less than {t.gap_minutes} minutes apart make a transition: {int(t.train_transitions)} of
        them in training. Counting transitions finds the true next song in the top 20 {pct(tr.hit[0], 1)} of the time
        against {pct(knn.hit[0], 1)} for co-listening similarity, and ranks it about {ratio(tr.mrr[0], knn.mrr[0])} higher
        (MRR {pct(tr.mrr[0], 1)} against {pct(knn.mrr[0], 1)}).
      </p>
      <p>
        When the next song is new to the listener the two tie, {pct(tr.hit_new[0], 1)} against {pct(knn.hit_new[0], 1)}. Transitions
        are better at the familiar next song, and that is the edge the routes walk along.
      </p>
      <p className="method">First-order transition counts with an ItemKNN back-off and a popularity discount tuned on validation (Ludewig &amp; Jannach 2018), scored on a random sample of held-out transitions.</p>
    </>}>
      <div className="tbl-scroll">
        <table className="data">
          <caption className="sr-only">Next-song prediction on held-out transitions</caption>
          <thead><tr><th scope="col">Model</th><th scope="col">Hit@20</th><th scope="col">MRR@20</th><th scope="col">Hit@20, new to listener</th><th scope="col">Hit@20, long tail</th></tr></thead>
          <tbody>
            {lab.next.map((m) => (
              <tr key={m.name} className={m === tr ? "on" : ""}>
                <td>{m.name.replace(" γ=0", "")}</td><td>{pct(m.hit[0], 1)}</td><td>{pct(m.mrr[0], 1)}</td><td>{pct(m.hit_new[0], 1)}</td><td>{pct(m.hit_tail[0], 1)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="figcaption muted" style={{ fontSize: 14 }}>
        Every 95% interval in this table is within ±{(Math.max(...lab.next.flatMap((m) => [m.hit, m.mrr, m.hit_new, m.hit_tail].map((c) => (c[2] - c[1]) / 2))) * 100).toFixed(1)} points;
        the full intervals are in the results file.
      </p>
    </Chapter>
  );
}

function Routes({ lab }: { lab: Lab }) {
  const R = lab.routes;
  const pop = by(R, "Top-20: PopPenalty β=0.5"), knn = by(R, "Top-20: ItemKNN"), mp = by(R, "Route: most probable");
  const sm = by(R, "Route: smooth audio (λ=1)"), g10 = by(R, "Route: personal γ=10 + smooth");
  const label: Record<string, [string, Pt["show"]]> = {
    "Top-20: ItemKNN": ["ItemKNN top 20", "r"], "Top-20: PopPenalty β=0.5": ["PopPenalty top 20", "b"],
    "Top-20: ItemKNN, one per artist": ["one per artist", "r"], "Random walk": ["random walk", "b"],
    "Route: smooth audio (λ=1)": ["route, γ = 0", "b"], "Route: personal γ=10 + smooth": ["route, γ = 10", "r"],
  };
  const pts: Pt[] = R.map((x) => ({
    id: x.name, label: label[x.name]?.[0] ?? x.name, x: x.tail_share, y: x.realism, on: x.name.startsWith("Route"), show: label[x.name]?.[1] ?? null,
    tip: [x.name, `recall@20 ${ci(x.recall)}`, `realism ${ci(x.realism)}`, `long-tail share ${pct(x.tail_share[0], 0)}`, `sound per step ${x.audio_step[0].toFixed(2)}`],
  }));
  const smoothPath = ["Route: smooth audio (λ=1)", "Route: personal + smooth", "Route: personal γ=3 + smooth", "Route: personal γ=10 + smooth"];
  return (
    <Chapter id="routes" n="5 · Routes" title="As deep as a popularity penalty, on steps people take" prose={<>
      <p>
        A route spends the same budget as a top-20 list ({spell(lab.routeSeeds)} seeds, {spell(lab.routeSteps)} steps each) under beam search on the next-song graph.
        <b> Realism</b> is the share of steps some listener made again in the held-out window.
      </p>
      <p>
        Routes reach as far into the long tail as PopPenalty, {pct(mp.tail_share[0], 0)} of stops against {pct(pop.tail_share[0], 0)},
        and {ratio(mp.realism[0], pop.realism[0])} as many of their steps are real ({pct(mp.realism[0], 1)} against {pct(pop.realism[0], 1)}).
        The smoothness term cuts the sound change per step from {mp.audio_step[0].toFixed(2)} to {sm.audio_step[0].toFixed(2)}.
      </p>
      <p>
        The cost is recall: routes leave the neighbourhood on purpose, so they rarely contain what the listener was
        about to play anyway ({pct(mp.recall[0])} against {pct(knn.recall[0])} for ItemKNN). Pull towards the listener’s
        taste buys it back: at γ = 10, {pct(g10.recall[0])}, with {pct(g10.tail_share[0], 0)} of stops in the tail.
        Whether people enjoy where a route ends is a question for an online test.
      </p>
      <p className="method">{int(lab.routeUsers)} test listeners; seeds are each listener’s most-played songs, one per artist. A route may not revisit a song, reuse an artist or land on anything the listener has heard.</p>
    </>}>
      <figure className="chart">
        <Scatter points={pts} paths={[{ ids: smoothPath, on: true }]} xLabel="share of stops in the long tail" yLabel="realism: steps listeners made again"
          xFmt={(v) => pct(v, 0)} yFmt={(v) => pct(v, 0)} xMin={0.2} xMax={0.65} desc="Lists and routes: long-tail share against realism." />
        <figcaption>Highlighted: routes. The line follows the pull setting γ from 0 to 10 with smoothing on; more pull, more realism, less depth.</figcaption>
        <DataTable caption="Lists and routes" head={["List", "Recall@20", "Long-tail share", "Realism", "Sound per step"]}
          rows={R.map((x) => ({ on: x.name.startsWith("Route"), cells: [x.name, ci(x.recall), pct(x.tail_share[0], 0), ci(x.realism), x.audio_step[0].toFixed(2)] }))} />
      </figure>
    </Chapter>
  );
}

function Limits({ lab, routes, tracks }: { lab: Lab; routes: RoutesFile; tracks: Tracks }) {
  const playable = new Set<number>();
  for (const r of routes.routes) {
    playable.add(r.seed);
    r.obvious.forEach((i) => playable.add(i));
    Object.values(r.grid).forEach((s) => s.forEach((x) => playable.add(x.to)));
  }
  const withPreview = [...playable].filter((i) => tracks[i]?.preview).length;
  const d = lab.dataset;
  const top = routes.routes.filter((r) => r.group === "top").map((r) => tracks[r.seed]);
  const topArtists = top.slice(0, 4).map((t) => t.artist);
  const topMax = Math.max(...top.map((t) => t.listeners));
  const indiaMax = Math.max(0, ...routes.routes.filter((r) => r.group === "india").map((r) => tracks[r.seed].listeners));
  return (
    <section className="chapter" id="limits" aria-labelledby="limits-h">
      <div className="prose">
        <p className="n">{lab.bandit ? 8 : 6} · Limits</p>
        <h2 id="limits-h">What this does not show</h2>
      </div>
      <div>
        <ul className="limits">
          <li>
            {spell(d.trainDays + d.testDays)} days of ListenBrainz listens ({d.trainDays} to train, {d.testDays} to test), from the people who
            choose to use ListenBrainz. The most played recordings are by {topArtists.join(", ")}; the best-played Indian seed
            has {int(indiaMax)} listeners against {int(topMax)} for the most played song.
          </li>
          <li>Offline recall rewards predicting what someone was going to play anyway. It cannot say whether a listener enjoys the far end of a route. That needs an online test.</li>
          <li>The app has no listener, so its pull is measured against the seed (songs that share its listeners), not against a listening history as in the benchmark. Its γ values are not the benchmark’s.</li>
          <li>Previews are matched to Apple’s catalogue by artist and title: {withPreview} of {playable.size} playable songs ({pct(withPreview / Math.max(playable.size, 1), 0)}) have one. The rest link to MusicBrainz.</li>
          <li>The benchmark counts each artist credit as its own artist, so “Korn feat. Skrillex” is not “Korn”, and its artist-diversity figures run a little high. The app’s routes use the lead artist instead.</li>
          {lab.bandit && <li>Chapters 6 and 7 are semi-synthetic: the listeners, candidates and rewards are real, but ListenBrainz has no logs of a recommender, so the logging policy is simulated. They show how the methods behave on this data, not what a deployed system would log.</li>}
          <li>AcousticBrainz stopped collecting in 2022, so newer recordings have no sound analysis; a step to or from one counts as a typical step.</li>
        </ul>
        <p className="figcaption" style={{ marginTop: 24 }}>
          Method, protocol and every table: <a href={`${REPO}#readme`}>README</a> · <a href={`${REPO}/blob/main/docs/PROTOCOL.md`}>PROTOCOL.md</a> · <a href={`${REPO}/tree/main/results`}>results/</a>
        </p>
      </div>
    </section>
  );
}

const SHORT: Record<string, string> = {
  "Logging policy (softmax ItemKNN)": "Logging policy itself",
  "ItemKNN, top pick": "ItemKNN top pick",
  "ALS, top pick": "ALS top pick",
  "PopPenalty β=0.5, top pick": "PopPenalty top pick",
  "Most popular candidate": "Most popular",
  "Long tail first": "Long tail first",
  "Uniform over candidates": "Uniform",
};
const within = (v: number, e: { q05: number; q95: number }) => v >= e.q05 && v <= e.q95;

function Ope({ b }: { b: Bandit }) {
  const n = 10_000;
  const rows = b.ope.filter((r) => r.n === n).sort((x, y) => y.ess - x.ess);
  const get = (p: string) => rows.find((r) => r.policy === p)!;
  const knn = get("ItemKNN, top pick"), tail = get("Long tail first"), pen = get("PopPenalty β=0.5, top pick"), als = get("ALS, top pick");
  const tailX = b.explore.rows.find((r) => r.policy === "Long tail first")!;
  const cost = b.explore.baseLoggingValue - b.explore.loggingValue;
  const dmMiss = rows.filter((r) => !within(r.truth, r.DM));
  return (
    <Chapter id="ope" n="6 · Off-policy evaluation" title="The policies worth trying are the hardest to judge from logs" prose={<>
      <p>
        A team that wants to try a new recommender usually has only the logs of the old one. Off-policy estimators reweight
        those logs to guess how the new one would do. ListenBrainz has no such logs, so here a logging policy is simulated on
        real listeners: it shows each one a single song from their {b.meta.K} ItemKNN candidates, mostly the top few.
        Because every candidate’s outcome is known, each estimate can be checked against the <b>true value</b>.
      </p>
      <p>
        For policies close to the logging policy, doubly robust estimation (<b>DR</b>) works: ItemKNN’s top pick is
        {" "}{pct(knn.truth)} and DR’s 90% range over {int(b.meta.reps)} logs of {int(n)} rounds is {pct(knn.DR.q05)}–{pct(knn.DR.q95)}.
        A policy that goes looking in the long tail picks songs the log rarely showed: {int(n)} logged rounds carry the
        information of about {int(tail.ess)}, and DR’s range for “long tail first” runs from {pct(tail.DR.q05)} to {pct(tail.DR.q95)} around
        a true {pct(tail.truth)}.{" "}
        {within(knn.truth, tail.DR) ? <>That range also contains ItemKNN’s true value, so the log cannot say which of the two is better</> : <>That range still excludes ItemKNN’s true value</>}
        {within(knn.truth, als.DR) && within(tail.truth, als.DR) ? <>; ALS ({pct(als.truth)}, ESS {int(als.ess)}) is just as unresolved.</> : "."}
      </p>
      <p>
        The model-based estimate (DM) is steady but can be steadily wrong: its range for PopPenalty, {pct(pen.DM.q05)}–{pct(pen.DM.q95)},
        {within(pen.truth, pen.DM) ? " contains" : " misses"} the true {pct(pen.truth)}{dmMiss.length > 1 ? `, and it misses the truth for ${dmMiss.length} of ${rows.length} policies` : ""}.
      </p>
      <p>
        Logging with more exploration helps, at a price paid up front. With {pct(b.explore.eps, 0)} of shows chosen at random instead
        of {pct(b.meta.logging.eps, 0)}, the logged system’s hit rate falls from {pct(b.explore.baseLoggingValue)} to {pct(b.explore.loggingValue)}
        {" "}({pct(cost / b.explore.baseLoggingValue, 0)} fewer hits), and DR’s error on “long tail first” falls from {(tail.DR.rmse * 100).toFixed(2)} to
        {" "}{(tailX.rmse * 100).toFixed(2)} points{tailX.rmse > Math.abs(knn.truth - tail.truth)
          ? <>, still larger than the {(Math.abs(knn.truth - tail.truth) * 100).toFixed(2)}-point gap between it and ItemKNN</> : <>, now smaller than the {(Math.abs(knn.truth - tail.truth) * 100).toFixed(2)}-point gap between it and ItemKNN</>}.
        A discovery-first product has to decide to pay for that evidence.
      </p>
      <p className="method">Supervised-to-bandit conversion (Dudík, Langford &amp; Li 2011). Logging: softmax over ItemKNN (τ = {b.meta.logging.tau}) with {pct(b.meta.logging.eps, 0)} uniform. Reward model cross-fitted on two folds. The exploration follow-up was added after the first result.</p>
    </>}>
      <figure className="chart">
        <Intervals rows={rows.map((r) => ({
          id: r.policy, label: SHORT[r.policy] ?? r.policy, truth: r.truth, note: `ESS ${int(r.ess)}`,
          bands: [{ name: "DR", lo: r.DR.q05, hi: r.DR.q95, mid: r.DR.mean, on: true }, { name: "DM", lo: r.DM.q05, hi: r.DM.q95, mid: r.DM.mean }],
          tip: [`true value ${pct(r.truth)}`, `DR ${pct(r.DR.q05)}–${pct(r.DR.q95)}`, `DM ${pct(r.DM.q05)}–${pct(r.DM.q95)}`, `IPS RMSE ${(r.IPS.rmse * 100).toFixed(2)} pts`, `effective sample ${int(r.ess)} of ${int(n)}`],
        }))} fmt={(v) => pct(v, 0)} axis="hit rate" desc="For each target policy, its true hit rate and the 90% range of doubly robust and direct-method estimates." />
        <figcaption>White tick: the true hit rate. Amber: the doubly robust estimate’s 90% range over {int(b.meta.reps)} logs of {int(n)} rounds; grey below it: the direct method’s. ESS is the effective sample size.</figcaption>
        <DataTable caption="Off-policy estimates at 10,000 logged rounds" head={["Policy", "True", "DM", "IPS", "SNIPS", "DR", "ESS"]}
          rows={rows.map((r) => ({ cells: [SHORT[r.policy] ?? r.policy, pct(r.truth), ...(["DM", "IPS", "SNIPS", "DR"] as const).map((e) => `${pct(r[e].mean)} ±${(r[e].rmse * 100).toFixed(2)}`), int(r.ess)] }))} />
      </figure>
    </Chapter>
  );
}

function Online({ b }: { b: Bandit }) {
  const o = b.online;
  const F = o.finals;
  const fixed = F["ItemKNN top pick (no learning)"], greedy = F["Greedy (learns, never explores)"];
  const ucbName = o.order.find((x) => x.startsWith("LinUCB"))!;
  const ucb = F[ucbName];
  const leastEverywhere = o.best.eps === Math.min(...o.grid.eps) && o.best.alpha === Math.min(...o.grid.alpha) && o.best.v === Math.min(...o.grid.v);
  const theta = Object.entries(o.hindsight_theta).filter(([k]) => k !== "bias").sort((x, y) => Math.abs(y[1]) - Math.abs(x[1]));
  const keep = ["Random", "ItemKNN top pick (no learning)", "Greedy (learns, never explores)", ucbName];
  const label: Record<string, string> = { Random: "random", "ItemKNN top pick (no learning)": "ItemKNN, fixed", "Greedy (learns, never explores)": "greedy learner", [ucbName]: "LinUCB" };
  return (
    <Chapter id="online" n="7 · Bandits" title="Learning from each hit beats the base ranker; exploring adds nothing here" prose={<>
      <p>
        A contextual bandit recommends one song, sees whether it was a hit, and updates. On the same candidates, a linear
        learner that sees only those single rewards climbs from ItemKNN’s fixed {pct(fixed.last_fifth)} to {pct(ucb.last_fifth)} by
        the last fifth of {int(o.rounds)} rounds ({ucbName}), {pct(ucb.last_fifth / fixed.last_fifth - 1, 0)} more hits. That is level with the same model
        fitted on all the feedback at once ({pct(o.hindsight_value)}).
      </p>
      <p>
        Exploration does not pay on this data. {leastEverywhere ? "Tuning chose the smallest exploration setting offered in every family, and " : ""}a
        greedy learner that never explores reaches {pct(greedy.last_fifth)}. That is consistent with Bastani, Bayati &amp; Khosravi (2021):
        when contexts vary enough, acting greedily already tries a spread of options.
      </p>
      {theta[0][0] === "als score" && (
        <p>
          Its largest weight is on the ALS score: the candidates come from ItemKNN, and the learner reorders them with what ALS knows.
        </p>
      )}
      <p className="method">Shared-parameter ε-greedy, LinUCB (Li et al. 2010) and linear Thompson sampling (Agrawal &amp; Goyal 2013). Tuned on {int(o.tune_listeners)} listeners, reported on the other {int(o.report_listeners)}; {o.seeds} seeds. Rewards don’t deplete.</p>
    </>}>
      <figure className="chart">
        <Lines series={keep.map((k) => ({ id: k, label: label[k], x: o.curves[k].t, y: o.curves[k].mean, on: k === ucbName }))}
          refs={[{ label: "hindsight fit", y: o.hindsight_value }]} yFmt={(v) => pct(v, 0)} xLabel="rounds (log scale)" yLabel="running hit rate"
          desc="Running hit rate over rounds for random, fixed ItemKNN, a greedy learner and LinUCB." />
        <figcaption>Running hit rate, mean of {o.seeds} seeds. Dashed: the same linear model fitted on all the feedback at once ({pct(o.hindsight_value)}).</figcaption>
        <DataTable caption="Online learners" head={["Learner", "All rounds", "Last fifth", "Hits short of hindsight fit"]}
          rows={o.order.map((k) => ({ on: k === ucbName, cells: [k, `${pct(F[k].mean)} ±${(F[k].sd * 100).toFixed(2)}`, `${pct(F[k].last_fifth)} ±${(F[k].last_fifth_sd * 100).toFixed(2)}`, int(F[k].gap_to_hindsight)] }))} />
      </figure>
    </Chapter>
  );
}

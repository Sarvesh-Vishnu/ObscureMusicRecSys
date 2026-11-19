import { useEffect, useMemo, useRef, useState } from "react";
import { featured, median, routeAt, routeCount, routeFor, routeHash, searchSeeds, stops as stopIds, summarise, type More, type RoutesFile, type SeedRoutes, type Step, type Track, type Tracks } from "./data";
import { fold, int, plural, reach, soundChange, spell, stepEvidence, testWindow, windowDays } from "./format";
import { Icon } from "./icons";
import { SAMPLES, usePlayer } from "./player";
import { RouteFigure } from "./RouteFigure";

const SMOOTH = ["Off", "Some", "Strong"];
const PULL = ["None", "Some", "Strong"];

export function RouteView({ data, tracks, more, seed, lam, gamma, alt }: {
  data: RoutesFile; tracks: Tracks; more: More | null; seed: number | null; lam: number | null; gamma: number | null; alt: number;
}) {
  const { meta, routes } = data;
  const r = useMemo(() => routes.find((x) => x.seed === seed) ?? featured(routes, tracks, meta), [routes, seed, tracks, meta]);
  const L = lam != null && meta.lams.includes(lam) ? lam : meta.default[0];
  const G = gamma != null && meta.gammas.includes(gamma) ? gamma : meta.default[1];
  const nRoutes = routeCount(r, L, G, more);
  const A = Math.min(alt, nRoutes - 1);
  const steps = routeAt(r, L, G, A, more);
  const ids = stopIds(r, steps);
  const stopTracks = ids.map((i) => tracks[i]);
  const sum = summarise(r, steps, tracks);
  const [selected, setSelected] = useState(1);
  const player = usePlayer();
  const seedT = tracks[r.seed];

  useEffect(() => { setSelected(1); }, [r.seed]);
  useEffect(() => { document.title = `${seedT.title} — Obscure`; }, [seedT.title]);

  const playable = ids.filter((i) => tracks[i].preview);
  const routePlaying = player.queue.length > 1 && player.queue.every((q) => ids.includes(q));
  const nowIdx = player.current != null ? ids.indexOf(player.current) : -1;
  /** Show another route; if this route is playing, the music follows to the same step of the new one. */
  const go = (s: number, l: number, g: number, a = 0) => {
    if (s === r.seed && routePlaying && player.playing) {
      const next = stopIds(r, routeAt(r, l, g, a, more));
      player.playQueue(next, Math.max(0, nowIdx));
    }
    location.hash = routeHash(s, l, g, meta, a);
  };
  /** A seed from the list: show its route without moving the page; if music is playing, play the new route. */
  const pick = (s: number, play: boolean) => {
    const to = routes.find((x) => x.seed === s);
    if (to && (play || player.playing)) player.playQueue(stopIds(to, routeFor(to, L, G)), 0);
    location.hash = routeHash(s, L, G, meta);
  };
  /** A stop on the figure: select it and play the route from there (or pause, if it's the one playing). */
  const fromStop = (i: number) => {
    setSelected(i);
    if (ids[i] === player.current && routePlaying) player.toggle();
    else player.playQueue(ids, i);
  };
  const obTracks = r.obvious.map((i) => tracks[i]);
  const obHead = obTracks.filter((t) => !t.tail).length;

  const obLs = obTracks.map((t) => t.listeners);
  const figLabel = `Route from ${seedT.title} by ${seedT.artist}: ` +
    stopTracks.slice(1).map((t, i) => `${i + 1}, ${t.title} by ${t.artist}, ${int(t.listeners)} listeners`).join("; ") +
    `. For comparison, a standard recommender's five picks have ${int(Math.min(...obLs))} to ${int(Math.max(...obLs))} listeners.`;

  return (
    <main className="wrap">
      <p className="lede">
        <strong>A route from a song you know to one almost nobody plays.</strong> Five steps, and each step is a
        move real ListenBrainz listeners made from one song to the next.
      </p>

      <div className="route-grid">
        <header className="seed-head a-head">
          <Cover t={seedT} size={168} className="cover" />
          <div className="seed-text">
          <p className="label">From</p>
          <h1>{seedT.title}</h1>
          <p className="by">{seedT.artist}</p>
          <p className="meta mono">{reach(seedT, meta)}</p>
          <div className="actions">
            <button className="btn primary" disabled={!playable.length}
              onClick={() => (routePlaying ? player.toggle() : player.playQueue(ids, 0))}>
              <Icon name={routePlaying && player.playing ? "pause" : "play"} />
              {routePlaying && player.playing ? "Pause route" : `Play the route`}
            </button>
            <a className="btn" href="#pick" onClick={(e) => { e.preventDefault(); document.getElementById("seed-search")?.focus(); }}>
              Start from another song
            </a>
          </div>
          <fieldset className="sample">
            <legend>Listen to each song for</legend>
            <div className="seg sm">
              {SAMPLES.map((v) => (
                <label key={v}><input type="radio" name="sample" checked={player.sample === v} onChange={() => player.setSample(v)} />{v} s</label>
              ))}
            </div>
          </fieldset>
          {playable.length > 0 && playable.length < ids.length && (
            <p className="note playable">
              {playable.length} of {ids.length} songs have a preview; the route plays those and skips the rest.
            </p>
          )}
          </div>
        </header>

        <div className="a-figure">
          <figure className="figure">
            <RouteFigure stops={stopTracks} meta={meta} selected={selected} onSelect={fromStop} label={figLabel}
              now={nowIdx >= 0 ? nowIdx : null} travelling={routePlaying} obvious={obTracks} />
          </figure>
          <p className="figcaption" aria-live="polite">
            {sum.endsInTail
              ? <>Ends on <b>{int(sum.endListeners)} listeners</b>, {fold(sum.fold)}. </>
              : <>Ends on <b>{int(sum.endListeners)} listeners</b>, still in the head: with this much pull the route stays near the seed. </>}
            {plural(sum.tailStops, "stop")} of 5 in the long tail. {sum.heldOut} of 5 steps were made again by a listener
            after the cutoff ({testWindow(meta)}), data no model saw.
          </p>
          <p className="figkey">
            <span className="ob-swatch" aria-hidden="true" /> Grey, at the right: the five songs a standard recommender
            returns next, {obHead === 5 ? "all" : `${obHead} of 5`} in the head.
          </p>
        </div>

        <section className="a-dials dials" aria-labelledby="dials-h">
          <h2 id="dials-h" className="sr-only">Tune the route</h2>
          <fieldset className="dial">
            <legend>Smoothness <span className="mono">λ = {L}</span></legend>
            <div className="seg">
              {meta.lams.map((v, i) => (
                <label key={v}><input type="radio" name="lam" checked={v === L} onChange={() => go(r.seed, v, G)} />{SMOOTH[i]}</label>
              ))}
            </div>
            <p>Penalises a big change in sound between one song and the next.</p>
          </fieldset>
          <fieldset className="dial">
            <legend>Pull <span className="mono">γ = {G}</span></legend>
            <div className="seg">
              {meta.gammas.map((v, i) => (
                <label key={v}><input type="radio" name="gamma" checked={v === G} onChange={() => go(r.seed, L, v)} />{PULL[i]}</label>
              ))}
            </div>
            <p>Keeps the route near songs that share listeners with the seed. Strong pull often never reaches the long tail.</p>
          </fieldset>
          {nRoutes > 1 && (
            <fieldset className="dial">
              <legend>Route <span className="mono">{A + 1} of {nRoutes}</span></legend>
              <div className="seg" style={{ gridTemplateColumns: `repeat(${nRoutes}, 1fr)` }}>
                {Array.from({ length: nRoutes }, (_, k) => (
                  <label key={k}><input type="radio" name="alt" checked={k === A} onChange={() => go(r.seed, L, G, k)} />
                    {["Best", "2nd", "3rd"][k] ?? `${k + 1}th`}</label>
                ))}
              </div>
              <p>{spell(nRoutes).replace(/^\w/, (c) => c.toUpperCase())} routes from this song that share no stop. If one misses, try another.</p>
            </fieldset>
          )}
        </section>

        <section className="a-stops" aria-labelledby="stops-h">
        <h2 id="stops-h" className="label stops-h">The route, stop by stop</h2>
        <ol className="stops" aria-label="Stops on this route">
          {ids.map((id, i) => (
            <StopRow key={`${i}-${id}`} n={i} t={tracks[id]} step={i ? steps[i - 1] : null} prev={i ? tracks[ids[i - 1]] : null}
              tracks={tracks} meta={data.meta} open={selected === i} onToggle={() => setSelected(selected === i ? -1 : i)}
              playing={player.current === id && player.playing} onPlay={() => player.toggle(id)} gamma={G}
              now={nowIdx === i} />
          ))}
        </ol>
        </section>

        <Obvious r={r} tracks={tracks} steps={steps} />
        <Picker routes={routes} tracks={tracks} current={r.seed} onPick={pick} defaults={meta.default} />
      </div>
    </main>
  );
}

function StopRow({ n, t, step, prev, tracks, meta, open, onToggle, playing, onPlay, gamma, now }: {
  n: number; t: Track; step: Step | null; prev: Track | null; tracks: Tracks; meta: RoutesFile["meta"];
  open: boolean; onToggle: () => void; playing: boolean; onPlay: () => void; gamma: number; now: boolean;
}) {
  const mb = `https://musicbrainz.org/recording/${t.mbid}`;
  const id = `why-${n}`;
  return (
    <li className={`stop-row${open ? " selected" : ""}${now ? " now" : ""}`} aria-current={now || undefined}>
      <div className="stop-main">
        <span className="stop-n" aria-hidden="true">{n === 0 ? "◦" : n}</span>
        <Cover t={t} size={44} />
        <button className="stop-title" onClick={onToggle} aria-expanded={open} aria-controls={id}>
          <span className="t">{n === 0 ? <span className="sr-only">Seed: </span> : <span className="sr-only">Step {n}: </span>}{t.title}</span>
          <span className="a">{t.artist}</span>
        </button>
        <span className="stop-reach mono">{int(t.listeners)}<span className="z">{t.tail ? "long tail" : "head"}</span></span>
        {t.preview
          ? <button className="icon-btn" onClick={onPlay} aria-pressed={playing} aria-label={`${playing ? "Pause" : "Play"} preview of ${t.title} by ${t.artist}`}><Icon name={playing ? "pause" : "play"} /></button>
          : <a className="icon-btn ext" href={mb} target="_blank" rel="noreferrer" aria-label={`${t.title} on MusicBrainz (no preview found)`} title="No preview found. Open on MusicBrainz">MB</a>}
      </div>
      {open && (
        <div className="why" id={id}>
          <dl>
            {step && prev && (<>
              <dt>Why here</dt>
              <dd>{stepEvidence(step, prev)} {step.rank === 1 ? "The single most common next song." : `Number ${step.rank} of the ${meta.fanout} likeliest next songs.`}</dd>
              <dt>After the cutoff</dt>
              <dd>{step.heldOut ? <>A listener made this same step between {testWindow(meta)}, after the model was frozen.</> : <>Not seen between {testWindow(meta)}. That window is only {spell(windowDays(meta))} days.</>}</dd>
              <dt>Sound</dt>
              <dd>
                <b>{soundChange(step)}</b>
                {prev.audio && t.audio ? <AudioCompare a={prev} b={t} /> : <span> No AcousticBrainz analysis for one of the two songs, so the step counts as typical.</span>}
              </dd>
            </>)}
            {!step && t.audio && (<>
              <dt>Sound</dt>
              <dd><AudioCompare a={null} b={t} /></dd>
            </>)}
            <dt>Reach</dt>
            <dd>{reach(t, meta)}, {t.tail ? "in the long tail" : "in the head"}.</dd>
            {step && step.alts.length > 0 && (<>
              <dt>Other ways on</dt>
              <dd>
                <ul className="alts">
                  {step.alts.map((a) => (
                    <li key={a.id}>{tracks[a.id].title} <span className="muted">· {tracks[a.id].artist} · {plural(a.count, "play")}</span></li>
                  ))}
                </ul>
              </dd>
            </>)}
            <dt>Links</dt>
            <dd className="ext">
              <a href={mb} target="_blank" rel="noreferrer">MusicBrainz ↗</a>
              {t.preview && <>{"  ·  "}<a href={t.preview.link} target="_blank" rel="noreferrer">Apple Music ↗</a></>}
              {t.preview && !t.preview.exact && <span>{"  ·  "}preview is “{t.preview.as}”</span>}
            </dd>
          </dl>
          {step && (
            <p className="terms mono">
              score = log p {Math.log(step.p).toFixed(2)}{gamma ? ` + γ·closeness ${gamma}×${step.rel.toFixed(2)}` : ""}{" "}
              − λ·sound {step.audio.toFixed(2)}
            </p>
          )}
        </div>
      )}
    </li>
  );
}

function AudioCompare({ a, b }: { a: Track | null; b: Track }) {
  const rows: [string, (t: Track) => string][] = [
    ["Tempo", (t) => `${t.audio!.bpm} bpm`],
    ["Key", (t) => t.audio!.key],
    ["Danceability", (t) => t.audio!.dance.toFixed(2)],
    ["Onsets / s", (t) => t.audio!.onsets.toFixed(1)],
  ];
  return (
    <table className="cmp">
      <tbody>
        {rows.map(([k, f]) => (
          <tr key={k}><th scope="row">{k}</th>{a && <td>{f(a)}</td>}{a && <td aria-hidden="true">→</td>}<td>{f(b)}</td></tr>
        ))}
      </tbody>
    </table>
  );
}

/** Album art from the matched preview, or an empty frame (never a stand-in picture). */
function Cover({ t, size, className = "art" }: { t: Track; size: number; className?: string }) {
  const src = t.preview?.art;
  return src
    ? <img className={className} src={src} alt="" width={size} height={size} loading="lazy" decoding="async" />
    : <span className={className} aria-hidden="true" />;
}

function PlayButton({ id, t, small }: { id: number; t: Track; small?: boolean }) {
  const player = usePlayer();
  const on = player.current === id && player.playing;
  if (!t.preview) {
    return <a className={`icon-btn ext${small ? " sm" : ""}`} href={`https://musicbrainz.org/recording/${t.mbid}`} target="_blank" rel="noreferrer"
      aria-label={`${t.title} on MusicBrainz (no preview found)`} title="No preview found. Open on MusicBrainz">MB</a>;
  }
  return (
    <button className={`icon-btn${small ? " sm" : ""}`} onClick={() => player.toggle(id)} aria-pressed={on}
      aria-label={`${on ? "Pause" : "Play"} preview of ${t.title} by ${t.artist}`}>
      <Icon name={on ? "pause" : "play"} size={small ? 15 : 18} />
    </button>
  );
}

function MiniRow({ id, t }: { id: number; t: Track }) {
  return (
    <li className="mini-row">
      <Cover t={t} size={36} className="art sm" />
      <span className="who">{t.title} <span>· {t.artist}</span></span>
      <span className="mono soft">{int(t.listeners)}</span>
      <PlayButton id={id} t={t} small />
    </li>
  );
}

function Obvious({ r, tracks, steps }: { r: SeedRoutes; tracks: Tracks; steps: Step[] }) {
  const ob = r.obvious.map((i) => tracks[i]);
  const mOb = median(ob.map((t) => t.listeners));
  const mRoute = median(steps.map((s) => tracks[s.to].listeners));
  const sameArtist = ob.filter((t) => t.artist === tracks[r.seed].artist).length;
  return (
    <section className="side-block a-obvious" aria-labelledby="obvious-h">
      <h2 id="obvious-h">What a standard recommender returns</h2>
      <p>
        The benchmark’s tuned ItemKNN, for a listener who has played only this song. Median {int(mOb)} listeners
        against the route’s {int(mRoute)}{sameArtist ? `; ${sameArtist} of 5 by the same artist` : ""}.
      </p>
      <ul className="mini">
        {r.obvious.map((id) => <MiniRow key={id} id={id} t={tracks[id]} />)}
      </ul>
    </section>
  );
}

function Picker({ routes, tracks, current, onPick, defaults }: {
  routes: SeedRoutes[]; tracks: Tracks; current: number; onPick: (s: number, play: boolean) => void; defaults: [number, number];
}) {
  const [q, setQ] = useState("");
  const list = useMemo(() => searchSeeds(routes, tracks, q), [routes, tracks, q]);
  const groups: [string, SeedRoutes[]][] = q
    ? [["Matches", list]]
    : [["From India", list.filter((x) => x.group === "india")], ["Most played on ListenBrainz", list.filter((x) => x.group === "top")]];
  const listRef = useRef<HTMLDivElement>(null);
  return (
    <section className="side-block a-picker" id="pick" aria-labelledby="pick-h">
      <h2 id="pick-h">Start from another song</h2>
      <p>{routes.length} seeds: the most played recordings with one per artist, and a few from India.</p>
      <label className="sr-only" htmlFor="seed-search">Search seeds by song or artist</label>
      <input id="seed-search" className="search" type="search" placeholder="Search song or artist" value={q}
        onChange={(e) => setQ(e.target.value)} autoComplete="off" />
      <div className="seeds" ref={listRef}>
        {groups.map(([name, items]) => items.length > 0 && (
          <div key={name}>
            <h3>{name}</h3>
            {name === "From India" && <p className="note">ListenBrainz has few listeners in India, so these routes rest on fewer plays.</p>}
            {items.map((x) => {
              const t = tracks[x.seed];
              const ids = stopIds(x, routeFor(x, ...defaults));
              const heard = ids.filter((i) => tracks[i].preview).length;
              return (
                <div key={x.seed} className="seed-row">
                  <button className="seed-pick" aria-current={x.seed === current} onClick={() => onPick(x.seed, false)}>
                    <Cover t={t} size={32} className="art xs" />
                    <span className="who">{t.title} <span>· {t.artist}</span>
                      {heard < ids.length && <span className="pv"> · {heard} of {ids.length} playable</span>}</span>
                    <span className="mono muted">{int(t.listeners)}</span>
                  </button>
                  {heard > 0 && (
                    <button className="icon-btn sm" onClick={() => onPick(x.seed, true)} aria-label={`Play the route from ${t.title} by ${t.artist}`}>
                      <Icon name="play" size={15} />
                    </button>
                  )}
                </div>
              );
            })}
          </div>
        ))}
        {!list.length && <p className="note" style={{ marginTop: 12 }}>No seed matches “{q}”. Seeds are limited to songs with a full route at every setting.</p>}
      </div>
    </section>
  );
}

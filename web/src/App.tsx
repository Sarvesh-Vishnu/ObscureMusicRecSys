import { useEffect, useState } from "react";
import { parseHash, type Lab, type RoutesFile, type Tracks, type View } from "./data";
import { Mark } from "./icons";
import { LabView } from "./LabView";
import { PlayerProvider } from "./player";
import { RouteView } from "./RouteView";

interface Loaded { routes: RoutesFile; tracks: Tracks; lab: Lab }

async function load(): Promise<Loaded> {
  const get = async <T,>(f: string): Promise<T> => {
    const r = await fetch(`${import.meta.env.BASE_URL}data/${f}`);
    if (!r.ok) throw new Error(`${f}: ${r.status}`);
    return r.json() as Promise<T>;
  };
  const [routes, tracks, lab] = await Promise.all([get<RoutesFile>("routes.json"), get<Tracks>("tracks.json"), get<Lab>("lab.json")]);
  return { routes, tracks, lab };
}

function useHashView(): View {
  const [v, setV] = useState(() => parseHash(location.hash));
  useEffect(() => {
    const on = () => setV(parseHash(location.hash));
    addEventListener("hashchange", on);
    return () => removeEventListener("hashchange", on);
  }, []);
  return v;
}

export function App() {
  const [data, setData] = useState<Loaded | null>(null);
  const [error, setError] = useState<string | null>(null);
  const view = useHashView();
  useEffect(() => { load().then(setData, (e) => setError(String(e))); }, []);

  return (
    <>
      <a className="skip" href="#content" onClick={(e) => { e.preventDefault(); document.getElementById("content")?.focus(); }}>Skip to content</a>
      <header className="top">
        <div className="wrap">
          <a className="mark" href="#/"><Mark />Obscure</a>
          <nav className="nav" aria-label="Views">
            <a href="#/" aria-current={view.name === "route" ? "page" : undefined}>Route</a>
            <a href="#/lab" aria-current={view.name === "lab" ? "page" : undefined}>Lab</a>
            <a href="https://github.com/Sarvesh-Vishnu/ObscureMusicRecSys">Code ↗</a>
          </nav>
        </div>
      </header>
      <div id="content" tabIndex={-1}>
        {error && <div className="wrap failed" role="alert">The data didn’t load ({error}). Reload to try again.</div>}
        {!data && !error && <div className="wrap loading" aria-live="polite">Loading routes…</div>}
        {data && (
          <PlayerProvider tracks={data.tracks}>
            {view.name === "route"
              ? <RouteView data={data.routes} tracks={data.tracks} seed={view.seed} lam={view.lam} gamma={view.gamma} />
              : <LabView lab={data.lab} routes={data.routes} tracks={data.tracks} section={view.section} />}
            <Footer />
          </PlayerProvider>
        )}
      </div>
    </>
  );
}

function Footer() {
  return (
    <footer className="foot">
      <div className="wrap">
        <p>
          Listening data: <a href="https://listenbrainz.org">ListenBrainz</a> and <a href="https://musicbrainz.org">MusicBrainz</a>, CC0.
          Audio analysis: <a href="https://acousticbrainz.org">AcousticBrainz</a>, CC0. Previews and artwork: Apple Music, linked on every song.
        </p>
        <p>
          No cookies, no analytics. Built by <a href="https://sarvesh-vishnu.github.io">Sarvesh Vishnu</a>.
        </p>
      </div>
    </footer>
  );
}

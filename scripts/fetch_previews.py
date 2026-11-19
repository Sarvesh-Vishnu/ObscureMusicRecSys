"""Match the web app's tracks to 30-second iTunes previews (run after export_web.py).

    python scripts/fetch_previews.py               # ~20 lookups a minute (Apple's limit); cached
    python scripts/fetch_previews.py --merge-only  # write what the cache already holds, no lookups
    python scripts/fetch_previews.py --fallback    # then retry the misses through the artist's catalogue

Only tracks a visitor can play are looked up: seeds, route stops and the "obvious"
picks. A match needs the same lead artist and the same title once tags such as
("Remastered 2011", "feat. ...") are set aside, and must not be a different version
(live, acoustic, remix...) unless ours is too; anything else stays unmatched and
the app links to MusicBrainz instead. Results are cached in data/itunes_cache.json
so a rerun only asks about new tracks.

Apple's search often leaves an artist's best-known recording out of its results
("Green Day American Idiot" returns covers and other Green Day songs). --fallback
retries each miss through the artist instead: find the artist whose name is the
lead artist exactly, list up to 200 of their songs, and apply the same match. It
tries the US store, then the Indian one. Its results go to data/itunes_fallback.json.
"""

from __future__ import annotations

import json
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "web" / "public" / "data"
CACHE = ROOT / "data" / "itunes_cache.json"
FALLBACK = ROOT / "data" / "itunes_fallback.json"
API = "https://itunes.apple.com/search"
LOOKUP = "https://itunes.apple.com/lookup"
STORES = ["US", "IN"]
PAUSE = 3.2                                   # seconds between requests
MATCHER = 2                                   # bump when the matching rules change, to re-check approximate matches once
HEADERS = {"User-Agent": "obscure-web-build/1.0"}


# MusicBrainz uses typographic punctuation ("Don’t", "a‐ha"); Apple mostly doesn't
TYPOGRAPHIC = str.maketrans({"‘": "'", "’": "'", "‛": "'", "′": "'", "“": '"', "”": '"',
                             "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "―": "-", "−": "-",
                             "…": "...", "\u00a0": " "})


def plain(s: str) -> str:
    return s.translate(TYPOGRAPHIC)


def norm(s: str) -> str:
    s = re.sub(r"(\w)'(\w)", r"\1\2", plain(s))          # don't -> dont, on both sides
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = s.lower().replace("&", " and ")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def core_title(s: str) -> str:
    s = re.sub(r"\s*[\(\[][^\)\]]*[\)\]]", "", s)          # (Remastered), [Live]
    s = re.split(r"\s+[-‐–—]\s+", s)[0]                    # "Song - 2011 Remaster"
    s = re.split(r"\s+(feat\.?|ft\.?|featuring)\s+", s, flags=re.I)[0]
    return norm(s)


def lead_artist(s: str) -> str:
    s = re.split(r"\s*(,|&| and | x | feat\.?| ft\.?| featuring | with )\s*", s, flags=re.I)[0]
    return norm(s)


VERSION = re.compile(r"\b(acoustic|live|remix|mix|instrumental|demo|karaoke|cover|tribute|unplugged|rehearsal|"
                     r"slowed|sped up|reverb|8d|lofi|lo fi|orchestral|piano version|re recorded|taylor s version)\b")


def versions(title: str) -> set[str]:
    return set(VERSION.findall(norm(title)))


def match(track: dict, results: list[dict]) -> dict | None:
    want_t, want_core, want_a = norm(track["title"]), core_title(track["title"]), lead_artist(track["artist"])
    best, best_score = None, 0
    for r in results:
        if not r.get("previewUrl"):
            continue
        a = norm(r.get("artistName", ""))
        if not want_a or not (want_a in a or lead_artist(r.get("artistName", "")) == want_a):
            continue
        t = r.get("trackName", "")
        if versions(t) - versions(track["title"]):       # a different version of the song
            continue
        score = 2 if norm(t) == want_t else 1 if core_title(t) == want_core and want_core else 0
        if score > best_score:
            best, best_score = r, score
    if best is None:
        return None
    art = best.get("artworkUrl100", "")
    return {"url": best["previewUrl"], "art": art.replace("100x100bb", "300x300bb") if art else "",
            "link": best.get("trackViewUrl", ""), "exact": best_score == 2, "v": MATCHER,
            "as": f"{best.get('artistName', '')} — {best.get('trackName', '')}"}


def lookup(track: dict) -> list[dict]:
    return get(API, {"term": f"{lead_artist(plain(track['artist']))} {core_title(plain(track['title']))}",
                     "media": "music", "entity": "song", "limit": 25, "country": "US"})


def get(url: str, params: dict) -> list[dict]:
    q = urllib.parse.urlencode(params)
    for attempt in range(5):
        try:
            req = urllib.request.Request(f"{url}?{q}", headers=HEADERS)
            with urllib.request.urlopen(req, timeout=20) as resp:
                return json.loads(resp.read().decode("utf-8")).get("results", [])
        except urllib.error.HTTPError as e:
            if e.code in (403, 429, 503):
                time.sleep(60 * (attempt + 1))
                continue
            raise
        except (urllib.error.URLError, TimeoutError):
            time.sleep(10)
    raise RuntimeError("iTunes kept refusing; try again later")


def artist_ids(name: str, store: str) -> list[int]:
    """Apple artists whose name is exactly this lead artist (after normalising)."""
    want = norm(name)
    found = get(API, {"term": name, "entity": "musicArtist", "limit": 10, "country": store})
    time.sleep(PAUSE)
    return [a["artistId"] for a in found if norm(a.get("artistName", "")) == want][:2]


def catalogue(aid: int, store: str) -> list[dict]:
    songs = get(LOOKUP, {"id": aid, "entity": "song", "limit": 200, "country": store})
    time.sleep(PAUSE)
    return [x for x in songs if x.get("wrapperType") == "track"]


def fallback(todo: list[dict], found: dict) -> None:
    """Retry misses through each lead artist's catalogue, one artist at a time."""
    by_artist: dict[str, list[dict]] = {}
    for t in todo:
        by_artist.setdefault(lead_artist(plain(t["artist"])), []).append(t)
    print(f"fallback: {len(todo)} misses across {len(by_artist)} artists", flush=True)
    for n, (name, ts) in enumerate(by_artist.items(), 1):
        left = list(ts)
        for store in STORES:
            if not left or not name:
                break
            for aid in artist_ids(name, store):
                songs = catalogue(aid, store)
                for t in list(left):
                    m = match(t, songs)
                    if m:
                        found[t["mbid"]] = {**m, "store": store}
                        left.remove(t)
        for t in left:
            found[t["mbid"]] = None
        if n % 10 == 0 or n == len(by_artist):
            FALLBACK.write_text(json.dumps(found, ensure_ascii=False))
            print(f"  {n}/{len(by_artist)} artists  recovered {sum(v is not None for v in found.values())}", flush=True)


def main() -> int:
    routes = json.loads((DATA / "routes.json").read_text())
    tracks = json.loads((DATA / "tracks.json").read_text())
    need: set[int] = set()
    for r in routes["routes"]:
        need.add(r["seed"])
        need.update(r["obvious"])
        for steps in r["grid"].values():
            need.update(s["to"] for s in steps)
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    # re-check approximate matches (the rules may have tightened since they were cached)
    def stale(t: dict) -> bool:
        if t["mbid"] not in cache:
            return True
        hit = cache[t["mbid"]]
        if hit is None:      # misses with typographic punctuation predate plain()
            return plain(t["artist"] + t["title"]) != t["artist"] + t["title"]
        return hit.get("exact") is False and hit.get("v") != MATCHER
    todo = [i for i in sorted(need) if stale(tracks[str(i)])]
    if "--merge-only" in sys.argv or "--fallback" in sys.argv:
        todo = []
    print(f"{len(need)} playable tracks, {len(todo)} to look up (~{len(todo) * PAUSE / 60:.0f} min)", flush=True)
    for n, i in enumerate(todo, 1):
        t = tracks[str(i)]
        cache[t["mbid"]] = match(t, lookup(t))
        if n % 25 == 0 or n == len(todo):
            CACHE.write_text(json.dumps(cache, ensure_ascii=False))
            hits = sum(v is not None for v in cache.values())
            print(f"  {n}/{len(todo)}  matched so far {hits}/{len(cache)}", flush=True)
        time.sleep(PAUSE)
    found = json.loads(FALLBACK.read_text()) if FALLBACK.exists() else {}
    if "--fallback" in sys.argv:
        misses = [tracks[str(i)] for i in sorted(need)
                  if cache.get(tracks[str(i)]["mbid"]) is None and tracks[str(i)]["mbid"] not in found]
        fallback(misses, found)
    for i in need:
        t = tracks[str(i)]
        m = cache.get(t["mbid"]) or found.get(t["mbid"])
        if m:
            t["preview"] = m
        else:
            t.pop("preview", None)
    (DATA / "tracks.json").write_text(json.dumps(tracks, separators=(",", ":"), ensure_ascii=False))
    hit = sum("preview" in tracks[str(i)] for i in need)
    print(f"previews for {hit}/{len(need)} playable tracks ({hit / len(need):.0%})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

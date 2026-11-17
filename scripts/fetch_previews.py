"""Match the web app's tracks to 30-second iTunes previews (run after export_web.py).

    python scripts/fetch_previews.py               # ~20 lookups a minute (Apple's limit); cached
    python scripts/fetch_previews.py --merge-only  # write what the cache already holds, no lookups

Only tracks a visitor can play are looked up: seeds, route stops and the "obvious"
picks. A match needs the same lead artist and the same title once tags such as
("Remastered 2011", "feat. ...") are set aside, and must not be a different version
(live, acoustic, remix...) unless ours is too; anything else stays unmatched and
the app links to MusicBrainz instead. Results are cached in data/itunes_cache.json
so a rerun only asks about new tracks.
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
API = "https://itunes.apple.com/search"
PAUSE = 3.2                                   # seconds between requests
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
            "link": best.get("trackViewUrl", ""), "exact": best_score == 2,
            "as": f"{best.get('artistName', '')} — {best.get('trackName', '')}"}


def lookup(track: dict) -> list[dict]:
    q = urllib.parse.urlencode({"term": f"{lead_artist(plain(track['artist']))} {core_title(plain(track['title']))}",
                                "media": "music", "entity": "song", "limit": 25, "country": "US"})
    for attempt in range(5):
        try:
            req = urllib.request.Request(f"{API}?{q}", headers=HEADERS)
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
        return hit.get("exact") is False
    todo = [i for i in sorted(need) if stale(tracks[str(i)])]
    if "--merge-only" in sys.argv:
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
    for i in need:
        t = tracks[str(i)]
        m = cache.get(t["mbid"])
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

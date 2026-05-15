"""Grow the catalog by walking MusicBrainz from existing seed artists.

Writes directly into the SQLite catalog (data/catalog.db). Run
`python scripts/build_index.py` after to refresh embeddings.

Usage:
    python scripts/ingest_musicbrainz.py --per-artist 5

MusicBrainz is CC0, no API key required. We respect their 1 req/sec policy.
Last.fm tag enrichment runs if LASTFM_API_KEY is configured.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import musicbrainzngs
import requests

from obscure import catalog as cat
from obscure import config


def _mb_setup() -> None:
    mb_cfg = config.musicbrainz()
    musicbrainzngs.set_useragent(mb_cfg.app_name, mb_cfg.app_version, mb_cfg.contact)


def _lastfm_tags(artist: str, track: str) -> list[str]:
    cfg = config.lastfm()
    if not cfg.enabled:
        return []
    try:
        r = requests.get(
            "https://ws.audioscrobbler.com/2.0/",
            params={"method": "track.getTopTags", "artist": artist, "track": track,
                    "api_key": cfg.api_key, "format": "json"},
            timeout=10,
        )
        return [t["name"].lower() for t in r.json().get("toptags", {}).get("tag", [])[:8]]
    except Exception:
        return []


def _lastfm_listeners(artist: str, track: str) -> int | None:
    cfg = config.lastfm()
    if not cfg.enabled:
        return None
    try:
        r = requests.get(
            "https://ws.audioscrobbler.com/2.0/",
            params={"method": "track.getInfo", "artist": artist, "track": track,
                    "api_key": cfg.api_key, "format": "json"},
            timeout=10,
        )
        n = r.json().get("track", {}).get("listeners")
        return int(n) if n is not None else None
    except Exception:
        return None


def _expand_artist(artist: str, per_artist: int) -> Iterable[cat.Track]:
    try:
        ar = musicbrainzngs.search_artists(artist=artist, limit=1)
        artists = ar.get("artist-list", [])
        if not artists:
            return
        artist_mbid = artists[0]["id"]
        time.sleep(1.0)

        recs = musicbrainzngs.browse_recordings(artist=artist_mbid, limit=per_artist)
        time.sleep(1.0)

        for rec in recs.get("recording-list", [])[:per_artist]:
            title = rec.get("title", "").strip()
            if not title:
                continue
            year_raw = rec.get("first-release-date") or ""
            try:
                year = int(year_raw[:4]) if year_raw else None
            except ValueError:
                year = None

            tags = _lastfm_tags(artist, title)
            listeners = _lastfm_listeners(artist, title)

            yield cat.Track(
                track_id=f"mb_{rec['id'][:12]}",
                title=title,
                artist=artist,
                tags=tags,
                description="",
                listener_count=listeners,
                year=year,
                mbid=rec["id"],
                source=cat.SOURCE_CRAWL,
            )
            time.sleep(1.0)
    except musicbrainzngs.WebServiceError as exc:
        print(f"  MusicBrainz error for {artist}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-artist", type=int, default=3,
                        help="Recordings to pull per seed artist")
    parser.add_argument("--artists", type=str, default="",
                        help="Comma-separated list of artists to crawl. "
                             "Defaults to all artists already in catalog.")
    args = parser.parse_args()

    _mb_setup()
    catalog = cat.open_catalog()

    if args.artists:
        artists = [a.strip() for a in args.artists.split(",") if a.strip()]
    else:
        all_tracks = catalog.all_tracks()
        artists = sorted({t.artist for t in all_tracks})

    print(f"Crawling {len(artists)} artists, up to {args.per_artist} tracks each")

    new_count = 0
    for i, artist in enumerate(artists, 1):
        print(f"[{i}/{len(artists)}] {artist}")
        for t in _expand_artist(artist, args.per_artist):
            if catalog.exists(t.title, t.artist):
                continue
            catalog.upsert(t)
            new_count += 1

    print(f"\nAdded {new_count} new tracks. Total: {catalog.count()}")


if __name__ == "__main__":
    main()

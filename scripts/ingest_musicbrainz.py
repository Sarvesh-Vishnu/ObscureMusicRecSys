"""Grow the catalog by walking MusicBrainz from existing seed tracks.

Usage:
    python scripts/ingest_musicbrainz.py --depth 1 --per-artist 5

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
            params={
                "method": "track.getTopTags",
                "artist": artist, "track": track,
                "api_key": cfg.api_key, "format": "json",
            },
            timeout=10,
        )
        tags = r.json().get("toptags", {}).get("tag", [])
        return [t["name"].lower() for t in tags[:8]]
    except Exception:
        return []


def _lastfm_listeners(artist: str, track: str) -> int | None:
    cfg = config.lastfm()
    if not cfg.enabled:
        return None
    try:
        r = requests.get(
            "https://ws.audioscrobbler.com/2.0/",
            params={
                "method": "track.getInfo",
                "artist": artist, "track": track,
                "api_key": cfg.api_key, "format": "json",
            },
            timeout=10,
        )
        n = r.json().get("track", {}).get("listeners")
        return int(n) if n is not None else None
    except Exception:
        return None


def _expand_artist(artist: str, per_artist: int) -> Iterable[cat.Track]:
    """Pull recordings for an artist; emit Track rows."""
    try:
        # find artist
        ar = musicbrainzngs.search_artists(artist=artist, limit=1)
        artists = ar.get("artist-list", [])
        if not artists:
            return
        artist_mbid = artists[0]["id"]
        time.sleep(1.0)

        # works by that artist
        recs = musicbrainzngs.browse_recordings(artist=artist_mbid, limit=per_artist)
        time.sleep(1.0)

        for i, rec in enumerate(recs.get("recording-list", [])[:per_artist]):
            title = rec.get("title", "").strip()
            if not title:
                continue
            year_raw = rec.get("first-release-date") or ""
            year = None
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
            )
            time.sleep(1.0)
    except musicbrainzngs.WebServiceError as exc:
        print(f"  MusicBrainz error for {artist}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-artist", type=int, default=3,
                        help="Recordings to pull per seed artist")
    parser.add_argument("--out", default=str(config.CATALOG_PATH),
                        help="Output parquet path")
    args = parser.parse_args()

    _mb_setup()

    seed_tracks = cat.load_seed()
    seed_artists = sorted({t.artist for t in seed_tracks})
    print(f"Seed: {len(seed_tracks)} tracks across {len(seed_artists)} artists")

    grown: list[cat.Track] = list(seed_tracks)
    seen_keys = {(t.title.lower(), t.artist.lower()) for t in grown}

    for i, artist in enumerate(seed_artists, 1):
        print(f"[{i}/{len(seed_artists)}] {artist}")
        for t in _expand_artist(artist, args.per_artist):
            key = (t.title.lower(), t.artist.lower())
            if key in seen_keys:
                continue
            seen_keys.add(key)
            grown.append(t)

    cat.save_parquet(grown, path=args.out)
    print(f"Wrote {len(grown)} tracks to {args.out}")


if __name__ == "__main__":
    main()

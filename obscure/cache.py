"""Warm-cache layer: look up tracks by title/artist; on cache miss, fetch from
MusicBrainz (and optionally enrich with Last.fm), embed, and persist.

This is what makes the catalog feel "all encompassing" — the hot core stays
small, but anything the user types is one MusicBrainz call away from being
recommendable on the next query.

Rate limits respected: MusicBrainz allows 1 req/s.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Optional

from . import catalog as cat
from . import config


@dataclass
class CacheResult:
    track: cat.Track
    hit: bool          # True if found in local catalog, False if just fetched
    source: str        # "catalog" | "musicbrainz" | "not_found"


def _stable_id(title: str, artist: str) -> str:
    h = hashlib.sha1(f"{title.lower()}::{artist.lower()}".encode()).hexdigest()[:12]
    return f"cache_{h}"


def lookup_or_fetch(title: str, artist: str,
                    catalog: Optional[cat.Catalog] = None,
                    enrich_lastfm: bool = True) -> Optional[CacheResult]:
    """Find (title, artist) in the catalog; if missing, fetch from MusicBrainz.

    Returns None if MusicBrainz has no match. On success, the track is persisted
    into the catalog with source='cache' and embedded into the FAISS index on
    the next index rebuild.

    `enrich_lastfm` is best-effort — silently skipped if no API key configured.
    """
    catalog = catalog or cat.open_catalog()

    # 1. exact match in local catalog
    if catalog.exists(title, artist):
        for t in catalog.by_text(f"{title} {artist}", limit=5):
            if t.title.lower() == title.lower() and t.artist.lower() == artist.lower():
                return CacheResult(track=t, hit=True, source="catalog")

    # 2. fall through to MusicBrainz
    track = _fetch_musicbrainz(title, artist, enrich=enrich_lastfm)
    if track is None:
        return None

    track.source = cat.SOURCE_CACHE
    catalog.upsert(track)
    return CacheResult(track=track, hit=False, source="musicbrainz")


def _fetch_musicbrainz(title: str, artist: str, enrich: bool = True) -> Optional[cat.Track]:
    try:
        import musicbrainzngs
    except ImportError:
        return None

    mb_cfg = config.musicbrainz()
    musicbrainzngs.set_useragent(mb_cfg.app_name, mb_cfg.app_version, mb_cfg.contact)

    try:
        res = musicbrainzngs.search_recordings(recording=title, artist=artist, limit=3)
        time.sleep(1.0)  # MB rate limit
    except Exception:
        return None

    recs = res.get("recording-list", []) or []
    if not recs:
        return None

    # pick the first recording with a release date if possible
    rec = next((r for r in recs if r.get("first-release-date")), recs[0])
    mbid = rec.get("id", "")
    year_raw = rec.get("first-release-date", "")[:4]
    try:
        year = int(year_raw) if year_raw else None
    except ValueError:
        year = None

    tags: list[str] = []
    listener_count: Optional[int] = None
    if enrich:
        tags = _lastfm_tags(artist, title)
        listener_count = _lastfm_listeners(artist, title)

    return cat.Track(
        track_id=_stable_id(title, artist),
        title=title,
        artist=artist,
        tags=tags,
        description="",
        listener_count=listener_count,
        year=year,
        mbid=mbid,
        source=cat.SOURCE_CACHE,
    )


def _lastfm_tags(artist: str, track: str) -> list[str]:
    cfg = config.lastfm()
    if not cfg.enabled:
        return []
    import requests
    try:
        r = requests.get(
            "https://ws.audioscrobbler.com/2.0/",
            params={"method": "track.getTopTags", "artist": artist, "track": track,
                    "api_key": cfg.api_key, "format": "json"},
            timeout=8,
        )
        return [t["name"].lower() for t in r.json().get("toptags", {}).get("tag", [])[:8]]
    except Exception:
        return []


def _lastfm_listeners(artist: str, track: str) -> Optional[int]:
    cfg = config.lastfm()
    if not cfg.enabled:
        return None
    import requests
    try:
        r = requests.get(
            "https://ws.audioscrobbler.com/2.0/",
            params={"method": "track.getInfo", "artist": artist, "track": track,
                    "api_key": cfg.api_key, "format": "json"},
            timeout=8,
        )
        n = r.json().get("track", {}).get("listeners")
        return int(n) if n else None
    except Exception:
        return None

"""Spotify as a display-only layer: album art + open-in-Spotify links.

This module deliberately avoids the deprecated /recommendations, audio-features,
and 30-second-preview endpoints. We only ask Spotify for things that still work
post-Nov-2024: search by title+artist to grab a track URL and an image.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from . import config


@dataclass
class SpotifyLink:
    url: str
    image_url: str = ""


_sp_cache: dict[str, Optional[SpotifyLink]] = {}


def _client():
    cfg = config.spotify()
    if not cfg.enabled:
        return None
    try:
        import spotipy
        from spotipy.oauth2 import SpotifyClientCredentials
        return spotipy.Spotify(auth_manager=SpotifyClientCredentials(
            client_id=cfg.client_id, client_secret=cfg.client_secret,
        ))
    except Exception:
        return None


def lookup(title: str, artist: str) -> Optional[SpotifyLink]:
    """Best-effort: find a Spotify link for `title` + `artist`. None on failure."""
    key = f"{title}::{artist}".lower()
    if key in _sp_cache:
        return _sp_cache[key]

    sp = _client()
    if sp is None:
        _sp_cache[key] = None
        return None

    try:
        q = f'track:"{title}" artist:"{artist}"'
        res = sp.search(q=q, type="track", limit=1)
        items = res.get("tracks", {}).get("items", [])
        if not items:
            _sp_cache[key] = None
            return None
        t = items[0]
        link = SpotifyLink(
            url=t.get("external_urls", {}).get("spotify", ""),
            image_url=(t.get("album", {}).get("images") or [{}])[0].get("url", ""),
        )
        _sp_cache[key] = link
        return link
    except Exception:
        _sp_cache[key] = None
        return None

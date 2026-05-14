"""Aggregated 'go listen to this' links.

Spotify is the primary surface when available (album art + direct link), but a
lot of long-tail catalog — old qawwali, Hindustani classical bootlegs, regional
cinema — simply isn't on Spotify. YouTube almost always is. We always emit a
YouTube search URL so the user can listen *something* even when Spotify misses.
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import quote_plus

from .catalog import Track
from . import spotify_display


@dataclass
class ListenLinks:
    youtube_search: str
    spotify_url: str = ""
    spotify_art: str = ""


def for_track(track: Track) -> ListenLinks:
    yt_query = quote_plus(f"{track.title} {track.artist}")
    yt = f"https://www.youtube.com/results?search_query={yt_query}"

    sp = spotify_display.lookup(track.title, track.artist)
    if sp is None:
        return ListenLinks(youtube_search=yt)
    return ListenLinks(youtube_search=yt, spotify_url=sp.url, spotify_art=sp.image_url)

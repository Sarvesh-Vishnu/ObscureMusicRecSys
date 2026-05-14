"""Secrets and runtime config.

Loads from Streamlit's `st.secrets` when running inside Streamlit, and from
environment variables otherwise. Never hard-code credentials.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
INDEX_PATH = DATA_DIR / "catalog.faiss"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"
CATALOG_PATH = DATA_DIR / "catalog.parquet"
SEED_CATALOG_PATH = DATA_DIR / "seed_catalog.json"


def _secrets() -> dict[str, Any]:
    try:
        import streamlit as st  # noqa: WPS433
        return dict(st.secrets)
    except Exception:
        return {}


def _get(section: str, key: str, env_var: str, default: str = "") -> str:
    sec = _secrets().get(section, {})
    if isinstance(sec, dict) and sec.get(key):
        return str(sec[key])
    return os.environ.get(env_var, default)


@dataclass(frozen=True)
class SpotifyConfig:
    client_id: str
    client_secret: str

    @property
    def enabled(self) -> bool:
        return bool(self.client_id and self.client_secret)


@dataclass(frozen=True)
class LastfmConfig:
    api_key: str

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)


@dataclass(frozen=True)
class OllamaConfig:
    host: str
    model: str


@dataclass(frozen=True)
class MusicBrainzConfig:
    app_name: str
    app_version: str
    contact: str


def spotify() -> SpotifyConfig:
    return SpotifyConfig(
        client_id=_get("spotify", "client_id", "SPOTIFY_CLIENT_ID"),
        client_secret=_get("spotify", "client_secret", "SPOTIFY_CLIENT_SECRET"),
    )


def lastfm() -> LastfmConfig:
    return LastfmConfig(api_key=_get("lastfm", "api_key", "LASTFM_API_KEY"))


def ollama() -> OllamaConfig:
    return OllamaConfig(
        host=_get("ollama", "host", "OLLAMA_HOST", "http://localhost:11434"),
        model=_get("ollama", "model", "OLLAMA_MODEL", "gemma3:4b"),
    )


def musicbrainz() -> MusicBrainzConfig:
    return MusicBrainzConfig(
        app_name=_get("musicbrainz", "app_name", "MB_APP_NAME", "ObscureMusicRecSys"),
        app_version=_get("musicbrainz", "app_version", "MB_APP_VERSION", "0.2.0"),
        contact=_get("musicbrainz", "contact", "MB_CONTACT", "anonymous@example.com"),
    )

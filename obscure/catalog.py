"""Catalog: tracks + tags + listener-count signal.

A `Track` is the unit of recommendation. The catalog is small and CC0-friendly:
we pull from MusicBrainz (no key) and optionally enrich with Last.fm tags. A
hand-curated seed catalog ships with the repo so the app is demo-able out of
the box.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd

from . import config


@dataclass
class Track:
    track_id: str
    title: str
    artist: str
    tags: list[str] = field(default_factory=list)
    description: str = ""
    listener_count: int | None = None
    year: int | None = None
    mbid: str = ""
    spotify_id: str = ""

    def tag_text(self) -> str:
        """Free-text representation used for embedding."""
        tag_part = ", ".join(self.tags) if self.tags else ""
        desc = self.description.strip()
        bits = [f"{self.title} by {self.artist}"]
        if tag_part:
            bits.append(f"tags: {tag_part}")
        if desc:
            bits.append(desc)
        if self.year:
            bits.append(f"year {self.year}")
        return ". ".join(bits)


def load_seed(path: Path = config.SEED_CATALOG_PATH) -> list[Track]:
    raw = json.loads(Path(path).read_text())
    return [Track(**row) for row in raw]


def to_dataframe(tracks: Iterable[Track]) -> pd.DataFrame:
    return pd.DataFrame([asdict(t) for t in tracks])


def from_dataframe(df: pd.DataFrame) -> list[Track]:
    return [Track(**{k: v for k, v in row.items() if k in Track.__dataclass_fields__})
            for row in df.to_dict(orient="records")]


def save_parquet(tracks: Iterable[Track], path: Path = config.CATALOG_PATH) -> None:
    df = to_dataframe(tracks)
    # tags is a list[str]; parquet handles it, but be safe across engines.
    df["tags"] = df["tags"].apply(list)
    df.to_parquet(path, index=False)


def load_parquet(path: Path = config.CATALOG_PATH) -> list[Track]:
    df = pd.read_parquet(path)
    df["tags"] = df["tags"].apply(list)
    return from_dataframe(df)


def obscurity_score(track: Track) -> float:
    """0.0 = mainstream, 1.0 = deeply obscure.

    Uses Last.fm-style listener count when available; falls back to 0.5 for
    unknown (NaN, None, or non-positive). Curve is log-scaled because listener
    counts span ~10^1 to ~10^7.
    """
    import math
    n = track.listener_count
    if n is None:
        return 0.5
    # NaN survives `is None` and `<= 0` checks; gate it explicitly.
    try:
        if math.isnan(float(n)):
            return 0.5
    except (TypeError, ValueError):
        return 0.5
    if n <= 0:
        return 0.5
    # ~10 listeners ~= 1.0, ~10M listeners ~= 0.0
    return max(0.0, min(1.0, 1.0 - (math.log10(n) - 1.0) / 6.0))

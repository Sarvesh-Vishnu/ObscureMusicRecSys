"""Catalog: tracks + tags + listener-count signal.

A `Track` is the unit of recommendation. The canonical store is SQLite
(`data/catalog.db`) with a FTS5 virtual table for keyword search over title,
artist, tags, and description. The seed JSON (`data/seed_catalog.json`) is the
bootstrap source; once migrated, all reads/writes go through `Catalog`.

The store is intentionally small even at scale because we keep listener_count
+ year as integers and tags as a JSON string. Combined with sentence-transformer
embeddings stored separately in FAISS (PQ-compressible for >10K tracks), 100K
tracks fits comfortably in <20 MB.
"""

from __future__ import annotations

import json
import math
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Optional

from . import config

DB_PATH = config.DATA_DIR / "catalog.db"

# Track source — useful for "is this in our curated core or pulled from cache?"
SOURCE_SEED = "seed"
SOURCE_CRAWL = "crawl"
SOURCE_CACHE = "cache"


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
    source: str = SOURCE_SEED

    def tag_text(self) -> str:
        """Free-text representation used for embedding."""
        bits = [f"{self.title} by {self.artist}"]
        if self.tags:
            bits.append("tags: " + ", ".join(self.tags))
        desc = self.description.strip()
        if desc:
            bits.append(desc)
        if self.year:
            bits.append(f"year {self.year}")
        return ". ".join(bits)


# --------------------------------------------------------------------- SQLite

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tracks (
    track_id        TEXT PRIMARY KEY,
    title           TEXT NOT NULL,
    artist          TEXT NOT NULL,
    tags            TEXT NOT NULL DEFAULT '[]',  -- JSON array
    description     TEXT NOT NULL DEFAULT '',
    listener_count  INTEGER,
    year            INTEGER,
    mbid            TEXT NOT NULL DEFAULT '',
    spotify_id      TEXT NOT NULL DEFAULT '',
    source          TEXT NOT NULL DEFAULT 'seed',
    created_at      REAL NOT NULL DEFAULT (strftime('%s','now'))
);

CREATE INDEX IF NOT EXISTS ix_tracks_artist ON tracks(artist);
CREATE INDEX IF NOT EXISTS ix_tracks_listener_count ON tracks(listener_count);

CREATE VIRTUAL TABLE IF NOT EXISTS tracks_fts USING fts5(
    title, artist, tags_text, description,
    content='tracks', content_rowid='rowid',
    tokenize='unicode61 remove_diacritics 2'
);

-- Sync triggers so FTS stays consistent
CREATE TRIGGER IF NOT EXISTS tracks_ai AFTER INSERT ON tracks BEGIN
    INSERT INTO tracks_fts(rowid, title, artist, tags_text, description)
    VALUES (new.rowid, new.title, new.artist,
            replace(replace(replace(new.tags, '[', ''), ']', ''), '"', ''),
            new.description);
END;
CREATE TRIGGER IF NOT EXISTS tracks_ad AFTER DELETE ON tracks BEGIN
    INSERT INTO tracks_fts(tracks_fts, rowid, title, artist, tags_text, description)
    VALUES('delete', old.rowid, old.title, old.artist,
           replace(replace(replace(old.tags, '[', ''), ']', ''), '"', ''),
           old.description);
END;
CREATE TRIGGER IF NOT EXISTS tracks_au AFTER UPDATE ON tracks BEGIN
    INSERT INTO tracks_fts(tracks_fts, rowid, title, artist, tags_text, description)
    VALUES('delete', old.rowid, old.title, old.artist,
           replace(replace(replace(old.tags, '[', ''), ']', ''), '"', ''),
           old.description);
    INSERT INTO tracks_fts(rowid, title, artist, tags_text, description)
    VALUES (new.rowid, new.title, new.artist,
            replace(replace(replace(new.tags, '[', ''), ']', ''), '"', ''),
            new.description);
END;
"""


def _row_to_track(row: sqlite3.Row) -> Track:
    return Track(
        track_id=row["track_id"],
        title=row["title"],
        artist=row["artist"],
        tags=json.loads(row["tags"] or "[]"),
        description=row["description"] or "",
        listener_count=row["listener_count"],
        year=row["year"],
        mbid=row["mbid"] or "",
        spotify_id=row["spotify_id"] or "",
        source=row["source"] or SOURCE_SEED,
    )


class Catalog:
    """SQLite-backed track store with FTS5 keyword search."""

    def __init__(self, path: Path = DB_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with self._conn() as c:
            c.executescript(_SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ---- read ---------------------------------------------------------
    def all_tracks(self) -> list[Track]:
        """Return every track in stable insertion order."""
        with self._conn() as c:
            rows = c.execute("SELECT * FROM tracks ORDER BY rowid").fetchall()
        return [_row_to_track(r) for r in rows]

    def by_id(self, track_id: str) -> Optional[Track]:
        with self._conn() as c:
            row = c.execute("SELECT * FROM tracks WHERE track_id = ?",
                            (track_id,)).fetchone()
        return _row_to_track(row) if row else None

    def by_text(self, query: str, limit: int = 20) -> list[Track]:
        """FTS5 search across title, artist, tags, description.

        Empty / unsafe query returns []. Uses prefix match for short queries.
        """
        q = query.strip()
        if not q:
            return []
        # escape FTS5 special characters by quoting each term
        safe = " ".join(f'"{w}"' for w in q.split() if w)
        with self._conn() as c:
            rows = c.execute(
                """
                SELECT tracks.* FROM tracks_fts
                JOIN tracks ON tracks.rowid = tracks_fts.rowid
                WHERE tracks_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (safe, limit),
            ).fetchall()
        return [_row_to_track(r) for r in rows]

    def count(self) -> int:
        with self._conn() as c:
            return c.execute("SELECT COUNT(*) FROM tracks").fetchone()[0]

    def count_by_source(self) -> dict[str, int]:
        with self._conn() as c:
            rows = c.execute("SELECT source, COUNT(*) FROM tracks GROUP BY source").fetchall()
        return {r[0]: r[1] for r in rows}

    # ---- write --------------------------------------------------------
    def upsert(self, track: Track) -> bool:
        """Insert or update by track_id. Returns True if new row inserted."""
        with self._conn() as c:
            existing = c.execute(
                "SELECT 1 FROM tracks WHERE track_id = ?", (track.track_id,)
            ).fetchone()
            if existing:
                c.execute(
                    """UPDATE tracks SET title=?, artist=?, tags=?, description=?,
                       listener_count=?, year=?, mbid=?, spotify_id=?, source=?
                       WHERE track_id=?""",
                    (track.title, track.artist, json.dumps(track.tags),
                     track.description, track.listener_count, track.year,
                     track.mbid, track.spotify_id, track.source, track.track_id),
                )
                return False
            c.execute(
                """INSERT INTO tracks (track_id, title, artist, tags, description,
                   listener_count, year, mbid, spotify_id, source)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (track.track_id, track.title, track.artist, json.dumps(track.tags),
                 track.description, track.listener_count, track.year,
                 track.mbid, track.spotify_id, track.source),
            )
            return True

    def upsert_many(self, tracks: Iterable[Track]) -> tuple[int, int]:
        """Returns (inserted, updated)."""
        ins = upd = 0
        for t in tracks:
            if self.upsert(t):
                ins += 1
            else:
                upd += 1
        return ins, upd

    def exists(self, title: str, artist: str) -> bool:
        with self._conn() as c:
            row = c.execute(
                "SELECT 1 FROM tracks WHERE lower(title)=lower(?) AND lower(artist)=lower(?)",
                (title, artist),
            ).fetchone()
        return row is not None


# --------------------------------------------------------------------- helpers

def load_seed(path: Path = config.SEED_CATALOG_PATH) -> list[Track]:
    raw = json.loads(Path(path).read_text())
    return [Track(**{**row, "source": SOURCE_SEED}) for row in raw]


def open_catalog(path: Path = DB_PATH) -> Catalog:
    """Open the canonical SQLite catalog. Bootstrap from seed JSON if empty."""
    cat = Catalog(path)
    if cat.count() == 0 and config.SEED_CATALOG_PATH.exists():
        cat.upsert_many(load_seed())
    return cat


# --------------------------------------------------------------------- legacy

def load_parquet(path: Path = config.CATALOG_PATH) -> list[Track]:
    """Back-compat: read the old parquet store. Used by migration script only."""
    import pandas as pd
    df = pd.read_parquet(path)
    df["tags"] = df["tags"].apply(list)
    return [Track(**{k: v for k, v in row.items()
                     if k in Track.__dataclass_fields__})
            for row in df.to_dict(orient="records")]


def to_dataframe(tracks: Iterable[Track]):
    """Back-compat for callers that still expect a DataFrame."""
    import pandas as pd
    return pd.DataFrame([asdict(t) for t in tracks])


# --------------------------------------------------------------------- scoring

def obscurity_score(track: Track) -> float:
    """0.0 = mainstream, 1.0 = deeply obscure.

    Uses Last.fm-style listener count when available; falls back to 0.5 for
    unknown (NaN, None, or non-positive). Log-scaled because listener counts
    span ~10^1 to ~10^7.
    """
    n = track.listener_count
    if n is None:
        return 0.5
    try:
        if math.isnan(float(n)):
            return 0.5
    except (TypeError, ValueError):
        return 0.5
    if n <= 0:
        return 0.5
    return max(0.0, min(1.0, 1.0 - (math.log10(n) - 1.0) / 6.0))

"""Fetch MusicBrainz's canonical-recording redirect table.

The same song often exists as several MusicBrainz recordings (album track,
single, remaster, compilation). MetaBrainz publishes a redirect from each
recording to one canonical recording; ListenBrainz uses it for the same
purpose. Merging on it stops one song's listeners being split across ids and
stops a re-listen to the "single version" counting as a discovery.

Source: https://data.metabrainz.org/pub/musicbrainz/canonical_data/ (CC0 1.0)
Output: data/canonical_redirect.parquet (recording_mbid -> canonical_recording_mbid)

Only the redirect CSV is extracted from the ~2.4 GB archive; the archive is
deleted afterwards.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

import duckdb
import zstandard

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "canonical"
OUT = ROOT / "data" / "canonical_redirect.parquet"
BASE = "https://data.metabrainz.org/pub/musicbrainz/canonical_data/"
UA = {"User-Agent": "ObscureMusicRecSys/0.3 (research; github.com/Sarvesh-Vishnu/ObscureMusicRecSys)"}


def _get(url: str) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        return r.read()


def main() -> int:
    dumps = sorted(set(re.findall(r'href="(musicbrainz-canonical-dump-[\d-]+)/"', _get(BASE).decode())))
    dump = dumps[-1]
    name = f"{dump}.tar.zst"
    url = f"{BASE}{dump}/{name}"
    RAW.mkdir(parents=True, exist_ok=True)
    archive = RAW / name

    if not archive.exists():
        print(f"downloading {name} ...", flush=True)
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r, \
                open(archive, "wb") as f:
            shutil.copyfileobj(r, f, length=1 << 20)
    h = hashlib.sha256()
    with open(archive, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    if h.hexdigest() != _get(url + ".sha256").decode().split()[0]:
        archive.unlink()
        raise SystemExit("checksum mismatch")

    csv = RAW / "canonical_recording_redirect.csv"
    with open(archive, "rb") as fh:
        reader = zstandard.ZstdDecompressor().stream_reader(fh)
        with tarfile.open(fileobj=reader, mode="r|") as tf:
            for m in tf:
                if m.name.endswith("canonical_recording_redirect.csv"):
                    if m.name.startswith("/") or ".." in Path(m.name).parts:
                        raise SystemExit(f"unsafe member {m.name}")
                    with tf.extractfile(m) as src, open(csv, "wb") as dst:
                        shutil.copyfileobj(src, dst, length=1 << 20)
                    break

    duckdb.sql(f"""
        COPY (SELECT recording_mbid, canonical_recording_mbid
              FROM read_csv('{csv}', header=true)
              WHERE recording_mbid <> canonical_recording_mbid)
        TO '{OUT}' (FORMAT parquet, COMPRESSION zstd)
    """)
    n = duckdb.sql(f"SELECT count(*) FROM '{OUT}'").fetchone()[0]
    shutil.rmtree(RAW)
    print(f"{OUT.name}: {n:,} redirects from {dump}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

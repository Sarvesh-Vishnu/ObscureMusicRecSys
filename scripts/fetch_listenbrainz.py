"""Download ListenBrainz daily incremental dumps and keep a compact subset.

Source: https://data.metabrainz.org/pub/musicbrainz/listenbrainz/incremental/
License: CC0 1.0 (the COPYING file inside every dump).

Each daily "spark" dump is ~200-300 MB of parquet holding every listen
*submitted* that day. For each dump we:
  1. verify its sha256,
  2. keep rows that have a MusicBrainz recording id (the join key to
     MusicBrainz metadata and AcousticBrainz audio features),
  3. write data/listens/<window-start-date>.parquet with only the columns the
     recommender needs,
  4. delete the raw tar, so disk use stays at roughly one dump at a time.

Dumps with an empty window (ListenBrainz publishes a second, near-empty dump
most days) are skipped. Re-running skips days already converted.

    python scripts/fetch_listenbrainz.py            # every available day
    python scripts/fetch_listenbrainz.py --days 3   # newest 3 days only
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "lb"
OUT = ROOT / "data" / "listens"
BASE = "https://data.metabrainz.org/pub/musicbrainz/listenbrainz/incremental/"
UA = {"User-Agent": "ObscureMusicRecSys/0.3 (research; github.com/Sarvesh-Vishnu/ObscureMusicRecSys)"}
MIN_BYTES = 10_000_000  # the near-empty companion dumps are a few KB


def _get(url: str) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        return r.read()


def _size(url: str) -> int:
    req = urllib.request.Request(url, headers=UA, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers.get("Content-Length", 0))


def list_dumps() -> list[str]:
    html = _get(BASE).decode()
    return sorted(set(re.findall(r'href="(listenbrainz-dump-\d+-\d+-\d+-incremental)/"', html)))


def _download(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def convert(dump: str) -> Path | None:
    name = dump.replace("listenbrainz-dump", "listenbrainz-spark-dump") + ".tar"
    url = f"{BASE}{dump}/{name}"
    if _size(url) < MIN_BYTES:
        return None

    RAW.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    tar_path = RAW / name
    if not tar_path.exists():
        print(f"  downloading {name} ...", flush=True)
        _download(url, tar_path)
    expected = _get(url + ".sha256").decode().split()[0]
    if _sha256(tar_path) != expected:
        tar_path.unlink()
        raise RuntimeError(f"checksum mismatch for {name}")

    work = RAW / dump
    with tarfile.open(tar_path) as tf:
        for m in tf.getmembers():  # refuse absolute paths / traversal
            if m.name.startswith("/") or ".." in Path(m.name).parts or not (m.isfile() or m.isdir()):
                raise RuntimeError(f"unsafe member in {name}: {m.name}")
        tf.extractall(work)
    inner = next(work.iterdir())
    start = (inner / "START_TIMESTAMP").read_text().strip()[:10]
    out = OUT / f"{start}.parquet"

    duckdb.sql(f"""
        COPY (
            SELECT user_id,
                   recording_mbid,
                   artist_credit_id,
                   artist_name,
                   recording_name,
                   release_name,
                   listened_at
            FROM read_parquet('{inner}/*.parquet')
            WHERE recording_mbid IS NOT NULL
        ) TO '{out}' (FORMAT parquet, COMPRESSION zstd)
    """)
    shutil.rmtree(work)
    tar_path.unlink()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=0, help="newest N dumps only (0 = all)")
    args = ap.parse_args()

    dumps = list_dumps()
    if args.days:
        dumps = dumps[-2 * args.days:]  # ~2 dumps per day, the empty one is skipped
    OUT.mkdir(parents=True, exist_ok=True)
    state = OUT / "_done.txt"  # dumps already handled (converted or empty)
    done = set(state.read_text().split()) if state.exists() else set()

    failed = 0
    for d in dumps:
        if d in done:
            continue
        print(d, flush=True)
        try:
            out = convert(d)
        except Exception as e:  # keep going; report at the end
            print(f"  FAILED: {e}", flush=True)
            failed += 1
            continue
        if out is None:
            print("  empty window, skipped", flush=True)
        else:
            n = duckdb.sql(f"SELECT count(*) FROM '{out}'").fetchone()[0]
            print(f"  -> {out.name}: {n:,} listens with a recording MBID", flush=True)
        with open(state, "a") as f:
            f.write(d + "\n")
    print(f"done; {failed} failed", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

"""Fetch AcousticBrainz low-level audio descriptors for the recordings people listen to.

AcousticBrainz (2015-2022, CC0 1.0) ran the Essentia audio analyser over
millions of user-submitted tracks and published per-recording descriptors.
It stopped accepting submissions in 2022, so recordings first released after
that have no features: coverage is measured, not assumed.

Source: https://data.metabrainz.org/pub/musicbrainz/acousticbrainz/dumps/
        acousticbrainz-lowlevel-features-20220623/  (three archives, ~3.3 GB)
Output: data/acousticbrainz.parquet, one row per canonical recording that
        appears anywhere in data/listens/.

Each recording can have several submissions (different rips of the same
song) and several MusicBrainz ids (merged by the canonical redirect). Numeric
descriptors take the median over all of them; key and scale take the mode.
Each archive's single CSV (~2.5 GB) is extracted, reduced, then deleted, so
peak extra disk is one CSV.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

import duckdb
import zstandard

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "acousticbrainz"
LISTENS = ROOT / "data" / "listens"
REDIRECT = ROOT / "data" / "canonical_redirect.parquet"
OUT = ROOT / "data" / "acousticbrainz.parquet"
DUMP = "acousticbrainz-lowlevel-features-20220623"
BASE = f"https://data.metabrainz.org/pub/musicbrainz/acousticbrainz/dumps/{DUMP}/"
UA = {"User-Agent": "ObscureMusicRecSys/0.3 (research; github.com/Sarvesh-Vishnu/ObscureMusicRecSys)"}

# part -> (numeric columns, categorical columns)
PARTS = {
    "lowlevel": (["average_loudness", "dynamic_complexity", "mfcc_zero_mean"], []),
    "rhythm": (["bpm", "bpm_histogram_first_peak_bpm_mean", "bpm_histogram_second_peak_bpm_mean",
                "danceability", "onset_rate"], []),
    "tonal": (["tuning_frequency", "tuning_equal_tempered_deviation"], ["key_key", "key_scale"]),
}


def _get(url: str) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        return r.read()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(part: str, sums: dict[str, str]) -> Path:
    name = f"{DUMP}-{part}.tar.zst"
    archive = RAW / name
    if not archive.exists():
        print(f"downloading {name} ...", flush=True)
        with urllib.request.urlopen(urllib.request.Request(BASE + name, headers=UA), timeout=120) as r, \
                open(archive, "wb") as f:
            shutil.copyfileobj(r, f, length=1 << 20)
    if _sha256(archive) != sums[name]:
        archive.unlink()
        raise SystemExit(f"checksum mismatch: {name}")
    return archive


def extract_csv(archive: Path) -> Path:
    with open(archive, "rb") as fh:
        reader = zstandard.ZstdDecompressor().stream_reader(fh)
        with tarfile.open(fileobj=reader, mode="r|") as tf:
            for m in tf:
                if m.isfile() and m.name.endswith(".csv"):
                    if m.name.startswith("/") or ".." in Path(m.name).parts:
                        raise SystemExit(f"unsafe member {m.name}")
                    out = RAW / Path(m.name).name
                    with tf.extractfile(m) as src, open(out, "wb") as dst:
                        shutil.copyfileobj(src, dst, length=1 << 20)
                    return out
    raise SystemExit(f"no csv in {archive.name}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-archives", action="store_true", help="don't delete the .tar.zst files")
    args = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    sums_file = RAW / "sha256sums"
    if not sums_file.exists():
        sums_file.write_bytes(_get(BASE + "sha256sums"))
    sums = {ln.split()[1]: ln.split()[0] for ln in sums_file.read_text().splitlines() if ln.strip()}

    con = duckdb.connect()
    tmp = RAW / "duckdb_tmp"
    con.execute(f"SET memory_limit='3GB'; SET threads=4; SET preserve_insertion_order=false; "
                f"SET temp_directory='{tmp}'")
    # canonical id for every recording anyone listened to
    con.execute(f"""
        CREATE TABLE wanted AS
        WITH ids AS (SELECT DISTINCT recording_mbid FROM read_parquet('{LISTENS}/*.parquet'))
        SELECT DISTINCT coalesce(r.canonical_recording_mbid, i.recording_mbid) AS mbid
        FROM ids i LEFT JOIN read_parquet('{REDIRECT}') r ON r.recording_mbid = i.recording_mbid
    """)
    print(f"recordings in listens: {con.execute('SELECT count(*) FROM wanted').fetchone()[0]:,}", flush=True)

    tables = []
    for part, (num, cat) in PARTS.items():
        csv = extract_csv(fetch(part, sums))
        aggs = [f"median({c})::FLOAT AS {c}" for c in num] + [f"mode({c}) AS {c}" for c in cat]
        con.execute(f"""
            CREATE TABLE {part} AS
            WITH raw AS (
              SELECT coalesce(r.canonical_recording_mbid, a.mbid) AS mbid, {", ".join("a." + c for c in num + cat)}
              FROM read_csv('{csv}', header=true, auto_detect=true) a
              LEFT JOIN read_parquet('{REDIRECT}') r ON r.recording_mbid = a.mbid
            )
            SELECT raw.mbid, count(*) AS n_{part}, {", ".join(aggs)}
            FROM raw SEMI JOIN wanted w ON w.mbid = raw.mbid
            GROUP BY raw.mbid
        """)
        n = con.execute(f"SELECT count(*) FROM {part}").fetchone()[0]
        print(f"{part}: {n:,} listened recordings with features", flush=True)
        csv.unlink()
        tables.append(part)

    con.execute(f"""
        COPY (SELECT l.mbid, l.n_lowlevel AS submissions,
                     l.* EXCLUDE (mbid, n_lowlevel), r.* EXCLUDE (mbid, n_rhythm), t.* EXCLUDE (mbid, n_tonal)
              FROM lowlevel l JOIN rhythm r USING (mbid) JOIN tonal t USING (mbid))
        TO '{OUT}' (FORMAT parquet, COMPRESSION zstd)
    """)
    n = con.execute(f"SELECT count(*) FROM '{OUT}'").fetchone()[0]
    print(f"{OUT.name}: {n:,} recordings", flush=True)
    shutil.rmtree(tmp, ignore_errors=True)
    if not args.keep_archives:
        for part in PARTS:
            (RAW / f"{DUMP}-{part}.tar.zst").unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

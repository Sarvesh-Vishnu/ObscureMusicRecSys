"""One-shot migration: build data/catalog.db from JSON seed + (optional)
existing catalog.parquet, deduping by (title, artist).

Usage:
    python scripts/migrate_to_sqlite.py [--drop]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from obscure import catalog as cat
from obscure import config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drop", action="store_true",
                        help="Delete catalog.db first; otherwise upserts.")
    args = parser.parse_args()

    if args.drop and cat.DB_PATH.exists():
        cat.DB_PATH.unlink()
        print(f"Dropped {cat.DB_PATH}")

    catalog = cat.Catalog()

    # 1. seed JSON (always — defines the curated core)
    seed = cat.load_seed()
    seed_ins, seed_upd = catalog.upsert_many(seed)
    print(f"Seed: {len(seed)} tracks ({seed_ins} new, {seed_upd} updated)")

    # 2. crawled parquet, if present
    if config.CATALOG_PATH.exists():
        try:
            crawled = cat.load_parquet()
            # only insert ones we don't already have (seed wins on conflict)
            to_add = [t for t in crawled if not catalog.exists(t.title, t.artist)]
            # tag the source
            for t in to_add:
                t.source = cat.SOURCE_CRAWL
            ins, upd = catalog.upsert_many(to_add)
            print(f"Crawled parquet: {len(crawled)} loaded → "
                  f"{ins} new, {len(crawled) - len(to_add)} skipped (dupe)")
        except Exception as exc:
            print(f"Skipping parquet ({exc})")

    print(f"\nFinal catalog: {catalog.count()} tracks")
    by_src = catalog.count_by_source()
    for src, n in sorted(by_src.items()):
        print(f"  {src}: {n}")


if __name__ == "__main__":
    main()

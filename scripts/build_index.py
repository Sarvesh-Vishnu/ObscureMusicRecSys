"""Embed the current catalog and write the FAISS index.

Usage:
    python scripts/build_index.py            # uses catalog.parquet if present, else seed
"""

from __future__ import annotations

import argparse
from pathlib import Path

from obscure import catalog as cat
from obscure import config
from obscure import embeddings as emb


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="auto",
                        help="'auto' (parquet if exists, else seed), 'seed', or path to parquet")
    args = parser.parse_args()

    if args.source == "seed":
        tracks = cat.load_seed()
        print(f"Loaded {len(tracks)} tracks from seed")
    elif args.source == "auto":
        if config.CATALOG_PATH.exists():
            tracks = cat.load_parquet()
            print(f"Loaded {len(tracks)} tracks from {config.CATALOG_PATH.name}")
        else:
            tracks = cat.load_seed()
            cat.save_parquet(tracks)
            print(f"Loaded {len(tracks)} tracks from seed (wrote to parquet)")
    else:
        tracks = cat.load_parquet(Path(args.source))
        print(f"Loaded {len(tracks)} tracks from {args.source}")

    emb.build_index(tracks)
    print(f"Wrote index to {config.INDEX_PATH}")


if __name__ == "__main__":
    main()

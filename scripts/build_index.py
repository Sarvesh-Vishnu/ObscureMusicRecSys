"""Embed the current catalog (SQLite) and write the FAISS index.

Usage:
    python scripts/build_index.py
    python scripts/build_index.py --pq    # use product-quantized index (for >10K tracks)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from obscure import catalog as cat
from obscure import config
from obscure import embeddings as emb


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pq", action="store_true",
                        help="Use product quantization (recommended for >10K tracks).")
    args = parser.parse_args()

    catalog = cat.open_catalog()
    tracks = catalog.all_tracks()
    print(f"Loaded {len(tracks)} tracks from catalog.db")

    if args.pq and len(tracks) < 256:
        print("PQ requires >=256 training vectors; falling back to flat index.")
        args.pq = False

    emb.build_index(tracks, use_pq=args.pq)
    print(f"Wrote index to {config.INDEX_PATH} (pq={args.pq})")


if __name__ == "__main__":
    main()

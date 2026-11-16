# Obscure Music RecSys

A semantic recommender for **long-tail music** — surfaces tracks you've probably never heard, but that are sonically and contextually adjacent to something you do love.

The previous version was a thin wrapper around Spotify's `/recommendations` and `/audio-features` endpoints; both were **deprecated for new apps on 2024-11-27**, along with the 30-second `preview_url`. This version owns its own catalog, embeds it, retrieves with FAISS, and uses Spotify (if configured) only as a *display layer* for album art and links.

## Pipeline

```
free-text seed ──▶ sentence-transformer embedding (all-MiniLM-L6-v2)
                          │
                          ▼
                   FAISS k-NN over catalog
                          │
                          ▼
      relevance pool (top 3k by similarity)
                          │
                          ▼
   blend similarity × obscurity, both scaled within the pool
                          │
                          ▼
   LinUCB re-rank (off until 10 ratings, full strength at 30)
                          │
                          ▼
              one track per artist
                          │
                          ▼
   Ollama gemma3:4b narrative ("why this is a kindred pick")
                          │
                          ▼
      optional Spotify link/art (display only)
```

Two seed modes:
- **Seed by track** — pick anything in the catalog; its tags + description become the query.
- **Seed by free text** — type *"hazy reverb-soaked 1970s Turkish psych with fuzz guitar"* and the embedder handles it.

A single slider trades **similarity ↔ obscurity** (default 0.4). It only reorders the closest matches, so a higher setting surfaces the most obscure *relevant* tracks rather than obscure unrelated ones.

## Ranking guards

Three fixes keep the default results on-topic (`obscure/recommender.py`, tested in `tests/test_ranking.py`):

- **Relevance pool.** Obscurity can reorder relevant tracks but can't pull in irrelevant ones.
- **Common scale.** Raw cosine similarity spans ~0.15 across the top 80 candidates while obscurity spans ~0.5, so an unscaled blend let obscurity outweigh relevance about 2:1. Both are now min-max scaled within the pool.
- **Confidence-gated bandit.** With no feedback, LinUCB's bonus is just the length of each track's feature vector, which reshuffled results arbitrarily. It now stays off until the session has 10 ratings.

## Offline discovery benchmark

Can a model predict the recordings people go on to **discover**? Six recommenders, one leakage-free temporal split over 16 days of [ListenBrainz](https://listenbrainz.org) listens (CC0): 33,496 listeners, 219,859 recordings, 10,000 evaluated listeners, 315,152 discoveries. Protocol in [docs/PROTOCOL.md](docs/PROTOCOL.md), full table with CIs in [results/phase1.md](results/phase1.md).

| Model | Recall@20 | Tail recall@20 | Novelty (bits) | Coverage |
|---|---|---|---|---|
| RandomTail (v1's idea) | 0.02% | 0.02% | 11.0 | — |
| Popularity | 1.02% | 0.00% | 4.8 | 0.0% |
| ItemKNN | 5.34% | **2.22%** | 8.5 | 22.5% |
| EASE (top 15k items) | **5.52%** | 0.00% | 7.0 | 5.8% |
| ALS | 5.15% | 0.13% | 6.7 | 5.5% |
| BPR | 3.70% | 1.34% | 9.0 | 28.7% |

ItemKNN, EASE and ALS are close on overall recall, but they differ roughly 17× on long-tail recall. Overall accuracy alone would pick the wrong model for a discovery product. Hyperparameters are untuned defaults for now.

```bash
pip install -r requirements-lab.txt
python scripts/fetch_listenbrainz.py      # ~4 GB download, ~1.5 GB kept
python scripts/run_eval.py --rebuild      # ~20 min on an 8 GB laptop
```

## Quickstart

```bash
git clone https://github.com/Sarvesh-Vishnu/ObscureMusicRecSys.git
cd ObscureMusicRecSys

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# (optional) configure Spotify for display layer + Last.fm for richer tags
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# edit .streamlit/secrets.toml

# (optional) install Ollama and pull gemma3:4b for the narrative blurbs
# https://ollama.com
ollama pull gemma3:4b

# build the index from the seed catalog (one-time, ~30s)
python scripts/build_index.py

streamlit run app.py

# tests
pip install -r requirements-dev.txt && python -m pytest -q
```

The app boots even if Ollama / Spotify / Last.fm aren't configured — the LLM blurb falls back to a tag list, and tracks render without album art.

## Growing the catalog

The seed catalog (`data/seed_catalog.json`) ships with 106 hand-curated tracks across continents, eras, and scenes. To expand:

```bash
python scripts/ingest_musicbrainz.py --per-artist 5
python scripts/build_index.py
```

This walks every seed artist, pulls up to N recordings each from MusicBrainz (CC0, no key needed), and — if `LASTFM_API_KEY` is configured — enriches each row with Last.fm tags and listener counts. MusicBrainz is rate-limited to 1 req/s, so 50 artists × 5 tracks takes a few minutes.

## Project layout

```
obscure/
  config.py            secrets + paths (st.secrets / env vars, never hard-coded)
  catalog.py           Track dataclass, parquet I/O, obscurity scoring
  embeddings.py        sentence-transformers + FAISS index build/query
  recommender.py       k-NN + obscurity-weighted blended ranking
  narrative.py         Ollama gemma3:4b explainer (best-effort, degrades gracefully)
  spotify_display.py   optional Spotify lookup for art + open-in-Spotify links
scripts/
  ingest_musicbrainz.py
  build_index.py
data/
  seed_catalog.json    50-track starter catalog
  catalog.parquet      grown catalog (gitignored)
  catalog.faiss        FAISS index (gitignored)
app.py
```

## Why a custom catalog?

Spotify's `/recommendations` is a black box and was the engine of the obscure-ness in v1 (`target_popularity=random.randint(1,50)`). With it deprecated and the underlying audio features endpoint gone too, building on Spotify is no longer viable. MusicBrainz + Last.fm cover the same ground for metadata, are open-licensed, and let us define what "obscure" means rather than asking a closed API for it.

## Roadmap

- [ ] Larger seed (target: 5k tracks via batched MusicBrainz crawl)
- [ ] Lyrics embeddings (Genius / MusixMatch) joined alongside tag embeddings
- [ ] User-feedback loop — thumbs-up/down adjusts the obscurity slider per session
- [ ] Hybrid retrieval: BM25 over tags + dense vector search

## Credits

Inspired by the original tutorial from [Avkash Chauhan](https://github.com/avkash). Built with Streamlit, MusicBrainz, Last.fm, sentence-transformers, FAISS, and Ollama.

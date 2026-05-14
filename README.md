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
        blend  cosine similarity × obscurity prior
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

A single slider trades **similarity ↔ obscurity** (default 0.4): higher means push deeper into the long tail at the cost of closer matches.

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

# build the index from the 50-track seed catalog (one-time, ~30s)
python scripts/build_index.py

streamlit run app.py
```

The app boots even if Ollama / Spotify / Last.fm aren't configured — the LLM blurb falls back to a tag list, and tracks render without album art.

## Growing the catalog

The seed catalog (`data/seed_catalog.json`) ships with 50 hand-curated tracks across continents, eras, and scenes. To expand:

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

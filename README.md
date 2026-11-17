# Obscure: a recommender that takes you somewhere new

**[Live app](https://sarvesh-vishnu.github.io/ObscureMusicRecSys/)** · **[The Lab](https://sarvesh-vishnu.github.io/ObscureMusicRecSys/#/lab)** · [Protocol](docs/PROTOCOL.md) · [Results](results/)

Most recommenders are scored on predicting what you were going to play anyway, so they keep you where you already are. This project asks whether a recommender can take a listener somewhere new, by steps real people take, and how to measure that honestly.

The answer is a **route**: five steps from a song you know to one almost nobody plays, each step a move ListenBrainz listeners really made. Behind it is a benchmark on one leakage-free split of public listening data (16 days, 33,575 listeners, 219,876 recordings, all CC0): a ladder of recommenders, re-rankers, cold start from audio, a next-song model, and off-policy evaluation with contextual bandits. Every model is tuned on a separate validation split and scored once; every comparison carries a bootstrap interval or a paired difference.

## What it found

1. **Similar recall, very different reach.** ItemKNN and ALS are statistically tied on recall@20 (6.87% and 6.71%), but ItemKNN finds 2.1× as many long-tail discoveries. EASE finds none.
2. **Reach costs a little recall.** A mild popularity penalty trades 0.22 points of recall for 1.13 points of long-tail recall (+44%). Calibration is free; diversity (MMR) is not the same as discovery.
3. **Sound alone barely finds anything.** For recordings too rare for collaborative filtering, 13 audio descriptors reach 0.10% recall (about 5× random); the artist's other songs reach 2.62%. Sound describes a song, not who will like it.
4. **What people play next is its own signal.** Back-to-back listens find the true next song 50.7% of the time in the top 20, against 46.4% for co-listening similarity, and rank it twice as high.
5. **Routes go as deep as a popularity penalty, on steps people take.** 59% of route stops are in the long tail (a penalised list: 60%), and 2.4× as many of their steps were made again by real listeners. The cost is recall, because a route leaves the neighbourhood on purpose; a pull setting buys it back.
6. **The policies worth trying are the hardest to judge from logs.** From a popularity-leaning system's logs, a "long tail first" policy keeps an effective sample of 43 in 10,000 rounds, too few for even doubly robust estimation to separate it from the base ranker. Online, a LinUCB bandit lifts hits 13% over the base ranker, and exploring adds nothing on this data.

What offline data cannot answer is whether people enjoy where a route ends. That needs an online test.

## Web app: routes and the lab

`web/` is a static site (Vite, React, TypeScript) with two views:

- **Route.** Pick one of 133 seed songs and see a route of five steps to a song almost nobody plays,
  drawn on a listeners axis so you can watch it leave the head. Each step shows how many training
  plays made that move, whether someone made it again after the cutoff, how the sound changed
  (AcousticBrainz tempo, key, danceability), and the other ways on. Two dials (smoothness λ, pull γ)
  step through a precomputed grid; next to the route, the tuned ItemKNN's top five for contrast.
  30-second previews come from Apple's iTunes Search API, matched by lead artist and title.
- **Lab.** The benchmark below as seven figures. Every number on the page is read from
  `results/phase*.json`; "tied" and "no measurable cost" are worded from whether a paired 95%
  interval spans zero.

The browser does no modelling. Every route is computed offline by the same `find_route` the
benchmark measured, on the phase-4 next-song graph. The app has no listener, so its pull is
closeness to the seed (two hops of ItemKNN), not to a listening history, and its routes treat
"Korn feat. Skrillex" as Korn (the benchmark keys artists by credit).

```bash
python scripts/export_web.py        # web/public/data/{routes,tracks,lab}.json (needs phase 2-4 caches)
python scripts/fetch_previews.py    # iTunes previews, ~20 lookups/min, cached in data/itunes_cache.json
cd web && npm install && npm run dev   # http://localhost:5192
npm test && npm run build              # vitest checks the exported routes keep the benchmark's rules
```

## The benchmark

Can a model predict the recordings people go on to **discover**? One leakage-free temporal split over 16 days of [ListenBrainz](https://listenbrainz.org) listens (CC0), merged to MusicBrainz canonical recordings: 33,575 listeners, 219,876 recordings, 10,000 evaluated listeners, 325,376 discoveries (46% long-tail). Protocol in [docs/PROTOCOL.md](docs/PROTOCOL.md).

Every model is tuned on a validation split inside the training window, then scored once on the test window. Full tables with 95% CIs and paired differences: [results/phase2.md](results/phase2.md) (tuned) and [results/phase1.md](results/phase1.md) (untuned defaults).

| Model (tuned) | Recall@20 | Tail recall@20 | Novelty (bits) | Coverage |
|---|---|---|---|---|
| Popularity | 1.02% | 0.00% | 4.8 | 0.0% |
| ItemKNN (k=15, shrink=20) | **6.87%** | **2.54%** | 8.3 | 20.3% |
| EASE (λ=300, top 15k items) | 5.68% | 0.00% | 7.1 | 5.9% |
| ALS (512 factors, α=40) | 6.71% | 1.21% | 7.5 | 14.3% |
| BPR (128 factors) | 3.99% | 0.63% | 8.6 | 20.4% |

ItemKNN and ALS are statistically tied on recall (paired difference −0.16 pp, CI [−0.39, +0.08]), but ItemKNN finds twice as much of the long tail. Overall accuracy alone can't separate them; a discovery product should.

**Re-rankers** take ItemKNN's top 200 and choose the final 20. Each has one knob, swept to trace the trade-off instead of picking a flattering point:

| On tuned ItemKNN | Δ Recall | Δ Tail recall | Δ Novelty |
|---|---|---|---|
| Calibrated λ=0.2 (match the listener's head/mid/tail mix) | +0.01 pp (tied) | **+0.19 pp** | +0.06 bits |
| PopPenalty β=0.25 (score ÷ popularity^β) | −0.22 pp | **+1.13 pp (+44%)** | +0.81 bits |
| PopPenalty β=1 | −1.90 pp | +2.98 pp (+117%) | +2.40 bits |
| MMR λ=0.5 (penalise similarity to picks so far) | −0.77 pp | +0.09 pp (tied) | +0.13 bits |

Calibration is a free win at low strength. A mild popularity penalty buys 44% more long-tail discoveries for 3% of recall. MMR raises artist diversity (37% → 61% distinct artists) but not obscurity: diversity and discovery are different goals.

### Cold start: recordings with fewer than 10 listeners

Collaborative models can't score a recording nobody has played. Phase 3 ranks 328,681 such recordings (all with [AcousticBrainz](https://acousticbrainz.org) audio descriptors) for 10,000 listeners, and splits the result by whether the recording's artist has any other recordings in the warm catalog. Full table: [results/phase3.md](results/phase3.md).

| Content used | Recall@20 | New-artist recall | Known-artist recall |
|---|---|---|---|
| Random | 0.02% | 0.02% | 0.02% |
| 13 audio descriptors → ALS factors | 0.10% | **0.11%** | 0.07% |
| Artist: "more from artists you play" | 2.54% | 0.00% | 3.43% |
| Artist: mean ALS factor of the artist's other recordings | **2.62%** | 0.00% | **3.62%** |
| Artist + audio → ALS factors | 2.24% | 0.00% | 3.12% |

- **The artist is the cold-start signal.** Knowing the artist is 25× better than knowing the audio, and a plain average of the artist's factors beats a learned mapping (paired −0.29 pp, CI [−0.48, −0.11]).
- **Low-level audio is real but faint.** 13 summary numbers (tempo, key, loudness, danceability …) predict 1.3% of the variance in ALS factors. That gives 5× random recall, and nothing on top of the artist (−0.09 pp, CI [−0.21, +0.03]). The features do describe sound: an item's nearest audio neighbour shares its artist 60× more often than chance. They encode how a recording sounds, not who will love it.
- **New artists are the gap.** Every artist-based model scores 0% on them by construction. Only audio reaches them at all. A discovery product has to give new artists slots on purpose; ranking won't surface them.

### Sessions and routes

Listens have timestamps, so histories are sequences. Back-to-back plays under 30 minutes apart give 20.8M training transitions. Full tables: [results/phase4.md](results/phase4.md).

**Next song** (100,000 held-out transitions): a first-order transition model with an ItemKNN back-off finds the true next song in its top 20 **50.7%** of the time (MRR 28.4%). Item-to-item ItemKNN gets 46.4% (MRR 14.2%) and popularity 0.3%. The two models tie on transitions to songs new to the listener (32.3% vs 32.1%).

**Routes.** From each of a listener's 4 favourite songs, beam search takes 5 steps through the next-song graph. A route never repeats an artist, never plays anything the listener has heard, and has to end in the long tail. Two knobs: γ pulls the route toward the listener's taste, and λ penalises audible jumps.

| 20 recordings per listener (10,000 listeners) | Recall@20 | Tail share | Realism | Audio step |
|---|---|---|---|---|
| Top-20 ItemKNN | 6.87% | 23% | 15.1% | 4.08 |
| Top-20 PopPenalty β=0.5 | 6.09% | 60% | 7.0% | 4.17 |
| Route: most probable | 0.52% | 59% | 16.6% | 4.02 |
| Route: smooth (λ=1) | 0.47% | 59% | 16.4% | **3.36** |
| Route: personal γ=3 + smooth | 1.28% | 53% | 18.5% | 3.36 |
| Route: personal γ=10 + smooth | 3.15% | 34% | **21.2%** | 3.45 |

*Realism* is the share of steps that some real listener made in the held-out window. For a top-20 list, the steps are its consecutive items played in order.

- **A route goes as deep as a debiased list, by real steps.** PopPenalty reaches 60% long-tail picks, but only 7% of its consecutive pairs are moves anyone made. Routes reach 59% with 16–21% real steps, and the audio penalty makes each step 16% smoother, with realism and recall inside the unpenalised route's confidence intervals.
- **Leaving the neighbourhood costs offline recall, and the dial shows how much.** A route is built to take the listener somewhere they weren't headed, so it rarely contains what they were about to play anyway (0.5% vs 6.1%). Turning up personal pull (γ) buys recall back at the price of depth. Offline recall can only score the first goal. Whether people *enjoy* the far end of a route needs an online test.
- **One song per artist costs a third of ItemKNN's hits** (6.87% → 4.59%). Many "discoveries" are another track by an artist the listener already plays.

### Off-policy evaluation and bandits

ListenBrainz has no logs of a recommender showing songs, so phase 6 uses the supervised-to-bandit conversion (Dudík et al. 2011). Each of 10,000 test listeners has 100 ItemKNN candidates, and the reward is 1 if a candidate is one of their real discoveries. A simulated logging policy shows one candidate per listener and reveals only that reward. Every policy's true value is known exactly, so every estimate can be checked. Semi-synthetic by construction: real listeners and rewards, simulated logging. Full tables: [results/phase6.md](results/phase6.md).

| Target policy (10,000 logged rounds) | True hit rate | DR 90% range | DM 90% range | Effective sample size |
|---|---|---|---|---|
| ItemKNN top pick | 6.46% | 5.80–6.99% | 6.30–7.33% | 3,142 |
| PopPenalty β=0.5 top pick | 5.46% | 3.54–9.89% | 4.10–4.97% | 103 |
| Long tail first | 4.69% | 2.22–10.21% | 3.63–4.99% | 43 |
| ALS top pick | 6.99% | 2.52–12.85% | 6.45–9.14% | 34 |

- **The policies worth trying are the hardest to judge from logs.** A log from a popularity-leaning recommender rarely shows long-tail songs. 10,000 rounds carry the information of about 43 for a "long tail first" policy, and doubly robust estimation can't tell it from ItemKNN. The direct method is steady but can be confidently wrong: its PopPenalty range misses the truth.
- **Exploring in the log helps, at a price paid up front.** Showing a random candidate 50% of the time instead of 10% costs 22% of the logged system's hits. It cuts DR's error on "long tail first" from 2.41 to 1.83 points, still more than the 1.77-point gap to ItemKNN. *(A follow-up added after the first result.)*
- **Learning from each hit beats the base ranker; exploring adds nothing here.** Linear bandits on six features (ItemKNN and ALS scores, popularity, long-tail and known-artist flags) climb from ItemKNN's fixed 6.67% to 7.56% (LinUCB, last fifth of 50,000 rounds). That's level with the same model fitted on all the feedback at once (7.50%). A greedy learner that never explores gets 7.55%, and tuning picked the smallest exploration in every family. This is consistent with Bastani, Bayati & Khosravi (2021) on greedy bandits with diverse contexts.

## Reproduce

Everything runs on an 8 GB laptop. Downloads are cached under `data/` (gitignored).

```bash
pip install -r requirements-lab.txt
python scripts/fetch_listenbrainz.py      # ~4 GB download, ~1.5 GB kept
python scripts/fetch_canonical.py         # 2.4 GB download, 60 MB kept
python scripts/run_eval.py                # phase 1: untuned baselines, ~20 min
python scripts/run_phase2.py --stage all  # tuning + re-rankers, several hours
python scripts/fetch_acousticbrainz.py    # 3.3 GB download, 72 MB kept
python scripts/run_phase3.py --stage all  # cold start, ~25 min
python scripts/run_phase4.py --stage all  # sessions and routes, ~1.5 h
python scripts/run_phase6.py --stage all  # off-policy evaluation and bandits, ~15 min
python -m pytest -q                       # 49 tests
```

Then the web app: see [Web app](#web-app-routes-and-the-lab).

## Repository layout

```
obscure/lab/        the benchmark: data and split, metrics, models, re-rankers,
                    cold start, sequences, routes, bandits
scripts/            fetchers, one runner per phase, web export, preview matcher
results/            every table (phase*.md) and its numbers (phase*.json)
docs/PROTOCOL.md    the evaluation protocol, written before each phase's results
web/                the static app (Vite, React, TypeScript); design contract in web/design.md
tests/              pytest for the lab; vitest for the app lives in web/src
obscure/*.py, app.py   the earlier Streamlit version (see below)
```

## What's next

- An online test of routes: whether listeners keep, skip or save where a route ends.
- A sequential neural model (SASRec) against the transition baseline.
- Lyrics and tags for recordings with no listeners and no audio analysis.
- Lead-artist parsing for credits like "X reinterpreted by Y", which the app's routes can still repeat.

## Earlier versions

**v1 (2024)** asked Spotify's `/recommendations` for obscure songs by passing `target_popularity=random.randint(1, 50)`. Spotify deprecated that endpoint, `/audio-features` and the 30-second `preview_url` for new apps on 2024-11-27.

**v2 (May 2026)** owned a small catalog: MusicBrainz and Last.fm tags embedded with a sentence transformer, FAISS retrieval, a similarity-obscurity blend, a LinUCB re-ranker and an Ollama-written blurb, in Streamlit. Its code is still here (`app.py`, `obscure/*.py`) and its notes follow. The benchmark above replaced it as the project's core, because a 353-track catalog with hand-entered listener counts could not be evaluated.

### Pipeline

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

### Ranking guards

Three fixes keep the default results on-topic (`obscure/recommender.py`, tested in `tests/test_ranking.py`):

- **Relevance pool.** Obscurity can reorder relevant tracks but can't pull in irrelevant ones.
- **Common scale.** Raw cosine similarity spans ~0.15 across the top 80 candidates while obscurity spans ~0.5, so an unscaled blend let obscurity outweigh relevance about 2:1. Both are now min-max scaled within the pool.
- **Confidence-gated bandit.** With no feedback, LinUCB's bonus is just the length of each track's feature vector, which reshuffled results arbitrarily. It now stays off until the session has 10 ratings.

### Quickstart

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

### Growing the catalog

The seed catalog (`data/seed_catalog.json`) ships with 106 hand-curated tracks across continents, eras, and scenes. To expand:

```bash
python scripts/ingest_musicbrainz.py --per-artist 5
python scripts/build_index.py
```

This walks every seed artist, pulls up to N recordings each from MusicBrainz (CC0, no key needed), and — if `LASTFM_API_KEY` is configured — enriches each row with Last.fm tags and listener counts. MusicBrainz is rate-limited to 1 req/s, so 50 artists × 5 tracks takes a few minutes.

### Why a custom catalog?

Spotify's `/recommendations` is a black box and was the engine of the obscure-ness in v1 (`target_popularity=random.randint(1,50)`). With it deprecated and the underlying audio features endpoint gone too, building on Spotify is no longer viable. MusicBrainz + Last.fm cover the same ground for metadata, are open-licensed, and let us define what "obscure" means rather than asking a closed API for it.

## Credits

Listening data: [ListenBrainz](https://listenbrainz.org) and [MusicBrainz](https://musicbrainz.org), CC0. Audio analysis: [AcousticBrainz](https://acousticbrainz.org), CC0. Previews and artwork in the app: Apple Music, linked on every song. The first version followed a tutorial by [Avkash Chauhan](https://github.com/avkash).

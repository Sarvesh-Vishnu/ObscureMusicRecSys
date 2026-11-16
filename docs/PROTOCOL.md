# Offline discovery benchmark: protocol

The question every model is scored on: **given what someone has listened to, can you predict the recordings they go on to discover?**

## Data

[ListenBrainz](https://listenbrainz.org) daily incremental dumps, CC0 1.0 ([source](https://data.metabrainz.org/pub/musicbrainz/listenbrainz/incremental/)). Each dump holds every listen *submitted* that day. `scripts/fetch_listenbrainz.py` verifies each dump's sha256, keeps rows with a MusicBrainz recording id, and stores one compact parquet file per submission day in `data/listens/`.

Across the 16 days (Oct 31 – Nov 15, 2026), 46.7M of the 62.4M listens with a recording id (75%) were *played* before the day they were submitted: people bulk-import their history. The protocol uses that history but never lets it leak.

## Split

A single global cutoff, `split = 2026-11-12`:

| | Submitted | Played | Used for |
|---|---|---|---|
| Train | before the split | before the split | model fitting, including imported history |
| Test | on/after the split | inside [split, end) | targets |
| Dropped | on/after the split | before the split | nothing: a model at the cutoff could not have seen it |

A user's **discoveries** are test-window recordings absent from that user's training history. Re-listens are not targets: a discovery system is judged on what is new to the listener.

Training is filtered once: recordings need ≥ 10 distinct training listeners, then users need ≥ 10 distinct kept recordings. Evaluation samples 10,000 users (seeded) from those with at least one discovery in the kept universe.

`tests/test_lab.py::test_split_has_no_leakage` pins these rules on a synthetic dataset.

## Long tail

Head = the most-listened recordings that together account for half of all training (user, recording) pairs. Everything else is the long tail. The definition is relative to this dataset: a "head" recording here is popular *among ListenBrainz users in these 12 days*.

## Metrics (k = 20)

| Metric | Definition | Why |
|---|---|---|
| Recall@k | hits / min(\|targets\|, k) | did the list contain what they discovered |
| NDCG@k | rank-discounted hits, normalised by the ideal list | were hits near the top |
| Tail recall@k | recall over long-tail targets only (users with ≥ 1) | the thesis: can it find obscure music people actually take to |
| Novelty | mean −log₂(listeners / users) of the list | how unfamiliar the list is, in bits |
| Tail share | fraction of the list in the long tail | |
| Artist diversity | distinct artists / k | one artist ten times is not discovery |
| Coverage | distinct items recommended across all users / catalog | does the system use the catalog or a few hits |

Every per-user metric is reported as a mean with a 95% percentile bootstrap CI over users (1,000 resamples). Model comparisons against Popularity use **paired** bootstrap differences over the same users, which are much tighter than overlapping marginal CIs.

## Known limitations

- **"New to the user" means new within the data we have.** Someone may have heard a "discovery" before they joined ListenBrainz.
- **One song, several ids.** The same song can appear under several MusicBrainz recording ids (album vs. single vs. remaster). That splits its signal and makes some "discoveries" re-listens. Canonical-recording mapping is a planned fix.
- **Hyperparameters are not tuned yet.** Phase 1 uses standard defaults. Tuning on a validation split carved from the training window comes next, because untuned baselines are a well-documented way to overstate progress ([Ferrari Dacrema et al., RecSys 2019](https://doi.org/10.1145/3298689.3347058)).
- **EASE covers only the 15,000 most popular recordings.** Its dense item-by-item inverse doesn't fit in 8 GB beyond that. The cap is visible in its tail metrics, and that is the point: model choice decides what can be recommended at all.
- **Twelve days of training data.** Short histories favour popularity and recency.

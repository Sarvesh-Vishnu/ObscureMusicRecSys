# Offline discovery benchmark: protocol

The question every model is scored on: **given what someone has listened to, can you predict the recordings they go on to discover?**

## Data

[ListenBrainz](https://listenbrainz.org) daily incremental dumps, CC0 1.0 ([source](https://data.metabrainz.org/pub/musicbrainz/listenbrainz/incremental/)). Each dump holds every listen *submitted* that day. `scripts/fetch_listenbrainz.py` verifies each dump's sha256, keeps rows with a MusicBrainz recording id, and stores one compact parquet file per submission day in `data/listens/`.

Across the 16 days (Oct 31 – Nov 15, 2026), 46.7M of the 62.4M listens with a recording id (75%) were *played* before the day they were submitted: people bulk-import their history. The protocol uses that history but never lets it leak.

**Canonical recordings (from phase 2).** The same song can exist as several MusicBrainz recordings (album track, single, remaster). `scripts/fetch_canonical.py` fetches MetaBrainz's canonical-recording redirect table (CC0, 7.95M redirects), and every listen is mapped to its canonical recording before counting. In practice ListenBrainz's own matching already resolves most listens to canonical ids. Only 3.7% of listens (4.1% of distinct ids, measured on Nov 10–15) needed redirecting, and the filtered catalog barely changed (219,859 → 219,876 recordings). The merge stays as cheap insurance.

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

## Cold start (phase 3)

The warm catalog drops every recording with fewer than 10 training listeners, and collaborative models have no row for those recordings. Phase 3 asks whether *content* can recommend them.

**Audio.** [AcousticBrainz](https://acousticbrainz.org) low-level descriptors (CC0, final 2022 dump), fetched by `scripts/fetch_acousticbrainz.py`. Every submission is mapped to its canonical recording. Numeric descriptors take the median over submissions and key/scale take the mode. Each recording becomes 13 numbers:
- loudness, dynamic complexity and MFCC₀
- log tempo and two tempo-histogram peaks
- danceability and onset rate
- tuning frequency and deviation
- key on the circle of fifths (sin, cos)
- scale (±1)

Numeric descriptors are clipped to their 1st–99th percentile and z-scored on warm recordings. 1.49M of the 3.50M recordings in the listens have features. AcousticBrainz stopped in 2022, so recent releases are missing by construction.

**Cold catalog.** A recording is in the cold catalog when all three hold:
- it is outside the warm catalog;
- it has audio features;
- someone played it in the test window.

Using test-window activity for *membership* mirrors a catalog where new recordings are added before anyone plays them. It reveals that a recording exists, not who will play it. Models rank the whole cold catalog for each user. Targets are a warm user's test-window plays of cold recordings that user had not played in training.

**Slices.** A target is **known-artist** when its artist credit has at least one warm recording, otherwise **new-artist**. On new-artist targets, artist metadata says nothing and only the audio is left. Artist credits are compared as MusicBrainz artist-credit ids, so "A feat. B" is a different credit from "A".

**Content → factors.** The strongest content models regress the tuned ALS item factors (512 dimensions) from content, then score a cold recording as user factor · predicted factor ([van den Oord et al., NeurIPS 2013](https://papers.nips.cc/paper/2013/hash/b3ba8f1bee1238a2f37603d90b58898d-Abstract.html)). Inputs are the audio vector, the artist's mean warm factor, or both. The regressor is a two-layer MLP trained on warm recordings that have audio, with early stopping on 5% of them held out. When training on warm recordings, each recording's artist mean **leaves that recording out**, as it would be for a recording the model has never seen. Width and weight decay are picked on the validation split's cold set.

## Sessions and routes (phase 4)

**Transitions.** Two consecutive listens by the same person, at most 30 minutes apart and both in the warm catalog, form a transition a → b. Repeats are dropped, and so are exact duplicate listens. A listen outside the catalog breaks the chain: a, X, b gives no a → b. Training transitions come only from listens played and submitted before the split. Test transitions come from listens played and submitted inside the test window, the same rule as everything else. `tests/test_routes.py` pins the gap, catalog and split rules.

**Next song.** Given the current recording, a model ranks every other recording, and a hit means the true next one is in the top 20. Ties count as a uniformly random order. The transition model's popularity discount γ is selected on the validation split, by hit@20 on transitions to recordings new to the listener.

**Routes.** Each evaluated listener's 4 most-played recordings, one per artist, are the seeds. From each seed the search takes 5 steps along the next-song graph (the 50 best next songs per recording), for 20 new recordings in total. That is the same budget as a top-20 list. A route:
- never revisits a recording;
- never reuses an artist within itself;
- never lands on anything the listener has heard;
- must end in the long tail.

Beam search (width 10) maximises Σ log p(step), with two optional terms: a personal pull γ·rel_u (the listener's ItemKNN score, scaled 0..1) and an audio penalty λ·(audio distance ÷ median edge distance).

**Route metrics.**
- **Realism:** the share of steps that some listener actually made in the held-out window.
- **Audio step:** the mean z-scored audio distance per step, over steps with audio on both ends.

For top-20 lists the "steps" are consecutive items, as if the list were played in order. Recall is measured against the listener's test-window discoveries, exactly as in phases 1–2. That is a strict test for a route, which is built to leave the listener's current neighbourhood.

## Known limitations

- **Audio covers 39% of cold plays.** 509,770 of the 832,136 test-window plays of cold recordings by warm users fall on recordings with no AcousticBrainz features, mostly post-2022 releases. Phase 3 results are conditional on having audio.

- **"New to the user" means new within the data we have.** Someone may have heard a "discovery" before they joined ListenBrainz.
- **Phase 1 hyperparameters were untuned defaults.** Phase 2 tunes every model on a validation split carved from the training window (train before Nov 8, discoveries Nov 8–11), selects by NDCG@20, and only then scores the test window once. Untuned baselines are a well-documented way to overstate progress ([Ferrari Dacrema et al., RecSys 2019](https://doi.org/10.1145/3298689.3347058)).
- **ItemKNN similarity caps very long histories.** When computing item-item similarity only, users with more than 2,000 recordings contribute their 2,000 most-played ones. The top 1% of users (bulk-imported libraries of up to ~32k recordings) account for 95% of co-occurrence work, Σ|u|². The cap cuts that from 3.0×10¹⁰ to 2.8×10⁹ and affects 1.2% of users. Their full history is still used for scoring and for excluding already-heard items. On the validation split the capped model fits 7× faster (299 s vs 2,205 s) and is slightly *more* accurate (paired NDCG@20 +0.27 pp, CI [+0.16, +0.40]); tail recall is tied (−0.12 pp, CI [−0.28, +0.01]). Very long imported libraries add noise to co-occurrence, not signal.
- **EASE covers only the 15,000 most popular recordings.** Its dense item-by-item inverse doesn't fit in 8 GB beyond that. The cap is visible in its tail metrics, and that is the point: model choice decides what can be recommended at all.
- **Twelve days of training data.** Short histories favour popularity and recency.

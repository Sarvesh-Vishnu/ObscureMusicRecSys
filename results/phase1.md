# Discovery benchmark: phase1

ListenBrainz listens submitted 2026-10-31 to 2026-11-11 (train) and 2026-11-12 to 2026-11-15 (test). 33,496 users × 219,859 recordings, 6,637,577 training pairs. Targets are **discoveries**: test-window listens of recordings the user never played in training. 10,000 sampled users, 315,152 discoveries, 46% of them long-tail. Long tail = recordings outside the most-listened set that holds half of all training pairs (187,022 of 219,859 items).

Means with 95% bootstrap CIs over users. Recall/NDCG/tail figures in %. k = 20.

| Model | Recall@20 | NDCG@20 | Tail recall@20 | Novelty (bits) | Tail share | Artist diversity | Coverage | Fit |
|---|---|---|---|---|---|---|---|---|
| RandomTail (v1 idea) | 0.02 [0.01, 0.03] | 0.01 [0.01, 0.02] | 0.02 [0.01, 0.04] | 11.01 [11.00, 11.01] | 100% | 100% | 55.8% | 0s |
| Popularity | 1.02 [0.94, 1.09] | 0.98 [0.90, 1.05] | 0.00 [0.00, 0.00] | 4.83 [4.83, 4.83] | 0% | 55% | 0.0% | 0s |
| ItemKNN | 5.34 [5.08, 5.61] | 4.67 [4.44, 4.90] | 2.22 [2.02, 2.42] | 8.48 [8.45, 8.50] | 29% | 47% | 22.5% | 632s |
| EASE | 5.52 [5.30, 5.77] | 5.36 [5.15, 5.59] | 0.00 [0.00, 0.00] | 7.01 [7.00, 7.03] | 0% | 47% | 5.8% | 111s |
| ALS | 5.15 [4.93, 5.41] | 4.93 [4.74, 5.16] | 0.13 [0.08, 0.17] | 6.65 [6.64, 6.67] | 1% | 46% | 5.5% | 111s |
| BPR | 3.70 [3.51, 3.89] | 3.37 [3.21, 3.53] | 1.34 [1.20, 1.48] | 9.03 [9.01, 9.05] | 35% | 70% | 28.7% | 308s |

Paired difference vs Popularity (percentage points, 95% CI):

| Model | Δ Recall | Δ Tail recall |
|---|---|---|
| RandomTail (v1 idea) | -1.00 [-1.08, -0.92] | 0.02 [0.01, 0.04] |
| ItemKNN | 4.32 [4.05, 4.60] | 2.22 [2.02, 2.42] |
| EASE | 4.51 [4.28, 4.76] | 0.00 [0.00, 0.00] |
| ALS | 4.13 [3.92, 4.37] | 0.13 [0.08, 0.17] |
| BPR | 2.68 [2.49, 2.89] | 1.34 [1.20, 1.48] |

Hyperparameters: ItemKNN {'k': 100, 'shrink': 10}; EASE {'lambda': 500, 'max_items': 15000}; ALS {'factors': 128, 'reg': 0.05, 'alpha': 10, 'iterations': 15}; BPR {'factors': 128, 'lr': 0.05, 'reg': 0.001, 'iterations': 100}

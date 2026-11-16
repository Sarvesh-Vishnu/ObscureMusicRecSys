# Discovery benchmark: phase 2 (tuned models + re-rankers)

Same protocol as [phase 1](phase1.md) (see [docs/PROTOCOL.md](../docs/PROTOCOL.md)), with recordings merged to MusicBrainz canonical recordings. 33,575 users × 219,876 recordings, 6,691,340 training pairs; 10,000 evaluated users, 325,376 discoveries (46% long-tail).

Hyperparameters were chosen on a **validation split inside the training window** (train before 2026-11-08, discoveries 2026-11-08 to 11-11) by NDCG@20, then refit on the full training window. A second, finer round was run for ItemKNN, ALS and BPR because round one's best settings sat on the edge of their grids. The test window was never used for selection. Grid and every trial: [phase2_tuning.json](phase2_tuning.json).

## Tuned models

| Model | Recall@20 | NDCG@20 | Tail recall@20 | Novelty | Tail share | Artist div. | Coverage |
|---|---|---|---|---|---|---|---|
| Popularity | 1.02 [0.94, 1.11] | 0.97 [0.90, 1.04] | 0.00 [0.00, 0.00] | 4.82 | 0% | 60% | 0.0% |
| ItemKNN (tuned) | 6.87 [6.60, 7.12] | 6.18 [5.95, 6.39] | 2.54 [2.32, 2.77] | 8.27 | 23% | 37% | 20.3% |
| EASE (tuned) | 5.68 [5.43, 5.91] | 5.51 [5.29, 5.73] | 0.00 [0.00, 0.00] | 7.07 | 0% | 48% | 5.9% |
| ALS (tuned) | 6.71 [6.44, 6.96] | 6.13 [5.91, 6.36] | 1.21 [1.07, 1.37] | 7.53 | 9% | 52% | 14.3% |
| BPR (tuned) | 3.99 [3.82, 4.17] | 3.76 [3.60, 3.93] | 0.63 [0.55, 0.71] | 8.57 | 24% | 63% | 20.4% |

Selected: ItemKNN {'k': 15, 'shrink': 20, 'max_user_items': 2000}; EASE {'lambda': 300, 'max_items': 15000}; ALS {'factors': 512, 'alpha': 40, 'reg': 0.05, 'iterations': 15}; BPR {'factors': 128, 'lr': 0.01, 'reg': 0.001, 'iterations': 100}

## Re-rankers on tuned ItemKNN

Each re-ranker chooses 20 of ItemKNN's top 200 candidates. One knob each, swept.

| Model | Recall@20 | NDCG@20 | Tail recall@20 | Novelty | Tail share | Artist div. | Coverage |
|---|---|---|---|---|---|---|---|
| PopPenalty β=0.25 | 6.65 [6.38, 6.91] | 5.94 [5.70, 6.16] | 3.67 [3.41, 3.93] | 9.09 | 41% | 36% | 26.0% |
| PopPenalty β=0.5 | 6.09 [5.82, 6.34] | 5.40 [5.18, 5.61] | 4.63 [4.34, 4.92] | 9.81 | 60% | 37% | 30.6% |
| PopPenalty β=0.75 | 5.45 [5.20, 5.70] | 4.80 [4.60, 4.99] | 5.25 [4.95, 5.55] | 10.32 | 76% | 39% | 32.9% |
| PopPenalty β=1 | 4.97 [4.74, 5.21] | 4.33 [4.14, 4.54] | 5.52 [5.21, 5.83] | 10.67 | 86% | 42% | 33.6% |
| Calibrated λ=0.2 | 6.89 [6.60, 7.15] | 6.18 [5.95, 6.39] | 2.73 [2.50, 2.96] | 8.34 | 25% | 38% | 21.1% |
| Calibrated λ=0.5 | 6.83 [6.55, 7.09] | 6.08 [5.84, 6.29] | 2.94 [2.70, 3.18] | 8.45 | 29% | 39% | 22.5% |
| Calibrated λ=0.8 | 6.78 [6.51, 7.03] | 5.99 [5.76, 6.19] | 3.30 [3.05, 3.55] | 8.62 | 35% | 40% | 24.3% |
| Calibrated λ=0.95 | 6.64 [6.37, 6.89] | 5.89 [5.66, 6.09] | 3.57 [3.33, 3.83] | 8.83 | 42% | 41% | 26.4% |
| MMR λ=0.1 | 6.84 [6.56, 7.10] | 6.16 [5.94, 6.38] | 2.57 [2.35, 2.80] | 8.30 | 24% | 40% | 20.9% |
| MMR λ=0.3 | 6.70 [6.43, 6.95] | 6.07 [5.85, 6.28] | 2.64 [2.42, 2.86] | 8.35 | 25% | 48% | 22.3% |
| MMR λ=0.5 | 6.10 [5.85, 6.33] | 5.64 [5.44, 5.83] | 2.62 [2.41, 2.85] | 8.40 | 27% | 61% | 24.1% |

## Paired differences vs tuned ItemKNN

Same users, 95% bootstrap CI. Recall and tail recall in percentage points, novelty in bits.

| | Δ Recall | Δ Tail recall | Δ Novelty |
|---|---|---|---|
| Popularity | -5.85 [-6.11, -5.56] | -2.54 [-2.77, -2.32] | -3.45 [-3.47, -3.43] |
| EASE (tuned) | -1.20 [-1.41, -0.96] | -2.54 [-2.77, -2.32] | -1.21 [-1.22, -1.19] |
| ALS (tuned) | -0.16 [-0.39, 0.08] | -1.33 [-1.53, -1.12] | -0.74 [-0.76, -0.73] |
| BPR (tuned) | -2.89 [-3.11, -2.63] | -1.91 [-2.13, -1.68] | +0.30 [+0.28, +0.31] |
| PopPenalty β=0.25 | -0.22 [-0.35, -0.09] | 1.13 [1.02, 1.25] | +0.81 [+0.80, +0.83] |
| PopPenalty β=0.5 | -0.79 [-0.94, -0.62] | 2.09 [1.94, 2.26] | +1.53 [+1.51, +1.55] |
| PopPenalty β=0.75 | -1.42 [-1.62, -1.21] | 2.72 [2.53, 2.91] | +2.05 [+2.03, +2.07] |
| PopPenalty β=1 | -1.90 [-2.11, -1.67] | 2.98 [2.78, 3.18] | +2.40 [+2.37, +2.42] |
| Calibrated λ=0.2 | 0.01 [-0.03, 0.06] | 0.19 [0.14, 0.25] | +0.06 [+0.06, +0.07] |
| Calibrated λ=0.5 | -0.04 [-0.10, 0.03] | 0.41 [0.33, 0.49] | +0.18 [+0.18, +0.18] |
| Calibrated λ=0.8 | -0.09 [-0.17, -0.00] | 0.76 [0.66, 0.88] | +0.35 [+0.34, +0.36] |
| Calibrated λ=0.95 | -0.23 [-0.35, -0.12] | 1.03 [0.90, 1.17] | +0.56 [+0.55, +0.57] |
| MMR λ=0.1 | -0.03 [-0.08, 0.02] | 0.03 [-0.03, 0.09] | +0.02 [+0.02, +0.03] |
| MMR λ=0.3 | -0.17 [-0.27, -0.07] | 0.10 [0.01, 0.18] | +0.07 [+0.07, +0.08] |
| MMR λ=0.5 | -0.77 [-0.92, -0.60] | 0.09 [-0.04, 0.23] | +0.13 [+0.12, +0.14] |

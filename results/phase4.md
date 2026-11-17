# Sessions and routes: phase 4

Consecutive listens by the same person less than 30 minutes apart are transitions. Training: 20,841,386 transitions over 10,564,292 distinct pairs, both ends in the warm catalog. Test window: 1,981,121 transitions, 758,019 of them to a recording new to the listener.

## Next song

100,000 held-out transitions by warm listeners. Given the current recording, rank every other recording; a hit is the true next one in the top 20. The popularity discount γ was picked on the validation split ([phase4_tuning.json](phase4_tuning.json)): γ = 0.

| Model | Hit@20 | MRR@20 | Hit@20, new to listener | Hit@20, long-tail next song |
|---|---|---|---|---|
| Popularity | 0.34 [0.30, 0.37] | 0.07 [0.06, 0.08] | 0.41 [0.35, 0.47] | 0.00 [0.00, 0.00] |
| ItemKNN neighbours | 46.41 [46.11, 46.71] | 14.19 [14.03, 14.36] | 32.11 [31.66, 32.56] | 34.66 [34.05, 35.23] |
| Transition γ=0 | 50.68 [50.40, 50.99] | 28.39 [28.15, 28.64] | 32.31 [31.83, 32.76] | 37.09 [36.51, 37.65] |

## Routes

10,000 test listeners. Each gets 4 routes of 5 steps from their 4 most-played recordings (one per artist): 20 new recordings, the same budget as a top-20 list. A route may not revisit a recording, reuse an artist or land on anything the listener has heard, and must end in the long tail. Next-song candidates: the 50 best per recording under the tuned transition model.

- **Realism**: share of steps that some listener actually made in the held-out window. For a top-20 list the steps are consecutive items, played as a playlist; for 'one jump' they are seed → item.
- **Audio step**: mean z-scored audio distance per step (lower is smoother; median over graph edges 3.92).

| Model | Recall@20 | Tail recall | Novelty | Tail share | Artist div. | Realism | Audio step | Length |
|---|---|---|---|---|---|---|---|---|
| Top-20: ItemKNN | 6.87 [6.60, 7.12] | 2.54 [2.32, 2.77] | 8.27 | 23% | 37% | 15.14 [14.88, 15.42] | 4.08 | 20.0 |
| Top-20: ItemKNN, one per artist | 4.59 [4.39, 4.77] | 1.88 [1.72, 2.05] | 8.25 | 27% | 100% | 5.48 [5.31, 5.65] | 4.19 | 20.0 |
| Top-20: PopPenalty β=0.5 | 6.09 [5.82, 6.34] | 4.63 [4.34, 4.92] | 9.81 | 60% | 37% | 6.99 [6.80, 7.17] | 4.17 | 20.0 |
| One jump from each seed | 1.54 [1.42, 1.66] | 0.46 [0.38, 0.54] | 8.24 | 27% | 93% | 13.26 [12.96, 13.57] | 4.01 | 19.7 |
| Random walk | 0.67 [0.61, 0.75] | 0.11 [0.07, 0.16] | 8.17 | 26% | 99% | 12.06 [11.82, 12.31] | 4.00 | 19.8 |
| Greedy walk | 0.82 [0.73, 0.91] | 0.24 [0.19, 0.30] | 8.58 | 33% | 99% | 17.40 [17.11, 17.68] | 4.00 | 19.8 |
| Route: most probable | 0.52 [0.45, 0.59] | 0.20 [0.15, 0.26] | 9.56 | 59% | 100% | 16.62 [16.32, 16.91] | 4.02 | 19.8 |
| Route: smooth audio (λ=1) | 0.47 [0.41, 0.54] | 0.20 [0.14, 0.26] | 9.57 | 59% | 100% | 16.40 [16.13, 16.70] | 3.36 | 19.8 |
| Route: personal (γ=1) | 0.65 [0.57, 0.73] | 0.24 [0.19, 0.31] | 9.52 | 58% | 100% | 17.08 [16.81, 17.38] | 4.02 | 19.8 |
| Route: personal + smooth | 0.60 [0.53, 0.66] | 0.23 [0.18, 0.29] | 9.52 | 58% | 100% | 16.91 [16.63, 17.21] | 3.36 | 19.8 |
| Route: personal γ=3 + smooth | 1.28 [1.18, 1.39] | 0.40 [0.33, 0.48] | 9.25 | 53% | 99% | 18.46 [18.19, 18.76] | 3.36 | 19.8 |
| Route: personal γ=10 + smooth | 3.15 [2.98, 3.30] | 0.70 [0.60, 0.82] | 8.25 | 34% | 93% | 21.21 [20.91, 21.50] | 3.45 | 19.8 |

## Paired differences vs Top-20: PopPenalty β=0.5

Same listeners, 95% bootstrap CI. Recall, tail recall and realism in percentage points; audio step in z-units (negative = smoother).

| | Δ Recall | Δ Tail recall | Δ Realism | Δ Audio step |
|---|---|---|---|---|
| Top-20: ItemKNN | 0.79 [0.62, 0.94] | -2.09 [-2.26, -1.94] | 8.15 [7.92, 8.38] | -0.09 [-0.11, -0.07] |
| Top-20: ItemKNN, one per artist | -1.49 [-1.72, -1.26] | -2.75 [-3.00, -2.53] | -1.51 [-1.70, -1.32] | +0.03 [+0.01, +0.05] |
| One jump from each seed | -4.55 [-4.81, -4.27] | -4.17 [-4.45, -3.91] | 6.26 [6.00, 6.55] | -0.17 [-0.19, -0.15] |
| Random walk | -5.41 [-5.66, -5.14] | -4.52 [-4.80, -4.25] | 5.07 [4.83, 5.30] | -0.17 [-0.20, -0.15] |
| Greedy walk | -5.26 [-5.51, -4.98] | -4.39 [-4.67, -4.11] | 10.40 [10.14, 10.69] | -0.18 [-0.20, -0.16] |
| Route: most probable | -5.57 [-5.83, -5.31] | -4.43 [-4.71, -4.16] | 9.62 [9.33, 9.90] | -0.15 [-0.17, -0.13] |
| Route: smooth audio (λ=1) | -5.61 [-5.87, -5.35] | -4.44 [-4.72, -4.16] | 9.41 [9.13, 9.68] | -0.82 [-0.84, -0.80] |
| Route: personal (γ=1) | -5.44 [-5.70, -5.18] | -4.39 [-4.67, -4.13] | 10.09 [9.80, 10.38] | -0.15 [-0.17, -0.13] |
| Route: personal + smooth | -5.49 [-5.75, -5.23] | -4.40 [-4.68, -4.14] | 9.91 [9.64, 10.21] | -0.82 [-0.84, -0.80] |
| Route: personal γ=3 + smooth | -4.81 [-5.05, -4.55] | -4.23 [-4.50, -3.97] | 11.47 [11.18, 11.75] | -0.82 [-0.84, -0.80] |
| Route: personal γ=10 + smooth | -2.94 [-3.18, -2.67] | -3.93 [-4.19, -3.67] | 14.21 [13.92, 14.48] | -0.73 [-0.75, -0.71] |

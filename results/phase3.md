# Cold-start benchmark: phase 3

Can content recommend recordings collaborative filtering can't see? The cold catalog is every recording **outside** the warm catalog (fewer than 10 training listeners) that has AcousticBrainz audio descriptors and was played by someone in the test window: 328,681 recordings, 23% of them with zero training listeners. Each model ranks all of them for 10,000 users; targets are the 153,473 cold recordings those users went on to play (36% by artists with no warm recording at all).

Protocol and limitations: [docs/PROTOCOL.md](../docs/PROTOCOL.md). Content-model settings were picked on the validation split ([phase3_tuning.json](phase3_tuning.json)).

| Model | Recall@20 | NDCG@20 | New-artist recall | Known-artist recall | Artist div. | Coverage |
|---|---|---|---|---|---|---|
| Random | 0.02 [0.00, 0.04] | 0.01 [0.00, 0.02] | 0.02 [0.00, 0.05] | 0.02 [0.00, 0.04] | 100% | 45.6% |
| Cold popularity | 0.01 [0.00, 0.02] | 0.01 [0.00, 0.01] | 0.00 [0.00, 0.00] | 0.02 [0.01, 0.03] | 100% | 2.6% |
| Artist affinity | 2.54 [2.28, 2.77] | 1.69 [1.55, 1.85] | 0.00 [0.00, 0.00] | 3.43 [3.14, 3.74] | 12% | 13.1% |
| Audio profile | 0.03 [0.02, 0.05] | 0.03 [0.02, 0.04] | 0.03 [0.01, 0.06] | 0.03 [0.02, 0.05] | 98% | 24.8% |
| Artist centroid | 2.62 [2.40, 2.83] | 1.72 [1.57, 1.86] | 0.00 [0.00, 0.00] | 3.62 [3.30, 3.92] | 21% | 16.9% |
| Audio → factors | 0.10 [0.07, 0.14] | 0.07 [0.05, 0.09] | 0.11 [0.06, 0.17] | 0.07 [0.04, 0.10] | 96% | 8.0% |
| Artist → factors | 2.33 [2.10, 2.55] | 1.48 [1.35, 1.61] | 0.00 [0.00, 0.00] | 3.19 [2.92, 3.47] | 15% | 14.8% |
| Audio + artist → factors | 2.24 [2.02, 2.43] | 1.46 [1.33, 1.58] | 0.00 [0.00, 0.00] | 3.12 [2.83, 3.41] | 16% | 12.9% |

Held-out R² of the content → ALS-factor mapping: Audio → factors 0.013; Artist → factors 0.266; Audio + artist → factors 0.264

## Paired differences

Same users, 95% bootstrap CI, percentage points.

| A − B | Question | Δ Recall | Δ New-artist recall | Δ Known-artist recall |
|---|---|---|---|---|
| Audio + artist → factors − Artist → factors | what audio adds on top of the artist | -0.09 [-0.21, 0.03] | 0.00 [0.00, 0.00] | -0.07 [-0.24, 0.09] |
| Audio → factors − Cold popularity | audio alone vs the cold-popularity baseline | 0.09 [0.05, 0.13] | 0.11 [0.06, 0.17] | 0.05 [0.02, 0.09] |
| Audio → factors − Random | audio alone vs random | 0.08 [0.03, 0.12] | 0.09 [0.03, 0.17] | 0.05 [0.02, 0.09] |
| Audio + artist → factors − Artist affinity | best content model vs 'more from artists you play' | -0.30 [-0.52, -0.06] | 0.00 [0.00, 0.00] | -0.31 [-0.62, -0.02] |
| Artist → factors − Artist centroid | learned vs averaged artist embedding | -0.29 [-0.48, -0.11] | 0.00 [0.00, 0.00] | -0.43 [-0.66, -0.19] |

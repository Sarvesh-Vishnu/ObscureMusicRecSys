# Off-policy evaluation and bandits: phase 6

Supervised-to-bandit conversion (Dudík et al. 2011) on the phase-2 test listeners. Each of 10,000 listeners has 100 candidates (ItemKNN's top 100, nothing already heard); reward 1 if the candidate is one of their discoveries in the test window. Mean hits per listener: 2.94; 62% of listeners have at least one. Because every candidate's reward is known, each policy's true value is exact.

Logging policy: softmax over ItemKNN scores (τ = 0.1) mixed with 10% uniform. Each log draws listeners with replacement and one action each; 100 logs per size. Values are hit rates in %; RMSE in percentage points.

## Off-policy estimates at 10,000 logged rounds

| Target policy | True value | DM | IPS | SNIPS | DR | Effective sample size |
|---|---|---|---|---|---|---|
| Logging policy (softmax ItemKNN) | 5.82 | 5.77 (RMSE 0.25) | 5.77 (RMSE 0.25) | 5.77 (RMSE 0.25) | 5.78 (RMSE 0.25) | 10,000 |
| ItemKNN, top pick | 6.46 | 6.79 (RMSE 0.47) | 6.37 (RMSE 0.37) | 6.37 (RMSE 0.38) | 6.37 (RMSE 0.38) | 3,142 |
| ALS, top pick | 6.99 | 7.66 (RMSE 1.07) | 6.98 (RMSE 3.68) | 6.94 (RMSE 3.49) | 6.95 (RMSE 3.50) | 34 |
| PopPenalty β=0.5, top pick | 5.46 | 4.53 (RMSE 0.97) | 5.20 (RMSE 2.11) | 5.37 (RMSE 2.05) | 5.30 (RMSE 2.04) | 103 |
| Most popular candidate | 3.93 | 4.29 (RMSE 0.74) | 3.34 (RMSE 4.04) | 3.32 (RMSE 3.92) | 3.21 (RMSE 3.98) | 16 |
| Long tail first | 4.69 | 4.25 (RMSE 0.60) | 4.39 (RMSE 2.51) | 4.53 (RMSE 2.50) | 4.50 (RMSE 2.41) | 43 |
| Uniform over candidates | 2.94 | 3.17 (RMSE 0.46) | 2.97 (RMSE 0.45) | 2.97 (RMSE 0.46) | 2.97 (RMSE 0.46) | 1,553 |

## RMSE by log size (percentage points)

| Target policy | Estimator | 1,000 | 3,000 | 10,000 | 30,000 |
|---|---|---|---|---|---|
| Logging policy (softmax ItemKNN) | DM | 0.83 | 0.42 | 0.25 | 0.14 |
| Logging policy (softmax ItemKNN) | IPS | 0.83 | 0.42 | 0.25 | 0.14 |
| Logging policy (softmax ItemKNN) | SNIPS | 0.83 | 0.42 | 0.25 | 0.14 |
| Logging policy (softmax ItemKNN) | DR | 0.83 | 0.43 | 0.25 | 0.14 |
| ItemKNN, top pick | DM | 1.04 | 0.66 | 0.47 | 0.46 |
| ItemKNN, top pick | IPS | 1.48 | 0.70 | 0.37 | 0.25 |
| ItemKNN, top pick | SNIPS | 1.45 | 0.69 | 0.38 | 0.24 |
| ItemKNN, top pick | DR | 1.45 | 0.70 | 0.38 | 0.24 |
| ALS, top pick | DM | 1.67 | 1.05 | 1.07 | 1.08 |
| ALS, top pick | IPS | 15.36 | 10.80 | 3.68 | 2.64 |
| ALS, top pick | SNIPS | 10.85 | 8.84 | 3.49 | 2.64 |
| ALS, top pick | DR | 15.16 | 10.12 | 3.50 | 2.68 |
| PopPenalty β=0.5, top pick | DM | 1.16 | 1.04 | 0.97 | 0.91 |
| PopPenalty β=0.5, top pick | IPS | 6.94 | 3.87 | 2.11 | 1.13 |
| PopPenalty β=0.5, top pick | SNIPS | 4.56 | 3.38 | 2.05 | 1.05 |
| PopPenalty β=0.5, top pick | DR | 6.31 | 3.77 | 2.04 | 1.09 |
| Most popular candidate | DM | 1.81 | 1.04 | 0.74 | 0.49 |
| Most popular candidate | IPS | 4.08 | 6.61 | 4.04 | 2.56 |
| Most popular candidate | SNIPS | 8.32 | 5.46 | 3.92 | 2.29 |
| Most popular candidate | DR | 6.31 | 6.73 | 3.98 | 2.47 |
| Long tail first | DM | 1.24 | 0.84 | 0.60 | 0.45 |
| Long tail first | IPS | 10.00 | 6.04 | 2.51 | 1.36 |
| Long tail first | SNIPS | 6.07 | 5.90 | 2.50 | 1.44 |
| Long tail first | DR | 9.76 | 5.96 | 2.41 | 1.36 |
| Uniform over candidates | DM | 1.17 | 0.68 | 0.46 | 0.31 |
| Uniform over candidates | IPS | 1.31 | 0.75 | 0.45 | 0.24 |
| Uniform over candidates | SNIPS | 1.30 | 0.75 | 0.46 | 0.24 |
| Uniform over candidates | DR | 1.29 | 0.75 | 0.46 | 0.23 |

## Follow-up: a logging policy that explores more (added after the first result)

Same softmax, but mixed with 50% uniform instead of 10%. Logging hit rate falls from 5.82% to 4.54%. 10,000 logged rounds, 100 logs.

| Target policy | True value | DR RMSE, 10% explore | DR RMSE, 50% explore | ESS, 10% | ESS, 50% |
|---|---|---|---|---|---|
| Logging policy (softmax ItemKNN) | 5.82 | 0.25 | 0.26 | 10,000 | 6,806 |
| ItemKNN, top pick | 6.46 | 0.38 | 0.56 | 3,142 | 1,810 |
| ALS, top pick | 6.99 | 3.50 | 2.73 | 34 | 89 |
| PopPenalty β=0.5, top pick | 5.46 | 2.04 | 1.49 | 103 | 166 |
| Most popular candidate | 3.93 | 3.98 | 2.18 | 16 | 58 |
| Long tail first | 4.69 | 2.41 | 1.83 | 43 | 104 |
| Uniform over candidates | 2.94 | 0.46 | 0.20 | 1,553 | 5,771 |

## Online learning

Hyperparameters tuned on 5,000 listeners, reported on the other 5,000: ε = 0.01, α = 0.05, v = 0.05. 50,000 rounds × 10 seeds; each round a random listener from the report half. References: the learners' own linear model fitted on all the feedback at once, played greedily, 7.50%; an oracle that always finds a hit when one exists 61.94%.

| Learner | Hit rate, all rounds | Hit rate, last fifth | Hits short of the hindsight fit |
|---|---|---|---|
| Random | 2.89 ± 0.05 | 2.91 ± 0.07 | 2,303 |
| ItemKNN top pick (no learning) | 6.60 ± 0.09 | 6.67 ± 0.21 | 450 |
| Greedy (learns, never explores) | 7.51 ± 0.18 | 7.55 ± 0.28 | -7 |
| ε-greedy (ε = 0.01) | 7.39 ± 0.23 | 7.47 ± 0.27 | 56 |
| LinUCB (α = 0.05) | 7.48 ± 0.25 | 7.56 ± 0.36 | 8 |
| Thompson (v = 0.05) | 7.33 ± 0.24 | 7.53 ± 0.35 | 85 |

Features: itemknn score, itemknn rank, als score, log popularity, long tail, known artist.
Hindsight ridge coefficients: bias +0.003, itemknn score +0.001, itemknn rank -0.018, als score +0.067, log popularity +0.003, long tail +0.009, known artist +0.009.
LinUCB's coefficients after one run: bias -0.089, itemknn score -0.025, itemknn rank -0.049, als score +0.117, log popularity +0.012, long tail +0.028, known artist +0.076.

# VSLAM variant benchmark (evaluation only)

`scripts/dev/slam_bench.py`: 300 s of autonomous exploration per scenario, scored against
ground truth. `seed k` turns the start heading by k × 72°. Cells: ATE / worst error (cm), frames lost.

| variant | arena s0 | arena s1 | arena s2 | home_a s0 | heldout_b s0 | heldout_c s0 | home_a_dim s0 | mean ATE (scenarios run) |
|---|---|---|---|---|---|---|---|---|
| before round-2 loss handling | 43.9 / 158 (100) | 1.1 / 3 (4) | 14.9 / 27 (0) | 1.9 / 4 (2) | 2.1 / 7 (0) | 2.4 / 5 (1746) | 1.5 / 4 (0) | 9.7 (7) |
| current defaults | 9.2 / 31 (117) | 1.4 / 3 (53) | 14.9 / 27 (0) | 2.4 / 10 (30) | 2.1 / 7 (0) | 2.4 / 5 (1746) | 1.5 / 4 (0) | 4.9 (7) |
| no static hypothesis while driving | 20.9 / 49 (263) | 9.3 / 22 (0) | — | 1.3 / 3 (0) | 5.3 / 19 (178) | — | — | 9.2 (4) |
| command-consistency score | 9.2 / 31 (117) | 1.4 / 3 (53) | — | 2.8 / 8 (0) | 2.1 / 7 (0) | — | — | 3.9 (4) |
| stricter relocalisation (60 inliers, 3 confirmations) | 8.6 / 29 (124) | 1.4 / 3 (1618) | — | 2.1 / 7 (49) | 2.1 / 7 (0) | — | — | 3.6 (4) |
| command score + stricter relocalisation | 8.6 / 29 (124) | 1.4 / 3 (1618) | — | 2.8 / 8 (0) | 2.1 / 7 (0) | — | — | 3.7 (4) |
| local re-acquisition first | 9.2 / 31 (117) | — | — | — | — | — | — | 9.2 (1) |

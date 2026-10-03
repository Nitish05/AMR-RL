# Learned policy vs simple baselines: pre-registered comparison (round 4)

The design, metrics, exclusions and decision rule were fixed in
[policy-comparison-prereg.md](policy-comparison-prereg.md) (commit `045fbe8`, map
addendum `8a4f3d3`) before any run.

**Run**
- Evidence: `work/evidence/policy-suite-20261003-s{0..4}`; report
  `work/evidence/policy-suite-20261003-report.md`.
- Map: `map-arena-20261003-nopano-s2`.
- Code: `6219155`.
- 75 phases of 480 s.

**Exclusions.** All 15 phases from start 1 were excluded as `not_relocalized`, as
anticipated in the addendum. That leaves **12 valid cells**: 3 rule permutations ×
starts 0, 2, 3 and 4.

## Primary result

Primary metric: true panel valence at 480 s, scored from the world's own fixture
events. A tie is a difference within 1.0.

| drive-clamped learned vs | valid cells | wins | ties | losses | median difference | decision rule met |
|---|---|---|---|---|---|---|
| random | 12 | 12 | 0 | 0 | +13.5 | **yes** |
| nearest | 12 | 11 | 0 | 1 | +13.0 | **yes** |
| fixed | 12 | 11 | 0 | 1 | +13.5 | **yes** |

Under the pre-registered rule, the project may now say that **in this arena, over
these rule permutations and start poses, the learned policy chooses better than
random, nearest-first and fixed-order baselines.** The comparison is made with the
engineered drive clamped, so choice quality is compared at equal opportunity to act.

The two losses are the cells where a baseline happened to be next to the rewarding
fixture:
- standard rules, start 0: `fixed` scored 33;
- swapped rules, start 0: `nearest` scored 38.

That adjacency effect is exactly what the round-3 single-layout comparison could not
separate from choice quality.

## Secondary (reported, not used for the claim)

Means over the 12 valid cells:

| policy | true panel valence | attempts | perceived valence / attempt | red panels (max in one run) | signals into a raised panel | idle fraction |
|---|---|---|---|---|---|---|
| drive-clamped learned (ablation) | 14.0 | 40.0 | 0.38 | 1.0 (1) | 8.7 | 0.01 |
| learned (deployed) | 8.2 | 17.2 | 0.49 | 0.6 (1) | 1.8 | 0.49 |
| random | −0.3 | 29.6 | 0.05 | 1.8 (5) | 0.6 | 0.00 |
| nearest | 3.0 | 40.2 | 0.07 | 0.2 (1) | 3.1 | 0.00 |
| fixed | 0.1 | 45.5 | 0.03 | 2.7 (32) | 5.0 | 0.00 |

- **The deployed policy, idling included, also beats every baseline on total true
  panel valence:** 12/12, 11/12 and 11/12 wins (median +9.0, +9.0, +8.5). It acts in
  about half of the run. Its engineered need allows roughly
  (0.6 + 0.004 · 480) / 0.35 ≈ 7 yellows per 480 s, and it scored 8.2 on average.
- **Over the first N attempts** (N = the fewest attempts of any policy in the
  cell), mean perceived valence:

  | learned | clamped | random | nearest | fixed |
  |---|---|---|---|---|
  | 6.8 | 4.9 | 1.1 | 1.0 | 0.3 |

- **Red panels:** neither learned variant took more than one per run; `fixed` took
  32 in one run.
- **Wasted signals:** the clamped ablation sends about 9 signals per run into an
  already raised panel. It does not wait out the 8 s hold, and the deployed policy
  wastes 1.8.

## Sensitivity

In four baseline phases the robot was stuck in a forced-avoid loop for at least
200 s: `nearest` and `random`, start 4, standard and swapped rules. The robot was
0.74 m from the disliked fixture, and the retreat goal was within the navigator's
arrival tolerance. The bug is fixed in `1572b07` (regression test in
`tests/test_avoid_giveup.py`). The pre-registration allows no exclusion for this,
so the primary table includes those cells. Without them:

| drive-clamped learned vs | cells | wins | losses | median difference |
|---|---|---|---|---|
| random | 10 | 10 | 0 | +15.0 |
| nearest | 10 | 9 | 1 | +13.5 |
| fixed | 12 | 11 | 1 | +13.5 |

The decision rule is met either way.

## Limits

- One arena, one map, fixed fixture positions.
- 12 cells, with start 1 excluded because it did not relocalise against this map.
- Engineered valences.
- Fixtures are detected by saturated uniform colours.
- No significance claims at this sample size, as pre-registered.
- The baselines are simple by design; they show the learned policy is not beaten by
  trivial strategies, not that it is optimal.

# Pre-registration: turn drift, round 8 (the near-surface tail)

Committed before any round-8 candidate result was read.
- Plan: `/Users/rrnitish/.claude/plans/tingly-painting-panda.md`.
- Round 7: [turn-drift.md](turn-drift.md).

## What changed from round 7

**Diagnosis.**
- The tail of the drift is wrong depth, not edge aliasing. Wall and box faces 0.15–0.3 m away are lifted onto the floor, and the camera's 0.125 m lever arm turns that error into extra rotation.
- Most of the points involved were created during the same turn.

**Bins.**
- Bins now use the nearest *surface* (`surface_m`, walls included); round 7 binned by objects only.
- Old captures are re-binned from the world configuration (`scripts/dev/capture_common.surface_distances`).

**New metrics** (`scripts/dev/turn_bench.py`, `turn_compare.py`):
- fraction of PREDICTED frames;
- jump frames: a heading-error step above 0.7° in one frame while tracking;
- "flagged but correct" frames.

**Fresh held-out capture.** The round-7 held-out set was used once.
- `turncap-r8-{heldout_b,heldout_c,home_a_dim,home_a_textured}-heldout-20261004`, seed 2, surface bins.
- 4 extra positions per world within 0.32 m of a wall.
- It is replayed only for the frozen finalists.

## Candidates

All are `VSLAMConfig` switches, off by default:
- **C9** `c9_parallax_check` (turn / always) × `c9_unmatched` (strict / lenient): check depth by parallax before lifting a point to the floor.
- **C10** `c10_weight`: provisional weight for unvalidated points made during the turn.
- **C8b** `turn_heading_gate` with `turn_gate_deg` 0.65 and `turn_gate_frac` 0.3, or `turn_gate_mode = window`:
  - `turn_max_prediction_s` 4 s, bounded by 60° of predicted rotation and an 8° heading sigma;
  - `turn_gate_keyframes`: keyframes allowed on gated frames, together with C9.
- **C11** `floor_patch_size`: floor-band ORB patch, with `floor_feature_share` 0.7. `fs07` (the share alone) is the matching control.
- Already available from round 7: C2a `floor_validation`, C5 `turn_closure`.

## Protocol

1. **Round 1.** Every candidate alone, at most 3 settings each, on the design split (arena, home_a, arena_textured), even-numbered positions, both replay modes.
2. **Round 2.** Greedy combination from the best single candidate. Planned order: C9 → +C10 → +C8b (with keyframes allowed) → +C2a → +C5.
   - Stop when e360 p90 improves by less than 0.5°, or a non-inferiority criterion fails.
   - At most 12 runs.
3. **Round 3.** The odd-numbered positions confirm the ranking. At most 2 frozen finalists then run once on the fresh held-out capture. The slip sweep follows (arena, reset mode, slip 0.75 and 1.0).
4. **Pinned code.** Every run comes from a pinned git worktree (`work/wt/r8*`).

## Pass criteria

Held-out run, finalists, pooled over worlds; on-map and fresh-map replays judged separately.

**Primary:**
- e360 p90 ≤ 5°;
- near-surface bin (< 0.4 m) e360 p90 ≤ 8°;
- e720 p90 ≤ 7°;
- `back` end error p90 ≤ 3°.

**Non-inferiority against the paired baseline:**
- lost frames ≤ +1 pp;
- PREDICTED frames ≤ +2 pp;
- confidently wrong tracking frames ≤ 0.5 % and not above baseline;
- jump-frame rate ≤ 30 % of baseline;
- far-bin e360 median ≤ +0.5°;
- slide p90 ≤ 5 cm;
- ms p95 ≤ 1.25×.

**Robustness:** the primary criteria also hold at slip 0.75 and 1.0.

**Choice:** among passing finalists, the one with the fewest switches whose primary metric is within 1° of the best.

## Live validation (only for a passing finalist)

These are L1–L5 as in round 7: coverage-pass builds, normal builds, navigation 4 rooms × 3 seeds, the relocalisation probe, and learning runs in arena_textured. Thresholds are unchanged from [turn-drift-prereg.md](turn-drift-prereg.md).

If all pass, the finalist and `RuntimeConfig.respect_degraded_heading` become defaults. Otherwise everything stays switchable and off, and the result is documented.

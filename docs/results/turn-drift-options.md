# Turn drift: where the error really comes from, and the options (research, 2026-10-04)

After the live failure of FAm ([turn-drift.md](turn-drift.md)), three research tracks were run:
- a literature survey;
- a code feasibility study;
- an analysis of the 30 live L1 map builds.

The L1 analysis scripts are in the session scratchpad (`liveanalysis/`). This page records the conclusions.

## Where the heading error arises in live runs

Baseline, 15 builds × 420 s, of which the last 150 s is a pass of survey turns. "Growth" is the sum over segments of the absolute net change in heading error.

| class | share of growth | rate | note |
|---|---|---|---|
| in-place turns, steady | **52 %** | 0.75°/rad | mostly unbiased, random per-turn scale error (0.6–0.7°/rad) away from surfaces |
| in-place turn starts (first 0.5 s) | 16 % (~5 % above the jitter floor) | | the baseline already tracks the wheel lag correctly |
| driving (straight and arcs) | 20 % | 3.8–4.0°/m | partly *corrects* turn error |
| relocalisation jumps, loop closures | 7 % | | 4.3° per relocalisation |

**The worst cases come from where the camera faces, not from where the robot stands.**
- When the central ray hits a surface **within 0.5 m**, the estimate over-rotates with a bias of +2.6°/rad, about 20 %.
- That situation is 4.6 % of turning time but produces 35 % of the 1 s windows with more than 3° of error.
- Both of the worst runs, 43° and 18°, jump during a single survey turn sweeping a shelf 0.34–0.45 m away.
- Removing those turns alone brings the worst run from 43° to 21° and the median per-run maximum from 10.3° to 8.6°.

**Losses (16 % of time) happen while driving, not while turning.**
- 10 of the 14 "motion inconsistent with commands" losses are *false*.
- They occur within about 1 s of a step into a sharp arc. Vision was right (turn-rate error 0.01–0.03 rad/s), while the commanded turn rate differed from the true one by 0.4–0.6 rad/s, because the wheels lag and sharp arcs reach only about 0.68 of the commanded rate.

**Measured wheel response:**
- About 0.1 s of dead time, then a 0.2–0.3 s ramp.
- Steady true/commanded turn rate is about 0.83 above 0.35 rad/s and 0.74 at 0.15–0.2 rad/s.

**FAm** helped only when facing a near surface (bias +2.6 → +0.8°/rad). It was worse at turn starts, in far-view steady turns, in arcs and in losses.

## Physical limits (literature track)

- **A differential drive cannot rotate about a camera that sits ahead of the axle.** The camera's lateral velocity ω·r cannot be cancelled by v. An in-place turn already gives the least camera translation, so "rotate about the camera" is not possible.
- **Near walls the floor almost vanishes from view.** It covers about 2 % of image rows at 0.18 m, 17 % at 0.25 m and 35 % at 0.40 m. Closer than about 0.25 m, heading is nearly unobservable from the camera alone. Over a short turn only the product (1 + r·inverse depth)·θ is observable. The same lack of floor breaks the learned-depth scale fit there.
- **Replacing the VSLAM does not help.**
  - GPL or non-commercial, so excluded: ORB-SLAM3, DSO, pySLAM, SVO, MASt3R-SLAM.
  - CUDA-only: DROID-SLAM, DPVO, VGGT-SLAM.
  - stella_vslam (BSD-2) runs on CPU but does not fix the near-wall physics.

## Options

RC = removes the root cause. M = mitigates.

| # | Option | Type | Leverage on the measured error | Effort | Notes |
|---|---|---|---|---|---|
| 1 | Fix the false losses at arc starts: model wheel lag and arc slip in the motion-consistency check, or relax it for about 1 s after a command step | availability | removes ~10 of 14 baseline losses and their relocalisation jumps | low | separate from heading, but high value |
| 2 | Do not sweep the camera across near surfaces: before survey, sweep or align turns, check the guard and the map for a surface within ~0.6 m in the turn's view sector; back off, turn the other way, or skip the panorama | RC, behaviour | worst case 43 → 21°, median maximum 10.3 → 8.6° | low–medium | engineered behaviour; uses the existing guard and map |
| 3 | Re-simulate turns from logged live command sequences (`turn_capture --from-log`) and stratify by rate, start and facing distance | method | prevents another FAm | low | precondition for any new estimator change |
| 4 | Floor mask from the existing depth model: lift a pixel only if the model depth agrees with the floor depth; keep the last good scale when little floor is visible | RC, perception | removes wrong floor depths at ≥ 0.3 m | low–medium | 0.2 s/keyframe on CPU; needs an MPS option; fails closer than ~0.25 m (pairs with #2) |
| 5 | Turn-scoped estimation: no committed landmarks during a turn; track against pre-turn landmarks; inverse-depth candidates with depth priors in a sliding window | RC, estimator | targets the lock-in | medium | the current local BA runs only at keyframes, too late |
| 6 | Calibrated kinematic prior: dead time + ramp + rate-dependent slip, fitted online from vision-trusted turns; a soft prior weighted by degeneracy, not a hard gate | M | addresses the unbiased per-turn scale drift (largest RMS share) | medium | what FAm should have been |
| 7 | Absolute heading from floor-contact lines (wall–floor junctions, box footprints, Manhattan modulo 90°) as pose-graph factors | M | bounds long-term drift | medium | needs rectilinear rooms |
| 8 | Global bundle adjustment after loop closures | M | map consistency | high | the open item from the loop-closure work |
| 9 | Spec changes: camera on the axle; wider FOV; rear camera; steeper pitch | RC / M | camera on the axle removes the lever-arm effect with certainty | high | the chassis blocks the view; invalidates all maps and thresholds; owner decision |

**Not recommended:**
- rotating about the camera (impossible);
- more tuning of the hard heading gate;
- replacing the VSLAM.

## Recommended order

1. **#1 and #3.** Cheap. #1 is a standalone availability gain, and #3 is needed before any estimator change is judged.
2. **#2.** It removes the measured worst case and can be checked live directly.
3. **#4 + #5.** The perception root-cause fix for the 0.3–0.6 m regime.
4. **#6, then #7.** These target the remaining unbiased drift.
5. **#9** only as an explicit decision. A camera-on-the-axle simulation A/B is still a cheap way to confirm the diagnosis.

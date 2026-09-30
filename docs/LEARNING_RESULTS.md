# Learning results (simulation, lockstep)

Evaluated separately from navigation ([NAVIGATION_RESULTS.md](NAVIGATION_RESULTS.md)).

**Protocol** (`scripts/eval/learning.py`): every experiment starts from a FRESH
agent memory in the arena (four engineered fixtures, rules hidden from the
robot, [LEARNING.md](LEARNING.md)) using the same saved map, which the robot
built itself from onboard RGB (`map-arena-20260929-225035`; built with the
earlier mapping code, i.e. before frontier tiling and the navigation
safeguards; a map rebuilt with current code was fragmented and is not used, see
its `NOTE.md`). Every phase is a new process: autonomy starts disabled and is
enabled only after the robot has relocalized from fresh images. Test phases
after a restart use **inert** consequences so that what is measured is the
choice made from memory, not a fresh reward. One seed and one start pose per
experiment; the simulator is deterministic, so identical configurations give
identical runs (`history_a` and `learned_standard` are the same configuration
and match exactly).

Regenerate the tables: `python scripts/eval/report.py --learning work/evidence/learning-20260930-022340 work/evidence/learning-20260930-042009 --out <file>`.

## What was learned, and how it changed behaviour

* **Opposite histories → opposite choices after restart.** `history_a`
  (standard rules: signalling bloom raises a yellow panel) learned
  P(yellow | bloom, signal, no panel) = 0.90 from 8 weighted outcomes;
  `history_b` (swapped: signalling stone raises yellow, bloom does nothing)
  learned P(yellow | stone, signal) = 0.92. After a process restart (authority
  not carried over; relocalized at 0.2 s; explicit enable), the first two
  interactions were **bloom/signal ×2** for A and **stone/signal ×2** for B.
  With **no memory**, the same start produced roller/signal and grump/signal
  (uninformed, nearest-first by cost). This is the clearest evidence that
  remembered, image-grounded outcomes change later choices.
* **Aversion — partial, limited by identity fragmentation.** In every learned
  run grump's red panel was observed once (`attach:red`, engineered valence −1);
  that entity became *disliked*, an `avoid` activity followed, and that entity
  was never engaged again. **However**, grump was also registered as a second,
  duplicate identity (seen from another place with a slightly different
  appearance), and the duplicate — having no bad history — was chosen again
  in every learned run (history_a 1×, noisy 3×, reversal_late 4×,
  reversal_early 5×). Every one of those attempts ended as a navigation
  failure, so no second red panel occurred, but that was not due to learning.
  Merging duplicate identities is on the next-gate list. (A superseded run also
  showed the dependence on perception: a detector bug classified the red panel
  as `none`; fixed and regression-tested.)
* **Context dependence.** Signalling bloom while its panel is still raised
  gives `none`; outcomes are stored under the visible context (`attach:none`
  vs `attach:yellow`), so these do not erase the learned yellow response.
* **Policy comparison (standard rules, same map and start).** Valence per
  observed outcome: learned **0.45**, fixed 0.38, random 0.18, nearest
  0.02. Total valence in 480 s: fixed 12.6, learned 5.0, random 3.0, nearest
  0.6. The fixed baseline repeats whichever option sorts first by entity id;
  in this run that was bloom/signal, the rewarding option, and it never rests.
  The learned policy **idles 27 % of the time** because the engineered
  stimulation need drops after useful outcomes; it is not a reward-rate
  maximiser. Learned vs random: fewer aversive outcomes (1 vs 2), higher
  valence (5.0 vs 3.0) from fewer attempts (19 vs 43).
* **Reversal.** Late switch (after 10 outcomes, t = 360 s): three failed
  bloom/signal tries, then stone/signal was tried and produced yellow 48 s
  after the switch (2 yellows). Early switch (after 6 outcomes, t = 189 s): three
  failed bloom/signal tries, then **no new useful option was found in the
  remaining 590 s** (8 navigation failures, 5 unintended contact episodes, 11
  operator re-enables in that run). Reversal adaptation: **1 of 2**.
* **Settling.** Inert world: 7 outcomes in the first half, 2 in the second,
  no useful outcomes; it stops probing but spends much of the time exploring
  rather than idle (idle 17 %). Noisy world (bloom yellow with p = 0.5): 2
  yellows from 6 bloom signals, then only 1 outcome in the second half — it
  backed off bloom although its true expected value stayed positive
  (change detection treated noise as change; a limitation).

## Safety and interventions during learning runs

Contact episodes are split into intended (inside a nudge interaction) and
unintended. Unintended contacts occurred only in the reversal runs:
reversal_early 5 episodes (one lasting ~60 s, pressing against bloom late in the
run during approach/explore) and reversal_late 2. Nudging the tall bloom
cylinder often filled the camera view and caused tracking loss → bounded
recovery → operator re-enable (the "operator re-enables" column); the option
is then suppressed after two such interruptions.

## Limits of this evidence

One seed per experiment, one arena, one start pose; engineered valences and
need dynamics; the map is from an earlier mapping-code state; fixture
detection relies on saturated uniform colours. The next gate is ≥ 10 seeds
with randomised start poses and fixture placements.


Evidence: `work/evidence/learning-20260930-022340`, `work/evidence/learning-20260930-042009`

Experiment → directory: baseline_fixed → `work/evidence/learning-20260930-022340`, baseline_nearest → `work/evidence/learning-20260930-022340`, baseline_random → `work/evidence/learning-20260930-022340`, history_a → `work/evidence/learning-20260930-022340`, history_b → `work/evidence/learning-20260930-022340`, no_memory → `work/evidence/learning-20260930-022340`, reversal_early → `work/evidence/learning-20260930-022340`, reversal_late → `work/evidence/learning-20260930-022340`, inert → `work/evidence/learning-20260930-042009`, learned_standard → `work/evidence/learning-20260930-042009`, noisy → `work/evidence/learning-20260930-042009`

### All phases (denominators)

| experiment | phase | policy | sim s | attempts | outcomes | useful | aversive | total valence | valence/outcome | ambiguous | nav failures | interrupted | idle frac | contact episodes intended/unintended | operator re-enables |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline_fixed | train | fixed | 480.300 | 50 | 33 | 13 | 0 | 12.600 | 0.382 | 0 | 9 | 3 | 0.000 | 5/0 | 2 |
| baseline_nearest | train | nearest | 480.300 | 41 | 34 | 1 | 0 | 0.600 | 0.018 | 0 | 5 | 1 | 0.000 | 2/0 | 1 |
| baseline_random | train | random | 480.300 | 43 | 17 | 5 | 2 | 3.000 | 0.176 | 0 | 10 | 11 | 0.000 | 3/0 | 6 |
| history_a | train | learned | 480.300 | 19 | 11 | 6 | 1 | 5.000 | 0.455 | 0 | 2 | 3 | 0.267 | 2/0 | 5 |
| history_a | test_after_restart | learned | 65.700 | 4 | 2 | 0 | 0 | 0.000 | 0.000 | 0 | 1 | 0 | 0.000 | 0/0 | 0 |
| history_b | train | learned | 480.300 | 19 | 13 | 7 | 1 | 6.000 | 0.462 | 0 | 2 | 0 | 0.366 | 1/0 | 0 |
| history_b | test_after_restart | learned | 20.700 | 2 | 2 | 0 | 0 | 0.000 | 0.000 | 0 | 0 | 0 | 0.000 | 0/0 | 0 |
| no_memory | test_fresh_memory | learned | 34.400 | 2 | 2 | 0 | 0 | 0.000 | 0.000 | 0 | 0 | 0 | 0.000 | 0/0 | 0 |
| reversal_early | train | learned | 780.300 | 26 | 10 | 2 | 1 | 1.000 | 0.100 | 0 | 8 | 5 | 0.359 | 3/5 | 11 |
| reversal_late | train | learned | 780.300 | 30 | 15 | 7 | 1 | 6.000 | 0.400 | 0 | 9 | 2 | 0.371 | 2/2 | 5 |
| inert | train | learned | 480.300 | 17 | 9 | 0 | 0 | 0.000 | 0.000 | 0 | 7 | 0 | 0.167 | 2/0 | 2 |
| learned_standard | train | learned | 480.300 | 19 | 11 | 6 | 1 | 5.000 | 0.455 | 0 | 2 | 3 | 0.267 | 2/0 | 5 |
| noisy | train | learned | 480.300 | 19 | 10 | 2 | 1 | 1.000 | 0.100 | 0 | 4 | 2 | 0.220 | 3/0 | 4 |

`useful` = valence ≥ 0.3 (yellow panel or moved); `aversive` = red panel. `attempts` counts engage/revisit interactions including cancelled/failed ones. Contact episodes are *intended* only when they fall inside a nudge interaction.

### Outcomes by true fixture / action / observed

* **baseline_fixed / train**: None/nudge/moved ×1, None/nudge/none ×5, bloom/signal/attach:yellow ×12, bloom/signal/none ×10, stone/nudge/none ×5
* **baseline_nearest / train**: None/nudge/moved ×1, None/nudge/none ×5, stone/nudge/none ×28
* **baseline_random / train**: None/signal/attach:yellow ×1, bloom/signal/attach:yellow ×4, grump/nudge/attach:red ×1, grump/signal/attach:red ×1, grump/signal/none ×1, roller/nudge/none ×1, roller/signal/none ×4, stone/nudge/none ×1, stone/signal/none ×3
* **history_a / train**: bloom/signal/attach:yellow ×6, bloom/signal/none ×2, grump/signal/attach:red ×1, roller/signal/none ×1, stone/signal/none ×1
* **history_a / test_after_restart**: bloom/signal/none ×2
* **history_b / train**: bloom/signal/none ×1, grump/signal/attach:red ×1, roller/signal/none ×1, stone/nudge/none ×1, stone/signal/attach:yellow ×7, stone/signal/none ×2
* **history_b / test_after_restart**: stone/signal/none ×2
* **no_memory / test_fresh_memory**: grump/signal/none ×1, roller/signal/none ×1
* **reversal_early / train**: bloom/signal/attach:yellow ×2, bloom/signal/none ×4, grump/signal/attach:red ×1, roller/signal/none ×1, stone/nudge/none ×1, stone/signal/none ×1
* **reversal_late / train**: bloom/signal/attach:yellow ×5, bloom/signal/none ×5, grump/signal/attach:red ×1, roller/signal/none ×1, stone/signal/attach:yellow ×2, stone/signal/none ×1
* **inert / train**: bloom/nudge/none ×1, bloom/signal/none ×1, grump/nudge/none ×1, grump/signal/none ×2, roller/signal/none ×1, stone/nudge/none ×1, stone/signal/none ×2
* **learned_standard / train**: bloom/signal/attach:yellow ×6, bloom/signal/none ×2, grump/signal/attach:red ×1, roller/signal/none ×1, stone/signal/none ×1
* **noisy / train**: bloom/signal/attach:yellow ×2, bloom/signal/none ×4, grump/signal/attach:red ×1, roller/signal/none ×1, stone/nudge/none ×1, stone/signal/none ×1

### Restart persistence and opposite histories (first interactions after restart)

| experiment | phase | prior sessions in memory | relocalized at s | autonomy at start | first interactions (true fixture/action) | first decisions (basis) |
|---|---|---|---|---|---|---|
| history_a | test_after_restart | 1 | 0.200 | False | bloom/signal; bloom/signal | revisit signal: 8 weighted outcomes; P(attach:yellow)=0.90; need 0.60; probes 3; revisit signal: 0 weighted outcomes; P(none)=0.60; need 0.64; probes 3 |
| history_b | test_after_restart | 1 | 0.200 | False | stone/signal; stone/signal | revisit signal: 9 weighted outcomes; P(attach:yellow)=0.92; need 0.60; probes 3; engage signal: 10 weighted outcomes; P(attach:yellow)=0.82; need 0.65; probes 2 |
| no_memory | test_fresh_memory | 0 | 0.200 | False | roller/signal; grump/signal | engage signal: 0 weighted outcomes; P(none)=0.60; need 0.63; probes 3; revisit signal: 0 weighted outcomes; P(none)=0.60; need 0.67; probes 3 |

### Policy comparison under the standard rules (same map, same start)

| run | policy | outcomes | useful | aversive | total valence | valence / 100 s | idle frac |
|---|---|---|---|---|---|---|---|
| learned_standard | learned | 11 | 6 | 1 | 5.000 | 1.041 | 0.267 |
| history_a | learned | 11 | 6 | 1 | 5.000 | 1.041 | 0.267 |
| baseline_random | random | 17 | 5 | 2 | 3.000 | 0.625 | 0.000 |
| baseline_nearest | nearest | 34 | 1 | 0 | 0.600 | 0.125 | 0.000 |
| baseline_fixed | fixed | 33 | 13 | 0 | 12.600 | 2.623 | 0.000 |

### Consequence changes (reversal) and settling

* **reversal_early**: switch to `swapped` at t = 189 s after 6 outcomes. Before: bloom/signal/attach:yellow ×2, bloom/signal/none ×1, grump/signal/attach:red ×1, roller/signal/none ×1, stone/signal/none ×1. After: bloom/signal/none ×3, stone/nudge/none ×1. Failed tries of the formerly useful option before first trying something else: 3. First useful outcome from a different option: never. Second half after switch: none.
* **reversal_late**: switch to `swapped` at t = 360 s after 10 outcomes. Before: bloom/signal/attach:yellow ×5, bloom/signal/none ×2, grump/signal/attach:red ×1, roller/signal/none ×1, stone/signal/none ×1. After: bloom/signal/none ×3, stone/signal/attach:yellow ×2. Failed tries of the formerly useful option before first trying something else: 3. First useful outcome from a different option: t = 408 s (stone/signal, 48 s after the switch). Second half after switch: none.
* **inert**: outcomes first half 7, second half 2; useful 0; idle fraction 0.167; by fixture/action: bloom/nudge/none ×1, bloom/signal/none ×1, grump/nudge/none ×1, grump/signal/none ×2, roller/signal/none ×1, stone/nudge/none ×1, stone/signal/none ×2
* **noisy**: outcomes first half 9, second half 1; useful 2; idle fraction 0.220; by fixture/action: bloom/signal/attach:yellow ×2, bloom/signal/none ×4, grump/signal/attach:red ×1, roller/signal/none ×1, stone/nudge/none ×1, stone/signal/none ×1

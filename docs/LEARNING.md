# Experience, memory and activity choice

This is an inspectable learning system with engineered motivations. It is **not**
a claim of consciousness, subjective emotion or self-created desire. "Attitude"
labels below are names for learned expected consequences, nothing more.

## The interaction experiment (engineered fixtures)

The arena contains four uniformly painted fixtures. Their response rules are
engineered, world-side and hidden from the robot (`configs/consequences/*.yaml`):

| Fixture (true name, for scoring only) | Appearance the robot sees | Standard rule |
|---|---|---|
| bloom | cyan cylinder 0.20 m | `signal` → a yellow panel rises for 8 s (not again while raised) |
| grump | green box 0.22 m | `signal` or `nudge` → a red panel rises for 8 s |
| stone | blue box 0.20 m | nothing |
| roller | magenta ball 0.12 m | physically rolls when nudged (no rule) |

Robot actions (physically plausible for wheels + camera + screen):

* **signal** — stand 0.75 m away facing the fixture and show the signal pattern
  on the screen for 2.2 s. The screen output is real; fixtures respond to it.
* **nudge** — from 0.52 m, creep straight at 0.06 m/s until 3 cm past the
  estimated contact, then retrace the approach at 0.10 m/s.

## From pixels to a learning update

1. **Perception** (`perception/entities.py`): saturated-colour regions in the
   onboard image; a region stacked on another is an *attachment* (visible state).
   Position from the object's floor contact row through the calibrated camera
   height and the current VSLAM pose.
2. **Identity** (`learning/identity.py`): appearance (hue, fill, metric size) +
   position in the current map version. Ambiguous → no identity, no learning.
   Two identities that look alike (within 1.5× the appearance gate), are
   remembered within 0.9 m of each other and have **never** been seen as two
   separate detections in one frame are merged into the older one, with all
   outcome history (a box seen at an angle can look 10 cm wider and was stored
   twice). Identities seen together in one frame are recorded as distinct and
   are never merged. Merges are recorded (`merges` table) and a receipt for a
   merged-away identity lands on the kept one.
3. **Decision** (`behavior/chooser.py`): value per option (below); the choice,
   its basis and the competing candidates are logged.
4. **Interaction** (`behavior/activities.py:Engage`): approach (navigation only;
   navigation failure is never an outcome) → **confirm** ≥ 3 fresh frames with
   the identified target (the visible state becomes the *context*) and record
   the **prediction** P(outcome | entity, action, context) → **act** → **observe**
   for 2.5 s.
5. **Outcome** (`learning/outcomes.py`): `moved` (displaced ≥ 15 cm and ≥ 3
   standard errors of the position scatter measured in the same frames, or former
   place in view and empty), `attach:<colour>` (a new attachment persisting ≥ 3
   frames), `none`; anything unstable or unobserved → ambiguous, nothing learned.
6. **Receipt → memory:** the receipt (event id, entity id + identity evidence,
   action, decision basis, prediction, context, before/after frame hashes and
   positions, observed outcome, map/calibration versions, authority generation)
   is written with its images immutably (`evidence/images.py`) **before** the
   SQLite transaction (`learning/memory.py`). Replays are no-ops; conflicting
   replays are rejected. Stop, heartbeat loss, stale frames or localization loss
   during an interaction cancel it: nothing is learned.

## What learns, and when

Updated only in `ExperienceMemory.record_outcome()` (step 6):

| Learned quantity | Model |
|---|---|
| P(outcome \| entity, action, context) | Dirichlet posterior over 10 outcome tokens; last 12 outcomes per option; each outcome weighted 0.5^(age/900 s); evidence before a detected change weighted ×0.25; context backs off to the entity-action aggregate |
| expected valence, uncertainty | derived from the posterior and the engineered valence table |
| probe budget, useful/failure streaks, stability | per (entity, action); budget 3, reset by a useful outcome |
| change hypotheses | 2 (early) or 6 (stable) consecutive useful outcomes, then a run of consecutive failures long enough to be unlikely (p < 0.05) under the success rate seen before it (at least 3, at most 8: 3 for a reliable effect, 5 for a 50 % one) → change epoch; suppressed alternatives get one probe; older evidence is discounted |
| entity `movable` flag | set when a `moved` outcome is observed |

Updated by perception (not outcome learning): entity appearance (running mean),
position in the current map version, labels from the semantic worker.

## Engineered (configured, never learned)

* **Valence table** (the motivation): `attach:yellow` +1.0, `moved` +0.6,
  other new attachment colours +0.2, `none` 0, `attach:red` −1.0.
* **Stimulation need** ∈ [0, 1]: grows 0.004/s; each positive outcome lowers it by
  0.35 × valence. **Habituation** per entity: +1 per positive outcome, half-life
  120 s; novelty = 1/(1 + habituation).
* Priors (P(none) = 0.6, strength 2), window, half-life, probe budget, curiosity
  weight 0.5 and cap 0.2, action costs (signal 0.03, nudge 0.06), travel cost
  0.05/m, explore weight, reconsideration limits.

## Choosing an activity

For each identified entity with a position in the current map and each action
with probes left:

```
value = need × E[valence] × novelty + curiosity × uncertainty × (0.3 + 0.7 need) − costs
```

Other activities: **investigate** (never-interacted entity, closer look),
**explore** (frontier, scaled by recent map growth), **avoid** (a disliked entity
within 0.75 m pre-empts everything), **revisit** (engage an entity that is not in
view, from its remembered position), **idle** (value 0; wins when nothing is
worth its cost, e.g. when the need is satisfied).

**Bounded reconsideration:** decisions happen when an activity ends, or when a
trigger fires (new entity evidence, ambiguous identity, outcome recorded) — at
most every 3 s, at most 6 switches per minute, only if the alternative beats the
current value by 0.08, and never inside an interaction's confirm/act/observe
phases. Navigation failures back the target off for 30 s × attempts; an option
interrupted twice by localization loss is suppressed for the session.

## Persistence lifecycles

| State | Survives restart? | Notes |
|---|---|---|
| Learned expectations, receipts, entity identities/appearance | yes | same agent id required |
| Entity positions | only within the same map version | a new map invalidates them; identity then relies on appearance only (ambiguous if two look alike) |
| Motion authority | **never** | every launch starts disabled; `enable_autonomy` needs fresh images + tracking |
| Need/habituation, route back-offs, interruption counts | no | session state |
| Explicit reset | `reset_memory` with `confirm: "RESET"` | deletes knowledge and issues a new agent id |

# Robot screen expression

Pip's screen shows its **operational state**: what it is doing, how uncertain it
is, and what it has *learned* about its current target. Everything on the face
is computed by one pure function,
`amr_rl.expression.policy.expression_from(state) -> ExpressionState`. It takes
the operator snapshot (`amr_rl.state.v1`, [INTERFACES.md](INTERFACES.md)) as
input. Then `amr_rl.expression.screen.render(expr, size=(192, 132))` draws it
deterministically as a `uint8` RGB array of shape `(132, 192, 3)`.
`size` is `(width, height)`.

## Honesty note

- The face displays state. It does **not** show emotions, and the robot has no
  emotion model. Face names like "happy" or "wary" are shorthand for display
  shapes. Each shape stands for a specific activity/attitude combination listed
  below. A "happy" face means "engaging or revisiting an entity whose learned
  outcomes are net-positive". It does not mean the robot feels anything.
- `liked` / `disliked` / `indifferent` only appear when the target entity's
  memory entry has `interactions >= 1`, which means at least one observed
  outcome was learned. Before that, the attitude is always shown as `unknown`
  (a hollow pip), whatever the entity entry says.
- No hidden variables are involved. The function does not read the clock,
  randomness, simulator ground truth, or any mood state. Identical snapshots
  give identical faces and pixels. Each output carries `basis`, which lists the
  snapshot fields it came from. The operator console shows these under "why
  this face?".
- The display constants are **engineered** and not learned. These are the
  per-face openness, colours, the 0.15 m sigma scale and the 0.65 rad gaze
  scale.

## Face selection (first matching rule wins)

| # | Condition (state fields) | Face | Openness |
|---|---|---|---|
| 1 | `localization.status` ∈ {`lost`, `relocalizing`} | `confused` | 0.85 |
| 2 | `authority.stopped == true` and `activity.name != "manual"` | `stopped` | 0.10 |
| 3 | `activity.name == "engage"` and `interaction.action == "signal"` and `interaction.status` ∈ {`acting`, `signalling`} | `signalling` | 0.90 |
| 4 | `activity.name == "engage"`, by displayed attitude: `liked`→`happy`, `disliked`→`wary`, `indifferent`→`neutral`, `unknown`/`none`→`curious` | as listed | per face |
| 5 | `activity.name == "revisit"`: `happy` if displayed attitude is `liked`, else `curious` | as listed | per face |
| 6 | `activity.name`: `idle`→`sleepy`, `explore`→`curious`, `investigate`→`focused`, `avoid`→`wary`, `recovering`→`confused`, `manual`→`neutral`, `stopped`→`stopped` | as listed | per face |
| 7 | anything else, missing, or `null` | `neutral` | 0.90 |

Openness per face (engineered): neutral 0.90, curious 1.00, focused 0.72, happy
0.90, wary 0.45, sleepy 0.35, confused 0.85, stopped 0.10, signalling 0.90.

## Visual element → state field

| Visual element | ExpressionState field | Snapshot field(s) | Rule |
|---|---|---|---|
| Eye shape, brows, glyphs (`?`, `z z`, red square, arcs) | `face` | see face table above | happy = upturned arcs; wary = narrowed eyes, slanted lids, inner-low brows; sleepy = dim, low lids, "z z"; confused = asymmetric eyes, raised brow and `?`; stopped = flat dim lines and a small red square; focused = slightly smaller eyes, larger pupils, level brows; curious = large round eyes, one raised brow; neutral = plain rounded eyes |
| Eyelid height | `openness` | `face` | fixed per face (table above) |
| Pupil and eye horizontal offset | `gaze[0]` | `localization.pose`, `activity.target_entity`, `entities[target].position` | `bearing = wrap(atan2(ey−y, ex−x) − θ)` (CCW-positive, meaning target on the robot's left); `gaze_x = −clamp(bearing / 0.65 rad, −1, 1)`. `gaze_x` is robot-frame: + means toward the robot's **right**. The screen faces outward, so the renderer mirrors it. The pupils move toward the target as a person in front of the robot sees it: a target on the robot's left moves the pupils to the viewer's right (image +x). With no target, a missing pose or position, or lost localization, `gaze_x = 0` |
| Vertical gaze | `gaze[1]` | – | always 0 (there is no vertical target bearing). The sleepy face draws its pupils low as part of its shape |
| Bottom bar, `?` label and fill width | `uncertainty` | `localization.position_sigma`, `localization.status`, `entities[target].uncertainty`, `.ambiguous`, `.identity_confidence` | `u = clamp(position_sigma / 0.15 m)`. `u = 1` if localization is missing, `position_sigma` is null, or status is lost/relocalizing. Then `u = max(u, target.uncertainty)`, and if the target is `ambiguous`, `u = max(u, 1 − identity_confidence)` (1 if confidence is null). Fill width ∝ `u`, and the colour runs teal → amber |
| Top-right pip | `attitude` | `activity.target_entity`, `entities[target].attitude`, `entities[target].interactions` | green = liked, red = disliked, grey = indifferent, hollow white ring = unknown, nothing = none (no target, or target id not in `entities`). Learned values need `interactions >= 1`, otherwise the pip shows `unknown` |
| Concentric yellow/magenta rings (large, centred between the eyes) | `signal_pattern` | `interaction.action`, `interaction.status` | drawn **if and only if** `interaction.action == "signal"` and `interaction.status` ∈ {`acting`, `signalling`}. The world can perceive this pattern with a camera, so it is tied exactly to the real signal action and to nothing else. It is drawn even when another rule (for example, stopped) picks the face |
| Caption (UI and `state.expression`, not drawn on the screen) | `caption` | `activity.name`, target `label`, `localization.status`, `authority.*` | "Lost - stopped" / "Lost" / "Relocalizing"; "Stopped" or "Stopped (<revoked_reason>)"; "Signalling <label>"; "Inspecting / Engaging / Revisiting / Avoiding <label>"; "Resting" (idle), "Exploring", "Recovering", "Manual drive"; at most 28 characters (truncated with "...") |

## Output

`ExpressionState.to_dict()` gives the JSON form for `state["expression"]`:
`face`, `gaze`, `openness`, `uncertainty`, `attitude`, `caption`,
`signal_pattern` and `basis` (element → list of snapshot field paths).
`screen.render_png(expr)` returns PNG bytes for `GET /api/screen.png`.

## Tests

`tests/test_expression.py` covers every rule above. That includes attitude
gating on `interactions >= 1`, the signal pattern's exact tie to interaction
state, gaze sign and saturation, and missing or malformed fields. It also
checks render shape, dtype and determinism, that every pair of faces renders
differently, that a left target moves the pupils to the viewer's right, that
the bar width is proportional to uncertainty, and the pip colours.

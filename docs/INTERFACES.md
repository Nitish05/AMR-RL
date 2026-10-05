# Runtime interfaces

## Operator state snapshot (`GET /api/state`, schema `amr_rl.state.v1`)

All coordinates are in the robot's **estimated** map frame (metres, radians),
never simulator ground truth. `null` means unavailable/unknown, never zero.

```jsonc
{
  "schema": "amr_rl.state.v1",
  "sim_time": 12.3,                 // simulation seconds (lockstep)
  "timing": "lockstep_simulation",  // physics pauses during perception
  "authority": {
    "autonomy_enabled": false,
    "manual_active": false,
    "stopped": true,
    "generation": 3,                // command generation; stale commands rejected
    "revoked_reason": "stop" ,      // null | stop | heartbeat_lost | stale_observation | localization_lost | restart | error
    "heartbeat_age": 0.4            // seconds since last operator heartbeat (null if none)
  },
  "localization": {
    "status": "tracking",           // initializing | tracking | lost | relocalizing
    "pose": [0.4, -0.2, 1.57],      // x, y, theta or null
    "position_sigma": 0.03,         // 1-sigma metres or null
    "inliers": 85, "landmarks": 1200, "keyframes": 34,
    "map_version": "map-1a2b3c",
    "odometry": {                   // round 9; {"source": "command"} for the camera-only robot
      "source": "imu_encoders",     // command | imu_encoders | command_model
      "calibration": "loaded (imu.yaml, scale 1.00249)",
      "gyro_bias_dps": 1.07, "gyro_bias_sigma_dps": 0.002, "bias_calibrated": true,
      "gyro_scale": 1.00249, "gyro_scale_sigma": 0.0002, "track_scale": 0.97,
      "slip": false, "slip_counts": {"zupt": 40, "slip_yaw": 1, "slip_accel": 0, "slip_cmd": 0},
      "fusion": {"frames": 600, "rot_downweighted": 3, "xy_downweighted": 9, "resyncs": 0, "predicted_odo": 0},
      "command_model_fallback_frames": 0
    }
  },
  "camera": {"frame_index": 120, "timestamp": 12.3, "age": 0.0, "fresh": true},
  "activity": {
    "name": "engage",               // explore | investigate | engage | revisit | avoid | idle | manual | stopped | recovering
    "target_entity": "ent-3f2a",    // or null
    "phase": "approaching",         // free text phase inside the activity
    "reason": "Expected +0.62 (learned) vs idle 0.00",
    "since": 10.1
  },
  "decision": {
    "chosen": {"activity": "engage", "entity_id": "ent-3f2a", "action": "signal", "value": 0.51},
    "candidates": [
      {"activity": "engage", "entity_id": "ent-3f2a", "action": "signal",
       "value": 0.51, "expected_value": 0.62, "information_value": 0.02, "cost": 0.13,
       "basis": "6 outcomes; P(yellow_flag)=0.83"}
    ],
    "reconsiderations": 4
  },
  "entities": [
    {"entity_id": "ent-3f2a", "label": "cyan cylinder", "visible": true,
     "identity_confidence": 0.93, "ambiguous": false, "position": [1.2, 0.4],
     "attitude": "liked",            // liked | disliked | indifferent | unknown
     "expected_value": 0.62, "uncertainty": 0.12, "interactions": 6}
  ],
  "interaction": {                  // current pending interaction or null
    "request_id": "…", "entity_id": "ent-3f2a", "action": "signal",
    "predicted": {"yellow_flag": 0.83, "none": 0.17}, "observed": null, "status": "observing"
  },
  "recent_outcomes": [
    {"event_id": "…", "entity_id": "ent-3f2a", "action": "signal", "context": "flag_down",
     "predicted": {"yellow_flag": 0.8, "none": 0.2}, "observed": "yellow_flag",
     "valence": 1.0, "update": {"before": 0.58, "after": 0.62}, "timestamp": 11.0}
  ],
  "motivation": {"stimulation_need": 0.4, "engineered": true},
  "expression": {"face": "happy", "gaze": [0.2, 0.0], "uncertainty": 0.1,
                 "attitude": "liked", "caption": "Greeting cyan cylinder"},
  "navigation": {"goal": [1.0, 0.5], "path": [[0.4, -0.2], [1.0, 0.5]],
                 "status": "following", "rejected_goals": 1, "last_rejection": "goal in unknown space"},
  "trajectory": [[0.0, 0.0], [0.1, 0.0]],
  "memory": {"agent_id": "pip-…", "db": "…", "outcomes": 12, "entities": 4}
}
```

Images: `GET /api/camera.jpg` (latest onboard RGB), `GET /api/map.png`
(occupancy: free / occupied / unknown + trajectory + entities),
`GET /api/screen.png` (robot face framebuffer),
`GET /api/inspect.jpg` (third-person inspection view; **human inspection only**).

Commands: `POST /api/command` JSON `{"action": ..., ...}`; response
`{"accepted": bool, "reason": str, "generation": int}`.

| action | fields | effect |
|---|---|---|
| `stop` | – | Always accepted. Revokes all motion authority, increments generation. |
| `heartbeat` | – | Refreshes operator liveness; never grants authority. |
| `manual` | `v` (m/s), `w` (rad/s), `generation` | Short-lived teleop command; rejected if generation is stale. |
| `enable_autonomy` | `generation` | Requires fresh camera frame and `tracking` localization. |
| `disable_autonomy` | – | Revokes autonomy (like stop, keeps manual idle). |
| `goal` | `x`, `y`, `generation` | Navigate to an estimated-map point; rejected when unsupported. |
| `reset_memory` | `confirm: "RESET"` | Deletes learned memory for this agent (explicit). |

## Proprioceptive input (round 9)

`RobotRuntime.on_proprio(ProprioBatch)` receives the raw samples that arrived since
the previous call, before each `on_frame`:

- `ImuSample(t, gyro[3] rad/s, accel[3] m/s²)`: quantised readings in the IMU frame;
- `EncoderSample(t, left, right)`: cumulative signed quadrature counts.

They are never poses or velocities from the simulator (`tests/test_privilege_boundary.py`).
The parts and their error models are in [ROBOT.md](ROBOT.md).

## Screen expression

`amr_rl.expression.policy.expression_from(state) -> ExpressionState` is a pure
function of the snapshot above. `amr_rl.expression.screen.render(expression,
size=(192, 132)) -> uint8 RGB` draws the face. Every visual element must map to
a real state field (see `docs/EXPRESSION.md`).

# The robot: "Pip", a compact indoor AMR (simulation asset)

All values are simulation design choices, not measurements of a physical robot.
Source of truth: [`assets/robot/amr_spec.yaml`](../assets/robot/amr_spec.yaml).
Regenerate derived assets with `scripts/amr.sh generate-robot` (deterministic;
`tests/test_robot_assets.py` checks the committed files match the spec).

![Pip in the arena, inspection view](media/robot-inspection.png)

## Geometry and mass

| Item | Value |
|---|---|
| Chassis | rounded rectangular shell 0.35 m (L) × 0.23 m (W) × 0.10 m (H), 45 mm plan-view corner radius, 15 mm edge bevel, 30 mm ground clearance |
| Overall footprint | 0.35 m long × 0.30 m wide (outside of tyres); 0.339 m tall (top of tilted screen) |
| Drive | two independently driven wheels, radius 0.05 m, width 0.03 m, track 0.27 m, axle 0.04 m ahead of the chassis centre |
| Support | passive ball-transfer caster, radius 0.02 m, 35 mm inside the rear, friction 0.02 |
| Mass | 5.08 kg total (chassis 4.0, wheels 2×0.3, caster 0.08, camera 0.05, screen 0.35) |
| Inertia | analytic box/cylinder/sphere tensors per link (see URDF) |
| Wheel actuation | Genesis velocity control, gain kv = 4.0, torque limit ±1.5 N·m, speed limit 12 rad/s; tyre friction 1.2 |
| Planning footprint | circumscribed radius 0.262 m about `base_link` (+0.03 m margin) |

Collision geometry is simplified to URDF primitives: chassis box, mast
cylinder, tyre cylinders, caster sphere, screen box. Visual meshes are GLB with
distinct materials: orange shell, black tyres, grey hubs, dark camera housing
with a blue lens, dark bezel and display.

## Frames (URDF links)

| Frame | Pose (in `base_link`) | Notes |
|---|---|---|
| `base_link` | origin | axle midpoint, 0.05 m above the floor; x forward, y left, z up |
| `left_wheel` / `right_wheel` | (0, ±0.135, 0) | continuous joints about +y |
| `caster_link` | (−0.18, 0, −0.03) | fixed |
| `camera_link` | (0.125, 0, 0.105), pitched down 12° | fixed; optical frame below |
| `screen_link` | (−0.075, 0, 0.235), tilted back 10° | fixed; display faces forward |

The screen sits 0.2 m behind the camera, so it never enters the camera view.

## Onboard camera (the only exteroceptive sensor; purchasable part in the next section)

| Property | Value |
|---|---|
| Model | pinhole, no distortion (the simulated lens has none) |
| Resolution | 320 × 240 RGB |
| Intrinsics K | fx = fy = 207.846 px, cx = 159.5, cy = 119.5 (OpenCV integer-pixel-centre convention) |
| Field of view | 60° vertical, 75.2° horizontal |
| Mounting | optical centre 0.155 m above the floor, 0.125 m ahead of the rotation axis, pitched 12° down |
| Optical frame | OpenCV: x right, y down, z forward. `T_base_cam` = [[0, −0.208, 0.978, 0.125], [−1, 0, 0, 0], [0, −0.978, −0.208, 0.105]] |
| Horizon row | 75.3 px; nearest visible floor 0.30 m ahead of `base_link` |
| Capture | 10 Hz in simulated time; each frame carries index + timestamp; the supervisor treats frames older than 0.35 s as stale |
| Calibration id | `cal-ac71bfd8f6b3` (hash of K, extrinsics, size, base height; maps record it and refuse a different calibration) |

Genesis reports its principal point in viewport coordinates (160, 120); AMR-RL
uses the equivalent OpenCV integer-centre value (159.5, 119.5), the same
half-pixel conversion BB8-RL established. The camera is attached to
`base_link` with `camera.attach()` and re-posed from the physical link pose at
every capture, so it moves with the body.

## Proprioceptive sensors (round 9): IMU and wheel encoders

Added on 2026-10-05 at the owner's request. Every sensor is a real part that can be bought today. The simulation models each part from its datasheet: `sim/sensors.py` applies the errors to the physical signal Genesis computes. The runtime receives only raw samples (`RobotRuntime.on_proprio`): integer encoder counts, and quantised gyro and accelerometer readings.

| Sensor | Part (where to buy, approx. price) | Modelled |
|---|---|---|
| IMU | ST LSM6DSOX 6-axis, Adafruit breakout 4438 (STEMMA QT/Qwiic, I2C/SPI; Adafruit, DigiKey, Mouser, The Pi Hut), about $12 | Datasheet DS12814 Rev 3, all "typ": ±250 dps range, 8.75 mdps/LSB, noise 3.8 mdps/√Hz, ODR 104 Hz with a 33 Hz filter, zero-rate offset ±1 dps, sensitivity ±1 %; accelerometer ±4 g, 75 µg/√Hz, offset ±20 mg. **Assumed** (not specified): cross-axis 1 %, bias random walk 0.002 dps/√s. Each unit's sensitivity and cross-axis errors are fixed by `imu.unit_serial`; offset and noise change per run |
| Wheel encoders | Pololu 4754: 70:1 metal gearmotor 37D×70L mm, 12 V, with a 64 CPR Hall-effect quadrature encoder, about $61 each | 4480 counts per wheel revolution (64 × 70, exact): 1.40 mrad and 0.070 mm of travel per count. Gearbox backlash 1° (**assumed**; Pololu gives no figure). 100 Hz sampling |
| Camera | Arducam B0394 (Sony IMX219 with an 88° diagonal, low-distortion M12 lens) matches the simulated 75° × 60° field of view | The vendor's FOV and focal length disagree, so calibrate with a checkerboard before relying on the intrinsics |

**Mounting.** The IMU sits on `base_link` at (−0.04, 0, 0.03), the chassis centre, with its z axis vertical.

**Gyro offset over time** (`sim/sensors.py`):
- a fixed offset per unit plus 0.1 dps per power-on (**assumed**);
- rate random walk 1.5e-4 dps/√s (**analogue**: BMI160 and BMI055 Allan measurements);
- **thermal warm-up**: the datasheet tempco (±0.010 dps/°C typ) times the electronics warming the IMU by 2–12 °C with τ 150–900 s (**assumed** ranges);
- the chip's temperature sensor (256 LSB/°C) is part of each sample.

The robot compensates the offset for temperature and recalibrates at stops.

**Calibration.** The gyro sensitivity tolerance (±1 %) would cost up to 3.6° per full turn, so the gyro scale is calibrated once per unit. `scripts/calibrate_imu.py` has the robot:
1. stand still to measure the offset;
2. turn in place 5 times each way;
3. measure the true total angle with the camera (the planar epipolar residual between the start and end images; no ground truth);
4. write `configs/calibration/imu.yaml`.

For unit 1 the calibration gives 1.00249 ± 0.0002. The same turns scored against simulator truth (evaluation only) give 1.00248. At runtime the robot also:
- stands still for 1 s at boot to measure the zero-rate offset;
- updates the offset whenever it stands still;
- refines the scale from full in-place turns that vision closes on the same view.

**Caveat (motor model).** The simulated drive still has a flat ±1.5 N·m torque limit. The Pololu 4754 stalls at 2.65 N·m at 12 V; on a linear speed–torque curve it gives about 0.63 N·m at 12 rad/s, and it is rated for 0.98 N·m continuous. The motor's speed–torque line is not modelled yet.

## Screen

A 0.16 m × 0.11 m display on a mast. `expression.screen.render()` draws a
192 × 132 face; the robot's `signal` action is a specific screen pattern.
In simulation the framebuffer is drawn onto the screen link as a textured debug
quad visible only to the third-person inspection camera (Genesis debug markers
are skipped by non-debug cameras), so the robot's own perception never sees it.
World fixtures "see" the signal through their engineered response rules.

## Control stack

`Supervisor` (authority) → `GenesisWheelBackend` (only actuator module) →
`control_dofs_velocity` on the two wheel joints. Body pose setters are used only
by `initialize_pose()` for explicit resets. There is no hardware adapter;
nothing connects to hardware on import or during simulation.

The BB8 movement policy (SAC/Dreamer on a free rolling body) was not
transferred; the AMR uses a classical pure-pursuit baseline on a conservative
grid plan. Learned low-level control remains a ledger item.

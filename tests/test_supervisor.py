"""Motion-authority contract: Stop priority, generations, heartbeat, freshness,
localization loss, bounded recovery and no automatic resumption."""

import math

import pytest

from amr_rl.control.contract import DriveCommand, DriveLimits, body_to_wheels
from amr_rl.control.supervisor import Supervisor, SupervisorConfig


class FakeBackend:
    def __init__(self):
        self.commands, self.stops = [], 0

    def command(self, command, *, now):
        self.commands.append(command)

    def stop(self):
        self.stops += 1


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def make(**cfg):
    clock = Clock()
    sup = Supervisor(FakeBackend(), SupervisorConfig(**cfg), wall_clock=clock)
    return sup, clock


def ready(sup, now=1.0):
    sup.heartbeat()
    sup.observe_frame(now)
    sup.observe_localization("tracking")


def test_starts_without_authority_after_restart():
    sup, _ = make()
    assert not sup.autonomy_enabled and sup.stopped and sup.revoked_reason == "restart"
    assert not sup.drive(0.1, 0, now=0.0, generation=sup.generation).accepted


def test_enable_requires_current_generation_fresh_frame_tracking_and_heartbeat():
    sup, clock = make()
    assert sup.handle({"action": "enable_autonomy", "generation": 0}, now=1.0).reason == "no_operator_heartbeat"
    sup.heartbeat()
    assert sup.handle({"action": "enable_autonomy", "generation": 0}, now=1.0).reason == "camera_observation_not_fresh"
    sup.observe_frame(1.0)
    assert sup.handle({"action": "enable_autonomy", "generation": 0}, now=1.0).reason == "localization_not_tracking"
    sup.observe_localization("tracking")
    assert sup.handle({"action": "enable_autonomy", "generation": 5}, now=1.0).reason == "stale_generation"
    ack = sup.handle({"action": "enable_autonomy", "generation": 0}, now=1.0)
    assert ack.accepted and ack.generation == 1 and sup.autonomy_enabled


def test_stop_wins_batch_and_stale_generation_cannot_resume():
    sup, _ = make()
    ready(sup)
    gen = sup.handle({"action": "enable_autonomy", "generation": 0}, now=1.0).generation
    acks = sup.handle_batch([{"action": "manual", "v": 0.2, "w": 0, "generation": gen}, {"action": "stop"},
                             {"action": "enable_autonomy", "generation": gen}], now=1.0)
    assert [a.reason for a in acks] == ["superseded_by_stop", "stop", "superseded_by_stop"]
    assert not sup.autonomy_enabled and sup.generation == gen + 1
    # A command issued before the Stop (old generation) is rejected afterwards.
    assert sup.handle({"action": "enable_autonomy", "generation": gen}, now=1.05).reason == "stale_generation"
    assert not sup.drive(0.1, 0.0, now=1.05, generation=gen).accepted


def test_heartbeat_loss_revokes_and_does_not_restore():
    sup, clock = make(heartbeat_seconds=2.0)
    ready(sup)
    sup.handle({"action": "enable_autonomy", "generation": 0}, now=1.0)
    clock.t += 2.5
    sup.observe_frame(1.1)
    assert sup.check(now=1.1) == "heartbeat_lost"
    assert not sup.autonomy_enabled
    sup.heartbeat()
    sup.observe_frame(1.2)
    assert sup.check(now=1.2) is None and not sup.autonomy_enabled


def test_stale_observation_revokes():
    sup, _ = make(max_observation_age=0.35)
    ready(sup)
    sup.handle({"action": "enable_autonomy", "generation": 0}, now=1.0)
    assert sup.drive(0.1, 0, now=1.2, generation=sup.generation).accepted
    assert sup.check(now=1.5) == "stale_observation"
    assert not sup.drive(0.1, 0, now=1.5, generation=sup.generation).accepted


def test_localization_loss_grants_only_bounded_recovery():
    sup, _ = make(recovery_max_angle=1.0, recovery_max_reverse=0.2)
    ready(sup)
    sup.handle({"action": "enable_autonomy", "generation": 0}, now=1.0)
    sup.observe_localization("lost")
    sup.observe_frame(1.1)
    assert sup.check(now=1.1) == "localization_lost"
    gen = sup.generation
    assert not sup.autonomy_enabled and sup.recovery is not None
    assert not sup.drive(0.1, 0.0, now=1.1, generation=gen).accepted  # no autonomy
    assert not sup.drive(0.1, 0.0, now=1.1, generation=gen, source="recovery").accepted  # no forward
    assert not sup.drive(-0.05, 0.3, now=1.1, generation=gen, source="recovery").accepted  # not both
    assert sup.drive(0.0, 0.4, now=1.1, generation=gen, source="recovery").accepted
    t = 1.1
    while sup.recovery is not None and t < 10:
        t += 0.1
        sup.observe_frame(t)
        sup.drive(0.0, 0.4, now=t, generation=gen, source="recovery")
        sup.check(now=t)
    assert sup.revoked_reason == "recovery_exhausted" and not sup.autonomy_enabled


def test_recovery_success_leaves_robot_stopped_until_explicit_enable():
    sup, _ = make()
    ready(sup)
    sup.handle({"action": "enable_autonomy", "generation": 0}, now=1.0)
    sup.observe_localization("lost")
    sup.observe_frame(1.1)
    sup.check(now=1.1)
    sup.observe_localization("tracking")
    sup.observe_frame(1.2)
    assert sup.check(now=1.2) == "recovery_succeeded"
    assert not sup.autonomy_enabled and sup.recovery is None


def test_manual_override_revokes_autonomy_and_requires_resend():
    sup, _ = make()
    ready(sup)
    gen = sup.handle({"action": "enable_autonomy", "generation": 0}, now=1.0).generation
    ack = sup.handle({"action": "manual", "v": 0.1, "w": 0, "generation": gen}, now=1.0)
    assert not ack.accepted and not sup.autonomy_enabled
    ack = sup.handle({"action": "manual", "v": 0.1, "w": 0, "generation": sup.generation}, now=1.01)
    assert ack.accepted


def test_manual_command_is_exposed_as_the_robots_own_motion_until_it_expires_or_stop():
    """The VSLAM chains relocalisation candidates through the robot's commanded motion;
    an operator turn must reach it (it used to be reported as standing still)."""
    sup, _ = make()
    ready(sup)
    assert sup.handle({"action": "manual", "v": 0.0, "w": 0.4, "generation": sup.generation}, now=2.0).accepted
    assert sup.manual_command == (0.0, 0.4) and 2.0 < sup.manual_until
    sup.stop()
    assert sup.manual_until == -math.inf  # Stop ends it at once


def test_frames_must_increase():
    sup, _ = make()
    sup.observe_frame(1.0)
    with pytest.raises(ValueError):
        sup.observe_frame(1.0)


def test_drive_command_contract_and_wheel_conversion():
    with pytest.raises(ValueError):
        DriveCommand(0.1, 0.0, 1.0, 0.9, 0)
    with pytest.raises(ValueError):
        DriveCommand(math.nan, 0.0, 1.0, 1.1, 0)
    limits = DriveLimits(max_linear=0.3, max_angular=1.0, max_wheel_speed=5.0)
    wheels = body_to_wheels(1.0, 0.0, wheel_radius=0.05, track=0.27, limits=limits)
    assert wheels.left == pytest.approx(5.0) and wheels.right == pytest.approx(5.0)
    turn = body_to_wheels(0.0, 1.0, wheel_radius=0.05, track=0.27, limits=limits)
    assert turn.left == pytest.approx(-turn.right) and turn.right > 0


def test_recovery_after_a_suspected_contact_backs_off_and_never_rotates():
    """Round 5: tracking was lost while the robot pressed on a box it could not see,
    and the rotate recovery swept the chassis into it for 18 s."""
    from amr_rl.runtime.robot import choose_recovery

    common = {"nudge_retrace": None, "turning_clearance": 1.0, "required_clearance": 0.28, "certified": True}
    stall = choose_recovery(loss_reason="visual_motion_inconsistent_with_commands", last_command=(0.19, 0.0), **common)
    assert stall["kind"] == "back_off" and stall["remaining"] <= 0.10
    other = choose_recovery(loss_reason="tracking_failed", last_command=(0.19, 0.0), **common)
    assert other["kind"] == "rotate"


def test_rotate_recovery_needs_turning_clearance_not_just_a_free_cell():
    from amr_rl.runtime.robot import choose_recovery

    tight = choose_recovery(loss_reason="tracking_failed", last_command=(0.0, 0.3), nudge_retrace=None,
                            turning_clearance=0.20, required_clearance=0.28, certified=True)
    assert tight["kind"] == "wait"
    nudge = choose_recovery(loss_reason="visual_motion_inconsistent_with_commands", last_command=(0.05, 0.0),
                            nudge_retrace=0.12, turning_clearance=1.0, required_clearance=0.28, certified=True)
    assert nudge["kind"] == "retrace"  # a nudge's verified approach is still retraced

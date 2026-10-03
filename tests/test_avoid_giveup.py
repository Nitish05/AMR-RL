"""Forced avoid: bounded give-up when a retreat is impossible (no simulator).

Regression (learning round 3, seed 0): next to the aversive fixture, 41 avoid
attempts in a row failed and the robot did nothing else for 470 s; with no retreat
pose at all, avoid would even be re-chosen every 0.1 s control tick.
"""

from types import SimpleNamespace

import numpy as np

from amr_rl.behavior.activities import Avoid
from amr_rl.learning.memory import ExperienceMemory
from amr_rl.sim.learner_bench import MAP_VERSION, BenchRuntime


def cornered_runtime(dist=0.5):
    memory = ExperienceMemory(":memory:")
    memory.upsert_entity("ent-red", appearance={"hue": 120.0, "fill": 0.9}, now=0.0, map_version=MAP_VERSION,
                         xy=(dist, 0.0), pos_sigma=0.05)
    for k in range(2):
        memory.record_outcome({"event_id": f"e{k}", "entity_id": "ent-red", "action": "signal",
                               "context": "attach:none", "observed": "attach:red", "timestamp": float(k),
                               "authorised": True, "images_retained": True})
    rt = BenchRuntime(memory, policy="learned", seed=0, room=5.0)
    rt.tracker.refresh()
    rt.pose = np.array([0.0, 0.0, 0.0])
    rt.planner = SimpleNamespace(traversable_xy=lambda grid, p: np.zeros(len(p), bool))  # nowhere to go
    rt.grid = None
    assert memory.attitude("ent-red", 2.0)["attitude"] == "disliked"
    return rt


def run(rt, seconds, t0=2.0):
    chooser, current, started, results = rt.chooser, None, [], []
    for k in range(int(seconds * 10)):
        now = t0 + 0.1 * k
        rt.motivation.update(now)
        act = chooser.decide(rt, now, current)
        if act is not None:
            current = act
            started.append((now, act.name, getattr(act, "hold_until", None)))
        v, w = current.step(rt, now)
        assert (v, w) == (0.0, 0.0)  # cornered: never moves, in particular never approaches
        if current.done and current.result not in [r[1] for r in results[-1:]]:
            results.append((now, current.result))
    return started, results


def test_impossible_retreat_gives_up_into_bounded_holds():
    rt = cornered_runtime()
    started, results = run(rt, 120.0)
    assert {name for _, name, _ in started} == {"avoid"}  # nothing else while within the radius
    failures = [r for _, r in results if r["status"] == "navigation_failed"]
    holds = [s for s in started if s[2] is not None]
    # 3 failures, then hold 20 s, retry, hold 40 s, retry, hold 60 s: not one attempt per tick
    assert len(started) <= 10 and 3 <= len(failures) <= 6 and len(holds) >= 2
    assert holds[1][2] - holds[1][0] > holds[0][2] - holds[0][0]  # backoff grows


def test_cancel_is_not_a_failure_and_leaving_the_radius_resets():
    rt = cornered_runtime()
    chooser = rt.chooser
    for k in range(3):
        act = chooser._instantiate(rt, {"activity": "avoid", "entity_id": "ent-red", "xy": np.array([0.5, 0.0]),
                                         "basis": "", "value": 5.0}, 2.0 + k)
        act.cancel("stop")
        chooser._note_avoid_result(2.0 + k)
    assert chooser._avoid_state.get("ent-red", {}).get("failures", 0) == 0
    chooser._avoid_state["ent-red"] = {"failures": 3, "holds": 1, "hold_until": 50.0}
    rt.pose = np.array([-1.5, 0.0, 0.0])  # far from the entity
    chooser.candidates_for(rt, 10.0)
    assert "ent-red" not in chooser._avoid_state


def test_unreachable_and_investigated_disliked_entity_is_still_avoided():
    rt = cornered_runtime()
    rt.note_unreachable("ent-red", 2.0)
    rt._investigated.add("ent-red")
    cands = rt.chooser.candidates_for(rt, 3.0)
    assert any(c["activity"] == "avoid" and c["entity_id"] == "ent-red" for c in cands)


def test_hold_mode_finishes_as_held():
    act = Avoid("ent-red", np.array([0.5, 0.0]), hold_until=5.0)
    assert act.step(None, 4.0) == (0.0, 0.0) and not act.done
    act.step(None, 5.0)
    assert act.done and act.result["status"] == "held"


class _InstantArrival:
    """A navigator that reports arrival at once without moving (a goal within its
    arrival tolerance)."""

    status, reason = "idle", ""

    def set_goal(self, grid, pose, goal, **kw):
        self.goal, self.status = np.asarray(goal, float), "arrived"
        return True

    def step(self, grid, pose, sigma, now):
        return 0.0, 0.0


def test_a_retreat_that_does_not_leave_the_radius_is_not_a_success():
    """Regression (policy suite, start 4): just inside the radius (0.74 m) the nearest
    retreat pose was within the arrival tolerance, so avoid 'completed' every tick
    without moving, which reset the give-up count: 2000 avoid decisions in 200 s."""
    rt = cornered_runtime(dist=0.74)
    rt.planner = SimpleNamespace(traversable_xy=lambda grid, p: np.ones(len(p), bool))
    rt.nav, rt.sigma, rt.loc_status = _InstantArrival(), 0.01, "tracking"
    rt.avoid_regions = lambda: []
    started, results = run(rt, 120.0)
    assert len(started) <= 10  # gives up into holds instead of looping every tick
    assert all(r["status"] != "completed" for _, r in results)
    act = Avoid("ent-red", np.array([0.74, 0.0]), radius=0.75)
    act.step(rt, 3.0)
    assert np.linalg.norm(act.goto.goal - np.array([0.74, 0.0])) >= 1.04  # goal clearly farther out

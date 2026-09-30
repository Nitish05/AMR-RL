"""Expression policy rules and screen rendering (no simulator required)."""

from __future__ import annotations

import copy
import json
import math

import numpy as np
import pytest

from amr_rl.expression import ExpressionState, expression_from
from amr_rl.expression.policy import CAPTION_MAX, FACES, OPENNESS
from amr_rl.expression.screen import render, render_png


def make_state(**over) -> dict:
    """A tracking, autonomous snapshot engaged with a learned-liked entity."""
    s = {
        "schema": "amr_rl.state.v1",
        "authority": {"autonomy_enabled": True, "manual_active": False, "stopped": False,
                      "generation": 3, "revoked_reason": None, "heartbeat_age": 0.2},
        "localization": {"status": "tracking", "pose": [0.0, 0.0, 0.0],
                         "position_sigma": 0.03, "inliers": 80, "landmarks": 900,
                         "keyframes": 12, "map_version": "map-1"},
        "activity": {"name": "engage", "target_entity": "ent-a", "phase": "approaching",
                     "reason": "r", "since": 1.0},
        "entities": [
            {"entity_id": "ent-a", "label": "cyan cylinder", "visible": True,
             "identity_confidence": 0.93, "ambiguous": False, "position": [1.0, 0.0],
             "attitude": "liked", "expected_value": 0.6, "uncertainty": 0.1,
             "interactions": 6},
            {"entity_id": "ent-b", "label": "green box", "visible": False,
             "identity_confidence": 0.8, "ambiguous": False, "position": [0.0, 2.0],
             "attitude": "disliked", "expected_value": -0.5, "uncertainty": 0.2,
             "interactions": 3},
        ],
        "interaction": None,
    }
    for k, v in over.items():
        s[k] = v
    return s


def with_activity(name, target="ent-a", **kw):
    s = make_state()
    s["activity"] = {"name": name, "target_entity": target}
    for k, v in kw.items():
        s[k] = v
    return s


def ent(s, eid):
    return next(e for e in s["entities"] if e["entity_id"] == eid)


# --------------------------------------------------------------- face rules


@pytest.mark.parametrize(
    "activity,face",
    [("idle", "sleepy"), ("explore", "curious"), ("investigate", "focused"),
     ("avoid", "wary"), ("recovering", "confused"), ("manual", "neutral"),
     ("something_new", "neutral")],
)
def test_activity_faces(activity, face):
    assert expression_from(with_activity(activity)).face == face


def test_idle_is_sleepy_with_low_openness():
    e = expression_from(with_activity("idle", target=None))
    assert e.face == "sleepy"
    assert e.openness == pytest.approx(0.35)
    assert e.caption == "Resting"


@pytest.mark.parametrize("att,face", [("liked", "happy"), ("disliked", "wary"),
                                      ("indifferent", "neutral"), ("unknown", "curious")])
def test_engage_face_follows_learned_attitude(att, face):
    s = make_state()
    ent(s, "ent-a")["attitude"] = att
    e = expression_from(s)
    assert e.face == face
    assert e.attitude == att


def test_revisit_happy_only_when_liked():
    assert expression_from(with_activity("revisit")).face == "happy"
    assert expression_from(with_activity("revisit", target="ent-b")).face == "curious"
    s = with_activity("revisit")
    ent(s, "ent-a")["interactions"] = 0
    assert expression_from(s).face == "curious"


def test_stopped_face_unless_manual():
    s = make_state()
    s["authority"]["stopped"] = True
    s["authority"]["revoked_reason"] = "heartbeat_lost"
    e = expression_from(s)
    assert e.face == "stopped"
    assert e.caption == "Stopped (heartbeat lost)"
    assert "authority.stopped" in e.basis["face"]
    s["activity"]["name"] = "manual"
    assert expression_from(s).face == "neutral"
    s["authority"]["revoked_reason"] = "stop"
    s["activity"]["name"] = "explore"
    assert expression_from(s).caption == "Stopped"


@pytest.mark.parametrize("status", ["lost", "relocalizing"])
def test_localization_lost_is_confused_even_when_stopped(status):
    s = make_state()
    s["localization"]["status"] = status
    s["localization"]["position_sigma"] = 0.01
    e = expression_from(s)
    assert e.face == "confused" and e.uncertainty == 1.0
    s["authority"]["stopped"] = True
    e = expression_from(s)
    assert e.face == "confused"
    assert e.caption == "Lost - stopped"
    assert e.gaze == (0.0, 0.0)


# --------------------------------------------------------------- attitude gating


def test_attitude_none_without_target_and_unknown_for_missing_entity():
    assert expression_from(with_activity("explore", target=None)).attitude == "none"
    assert expression_from(with_activity("investigate", target="ent-zzz")).attitude == "none"


@pytest.mark.parametrize("att", ["liked", "disliked", "indifferent"])
@pytest.mark.parametrize("n", [0, None, "3"])
def test_learned_attitude_requires_an_interaction(att, n):
    s = make_state()
    ent(s, "ent-a")["attitude"] = att
    if n is None:
        del ent(s, "ent-a")["interactions"]
    else:
        ent(s, "ent-a")["interactions"] = n
    e = expression_from(s)
    assert e.attitude == "unknown"
    assert e.face == "curious"  # engage + unknown


def test_attitude_shown_after_one_interaction():
    s = make_state()
    ent(s, "ent-a")["interactions"] = 1
    ent(s, "ent-a")["attitude"] = "disliked"
    e = expression_from(s)
    assert e.attitude == "disliked" and e.face == "wary"
    assert "entities[target].interactions" in e.basis["attitude"]


def test_bogus_attitude_string_is_unknown():
    s = make_state()
    ent(s, "ent-a")["attitude"] = "adores"
    assert expression_from(s).attitude == "unknown"


# --------------------------------------------------------------- signal pattern


@pytest.mark.parametrize(
    "action,status,expected",
    [("signal", "acting", True), ("signal", "signalling", True),
     ("signal", "observing", False), ("signal", "done", False), ("signal", None, False),
     ("push", "acting", False), (None, "acting", False)],
)
def test_signal_pattern_exactly_tied_to_interaction(action, status, expected):
    s = make_state(interaction={"entity_id": "ent-a", "action": action, "status": status,
                                "predicted": {"yellow_flag": 0.8}, "observed": None})
    e = expression_from(s)
    assert e.signal_pattern is expected
    assert (e.face == "signalling") is expected
    if expected:
        assert e.caption == "Signalling cyan cylinder"


def test_signal_pattern_independent_of_activity_face():
    # The pattern mirrors the real emitted action even if the face is overridden.
    s = make_state(interaction={"action": "signal", "status": "acting"})
    s["authority"]["stopped"] = True
    e = expression_from(s)
    assert e.face == "stopped" and e.signal_pattern is True
    assert expression_from(make_state(interaction=None)).signal_pattern is False


# --------------------------------------------------------------- uncertainty


def test_uncertainty_from_sigma():
    s = with_activity("explore", target=None)
    s["localization"]["position_sigma"] = 0.075
    assert expression_from(s).uncertainty == pytest.approx(0.5)
    s["localization"]["position_sigma"] = 0.9
    assert expression_from(s).uncertainty == 1.0
    s["localization"]["position_sigma"] = None
    assert expression_from(s).uncertainty == 1.0


def test_uncertainty_max_with_target_and_ambiguity():
    s = make_state()
    s["localization"]["position_sigma"] = 0.015  # 0.1
    ent(s, "ent-a")["uncertainty"] = 0.4
    assert expression_from(s).uncertainty == pytest.approx(0.4)
    ent(s, "ent-a")["ambiguous"] = True
    ent(s, "ent-a")["identity_confidence"] = 0.3
    e = expression_from(s)
    assert e.uncertainty == pytest.approx(0.7)
    assert "entities[target].ambiguous" in e.basis["uncertainty"]
    ent(s, "ent-a")["identity_confidence"] = None
    assert expression_from(s).uncertainty == 1.0


# --------------------------------------------------------------- gaze


@pytest.mark.parametrize(
    "pos,theta,gx",
    [([1.0, 0.0], 0.0, 0.0),
     ([1.0, math.tan(0.325)], 0.0, -0.5),     # target 0.325 rad to robot's left
     ([1.0, -math.tan(0.325)], 0.0, 0.5),     # to the right
     ([0.0, 1.0], 0.0, -1.0),                 # 90 deg left, saturates
     ([0.0, 1.0], math.pi / 2, 0.0),          # robot facing it
     ([-1.0, 0.01], 0.0, -1.0),               # behind-left wraps positive bearing
     ([-1.0, -0.01], 0.0, 1.0)],
)
def test_gaze_from_bearing(pos, theta, gx):
    s = make_state()
    s["localization"]["pose"] = [0.0, 0.0, theta]
    ent(s, "ent-a")["position"] = pos
    e = expression_from(s)
    assert e.gaze[0] == pytest.approx(gx, abs=1e-9)
    assert e.gaze[1] == 0.0
    assert -1.0 <= e.gaze[0] <= 1.0


def test_gaze_centred_without_target_or_pose():
    assert expression_from(with_activity("explore", target=None)).gaze == (0.0, 0.0)
    s = make_state()
    s["localization"]["pose"] = None
    assert expression_from(s).gaze == (0.0, 0.0)
    s = make_state()
    ent(s, "ent-a")["position"] = None
    assert expression_from(s).gaze == (0.0, 0.0)


# --------------------------------------------------------------- captions & robustness


def test_captions():
    assert expression_from(with_activity("explore", target=None)).caption == "Exploring"
    assert expression_from(with_activity("avoid", target="ent-b")).caption == \
        "Avoiding green box"
    assert expression_from(with_activity("investigate")).caption == "Inspecting cyan cylinder"
    s = make_state()
    ent(s, "ent-a")["label"] = "an extraordinarily long fixture label here"
    cap = expression_from(s).caption
    assert len(cap) <= CAPTION_MAX and cap.endswith("...")


@pytest.mark.parametrize(
    "state",
    [None, {}, {"schema": "amr_rl.state.v1"},
     {"activity": None, "localization": None, "authority": None, "entities": None,
      "interaction": None},
     {"activity": {"name": None, "target_entity": "x"}, "entities": [None, 3, "x"]},
     {"localization": {"status": "tracking", "position_sigma": float("nan")},
      "activity": {"name": "explore"}},
     {"localization": {"status": "tracking", "pose": ["a", 0, 0], "position_sigma": True},
      "activity": {"name": "engage", "target_entity": "e"},
      "entities": [{"entity_id": "e", "position": [1]}]}],
)
def test_missing_fields_are_graceful(state):
    e = expression_from(state)
    assert e.face in FACES
    assert 0.0 <= e.uncertainty <= 1.0
    assert e.uncertainty == 1.0  # none of these has a usable position_sigma
    assert e.attitude in ("unknown", "none")
    assert e.signal_pattern is False
    assert len(e.caption) <= CAPTION_MAX
    json.dumps(e.to_dict())


def test_missing_everything_is_neutral():
    e = expression_from({})
    assert e.face == "neutral" and e.uncertainty == 1.0 and e.attitude == "none"


def test_pure_and_does_not_mutate():
    s = make_state(interaction={"action": "signal", "status": "acting"})
    before = copy.deepcopy(s)
    a, b = expression_from(s), expression_from(s)
    assert s == before
    assert a == b
    assert a.openness == OPENNESS[a.face]
    for key in ("face", "gaze", "uncertainty", "attitude", "caption", "signal_pattern"):
        assert key in a.basis


def test_to_dict_matches_interface_keys():
    d = expression_from(make_state()).to_dict()
    for key in ("face", "gaze", "uncertainty", "attitude", "caption"):
        assert key in d
    assert d["face"] == "happy" and d["attitude"] == "liked"


# --------------------------------------------------------------- rendering


def _expr(face, **kw):
    kw.setdefault("openness", OPENNESS[face])
    return ExpressionState(face=face, **kw)


def test_render_shape_dtype_and_size():
    img = render(_expr("neutral"))
    assert img.shape == (132, 192, 3) and img.dtype == np.uint8
    assert render(_expr("happy"), size=(96, 66)).shape == (66, 96, 3)
    png = render_png(_expr("curious"))
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_render_deterministic():
    e = expression_from(make_state(interaction={"action": "signal", "status": "acting"}))
    assert np.array_equal(render(e), render(e))


def test_faces_render_differently():
    imgs = {f: render(_expr(f, uncertainty=0.3, attitude="none",
                            signal_pattern=f == "signalling")) for f in FACES}
    faces = list(imgs)
    for i, a in enumerate(faces):
        for b in faces[i + 1:]:
            diff = np.abs(imgs[a].astype(int) - imgs[b].astype(int)).sum(axis=2)
            assert (diff > 60).mean() > 0.01, (a, b)


def test_gaze_moves_pupils():
    left = render(_expr("neutral", gaze=(-1.0, 0.0)))
    right = render(_expr("neutral", gaze=(1.0, 0.0)))
    assert not np.array_equal(left, right)
    # mirrored gaze gives (almost exactly) mirrored eyes region
    eyes = slice(20, 100)
    mirrored = right[eyes, ::-1].astype(int)
    assert np.abs(left[eyes].astype(int) - mirrored).mean() < 25


def _pupil_x(img):
    """Mean column of the near-white pupil glints inside the eye band."""
    band = img[25:95]
    glint = (band > 225).all(axis=2)
    assert glint.sum() > 4
    return np.nonzero(glint)[1].mean()


def test_left_target_moves_pupils_to_viewer_right():
    # Target 0.3 rad to the robot's LEFT (CCW). The screen faces outward, so the
    # robot's left is the viewer's right: pupils must move to larger image x.
    s = make_state()
    ent(s, "ent-a")["position"] = [1.0, math.tan(0.3)]
    ent(s, "ent-a")["attitude"] = "unknown"  # curious face: round eyes with pupils
    left = expression_from(s)
    assert left.gaze[0] < 0
    ent(s, "ent-a")["position"] = [1.0, 0.0]
    centre = expression_from(s)
    ent(s, "ent-a")["position"] = [1.0, -math.tan(0.3)]
    right = expression_from(s)
    xs = [_pupil_x(render(e)) for e in (left, centre, right)]
    assert xs[0] > xs[1] + 3 > xs[2] + 6


def test_uncertainty_bar_width_proportional():
    def filled(u):
        row = render(_expr("neutral", uncertainty=u))[132 - 7, 17:170].astype(int)
        return int((np.abs(row - np.array([38, 46, 60])).sum(axis=1) > 60).sum())

    a, b, c = filled(0.0), filled(0.5), filled(1.0)
    assert a == 0
    assert 0.35 < b / c < 0.65


def test_attitude_pip_colours():
    def pip(att):
        return render(_expr("neutral", attitude=att))[10, 192 - 10].astype(int)

    liked, disliked = pip("liked"), pip("disliked")
    assert liked[1] > 150 and liked[0] < 120
    assert disliked[0] > 150 and disliked[1] < 100
    none, unknown = pip("none"), pip("unknown")
    assert none.max() < 40  # background
    assert unknown.max() < 60  # outline only: centre stays dark
    ring = render(_expr("neutral", attitude="unknown"))[5:16, 192 - 16:192 - 4]
    assert ring.max() > 200


def test_signal_pattern_is_large_and_only_when_flagged():
    on = render(_expr("signalling", signal_pattern=True))
    off = render(_expr("signalling", signal_pattern=False))
    yellow = (on[..., 0] > 200) & (on[..., 1] > 180) & (on[..., 2] < 80)
    magenta = (on[..., 0] > 200) & (on[..., 1] < 90) & (on[..., 2] > 130)
    assert (yellow | magenta).mean() > 0.2
    y_off = (off[..., 0] > 200) & (off[..., 1] > 180) & (off[..., 2] < 80)
    assert y_off.mean() < 0.01

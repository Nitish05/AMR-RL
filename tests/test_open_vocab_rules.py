"""The engineered rules around the open-vocabulary detector (no model weights needed):
floor contact, flag attachment and flag colour (perception/open_vocab.py)."""

import numpy as np
import pytest

from amr_rl.perception.camera_model import CameraModel
from amr_rl.perception.open_vocab import OpenVocabDetector
from amr_rl.robot.spec import RobotSpec


@pytest.fixture(scope="module")
def det():
    d = OpenVocabDetector.__new__(OpenVocabDetector)  # rules only: no network weights
    d.model = CameraModel.from_spec(RobotSpec.load())
    d.labels, d.flag_label, d.flag_mode = ["box", "cylinder", "ball"], "flag", "detect"
    d.threshold, d.flag_threshold, d.flag_color_share = 0.25, 0.15, 0.25
    d.max_range, d.min_box_px, d.nms_iou = 3.0, 6, 0.5
    d.flag_band, d.flag_min_share = 0.7, 0.12
    return d


def frame(det, top_color=None):
    H, W = det.model.height, det.model.width
    rgb = np.full((H, W, 3), 120, np.uint8)
    rgb[150:200, 140:180] = (40, 160, 200)  # object
    if top_color is not None:
        rgb[120:150, 150:170] = top_color  # flag standing on it
    return rgb, (140, 150, 180, 200), (150, 120, 170, 152)


def run(det, rgb, props):
    det.proposals = lambda _rgb: props
    return det.detect(rgb, pose=(0.0, 0.0, 0.0))


def test_flag_box_on_the_object_top_is_its_attachment(det):
    rgb, obj, flag = frame(det, (240, 200, 20))
    out = run(det, rgb, [(np.array(obj, float), 0.6, "box"), (np.array(flag, float), 0.3, "flag")])
    assert len(out) == 1 and out[0].state_token == "attach:yellow"
    rgb, obj, flag = frame(det, (220, 20, 15))
    out = run(det, rgb, [(np.array(obj, float), 0.6, "box"), (np.array(flag, float), 0.3, "flag")])
    assert out[0].state_token == "attach:red"


def test_flag_box_elsewhere_or_without_flag_colour_is_ignored(det):
    rgb, obj, _ = frame(det, (240, 200, 20))
    far = np.array([20, 120, 40, 152], float)  # not above the object
    out = run(det, rgb, [(np.array(obj, float), 0.6, "box"), (far, 0.5, "flag")])
    assert out[0].state_token == "attach:none"
    rgb, obj, flag = frame(det, (40, 160, 200))  # a "flag" box over the object's own colour
    out = run(det, rgb, [(np.array(obj, float), 0.6, "box"), (np.array(flag, float), 0.5, "flag")])
    assert out[0].state_token == "attach:none"


def test_objects_must_stand_on_the_floor_within_range(det):
    rgb, obj, _ = frame(det)
    poster = np.array([140, 10, 200, 40], float)  # bottom edge above the horizon: no floor contact
    out = run(det, rgb, [(np.array(obj, float), 0.6, "box"), (poster, 0.6, "box")])
    assert [d.bbox for d in out] == [tuple(obj)]
    assert out[0].position is not None and 0.2 < out[0].range_m < 3.0

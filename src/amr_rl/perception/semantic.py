"""Asynchronous semantic annotation with stale/authority rejection.

Semantic backends run off the control path in a worker thread. They receive an
image crop and return ONLY a short label and attribute strings. Results are
admitted only if (a) the job's authority generation is still current, (b) the
result arrives within ``max_age`` of its source frame (in the runtime's clock),
and (c) the entity still exists. Semantic output can never carry coordinates or
commands: the result schema is validated and anything else is rejected.

The default backend is the engineered ``FixtureDescriber`` (colour name + coarse
shape from the crop). It is cheap; a slower learned model can be plugged in via
the same interface, and tests inject slow/misbehaving backends.
"""

from __future__ import annotations

import queue
import re
import threading
from dataclasses import dataclass

import cv2
import numpy as np

from .entities import circular_mean_deg, hue_name

LABEL_RE = re.compile(r"^[a-z][a-z \-]{0,40}$")


@dataclass
class SemanticJob:
    job_id: int
    entity_id: str
    crop: np.ndarray
    frame_time: float
    generation: int


@dataclass
class SemanticResult:
    job: SemanticJob
    label: str | None
    attributes: dict
    error: str | None = None


class FixtureDescriber:
    name = "fixture-describer-v1 (engineered)"

    def describe(self, crop: np.ndarray) -> dict:
        hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
        mask = (hsv[..., 1] > 120) & (hsv[..., 2] > 45)
        if mask.sum() < 20:
            return {"label": "unclear object", "attributes": {"shape": "unknown"}}
        color = hue_name(circular_mean_deg(hsv[..., 0][mask].astype(float) * 2))
        ys, xs = np.nonzero(mask)
        h, w = int(np.ptp(ys)) + 1, int(np.ptp(xs)) + 1
        fill = mask.sum() / float(h * w)
        rows = [mask[y].sum() for y in range(ys.min(), ys.max() + 1)]
        taper = (rows[len(rows) // 2] + 1) / (max(rows[0], rows[-1]) + 1)
        if fill < 0.83 and taper > 1.25:
            shape = "ball"
        elif fill > 0.9 and abs(w / h - 1) < 0.5:
            shape = "block"
        else:
            shape = "cylinder" if fill < 0.93 else "block"
        return {"label": f"{color} {shape}", "attributes": {"shape": shape, "color": color}}


class SemanticWorker:
    def __init__(self, backend=None, *, max_age: float = 3.0, max_pending: int = 4):
        self.backend = backend or FixtureDescriber()
        self.max_age = max_age
        self.jobs: queue.Queue = queue.Queue(maxsize=max_pending)
        self.results: queue.Queue = queue.Queue()
        self.counter = 0
        self.stats = {"submitted": 0, "accepted": 0, "rejected_stale": 0, "rejected_authority": 0,
                      "rejected_schema": 0, "rejected_unknown_entity": 0, "dropped_busy": 0, "errors": 0}
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._run, name="semantic-worker", daemon=True)
        self.thread.start()

    def submit(self, entity_id, crop, frame_time, generation) -> bool:
        self.counter += 1
        job = SemanticJob(self.counter, entity_id, np.ascontiguousarray(crop), frame_time, generation)
        try:
            self.jobs.put_nowait(job)
        except queue.Full:
            self.stats["dropped_busy"] += 1
            return False
        self.stats["submitted"] += 1
        return True

    def _run(self):
        while not self._stop.is_set():
            try:
                job = self.jobs.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                out = self.backend.describe(job.crop)
                self.results.put(SemanticResult(job, out.get("label"), out.get("attributes", {})))
            except Exception as error:  # noqa: BLE001 - backend faults are contained
                self.results.put(SemanticResult(job, None, {}, error=f"{type(error).__name__}: {error}"))

    @staticmethod
    def _valid_schema(result: SemanticResult) -> bool:
        if result.label is None or not isinstance(result.label, str) or not LABEL_RE.match(result.label):
            return False
        attrs = result.attributes
        if not isinstance(attrs, dict) or len(attrs) > 8:
            return False
        return all(isinstance(k, str) and isinstance(v, str) and len(v) <= 32 and LABEL_RE.match(v)
                   for k, v in attrs.items())

    def collect(self, *, now: float, generation: int, known_entities) -> list[SemanticResult]:
        admitted = []
        while True:
            try:
                result = self.results.get_nowait()
            except queue.Empty:
                break
            if result.error:
                self.stats["errors"] += 1
            elif result.job.generation != generation:
                self.stats["rejected_authority"] += 1
            elif now - result.job.frame_time > self.max_age:
                self.stats["rejected_stale"] += 1
            elif result.job.entity_id not in known_entities:
                self.stats["rejected_unknown_entity"] += 1
            elif not self._valid_schema(result):
                self.stats["rejected_schema"] += 1
            else:
                self.stats["accepted"] += 1
                admitted.append(result)
        return admitted

    def close(self):
        self._stop.set()
        self.thread.join(timeout=1.0)

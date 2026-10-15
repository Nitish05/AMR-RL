"""Session-scoped ledger of approaches that could not confirm their target (engineered).

Two holds, both switchable in ``ChooserConfig`` and off until the pre-registered suite of
docs/results/r9-followup.md §12 decides; every event is logged (``events``) and the learning
evaluation stores them with the run.

* ``unconfirmed_hold`` (count, 0 = off): an option (entity, action) whose last
  ``unconfirmed_hold`` approaches in a row ended without the target confirmed in fresh images,
  and which was never confirmed in this session, is not offered again this session. An option
  confirmed earlier keeps only the ordinary unreachable back-off: its misses are chance, not a
  property of the object. In the 2026-10-12 suite the textured crate was approached 155 times
  and confirmed never (at the 0.75 m standoff its top is cut by the image border); on that
  suite's interaction records two misses would have held 48 such entities and no fixture that
  later completed an outcome.
* ``absent_hold`` (bool): an entity whose remembered place was in view through the whole
  confirmation window without one sighting of it is "absent" from that place: not offered
  while it is still remembered there (within ``ABSENT_RADIUS``) and not currently seen. A
  sighting, a new remembered position or a confirmed interaction lifts the hold (the roller
  rolls away; detector slivers vanish).
"""

from __future__ import annotations

import math

ABSENT_RADIUS = 0.3  # m: "the same remembered place"
ABSENT_FRAMES = 5  # confirmation frames with the place in view and nothing seen (ticks of 0.1 s)


class TargetLedger:
    def __init__(self, cfg):
        self.cfg = cfg
        self._misses = {}  # (entity_id, action) -> consecutive approaches that did not confirm
        self._confirmed = set()  # (entity_id, action) confirmed in this session
        self._absent = {}  # entity_id -> remembered (x, y) where nothing was found
        self.events = []

    def note_unconfirmed(self, entity_id, action, now, xy=None, absent=False):
        key = (entity_id, action)
        self._misses[key] = self._misses.get(key, 0) + 1
        event = {"t": float(now), "entity_id": entity_id, "action": action,
                 "event": "absent" if absent else "unconfirmed", "count": self._misses[key]}
        if absent and self.cfg.absent_hold and xy is not None:
            self._absent[entity_id] = (float(xy[0]), float(xy[1]))
        if self.held(entity_id, action) and self._misses[key] == self.cfg.unconfirmed_hold:
            event["held"] = True
        self.events.append(event)

    def note_confirmed(self, entity_id, action):
        """The target was confirmed in fresh images at this action's standoff."""
        self._confirmed.add((entity_id, action))
        self._misses.pop((entity_id, action), None)
        self._absent.pop(entity_id, None)

    def held(self, entity_id, action):
        hold = int(self.cfg.unconfirmed_hold or 0)
        return hold > 0 and (entity_id, action) not in self._confirmed and \
            self._misses.get((entity_id, action), 0) >= hold

    def absent(self, entity_id, xy, visible):
        if not self.cfg.absent_hold or visible:
            return False
        at = self._absent.get(entity_id)
        return at is not None and math.hypot(float(xy[0]) - at[0], float(xy[1]) - at[1]) < ABSENT_RADIUS

    def absence_verdict(self, seen, empty_frames):
        """Was a confirmation window that ran out an absence? Only with the switch on."""
        return bool(self.cfg.absent_hold) and not seen and empty_frames >= ABSENT_FRAMES

    def snapshot(self):
        return {"held": sorted(f"{e}:{a}" for e, a in self._misses if self.held(e, a)),
                "absent": sorted(self._absent)}

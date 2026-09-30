"""Persistent experience memory and outcome learning (SQLite).

Generalised from BB8-RL ``purpose.py`` (fixed stations x scalar resource) to
visually identified entities x actions x visible context -> discrete observed
consequences. Retained mechanisms from BB8: atomic event insertion with durable
duplicate-receipt rejection, windowed evidence, bounded probe budgets (no
endless probing of ineffective options), early/late change hypotheses that
reopen suppressed alternatives, and a capped one-step value of information.
Added: a time-decayed evidence window (non-stationarity prior), context backoff,
entity identity records scoped by map version, and a learning-update log.

What LEARNS here (updated only in ``record_outcome`` after a validated,
image-grounded, authorised interaction receipt has been retained):
  * P(outcome | entity, action, context) - windowed, time-decayed Dirichlet
  * per-option probe budget, success/failure streaks, stability flag
  * derived: expected valence per option and attitude per entity
What is ENGINEERED (configuration, never learned): the outcome valence table,
priors, window, half-life, probe budget, thresholds, action costs.
Entity appearance/position records are updated by perception (identity), which
is separate from outcome learning.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path

SCHEMA_VERSION = 1
COLORS = ("red", "orange", "yellow", "green", "cyan", "blue", "violet", "magenta")
OUTCOMES = ("none", "moved") + tuple(f"attach:{c}" for c in COLORS)
ACTIONS = ("signal", "nudge")


@dataclass
class LearningConfig:
    valence: dict = field(default_factory=lambda: {
        "none": 0.0, "moved": 0.6, "attach:yellow": 1.0, "attach:red": -1.0,
        **{f"attach:{c}": 0.2 for c in COLORS if c not in ("yellow", "red")},
    })
    prior_none: float = 0.6
    prior_strength: float = 2.0
    context_backoff: float = 2.0
    window: int = 12
    half_life: float = 900.0  # sim seconds; evidence weight halves
    useful_valence: float = 0.3
    initial_probes: int = 3
    early_change_useful: int = 2
    change_failures: int = 3
    stable_useful: int = 6
    max_information_value: float = 0.2
    curiosity: float = 0.5  # engineered weight on outcome uncertainty (exploration bonus)
    change_discount: float = 0.25  # weight of evidence older than a detected consequence change
    entity_backoff: float = 1.5  # pseudo-count weight of this entity's other actions (generalisation)


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


class ExperienceMemory:
    """One agent's durable memory. Knowledge survives restart; authority does not
    live here at all."""

    def __init__(self, path, *, agent_id: str | None = None, config: LearningConfig | None = None):
        self.cfg = config or LearningConfig()
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(self.path, timeout=10, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA busy_timeout=10000")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS entities(
                entity_id TEXT PRIMARY KEY, created REAL NOT NULL, appearance TEXT NOT NULL,
                label TEXT, movable INTEGER NOT NULL DEFAULT 0, n_obs INTEGER NOT NULL DEFAULT 0,
                last_seen REAL, map_version TEXT, x REAL, y REAL, pos_sigma REAL);
            CREATE TABLE IF NOT EXISTS options(
                entity_id TEXT NOT NULL, action TEXT NOT NULL, context TEXT NOT NULL,
                window TEXT NOT NULL DEFAULT '[]', outcomes INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY(entity_id, action, context));
            CREATE TABLE IF NOT EXISTS budgets(
                entity_id TEXT NOT NULL, action TEXT NOT NULL,
                probes INTEGER NOT NULL, useful_streak INTEGER NOT NULL DEFAULT 0,
                failure_streak INTEGER NOT NULL DEFAULT 0, stable INTEGER NOT NULL DEFAULT 0,
                last_outcome REAL, last_renewal REAL, proposals INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY(entity_id, action));
            CREATE TABLE IF NOT EXISTS events(
                seq INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT NOT NULL UNIQUE,
                entity_id TEXT NOT NULL, action TEXT NOT NULL, context TEXT NOT NULL,
                observed TEXT NOT NULL, valence REAL NOT NULL, t REAL NOT NULL,
                receipt_sha256 TEXT NOT NULL, receipt TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS updates(
                seq INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT NOT NULL,
                entity_id TEXT NOT NULL, action TEXT NOT NULL, detail TEXT NOT NULL);
        """)
        with self.db:
            existing = self._meta("schema_version")
            if existing is None:
                self._set_meta("schema_version", str(SCHEMA_VERSION))
                self._set_meta("agent_id", agent_id or f"pip-{uuid.uuid4().hex[:8]}")
                self._set_meta("change_epoch", "0")
                self._set_meta("sessions", "0")
            elif int(existing) != SCHEMA_VERSION:
                raise ValueError(f"Unsupported memory schema {existing}")
            elif agent_id is not None and self._meta("agent_id") != agent_id:
                raise ValueError("Memory belongs to a different agent identity")
            self._set_meta("sessions", str(int(self._meta("sessions")) + 1))
        self.agent_id = self._meta("agent_id")

    # ------------------------------------------------------------ meta
    def _meta(self, key):
        row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return None if row is None else row[0]

    def _set_meta(self, key, value):
        self.db.execute("INSERT INTO meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                        (key, value))

    def close(self):
        with self.lock:
            self.db.close()

    @property
    def sessions(self) -> int:
        return int(self._meta("sessions"))

    # ------------------------------------------------------------ entities
    def upsert_entity(self, entity_id, *, appearance, now, map_version=None, xy=None, pos_sigma=None,
                      label=None, n_obs_increment=1):
        with self.lock, self.db:
            row = self.db.execute("SELECT * FROM entities WHERE entity_id=?", (entity_id,)).fetchone()
            if row is None:
                self.db.execute(
                    "INSERT INTO entities(entity_id,created,appearance,label,n_obs,last_seen,map_version,x,y,pos_sigma)"
                    " VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (entity_id, now, canonical(appearance), label, n_obs_increment, now, map_version,
                     None if xy is None else float(xy[0]), None if xy is None else float(xy[1]), pos_sigma))
            else:
                self.db.execute(
                    "UPDATE entities SET appearance=?, n_obs=n_obs+?, last_seen=?, label=COALESCE(?,label),"
                    " map_version=COALESCE(?,map_version), x=COALESCE(?,x), y=COALESCE(?,y),"
                    " pos_sigma=COALESCE(?,pos_sigma) WHERE entity_id=?",
                    (canonical(appearance), n_obs_increment, now, label, map_version,
                     None if xy is None else float(xy[0]), None if xy is None else float(xy[1]), pos_sigma,
                     entity_id))

    def set_label(self, entity_id, label):
        with self.lock, self.db:
            self.db.execute("UPDATE entities SET label=? WHERE entity_id=?", (label, entity_id))

    def mark_movable(self, entity_id):
        with self.lock, self.db:
            self.db.execute("UPDATE entities SET movable=1 WHERE entity_id=?", (entity_id,))

    def entities(self):
        with self.lock:
            rows = self.db.execute("SELECT * FROM entities ORDER BY created").fetchall()
        out = []
        for row in rows:
            item = dict(row)
            item["appearance"] = json.loads(item["appearance"])
            out.append(item)
        return out

    def invalidate_positions(self, map_version):
        """A new map version makes remembered positions meaningless; identities,
        appearances and learned expectations survive."""
        with self.lock, self.db:
            self.db.execute("UPDATE entities SET x=NULL, y=NULL, pos_sigma=NULL WHERE map_version IS NOT ?",
                            (map_version,))

    # ------------------------------------------------------------ prediction
    def _budget(self, entity_id, action):
        row = self.db.execute("SELECT * FROM budgets WHERE entity_id=? AND action=?", (entity_id, action)).fetchone()
        if row is None:
            self.db.execute("INSERT INTO budgets(entity_id,action,probes) VALUES(?,?,?)",
                            (entity_id, action, self.cfg.initial_probes))
            row = self.db.execute("SELECT * FROM budgets WHERE entity_id=? AND action=?",
                                  (entity_id, action)).fetchone()
        return row

    def _weighted_counts(self, windows, now):
        counts = {o: 0.0 for o in OUTCOMES}
        total = 0.0
        change_time = float(self._meta("change_time") or "-inf")
        for window in windows:
            for token, _, t in window:
                w = 0.5 ** (max(0.0, now - t) / self.cfg.half_life)
                if t < change_time:
                    w *= self.cfg.change_discount  # evidence from before a detected change
                counts[token] = counts.get(token, 0.0) + w
                total += w
        return counts, total

    def predict(self, entity_id, action, context, now):
        """Posterior predictive over OUTCOMES with hierarchical backoff:
        (entity, action, context) -> (entity, action) -> (entity, any action) -> prior."""
        with self.lock:
            rows = self.db.execute("SELECT action, context, window FROM options WHERE entity_id=?",
                                   (entity_id,)).fetchall()
        windows = {r["context"]: json.loads(r["window"]) for r in rows if r["action"] == action}
        others = [json.loads(r["window"]) for r in rows if r["action"] != action]
        prior = {o: (self.cfg.prior_none if o == "none" else (1 - self.cfg.prior_none) / (len(OUTCOMES) - 1))
                 for o in OUTCOMES}
        ent, n_ent = self._weighted_counts(others, now)
        ke = self.cfg.entity_backoff
        entity_prior = {o: (ke * ent[o] / max(n_ent, 1e-9) * min(1.0, n_ent) + self.cfg.prior_strength * prior[o])
                        / (ke * min(1.0, n_ent) + self.cfg.prior_strength) for o in OUTCOMES}
        agg, n_agg = self._weighted_counts(windows.values(), now)
        k0 = self.cfg.prior_strength
        base = {o: (agg[o] + k0 * entity_prior[o]) / (n_agg + k0) for o in OUTCOMES}
        ctx, n_ctx = self._weighted_counts([windows.get(context, [])], now)
        kb = self.cfg.context_backoff
        probs = {o: (ctx[o] + kb * base[o]) / (n_ctx + kb) for o in OUTCOMES}
        ev = sum(p * self.cfg.valence.get(o, 0.0) for o, p in probs.items())
        second = sum(p * self.cfg.valence.get(o, 0.0) ** 2 for o, p in probs.items())
        spread = math.sqrt(max(0.0, second - ev * ev))
        effective = n_ctx + 0.5 * (n_agg - n_ctx)
        uncertainty = spread / math.sqrt(1.0 + effective)
        return {"probabilities": probs, "expected_valence": ev, "uncertainty": uncertainty,
                "evidence": {"context": n_ctx, "all_contexts": n_agg}}

    def option_state(self, entity_id, action, now):
        with self.lock, self.db:
            row = self._budget(entity_id, action)
            # Bounded renewal: evidence older than one half-life earns one fresh probe,
            # at most once per half-life (non-stationarity, not a wandering timer).
            if row["probes"] == 0 and row["last_outcome"] is not None:
                last = row["last_renewal"] if row["last_renewal"] is not None else row["last_outcome"]
                if now - max(last, row["last_outcome"]) >= self.cfg.half_life:
                    self.db.execute("UPDATE budgets SET probes=1, last_renewal=? WHERE entity_id=? AND action=?",
                                    (now, entity_id, action))
                    row = self._budget(entity_id, action)
            return dict(row)

    def information_value(self, prediction, best_other):
        """Capped value of information: one-step lookahead plus an engineered
        uncertainty bonus. Only the uncertainty itself is learned (it shrinks with
        evidence); the weight and cap are configuration."""
        p = prediction["probabilities"]
        ev = prediction["expected_valence"]
        n = prediction["evidence"]["context"] + self.cfg.context_backoff
        after = 0.0
        for o, po in p.items():
            v_o = self.cfg.valence.get(o, 0.0)
            ev_after = (ev * n + v_o) / (n + 1)
            after += po * max(best_other, ev_after)
        gain = after - max(best_other, ev)
        bonus = self.cfg.curiosity * prediction["uncertainty"]
        return max(0.0, min(self.cfg.max_information_value, gain + bonus))

    def note_proposal(self, entity_id, action):
        with self.lock, self.db:
            self._budget(entity_id, action)
            self.db.execute("UPDATE budgets SET proposals=proposals+1 WHERE entity_id=? AND action=?",
                            (entity_id, action))

    # ------------------------------------------------------------ learning
    def record_outcome(self, receipt: dict) -> dict | None:
        """Atomically store one validated receipt and update beliefs.

        Returns the learning update, or None for an exact replay of an already
        stored receipt. A conflicting receipt with the same event id raises.
        """
        required = {"event_id", "entity_id", "action", "context", "observed", "timestamp"}
        if not required.issubset(receipt):
            raise ValueError("Receipt missing required fields")
        if receipt["action"] not in ACTIONS or receipt["observed"] not in OUTCOMES:
            raise ValueError("Unknown action or outcome token")
        if not receipt.get("authorised") or not receipt.get("images_retained"):
            raise ValueError("Only authorised receipts with retained image evidence can train")
        payload = canonical(receipt)
        digest = hashlib.sha256(payload.encode()).hexdigest()
        e, a, c, o, t = (receipt[k] for k in ("entity_id", "action", "context", "observed", "timestamp"))
        valence = float(self.cfg.valence.get(o, 0.0))
        with self.lock, self.db:
            self.db.execute("BEGIN IMMEDIATE")
            old = self.db.execute("SELECT receipt_sha256 FROM events WHERE event_id=?",
                                  (receipt["event_id"],)).fetchone()
            if old is not None:
                if old[0] != digest:
                    raise ValueError("Event id already stored with a conflicting receipt")
                return None
            if self.db.execute("SELECT 1 FROM entities WHERE entity_id=?", (e,)).fetchone() is None:
                raise ValueError("Receipt refers to an unknown entity")
            before = self.predict(e, a, c, t)
            self.db.execute(
                "INSERT INTO events(event_id,entity_id,action,context,observed,valence,t,receipt_sha256,receipt)"
                " VALUES(?,?,?,?,?,?,?,?,?)", (receipt["event_id"], e, a, c, o, valence, t, digest, payload))
            row = self.db.execute("SELECT window FROM options WHERE entity_id=? AND action=? AND context=?",
                                  (e, a, c)).fetchone()
            window = json.loads(row["window"]) if row else []
            window = (window + [[o, valence, t]])[-self.cfg.window:]
            self.db.execute(
                "INSERT INTO options(entity_id,action,context,window,outcomes) VALUES(?,?,?,?,1)"
                " ON CONFLICT(entity_id,action,context) DO UPDATE SET window=excluded.window, outcomes=outcomes+1",
                (e, a, c, canonical(window)))
            budget = dict(self._budget(e, a))
            useful = valence >= self.cfg.useful_valence
            useful_streak = budget["useful_streak"] + 1 if useful else 0
            failure_streak = 0 if useful else budget["failure_streak"] + 1
            established = budget["stable"] or budget["useful_streak"] >= self.cfg.early_change_useful
            stable = bool(budget["stable"]) or useful_streak >= self.cfg.stable_useful
            probes = self.cfg.initial_probes if useful else max(0, budget["probes"] - 1)
            change = False
            if failure_streak == 1 and established:
                # Remember that this failure run follows an established effect.
                self._set_meta(f"armed:{e}:{a}", "1")
                self._set_meta(f"armed_time:{e}:{a}", repr(float(t)))
            if failure_streak == self.cfg.change_failures and self._meta(f"armed:{e}:{a}") == "1":
                change = True
                stable = False
                self._set_meta(f"armed:{e}:{a}", "0")
                epoch = int(self._meta("change_epoch")) + 1
                self._set_meta("change_epoch", str(epoch))
                self._set_meta("change_time", self._meta(f"armed_time:{e}:{a}") or repr(float(t)))
                self.db.execute("UPDATE budgets SET probes=1 WHERE probes=0 AND NOT (entity_id=? AND action=?)",
                                (e, a))
            if useful:
                self._set_meta(f"armed:{e}:{a}", "0")
            self.db.execute(
                "UPDATE budgets SET probes=?, useful_streak=?, failure_streak=?, stable=?, last_outcome=?"
                " WHERE entity_id=? AND action=?",
                (probes, useful_streak, failure_streak, int(stable), t, e, a))
            if o == "moved":
                self.db.execute("UPDATE entities SET movable=1 WHERE entity_id=?", (e,))
            after = self.predict(e, a, c, t)
            detail = {
                "expected_valence": {"before": before["expected_valence"], "after": after["expected_valence"]},
                "p_observed": {"before": before["probabilities"][o], "after": after["probabilities"][o]},
                "probes": {"before": budget["probes"], "after": probes},
                "useful": useful, "valence": valence, "change_hypothesis": change,
                "surprise": -math.log(max(before["probabilities"][o], 1e-6)),
            }
            self.db.execute("INSERT INTO updates(event_id,entity_id,action,detail) VALUES(?,?,?,?)",
                            (receipt["event_id"], e, a, canonical(detail)))
        return detail

    # ------------------------------------------------------------ summaries
    def attitude(self, entity_id, now, context="attach:none"):
        with self.lock:
            n = self.db.execute("SELECT COUNT(*) FROM events WHERE entity_id=?", (entity_id,)).fetchone()[0]
        if n == 0:
            return {"attitude": "unknown", "expected_value": None, "uncertainty": None, "interactions": 0,
                    "best_action": None}
        best = None
        for action in ACTIONS:
            pred = self.predict(entity_id, action, context, now)
            with self.lock:
                tried = self.db.execute("SELECT COUNT(*) FROM events WHERE entity_id=? AND action=?",
                                        (entity_id, action)).fetchone()[0]
            if tried == 0:
                continue
            if best is None or pred["expected_valence"] > best[1]["expected_valence"]:
                best = (action, pred)
        worst = min((self.predict(entity_id, a, context, now)["expected_valence"] for a in ACTIONS), default=0.0)
        action, pred = best
        ev = pred["expected_valence"]
        if ev >= 0.3:
            label = "liked"
        elif ev <= -0.25 or worst <= -0.4:
            label = "disliked"
        elif n >= 3:
            label = "indifferent"
        else:
            label = "unknown"
        return {"attitude": label, "expected_value": ev, "uncertainty": pred["uncertainty"],
                "interactions": n, "best_action": action}

    def recent_events(self, limit=12):
        with self.lock:
            rows = self.db.execute(
                "SELECT e.*, u.detail FROM events e LEFT JOIN updates u ON u.event_id=e.event_id"
                " ORDER BY e.seq DESC LIMIT ?", (limit,)).fetchall()
        out = []
        for row in rows:
            receipt = json.loads(row["receipt"])
            out.append({
                "event_id": row["event_id"], "entity_id": row["entity_id"], "action": row["action"],
                "context": row["context"], "observed": row["observed"], "valence": row["valence"],
                "timestamp": row["t"], "predicted": receipt.get("predicted"),
                "update": json.loads(row["detail"]) if row["detail"] else None,
                "receipt_sha256": row["receipt_sha256"],
            })
        return out

    def counts(self):
        with self.lock:
            return {
                "outcomes": self.db.execute("SELECT COUNT(*) FROM events").fetchone()[0],
                "entities": self.db.execute("SELECT COUNT(*) FROM entities").fetchone()[0],
                "change_epoch": int(self._meta("change_epoch")),
                "sessions": self.sessions,
            }

    def reset(self, confirm: str):
        """Explicit, irreversible memory reset for this agent (new identity)."""
        if confirm != "RESET":
            raise ValueError("Memory reset requires explicit confirmation")
        with self.lock, self.db:
            for table in ("entities", "options", "budgets", "events", "updates", "meta"):
                self.db.execute(f"DELETE FROM {table}")
            self._set_meta("schema_version", str(SCHEMA_VERSION))
            self._set_meta("agent_id", f"pip-{uuid.uuid4().hex[:8]}")
            self._set_meta("change_epoch", "0")
            self._set_meta("sessions", "1")
        self.agent_id = self._meta("agent_id")

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
    change_failures: int = 3  # minimum consecutive failures before a change is hypothesised
    change_alpha: float = 0.05  # ...and enough that the run is this unlikely under the old rate
    change_failures_max: int = 8
    stable_useful: int = 6
    max_information_value: float = 0.2
    curiosity: float = 0.5  # engineered weight on outcome uncertainty (exploration bonus)
    # Character (behavior/character.py; set by RobotRuntime when character_active). surprise_ratio:
    # evidence weight of a better-than-expected outcome over a worse-than-expected one, for
    # self-chosen actions only (bounded [0.5, 2]; 1 = unchanged). curiosity_lp: the share of the
    # uncertainty bonus that learning progress replaces (0 = unchanged; 1 = pure learning progress).
    surprise_ratio: float = 1.0
    curiosity_lp: float = 0.0
    lp_fast: float = 0.5  # EMA rates of the outcome-prediction error; learning progress = slow - fast
    lp_slow: float = 0.2
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
            CREATE TABLE IF NOT EXISTS distinct_pairs(
                a TEXT NOT NULL, b TEXT NOT NULL, t REAL NOT NULL, PRIMARY KEY(a, b));
            CREATE TABLE IF NOT EXISTS merges(
                seq INTEGER PRIMARY KEY AUTOINCREMENT, kept TEXT NOT NULL, dropped TEXT NOT NULL UNIQUE,
                t REAL NOT NULL, reason TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS character(
                key TEXT PRIMARY KEY, value REAL NOT NULL, lo REAL NOT NULL, hi REAL NOT NULL,
                anchor REAL NOT NULL, provenance TEXT NOT NULL, updated REAL NOT NULL, session INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS character_log(
                seq INTEGER PRIMARY KEY AUTOINCREMENT, t REAL NOT NULL, session INTEGER NOT NULL, key TEXT NOT NULL,
                before REAL, after REAL NOT NULL, cause TEXT NOT NULL, provenance TEXT NOT NULL, detail TEXT);
            CREATE TABLE IF NOT EXISTS character_archive(
                seq INTEGER PRIMARY KEY AUTOINCREMENT, t REAL NOT NULL, session INTEGER NOT NULL,
                agent_id TEXT NOT NULL, reason TEXT NOT NULL, snapshot TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS boots(
                seq INTEGER PRIMARY KEY AUTOINCREMENT, t REAL NOT NULL, session INTEGER NOT NULL,
                event TEXT NOT NULL, detail TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS advice(
                seq INTEGER PRIMARY KEY AUTOINCREMENT, t REAL NOT NULL, session INTEGER NOT NULL,
                trigger TEXT NOT NULL, signature TEXT NOT NULL, payload TEXT NOT NULL, proposal TEXT,
                checks TEXT NOT NULL, status TEXT NOT NULL, backend TEXT NOT NULL, baseline TEXT, verdict TEXT);
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
            # where this session's events start (the consolidation rule reads only this session)
            last = self.db.execute("SELECT COALESCE(MAX(seq), 0) FROM events").fetchone()[0]
            self._set_meta("session_start_seq", str(int(last)))
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

    # ------------------------------------------------------------ identity maintenance
    def note_distinct(self, a, b, now):
        """Two identities were seen as two separate detections in one frame: they
        are different objects and must never be merged."""
        if a == b:
            return
        a, b = sorted((a, b))
        with self.lock, self.db:
            self.db.execute("INSERT OR IGNORE INTO distinct_pairs(a,b,t) VALUES(?,?,?)", (a, b, now))

    def are_distinct(self, a, b):
        a, b = sorted((a, b))
        return self.db.execute("SELECT 1 FROM distinct_pairs WHERE a=? AND b=?", (a, b)).fetchone() is not None

    def resolve(self, entity_id):
        """Follow merges: the identity an old (merged-away) id now refers to."""
        seen = set()
        while entity_id not in seen:
            seen.add(entity_id)
            row = self.db.execute("SELECT kept FROM merges WHERE dropped=?", (entity_id,)).fetchone()
            if row is None:
                return entity_id
            entity_id = row[0]
        return entity_id

    def merge_entities(self, keep, drop, *, now, reason):
        """Fold identity ``drop`` into ``keep``: outcome windows, events, updates and
        budgets move to ``keep`` (receipts are stored unchanged; the merge is
        recorded). Refused for pairs known to be distinct."""
        if keep == drop or self.are_distinct(keep, drop):
            raise ValueError("Refusing to merge distinct or identical identities")
        with self.lock, self.db:
            self.db.execute("BEGIN IMMEDIATE")
            rows = {r["entity_id"]: dict(r) for r in self.db.execute(
                "SELECT * FROM entities WHERE entity_id IN (?,?)", (keep, drop))}
            if len(rows) != 2:
                raise ValueError("Unknown entity in merge")
            for opt in self.db.execute("SELECT * FROM options WHERE entity_id=?", (drop,)).fetchall():
                mine = self.db.execute("SELECT * FROM options WHERE entity_id=? AND action=? AND context=?",
                                       (keep, opt["action"], opt["context"])).fetchone()
                window = json.loads(opt["window"]) + (json.loads(mine["window"]) if mine else [])
                window = sorted(window, key=lambda w: w[2])[-self.cfg.window:]
                outcomes = opt["outcomes"] + (mine["outcomes"] if mine else 0)
                self.db.execute(
                    "INSERT INTO options(entity_id,action,context,window,outcomes) VALUES(?,?,?,?,?)"
                    " ON CONFLICT(entity_id,action,context) DO UPDATE SET window=excluded.window,"
                    " outcomes=excluded.outcomes", (keep, opt["action"], opt["context"], canonical(window), outcomes))
            self.db.execute("DELETE FROM options WHERE entity_id=?", (drop,))
            for bud in self.db.execute("SELECT * FROM budgets WHERE entity_id=?", (drop,)).fetchall():
                mine = self.db.execute("SELECT * FROM budgets WHERE entity_id=? AND action=?",
                                       (keep, bud["action"])).fetchone()
                if mine is None:
                    self.db.execute("UPDATE budgets SET entity_id=? WHERE entity_id=? AND action=?",
                                    (keep, drop, bud["action"]))
                elif (bud["last_outcome"] or -1) > (mine["last_outcome"] or -1):
                    # the more recent history wins the streak/probe state
                    self.db.execute("DELETE FROM budgets WHERE entity_id=? AND action=?", (keep, bud["action"]))
                    self.db.execute("UPDATE budgets SET entity_id=? WHERE entity_id=? AND action=?",
                                    (keep, drop, bud["action"]))
            self.db.execute("DELETE FROM budgets WHERE entity_id=?", (drop,))
            self.db.execute("UPDATE events SET entity_id=? WHERE entity_id=?", (keep, drop))
            self.db.execute("UPDATE updates SET entity_id=? WHERE entity_id=?", (keep, drop))
            k, d = rows[keep], rows[drop]
            self.db.execute("UPDATE entities SET created=?, movable=?, n_obs=? WHERE entity_id=?",
                            (min(k["created"], d["created"]), max(k["movable"], d["movable"]),
                             k["n_obs"] + d["n_obs"], keep))
            self.db.execute("DELETE FROM entities WHERE entity_id=?", (drop,))
            pairs = self.db.execute("SELECT a, b, t FROM distinct_pairs WHERE a=? OR b=?", (drop, drop)).fetchall()
            self.db.execute("DELETE FROM distinct_pairs WHERE a=? OR b=?", (drop, drop))
            for a, b, t_pair in pairs:
                other = b if a == drop else a
                if other != keep:
                    x, y = sorted((keep, other))
                    self.db.execute("INSERT OR IGNORE INTO distinct_pairs(a,b,t) VALUES(?,?,?)", (x, y, t_pair))
            self.db.execute("UPDATE merges SET kept=? WHERE kept=?", (keep, drop))
            self.db.execute("INSERT INTO merges(kept,dropped,t,reason) VALUES(?,?,?,?)", (keep, drop, now, reason))

    def merges(self):
        return [dict(r) for r in self.db.execute("SELECT * FROM merges ORDER BY seq")]

    def relocate_entities(self, positions: dict):
        """Move remembered positions after a map correction (loop closure): the
        entities did not move, the map frame under them was corrected."""
        with self.lock, self.db:
            for entity_id, (x, y) in positions.items():
                self.db.execute("UPDATE entities SET x=?, y=? WHERE entity_id=?", (float(x), float(y), entity_id))

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
            for token, _, t, *extra in window:
                w = 0.5 ** (max(0.0, now - t) / self.cfg.half_life)
                if t < change_time:
                    w *= self.cfg.change_discount  # evidence from before a detected change
                if extra:
                    w *= float(extra[0])  # surprise weighting stored with the outcome (character)
                counts[token] = counts.get(token, 0.0) + w
                total += w
        return counts, total

    def predict(self, entity_id, action, context, now):
        """Posterior predictive over OUTCOMES with hierarchical backoff:
        (entity, action, context) -> (entity, action) -> (entity, any action) -> prior."""
        with self.lock:
            rows = self.db.execute("SELECT action, context, window FROM options WHERE entity_id=?",
                                   (entity_id,)).fetchall()
            lp = self.learning_progress(entity_id, action) if self.cfg.curiosity_lp > 0 else 0.0
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
                "evidence": {"context": n_ctx, "all_contexts": n_agg}, "learning_progress": lp}

    def learning_progress(self, entity_id, action):
        """Oudeyer-style learning progress of one option: the slow EMA of the outcome-
        prediction error minus the fast one (how much the predictions have been improving),
        floored at 0. Learned statistic; the EMA rates are engineered."""
        raw = self._meta(f"lp:{entity_id}:{action}")
        if not raw:
            return 0.0
        fast, slow = json.loads(raw)
        return max(0.0, slow - fast)

    def _update_learning_progress(self, entity_id, action, error):
        key = f"lp:{entity_id}:{action}"
        raw = self._meta(key)
        fast, slow = json.loads(raw) if raw else (error, error)
        fast += self.cfg.lp_fast * (error - fast)
        slow += self.cfg.lp_slow * (error - slow)
        self._set_meta(key, json.dumps([fast, slow]))
        return max(0.0, slow - fast)

    def _evidence_weight(self, receipt, valence, expected):
        """Surprise weighting (character): for a self-chosen action a better-than-expected
        outcome weighs sqrt(ratio) and a worse one 1/sqrt(ratio), so the ratio of the two is
        ``surprise_ratio``; 1 for every other outcome and whenever the ratio is 1."""
        ratio = float(self.cfg.surprise_ratio)
        if ratio == 1.0 or (receipt.get("decision") or {}).get("policy") != "learned":
            return 1.0
        surprise = valence - expected
        if abs(surprise) < 1e-9:
            return 1.0
        return math.sqrt(ratio) if surprise > 0 else 1.0 / math.sqrt(ratio)

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

    def renew_probes(self, now, min_age):
        """One fresh probe for every exhausted option whose last outcome (and renewal) is
        at least ``min_age`` old (ActivityChooser frustration renewal). Returns how many."""
        with self.lock, self.db:
            rows = self.db.execute("SELECT entity_id, action, last_outcome, last_renewal FROM budgets "
                                   "WHERE probes=0 AND last_outcome IS NOT NULL").fetchall()
            n = 0
            for r in rows:
                last = max(r["last_outcome"], r["last_renewal"] if r["last_renewal"] is not None else r["last_outcome"])
                if now - last >= min_age:
                    self.db.execute("UPDATE budgets SET probes=1, last_renewal=? WHERE entity_id=? AND action=?",
                                    (now, r["entity_id"], r["action"]))
                    n += 1
            return n

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
        lp = self.cfg.curiosity_lp  # curiosity type (character): 0 = uncertainty only, 1 = learning progress only
        bonus = self.cfg.curiosity * ((1.0 - lp) * prediction["uncertainty"]
                                      + lp * prediction.get("learning_progress", 0.0))
        return max(0.0, min(self.cfg.max_information_value, gain + bonus))

    def note_proposal(self, entity_id, action):
        with self.lock, self.db:
            self._budget(entity_id, action)
            self.db.execute("UPDATE budgets SET proposals=proposals+1 WHERE entity_id=? AND action=?",
                            (entity_id, action))

    # ------------------------------------------------------------ learning
    def _failures_needed(self, entity_id, action):
        """Consecutive failures that make a consequence change credible: at least
        ``change_failures``, and enough that such a run has probability < change_alpha
        under the success rate observed before it (Laplace-smoothed). Called when the
        current outcome (the first failure) is already stored."""
        rows = self.db.execute("SELECT valence FROM events WHERE entity_id=? AND action=? ORDER BY seq DESC LIMIT ?",
                               (entity_id, action, self.cfg.window + 1)).fetchall()[1:]
        useful = sum(r[0] >= self.cfg.useful_valence for r in rows)
        p = (useful + 1) / (len(rows) + 2)
        if p >= 1.0:
            return self.cfg.change_failures
        k = math.ceil(math.log(self.cfg.change_alpha) / math.log(max(1e-9, 1.0 - p)))
        return int(min(self.cfg.change_failures_max, max(self.cfg.change_failures, k)))

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
        e = self.resolve(e)  # an identity merged away during the interaction
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
            weight = self._evidence_weight(receipt, valence, before["expected_valence"])
            progress = self._update_learning_progress(e, a, abs(valence - before["expected_valence"]))
            self.db.execute(
                "INSERT INTO events(event_id,entity_id,action,context,observed,valence,t,receipt_sha256,receipt)"
                " VALUES(?,?,?,?,?,?,?,?,?)", (receipt["event_id"], e, a, c, o, valence, t, digest, payload))
            row = self.db.execute("SELECT window FROM options WHERE entity_id=? AND action=? AND context=?",
                                  (e, a, c)).fetchone()
            window = json.loads(row["window"]) if row else []
            entry = [o, valence, t] if weight == 1.0 else [o, valence, t, weight]
            window = (window + [entry])[-self.cfg.window:]
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
                # Remember that this failure run follows an established effect, and how
                # reliable that effect was (a 50 % effect produces failure runs by chance).
                self._set_meta(f"armed:{e}:{a}", "1")
                self._set_meta(f"armed_time:{e}:{a}", repr(float(t)))
                self._set_meta(f"armed_need:{e}:{a}", str(self._failures_needed(e, a)))
            needed = int(self._meta(f"armed_need:{e}:{a}") or self.cfg.change_failures)
            if failure_streak == needed and self._meta(f"armed:{e}:{a}") == "1":
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
                "evidence_weight": weight, "learning_progress": progress,
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
                "merges": self.db.execute("SELECT COUNT(*) FROM merges").fetchone()[0],
                "change_epoch": int(self._meta("change_epoch")),
                "sessions": self.sessions,
            }

    def reset(self, confirm: str, level: str = "all", t: float = 0.0):
        """Explicit memory reset. ``level="objects"``: forget objects, places and outcomes
        but keep the identity and the character (its traits, bounds and log). ``level="all"``:
        a new identity; the character is archived (not deleted) first, so a character can be
        restored from the record. Both are recorded in the boots table."""
        if confirm != "RESET":
            raise ValueError("Memory reset requires explicit confirmation")
        if level not in ("objects", "all"):
            raise ValueError(f"unknown reset level {level!r}")
        with self.lock, self.db:
            if level == "all":
                self._archive_character(t, "reset:all")
            for table in ("entities", "options", "budgets", "events", "updates", "distinct_pairs", "merges"):
                self.db.execute(f"DELETE FROM {table}")
            if level == "all":
                for table in ("character", "character_log", "meta"):
                    self.db.execute(f"DELETE FROM {table}")
                self._set_meta("schema_version", str(SCHEMA_VERSION))
                self._set_meta("agent_id", f"pip-{uuid.uuid4().hex[:8]}")
                self._set_meta("change_epoch", "0")
                self._set_meta("sessions", "1")
            else:
                for key in [r[0] for r in self.db.execute("SELECT key FROM meta WHERE key LIKE 'armed%'")]:
                    self.db.execute("DELETE FROM meta WHERE key=?", (key,))
                self.db.execute("DELETE FROM meta WHERE key='change_time'")
            self.db.execute("INSERT INTO boots(t,session,event,detail) VALUES(?,?,?,?)",
                            (float(t), self.sessions, f"reset:{level}", "{}"))
        self.agent_id = self._meta("agent_id")

    # ------------------------------------------------------------ character store
    # Owner decisions 2026-10-12 (behavior/character.py): the engineered traits live here so
    # that they persist across restarts with the rest of the robot's memory, with bounds, an
    # anchor (the value they started from), the provenance of the last write and a log of
    # every change. Nothing reads them yet.
    def ensure_character(self, cfg, t: float = 0.0):
        """Create any missing trait rows from the config (defaults or the owner's preset).
        Existing rows are kept: the character persists; the config only seeds it."""
        self.character_cfg = cfg
        with self.lock, self.db:
            if self._meta("character_preset") is None:
                self._set_meta("character_preset", cfg.preset)
            present = {r[0] for r in self.db.execute("SELECT key FROM character")}
            for spec in cfg.traits:
                if spec.key in present:
                    continue
                value = cfg.initial(spec.key)
                prov = "owner" if spec.key in cfg.values else "engineered_default"
                self.db.execute("INSERT INTO character(key,value,lo,hi,anchor,provenance,updated,session)"
                                " VALUES(?,?,?,?,?,?,?,?)",
                                (spec.key, value, spec.lo, spec.hi, value, prov, float(t), self.sessions))
                self.db.execute("INSERT INTO character_log(t,session,key,before,after,cause,provenance,detail)"
                                " VALUES(?,?,?,?,?,?,?,?)",
                                (float(t), self.sessions, spec.key, None, value, "created", prov,
                                 json.dumps({"preset": cfg.preset})))

    def character(self):
        with self.lock:
            rows = self.db.execute("SELECT * FROM character ORDER BY key").fetchall()
        return {r["key"]: dict(r) for r in rows}

    def set_trait(self, key, value, *, cause, provenance, t, detail=None):
        """Write a trait within its bounds and its provenance's cap (character.clip_write);
        returns the value applied. Every write is logged with its cause."""
        from ..behavior.character import clip_write

        cfg = getattr(self, "character_cfg", None)
        if cfg is None:
            raise ValueError("ensure_character() first")
        with self.lock, self.db:
            row = self.db.execute("SELECT * FROM character WHERE key=?", (key,)).fetchone()
            if row is None:
                raise KeyError(key)
            applied, clipped = clip_write(row["value"], value, row["lo"], row["hi"], provenance, cfg)
            self.db.execute("UPDATE character SET value=?, provenance=?, updated=?, session=? WHERE key=?",
                            (applied, provenance, float(t), self.sessions, key))
            self.db.execute("INSERT INTO character_log(t,session,key,before,after,cause,provenance,detail)"
                            " VALUES(?,?,?,?,?,?,?,?)",
                            (float(t), self.sessions, key, row["value"], applied, cause, provenance,
                             json.dumps({**(detail or {}), "proposed": float(value), "clipped": bool(clipped)})))
        return applied

    # ------------------------------------------------------------ session-end consolidation
    def session_statistics(self):
        """Logged statistics of this session (events since the memory was opened), the inputs
        of the consolidation rule. Observer-measurable; no inner state."""
        start = int(self._meta("session_start_seq") or 0)
        with self.lock:
            rows = self.db.execute("SELECT seq, event_id, entity_id, action, valence FROM events WHERE seq>? "
                                   "ORDER BY seq", (start,)).fetchall()
            ids = [r["event_id"] for r in rows]
            changes = 0
            if ids:
                marks = ",".join("?" * len(ids))
                for (detail,) in self.db.execute(f"SELECT detail FROM updates WHERE event_id IN ({marks})", ids):
                    changes += bool(json.loads(detail).get("change_hypothesis"))
            # a probe: one of the first initial_probes outcomes an option ever had
            rank = {}
            probes = []
            for r in rows:
                n_before = self.db.execute("SELECT COUNT(*) FROM events WHERE entity_id=? AND action=? AND seq<?",
                                           (r["entity_id"], r["action"], r["seq"])).fetchone()[0]
                rank[r["seq"]] = n_before
                if n_before < self.cfg.initial_probes:
                    probes.append(r)
            per_option = {}
            for r in rows:
                per_option.setdefault((r["entity_id"], r["action"]), []).append(r["valence"])
        useful = sum(r["valence"] >= self.cfg.useful_valence for r in rows)
        aversive = sum(r["valence"] < 0 for r in rows)
        engaged = [(k, len(v)) for k, v in per_option.items() if len(v) >= 3]
        zero_lp = [k for k, _ in engaged if self.learning_progress(*k) < 0.01]
        return {"outcomes": len(rows), "useful": useful, "aversive": aversive, "change_hypotheses": changes,
                "probes": len(probes), "probes_useful": sum(r["valence"] >= self.cfg.useful_valence for r in probes),
                "options_engaged_3plus": len(engaged), "options_zero_progress": len(zero_lp),
                "distinct_options": len(per_option)}

    def consolidate_character(self, t, eta=0.1):
        """Session-end evolution of the character (owner decision 2026-10-12; docs/research/
        PERSONALITY_RESEARCH_2.md section 2): for each trait with a rule, theta <- theta + eta x
        (target(statistic) - theta), then clipped to one per-session cap (CharacterConfig.
        session_cap, by set_trait) and to the bounds. No mean reversion: a session with no
        evidence for a rule leaves its trait where it is. Every change is logged with the
        statistic, the target and the rule. Engineered rules; the statistics are logged
        behaviour. Returns the list of writes (also the no-ops, with their reason)."""
        from ..behavior.character import EVOLVES

        if getattr(self, "character_cfg", None) is None:
            return []
        s = self.session_statistics()
        n = s["outcomes"]
        current = {k: r["value"] for k, r in self.character().items()}
        targets = {}
        # caution: a bounded "reserve" of good against bad outcomes (asset protection); empty or
        # benign sessions let the reserve decay toward 0, so caution returns toward 0.5
        x = (s["useful"] - s["aversive"]) / n if n else 0.0
        reserve = float(self._meta("reserve") or 0.0)
        reserve += 0.3 * (x - reserve)
        with self.lock, self.db:
            self._set_meta("reserve", repr(reserve))
        targets["caution"] = (0.5 + 0.5 * max(-1.0, min(1.0, reserve)), {"reserve": reserve, "net_rate": x})
        # surprise weighting: symmetric evidence (ratio 1 = trait 0.5) when a rule change was
        # detected; a mild confirmation bias when the world stayed stable and paid
        if s["change_hypotheses"] > 0:
            targets["surprise_weighting"] = (0.5, {"change_hypotheses": s["change_hypotheses"]})
        elif n:
            targets["surprise_weighting"] = (0.5 + 0.1 * s["useful"] / n, {"useful_rate": s["useful"] / n})
        # curiosity type: toward learning progress when engaged objects stop being learnable
        if s["options_engaged_3plus"]:
            frac = s["options_zero_progress"] / s["options_engaged_3plus"]
            targets["curiosity_type"] = (frac, {"zero_progress_fraction": frac})
        # novelty bias: the fraction of this session's probes that paid off
        if s["probes"]:
            frac = s["probes_useful"] / s["probes"]
            targets["novelty_bias"] = (frac, {"probes_useful_fraction": frac})
        # approach_tempo and persistence: no rule yet (the perturbation meta-rule is pending)
        writes = []
        from ..behavior.character import RETIRED

        for key, (target, stat) in targets.items():
            if key not in current or key in RETIRED:  # a retired trait never changes
                continue
            proposed = current[key] + eta * (target - current[key])
            applied = self.set_trait(key, proposed, cause="session_end", provenance="consolidation", t=t,
                                     detail={"rule": EVOLVES[key], "statistic": stat, "target": target, "eta": eta,
                                             "session_statistics": s})
            writes.append({"key": key, "before": current[key], "after": applied, "target": target, "statistic": stat})
        return writes

    # ------------------------------------------------------------ advice (shadow records)
    def note(self, key, value):
        """Advisory bookkeeping (budgets, previous-session counters); never behaviour."""
        with self.lock, self.db:
            self._set_meta(f"note:{key}", str(value))

    def noted(self, key):
        return self._meta(f"note:{key}")

    def record_advice(self, t, trigger, signature, payload, proposal, checks, status, backend, baseline):
        with self.lock, self.db:
            cur = self.db.execute(
                "INSERT INTO advice(t,session,trigger,signature,payload,proposal,checks,status,backend,baseline)"
                " VALUES(?,?,?,?,?,?,?,?,?,?)",
                (float(t), self.sessions, trigger, signature, json.dumps(payload, sort_keys=True, default=str),
                 None if proposal is None else json.dumps(proposal, sort_keys=True, default=str),
                 json.dumps(checks, sort_keys=True, default=str), status, backend,
                 None if baseline is None else json.dumps(baseline, sort_keys=True, default=str)))
            return cur.lastrowid

    def advice_rows(self, limit=50):
        with self.lock:
            rows = self.db.execute("SELECT * FROM advice ORDER BY seq DESC LIMIT ?", (int(limit),)).fetchall()
        return [dict(r) for r in rows]

    def advice_by_signature(self, signature):
        with self.lock:
            row = self.db.execute("SELECT * FROM advice WHERE signature=? AND status != 'cached' ORDER BY seq DESC "
                                  "LIMIT 1", (signature,)).fetchone()
        return None if row is None else dict(row)

    def set_advice_status(self, seq, status, verdict=None):
        with self.lock, self.db:
            self.db.execute("UPDATE advice SET status=?, verdict=? WHERE seq=?",
                            (status, None if verdict is None else json.dumps(verdict, default=str), int(seq)))

    def character_log(self, limit=50):
        with self.lock:
            rows = self.db.execute("SELECT * FROM character_log ORDER BY seq DESC LIMIT ?", (int(limit),)).fetchall()
        return [dict(r) for r in rows]

    def record_boot(self, t, event, **detail):
        with self.lock, self.db:
            self.db.execute("INSERT INTO boots(t,session,event,detail) VALUES(?,?,?,?)",
                            (float(t), self.sessions, str(event), json.dumps(detail, default=float)))

    def boots(self, limit=10):
        with self.lock:
            rows = self.db.execute("SELECT * FROM boots ORDER BY seq DESC LIMIT ?", (int(limit),)).fetchall()
        return [{**dict(r), "detail": json.loads(r["detail"])} for r in rows]

    def _archive_character(self, t, reason):
        snap = {"agent_id": self.agent_id, "preset": self._meta("character_preset"), "traits": self.character(),
                "log": self.character_log(limit=1000)}
        self.db.execute("INSERT INTO character_archive(t,session,agent_id,reason,snapshot) VALUES(?,?,?,?,?)",
                        (float(t), self.sessions, self.agent_id or "", reason, json.dumps(snap, default=float)))

    def character_inspector(self, limit=20):
        """Everything the console shows about the character: values with bounds, anchors,
        the provenance of the last write, the recent log and boots. Honesty line included:
        the traits are engineered parameters, and nothing reads them yet."""
        cfg = getattr(self, "character_cfg", None)
        specs = {s.key: s for s in (cfg.traits if cfg is not None else ())}
        traits = []
        for key, r in self.character().items():
            s = specs.get(key)
            traits.append({"key": key, "value": r["value"], "lo": r["lo"], "hi": r["hi"], "anchor": r["anchor"],
                           "drift": r["value"] - r["anchor"], "provenance": r["provenance"], "updated": r["updated"],
                           "sets": None if s is None else s.sets, "analogue": None if s is None else s.analogue,
                           "retired": (s.retired or None) if s is not None else None,
                           "kind": "engineered parameter" if r["provenance"] in ("engineered_default", "owner", "migration")
                           else ("learned via an engineered update" if r["provenance"] == "consolidation"
                                 else f"{r['provenance']} (bounded, logged)")})
        log = self.character_log(limit=limit)
        with self.lock:
            counts = {r[0]: r[1] for r in self.db.execute(
                "SELECT provenance, COUNT(*) FROM character_log WHERE cause != 'created' GROUP BY provenance")}
        from ..behavior.character import EVOLVES

        advice = self.advice_rows(limit=200)
        by_status = {}
        for r in advice:
            by_status[r["status"]] = by_status.get(r["status"], 0) + 1
        last = advice[0] if advice else None
        return {"preset": self._meta("character_preset"), "traits": traits, "log": log,
                "writes_by_provenance": counts, "boots": self.boots(limit=5), "evolution": EVOLVES,
                "advice": {"mode": "shadow", "records": len(advice), "by_status": by_status, "applied": counts.get("llm_advised", 0),
                           "last": None if last is None else {k: last[k] for k in ("t", "session", "trigger", "status", "backend")},
                           "note": "proposals are stored and scored against the next session's logged statistics; "
                                   "none has been applied"},
                "reads": "no behaviour reads these values yet (store and audit only)"}

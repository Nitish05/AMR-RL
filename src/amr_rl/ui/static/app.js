/* AMR-RL operator console (vanilla JS, no build step, no external resources).
 *
 * Polls GET /api/state (schema amr_rl.state.v1, see docs/INTERFACES.md) every
 * 250 ms, refreshes the four images, sends a heartbeat every 1 s and posts
 * commands to POST /api/command. Pure helpers are exported on window.AMRUI so
 * tests/test_ui_static.js can exercise them under Node without a DOM.
 *
 * Safety conventions:
 *  - STOP is never disabled and never depends on known state.
 *  - manual / goal / enable_autonomy always carry the latest known
 *    authority.generation; without one they are not sent.
 *  - Manual drive sends at 5 Hz only while a key/button is held. If the
 *    generation changes while held (Stop, revocation), sending latches off
 *    until every drive key is released and pressed again.
 */
(function (root) {
  'use strict';

  var POLL_MS = 250;
  var HEARTBEAT_MS = 1000;
  var MANUAL_MS = 200; // 5 Hz
  var TURN_RATE = 0.8; // rad/s at full turn input

  // ------------------------------------------------------------------ helpers

  function isNum(v) {
    return typeof v === 'number' && isFinite(v);
  }

  function isGen(g) {
    return typeof g === 'number' && isFinite(g) && Math.floor(g) === g;
  }

  function clamp(v, lo, hi) {
    return v < lo ? lo : v > hi ? hi : v;
  }

  var DASH = '—';

  function fmt(v, digits) {
    if (!isNum(v)) return DASH;
    return v.toFixed(digits === undefined ? 2 : digits);
  }

  function fmtSigned(v, digits) {
    if (!isNum(v)) return DASH;
    var s = v.toFixed(digits === undefined ? 2 : digits);
    return v > 0 ? '+' + s : s;
  }

  function fmtPct(v) {
    return isNum(v) ? Math.round(v * 100) + '%' : DASH;
  }

  function fmtPose(pose) {
    if (!Array.isArray(pose) || pose.length < 3 || !pose.slice(0, 3).every(isNum)) return DASH;
    return 'x ' + pose[0].toFixed(2) + '  y ' + pose[1].toFixed(2) + '  θ ' +
      (pose[2] * 180 / Math.PI).toFixed(1) + '°';
  }

  function fmtXY(p) {
    if (!Array.isArray(p) || p.length < 2 || !isNum(p[0]) || !isNum(p[1])) return DASH;
    return '(' + p[0].toFixed(2) + ', ' + p[1].toFixed(2) + ')';
  }

  function fmtAge(sec) {
    if (!isNum(sec)) return DASH;
    return sec < 10 ? sec.toFixed(1) + ' s' : Math.round(sec) + ' s';
  }

  function escapeHtml(s) {
    if (s === null || s === undefined) return '';
    return String(s).replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
  }

  function locStatusClass(status) {
    if (status === 'tracking') return 'ok';
    if (status === 'initializing' || status === 'relocalizing') return 'warn';
    if (status === 'lost') return 'bad';
    return 'unknown';
  }

  /** Connection health from seconds since the last good /api/state. */
  function connectionLevel(ageSec) {
    if (!isNum(ageSec)) return 'bad';
    if (ageSec < 1.0) return 'ok';
    if (ageSec < 3.0) return 'warn';
    return 'bad';
  }

  function attitudeClass(att) {
    return ['liked', 'disliked', 'indifferent', 'unknown'].indexOf(att) >= 0 ? 'att-' + att
      : 'att-none';
  }

  function imgUrl(path, t) {
    return path + (path.indexOf('?') >= 0 ? '&' : '?') + 't=' + encodeURIComponent(String(t));
  }

  // ------------------------------------------------------------------ map geometry

  function validMapMeta(meta) {
    return !!meta && Array.isArray(meta.origin) && meta.origin.length >= 2 &&
      isNum(meta.origin[0]) && isNum(meta.origin[1]) && isNum(meta.resolution) &&
      meta.resolution > 0 && isNum(meta.width_px) && meta.width_px > 0 &&
      isNum(meta.height_px) && meta.height_px > 0;
  }

  /**
   * Convert a click inside the rendered map <img> box to estimated-map metres.
   *
   * clickX/clickY: offset from the element's top-left (CSS px).
   * boxW/boxH: rendered element size. natW/natH: image natural size (the image
   * is drawn with object-fit: contain, so it may be letterboxed); 0/undefined
   * means "same aspect as the map".
   * meta: state.map_meta {origin:[x0,y0], resolution, width_px, height_px}.
   * origin is the map-frame position of the image's bottom-left corner; image
   * row 0 is the maximum y (north up).
   * Returns {x, y, u, v} (u,v in map pixels) or null when outside/unavailable.
   */
  function clickToMap(clickX, clickY, boxW, boxH, natW, natH, meta) {
    if (!validMapMeta(meta) || !isNum(clickX) || !isNum(clickY) ||
        !(boxW > 0) || !(boxH > 0)) return null;
    var nw = natW > 0 ? natW : meta.width_px;
    var nh = natH > 0 ? natH : meta.height_px;
    var s = Math.min(boxW / nw, boxH / nh);
    var cw = nw * s, ch = nh * s;
    var fx = (clickX - (boxW - cw) / 2) / cw;
    var fy = (clickY - (boxH - ch) / 2) / ch;
    if (fx < 0 || fx > 1 || fy < 0 || fy > 1) return null;
    var u = fx * meta.width_px;
    var v = fy * meta.height_px;
    return {
      x: meta.origin[0] + u * meta.resolution,
      y: meta.origin[1] + (meta.height_px - v) * meta.resolution,
      u: u,
      v: v,
    };
  }

  /** Inverse of clickToMap's pixel step: map metres -> map pixel (u right, v down). */
  function mapToPixel(x, y, meta) {
    if (!validMapMeta(meta) || !isNum(x) || !isNum(y)) return null;
    return {
      u: (x - meta.origin[0]) / meta.resolution,
      v: meta.height_px - (y - meta.origin[1]) / meta.resolution,
    };
  }

  // ------------------------------------------------------------------ command payloads

  function stopCmd() { return { action: 'stop' }; }
  function heartbeatCmd() { return { action: 'heartbeat' }; }
  function disableCmd() { return { action: 'disable_autonomy' }; }

  function enableCmd(gen) {
    return isGen(gen) ? { action: 'enable_autonomy', generation: gen } : null;
  }

  function manualCmd(v, w, gen) {
    if (!isGen(gen) || !isNum(v) || !isNum(w)) return null;
    return { action: 'manual', v: v, w: w, generation: gen };
  }

  function goalCmd(x, y, gen) {
    if (!isGen(gen) || !isNum(x) || !isNum(y)) return null;
    return { action: 'goal', x: x, y: y, generation: gen };
  }

  function resetCmd(typed) {
    return typed === 'RESET' ? { action: 'reset_memory', confirm: 'RESET' } : null;
  }

  // ------------------------------------------------------------------ manual drive

  var KEY_DIRS = {
    arrowup: 'fwd', w: 'fwd',
    arrowdown: 'back', s: 'back',
    arrowleft: 'left', a: 'left',
    arrowright: 'right', d: 'right',
  };

  function keyToDir(key) {
    if (typeof key !== 'string') return null;
    return KEY_DIRS[key.toLowerCase()] || null;
  }

  /** held: iterable of 'fwd'|'back'|'left'|'right'. w > 0 turns left (CCW). */
  function driveFromDirs(held, speed, turn) {
    var set = {};
    (held ? Array.from(held) : []).forEach(function (d) { set[d] = true; });
    var fb = (set.fwd ? 1 : 0) - (set.back ? 1 : 0);
    var lr = (set.left ? 1 : 0) - (set.right ? 1 : 0);
    var v = fb * (isNum(speed) ? speed : 0.25);
    var w = lr * (isNum(turn) ? turn : TURN_RATE);
    return { v: v === 0 ? 0 : v, w: w === 0 ? 0 : w };
  }

  /**
   * Manual-session step. session: {gen, latched}. Returns the payload to send
   * this tick or null. A generation change during a held session latches it
   * off (Stop or revocation must not be undone by a key that is still down).
   */
  function manualTick(session, latestGen, held, speed, turn) {
    if (!session || session.latched) return null;
    if (!held || Array.from(held).length === 0) return null;
    if (!isGen(session.gen) || latestGen !== session.gen) {
      session.latched = true;
      return null;
    }
    var d = driveFromDirs(held, speed, turn);
    return manualCmd(d.v, d.w, session.gen);
  }

  // ------------------------------------------------------------------ view models

  function isChosen(cand, chosen) {
    if (!cand || !chosen) return false;
    var n = function (x) { return x === undefined ? null : x; };
    return cand.activity === chosen.activity && n(cand.entity_id) === n(chosen.entity_id) &&
      n(cand.action) === n(chosen.action);
  }

  /** Predicted distribution rows, highest first; observed outcome always included. */
  function distributionRows(predicted, observed) {
    var rows = [];
    var seen = {};
    if (predicted && typeof predicted === 'object') {
      Object.keys(predicted).forEach(function (k) {
        var p = isNum(predicted[k]) ? clamp(predicted[k], 0, 1) : 0;
        rows.push({ outcome: k, p: p, observed: observed === k });
        seen[k] = true;
      });
    }
    if (observed !== null && observed !== undefined && !seen[observed]) {
      rows.push({ outcome: String(observed), p: 0, observed: true });
    }
    rows.sort(function (a, b) { return b.p - a.p || (a.outcome < b.outcome ? -1 : 1); });
    return rows;
  }

  function topOutcome(predicted) {
    var rows = distributionRows(predicted, null);
    return rows.length ? rows[0] : null;
  }

  function describeResponse(cmd, resp) {
    var action = cmd && cmd.action ? cmd.action : '?';
    if (!resp || typeof resp !== 'object') return { ok: false, text: action + ': no response' };
    var ok = resp.accepted === true;
    var text = action + (ok ? ' accepted' : ' rejected');
    if (resp.reason) text += ' — ' + resp.reason;
    if (isGen(resp.generation)) text += ' (gen ' + resp.generation + ')';
    return { ok: ok, text: text };
  }

  var AMRUI = {
    POLL_MS: POLL_MS, HEARTBEAT_MS: HEARTBEAT_MS, MANUAL_MS: MANUAL_MS,
    isNum: isNum, isGen: isGen, clamp: clamp, fmt: fmt, fmtSigned: fmtSigned, fmtPct: fmtPct,
    fmtPose: fmtPose, fmtXY: fmtXY, fmtAge: fmtAge, escapeHtml: escapeHtml,
    locStatusClass: locStatusClass, connectionLevel: connectionLevel,
    attitudeClass: attitudeClass, imgUrl: imgUrl,
    validMapMeta: validMapMeta, clickToMap: clickToMap, mapToPixel: mapToPixel,
    stopCmd: stopCmd, heartbeatCmd: heartbeatCmd, disableCmd: disableCmd,
    enableCmd: enableCmd, manualCmd: manualCmd, goalCmd: goalCmd, resetCmd: resetCmd,
    keyToDir: keyToDir, driveFromDirs: driveFromDirs, manualTick: manualTick,
    isChosen: isChosen, distributionRows: distributionRows, topOutcome: topOutcome,
    describeResponse: describeResponse,
  };
  root.AMRUI = AMRUI;

  if (typeof document === 'undefined' || !document.getElementById) return;

  // ================================================================== DOM app

  var app = {
    state: null,
    lastStateAt: null, // performance.now() ms of last good state
    generation: null,
    polling: false,
    held: new Set(),
    session: null,
    manualTimer: null,
    lastHeartbeat: null, // {at, ok}
    lastManualRejectToast: 0,
  };

  function byId(id) { return document.getElementById(id); }

  function setText(id, text) {
    var el = byId(id);
    if (el) el.textContent = text;
  }

  function nowMs() { return (root.performance && performance.now) ? performance.now() : Date.now(); }

  // ---------------------------------------------------------------- toasts

  function toast(text, level) {
    var box = byId('toasts');
    if (!box) return;
    var el = document.createElement('div');
    el.className = 'toast ' + (level || 'info');
    var t = new Date();
    el.textContent = t.toTimeString().slice(0, 8) + '  ' + text;
    box.insertBefore(el, box.firstChild);
    while (box.children.length > 7) box.removeChild(box.lastChild);
    setTimeout(function () { el.classList.add('fade'); }, 7000);
    setTimeout(function () { if (el.parentNode) el.parentNode.removeChild(el); }, 8000);
  }

  // ---------------------------------------------------------------- network

  function fetchJson(url, opts, timeoutMs) {
    var ctrl = typeof AbortController !== 'undefined' ? new AbortController() : null;
    var timer = ctrl ? setTimeout(function () { ctrl.abort(); }, timeoutMs || 2000) : null;
    var o = Object.assign({ cache: 'no-store' }, opts || {});
    if (ctrl) o.signal = ctrl.signal;
    return fetch(url, o).then(function (r) {
      return r.json().catch(function () { return null; }).then(function (body) {
        if (!r.ok && !body) throw new Error('HTTP ' + r.status);
        return body;
      });
    }).finally(function () { if (timer) clearTimeout(timer); });
  }

  /** opts.quiet: only toast failures. */
  function post(cmd, opts) {
    opts = opts || {};
    if (!cmd) {
      if (!opts.quiet) toast('not sent: no authority generation known yet', 'bad');
      return Promise.resolve(null);
    }
    return fetchJson('/api/command', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(cmd),
    }, 2000).then(function (resp) {
      if (resp && isGen(resp.generation)) app.generation = resp.generation;
      var d = describeResponse(cmd, resp);
      if (!d.ok || !opts.quiet) {
        if (cmd.action === 'manual' && !d.ok) {
          if (nowMs() - app.lastManualRejectToast < 1500) return resp;
          app.lastManualRejectToast = nowMs();
        }
        toast(d.text, d.ok ? 'ok' : 'bad');
      }
      return resp;
    }).catch(function (err) {
      toast(cmd.action + ' failed: ' + (err && err.message ? err.message : err), 'bad');
      return null;
    });
  }

  // ---------------------------------------------------------------- images

  function refreshImage(id, path) {
    var img = byId(id);
    if (!img) return;
    var started = Number(img.dataset.started || 0);
    if (img.dataset.loading === '1' && nowMs() - started < 3000) return;
    img.dataset.loading = '1';
    img.dataset.started = String(nowMs());
    img.src = imgUrl(path, Date.now());
  }

  function wireImage(id) {
    var img = byId(id);
    if (!img) return;
    img.addEventListener('load', function () {
      img.dataset.loading = '0';
      img.parentNode.classList.remove('noimg');
    });
    img.addEventListener('error', function () {
      img.dataset.loading = '0';
      img.parentNode.classList.add('noimg');
    });
  }

  // ---------------------------------------------------------------- polling

  function poll() {
    if (app.polling) return;
    app.polling = true;
    fetchJson('/api/state', null, 2000).then(function (s) {
      if (s && typeof s === 'object') {
        app.state = s;
        app.lastStateAt = nowMs();
        var g = s.authority && s.authority.generation;
        if (isGen(g)) app.generation = g;
        render(s);
      }
    }).catch(function () { /* shown by connection indicator */ })
      .finally(function () { app.polling = false; });
    refreshImage('img-camera', '/api/camera.jpg');
    refreshImage('img-map', '/api/map.png');
    refreshImage('img-screen', '/api/screen.png');
    refreshImage('img-inspect', '/api/inspect.jpg');
    updateConnection();
  }

  function updateConnection() {
    var age = app.lastStateAt === null ? null : (nowMs() - app.lastStateAt) / 1000;
    var lvl = connectionLevel(age);
    var dot = byId('conn-dot');
    if (dot) dot.className = 'dot ' + lvl;
    setText('conn-text', age === null ? 'no state yet' : lvl === 'ok' ? 'live' :
      'state ' + fmtAge(age) + ' old');
    document.body.classList.toggle('stale', lvl === 'bad');
    var hb = app.lastHeartbeat;
    var hbDot = byId('hb-dot');
    if (hbDot) {
      var hbAge = hb ? (nowMs() - hb.at) / 1000 : null;
      hbDot.className = 'dot ' + (!hb ? 'bad' : !hb.ok ? 'bad' : connectionLevel(hbAge - 1));
    }
    setText('hb-text', !app.lastHeartbeat ? 'heartbeat —' :
      app.lastHeartbeat.ok ? 'heartbeat ok' : 'heartbeat failing');
  }

  function heartbeat() {
    post(heartbeatCmd(), { quiet: true }).then(function (resp) {
      app.lastHeartbeat = { at: nowMs(), ok: !!(resp && resp.accepted) };
    });
  }

  // ---------------------------------------------------------------- render

  function kv(rows) {
    return rows.map(function (r) {
      return '<div class="k">' + escapeHtml(r[0]) + '</div><div class="v' +
        (r[2] ? ' ' + r[2] : '') + '">' + escapeHtml(r[1]) + '</div>';
    }).join('');
  }

  function yesNo(b) { return b === true ? 'yes' : b === false ? 'no' : DASH; }

  function entityLabels(s) {
    var m = {};
    (Array.isArray(s.entities) ? s.entities : []).forEach(function (e) {
      if (e && e.entity_id) m[e.entity_id] = e.label || e.entity_id;
    });
    return m;
  }

  function render(s) {
    var a = s.authority || {};
    var loc = s.localization || {};
    var act = s.activity || {};
    var labels = entityLabels(s);

    // header
    setText('sim-time', 'sim ' + fmt(s.sim_time, 1) + ' s');
    setText('gen-badge', 'gen ' + (isGen(a.generation) ? a.generation : DASH));
    var auto = byId('btn-autonomy');
    if (auto) {
      auto.disabled = false;
      auto.classList.toggle('on', a.autonomy_enabled === true);
      auto.textContent = a.autonomy_enabled === true ? 'Autonomy ON — disable'
        : 'Autonomy OFF — enable';
    }
    var mode = a.stopped ? 'STOPPED' : a.manual_active ? 'MANUAL' :
      a.autonomy_enabled ? 'AUTONOMOUS' : 'IDLE';
    var modeEl = byId('mode-badge');
    if (modeEl) {
      modeEl.textContent = mode;
      modeEl.className = 'badge mode-' + mode.toLowerCase();
    }

    // camera meta
    var cam = s.camera || {};
    setText('cam-meta', 'frame ' + (isNum(cam.frame_index) ? cam.frame_index : DASH) +
      ' · age ' + fmtAge(cam.age) + ' · ' + (cam.fresh === true ? 'fresh' : 'NOT fresh'));
    byId('cam-meta').className = 'meta ' + (cam.fresh === true ? '' : 'warn-text');

    // localization
    var st = loc.status || 'unknown';
    var badge = byId('loc-badge');
    badge.textContent = st;
    badge.className = 'badge status-' + locStatusClass(loc.status);
    byId('loc-kv').innerHTML = kv([
      ['pose', fmtPose(loc.pose)],
      ['σ position', isNum(loc.position_sigma) ? fmt(loc.position_sigma, 3) + ' m' : DASH],
      ['inliers', isNum(loc.inliers) ? String(loc.inliers) : DASH],
      ['landmarks', isNum(loc.landmarks) ? String(loc.landmarks) : DASH],
      ['keyframes', isNum(loc.keyframes) ? String(loc.keyframes) : DASH],
      ['map version', loc.map_version || DASH],
    ]);

    // navigation line under map
    var nav = s.navigation || {};
    setText('nav-meta', 'nav ' + (nav.status || DASH) + ' · goal ' + fmtXY(nav.goal) +
      ' · rejected ' + (isNum(nav.rejected_goals) ? nav.rejected_goals : DASH) +
      (nav.last_rejection ? ' (last: ' + nav.last_rejection + ')' : ''));
    byId('map-box').classList.toggle('clickable', validMapMeta(s.map_meta));

    // authority
    byId('auth-kv').innerHTML = kv([
      ['autonomy', yesNo(a.autonomy_enabled)],
      ['manual active', yesNo(a.manual_active)],
      ['stopped', yesNo(a.stopped), a.stopped ? 'bad-text' : ''],
      ['generation', isGen(a.generation) ? String(a.generation) : DASH],
      ['revoked reason', a.revoked_reason || DASH, a.revoked_reason ? 'warn-text' : ''],
      ['heartbeat age', fmtAge(a.heartbeat_age),
        isNum(a.heartbeat_age) && a.heartbeat_age > 2 ? 'warn-text' : ''],
    ]);

    // expression readout
    var ex = s.expression || {};
    byId('expr-kv').innerHTML = kv([
      ['caption', ex.caption || DASH],
      ['face', ex.face || DASH],
      ['attitude', ex.attitude || DASH],
      ['uncertainty', fmt(ex.uncertainty, 2)],
      ['gaze', Array.isArray(ex.gaze) ? fmtSigned(ex.gaze[0], 2) + ', ' +
        fmtSigned(ex.gaze[1], 2) : DASH],
      ['signal pattern', yesNo(ex.signal_pattern), ex.signal_pattern ? 'warn-text' : ''],
    ]);
    var basisEl = byId('expr-basis');
    if (ex.basis && typeof ex.basis === 'object') {
      basisEl.innerHTML = Object.keys(ex.basis).map(function (k) {
        var v = ex.basis[k];
        return '<div><b>' + escapeHtml(k) + '</b> ← ' +
          escapeHtml(Array.isArray(v) ? v.join(', ') : String(v)) + '</div>';
      }).join('');
    } else {
      basisEl.textContent = 'no basis reported';
    }

    // motivation
    var mot = s.motivation || {};
    var need = isNum(mot.stimulation_need) ? clamp(mot.stimulation_need, 0, 1) : null;
    byId('mot-fill').style.width = need === null ? '0%' : (need * 100).toFixed(1) + '%';
    setText('mot-val', fmt(mot.stimulation_need, 2));
    setText('mot-tag', mot.engineered === false ? 'not engineered?' : 'engineered');

    // memory
    var mem = s.memory || {};
    setText('mem-meta', 'agent ' + (mem.agent_id || DASH) + ' · ' +
      (isNum(mem.outcomes) ? mem.outcomes : DASH) + ' outcomes · ' +
      (isNum(mem.entities) ? mem.entities : DASH) + ' entities');

    renderActivity(s, act, labels);
    renderEntities(s, act);
    renderInteraction(s, labels);
    renderOutcomes(s, labels);
  }

  function renderActivity(s, act, labels) {
    setText('act-name', act.name || DASH);
    setText('act-phase', act.phase || '');
    setText('act-reason', act.reason || '');
    var dur = isNum(s.sim_time) && isNum(act.since) ? s.sim_time - act.since : null;
    var dec = s.decision || {};
    setText('act-meta', 'target ' + (act.target_entity ? (labels[act.target_entity] ||
      act.target_entity) : DASH) + ' · for ' + fmtAge(dur) + ' · reconsiderations ' +
      (isNum(dec.reconsiderations) ? dec.reconsiderations : DASH));
    var cands = Array.isArray(dec.candidates) ? dec.candidates.slice() : [];
    cands.sort(function (x, y) { return (isNum(y.value) ? y.value : -1e9) - (isNum(x.value) ? x.value : -1e9); });
    var body = cands.map(function (c) {
      var chosen = isChosen(c, dec.chosen);
      return '<tr class="' + (chosen ? 'chosen' : '') + '">' +
        '<td>' + (chosen ? '▶ ' : '') + escapeHtml(c.activity || DASH) + '</td>' +
        '<td>' + escapeHtml(c.entity_id ? (labels[c.entity_id] || c.entity_id) : DASH) + '</td>' +
        '<td>' + escapeHtml(c.action || DASH) + '</td>' +
        '<td class="num">' + fmtSigned(c.value) + '</td>' +
        '<td class="num">' + fmtSigned(c.expected_value) + '</td>' +
        '<td class="num">' + fmt(c.information_value) + '</td>' +
        '<td class="num">' + fmt(c.cost) + '</td>' +
        '<td class="basis">' + escapeHtml(c.basis || '') + '</td></tr>';
    }).join('');
    byId('cand-body').innerHTML = body ||
      '<tr><td colspan="8" class="empty">no candidates reported</td></tr>';
  }

  function renderEntities(s, act) {
    var ents = Array.isArray(s.entities) ? s.entities.filter(Boolean) : [];
    var body = ents.map(function (e) {
      var learned = isNum(e.interactions) && e.interactions >= 1;
      var att = e.attitude || 'unknown';
      var attNote = !learned && att !== 'unknown' ? ' (unlearned)' : '';
      return '<tr class="' + (e.entity_id === act.target_entity ? 'target' : '') + '">' +
        '<td class="mono">' + escapeHtml(e.entity_id) + '</td>' +
        '<td>' + escapeHtml(e.label || DASH) + '</td>' +
        '<td>' + (e.visible ? '<span class="dot ok small"></span>' :
          '<span class="dot off small"></span>') + '</td>' +
        '<td class="num">' + fmtPct(e.identity_confidence) + '</td>' +
        '<td>' + (e.ambiguous ? '<span class="warn-text">ambiguous</span>' : '') + '</td>' +
        '<td><span class="pill ' + attitudeClass(att) + (learned ? '' : ' dim') + '">' +
        escapeHtml(att + attNote) + '</span></td>' +
        '<td class="num">' + fmtSigned(e.expected_value) + ' ± ' + fmt(e.uncertainty) +
        '</td>' +
        '<td class="num">' + (isNum(e.interactions) ? e.interactions : DASH) + '</td></tr>';
    }).join('');
    byId('ent-body').innerHTML = body ||
      '<tr><td colspan="8" class="empty">no entities remembered</td></tr>';
  }

  function barsHtml(predicted, observed) {
    var rows = distributionRows(predicted, observed);
    if (!rows.length) return '<div class="empty">no prediction</div>';
    return rows.map(function (r) {
      return '<div class="bar-row' + (r.observed ? ' observed' : '') + '">' +
        '<span class="bar-label">' + escapeHtml(r.outcome) + '</span>' +
        '<span class="bar-track"><span class="bar-fill" style="width:' +
        (r.p * 100).toFixed(1) + '%"></span></span>' +
        '<span class="bar-val">' + fmt(r.p) + (r.observed ? ' ✓ observed' : '') +
        '</span></div>';
    }).join('');
  }

  function renderInteraction(s, labels) {
    var it = s.interaction;
    var title, pred, obs, meta;
    if (it && typeof it === 'object') {
      title = 'Pending interaction';
      pred = it.predicted;
      obs = it.observed;
      meta = (labels[it.entity_id] || it.entity_id || DASH) + ' · action ' +
        (it.action || DASH) + ' · status ' + (it.status || DASH) +
        (it.request_id ? ' · ' + String(it.request_id).slice(0, 12) : '');
    } else {
      var outs = Array.isArray(s.recent_outcomes) ? s.recent_outcomes : [];
      var last = outs.length ? outs[outs.length - 1] : null;
      if (last) {
        title = 'Last completed interaction';
        pred = last.predicted;
        obs = last.observed;
        meta = (labels[last.entity_id] || last.entity_id || DASH) + ' · action ' +
          (last.action || DASH) + (last.context ? ' · context ' + last.context : '');
      } else {
        title = 'No interaction yet';
        pred = null;
        obs = null;
        meta = '';
      }
    }
    setText('int-title', title);
    setText('int-meta', meta);
    setText('int-observed', 'observed: ' + (obs === null || obs === undefined ? 'pending' : obs));
    byId('int-bars').innerHTML = barsHtml(pred, obs);
  }

  function renderOutcomes(s, labels) {
    var outs = Array.isArray(s.recent_outcomes) ? s.recent_outcomes.slice().reverse() : [];
    var html = outs.slice(0, 14).map(function (o) {
      var top = topOutcome(o.predicted);
      var pObs = o.predicted && isNum(o.predicted[o.observed]) ? o.predicted[o.observed] : 0;
      var match = top && top.outcome === o.observed;
      var up = o.update || {};
      var delta = isNum(up.before) && isNum(up.after) ? up.after - up.before : null;
      var val = isNum(o.valence) ? o.valence : null;
      return '<li>' +
        '<div class="o-line">' +
        '<span class="o-time mono">' + fmt(o.timestamp, 1) + '</span>' +
        '<span class="o-what">' + escapeHtml(labels[o.entity_id] || o.entity_id || DASH) +
        ' · ' + escapeHtml(o.action || DASH) +
        (o.context ? ' <span class="muted">[' + escapeHtml(o.context) + ']</span>' : '') +
        '</span>' +
        '<span class="o-val ' + (val > 0 ? 'ok-text' : val < 0 ? 'bad-text' : '') +
        '">valence ' + fmtSigned(val, 1) + '</span></div>' +
        '<div class="o-line sub">' +
        '<span>pred ' + escapeHtml(top ? top.outcome + ' ' + fmt(top.p) : DASH) + '</span>' +
        '<span class="' + (match ? 'ok-text' : 'warn-text') + '">obs ' +
        escapeHtml(o.observed === null || o.observed === undefined ? DASH : o.observed) +
        ' (p ' + fmt(pObs) + ')' + (match ? '' : ' surprise') + '</span>' +
        '<span class="o-upd mono">value ' + fmt(up.before) + ' → ' + fmt(up.after) +
        (delta !== null ? ' <span class="' + (delta >= 0 ? 'ok-text' : 'bad-text') + '">(' +
          fmtSigned(delta) + ')</span>' : '') + '</span></div></li>';
    }).join('');
    byId('out-list').innerHTML = html || '<li class="empty">no outcomes recorded yet</li>';
  }

  // ---------------------------------------------------------------- manual drive

  function speed() {
    var el = byId('speed');
    var v = el ? Number(el.value) : 0.25;
    return isNum(v) ? v : 0.25;
  }

  function manualStatus(text, cls) {
    var el = byId('drive-status');
    if (!el) return;
    el.textContent = text;
    el.className = 'meta ' + (cls || '');
  }

  function stopManualLoop() {
    if (app.manualTimer !== null) clearInterval(app.manualTimer);
    app.manualTimer = null;
  }

  function manualStep() {
    if (app.held.size === 0) {
      stopManualLoop();
      app.session = null;
      manualStatus('idle — hold WASD / arrows or the pad', '');
      return;
    }
    var cmd = manualTick(app.session, app.generation, app.held, speed(), TURN_RATE);
    if (!cmd) {
      stopManualLoop();
      manualStatus(app.session && isGen(app.session.gen) ?
        'authority changed — release all drive keys, then press again' :
        'no authority generation known yet — release and retry', 'warn-text');
      return;
    }
    manualStatus('sending v ' + fmtSigned(cmd.v) + ' m/s, w ' + fmtSigned(cmd.w) +
      ' rad/s (gen ' + cmd.generation + ')', 'ok-text');
    post(cmd, { quiet: true });
  }

  function pressDir(dir) {
    if (!dir) return;
    if (app.held.has(dir)) return;
    app.held.add(dir);
    updatePadHighlight();
    if (!app.session) app.session = { gen: app.generation, latched: false };
    if (app.manualTimer === null && !app.session.latched) {
      manualStep();
      if (app.held.size > 0 && app.session && !app.session.latched && app.manualTimer === null &&
          isGen(app.session.gen)) {
        app.manualTimer = setInterval(manualStep, MANUAL_MS);
      }
    }
  }

  function releaseDir(dir) {
    if (!app.held.delete(dir)) return;
    updatePadHighlight();
    if (app.held.size === 0) {
      stopManualLoop();
      app.session = null; // latch clears only once everything is released
      manualStatus('idle — hold WASD / arrows or the pad', '');
    }
  }

  function releaseAll() {
    Array.from(app.held).forEach(releaseDir);
  }

  function updatePadHighlight() {
    ['fwd', 'back', 'left', 'right'].forEach(function (d) {
      var b = byId('pad-' + d);
      if (b) b.classList.toggle('held', app.held.has(d));
    });
  }

  function doStop() {
    if (app.session) app.session.latched = true;
    stopManualLoop();
    post(stopCmd());
  }

  function isTyping(ev) {
    var t = ev.target;
    var tag = t && t.tagName ? t.tagName.toLowerCase() : '';
    return tag === 'input' || tag === 'textarea' || tag === 'select' || (t && t.isContentEditable);
  }

  // ---------------------------------------------------------------- wiring

  function init() {
    ['img-camera', 'img-map', 'img-screen', 'img-inspect'].forEach(wireImage);

    byId('btn-stop').addEventListener('click', doStop);

    byId('btn-autonomy').addEventListener('click', function () {
      var a = app.state && app.state.authority;
      if (a && a.autonomy_enabled === true) post(disableCmd());
      else post(enableCmd(app.generation));
    });

    // keyboard drive; Space / Escape = STOP
    document.addEventListener('keydown', function (ev) {
      if (ev.key === 'Escape') {
        // Escape stops even while typing (the reset field swallows its own Escape).
        if (!ev.repeat) doStop();
        return;
      }
      if (ev.ctrlKey || ev.metaKey || ev.altKey || isTyping(ev)) return;
      if (ev.key === ' ') {
        ev.preventDefault();
        if (!ev.repeat) doStop();
        return;
      }
      var dir = keyToDir(ev.key);
      if (!dir) return;
      ev.preventDefault();
      pressDir(dir);
    });
    document.addEventListener('keyup', function (ev) {
      var dir = keyToDir(ev.key);
      if (dir) releaseDir(dir);
    });
    root.addEventListener('blur', releaseAll);
    document.addEventListener('visibilitychange', function () {
      if (document.hidden) releaseAll();
    });

    ['fwd', 'back', 'left', 'right'].forEach(function (d) {
      var b = byId('pad-' + d);
      b.addEventListener('pointerdown', function (ev) {
        ev.preventDefault();
        if (b.setPointerCapture) b.setPointerCapture(ev.pointerId);
        pressDir(d);
      });
      ['pointerup', 'pointercancel', 'lostpointercapture'].forEach(function (evn) {
        b.addEventListener(evn, function () { releaseDir(d); });
      });
      b.addEventListener('contextmenu', function (ev) { ev.preventDefault(); });
    });
    byId('pad-stop').addEventListener('click', doStop);
    byId('speed').addEventListener('input', function () {
      setText('speed-val', fmt(speed(), 2) + ' m/s');
    });

    // goal form
    byId('goal-form').addEventListener('submit', function (ev) {
      ev.preventDefault();
      var x = parseFloat(byId('goal-x').value);
      var y = parseFloat(byId('goal-y').value);
      if (!isNum(x) || !isNum(y)) { toast('goal needs numeric x and y', 'bad'); return; }
      post(goalCmd(x, y, app.generation));
    });

    // map click -> goal
    var mapImg = byId('img-map');
    function mapPoint(ev) {
      var meta = app.state && app.state.map_meta;
      var r = mapImg.getBoundingClientRect();
      return clickToMap(ev.clientX - r.left, ev.clientY - r.top, r.width, r.height,
        mapImg.naturalWidth, mapImg.naturalHeight, meta);
    }
    mapImg.addEventListener('click', function (ev) {
      var p = mapPoint(ev);
      if (!p) {
        if (!validMapMeta(app.state && app.state.map_meta)) toast('map click ignored: no map_meta', 'bad');
        return;
      }
      byId('goal-x').value = p.x.toFixed(2);
      byId('goal-y').value = p.y.toFixed(2);
      toast('goal from map click (' + p.x.toFixed(2) + ', ' + p.y.toFixed(2) + ')', 'info');
      post(goalCmd(p.x, p.y, app.generation));
    });
    mapImg.addEventListener('mousemove', function (ev) {
      var p = mapPoint(ev);
      setText('map-hover', p ? 'cursor (' + p.x.toFixed(2) + ', ' + p.y.toFixed(2) + ')' : '');
    });
    mapImg.addEventListener('mouseleave', function () { setText('map-hover', ''); });

    // reset memory: inline typed confirmation
    var resetRow = byId('reset-row');
    var resetInput = byId('reset-input');
    var resetGo = byId('reset-go');
    function closeReset() {
      resetRow.hidden = true;
      resetInput.value = '';
      resetGo.disabled = true;
    }
    byId('btn-reset').addEventListener('click', function () {
      resetRow.hidden = !resetRow.hidden;
      if (!resetRow.hidden) resetInput.focus();
    });
    resetInput.addEventListener('input', function () {
      resetGo.disabled = resetCmd(resetInput.value) === null;
    });
    resetGo.addEventListener('click', function () {
      var cmd = resetCmd(resetInput.value);
      if (!cmd) return;
      post(cmd);
      closeReset();
    });
    byId('reset-cancel').addEventListener('click', closeReset);
    resetInput.addEventListener('keydown', function (ev) {
      if (ev.key === 'Enter') { ev.preventDefault(); resetGo.click(); }
      if (ev.key === 'Escape') { ev.preventDefault(); ev.stopPropagation(); closeReset(); }
    });

    manualStatus('idle — hold WASD / arrows or the pad', '');
    setText('speed-val', fmt(speed(), 2) + ' m/s');
    poll();
    heartbeat();
    setInterval(poll, POLL_MS);
    setInterval(heartbeat, HEARTBEAT_MS);
    setInterval(updateConnection, 500);
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})(typeof window !== 'undefined' ? window : globalThis);

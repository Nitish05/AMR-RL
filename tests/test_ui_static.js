// Static checks + pure-helper unit tests for the operator console.
// Run: node tests/test_ui_static.js   (no npm dependencies)
'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const STATIC = path.join(__dirname, '..', 'src', 'amr_rl', 'ui', 'static');
const appSrc = fs.readFileSync(path.join(STATIC, 'app.js'), 'utf8');
const html = fs.readFileSync(path.join(STATIC, 'index.html'), 'utf8');
const css = fs.readFileSync(path.join(STATIC, 'style.css'), 'utf8');

let passed = 0;
function test(name, fn) {
  try {
    fn();
    passed += 1;
  } catch (err) {
    console.error('FAIL ' + name);
    throw err;
  }
}
// Values created inside the vm context have that realm's prototypes; compare as plain JSON.
const plain = (x) => JSON.parse(JSON.stringify(x));
const near = (a, b, eps = 1e-9) => assert.ok(Math.abs(a - b) <= eps, `${a} != ${b}`);

// ------------------------------------------------------------------ load
new vm.Script(appSrc, { filename: 'app.js' }); // throws on syntax errors
const ctx = vm.createContext({ console });
vm.runInContext(appSrc, ctx, { filename: 'app.js' });
const U = ctx.AMRUI;

test('exports helpers without a DOM', () => {
  assert.ok(U, 'AMRUI global missing');
  for (const k of ['clickToMap', 'mapToPixel', 'manualCmd', 'goalCmd', 'enableCmd', 'stopCmd',
    'heartbeatCmd', 'resetCmd', 'fmt', 'driveFromDirs', 'manualTick', 'distributionRows']) {
    assert.strictEqual(typeof U[k], 'function', k);
  }
  assert.strictEqual(U.POLL_MS, 250);
  assert.strictEqual(U.HEARTBEAT_MS, 1000);
  assert.strictEqual(U.MANUAL_MS, 200);
});

// ------------------------------------------------------------------ static hygiene
test('no alert/confirm/prompt, no external resources', () => {
  for (const src of [appSrc, html, css]) {
    assert.ok(!/\b(alert|confirm|prompt)\s*\(/.test(src.replace(/\/\/.*$/gm, '')),
      'dialog call found');
    assert.ok(!/(src|href)\s*=\s*["']?(https?:)?\/\//i.test(src), 'external URL found');
    assert.ok(!/@import|url\(\s*["']?https?:/i.test(src), 'external CSS import');
  }
  assert.ok(!/\beval\s*\(/.test(appSrc));
});

test('every id used by app.js exists in index.html', () => {
  const used = new Set();
  for (const m of appSrc.matchAll(/(?:byId|setText)\('([a-z0-9-]+)'/g)) used.add(m[1]);
  for (const m of appSrc.matchAll(/'(pad-)' \+ d/g)) used.add(m[1]);
  used.delete('pad-');
  for (const d of ['fwd', 'back', 'left', 'right']) used.add('pad-' + d);
  const ids = new Set([...html.matchAll(/\sid="([^"]+)"/g)].map((m) => m[1]));
  const missing = [...used].filter((id) => !ids.has(id));
  assert.deepStrictEqual(missing, []);
  assert.ok(used.size > 40);
});

test('required labels and controls present', () => {
  assert.ok(html.includes('Onboard RGB — operational input'));
  assert.ok(html.includes('third-person inspection only — not used by the robot'));
  assert.ok(html.includes('id="btn-stop"'));
  assert.ok(!/id="btn-stop"[^>]*disabled/.test(html), 'STOP must never be disabled');
  assert.ok(/<script src="app\.js"><\/script>/.test(html));
  assert.ok(html.includes('engineered'));
  assert.ok(appSrc.includes("'/api/state'"));
  for (const p of ['/api/camera.jpg', '/api/map.png', '/api/screen.png', '/api/inspect.jpg']) {
    assert.ok(appSrc.includes("'" + p + "'"), p);
  }
});

// ------------------------------------------------------------------ formatting
test('formatting', () => {
  assert.strictEqual(U.fmt(null), '—');
  assert.strictEqual(U.fmt(undefined), '—');
  assert.strictEqual(U.fmt(NaN), '—');
  assert.strictEqual(U.fmt(0), '0.00');
  assert.strictEqual(U.fmt(1.23456, 3), '1.235');
  assert.strictEqual(U.fmtSigned(0.5), '+0.50');
  assert.strictEqual(U.fmtSigned(-0.5), '-0.50');
  assert.strictEqual(U.fmtSigned(0), '0.00');
  assert.strictEqual(U.fmtPct(0.934), '93%');
  assert.strictEqual(U.fmtPose([0.4, -0.2, Math.PI / 2]), 'x 0.40  y -0.20  θ 90.0°');
  assert.strictEqual(U.fmtPose(null), '—');
  assert.strictEqual(U.fmtPose([1, 2]), '—');
  assert.strictEqual(U.fmtXY([1, 2.5]), '(1.00, 2.50)');
  assert.strictEqual(U.fmtAge(0.42), '0.4 s');
  assert.strictEqual(U.escapeHtml('<b a="1">&\''), '&lt;b a=&quot;1&quot;&gt;&amp;&#39;');
  assert.strictEqual(U.escapeHtml(null), '');
  assert.strictEqual(U.imgUrl('/api/map.png', 123), '/api/map.png?t=123');
  assert.strictEqual(U.imgUrl('/x?a=1', 5), '/x?a=1&t=5');
});

test('status classes', () => {
  assert.strictEqual(U.locStatusClass('tracking'), 'ok');
  assert.strictEqual(U.locStatusClass('relocalizing'), 'warn');
  assert.strictEqual(U.locStatusClass('initializing'), 'warn');
  assert.strictEqual(U.locStatusClass('lost'), 'bad');
  assert.strictEqual(U.locStatusClass(null), 'unknown');
  assert.strictEqual(U.connectionLevel(0.3), 'ok');
  assert.strictEqual(U.connectionLevel(1.5), 'warn');
  assert.strictEqual(U.connectionLevel(5), 'bad');
  assert.strictEqual(U.connectionLevel(null), 'bad');
  assert.strictEqual(U.attitudeClass('liked'), 'att-liked');
  assert.strictEqual(U.attitudeClass('weird'), 'att-none');
});

// ------------------------------------------------------------------ map geometry
const META = { origin: [-2.0, -1.0], resolution: 0.05, width_px: 200, height_px: 100 };

test('clickToMap: corners and north-up', () => {
  // element exactly matches the image aspect (2:1), displayed at 2x
  const tl = U.clickToMap(0, 0, 400, 200, 200, 100, META);
  near(tl.x, -2.0); near(tl.y, -1.0 + 100 * 0.05); // row 0 = max y
  const br = U.clickToMap(400, 200, 400, 200, 200, 100, META);
  near(br.x, -2.0 + 200 * 0.05); near(br.y, -1.0);
  const c = U.clickToMap(200, 100, 400, 200, 200, 100, META);
  near(c.x, 3.0); near(c.y, 1.5); near(c.u, 100); near(c.v, 50);
  // moving down the image decreases y
  assert.ok(U.clickToMap(200, 150, 400, 200, 200, 100, META).y < c.y);
});

test('clickToMap: letterboxed (object-fit: contain)', () => {
  // 2:1 image in a 400x400 box -> 400x200 content, 100px bars top and bottom
  assert.strictEqual(U.clickToMap(200, 50, 400, 400, 200, 100, META), null);
  assert.strictEqual(U.clickToMap(200, 350, 400, 400, 200, 100, META), null);
  const c = U.clickToMap(200, 200, 400, 400, 200, 100, META);
  near(c.x, 3.0); near(c.y, 1.5);
  // natural size unknown (0) -> assume map aspect
  const d = U.clickToMap(200, 200, 400, 400, 0, 0, META);
  near(d.x, 3.0); near(d.y, 1.5);
  // PNG upscaled 3x relative to the grid still maps through width_px/height_px
  const e = U.clickToMap(100, 50, 200, 100, 600, 300, META);
  near(e.x, 3.0); near(e.y, 1.5);
});

test('clickToMap: rejects missing/invalid meta', () => {
  assert.strictEqual(U.clickToMap(10, 10, 100, 100, 100, 100, null), null);
  assert.strictEqual(U.clickToMap(10, 10, 100, 100, 100, 100, undefined), null);
  assert.strictEqual(U.clickToMap(10, 10, 100, 100, 100, 100, { ...META, resolution: 0 }), null);
  assert.strictEqual(U.clickToMap(10, 10, 100, 100, 100, 100, { ...META, origin: null }), null);
  assert.strictEqual(U.clickToMap(10, 10, 0, 0, 100, 100, META), null);
  assert.strictEqual(U.validMapMeta(META), true);
});

test('mapToPixel inverts clickToMap', () => {
  for (const [cx, cy] of [[0, 0], [37, 81], [399, 1], [123.5, 199]]) {
    const p = U.clickToMap(cx, cy, 400, 200, 200, 100, META);
    const px = U.mapToPixel(p.x, p.y, META);
    near(px.u, p.u, 1e-9); near(px.v, p.v, 1e-9);
  }
  assert.strictEqual(U.mapToPixel(1, 1, null), null);
});

// ------------------------------------------------------------------ commands
test('command payloads carry generation', () => {
  assert.deepStrictEqual({ ...U.manualCmd(0.2, -0.5, 7) },
    { action: 'manual', v: 0.2, w: -0.5, generation: 7 });
  assert.deepStrictEqual({ ...U.goalCmd(1.5, -2, 7) },
    { action: 'goal', x: 1.5, y: -2, generation: 7 });
  assert.deepStrictEqual({ ...U.enableCmd(4) }, { action: 'enable_autonomy', generation: 4 });
  assert.deepStrictEqual({ ...U.stopCmd() }, { action: 'stop' });
  assert.deepStrictEqual({ ...U.heartbeatCmd() }, { action: 'heartbeat' });
  assert.deepStrictEqual({ ...U.disableCmd() }, { action: 'disable_autonomy' });
  // no generation known -> nothing to send
  for (const g of [null, undefined, NaN, 1.5, '3']) {
    assert.strictEqual(U.manualCmd(0.1, 0, g), null);
    assert.strictEqual(U.goalCmd(1, 1, g), null);
    assert.strictEqual(U.enableCmd(g), null);
  }
  assert.strictEqual(U.goalCmd(NaN, 1, 3), null);
  assert.strictEqual(U.manualCmd(0.1, Infinity, 3), null);
  assert.strictEqual(U.manualCmd(0.1, 0, 0).generation, 0);
});

test('reset requires typing RESET exactly', () => {
  assert.deepStrictEqual({ ...U.resetCmd('RESET') }, { action: 'reset_memory', confirm: 'RESET' });
  for (const t of ['', 'reset', 'RESET ', ' RESET', 'yes', null, undefined]) {
    assert.strictEqual(U.resetCmd(t), null, String(t));
  }
});

test('describeResponse', () => {
  const ok = U.describeResponse({ action: 'goal' }, { accepted: true, reason: 'planned', generation: 3 });
  assert.strictEqual(ok.ok, true);
  assert.strictEqual(ok.text, 'goal accepted — planned (gen 3)');
  const bad = U.describeResponse({ action: 'manual' }, { accepted: false, reason: 'stale generation' });
  assert.strictEqual(bad.ok, false);
  assert.ok(bad.text.includes('rejected') && bad.text.includes('stale generation'));
  assert.strictEqual(U.describeResponse({ action: 'stop' }, null).ok, false);
});

// ------------------------------------------------------------------ manual drive
test('key mapping and drive vector', () => {
  assert.strictEqual(U.keyToDir('ArrowUp'), 'fwd');
  assert.strictEqual(U.keyToDir('W'), 'fwd');
  assert.strictEqual(U.keyToDir('s'), 'back');
  assert.strictEqual(U.keyToDir('a'), 'left');
  assert.strictEqual(U.keyToDir('ArrowRight'), 'right');
  assert.strictEqual(U.keyToDir('q'), null);
  assert.strictEqual(U.keyToDir(undefined), null);
  const d = (dirs) => ({ ...U.driveFromDirs(new Set(dirs), 0.3, 0.8) });
  assert.deepStrictEqual(d(['fwd']), { v: 0.3, w: 0 });
  assert.deepStrictEqual(d(['back']), { v: -0.3, w: 0 });
  assert.deepStrictEqual(d(['left']), { v: 0, w: 0.8 }); // CCW positive
  assert.deepStrictEqual(d(['right']), { v: 0, w: -0.8 });
  assert.deepStrictEqual(d(['fwd', 'right']), { v: 0.3, w: -0.8 });
  assert.deepStrictEqual(d(['fwd', 'back', 'left', 'right']), { v: 0, w: 0 });
  assert.deepStrictEqual(d([]), { v: 0, w: 0 });
});

test('manualTick sends with session generation and latches on change', () => {
  const held = new Set(['fwd']);
  const s = { gen: 5, latched: false };
  const c = U.manualTick(s, 5, held, 0.25, 0.8);
  assert.deepStrictEqual({ ...c }, { action: 'manual', v: 0.25, w: 0, generation: 5 });
  // Stop bumps generation -> latched, no command, stays latched
  assert.strictEqual(U.manualTick(s, 6, held, 0.25, 0.8), null);
  assert.strictEqual(s.latched, true);
  assert.strictEqual(U.manualTick(s, 5, held, 0.25, 0.8), null);
  // nothing held -> nothing sent
  assert.strictEqual(U.manualTick({ gen: 5, latched: false }, 5, new Set(), 0.25, 0.8), null);
  // unknown generation -> latched, not sent
  const s2 = { gen: null, latched: false };
  assert.strictEqual(U.manualTick(s2, null, held, 0.25, 0.8), null);
  assert.strictEqual(s2.latched, true);
});

// ------------------------------------------------------------------ view models
test('isChosen', () => {
  const chosen = { activity: 'engage', entity_id: 'e1', action: 'signal' };
  assert.ok(U.isChosen({ activity: 'engage', entity_id: 'e1', action: 'signal', value: 1 }, chosen));
  assert.ok(!U.isChosen({ activity: 'engage', entity_id: 'e2', action: 'signal' }, chosen));
  assert.ok(U.isChosen({ activity: 'idle' }, { activity: 'idle', entity_id: null }));
  assert.ok(!U.isChosen({ activity: 'idle' }, null));
});

test('distributionRows sorts and always includes observed', () => {
  const rows = U.distributionRows({ none: 0.17, yellow_flag: 0.83 }, 'yellow_flag');
  assert.deepStrictEqual(plain(rows.map((r) => r.outcome)), ['yellow_flag', 'none']);
  assert.strictEqual(rows[0].observed, true);
  assert.strictEqual(rows[1].observed, false);
  const surprise = U.distributionRows({ a: 0.9, b: 0.1 }, 'c');
  assert.deepStrictEqual(plain(surprise.map((r) => [r.outcome, r.p, r.observed])),
    [['a', 0.9, false], ['b', 0.1, false], ['c', 0, true]]);
  assert.deepStrictEqual(U.distributionRows(null, null).length, 0);
  assert.strictEqual(U.distributionRows({ x: 1.7, y: 'bad' }, null)[0].p, 1);
  assert.strictEqual(U.topOutcome({ a: 0.2, b: 0.8 }).outcome, 'b');
  assert.strictEqual(U.topOutcome(null), null);
});

console.log(`test_ui_static.js: ${passed} passed`);

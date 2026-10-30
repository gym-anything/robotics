"use strict";
// Keyboard chords -> the backend's `keyboard` action, through the local proxy (barcode_scanning/manual.py), which
// holds the GAR key. At most one robot request is in flight; held time accumulates as queued chords meanwhile and the
// whole queue is sent as ONE request (one camera render).
const $ = id => document.getElementById(id);
const RATE = 30;                          // keyboard chord ticks per second (rate_hz)
const CODES = {KeyW:"W",KeyS:"S",KeyA:"A",KeyD:"D",KeyR:"R",KeyF:"F",KeyI:"I",KeyK:"K",KeyJ:"J",KeyL:"L",KeyU:"U",
  KeyO:"O",KeyQ:"Q",KeyE:"E",KeyH:"H",KeyG:"G",KeyB:"B",KeyP:"P",KeyT:"T",ArrowUp:"W",ArrowDown:"S",ArrowLeft:"A",
  ArrowRight:"D",AltLeft:"ALT",AltRight:"ALT",ShiftLeft:"SHIFT",ShiftRight:"SHIFT"};
const CONFLICTS = [["A","D"],["W","S"],["R","F"],["I","K"],["J","L"],["U","O"],["Q","E"],["G","B"],["P","T"],["ALT","SHIFT"],["H","E"]];
const MOTION = new Set("ADWSRFIKJLUO".split(""));
const GATHER_MS = 300;                    // wait this long after the last release before sending (batches taps)

const token = (() => {
  const m = location.hash.match(/token=([^&]+)/);
  if (m) { sessionStorage.setItem("manualToken", decodeURIComponent(m[1])); history.replaceState(null, "", location.pathname); }
  return sessionStorage.getItem("manualToken") || "";
})();

let info = null, inflight = null, open = null, saturated = "", lastRelease = 0, frameUrls = [], lastObs = null, lastWall = null;
const held = new Map();                   // source (key code or on-screen button) -> {key, t}
const queue = [];                         // {keys, arm, ticks, sig}
const log = [];

async function api(path, body, binary=false) {
  const options = {headers: {"X-Manual-Token": token}, cache: "no-store"};
  if (body !== undefined) { options.method = "POST"; options.headers["Content-Type"] = "application/json"; options.body = JSON.stringify(body); }
  const r = await fetch(path, options);
  if (binary && r.ok) return r.blob();
  const data = await r.json().catch(() => ({ok: false, error: "HTTP " + r.status}));
  if (!r.ok || data.ok === false) { const e = new Error(data.error || ("HTTP " + r.status)); e.status = r.status; e.detail = data.detail; throw e; }
  return data;
}
function notice(text, error=false) { $("notice").textContent = text; $("notice").classList.toggle("error", error); }
const fmt = (v, d=1) => (v === null || v === undefined || Number.isNaN(v)) ? "–" : Number(v).toFixed(d);
const signed = (v, d=2) => (v >= 0 ? "+" : "") + Number(v).toFixed(d);
function num(id, lo, hi) { const v = Number($(id).value); if (!Number.isFinite(v) || v < lo || v > hi) throw new Error(`${$(id).previousSibling.textContent.trim()} must be within ${lo}…${hi}`); return v; }
function arm() { return $("arm").value; }
function enabled() { return !!info && $("enabled").checked; }
function typing(el) { return el && el.closest && el.closest("input,textarea,select,[contenteditable='true']"); }

// ------------------------------------------------------------------------------------------ chords from held keys
function chord() {
  const order = [...held.values()].sort((a, b) => a.t - b.t).map(h => h.key);
  const set = new Set();
  for (const k of order) {                // a later press wins a conflicting pair (never send conflicting keys)
    for (const [a, b] of CONFLICTS) { if (k === a) set.delete(b); if (k === b) set.delete(a); }
    set.add(k);
  }
  if (!set.has("H")) { set.delete("G"); set.delete("B"); }
  if (set.has("H")) set.delete("Q");
  if (!set.has("Q") && !set.has("E")) { set.delete("P"); set.delete("T"); }
  const moving = [...set].some(k => MOTION.has(k));
  if (!moving) { set.delete("ALT"); set.delete("SHIFT"); }
  if (!moving && !set.has("Q") && !set.has("E") && !set.has("H")) return [];
  return [...set].sort();
}
function queuedSeconds() { return queue.reduce((s, c) => s + c.ticks / RATE, 0); }
function settings() {
  return {minChord: num("minchord", 1 / RATE, 2), maxQueue: num("maxqueue", .2, 10)};
}
function pushChord(keys, armName, seconds) {
  const ticks = Math.max(1, Math.round(seconds * RATE));
  const sig = keys.join("+") + "@" + armName;
  const last = queue[queue.length - 1];
  if (last && last.sig === sig && (last.ticks + ticks) / RATE <= 10) last.ticks += ticks;
  else queue.push({keys, arm: armName, ticks, sig});
}
function closeOpen(now, cap) {
  if (!open) return;
  const {minChord} = settings();
  let s = Math.max(minChord, (now - open.start) / 1000);
  if (cap !== undefined) s = Math.min(s, cap);
  if (s * RATE >= 1) pushChord(open.keys, open.arm, s);
  open = null; lastRelease = now;
}
function sample() {
  const now = performance.now();
  let c = [];
  try { c = enabled() ? chord() : []; } catch (_) { c = []; }
  const sig = c.length ? c.join("+") + "@" + arm() : "";
  if (sig !== saturated) saturated = "";
  if (open && open.sig !== sig) closeOpen(now);
  let maxQueue = 2;
  try { maxQueue = settings().maxQueue; } catch (_) {}
  if (!open && sig && !saturated) {
    if (queuedSeconds() + 1 / RATE <= maxQueue) open = {keys: c, arm: arm(), sig, start: now};
    else saturated = sig;
  }
  if (open && queuedSeconds() + (now - open.start) / 1000 >= maxQueue) {
    closeOpen(now, maxQueue - queuedSeconds()); saturated = sig;   // held time beyond the cap is dropped (shown)
  }
  if (!inflight && queue.length && !open && (now - lastRelease >= GATHER_MS || saturated)) dispatch();
  showHeld(c, now);
}
setInterval(sample, 1000 / RATE);

function showHeld(c, now) {
  const keys = new Set([...held.values()].map(h => h.key));
  for (const b of document.querySelectorAll("[data-key]")) b.classList.toggle("down", keys.has(b.dataset.key));
  $("held").textContent = !enabled() ? "Keyboard control off" : c.length
    ? `Holding ${c.join(" + ")} · ${arm()} arm · ${open ? ((now - open.start) / 1000).toFixed(2) + " s" : "queue full: extra hold time is dropped"}`
    : (keys.size ? `Held ${[...keys].join(" + ")} (no command: modifier alone or conflicting)` : "No keys held");
  const items = queue.map(q => `<li>${q.keys.join(" + ")} <span class="muted">· ${q.arm} · ${(q.ticks / RATE).toFixed(2)} s</span></li>`);
  if (open) items.push(`<li class="live">${open.keys.join(" + ")} <span class="muted">· ${open.arm} · ${((now - open.start) / 1000).toFixed(2)} s held…</span></li>`);
  const html = items.length ? items.join("") : '<li class="muted">Nothing queued</li>';
  if ($("queue").innerHTML !== html) $("queue").innerHTML = html;
  const note = inflight ? `${inflight.label} in flight for ${((now - inflight.t0) / 1000).toFixed(0)} s; queued chords go next as one request.`
    : (queue.length ? "Sending when keys are released…" : "");
  if ($("queue-note").textContent !== note) $("queue-note").textContent = note;
  if (inflight) $("render-status").textContent = `${inflight.physics ? "Physics + camera render" : "Camera render"}… ${((now - inflight.t0) / 1000).toFixed(0)} s`;
}

// ------------------------------------------------------------------------------------------ requests
async function dispatch() {
  let body;
  try {
    body = {chords: queue.map(({keys, arm, ticks}) => ({keys, arm, ticks})), rate_hz: RATE,
            linear_speed_m_s: num("lin", .1, 600) / 1000, angular_speed_rad_s: num("ang", .01, 68) * Math.PI / 180,
            gripper_half_speed_m_s: num("grip", .1, 200) / 1000};
  } catch (e) { notice(e.message + " Queue kept; fix the setting.", true); return; }
  const sent = queue.splice(0);
  saturated = "";                         // a key still held keeps accumulating into the next request
  const label = sent.map(c => `${c.keys.join("+")} ${(c.ticks / RATE).toFixed(2)} s`).join(", ");
  await run(`Keys ${label}`, async () => {
    let r;
    try { r = await api("/api/keyboard", body); }
    catch (e) {
      if (e.status === 429) { queue.unshift(...sent); addLog(label, "proxy busy; chords kept in the queue"); return; }
      throw e;
    }
    await showResult(r, sent);
  });
}
async function run(label, job, physics=true) {
  if (inflight) return;
  inflight = {label, t0: performance.now(), physics};
  $("refresh").disabled = true; $("frames").classList.add("rendering");
  const typical = lastWall ? `the last request took ${fmt(lastWall, 0)} s` : "typically 20–50 s on a shared GPU";
  notice(physics ? `${label}: sent. Physics, then the camera render (${typical})…`
                 : `${label}: camera render only, no physics step (${typical})…`);
  try { await job(); }
  catch (e) {
    notice(e.message, true);
    addLog(label, "refused/failed: " + e.message);
    const h = e.detail && e.detail.health; if (h) showHealth(h);
  } finally {
    inflight = null; $("frames").classList.remove("rendering"); $("refresh").disabled = !info;
    if (lastObs) $("render-status").textContent = `Image rendered for observation ${lastObs.observation_id}`;
    if (info && info.role === "owner") ownerRefresh().catch(() => {});
    else api("/api/health").then(r => showHealth(r.health)).catch(() => {});
  }
}
async function showObservation(obs) {
  const fresh = [];
  try {
    for (const name of obs.cameras) {
      const blob = await api(`/api/camera/${encodeURIComponent(name)}?observation_id=${obs.observation_id}`, undefined, true);
      fresh.push([name, URL.createObjectURL(blob)]);
    }
  } catch (e) { for (const [, u] of fresh) URL.revokeObjectURL(u); throw e; }
  for (const u of frameUrls) URL.revokeObjectURL(u);
  frameUrls = fresh.map(f => f[1]);
  $("frames").replaceChildren(...fresh.map(([name, url]) => {
    const fig = document.createElement("figure"), img = document.createElement("img"), cap = document.createElement("figcaption");
    img.src = url; img.alt = `${name} at observation ${obs.observation_id}`; img.dataset.observation = obs.observation_id;
    cap.textContent = `${name} · observation ${obs.observation_id} · sim ${fmt(obs.sim_time_s, 3)} s`;
    fig.append(img, cap); return fig;
  }));
  if (!fresh.length) $("frames").innerHTML = '<div class="empty">This session has no robot cameras.</div>';
  $("frames").classList.toggle("multi", fresh.length > 1);
  $("frames").dataset.observation = obs.observation_id;
  lastObs = obs;
  $("stamp").textContent = `Observation ${obs.observation_id} · simulated time ${fmt(obs.sim_time_s, 3)} s · command sequence ${obs.sequence} · request wall time ${fmt(obs.wall_s, 1)} s`;
  if (obs.wall_s) lastWall = obs.wall_s;
  $("render-status").textContent = `Image rendered for observation ${obs.observation_id}`;
  showTcp(obs.tcp, obs.joints);
}
function showTcp(tcp, joints) {
  if (!tcp) { $("tcp").textContent = info && !info.tcp_available ? `TCP unavailable: ${info.tcp_error}` : "No observation yet"; return; }
  const lines = [];
  // roll is shown in [0, 360) when |roll| > 90 deg so a downward-pointing hand (roll ~ 180) does not flip between
  // +180 and -180 on the Euler wrap
  const roll = r => Math.abs(r) > 90 ? (r + 360) % 360 : r;
  for (const a of Object.keys(tcp)) {
    const t = tcp[a], p = t.position_m.map(v => v * 1000), r = t.rpy_deg;
    lines.push(`${a === arm() ? "▶" : " "} ${a.padEnd(6)}x ${fmt(p[0]).padStart(7)}  y ${fmt(p[1]).padStart(7)}  z ${fmt(p[2]).padStart(7)} mm`);
    lines.push(`         roll ${fmt(roll(r[0]))}°  pitch ${fmt(r[1])}°  yaw ${fmt(r[2])}°`);
    lines.push(`         finger opening ${fmt(t.finger_gap_m * 1000, 2)} mm`);
  }
  $("tcp").textContent = lines.join("\n");
}
async function showResult(r, sent) {
  const f = [];
  const prevSim = lastObs ? lastObs.sim_time_s : null;
  if (r.observation) await showObservation(r.observation);
  f.push(`Sent ${sent.length} chord${sent.length > 1 ? "s" : ""} in one request (${fmt(r.sim_s, 3)} s simulated)`);
  for (const c of sent) f.push(`  ${c.keys.join(" + ")} · ${c.arm} arm · ${(c.ticks / RATE).toFixed(3)} s`);
  f.push(`Wall time ${fmt(r.wall_s, 1)} s (physics + camera render)`);
  if (r.observation && prevSim !== null) f.push(`Simulated clock ${fmt(prevSim, 3)} → ${fmt(r.observation.sim_time_s, 3)} s`);
  if (r.observation_error) f.push(`Committed, but no image: ${r.observation_error.message}. Do not resend.`);
  for (const a of new Set(sent.map(c => c.arm))) {
    const d = r.tcp_delta && r.tcp_delta[a];
    if (d) f.push(`${a} TCP moved ${d.position_mm.map(v => signed(v)).join(", ")} mm · turned ${d.rotation_deg.map(v => signed(v)).join(", ")}° (rotation vector) · fingers ${signed(d.finger_gap_mm)} mm`);
  }
  if (r.feedback) {
    const fb = r.feedback;
    f.push(`Reflex stops ${fb.reflexes} · refused joint targets ${fb.refused_targets} · stopped arms ${JSON.stringify(fb.arms_stopped_s || {})}`);
    f.push(`Scanner frames ${fb.scanner_frames} · scanner reads ${fb.scanner_reads} · physics ticks ${fb.executed_ticks}`);
    if (fb.reflex_details_last_chord && fb.reflex_details_last_chord.length) f.push("Reflex: " + JSON.stringify(fb.reflex_details_last_chord));
  } else {
    f.push("Reflex / refused-target / scanner step feedback: the GAR API returns it to the owner key only.");
  }
  $("feedback").textContent = f.join("\n");
  addLog(sent.map(c => `${c.keys.join("+")} ${(c.ticks / RATE).toFixed(2)} s`).join(", "), `${fmt(r.wall_s, 1)} s wall`);
  notice(`Done in ${fmt(r.wall_s, 1)} s. ${r.feedback && r.feedback.reflexes ? "A reflex stopped an arm." : ""}`);
}
function addLog(what, result) {
  log.unshift(`${new Date().toLocaleTimeString()}  ${what} → ${result}`);
  log.length = Math.min(log.length, 30);
  $("log").replaceChildren(...log.map(t => { const li = document.createElement("li"); li.textContent = t; return li; }));
}
function showHealth(h) {
  const bad = h && h.runtime_ok === false;
  $("fault").hidden = !bad;
  if (bad) {
    $("fault-text").textContent = `The session latched a ${h.runtime_fault_type || "runtime"} fault; it refuses further motion until the owner recovers it.` +
      (info && info.role === "owner" ? "" : " This page holds the policy key, which cannot recover it: ask the session owner.");
    $("recover").hidden = !(info && info.role === "owner");
  }
}

// ------------------------------------------------------------------------------------------ owner panel
function row(k, v, cls="") { return `<div class="kv ${cls}"><span>${k}</span><b>${v}</b></div>`; }
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
async function ownerRefresh() {
  const s = await api("/api/owner");
  const sc = s.scanner || {}, ev = s.evaluation || {}, host = s.host || {}, rec = host.record || {}, screen = host.screen || {};
  const checks = Object.entries(ev.checks || {}).map(([k, v]) => `<li class="${v ? "pass" : "fail"}">${v ? "✓" : "✗"} ${esc(k)}</li>`).join("");
  $("owner-body").innerHTML =
    `<h3>Scanner</h3>` + row("Firmware state", esc(sc.firmware_state)) + row("Trigger travel", `${fmt(sc.trigger_travel_mm, 2)} mm`) +
    row("Trigger switch", sc.trigger_switch ? "closed" : "open") + row("Seat sensor", sc.seated ? "seated in cradle" : "not seated") +
    row("Imager frames / reads", `${sc.frames} / ${sc.reads}`) + row("Keypad keys pressed", esc((s.keys_pressed || []).join(", ") || "none")) +
    `<h3>Host application</h3>` + row("Screen", esc(`${screen.title || ""}${(screen.lines || []).length ? " · " + screen.lines.join(" · ") : ""}`)) +
    row("Location recorded", esc(rec.location ?? "none")) + row("Counts", esc(JSON.stringify(rec.counts || {}))) +
    row("Entries / errors", esc(`${(rec.entries || []).length} / ${(rec.errors || []).length}`)) +
    `<h3>Evaluation</h3>` + row("Success", ev.success ? "yes" : "no", ev.success ? "pass" : "fail") + row("Score", fmt(ev.score, 2)) +
    `<ul class="checks">${checks}</ul>` +
    `<h3>Simulator</h3>` + row("Tick / time", `${s.tick} / ${fmt(s.time_s, 3)} s`) + row("Reflex stops (total)", s.reflexes) +
    row("Refused joint targets (total)", s.refused_targets) + row("Fault", esc(s.fault ?? "none")) +
    row("Session", esc(`${(s.session || {}).paused ? "paused" : "accepting commands"} · ${(s.session || {}).clock || ""}`));
  const h = (await api("/api/health")).health; showHealth(h);
}

// ------------------------------------------------------------------------------------------ input
function press(source, key) { if (!enabled() || held.has(source)) return; held.set(source, {key, t: performance.now()}); }
function release(source) { held.delete(source); }
function releaseAll() { held.clear(); }
window.addEventListener("keydown", e => {
  if (!info || typing(e.target) || e.ctrlKey || e.metaKey) return;
  if (e.code === "Escape") { e.preventDefault(); releaseAll(); open = null; queue.length = 0; saturated = ""; notice("Queued chords dropped. A request already in flight still completes."); return; }
  if (e.code === "Digit1" || e.code === "Digit2") { e.preventDefault(); $("arm").value = e.code === "Digit1" ? "left" : "right"; showTcp(lastObs && lastObs.tcp); return; }
  const key = CODES[e.code];
  if (!key || !enabled()) return;
  e.preventDefault();
  press(e.code, key);
});
window.addEventListener("keyup", e => { if (CODES[e.code]) { release(e.code); e.preventDefault(); } });
window.addEventListener("blur", releaseAll);
document.addEventListener("visibilitychange", () => { if (document.hidden) releaseAll(); });
document.addEventListener("focusin", e => { if (typing(e.target)) releaseAll(); });
for (const b of document.querySelectorAll("[data-key]")) {
  b.addEventListener("pointerdown", e => { e.preventDefault(); press("btn:" + b.dataset.key, b.dataset.key); try { b.setPointerCapture(e.pointerId); } catch (_) {} });
  for (const ev of ["pointerup", "pointercancel", "lostpointercapture"]) b.addEventListener(ev, () => release("btn:" + b.dataset.key));
}
$("arm").addEventListener("change", () => showTcp(lastObs && lastObs.tcp));
$("refresh").onclick = () => run("Refresh camera", async () => {
  const r = await api("/api/observe", {});
  await showObservation(r.observation);
  $("feedback").textContent = `Camera refreshed: observation ${r.observation.observation_id}, no physics step.\nWall time ${fmt(r.observation.wall_s, 1)} s (render only).`;
  addLog("Refresh camera", `${fmt(r.observation.wall_s, 1)} s wall`);
  notice("Fresh camera image; physics did not advance.");
}, false);
$("recover").onclick = () => run("Owner recovery", async () => {
  const r = await api("/api/recover", {});
  if (r.observation) await showObservation(r.observation);
  showHealth(r.health);
  addLog("Owner recovery", `previous fault ${JSON.stringify(r.previous_fault)}`);
  notice("Recovered without reset or replay. Robot state is unchanged.");
}, false);
$("owner-refresh").onclick = () => ownerRefresh().catch(e => notice(e.message, true));

// ------------------------------------------------------------------------------------------ connect
(async () => {
  if (!token) { notice("Missing page token: open the full URL printed by manual.py (it ends with #token=…).", true); $("badge").textContent = "Not connected"; return; }
  try {
    const s = await api("/api/session");
    info = s.session;
    $("badge").textContent = info.role === "owner" ? "Owner key · privileged panels" : "Policy key · robot view";
    $("badge").classList.add(info.role);
    $("task").textContent = info.instruction || "No instruction";
    $("identity").textContent = `${info.robot} · arms ${info.arms.join(", ")} · cameras ${info.cameras.join(", ") || "none"} · physics tick ${info.dt_s ? info.dt_s * 1000 + " ms" : "1 ms"}`;
    $("owner").hidden = info.role !== "owner";
    showHealth(s.health);
    if (s.last_observation) await showObservation(s.last_observation);
    if (info.role === "owner") ownerRefresh().catch(e => notice(e.message, true));
    $("refresh").disabled = false;
    if (!s.last_observation && !s.busy) {
      await run("First camera image", async () => {
        const r = await api("/api/observe", {});
        await showObservation(r.observation);
        $("feedback").textContent = `First observation ${r.observation.observation_id}: render only, no physics step (${fmt(r.observation.wall_s, 1)} s).`;
        addLog("First camera image", `${fmt(r.observation.wall_s, 1)} s wall`);
      }, false);
    }
    notice("Connected. Hold keys to drive the selected arm; chords queue while a request runs.");
  } catch (e) { notice(e.message, true); $("badge").textContent = "Not connected"; }
})();

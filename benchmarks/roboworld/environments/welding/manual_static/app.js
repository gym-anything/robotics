"use strict";
// Welding manual page. Talks only to the local proxy (welding/manual.py), which holds the GAR key; the page token comes
// from the URL fragment. One robot request at a time: a key pressed while a command runs is not queued. Physics advances
// only when a command is sent; camera images are fetched only after a command or when asked for.
const $ = id => document.getElementById(id);
const token = (() => {
  const m = location.hash.match(/token=([^&]+)/);
  if (m) { sessionStorage.setItem("weldingManualToken", decodeURIComponent(m[1])); history.replaceState(null, "", location.pathname); }
  return sessionStorage.getItem("weldingManualToken") || "";
})();

let info = null, inflight = null, lastObs = null, lastWall = null, frameUrls = [], viewer = null, viewerLoading = null,
    owner = null, diagUrl = null;
const logLines = [];

// ------------------------------------------------------------------------------------------------- helpers
async function api(path, body, binary = false) {
  const options = {headers: {"X-Manual-Token": token}, cache: "no-store"};
  if (body !== undefined) { options.method = "POST"; options.headers["Content-Type"] = "application/json"; options.body = JSON.stringify(body); }
  const r = await fetch(path, options);
  if (binary && r.ok) return {blob: await r.blob(), headers: r.headers};
  const data = await r.json().catch(() => ({ok: false, error: "HTTP " + r.status}));
  if (!r.ok || data.ok === false) {
    const e = new Error(data.error || ("HTTP " + r.status)); e.status = r.status; e.detail = data.detail; e.busy = !!data.busy; throw e;
  }
  return data;
}
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
const fmt = (v, d = 1) => (v === null || v === undefined || Number.isNaN(Number(v))) ? "–" : Number(v).toFixed(d);
const signed = (v, d = 2) => (Number(v) >= 0 ? "+" : "") + Number(v).toFixed(d);
const row = (k, v, cls = "") => `<div class="kv ${cls}"><span>${esc(k)}</span><b>${v}</b></div>`;
function notice(text, error = false) { $("notice").textContent = text; $("notice").classList.toggle("error", error); }
// a field that takes typing: text-like inputs, text areas, selects, editable content (a checkbox or a button does not)
function typing(el) {
  if (!el || !el.closest) return false;
  if (el.closest("textarea,select,[contenteditable='true']")) return true;
  const input = el.closest("input");
  return !!input && !["checkbox", "radio", "button", "submit", "reset", "range", "color"].includes(input.type);
}
function num(id, label) {
  const raw = $(id).value.trim(), v = Number(raw);
  if (raw === "" || !Number.isFinite(v)) throw new Error(`${label} must be a number`);
  return v;
}
function arm() { return $("arm").value; }
function handArm() { return info && info.hand_arms.length ? info.hand_arms[0] : null; }
function duration() { return num("dur", "Duration"); }
function addLog(what, result, cls = "") {
  logLines.unshift({t: new Date().toLocaleTimeString(), what, result, cls});
  logLines.length = Math.min(logLines.length, 60);
  $("log").replaceChildren(...logLines.map(l => {
    const li = document.createElement("li");
    if (l.cls) li.className = l.cls;
    li.textContent = `${l.t}  ${l.what} → ${l.result}`;
    return li;
  }));
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

// ------------------------------------------------------------------------------------------------- requests
async function run(label, job, physics = true) {
  if (!info) return false;
  if (inflight) {
    notice(`“${inflight.label}” is still running (${((performance.now() - inflight.t0) / 1000).toFixed(0)} s); “${label}” was not sent and is not queued.`, true);
    return false;
  }
  inflight = {label, t0: performance.now(), physics};
  document.body.classList.add("busy");
  $("refresh").disabled = true; $("frames").classList.add("rendering");
  const typical = lastWall ? `the last request took ${fmt(lastWall, 0)} s` : "about 4 s per camera image on the shared GPU";
  notice(physics ? `${label}: sent. Physics, then the camera render (${typical})…` : `${label}: no physics step (${typical})…`);
  const ticker = setInterval(() => {
    if (inflight) $("held").textContent = `Running: ${inflight.label} · ${((performance.now() - inflight.t0) / 1000).toFixed(0)} s`;
  }, 500);
  let ok = true;
  try { await job(); }
  catch (e) { ok = false; showRefusal(label, e); }
  finally {
    clearInterval(ticker);
    inflight = null; document.body.classList.remove("busy");
    $("frames").classList.remove("rendering"); $("refresh").disabled = !info;
    $("held").textContent = "Ready";
    if (info.role === "owner") await ownerRefresh().catch(e => notice(e.message, true));
    else api("/api/health").then(r => showHealth(r.health)).catch(() => {});
  }
  return ok;
}
function showRefusal(label, e) {
  if (e.busy) { notice(`${label}: not sent; ${e.message}.`, true); addLog(label, "not sent (busy)", "refused"); return; }
  const d = e.detail || {};
  const lines = [`${label}`, `REFUSED: ${e.message}`];
  if (d.committed === false) lines.push("committed: false — nothing ran, the simulated clock did not advance.");
  if (d.page_hint) lines.push(`Page reading of the documented ranges: ${d.page_hint}`);
  if (d.sent) lines.push("Sent: " + JSON.stringify(d.sent));
  if (d.health) { lines.push(`Session health after: ready ${d.health.ready}, runtime_ok ${d.health.runtime_ok}`); showHealth(d.health); }
  $("feedback").textContent = lines.join("\n");
  $("feedback").classList.add("refused");
  notice(`${label}: ${e.message}`, true);
  addLog(label, "refused: " + e.message, "refused");
}
async function command(req, label) {
  return run(label, async () => {
    const r = await api("/api/command", req);
    await showResult(r);
  });
}
async function showResult(r) {
  $("feedback").classList.remove("refused");
  const f = [r.label, "Sent: " + JSON.stringify(r.sent)];
  f.push(`Committed: ${r.committed} · request ${r.request_id || "–"} · wall time ${fmt(r.wall_s, 1)} s (physics + camera render)`);
  if (r.clock) f.push(`Simulated clock ${fmt(r.clock.before_s, 3)} → ${fmt(r.clock.after_s, 3)} s (${r.clock.ticks} ticks)`);
  if (r.info) {
    f.push(`Backend: requested ${r.info.requested_ticks} ticks, executed ${r.info.executed_ticks}${r.info.termination ? " · " + r.info.termination : ""}${r.info.physics_s !== null && r.info.physics_s !== undefined ? ` · physics ${fmt(r.info.physics_s, 1)} s` : ""}`);
    const ev = (r.info.events || []).slice(-4);
    if (ev.length) f.push("Last events: " + ev.map(e => JSON.stringify(e)).join(" · "));
  }
  for (const [a, d] of Object.entries(r.tcp_delta || {})) {
    const moved = Math.hypot(...d.position_mm), turned = Math.hypot(...d.rotation_deg);
    if (moved < 0.005 && turned < 0.005 && !(d.finger_opening_mm && Math.abs(d.finger_opening_mm) > 0.005)) continue;
    f.push(`${a}: TCP ${d.position_mm.map(v => signed(v)).join(", ")} mm · turned ${d.rotation_deg.map(v => signed(v)).join(", ")}° (rotation vector)` +
           (d.finger_opening_mm !== undefined ? ` · fingers ${signed(d.finger_opening_mm)} mm` : ""));
  }
  if (r.observation_error) f.push(`Committed, but no image: ${r.observation_error.message} Do not resend.`);
  $("feedback").textContent = f.join("\n");
  if (r.observation) await showObservation(r.observation);
  addLog(r.label, `${r.clock ? r.clock.ticks + " ticks, " : ""}${fmt(r.wall_s, 1)} s wall`);
  notice(`Done in ${fmt(r.wall_s, 1)} s${r.clock ? `; the clock advanced ${r.clock.ticks} ticks` : ""}.`);
  if (r.wall_s) lastWall = r.wall_s;
}

// ------------------------------------------------------------------------------------------------- observation display
async function showObservation(obs) {
  const fresh = [];
  try {
    for (const name of obs.cameras) {
      const {blob} = await api(`/api/camera/${encodeURIComponent(name)}?observation_id=${obs.observation_id}`, undefined, true);
      fresh.push([name, URL.createObjectURL(blob)]);
    }
  } catch (e) { for (const [, u] of fresh) URL.revokeObjectURL(u); throw e; }
  for (const u of frameUrls) URL.revokeObjectURL(u);
  frameUrls = fresh.map(f => f[1]);
  $("frames").replaceChildren(...fresh.map(([name, url]) => {
    const fig = document.createElement("figure"), img = document.createElement("img"), cap = document.createElement("figcaption");
    img.src = url; img.alt = `${name} camera at observation ${obs.observation_id}`; img.dataset.camera = name; img.dataset.observation = obs.observation_id;
    cap.textContent = `${name}${name === "weld_camera" ? " (process camera: dark unless the arc burns)" : ""} · observation ${obs.observation_id} · sim ${fmt(obs.sim_time_s, 3)} s`;
    fig.append(img, cap); return fig;
  }));
  if (!fresh.length) $("frames").innerHTML = '<div class="empty">This session has no robot cameras (launch it with --backend-options-json \'{"cameras":["bench","side","weld_camera"]}\').</div>';
  $("frames").classList.toggle("multi", fresh.length > 1);
  $("frames").dataset.observation = obs.observation_id;
  lastObs = obs;
  if (obs.wall_s) lastWall = obs.wall_s;
  const tick = info ? Math.round(obs.sim_time_s / info.dt_s) : null;
  $("stamp").textContent = `Observation ${obs.observation_id} · simulated time ${fmt(obs.sim_time_s, 3)} s (tick ${tick}) · command sequence ${obs.sequence} · request wall time ${fmt(obs.wall_s, 1)} s`;
  $("render-status").textContent = `Images of observation ${obs.observation_id}`;
  showTool(obs.tool); showRobot(obs); showJoints(obs.joints);
}
function showTool(t) {
  if (!t) { $("tool").innerHTML = '<p class="muted">No source feedback in this observation.</p>'; return; }
  const on = v => Number(v) > 0.5;
  $("tool").innerHTML =
    row("Arc", on(t.arc_on) ? "ON" : "off", on(t.arc_on) ? "hot" : "") + row("Current", `${fmt(t.current_a, 1)} A`) +
    row("Voltage", `${fmt(t.voltage_v, 2)} V`) + row("Wire feed", `${fmt(t.wire_feed_m_min, 2)} m/min`) +
    row("Wire touching work", on(t.touch) ? "yes" : "no", on(t.touch) ? "pass" : "") +
    row("Gas", on(t.gas_on) ? "on" : "off") + row("Fault", on(t.fault) ? "FAULT" : "none", on(t.fault) ? "fail" : "");
}
function showRobot(obs) {
  const tcp = obs.tcp, rows = [row("Simulated clock", `${fmt(obs.sim_time_s, 3)} s · tick ${info ? Math.round(obs.sim_time_s / info.dt_s) : "?"}`)];
  if (!tcp) rows.push(row("TCP", info && !info.tcp_available ? `unavailable (${esc(info.tcp_error)})` : "–"));
  for (const [a, t] of Object.entries(tcp || {})) {
    const p = t.position_m.map(v => fmt(v * 1000, 1)).join(", ");
    const title = a === info.torch_arm ? `Torch work point (${a})` : `Hand TCP (${a})`;
    rows.push(row((a === arm() ? "▶ " : "") + title, `${p} mm`));
    if (t.work_deg !== undefined) rows.push(row("  torch angles", `work ${fmt(t.work_deg, 1)}° · push ${fmt(t.push_deg, 1)}°`));
    if (t.finger_opening_m !== undefined && t.finger_opening_m !== null) rows.push(row("  finger opening", `${fmt(t.finger_opening_m * 1000, 2)} mm`));
  }
  $("robot").innerHTML = rows.join("");
}
function showJoints(joints) {
  if (!joints || !info) return;
  const ownerView = info.role === "owner";
  const head = `<tr><th>Joint</th><th>Position</th><th>Velocity</th>${ownerView ? "<th>Target</th>" : ""}<th>Limits</th></tr>`;
  const body = Object.entries(joints).map(([n, j]) => {
    const spec = info.joints[n] || {}, prismatic = n.includes("finger");
    const v = x => x === undefined || x === null ? "–" : prismatic ? `${fmt(x * 1000, 2)} mm` : `${fmt(x, 4)} (${fmt(x * 180 / Math.PI, 1)}°)`;
    const vel = j.velocity === undefined ? "–" : prismatic ? `${fmt(j.velocity * 1000, 1)} mm/s` : `${fmt(j.velocity, 3)} rad/s`;
    const lim = spec.lower === undefined || spec.lower === null ? "–" : prismatic ? `${fmt(spec.lower * 1000, 1)}…${fmt(spec.upper * 1000, 1)} mm` : `${fmt(spec.lower, 3)}…${fmt(spec.upper, 3)} rad`;
    return `<tr><td>${esc(n)}</td><td>${v(j.position)}</td><td>${vel}</td>${ownerView ? `<td>${v(j.target)}</td>` : ""}<td>${lim}</td></tr>`;
  }).join("");
  $("joints").innerHTML = `<table>${head}${body}</table>`;
  $("joints-note").textContent = ownerView ? "observation proprioception; targets are owner-only" : "observation proprioception";
}

// ------------------------------------------------------------------------------------------------- owner panels
async function ownerRefresh() {
  if (!info || info.role !== "owner") return;
  const s = await api(`/api/owner?pose=${viewer ? 1 : 0}`);
  owner = s;
  const ev = s.evaluation || {};
  const checks = Object.entries(ev.checks || {}).map(([k, c]) => {
    const values = Object.entries(c).filter(([kk]) => kk !== "pass").map(([kk, vv]) => `${kk} ${typeof vv === "number" ? fmt(vv, 3) : JSON.stringify(vv)}`).join(" · ");
    return `<li class="${c.pass ? "pass" : "fail"}"><span>${c.pass ? "✓" : "✗"} ${esc(k)}</span><small>${esc(values)}</small></li>`;
  }).join("");
  $("eval").innerHTML = row("Task", esc(`${(s.config || {}).task} · seed ${(s.config || {}).seed} · ${(s.config || {}).robot}`)) +
    row("Success", ev.success ? "yes" : "no", ev.success ? "pass" : "fail") +
    row("Score", ev.reward ? `${fmt(ev.score, 3)} (${esc(ev.reward.score_rule)}; progress ${fmt(ev.progress, 3)})`
                           : `${fmt(ev.score, 3)} (fraction of checks passed)`) +
    `<ul class="checks">${checks || '<li class="muted">No checks reported</li>'}</ul>` +
    row("Evaluated at", `tick ${s.tick} · ${fmt(s.time_s, 3)} s${s.terminated ? " · time limit reached" : ""}`);
  const t = s.tool || {}, c = t.commanded || {}, p = s.process || {}, b = s.bond || {}, rb = s.robot || {}, h = s.history || {};
  $("state").innerHTML =
    `<h3>Source (commanded by the robot)</h3>` +
    row("Wire feed / voltage", esc(`${fmt(c.wire_feed_m_min, 2)} m/min · ${fmt(c.voltage_v, 1)} V`)) +
    row("Arc on / touch sense / inch", esc(`${c.arc_on} · ${c.touch_sense} · ${c.wire_inch}`)) +
    row("Wire stick-out", `${fmt(t.stickout_mm, 2)} mm`) + row("Source fault", esc(t.fault ?? "none"), t.fault ? "fail" : "") +
    `<h3>Process</h3>` +
    row("Welding circuit closed", p.circuit_closed ? "yes" : "no") + row("Arc length", `${fmt(p.arc_length_mm, 2)} mm`) +
    row("Transfer mode / air under the arc", esc(`${p.transfer_mode ?? "no arc"} · ${fmt(100*(p.air_fraction ?? 0), 1)} %`)) +
    row("Deposited filler", `${fmt(p.deposited_mm3, 1)} mm³`) + row("Liquid sections", esc(p.liquid_sections ?? "–")) +
    row("Energy/mass ledger residual", fmt(p.ledger_residual_rel, 6)) +
    row("Bond", esc(`${b.active ? "active" : "none"}${b.broken ? " (broken)" : ""} · capacity ${fmt(b.capacity_n, 0)} N`)) +
    `<h3>History</h3>` +
    row("Arc time / strikes / max current", esc(`${fmt(h.arc_s, 3)} s · ${h.strikes} · ${fmt(h.max_current_a, 0)} A`)) +
    row("Wire touched work / gun contact", esc(`${h.wire_touch} · ${h.gun_contact_ticks} ticks`)) +
    row("Source faults", esc((h.faults || []).join(", ") || "none")) +
    `<h3>Robot</h3>` +
    row("Protective stops (last 10)", esc((rb.reflexes || []).length ? JSON.stringify(rb.reflexes.slice(-3)) : "none")) +
    row("Refused joint targets / stopped arms", esc(`${rb.refused_targets} · ${(rb.stopped_arms || []).join(", ") || "none"}`)) +
    `<h3>Simulator</h3>` +
    row("Tick / time / generation", `${s.tick} · ${fmt(s.time_s, 3)} s · ${s.generation}`) + row("Fault", esc(s.fault ?? "none"), s.fault ? "fail" : "") +
    row("Session", esc(s.session ? `${s.session.paused ? "paused" : "accepting commands"} · ${s.session.clock}` : "–")) +
    row("Coupon offset (dx, dy, yaw)", esc(JSON.stringify((s.config || {}).coupon_offset))) +
    `<h3>Last events</h3><pre class="events">${esc((s.events || []).slice(-6).map(e => JSON.stringify(e)).join("\n") || "none")}</pre>`;
  if (viewer && s.pose) { viewer.update(s.pose); $("view-status").textContent = `Posed at tick ${s.pose.tick} · ${fmt(s.pose.time_s, 3)} s`; }
  api("/api/health").then(r => showHealth(r.health)).catch(() => {});
}
async function checkpointList() {
  const r = await api("/api/checkpoints");
  const list = r.checkpoints || [];
  $("cp-list").replaceChildren(...(list.length ? list.map(c => new Option(`${c.label} · tick ${c.tick}`, c.label)) : [new Option("None saved", "")]));
}

// ------------------------------------------------------------------------------------------------- tabs, 3D, diagnostic
function tab(which) {
  for (const [t, p] of [["tab-cameras", "pane-cameras"], ["tab-3d", "pane-3d"], ["tab-diag", "pane-diag"]]) {
    const on = t === which;
    $(t).setAttribute("aria-selected", String(on)); $(p).hidden = !on;
  }
  if (which === "tab-3d") open3d();
}
async function open3d() {
  if (viewer) { viewer.resize(); return; }
  if (viewerLoading) return;
  viewerLoading = (async () => {
    $("view-status").textContent = "Loading three.js and the cell's colliders…";
    try {
      const {CellViewer} = await import("/viewer.js");
      const scene = await api("/api/scene");
      viewer = new CellViewer($("viewport"), scene);
      $("view-note").textContent = scene.note + " Drag: orbit · right-drag: pan · wheel: zoom. Moving the view never steps physics.";
      $("view-status").textContent = "Loaded; waiting for the session's pose…";
      if (!inflight) await ownerRefresh();       // sets "Posed at tick N · t s" (the one status format)
    } catch (e) { $("view-status").textContent = "3D view failed: " + e.message; viewerLoading = null; throw e; }
  })();
  await viewerLoading.catch(e => notice(e.message, true));
}
for (const b of document.querySelectorAll("[data-view]")) b.addEventListener("click", () => viewer && viewer.preset(b.dataset.view));
$("show-robots").addEventListener("change", () => viewer && viewer.showRobots($("show-robots").checked));
$("show-screens").addEventListener("change", () => viewer && viewer.showScreens($("show-screens").checked));
$("tab-cameras").onclick = () => tab("tab-cameras");
$("tab-3d").onclick = () => tab("tab-3d");
$("tab-diag").onclick = () => tab("tab-diag");
$("diag-render").onclick = () => run("Diagnostic render", async () => {
  const before = owner ? owner.tick : null;
  const {blob, headers} = await api("/api/render", undefined, true);
  if (diagUrl) URL.revokeObjectURL(diagUrl);
  diagUrl = URL.createObjectURL(blob);
  $("diag-image").src = diagUrl; $("diag-image").hidden = false; $("diag-empty").hidden = true;
  await $("diag-image").decode().catch(() => {});
  const wall = headers.get("X-Wall-Seconds");
  $("diag-status").textContent = `Rendered in ${fmt(wall, 1)} s at tick ${before ?? "?"}; no physics step.`;
  addLog("Diagnostic render", `${fmt(wall, 1)} s wall, no physics`);
  notice("Diagnostic render done; physics did not advance.");
}, false);

// ------------------------------------------------------------------------------------------------- commands
function mult(ev) { return ev && ev.shiftKey ? 4 : ev && ev.altKey ? 0.2 : 1; }
function jog(axis, sign, ev) {
  const a = arm(), step = num("step", "Step") * mult(ev), d = duration();
  return command({kind: "jog", arm: a, axis, sign, step_mm: step, duration_s: d},
                 `${a}: ${sign > 0 ? "+" : "−"}${"xyz"[axis]} ${+step.toFixed(3)} mm`);
}
function turn(axis, sign, ev) {
  const a = arm(), deg = num("turn", "Turn") * mult(ev), d = duration();
  return command({kind: "rotate", arm: a, axis, sign, step_deg: deg, duration_s: d},
                 `${a}: turn ${sign > 0 ? "+" : "−"}${+deg.toFixed(3)}° about ${"xyz"[axis]}`);
}
function hand(mode) {
  const h = handArm();
  if (!h) { notice("This deployment has no hand.", true); return; }
  const req = {kind: "gripper", arm: h, mode, duration_s: duration()};
  if (mode === "force") req.effort_n = num("grip-force", "Grip force");
  return command(req, `hand ${mode}${mode === "force" ? " " + req.effort_n + " N" : ""}`);
}
function tool(outputs, label) { return command({kind: "tool", outputs, ticks: 1}, label); }
function inch(direction) { return command({kind: "inch", direction, duration_s: duration()}, `inch ${direction} ${duration()} s`); }
function advance(seconds) { return command({kind: "hold", duration_s: seconds}, `advance ${seconds} s`); }
function safely(fn) { return (ev) => { try { const p = fn(ev); if (p && p.catch) p.catch(e => notice(e.message, true)); } catch (e) { notice(e.message, true); } }; }

const KEYS = {KeyW: ["jog", 1, -1], KeyS: ["jog", 1, 1], KeyA: ["jog", 0, 1], KeyD: ["jog", 0, -1], KeyR: ["jog", 2, 1], KeyF: ["jog", 2, -1],
  KeyI: ["turn", 0, 1], KeyK: ["turn", 0, -1], KeyJ: ["turn", 1, 1], KeyL: ["turn", 1, -1], KeyU: ["turn", 2, 1], KeyO: ["turn", 2, -1],
  KeyQ: ["hand", "close"], KeyE: ["hand", "open"], KeyH: ["hand", "force"], BracketLeft: ["inch", "backward"], BracketRight: ["inch", "forward"],
  Space: ["advance"]};
const BUTTON_CODES = {w: "KeyW", s: "KeyS", a: "KeyA", d: "KeyD", r: "KeyR", f: "KeyF", i: "KeyI", k: "KeyK", j: "KeyJ", l: "KeyL", u: "KeyU",
  o: "KeyO", q: "KeyQ", e: "KeyE", h: "KeyH", "[": "BracketLeft", "]": "BracketRight", " ": "Space"};
function dispatchKey(code, ev) {
  const k = KEYS[code];
  if (!k) return false;
  if (k[0] === "jog") return jog(k[1], k[2], ev);
  if (k[0] === "turn") return turn(k[1], k[2], ev);
  if (k[0] === "hand") return hand(k[1]);
  if (k[0] === "inch") return inch(k[1]);
  if (k[0] === "advance") return advance(duration());
  return false;
}
window.addEventListener("keydown", e => {
  if (!info || typing(e.target) || e.ctrlKey || e.metaKey) return;
  if (e.code === "Digit1" || e.code === "Digit2") {
    const want = e.code === "Digit1" ? info.torch_arm : handArm();
    if (want) { $("arm").value = want; if (lastObs) showRobot(lastObs); }
    e.preventDefault(); return;
  }
  if (!KEYS[e.code] || !$("enabled").checked) return;
  e.preventDefault();
  if (e.repeat) return;
  for (const b of document.querySelectorAll("[data-key]")) b.classList.toggle("down", BUTTON_CODES[b.dataset.key] === e.code);
  setTimeout(() => { for (const b of document.querySelectorAll("[data-key].down")) b.classList.remove("down"); }, 250);
  safely(ev => dispatchKey(e.code, ev))(e);
});
// a bound key pressed while a button has focus must not also click that button (Space activates buttons on keyup)
window.addEventListener("keyup", e => {
  if (info && KEYS[e.code] && !typing(e.target) && $("enabled").checked && e.target && e.target.tagName === "BUTTON") e.preventDefault();
});
for (const b of document.querySelectorAll("[data-key]")) b.addEventListener("click", safely(ev => dispatchKey(BUTTON_CODES[b.dataset.key], ev)));
$("hand-open").onclick = safely(() => hand("open"));
$("hand-close").onclick = safely(() => hand("close"));
$("hand-force").onclick = safely(() => hand("force"));
$("preset").onclick = safely(() => {
  const p = info.schedule_preset;
  $("wfs").value = p.wire_feed_m_min; $("volts").value = p.voltage_v;
  return tool({wire_feed_m_min: p.wire_feed_m_min, voltage_v: p.voltage_v}, `schedule preset (${p.wire_feed_m_min} m/min, ${p.voltage_v} V)`);
});
$("schedule-send").onclick = safely(() => {
  const o = {wire_feed_m_min: num("wfs", "Wire feed"), voltage_v: num("volts", "Voltage")};
  return tool(o, `set points ${o.wire_feed_m_min} m/min, ${o.voltage_v} V`);
});
$("gas-on").onclick = safely(() => tool({gas_on: true}, "gas on"));
$("gas-off").onclick = safely(() => tool({gas_on: false}, "gas off"));
$("touch-on").onclick = safely(() => tool({touch_sense: true}, "touch sense on"));
$("touch-off").onclick = safely(() => tool({touch_sense: false}, "touch sense off"));
$("inch-fwd").onclick = safely(() => inch("forward"));
$("inch-back").onclick = safely(() => inch("backward"));
$("inch-stop").onclick = safely(() => tool({wire_inch: "stop"}, "stop inching"));
$("arc-on").onclick = safely(() => tool({arc_on: true}, "ARC ON (weld start)"));
$("arc-off").onclick = safely(() => tool({arc_on: false}, "arc off"));
$("advance").onclick = safely(() => advance(num("advance-s", "Advance")));
$("use-tcp").onclick = () => {
  const t = lastObs && lastObs.tcp && lastObs.tcp[arm()];
  if (!t) { notice("No TCP yet: refresh the cameras first.", true); return; }
  ["tx", "ty", "tz"].forEach((id, i) => $(id).value = (t.position_m[i] * 1000).toFixed(2));
  if (t.work_deg !== undefined) { $("work").value = t.work_deg.toFixed(1); $("push").value = t.push_deg.toFixed(1); }
};
$("goto").onclick = safely(() => {
  const p = [num("tx", "x"), num("ty", "y"), num("tz", "z")];
  return command({kind: "goto", arm: arm(), position_mm: p, duration_s: duration()}, `${arm()} to (${p.join(", ")}) mm`);
});
$("orient").onclick = safely(() => {
  const w = num("work", "Work angle"), p = num("push", "Push angle");
  return command({kind: "orient", arm: info.torch_arm, work_deg: w, push_deg: p, duration_s: duration()}, `torch to ${w}° work, ${p}° push`);
});
function jointChanged() {
  const n = $("joint").value, spec = info.joints[n] || {}, prismatic = n.includes("finger");
  const cur = lastObs && lastObs.joints && lastObs.joints[n] ? lastObs.joints[n].position : null;
  $("joint-limits").textContent = (spec.lower === undefined || spec.lower === null ? "Limits unknown" :
    `Limits ${spec.lower} … ${spec.upper} ${prismatic ? "m (half gap of one finger)" : "rad"}`) +
    (cur !== null ? ` · now ${cur.toFixed(5)}` : "") + ` · named_joint_target over the Jog duration; the robot refuses a target outside the limits.`;
  if (cur !== null) $("joint-value").value = cur.toFixed(5);
}
$("joint").onchange = jointChanged;
$("joint-send").onclick = safely(() => {
  const n = $("joint").value, v = num("joint-value", "Target");
  return command({kind: "joint", name: n, value: v, duration_s: duration()}, `${n} → ${v}`);
});
$("arm").addEventListener("change", () => lastObs && showRobot(lastObs));
$("refresh").onclick = () => run("Refresh cameras", async () => {
  const r = await api("/api/observe", {});
  await showObservation(r.observation);
  $("feedback").classList.remove("refused");
  $("feedback").textContent = `Cameras refreshed: observation ${r.observation.observation_id}, no physics step.\nWall time ${fmt(r.observation.wall_s, 1)} s (render only).`;
  addLog("Refresh cameras", `${fmt(r.observation.wall_s, 1)} s wall, no physics`);
  notice("Fresh camera images; physics did not advance.");
}, false);
$("recover").onclick = () => run("Owner recovery", async () => {
  const r = await api("/api/recover", {});
  if (r.observation) await showObservation(r.observation);
  showHealth(r.health);
  addLog("Owner recovery", `previous fault ${JSON.stringify(r.previous_fault)}`);
  notice("Recovered without reset or replay. Robot state is unchanged.");
}, false);
$("owner-refresh").onclick = () => ownerRefresh().catch(e => notice(e.message, true));
$("cp-save").onclick = () => run("Save checkpoint", async () => {
  const label = $("cp-label").value.trim();
  if (!label) throw new Error("Enter a checkpoint label");
  const r = await api("/api/checkpoint/save", {label});
  await checkpointList();
  $("cp-list").value = label;
  $("feedback").classList.remove("refused");
  $("feedback").textContent = `Saved checkpoint “${r.checkpoint.label}” at tick ${r.checkpoint.tick} (generation ${r.checkpoint.generation}); physics did not advance.`;
  addLog(`Save checkpoint ${label}`, `tick ${r.checkpoint.tick}`);
  notice(`Saved “${label}” at tick ${r.checkpoint.tick}.`);
}, false);
$("cp-restore").onclick = () => run("Restore checkpoint", async () => {
  const cp = $("cp-list").value;
  if (!cp) throw new Error("Choose a checkpoint");
  const r = await api("/api/checkpoint/restore", {checkpoint: cp});
  if (r.observation) await showObservation(r.observation);
  const i = r.info || {};
  $("feedback").classList.remove("refused");
  $("feedback").textContent = `Restored “${cp}”: tick ${i.tick}, generation ${i.generation}, ${i.continuity}. A new timeline starts here.`;
  addLog(`Restore ${cp}`, `tick ${i.tick}, generation ${i.generation}`);
  notice(`Restored “${cp}” (tick ${i.tick}). This is a discontinuity, not physics.`);
}, false);

// ------------------------------------------------------------------------------------------------- connect
(async () => {
  if (!token) { notice("Missing page token: open the full URL printed by manual.py (it ends with #token=…).", true); $("badge").textContent = "Not connected"; return; }
  try {
    const s = await api("/api/session");
    info = s.session;
    const ownerView = info.role === "owner";
    $("badge").textContent = ownerView ? "Owner key · privileged panels" : "Policy key · robot view";
    $("badge").classList.add(info.role);
    $("robot-tag").textContent = `${info.robot} · policy actions only`;
    $("task").textContent = info.instruction || "No instruction";
    $("identity").textContent = `${info.robot} · arms ${info.arms.join(", ")} (torch on ${info.torch_arm}) · cameras ${info.cameras.join(", ") || "none"} · physics tick ${info.dt_s * 1000} ms · actions ${info.action_types.join(", ")}`;
    $("arm").replaceChildren(...info.arms.map(a => new Option(a === info.torch_arm ? `Torch work point (${a}) · 1` : `Hand TCP (${a}) · 2`, a)));
    $("arm").value = info.torch_arm;
    const h = handArm();
    $("hand-panel").hidden = !h; $("hand-keys").hidden = !h;
    if (h) {
      $("hand-arm").textContent = `on '${h}'`;
      if (info.grip_max_n) { $("grip-force").max = info.grip_max_n; $("hand-hint").innerHTML += ` Force mode up to ${esc(info.grip_max_n)} N total (the hand's finger effort limit).`; }
    }
    $("orient-box").hidden = !info.tcp_available;
    $("preset").textContent = `Schedule preset: ${info.schedule_preset.wire_feed_m_min} m/min, ${info.schedule_preset.voltage_v} V`;
    $("preset").title = `${info.schedule_preset.label} — ${info.schedule_preset.source}`;
    $("joint").replaceChildren(...Object.keys(info.joints).map(n => new Option(n, n)));
    for (const id of ["owner-eval", "owner-state", "owner-cp"]) $(id).hidden = !ownerView;
    $("tab-diag").hidden = !ownerView;
    $("tab-3d").hidden = !(s.viewer && s.viewer.available);
    if (ownerView && s.viewer && !s.viewer.available) $("tab-3d").title = s.viewer.reason || "";
    showHealth(s.health);
    if (s.last_observation) await showObservation(s.last_observation);
    if (ownerView) { await ownerRefresh().catch(e => notice(e.message, true)); await checkpointList().catch(() => {}); }
    $("refresh").disabled = false;
    jointChanged();
    if (!s.last_observation && !s.busy) {
      await run("First camera images", async () => {
        const r = await api("/api/observe", {});
        await showObservation(r.observation);
        $("feedback").textContent = `First observation ${r.observation.observation_id}: render only, no physics step (${fmt(r.observation.wall_s, 1)} s).`;
        addLog("First camera images", `${fmt(r.observation.wall_s, 1)} s wall, no physics`);
      }, false);
      jointChanged();
    }
    notice("Connected. Press a key or a button to send one command; nothing moves otherwise.");
    document.body.dataset.connected = "true";
  } catch (e) { notice(e.message, true); $("badge").textContent = "Not connected"; }
})();

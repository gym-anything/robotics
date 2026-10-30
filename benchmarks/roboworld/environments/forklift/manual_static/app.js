"use strict";
// Forklift manual page: talks only to the local proxy (forklift/manual.py) with the page token from the URL fragment.
const token = new URLSearchParams(location.hash.slice(1)).get("token") || "";
const $ = (id) => document.getElementById(id);
let session = null, joints = null, busy = false, groupIndex = 0, motorIndex = 0;

async function api(path, body) {
  const options = { headers: { "X-Manual-Token": token } };
  if (body !== undefined) {
    options.method = "POST";
    options.headers["Content-Type"] = "application/json";
    options.body = JSON.stringify(body);
  }
  const r = await fetch(path, options);
  const data = await r.json().catch(() => ({ ok: false, error: `HTTP ${r.status}` }));
  if (!data.ok) throw new Error(data.error || `HTTP ${r.status}`);
  return data;
}

function notice(text) { $("notice").textContent = text; }
function log(title, detail) {
  const li = document.createElement("li");
  li.textContent = `${new Date().toLocaleTimeString()} · ${title}${detail ? " · " + detail : ""}`;
  const ol = $("log");
  if (ol.firstElementChild && ol.firstElementChild.classList.contains("muted")) ol.innerHTML = "";
  ol.prepend(li);
  while (ol.children.length > 60) ol.lastElementChild.remove();
}

const short = (n) => n.split("/").pop().replace(/_joint$/, "");
const fmt = (v, d) => (typeof v === "number" ? v.toFixed(d) : "–");
function group() { return session ? session.groups[groupIndex] : null; }
function motorName() { const g = group(); return g ? g.motors[motorIndex] : null; }

function cell(tr, text) { const td = document.createElement("td"); td.textContent = text; tr.append(td); return td; }
function renderJoints() {
  const g = group();
  if (!g) return;
  $("group-title").textContent = `· ${g.name}`;
  const body = $("joint-rows");
  body.innerHTML = "";
  for (const n of g.motors) {
    const tr = document.createElement("tr");
    if (n === motorName()) tr.className = "selected";
    const j = joints && joints[n] ? joints[n] : [null, null, null];
    const m = session.motors[n];
    cell(tr, short(n)); cell(tr, fmt(j[0], 4)); cell(tr, fmt(j[1], 3)); cell(tr, fmt(j[2], 2));
    cell(tr, `${fmt(m.lower, 3)} … ${fmt(m.upper, 3)}`);
    tr.addEventListener("click", () => { motorIndex = g.motors.indexOf(n); showMotor(); });
    body.append(tr);
  }
}
function showMotor() {
  const n = motorName();
  if (!n) return;
  $("motor").value = n;
  const p = joints && joints[n] ? joints[n][0] : null;
  const m = session.motors[n];
  $("motor-pos").textContent = `position: ${p === null ? "–" : p.toFixed(4) + " rad"} · limits ${m.lower.toFixed(3)} … ${m.upper.toFixed(3)}`;
  renderJoints();
}
function selectMotor(name) {
  const gi = session.groups.findIndex((g) => g.motors.includes(name));
  if (gi < 0) return;
  groupIndex = gi; motorIndex = session.groups[gi].motors.indexOf(name);
  showMotor();
}

async function showObservation(obs) {
  if (!obs) return;
  joints = obs.joints || joints;
  const cf = obs.camera_frame;
  const frame = cf ? ` · camera frame ${cf.index} captured at ${Number(cf.capture_time_s).toFixed(3)} s` : "";
  $("stamp").textContent = `sim ${obs.sim_time_s !== undefined ? Number(obs.sim_time_s).toFixed(3) : "?"} s${frame} · observation ${obs.observation_id} · ${obs.wall_s} s wall`;
  const imu = obs.imu;
  $("imu").textContent = imu && imu.rpy_rad ? `pelvis IMU roll ${imu.rpy_rad[0].toFixed(3)} · pitch ${imu.rpy_rad[1].toFixed(3)} · yaw ${imu.rpy_rad[2].toFixed(3)} rad` : "IMU: –";
  const frames = $("frames");
  frames.innerHTML = "";
  for (const cam of obs.cameras || []) {
    const r = await fetch(`/api/camera/${encodeURIComponent(cam)}?observation_id=${obs.observation_id}`, { headers: { "X-Manual-Token": token } });
    if (!r.ok) { frames.textContent = `camera ${cam}: HTTP ${r.status}`; continue; }
    const img = document.createElement("img");
    img.src = URL.createObjectURL(await r.blob());
    img.alt = cam;
    const fig = document.createElement("figure");
    const cap = document.createElement("figcaption");
    cap.textContent = cam;
    fig.append(img, cap);
    frames.append(fig);
  }
  showMotor();
}

async function run(title, fn) {
  if (busy) { notice("A request is in flight; wait for it to finish."); return; }
  busy = true; $("badge").textContent = "Working…";
  try {
    const r = await fn();
    if (r && r.observation) await showObservation(r.observation);
    if (r && r.targets !== undefined) {
      const fb = { targets: r.targets, sim_s: r.sim_s, wall_s: r.wall_s, wall_per_sim_s: r.wall_per_sim_s, committed: r.committed,
                   observation_error: r.observation_error };
      if (r.physics_wall_per_sim_s !== undefined) fb.physics_wall_per_sim_s = r.physics_wall_per_sim_s;
      $("feedback").textContent = JSON.stringify(fb, null, 1);
      log(title, `${r.sim_s} s simulated, ${r.wall_s} s wall`);
    }
    notice(`${title}: done.`);
    if (session && session.role === "owner") await ownerRefresh();
  } catch (e) {
    notice(`${title}: ${e.message}`);
    log(title, `refused: ${e.message}`);
  } finally {
    busy = false; $("badge").textContent = session ? session.role : "?";
  }
}

function ticks() {
  const dt = session.dt_s;
  const s = Math.min(session.max_ticks * dt, Math.max(session.min_ticks * dt, Number($("dur").value) || 0.25));
  return Math.min(session.max_ticks, Math.max(session.min_ticks, Math.round(s / dt)));
}
function step(mult) { return Math.min(session.max_delta, Math.max(0.005, Number($("step").value) || 0.05) * mult); }

function command(kind, mult) {
  if (!session) return;
  if (kind === "prev" || kind === "next") {
    const g = group();
    motorIndex = (motorIndex + (kind === "next" ? 1 : g.motors.length - 1)) % g.motors.length;
    showMotor(); return;
  }
  if (kind.startsWith("g")) {
    const gi = Number(kind.slice(1)) - 1;
    if (gi >= 0 && gi < session.groups.length) { groupIndex = gi; motorIndex = 0; showMotor(); }
    return;
  }
  let body;
  if (kind === "up" || kind === "down") body = { motor: motorName(), delta: (kind === "up" ? 1 : -1) * step(mult), ticks: ticks() };
  else if (kind === "torsop" || kind === "torsom") {
    if (!session.torso) return;
    body = { motor: session.torso, delta: (kind === "torsop" ? 1 : -1) * step(mult), ticks: ticks() };
  }
  else if (["lclose", "lopen", "rclose", "ropen"].includes(kind))
    body = { hand: kind[0] === "l" ? "left" : "right", grip: kind.slice(1), ticks: ticks() };
  else if (kind === "hold") body = { ticks: ticks() };
  else return;
  const title = kind === "hold" ? "Hold" : body.hand ? `${body.grip} ${body.hand} hand` : `${kind} ${short(body.motor)}`;
  run(title, () => api("/api/command", body));
}

async function ownerRefresh() {
  try {
    const r = await api("/api/owner");
    const ev = r.evaluation || {};
    const parts = [`tick ${r.tick}`, `sim ${fmt(r.sim_time_s, 3)} s`, `finite ${r.finite}`, `fault ${r.fault === null ? "none" : r.fault}`];
    if (ev.error) parts.push(`evaluator: ${ev.error}`);
    else parts.push(`task ${ev.task} · success ${ev.success} · score ${fmt(ev.score, 3)}`);
    if (ev.strikes && ev.strikes.length) parts.push(`strikes ${JSON.stringify(ev.strikes)}`);
    if (ev.dropped && ev.dropped.length) parts.push(`dropped ${JSON.stringify(ev.dropped)}`);
    if (r.start_problems && r.start_problems.length) parts.push(`start problems ${r.start_problems.join("; ")}`);
    $("owner-summary").textContent = parts.join(" · ");
    const body = $("owner-rows");
    body.innerHTML = "";
    for (const c of ev.criteria || []) {
      const tr = document.createElement("tr");
      tr.className = c.passed ? "pass" : "fail";
      const m = typeof c.measured === "number" ? `${Number(c.measured.toPrecision(4))} ${c.unit}` : `${JSON.stringify(c.measured)} ${c.unit}`;
      cell(tr, c.name); cell(tr, m); cell(tr, c.threshold); cell(tr, c.passed ? "pass" : "fail");
      body.append(tr);
    }
  } catch (e) { $("owner-summary").textContent = e.message; }
}

const KEYS = { ArrowUp: "up", ArrowDown: "down", "[": "prev", "]": "next", w: "torsop", s: "torsom", q: "lclose", a: "lopen",
               e: "rclose", d: "ropen", " ": "hold", 1: "g1", 2: "g2", 3: "g3", 4: "g4", 5: "g5", 6: "g6", 7: "g7" };
document.addEventListener("keydown", (ev) => {
  if (!$("enabled").checked || ["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement.tagName)) return;
  if (ev.ctrlKey || ev.metaKey) return;                       // browser shortcuts stay the browser's
  const arrow = ev.key === "ArrowUp" || ev.key === "ArrowDown";
  if (ev.altKey && !arrow) return;                            // Alt+digit switches tabs in some browsers
  const kind = KEYS[ev.key] || KEYS[ev.key.toLowerCase()] || (ev.code.startsWith("Digit") ? KEYS[ev.code.slice(5)] : undefined);
  if (!kind) return;
  ev.preventDefault();
  if (ev.repeat) return;
  command(kind, ev.shiftKey ? 4 : ev.altKey ? 0.2 : 1);
});
for (const b of document.querySelectorAll("button[data-cmd]")) b.addEventListener("click", (ev) => { b.blur(); command(b.dataset.cmd, ev.shiftKey ? 4 : 1); });
$("motor").addEventListener("change", () => { selectMotor($("motor").value); $("motor").blur(); });
$("refresh").addEventListener("click", () => { $("refresh").blur(); run("Camera", () => api("/api/observe", {})); });
$("owner-refresh").addEventListener("click", ownerRefresh);

(async () => {
  try {
    const r = await api("/api/session");
    session = r.session;
    $("task").textContent = session.instruction || "(no instruction)";
    const n = Object.keys(session.motors).length;
    $("identity").textContent = `${session.robot} · ${n} motors · role ${session.role} · tick ${session.dt_s * 1e3} ms · command ${session.min_ticks * session.dt_s * 1e3} ms … ${session.max_ticks * session.dt_s} s`;
    for (const g of session.groups) {
      const og = document.createElement("optgroup");
      og.label = g.name;
      for (const m of g.motors) {
        const o = document.createElement("option");
        o.value = m; o.textContent = short(m);
        og.append(o);
      }
      $("motor").append(og);
    }
    $("group-keys").textContent = session.groups.slice(0, 7).map((g, i) => `${i + 1} ${g.name}`).join(" · ");
    $("dur").min = String(session.min_ticks * session.dt_s);
    $("badge").textContent = session.role;
    $("refresh").disabled = false;
    $("owner").hidden = session.role !== "owner";
    showMotor();
    notice("Connected. Take a camera image first (it reads the measured joint positions the nudges start from).");
    if (r.last_observation) await showObservation(r.last_observation);
    if (session.role === "owner") ownerRefresh();
  } catch (e) { notice(`Not connected: ${e.message}`); }
})();

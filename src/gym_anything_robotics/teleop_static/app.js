const { InputIntentQueue, RefreshLatch, isEditableElement } = TeleopControlRuntime;
const inputIntents = new InputIntentQueue();
const frameRefresh = new RefreshLatch();
const browserClientId = (() => {
  const key = "gym-anything-browser-client-id";
  try {
    const existing = sessionStorage.getItem(key);
    if (existing) return existing;
    const created = crypto.randomUUID();
    sessionStorage.setItem(key, created);
    return created;
  } catch (_) {
    return `dashboard-${Date.now()}-${Math.random().toString(16).slice(2)}`;
  }
})();

const state = {
  arm: "right",
  view: "free_camera",
  pressed: new Set(),
  releaseTail: false,
  lastMotionKeys: [],
  controlGeneration: 0,
  motionLevel: 0,
  commandBusy: false,
  frameBusy: false,
  statusBusy: false,
  statusRefreshPending: false,
  workScheduled: false,
  frameBeforeControl: false,
  actionsSinceFrame: 0,
  browserSyncBusy: false,
  lastBrowserCommandId: null,
  lastBrowserCommandError: null,
  lastFrameAt: 0,
  lastFrameTick: null,
  lastFrameWallMs: null,
  frameSequence: 0,
  freeFrameDirty: false,
  renderedFreeCamera: null,
  latest: null,
};

const ACTIONS_PER_RENDER = 4;
const TARGET_VIEW_COUNT = 2;

const conflictPairs = [
  ["A", "D"], ["W", "S"], ["R", "F"],
  ["I", "K"], ["J", "L"], ["U", "O"], ["Q", "E"],
];
const motionKeys = new Set("ADWSRFIKJLUOQE".split(""));
const armDirectionKeys = new Set("ADWSRFIKJLUO".split(""));
const armControlKeys = new Set([...armDirectionKeys, "ALT", "SHIFT"]);
const controllerKeys = new Set([...armControlKeys, "Q", "E"]);
const browserKeyMap = {
  ARROWLEFT: "A",
  ARROWRIGHT: "D",
  ARROWUP: "W",
  ARROWDOWN: "S",
  " ": "R",
  SPACEBAR: "R",
  C: "F",
};
const viewNames = {
  free_camera: ["Free camera", "Operator-only diagnostic camera"],
  diagnostic_scene: ["Diagnostic scene", "Privileged diagnostic view"],
  top_camera_d405: ["Top D405", "Robot-mounted deployable camera"],
  left_camera_d405: ["Left D405", "Robot-mounted deployable camera"],
  right_camera_d405: ["Right D405", "Robot-mounted deployable camera"],
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));

const freeCameraDefaults = {
  eye: [1.12, -0.47, 1.07],
  target: [0.575, 0.0, 0.8],
  fov: 42,
};
const freeCamera = {
  eye: [...freeCameraDefaults.eye],
  target: [...freeCameraDefaults.target],
  fov: freeCameraDefaults.fov,
  yaw: 0,
  pitch: 0,
  distance: 1,
  drag: null,
};
const targetPicker = {
  enabled: false,
  samples: [],
  position: null,
  rmsResidualM: null,
  conditionNumber: null,
  lastMoveErrorM: null,
  geometryError: null,
  activeMarker: null,
  moveStatus: "No move requested",
  moveFailed: false,
  busy: false,
};

function syncFreeCameraSpherical() {
  const offset = freeCamera.eye.map((value, axis) => value - freeCamera.target[axis]);
  freeCamera.distance = Math.max(0.05, Math.hypot(...offset));
  freeCamera.yaw = Math.atan2(offset[1], offset[0]);
  freeCamera.pitch = Math.asin(Math.max(-1, Math.min(1, offset[2] / freeCamera.distance)));
}

function updateFreeCameraEye() {
  const planar = freeCamera.distance * Math.cos(freeCamera.pitch);
  freeCamera.eye = [
    freeCamera.target[0] + planar * Math.cos(freeCamera.yaw),
    freeCamera.target[1] + planar * Math.sin(freeCamera.yaw),
    freeCamera.target[2] + freeCamera.distance * Math.sin(freeCamera.pitch),
  ];
}

function resetFreeCamera() {
  freeCamera.eye = [...freeCameraDefaults.eye];
  freeCamera.target = [...freeCameraDefaults.target];
  freeCamera.fov = freeCameraDefaults.fov;
  syncFreeCameraSpherical();
}

function scheduleFreeCameraFrame() {
  if (state.view !== "free_camera") return;
  targetPicker.activeMarker = null;
  renderTargetOverlay();
  state.freeFrameDirty = true;
  requestFrameRefresh();
}

function freeCameraQuery(camera = freeCamera) {
  const encodeVector = (values) => values.map((value) => Number(value).toFixed(6)).join(",");
  return `&eye=${encodeURIComponent(encodeVector(camera.eye))}`
    + `&target=${encodeURIComponent(encodeVector(camera.target))}`
    + `&fov=${encodeURIComponent(Number(camera.fov).toFixed(3))}`;
}

resetFreeCamera();

function normalizeVector(values) {
  const norm = Math.hypot(...values);
  if (!Number.isFinite(norm) || norm < 1e-12) throw new Error("Cannot construct camera ray");
  return values.map((value) => value / norm);
}

function crossVector(left, right) {
  return [
    left[1] * right[2] - left[2] * right[1],
    left[2] * right[0] - left[0] * right[2],
    left[0] * right[1] - left[1] * right[0],
  ];
}

function cameraRayAt(clientX, clientY) {
  const camera = state.renderedFreeCamera;
  if (!camera) throw new Error("Wait for the free-camera frame before selecting");
  const bounds = $("#viewport").getBoundingClientRect();
  const pixelX = clientX - bounds.left;
  const pixelY = clientY - bounds.top;
  if (pixelX < 0 || pixelX > bounds.width || pixelY < 0 || pixelY > bounds.height) {
    throw new Error("Selected pixel lies outside the camera image");
  }
  const front = normalizeVector(
    camera.target.map((value, axis) => value - camera.eye[axis])
  );
  const right = normalizeVector(crossVector(front, [0, 0, 1]));
  const up = normalizeVector(crossVector(right, front));
  const u = 2 * pixelX / bounds.width - 1;
  const v = 1 - 2 * pixelY / bounds.height;
  const alpha = Math.tan(camera.fov * Math.PI / 360);
  const aspect = bounds.width / bounds.height;
  const direction = normalizeVector(front.map(
    (value, axis) => value + right[axis] * u * alpha * aspect + up[axis] * v * alpha
  ));
  return { origin_m: [...camera.eye], direction };
}

function projectWorldPoint(position) {
  const camera = state.renderedFreeCamera;
  if (!camera) return null;
  const bounds = $("#viewport").getBoundingClientRect();
  const front = normalizeVector(camera.target.map((value, axis) => value - camera.eye[axis]));
  const right = normalizeVector(crossVector(front, [0, 0, 1]));
  const up = normalizeVector(crossVector(right, front));
  const relative = position.map((value, axis) => value - camera.eye[axis]);
  const dot = (left, rightVector) => left.reduce(
    (total, value, axis) => total + value * rightVector[axis], 0
  );
  const depth = dot(relative, front);
  if (depth <= 0) return null;
  const alpha = Math.tan(camera.fov * Math.PI / 360);
  const u = dot(relative, right) / (depth * alpha * (bounds.width / bounds.height));
  const v = dot(relative, up) / (depth * alpha);
  if (Math.abs(u) > 1 || Math.abs(v) > 1) return null;
  return { x: (u + 1) * bounds.width / 2, y: (1 - v) * bounds.height / 2 };
}

function placePointMarker(selector, point, label) {
  const marker = $(selector);
  marker.hidden = !point;
  if (!point) return;
  marker.style.left = `${point.x}px`;
  marker.style.top = `${point.y}px`;
  marker.querySelector("b").textContent = label;
}

function renderTargetOverlay() {
  if (state.view !== "free_camera") {
    placePointMarker("#selected-point-marker", null, "");
    placePointMarker("#estimated-point-marker", null, "");
    return;
  }
  const bounds = $("#viewport").getBoundingClientRect();
  const selected = targetPicker.activeMarker
    ? { x: targetPicker.activeMarker.u * bounds.width, y: targetPicker.activeMarker.v * bounds.height }
    : null;
  placePointMarker(
    "#selected-point-marker",
    selected,
    targetPicker.activeMarker?.label || "",
  );
  placePointMarker(
    "#estimated-point-marker",
    targetPicker.position ? projectWorldPoint(targetPicker.position) : null,
    targetPicker.samples.length === TARGET_VIEW_COUNT ? "XYZ" : "Preview",
  );
}

function captureTargetThumbnail(pixelX, pixelY, label) {
  const source = $("#live-frame");
  const bounds = $("#viewport").getBoundingClientRect();
  const canvas = document.createElement("canvas");
  canvas.width = 320;
  canvas.height = 180;
  const context = canvas.getContext("2d");
  context.drawImage(source, 0, 0, canvas.width, canvas.height);
  const x = pixelX / bounds.width * canvas.width;
  const y = pixelY / bounds.height * canvas.height;
  context.strokeStyle = "#ffffff";
  context.fillStyle = "#c8ff41";
  context.lineWidth = 4;
  context.beginPath();
  context.arc(x, y, 13, 0, Math.PI * 2);
  context.moveTo(x - 22, y); context.lineTo(x + 22, y);
  context.moveTo(x, y - 22); context.lineTo(x, y + 22);
  context.stroke();
  context.beginPath();
  context.arc(x, y, 5, 0, Math.PI * 2);
  context.fill();
  context.font = "bold 22px sans-serif";
  context.fillText(String(label), Math.min(canvas.width - 24, x + 16), Math.max(24, y - 14));
  return canvas.toDataURL("image/jpeg", 0.82);
}

function maximumRayAngleDegrees() {
  let maximum = 0;
  for (let first = 0; first < targetPicker.samples.length; first += 1) {
    for (let second = first + 1; second < targetPicker.samples.length; second += 1) {
      const a = targetPicker.samples[first].ray.direction;
      const b = targetPicker.samples[second].ray.direction;
      const dot = Math.max(-1, Math.min(1, a.reduce(
        (total, value, axis) => total + value * b[axis], 0
      )));
      maximum = Math.max(maximum, Math.acos(dot) * 180 / Math.PI);
    }
  }
  return maximum;
}

function renderTargetPicker() {
  const button = $("#target-mode");
  button.classList.toggle("active", targetPicker.enabled);
  const nextView = Math.min(TARGET_VIEW_COUNT, targetPicker.samples.length + 1);
  button.textContent = targetPicker.busy
    ? "Computing geometry…"
    : targetPicker.samples.length >= TARGET_VIEW_COUNT
      ? "Two views captured"
      : targetPicker.enabled
        ? `Click point for view ${nextView} · Cancel`
        : `Select point for view ${nextView}`;
  button.disabled = targetPicker.busy || targetPicker.samples.length >= TARGET_VIEW_COUNT;
  $("#viewport").classList.toggle("target-picking", targetPicker.enabled);
  if (state.view === "free_camera") {
    $("#viewport-hint").textContent = targetPicker.enabled
      ? `Click point for view ${nextView} · drag still rotates · right/middle pans`
      : "Left drag rotate · right/middle drag pan · wheel zoom";
  }
  $$("#target-samples article").forEach((card, index) => {
    const sample = targetPicker.samples[index];
    card.classList.toggle("filled", Boolean(sample));
    const image = card.querySelector("img");
    const status = card.querySelector("small");
    if (sample) {
      image.src = sample.thumbnail;
      status.textContent = `${Math.round(sample.pixelX)} × ${Math.round(sample.pixelY)} px`;
    } else {
      image.removeAttribute("src");
      status.textContent = "Waiting";
    }
  });
  const result = $("#target-result");
  const quality = $("#target-quality");
  const detail = $("#target-selection-detail");
  const move = $("#move-to-target");
  const moveStatus = $("#target-move-status");
  if (targetPicker.position) {
    const prefix = targetPicker.samples.length === TARGET_VIEW_COUNT ? "Final" : "Preview";
    result.textContent = `${prefix} · ` + targetPicker.position
      .map((value, axis) => `${"XYZ"[axis]} ${(value * 1000).toFixed(1)}`)
      .join(" · ") + " mm";
    const reached = targetPicker.lastMoveErrorM == null
      ? ""
      : `Reached error ${(targetPicker.lastMoveErrorM * 1000).toFixed(2)} mm · `;
    const stage = targetPicker.samples.length === TARGET_VIEW_COUNT ? "Final" : "Preliminary";
    quality.textContent = `${reached}${stage} · parallax ${maximumRayAngleDegrees().toFixed(1)}° · ray mismatch ${(targetPicker.rmsResidualM * 1000).toFixed(2)} mm`;
    quality.className = `target-quality ${targetPicker.rmsResidualM > 0.01 ? "bad" : ""}`;
  } else if (targetPicker.geometryError) {
    result.textContent = `${targetPicker.samples.length}/${TARGET_VIEW_COUNT} views captured`;
    quality.textContent = targetPicker.geometryError;
    quality.className = "target-quality bad";
  } else if (targetPicker.samples.length === 1) {
    result.textContent = "View 1 captured · depth unresolved";
    quality.textContent = "One camera click defines a ray; rotate for a second view";
    quality.className = "target-quality";
  } else {
    result.textContent = targetPicker.busy
      ? "Computing geometry…"
      : "No target selected";
    quality.textContent = "No camera rays captured";
    quality.className = "target-quality";
  }
  const latest = targetPicker.samples.at(-1);
  detail.textContent = latest
    ? `View ${targetPicker.samples.length} · pixel [${latest.pixelX.toFixed(1)}, ${latest.pixelY.toFixed(1)}] · camera [${latest.ray.origin_m.map((value) => value.toFixed(3)).join(", ")}] m`
    : "No camera selections captured";
  move.disabled = targetPicker.samples.length !== TARGET_VIEW_COUNT || !targetPicker.position || targetPicker.busy;
  move.textContent = `Move ${state.arm} arm`;
  moveStatus.textContent = targetPicker.moveStatus;
  moveStatus.className = `target-move-status ${targetPicker.moveFailed ? "bad" : targetPicker.lastMoveErrorM == null ? "" : "ok"}`;
  renderTargetOverlay();
}

function clearTargetPicker() {
  targetPicker.enabled = false;
  targetPicker.samples = [];
  targetPicker.position = null;
  targetPicker.rmsResidualM = null;
  targetPicker.conditionNumber = null;
  targetPicker.lastMoveErrorM = null;
  targetPicker.geometryError = null;
  targetPicker.activeMarker = null;
  targetPicker.moveStatus = "No move requested";
  targetPicker.moveFailed = false;
  targetPicker.busy = false;
  renderTargetPicker();
}

async function addTargetSample(clientX, clientY) {
  if (!targetPicker.enabled || targetPicker.busy || targetPicker.samples.length >= TARGET_VIEW_COUNT) return;
  const bounds = $("#viewport").getBoundingClientRect();
  const pixelX = clientX - bounds.left;
  const pixelY = clientY - bounds.top;
  const label = targetPicker.samples.length + 1;
  try {
    const ray = cameraRayAt(clientX, clientY);
    targetPicker.samples.push({
      ray,
      pixelX,
      pixelY,
      thumbnail: captureTargetThumbnail(pixelX, pixelY, label),
    });
    targetPicker.enabled = false;
    targetPicker.activeMarker = {
      u: pixelX / bounds.width,
      v: pixelY / bounds.height,
      label: String(label),
    };
    targetPicker.position = null;
    targetPicker.rmsResidualM = null;
    targetPicker.conditionNumber = null;
    targetPicker.lastMoveErrorM = null;
    targetPicker.geometryError = null;
    targetPicker.moveStatus = "No move requested";
    targetPicker.moveFailed = false;
    renderTargetPicker();
    if (targetPicker.samples.length < 2) return;
    targetPicker.busy = true;
    renderTargetPicker();
    const result = await api("/api/triangulate", {
      method: "POST",
      body: JSON.stringify({ rays: targetPicker.samples.map((sample) => sample.ray) }),
    });
    targetPicker.position = result.position_m;
    targetPicker.rmsResidualM = result.rms_residual_m;
    targetPicker.conditionNumber = result.condition_number;
    setError();
  } catch (error) {
    targetPicker.position = null;
    targetPicker.geometryError = error.message;
  } finally {
    targetPicker.busy = false;
    renderTargetPicker();
  }
}

function effectiveKeys() {
  const keys = new Set(state.pressed);
  for (const [first, second] of conflictPairs) {
    if (keys.has(first) && keys.has(second)) {
      keys.delete(first);
      keys.delete(second);
    }
  }
  return [...keys].sort();
}

function effectiveMotionKeys() {
  return effectiveKeys().filter((key) => controllerKeys.has(key));
}

function hasMotionDirection(keys) {
  return keys.some((key) => motionKeys.has(key));
}

function hasArmDirection(keys) {
  return keys.some((key) => armDirectionKeys.has(key));
}

function setError(message = "") {
  const banner = $("#error-banner");
  banner.hidden = !message;
  banner.textContent = message;
  if (message) {
    $("#connection-badge").className = "badge error";
    $("#connection-badge").innerHTML = "<i></i> Controller error";
  }
}

function finiteNumber(value, fallback) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function gameKey(event) {
  const raw = event.key.toUpperCase();
  return browserKeyMap[raw] || raw;
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    cache: "no-store",
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  const payload = await response.json();
  if (!response.ok || payload.ok === false) throw new Error(payload.error || `HTTP ${response.status}`);
  return payload;
}

function positionText(values) {
  if (!Array.isArray(values) || values.length !== 3) return "—";
  return values.map((value) => `${(Number(value) * 1000).toFixed(1)}`).join(" · ") + " mm";
}

function updateTelemetry(next) {
  if (!next) return;
  state.latest = next;
  $("#tick-badge").textContent = `Tick ${next.tick ?? "—"}`;
  $("#sim-time").textContent = `Simulation ${finiteNumber(next.sim_time_s, 0).toFixed(3)} s`;
  $("#left-position").textContent = positionText(next.tcp?.left?.position_m);
  $("#right-position").textContent = positionText(next.tcp?.right?.position_m);
  for (const arm of ["left", "right"]) {
    const gap = next.gripper_half_gap_m?.[arm];
    const effort = next.gripper_inward_effort_n?.[arm];
    $(`#${arm}-gripper`).textContent = `Half-gap ${gap == null ? "—" : (gap * 1000).toFixed(2) + " mm"} · Force ${effort == null ? "—" : Number(effort).toFixed(2) + " N"}`;
  }
  $("#connector-state").textContent = Array.isArray(next.connector_active)
    ? next.connector_active.map((value) => value ? "1" : "0").join("")
    : "—";
  const integrity = $("#integrity-badge");
  const valid = next.finite === true && next.environment_hashes_unchanged === true;
  integrity.textContent = valid ? "Finite · environment unchanged" : "Integrity check failed";
  integrity.className = `integrity ${valid ? "ok" : "bad"}`;
  $("#connection-badge").className = "badge online";
  $("#connection-badge").innerHTML = "<i></i> Controller connected";
}

function controllerTransportBusy() {
  return state.commandBusy || state.frameBusy || state.statusBusy;
}

function currentControlIntent() {
  const heldKeys = effectiveMotionKeys();
  const released = !hasArmDirection(heldKeys) && state.releaseTail;
  if (!hasMotionDirection(heldKeys) && !released) return null;
  return {
    arm: state.arm,
    keys: released
      ? [...new Set([
        ...state.lastMotionKeys,
        ...heldKeys.filter((key) => key === "Q" || key === "E"),
      ])]
      : heldKeys,
    released,
    generation: state.controlGeneration,
  };
}

function enqueueCurrentControlIntent() {
  const intent = currentControlIntent();
  if (intent) inputIntents.push(intent);
}

function hasControlWork() {
  return inputIntents.length > 0 || currentControlIntent() !== null;
}

function requestFrameRefresh({ beforeControl = false } = {}) {
  frameRefresh.request();
  if (beforeControl) state.frameBeforeControl = true;
  scheduleWork();
}

function requestStatusRefresh() {
  state.statusRefreshPending = true;
  scheduleWork();
}

function scheduleWork() {
  if (state.workScheduled) return;
  state.workScheduled = true;
  queueMicrotask(() => {
    state.workScheduled = false;
    runNextWork();
  });
}

function runNextWork() {
  if (controllerTransportBusy()) return;
  if (state.latest == null && state.statusRefreshPending) {
    refreshStatus();
    return;
  }
  if (state.frameBeforeControl && frameRefresh.pending) {
    refreshFrame();
    return;
  }
  if (hasControlWork()) {
    pumpCommands();
    return;
  }
  if (state.statusRefreshPending) {
    refreshStatus();
    return;
  }
  if (frameRefresh.pending) refreshFrame();
}

async function refreshStatus() {
  if (controllerTransportBusy()) {
    state.statusRefreshPending = true;
    return;
  }
  state.statusRefreshPending = false;
  state.statusBusy = true;
  try {
    const previousTick = state.latest?.tick;
    const payload = await api("/api/status");
    updateTelemetry(payload.state);
    setError();
    if (previousTick != null && payload.state?.tick !== previousTick) {
      frameRefresh.request();
    }
  } catch (error) {
    setError(error.message);
  } finally {
    state.statusBusy = false;
    scheduleWork();
  }
}

async function refreshCheckpoints() {
  const select = $("#checkpoint-list");
  try {
    const payload = await api("/api/checkpoints");
    const checkpoints = Array.isArray(payload.checkpoints) ? payload.checkpoints : [];
    const previous = select.value;
    select.replaceChildren();
    if (!checkpoints.length) {
      select.append(new Option("No checkpoints", ""));
    } else {
      for (const checkpoint of checkpoints) {
        const sizeMiB = Number(checkpoint.size_bytes || 0) / (1024 * 1024);
        select.append(new Option(`${checkpoint.name} · ${sizeMiB.toFixed(1)} MiB`, checkpoint.name));
      }
      if (checkpoints.some((checkpoint) => checkpoint.name === previous)) select.value = previous;
    }
    $("#restore-checkpoint").disabled = !select.value;
  } catch (error) {
    const status = $("#checkpoint-status");
    status.textContent = error.message;
    status.className = "checkpoint-status bad";
  }
}

async function refreshFrame(force = false) {
  if (force) frameRefresh.request();
  const age = performance.now() - state.lastFrameAt;
  if (!force && !frameRefresh.pending && age < 250) return;
  if (controllerTransportBusy() || !frameRefresh.begin()) return;
  state.frameBusy = true;
  if (state.view === "free_camera") state.freeFrameDirty = false;
  const requestedFreeCamera = state.view === "free_camera" ? {
    eye: [...freeCamera.eye],
    target: [...freeCamera.target],
    fov: freeCamera.fov,
  } : null;
  const started = performance.now();
  const loading = $("#frame-loading");
  if (!$("#live-frame").src) loading.classList.remove("hidden");
  try {
    const camera = requestedFreeCamera ? freeCameraQuery(requestedFreeCamera) : "";
    const response = await fetch(`/api/frame?view=${encodeURIComponent(state.view)}${camera}&nonce=${Date.now()}`, { cache: "no-store" });
    if (!response.ok) {
      const payload = await response.json();
      throw new Error(payload.error || `Frame HTTP ${response.status}`);
    }
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const image = $("#live-frame");
    const previous = image.dataset.objectUrl;
    await new Promise((resolve, reject) => {
      image.onload = resolve;
      image.onerror = () => reject(new Error("Browser could not decode the camera frame"));
      image.src = url;
    });
    image.onload = null;
    image.onerror = null;
    image.dataset.objectUrl = url;
    if (previous) URL.revokeObjectURL(previous);
    state.lastFrameAt = performance.now();
    state.lastFrameWallMs = state.lastFrameAt - started;
    state.lastFrameTick = Number(response.headers.get("X-Simulation-Tick"));
    state.frameSequence += 1;
    if (requestedFreeCamera) {
      state.renderedFreeCamera = requestedFreeCamera;
      renderTargetOverlay();
    }
    $("#frame-badge").textContent = `Frame ${state.lastFrameWallMs.toFixed(0)} ms`;
    loading.classList.add("hidden");
    setError();
  } catch (error) {
    setError(error.message);
    loading.textContent = "View unavailable";
    loading.classList.remove("hidden");
  } finally {
    state.frameBusy = false;
    if (state.view === "free_camera" && state.freeFrameDirty) frameRefresh.request();
    frameRefresh.finish();
    state.frameBeforeControl = false;
    state.actionsSinceFrame = 0;
    scheduleWork();
  }
}

function keyboardPayload(intent) {
  return {
    arm: intent.arm,
    keys: intent.keys,
    ticks: Math.trunc(finiteNumber($("#ticks-input").value, 1)),
    rate_hz: finiteNumber($("#rate-input").value, 120),
    linear_speed_m_s: finiteNumber($("#linear-speed").value, 3000) / 1000,
    angular_speed_rad_s: finiteNumber($("#angular-speed").value, 60) * Math.PI / 180,
    gripper_half_speed_m_s: 0.025,
    released: intent.released,
  };
}

async function pumpCommands() {
  if (controllerTransportBusy()) return;
  const generation = state.controlGeneration;
  const intent = inputIntents.shift(generation) || currentControlIntent();
  if (!intent) return;
  state.commandBusy = true;
  let actionCompleted = false;
  try {
    const payload = await api("/api/keyboard", {
      method: "POST",
      body: JSON.stringify(keyboardPayload(intent)),
    });
    actionCompleted = true;
    if (generation !== state.controlGeneration) return;
    updateTelemetry(payload.state);
    state.motionLevel = finiteNumber(payload.action?.motion_level, 0);
    state.releaseTail = (
      !hasMotionDirection(effectiveMotionKeys()) &&
      payload.action?.motion_active === true
    );
    renderPressed();
    $("#latency-badge").textContent = `Action ${(payload.wall_time_s * 1000).toFixed(0)} ms`;
    setError();
  } catch (error) {
    setError(error.message);
    releaseAll();
  } finally {
    state.commandBusy = false;
    if (actionCompleted) {
      state.actionsSinceFrame += 1;
      frameRefresh.request();
      if (hasControlWork() && state.actionsSinceFrame >= ACTIONS_PER_RENDER) {
        state.frameBeforeControl = true;
      }
    }
    scheduleWork();
  }
}

function renderPressed() {
  const keys = effectiveKeys();
  $("#active-chord").textContent = keys.length
    ? `Held: ${keys.join(" + ")}`
    : (state.releaseTail ? `Releasing · ${(state.motionLevel * 100).toFixed(0)}%` : "No keys held");
  $$("[data-key]").forEach((button) => button.classList.toggle("pressed", state.pressed.has(button.dataset.key)));
}

function pressKey(key) {
  if (!motionKeys.has(key) && key !== "ALT" && key !== "SHIFT") return;
  if (state.pressed.has(key)) return;
  state.pressed.add(key);
  const keys = effectiveMotionKeys();
  if (hasArmDirection(keys)) {
    state.lastMotionKeys = keys.filter((candidate) => armControlKeys.has(candidate));
    state.releaseTail = false;
  }
  enqueueCurrentControlIntent();
  renderPressed();
  scheduleWork();
}

function releaseKey(key) {
  if (!state.pressed.has(key)) return;
  const before = effectiveMotionKeys();
  state.pressed.delete(key);
  const after = effectiveMotionKeys();
  if (hasArmDirection(before) && !hasArmDirection(after)) {
    state.lastMotionKeys = before.filter((candidate) => armControlKeys.has(candidate));
    state.releaseTail = true;
  } else if (hasArmDirection(after)) {
    state.lastMotionKeys = after.filter((candidate) => armControlKeys.has(candidate));
  }
  enqueueCurrentControlIntent();
  renderPressed();
  scheduleWork();
}

function releaseAll() {
  state.controlGeneration += 1;
  inputIntents.clear();
  state.pressed.clear();
  state.releaseTail = false;
  state.lastMotionKeys = [];
  state.motionLevel = 0;
  state.frameBeforeControl = false;
  renderPressed();
}

async function hardStop(arm = state.arm) {
  releaseAll();
  try {
    await api("/api/stop", {
      method: "POST",
      body: JSON.stringify({ arm }),
    });
  } catch (error) {
    setError(error.message);
  }
}

function selectArm(arm) {
  const previousArm = state.arm;
  hardStop(previousArm);
  state.arm = arm;
  $$("[data-arm]").forEach((button) => button.classList.toggle("active", button.dataset.arm === arm));
  renderTargetPicker();
  $("#viewport").focus();
}

$$('[data-arm]').forEach((button) => button.addEventListener("click", () => selectArm(button.dataset.arm)));
function selectView(view) {
  const button = $(`[data-view="${view}"]`);
  if (!button || !Object.hasOwn(viewNames, view)) throw new Error(`Unknown view ${view}`);
  state.view = view;
  if (state.view !== "free_camera") targetPicker.enabled = false;
  $$("[data-view]").forEach((candidate) => candidate.classList.toggle("active", candidate === button));
  $("#view-title").textContent = viewNames[state.view][0];
  $("#camera-kind").textContent = viewNames[state.view][1];
  $("#viewport").classList.toggle("free-camera", state.view === "free_camera");
  $("#reset-camera").hidden = state.view !== "free_camera";
  $("#viewport-hint").textContent = state.view === "free_camera"
    ? (targetPicker.enabled
      ? "Click target · left drag rotate · right/middle drag pan · wheel zoom"
      : "Left drag rotate · right/middle drag pan · wheel zoom")
    : "Controls are live · hold WASD or arrow keys";
  renderTargetPicker();
  state.lastFrameAt = 0;
  requestFrameRefresh({ beforeControl: true });
}

$$('[data-view]').forEach((button) => button.addEventListener("click", () => selectView(button.dataset.view)));

const viewport = $("#viewport");
viewport.classList.add("free-camera");
viewport.addEventListener("contextmenu", (event) => event.preventDefault());
viewport.addEventListener("pointerdown", (event) => {
  if (state.view !== "free_camera" || event.target === $("#reset-camera")) return;
  if (![0, 1, 2].includes(event.button)) return;
  event.preventDefault();
  viewport.focus();
  viewport.setPointerCapture(event.pointerId);
  freeCamera.drag = {
    pointerId: event.pointerId,
    button: event.button,
    mode: event.button === 0 && !event.shiftKey ? "orbit" : "pan",
    startX: event.clientX,
    startY: event.clientY,
    movedSquared: 0,
    x: event.clientX,
    y: event.clientY,
  };
  viewport.classList.add("grabbing");
});
viewport.addEventListener("pointermove", (event) => {
  const drag = freeCamera.drag;
  if (state.view !== "free_camera" || !drag || drag.pointerId !== event.pointerId) return;
  const dx = event.clientX - drag.x;
  const dy = event.clientY - drag.y;
  drag.movedSquared = Math.max(
    drag.movedSquared,
    (event.clientX - drag.startX) ** 2 + (event.clientY - drag.startY) ** 2,
  );
  drag.x = event.clientX;
  drag.y = event.clientY;
  if (drag.mode === "orbit") {
    freeCamera.yaw -= dx * 0.008;
    freeCamera.pitch = Math.max(-1.553, Math.min(1.553, freeCamera.pitch + dy * 0.008));
    updateFreeCameraEye();
  } else {
    const forward = freeCamera.target.map((value, axis) => value - freeCamera.eye[axis]);
    const forwardNorm = Math.hypot(...forward);
    const f = forward.map((value) => value / forwardNorm);
    let right = [f[1], -f[0], 0];
    const rightNorm = Math.max(1e-9, Math.hypot(...right));
    right = right.map((value) => value / rightNorm);
    const up = [
      right[1] * f[2] - right[2] * f[1],
      right[2] * f[0] - right[0] * f[2],
      right[0] * f[1] - right[1] * f[0],
    ];
    const scale = freeCamera.distance * 0.0016;
    const translation = right.map(
      (value, axis) => -dx * scale * value + dy * scale * up[axis]
    );
    freeCamera.eye = freeCamera.eye.map((value, axis) => value + translation[axis]);
    freeCamera.target = freeCamera.target.map((value, axis) => value + translation[axis]);
  }
  scheduleFreeCameraFrame();
});
for (const eventName of ["pointerup", "pointercancel", "lostpointercapture"]) {
  viewport.addEventListener(eventName, (event) => {
    if (!freeCamera.drag || freeCamera.drag.pointerId !== event.pointerId) return;
    const drag = freeCamera.drag;
    const selectTarget = eventName === "pointerup"
      && targetPicker.enabled
      && drag.button === 0
      && drag.mode === "orbit"
      && drag.movedSquared < 25;
    freeCamera.drag = null;
    viewport.classList.remove("grabbing");
    if (selectTarget) addTargetSample(event.clientX, event.clientY);
  });
}
viewport.addEventListener("wheel", (event) => {
  if (state.view !== "free_camera") return;
  event.preventDefault();
  freeCamera.distance = Math.max(
    0.12,
    Math.min(5.0, freeCamera.distance * Math.exp(event.deltaY * 0.0015)),
  );
  updateFreeCameraEye();
  scheduleFreeCameraFrame();
}, { passive: false });
$("#reset-camera").addEventListener("click", (event) => {
  event.stopPropagation();
  resetFreeCamera();
  scheduleFreeCameraFrame();
});

$("#target-mode").addEventListener("click", () => {
  if (state.view !== "free_camera") {
    setError("Select Free camera before target picking");
    return;
  }
  targetPicker.enabled = !targetPicker.enabled;
  $("#viewport-hint").textContent = targetPicker.enabled
    ? "Click target · left drag rotate · right/middle drag pan · wheel zoom"
    : "Left drag rotate · right/middle drag pan · wheel zoom";
  renderTargetPicker();
  viewport.focus();
});

$("#clear-target").addEventListener("click", clearTargetPicker);

$("#move-to-target").addEventListener("click", async () => {
  if (!targetPicker.position || targetPicker.busy) return;
  const durationS = finiteNumber($("#target-duration").value, 0.5);
  if (!(durationS > 0 && durationS <= 10)) {
    setError("Move time must be greater than 0 and at most 10 sim s");
    return;
  }
  targetPicker.busy = true;
  targetPicker.moveStatus = `Preparing ${state.arm} arm…`;
  targetPicker.moveFailed = false;
  state.commandBusy = true;
  renderTargetPicker();
  await hardStop();
  try {
    targetPicker.moveStatus = `Moving ${state.arm} arm to selected XYZ…`;
    renderTargetPicker();
    const payload = await api("/api/go-to-target", {
      method: "POST",
      body: JSON.stringify({
        arm: state.arm,
        position_m: targetPicker.position,
        duration_s: durationS,
      }),
    });
    updateTelemetry(payload.state);
    targetPicker.lastMoveErrorM = payload.actual_error_m;
    const movedDistanceM = Math.hypot(...payload.actual_position_m.map(
      (value, axis) => value - payload.start_position_m[axis]
    ));
    const physicsSteps = payload.controller_result?.executed_physics_steps;
    targetPicker.moveStatus = `Moved ${(movedDistanceM * 1000).toFixed(1)} mm · final error ${(payload.actual_error_m * 1000).toFixed(1)} mm${physicsSteps == null ? "" : ` · ${physicsSteps} physics steps`}`;
    $("#latency-badge").textContent = `Action ${(payload.wall_time_s * 1000).toFixed(0)} ms`;
    setError();
    requestFrameRefresh({ beforeControl: true });
  } catch (error) {
    targetPicker.moveStatus = `Move failed · ${error.message}`;
    targetPicker.moveFailed = true;
    setError(error.message);
  } finally {
    state.commandBusy = false;
    targetPicker.busy = false;
    renderTargetPicker();
    scheduleWork();
  }
});

$$('[data-key]').forEach((button) => {
  button.addEventListener("pointerdown", (event) => {
    event.preventDefault();
    pressKey(button.dataset.key);
    try { button.setPointerCapture(event.pointerId); } catch (_) { /* release still arrives on window */ }
  });
  for (const eventName of ["pointerup", "pointercancel", "lostpointercapture"]) {
    button.addEventListener(eventName, () => releaseKey(button.dataset.key));
  }
});

window.addEventListener("keydown", (event) => {
  if (isEditableElement(event.target)) return;
  if (event.metaKey || (event.ctrlKey && event.key.toUpperCase() !== "CONTROL")) return;
  if (event.key === "Escape") { event.preventDefault(); hardStop(); return; }
  if (event.key === "Tab") { event.preventDefault(); selectArm(state.arm === "left" ? "right" : "left"); return; }
  if (event.key === "1") { event.preventDefault(); selectArm("left"); return; }
  if (event.key === "2") { event.preventDefault(); selectArm("right"); return; }
  const key = gameKey(event);
  if (key === "ALT" || key === "SHIFT") {
    event.preventDefault();
    pressKey(key);
    return;
  }
  if (motionKeys.has(key)) {
    event.preventDefault();
    if (document.activeElement && document.activeElement !== document.body) document.activeElement.blur();
    if (event.altKey) pressKey("ALT");
    if (event.shiftKey) pressKey("SHIFT");
    pressKey(key);
  }
});
window.addEventListener("keyup", (event) => {
  if (isEditableElement(event.target)) return;
  releaseKey(gameKey(event));
  if (!event.altKey) releaseKey("ALT");
  if (!event.shiftKey) releaseKey("SHIFT");
});
document.addEventListener("focusin", (event) => {
  if (isEditableElement(event.target)) releaseAll();
});
window.addEventListener("blur", () => hardStop());
document.addEventListener("visibilitychange", () => { if (document.hidden) hardStop(); });

for (const [selector, key] of [["#close-gripper", "Q"], ["#open-gripper", "E"]]) {
  const button = $(selector);
  button.addEventListener("pointerdown", (event) => {
    event.preventDefault();
    pressKey(key);
    try { button.setPointerCapture(event.pointerId); } catch (_) { /* release still arrives on window */ }
  });
  for (const eventName of ["pointerup", "pointercancel", "lostpointercapture"]) {
    button.addEventListener(eventName, () => releaseKey(key));
  }
}
$("#reset-button").addEventListener("click", async () => {
  releaseAll();
  if (!window.confirm("Reset destroys the current simulator state and reconstructs the scene. Continue?")) return;
  state.commandBusy = true;
  try {
    const payload = await api("/api/reset", { method: "POST", body: "{}" });
    updateTelemetry(payload.state);
    state.lastFrameAt = 0;
    requestFrameRefresh({ beforeControl: true });
    setError();
  } catch (error) {
    setError(error.message);
  } finally {
    state.commandBusy = false;
    scheduleWork();
  }
});

$("#advance-simulation").addEventListener("click", async () => {
  const durationS = finiteNumber($("#advance-duration").value, NaN);
  const status = $("#advance-status");
  const button = $("#advance-simulation");
  if (!Number.isFinite(durationS) || !(durationS > 0 && durationS <= 10)) {
    status.textContent = "Duration must be greater than 0 and at most 10 sim s";
    status.className = "advance-status bad";
    return;
  }
  releaseAll();
  state.commandBusy = true;
  button.disabled = true;
  status.textContent = `Advancing ${durationS.toFixed(2)} sim s with unchanged targetsâ€¦`;
  status.className = "advance-status";
  const startTick = state.latest?.tick;
  try {
    const payload = await api("/api/advance", {
      method: "POST",
      body: JSON.stringify({ duration_s: durationS }),
    });
    updateTelemetry(payload.state);
    const tickDelta = Number.isFinite(startTick)
      ? Number(payload.state.tick) - Number(startTick)
      : null;
    status.textContent = `Advanced ${durationS.toFixed(2)} sim s${tickDelta == null ? "" : ` Â· ${tickDelta} physics ticks`} Â· targets unchanged`;
    status.className = "advance-status ok";
    $("#latency-badge").textContent = `Advance ${(payload.wall_time_s * 1000).toFixed(0)} ms`;
    requestFrameRefresh({ beforeControl: true });
    setError();
  } catch (error) {
    status.textContent = error.message;
    status.className = "advance-status bad";
    setError(error.message);
  } finally {
    state.commandBusy = false;
    button.disabled = false;
    scheduleWork();
  }
});

$("#checkpoint-list").addEventListener("change", () => {
  $("#restore-checkpoint").disabled = !$("#checkpoint-list").value;
});

$("#save-checkpoint").addEventListener("click", async () => {
  releaseAll();
  const status = $("#checkpoint-status");
  const label = $("#checkpoint-label").value.trim();
  if (!label) { status.textContent = "Enter a checkpoint name"; status.className = "checkpoint-status bad"; return; }
  state.commandBusy = true;
  status.textContent = "Saving complete physics state…";
  status.className = "checkpoint-status";
  try {
    const payload = await api("/api/checkpoints/save", {
      method: "POST",
      body: JSON.stringify({ label }),
    });
    updateTelemetry(payload.state);
    status.textContent = `Saved ${label} · ${(Number(payload.checkpoint.size_bytes) / (1024 * 1024)).toFixed(1)} MiB · tick ${payload.checkpoint.source_tick}`;
    status.className = "checkpoint-status ok";
    await refreshCheckpoints();
  } catch (error) {
    status.textContent = error.message;
    status.className = "checkpoint-status bad";
    setError(error.message);
  } finally {
    state.commandBusy = false;
    scheduleWork();
  }
});

$("#restore-checkpoint").addEventListener("click", async () => {
  releaseAll();
  const checkpoint = $("#checkpoint-list").value;
  if (!checkpoint) return;
  if (!window.confirm(`Discard the current state and restore “${checkpoint}”?`)) return;
  const status = $("#checkpoint-status");
  state.commandBusy = true;
  status.textContent = `Restoring ${checkpoint}…`;
  status.className = "checkpoint-status";
  try {
    const payload = await api("/api/checkpoints/restore", {
      method: "POST",
      body: JSON.stringify({ checkpoint }),
    });
    updateTelemetry(payload.state);
    state.lastFrameAt = 0;
    clearTargetPicker();
    requestFrameRefresh({ beforeControl: true });
    status.textContent = `Restored ${checkpoint} · tick ${payload.checkpoint.restored_tick} · byte-exact arrays`;
    status.className = "checkpoint-status ok";
    setError();
  } catch (error) {
    status.textContent = error.message;
    status.className = "checkpoint-status bad";
    setError(error.message);
  } finally {
    state.commandBusy = false;
    scheduleWork();
  }
});

function finiteVector(values, length) {
  return Array.isArray(values)
    && values.length === length
    && values.every((value) => Number.isFinite(Number(value)));
}

function browserRelayState() {
  return {
    url: window.location.href,
    title: document.title,
    visible: document.visibilityState === "visible",
    active_arm: state.arm,
    view: state.view,
    free_camera: {
      eye_m: [...freeCamera.eye],
      target_m: [...freeCamera.target],
      fov_degrees: freeCamera.fov,
    },
    rendered_free_camera: state.renderedFreeCamera ? {
      eye_m: [...state.renderedFreeCamera.eye],
      target_m: [...state.renderedFreeCamera.target],
      fov_degrees: state.renderedFreeCamera.fov,
    } : null,
    simulator_tick: state.latest?.tick ?? null,
    last_frame_tick: state.lastFrameTick,
    frame_sequence: state.frameSequence,
    command_busy: state.commandBusy,
    frame_busy: state.frameBusy,
    pressed_keys: effectiveKeys(),
    target_picker: {
      enabled: targetPicker.enabled,
      sample_count: targetPicker.samples.length,
      position_m: targetPicker.position ? [...targetPicker.position] : null,
    },
    viewport: {
      width_px: $("#viewport").clientWidth,
      height_px: $("#viewport").clientHeight,
      window_width_px: window.innerWidth,
      window_height_px: window.innerHeight,
      scroll_x_px: window.scrollX,
      scroll_y_px: window.scrollY,
    },
    status_text: {
      checkpoint: $("#checkpoint-status").textContent,
      advance: $("#advance-status").textContent,
      error: $("#error-banner").hidden ? "" : $("#error-banner").textContent,
    },
    last_remote_command_id: state.lastBrowserCommandId,
    last_remote_command_error: state.lastBrowserCommandError,
  };
}

function applyBrowserCommand(command) {
  if (!command || !Number.isInteger(command.id)) throw new Error("Remote command has no integer id");
  if (command.type === "set_arm") {
    if (!["left", "right"].includes(command.arm)) throw new Error("set_arm requires left or right");
    selectArm(command.arm);
  } else if (command.type === "set_view") {
    selectView(String(command.view));
  } else if (command.type === "set_free_camera") {
    if (!finiteVector(command.eye_m, 3) || !finiteVector(command.target_m, 3)) {
      throw new Error("set_free_camera requires finite eye_m and target_m vectors");
    }
    const fov = Number(command.fov_degrees);
    if (!Number.isFinite(fov) || !(fov >= 10 && fov <= 100)) {
      throw new Error("set_free_camera fov_degrees must be in [10, 100]");
    }
    freeCamera.eye = command.eye_m.map(Number);
    freeCamera.target = command.target_m.map(Number);
    freeCamera.fov = fov;
    syncFreeCameraSpherical();
    selectView("free_camera");
  } else if (command.type === "refresh_frame") {
    requestFrameRefresh({ beforeControl: true });
  } else {
    throw new Error(`Unsupported remote command ${command.type}`);
  }
}

async function syncBrowserSession() {
  if (state.browserSyncBusy) return;
  state.browserSyncBusy = true;
  let handledCommand = false;
  try {
    const response = await fetch("/api/browser/sync", {
      method: "POST",
      cache: "no-store",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        client_id: browserClientId,
        state: browserRelayState(),
      }),
    });
    const payload = await response.json();
    if (!response.ok || payload.ok === false) throw new Error(payload.error || `HTTP ${response.status}`);
    for (const command of payload.commands || []) {
      handledCommand = true;
      try {
        applyBrowserCommand(command);
        state.lastBrowserCommandId = command.id;
        state.lastBrowserCommandError = null;
      } catch (error) {
        state.lastBrowserCommandId = command.id;
        state.lastBrowserCommandError = error.message;
      }
    }
  } catch (_) {
    // The simulator controls remain usable if the optional local relay is unavailable.
  } finally {
    state.browserSyncBusy = false;
    if (handledCommand) setTimeout(syncBrowserSession, 0);
  }
}

renderTargetPicker();
requestStatusRefresh();
requestFrameRefresh();
refreshCheckpoints();
setInterval(requestStatusRefresh, 1500);
syncBrowserSession();
setInterval(syncBrowserSession, 500);

window.__teleopDiagnostics = () => ({
  commandBusy: state.commandBusy,
  frameBusy: state.frameBusy,
  statusBusy: state.statusBusy,
  queuedInputCount: inputIntents.length,
  queuedInputs: inputIntents.snapshot(),
  frameRefreshPending: frameRefresh.pending,
  frameBeforeControl: state.frameBeforeControl,
  frameSequence: state.frameSequence,
  lastFrameTick: state.lastFrameTick,
  lastFrameWallMs: state.lastFrameWallMs,
  pressed: effectiveKeys(),
  releaseTail: state.releaseTail,
  motionLevel: state.motionLevel,
  stateTick: state.latest?.tick ?? null,
  freeCamera: {
    eye: [...freeCamera.eye],
    target: [...freeCamera.target],
    distance: freeCamera.distance,
    yaw: freeCamera.yaw,
    pitch: freeCamera.pitch,
    fov: freeCamera.fov,
  },
  targetPicker: {
    enabled: targetPicker.enabled,
    sampleCount: targetPicker.samples.length,
    position: targetPicker.position ? [...targetPicker.position] : null,
    rmsResidualM: targetPicker.rmsResidualM,
    conditionNumber: targetPicker.conditionNumber,
    lastMoveErrorM: targetPicker.lastMoveErrorM,
    busy: targetPicker.busy,
  },
  browserRelay: {
    clientId: browserClientId,
    syncBusy: state.browserSyncBusy,
    lastCommandId: state.lastBrowserCommandId,
    lastCommandError: state.lastBrowserCommandError,
  },
});

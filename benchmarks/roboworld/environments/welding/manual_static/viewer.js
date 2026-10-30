// Operator 3D view of the welding cell's collision geometry (owner key), posed from the session's reported state.
// It draws poses only: it never simulates, interpolates or steps anything, and it renders on demand (when the view
// moves or a new pose arrives). three.js r180 is served locally by manual.py from whiteboard/.cache/web_vendor.
import * as THREE from "three";
import {OrbitControls} from "/vendor/OrbitControls.js";

function geometryFor(part, meshes) {
  let g;
  switch (part.kind) {
    case "mesh": {
      const m = meshes[part.mesh];
      g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.Float32BufferAttribute(m.vertices, 3));
      g.setIndex(m.indices);
      g = g.toNonIndexed();               // flat facets: the collision meshes are coarse hulls
      g.computeVertexNormals();
      return g;
    }
    case "box": return new THREE.BoxGeometry(...part.size);
    case "sphere": return new THREE.SphereGeometry(part.radius, 24, 16);
    case "cylinder": g = new THREE.CylinderGeometry(part.radius, part.radius, part.length, 32); break;   // URDF: along z
    case "zcylinder": g = new THREE.CylinderGeometry(part.radius, part.radius, 2 * part.half_height, 32); break;
    case "zcapsule": g = new THREE.CapsuleGeometry(part.radius, 2 * part.half_height, 8, 20); break;
    case "zfrustum": {                    // the simulator's polygon circumscribes the cone (welding/scene.py _frustum)
      const k = 1 / Math.cos(Math.PI / part.sides);
      g = new THREE.CylinderGeometry(k * part.radius_tip, k * part.radius_base, 2 * part.half_height, part.sides); break;
    }
    default: throw new Error(`unsupported collider kind ${part.kind}`);
  }
  g.rotateX(Math.PI / 2);                 // three.js cylinders run along y; the colliders' axis is local z
  return g;
}

export class CellViewer {
  constructor(element, scene) {
    this.el = element;
    this.data = scene;
    this.renderer = new THREE.WebGLRenderer({antialias: true});
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    element.replaceChildren(this.renderer.domElement);
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color("#1b242d");
    this.camera = new THREE.PerspectiveCamera(45, 1, 0.005, 40);
    this.camera.up.set(0, 0, 1);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.addEventListener("change", () => this.draw());
    this.scene.add(new THREE.HemisphereLight(0xdfe9f3, 0x3a342c, 2.0));
    const sun = new THREE.DirectionalLight(0xffffff, 2.2);
    sun.position.set(1.5, -2.0, 3.0);
    this.scene.add(sun);
    const grid = new THREE.GridHelper(4, 40, 0x51606e, 0x2c3843);
    grid.rotateX(Math.PI / 2);
    this.scene.add(grid);
    const axes = new THREE.AxesHelper(0.15);
    axes.position.set(-0.6, -0.4, 0.851);
    this.scene.add(axes);

    this.materials = {};
    for (const [name, [rgb, rough, metal]] of Object.entries(scene.materials))
      this.materials[name] = new THREE.MeshStandardMaterial({color: new THREE.Color(...rgb), roughness: rough, metalness: metal,
        ...(name === "screen" ? {transparent: true, opacity: 0.18, depthWrite: false, side: THREE.DoubleSide} : {})});
    const material = name => this.materials[name] || this.materials.table;
    const meshes = scene.meshes;
    const add = (parent, part) => {
      const mesh = new THREE.Mesh(geometryFor(part, meshes), material(part.material));
      mesh.position.fromArray(part.position || [0, 0, 0]);
      if (part.quat_xyzw) mesh.quaternion.fromArray(part.quat_xyzw);
      mesh.name = part.name || "";
      parent.add(mesh);
      return mesh;
    };
    this.screens = [];
    for (const item of scene.static) { const m = add(this.scene, item); if (item.material === "screen") this.screens.push(m); }
    this.bodies = {};
    for (const [name, b] of Object.entries(scene.bodies)) {
      if (!b.size) continue;
      const group = new THREE.Group();
      add(group, {kind: "box", size: b.size, material: b.material, name});
      group.visible = false;              // shown once the session reports its pose
      this.bodies[name] = group;
      this.scene.add(group);
    }
    this.links = {};
    this.robotRoot = new THREE.Group();
    this.scene.add(this.robotRoot);
    for (const link of scene.links) {
      const group = new THREE.Group();
      for (const p of link.parts) add(group, p);
      group.visible = false;
      this.links[link.id] = group;
      this.robotRoot.add(group);
    }
    const lineMat = color => new THREE.LineBasicMaterial({color});
    this.wire = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]), lineMat(0xffb35c));
    this.seam = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]), lineMat(0x6fe3ff));
    this.work = new THREE.Mesh(new THREE.SphereGeometry(0.0025, 16, 12), new THREE.MeshBasicMaterial({color: 0xa0ffd2}));
    this.work.add(new THREE.AxesHelper(0.03));
    for (const o of [this.wire, this.seam, this.work]) { o.visible = false; this.scene.add(o); }
    this.lastPose = null;
    new ResizeObserver(() => this.resize()).observe(element);
    this.resize();
    this.preset("cell");
  }

  update(pose) {
    this.lastPose = pose;
    for (const [id, p] of Object.entries(pose.links || {})) {
      const g = this.links[id];
      if (!g) continue;
      g.position.fromArray(p); g.quaternion.fromArray(p, 3); g.visible = true;
    }
    for (const [name, p] of Object.entries(pose.bodies || {})) {
      const g = this.bodies[name];
      if (!g) continue;
      g.position.fromArray(p); g.quaternion.fromArray(p, 3); g.visible = true;
    }
    if (pose.wire) {
      const w = pose.wire;
      this.wire.geometry.setFromPoints([new THREE.Vector3(...w.contact_tip_m), new THREE.Vector3(...w.wire_tip_m)]);
      this.wire.visible = true;
      this.work.position.fromArray(w.work_point_m); this.work.visible = true;
    }
    if (pose.weld_frame && pose.coupon) {
      const o = new THREE.Vector3(...pose.weld_frame.origin_m), R = pose.weld_frame.axes_columns;
      const x = new THREE.Vector3(R[0][0], R[1][0], R[2][0]);
      this.seam.geometry.setFromPoints([o, o.clone().addScaledVector(x, pose.coupon.weld_length_m)]);
      this.seam.visible = true;
    }
    this.draw();
  }

  showRobots(on) { this.robotRoot.visible = on; this.draw(); }

  showScreens(on) { for (const m of this.screens) m.visible = on; this.draw(); }

  look(eye, target, hfovDeg) {
    const aspect = this.camera.aspect || 1.6;
    this.camera.fov = hfovDeg ? 2 * Math.atan(Math.tan(hfovDeg * Math.PI / 360) / aspect) * 180 / Math.PI : 45;
    this.camera.position.set(...eye);
    this.controls.target.set(...target);
    this.camera.updateProjectionMatrix();
    this.controls.update();
    this.draw();
  }

  preset(name) {
    const cams = this.data.cameras || {};
    if (name === "bench" || name === "side") { const c = cams[name]; return this.look(c.eye, c.target, c.hfov_deg); }
    if (name === "seam") return this.look([0.30, -0.42, 1.10], [0.0, -0.06, 0.87]);
    if (name === "top") return this.look([0.0, -0.36, 2.30], [0.0, -0.06, 0.86]);
    if (name === "torch" && this.lastPose && this.lastPose.wire) {
      const p = this.lastPose.wire.work_point_m;
      return this.look([p[0] + 0.22, p[1] - 0.30, p[2] + 0.16], p);
    }
    return this.look([1.55, -1.75, 1.85], [0.0, 0.05, 0.92]);
  }

  resize() {
    const w = this.el.clientWidth, h = this.el.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.draw();
  }

  draw() { this.renderer.render(this.scene, this.camera); }
}

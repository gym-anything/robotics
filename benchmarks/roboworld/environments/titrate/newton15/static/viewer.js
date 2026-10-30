import * as THREE from 'three';
import {OrbitControls} from '/vendor/OrbitControls.js';

export class Viewer {
  constructor(element) {
    this.element=element; this.bodies=new Map();this.cutaway=false;
    this.scene=new THREE.Scene();this.scene.background=new THREE.Color('#e1e7e2');
    this.renderer=new THREE.WebGLRenderer({antialias:true});this.renderer.setPixelRatio(Math.min(devicePixelRatio,2));
    this.renderer.outputColorSpace=THREE.SRGBColorSpace;this.renderer.shadowMap.enabled=true;
    element.append(this.renderer.domElement);
    this.camera=new THREE.PerspectiveCamera(42,1,.01,30);this.camera.up.set(0,0,1);
    this.controls=new OrbitControls(this.camera,this.renderer.domElement);this.controls.enableDamping=true;
    this.home();this.scene.add(new THREE.HemisphereLight(0xffffff,0x8c9b8e,2.2));
    const key=new THREE.DirectionalLight(0xffffff,3);key.position.set(1,-2,4);this.scene.add(key);
    this.root=new THREE.Group();this.scene.add(this.root);
    new ResizeObserver(()=>this.resize()).observe(element);this.resize();
    this.renderer.setAnimationLoop(()=>{this.controls.update();this.renderer.render(this.scene,this.camera);});
  }
  home(){this.camera.position.set(1.6,-2,1.9);this.controls.target.set(0,.05,1.25);this.controls.update();}
  resize(){const w=this.element.clientWidth,h=this.element.clientHeight;if(!w||!h)return;this.renderer.setSize(w,h);this.camera.aspect=w/h;this.camera.updateProjectionMatrix();}
  load(data){
    this.root.traverse(o=>{if(o.isMesh){o.geometry.dispose();o.material.dispose();}});this.root.clear();this.bodies.clear();
    for(const shape of data.geometries){
      let geometry;const s=shape.scale;
      if(shape.asset_id){const a=data.assets[shape.asset_id];geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.Float32BufferAttribute(a.vertices_local_m.flat(),3));geometry.setIndex(a.triangles.flat());geometry.computeVertexNormals();}
      else if(shape.type==='BOX')geometry=new THREE.BoxGeometry(2*s[0],2*s[1],2*s[2]);
      else if(shape.type==='SPHERE')geometry=new THREE.SphereGeometry(s[0],24,16);
      else if(shape.type==='CYLINDER'){geometry=new THREE.CylinderGeometry(s[0],s[0],2*s[1],32);geometry.rotateX(Math.PI/2);}
      else if(shape.type==='CAPSULE'){geometry=new THREE.CapsuleGeometry(s[0],2*s[1],8,20);geometry.rotateX(Math.PI/2);}
      else throw new Error(`Unsupported visible geometry: ${shape.name} (${shape.type})`);
      const m=shape.preview_material;
      const material=new THREE.MeshStandardMaterial({color:new THREE.Color(...m.linear_rgb),roughness:m.roughness,metalness:m.metallic,transparent:m.transparent,opacity:m.transparent?.18:1,depthWrite:!m.transparent,side:THREE.DoubleSide});
      const mesh=new THREE.Mesh(geometry,material);mesh.name=shape.name;
      mesh.userData.architecture=/^(ceiling|side_wall|right_wall|rear_|front_|window_|wall_cabinet|.*skirting|luminaire|entry_)/.test(shape.name);
      mesh.visible=!(this.cutaway&&mesh.userData.architecture);
      mesh.position.fromArray(shape.pose_local_xyzw);mesh.quaternion.fromArray(shape.pose_local_xyzw,3);
      if(shape.asset_id)mesh.scale.fromArray(s);
      if(shape.body_id<0)this.root.add(mesh);
      else{if(!this.bodies.has(shape.body_id)){const group=new THREE.Group();this.bodies.set(shape.body_id,group);this.root.add(group);}this.bodies.get(shape.body_id).add(mesh);}
    }
  }
  update(state){for(const [i,body] of this.bodies){body.position.fromArray(state.bodies[i]);body.quaternion.fromArray(state.bodies[i],3);}}
  setCutaway(enabled){this.cutaway=enabled;this.root.traverse(o=>{if(o.isMesh&&o.userData.architecture)o.visible=!enabled;});}
}

import * as THREE from 'three';
import {OrbitControls} from '/vendor/OrbitControls.js';

const $ = id => document.getElementById(id);
let key='', current=null, busy=false, polling=null, sceneReady=false, inkRevision=-1, plan=null;
let generation=0, physicsDt=.0005;
let objects=[], hardware={}, boardMesh=null, lastFrameTime=performance.now(), cameraFrames=0;
const held=new Set(), keySince=new Map();
const viewport=$('viewport');
const renderer=new THREE.WebGLRenderer({antialias:true,alpha:false});
renderer.setPixelRatio(Math.min(devicePixelRatio,2));
renderer.outputColorSpace=THREE.SRGBColorSpace;
renderer.toneMapping=THREE.ACESFilmicToneMapping; renderer.toneMappingExposure=1.25;
renderer.shadowMap.enabled=true; renderer.shadowMap.type=THREE.PCFSoftShadowMap;
viewport.appendChild(renderer.domElement);
const scene=new THREE.Scene(); scene.background=new THREE.Color('#5d706a');
const camera=new THREE.PerspectiveCamera(52,1,.01,40); camera.up.set(0,0,1);
const orbit=new OrbitControls(camera,renderer.domElement); orbit.enableDamping=true;
orbit.minDistance=.07; orbit.maxDistance=18; orbit.target.set(-.3,.6,1.2);
camera.position.set(2.5,3.4,2.1); orbit.update();
scene.add(new THREE.HemisphereLight(0xe7f2ff,0x464038,2.2));
const sun=new THREE.DirectionalLight(0xfff1d8,3.2);sun.position.set(2.8,2.5,3.1);sun.target.position.set(0,0,1);
sun.castShadow=true;sun.shadow.mapSize.set(2048,2048);sun.shadow.camera.left=-4;sun.shadow.camera.right=4;
sun.shadow.camera.top=4;sun.shadow.camera.bottom=-4;sun.shadow.normalBias=.0008;scene.add(sun,sun.target);
const targetBall=new THREE.Mesh(new THREE.SphereGeometry(.006,16,12),new THREE.MeshBasicMaterial({color:0xa0ffd2}));
targetBall.visible=false;scene.add(targetBall);
const targetAxes=new THREE.AxesHelper(.06); targetBall.add(targetAxes);
const observer=new ResizeObserver(()=>{const r=viewport.getBoundingClientRect();renderer.setSize(r.width,r.height,false);camera.aspect=r.width/r.height;camera.updateProjectionMatrix();});observer.observe(viewport);
function animate(){requestAnimationFrame(animate);orbit.update();renderer.render(scene,camera);cameraFrames++;const now=performance.now();if(now-lastFrameTime>1000){$('timing').textContent=`Camera ${Math.round(cameraFrames*1000/(now-lastFrameTime))} fps · poses are acknowledged physics`;lastFrameTime=now;cameraFrames=0;}}
animate();

async function api(route,body){
  const response=await fetch(route,{method:body===undefined?'GET':'POST',headers:{Authorization:`Bearer ${key}`,'Content-Type':'application/json'},body:body===undefined?undefined:JSON.stringify(body)});
  const data=await response.json();if(!response.ok||data.ok===false)throw new Error(data.error||`HTTP ${response.status}`);return data;
}
function message(text,error=false){$('message').textContent=text;$('message').style.color=error?'#ffba9c':'';}
function arm(){return $('arm').value;}
function number(id){const raw=$(id).value;const value=Number(raw);if(raw.trim()===''||!Number.isFinite(value))throw new Error(`${id} must be a finite number`);return value;}
function safely(operation){try{return Promise.resolve(operation()).catch(error=>{held.clear();message(error.message,true);});}catch(error){held.clear();message(error.message,true);}}
function ticks(seconds){const value=seconds/physicsDt;if(value<=0||Math.abs(value-Math.round(value))>1e-7)throw new Error(`Time must be an exact positive multiple of ${physicsDt} seconds`);return Math.round(value);}
function invalidatePlan(){plan=null;$('move').disabled=true;targetBall.visible=true;targetBall.position.set(number('x'),number('y'),number('z'));$('plan').textContent='Target changed. Preview IK before moving.';}
for(const id of ['x','y','z'])$(id).addEventListener('input',()=>safely(invalidatePlan));
function useTCP(){if(!current)return;const p=current[arm()].position_m;['x','y','z'].forEach((id,i)=>$(id).value=p[i].toFixed(5));invalidatePlan();}
$('use-tcp').onclick=useTCP;

async function populate(description){
  const oldGeometry=new Set(),oldMaterials=new Set();
  for(const group of objects){group.traverse(item=>{if(item.isMesh){oldGeometry.add(item.geometry);oldMaterials.add(item.material);}});scene.remove(group);} objects=[];
  for(const geometry of oldGeometry)geometry.dispose();for(const material of oldMaterials)material.dispose();
  const materials={};for(const [name,[rgb,rough,metal]] of Object.entries(description.materials))materials[name]=new THREE.MeshStandardMaterial({color:new THREE.Color(...rgb),roughness:rough,metalness:metal,side:THREE.DoubleSide});
  const meshes={};for(const [id,data] of Object.entries(description.meshes)){const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.Float32BufferAttribute(data.vertices,3));geometry.setIndex(data.indices);geometry.computeVertexNormals();meshes[id]=geometry;}
  for(const object of description.objects){
    const group=new THREE.Group();group.position.fromArray(object.position);group.quaternion.fromArray(object.rotation);group.userData.body=object.body;
    for(const part of object.parts){let geometry;if(part.kind==='mesh')geometry=part.name==='elastic_material'?meshes[part.mesh].clone():meshes[part.mesh];else if(part.kind==='box')geometry=new THREE.BoxGeometry(...part.size);else{geometry=new THREE.CylinderGeometry(part.size[0],part.size[0],part.size[1],32);geometry.rotateX(Math.PI/2);}
      const mesh=new THREE.Mesh(geometry,materials[part.material]);mesh.position.fromArray(part.position);mesh.quaternion.fromArray(part.rotation);mesh.castShadow=true;mesh.receiveShadow=true;mesh.name=`${object.name}/${part.name}`;group.add(mesh);}
    scene.add(group);objects.push(group);
  }
  if(boardMesh){scene.remove(boardMesh);boardMesh.geometry.dispose();boardMesh.material.map?.dispose();boardMesh.material.dispose();}
  const b=description.board;boardMesh=new THREE.Mesh(new THREE.PlaneGeometry(b.width,b.height),new THREE.MeshStandardMaterial({color:0xffffff,roughness:.13,side:THREE.DoubleSide}));
  boardMesh.position.set(0,b.face_y+.0001,b.center_z);boardMesh.rotation.x=Math.PI/2;boardMesh.receiveShadow=true;scene.add(boardMesh);sceneReady=true;inkRevision=-1;
}
async function update(fresh=false){
  if(polling){await polling;if(!fresh)return;}
  if(!key||!sceneReady||busy&&!fresh)return;
  const requestedGeneration=generation;
  polling=(async()=>{
  try{const state=await api('/v1/manual/state');if(requestedGeneration!==generation)return;if(current&&state.episode_id===current.episode_id&&state.tick<current.tick)throw new Error('Unannounced clock rewind');current=state;
    for(const group of objects){const p=state.poses[group.userData.body];if(p){group.position.fromArray(p);group.quaternion.fromArray(p.slice(3));}}
    for(const [name,vertices] of Object.entries(state.elastic_vertices||{})){const mesh=scene.getObjectByName(name);if(mesh){mesh.geometry.attributes.position.array.set(vertices);mesh.geometry.attributes.position.needsUpdate=true;mesh.geometry.computeVertexNormals();mesh.geometry.computeBoundingSphere();}}
    $('health').textContent=`${state.ready?'Ready':'Stopped'} · tick ${state.tick.toLocaleString()} · ${state.time_s.toFixed(3)} s`;
    $('tcp').textContent=`Measured TCP (${arm()})\n${state[arm()].position_m.map((v,i)=>`${'XYZ'[i]} ${(v*1000).toFixed(1)} mm`).join(' · ')}`;
    if(inkRevision!==state.ink_revision){const ink=await api('/v1/manual/ink');const texture=await new THREE.TextureLoader().loadAsync(`data:image/png;base64,${ink.png}`);if(requestedGeneration!==generation){texture.dispose();return;}texture.colorSpace=THREE.SRGBColorSpace;if(boardMesh.material.map)boardMesh.material.map.dispose();boardMesh.material.map=texture;boardMesh.material.needsUpdate=true;inkRevision=ink.revision;}
  }catch(error){message(error.message,true);held.clear();$('keyboard').checked=false;if(fresh)throw error;}
  })();
  try{await polling;}finally{polling=null;}
}
async function checkpointList(){const result=await api('/v1/checkpoints');$('checkpoints').replaceChildren(...result.names.map(name=>new Option(name,name)));}
$('connect').onclick=async()=>{if(busy)return;busy=true;held.clear();generation++;current=null;sceneReady=false;key=$('key').value;message('Loading authored scene and current state…');try{physicsDt=(await api('/v1/help')).physics_dt_s;hardware=await api('/v1/robot');await populate(await api('/v1/manual/scene'));$('joint').replaceChildren(...Object.keys(hardware.joints).map(name=>new Option(name,name)));jointChanged();await update(true);useTCP();await checkpointList();message('Connected. Physics is paused between explicit commands.');}catch(error){key='';message(error.message,true);}finally{busy=false;}};

async function command(route,payload){
  if(busy||!current)return;busy=true;generation++;$('busy').textContent='Executing · no queued commands';message('Command sent; waiting for authoritative state…');
  const started=performance.now();try{const result=await api(route,{...payload,id:crypto.randomUUID(),episode_id:current.episode_id});if(result.episode_id!==current.episode_id)current=null;
    await update(true);const c=result.control;message(c?`Executed ${result.ticks_executed} ticks · ${c.stop_reason||'duration complete'}${c.actual_position_error_m!==undefined?' · TCP error '+(c.actual_position_error_m*1000).toFixed(2)+' mm':''}`:`Executed ${result.ticks_executed||0} ticks · ${(performance.now()-started).toFixed(0)} ms wall time`);
    if(c)$('plan').textContent=JSON.stringify({stop:c.stop_reason,actual_position_error_m:c.actual_position_error_m,preview_error_m:c.plan?.position_error_m},null,2);
    plan=null;$('move').disabled=true;return result;
  }catch(error){held.clear();$('keyboard').checked=false;message(`${error.message}. No automatic retry.`,true);await update(true);}finally{busy=false;$('busy').textContent='Idle · no queued commands';}
}
function delta(axis,sign,multiplier=1){const d=[0,0,0];d[axis]=sign*number('step')/1000*multiplier;return command('/v1/cartesian',{action:{type:'cartesian_delta',arm:arm(),delta_m:d,duration_s:number('duration')}});}
for(const button of document.querySelectorAll('[data-axis]'))button.onclick=()=>delta(Number(button.dataset.axis),Number(button.dataset.sign));
function fingers(close){const sign=arm()==='left'?-1:1;const values={thumb_0:0,thumb_1:close?(arm()==='left'?.7:-.7):0,thumb_2:close?(arm()==='left'?.9:-.9):0,index_0:close?sign:0,index_1:close?sign:0,middle_0:close?sign:0,middle_1:close?sign:0};const targets={};for(const [suffix,value] of Object.entries(values))targets[`${arm()}_hand_${suffix}_joint`]=value*1.49;return command('/v1/action',{targets,ticks:ticks(number('duration'))});}
$('open-hand').onclick=()=>fingers(false);$('pinch').onclick=()=>fingers(true);
$('preview').onclick=async()=>{if(!current||busy)return;try{const position=['x','y','z'].map(number);plan=await api('/v1/ik/preview',{arm:arm(),position_m:position});plan.tick=current.tick;plan.arm=arm();$('move').disabled=!plan.reachable;$('plan').textContent=`${plan.reachable?'Preview valid':'Preview rejected'} · ${plan.path_samples} path samples\nPosition error ${(plan.position_error_m*1000).toFixed(3)} mm\nOrientation error ${(plan.orientation_error_rad*180/Math.PI).toFixed(2)}°\n${plan.collisions.length?'Collision: '+JSON.stringify(plan.collisions[0]):'No requested-guard collision in sampled path'}\nDynamics can differ; final measured error is reported.`;targetBall.visible=true;targetBall.position.fromArray(position);}catch(error){message(error.message,true);}};
$('move').onclick=()=>{if(!plan||plan.tick!==current.tick||plan.arm!==arm()){invalidatePlan();return;}command('/v1/cartesian',{action:{type:'cartesian_pose',arm:arm(),position_m:plan.target_m,orientation_wxyz:plan.target_orientation_wxyz,duration_s:number('duration')}});};
function jointChanged(){const joint=hardware.joints[$('joint').value];if(!joint)return;$('joint-limits').textContent=`Range ${joint.lower.toFixed(3)} … ${joint.upper.toFixed(3)} rad · effort limit ${joint.effort_nm} Nm`;$('joint-value').value=current?.joints[$('joint').value]?.position_rad.toFixed(4)||'0';}
$('joint').onchange=jointChanged;$('joint-send').onclick=()=>command('/v1/action',{targets:{[$('joint').value]:number('joint-value')},ticks:ticks(number('duration'))});
$('arm').onchange=()=>{held.clear();useTCP();update();};
$('advance').onclick=()=>command('/v1/action',{targets:{},ticks:ticks(number('advance-time'))});
$('save').onclick=async()=>{if(busy)return;try{const r=await api('/v1/checkpoints/save',{name:$('checkpoint-name').value});message(`Saved '${r.name}' at tick ${r.tick}; physics unchanged.`);await checkpointList();}catch(error){message(error.message,true);}};
$('restore').onclick=()=>command('/v1/checkpoints/restore',{name:$('checkpoints').value});$('reset').onclick=()=>command('/v1/reset',{});
$('board-view').onclick=()=>{camera.position.set(2.5,3.4,2.1);orbit.target.set(-.3,.6,1.2);};
$('room-view').onclick=()=>{camera.position.set(3.1,6.5,2.1);orbit.target.set(-.3,.6,1.3);};
$('hand-view').onclick=()=>{if(!current)return;const p=new THREE.Vector3(...current[arm()].position_m);orbit.target.copy(p);camera.position.copy(p).add(new THREE.Vector3(.27,.38,.19));};
$('rgb-view').onclick=async()=>{if(busy)return;message('Capturing physical onboard RGB; this does not advance physics…');try{const r=await api('/v1/observe',{cameras:['g1_rgb']});$('rgb-image').src=`data:image/png;base64,${r.images.g1_rgb}`;$('rgb-panel').hidden=false;message(`Onboard RGB at tick ${r.tick}; zero physics steps.`);}catch(error){message(error.message,true);}};
$('close-rgb').onclick=()=>{$('rgb-panel').hidden=true;};
const ray=new THREE.Raycaster(),pointer=new THREE.Vector2();let pointerStart=null;
renderer.domElement.addEventListener('pointerdown',e=>{pointerStart=[e.clientX,e.clientY];});
renderer.domElement.addEventListener('pointerup',e=>{if(!e.altKey||!pointerStart||Math.hypot(e.clientX-pointerStart[0],e.clientY-pointerStart[1])>4)return;const r=renderer.domElement.getBoundingClientRect();pointer.set((e.clientX-r.left)/r.width*2-1,-(e.clientY-r.top)/r.height*2+1);ray.setFromCamera(pointer,camera);const hits=ray.intersectObjects(objects,true);if(hits.length){const p=hits[0].point;['x','y','z'].forEach((id,i)=>$(id).value=p.getComponent(i).toFixed(5));invalidatePlan();message(`Selected ${hits[0].object.name}; edit offset, then preview IK.`);}});
const mapping={KeyA:[0,-1],KeyD:[0,1],KeyW:[1,1],KeyS:[1,-1],Space:[2,1],KeyC:[2,-1]};
function typing(element){return element?.closest('input,textarea,select,[contenteditable="true"]');}
window.addEventListener('keydown',event=>{if(!$('keyboard').checked||typing(event.target)||event.altKey||event.ctrlKey||event.metaKey)return;if(mapping[event.code]||['KeyQ','KeyE'].includes(event.code)){event.preventDefault();if(!held.has(event.code))keySince.set(event.code,performance.now());held.add(event.code);}});
window.addEventListener('keyup',event=>{held.delete(event.code);keySince.delete(event.code);});
window.addEventListener('blur',()=>{held.clear();keySince.clear();});document.addEventListener('visibilitychange',()=>{if(document.hidden){held.clear();keySince.clear();}});document.addEventListener('focusin',event=>{if(typing(event.target)){held.clear();keySince.clear();}});
setInterval(()=>{if(!busy&&!polling)update();},1000);
setInterval(()=>safely(()=>{if(busy||!current||!$('keyboard').checked||!held.size)return;const key=[...held][0];if(key==='KeyQ')return fingers(true);else if(key==='KeyE')return fingers(false);else if(mapping[key]){const multiplier=Math.min(4,1+(performance.now()-keySince.get(key))/800);return delta(...mapping[key],multiplier);}}),60);
for(const button of document.querySelectorAll('button')){const handler=button.onclick;if(handler)button.onclick=function(event){return safely(()=>handler.call(this,event));};}

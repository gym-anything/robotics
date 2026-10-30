"use strict";
const el = id => document.getElementById(id);
let key = "", busy = false, owner = false, paused = false, blobs = [];
function controls() {
  for (const id of ["refresh","send","pause","resume"]) el(id).disabled = !key || busy;
  el("send").disabled = !key || busy || paused;
  el("pause").disabled = !key || busy || paused;
  el("resume").disabled = !key || busy || !paused;
  el("connect").disabled = busy;
  el("disconnect").disabled = !key || busy;
}
function notice(text, error=false) { el("notice").textContent=text; el("notice").classList.toggle("error",error); }
async function api(path, payload, binary=false) {
  const options = {headers:{Authorization:"Bearer "+key}, cache:"no-store", redirect:"error"};
  if (payload !== undefined) {
    options.method="POST"; options.headers["Content-Type"]="application/json"; options.body=JSON.stringify(payload);
    if (path==="/v1/action") options.headers["Idempotency-Key"]=Array.from(crypto.getRandomValues(new Uint8Array(24)),n=>n.toString(16).padStart(2,"0")).join("");
  }
  const response=await fetch(path,options);
  if (!response.ok) {
    const text=await response.text();
    throw new Error("HTTP "+response.status+": "+text);
  }
  return binary ? response.blob() : response.json();
}
async function perform(job) {
  if (busy) return;
  busy=true; controls(); notice("Request in progress…");
  try { await job(); } catch(error) { notice(error.message+" No command was retried.",true); }
  finally { busy=false; controls(); }
}
function clearFrames() { for (const url of blobs) URL.revokeObjectURL(url); blobs=[]; el("frames").replaceChildren(); }
let references = [];
async function showTask(task) {
  el("task").textContent=task.instruction || task.task || task.description || JSON.stringify(task);
  for (const url of references) URL.revokeObjectURL(url);
  references=[]; el("reference").replaceChildren();
  for (const [name, info] of Object.entries(task.reference_images || {})) {
    if (!info.url.startsWith("/v1/task/images/")) throw new Error("Invalid task picture URL");
    const url=URL.createObjectURL(await api(info.url,undefined,true)); references.push(url);
    const figure=document.createElement("figure"), image=document.createElement("img"), caption=document.createElement("figcaption");
    image.src=url; image.alt=info.description || name; image.style.width="100%"; caption.textContent=info.description || name;
    figure.append(image,caption); el("reference").append(figure);
  }
}
async function showObservation(obs) {
  const fresh=[];
  try {
    for (const [name, image] of Object.entries(obs.images)) {
      const url=new URL(image.url,location.origin);
      if (url.origin!==location.origin || !url.pathname.startsWith("/v1/cameras/")) throw new Error("Invalid camera URL");
      const blob=await api(url.pathname+url.search,undefined,true);
      fresh.push([name,URL.createObjectURL(blob)]);
    }
  } catch(error) { for(const [,url] of fresh) URL.revokeObjectURL(url); throw error; }
  clearFrames();
  for(const [name,url] of fresh) {
    blobs.push(url);
    const figure=document.createElement("figure"), image=document.createElement("img"), caption=document.createElement("figcaption");
    image.src=url; image.alt=name+" at observation "+obs.observation_id; caption.textContent=name+" · observation "+obs.observation_id;
    figure.append(image,caption); el("frames").append(figure);
  }
  if (!fresh.length) el("frames").textContent="No cameras enabled for this session.";
  el("stamp").textContent="Observation "+obs.observation_id+" · simulation "+obs.sim_time_s+" s · command sequence "+obs.sequence;
  el("proprio").textContent=JSON.stringify(obs.proprioception,null,2);
}
async function sessionStatus() {
  if (!owner) return;
  const state=await api("/v1/session");
  paused=state.paused;
  el("identity").textContent="Session "+state.session_id;
  el("clock").textContent=paused ? "Paused · new commands are blocked. Physics state is retained." : "Command-driven · no physics advances while idle.";
  el("connection").textContent=state.health.ready ? (paused ? "Owner · paused" : "Owner · ready") : "Owner · not ready";
}
async function refresh() { await showObservation((await api("/v1/observation")).observation); await sessionStatus(); }
// Ticks of the pre-filled hold: the shortest command this session takes. No one number is legal in every environment
// (a session with a command floor refuses one tick; the floor follows the session's cameras), so it is read from the
// session: a policy's capabilities give hardware.min_command_s and the tick hardware.dt_s, an owner's (which carry no
// hardware block) notes.min_command_ticks where the backend publishes it. A session that publishes neither has no
// floor to meet and gets one tick, as before.
function shortestTicks(caps) {
  const hardware=(caps&&caps.hardware)||{}, notes=(caps&&caps.notes)||{};
  for (const n of [Number(hardware.min_command_s)/Number(hardware.dt_s), Number(notes.min_command_ticks)])
    if (Number.isFinite(n) && n>=1 && n<=1e7) return Math.ceil(n-1e-9);
  return 1;
}
el("connect").onclick=()=>perform(async()=>{
  key=el("key").value.trim(); if(!key) throw new Error("Enter an API key");
  clearFrames(); el("proprio").textContent=""; el("result").textContent=""; el("help").textContent="";
  let help;
  try { help=await api("/v1/help"); } catch(error) { key=""; owner=false; el("owner").hidden=true; el("connection").textContent="Disconnected"; throw error; }
  owner=help.role==="privileged"; paused=false; el("owner").hidden=!owner;
  el("help").textContent=JSON.stringify({actions:help.action_types,capabilities:help.effective_capabilities||help.capabilities},null,2);
  await showTask(await api("/v1/task"));
  el("connection").textContent=owner?"Owner connected":"Restricted policy connected";
  const caps=(await api("/v1/capabilities")).capabilities;
  if (!el("action").value) {
    const types=caps.action_types||[], ticks=shortestTicks(caps);
    if(types.includes("named_joint_target")) el("action").value=JSON.stringify({type:"named_joint_target",targets_si:{},ticks:ticks},null,2);
    else if(types.includes("named_joint_position")) el("action").value=JSON.stringify({type:"named_joint_position",targets_rad:{},ticks:ticks},null,2);
  }
  el("key").value=""; await refresh(); notice("Connected. No reset or physics step was requested.");
});
el("disconnect").onclick=()=>{key="";owner=false;paused=false;el("owner").hidden=true;el("connection").textContent="Disconnected";clearFrames();el("frames").textContent="Disconnected. The environment remains running.";el("stamp").textContent="No current observation";el("proprio").textContent="";el("task").textContent="Not connected";for(const url of references)URL.revokeObjectURL(url);references=[];el("reference").replaceChildren();el("identity").textContent="";el("help").textContent="";el("result").textContent="";controls();notice("Disconnected only. Server and physics state are preserved.");};
el("refresh").onclick=()=>perform(async()=>{await refresh();notice("Fresh observation received. No physics step requested.");});
el("send").onclick=()=>perform(async()=>{
  const action=JSON.parse(el("action").value);
  const result=await api("/v1/action",action);
  el("result").textContent=JSON.stringify({committed:result.committed,request_id:result.request_id,reward:result.reward,terminated:result.terminated,observation_error:result.observation_error},null,2);
  if(result.observation) await showObservation(result.observation);
  await sessionStatus();
  notice(result.observation_error ? "Command committed; camera unavailable. Do not resend." : "Command committed. Returned observation displayed.",!!result.observation_error);
});
for(const operation of ["pause","resume"]) el(operation).onclick=()=>perform(async()=>{
  await api("/v1/session/"+operation,{}); await sessionStatus();
  notice(operation==="pause"?"Paused after the in-flight command. No reset.":"Commands resumed. No physics step requested.");
});
controls();

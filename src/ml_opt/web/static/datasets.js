import {parseProject} from "./project-io.js";

const el=id=>document.getElementById(id), form=el("dataset-form");
const state={cases:[],busy:false,externalBusy:false,ready:false,runId:null};
const field=name=>form.elements.namedItem(name);
function error(message="") {el("dataset-error").textContent=message;el("dataset-error").hidden=!message;}
async function json(response) {
  const result=await response.json();
  if(!response.ok) throw new Error(typeof result.detail==="string" ? result.detail : result.detail?.map(item=>item.msg).join("; ") || "Check the dataset settings");
  return result;
}
function controls() {
  const locked=state.busy || state.externalBusy;
  el("dataset-settings").disabled=locked;
  el("dataset-load-cases").disabled=locked;
  el("dataset-prepare").disabled=locked || !state.ready || !state.cases.some(item=>item.selected);
  el("dataset-stop").disabled=!state.busy || !state.runId;
}
function renderCases() {
  const target=el("dataset-target"), previous=target.value;
  target.replaceChildren();el("dataset-sources").replaceChildren();
  for(const [index,item] of state.cases.entries()) {
    const label=document.createElement("label"), check=document.createElement("input"), name=document.createElement("span");
    label.className="dataset-source";check.type="checkbox";check.checked=item.selected;
    check.addEventListener("change",()=>{item.selected=check.checked;controls();});
    name.textContent=item.name;label.append(check,name);el("dataset-sources").append(label);
    const option=document.createElement("option");option.value=String(index);option.textContent=item.name;target.append(option);
  }
  if([...target.options].some(option=>option.value===previous)) target.value=previous;
  el("dataset-no-cases").hidden=!!state.cases.length;controls();
}
el("dataset-load-cases").addEventListener("click",()=>el("dataset-case-files").click());
el("dataset-case-files").addEventListener("change",async()=>{
  const files=[...el("dataset-case-files").files];el("dataset-case-files").value="";error();
  const failures=[];
  for(const file of files) {
    try {
      if(file.size>64*1024*1024) throw new Error("The project exceeds the 64 MiB import limit");
      const {model}=parseProject(await file.text());
      const checked=await json(await fetch("/api/project/validate",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(model)}));
      if(state.busy || state.externalBusy) throw new Error("Wait for the active run to finish");
      if(state.cases.some(item=>JSON.stringify(item.problem)===JSON.stringify(checked.model))) continue;
      state.cases.push({name:file.name.replace(/\.json$/i,"").slice(0,120),problem:checked.model,selected:true});
    } catch(e) {failures.push(`${file.name}: ${e.message}`);}
  }
  renderCases();error(failures.join(" "));
});
async function refresh() {
  const [data,health]=await Promise.all([fetch("/api/design-regression/datasets").then(json),fetch("/api/health").then(json)]);
  state.ready=!!health.cross_case_designs_supported;
  if(!state.busy) state.externalBusy=!!health.running;
  const library=el("dataset-library");library.replaceChildren();
  if(!data.datasets.length) {const note=document.createElement("p");note.className="model-message";note.textContent="No saved datasets";library.append(note);}
  for(const item of data.datasets) {
    const row=document.createElement("article"), title=document.createElement("h3"), note=document.createElement("p"), actions=document.createElement("div");
    row.className="dataset-record";title.textContent=item.name || item.id.slice(0,8);
    note.textContent=`${item.case_count} / ${item.expected_cases} designs · ${item.family_count} base designs · ${item.status}`;
    if(item.failed_cases || item.failed_families) note.textContent+=` · ${item.usable_cases} valid labels · ${item.failed_cases} failed evaluations · ${item.failed_families} failed optimisations`;
    actions.className="run-actions";
    const train=document.createElement("button");train.type="button";train.className="button secondary";train.textContent="Use for training";
    train.disabled=state.busy || state.externalBusy || (item.usable_cases ?? item.case_count)===0 || item.status==="generating";
    train.addEventListener("click",()=>{document.dispatchEvent(new CustomEvent("datasetprepared",{detail:item.id}));location.hash="#ml/regression/design";});
    const download=document.createElement("a");download.textContent="Download dataset";download.href=`/api/design-regression/${item.id}/artifacts/dataset`;
    actions.append(train,download);row.append(title,note,actions);library.append(row);
  }
  if(!state.ready) el("dataset-status").textContent="Restart TOptVis to enable dataset preparation";
  controls();
}
form.addEventListener("submit",async event=>{
  event.preventDefault();if(state.busy || state.externalBusy || !state.ready || !form.reportValidity()) return;
  const target=state.cases[Number(el("dataset-target").value)], sources=state.cases.filter(item=>item.selected);
  if(!target || !sources.length) {error("Choose a target and at least one source case");return;}
  const saved=item=>({name:item.name,problem:item.problem});
  const payload={name:field("name").value.trim(),target:saved(target),sources:sources.map(saved)};
  for(const name of ["volume_min","volume_max","volume_count","variants","amplitude","seed"]) payload[name]=Number(field(name).value);
  error();state.busy=true;state.runId=null;controls();
  document.dispatchEvent(new CustomEvent("solveractivity",{detail:{workspace:"datasets",running:true}}));
  el("dataset-status").textContent="Preparing dataset…";
  let terminal=false;
  try {
    const response=await fetch("/api/design-regression/prepare",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)});
    if(!response.ok) await json(response);
    const reader=response.body.getReader(), decoder=new TextDecoder();let buffer="";
    const receive=line=>{
      if(!line.trim()) return;
      const event=JSON.parse(line);
      if(event.event==="error") throw new Error(event.message);
      if(event.event==="started") {state.runId=event.run_id;controls();}
      if(event.event==="design_optimization") el("dataset-status").textContent=`Optimising design ${event.family+1} / ${event.family_count} · iteration ${event.iteration}`;
      if(event.event==="design_case") el("dataset-status").textContent=`Evaluated ${event.completed_cases} / ${event.case_count} designs`;
      if(event.event==="complete") {terminal=true;el("dataset-status").textContent=event.stop_reason==="cancelled" ? "Stopped · partial dataset saved" : "Dataset saved";}
    };
    try {
      while(true) {
        const {value,done}=await reader.read();buffer+=decoder.decode(value || new Uint8Array(),{stream:!done});
        let end;while((end=buffer.indexOf("\n"))>=0) {receive(buffer.slice(0,end));buffer=buffer.slice(end+1);}
        if(done) break;
      }
      if(buffer.trim()) receive(buffer);
      if(!terminal) throw new Error("Connection closed before completion; saved results are retained");
    } finally {await reader.cancel();}
  } catch(e) {error(e.message);el("dataset-status").textContent="Error";}
  finally {
    state.busy=false;state.runId=null;
    document.dispatchEvent(new CustomEvent("solveractivity",{detail:{workspace:"datasets",running:false}}));
    await refresh().catch(e=>error(e.message));controls();
  }
});
el("dataset-stop").addEventListener("click",async()=>{
  try {await json(await fetch(`/api/run/${state.runId}/stop`,{method:"POST"}));el("dataset-status").textContent="Stopping…";} catch(e) {error(e.message);}
});
el("dataset-refresh").addEventListener("click",()=>refresh().catch(e=>error(e.message)));
document.addEventListener("solveractivity",event=>{
  if(event.detail.workspace!=="datasets") {state.externalBusy=event.detail.running;controls();if(!event.detail.running) refresh().catch(e=>error(e.message));}
});
document.addEventListener("mlmethodchange",event=>{if(event.detail==="datasets") refresh().catch(e=>error(e.message));});
controls();

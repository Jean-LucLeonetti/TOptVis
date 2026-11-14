import {drawTopology} from "/static/viewer.js";

const $ = id => document.getElementById(id);
const trainForm=$("ml-train-form"), dataForm=$("ml-data-form"), validationForm=$("ml-validation-form"), experimentForm=$("ml-experiment-form");
const pod=new Set(["pod","energy_pod","local_pod","adaptive_pod"]);
const state={ready:false,busy:false,externalBusy:false,jobId:null,catalog:[],family:"fno",datasets:[],models:[],
  selectedId:null,selected:null,checked:new Set(),history:[],kind:null,density:null,preview:null,terminal:null};
const familyName=id => state.catalog.find(item=>item.id===id)?.name || id;
const usable=model => model?.checkpoint_ready && ["trained","cancelled"].includes(model.status);
const percent=value => value==null ? "undefined" : value!==0&&Math.abs(100*value)<.001 ? `${(100*value).toExponential(2)}%` : `${(100*value).toLocaleString("en",{maximumSignificantDigits:4})}%`;
const seconds=value => `${Number(value).toFixed(3)} s`;
function error(message="") {$("ml-error").textContent=message;$("ml-error").hidden=!message;}
function status(message,kind="ready") {$("ml-status").textContent=message;$("ml-status").dataset.status=kind;}
function selectedCases() {return [...state.checked].filter(id=>usable(state.models.find(model=>model.id===id)));}

function buttons() {
  const disabled=!state.ready||state.busy||state.externalBusy;
  $("ml-train").disabled=disabled||!$("ml-dataset").value;
  $("ml-generate").disabled=disabled;
  $("ml-validate").disabled=disabled||!selectedCases().length;
  $("ml-validate-selected").disabled=disabled||!usable(state.selected);
  $("ml-experiment").disabled=disabled||!usable(state.selected);
  $("ml-refresh").disabled=state.busy;
  $("ml-stop").disabled=!state.busy||!state.jobId;
  for (const id of ["ml-train-settings","ml-data-settings","ml-validation-settings","ml-experiment-settings"]) $(id).disabled=state.busy;
  for (const button of $("ml-families").querySelectorAll("button")) button.disabled=state.busy;
  for (const button of $("ml-model-rows").querySelectorAll("button,input")) button.disabled=state.busy||!usable(state.models.find(model=>model.id===button.dataset.model));
  $("ml-availability").textContent=!state.ready ? "Restart the local TOptVis server and refresh to enable Model Lab." : state.externalBusy ?
    "Another workspace is using the solver worker. Finish or stop that job before starting another." :
    "Local experiments · shared FOM snapshots · case-separated fitting · independent physics validation.";
}

function busy(value) {
  state.busy=value; buttons();
  document.dispatchEvent(new CustomEvent("solveractivity",{detail:{workspace:"model-lab",running:value}}));
}

function selectFamily(id) {
  state.family=id;
  for (const button of $("ml-families").children) button.setAttribute("aria-pressed",String(button.dataset.family===id));
  $("ml-family-description").textContent=state.catalog.find(item=>item.id===id)?.description || "";
  for (const element of document.querySelectorAll("[data-ml-neural]")) element.hidden=pod.has(id);
  for (const element of document.querySelectorAll("[data-ml-pod]")) element.hidden=!pod.has(id)&&id!=="pod_mlp";
  for (const [key,family] of [["local","local_pod"],["adaptive","adaptive_pod"],["fno","fno"],["latent","deeponet"]]) {
    for (const element of document.querySelectorAll(`[data-ml-${key}]`)) element.hidden=id!==family;
  }
  $("ml-train").textContent=`${pod.has(id)?"Build":"Train"} ${familyName(id)}`;
}

function dataNote() {
  const record=state.datasets.find(item=>item.id===$("ml-dataset").value);
  $("ml-dataset-note").textContent=record ? `${record.settings.nx} × ${record.settings.ny} elements · ${record.snapshot_count} states · ${record.case_count} source cases. Entire cases are held out; snapshots are never randomly split across a trajectory.` : "Generate snapshots here or reuse a ROM Training dataset.";
}

function chart() {
  const svg=$("ml-loss-chart");
  const rows=state.history.filter(row=>Number.isFinite(row.train_loss)&&Number.isFinite(row.validation_loss));
  if (!rows.length) {svg.innerHTML='<text x="410" y="95" text-anchor="middle" fill="#91a0a6">Training curves will appear here</text>';return;}
  const values=rows.flatMap(row=>[row.train_loss,row.validation_loss]).map(value=>Math.log10(Math.max(value,1e-12)));
  const low=Math.min(...values)-.2, high=Math.max(low+1,...values)+.2;
  const x=index=>54+(rows.length===1?0:index/(rows.length-1))*730;
  const y=value=>148-(Math.log10(Math.max(value,1e-12))-low)/(high-low)*126;
  const path=key=>rows.map((row,index)=>`${index?"L":"M"}${x(index).toFixed(2)} ${y(row[key]).toFixed(2)}`).join(" ");
  svg.innerHTML=`<title>Loss through epoch ${rows.at(-1).epoch}. Training objective ${rows.at(-1).train_loss.toPrecision(4)}, case-validation displacement MSE ${rows.at(-1).validation_loss.toPrecision(4)}.</title>
    <path d="M54 18V148H788" fill="none" stroke="#d6e1df"/>
    <text x="8" y="25" fill="#81969a">10^${high.toFixed(1)}</text><text x="8" y="149" fill="#81969a">10^${low.toFixed(1)}</text>
    <text x="54" y="172" fill="#81969a">Epoch ${rows[0].epoch}</text><text x="784" y="172" text-anchor="end" fill="#81969a">${rows.at(-1).epoch}</text>
    <path d="${path("train_loss")}" fill="none" stroke="#196964" stroke-width="2"/><path d="${path("validation_loss")}" fill="none" stroke="#c18a41" stroke-width="2"/>
    <circle cx="${x(rows.length-1)}" cy="${y(rows.at(-1).validation_loss)}" r="3" fill="#c18a41"/>`;
}

function redraw() {
  if (state.preview&&state.density&&$("ml-preview").getClientRects().length) drawTopology($("ml-preview"),state.preview,state.density);
}
function preview(config,density) {state.preview=config;state.density=density;$("ml-preview-empty").hidden=true;redraw();}

function evidenceCell(model,mode) {
  const cell=document.createElement("td"), validation=model.validation, summary=validation?.summaries?.[mode];
  if (!summary) {cell.textContent="Not validated";return cell;}
  cell.textContent=`C ${percent(summary.metrics.compliance.max)} / ∇ ${percent(summary.metrics.gradient.max)}`;
  cell.dataset.passed=String(summary.meets_tolerances);
  const note=document.createElement("small");
  note.textContent=`u ${percent(summary.metrics.displacement.max)} · ${summary.passed_states}/${summary.state_count} pass · ${summary.solver_statistics.fom_solves} FOM solves`;
  cell.append(note);
  const cases=document.createElement("small");cases.textContent=`${validation.status} · cases ${validation.case_signature.slice(0,8)}`;cell.append(cases);
  return cell;
}

function library() {
  $("ml-model-rows").replaceChildren();
  $("ml-model-count").textContent=`${state.models.length} models`;
  $("ml-library-empty").hidden=!!state.models.length;
  for (const model of state.models) {
    const row=document.createElement("tr");row.dataset.selected=String(model.id===state.selectedId);
    const checkCell=document.createElement("td"), check=document.createElement("input");
    check.type="checkbox";check.dataset.model=model.id;check.checked=state.checked.has(model.id);
    check.setAttribute("aria-label",`Compare ${familyName(model.family)} ${model.id.slice(0,8)}`);
    check.addEventListener("change",()=>{check.checked?state.checked.add(model.id):state.checked.delete(model.id);buttons();});
    checkCell.append(check);
    const identity=document.createElement("td"), strong=document.createElement("strong"), note=document.createElement("small");
    strong.textContent=`${familyName(model.family)} · ${model.id.slice(0,8)}`;
    note.textContent=model.config ? `${model.config.nx}×${model.config.ny} · dataset ${model.source_training_id?.slice(0,8)} · ${model.status}` : `${model.status} · ${model.error||"No checkpoint"}`;
    identity.append(strong,note);
    const fit=document.createElement("td");fit.textContent=`${model.train_states??"—"} train / ${model.validation_states??"—"} validation states`;
    const fitNote=document.createElement("small");fitNote.textContent=model.ranks?`${model.ranks.join(" / ")} modes · CPU`:model.best_epoch?`Best epoch ${model.best_epoch} · ${model.device}`:"No saved fit";fit.append(fitNote);
    const action=document.createElement("td"), button=document.createElement("button");
    button.type="button";button.className="button secondary";button.textContent="Inspect";button.dataset.model=model.id;
    button.setAttribute("aria-label",`Inspect ${familyName(model.family)} ${model.id.slice(0,8)}`);
    button.addEventListener("click",()=>inspect(model.id).catch(e=>error(e.message)));action.append(button);
    row.append(checkCell,identity,fit,evidenceCell(model,"surrogate"),evidenceCell(model,"corrected"),action);
    $("ml-model-rows").append(row);
  }
  buttons();
}

function downloads(kind,model) {
  const metadata=kind==="model"?model:model[kind];
  $(kind==="model"?"ml-model-downloads":`ml-${kind}-downloads`).hidden=!metadata;
  if (!metadata) return;
  const prefix=`/api/ml/models/${model.id}/artifacts/`;
  if (kind==="model") {
    $("ml-download-model").href=prefix+"model";$("ml-download-metadata").href=prefix+"metadata";
    $("ml-model-identity").textContent=`${familyName(model.family)} · checkpoint ${model.model_sha256?.slice(0,12)}…`;
  } else {
    $(`ml-download-${kind}`).href=prefix+kind;$(`ml-download-${kind}-arrays`).href=prefix+kind+"_arrays";
    $(`ml-${kind}-identity`).textContent=`${kind} ${metadata.id.slice(0,8)} · ${metadata.status}`;
  }
}

function experimentSummary(report) {
  const check=report.final_fom_check, stats=report.solver_statistics;
  $("ml-experiment-result").textContent=`${report.status} · ${report.iterations} iterations · ${report.mode} · model C ${check.rom_compliance.toFixed(3)}, FOM C ${check.full_compliance.toFixed(3)} · errors: u ${percent(check.displacement_error)}, C ${percent(check.compliance_error)}, ∇ ${percent(check.gradient_error)} · ${check.passed?"within":"outside"} comparison tolerances. Optimisation ${seconds(report.elapsed_seconds)} + final FOM check ${seconds(report.final_fom_check_seconds)}. ${stats.fom_solves} FOM solves during optimisation; ${stats.correction_iterations} correction iterations; ${stats.lu_fallbacks} LU fallbacks; ${stats.basis_updates} basis updates. Full-order volume ${(100*report.full_volume).toFixed(2)}%.`;
}

async function inspect(id,{syncTraining=true}={}) {
  state.selectedId=id;
  const response=await fetch(`/api/ml/models/${id}`);
  if (!response.ok) throw new Error("Could not load this saved model.");
  const model=await response.json();
  if (state.selectedId!==id) return;
  state.selected=model;state.history=model.history||[];chart();
  if (syncTraining) {
    selectFamily(model.family);
    for (const [name,value] of Object.entries(model.options||{})) {
      const field=trainForm.elements.namedItem(name);
      if (field) field.value=["validation_fraction","energy_target"].includes(name)?100*value:value;
    }
    if (state.datasets.some(item=>item.id===model.source_training_id)) $("ml-dataset").value=model.source_training_id;
  }
  if ($( "ml-device").value==="cuda"&&$("ml-cuda-option").disabled) $("ml-device").value="cpu";
  dataNote();
  $("ml-epoch").textContent=state.history.length?`Epoch ${state.history.at(-1).epoch}`:"No epochs";
  $("ml-training-note").textContent=`${familyName(model.family)} · ${model.status} · ${model.elapsed_seconds?seconds(model.elapsed_seconds):"fit in progress"}${model.best_epoch?` · best checkpoint at epoch ${model.best_epoch}`:""}.`;
  $("ml-split-note").textContent=model.split?`Train cases: ${model.split.train_case_indices.map(i=>i+1).join(", ")}. Validation cases: ${model.split.validation_case_indices.map(i=>i+1).join(", ")}. ${model.split.scope}.`:"No split available.";
  $("ml-experiment-model").textContent=`${familyName(model.family)} · ${id.slice(0,8)}`;
  for (const kind of ["model","validation","experiment"]) downloads(kind,model);
  if (model.validation) {
    $("ml-validation-result").textContent=Object.entries(model.validation.summaries).map(([mode,summary])=>`${mode}: ${summary.passed_states}/${summary.state_count} within all limits; max errors u ${percent(summary.metrics.displacement.max)}, C ${percent(summary.metrics.compliance.max)}, ∇ ${percent(summary.metrics.gradient.max)}; ${summary.solver_statistics.fom_solves} FOM solves. Model evaluations ${seconds(summary.model_evaluation_seconds)}, reference evaluations ${seconds(summary.full_evaluation_seconds)}.`).join(" ");
  } else $("ml-validation-result").textContent="This model has no independent unseen-case validation yet.";
  if (model.experiment) experimentSummary(model.experiment);
  else $("ml-experiment-result").textContent="Ready to try this model. FEM-corrected mode is selected by default; raw model runs are experimental.";
  library();
}

async function refresh({preferredDataset=null,preferredModel=null}={}) {
  const [healthResponse,dataResponse,modelResponse]=await Promise.all([fetch("/api/health"),fetch("/api/rom/trainings"),fetch("/api/ml/models")]);
  if (!healthResponse.ok||!dataResponse.ok||!modelResponse.ok) throw new Error("Model Lab is unavailable. Restart TOptVis and refresh.");
  const health=await healthResponse.json();state.ready=!!health.model_lab_supported;
  state.externalBusy=health.running&&!state.busy;
  state.datasets=(await dataResponse.json()).trainings.filter(item=>item.snapshot_count>0&&["snapshots_ready","basis_ready","cancelled"].includes(item.status));
  state.models=(await modelResponse.json()).models;
  const previous=preferredDataset||$("ml-dataset").value;
  $("ml-dataset").replaceChildren();
  for (const record of state.datasets) $("ml-dataset").add(new Option(`${record.settings.nx}×${record.settings.ny} · ${record.snapshot_count} states · ${record.id.slice(0,8)}`,record.id));
  if (!state.datasets.length) $("ml-dataset").add(new Option("No saved datasets",""));
  if (state.datasets.some(item=>item.id===previous)) $("ml-dataset").value=previous;
  dataNote();
  const id=preferredModel||state.selectedId||state.models.find(usable)?.id;
  if (preferredModel) state.checked.add(preferredModel);
  library();
  if (id&&state.models.some(item=>item.id===id)) await inspect(id,{syncTraining:!!preferredModel&&!preferredDataset});
  buttons();
}

function formValues(form,{lists=[],strings=[]}={}) {
  return Object.fromEntries([...new FormData(form)].map(([key,value])=>{
    if (strings.includes(key)) return [key,value];
    if (lists.includes(key)) {
      const entries=value.split(",").map(item=>item.trim());
      if (entries.some(item=>!item||!Number.isFinite(Number(item)))) throw new Error("Enter comma-separated numbers without empty entries.");
      return [key,entries.map(Number)];
    }
    return [key,Number(value)];
  }));
}

async function receive(event) {
  if (event.event==="error") throw new Error(event.message);
  if (event.event==="started") {state.jobId=event.run_id;buttons();}
  else if (event.event==="ml_phase") $("ml-training-note").textContent=event.message;
  else if (event.event==="ml_epoch") {
    state.history.push(event);chart();$("ml-epoch").textContent=`Epoch ${event.epoch}`;
    $("ml-progress").value=Math.min(99,100*event.epoch/Number(trainForm.elements.namedItem("epochs").value));
    $("ml-training-note").textContent=`${familyName(state.family)} · training objective ${event.train_loss.toPrecision(4)} · case-validation displacement error ${percent(event.validation_displacement_error)} · ${seconds(event.seconds)}.`;
  } else if (event.event==="case_started") {state.preview=event.config;status("Generating FOM snapshots","running");}
  else if (event.event==="snapshot") {preview(state.preview,event.density);$("ml-progress").value=100*event.snapshot_count/event.maximum_snapshots;$("ml-training-note").textContent=`Collected ${event.snapshot_count} full-order snapshots.`;}
  else if (event.event==="ml_validation_case") status(`Validating case ${event.case_index+1}/${event.case_count}`,"running");
  else if (event.event==="ml_validation_state") {
    preview(event.config,event.density);
    state.validationSeen++;
    $("ml-progress").value=Math.min(99,100*state.validationSeen/state.validationMaximum);
    const row=event.result;
    $("ml-validation-result").textContent=`${familyName(event.family)} · ${row.mode} · case ${row.case_index+1}, iteration ${row.iteration}: errors u ${percent(row.displacement_error)}, C ${percent(row.compliance_error)}, ∇ ${percent(row.gradient_error)}.`;
  } else if (event.event==="ml_model_validated") {
    $("ml-validation-result").textContent=`Saved ${familyName(event.report.family)} comparison · cases ${event.report.case_signature.slice(0,8)}. Continuing with the remaining models…`;
  } else if (event.event==="ml_iteration") {
    preview(event.config,event.density);$("ml-progress").value=100*event.history.iteration/event.config.max_iterations;
    $("ml-experiment-result").textContent=`Iteration ${event.history.iteration}/${event.config.max_iterations} · ${event.history.state_solver} · C ${event.history.compliance.toFixed(3)} · volume ${(100*event.history.volume_fraction).toFixed(2)}% · residual ${event.history.linear_residual.toExponential(2)} · ${event.solver_statistics.fom_solves} FOM solves so far. Final FOM comparison pending.`;
  } else if (event.event==="complete") {
    state.terminal=event;status(event.stop_reason==="cancelled"?"Stopped":"Complete",event.stop_reason==="cancelled"?"ready":"complete");
    if (event.stop_reason!=="cancelled") $("ml-progress").value=100;
    if (event.experiment) experimentSummary(event.experiment);
  }
}

async function run(url,payload,kind) {
  if (state.busy||state.externalBusy||!state.ready) return;
  error();state.kind=kind;state.jobId=null;state.terminal=null;$("ml-progress").value=0;
  if (kind==="validate") {
    state.validationSeen=0;
    state.validationMaximum=(payload.model_ids?.length||1)*payload.modes.length*payload.load_heights.length*
      payload.load_angles.length*payload.volume_fractions.length*(1+Math.ceil(payload.iterations_per_case/payload.comparison_stride));
  }
  if (kind==="train") {state.history=[];chart();}
  busy(true);status(kind==="train"?"Training":kind==="data"?"Generating data":kind==="experiment"?"Optimising":"Validating","running");
  let reader;
  try {
    const response=await fetch(url,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)});
    if (!response.ok) {
      const body=await response.json();throw new Error(typeof body.detail==="string"?body.detail:JSON.stringify(body.detail));
    }
    reader=response.body.getReader();const decoder=new TextDecoder();let buffer="";
    while (true) {
      const {done,value}=await reader.read();buffer+=decoder.decode(value||new Uint8Array(),{stream:!done});
      const lines=buffer.split("\n");buffer=lines.pop();
      for (const line of lines) if (line.trim()) await receive(JSON.parse(line));
      if (done) {if (buffer.trim()) await receive(JSON.parse(buffer));break;}
    }
    if (!state.terminal) throw new Error("The worker disconnected before a terminal result. Any saved checkpoints remain in the library.");
  } catch (e) {error(e.message);status("Job stopped","ready");if (reader) await reader.cancel().catch(()=>{});}
  finally {
    if (reader) reader.releaseLock();busy(false);state.jobId=null;
    try {await refresh({preferredDataset:state.terminal?.training_id,preferredModel:kind==="train"?state.terminal?.model_id:null});}
    catch (e) {error(e.message);}
  }
}

trainForm.addEventListener("submit",event=>{
  event.preventDefault();try {
    const options=formValues(trainForm,{strings:["device"]});options.family=state.family;
    options.validation_fraction/=100;options.energy_target/=100;
    run("/api/ml/train",{training_id:$("ml-dataset").value,options},"train");
  } catch (e) {error(e.message);}
});
dataForm.addEventListener("submit",event=>{event.preventDefault();try {run("/api/rom/snapshots",{...formValues(dataForm,{lists:["load_heights","load_angles","volume_fractions"]}),projection_eta:.5},"data");}catch(e){error(e.message);}});
function validationPayload() {
  const value=formValues(validationForm,{lists:["load_heights","load_angles","volume_fractions"],strings:["evaluation"]});
  value.modes=value.evaluation==="both"?["surrogate","corrected"]:[value.evaluation];delete value.evaluation;
  for (const key of ["displacement_tolerance","compliance_tolerance","gradient_tolerance"]) value[key]/=100;
  return value;
}
validationForm.addEventListener("submit",event=>{event.preventDefault();try {run("/api/ml/compare",{...validationPayload(),model_ids:selectedCases()},"validate");}catch(e){error(e.message);}});
$("ml-validate-selected").addEventListener("click",()=>{if (!validationForm.reportValidity())return;try {run(`/api/ml/models/${state.selectedId}/validate`,validationPayload(),"validate");}catch(e){error(e.message);}});
experimentForm.addEventListener("submit",event=>{event.preventDefault();run(`/api/ml/models/${state.selectedId}/experiment`,formValues(experimentForm,{strings:["mode","optimizer"]}),"experiment");});
$("ml-dataset").addEventListener("change",dataNote);
$("ml-refresh").addEventListener("click",()=>refresh().catch(e=>error(e.message)));
$("ml-stop").addEventListener("click",async()=>{if (!state.jobId)return;$("ml-stop").disabled=true;try {const response=await fetch(`/api/run/${state.jobId}/stop`,{method:"POST"});if (!response.ok)throw new Error("This job is no longer active.");status("Stopping…","running");}catch(e){error(e.message);buttons();}});
document.addEventListener("solveractivity",event=>{if (event.detail.workspace!=="model-lab"){state.externalBusy=event.detail.running;buttons();}});
document.addEventListener("workspacechange",event=>{if (event.detail==="model-lab"){redraw();if (!state.busy)refresh().catch(e=>error(e.message));}});
new ResizeObserver(redraw).observe($("ml-preview"));

async function init() {
  try {
    const response=await fetch("/api/ml/catalog");if (!response.ok)throw new Error("Model Lab requires the updated local app server.");
    const catalog=await response.json();state.catalog=catalog.families;
    for (const family of state.catalog) {
      const button=document.createElement("button");button.type="button";button.className="ml-family";button.dataset.family=family.id;
      const group=document.createElement("small"),name=document.createElement("strong");group.textContent=family.group;name.textContent=family.name;
      button.append(group,name);button.addEventListener("click",()=>selectFamily(family.id));$("ml-families").append(button);
    }
    $("ml-cuda-option").disabled=!catalog.cuda_available;$("ml-cuda-option").textContent=catalog.cuda_available?"CUDA GPU":"CUDA GPU · unavailable";
    $("ml-device-note").textContent=catalog.cuda_available?"CUDA GPU available. Saved models are evaluated on CPU for these local comparisons.":"CPU PyTorch environment. CUDA training needs a GPU-enabled PyTorch installation and compatible hardware.";
    selectFamily(state.family);chart();await refresh();
  } catch(e) {error(e.message);buttons();}
}
init();

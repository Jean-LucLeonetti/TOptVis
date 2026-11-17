import {drawTopology,pickElement} from "./viewer.js";
import {createTrainingChart} from "./network-training.js";
import {predictDesign,editDensity,designModelName,comparableReports} from "./design-models.js";

const el=id=>document.getElementById(id), generate=el("design-generate-form"), fit=el("design-fit-form");
const input=(form,name)=>form.elements.namedItem(name);
const number=value=>Number.isFinite(value) ? value.toLocaleString("en",{maximumFractionDigits:3}) : "Unavailable";
const percent=value=>Number.isFinite(value) ? `${(value*100).toFixed(2)}%` : "—";
const state={dataset:null,reports:{},models:new Map(),busy:false,externalBusy:false,ready:false,runId:null,problem:null,originalDatasetId:null,crossSupported:false,drafts:new Map(),checks:new Map(),loading:null,history:[],active:false};
const learning=createTrainingChart("design");
const selected=()=>state.models.get(el("design-model").value);
const reference=()=>state.dataset?.cases.find(item=>String(item.index)===el("design-case").value);
const density=()=>state.drafts.get(reference()?.index) || reference()?.density;
const use=(report,index)=>report?.test_indices.includes(index) ? "Test" : report?.validation_indices.includes(index) ? "Validation" : report?.train_indices.includes(index) ? "Training" : "Reference";
function error(message="") {el("design-error").textContent=message;el("design-error").hidden=!message;el("dataset-legacy-error").textContent=message;el("dataset-legacy-error").hidden=!message;}
function status(message) {el("design-status").textContent=message;el("dataset-legacy-status").textContent=message;}
async function json(response) {
  const result=await response.json();
  if(!response.ok) throw new Error(typeof result.detail==="string" ? result.detail : "Check the experiment settings");
  return result;
}
function options(select,items,value) {
  select.replaceChildren(...items.map(([id,label])=>{const option=document.createElement("option");option.value=id;option.textContent=label;return option;}));
  if([...select.options].some(option=>option.value===value)) select.value=value;
}
function controls() {
  const locked=state.busy || state.externalBusy || !state.ready;
  el("design-generate-settings").disabled=locked;
  el("design-fit-settings").disabled=locked || !state.dataset;
  const cross=input(generate,"generation_mode").value==="cross";
  el("design-generate").disabled=locked || (cross && !state.crossSupported) || (!input(generate,"source_id").value && !(cross && (state.problem || state.originalDatasetId)));
  el("design-use-current").disabled=locked;
  input(generate,"source_id").required=!(cross && (state.problem || state.originalDatasetId));
  el("design-fit").disabled=locked || !state.dataset;
  el("design-stop").disabled=!state.busy || !state.runId;
  el("dataset-legacy-stop").disabled=el("design-stop").disabled;
  for(const id of ["design-dataset","design-refresh"]) el(id).disabled=state.busy;
  el("design-case").disabled=state.busy || !reference();
  el("design-model").disabled=state.busy || !state.models.size;
  for(const id of ["design-element","design-value","design-apply","design-reset"]) el(id).disabled=state.busy || !density();
  el("design-evaluate").disabled=locked || !density();
}
function generationMode() {
  const cross=input(generate,"generation_mode").value==="cross";
  el("design-cross-options").hidden=!cross;el("design-source-label").textContent=cross ? "Original case" : "Base designs";
  for(const field of el("design-cross-options").querySelectorAll("input")) field.disabled=!cross;
  controls();
}
function method() {
  const current=input(fit,"method").value;
  for(const [id,shown] of [["design-order-options",current!=="neural"],["design-ridge-options",current==="ridge"],["design-neural-options",current==="neural"]]) {
    el(id).hidden=!shown;for(const field of el(id).querySelectorAll("input")) field.disabled=!shown;
  }
}
function preview() {
  const item=reference(), values=density(), fitted=selected(), config=state.dataset?.fixed_config;
  const family=state.dataset?.families?.find(f=>String(f.index)===String(item?.family));
  const edited=state.drafts.has(item?.index), check=state.checks.get(item?.index);
  const actual=check?.compliance ?? (!edited ? item?.compliance : null);
  let predicted=null;
  if(fitted && values) {try {predicted=predictDesign(fitted.model,values);} catch { /* Show unavailable; FE check remains available. */ }}
  el("design-prediction").textContent=number(predicted);
  el("design-actual").textContent=number(actual);
  el("design-relative").textContent=actual>0 && predicted!=null ? percent(Math.abs(predicted-actual)/actual) : "Unavailable";
  el("design-volume").textContent=values ? percent(values.reduce((sum,v)=>sum+v,0)/values.length) : "Unavailable";
  el("design-density-empty").hidden=!!values;
  el("design-case-note").textContent=item ? `Family ${item.family} · ${item.variant===0 ? "Base design" : `Variation ${item.variant}`} · ${use(fitted?.report,item.index)}${edited ? " · Edited" : ""}` : "";
  el("design-check-note").textContent=check ? `Equilibrium residual ${check.residual.toExponential(2)}. ${state.dataset.source_robust ? "Nominal material checked." : ""}` : edited ? "Edited design: check structure to obtain its actual objective." : "";
  if(family) el("design-case-note").textContent+=` · ${family.name} · ${family.converged ? "Converged" : family.stop_reason || "Unfinished"}${family.feasible===false ? " · Source constraints unmet" : ""}`;
  el("design-setup-view").hidden=!state.dataset?.families;
  el("design-element").max=values?.length || 1;
  if(config && el("design-density").getClientRects().length) drawTopology(el("design-density"),{...(el("design-view-setup").value==="source" && family?.config ? family.config : config),filter_radius:0,max_solid_size:0,max_void_size:0},values);
  const connection=check || (!edited ? item : null);
  if(connection?.connection_constraints) {
    const pass=Object.values(connection.connection_constraints).every(value=>value<=(state.dataset.fixed_config.feasibility_tolerance ?? 1e-4)), path=Object.keys(connection.connection_constraints).every(key=>connection.connection_connected?.[key]===true);
    el("design-check-note").textContent+=` Connection limit ${pass ? "satisfied" : "exceeded"}; path at ρ ≥ 0.5 ${path ? "connected" : "disconnected"}.`;
  }
  controls();
}
function chart() {
  const fitted=selected(), svg=el("design-chart");
  el("design-chart-empty").hidden=!!fitted;
  el("design-axis-note").hidden=true;
  if(!fitted) {svg.replaceChildren();return;}
  const points=fitted.model.cases, logarithmic=el("design-log-axis").checked;
  const valid=points.filter(p=>p.reference>0 && p.predicted>0);
  const shown=logarithmic ? valid : points;
  if(logarithmic && valid.length<points.length) {el("design-axis-note").hidden=false;el("design-axis-note").textContent=`${points.length-valid.length} nonpositive predictions omitted from log axes. Use linear axes to inspect them.`;}
  if(!shown.length) {svg.replaceChildren();return;}
  const transform=v=>logarithmic ? Math.log10(v) : v;
  const values=shown.flatMap(p=>[transform(p.reference),transform(p.predicted)]);
  let low=Math.min(...values),high=Math.max(...values);const pad=Math.max((high-low)*.08,Math.abs(high)*.01,1e-6);low-=pad;high+=pad;
  const width=Math.max(360,svg.clientWidth || 800),height=350,left=90,right=width-32,top=28,bottom=290;
  const x=v=>left+(right-left)*(v-low)/(high-low),y=v=>bottom-(bottom-top)*(v-low)/(high-low);
  svg.setAttribute("viewBox",`0 0 ${width} ${height}`);
  let content="<title>Predicted versus actual objective; diagonal indicates exact prediction</title>";
  for(let i=0;i<=4;i++) {
    const v=low+(high-low)*i/4,value=logarithmic ? 10**v : v;
    const label=Math.abs(value)>1e5 || Math.abs(value)<.001 ? value.toExponential(1) : number(value);
    content+=`<path d="M${left} ${y(v)}H${right}M${x(v)} ${top}V${bottom}" stroke="#e4e9ed"/><text x="${left-10}" y="${y(v)+5}" text-anchor="end">${label}</text><text x="${x(v)}" y="${bottom+23}" text-anchor="middle">${label}</text>`;
  }
  content+=`<path d="M${left} ${bottom}L${right} ${top}" stroke="#7961b2" stroke-dasharray="5 4"/>`;
  for(const p of shown) {
    const group=use(fitted.report,p.index),color={Training:"#196964",Validation:"#4b7eaf",Test:"#df7a40"}[group];
    const label=`Case ${p.index+1}, ${group}, actual ${number(p.reference)}, predicted ${number(p.predicted)}`;
    content+=`<circle class="reg-chart-point" data-case="${p.index}" cx="${x(transform(p.reference))}" cy="${y(transform(p.predicted))}" r="${String(p.index)===el("design-case").value ? 7 : 5}" fill="${color}" stroke="white" tabindex="0" role="button" aria-label="${label}"><title>${label}</title></circle>`;
  }
  content+=`<text x="${(left+right)/2}" y="340" text-anchor="middle">Actual${logarithmic ? " (log scale)" : ""}</text><text x="${left}" y="18">Predicted${logarithmic ? " (log scale)" : ""}</text>`;
  svg.innerHTML=content;
  svg.querySelectorAll("[data-case]").forEach(point=>{
    const select=()=>{el("design-case").value=point.dataset.case;selectCase();chart();};
    point.addEventListener("click",select);point.addEventListener("keydown",event=>{if(event.key==="Enter" || event.key===" ") {event.preventDefault();select();}});
  });
}
function row(name,scores,active=false) {
  const tr=document.createElement("tr");tr.dataset.active=String(active);
  for(const text of [name,...["train","validation","test"].map(key=>percent(scores[key]?.mean_relative_error))]) {
    const cell=document.createElement("td");cell.textContent=text;tr.append(cell);
  }
  return tr;
}
function render() {
  const previous=state.preferredModel ?? el("design-model").value;state.preferredModel=null;state.models.clear();
  for(const report of Object.values(state.reports)) if(report) for(const [key,model] of Object.entries(report.models)) state.models.set(key,{model,report});
  options(el("design-model"),[...state.models].map(([key,{model}])=>[key,designModelName(model)]),previous);
  const oldCase=el("design-case").value;
  options(el("design-case"),(state.dataset?.cases || []).filter(c=>c.status==="complete").map(c=>[String(c.index),`Case ${c.index+1} · family ${c.family}`]),oldCase);
  const fitted=selected(), scores=el("design-scores");scores.replaceChildren();
  for(const [key,{model}] of state.models) scores.append(row(designModelName(model),model,key===el("design-model").value));
  if(fitted) {
    scores.append(row("Training mean",fitted.report.baseline),row("Volume only",fitted.report.volume_baseline));
    const r=fitted.report;
    const mismatch=[...state.models.values()].some(item=>!comparableReports(r,item.report));
    el("design-fit-note").textContent=`${r.train_indices.length} training / ${r.validation_indices.length} validation / ${r.test_indices.length} test cases. ${mismatch ? "Saved models use different splits; scores are not directly comparable." : "Whole design families are held out."}`;
  } else el("design-fit-note").textContent="At least three independent base designs are needed.";
  const data=state.dataset;
  el("design-dataset-note").textContent=data ? `${data.cases.length} / ${data.expected_cases} cases · ${data.family_count} base designs · ${data.status}` : "";
  const objective=data?.objective_quantity==="elastic_work" ? "elastic work" : "compliance";
  el("design-prediction-title").textContent=data?.source_kind==="cross_cases" ? `Original-case ${objective} prediction` : `${objective[0].toUpperCase()+objective.slice(1)} prediction`;
  el("design-downloads").hidden=!data;
  if(data) el("design-download-data").href=`/api/design-regression/${data.id}/artifacts/dataset`;
  el("design-download-models").hidden=!fitted;
  if(fitted) el("design-download-models").href=`/api/design-regression/${data.id}/artifacts/models-${fitted.report.method}`;
  const model=fitted?.model;learning.update(state.history.length ? state.history : model?.training_history,state.history.length ? null : model);
  chart();preview();
}
function selectCase() {
  const values=density(), element=Math.max(0,Math.min((values?.length || 1)-1,Number(el("design-element").value)-1));
  el("design-element").value=element+1;el("design-value").value=values ? Number(values[element].toPrecision(6)) : 1;preview();
}
async function load(id) {
  if(!id) {state.dataset=null;state.reports={};state.drafts.clear();state.checks.clear();render();return;}
  state.loading=id;
  const result=await json(await fetch(`/api/design-regression/${id}`)).finally(()=>{if(state.loading===id) state.loading=null;});
  if(state.dataset?.id!==id) {
    state.drafts.clear();state.checks.clear();
    if(result.dataset.source_kind==="cross_cases") {
      state.problem=null;state.originalDatasetId=id;input(generate,"source_id").value="";
      input(generate,"generation_mode").value="cross";
      for(const [key,value] of Object.entries(result.dataset.settings)) if(input(generate,key)) input(generate,key).value=value;
      el("design-original-note").textContent="Saved original optimisation setup selected";generationMode();
    }
  }
  state.dataset=result.dataset;state.reports=result.reports;
  const failures=[...result.dataset.cases.filter(item=>item.status==="failed"),...(result.dataset.families || []).filter(item=>item.status==="failed")];
  const issues=[...Object.values(result.errors),...(failures.length ? [`${failures.length} structural evaluations failed. ${failures[0].message}`] : [])];
  error(issues.join(" "));render();selectCase();
}
async function refresh(autoload=false) {
  const [designs,sources,health]=await Promise.all([fetch("/api/design-regression/datasets").then(json),fetch("/api/regression/datasets").then(json),fetch("/api/health").then(json)]);
  state.ready=!!health.design_regression_supported;state.crossSupported=!!health.cross_case_designs_supported;
  if(!state.busy) state.externalBusy=!!health.running;
  const saved=state.dataset?.id || el("design-dataset").value;
  options(el("design-dataset"),[["","New experiment"],...designs.datasets.map(d=>[d.id,`${d.name || d.id.slice(0,8)} · ${d.case_count} cases`])],saved);
  options(input(generate,"source_id"),[["","Select a saved regression dataset"],...sources.datasets.map(d=>[d.id,`${d.id.slice(0,8)} · ${d.completed_cases} designs`])],input(generate,"source_id").value);
  if(autoload && !state.dataset && !state.loading && designs.datasets.length) {el("design-dataset").value=designs.datasets[0].id;await load(designs.datasets[0].id);}
  if(!state.ready) status("Restart TOptVis to enable this experiment");
  controls();
}
async function run(path,payload,kind) {
  error();state.busy=true;state.runId=null;
  if(kind==="fit") state.history=[];
  controls();document.dispatchEvent(new CustomEvent("solveractivity",{detail:{workspace:"design-regression",running:true}}));
  status(kind==="generate" ? "Generating designs…" : kind==="fit" ? "Training…" : "Checking structure…");
  let datasetId=null,terminal=false,cancelled=false,failure="";
  try {
    const response=await fetch(path,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)});
    if(!response.ok) await json(response);
    const reader=response.body.getReader(),decoder=new TextDecoder();let buffer="";
    const receive=line=>{
      if(!line.trim()) return;
      const event=JSON.parse(line);
      if(event.event==="error") throw new Error(event.message);
      if(event.event==="started") {state.runId=event.run_id;datasetId=event.dataset_id;controls();}
      if(event.event==="design_optimization") status(`Optimising case ${event.family+1} / ${event.family_count} · iteration ${event.iteration} / ${event.max_iterations}`);
      if(event.event==="design_case") status(`Evaluated ${event.completed_cases} / ${event.case_count} designs`);
      if(event.event==="neural_training") {state.history.push(event);learning.update(state.history);}
      if(event.event==="design_evaluation") state.checks.set(reference().index,event.evaluation);
      if(event.event==="complete") {terminal=true;cancelled=event.stop_reason==="cancelled";}
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
    status(cancelled ? "Stopped" : "Complete");
  } catch(e) {failure=e.message;error(failure);status("Error");}
  finally {
    state.busy=false;state.runId=null;state.history=[];
    document.dispatchEvent(new CustomEvent("solveractivity",{detail:{workspace:"design-regression",running:false}}));
    if(kind==="fit" && terminal && !cancelled) state.preferredModel=payload.method==="neural" ? "neural_network" : `${payload.method}_${payload.degrees[0]}`;
    try {if(datasetId) await load(datasetId);await refresh();} catch(e) {error(e.message);}
    if(failure) error(failure);
    controls();
  }
}
generate.addEventListener("submit",event=>{
  event.preventDefault();if(state.busy || state.externalBusy || !generate.reportValidity()) return;
  const payload=Object.fromEntries(["variants","amplitude","seed"].map(key=>[key,Number(input(generate,key).value)]));  const cross=input(generate,"generation_mode").value==="cross";
  if(cross && state.problem) payload.problem=state.problem;
  else if(cross && state.originalDatasetId) payload.original_dataset_id=state.originalDatasetId;
  else payload.source_id=input(generate,"source_id").value;
  if(cross) for(const key of ["case_count","max_iterations"]) payload[key]=Number(input(generate,key).value);
  run(`/api/design-regression/${cross ? "cross-cases" : "generate"}`,payload,"generate");
});
fit.addEventListener("submit",event=>{
  event.preventDefault();if(state.busy || state.externalBusy || !state.dataset || !fit.reportValidity()) return;
  const payload={method:input(fit,"method").value,log_target:input(fit,"log_target").checked};
  for(const key of ["seed"]) payload[key]=Number(input(fit,key).value);
  payload.test_fraction=Number(input(fit,"test_percent").value)/100;payload.validation_fraction=Number(input(fit,"validation_percent").value)/100;
  if(payload.method==="neural") for(const key of ["neurons","epochs","learning_rate","weight_penalty","network_seed"]) payload[key]=Number(input(fit,key).value);
  else {
    const degree=Number(input(fit,"max_degree").value);
    if(!Number.isSafeInteger(degree) || degree<1) {error("Choose a positive integer polynomial order");return;}
    try {payload.degrees=Array.from({length:degree},(_,i)=>i+1);} catch {error("Requested order exceeds available memory");return;}
    payload.ridge_alpha=Number(input(fit,"ridge_alpha").value);
  }
  run(`/api/design-regression/${state.dataset.id}/fit`,payload,"fit");
});
el("dataset-legacy-stop").addEventListener("click",()=>el("design-stop").click());
input(fit,"method").addEventListener("change",method);
input(generate,"source_id").addEventListener("change",()=>{state.problem=null;state.originalDatasetId=null;el("design-original-note").textContent="";controls();});
input(generate,"generation_mode").addEventListener("change",generationMode);
el("design-use-current").addEventListener("click",()=>{
  let received=false;
  document.dispatchEvent(new CustomEvent("designregressionuseproblem",{detail:{accept:problem=>{
    received=true;state.problem=problem;state.originalDatasetId=null;input(generate,"source_id").value="";el("design-original-note").textContent="Current optimisation setup selected";error();controls();
  },reject:message=>{received=true;error(message);}}}));
  if(!received) error("The optimisation setup is unavailable; select a saved experiment");
});
el("design-view-setup").addEventListener("change",preview);
el("design-refresh").addEventListener("click",()=>refresh().catch(e=>error(e.message)));
el("design-dataset").addEventListener("change",()=>load(el("design-dataset").value).catch(e=>error(e.message)));
el("design-model").addEventListener("change",render);
el("design-case").addEventListener("change",()=>{selectCase();chart();});
el("design-log-axis").addEventListener("change",chart);
el("design-element").addEventListener("change",selectCase);
el("design-density").addEventListener("click",event=>{
  if(!state.dataset || state.busy) return;
  const index=pickElement(el("design-density"),state.dataset.fixed_config,event);
  if(index!=null) {el("design-element").value=index+1;selectCase();}
});
el("design-edit-form").addEventListener("submit",event=>{
  event.preventDefault();if(state.busy || !density()) return;
  try {state.drafts.set(reference().index,editDensity(density(),state.dataset.fixed_config,Number(el("design-element").value)-1,Number(el("design-value").value)));state.checks.delete(reference().index);error();preview();} catch(e) {error(e.message);}
});
el("design-reset").addEventListener("click",()=>{state.drafts.delete(reference().index);state.checks.delete(reference().index);error();selectCase();});
el("design-evaluate").addEventListener("click",()=>{if(!state.busy && !state.externalBusy && density()) run(`/api/design-regression/${state.dataset.id}/evaluate`,{density:density()},"evaluate");});
el("design-stop").addEventListener("click",async()=>{try {await json(await fetch(`/api/run/${state.runId}/stop`,{method:"POST"}));status("Stopping…");} catch(e) {error(e.message);}});
document.addEventListener("solveractivity",event=>{if(event.detail.workspace!=="design-regression") {state.externalBusy=event.detail.running;controls();}});
document.addEventListener("datasetprepared",event=>load(event.detail).then(()=>refresh()).catch(e=>error(e.message)));
document.addEventListener("mlmethodchange",event=>{if(event.detail==="datasets") refresh().catch(e=>error(e.message));});
document.addEventListener("regressionviewchange",event=>{state.active=event.detail==="design";if(state.active) {render();if(!state.busy) refresh(true).catch(e=>error(e.message));}});
new ResizeObserver(()=>{if(state.active) {chart();preview();}}).observe(el("design-chart").parentElement);
method();generationMode();controls();

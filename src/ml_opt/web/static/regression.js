import {drawTopology} from "./viewer.js";
import {parseProject} from "./project-io.js";
import {createTopologyRegression} from "./topology-regression.js";
import {predictNeural, modelName, mergeReports, selectReport, comparisonNote} from "./regression-models.js";
import {createTrainingChart} from "./network-training.js";

const el = id => document.getElementById(id);
const generateForm = el("reg-generate-form"), fitForm = el("reg-fit-form");
const canvas = el("reg-topology"), chart = el("reg-chart");
const state = {ready:false, busy:false, externalBusy:false, jobId:null, dataset:null, models:null,
  selected:null, liveConfig:null, liveDensity:null, loading:0, problem:null, importing:false,
  importSupported:false, topologySupported:false, topologyModels:null, mode:"compliance", fitOptions:{}, fitIdentity:{},
  preferred:{}, liveHistory:{compliance:[],topology:[]}, trainingTarget:null};
const learningCharts={compliance:createTrainingChart("reg"),topology:createTrainingChart("top")};
const topologyPanel=createTopologyRegression({run,onModelChange:report=>{
  state.topologyModels=report;restoreFitOptions();renderLearning();
}});
const activeReport=()=>state.mode==="topology" ? state.topologyModels : state.models;
const number = value => Math.abs(value)>=1e6 ? Number(value).toExponential(3) : Number(value).toLocaleString("en", {maximumFractionDigits:3});
const percent = value => `${(100*value).toFixed(2)}%`;
const evaluate = (model, volume) => {
  if(model.kind==="neural_network") return predictNeural(model,volume);
  const scaled = model.scaled_coefficients;
  const input = scaled ? (volume-model.input_offset)/model.input_scale : volume;
  return (scaled || model.coefficients).reduceRight((sum, coefficient) => sum*input+coefficient, 0);
};
const degreeOf = item => item.degree ?? item.coefficients.length-1;
const modelLabel = modelName;
const model = () => state.models?.models[el("reg-model").value];
const error = (message = "") => { el("reg-error").textContent = message; el("reg-error").hidden = !message; };
const status = (message, kind = "ready") => { el("reg-status").textContent = message; el("reg-status").dataset.status = kind; };
const problemConfig = () => state.liveConfig || state.dataset?.fixed_config ||
  state.problem?.model;
const objectiveLabel = () => (problemConfig()?.boundary_conditions || []).some(item =>
  item.kind === "displacement" && (((item.components || "xy").includes("x") && item.ux) ||
    ((item.components || "xy").includes("y") && item.uy))) ? "Elastic work" : "Compliance";

function quality(item) {
  if (item.stop_reason === "failed") return "Failed";
  if (item.stop_reason === "cancelled") return "Stopped";
  if (!item.feasible) return "Infeasible";
  return item.converged ? "Converged" : "Iteration limit";
}

function useOf(item) {
  if (state.models?.test_indices.includes(item.index)) return "Test";
  if (state.models?.validation_indices?.includes(item.index)) return "Validation";
  if (state.models?.train_indices.includes(item.index)) return "Training";
  return state.models ? "Excluded" : "Reference";
}

function buttons() {
  const locked = state.busy || state.externalBusy || state.importing;
  el("reg-generate").disabled = !state.ready || locked || !state.problem || !state.importSupported;
  el("reg-fit").disabled = !state.ready || locked || !state.dataset?.cases.length || (state.mode==="topology" && !state.topologySupported);
  el("reg-generate-settings").disabled = state.busy || state.importing;
  el("reg-fit-settings").disabled = state.busy || state.importing;
  ridgeControls();
  el("reg-dataset").disabled = state.busy || state.importing;
  el("reg-refresh").disabled = state.busy || state.importing;
  el("reg-stop").disabled = !state.busy || !state.jobId;
  el("reg-load-problem").disabled = state.busy || state.importing || !state.importSupported;
  el("reg-problem-name").textContent = state.importing ? "Loading problem..." :
    !state.importSupported ? "Restart TOptVis to enable saved-problem imports." : state.problem ?
      state.problem.name : "No setup loaded";
  if (!state.busy && state.externalBusy) status("Solver busy");
  topologyPanel.update({dataset:state.dataset,report:state.topologyModels,locked,
    ready:state.ready && state.topologySupported});
}

function ridgeControls() {
  const method=fitForm.elements.namedItem("method").value, ridge=method==="ridge", neural=method==="neural";
  el("reg-ridge-options").hidden=!ridge;
  fitForm.elements.namedItem("ridge_alpha").disabled=!ridge;
  el("reg-polynomial-options").hidden=neural;
  fitForm.elements.namedItem("max_degree").disabled=neural;
  el("reg-neural-options").hidden=!neural;
  for(const name of ["neurons","epochs","learning_rate","weight_penalty","network_seed","validation_percent"]) fitForm.elements.namedItem(name).disabled=!neural;
}

function restoreFitOptions() {
  const report=activeReport(), options=state.fitOptions[state.mode] || {
    test_percent:25, seed:42, include_unconverged:false, max_degree:2, method:"ols", ridge_alpha:.01,
    neurons:8,epochs:2000,learning_rate:.01,weight_penalty:0,network_seed:42,validation_percent:20};
  const identity=`${state.dataset?.id || ""}/${report?.trained_at || JSON.stringify(report && [report.degrees,report.seed,report.test_fraction,report.method,report.ridge_alpha])}`;
  if(report && identity!==state.fitIdentity[state.mode]) {
    options.test_percent=report.test_fraction*100;options.seed=report.seed;
    options.include_unconverged=report.include_unconverged;
    if(report.method!=="neural") options.max_degree=Math.max(...(report.degrees || Object.values(report.models).filter(item=>item.kind!=="neural_network").map(degreeOf)));
    options.method=report.method || "ols";options.ridge_alpha=report.method==="ridge" ? report.ridge_alpha ?? .01 : .01;
    if(report.method==="neural") {
      for(const name of ["neurons","epochs","learning_rate","weight_penalty","network_seed"]) options[name]=report[name];
      options.validation_percent=report.validation_fraction*100;
    }
  }
  state.fitIdentity[state.mode]=identity;
  state.fitOptions[state.mode]={...options};
  for(const [name,value] of Object.entries(options)) {
    const control=fitForm.elements.namedItem(name);
    if(name==="include_unconverged") control.checked=value;else control.value=value;
  }
  ridgeControls();
}

function rememberFitOptions() {
  state.fitOptions[state.mode]=Object.fromEntries(["test_percent","seed","max_degree","include_unconverged","method","ridge_alpha",
    "neurons","epochs","learning_rate","weight_penalty","network_seed","validation_percent"].map(name=>{
    const control=fitForm.elements.namedItem(name);return [name,name==="include_unconverged" ? control.checked : control.value];
  }));
}

function selectView(view) {
  el("reg-existing-view").hidden=view==="design";
  el("reg-design-view").hidden=view!=="design";
  if(!["compliance","topology"].includes(view) || view===state.mode) return;
  rememberFitOptions();
  state.mode=view;restoreFitOptions();
  el("reg-title").textContent=`Regression · ${view==="topology" ? "Topology" : "Compliance"}`;
  el("reg-compliance-results").hidden=view!=="compliance";
  el("reg-topology-results").hidden=view!=="topology";
  el("reg-compliance-help").hidden=view!=="compliance";
  el("reg-topology-help").hidden=view!=="topology";
  render();
}

function renderLearning() {
  for(const target of ["compliance","topology"]) {
    const report=target==="compliance" ? state.models : state.topologyModels;
    const fitted=report?.method==="neural" ? report.models.neural_network : null;
    const live=state.trainingTarget===target ? state.liveHistory[target] : [];
    learningCharts[target].update(live.length ? live : fitted?.training_history,live.length ? null : fitted);
  }
}

function setBusy(busy) {
  state.busy = busy;
  buttons();
  document.dispatchEvent(new CustomEvent("solveractivity", {detail:{workspace:"regression", running:busy}}));
}

function topology() {
  if (!canvas.getClientRects().length) return;
  const item = state.dataset?.cases.find(item => item.index === state.selected);
  const config = problemConfig();
  const density = state.liveDensity || item?.density;
  el("reg-topology-empty").hidden = !!config;
  if (config) {
    drawTopology(canvas, config, density || null);
    canvas.setAttribute("aria-label", !density ? "Optimisation problem geometry and boundaries" : state.liveConfig ? "Live reference topology during optimisation" :
      `Reference topology at volume ${number(item.volume_fraction)}, ${objectiveLabel().toLowerCase()} ${number(item.compliance)}`);
  } else {
    canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
  }
}

function selectCase(index) {
  state.selected = Number(index);
  state.liveConfig = state.liveDensity = null;
  el("reg-case").value = String(index);
  const item = state.dataset?.cases.find(item => item.index === state.selected);
  if (item) el("reg-case-note").textContent = `Volume ${number(item.volume_fraction)} / ${objectiveLabel()} ${item.compliance ? number(item.compliance) : "unavailable"} / ${quality(item)} / ${useOf(item)}${item.message ? `: ${item.message}` : ""}`;
  topology();
  plot();
}

function domain() {
  const range = [state.dataset?.settings.volume_min ?? .25, state.dataset?.settings.volume_max ?? .65];
  if (state.models) {
    range[0] = Math.min(range[0], state.models.training_range[0]);
    range[1] = Math.max(range[1], state.models.training_range[1]);
  }
  const pad = (range[1]-range[0])*.15;
  return [Math.max(.001, range[0]-pad), Math.min(.999, range[1]+pad)];
}

function prediction() {
  const fitted = model(), volume = Number(el("reg-volume").value);
  el("reg-volume-value").textContent = volume.toFixed(3);
  el("reg-volume").disabled = !fitted;
  if (!fitted) {
    el("reg-prediction").textContent = "Unavailable";
    el("reg-prediction-note").textContent = "Train a model to explore predictions.";
    el("reg-prediction-note").dataset.warning = "false";
    return;
  }
  const value = evaluate(fitted, volume), [low, high] = state.models.training_range;
  if(!Number.isFinite(value)) {
    el("reg-prediction").textContent="Unavailable";
    el("reg-prediction-note").textContent="Prediction exceeds numerical range at this volume.";
    el("reg-prediction-note").dataset.warning="true";
    return;
  }
  const outside = volume < low || volume > high;
  el("reg-prediction").textContent = number(value);
  el("reg-prediction-note").textContent = value <= 0 ? `Nonphysical prediction: ${objectiveLabel().toLowerCase()} must be positive.` :
    outside ? "Extrapolation: outside the training range." : "Within the training range.";
  el("reg-prediction-note").dataset.warning = String(outside || value <= 0);
}

function plot() {
  const objective = objectiveLabel();
  el("reg-prediction-title").textContent = `${objective} prediction`;
  el("reg-prediction-label").textContent = `Predicted ${objective.toLowerCase()}`;
  el("reg-case-objective").textContent = objective;
  chart.setAttribute("aria-label", `${objective} versus permitted volume. Select a reference point to inspect its topology.`);
  const cases = state.dataset?.cases.filter(item => Number.isFinite(item.compliance)) || [];
  el("reg-chart-empty").hidden = cases.length > 0;
  el("reg-legend").hidden = !cases.length;
  if (!cases.length) { chart.replaceChildren(); return; }
  const fitted = model(), [low, high] = domain();
  const width = Math.max(320, chart.clientWidth || 800), height = Math.max(250, chart.clientHeight || 350);
  const right = width-26, base = height-58;
  chart.setAttribute("viewBox", `0 0 ${width} ${height}`);
  const volumes = Array.from({length:201}, (_, i) => low+(high-low)*i/200);
  const values = fitted ? volumes.map(volume => evaluate(fitted, volume)) : [];
  const finite=values.filter(Number.isFinite);
  const magnitude=Math.max(1,...cases.map(item=>Math.abs(item.compliance)),...finite.map(Math.abs));
  const min = Math.min(0, ...finite.map(value=>value/magnitude));
  const max = Math.max(...cases.map(item => item.compliance/magnitude), ...finite.map(value=>value/magnitude), 1/magnitude);
  const span = max-min || 1, bottom = min-.04*span, top = max+.08*span;
  const x = volume => 85+(volume-low)/(high-low)*(right-85);
  const y = value => base-(value/magnitude-bottom)/(top-bottom)*(base-40);
  let svg = `<title>Reference ${objective.toLowerCase()} and regression prediction versus permitted volume</title>`;
  if (fitted) {
    const [start, end] = state.models.training_range;
    svg += `<rect x="85" y="30" width="${Math.max(0,x(start)-85)}" height="${base-30}" fill="#fff4e8"/><rect x="${x(end)}" y="30" width="${Math.max(0,right-x(end))}" height="${base-30}" fill="#fff4e8"/>`;
  }
  for (let i=0; i<=4; i++) {
    const value = (min+(max-min)*i/4)*magnitude, volume = low+(high-low)*i/4;
    svg += `<path d="M85 ${y(value)}H${right}" stroke="#e4e9ed"/><text x="73" y="${y(value)+5}" text-anchor="end" fill="#606e7b">${number(value)}</text><text x="${x(volume)}" y="${base+23}" text-anchor="middle" fill="#606e7b">${volume.toFixed(2)}</text>`;
  }
  svg += `<path d="M85 30V${base}H${right}" fill="none" stroke="#9cabb7"/><text x="${(85+right)/2}" y="${height-8}" text-anchor="middle" fill="#606e7b">Permitted volume fraction</text><text x="85" y="20" fill="#606e7b">${objective}</text>`;
  if (fitted) {
    const [start, end] = state.models.training_range;
    for (const inside of [false, true]) {
      let d = "", drawing = false;
      volumes.forEach((volume, i) => {
        if (Number.isFinite(values[i]) && (volume >= start && volume <= end) === inside) {
          d += `${drawing ? "L" : "M"}${x(volume)},${y(values[i])}`;
          drawing = true;
        } else drawing = false;
      });
      svg += `<path d="${d}" fill="none" stroke="#7961b2" stroke-width="2.5" ${inside ? "" : 'stroke-dasharray="6 5"'}/>`;
    }
    const volume = Number(el("reg-volume").value), value = evaluate(fitted, volume);
    if(Number.isFinite(value)) svg += `<path d="M${x(volume)} 30V${base}" stroke="#7961b2" stroke-dasharray="3 4" opacity=".5"/><circle cx="${x(volume)}" cy="${y(value)}" r="4" fill="#7961b2"/>`;
  }
  cases.forEach(item => {
    const use = useOf(item), test = use === "Test", excluded = use === "Excluded";
    const color = test ? "#df7a40" : use==="Validation" ? "#4b7eaf" : excluded ? "#8b929a" : "#196964";
    const attrs = `class="reg-chart-point" data-case="${item.index}" tabindex="0" role="button" aria-label="Volume ${number(item.volume_fraction)}, ${objective.toLowerCase()} ${number(item.compliance)}, ${use}, ${quality(item)}" fill="${excluded ? "white" : color}" stroke="${state.selected === item.index ? "#152c40" : color}"`;
    const cx=x(item.volume_fraction), cy=y(item.compliance), radius=state.selected === item.index ? 7 : 5;
    svg += test ? `<rect x="${cx-radius}" y="${cy-radius}" width="${2*radius}" height="${2*radius}" ${attrs}/>` : `<circle cx="${cx}" cy="${cy}" r="${radius}" ${attrs}/>`;
  });
  // All inserted values above are numbers or fixed labels, never user-provided HTML.
  chart.innerHTML = svg;
  el("reg-train-label").textContent = fitted ? "Training" : "Reference";
  el("reg-test-legend").hidden = !fitted;
  el("reg-validation-legend").hidden = !state.models?.validation_indices?.length;
  el("reg-curve-legend").hidden = !fitted;
  el("reg-excluded-legend").hidden = !state.models?.excluded_indices.length;
}

function scoreRow(label, training, test, active=false, validation="—") {
  const row=document.createElement("tr"); row.dataset.active=String(active);
  for (const value of [label, training, validation, test]) { const td=document.createElement("td"); td.textContent=value; row.append(td); }
  return row;
}

function renderModels() {
  const select=el("reg-model"), previous=state.models?.selected_model || select.value;
  state.models=selectReport(state.models,previous);
  const report=state.models;
  select.replaceChildren();
  for (const [name, item] of Object.entries(report?.models || {})) select.add(new Option(modelLabel(item), name));
  select.value = report?.models[previous] ? previous : Object.keys(report?.models || {})[0] || "";
  select.disabled = !report;
  const fitted=model();
  const scores=el("reg-scores"); scores.replaceChildren();
  el("reg-coefficients").replaceChildren();
  el("reg-training-time").textContent = fitted ? `${(fitted.training_seconds*1000).toPrecision(3)} ms` : "Unavailable";
  el("reg-prediction-time").textContent = fitted ? `${(fitted.prediction_seconds*1000).toPrecision(3)} ms` : "Unavailable";
  if (report) {
    const count=report.unfinished_indices.length;
    el("reg-fit-note").textContent = `${report.train_indices.length} training${report.validation_indices ? ` / ${report.validation_indices.length} validation` : ""} / ${report.test_indices.length} test / ${report.excluded_indices.length} excluded. Split seed ${report.seed}.${count ? ` Includes ${count} feasible unfinished runs.` : ""}${comparisonNote(report)}`;
    scores.append(scoreRow("Training mean", "-", percent(report.baseline.test.mean_relative_error)));
    for (const [name, item] of Object.entries(report.models)) scores.append(scoreRow(modelLabel(item), percent(item.train.mean_relative_error), percent(item.test.mean_relative_error), name === el("reg-model").value,item.validation ? percent(item.validation.mean_relative_error) : "—"));
    (fitted.coefficients || fitted.scaled_coefficients || []).forEach((value,index) => {
      const div=document.createElement("div"), dt=document.createElement("dt"), dd=document.createElement("dd");
      const variable=fitted.coefficients ? "Volume" : "Scaled volume";
      dt.textContent=index===0 ? "Intercept (a)" : index===1 ? `${variable} (b)` : index===2 ? `${variable} squared (c)` : `${variable} power ${index}`; dd.textContent=number(value);
      div.append(dt,dd); el("reg-coefficients").append(div);
    });
  } else {
    scores.append(scoreRow("No fitted models", "-", "-"));
    el("reg-fit-note").textContent="At least six eligible cases are needed. Unfinished and infeasible runs are excluded by default.";
  }
  prediction(); plot();renderLearning();
}

function renderCases() {
  const data=state.dataset, cases=data?.cases || [];
  el("reg-case").replaceChildren(); el("reg-case-rows").replaceChildren();
  for (const item of cases) {
    const option=document.createElement("option"); option.value=item.index; option.textContent=number(item.volume_fraction); el("reg-case").append(option);
    const row=document.createElement("tr"), cell=document.createElement("td"), button=document.createElement("button");
    button.type="button"; button.textContent=number(item.volume_fraction); button.addEventListener("click",()=>selectCase(item.index)); cell.append(button); row.append(cell);
    for (const value of [item.compliance ? number(item.compliance) : "Unavailable", quality(item), useOf(item)]) {
      const td=document.createElement("td"); td.textContent=value; if(item.message) td.title=item.message; row.append(td);
    }
    el("reg-case-rows").append(row);
  }
  el("reg-case").disabled=!cases.length;
  el("reg-case-count").textContent=`${cases.length}${data ? ` / ${data.settings.case_count}` : ""}`;
  if (cases.length) {
    if (!cases.some(item=>item.index === state.selected)) state.selected=cases[0].index;
    selectCase(state.selected);
  } else {
    state.selected=null;
    el("reg-case-note").textContent="";
    topology();
  }
}

function render() {
  const data=state.dataset;
  el("reg-downloads").hidden=!data;
  el("reg-download-models").hidden=!state.models;
  if (data) {
    const good=data.cases.filter(item=>item.converged && item.feasible).length;
    el("reg-dataset-note").textContent=`${data.cases.length}/${data.settings.case_count} cases saved. ${good} converged and feasible. ${data.status.replaceAll("_"," ")}.`;
    el("reg-generation-time").textContent=data.status === "generating" ? "In progress" : `${number(data.generation_seconds)} s`;
    el("reg-download-data").href=`/api/regression/${data.id}/artifacts/dataset`;
    el("reg-download-models").href=`/api/regression/${data.id}/artifacts/${state.models?.method==="neural" ? "neural-compliance" : "models"}`;
    const [low,high]=domain(); el("reg-volume").min=low; el("reg-volume").max=high;
  } else {
    el("reg-dataset-note").textContent=""; el("reg-generation-time").textContent="Unavailable";
  }
  renderModels(); renderCases(); buttons();
}

async function json(response) {
  const data=await response.json();
  if(!response.ok) throw new Error(Array.isArray(data.detail) ? data.detail.map(item=>item.msg).join("; ") : data.detail || "Request failed");
  return data;
}

async function load(id) {
  const token=++state.loading;
  if (!id) { state.dataset=state.models=state.topologyModels=null; state.liveConfig=state.liveDensity=null; render(); status(state.problem ? "Setup loaded" : "Ready"); return; }
  const result=await json(await fetch(`/api/regression/${id}`));
  if(token!==state.loading) return;
  const same=state.dataset?.id===result.dataset.id;
  state.dataset=result.dataset;
  state.models=mergeReports(result.models,result.neural_models?.compliance,state.preferred.compliance || (same ? el("reg-model").value : ""));
  state.topologyModels=mergeReports(result.topology_models,result.neural_models?.topology,state.preferred.topology || (same ? el("top-model").value : ""));
  state.preferred={};state.liveConfig=state.liveDensity=null;
  if(result.neural_errors?.[state.mode]) error(result.neural_errors[state.mode]);
  if(result.topology_model_error && state.mode==="topology") error(result.topology_model_error);
  state.problem=state.dataset.settings.source_problem ? {
    name:state.dataset.settings.problem_name || "Saved problem",model:state.dataset.settings.source_problem} : null;
  for(const [name,value] of Object.entries(state.dataset.settings)) {
    const control=generateForm.elements.namedItem(name);
    if(control) control.value=value;
  }
  restoreFitOptions();
  render(); status(activeReport() ? "Models ready" : state.dataset.status === "complete" ? "Dataset ready" : "Partial dataset");
}

async function refresh(selected=el("reg-dataset").value) {
  const token=++state.loading;
  const result=await json(await fetch("/api/regression/datasets"));
  if(token!==state.loading || state.busy || state.importing) return;
  const select=el("reg-dataset"); select.replaceChildren(new Option("New experiment", ""));
  for (const record of result.datasets) select.add(new Option(`${record.created_at.slice(0,16).replace("T"," ")} / ${record.completed_cases} cases / ${record.id.slice(0,6)}`,record.id));
  select.value=result.datasets.some(record=>record.id===selected) ? selected : "";
  await load(select.value);
}

async function availability(refreshData=false) {
  try {
    const health=await json(await fetch("/api/health"));
    state.ready=!!health.regression_supported;
    state.importSupported=!!health.regression_problem_import_supported;
    state.topologySupported=!!health.topology_regression_supported;
    if(!state.busy) state.externalBusy=health.running;
    if(!state.ready) status("Restart TOptVis to enable regression");
    else if(refreshData && !state.busy && !state.importing) await refresh();
  } catch (e) {state.ready=false; error(e.message);}
  buttons();
}

function receive(event) {
  if(event.event === "error") throw new Error(event.message);
  if(event.event === "started") {state.jobId=event.run_id; buttons();}
  if(event.event === "topology_evaluation") topologyPanel.acceptEvaluation(event.evaluation);
  if(event.event === "neural_training") {
    state.liveHistory[event.target].push(event);renderLearning();
    status(`Training network · epoch ${event.epoch}/${event.epochs}`,"running");
  }
  if(event.event === "regression_case_started") {
    state.liveConfig=event.config; state.liveDensity=null;
    state.dataset.fixed_config ??= event.config;
    status(`Case ${event.case_index+1}/${event.case_count}`,"running");
  }
  if(event.event === "regression_iteration") {
    state.liveDensity=event.density; topology();
    el("reg-case-note").textContent=`Case ${event.case_index+1} / Iteration ${event.iteration} / ${objectiveLabel()} ${number(event.compliance)}`;
  }
  if(event.event === "regression_case") {
    state.dataset.cases.push(event.case); state.selected=event.case.index;
    render();
  }
}

async function run(endpoint, payload, kind) {
  if(state.busy || state.externalBusy || state.importing || !state.ready) return;
  state.trainingTarget=kind==="fit" && payload.method==="neural" ? state.mode : null;
  if(state.trainingTarget) {state.liveHistory[state.trainingTarget]=[];state.preferred[state.trainingTarget]="neural_network";}
  else if(kind==="fit" && activeReport()?.method==="neural") state.preferred[state.mode]="linear";
  ++state.loading;
  error(); state.jobId=null; setBusy(true); status(kind === "generate" ? "Generating dataset" : kind === "evaluate" ? "Checking structure" : "Training models","running");
  let completedId=null, terminal=false, cancelled=false;
  if(kind === "generate") {
    state.models=state.topologyModels=null; state.selected=null; state.liveConfig=state.liveDensity=null;
    state.dataset={id:null,cases:[],settings:payload,generation_seconds:0,status:"generating"};
    render(); el("reg-downloads").hidden=true;
  }
  try {
    const response=await fetch(endpoint,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)});
    if(!response.ok) await json(response);
    const reader=response.body.getReader(), decoder=new TextDecoder(); let buffer="";
    const consume = line => {
      if(!line.trim()) return;
      const event=JSON.parse(line);
      if(event.event === "started") {
        completedId=event.dataset_id;
        if(kind === "generate") state.dataset.id=completedId;
      }
      if(event.event === "complete") {terminal=true; cancelled=event.stop_reason === "cancelled";}
      receive(event);
    };
    try {
      while(true) {
        const {value,done}=await reader.read(); buffer+=decoder.decode(value || new Uint8Array(),{stream:!done});
        let index; while((index=buffer.indexOf("\n"))>=0) {consume(buffer.slice(0,index)); buffer=buffer.slice(index+1);}
        if(done) break;
      }
      if(buffer.trim()) consume(buffer);
      if(!terminal) throw new Error("The connection closed before the job completed. Saved cases are retained.");
    } finally {await reader.cancel();}
  } catch(e) {error(e.message); status("Error");}
  finally {
    state.jobId=null;state.trainingTarget=null; setBusy(false);
    try {await refresh(completedId || el("reg-dataset").value);} catch(e) {error(e.message);}
    await availability();
    if(terminal) status(cancelled ? "Stopped / saved progress" : kind === "generate" ? "Dataset ready" : kind === "evaluate" ? "Structure checked" : "Models ready");
  }
}

generateForm.addEventListener("submit",event=>{
  event.preventDefault();
  const settings=Object.fromEntries(Array.from(new FormData(generateForm),([name,value])=>[name,name === "material_model" ? value : Number(value)]));
  if(settings.volume_min >= settings.volume_max) {error("Minimum volume must be smaller than maximum volume.");return;}
  if(!state.problem) {error("Load an optimisation setup before generating cases.");return;}
  settings.problem=state.problem.model;
  settings.problem_name=state.problem.name;
  run("/api/regression/generate",settings,"generate");
});
el("reg-load-problem").addEventListener("click",()=>{
  if(!state.busy && !state.importing) el("reg-problem-file").click();
});
el("reg-problem-file").addEventListener("change",async()=>{
  const file=el("reg-problem-file").files[0]; el("reg-problem-file").value="";
  if(!file || state.busy || state.importing) return;
  state.importing=true; ++state.loading; error(); buttons();
  try {
    if(file.size>64*1024*1024) throw new Error("The project exceeds the 64 MiB import limit.");
    const project=parseProject(await file.text());
    const result=await json(await fetch("/api/project/validate",{method:"POST",
      headers:{"Content-Type":"application/json"},body:JSON.stringify(project.model)}));
    state.problem={name:file.name,model:result.model};
    el("reg-dataset").value=""; ++state.loading;
    state.dataset=state.models=state.topologyModels=null; state.selected=null;
    state.liveConfig=result.model; state.liveDensity=null;
    render(); status("Setup loaded");
  } catch(e) {error(e.message || "Could not load the optimisation problem.");}
  finally {state.importing=false;buttons();}
});
generateForm.addEventListener("invalid",event=>{const details=event.target.closest("details");if(details) details.open=true;},true);
fitForm.addEventListener("submit",event=>{
  event.preventDefault(); if(!state.dataset?.id) return;
  const method=fitForm.elements.namedItem("method").value;
  const common={test_fraction:Number(fitForm.elements.namedItem("test_percent").value)/100,
    seed:Number(fitForm.elements.namedItem("seed").value),include_unconverged:fitForm.elements.namedItem("include_unconverged").checked};
  if(method==="neural") {
    const options=Object.fromEntries(["neurons","epochs","learning_rate","weight_penalty","network_seed"].map(name=>[name,Number(fitForm.elements.namedItem(name).value)]));
    options.validation_fraction=Number(fitForm.elements.namedItem("validation_percent").value)/100;
    if(!Number.isSafeInteger(options.neurons) || options.neurons<1 || !Number.isSafeInteger(options.epochs) || options.epochs<1 ||
       !Number.isFinite(options.learning_rate) || options.learning_rate<=0 || !Number.isFinite(options.weight_penalty) || options.weight_penalty<0) {
      error("Choose positive integer neurons and epochs, a positive learning rate, and a nonnegative weight penalty.");return;
    }
    run(`/api/regression/${state.dataset.id}/${state.mode==="topology" ? "topology/fit" : "fit"}`,{...common,method,...options},"fit");return;
  }
  const maximum=Number(fitForm.elements.namedItem("max_degree").value);
  if(!Number.isSafeInteger(maximum) || maximum<1) {error("Choose a positive integer polynomial order.");return;}
  const ridgeAlpha=Number(fitForm.elements.namedItem("ridge_alpha").value);
  if(method==="ridge" && (!Number.isFinite(ridgeAlpha) || ridgeAlpha<0)) {error("Ridge penalty must be finite and nonnegative.");return;}
  run(`/api/regression/${state.dataset.id}/${state.mode==="topology" ? "topology/fit" : "fit"}`,{test_fraction:Number(fitForm.elements.namedItem("test_percent").value)/100,
    seed:Number(fitForm.elements.namedItem("seed").value),include_unconverged:fitForm.elements.namedItem("include_unconverged").checked,
    method,ridge_alpha:method==="ridge" ? ridgeAlpha : .01,
    degrees:Array.from({length:maximum},(_,index)=>index+1)},"fit");
});
fitForm.addEventListener("input",rememberFitOptions);
fitForm.addEventListener("change",()=>{rememberFitOptions();ridgeControls();});
el("reg-model").addEventListener("change",()=>{
  state.models=selectReport(state.models,el("reg-model").value);restoreFitOptions();render();
});
el("reg-volume").addEventListener("input",()=>{prediction(); plot();});
el("reg-case").addEventListener("change",event=>selectCase(event.target.value));
el("reg-dataset").addEventListener("change",()=>{error();load(el("reg-dataset").value).catch(e=>error(e.message));});
el("reg-refresh").addEventListener("click",()=>{error();availability(true);});
el("reg-stop").addEventListener("click",async()=>{
  if(!state.jobId) return; el("reg-stop").disabled=true;
  try {await json(await fetch(`/api/run/${state.jobId}/stop`,{method:"POST"}));status("Stopping","running");}
  catch(e) {error(e.message);buttons();}
});
chart.addEventListener("click",event=>{const point=event.target.closest("[data-case]");if(point) selectCase(point.dataset.case);});
chart.addEventListener("keydown",event=>{
  const point=event.target.closest("[data-case]"); if(!point) return;
  if(["Enter"," ","ArrowLeft","ArrowRight","Home","End"].includes(event.key)) {
    event.preventDefault();
    const points=Array.from(chart.querySelectorAll("[data-case]")), index=points.indexOf(point);
    const next=event.key === "Home" ? 0 : event.key === "End" ? points.length-1 : event.key === "ArrowLeft" ? Math.max(0,index-1) : event.key === "ArrowRight" ? Math.min(points.length-1,index+1) : index;
    const id=points[next].dataset.case;selectCase(id);chart.querySelector(`[data-case="${id}"]`).focus();
  }
});
document.addEventListener("solveractivity",event=>{
  if(event.detail.workspace!=="regression") {state.externalBusy=event.detail.running;buttons();}
});
document.addEventListener("mlmethodchange",event=>{
  if(event.detail === "regression") {topology();if(!state.busy) availability(true);}
});
document.addEventListener("regressionviewchange",event=>selectView(event.detail));
new ResizeObserver(()=>{topology();plot();}).observe(chart.parentElement);
new ResizeObserver(topology).observe(canvas.parentElement);
render();availability(true);

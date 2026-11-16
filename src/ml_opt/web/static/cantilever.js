import {drawModel, drawTopology} from "/static/viewer.js";
import {ConvergencePlots} from "/static/convergence.js?v=individual-log-scales";
import {RomChoice} from "/static/rom-choice.js";
import {BoundaryEditor} from "/static/boundary-editor.js";
import {ProjectIO} from "/static/project-io.js";
import {initialiseReferences} from "/static/method-references.js";

const byId = id => document.getElementById(id);
const form = byId("model-form"), modelCanvas = byId("model-canvas"), topologyCanvas = byId("topology-canvas");
const state = {model: null, density: null, fields: null, fieldRow: null, running: false, ready: false, externalBusy: false, runId: null, initial: null,
  hover: null, history: [], runConfig: null, activePanel: "panel-geometry", romSupported: false};
let boundaryEditor = null;
let projectIO = null;
const romChoice = new RomChoice({onChange: () => updateModel(), onStateChange: updateRunAvailability, isRunning: () => state.running,
  getModel: () => state.model || readModel()});
const convergencePlots = new ConvergencePlots();
let convergenceFrame = null;
const format = value => new Intl.NumberFormat("en", {maximumFractionDigits: 2}).format(value);
const filterFormat = value => new Intl.NumberFormat("en", {maximumSignificantDigits: 4}).format(value);

const volumeRelation = mode => ({lt: "<", gt: ">", eq: "="})[mode || "lt"];

function readModel() {
  const model = {};
  for (const [key, value] of new FormData(form)) {
    model[key] = key === "robust" ? value === "true" :
      ["optimizer", "field_execution", "state_solver", "linear_solver", "rom_training_id", "material_model", "projection_schedule", "volume_constraint"].includes(key) ? value : Number(value);
  }
  if (model.state_solver !== "rom") delete model.rom_training_id;
  if (boundaryEditor?.conditions !== null && boundaryEditor?.conditions !== undefined) model.boundary_conditions = structuredClone(boundaryEditor.conditions);
  if (boundaryEditor) { model.load_cases = structuredClone(boundaryEditor.cases); model.load_aggregation = boundaryEditor.aggregation; model.non_design_regions = structuredClone(boundaryEditor.regions); }
  return model;
}

function updateRunAvailability() {
  const boundaryIssue = boundaryEditor?.validationError || "";
  const incompatibleModel = byId("state_solver").value === "rom" && (!state.romSupported || !romChoice.allowed);
  byId("run").disabled = state.running || !state.ready || state.externalBusy || !!boundaryIssue || incompatibleModel;
  byId("run").title = boundaryIssue || (incompatibleModel ? "Choose a reduced model matching the mesh and supports, or use Full model." : "");
  if (!state.running) byId("run-label").textContent = "Run optimisation";
}

function updateSolverControls() {
  byId("fom-linear-options").hidden = state.model?.state_solver === "rom";
}

const configTabs = Array.from(document.querySelectorAll('#workspace-cantilever .config-tabs [role="tab"]'));
function selectPanel(panelId, focus = false) {
  state.activePanel = panelId;
  for (const tab of configTabs) {
    const selected = tab.dataset.panel === panelId;
    tab.setAttribute("aria-selected", String(selected));
    tab.tabIndex = selected ? 0 : -1;
    byId(tab.dataset.panel).hidden = !selected;
    if (selected) {
      byId("config-title").textContent = tab.dataset.title;
      if (focus) tab.focus();
    }
  }
  const charts = panelId === "panel-visualisation";
  byId("model-view").hidden = charts;
  byId("convergence-view").hidden = !charts;
  redraw();
}
for (const [index, tab] of configTabs.entries()) {
  tab.addEventListener("click", () => selectPanel(tab.dataset.panel));
  tab.addEventListener("keydown", event => {
    let next;
    if (event.key === "ArrowDown" || event.key === "ArrowRight") next = (index + 1) % configTabs.length;
    if (event.key === "ArrowUp" || event.key === "ArrowLeft") next = (index + configTabs.length - 1) % configTabs.length;
    if (event.key === "Home") next = 0;
    if (event.key === "End") next = configTabs.length - 1;
    if (next !== undefined) { event.preventDefault(); selectPanel(configTabs[next].dataset.panel, true); }
  });
}
form.addEventListener("invalid", event => {
  const invalid = form.querySelector("input:invalid, select:invalid") || event.target;
  const panel = invalid.closest('[role="tabpanel"]');
  if (panel) selectPanel(panel.id);
}, true);

function showError(message = "") {
  byId("error-message").textContent = message;
  byId("error").hidden = !message;
}
byId("dismiss-error").addEventListener("click", () => showError());

function resultMessage(message = "") {
  byId("topology-description").textContent = message;
  byId("topology-description").title = message;
}

function status(label, kind) {
  for (const id of ["run-status", "convergence-status"]) {
    const badge = byId(id);
    badge.dataset.status = kind;
    badge.replaceChildren(document.createElement("i"), document.createTextNode(label));
  }
}

function updateConvergence() {
  const latest = state.history.at(-1), config = state.runConfig || state.model;
  byId("history-iteration").textContent = latest ? String(latest.iteration) : "—";
  byId("history-compliance").textContent = latest ? format(latest.compliance) : "—";
  byId("history-volume").textContent = latest ? `${(100*latest.volume_fraction).toFixed(2)}%` : "—";
  byId("history-elapsed").textContent = Number.isFinite(latest?.elapsed_seconds) ? `${filterFormat(latest.elapsed_seconds)} s` : "—";
  byId("history-max-size-summary").hidden = !(config.max_solid_size > 0);
  byId("history-max-size").textContent = Number.isFinite(latest?.max_size_constraint) ? filterFormat(latest.max_size_constraint) : "—";
  byId("history-max-void-summary").hidden = !(config.max_void_size > 0);
  byId("history-max-void").textContent = Number.isFinite(latest?.max_void_constraint) ? filterFormat(latest.max_void_constraint) : "—";
  byId("volume-reference").textContent = `Volume ${volumeRelation(config.volume_constraint)} ${filterFormat(100*config.volume_fraction)}%`;
  byId("design-reference").textContent = Number.isFinite(config.design_tolerance) ?
    `Tolerance ${filterFormat(config.design_tolerance)}` : "Stopping tolerance";
  byId("objective-reference").textContent = Number.isFinite(config.objective_tolerance) ?
    `Tolerance ${filterFormat(config.objective_tolerance)}` : "Stopping tolerance";
  if (byId("workspace-cantilever").hidden || byId("convergence-view").hidden || convergenceFrame !== null) return;
  convergenceFrame = requestAnimationFrame(() => {
    convergenceFrame = null;
    if (!byId("workspace-cantilever").hidden && !byId("convergence-view").hidden) convergencePlots.render(state.history, state.runConfig || state.model);
  });
}

function redraw() {
  if (!state.model || byId("workspace-cantilever").hidden) return;
  if (!byId("model-view").hidden) {
    const viewModel = {...state.model, display_load_case: boundaryEditor?.activeCase};
    drawModel(modelCanvas, viewModel, null, boundaryEditor?.overlay);
    drawTopology(topologyCanvas, viewModel, state.density);
  }
  updateConvergence();
}

function syncControls() {
  const gcmma = byId("optimizer").value === "gcmma";
  byId("gcmma-options").hidden = !gcmma;
  byId("max_inner_iterations").disabled = !gcmma;
  const size = Number(byId("max_solid_size").value);
  const spacing = Math.max(Number(byId("length").value)/Number(byId("nx").value), Number(byId("height").value)/Number(byId("ny").value));
  byId("max_solid_size").setCustomValidity(size > 0 && size <= 2*spacing ? "Maximum diameter must exceed two element spacings in both directions. Increase the diameter or refine the mesh." : "");
  byId("max-size-options").hidden = !(size > 0);
  const voidSize = Number(byId("max_void_size").value);
  byId("max_void_size").setCustomValidity(voidSize > 0 && voidSize <= 2*spacing ? "Maximum void diameter must exceed two element spacings in both directions. Increase the diameter or refine the mesh." : "");
  byId("max-void-options").hidden = !(voidSize > 0);
  const ramp = byId("material_model").value === "ramp";
  byId("penalisation").min = ramp ? "0" : "1";
  byId("penalty-label").textContent = ramp ? "Penalty parameter" : "Penalty exponent";
  const scheduled = byId("projection_schedule").value === "geometric";
  byId("beta-schedule-options").hidden = !scheduled;
  const beta = Number(byId("projection_beta").value), maximum = Number(byId("projection_beta_max").value);
  byId("projection_beta").setCustomValidity(scheduled && beta <= 0 ? "Scheduled projection needs a positive starting β." : "");
  byId("projection_beta_max").setCustomValidity(maximum <= 0 || (scheduled && maximum < beta) ? "Maximum β must be positive and at least the starting β." : "");
  byId("projection_beta_multiplier").setCustomValidity(Number(byId("projection_beta_multiplier").value) <= 1 ? "The β multiplier must be greater than 1." : "");
}

function updateMaxSizeStatus() {
  const row = state.history.at(-1), m = state.model;
  byId("max-size-status").hidden = !(m.max_solid_size > 0 && Number.isFinite(row?.max_size_constraint));
  byId("max-size-status").textContent = Number.isFinite(row?.max_size_constraint) ?
    `Solid size: ${row.max_size_constraint <= (state.runConfig?.feasibility_tolerance ?? 1e-4) ? "satisfied" : "exceeded"}` : "";
  byId("max-void-status").hidden = !(m.max_void_size > 0 && Number.isFinite(row?.max_void_constraint));
  byId("max-void-status").textContent = Number.isFinite(row?.max_void_constraint) ?
    `Void size: ${row.max_void_constraint <= (state.runConfig?.feasibility_tolerance ?? 1e-4) ? "satisfied" : "exceeded"}` : "";
}

function updateLabels(resizeFilterRange = true) {
  const m = state.model;
  byId("fixed-solid-legend").hidden = !(m.non_design_regions || []).some(item => item.kind === "solid");
  byId("fixed-void-legend").hidden = !(m.non_design_regions || []).some(item => item.kind === "void");
  byId("volume-label").textContent = `${Math.round(m.volume_fraction * 100)}%`;
  byId("volume-target").textContent = `Volume ${volumeRelation(m.volume_constraint)} ${Math.round(m.volume_fraction * 100)}%`;
  const radius = m.filter_radius;
  const slider = byId("filter-slider");
  const sliderMax = resizeFilterRange ? Math.max(Math.min(m.length, m.height) / 2, radius) : Number(slider.max);
  slider.max = sliderMax;
  slider.step = "any";
  slider.value = radius;
  const beta = m.projection_beta;
  const projectionSlider = byId("projection-slider");
  const projectionMax = resizeFilterRange ? Math.max(16, beta) : Number(projectionSlider.max);
  projectionSlider.max = projectionMax;
  projectionSlider.value = beta;
  byId("robust-options").hidden = !m.robust;
  byId("solver-field-options").hidden = !m.robust;
  byId("field-view").hidden = !m.robust;
  const imposed = (m.boundary_conditions || []).some(item => item.kind === "displacement" &&
    (((item.components || "xy").includes("x") && item.ux) || ((item.components || "xy").includes("y") && item.uy)));
  byId("compliance-label").textContent = imposed ? "Elastic work" : m.robust ? "Worst compliance" : "Compliance";
  byId("objective-title").textContent = imposed ? "Minimise elastic work" : "Minimise compliance";
  byId("compliance-chart-title").textContent = imposed ? "Elastic work" : "Compliance";
  byId("objective-description").textContent = imposed ? "Minimise total elastic work, including prescribed-motion reactions. Under imposed motion this can favour a softer structure." : m.robust ?
    "Maximise stiffness across the eroded, nominal and dilated designs." :
    "Maximise stiffness under the applied load.";
  byId("material-volume-label").textContent = m.robust ? "Dilated volume" : "Material volume";
  byId("filter-legend").hidden = radius === 0;
  updateMaxSizeStatus();
}

function clearResult() {
  state.density = null; state.fields = null; state.fieldRow = null; state.initial = null;
  byId("field-metrics").textContent = "";
  state.history = []; state.runConfig = null;
  boundaryEditor?.setResults(null);
  updateMaxSizeStatus();
  byId("empty-state").hidden = false;
  byId("compliance").textContent = "—";
  byId("improvement").textContent = "";
  byId("improvement").className = "";
  byId("current-volume").textContent = "—";
  byId("iteration").innerHTML = `—<em> / ${state.model.max_iterations}</em>`;
  for (const id of ["compliance", "current-volume", "iteration", "improvement"]) byId(id).title = byId(id).textContent;
  byId("progress-bar").style.width = "0%";
  resultMessage();
  status("Ready", "ready");
  updateSolverControls();
  updateConvergence();
}

function selectField(redrawView = true) {
  const name = byId("display-field").value;
  if (state.fields) {
    state.density = state.fields[name];
    const row = state.fieldRow;
    byId("field-metrics").textContent = `Compliance ${format(row.field_compliances[name])} · volume ${(100 * row.field_volumes[name]).toFixed(2)}%`;
    byId("topology-canvas").setAttribute("aria-label", `${name} material density with Helmholtz length reference`);
  } else {
    byId("topology-canvas").setAttribute("aria-label", "Physical material density with Helmholtz length reference");
  }
  if (redrawView) redraw();
}

function updateModel(event) {
  if (state.running) return;
  syncControls();
  if (event?.target.id === "nx" || event?.target.id === "ny") {
    const nx = Number(byId("nx").value), ny = Number(byId("ny").value);
    if (Number.isInteger(nx) && nx >= 1 && Number.isInteger(ny) && ny >= 1) {
      byId("load_node_x").max = nx;
      byId("load_node_y").max = ny;
      byId("load_node_x").value = Math.max(1, Math.round(state.model.load_node_x / state.model.nx * nx));
      byId("load_node_y").value = Math.round(state.model.load_node_y / state.model.ny * ny);
    }
  }
  if (!form.checkValidity()) return;
  state.model = readModel(); state.hover = null;
  if (boundaryEditor) { boundaryEditor.render(); showError(boundaryEditor.validationError); }
  else showError();
  romChoice.sync();
  if (state.model.state_solver === "rom" && !romChoice.allowed && !boundaryEditor?.validationError) {
    showError("Choose a reduced model matching this mesh and supports, or use Full model in Solver.");
  }
  updateLabels(!["filter-slider", "projection-slider"].includes(event?.target.id)); clearResult(); redraw();
}

function setRunning(running) {
  state.running = running;
  byId("settings").disabled = running;
  updateRunAvailability();
  byId("run-label").textContent = running ? "Optimising..." : "Run optimisation";
  byId("stop").disabled = !running || !state.runId;
  modelCanvas.style.cursor = running ? "default" : "crosshair";
  boundaryEditor?.render();
  projectIO?.setRunning(running);
  romChoice.sync();
  document.dispatchEvent(new CustomEvent("solveractivity", {detail: {workspace: "cantilever", running}}));
}

function receive(event) {
  if (event.event === "started") {
    state.runId = event.run_id;
    state.runConfig = event.config;
    updateSolverControls();
    updateConvergence();
    byId("stop").disabled = false;
    return;
  }
  if (event.event === "iteration") {
    const row = event.history;
    state.history.push(row);
    updateMaxSizeStatus();
    updateSolverControls();
    if (state.initial === null) state.initial = row.compliance;
    state.fields = event.field_densities || null;
    state.fieldRow = row;
    state.density = event.density;
    selectField(false);
    byId("empty-state").hidden = true;
    byId("compliance").textContent = format(row.compliance);
    const reduction = 100 * (1 - row.compliance / state.initial);
    byId("improvement").textContent = row.iteration ? `${Math.abs(reduction).toFixed(1)}% ${reduction >= 0 ? "lower" : "higher"} than initial` : "Initial design";
    byId("improvement").className = reduction > 0 ? "positive" : "";
    byId("current-volume").textContent = `${(100 * row.volume_fraction).toFixed(2)}%`;
    byId("iteration").innerHTML = `${row.iteration}<em> / ${state.model.max_iterations}</em>`;
    for (const id of ["compliance", "current-volume", "iteration", "improvement"]) byId(id).title = byId(id).textContent;
    byId("progress-bar").style.width = `${100 * row.iteration / state.model.max_iterations}%`;
    boundaryEditor?.setResults(row.load_case_compliances);
    resultMessage(state.model.state_solver === "rom" ? "Reduced model: approximate results" : "");
    status("Running", "running");
    if (topologyCanvas.getClientRects().length) drawTopology(topologyCanvas, {...state.model,display_load_case:boundaryEditor?.activeCase}, state.density);
    updateConvergence();
    return;
  }
  if (event.event === "complete") {
    const labels = {cancelled: "Stopped", maximum_iterations: "Iteration limit", design_and_objective_stable: "Complete"};
    status(labels[event.stop_reason] || "Finished", event.stop_reason === "cancelled" ? "ready" : "complete");
    let message = event.stop_reason === "cancelled" ? "Stopped at the last evaluated design." :
      event.stop_reason === "maximum_iterations" ? "Iteration limit reached." : "Stopping tolerances reached.";
    if (state.runConfig?.state_solver === "rom") message += " Reduced model: approximate results.";
    resultMessage(message);
    if (!event.feasible) showError(event.max_size_feasible === false ?
      "The final design has not met the maximum solid-size constraint. The last evaluated design is shown." : event.max_void_feasible === false ?
      "The final design has not met the maximum void-size constraint. The last evaluated design is shown." : event.volume_feasible !== false && event.epigraph_feasible === false ?
      "The final epigraph bound is below a field's compliance. Increase the iteration limit and run again." :
      "The final design has not met the volume constraint. Increase the iteration limit and run again.");
    return;
  }
  if (event.event === "error") throw new Error(event.message);
}

async function start(event) {
  event.preventDefault();
  if (state.running || state.externalBusy || !state.ready || !romChoice.allowed || boundaryEditor?.validationError || !form.reportValidity()) return;
  state.model = readModel(); state.runId = null;
  clearResult(); showError(); setRunning(true); status("Starting", "running");
  resultMessage("Starting the solver...");
  const resultView = document.querySelector(state.activePanel === "panel-visualisation" ? "#convergence-view" : ".topology-panel");
  if (window.matchMedia("(min-width: 701px)").matches) {
    const column = document.querySelector(".central-workspace");
    column.scrollTo({top: column.scrollTop + resultView.getBoundingClientRect().top - column.getBoundingClientRect().top, behavior: "smooth"});
  } else {
    resultView.scrollIntoView({behavior: "smooth", block: "start"});
  }
  let terminal = false;
  try {
    const payload = {...state.model};
    if (!state.romSupported) delete payload.state_solver;
    const response = await fetch("/api/run", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload)});
    if (!response.ok) {
      const error = await response.json();
      const details = error.detail;
      if (Array.isArray(details) && details.some(item => item.type === "extra_forbidden" &&
          ["max_inner_iterations", "volume_constraint", "linear_solver", "max_void_size", "max_void_beta", "max_void_tolerance", "max_solid_size", "max_size_beta", "max_size_tolerance", "projection_beta", "projection_eta", "robust", "robust_eta_offset", "field_execution", "material_model", "penalisation", "projection_schedule", "projection_beta_interval", "projection_beta_multiplier", "projection_beta_max"].includes(item.loc?.at(-1)))) {
        throw new Error("Open Start TOptVis.bat again to launch the updated server. Use the browser page it opens.");
      }
      throw new Error(typeof details === "string" ? details : details?.map(item => `${item.loc?.at(-1) || "Input"}: ${item.msg}`).join("; ") || "Could not start the solver.");
    }
    const reader = response.body.getReader(), decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const {value, done} = await reader.read();
      buffer += decoder.decode(value, {stream: !done});
      const lines = buffer.split("\n"); buffer = lines.pop();
      for (const line of lines) {
        if (!line.trim()) continue;
        const message = JSON.parse(line);
        receive(message);
        if (message.event === "complete") terminal = true;
      }
      if (done) {
        if (buffer.trim()) {
          const message = JSON.parse(buffer); receive(message);
          if (message.event === "complete") terminal = true;
        }
        break;
      }
      await new Promise(requestAnimationFrame);
    }
    if (!terminal) throw new Error("The live connection ended before the run finished. Please run again.");
  } catch (error) {
    showError(error.message || "Could not connect to the local solver.");
    status("Error", "error");
    resultMessage("The run could not finish.");
    if (state.runId) fetch(`/api/run/${state.runId}/stop`, {method: "POST"}).catch(() => {});
  } finally {
    state.runId = null; setRunning(false);
  }
}

form.addEventListener("submit", start);
byId("display-field").addEventListener("change", () => selectField());
byId("show-model").addEventListener("click", () => selectPanel("panel-geometry", true));
form.addEventListener("input", updateModel);
byId("filter-slider").addEventListener("input", event => {
  byId("filter_radius").value = Number(Number(event.target.value).toPrecision(4));
  updateModel(event);
});
byId("projection-slider").addEventListener("input", event => {
  byId("projection_beta").value = Number(Number(event.target.value).toPrecision(4));
  updateModel(event);
});
byId("stop").addEventListener("click", async () => {
  if (!state.runId) return;
  byId("stop").disabled = true; status("Stopping", "running");
  try {
    const response = await fetch(`/api/run/${state.runId}/stop`, {method: "POST"});
    if (!response.ok && response.status !== 404) throw new Error("Could not stop the run.");
  } catch (error) { showError(error.message); byId("stop").disabled = false; }
});
const viewerResizeObserver = new ResizeObserver(redraw);
for (const element of [document.querySelector(".central-workspace"), modelCanvas.parentElement, topologyCanvas.parentElement]) {
  viewerResizeObserver.observe(element);
}
document.addEventListener("solveractivity", event => {
  if (event.detail.workspace !== "cantilever") {
    state.externalBusy = event.detail.running;
    updateRunAvailability();
    if (!event.detail.running && state.romSupported) romChoice.refresh();
  }
});
document.addEventListener("workspacechange", event => {
  if (event.detail === "cantilever-2d") { requestAnimationFrame(redraw); if (state.romSupported) romChoice.refresh(); }
});
syncControls(); state.model = readModel();
boundaryEditor = new BoundaryEditor({canvas: modelCanvas, getModel: () => state.model, isRunning: () => state.running,
  onChange: () => updateModel(), onError: showError, onOpen: () => selectPanel("panel-load"), redraw});
updateLabels(); clearResult(); redraw();
initialiseReferences();
projectIO = new ProjectIO({isRunning: () => state.running, onError: showError,
  getModel: () => {
    if (!state.running && !form.reportValidity()) throw new Error("Correct the model inputs before saving.");
    if (boundaryEditor.validationError) throw new Error(boundaryEditor.validationError);
    return structuredClone(state.model);
  },
  getResult: () => state.density ? {density:state.density,fields:state.fields,history:state.history,
    stopping_tolerances:Object.fromEntries(["design_tolerance","objective_tolerance","feasibility_tolerance"].filter(key => Number.isFinite(state.runConfig?.[key])).map(key => [key,state.runConfig[key]]))} : null,
  load: (model, saved) => {
    for (const control of form.elements) {
      if (!control.name || !Object.hasOwn(model,control.name) || model[control.name] == null) continue;
      if (control.name === "rom_training_id" && ![...control.options].some(option => option.value === model.rom_training_id)) control.add(new Option("Saved model unavailable",model.rom_training_id));
      control.value = String(model[control.name]);
    }
    state.model = structuredClone(model); state.runId = null;
    boundaryEditor.loadModel(model); syncControls(); updateLabels(); clearResult(); showError(); romChoice.sync();
    if (saved) {
      state.density = saved.density; state.fields = saved.fields || null; state.history = saved.history;
      state.fieldRow = saved.history.at(-1); state.initial = saved.history[0].compliance;
      state.runConfig = {...model,...saved.stopping_tolerances};
      byId("empty-state").hidden = true;
      byId("compliance").textContent = format(state.fieldRow.compliance);
      byId("current-volume").textContent = `${(100*state.fieldRow.volume_fraction).toFixed(2)}%`;
      byId("iteration").innerHTML = `${state.fieldRow.iteration}<em> / ${model.max_iterations}</em>`;
      for (const id of ["compliance","current-volume","iteration"]) byId(id).title = byId(id).textContent;
      byId("progress-bar").style.width = `${100*state.fieldRow.iteration/model.max_iterations}%`;
      boundaryEditor.setResults(state.fieldRow.load_case_compliances);
      selectField(false); updateMaxSizeStatus(); updateConvergence();
      status("Saved snapshot","ready"); resultMessage("Saved result; start a new run to recompute.");
    }
    selectPanel("panel-geometry"); updateRunAvailability(); redraw();
  },
});
async function connectSolver() {
  try {
    const response = await fetch("/api/health");
    if (!response.ok) throw new Error("Local solver unavailable");
    let health = await response.json();
    state.romSupported = !!health.rom_optimisation_supported;
    await romChoice.refresh(state.romSupported);
    if (!health.non_design_regions_supported || !health.extended_boundaries_supported || !health.multiple_load_cases_supported || !health.boundary_conditions_supported || !health.projection_supported || !health.robust_supported || !health.material_schedule_supported || !health.max_size_supported || !health.max_void_supported || !health.linear_solver_supported || !health.volume_modes_supported || !health.gcmma_inner_limit_supported) {
      byId("connection").lastElementChild.textContent = "Server restart needed";
      showError("Open Start TOptVis.bat again to launch the updated server with the current solver and size controls. Use the browser page it opens.");
      setRunning(false);
    } else {
      while (!health.worker_ready) {
        byId("connection").lastElementChild.textContent = "Preparing solver…";
        state.ready = false;
        setRunning(false);
        await new Promise(resolve => setTimeout(resolve, 500));
        const readyResponse = await fetch("/api/health", {cache: "no-store"});
        if (!readyResponse.ok) throw new Error("Solver warm-up unavailable");
        health = await readyResponse.json();
      }
      state.ready = true;
      byId("connection").dataset.ready = "true";
      byId("connection").lastElementChild.textContent = "Local solver connected";
      if (health.running) showError("Another run is active in this server. Wait for it to finish before starting.");
      setRunning(false);
    }
  } catch {
    byId("connection").lastElementChild.textContent = "Solver unavailable";
    showError("Could not reach the local solver. Start it with python -m ml_opt.web, then reload this page.");
  }
}
connectSolver();

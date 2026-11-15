import {drawModel, drawTopology, pickNode} from "/static/viewer.js";
import {ConvergencePlots} from "/static/convergence.js";
import {RomChoice} from "/static/rom-choice.js";

const byId = id => document.getElementById(id);
const form = byId("model-form"), modelCanvas = byId("model-canvas"), topologyCanvas = byId("topology-canvas");
const state = {model: null, density: null, fields: null, fieldRow: null, running: false, ready: false, externalBusy: false, runId: null, initial: null,
  hover: null, history: [], runConfig: null, activePanel: "panel-geometry", romSupported: false};
const romChoice = new RomChoice({onChange: () => updateModel(), onStateChange: updateRunAvailability, isRunning: () => state.running});
const convergencePlots = new ConvergencePlots();
let convergenceFrame = null;
const format = value => new Intl.NumberFormat("en", {maximumFractionDigits: 2}).format(value);
const filterFormat = value => new Intl.NumberFormat("en", {maximumSignificantDigits: 4}).format(value);
const signed = value => (Math.abs(value) < 1e-12 ? "0.00" : value.toFixed(2)).replace("-", "−");

const volumeRelation = mode => ({lt: "<", gt: ">", eq: "="})[mode || "lt"];

function readModel() {
  const model = {};
  for (const [key, value] of new FormData(form)) {
    model[key] = key === "robust" ? value === "true" :
      ["optimizer", "field_execution", "state_solver", "linear_solver", "rom_training_id", "material_model", "projection_schedule", "volume_constraint"].includes(key) ? value : Number(value);
  }
  if (model.state_solver !== "rom") delete model.rom_training_id;
  return model;
}

function updateRunAvailability() {
  byId("run").disabled = state.running || !state.ready || state.externalBusy || (byId("state_solver").value === "rom" && (!state.romSupported || !romChoice.allowed));
  if (!state.running) byId("run-label").textContent = byId("state_solver").value === "rom" ? "Run ROM optimisation" : "Run optimisation";
}

function updateSolverDescription() {
  const config = state.runConfig || state.model;
  const latest = state.history.at(-1);
  byId("run-solver-description").textContent = config?.state_solver === "rom" ?
    `ROM · ${latest?.rom_basis_rank || romChoice.selected?.basis.rank || "—"} modes · basis ${config.rom_basis_sha256?.slice(0,12) || romChoice.selected?.id.slice(0,8) || "not selected"} · approximate compliance and sensitivities` :
    `FOM · ${(config?.linear_solver || "lu").toUpperCase()} · full-order elasticity`;
  byId("fom-linear-options").hidden = state.model?.state_solver === "rom";
  byId("fom-method-hint").hidden = state.model?.state_solver === "rom";
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
  byId("error").textContent = message;
  byId("error").hidden = !message;
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
  byId("history-beta").textContent = Number.isFinite(latest?.projection_beta) ? filterFormat(latest.projection_beta) : "—";
  byId("history-max-size-summary").hidden = !(config.max_solid_size > 0);
  byId("history-max-size").textContent = Number.isFinite(latest?.max_size_constraint) ? filterFormat(latest.max_size_constraint) : "—";
  byId("history-max-void-summary").hidden = !(config.max_void_size > 0);
  byId("history-max-void").textContent = Number.isFinite(latest?.max_void_constraint) ? filterFormat(latest.max_void_constraint) : "—";
  byId("convergence-description").textContent = latest ?
    `${state.model.optimizer.toUpperCase()} · ${(config.state_solver || "fom").toUpperCase()} · ${state.history.length} evaluated history ${state.history.length === 1 ? "point" : "points"} · iterations 0–${latest.iteration}` :
    "Run an optimisation to build the convergence history.";
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
    drawModel(modelCanvas, state.model, state.hover);
    drawTopology(topologyCanvas, state.model, state.density);
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
  byId("penalty-label").textContent = ramp ? "Penalty parameter · q" : "Penalty exponent · p";
  byId("material-formula").textContent = ramp ? "E(ρ) = E_min + (E₀ − E_min) ρ / [1 + q(1 − ρ)]" : "E(ρ) = E_min + (E₀ − E_min) ρᵖ";
  byId("material-description").textContent = ramp ?
    "RAMP: q ≥ 0. Larger q penalises intermediate densities; q = 0 is linear." :
    "SIMP: p ≥ 1. Larger p penalises intermediate densities; p = 1 is linear.";
  const scheduled = byId("projection_schedule").value === "geometric";
  byId("beta-schedule-options").hidden = !scheduled;
  const beta = Number(byId("projection_beta").value), maximum = Number(byId("projection_beta_max").value);
  byId("projection_beta").setCustomValidity(scheduled && beta <= 0 ? "Scheduled projection needs a positive starting β." : "");
  byId("projection_beta_max").setCustomValidity(maximum <= 0 || (scheduled && maximum < beta) ? "Maximum β must be positive and at least the starting β." : "");
  byId("projection_beta_multiplier").setCustomValidity(Number(byId("projection_beta_multiplier").value) <= 1 ? "The β multiplier must be greater than 1." : "");
}

function updateProjectionStatus() {
  const m = state.model, active = state.history.at(-1)?.projection_beta ?? m.projection_beta;
  byId("projection-description").textContent = active > 0 ?
    `β = ${filterFormat(active)} · η = ${filterFormat(m.projection_eta)} · ${m.projection_schedule === "geometric" ? `scheduled to ${filterFormat(m.projection_beta_max)}` : "fixed during run"}` :
    "Projection off · smoothed densities used directly";
  byId("model-filter-summary").textContent = `${m.filter_radius > 0 ? `ℓ = ${filterFormat(m.filter_radius)}` : "Smoothing off"} · ${active > 0 ? `β = ${filterFormat(active)}` : "Projection off"}`;
}

function updateMaxSizeStatus() {
  const row = state.history.at(-1), m = state.model;
  byId("max-size-description").textContent = m.max_solid_size > 0 ?
    `Diameter ${filterFormat(m.max_solid_size)} model units · ${m.robust ? "dilated" : "physical"} field · allowed eroded fraction ${filterFormat(100*m.max_size_tolerance)}%. Outside the domain is void.` :
    "Maximum-size constraint off";
  byId("max-size-status").textContent = Number.isFinite(row?.max_size_constraint) ?
    `Eroded fraction ${filterFormat(100*row.max_size_eroded_volume)}% · ${row.max_size_constraint <= (state.runConfig?.feasibility_tolerance ?? 1e-4) ? "satisfied" : "violated"} · g = ${filterFormat(row.max_size_constraint)}` : "";
  byId("max-void-description").textContent = m.max_void_size > 0 ?
    `Diameter ${filterFormat(m.max_void_size)} model units · ${m.robust ? "eroded material" : "physical"} field · allowed remaining void ${filterFormat(100*m.max_void_tolerance)}%. Outside the domain is void.` :
    "Maximum void-size constraint off";
  byId("max-void-status").textContent = Number.isFinite(row?.max_void_constraint) ?
    `Remaining void ${filterFormat(100*row.max_void_remaining_volume)}% · ${row.max_void_constraint <= (state.runConfig?.feasibility_tolerance ?? 1e-4) ? "satisfied" : "violated"} · g = ${filterFormat(row.max_void_constraint)}` : "";
}

function updateLabels(resizeFilterRange = true) {
  const m = state.model;
  byId("mesh-description").textContent = `${format(m.nx * m.ny)} elements · ${format((m.nx + 1) * (m.ny + 1))} nodes`;
  byId("selection-description").textContent = `Load at node (${m.load_node_x}, ${m.load_node_y}) · x = ${format(m.load_node_x / m.nx * m.length)}, y = ${format(m.load_node_y / m.ny * m.height)}`;
  byId("volume-label").textContent = `${Math.round(m.volume_fraction * 100)}%`;
  byId("volume-target").textContent = `Volume ${volumeRelation(m.volume_constraint)} ${Math.round(m.volume_fraction * 100)}%`;
  byId("angle-slider").value = m.load_angle;
  const radius = m.filter_radius;
  const slider = byId("filter-slider");
  const sliderMax = resizeFilterRange ? Math.max(Math.min(m.length, m.height) / 2, radius) : Number(slider.max);
  slider.max = sliderMax;
  slider.step = "any";
  slider.value = radius;
  byId("filter-range-max").textContent = filterFormat(sliderMax);
  byId("filter-description").textContent = radius > 0 ?
    `ℓ = ${filterFormat(radius)} · Ø = ${filterFormat(2 * radius)} model units · radius ${filterFormat(radius * m.nx / m.length)} × ${filterFormat(radius * m.ny / m.height)} cells` : "Filter off · design densities used directly";
  const beta = m.projection_beta, eta = m.projection_eta;
  const projectionSlider = byId("projection-slider");
  const projectionMax = resizeFilterRange ? Math.max(16, beta) : Number(projectionSlider.max);
  projectionSlider.max = projectionMax;
  projectionSlider.value = beta;
  byId("projection-range-max").textContent = filterFormat(projectionMax);
  const scheduled = m.projection_schedule === "geometric";
  const increases = scheduled && beta > 0 ? Math.max(0, Math.ceil(Math.log(m.projection_beta_max / beta) / Math.log(m.projection_beta_multiplier))) : 0;
  const finalStage = increases ? increases * m.projection_beta_interval + 1 : 0;
  byId("beta-schedule-description").textContent =
    `Start β = ${filterFormat(beta)}; multiply by ${filterFormat(m.projection_beta_multiplier)} after every ${m.projection_beta_interval} completed iterations, capped at ${filterFormat(m.projection_beta_max)}. ` +
    (finalStage > m.max_iterations ? `The iteration limit will be reached before the maximum β (first used at iteration ${format(finalStage)}).` :
      `Maximum β is first used at iteration ${format(finalStage)}. Convergence stopping waits for this stage.`);
  byId("robust-options").hidden = !m.robust;
  byId("solver-field-options").hidden = !m.robust;
  byId("robust-description").hidden = !m.robust;
  byId("field-view").hidden = !m.robust;
  byId("compliance-label").textContent = `${m.state_solver === "rom" ? "ROM " : ""}${m.robust ? "WORST COMPLIANCE" : "COMPLIANCE"}`;
  byId("objective-description").textContent = m.robust ?
    "Minimise the worst compliance across the eroded, nominal and dilated material fields for the applied load." :
    "Minimise compliance for the applied load using the physical material after filtering and projection.";
  byId("material-volume-label").textContent = m.robust ? "DILATED VOLUME" : "MATERIAL VOLUME";
  byId("volume-description").textContent =
    `${m.robust ? "Dilated-field" : "Physical material"} volume ${volumeRelation(m.volume_constraint)} ${filterFormat(100*m.volume_fraction)}%. Applied after filtering and projection, within the solver's feasibility tolerance.` +
    (m.volume_constraint === "gt" ? " A lower bound allows the optimiser to use more material, up to a fully solid domain." : "") +
    (m.robust ? " This target applies to the dilated field for all three modes." : "");
  if (m.robust) byId("robust-description").textContent =
    `Thresholds: eroded ${filterFormat(eta + m.robust_eta_offset)}, nominal ${filterFormat(eta)}, dilated ${filterFormat(eta - m.robust_eta_offset)}. Worst compliance and dilated-volume constraint. Requires β > 0 and all thresholds inside (0, 1).`;
  byId("filter-status").textContent = [radius > 0 ? "Helmholtz" : null, beta > 0 ? "Projection" : null].filter(Boolean).join(" + ") || "Filters off";
  updateProjectionStatus();
  byId("filter-legend").hidden = radius === 0;
  updateMaxSizeStatus();
  const rad = m.load_angle * Math.PI / 180;
  byId("force-components").textContent = `F = (${signed(m.load * Math.cos(rad))}, ${signed(m.load * Math.sin(rad))})`;
}

function clearResult() {
  state.density = null; state.fields = null; state.fieldRow = null; state.initial = null;
  byId("field-metrics").textContent = "";
  state.history = []; state.runConfig = null;
  updateProjectionStatus();
  updateMaxSizeStatus();
  byId("empty-state").hidden = false;
  byId("compliance").textContent = "—";
  byId("improvement").textContent = "Waiting for first solve";
  byId("improvement").className = "";
  byId("current-volume").textContent = "—";
  byId("iteration").innerHTML = `—<em> / ${state.model.max_iterations}</em>`;
  byId("density-change").textContent = "No iterations yet";
  byId("progress-bar").style.width = "0%";
  byId("topology-description").textContent = "Run optimisation to see the material distribution.";
  byId("live-note").textContent = "Updates after every evaluated iteration";
  status("Ready", "ready");
  updateSolverDescription();
  updateConvergence();
}

function selectField(redrawView = true) {
  const name = byId("display-field").value;
  if (state.fields) {
    state.density = state.fields[name];
    const row = state.fieldRow;
    byId("field-metrics").textContent = `Compliance ${format(row.field_compliances[name])} · volume ${(100 * row.field_volumes[name]).toFixed(2)}%`;
    if (Number.isFinite(row.epigraph_compliance)) byId("field-metrics").textContent += ` · epigraph t ${format(row.epigraph_compliance)}`;
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
  romChoice.sync();
  if (!form.checkValidity()) return;
  state.model = readModel(); state.hover = null;
  showError(); updateLabels(!["filter-slider", "projection-slider"].includes(event?.target.id)); clearResult(); redraw();
}

function setRunning(running) {
  state.running = running;
  byId("settings").disabled = running;
  updateRunAvailability();
  byId("run-label").textContent = running ? "Optimising…" : romChoice.isRom ? "Run ROM optimisation" : "Run optimisation";
  byId("stop").disabled = !running || !state.runId;
  modelCanvas.style.cursor = running ? "default" : "crosshair";
  romChoice.sync();
  document.dispatchEvent(new CustomEvent("solveractivity", {detail: {workspace: "cantilever", running}}));
}

function receive(event) {
  if (event.event === "started") {
    state.runId = event.run_id;
    state.runConfig = event.config;
    updateSolverDescription();
    updateConvergence();
    byId("stop").disabled = false;
    return;
  }
  if (event.event === "iteration") {
    const row = event.history;
    state.history.push(row);
    updateMaxSizeStatus();
    updateSolverDescription();
    updateProjectionStatus();
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
    byId("density-change").textContent = `Density change ${row.design_change.toFixed(4)}`;
    byId("progress-bar").style.width = `${100 * row.iteration / state.model.max_iterations}%`;
    byId("topology-description").textContent = state.model.robust ?
      `${state.model.optimizer.toUpperCase()} · Three fields · worst: ${row.worst_field} · ${state.model.field_execution}` :
      `${state.model.optimizer.toUpperCase()} · ${state.model.projection_beta > 0 ? "Projected" : state.model.filter_radius > 0 ? "Filtered" : "Unfiltered"} physical densities`;
    byId("live-note").textContent = `Iteration ${row.iteration} · β = ${filterFormat(row.projection_beta)} · ${state.model.material_model.toUpperCase()} · ${row.state_solves} ${(row.state_solver || "fom").toUpperCase()} state solves · residual ${row.linear_residual.toExponential(2)}`;
    status("Running", "running");
    if (topologyCanvas.getClientRects().length) drawTopology(topologyCanvas, state.model, state.density);
    updateConvergence();
    return;
  }
  if (event.event === "complete") {
    const labels = {cancelled: "Stopped", maximum_iterations: "Iteration limit", design_and_objective_stable: "Complete"};
    status(labels[event.stop_reason] || "Finished", event.stop_reason === "cancelled" ? "ready" : "complete");
    byId("topology-description").textContent = event.stop_reason === "cancelled" ? "Run stopped. Showing the last evaluated design." :
      event.stop_reason === "maximum_iterations" ? "Iteration limit reached. Showing the last evaluated design." : "Design and objective reached the stopping tolerances.";
    if (state.runConfig?.state_solver === "rom") byId("topology-description").textContent += " ROM objective and sensitivities; full-order accuracy is not certified by this stopping test.";
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
  if (state.running || state.externalBusy || !state.ready || !romChoice.allowed || !form.reportValidity()) return;
  state.model = readModel(); state.runId = null;
  clearResult(); showError(); setRunning(true); status("Starting", "running");
  byId("topology-description").textContent = "Starting the solver…";
  document.querySelector(state.activePanel === "panel-visualisation" ? "#convergence-view" : ".topology-panel")
    .scrollIntoView({behavior: "smooth", block: "start"});
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
    byId("topology-description").textContent = "The run could not finish.";
    if (state.runId) fetch(`/api/run/${state.runId}/stop`, {method: "POST"}).catch(() => {});
  } finally {
    state.runId = null; setRunning(false);
  }
}

form.addEventListener("submit", start);
byId("display-field").addEventListener("change", () => selectField());
byId("show-model").addEventListener("click", () => selectPanel("panel-geometry", true));
form.addEventListener("input", updateModel);
byId("angle-slider").addEventListener("input", event => {
  byId("load_angle").value = event.target.value;
  updateModel();
});
byId("filter-slider").addEventListener("input", event => {
  byId("filter_radius").value = Number(Number(event.target.value).toPrecision(4));
  updateModel(event);
});
byId("projection-slider").addEventListener("input", event => {
  byId("projection_beta").value = Number(Number(event.target.value).toPrecision(4));
  updateModel(event);
});
modelCanvas.addEventListener("click", event => {
  if (state.running) return;
  const node = pickNode(modelCanvas, state.model, event);
  if (!node) { redraw(); return; }
  if (node.x === 0) { showError("The left edge is fixed. Select a free mesh node for the load."); redraw(); return; }
  byId("load_node_x").value = node.x; byId("load_node_y").value = node.y;
  updateModel();
});
modelCanvas.addEventListener("pointermove", event => {
  if (state.running) return;
  state.hover = pickNode(modelCanvas, state.model, event);
  drawModel(modelCanvas, state.model, state.hover);
});
modelCanvas.addEventListener("pointerleave", () => { state.hover = null; drawModel(modelCanvas, state.model); });
byId("stop").addEventListener("click", async () => {
  if (!state.runId) return;
  byId("stop").disabled = true; status("Stopping", "running");
  try {
    const response = await fetch(`/api/run/${state.runId}/stop`, {method: "POST"});
    if (!response.ok && response.status !== 404) throw new Error("Could not stop the run.");
  } catch (error) { showError(error.message); byId("stop").disabled = false; }
});
new ResizeObserver(redraw).observe(document.querySelector(".central-workspace"));
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
syncControls(); state.model = readModel(); updateLabels(); clearResult(); redraw();
async function connectSolver() {
  try {
    const response = await fetch("/api/health");
    if (!response.ok) throw new Error("Local solver unavailable");
    let health = await response.json();
    state.romSupported = !!health.rom_optimisation_supported;
    await romChoice.refresh(state.romSupported);
    if (!health.projection_supported || !health.robust_supported || !health.material_schedule_supported || !health.max_size_supported || !health.max_void_supported || !health.linear_solver_supported || !health.volume_modes_supported || !health.gcmma_inner_limit_supported) {
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

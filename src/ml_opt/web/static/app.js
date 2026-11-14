import {drawModel, drawTopology, pickNode} from "/static/viewer.js";

const byId = id => document.getElementById(id);
const form = byId("model-form"), modelCanvas = byId("model-canvas"), topologyCanvas = byId("topology-canvas");
const state = {model: null, density: null, running: false, ready: false, runId: null, initial: null, hover: null};
const format = value => new Intl.NumberFormat("en", {maximumFractionDigits: 2}).format(value);
const filterFormat = value => new Intl.NumberFormat("en", {maximumSignificantDigits: 4}).format(value);
const signed = value => (Math.abs(value) < 1e-12 ? "0.00" : value.toFixed(2)).replace("-", "−");

function readModel() {
  const model = {};
  for (const [key, value] of new FormData(form)) model[key] = key === "optimizer" ? value : Number(value);
  return model;
}

function showError(message = "") {
  byId("error").textContent = message;
  byId("error").hidden = !message;
}

function status(label, kind) {
  const badge = byId("run-status");
  badge.dataset.status = kind;
  badge.replaceChildren(Object.assign(document.createElement("i"), {}), document.createTextNode(label));
}

function redraw() {
  if (!state.model) return;
  drawModel(modelCanvas, state.model, state.hover);
  drawTopology(topologyCanvas, state.model, state.density);
}

function updateLabels(resizeFilterRange = true) {
  const m = state.model;
  byId("mesh-description").textContent = `${format(m.nx * m.ny)} elements · ${format((m.nx + 1) * (m.ny + 1))} nodes`;
  byId("selection-description").textContent = `Load at node (${m.load_node_x}, ${m.load_node_y}) · x = ${format(m.load_node_x / m.nx * m.length)}, y = ${format(m.load_node_y / m.ny * m.height)}`;
  byId("volume-label").textContent = `${Math.round(m.volume_fraction * 100)}%`;
  byId("volume-target").textContent = `Target ${Math.round(m.volume_fraction * 100)}%`;
  byId("angle-slider").value = m.load_angle;
  const radius = m.filter_radius;
  const slider = byId("filter-slider");
  const sliderMax = resizeFilterRange ? Math.max(Math.min(m.length, m.height) / 2, radius) : Number(slider.max);
  slider.max = sliderMax;
  slider.step = "any";
  slider.value = radius;
  byId("filter-range-max").textContent = filterFormat(sliderMax);
  byId("filter-description").textContent = radius > 0 ?
    `ℓ = ${filterFormat(radius)} model units · ${filterFormat(radius * m.nx / m.length)} × ${filterFormat(radius * m.ny / m.height)} cells` : "Filter off · design densities used directly";
  byId("filter-status").textContent = radius > 0 ? "Helmholtz filter" : "Filter off";
  byId("filter-legend").hidden = radius === 0;
  const rad = m.load_angle * Math.PI / 180;
  byId("force-components").textContent = `F = (${signed(m.load * Math.cos(rad))}, ${signed(m.load * Math.sin(rad))})`;
}

function clearResult() {
  state.density = null; state.initial = null;
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
}

function updateModel(event) {
  if (state.running) return;
  if (event?.target.id === "nx" || event?.target.id === "ny") {
    const nx = Number(byId("nx").value), ny = Number(byId("ny").value);
    if (nx >= 2 && nx <= 160 && ny >= 2 && ny <= 100) {
      byId("load_node_x").max = nx;
      byId("load_node_y").max = ny;
      byId("load_node_x").value = Math.max(1, Math.round(state.model.load_node_x / state.model.nx * nx));
      byId("load_node_y").value = Math.round(state.model.load_node_y / state.model.ny * ny);
    }
  }
  if (!form.checkValidity()) return;
  state.model = readModel(); state.hover = null;
  showError(); updateLabels(event?.target.id !== "filter-slider"); clearResult(); redraw();
}

function setRunning(running) {
  state.running = running;
  byId("settings").disabled = running;
  byId("filter-settings").disabled = running;
  byId("run").disabled = running || !state.ready;
  byId("run-label").textContent = running ? "Optimising…" : "Run optimisation";
  byId("stop").disabled = !running || !state.runId;
  modelCanvas.style.cursor = running ? "default" : "crosshair";
}

function receive(event) {
  if (event.event === "started") {
    state.runId = event.run_id;
    byId("stop").disabled = false;
    return;
  }
  if (event.event === "iteration") {
    const row = event.history;
    if (state.initial === null) state.initial = row.compliance;
    state.density = event.density;
    byId("empty-state").hidden = true;
    byId("compliance").textContent = format(row.compliance);
    const reduction = 100 * (1 - row.compliance / state.initial);
    byId("improvement").textContent = row.iteration ? `${Math.abs(reduction).toFixed(1)}% ${reduction >= 0 ? "lower" : "higher"} than initial` : "Initial design";
    byId("improvement").className = reduction > 0 ? "positive" : "";
    byId("current-volume").textContent = `${(100 * row.volume_fraction).toFixed(2)}%`;
    byId("iteration").innerHTML = `${row.iteration}<em> / ${state.model.max_iterations}</em>`;
    byId("density-change").textContent = `Density change ${row.design_change.toFixed(4)}`;
    byId("progress-bar").style.width = `${100 * row.iteration / state.model.max_iterations}%`;
    byId("topology-description").textContent = `${state.model.optimizer.toUpperCase()} · ${state.model.filter_radius > 0 ? "Filtered" : "Unfiltered"} element densities`;
    byId("live-note").textContent = `Iteration ${row.iteration} · ${row.state_solves} state solves`;
    status("Running", "running");
    drawTopology(topologyCanvas, state.model, state.density);
    return;
  }
  if (event.event === "complete") {
    const labels = {cancelled: "Stopped", maximum_iterations: "Iteration limit", design_and_objective_stable: "Complete"};
    status(labels[event.stop_reason] || "Finished", event.stop_reason === "cancelled" ? "ready" : "complete");
    byId("topology-description").textContent = event.stop_reason === "cancelled" ? "Run stopped. Showing the last evaluated design." :
      event.stop_reason === "maximum_iterations" ? "Iteration limit reached. Showing the last evaluated design." : "Design and objective reached the stopping tolerances.";
    if (!event.feasible) showError("The final design has not met the volume constraint. Increase the iteration limit and run again.");
    return;
  }
  if (event.event === "error") throw new Error(event.message);
}

async function start(event) {
  event.preventDefault();
  if (state.running || !state.ready || !form.reportValidity()) return;
  state.model = readModel(); state.runId = null;
  clearResult(); showError(); setRunning(true); status("Starting", "running");
  byId("topology-description").textContent = "Starting the solver…";
  document.querySelector(".topology-panel").scrollIntoView({behavior: "smooth", block: "start"});
  let terminal = false;
  try {
    const response = await fetch("/api/run", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(state.model)});
    if (!response.ok) {
      const error = await response.json();
      throw new Error(typeof error.detail === "string" ? error.detail : error.detail?.map(item => item.msg).join("; ") || "Could not start the solver.");
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
form.addEventListener("input", updateModel);
byId("angle-slider").addEventListener("input", event => {
  byId("load_angle").value = event.target.value;
  updateModel();
});
byId("filter-slider").addEventListener("input", event => {
  byId("filter_radius").value = Number(Number(event.target.value).toPrecision(4));
  updateModel(event);
});
byId("filter_radius").addEventListener("input", updateModel);
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
new ResizeObserver(redraw).observe(document.querySelector(".viewers"));
state.model = readModel(); updateLabels(); clearResult(); redraw();
try {
  const response = await fetch("/api/health");
  if (!response.ok) throw new Error("Local solver unavailable");
  const health = await response.json();
  state.ready = true;
  byId("connection").dataset.ready = "true";
  byId("connection").lastElementChild.textContent = "Local solver connected";
  if (health.running) showError("Another run is active in this server. Wait for it to finish before starting.");
  setRunning(false);
} catch {
  byId("connection").lastElementChild.textContent = "Solver unavailable";
  showError("Could not reach the local solver. Start it with python -m ml_opt.web, then reload this page.");
}

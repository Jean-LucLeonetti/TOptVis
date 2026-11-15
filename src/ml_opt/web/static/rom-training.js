import {drawTopology} from "/static/viewer.js";
import {ValidationView} from "/static/rom-validation.js";

const byId = id => document.getElementById(id);
const snapshotForm = byId("rom-snapshot-form"), basisForm = byId("rom-basis-form");
const dataset = byId("rom-dataset"), canvas = byId("rom-preview");
const state = {ready: false, busy: false, externalBusy: false, jobId: null,
  records: [], selected: null, previewModel: null, density: null, completedId: null, liveId: null,
  validationSupported: false, kind: null, terminal: null};
const validationView = new ValidationView({run, getDataset: () => state.selected, onError: error});
const number = value => new Intl.NumberFormat("en", {maximumFractionDigits: 2}).format(value);
const percent = value => `${(100 * value).toLocaleString("en", {maximumSignificantDigits: 5})}%`;

// Keep every field in one form so tab changes preserve the complete snapshot plan.
const snapshotTabs = Array.from(document.querySelectorAll('.rom-config-tabs [role="tab"]'));
function selectSnapshotPanel(panelId, focus = false) {
  for (const tab of snapshotTabs) {
    const selected = tab.dataset.panel === panelId;
    tab.setAttribute("aria-selected", String(selected));
    tab.tabIndex = selected ? 0 : -1;
    byId(tab.dataset.panel).hidden = !selected;
    if (selected) {
      byId("rom-config-title").textContent = tab.dataset.title;
      if (focus) tab.focus();
    }
  }
}
for (const [index, tab] of snapshotTabs.entries()) {
  tab.addEventListener("click", () => selectSnapshotPanel(tab.dataset.panel));
  tab.addEventListener("keydown", event => {
    let next;
    if (event.key === "ArrowDown" || event.key === "ArrowRight") next = (index + 1) % snapshotTabs.length;
    if (event.key === "ArrowUp" || event.key === "ArrowLeft") next = (index + snapshotTabs.length - 1) % snapshotTabs.length;
    if (event.key === "Home") next = 0;
    if (event.key === "End") next = snapshotTabs.length - 1;
    if (next !== undefined) { event.preventDefault(); selectSnapshotPanel(snapshotTabs[next].dataset.panel, true); }
  });
}
snapshotForm.addEventListener("invalid", event => {
  const invalid = snapshotForm.querySelector("input:invalid") || event.target;
  const panel = invalid.closest('[role="tabpanel"]');
  if (panel) selectSnapshotPanel(panel.id);
}, true);

function error(message = "") {
  byId("rom-error").textContent = message;
  byId("rom-error").hidden = !message;
}

function status(text, kind = "ready") {
  byId("rom-status").textContent = text;
  byId("rom-status").dataset.status = kind;
}

function updateButtons() {
  const disabled = !state.ready || state.busy || state.externalBusy;
  byId("rom-generate").disabled = disabled;
  byId("rom-build").disabled = disabled || !state.selected?.snapshot_count ||
    !["snapshots_ready", "cancelled", "basis_ready"].includes(state.selected.status);
  byId("rom-snapshot-settings").disabled = state.busy;
  byId("rom-basis-settings").disabled = state.busy;
  byId("rom-refresh").disabled = state.busy;
  byId("rom-stop").disabled = !state.busy || !state.jobId;
  validationView.setDisabled(disabled, state.validationSupported);
  byId("rom-availability").textContent = !state.ready ? "Restart TOptVis, then refresh this page to enable offline ROM training." :
    state.externalBusy ? "The solver worker is busy. Finish or stop the current job before starting training." :
    "Generate snapshots, then build a basis. Datasets and bases are saved locally and remain available after restart.";
}

function setBusy(busy) {
  state.busy = busy;
  updateButtons();
  document.dispatchEvent(new CustomEvent("solveractivity", {detail: {workspace: "rom-training", running: busy}}));
}

function readSettings() {
  const lists = ["load_heights", "load_angles", "volume_fractions"];
  return Object.fromEntries(Array.from(new FormData(snapshotForm), ([key, value]) => {
    if (!lists.includes(key)) return [key, Number(value)];
    const entries = value.split(",").map(item => item.trim());
    if (entries.some(item => !item || !Number.isFinite(Number(item)))) {
      throw new Error("Enter comma-separated numbers for load heights, angles, and volume fractions.");
    }
    return [key, entries.map(Number)];
  }));
}

function estimate() {
  try {
    const config = readSettings();
    const cases = config.load_heights.length * config.load_angles.length * config.volume_fractions.length;
    const snapshots = cases * (1 + Math.ceil(config.iterations_per_case / config.snapshot_stride));
    const bytes = 8 * (2*config.nx*(config.ny+1) + config.nx*config.ny) * snapshots;
    const power = bytes >= 2**40 ? 40 : bytes >= 2**30 ? 30 : bytes >= 2**20 ? 20 : bytes >= 2**10 ? 10 : 0;
    const unit = {0:"B",10:"KiB",20:"MiB",30:"GiB",40:"TiB"}[power];
    const storage = Number.isFinite(bytes) ? `${number(bytes/2**power)} ${unit}` : "estimate unavailable";
    byId("rom-estimate").textContent = `${number(cases)} case${cases === 1 ? "" : "s"} · up to ${number(snapshots)} snapshots · ${storage} of displacement/density arrays. No fixed size caps. Saving and POD require additional RAM.`;
  } catch (e) { byId("rom-estimate").textContent = e.message; }
}

function redraw() {
  if (!state.previewModel || !canvas.getClientRects().length) return;
  drawTopology(canvas, state.previewModel, state.density);
}

function spectrum(basis) {
  const values = basis.cumulative_energy;
  const x = mode => 45 + 520 * (mode - 1) / Math.max(values.length - 1, 1);
  const y = value => 145 - 120 * value;
  const path = values.map((value, index) => `${index ? "L" : "M"}${x(index+1)},${y(value)}`).join(" ");
  byId("rom-spectrum").innerHTML = `<title>Cumulative training snapshot energy. ${basis.rank} modes retain ${percent(basis.captured_energy)}.</title>
    <path d="M45 20V145H565" fill="none" stroke="#ccd9dc"/>
    <path d="M45 ${y(basis.energy_target)}H565" fill="none" stroke="#b9a8d2" stroke-dasharray="4 4"/>
    <path d="M${x(basis.rank)} 20V145" fill="none" stroke="#93b9ad" stroke-dasharray="4 4"/>
    <path d="${path}" fill="none" stroke="#196964" stroke-width="2.5"/>
    <circle cx="${x(basis.rank)}" cy="${y(basis.captured_energy)}" r="4" fill="#196964"/>
    <g fill="#74838c"><text x="5" y="28">100%</text><text x="18" y="148">0%</text>
    <text x="45" y="165">1</text><text x="565" y="165" text-anchor="end">${values.length}</text>
    <text x="300" y="184" text-anchor="middle">Modes · cumulative POD energy</text></g>`;
}

function renderDataset() {
  state.selected = state.records.find(record => record.id === dataset.value) || null;
  const record = state.selected, basis = record?.basis;
  byId("rom-basis-result").hidden = !basis;
  byId("rom-downloads").hidden = !record?.snapshot_count;
  byId("rom-download-basis").hidden = !basis;
  if (record) {
    const cfg = record.settings;
    byId("rom-dataset-description").textContent = `${cfg.nx} × ${cfg.ny} mesh · ${record.snapshot_count} snapshots · ${record.completed_cases}/${record.case_count} cases completed · ${record.status.replaceAll("_", " ")}.`;
    byId("rom-dataset-id").textContent = record.id.slice(0, 8);
    for (const kind of ["snapshots", "basis", "metadata"]) {
      byId(`rom-download-${kind}`).href = `/api/rom/${record.id}/artifacts/${kind}`;
    }
    byId("rom-snapshot-count").textContent = number(record.snapshot_count);
    byId("rom-case-count").textContent = `${record.completed_cases}/${record.case_count} cases completed`;
    if (record.id !== state.liveId) {
      state.previewModel = null; state.density = null;
      canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
      byId("rom-preview-empty").hidden = false;
      byId("rom-preview-empty").textContent = "Live preview appears during generation. Download saved snapshots below.";
      byId("rom-compliance").textContent = record.last_snapshot ? number(record.last_snapshot.compliance) : "—";
      byId("rom-residual").textContent = record.last_snapshot ? record.last_snapshot.residual.toExponential(1) : "—";
      byId("rom-iteration").textContent = record.last_snapshot ? `Saved snapshot at iteration ${record.last_snapshot.iteration}` : "No live snapshot loaded";
    }
    byId("rom-case-label").textContent = `Saved dataset ${record.id.slice(0, 8)} · ${record.snapshot_count} full-order snapshots`;
    byId("rom-progress").value = 100 * record.completed_cases / record.case_count;
    status(basis ? "Basis ready" : record.status === "snapshots_ready" ? "Snapshots ready" : record.status.replaceAll("_", " "),
      basis || record.status === "snapshots_ready" ? "complete" : "ready");
  } else {
    byId("rom-dataset-description").textContent = "Generate snapshots to create your first dataset.";
  }
  if (basis) {
    byId("rom-rank").textContent = basis.rank;
    byId("rom-rank-description").textContent = `of ${basis.numerical_rank} independent modes`;
    byId("rom-energy").textContent = percent(basis.captured_energy);
    byId("rom-target-status").textContent = basis.target_met ? `Target ${percent(basis.energy_target)} met` : "Mode limit reached · energy target not met";
    byId("rom-reconstruction").textContent = percent(basis.max_training_error);
    spectrum(basis);
  }
  updateButtons();
  validationView.render(record);
}

async function refreshLibrary(preferred = dataset.value) {
  if (state.busy) return;
  const response = await fetch("/api/rom/trainings");
  if (!response.ok) throw new Error("Could not load training datasets. Restart TOptVis if the server has not been updated.");
  state.records = (await response.json()).trainings;
  dataset.replaceChildren();
  for (const record of state.records) {
    const option = document.createElement("option");
    option.value = record.id;
    option.textContent = `${new Date(record.created_at).toLocaleString()} · ${record.settings.nx}×${record.settings.ny} · ${record.snapshot_count} snapshots · ${record.status.replaceAll("_", " ")}`;
    dataset.append(option);
  }
  if (!state.records.length) dataset.add(new Option("No saved datasets", ""));
  if (state.records.some(record => record.id === preferred)) dataset.value = preferred;
  renderDataset();
}

async function checkAvailability() {
  try {
    const response = await fetch("/api/health");
    if (!response.ok) throw new Error("The local solver is unavailable.");
    const health = await response.json();
    state.ready = !!health.rom_training_supported;
    state.validationSupported = !!health.rom_validation_supported;
    if (!state.busy) state.externalBusy = health.running;
    if (state.ready && !state.busy) await refreshLibrary();
  } catch (e) { state.ready = false; error(e.message); }
  updateButtons();
}

function receive(event) {
  if (event.event === "error") throw new Error(event.message);
  if (event.event === "started") {
    state.jobId = event.run_id;
    state.completedId = event.training_id;
    if (event.kind === "snapshots" || event.kind === "validation") state.liveId = event.training_id;
    updateButtons();
  } else if (event.event === "validation_case_started") {
    state.previewModel = event.config;
    validationView.caseStarted(event);
    byId("rom-case-label").textContent = `Validation case ${event.case_index+1}/${event.case_count} · node ${event.config.load_node_y} · ${event.config.load_angle}° · volume ${percent(event.config.volume_fraction)}`;
  } else if (event.event === "validation_state") {
    validationView.addResult(event);
    state.density = event.density;
    byId("rom-preview-empty").hidden = true;
    byId("rom-compliance").textContent = number(event.result.full_compliance);
    byId("rom-residual").textContent = event.result.full_residual.toExponential(1);
    byId("rom-iteration").textContent = `Validation at iteration ${event.result.iteration}`;
    byId("rom-progress").value = 100 * event.state_count / event.maximum_states;
    byId("rom-progress-note").textContent = `${event.state_count} held-out designs compared…`;
    redraw();
  } else if (event.event === "case_started") {
    state.previewModel = event.config;
    byId("rom-case-label").textContent = `Case ${event.case_index+1}/${event.case_count} · node (${event.config.load_node_x}, ${event.config.load_node_y}) · ${event.config.load_angle}° · volume ${percent(event.config.volume_fraction)}`;
  } else if (event.event === "snapshot") {
    state.density = event.density;
    byId("rom-preview-empty").hidden = true;
    byId("rom-snapshot-count").textContent = event.snapshot_count;
    byId("rom-case-count").textContent = `Case ${event.case_index+1}/${event.case_count}`;
    byId("rom-compliance").textContent = number(event.compliance);
    byId("rom-iteration").textContent = `Snapshot at iteration ${event.iteration}`;
    byId("rom-residual").textContent = event.residual.toExponential(1);
    byId("rom-progress").value = 100 * event.snapshot_count / event.maximum_snapshots;
    byId("rom-progress-note").textContent = "Collecting full-order solutions…";
    redraw();
  } else if (event.event === "saving_snapshots") {
    status("Saving snapshots", "running");
  } else if (event.event === "basis_started") {
    status("Building basis", "running");
    byId("rom-progress-note").textContent = `Computing POD from ${event.snapshot_count} displacement snapshots…`;
  } else if (event.event === "operators_started") {
    status("Preparing reduced model", "running");
    byId("rom-progress-note").textContent = `Preparing element stiffness operators for ${event.rank} modes…`;
  } else if (event.event === "complete") {
    state.terminal = event;
    complete(event);
  }
}

function complete(event) {
  const stopped = event.stop_reason === "cancelled";
  if (state.kind === "validation") {
    const report = event.training?.validation;
    status(stopped ? "Validation stopped" : report?.meets_tolerances ? "Within tolerances" : "Outside tolerances", stopped ? "ready" : "complete");
    byId("rom-progress-note").textContent = stopped ? "Validation stopped. Any collected comparisons are saved as a partial report; existing results are retained if none were collected." :
      `Validation saved · ${report?.passed_states}/${report?.state_count} designs within all tolerances.`;
    if (!stopped) byId("rom-progress").value = 100;
    return;
  }
  status(stopped ? "Stopped" : event.training?.basis ? "Basis ready" : "Snapshots ready", stopped ? "ready" : "complete");
  byId("rom-progress-note").textContent = stopped ? "Stopped. Previously saved data and collected snapshots are retained." :
    event.training?.basis ? "Basis and reduced element operators saved. Choose this model in Cantilever 2D to optimise with it." : "Snapshots saved. Build the reduced basis below.";
  if (!stopped) byId("rom-progress").value = 100;
}

async function run(endpoint, payload, kind) {
  if (state.busy || !state.ready || state.externalBusy) return;
  error(); state.jobId = null; state.completedId = null; state.terminal = null; state.kind = kind; setBusy(true);
  status(kind === "snapshots" ? "Generating snapshots" : kind === "validation" ? "Validating basis" : "Building basis", "running");
  if (kind === "validation") validationView.begin();
  if (kind === "snapshots") {
    state.density = null; state.previewModel = null;
    byId("rom-preview-empty").hidden = false;
    byId("rom-preview-empty").textContent = "A material snapshot will appear here during generation.";
    byId("rom-snapshot-count").textContent = "0";
    byId("rom-compliance").textContent = "—";
    byId("rom-residual").textContent = "—";
    byId("rom-progress").value = 0;
    canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
  }
  let terminal = false, failed = false;
  try {
    const response = await fetch(endpoint, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload)});
    if (!response.ok) {
      const details = (await response.json()).detail;
      throw new Error(Array.isArray(details) ? details.map(item => item.msg).join("; ") : details || "Could not start training.");
    }
    const reader = response.body.getReader(), decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const {value, done} = await reader.read();
      buffer += decoder.decode(value, {stream: !done});
      const lines = buffer.split("\n"); buffer = lines.pop();
      if (done && buffer.trim()) lines.push(buffer);
      for (const line of lines) {
        if (!line.trim()) continue;
        const event = JSON.parse(line); receive(event);
        if (event.event === "complete") terminal = true;
      }
      if (done) break;
      await new Promise(requestAnimationFrame);
    }
    if (!terminal) throw new Error("The connection ended before training finished. Refresh datasets to inspect saved work.");
  } catch (e) {
    failed = true;
    error(e.message); status("Error", "error");
    if (state.jobId) fetch(`/api/run/${state.jobId}/stop`, {method: "POST"}).catch(() => {});
  } finally {
    state.jobId = null; setBusy(false);
    try { await refreshLibrary(state.completedId || dataset.value); } catch (e) { error(e.message); }
    if (state.terminal) complete(state.terminal);
    if (failed) status("Error", "error");
  }
}

snapshotForm.addEventListener("input", estimate);
snapshotForm.addEventListener("submit", event => {
  event.preventDefault();
  if (!snapshotForm.reportValidity()) return;
  try { run("/api/rom/snapshots", readSettings(), "snapshots"); } catch (e) { error(e.message); }
});
basisForm.addEventListener("submit", event => {
  event.preventDefault();
  if (!basisForm.reportValidity() || !state.selected) return;
  const data = new FormData(basisForm);
  run(`/api/rom/${state.selected.id}/basis`, {energy_target: Number(data.get("energy_percent"))/100,
    max_modes: Number(data.get("max_modes")), normalize: data.has("normalize")}, "basis");
});
dataset.addEventListener("change", renderDataset);
byId("rom-refresh").addEventListener("click", () => checkAvailability());
byId("rom-stop").addEventListener("click", async () => {
  if (!state.jobId) return;
  byId("rom-stop").disabled = true;
  try {
    const response = await fetch(`/api/run/${state.jobId}/stop`, {method: "POST"});
    if (!response.ok) throw new Error("Could not stop the training job.");
    status("Stopping", "running");
  } catch (e) { error(e.message); updateButtons(); }
});
document.addEventListener("solveractivity", event => {
  if (event.detail.workspace !== "rom-training") {
    state.externalBusy = event.detail.running;
    updateButtons();
  }
});
document.addEventListener("mlmethodchange", event => {
  if (event.detail === "rom") { redraw(); if (!state.busy) checkAvailability(); }
});
new ResizeObserver(redraw).observe(canvas.parentElement);
estimate(); checkAvailability();

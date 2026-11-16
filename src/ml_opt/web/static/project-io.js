const FORMAT = "toptvis-project";

export function parseProject(text) {
  const project = JSON.parse(text);
  if (!project || project.format !== FORMAT || project.version !== 1 || !project.model || typeof project.model !== "object") throw new Error("Choose a TOptVis project file (version 1).");
  const model = project.model;
  if (!Number.isInteger(model.nx) || !Number.isInteger(model.ny) || model.nx < 1 || model.ny < 1 || model.nx*model.ny > 10_000_000) throw new Error("The project mesh is invalid or too large to import.");
  const result = project.result || null, count = model.nx*model.ny;
  const density = values => Array.isArray(values) && values.length === count && values.every(value => Number.isFinite(value) && value >= 0 && value <= 1);
  if (result) {
    if (!density(result.density) || !Array.isArray(result.history) || !result.history.length || result.history.length > 1001) throw new Error("The saved result does not match the project mesh or history.");
    let previous = -1, elapsed = -1;
    for (const row of result.history) {
      if (!row || !Number.isInteger(row.iteration) || row.iteration <= previous || row.iteration > model.max_iterations || !Number.isFinite(row.compliance) || row.compliance < 0 || !Number.isFinite(row.volume_fraction) || row.volume_fraction < 0 || row.volume_fraction > 1) throw new Error("The saved convergence history is invalid.");
      for (const value of Object.values(row)) {
        if (typeof value === "number" && !Number.isFinite(value)) throw new Error("The saved history contains nonfinite values.");
        if (value && typeof value === "object" && Object.values(value).some(item => typeof item !== "number" || !Number.isFinite(item))) throw new Error("The saved history contains an invalid result map.");
      }
      if (row.elapsed_seconds != null && (row.elapsed_seconds < elapsed || row.elapsed_seconds < 0)) throw new Error("The saved elapsed times are invalid.");
      previous = row.iteration; elapsed = row.elapsed_seconds ?? elapsed;
    }
    if (result.fields && (!model.robust || Object.keys(result.fields).length !== 3 || !["eroded","nominal","dilated"].every(key => density(result.fields[key])))) throw new Error("The saved robust fields are invalid.");
    if (model.robust && !result.fields) throw new Error("The saved robust result is missing material fields.");
    if (model.robust && !["field_compliances","field_volumes"].every(key => ["eroded","nominal","dilated"].every(field => Number.isFinite(result.history.at(-1)[key]?.[field])))) throw new Error("The saved robust result is missing field metrics.");
    if (result.stopping_tolerances && Object.entries(result.stopping_tolerances).some(([key,value]) => !["design_tolerance","objective_tolerance","feasibility_tolerance"].includes(key) || !Number.isFinite(value) || value <= 0)) throw new Error("The saved stopping tolerances are invalid.");
  }
  return {model,result};
}

export function serialiseProject(model, result) {
  return JSON.stringify({format:FORMAT,version:1,saved_at:new Date().toISOString(),model,result},null,2);
}

export class ProjectIO {
  constructor({getModel,getResult,load,isRunning,onError}) {
    this.load = load; this.onError = onError; this.isRunning = isRunning;
    const save = document.getElementById("project-save"), open = document.getElementById("project-load");
    const file = document.getElementById("project-file");
    this.openButton = open;
    save.addEventListener("click", () => {
      try {
        const blob = new Blob([serialiseProject(getModel(),getResult())],{type:"application/json"});
        const url = URL.createObjectURL(blob), link = document.createElement("a");
        link.href = url; link.download = "TOptVis-project.json"; document.body.append(link); link.click(); link.remove();
        setTimeout(() => URL.revokeObjectURL(url),1000);
      } catch (error) { onError(error.message); }
    });
    open.addEventListener("click", () => { if (!isRunning()) file.click(); });
    file.addEventListener("change", async () => {
      const selected = file.files[0]; file.value = "";
      if (!selected || isRunning()) return;
      try {
        if (selected.size > 64*1024*1024) throw new Error("The project exceeds the 64 MiB import limit.");
        const project = parseProject(await selected.text());
        const response = await fetch("/api/project/validate",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(project.model)});
        if (!response.ok) {
          const error = await response.json();
          throw new Error(typeof error.detail === "string" ? error.detail : error.detail?.map(item => item.msg).join("; ") || "The project settings are invalid.");
        }
        const {model} = await response.json();
        if (isRunning()) throw new Error("Stop the active optimisation before loading a project.");
        const checked = parseProject(serialiseProject(model,project.result));
        load(checked.model,checked.result);
      } catch (error) { onError(error.message || "Could not load the project."); }
    });
  }
  setRunning(running) { this.openButton.disabled = running; }
}

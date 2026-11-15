const byId = id => document.getElementById(id);

export class RomChoice {
  constructor({onChange, onStateChange, isRunning}) {
    this.onChange = onChange; this.onStateChange = onStateChange; this.isRunning = isRunning;
    this.records = []; this.supported = false;
    this.mode = byId("state_solver"); this.models = byId("rom_training_id");
    for (const control of [this.mode, this.models]) control.addEventListener("input", () => this.sync());
    byId("rom-refresh-models").addEventListener("click", () => this.refresh());
    byId("rom-use-mesh").addEventListener("click", () => {
      const record = this.selected;
      if (!record || isRunning()) return;
      const oldX = Number(byId("nx").value), oldY = Number(byId("ny").value);
      const loadX = Number(byId("load_node_x").value), loadY = Number(byId("load_node_y").value);
      for (const key of ["nx", "ny", "length", "height"]) byId(key).value = record.settings[key];
      const nx = record.settings.nx, ny = record.settings.ny;
      byId("load_node_x").max = nx; byId("load_node_y").max = ny;
      byId("load_node_x").value = Math.max(1, Math.min(nx, Math.round(loadX/oldX*nx)));
      byId("load_node_y").value = Math.max(0, Math.min(ny, Math.round(loadY/oldY*ny)));
      this.sync(); onChange();
    });
    this.sync();
  }

  get isRom() { return this.mode.value === "rom"; }
  get selected() { return this.records.find(record => record.id === this.models.value); }
  get compatible() {
    const record = this.selected;
    return !!record && ["nx", "ny", "length", "height"].every(key => Number(byId(key).value) === record.settings[key]);
  }
  get allowed() { return !this.isRom || (this.supported && this.compatible); }
  signature() { return `${this.selected?.id || ""}/${this.selected?.basis?.built_at || ""}`; }

  sync() {
    byId("rom-solver-option").disabled = !this.supported;
    byId("rom-solver-options").hidden = !this.isRom;
    this.models.disabled = !this.isRom;
    this.models.required = this.isRom;
    byId("rom-refresh-models").disabled = this.isRunning();
    byId("rom-use-mesh").hidden = !this.selected || this.compatible;
    byId("rom-use-mesh").disabled = this.isRunning();
    const record = this.selected;
    byId("rom-model-description").textContent = !this.supported ? "Restart TOptVis to enable ROM optimisation." :
      !record ? "No saved basis is available. Build one in ML → ROM, then refresh ROMs." :
      !this.compatible ? `This ROM uses ${record.settings.nx} × ${record.settings.ny} elements on ${record.settings.length} × ${record.settings.height}. Use its mesh or choose a matching model.` :
      `${record.basis.rank} modes · ${record.validation ? record.validation.status === "cancelled" ? "partial validation" : record.validation.meets_tolerances ? "within tested validation tolerances" : "outside tested validation tolerances" : "not validated"}. Reduced compliance and gradients are approximate; coverage depends on training cases.`;
    this.onStateChange();
  }

  async refresh(supported = this.supported) {
    this.supported = supported;
    if (this.isRunning()) return;
    if (!supported) { this.sync(); return; }
    const previous = this.signature(), selectedId = this.models.value;
    try {
      const response = await fetch("/api/rom/trainings");
      if (!response.ok) throw new Error("Could not load saved ROM models. Refresh ROMs to retry.");
      this.records = (await response.json()).trainings.filter(record => record.status === "basis_ready" && record.basis);
      this.models.replaceChildren();
      for (const record of this.records) {
        const option = document.createElement("option"); option.value = record.id;
        option.textContent = `${record.id.slice(0,8)} · ${record.settings.nx}×${record.settings.ny} · ${record.basis.rank} modes`;
        this.models.append(option);
      }
      if (!this.records.length) this.models.add(new Option("No trained ROMs", ""));
      if (this.records.some(record => record.id === selectedId)) this.models.value = selectedId;
      else {
        const matching = this.records.find(record => ["nx", "ny", "length", "height"].every(key => Number(byId(key).value) === record.settings[key]));
        if (matching) this.models.value = matching.id;
      }
      this.sync();
      if (this.isRom && previous !== this.signature()) this.onChange();
    } catch (error) {
      this.records = [];
      this.sync();
      byId("rom-model-description").textContent = error.message;
    }
  }
}

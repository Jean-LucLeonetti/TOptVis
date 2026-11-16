const byId = id => document.getElementById(id);
const percentage = value => value == null ? "Undefined" : `${(100*value).toLocaleString("en", {maximumSignificantDigits: 4})}%`;
const number = value => value.toLocaleString("en", {maximumSignificantDigits: 5});

export class ValidationView {
  constructor({run, getDataset, onError}) {
    this.getDataset = getDataset; this.onError = onError;
    this.rows = []; this.cases = []; this.key = null; this.supported = false;
    this.form = byId("rom-validation-form");
    this.form.addEventListener("submit", event => {
      event.preventDefault();
      if (!this.form.reportValidity() || !getDataset()?.basis) return;
      try {
        const payload = Object.fromEntries(Array.from(new FormData(this.form), ([name, value]) => {
          if (["load_heights", "load_angles", "volume_fractions"].includes(name)) {
            const values = value.split(",").map(part => part.trim());
            if (values.some(part => !part || !Number.isFinite(Number(part)))) throw new Error("Validation cases need comma-separated numbers.");
            return [name, values.map(Number)];
          }
          return [name, Number(value) / (name.endsWith("_tolerance") ? 100 : 1)];
        }));
        run(`/api/rom/${getDataset().id}/validate`, payload, "validation");
      } catch (error) { onError(error.message); }
    });
  }

  setDisabled(disabled, supported) {
    this.supported = supported;
    byId("rom-validation-settings").disabled = disabled;
    byId("rom-validate").disabled = disabled || !supported || !this.getDataset()?.basis;
    if (!supported) byId("rom-validation-scope").textContent = "Restart TOptVis, then refresh to enable held-out validation.";
  }

  begin() {
    this.key = null; this.rows = []; this.cases = [];
    byId("rom-validation-rows").replaceChildren();
    byId("rom-validation-results").hidden = false;
    byId("rom-validation-downloads").hidden = true;
    this.summary(null);
    byId("rom-validation-note").textContent = "Comparing validation cases...";
    byId("rom-validation-verdict").textContent = "Validating";
  }

  caseStarted(event) { this.cases[event.case_index] = event.config; }

  addResult(event) {
    this.rows.push(event.result); this.appendRow(event.result);
    const metrics = {};
    for (const name of ["displacement", "compliance", "gradient"]) {
      const values = this.rows.map(row => row[`${name}_error`]).filter(value => value != null);
      metrics[name] = {max: values.length ? Math.max(...values) : null,
        mean: values.length ? values.reduce((a,b) => a+b, 0)/values.length : null};
    }
    this.summary({metrics});
    byId("rom-validation-note").textContent = `${this.rows.length} designs compared · ${this.rows.filter(row => row.passed).length} within tolerances so far.`;
  }

  summary(report) {
    for (const name of ["displacement", "compliance", "gradient"]) {
      byId(`rom-validation-${name}`).textContent = report ? percentage(report.metrics[name].max) : "—";
      byId(`rom-validation-${name}-mean`).textContent = report ? `Mean ${percentage(report.metrics[name].mean)}` : "Waiting for comparisons";
    }
  }

  appendRow(row) {
    const config = this.cases[row.case_index];
    const label = config ? `${row.case_index+1} · node ${config.load_node_y} · ${config.load_angle}° · ${percentage(config.volume_fraction)}` : `Case ${row.case_index+1}`;
    const tr = document.createElement("tr");
    const values = [label, row.iteration, number(row.full_compliance), number(row.rom_compliance),
      percentage(row.displacement_error), percentage(row.compliance_error), percentage(row.gradient_error),
      row.rom_residual.toExponential(2), row.passed ? "Within" : "Outside"];
    for (const [index, value] of values.entries()) {
      const cell = document.createElement("td"); cell.textContent = value;
      if (index === values.length-1) cell.dataset.passed = row.passed;
      tr.append(cell);
    }
    byId("rom-validation-rows").append(tr);
  }

  async render(record) {
    const summary = record?.validation, key = summary ? `${record.id}/${summary.id}` : null;
    if (this.supported) byId("rom-validation-scope").textContent = record?.basis ?
      "" : record ? "Build a basis for this dataset to validate it." : "";
    if (key && key === this.key) return;
    this.key = key; this.rows = []; this.cases = [];
    byId("rom-validation-rows").replaceChildren();
    byId("rom-validation-results").hidden = !summary;
    byId("rom-validation-downloads").hidden = !summary;
    if (!summary) {
      byId("rom-validation-verdict").textContent = "Not validated";
      byId("rom-validation-note").textContent = "";
      return;
    }
    this.summary(summary);
    const verdict = summary.status === "cancelled" ? "Partial validation" : summary.meets_tolerances ? "Within tolerances" : "Outside tolerances";
    byId("rom-validation-verdict").textContent = verdict;
    byId("rom-validation-verdict").dataset.status = summary.meets_tolerances ? "complete" : "ready";
    byId("rom-validation-note").textContent = `${summary.passed_states}/${summary.state_count} designs within tolerances. ${summary.completed_cases}/${summary.case_count} cases completed. Results apply to tested designs.`;
    byId("rom-validation-note").title = `Tested tolerances: displacement ${percentage(summary.options.displacement_tolerance)}, compliance ${percentage(summary.options.compliance_tolerance)}, gradient ${percentage(summary.options.gradient_tolerance)}.`;
    byId("rom-validation-provenance").textContent = "";
    for (const artifact of ["validation", "comparisons"]) byId(`rom-download-${artifact}`).href = `/api/rom/${record.id}/artifacts/${artifact}`;
    try {
      const response = await fetch(`/api/rom/${record.id}/validation`);
      if (!response.ok) throw new Error("Could not load the saved validation report.");
      const report = await response.json();
      if (this.key !== key) return;
      this.cases = report.cases; this.rows = report.results;
      for (const row of report.results) this.appendRow(row);
    } catch (error) { if (this.key === key) this.onError(error.message); }
  }
}

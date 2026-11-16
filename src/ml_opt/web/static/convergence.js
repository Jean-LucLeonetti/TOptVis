// Convergence uses the evaluated history rows from the solver's live stream.
// Only scalar history is retained; density frames remain in the topology view.
const number = value => new Intl.NumberFormat("en", {maximumSignificantDigits: 5}).format(value);
const tick = value => value !== 0 && (Math.abs(value) < .001 || Math.abs(value) >= 100000)
  ? value.toExponential(1).replace(".0e", "e").replace("e-", "e−") : number(value);
const clamp = (value, lower, upper) => Math.max(lower, Math.min(upper, value));
const specifications = [
  {id: "compliance", label: "Compliance", key: "compliance", color: "#196964", zero: true,
   value: row => row.compliance, format: number},
  {id: "volume", label: "Physical material volume", key: "volume_fraction", color: "#7961b2",
   value: row => 100 * row.volume_fraction, format: value => `${number(value)}%`,
   reference: model => 100 * model.volume_fraction},
  {id: "design", label: "Design change", key: "design_change", color: "#d27942", zero: true, changes: true,
   value: row => row.design_change, format: number, reference: model => model.design_tolerance},
  {id: "objective", label: "Relative objective change", key: "relative_objective_change", color: "#4b7eaf", zero: true, changes: true,
   value: row => row.relative_objective_change, format: number, reference: model => model.objective_tolerance},
  {id: "elapsed", label: "Iteration time in seconds", color: "#748c41", zero: true, changes: true, bars: true,
   value: (row, times) => times.get(row.iteration), format: value => `${number(value)} s`,
   initialText: "Iteration 0 · setup and initial evaluation are excluded",
   unavailable: "Timing unavailable for this iteration"},
];

function setText(element, text) {
  if (element.textContent !== text) element.textContent = text;
}

export class ConvergencePlots {
  constructor() {
    this.history = [];
    this.iterationTimes = new Map();
    this.model = {};
    this.selected = null;
    this.plots = specifications.map(specification => ({...specification,
      canvas: document.getElementById(`convergence-${specification.id}`),
      readout: document.getElementById(`convergence-${specification.id}-readout`),
      logToggle: document.getElementById(`convergence-${specification.id}-log`),
      logarithmic: false,
    }));
    for (const plot of this.plots) {
      plot.logarithmic = plot.logToggle.checked;
      plot.logToggle.addEventListener("change", () => {
        plot.logarithmic = plot.logToggle.checked;
        this.drawPlot(plot);
      });
      plot.canvas.addEventListener("pointermove", event => {
        if (!plot.area || !this.history.length) return;
        const x = event.clientX - plot.canvas.getBoundingClientRect().left;
        const iteration = plot.area.minIteration + clamp((x-plot.area.left)/plot.area.width, 0, 1) * (plot.area.maxIteration-plot.area.minIteration);
        this.selected = this.history.reduce((best, row, index) =>
          Math.abs(row.iteration-iteration) < Math.abs(this.history[best].iteration-iteration) ? index : best, 0);
        this.draw();
      });
      plot.canvas.addEventListener("pointerleave", () => { this.selected = null; this.draw(); });
      plot.canvas.addEventListener("focus", () => {
        this.selected = this.history.length ? this.history.length-1 : null;
        this.draw();
      });
      plot.canvas.addEventListener("blur", () => { this.selected = null; this.draw(); });
      plot.canvas.addEventListener("keydown", event => {
        if (!this.history.length || !["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
        event.preventDefault();
        const current = this.selected ?? this.history.length-1;
        this.selected = event.key === "Home" ? 0 : event.key === "End" ? this.history.length-1 :
          clamp(current + (event.key === "ArrowRight" ? 1 : -1), 0, this.history.length-1);
        this.draw();
      });
    }
  }

  render(history, model) {
    if (history !== this.history || this.selected >= history.length) this.selected = null;
    this.history = history;
    this.iterationTimes = new Map(history.map((row, index) => {
      const previous = history[index-1];
      const valid = previous && row.iteration === previous.iteration+1 &&
        Number.isFinite(row.elapsed_seconds) && Number.isFinite(previous.elapsed_seconds) &&
        row.elapsed_seconds >= previous.elapsed_seconds;
      return [row.iteration, valid ? row.elapsed_seconds-previous.elapsed_seconds : NaN];
    }));
    this.model = model;
    this.draw();
  }

  draw() {
    for (const plot of this.plots) this.drawPlot(plot);
  }

  drawPlot(plot) {
    const {canvas, readout} = plot;
    const bounds = canvas.getBoundingClientRect();
    const width = bounds.width, height = bounds.height;
    if (width < 10 || height < 10) return;
    const ratio = window.devicePixelRatio || 1;
    const pixelWidth = Math.round(width*ratio), pixelHeight = Math.round(height*ratio);
    if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
      canvas.width = pixelWidth; canvas.height = pixelHeight;
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, width, height);
    const valueOf = row => plot.value(row, this.iterationTimes);
    const valid = row => (!plot.changes || row.iteration > 0) && Number.isFinite(valueOf(row)) &&
      (!plot.logarithmic || valueOf(row) > 0);
    const points = this.history.filter(valid);
    const reference = plot.reference?.(this.model);
    const showReference = Number.isFinite(reference) && (!plot.logarithmic || reference > 0);
    const left = 66, right = 18, top = 18, bottom = 44;
    const plotWidth = Math.max(1, width-left-right), plotHeight = Math.max(1, height-top-bottom);
    const maxIteration = Math.max(1, this.history.at(-1)?.iteration || 0);
    const minX = plot.bars ? .5 : 0, maxX = maxIteration + (plot.bars ? .5 : 0);
    plot.area = {left, width: plotWidth, minIteration: minX, maxIteration: maxX};
    const values = points.map(valueOf);
    if (showReference) values.push(reference);
    let low = values.length ? Math.min(...values) : 0;
    let high = values.length ? Math.max(...values) : 1;
    const padding = high-low > Math.max(1e-14, Math.abs(high)*1e-12) ? (high-low)*.1 : Math.max(Math.abs(high)*.06, .001);
    low = plot.zero ? 0 : Math.max(0, low-padding);
    high = Math.max(high+padding, low+1e-12);
    if (plot.logarithmic) {
      const logs = values.map(Math.log10);
      low = logs.length ? Math.min(...logs) : 0;
      high = logs.length ? Math.max(...logs) : 1;
      const logPadding = high > low ? (high-low)*.1 : .5;
      low = Math.max(Math.log10(Number.MIN_VALUE), low-logPadding);
      high = Math.min(Math.log10(Number.MAX_VALUE), high+logPadding);
    }
    const x = iteration => left + (iteration-minX)/(maxX-minX)*plotWidth;
    const y = value => top + (high-(plot.logarithmic ? Math.log10(value) : value))/(high-low)*plotHeight;

    ctx.font = "12px 'Segoe UI',sans-serif";
    ctx.lineWidth = 1;
    ctx.textAlign = "right";
    for (let index = 0; index <= 4; index++) {
      const axisValue = low+(high-low)*index/4;
      const value = plot.logarithmic ? Math.min(10**axisValue,Number.MAX_VALUE) : axisValue;
      const py = top + (1-index/4)*plotHeight;
      ctx.strokeStyle = "#eaf0f2";
      ctx.beginPath(); ctx.moveTo(left, py); ctx.lineTo(left+plotWidth, py); ctx.stroke();
      ctx.fillStyle = "#81949c"; ctx.fillText(tick(value), left-8, py+3);
    }
    ctx.textAlign = "center";
    const firstTick = plot.bars ? 1 : 0;
    const tickCount = Math.min(4, maxIteration-firstTick);
    for (let index = 0; index <= tickCount; index++) {
      const iteration = firstTick + Math.round((maxIteration-firstTick)*index/Math.max(1,tickCount));
      ctx.fillStyle = "#81949c"; ctx.fillText(String(iteration), x(iteration), top+plotHeight+16);
    }
    ctx.fillText("Outer iteration", left+plotWidth/2, height-5);

    if (showReference) {
      ctx.strokeStyle = "#a39a85"; ctx.setLineDash([5, 4]);
      ctx.beginPath(); ctx.moveTo(left, y(reference)); ctx.lineTo(left+plotWidth, y(reference)); ctx.stroke();
      ctx.setLineDash([]);
    }
    if (points.length && plot.bars) {
      const barWidth = Math.max(.5, Math.min(36, .8*plotWidth/maxIteration));
      ctx.fillStyle = plot.color;
      const baseline = plot.logarithmic ? top+plotHeight : y(0);
      for (const row of points) ctx.fillRect(x(row.iteration)-barWidth/2, y(valueOf(row)), barWidth, baseline-y(valueOf(row)));
    } else if (points.length) {
      ctx.strokeStyle = plot.color; ctx.lineWidth = 2;
      ctx.beginPath();
      let connected = false;
      for (const row of this.history) {
        if (!valid(row)) { connected = false; continue; }
        if (connected) ctx.lineTo(x(row.iteration), y(valueOf(row)));
        else ctx.moveTo(x(row.iteration), y(valueOf(row)));
        connected = true;
      }
      ctx.stroke();
      const latest = points.at(-1);
      ctx.beginPath(); ctx.arc(x(latest.iteration), y(valueOf(latest)), 3, 0, 2*Math.PI);
      ctx.fillStyle = plot.color; ctx.fill();
    } else {
      ctx.fillStyle = "#93a2a9";
      ctx.fillText(plot.logarithmic && this.history.some(row => (!plot.changes || row.iteration > 0) && Number.isFinite(valueOf(row))) ? "No positive values for log scale" :
        plot.changes && this.history.length === 1 ? "Waiting for iteration 1" :
        plot.unavailable && this.history.length ? "Timing unavailable for this run" :
        plot.changes && this.history.length ? "Waiting for iteration 1" : "Run an optimisation to see this graph", left+plotWidth/2, top+plotHeight/2);
    }

    const selected = this.history[this.selected ?? this.history.length-1];
    if (this.selected !== null && selected && (!plot.bars || selected.iteration > 0)) {
      ctx.strokeStyle = "#b4c3ca"; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(x(selected.iteration), top); ctx.lineTo(x(selected.iteration), top+plotHeight); ctx.stroke();
      if (valid(selected)) {
        ctx.beginPath(); ctx.arc(x(selected.iteration), y(valueOf(selected)), 4, 0, 2*Math.PI);
        ctx.fillStyle = plot.color; ctx.fill(); ctx.strokeStyle = "white"; ctx.lineWidth = 1.5; ctx.stroke();
      }
    }
    const text = !selected ? "No evaluated iterations yet" : plot.changes && selected.iteration === 0 ?
      plot.initialText || "Iteration 0 · changes start at iteration 1" : !Number.isFinite(valueOf(selected)) ?
      plot.unavailable || "Value unavailable for this iteration" : `Iteration ${selected.iteration} · ${plot.format(valueOf(selected))}`;
    const scaleNote = plot.logarithmic && selected && (!plot.changes || selected.iteration > 0) &&
      Number.isFinite(valueOf(selected)) && valueOf(selected) <= 0 ? " · Not shown on log scale" : "";
    setText(readout, text+scaleNote);
    canvas.setAttribute("aria-label", `${plot.label} against outer iteration. ${plot.logarithmic ? "Logarithmic" : "Linear"} vertical scale. ${text}${scaleNote}. Use left and right arrows to inspect iterations.`);
  }
}

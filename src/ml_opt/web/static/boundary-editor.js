import {pickNode} from "./viewer.js";
import {makeSelection, selectionBounds, selectionNodes, defaultConditions, boundaryError, regionElements, nonDesignError} from "./mesh-selection.js";

import {connectionError,connectionElements} from "./connections.js";

const byId = id => document.getElementById(id);

export class BoundaryEditor {
  constructor({canvas, getModel, isRunning, onChange, onError, onOpen, redraw}) {
    Object.assign(this, {canvas, getModel, isRunning, onChange, onError, onOpen, redraw});
    this.conditions = null; this.selection = null; this.anchor = null; this.hover = null;
    this.regions = []; this.editingRegion = null;
    this.connections = []; this.editingConnection = null;
    this.cases = [{id:"case-1",name:"Load case 1",weight:1}]; this.activeCase = "case-1"; this.results = {};
    byId("load-aggregation").value = "weighted"; byId("boundary-components").value = "xy";
    this.mode = "point"; this.editing = null; this.drag = null; this.suppressClick = false;
    this.cursor = {x: getModel().nx, y: Math.round(getModel().ny/2)};
    this.tools = [...document.querySelectorAll('[data-selection-mode]')];
    for (const tool of this.tools) tool.addEventListener("click", () => {
      if (isRunning() || (["solid","void"].includes(byId("boundary-kind").value) && tool.dataset.selectionMode !== "surface")) return;
      const editing = this.editing, editingRegion = this.editingRegion, editingConnection = this.editingConnection;
      this.mode = tool.dataset.selectionMode;
      this.clearSelection(); this.editing = editing; this.editingRegion = editingRegion; this.editingConnection = editingConnection; this.render(); onOpen();
    });
    byId("selection-clear").addEventListener("click", () => this.clearSelection());
    byId("boundary-kind").addEventListener("change", () => {
      if (["solid","void"].includes(byId("boundary-kind").value)) {
        this.mode = "surface";
        if (this.selection?.kind !== "surface") this.clearSelection();
      }
      this.render(); this.redraw();
    });
    byId("boundary-components").addEventListener("change", () => this.render());
    byId("active-load-case").addEventListener("change", () => { this.activeCase = byId("active-load-case").value; this.editing = null; this.render(); redraw(); });
    byId("case-add").addEventListener("click", () => this.addCase());
    byId("case-remove").addEventListener("click", () => this.removeCase());
    for (const id of ["case-name","case-weight","load-aggregation"]) byId(id).addEventListener("change", () => {
      if (isRunning() || !byId("case-name").reportValidity() || !byId("case-weight").reportValidity()) return;
      const item = this.cases.find(item => item.id === this.activeCase);
      if (!byId("case-name").value.trim()) { onError("Enter a load case name."); return; }
      item.name = byId("case-name").value.trim(); item.weight = Number(byId("case-weight").value);
      onChange(); this.render();
    });
    byId("boundary-apply").addEventListener("click", () => this.apply());
    byId("boundary-reset").addEventListener("click", () => {
      if (isRunning()) return;
      this.conditions = null; this.cases = [{id:"case-1",name:"Load case 1",weight:1}]; this.activeCase = "case-1";
      byId("load-aggregation").value = "weighted"; this.editing = null; this.clearSelection(); onChange();
    });
    byId("boundary-select-coordinates").addEventListener("click", () => this.selectCoordinates());
    byId("panel-load").addEventListener("input", event => {if(!event.target.name?.startsWith("connection_")) event.stopPropagation();});
    byId("boundary-editor-form").addEventListener("submit", event => event.preventDefault());
    canvas.addEventListener("click", event => {
      if (this.suppressClick) { this.suppressClick = false; return; }
      const node = pickNode(canvas, getModel(), event);
      if (!isRunning() && node) this.choose(node);
    });
    canvas.addEventListener("pointerdown", event => {
      if (isRunning() || event.button !== 0) return;
      const node = pickNode(canvas, getModel(), event);
      if (!node) return;
      this.suppressClick = false;
      this.drag = {node, x: event.clientX, y: event.clientY, moved: false};
      canvas.setPointerCapture(event.pointerId);
    });
    canvas.addEventListener("pointermove", event => {
      if (isRunning()) return;
      this.hover = pickNode(canvas, getModel(), event);
      if (this.drag && Math.hypot(event.clientX-this.drag.x, event.clientY-this.drag.y) > 4) this.drag.moved = true;
      redraw();
    });
    canvas.addEventListener("pointerup", event => {
      const drag = this.drag;
      const node = pickNode(canvas, getModel(), event);
      this.drag = null;
      if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
      if (!drag || !drag.moved || isRunning()) return;
      this.suppressClick = true;
      if (node) {
        this.selection = makeSelection(this.mode, this.mode === "point" ? node : drag.node, node, getModel());
        this.anchor = null; this.render(); redraw(); onOpen();
      } else { this.render(); redraw(); }
    });
    canvas.addEventListener("pointercancel", () => { this.drag = null; this.suppressClick = true; this.anchor = null; redraw(); });
    canvas.addEventListener("pointerleave", () => { if (!this.drag) this.hover = null; redraw(); });
    canvas.addEventListener("keydown", event => {
      if (isRunning()) return;
      if (event.key === "Escape") { event.preventDefault(); this.clearSelection(); return; }
      const movement = {ArrowLeft: [-1,0], ArrowRight: [1,0], ArrowUp: [0,1], ArrowDown: [0,-1]}[event.key];
      if (movement) {
        event.preventDefault();
        this.cursor = {x: Math.max(0,Math.min(getModel().nx,this.cursor.x+movement[0])), y: Math.max(0,Math.min(getModel().ny,this.cursor.y+movement[1]))};
        this.hover = {...this.cursor}; redraw();
      } else if (event.key === "Enter" || event.key === " ") { event.preventDefault(); this.choose(this.cursor); }
    });
    this.render();
  }

  get aggregation() { return byId("load-aggregation").value; }
  get validationError() { return boundaryError(this.conditions, {...this.getModel(),load_cases:this.cases}) || nonDesignError(this.regions,this.getModel()) || connectionError(this.connections,this.getModel(),this.regions); }
  addCase() {
    if (this.isRunning() || this.cases.length >= 16) return;
    const current = this.conditions || defaultConditions(this.getModel());
    if (current.length + current.filter(item => item.kind === "force" && (item.case_id || this.cases[0].id) === this.activeCase).length > 128) { this.onError("At most 128 boundary conditions can be assigned."); return; }
    this.conditions = structuredClone(this.conditions || defaultConditions(this.getModel()));
    for (const condition of this.conditions) if (condition.kind === "force" && !condition.case_id) condition.case_id = this.cases[0].id;
    let index = 1; while (this.cases.some(item => item.id === `case-${index}`)) index++;
    const id = `case-${index}`;
    this.conditions.push(...this.conditions.filter(item => item.kind === "force" && item.case_id === this.activeCase).map(item => ({...structuredClone(item),case_id:id})));
    this.cases.push({id,name:`Load case ${index}`,weight:1}); this.activeCase = id; this.editing = null;
    this.onChange(); this.render(); this.redraw();
  }
  removeCase() {
    if (this.isRunning() || this.cases.length === 1) return;
    const removed = this.activeCase, first = this.cases[0].id;
    this.conditions = structuredClone(this.conditions || defaultConditions(this.getModel())).filter(item => item.kind !== "force" || (item.case_id || first) !== removed);
    this.cases = this.cases.filter(item => item.id !== removed); this.activeCase = this.cases[0].id;
    this.editing = null; this.onChange(); this.render(); this.redraw();
  }
  loadModel(model) {
    this.conditions = structuredClone(model.boundary_conditions || null);
    this.regions = structuredClone(model.non_design_regions || []);
    this.connections = [...(model.connection_anchors || []).map(selection=>({kind:"connection_anchor",selection:structuredClone(selection)})),
      ...(model.connection_targets || []).map(selection=>({kind:"connection_target",selection:structuredClone(selection)}))];
    this.cases = structuredClone(model.load_cases || [{id:"case-1",name:"Load case 1",weight:1}]);
    this.activeCase = this.cases[0].id; byId("load-aggregation").value = model.load_aggregation || "weighted";
    this.clearSelection();
  }
  setResults(values) {
    this.results = values || {};
    const value = this.results[this.activeCase];
    byId("case-result").textContent = Number.isFinite(value) ? `Case objective: ${new Intl.NumberFormat("en",{maximumSignificantDigits:6}).format(value)}` : "";
  }
  get overlay() {
    const start = this.drag?.moved ? this.drag.node : this.anchor;
    return {selection: start && this.hover ? makeSelection(this.mode, start, this.hover, this.getModel()) : this.selection,
      hover: this.hover || this.anchor, draft: !!start};
  }
  clearSelection() {
    this.selection = null; this.anchor = null; this.drag = null; this.hover = null; this.editing = null;
    this.editingRegion = null; this.editingConnection = null;
    this.render(); this.redraw();
  }
  choose(node) {
    this.cursor = {...node};
    if (this.mode === "point") this.selection = makeSelection(this.mode, node, node, this.getModel());
    else if (!this.anchor) { this.anchor = {...node}; this.selection = null; }
    else {
      const selection = makeSelection(this.mode, this.anchor, node, this.getModel());
      if (selection) { this.selection = selection; this.anchor = null; }
    }
    this.render(); this.redraw(); this.onOpen();
  }
  selectCoordinates() {
    if (this.isRunning()) return;
    const fields = ["selection-x0", "selection-y0", "selection-x1", "selection-y1"].map(byId);
    const values = fields.map(field => Number(field.value));
    if (values.some((value, index) => !fields[index].value.trim() || !Number.isInteger(value) || value < 0 || value > (index%2 ? this.getModel().ny : this.getModel().nx))) {
      this.onError("Selection coordinates must be valid mesh columns and rows."); return;
    }
    const selection = makeSelection(this.mode, {x:values[0],y:values[1]}, {x:values[2],y:values[3]}, this.getModel());
    if (!selection) { this.onError(this.mode === "surface" ? "Select opposite corners spanning at least one element." : "Select distinct endpoints along a mesh line."); return; }
    this.selection = selection;
    this.anchor = null; this.render(); this.redraw();
  }
  apply() {
    if (this.isRunning() || !this.selection) return;
    const condition = {kind: byId("boundary-kind").value, selection: structuredClone(this.selection)};
    if (["connection_anchor","connection_target"].includes(condition.kind)) {
      const items=structuredClone(this.connections);
      if(this.editingConnection===null) items.push(condition);else items[this.editingConnection]=condition;
      if(items.filter(item=>item.kind===condition.kind).length>16) {this.onError("At most 16 connection regions of each type can be assigned.");return;}
      if(items.some(item=>item.kind==="connection_anchor") && items.some(item=>item.kind==="connection_target")) {
        const issue=connectionError(items,this.getModel(),this.regions);if(issue) {this.onError(issue);return;}
      }
      this.connections=items;this.editingConnection=null;this.editingRegion=null;this.editing=null;
      this.onChange();this.render();this.redraw();return;
    }
    this.editingConnection=null;
    if (["solid","void"].includes(condition.kind)) {
      const regions = structuredClone(this.regions);
      if (this.editingRegion === null && regions.length >= 128) { this.onError("At most 128 non-design regions can be assigned."); return; }
      if (this.editingRegion === null) regions.push(condition);
      else regions[this.editingRegion] = condition;
      const issue = nonDesignError(regions,this.getModel());
      if (issue) { this.onError(issue); return; }
      this.regions = regions; this.editingRegion = null; this.editing = null;
      this.onChange(); this.render(); this.redraw(); return;
    }
    if (condition.kind === "force") {
      const magnitude = byId("boundary-magnitude"), angle = byId("boundary-angle");
      if (!magnitude.reportValidity() || !angle.reportValidity()) return;
      condition.magnitude = Number(magnitude.value); condition.angle = Number(angle.value);
      condition.case_id = this.activeCase;
    } else if (condition.kind === "displacement") {
      for (const id of ["boundary-ux","boundary-uy"]) if (!byId(id).disabled && !byId(id).reportValidity()) return;
      condition.components = byId("boundary-components").value;
      condition.ux = condition.components.includes("x") ? Number(byId("boundary-ux").value) : 0;
      condition.uy = condition.components.includes("y") ? Number(byId("boundary-uy").value) : 0;
    }
    const conditions = structuredClone(this.conditions || defaultConditions(this.getModel()));
    if (this.editing === null && conditions.length >= 128) { this.onError("At most 128 boundary conditions can be assigned."); return; }
    if (this.editing !== null) conditions[this.editing] = condition;
    else conditions.push(condition);
    this.conditions = conditions; this.editing = null; this.editingRegion = null;
    this.onChange(); this.render(); this.redraw();
  }
  render() {
    const model = this.getModel(), running = this.isRunning();
    const caseSelect = byId("active-load-case"); caseSelect.replaceChildren();
    for (const item of this.cases) { const option = document.createElement("option"); option.value = item.id; option.textContent = item.name; caseSelect.append(option); }
    caseSelect.value = this.activeCase; caseSelect.disabled = running;
    const active = this.cases.find(item => item.id === this.activeCase);
    byId("case-name").value = active.name; byId("case-weight").value = active.weight;
    byId("case-weight").disabled = running || this.aggregation === "worst";
    byId("case-add").disabled = running || this.cases.length >= 16;
    byId("case-remove").disabled = running || this.cases.length === 1;
    this.setResults(this.results);
    this.cursor = {x: Math.min(model.nx,this.cursor.x), y: Math.min(model.ny,this.cursor.y)};
    const meshKey = `${model.nx}/${model.ny}`;
    if (this.meshKey && this.meshKey !== meshKey) { this.anchor = null; this.hover = null; this.drag = null; }
    this.meshKey = meshKey;
    const region = ["solid","void"].includes(byId("boundary-kind").value);
    for (const tool of this.tools) { tool.setAttribute("aria-pressed", String(tool.dataset.selectionMode === this.mode)); tool.disabled = running || (region && tool.dataset.selectionMode !== "surface"); }
    byId("selection-clear").disabled = running || (!this.selection && !this.anchor);
    byId("boundary-apply").disabled = running || !this.selection;
    byId("boundary-apply").textContent = this.editing === null && this.editingRegion === null && this.editingConnection === null ? "Assign" : "Update";
    byId("boundary-reset").disabled = running || (this.conditions === null && this.cases.length === 1);
    byId("selection-end-fields").hidden = this.mode === "point";
    byId("boundary-force-fields").hidden = byId("boundary-kind").value !== "force";
    byId("boundary-magnitude").disabled = running || byId("boundary-kind").value !== "force";
    byId("boundary-angle").disabled = running || byId("boundary-kind").value !== "force";
    const motion = byId("boundary-kind").value === "displacement", components = byId("boundary-components").value;
    byId("boundary-displacement-fields").hidden = !motion;
    byId("boundary-ux").disabled = running || !motion || !components.includes("x");
    byId("boundary-uy").disabled = running || !motion || !components.includes("y");
    byId("boundary-select-coordinates").textContent = region ? "Select elements" : "Select nodes";
    byId("selection-count").textContent = this.selection ? region ? `${regionElements(this.selection,model).length} elements selected` : `${selectionNodes(this.selection,model).length} nodes selected` : this.anchor ? "Select an endpoint" : "";
    const bounds = this.selection ? selectionBounds(this.selection,model) : {x0:0,y0:0,x1:0,y1:0};
    for (const [id,key] of [["selection-x0","x0"],["selection-y0","y0"],["selection-x1","x1"],["selection-y1","y1"]]) {
      byId(id).max = key[0] === "x" ? model.nx : model.ny;
      byId(id).value = bounds[key];
    }
    const list = byId("boundary-list"); list.replaceChildren();
    for (const [index, condition] of (this.conditions || defaultConditions(model)).entries()) {
      if (condition.kind === "force" && (condition.case_id || this.cases[0].id) !== this.activeCase) continue;
      const row = document.createElement("li"); row.className = "boundary-row";
      const edit = document.createElement("button"); edit.type = "button"; edit.className = "boundary-item"; edit.disabled = running;
      edit.setAttribute("aria-pressed", String(this.editing === index));
      const title = document.createElement("strong"); title.textContent = {fixed:"Fixed support",restrain_x:"Restrain x",restrain_y:"Restrain y",displacement:"Prescribed motion",force:"Force"}[condition.kind];
      const detail = document.createElement("span"); detail.textContent = `${condition.selection.kind} / ${selectionNodes(condition.selection,model).length} nodes${condition.kind === "force" ? ` / ${condition.magnitude} at ${condition.angle} degrees` : condition.kind === "displacement" ? ` / ${condition.components}: (${condition.ux}, ${condition.uy})` : ""}`;
      edit.append(title,detail);
      edit.addEventListener("click", () => {
        if (this.isRunning()) return;
        this.editing = index; this.selection = structuredClone(condition.selection); this.anchor = null;
        this.editingRegion = null;this.editingConnection=null;
        this.mode = condition.selection.kind; byId("boundary-kind").value = condition.kind;
        byId("boundary-magnitude").value = condition.magnitude ?? 1; byId("boundary-angle").value = condition.angle ?? -90;
        byId("boundary-components").value = condition.components || "xy";
        byId("boundary-ux").value = condition.ux || 0; byId("boundary-uy").value = condition.uy || 0;
        this.render(); this.redraw();
      });
      const remove = document.createElement("button"); remove.type = "button"; remove.className = "boundary-remove";
      remove.setAttribute("aria-label", `Remove ${title.textContent.toLowerCase()} ${index+1}`);
      remove.textContent = "Remove"; remove.disabled = running;
      remove.addEventListener("click", () => {
        if (this.isRunning()) return;
        this.conditions = structuredClone(this.conditions || defaultConditions(this.getModel()));
        this.conditions.splice(index,1); this.editing = null; this.onChange(); this.render(); this.redraw();
      });
      row.append(edit,remove); list.append(row);
    }
    const regionList = byId("non-design-list"); regionList.replaceChildren();
    for (const [index,item] of this.regions.entries()) {
      const row=document.createElement("li"); row.className="boundary-row";
      const edit=document.createElement("button"); edit.type="button"; edit.className="boundary-item"; edit.disabled=running;
      edit.setAttribute("aria-pressed",String(this.editingRegion===index));
      const title=document.createElement("strong"); title.textContent=item.kind==="solid" ? "Fixed solid" : "Fixed void";
      const detail=document.createElement("span"); detail.textContent=`${regionElements(item.selection,model).length} elements`;
      edit.append(title,detail);
      edit.addEventListener("click",()=>{
        if(this.isRunning()) return;
        this.editingRegion=index;this.editing=null;this.editingConnection=null;this.selection=structuredClone(item.selection);this.anchor=null;
        this.mode="surface";byId("boundary-kind").value=item.kind;this.render();this.redraw();
      });
      const remove=document.createElement("button");remove.type="button";remove.className="boundary-remove";remove.textContent="Remove";remove.disabled=running;
      remove.setAttribute("aria-label",`Remove ${title.textContent.toLowerCase()} region ${index+1}`);
      remove.addEventListener("click",()=>{
        if(this.isRunning()) return;
        this.regions.splice(index,1);this.editingRegion=null;this.onChange();this.render();this.redraw();
      });
      row.append(edit,remove);regionList.append(row);
    }
    const connectionList=byId("connection-list");connectionList.replaceChildren();
    for(const [index,item] of this.connections.entries()) {
      const row=document.createElement("li");row.className="boundary-row";
      const edit=document.createElement("button");edit.type="button";edit.className="boundary-item";edit.disabled=running;
      edit.setAttribute("aria-pressed",String(this.editingConnection===index));
      const title=document.createElement("strong");title.textContent=item.kind==="connection_anchor" ? "Connection anchor" : "Connection target";
      const detail=document.createElement("span");detail.textContent=`${item.selection.kind} / ${connectionElements(item.selection,model).length} cells`;
      edit.append(title,detail);edit.addEventListener("click",()=>{
        if(this.isRunning())return;
        this.editingConnection=index;this.editing=this.editingRegion=null;this.selection=structuredClone(item.selection);this.anchor=null;
        this.mode=item.selection.kind;byId("boundary-kind").value=item.kind;this.render();this.redraw();
      });
      const remove=document.createElement("button");remove.type="button";remove.className="boundary-remove";remove.textContent="Remove";remove.disabled=running;
      remove.setAttribute("aria-label",`Remove ${title.textContent.toLowerCase()} ${index+1}`);
      remove.addEventListener("click",()=>{if(this.isRunning())return;this.connections.splice(index,1);this.editingConnection=null;this.onChange();this.render();this.redraw();});
      row.append(edit,remove);connectionList.append(row);
    }
  }
}

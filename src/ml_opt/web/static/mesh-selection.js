// Geometry and boundary data shared by the editor and canvas.
export function selectionBounds(selection, model) {
  const start = selection.start.map((value, index) => Math.round(value * [model.nx, model.ny][index]));
  const end = (selection.end || selection.start).map((value, index) => Math.round(value * [model.nx, model.ny][index]));
  return {x0: Math.min(start[0], end[0]), y0: Math.min(start[1], end[1]),
    x1: Math.max(start[0], end[0]), y1: Math.max(start[1], end[1])};
}

export function selectionNodes(selection, model) {
  const {x0, y0, x1, y1} = selectionBounds(selection, model);
  const nodes = [], total = Math.max(1, x1-x0) * Math.max(1, y1-y0);
  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      const wx = x0 !== x1 && (x === x0 || x === x1) ? .5 : 1;
      const wy = y0 !== y1 && (y === y0 || y === y1) ? .5 : 1;
      nodes.push({x, y, index: y*(model.nx+1)+x, weight: wx*wy/total});
    }
  }
  return nodes;
}

export function regionElements(selection, model) {
  const {x0,y0,x1,y1} = selectionBounds(selection,model), elements=[];
  if (selection.kind !== "surface" || x0===x1 || y0===y1) return elements;
  for(let y=y0;y<y1;y++) for(let x=x0;x<x1;x++) elements.push(y*model.nx+x);
  return elements;
}

export function nonDesignError(regions, model) {
  const fixed=new Map();
  for(const region of regions) {
    const elements=regionElements(region.selection,model);
    if(!elements.length) return "Select a surface covering at least one element for a non-design region.";
    for(const index of elements) {
      if(fixed.has(index) && fixed.get(index)!==region.kind) return "Solid and void non-design regions overlap.";
      fixed.set(index,region.kind);
    }
  }
  const count=model.nx*model.ny;
  if(fixed.size===count) return "Leave at least one element available for design.";
  const solid=[...fixed.values()].filter(kind=>kind==="solid").length;
  const voids=fixed.size-solid, target=model.volume_fraction, tolerance=model.feasibility_tolerance ?? 1e-4;
  if(model.volume_constraint!=="gt" && solid/count>target*(1+tolerance)) return "Fixed solid regions exceed the permitted material volume.";
  if(["gt","eq"].includes(model.volume_constraint) && 1-voids/count<target*(1-tolerance)) return "Fixed void regions prevent the required material volume.";
  return "";
}

export function makeSelection(kind, start, end, model) {
  let target = {...end};
  if (kind === "point") target = {...start};
  if (kind === "line") {
    if (Math.abs((end.x-start.x)*model.length/model.nx) >= Math.abs((end.y-start.y)*model.height/model.ny)) target.y = start.y;
    else target.x = start.x;
  }
  if (kind !== "point" && start.x === target.x && start.y === target.y) return null;
  if (kind === "surface" && (start.x === target.x || start.y === target.y)) return null;
  return {kind, start: [Math.min(start.x,target.x)/model.nx, Math.min(start.y,target.y)/model.ny],
    end: kind === "point" ? null : [Math.max(start.x,target.x)/model.nx, Math.max(start.y,target.y)/model.ny]};
}

export function defaultConditions(model) {
  return [
    {kind: "fixed", selection: {kind: "line", start: [0,0], end: [0,1]}},
    {kind: "force", selection: {kind: "point", start: [model.load_node_x/model.nx, model.load_node_y/model.ny], end: null},
      magnitude: model.load, angle: model.load_angle},
  ];
}

export function constrainedComponents(condition) {
  return {fixed: "xy", restrain_x: "x", restrain_y: "y"}[condition.kind] || condition.components || "xy";
}

export function loadCases(model) {
  return model.load_cases || [{id:"case-1",name:"Load case 1",weight:1}];
}

export function boundaryError(conditions, model) {
  if (conditions === null) return "";
  const fixed = new Map(), forces = new Map(loadCases(model).map(item => [item.id,new Map()]));
  const cases = loadCases(model), first = cases[0].id;
  for (const condition of conditions) {
    const bounds = selectionBounds(condition.selection, model);
    if (condition.selection.kind === "line" && bounds.x0 === bounds.x1 && bounds.y0 === bounds.y1) return "A selected line is smaller than one mesh edge. Refine the mesh or replace the selection.";
    if (condition.selection.kind === "surface" && (bounds.x0 === bounds.x1 || bounds.y0 === bounds.y1)) return "A selected surface is smaller than one element. Refine the mesh or replace the selection.";
    const target = forces.get(condition.case_id || first);
    if (condition.kind === "force" && !target) return "A force refers to an unknown load case.";
    for (const node of selectionNodes(condition.selection, model)) {
      if (condition.kind !== "force") {
        for (const [component,letter] of [..."xy"].entries()) {
          if (!constrainedComponents(condition).includes(letter)) continue;
          const value = condition.kind === "displacement" ? Number(condition[component ? "uy" : "ux"] || 0) : 0;
          const dof = 2*node.index+component;
          if (fixed.has(dof) && Math.abs(fixed.get(dof)-value)>1e-12) return "Conflicting prescribed displacements on the same node and direction.";
          fixed.set(dof,value);
        }
      } else {
        const angle = condition.angle*Math.PI/180;
        const direction = [Math.cos(angle),Math.sin(angle)].map(value => Math.abs(value)<1e-14 ? 0 : value);
        const force = target.get(node.index) || [0,0];
        target.set(node.index,force.map((value,index) => value+direction[index]*condition.magnitude*node.weight));
      }
    }
  }
  const modes=[];
  for (const dof of fixed.keys()) {
    const node=Math.floor(dof/2), x=node%(model.nx+1)/model.nx*model.length, y=Math.floor(node/(model.nx+1))/model.ny*model.height;
    modes.push(dof%2 ? [0,1,x] : [1,0,-y]);
  }
  // Three-column Gaussian elimination checks the planar rigid-body modes.
  let rank=0;
  for (let column=0;column<3;column++) {
    let pivot=rank;
    for (let row=rank;row<modes.length;row++) if (Math.abs(modes[row][column])>Math.abs(modes[pivot]?.[column] || 0)) pivot=row;
    if (!modes[pivot] || Math.abs(modes[pivot][column])<1e-10) continue;
    [modes[rank],modes[pivot]]=[modes[pivot],modes[rank]];
    const value=modes[rank][column];
    for (let c=column;c<3;c++) modes[rank][c]/=value;
    for (let row=rank+1;row<modes.length;row++) {
      const ratio=modes[row][column];
      for (let c=column;c<3;c++) modes[row][c]-=ratio*modes[rank][c];
    }
    rank++;
  }
  if (rank<3) return "Restrain rigid motion using at least two distinct nodes and suitable directions.";
  if (fixed.size===2*(model.nx+1)*(model.ny+1)) return "Leave at least one node unconstrained.";
  const imposed=[...fixed.values()].some(value => value!==0);
  for (const item of cases) {
    if (!imposed && ![...forces.get(item.id)].some(([node,force]) => force.some((value,index) => value!==0 && !fixed.has(2*node+index)))) return `${item.name}: apply a force to at least one unconstrained direction.`;
  }
  return "";
}

export function cantileverSupports(conditions, model) {
  if (loadCases(model).length>1) return false;
  if (!conditions) return true;
  if (conditions.some(item => item.kind!=="fixed" && item.kind!=="force")) return false;
  const fixed=new Set();
  for (const condition of conditions.filter(item => item.kind === "fixed")) for (const node of selectionNodes(condition.selection,model)) fixed.add(node.index);
  return fixed.size === model.ny+1 && [...fixed].every(node => node%(model.nx+1)===0);
}

import {selectionBounds, selectionNodes, constrainedComponents, loadCases} from "./mesh-selection.js";

const FORCE = "#df7a40";
const clamp = (value, lower, upper) => Math.min(upper, Math.max(lower, value));

function surface(canvas, model, clear = true) {
  const {width, height} = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  const pixelWidth = Math.round(width * ratio), pixelHeight = Math.round(height * ratio);
  if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
    canvas.width = pixelWidth; canvas.height = pixelHeight;
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  if (clear) ctx.clearRect(0, 0, width, height);
  const scale = Math.min((width - 120) / model.length, (height - 82) / model.height);
  const w = model.length * scale, h = model.height * scale;
  return {ctx, width, height, w, h, left: (width - w) / 2, top: (height - h) / 2 - 3};
}

function support(view) {
  const {ctx, left, top, h} = view;
  ctx.fillStyle = "#e6edf0";
  ctx.fillRect(left - 11, top, 11, h);
  ctx.save();
  ctx.beginPath(); ctx.rect(left - 11, top, 11, h); ctx.clip();
  ctx.strokeStyle = "#9db0b8"; ctx.lineWidth = 1;
  ctx.beginPath();
  for (let y = top - 12; y < top + h + 12; y += 6) {
    ctx.moveTo(left - 12, y + 12); ctx.lineTo(left + 1, y - 1);
  }
  ctx.stroke(); ctx.restore();
  ctx.strokeStyle = "#397a91"; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(left,top); ctx.lineTo(left,top+h); ctx.stroke();
}

function outline(view) {
  const {ctx, left, top, w, h} = view;
  ctx.strokeStyle = "#829ca4"; ctx.lineWidth = 1;
  ctx.strokeRect(left, top, w, h);
}

function load(view, model, node = {x: model.load_node_x, y: model.load_node_y}, magnitude = model.load, degrees = model.load_angle, color = FORCE) {
  const {ctx, width, height, left, top, w, h} = view;
  const x = left + node.x / model.nx * w;
  const y = top + h - node.y / model.ny * h;
  const angle = -degrees * Math.PI / 180;
  const dx = Math.cos(angle), dy = Math.sin(angle);
  const desired = clamp(24 + 14 * Math.sqrt(magnitude), 30, 74);
  let limit = desired;
  if (dx > 1e-8) limit = Math.min(limit, (width - 15 - x) / dx);
  if (dx < -1e-8) limit = Math.min(limit, (15 - x) / dx);
  if (dy > 1e-8) limit = Math.min(limit, (height - 13 - y) / dy);
  if (dy < -1e-8) limit = Math.min(limit, (13 - y) / dy);
  const endX = x + dx * limit, endY = y + dy * limit;
  ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(endX, endY); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(endX, endY);
  ctx.lineTo(endX - 8 * Math.cos(angle - .4), endY - 8 * Math.sin(angle - .4));
  ctx.lineTo(endX - 8 * Math.cos(angle + .4), endY - 8 * Math.sin(angle + .4));
  ctx.closePath(); ctx.fill();
  ctx.beginPath(); ctx.arc(x, y, 7, 0, Math.PI * 2); ctx.fillStyle = "#ffffffdc"; ctx.fill();
  ctx.beginPath(); ctx.arc(x, y, 3.5, 0, Math.PI * 2); ctx.fillStyle = color; ctx.fill();
}

function dimensions(view, model) {
  const {ctx, left, top, w, h} = view;
  const bottom = top + h;
  ctx.strokeStyle = "#d4dfe3"; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(left, bottom + 23); ctx.lineTo(left + w, bottom + 23);
  ctx.moveTo(left, bottom + 19); ctx.lineTo(left, bottom + 27);
  ctx.moveTo(left + w, bottom + 19); ctx.lineTo(left + w, bottom + 27); ctx.stroke();
  ctx.font = "12px 'Segoe UI',sans-serif"; ctx.textAlign = "center";
  const text = `L = ${model.length}`;
  const textWidth = ctx.measureText(text).width;
  ctx.fillStyle = "#fcfdfe"; ctx.fillRect(left + w / 2 - textWidth / 2 - 6, bottom + 16, textWidth + 12, 13);
  ctx.fillStyle = "#606e7b"; ctx.fillText(text, left + w / 2, bottom + 26);
  ctx.save(); ctx.translate(left - 30, top + h / 2); ctx.rotate(-Math.PI / 2);
  ctx.fillText(`H = ${model.height}`, 0, 0); ctx.restore();
}

function filterReference(view, model, fill = true) {
  const radius = model.filter_radius || 0;
  if (radius <= 0) return;
  const {ctx, width, height, left, top, w, h} = view;
  const x = left + w / 2, y = top + h / 2;
  // Show the PDE length ell directly, at the same physical scale on both axes.
  const pixels = radius * w / model.length;
  ctx.save();
  ctx.beginPath(); ctx.rect(0, 0, width, height); ctx.clip();
  ctx.beginPath(); ctx.arc(x, y, pixels, 0, 2 * Math.PI);
  if (fill) { ctx.fillStyle = "#8272c41a"; ctx.fill(); }
  ctx.strokeStyle = "#7961b2"; ctx.lineWidth = 1.6;
  ctx.setLineDash([5, 3]); ctx.stroke(); ctx.setLineDash([]);
  ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x + pixels, y);
  ctx.moveTo(x, y - 3); ctx.lineTo(x, y + 3);
  ctx.moveTo(x + pixels, y - 3); ctx.lineTo(x + pixels, y + 3); ctx.stroke();
  ctx.restore();
}

function maximumSizeReference(view, model) {
  const {ctx, width, height, left, top, w, h} = view;
  const x = left + w/2, y = top + h/2;
  for (const [size, color, phase] of [[model.max_solid_size, "#3985b6", "solid"], [model.max_void_size, "#4c915e", "void"]]) {
    if (!(size > 0)) continue;
    const radius = size/2 * w/model.length;
    ctx.save();
    ctx.strokeStyle = color; ctx.lineWidth = 1.6; ctx.setLineDash(phase === "solid" ? [2, 3] : [7, 4]);
    ctx.beginPath(); ctx.arc(x, y, radius, 0, 2*Math.PI); ctx.stroke(); ctx.setLineDash([]);
    const label = phase === "solid" ? "Solid limit" : "Void limit";
    ctx.font = "12px 'Segoe UI',sans-serif"; ctx.textAlign = "center";
    const textWidth = ctx.measureText(label).width;
    const labelY = phase === "solid" ? Math.min(height-8, y+radius+16) : Math.max(18, y-radius-17);
    const labelX = clamp(x, textWidth/2+8, width-textWidth/2-8);
    ctx.fillStyle = "#f5fbffed"; ctx.fillRect(labelX-textWidth/2-5, labelY-11, textWidth+10, 16);
    ctx.fillStyle = color; ctx.fillText(label, labelX, labelY);
    ctx.restore();
  }
}

function selectionShape(view, model, selection, color, fill) {
  const {ctx, left, top, w, h} = view;
  const {x0,y0,x1,y1} = selectionBounds(selection,model);
  const x = left+x0/model.nx*w, y = top+h-y1/model.ny*h;
  const width = (x1-x0)/model.nx*w, height = (y1-y0)/model.ny*h;
  ctx.strokeStyle = color; ctx.fillStyle = fill; ctx.lineWidth = 2;
  if (selection.kind === "point") {
    ctx.beginPath(); ctx.arc(x,y,7,0,2*Math.PI); ctx.fill(); ctx.stroke();
  } else if (selection.kind === "surface") { ctx.fillRect(x,y,width,height); ctx.strokeRect(x,y,width,height); }
  else { ctx.beginPath(); ctx.moveTo(x,y+height); ctx.lineTo(x+width,y); ctx.stroke(); }
}

function boundaries(view, model, arrows = true) {
  if (!model.boundary_conditions) { support(view); if (arrows) load(view,model); return; }
  const {ctx,left,top,w,h} = view;
  for (const condition of model.boundary_conditions) {
    const fixed = condition.kind !== "force";
    if (!fixed && (condition.case_id || loadCases(model)[0].id) !== (model.display_load_case || loadCases(model)[0].id)) continue;
    const color = condition.kind === "displacement" ? "#8558a3" : fixed ? "#397a91" : FORCE;
    selectionShape(view,model,condition.selection,color,fixed ? "#397a9115" : "#df7a4010");
    const nodes = selectionNodes(condition.selection,model);
    const stride = Math.max(1,Math.ceil(nodes.length/(fixed ? 48 : 7)));
    for (let index=0; index<nodes.length; index+=stride) {
      const node = nodes[index];
      if (fixed) {
        const x = left+node.x/model.nx*w, y = top+h-node.y/model.ny*h, components = constrainedComponents(condition);
        ctx.fillStyle = color; ctx.strokeStyle = color; ctx.lineWidth = 2;
        if (components === "xy") ctx.fillRect(x-3,y-3,6,6);
        else { ctx.beginPath(); ctx.moveTo(x-(components === "y" ? 6 : 0),y-(components === "x" ? 6 : 0)); ctx.lineTo(x+(components === "y" ? 6 : 0),y+(components === "x" ? 6 : 0)); ctx.stroke(); }
        if (condition.kind === "displacement" && index === 0 && arrows) {
          if (components.includes("x") && condition.ux) load(view,model,node,Math.abs(condition.ux),condition.ux>0 ? 0 : 180,color);
          if (components.includes("y") && condition.uy) load(view,model,node,Math.abs(condition.uy),condition.uy>0 ? 90 : -90,color);
        }
      } else if (arrows) load(view,model,node,condition.magnitude/Math.min(nodes.length,7),condition.angle);
    }
  }
}

function selectionOverlay(view, model, overlay) {
  if (!overlay) return;
  const {ctx,left,top,w,h} = view;
  ctx.save();
  if (overlay.draft) ctx.setLineDash([5,4]);
  if (overlay.selection) {
    selectionShape(view,model,overlay.selection,"#127fba","#168dd82a");
    const nodes = selectionNodes(overlay.selection,model), stride = Math.max(1,Math.ceil(nodes.length/120));
    ctx.setLineDash([]); ctx.fillStyle = "#127fba";
    for (let index=0; index<nodes.length; index+=stride) {
      const node = nodes[index]; ctx.beginPath();
      ctx.arc(left+node.x/model.nx*w,top+h-node.y/model.ny*h,2.5,0,2*Math.PI); ctx.fill();
    }
  }
  if (overlay.hover) {
    ctx.setLineDash([]); ctx.strokeStyle = "#127fba"; ctx.lineWidth = 1.5; ctx.beginPath();
    ctx.arc(left+w*overlay.hover.x/model.nx,top+h-h*overlay.hover.y/model.ny,5,0,2*Math.PI); ctx.stroke();
  }
  ctx.restore();
}

function nonDesignRegions(view, model, fill = true) {
  for (const region of model.non_design_regions || []) {
    const solid = region.kind === "solid";
    selectionShape(view,model,region.selection,solid ? "#466e63" : "#a56b39",fill ? solid ? "#c8ddd4" : "#f3e4d4" : "transparent");
  }
}

function connectionRegions(view,model) {
  for(const [key,color] of [["connection_anchors","#167dbe"],["connection_targets","#924db6"]]) {
    for(const selection of model[key] || []) selectionShape(view,model,selection,color,`${color}24`);
  }
}

export function drawModel(canvas, model, hover = null, overlay = null) {
  const view = surface(canvas, model);
  const {ctx, left, top, w, h} = view;
  ctx.fillStyle = "#f0f6f5"; ctx.fillRect(left, top, w, h);
  nonDesignRegions(view,model);
  ctx.strokeStyle = "#b8cdcf"; ctx.lineWidth = .55; ctx.beginPath();
  for (let x = 0; x <= model.nx; x++) {
    const px = left + w * x / model.nx; ctx.moveTo(px, top); ctx.lineTo(px, top + h);
  }
  for (let y = 0; y <= model.ny; y++) {
    const py = top + h * y / model.ny; ctx.moveTo(left, py); ctx.lineTo(left + w, py);
  }
  ctx.stroke();
  ctx.fillStyle = "#6b9498";
  for (let y = 0; y <= model.ny; y++) {
    for (let x = 0; x <= model.nx; x++) {
      ctx.beginPath(); ctx.arc(left + w * x / model.nx, top + h - h * y / model.ny, .85, 0, Math.PI * 2); ctx.fill();
    }
  }
  outline(view); dimensions(view, model);
  filterReference(view, model);
  maximumSizeReference(view, model);
  if (hover) {
    ctx.beginPath(); ctx.arc(left + w * hover.x / model.nx, top + h - h * hover.y / model.ny, 5, 0, 2 * Math.PI);
    ctx.strokeStyle = "#45978d"; ctx.lineWidth = 1.5; ctx.stroke();
  }
  boundaries(view, model);
  connectionRegions(view,model);
  selectionOverlay(view, model, overlay);
  return view;
}

export function drawTopology(canvas, model, density) {
  const view = surface(canvas, model);
  const {ctx, left, top, w, h} = view;
  const cellWidth = w / model.nx, cellHeight = h / model.ny;
  if (density) {
    for (let iy = 0; iy < model.ny; iy++) {
      for (let ix = 0; ix < model.nx; ix++) {
        const rho = clamp(density[iy * model.nx + ix], 0, 1);
        ctx.fillStyle = `rgb(${Math.round(247 - 212 * rho)},${Math.round(250 - 176 * rho)},${Math.round(250 - 174 * rho)})`;
        ctx.fillRect(left + ix * cellWidth, top + (model.ny - 1 - iy) * cellHeight, cellWidth + .2, cellHeight + .2);
      }
    }
  } else {
    ctx.fillStyle = "#f7fafb"; ctx.fillRect(left, top, w, h);
  }
  outline(view); dimensions(view, model);
  if (density) filterReference(view, model, false);
  nonDesignRegions(view,model,!density);
  boundaries(view, model, !!density);
  connectionRegions(view,model);
  maximumSizeReference(view, model);
}

export function drawDensityDifference(canvas, model, difference) {
  const view = surface(canvas, model);
  const {ctx, left, top, w, h} = view;
  const cellWidth = w / model.nx, cellHeight = h / model.ny;
  difference.forEach((value, index) => {
    const error = clamp(value, 0, 1);
    ctx.fillStyle = `rgb(${Math.round(255-65*error)},${Math.round(255-207*error)},${Math.round(255-213*error)})`;
    ctx.fillRect(left+(index%model.nx)*cellWidth,
      top+(model.ny-1-Math.floor(index/model.nx))*cellHeight,cellWidth+.2,cellHeight+.2);
  });
  outline(view); dimensions(view, model);
}

export function pickNode(canvas, model, event) {
  const view = surface(canvas, model, false);
  const bounds = canvas.getBoundingClientRect();
  const x = event.clientX - bounds.left, y = event.clientY - bounds.top;
  if (x < view.left - 8 || x > view.left + view.w + 8 || y < view.top - 8 || y > view.top + view.h + 8) return null;
  return {x: clamp(Math.round((x - view.left) / view.w * model.nx), 0, model.nx),
          y: clamp(Math.round((view.top + view.h - y) / view.h * model.ny), 0, model.ny)};
}

// Element indexing follows drawTopology, with the first row at the bottom.
export function pickElement(canvas, model, event) {
  const {width,height,left,top,w,h} = surface(canvas,model,false);
  const rect=canvas.getBoundingClientRect();
  const x=(event.clientX-rect.left)*width/rect.width;
  const y=(event.clientY-rect.top)*height/rect.height;
  if(x<left || x>=left+w || y<top || y>=top+h) return null;
  return Math.min(model.ny-1,Math.floor((top+h-y)/h*model.ny))*model.nx+Math.floor((x-left)/w*model.nx);
}

export function drawPseudoTemperature(canvas,model,temperature,limit) {
  const view=surface(canvas,model), {ctx,left,top,w,h}=view;
  for(let y=0;y<model.ny;y++) for(let x=0;x<model.nx;x++) {
    const value=Math.min(1,Math.max(0,temperature[y*model.nx+x]/limit));
    ctx.fillStyle=`rgb(${Math.round(242-20*value)},${Math.round(248-195*value)},${Math.round(255-197*value)})`;
    ctx.fillRect(left+x*w/model.nx,top+(model.ny-1-y)*h/model.ny,w/model.nx+.2,h/model.ny+.2);
  }
  outline(view);dimensions(view,model);connectionRegions(view,model);
}

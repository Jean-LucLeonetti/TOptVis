const FORCE = "#df7a40";
const clamp = (value, lower, upper) => Math.min(upper, Math.max(lower, value));

function surface(canvas, model) {
  const {width, height} = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  const pixelWidth = Math.round(width * ratio), pixelHeight = Math.round(height * ratio);
  if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
    canvas.width = pixelWidth; canvas.height = pixelHeight;
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);
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
}

function outline(view) {
  const {ctx, left, top, w, h} = view;
  ctx.strokeStyle = "#829ca4"; ctx.lineWidth = 1;
  ctx.strokeRect(left, top, w, h);
}

function load(view, model) {
  const {ctx, width, height, left, top, w, h} = view;
  const x = left + model.load_node_x / model.nx * w;
  const y = top + h - model.load_node_y / model.ny * h;
  const angle = -model.load_angle * Math.PI / 180;
  const dx = Math.cos(angle), dy = Math.sin(angle);
  const desired = clamp(24 + 14 * Math.sqrt(model.load), 30, 74);
  let limit = desired;
  if (dx > 1e-8) limit = Math.min(limit, (width - 15 - x) / dx);
  if (dx < -1e-8) limit = Math.min(limit, (15 - x) / dx);
  if (dy > 1e-8) limit = Math.min(limit, (height - 13 - y) / dy);
  if (dy < -1e-8) limit = Math.min(limit, (13 - y) / dy);
  const endX = x + dx * limit, endY = y + dy * limit;
  ctx.strokeStyle = FORCE; ctx.fillStyle = FORCE; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(endX, endY); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(endX, endY);
  ctx.lineTo(endX - 8 * Math.cos(angle - .4), endY - 8 * Math.sin(angle - .4));
  ctx.lineTo(endX - 8 * Math.cos(angle + .4), endY - 8 * Math.sin(angle + .4));
  ctx.closePath(); ctx.fill();
  ctx.beginPath(); ctx.arc(x, y, 7, 0, Math.PI * 2); ctx.fillStyle = "#ffffffdc"; ctx.fill();
  ctx.beginPath(); ctx.arc(x, y, 3.5, 0, Math.PI * 2); ctx.fillStyle = FORCE; ctx.fill();
}

function dimensions(view, model) {
  const {ctx, left, top, w, h} = view;
  const bottom = top + h;
  ctx.strokeStyle = "#d4dfe3"; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(left, bottom + 23); ctx.lineTo(left + w, bottom + 23);
  ctx.moveTo(left, bottom + 19); ctx.lineTo(left, bottom + 27);
  ctx.moveTo(left + w, bottom + 19); ctx.lineTo(left + w, bottom + 27); ctx.stroke();
  ctx.font = "10px 'Segoe UI',sans-serif"; ctx.textAlign = "center";
  const text = `L = ${model.length}`;
  const textWidth = ctx.measureText(text).width;
  ctx.fillStyle = "#fcfdfe"; ctx.fillRect(left + w / 2 - textWidth / 2 - 6, bottom + 16, textWidth + 12, 13);
  ctx.fillStyle = "#8ca0a8"; ctx.fillText(text, left + w / 2, bottom + 26);
  ctx.save(); ctx.translate(left - 30, top + h / 2); ctx.rotate(-Math.PI / 2);
  ctx.fillText(`H = ${model.height}`, 0, 0); ctx.restore();
}

export function drawModel(canvas, model, hover = null) {
  const view = surface(canvas, model);
  const {ctx, left, top, w, h} = view;
  ctx.fillStyle = "#f0f6f5"; ctx.fillRect(left, top, w, h);
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
    for (let x = 1; x <= model.nx; x++) {
      ctx.beginPath(); ctx.arc(left + w * x / model.nx, top + h - h * y / model.ny, .85, 0, Math.PI * 2); ctx.fill();
    }
  }
  support(view); outline(view); dimensions(view, model);
  if (hover && hover.x > 0) {
    ctx.beginPath(); ctx.arc(left + w * hover.x / model.nx, top + h - h * hover.y / model.ny, 5, 0, 2 * Math.PI);
    ctx.strokeStyle = "#45978d"; ctx.lineWidth = 1.5; ctx.stroke();
  }
  load(view, model);
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
  support(view); outline(view); dimensions(view, model);
  if (density) load(view, model);
}

export function pickNode(canvas, model, event) {
  const view = surface(canvas, model);
  const bounds = canvas.getBoundingClientRect();
  const x = event.clientX - bounds.left, y = event.clientY - bounds.top;
  if (x < view.left - 8 || x > view.left + view.w + 8 || y < view.top - 8 || y > view.top + view.h + 8) return null;
  return {x: clamp(Math.round((x - view.left) / view.w * model.nx), 0, model.nx),
          y: clamp(Math.round((view.top + view.h - y) / view.h * model.ny), 0, model.ny)};
}

import {selectionBounds,regionElements} from "./mesh-selection.js";

export function connectionElements(selection,model) {
  const {x0,y0,x1,y1}=selectionBounds(selection,model), result=[];
  const xa=x0===x1 ? Math.max(0,x0-1) : x0, xb=x0===x1 ? Math.min(model.nx,x0+1) : x1;
  const ya=y0===y1 ? Math.max(0,y0-1) : y0, yb=y0===y1 ? Math.min(model.ny,y0+1) : y1;
  if(selection.kind!=="point" && ((selection.kind==="surface" && (x0===x1 || y0===y1)) || (x0===x1 && y0===y1))) return [];
  for(let y=ya;y<yb;y++) for(let x=xa;x<xb;x++) result.push(y*model.nx+x);
  return result;
}
export function connectionError(items,model,regions=[]) {
  if(!items.length) return "";
  const anchors=items.filter(item=>item.kind==="connection_anchor"), targets=items.filter(item=>item.kind==="connection_target");
  if(!anchors.length || !targets.length) return "Assign a connection anchor and at least one connection target.";
  if(anchors.length>16 || targets.length>16) return "At most 16 connection anchors and 16 targets can be assigned.";
  const voids=new Set(regions.filter(r=>r.kind==="void").flatMap(r=>regionElements(r.selection,model)));
  const cells=items.map(item=>connectionElements(item.selection,model));
  if(cells.some(indices=>!indices.length)) return "A connection selection is smaller than a mesh cell. Refine the mesh or replace the selection.";
  if(cells.some(indices=>indices.some(index=>voids.has(index)))) return "Connection regions cannot include fixed void cells.";
  const sink=new Set(anchors.flatMap(item=>connectionElements(item.selection,model)));
  if(targets.some(item=>connectionElements(item.selection,model).some(index=>sink.has(index)))) return "Connection anchors and targets must occupy separate mesh cells.";
  return "";
}

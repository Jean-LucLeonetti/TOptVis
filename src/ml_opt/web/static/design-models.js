import {predictNeural} from "./regression-models.js";

export function predictDesign(model,density) {
  if(!density.every(value=>Number.isFinite(value) && value>=0 && value<=1)) throw new Error("Invalid physical densities");
  let value;
  if(model.kind==="neural_network") value=predictNeural(model,density);
  else {
    const inputs=model.input_indices.map(i=>2*density[i]-1);
    const kernel=model.training_inputs.map(row=>{
      const dot=row.reduce((sum,rho,j)=>sum+(2*rho-1)*inputs[j],0)/inputs.length;
      return Math.max(0,Math.min(1,(1+dot)/2))**model.degree;
    });
    const mean=kernel.reduce((sum,k)=>sum+k,0)/kernel.length;
    value=kernel.reduce((sum,k,i)=>sum+(k-mean-model.kernel_column_mean[i]+model.kernel_mean)*model.dual_coefficients[i],model.target_mean);
  }
  if(model.target_transform==="log") value=Math.exp(value);
  if(!Number.isFinite(value)) throw new Error("Prediction exceeds numerical range");
  return value;
}

export function fixedElements(config) {
  const values=new Map();
  for(const region of config.non_design_regions || []) {
    const [x0,y0]=region.selection.start.map((v,i)=>Math.floor(v*[config.nx,config.ny][i]+.5));
    const [x1,y1]=region.selection.end.map((v,i)=>Math.floor(v*[config.nx,config.ny][i]+.5));
    for(let y=y0;y<y1;y++) for(let x=x0;x<x1;x++) values.set(y*config.nx+x,region.kind==="solid" ? 1 : 0);
  }
  return values;
}

export function editDensity(density,config,index,value) {
  if(!Number.isInteger(index) || index<0 || index>=density.length) throw new Error("Select a mesh element");
  if(!Number.isFinite(value) || value<0 || value>1) throw new Error("Density must be between 0 and 1");
  if(fixedElements(config).has(index)) throw new Error("This element belongs to a fixed solid or void region");
  const result=[...density];result[index]=value;return result;
}

export const designModelName=model=>model.kind==="neural_network" ? "Neural network" :
  `${model.method==="ridge" ? "Ridge" : "Polynomial"} · order ${model.degree}`;

export function comparableReports(a,b) {
  return ["train_indices","validation_indices","test_indices"].every(key=>
    JSON.stringify([...(a[key] || [])].sort((x,y)=>x-y))===JSON.stringify([...(b[key] || [])].sort((x,y)=>x-y)));
}

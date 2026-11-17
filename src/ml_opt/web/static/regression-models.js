// Portable neural weights use the same tanh/linear/sigmoid forward map as PyTorch.
export function predictNeural(model, volume) {
  const input=model.input_kind==="density" ? model.input_indices.map(i=>volume[i]) : [volume];
  const x=input.map(value=>(value-model.input_offset)/model.input_scale), w=model.weights;
  const hidden=w.hidden_weight.map((row,i)=>Math.tanh(row.reduce((sum,weight,j)=>sum+weight*x[j],w.hidden_bias[i])));
  const values=w.output_weight.map((row,i)=>{
    let value=row.reduce((sum,weight,j)=>sum+weight*hidden[j],w.output_bias[i]);
    if(model.output_activation==="sigmoid") value=value>=0 ? 1/(1+Math.exp(-value)) : Math.exp(value)/(1+Math.exp(value));
    return value*model.target_scale+model.target_offset;
  });
  if(model.target==="topology") {
    const density=[...model.fixed_density];model.output_indices.forEach((index,i)=>{density[index]=values[i];});return density;
  }
  return values[0];
}
export const modelName = item => item.kind==="neural_network" ? "Neural network" :
  ({1:"Linear",2:"Quadratic",3:"Cubic"}[item.degree ?? item.coefficients.length-1] || `Order ${item.degree}`);
export function mergeReports(polynomial, neural, selected="") {
  if(!polynomial && !neural) return null;
  const preferNeural=selected==="neural_network" || (!selected && neural && (!polynomial || neural.trained_at>(polynomial.trained_at || "")));
  const base=preferNeural && neural ? neural : polynomial || neural;
  const models={...polynomial?.models,...neural?.models};
  return {...base,models,selected_model:models[selected] ? selected : base.method==="neural" ? "neural_network" : Object.keys(models)[0],sources:{polynomial,neural}};
}
export function selectReport(report, key) {
  return report?.sources ? mergeReports(report.sources.polynomial,report.sources.neural,key) : report;
}
export function comparisonNote(report) {
  const {polynomial,neural}=report?.sources || {};
  return polynomial && neural && JSON.stringify([...polynomial.test_indices].sort((a,b)=>a-b))!==JSON.stringify([...neural.test_indices].sort((a,b)=>a-b))
    ? " Saved models use different test cases; scores are not directly comparable." : "";
}

import {drawTopology, drawDensityDifference} from "./viewer.js";

const el = id => document.getElementById(id);
const number = value => Number(value).toLocaleString("en", {maximumFractionDigits:4});
const label = degree => ({1:"Linear",2:"Quadratic",3:"Cubic"}[degree] || `Order ${degree}`);
export function predictDensity(model, volume, bounded=true) {
  const input = (volume-model.input_offset)/model.input_scale;
  return model.coefficients[0].map((_,element) => {
    const value = model.coefficients.reduceRight((sum,row)=>sum*input+row[element],0);
    return bounded ? Math.min(1,Math.max(0,value)) : value;
  });
}

export function createTopologyRegression({run}) {
  let dataset=null, report=null, locked=true, ready=false, evaluation=null, identity="";
  const selectedModel = () => report?.models[el("top-model").value];
  const reference = () => dataset?.cases.find(item=>String(item.index)===el("top-case").value);
  const volume = () => Number(el("top-volume").value);
  const matches = item => item && Math.abs(item.volume_fraction-volume())<1e-12;
  const use = item => report?.test_indices.includes(item.index) ? "Test" :
    report?.train_indices.includes(item.index) ? "Training" : report ? "Excluded" : "Reference";
  const row = (values, active=false) => {
    const tr=document.createElement("tr");tr.dataset.active=String(active);
    for(const value of values) {const td=document.createElement("td");td.textContent=value;tr.append(td);}
    return tr;
  };
  function draw(id, density, difference=false) {
    const canvas=el(id), config=dataset?.fixed_config;
    if(!canvas.getClientRects().length) return;
    if(!config || !density) {canvas.getContext("2d").clearRect(0,0,canvas.width,canvas.height);return;}
    const display={...config,filter_radius:0,max_solid_size:0,max_void_size:0};
    (difference ? drawDensityDifference : drawTopology)(canvas,display,density);
  }
  function structure() {
    const target=el("top-evaluation");target.replaceChildren();
    const current=evaluation && evaluation.degree===selectedModel()?.degree && Math.abs(evaluation.volume_fraction-volume())<1e-12;
    if(!current) {
      el("top-evaluation-note").textContent="Check structure evaluates the current prediction; results are cleared when the prediction changes.";
      return;
    }
    const p=evaluation.prediction, r=evaluation.reference;
    const objective=evaluation.objective_quantity==="elastic_work" ? "Elastic work" : "Compliance";
    for(const values of [[objective,number(p.objective),r ? number(r.objective) : "—"],
      ["Physical volume",number(p.physical_volume),r ? number(r.physical_volume) : "—"],
      ["Equilibrium residual",p.residual.toExponential(2),r ? r.residual.toExponential(2) : "—"],
      ["Nominal constraints",p.nominal_feasible ? "Satisfied" : "Violated",r ? r.nominal_feasible ? "Satisfied" : "Violated" : "—"]]) target.append(row(values));
    const change=evaluation.relative_objective_change;
    el("top-evaluation-note").textContent=`${evaluation.robust_dataset ? "Nominal material only; robust fields are not checked. " : ""}${change == null ? "No saved reference at this volume. " : `Objective change from the reference: ${(100*change).toFixed(2)}%. `}Checked in ${number(evaluation.evaluation_seconds)} s. These checks do not establish optimality.`;
  }
  function preview() {
    const fitted=selectedModel(), item=reference(), v=volume();
    const raw=fitted ? predictDensity(fitted,v,false) : null;
    const predicted=raw?.map(value=>Math.min(1,Math.max(0,value)));
    const comparable=predicted && matches(item);
    el("top-volume-value").textContent=v.toFixed(3);
    el("top-predicted-empty").hidden=!!predicted;
    el("top-reference-empty").hidden=!!item?.density;
    el("top-difference-empty").hidden=!!comparable;
    let note="Train topology models to explore predictions.", warning=false;
    if(fitted) {
      const clipped=raw.filter(value=>value < -1e-10 || value > 1+1e-10).length;
      const outside=v<report.training_range[0] || v>report.training_range[1];
      note=`${outside ? "Extrapolation: outside the training range." : "Within the training range."}${clipped ? ` ${clipped} of ${raw.length} elements clipped to [0, 1].` : ""}`;
      warning=outside || clipped>0;
    }
    el("top-prediction-note").textContent=note;el("top-prediction-note").dataset.warning=String(warning);
    el("top-predicted-note").textContent=predicted ? `Physical volume ${number(predicted.reduce((sum,value)=>sum+value,0)/predicted.length)}` : "Physical density";
    el("top-reference-note").textContent=item ? `Volume ${number(item.volume_fraction)} / ${use(item)}${!item.converged ? " / Unfinished" : ""}` : "";
    const difference=comparable ? predicted.map((value,index)=>Math.abs(value-item.density[index])) : null;
    el("top-difference-note").textContent=difference ? `Mean absolute error ${number(difference.reduce((sum,value)=>sum+value,0)/difference.length)}` : "Matching volumes required";
    draw("top-predicted",predicted);draw("top-reference",item?.density);draw("top-difference",difference,true);
    el("top-model").disabled=locked || !report;
    el("top-volume").disabled=locked || !fitted;
    el("top-case").disabled=locked || !dataset?.cases.some(item=>item.density);
    el("top-evaluate").disabled=locked || !ready || !fitted;
    structure();
  }
  function render() {
    const select=el("top-model"), previous=select.value;
    select.replaceChildren();
    for(const [key,item] of Object.entries(report?.models || {})) select.add(new Option(label(item.degree),key));
    select.value=report?.models[previous] ? previous : Object.keys(report?.models || {})[0] || "";
    const cases=el("top-case"), selected=cases.value;cases.replaceChildren();
    const eligible=dataset?.cases.filter(item=>item.density) || [];
    for(const item of eligible) cases.add(new Option(`${number(item.volume_fraction)} · ${use(item)}`,item.index));
    cases.value=eligible.some(item=>String(item.index)===selected) ? selected : String(
      eligible.find(item=>report?.test_indices.includes(item.index))?.index ?? eligible[0]?.index ?? "");
    const scores=el("top-scores");scores.replaceChildren();
    if(report) {
      el("top-fit-note").textContent=`${report.train_indices.length} training / ${report.test_indices.length} test / ${report.excluded_indices.length} excluded.${report.unfinished_indices.length ? ` Includes ${report.unfinished_indices.length} feasible unfinished runs.` : ""}`;
      scores.append(row(["Mean training topology",number(report.baseline.train.mae),number(report.baseline.test.mae)]));
      for(const [key,item] of Object.entries(report.models)) scores.append(row([label(item.degree),number(item.train.mae),number(item.test.mae)],key===select.value));
    } else {
      el("top-fit-note").textContent="At least six eligible cases with saved density fields are needed.";
      scores.append(row(["No fitted models","—","—"]));
    }
    el("top-download-models").hidden=!report;
    if(dataset) el("top-download-models").href=`/api/regression/${dataset.id}/artifacts/topology-models`;
    preview();
  }
  el("top-model").addEventListener("change",()=>{evaluation=null;render();});
  el("top-volume").addEventListener("input",()=>{evaluation=null;preview();});
  el("top-case").addEventListener("change",()=>{
    const item=reference();if(item) el("top-volume").value=item.volume_fraction;
    evaluation=null;preview();
  });
  el("top-evaluate").addEventListener("click",()=>{
    if(!dataset || !selectedModel() || locked || !ready) return;
    evaluation=null;structure();
    run(`/api/regression/${dataset.id}/topology/evaluate`,{degree:selectedModel().degree,volume:volume()},"evaluate");
  });
  for(const id of ["top-predicted","top-reference","top-difference"]) new ResizeObserver(preview).observe(el(id).parentElement);
  return {
    update(options) {
      const next=`${options.dataset?.id || ""}/${options.report?.trained_at || ""}`;
      const changed=next!==identity;
      dataset=options.dataset;report=options.report;locked=options.locked;ready=options.ready;
      if(changed) {
        identity=next;evaluation=null;el("top-case").value="";
        const range=dataset ? [dataset.settings.volume_min,dataset.settings.volume_max] : [.25,.65];
        const pad=(range[1]-range[0])*.15;
        el("top-volume").min=Math.max(.001,range[0]-pad);el("top-volume").max=Math.min(.999,range[1]+pad);
        el("top-volume").value=(range[0]+range[1])/2;
      }
      render();
      if(changed && report && reference()) {el("top-volume").value=reference().volume_fraction;preview();}
    },
    acceptEvaluation(result) {if(result.dataset_id===dataset?.id) {evaluation=result;preview();}},
    preview,
  };
}

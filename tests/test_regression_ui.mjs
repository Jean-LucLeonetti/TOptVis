import assert from "node:assert/strict";
import {readFile} from "node:fs/promises";
import test from "node:test";

// Exercise the real saved-experiment and training handlers without a browser.
class Element {
  constructor() { this.listeners=new Map(); this.children=[]; this.dataset={}; this._value=""; this.hidden=false; }
  set value(value) { this._value=String(value); }
  get value() { return this._value; }
  addEventListener(kind, callback) { this.listeners.set(kind,[...(this.listeners.get(kind)||[]),callback]); }
  emit(kind) { for(const callback of this.listeners.get(kind)||[]) callback({preventDefault(){}}); }
  append(...children) { this.children.push(...children); }
  add(child) { this.append(child); }
  replaceChildren(...children) { this.children=[...children]; }
  getClientRects() { return []; }
  getContext() { return {clearRect(){}}; }
  setAttribute() {}
}

test("legacy regression experiments restore their volume sweep and training options", async () => {
  const html=await readFile(new URL("../src/ml_opt/web/static/index.html",import.meta.url),"utf8");
  const elements=new Map([...html.matchAll(/id="([^"]+)"/g)].map(match=>[match[1],new Element()]));
  const node=id=>elements.get(id);
  const fields=new Map(), training=new Map();
  for(const [id, collection] of [["reg-generate-form",fields],["reg-fit-form",training]]) {
    const form=html.slice(html.indexOf(`id="${id}"`)).split("</form>")[0];
    for(const match of form.matchAll(/name="([^"]+)"/g)) collection.set(match[1],new Element());
    node(id).elements={length:17,namedItem:name=>collection.get(name) || null};
  }
  node("reg-model").value="linear";
  node("reg-volume").value=.45;
  const id="b".repeat(32);
  const settings={length:3,height:1,nx:30,ny:10,volume_min:.25,volume_max:.65,case_count:6};
  const dataset={id,settings,fixed_config:{},status:"complete",generation_seconds:2,
    cases:Array.from({length:6},(_,index)=>({index,volume_fraction:.25+.08*index,compliance:50-index,
      converged:true,feasible:true,stop_reason:"design_and_objective_stable",density:[.4]}))};
  const fitted={coefficients:[50,-10],training_seconds:.001,prediction_seconds:.00001,
    train:{mean_relative_error:.1},test:{mean_relative_error:.2}};
  let models={test_fraction:.35,seed:7,include_unconverged:true,train_indices:[0,2,4,5],
    test_indices:[1,3],excluded_indices:[],unfinished_indices:[],training_range:[.25,.65],
    baseline:{test:{mean_relative_error:.3}},models:{linear:fitted,quadratic:{...fitted,coefficients:[50,-10,2]}}};
  const requests=[], documentEvents=new Map();
  const globals=["document","fetch","ResizeObserver","Option"];
  const previous=new Map(globals.map(name=>[name,Object.getOwnPropertyDescriptor(globalThis,name)]));
  const flush=()=>new Promise(resolve=>setImmediate(resolve));
  try {
    globalThis.document={getElementById:id=>{
      assert.ok(elements.has(id),`Missing element ${id}`);return node(id);
    },createElement:()=>new Element(),addEventListener:(name,fn)=>documentEvents.set(name,fn),dispatchEvent(){}};
    globalThis.ResizeObserver=class {observe(){}};
    globalThis.Option=class extends Element {constructor(text,value){super();this.textContent=text;this.value=value;}};
    globalThis.fetch=async (url, options={})=>{
      requests.push({url,...options});
      if(options.method==="POST") {
        const payload=JSON.parse(options.body), degrees=payload.degrees;
        if(!url.includes("/topology/")) models={...models,method:payload.method,
          ridge_alpha:payload.ridge_alpha,degrees,trained_at:String(requests.length)};
        if(degrees.includes(3)) models={...models,degrees,models:{...models.models,
          polynomial_3:{...fitted,degree:3,coefficients:[50,-10,2,1],scaled_coefficients:[10,2,-1,.5],input_offset:.45,input_scale:.2},
          polynomial_4:{...fitted,degree:4,coefficients:[50,-10,2,1,1]},
          polynomial_5:{...fitted,degree:5,coefficients:[50,-10,2,1,1,1]}}};
        return new Response(JSON.stringify({event:"started",run_id:"job",dataset_id:id})+"\n"+
          JSON.stringify({event:"complete",dataset_id:id,stop_reason:"complete"})+"\n");
      }
      const data=url==="/api/health" ? {regression_supported:true,topology_regression_supported:true,running:false} :
        url==="/api/regression/datasets" ? {datasets:[{id,created_at:"2026-11-16T12:00",completed_cases:6}]} :
        {dataset,models};
      return new Response(JSON.stringify(data));
    };
    await import("../src/ml_opt/web/static/regression.js");
    await flush();
    node("reg-dataset").value=id;node("reg-dataset").emit("change");
    await flush();
    assert.equal(node("reg-error").hidden,true,node("reg-error").textContent);
    assert.equal(node("reg-generate-form").elements.length,17);
    assert.equal(fields.get("volume_min").value,"0.25");
    assert.equal(fields.get("case_count").value,"6");
    assert.equal(training.get("test_percent").value,"35");
    assert.equal(training.get("seed").value,"7");
    assert.equal(training.get("include_unconverged").checked,true);
    assert.equal(training.get("max_degree").value,"2");
    assert.equal(node("reg-status").textContent,"Models ready");
    assert.equal(node("reg-case").children.length,6);
    assert.equal(node("reg-volume").disabled,false);
    node("reg-fit-form").emit("submit");
    await flush();
    const request=requests.find(item=>item.method==="POST");
    assert.equal(request.url,`/api/regression/${id}/fit`);
    assert.deepEqual(JSON.parse(request.body),{test_fraction:.35,seed:7,include_unconverged:true,degrees:[1,2],method:"ols",ridge_alpha:.01});
    assert.equal(node("reg-error").hidden,true,node("reg-error").textContent);
    assert.equal(node("reg-status").textContent,"Models ready");
    training.get("max_degree").value=5;
    node("reg-fit-form").emit("submit");await flush();
    assert.deepEqual(JSON.parse(requests.filter(item=>item.method==="POST").at(-1).body).degrees,[1,2,3,4,5]);
    assert.equal(training.get("max_degree").value,"5");
    assert.deepEqual(node("reg-model").children.map(option=>option.textContent),["Linear","Quadratic","Cubic","Order 4","Order 5"]);
    node("reg-model").value="polynomial_3";node("reg-model").emit("change");
    assert.equal(node("reg-prediction").textContent,"10");
    assert.equal(node("reg-coefficients").children[3].children[0].textContent,"Volume power 3");
    node("reg-model").value="polynomial_5";node("reg-model").emit("change");
    assert.equal(node("reg-coefficients").children.length,6);
    assert.equal(node("reg-scores").children.at(-1).children[0].textContent,"Order 5");
    const count=requests.filter(item=>item.method==="POST").length;
    training.get("max_degree").value=0;
    node("reg-fit-form").emit("submit");await flush();
    assert.match(node("reg-error").textContent,/positive integer/);
    assert.equal(requests.filter(item=>item.method==="POST").length,count);
    documentEvents.get("regressionviewchange")({detail:"topology"});
    assert.equal(node("reg-compliance-results").hidden,true);
    assert.equal(node("reg-topology-results").hidden,false);
    training.get("max_degree").value=2;
    node("reg-fit-form").emit("submit");await flush();
    assert.equal(requests.filter(item=>item.method==="POST").at(-1).url,`/api/regression/${id}/topology/fit`);
    documentEvents.get("regressionviewchange")({detail:"compliance"});
    assert.equal(node("reg-compliance-results").hidden,false);
    assert.equal(node("reg-topology-results").hidden,true);
    assert.equal(node("reg-model").value,"polynomial_5");
    assert.equal(training.get("method").value,"ols");
    assert.equal(node("reg-ridge-options").hidden,true);
    training.get("max_degree").value=5;
    training.get("method").value="ridge";training.get("ridge_alpha").value=-1;
    node("reg-fit-form").emit("change");
    assert.equal(node("reg-ridge-options").hidden,false);
    assert.equal(training.get("ridge_alpha").disabled,false);
    node("reg-fit-form").emit("submit");await flush();
    assert.match(node("reg-error").textContent,/finite and nonnegative/);
    training.get("ridge_alpha").value=.25;training.get("max_degree").value=12;
    node("reg-fit-form").emit("submit");await flush();
    const ridgeRequest=JSON.parse(requests.filter(item=>item.method==="POST").at(-1).body);
    assert.equal(ridgeRequest.method,"ridge");assert.equal(ridgeRequest.ridge_alpha,.25);
    assert.deepEqual(ridgeRequest.degrees,Array.from({length:12},(_,i)=>i+1));
    assert.equal(training.get("max_degree").value,"12");
    assert.equal(training.get("method").value,"ridge");
    assert.equal(training.get("ridge_alpha").value,"0.25");
    documentEvents.get("regressionviewchange")({detail:"topology"});
    assert.equal(training.get("method").value,"ols");
    documentEvents.get("regressionviewchange")({detail:"compliance"});
    assert.equal(training.get("method").value,"ridge");assert.equal(training.get("ridge_alpha").value,"0.25");
    models.models.polynomial_5={...models.models.polynomial_5,coefficients:null,
      scaled_coefficients:[10,2,-1,.5,0,0],input_offset:.45,input_scale:.2};
    node("reg-dataset").emit("change");await flush();
    node("reg-model").emit("change");
    assert.equal(node("reg-coefficients").children[1].children[0].textContent,"Scaled volume (b)");
    models.models.polynomial_5.scaled_coefficients=[1e308,1e308,0,0,0,0];
    node("reg-dataset").emit("change");await flush();
    node("reg-volume").value=.7;node("reg-volume").emit("input");
    assert.equal(node("reg-prediction").textContent,"Unavailable");
    assert.match(node("reg-prediction-note").textContent,/numerical range/);
    assert.doesNotMatch(node("reg-chart").innerHTML,/NaN|Infinity/);
  } finally {
    for(const name of globals) {
      const descriptor=previous.get(name);
      if(descriptor) Object.defineProperty(globalThis,name,descriptor);else delete globalThis[name];
    }
  }
});

test("saved optimisation files validate, generate with unchanged settings, and restore on reload", async () => {
  const html=await readFile(new URL("../src/ml_opt/web/static/index.html",import.meta.url),"utf8");
  const elements=new Map([...html.matchAll(/id="([^"]+)"/g)].map(match=>[match[1],new Element()]));
  const node=id=>elements.get(id), fields=new Map();
  for(const id of ["reg-generate-form","reg-fit-form"]) {
    const controls=new Map(), form=html.slice(html.indexOf(`id="${id}"`)).split("</form>")[0];
    for(const match of form.matchAll(/<(?:input|select)\b[^>]*name="([^"]+)"[^>]*>/g)) {
      const control=new Element();
      control.value=match[0].match(/value="([^"]+)"/)?.[1] || (match[1]==="material_model" ? "simp" : "");
      controls.set(match[1],control);
      if(id==="reg-generate-form") fields.set(match[1],control);
    }
    node(id).elements={length:controls.size,namedItem:name=>controls.get(name)};
  }
  node("reg-model").value="linear";node("reg-volume").value=.45;
  const problem={nx:6,ny:2,length:3,height:1,max_iterations:2,robust:false,load:3,
    load_node_x:6,load_node_y:1,load_angle:-60,filter_radius:.05,material_model:"ramp",penalisation:2,
    projection_beta:1,projection_schedule:"geometric",optimizer:"gcmma",
    boundary_conditions:[{kind:"fixed",selection:{kind:"line",start:[0,0],end:[0,1]}},
      {kind:"force",selection:{kind:"point",start:[1,1]},magnitude:3,angle:-60}]};
  const id="c".repeat(32), requests=[], documentEvents=new Map();
  let generated=null;
  const names=["document","fetch","ResizeObserver","Option","FormData","CustomEvent","window"];
  const previous=new Map(names.map(name=>[name,Object.getOwnPropertyDescriptor(globalThis,name)]));
  const flush=()=>new Promise(resolve=>setImmediate(resolve));
  const choose=async file=>{
    node("reg-problem-file").files=[file];node("reg-problem-file").emit("change");await flush();
  };
  const file=(model=problem)=>({name:"Cantilever.json",size:1000,text:async()=>JSON.stringify({format:"toptvis-project",version:1,model,result:null})});
  try {
    globalThis.document={getElementById:id=>{assert.ok(elements.has(id),`Missing ${id}`);return node(id);},
      createElement:()=>new Element(),addEventListener:(kind,callback)=>documentEvents.set(kind,callback),dispatchEvent(){}};
    globalThis.ResizeObserver=class {observe(){}};
    globalThis.Option=class extends Element {constructor(text,value){super();this.textContent=text;this.value=value;}};
    globalThis.CustomEvent=class {constructor(type,options){this.type=type;this.detail=options.detail;}};
    globalThis.FormData=class {
      *[Symbol.iterator]() {
        for(const [name,control] of fields) {
          yield [name,control.value];
        }
      }
    };
    globalThis.window={devicePixelRatio:1};
    const context={measureText:text=>({width:text.length*6})};
    for(const name of ["clearRect","setTransform","save","restore","beginPath","rect","clip","stroke","fill","fillRect","strokeRect","moveTo","lineTo","closePath","arc","translate","rotate","fillText","setLineDash"]) context[name]=()=>{};
    node("reg-topology").getClientRects=()=>[{}];
    node("reg-topology").getBoundingClientRect=()=>({width:720,height:300,left:0,top:0});
    node("reg-topology").getContext=()=>context;
    globalThis.fetch=async (url,options={})=>{
      requests.push({url,...options});
      if(url==="/api/project/validate") {
        const model=JSON.parse(options.body);
        return new Response(JSON.stringify(model.nx===0 ? {detail:"Invalid mesh"} : {model}),{status:model.nx===0 ? 422 : 200});
      }
      if(url==="/api/regression/generate") {
        const settings=JSON.parse(options.body);
        generated={id,settings:{...settings,source_problem:settings.problem},fixed_config:settings.problem,
          status:"complete",generation_seconds:1,cases:[]};
        return new Response(JSON.stringify({event:"started",run_id:"job",dataset_id:id})+"\n"+
          JSON.stringify({event:"complete",stop_reason:"complete"})+"\n");
      }
      const data=url==="/api/health" ? {regression_supported:true,regression_problem_import_supported:true,running:false} :
        url==="/api/regression/datasets" ? {datasets:generated ? [{id,created_at:"2026-11-16T12:00",completed_cases:0}] : []} :
        {dataset:generated,models:null};
      return new Response(JSON.stringify(data));
    };
    await import("../src/ml_opt/web/static/regression.js?saved-setup-test");await flush();
    assert.equal(node("reg-generate").disabled,true);
    node("reg-generate-form").emit("submit");
    assert.match(node("reg-error").textContent,/Load an optimisation setup/);
    assert.equal(requests.some(item=>item.url==="/api/regression/generate"),false);
    await choose(file());
    assert.equal(node("reg-error").hidden,true,node("reg-error").textContent);
    assert.equal(node("reg-problem-name").textContent,"Cantilever.json");
    assert.equal(node("reg-generate").disabled,false);
    assert.equal(node("reg-topology-empty").hidden,true);
    // Navigating away and returning refreshes the list without losing the setup.
    documentEvents.get("mlmethodchange")({detail:"regression"});await flush();
    assert.equal(node("reg-topology-empty").hidden,true);
    await choose(file({...problem,nx:0}));
    assert.equal(node("reg-error").hidden,false);
    assert.equal(node("reg-problem-name").textContent,"Cantilever.json");
    const before=requests.length;
    await choose({...file(),size:65*1024*1024});
    assert.match(node("reg-error").textContent,/64 MiB/);
    assert.equal(requests.length,before);
    fields.get("case_count").value=6;
    node("reg-generate-form").emit("submit");await flush();
    const request=requests.find(item=>item.url==="/api/regression/generate");
    assert.ok(request);
    assert.deepEqual(JSON.parse(request.body),{volume_min:.25,volume_max:.65,case_count:6,
      problem,problem_name:"Cantilever.json"});
    assert.equal(node("reg-status").textContent,"Dataset ready");
    assert.equal(node("reg-error").hidden,true,node("reg-error").textContent);
    node("reg-dataset").value=id;node("reg-dataset").emit("change");await flush();
    assert.equal(node("reg-problem-name").textContent,"Cantilever.json");
    await choose(file({...problem,boundary_conditions:[...problem.boundary_conditions,
      {kind:"displacement",selection:{kind:"point",start:[1,0]},components:"x",ux:.03}]}));
    assert.equal(node("reg-prediction-title").textContent,"Elastic work prediction");
    assert.equal(node("reg-case-objective").textContent,"Elastic work");
  } finally {
    for(const name of names) {
      const descriptor=previous.get(name);
      if(descriptor) Object.defineProperty(globalThis,name,descriptor);else delete globalThis[name];
    }
  }
});

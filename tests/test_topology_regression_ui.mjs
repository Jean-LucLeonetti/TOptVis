import assert from "node:assert/strict";
import {readFile} from "node:fs/promises";
import test from "node:test";
import {createTopologyRegression,predictDensity} from "../src/ml_opt/web/static/topology-regression.js";

class Element {
  constructor(){this.children=[];this.listeners=new Map();this.dataset={};this._value="";}
  set value(value){this._value=String(value);}get value(){return this._value;}
  replaceChildren(...items){this.children=items;}add(item){this.children.push(item);}append(item){this.children.push(item);}
  addEventListener(name,fn){this.listeners.set(name,fn);}emit(name){this.listeners.get(name)();}
  getClientRects(){return [];}
}

test("density predictions use the fitted scaled basis and explicit clipping",()=>{
  const fitted={input_offset:.5,input_scale:.5,coefficients:[[.4,.5],[.2,-.2]]};
  assert.deepEqual(predictDensity(fitted,.5),[.4,.5]);
  assert.deepEqual(predictDensity(fitted,5),[1,0]);
  assert.ok(predictDensity(fitted,5,false)[0]>1);
});

test("topology comparison selects held-out cases, checks the current prediction, and clears stale checks",async()=>{
  const html=await readFile(new URL("../src/ml_opt/web/static/index.html",import.meta.url),"utf8");
  const nodes=new Map([...html.matchAll(/id="([^\"]+)"/g)].map(match=>[match[1],new Element()]));
  const node=id=>nodes.get(id),names=["document","ResizeObserver","Option"];
  const previous=new Map(names.map(name=>[name,Object.getOwnPropertyDescriptor(globalThis,name)]));
  const fitted={degree:1,input_offset:.5,input_scale:.5,coefficients:[[.4,.5],[.2,-.2]],train:{mae:.01},test:{mae:.02}};
  const dataset={id:"example",settings:{volume_min:.25,volume_max:.75},fixed_config:{nx:2,ny:1},
    cases:[{index:0,volume_fraction:.25,density:[.3,.6],converged:true},
      {index:1,volume_fraction:.5,density:[.4,.5],converged:true},
      {index:2,volume_fraction:.75,density:[.5,.4],converged:true}]};
  const report={trained_at:"saved",train_indices:[0,2],test_indices:[1],excluded_indices:[],unfinished_indices:[],
    training_range:[.25,.75],models:{1:fitted},baseline:{train:{mae:.1},test:{mae:.2}}};
  const requests=[];
  try {
    globalThis.document={getElementById:id=>{assert.ok(nodes.has(id),id);return node(id);},createElement:()=>new Element()};
    globalThis.ResizeObserver=class{observe(){}};
    globalThis.Option=class extends Element {constructor(text,value){super();this.textContent=text;this.value=value;}};
    const panel=createTopologyRegression({run:(...args)=>requests.push(args)});
    panel.update({dataset,report,locked:false,ready:true});
    assert.equal(node("top-case").value,"1");assert.equal(node("top-volume").value,"0.5");
    assert.match(node("top-reference-note").textContent,/Test/);
    assert.equal(node("top-difference-empty").hidden,true);
    assert.equal(node("top-reference-empty").hidden,true);
    assert.equal(node("top-difference-note").textContent,"Mean absolute error 0");
    assert.equal(node("top-evaluate").disabled,false);
    node("top-evaluate").emit("click");
    assert.deepEqual(requests[0],["/api/regression/example/topology/evaluate",{degree:1,volume:.5},"evaluate"]);
    const checked={dataset_id:"example",degree:1,volume_fraction:.5,prediction:{objective:20,physical_volume:.45,residual:1e-12,nominal_feasible:true},
      reference:{objective:20,physical_volume:.45,residual:1e-12,nominal_feasible:true},relative_objective_change:0,evaluation_seconds:.01};
    panel.acceptEvaluation(checked);assert.equal(node("top-evaluation").children.length,4);
    // Reloading the same artifacts preserves this check.
    panel.update({dataset:structuredClone(dataset),report:structuredClone(report),locked:false,ready:true});
    assert.equal(node("top-evaluation").children.length,4);
    node("top-volume").value=.6;node("top-volume").emit("input");
    assert.equal(node("top-difference-empty").hidden,false);assert.equal(node("top-evaluation").children.length,0);
    node("top-case").value=0;node("top-case").emit("change");assert.equal(node("top-volume").value,"0.25");
    assert.match(node("top-reference-note").textContent,/Training/);
    node("top-volume").value=.8;node("top-volume").emit("input");assert.match(node("top-prediction-note").textContent,/Extrapolation/);
    panel.update({dataset,report,locked:true,ready:true});assert.equal(node("top-evaluate").disabled,true);
    assert.equal(node("top-case").disabled,true);
    panel.update({dataset:null,report:null,locked:false,ready:true});assert.equal(node("top-evaluate").disabled,true);
    assert.equal(node("top-download-models").hidden,true);
    assert.equal(node("top-reference-empty").hidden,false);
  } finally {
    for(const name of names){const descriptor=previous.get(name);if(descriptor)Object.defineProperty(globalThis,name,descriptor);else delete globalThis[name];}
  }
});

import assert from "node:assert/strict";
import {readFile} from "node:fs/promises";
import test from "node:test";
import {mergeReports,selectReport,comparisonNote,predictNeural} from "../src/ml_opt/web/static/regression-models.js";

class Element {
  constructor(){this.children=[];this.listeners=new Map();this.dataset={};this._value="";this.hidden=false;}
  set value(value){this._value=String(value);}get value(){return this._value;}
  replaceChildren(...children){this.children=children;}append(...children){this.children.push(...children);}add(child){this.append(child);}
  addEventListener(name,fn){this.listeners.set(name,[...(this.listeners.get(name)||[]),fn]);}
  emit(name){for(const fn of this.listeners.get(name)||[])fn({preventDefault(){}});}
  getClientRects(){return [];}getContext(){return {clearRect(){}};}setAttribute(name,value){this[name]=value;}
}
const metric={mae:.1,rmse:.1,mean_relative_error:.1};
const neuralModel={kind:"neural_network",target:"compliance",neurons:1,output_activation:"linear",input_offset:.5,input_scale:.5,
  target_offset:10,target_scale:2,weights:{hidden_weight:[[1]],hidden_bias:[0],output_weight:[[2]],output_bias:[1]},
  train:metric,validation:metric,test:metric,training_seconds:.1,prediction_seconds:.001,best_epoch:1,
  training_history:[{epoch:0,train_loss:1,validation_loss:2},{epoch:1,train_loss:.1,validation_loss:.2}]};
const base={trained_at:"2026-11-17T01:00:00",test_fraction:.25,seed:42,include_unconverged:false,train_indices:[0,2,3,4,5,7],
  test_indices:[1,6],excluded_indices:[],unfinished_indices:[],training_range:[.25,.65],baseline:{value:10,train:metric,test:metric}};
const polynomial={...base,method:"ridge",degrees:[1],ridge_alpha:.01,models:{linear:{degree:1,coefficients:[10,2],train:metric,test:metric,training_seconds:.01,prediction_seconds:.001}}};
const neural={...base,trained_at:"2026-11-17T02:00:00",method:"neural",train_indices:[0,3,4,5,7],validation_indices:[2],
  neurons:8,epochs:20,learning_rate:.01,weight_penalty:0,network_seed:42,validation_fraction:.2,models:{neural_network:neuralModel}};

test("comparison selects the model's own split and portable inference",()=>{
  const merged=mergeReports(polynomial,neural);
  assert.equal(merged.method,"neural");assert.equal(merged.selected_model,"neural_network");
  assert.deepEqual(merged.validation_indices,[2]);assert.deepEqual(Object.keys(merged.models),["linear","neural_network"]);
  const scalar=selectReport(merged,"linear");assert.equal(scalar.method,"ridge");assert.equal(scalar.validation_indices,undefined);
  assert.equal(comparisonNote(merged),"");
  assert.match(comparisonNote(mergeReports(polynomial,{...neural,test_indices:[2,6]})),/different test cases/);
  assert.equal(predictNeural(neuralModel,.5),12);
  const density={...neuralModel,target:"topology",output_activation:"sigmoid",target_offset:0,target_scale:1,
    fixed_density:[1,0,0],output_indices:[1]};
  assert.deepEqual(predictNeural(density,.5),[1,1/(1+Math.exp(-1)),0]);
});

test("neural controls train both targets, retain polynomial choices and show training curves",async()=>{
  const html=await readFile(new URL("../src/ml_opt/web/static/index.html",import.meta.url),"utf8");
  const nodes=new Map([...html.matchAll(/id="([^\"]+)"/g)].map(match=>[match[1],new Element()])),node=id=>nodes.get(id);
  const training=new Map(),generating=new Map();
  for(const [id,fields] of [["reg-fit-form",training],["reg-generate-form",generating]]) {
    const form=html.slice(html.indexOf(`id="${id}"`)).split("</form>")[0];
    for(const match of form.matchAll(/name="([^\"]+)"/g))fields.set(match[1],new Element());
    node(id).elements={namedItem:name=>fields.get(name)};
  }
  node("reg-volume").value=.5;node("top-volume").value=.5;
  const id="d".repeat(32),dataset={id,settings:{volume_min:.25,volume_max:.65,case_count:8},fixed_config:{nx:1,ny:1},
    status:"complete",generation_seconds:1,cases:Array.from({length:8},(_,index)=>({index,volume_fraction:.25+index*.4/7,
      compliance:10+index,density:[.4],feasible:true,converged:true,stop_reason:"design_and_objective_stable"}))};
  const events=new Map(),requests=[],saved={compliance:null,topology:null};
  const names=["document","fetch","Option","ResizeObserver","CustomEvent"],previous=new Map(names.map(name=>[name,Object.getOwnPropertyDescriptor(globalThis,name)]));
  const flush=()=>new Promise(resolve=>setImmediate(resolve));
  try {
    globalThis.document={getElementById:id=>{assert.ok(nodes.has(id),id);return node(id);},createElement:()=>new Element(),
      addEventListener:(name,fn)=>events.set(name,fn),dispatchEvent(){}};
    globalThis.Option=class extends Element{constructor(text,value){super();this.textContent=text;this.value=value;}};
    globalThis.ResizeObserver=class{observe(){}};
    globalThis.CustomEvent=class{constructor(type,options){this.type=type;this.detail=options.detail;}};
    globalThis.fetch=async(url,options={})=>{
      requests.push({url,...options});
      if(options.method==="POST") {
        const payload=JSON.parse(options.body),target=url.includes("/topology/") ? "topology" : "compliance";
        const model=target==="topology" ? {...neuralModel,target,output_activation:"sigmoid",target_offset:0,target_scale:1,
          output_indices:[0],fixed_density:[0]} : neuralModel;
        saved[target]={...neural,...payload,models:{neural_network:model},trained_at:String(requests.length)};
        return new Response([ {event:"started",run_id:"job",dataset_id:id},
          {event:"neural_training",target,epochs:20,epoch:0,train_loss:1,validation_loss:2},
          {event:"neural_training",target,epochs:20,epoch:20,train_loss:.1,validation_loss:.2},
          {event:"complete",dataset_id:id,stop_reason:"complete"}].map(JSON.stringify).join("\n"));
      }
      const result=url==="/api/health" ? {regression_supported:true,topology_regression_supported:true,running:false} :
        url==="/api/regression/datasets" ? {datasets:[{id,created_at:"2026-11-17T00:00",completed_cases:8}]} :
        {dataset,models:polynomial,topology_models:null,neural_models:saved};
      return new Response(JSON.stringify(result));
    };
    await import("../src/ml_opt/web/static/regression.js?neural-test");await flush();
    node("reg-dataset").value=id;node("reg-dataset").emit("change");await flush();
    assert.equal(training.get("method").value,"ridge");
    training.get("method").value="neural";training.get("epochs").value=20;node("reg-fit-form").emit("change");
    assert.equal(node("reg-neural-options").hidden,false);assert.equal(training.get("max_degree").disabled,true);
    node("reg-fit-form").emit("submit");await flush();
    const request=JSON.parse(requests.find(item=>item.method==="POST").body);
    assert.equal(request.method,"neural");assert.equal(request.epochs,20);assert.equal(request.neurons,8);
    assert.equal(request.validation_fraction,.2);assert.equal(request.degrees,undefined);
    assert.equal(node("reg-model").value,"neural_network");
    assert.deepEqual(node("reg-model").children.map(item=>item.textContent),["Linear","Neural network"]);
    assert.equal(node("reg-coefficients").children.length,0);assert.match(node("reg-fit-note").textContent,/validation/);
    assert.equal(node("reg-learning-panel").hidden,false);assert.match(node("reg-learning-note").textContent,/Using epoch 1/);
    assert.match(node("reg-download-models").href,/neural-compliance$/);
    node("reg-model").value="linear";node("reg-model").emit("change");
    assert.equal(training.get("method").value,"ridge");assert.equal(node("reg-learning-panel").hidden,true);
    events.get("regressionviewchange")({detail:"topology"});
    training.get("method").value="neural";training.get("epochs").value=20;node("reg-fit-form").emit("change");
    node("reg-fit-form").emit("submit");await flush();
    assert.match(requests.filter(item=>item.method==="POST").at(-1).url,/topology\/fit$/);
    assert.equal(node("top-model").value,"neural_network");assert.equal(node("top-learning-panel").hidden,false);
    assert.match(node("top-download-models").href,/neural-topology$/);
    node("top-evaluate").emit("click");await flush();
    const check=requests.filter(item=>item.method==="POST").at(-1);
    assert.match(check.url,/topology\/evaluate$/);assert.equal(JSON.parse(check.body).model_key,"neural_network");
  } finally {
    for(const name of names){const descriptor=previous.get(name);if(descriptor)Object.defineProperty(globalThis,name,descriptor);else delete globalThis[name];}
  }
});

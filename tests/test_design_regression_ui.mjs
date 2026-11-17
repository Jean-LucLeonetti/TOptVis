import assert from "node:assert/strict";
import {readFile} from "node:fs/promises";
import test from "node:test";

class Element {
  constructor(){this.children=[];this.listeners=new Map();this.dataset={};this._value="";this.hidden=false;this.checked=false;this.parentElement={};}
  set value(value){this._value=String(value);}get value(){return this._value || this.children[0]?.value || "";}
  get options(){return this.children;}
  replaceChildren(...children){this.children=children;this._value="";}append(...children){this.children.push(...children);}
  addEventListener(name,fn){this.listeners.set(name,[...(this.listeners.get(name)||[]),fn]);}
  emit(name){for(const fn of this.listeners.get(name)||[])fn({preventDefault(){}});}
  getClientRects(){return [];}getContext(){return {clearRect(){}};}setAttribute(name,value){this[name]=value;}
  querySelectorAll(){return [];}reportValidity(){return true;}set innerHTML(value){this.html=value;}
}
const metric={mean_relative_error:.1};
const model={kind:"neural_network",target:"compliance",input_kind:"density",input_indices:[1,2],input_offset:.5,input_scale:.5,
  target_offset:2,target_scale:1,target_transform:"log",output_activation:"linear",method:"neural",
  weights:{hidden_weight:[[1,2]],hidden_bias:[0],output_weight:[[1]],output_bias:[0]},
  train:metric,validation:metric,test:metric,cases:[{index:0,reference:10,predicted:9}],best_epoch:1,
  training_history:[{epoch:0,train_loss:1,validation_loss:2},{epoch:1,train_loss:.1,validation_loss:.2}]};

test("design regression trains, restores, edits, checks, resets and shares worker locking",async()=>{
  const html=await readFile(new URL("../src/ml_opt/web/static/index.html",import.meta.url),"utf8");
  const ids=[...html.matchAll(/id="([^\"]+)"/g)].map(match=>match[1]);assert.equal(ids.length,new Set(ids).size);
  const nodes=new Map(ids.map(id=>[id,new Element()])),node=id=>nodes.get(id),forms=new Map();
  for(const id of ["design-fit-form","design-generate-form"]) {
    const fields=new Map(),form=html.slice(html.indexOf(`id="${id}"`)).split("</form>")[0];
    for(const match of form.matchAll(/name="([^\"]+)"[^>]*>/g)) {
      const field=new Element();field.value=match[0].match(/value="([^\"]+)"/)?.[1] || "ridge";fields.set(match[1],field);
    }
    forms.set(id,fields);node(id).elements={namedItem:name=>fields.get(name)};
  }
  const fit=forms.get("design-fit-form"),generate=forms.get("design-generate-form");
  fit.get("method").value="neural";fit.get("epochs").value="20";fit.get("log_target").checked=true;
  const identifier="c".repeat(32),data={id:identifier,status:"complete",expected_cases:1,family_count:3,source_robust:true,
    fixed_config:{nx:3,ny:1,non_design_regions:[{kind:"solid",selection:{start:[0,0],end:[1/3,1]}}]},
    cases:[{index:0,status:"complete",family:"base",variant:0,density:[1,.2,.8],compliance:10}]};
  const report={method:"neural",train_indices:[0],validation_indices:[],test_indices:[],models:{neural_network:model},baseline:{train:metric,validation:metric,test:metric},volume_baseline:{train:metric,validation:metric,test:metric}};
  const events=new Map(),requests=[];let trained=false,workerFailure=false;
  const names=["document","fetch","ResizeObserver","CustomEvent"],previous=new Map(names.map(name=>[name,Object.getOwnPropertyDescriptor(globalThis,name)]));
  const flush=async()=>{for(let i=0;i<5;i++)await new Promise(resolve=>setImmediate(resolve));};
  try {
    globalThis.document={getElementById:id=>{assert.ok(nodes.has(id),id);return node(id);},createElement:()=>new Element(),addEventListener:(name,fn)=>events.set(name,fn),dispatchEvent(){}};
    globalThis.ResizeObserver=class{observe(){}};
    globalThis.CustomEvent=class{constructor(type,{detail}){this.type=type;this.detail=detail;}};
    globalThis.fetch=async(url,options={})=>{
      requests.push({url,...options});
      if(options.method==="POST") {
        if(workerFailure) return new Response([{event:"started",run_id:"job",dataset_id:identifier},{event:"error",message:"Training diverged"}].map(JSON.stringify).join("\n"));
        const check=url.endsWith("/evaluate");trained=true;
        return new Response([{event:"started",run_id:"job",dataset_id:identifier},...(check ? [{event:"design_evaluation",evaluation:{compliance:12,residual:1e-12}}] : [{event:"neural_training",epoch:0,train_loss:1,validation_loss:2}]),{event:"complete",stop_reason:"complete"}].map(JSON.stringify).join("\n"));
      }
      const result=url==="/api/health" ? {design_regression_supported:true} : url==="/api/design-regression/datasets" ? {datasets:[{id:identifier,case_count:1}]} :
        url==="/api/regression/datasets" ? {datasets:[{id:identifier,completed_cases:3}]} : {dataset:data,reports:{neural:trained ? report : null},errors:{}};
      return new Response(JSON.stringify(result));
    };
    await import("../src/ml_opt/web/static/design-regression.js?test");
    events.get("regressionviewchange")({detail:"design"});await flush();
    assert.equal(node("design-actual").textContent,"10");assert.equal(node("design-fit").disabled,false);
    node("design-fit-form").emit("submit");await flush();
    const payload=JSON.parse(requests.find(r=>r.method==="POST").body);
    assert.equal(payload.method,"neural");assert.equal(payload.validation_fraction,.2);assert.equal(payload.epochs,20);
    assert.equal(node("design-model").value,"neural_network");assert.equal(node("design-learning-panel").hidden,false);
    assert.match(node("design-download-models").href,/models-neural$/);
    node("design-element").value=2;node("design-value").value=.1;node("design-edit-form").emit("submit");
    assert.equal(node("design-actual").textContent,"Unavailable");assert.match(node("design-check-note").textContent,/Edited/);
    node("design-evaluate").emit("click");await flush();
    const checked=requests.filter(r=>r.method==="POST").at(-1);assert.deepEqual(JSON.parse(checked.body).density,[1,.1,.8]);
    assert.equal(node("design-actual").textContent,"12");assert.match(node("design-check-note").textContent,/Nominal/);
    node("design-reset").emit("click");assert.equal(node("design-actual").textContent,"10");
    node("design-element").value=1;node("design-value").value=0;node("design-edit-form").emit("submit");
    assert.match(node("design-error").textContent,/fixed/);assert.equal(data.cases[0].density[0],1);
    events.get("solveractivity")({detail:{workspace:"regression",running:true}});assert.equal(node("design-evaluate").disabled,true);
    events.get("solveractivity")({detail:{workspace:"regression",running:false}});assert.equal(node("design-evaluate").disabled,false);
    workerFailure=true;node("design-fit-form").emit("submit");await flush();
    assert.equal(node("design-error").textContent,"Training diverged");assert.equal(node("design-fit").disabled,false);workerFailure=false;
    generate.get("source_id").value=identifier;node("design-generate-form").emit("submit");await flush();
    const generated=requests.filter(r=>r.method==="POST").at(-1);assert.equal(generated.url,"/api/design-regression/generate");assert.equal(JSON.parse(generated.body).variants,4);
  } finally {
    for(const name of names){const descriptor=previous.get(name);if(descriptor)Object.defineProperty(globalThis,name,descriptor);else delete globalThis[name];}
  }
});

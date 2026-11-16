import assert from "node:assert/strict";
import {readFile} from "node:fs/promises";
import test from "node:test";
import {ConvergencePlots} from "../src/ml_opt/web/static/convergence.js";

class Element {
  constructor() {this.listeners=new Map();this.attributes={};this.checked=false;this.textContent="";}
  addEventListener(kind,callback) {this.listeners.set(kind,callback);}
  emit(kind,event={}) {this.listeners.get(kind)?.({preventDefault(){},...event});}
  getBoundingClientRect() {return {left:0,width:600,height:240};}
  setAttribute(name,value) {this.attributes[name]=value;}
}

class Context {
  constructor() {this.reset();}
  reset() {this.strokes=[];this.rectangles=[];this.labels=[];this.path=[];}
  numbers(values) {assert.ok(values.every(Number.isFinite),`Nonfinite drawing coordinate: ${values}`);}
  setTransform(...values) {this.numbers(values);}
  clearRect() {this.reset();}
  beginPath() {this.path=[];}
  moveTo(...values) {this.numbers(values);this.path.push(["M",...values]);}
  lineTo(...values) {this.numbers(values);this.path.push(["L",...values]);}
  arc(...values) {this.numbers(values);}
  stroke() {this.strokes.push({color:this.strokeStyle,path:structuredClone(this.path)});}
  fill() {}
  setLineDash() {}
  fillText(text,...values) {this.numbers(values);this.labels.push(text);}
  fillRect(...values) {this.numbers(values);assert.ok(values[3]>=0);this.rectangles.push(values);}
}

async function fixture(t) {
  const html=await readFile(new URL("../src/ml_opt/web/static/index.html",import.meta.url),"utf8");
  const elements=new Map([...html.matchAll(/id="([^"]+)"/g)].map(match=>[match[1],new Element()]));
  const previous=["document","window"].map(key=>[key,Object.getOwnPropertyDescriptor(globalThis,key)]);
  t.after(()=>{for(const [key,value] of previous) {if(value) Object.defineProperty(globalThis,key,value);else delete globalThis[key];}});
  globalThis.document={getElementById:id=>{assert.ok(elements.has(id),`Missing ${id}`);return elements.get(id);}};
  globalThis.window={devicePixelRatio:2};
  for(const key of ["compliance","volume","design","objective","elapsed"]) {
    const context=new Context();elements.get(`convergence-${key}`).getContext=()=>context;
  }
  const plots=new ConvergencePlots();
  return {plots,node:id=>elements.get(id),toggle:key=>{
    const control=elements.get(`convergence-${key}-log`);control.checked=!control.checked;control.emit("change");
  }};
}

const model={volume_fraction:.4,design_tolerance:.005,objective_tolerance:.0001};
const rows=()=>[1,10,100].map((value,iteration)=>({iteration,compliance:value,
  volume_fraction:.4,design_change:iteration ? 1/value : 0,
  relative_objective_change:iteration ? .1/value : 0,elapsed_seconds:[0,.1,1.1][iteration]}));
const curve=plot=>plot.canvas.getContext("2d").strokes.find(item=>item.color===plot.color)?.path;

test("each graph switches independently, uses logarithmic spacing, and retains its scale on updates",async t=>{
  const {plots,toggle}=await fixture(t);plots.render(rows(),model);
  const compliance=plots.plots[0],linear=curve(compliance);
  assert.ok(Math.abs((linear[0][2]-linear[1][2])-(linear[1][2]-linear[2][2]))>10);
  toggle("compliance");
  const logarithmic=curve(compliance);
  assert.ok(Math.abs((logarithmic[0][2]-logarithmic[1][2])-(logarithmic[1][2]-logarithmic[2][2]))<1e-10);
  assert.deepEqual(plots.plots.map(plot=>plot.logarithmic),[true,false,false,false,false]);
  toggle("design");
  assert.deepEqual(plots.plots.map(plot=>plot.logarithmic),[true,false,true,false,false]);
  plots.render(rows(),model);
  assert.equal(compliance.logarithmic,true);
  assert.match(compliance.canvas.attributes["aria-label"],/Logarithmic vertical scale/);
  assert.equal(compliance.readout.textContent,"Iteration 2 · 100");
  toggle("compliance");
  assert.deepEqual(curve(compliance),linear);
  assert.equal(plots.plots[2].logarithmic,true);
  plots.render(rows().map((row,index)=>({...row,volume_fraction:[.1,.4,.8][index]})),model);
  toggle("volume");
  const volume=plots.plots[1];
  const reference=volume.canvas.getContext("2d").strokes.find(item=>item.color==="#a39a85");
  assert.equal(reference.path[0][2],curve(volume)[1][2]);
});

test("log plots leave gaps at nonpositive samples and report the original inspected value",async t=>{
  const {plots,toggle}=await fixture(t);
  const history=[1,0,100,-1,1000].map((compliance,iteration)=>({...rows()[0],iteration,compliance}));
  plots.render(history,model);toggle("compliance");
  const plot=plots.plots[0];
  assert.deepEqual(curve(plot).map(point=>point[0]),["M","M","M"]);
  plot.canvas.emit("focus");
  plot.canvas.emit("keydown",{key:"ArrowLeft"});
  assert.equal(plot.readout.textContent,"Iteration 3 · -1 · Not shown on log scale");
  assert.equal(plots.selected,3);
  plot.canvas.emit("keydown",{key:"Home"});
  assert.equal(plot.readout.textContent,"Iteration 0 · 1");
  plots.render([{...history[0],compliance:0}],model);
  assert.ok(plot.canvas.getContext("2d").labels.includes("No positive values for log scale"));
  assert.match(plot.readout.textContent,/Iteration 0 · 0/);
});

test("log time bars use a positive baseline and tolerate zero, absent, and very small values",async t=>{
  const {plots,toggle}=await fixture(t);plots.render(rows(),model);toggle("elapsed");
  const elapsed=plots.plots.at(-1),context=elapsed.canvas.getContext("2d");
  assert.equal(context.rectangles.length,2);
  assert.ok(context.rectangles[1][3]>context.rectangles[0][3]);
  const history=rows();history[2].elapsed_seconds=history[1].elapsed_seconds;
  plots.render(history,model);
  assert.equal(context.rectangles.length,1);
  assert.match(elapsed.readout.textContent,/0 s · Not shown/);
  for(const plot of plots.plots) if(!plot.logarithmic) toggle(plot.id);
  plots.render([],{});
  plots.render([{...rows()[0],compliance:Number.MIN_VALUE,volume_fraction:0}],model);
  plots.render([{...rows()[0],compliance:Number.MAX_VALUE,volume_fraction:0}],model);
  const missing=rows().map(({elapsed_seconds,...row})=>row);plots.render(missing,model);
  assert.equal(elapsed.readout.textContent,"Timing unavailable for this iteration");
});

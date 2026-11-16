import assert from "node:assert/strict";
import test from "node:test";
import {selectionBounds, selectionNodes, makeSelection, defaultConditions, boundaryError, cantileverSupports}
  from "../src/ml_opt/web/static/mesh-selection.js";
import {regionElements,nonDesignError} from "../src/ml_opt/web/static/mesh-selection.js";

const model = {nx:8,ny:4,length:3,height:1,load_node_x:8,load_node_y:2,load:1,load_angle:-90};

test("non-design surfaces map to elements and reject collisions and impossible volumes",()=>{
  const region={kind:"solid",selection:{kind:"surface",start:[0,0],end:[.25,.5]}};
  assert.deepEqual(regionElements(region.selection,model),[0,1,8,9]);
  assert.equal(nonDesignError([region],{...model,volume_fraction:.4}),"");
  assert.match(nonDesignError([region,{...region,kind:"void"}],model),/overlap/);
  assert.match(nonDesignError([region],{...model,volume_fraction:.1}),/exceed/);
  assert.match(nonDesignError([{...region,kind:"void"}],{...model,volume_constraint:"eq",volume_fraction:.9}),/prevent/);
  assert.match(nonDesignError([{...region,selection:{...region.selection,end:[.01,.5]}}],model),/one element/);
});

test("point, horizontal/vertical line, and surface selections snap to the mesh", () => {
  assert.deepEqual(makeSelection("point", {x:3,y:2}, {x:8,y:4}, model), {kind:"point",start:[.375,.5],end:null});
  const horizontal = makeSelection("line", {x:1,y:1}, {x:7,y:3}, model);
  assert.deepEqual(selectionBounds(horizontal,model), {x0:1,y0:1,x1:7,y1:1});
  assert.deepEqual(selectionBounds(makeSelection("line", {x:2,y:4}, {x:2,y:0}, model),model), {x0:2,y0:0,x1:2,y1:4});
  const surface = makeSelection("surface", {x:6,y:3}, {x:2,y:1}, model);
  assert.equal(selectionNodes(surface,model).length,15);
  assert.equal(makeSelection("surface", {x:0,y:0}, {x:0,y:4}, model),null);
});

test("distributed weights conserve a load over a selected region", () => {
  for (const kind of ["point","line","surface"]) {
    const selection = makeSelection(kind,{x:0,y:0},{x:8,y:4},model);
    assert.ok(Math.abs(selectionNodes(selection,model).reduce((sum,node) => sum+node.weight,0)-1)<1e-14);
  }
});

test("boundary validation detects instability, cancellation, and changed ROM supports", () => {
  const defaults=defaultConditions(model);
  assert.equal(boundaryError(defaults,model),"");
  assert.equal(cantileverSupports(defaults,model),true);
  assert.match(boundaryError([defaults[1]],model),/two distinct/);
  assert.match(boundaryError([defaults[0]],model),/force/);
  assert.match(boundaryError([...defaults,{...defaults[1],angle:90}],model),/force/);
  const changed=[{kind:"fixed",selection:{kind:"line",start:[0,0],end:[1,0]}},defaults[1]];
  assert.equal(boundaryError(changed,model),"");
  assert.equal(cantileverSupports(changed,model),false);
});

test("small selections cannot silently collapse when remeshing", () => {
  const defaults=defaultConditions(model);
  assert.match(boundaryError([...defaults,{kind:"force",selection:{kind:"surface",start:[.1,.1],end:[.11,.11]},magnitude:1,angle:-90}],model),/smaller/);
});

test("canvas hit testing includes boundary nodes without erasing the drawing", async () => {
  const viewer=await import("../src/ml_opt/web/static/viewer.js");
  globalThis.window={devicePixelRatio:2};
  let clears=0;
  const context={clearRect(){clears++;}, measureText:text=>({width:text.length*6})};
  for (const method of ["setTransform","save","restore","beginPath","rect","clip","stroke","fill","fillRect","strokeRect","moveTo","lineTo","closePath","arc","translate","rotate","fillText","setLineDash"]) context[method]=()=>{};
  const canvas={getBoundingClientRect:()=>({width:720,height:300,left:0,top:0}),getContext:()=>context};
  const custom={...model,boundary_conditions:defaultConditions(model)};
  const geometry=viewer.drawModel(canvas,custom,null,{selection:{kind:"surface",start:[.25,.25],end:[.75,.75]},draft:false});
  assert.equal(clears,1);
  assert.deepEqual(viewer.pickNode(canvas,custom,{clientX:geometry.left,clientY:geometry.top+geometry.h}),{x:0,y:0});
  assert.deepEqual(viewer.pickNode(canvas,custom,{clientX:geometry.left+geometry.w,clientY:geometry.top}),{x:8,y:4});
  assert.equal(viewer.pickNode(canvas,custom,{clientX:0,clientY:0}),null);
  assert.equal(clears,1);
  viewer.drawTopology(canvas,custom,new Array(model.nx*model.ny).fill(.5));
  assert.equal(clears,2);
  delete globalThis.window;
});

import assert from "node:assert/strict";
import {readFile} from "node:fs/promises";
import test from "node:test";

// Exercise the editor and canvas hit testing together with an in-memory DOM.
class Element {
  constructor() { this.listeners=new Map(); this.children=[]; this.dataset={}; this.attributes={}; this._value=""; }
  set value(value) { this._value=String(value); }
  get value() { return this._value; }
  addEventListener(kind, callback) { this.listeners.set(kind,[...(this.listeners.get(kind)||[]),callback]); }
  emit(kind, data={}) {
    if (data.node && this.coordinates) data={...data,...this.coordinates(data.node)};
    for (const callback of this.listeners.get(kind)||[]) callback({preventDefault(){},stopPropagation(){},...data});
  }
  setAttribute(name,value) { this.attributes[name]=String(value); }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children=[...children]; }
  reportValidity() { return true; }
  setPointerCapture() { this.capture=true; }
  hasPointerCapture() { return !!this.capture; }
  releasePointerCapture() { this.capture=false; }
}

test("selection, assignment, editing, deletion, keyboard control, and run locking", async () => {
  const html=await readFile(new URL("../src/ml_opt/web/static/index.html",import.meta.url),"utf8");
  const elements=new Map([...html.matchAll(/id="([^"]+)"/g)].map(match=>[match[1],new Element()]));
  const tools=["point","line","surface"].map(mode=>Object.assign(new Element(),{dataset:{selectionMode:mode}}));
  globalThis.document={getElementById:id=>{
    assert.ok(elements.has(id),`Missing element ${id}`); return elements.get(id);
  }, querySelectorAll:()=>tools, createElement:()=>new Element()};
  const node=id=>document.getElementById(id), canvas=node("model-canvas");
  node("boundary-kind").value="fixed";
  node("boundary-magnitude").value=1; node("boundary-angle").value=-90;
  const {BoundaryEditor}=await import("../src/ml_opt/web/static/boundary-editor.js");
  const {drawModel}=await import("../src/ml_opt/web/static/viewer.js");
  const model={nx:8,ny:4,length:3,height:1,load_node_x:8,load_node_y:2,load:1,load_angle:-90};
  globalThis.window={devicePixelRatio:2};
  const context={measureText:text=>({width:text.length*6})};
  for (const method of ["clearRect","setTransform","save","restore","beginPath","rect","clip","stroke","fill","fillRect","strokeRect","moveTo","lineTo","closePath","arc","translate","rotate","fillText","setLineDash"]) context[method]=()=>{};
  canvas.getBoundingClientRect=()=>({width:720,height:300,left:0,top:0});
  canvas.getContext=()=>context;
  canvas.coordinates=node=>{
    const view=drawModel(canvas,model);
    return {clientX:view.left+view.w*node.x/model.nx,clientY:view.top+view.h*(1-node.y/model.ny)};
  };
  let running=false,changes=0,error="";
  const editor=new BoundaryEditor({canvas,getModel:()=>model,isRunning:()=>running,
    onChange:()=>{changes++; model.boundary_conditions=editor.conditions;},onError:value=>{error=value;},onOpen(){},redraw(){}});
  assert.equal(node("boundary-list").children.length,2);
  assert.equal(editor.validationError,"");

  tools[1].emit("click");
  canvas.emit("click",{node:{x:2,y:4}}); canvas.emit("click",{node:{x:6,y:4}});
  assert.equal(node("selection-count").textContent,"5 nodes selected");
  node("boundary-kind").value="force"; node("boundary-kind").emit("change");
  node("boundary-magnitude").value=3; node("boundary-apply").emit("click");
  assert.equal(editor.conditions.length,3); assert.equal(editor.conditions[2].magnitude,3);
  assert.equal(editor.validationError,"");

  node("boundary-list").children[1].children[0].emit("click");
  canvas.emit("click",{node:{x:8,y:3}});
  assert.equal(editor.editing,1);
  node("boundary-apply").emit("click");
  assert.equal(editor.conditions.length,3);
  assert.deepEqual(editor.conditions[1].selection.start,[1,.75]);

  tools[2].emit("click");
  canvas.emit("pointerdown",{button:0,pointerId:1,clientX:10,clientY:10,node:{x:2,y:1}});
  canvas.emit("pointermove",{pointerId:1,clientX:30,clientY:30,node:{x:6,y:3}});
  canvas.emit("pointerup",{pointerId:1,clientX:30,clientY:30,node:{x:6,y:3}});
  canvas.emit("click",{node:{x:6,y:3}});
  assert.equal(editor.selection.kind,"surface");
  assert.equal(node("selection-count").textContent,"15 nodes selected");

  const before=changes; running=true; editor.render();
  node("boundary-apply").emit("click"); tools[0].emit("click");
  node("boundary-list").children[0].children[1].emit("click");
  assert.equal(changes,before); assert.equal(editor.conditions.length,3);
  running=false; editor.render();

  node("boundary-reset").emit("click"); assert.equal(editor.conditions,null);
  tools[0].emit("click");
  canvas.emit("keydown",{key:"ArrowLeft"}); canvas.emit("keydown",{key:"Enter"});
  assert.equal(editor.selection.kind,"point");
  canvas.emit("keydown",{key:"Escape"}); assert.equal(editor.selection,null);
  node("boundary-list").children[0].children[1].emit("click");
  assert.match(editor.validationError,/two distinct/);
  assert.equal(node("boundary-list").children.length,1);
  node("boundary-reset").emit("click");

  tools[2].emit("click");
  node("selection-x0").value="2"; node("selection-y0").value="1";
  node("selection-x1").value="6"; node("selection-y1").value="3";
  node("boundary-select-coordinates").emit("click");
  assert.equal(node("selection-count").textContent,"15 nodes selected");
  model.nx=16; model.ny=8; editor.render();
  assert.equal(node("selection-count").textContent,"45 nodes selected");
  node("selection-x0").value="999"; node("boundary-select-coordinates").emit("click");
  assert.match(error,/valid mesh/);
  node("boundary-reset").emit("click");
  node("case-add").emit("click");
  assert.equal(editor.cases.length,2);
  assert.equal(editor.activeCase,"case-2");
  assert.equal(editor.validationError,"");
  assert.ok(editor.conditions.some(item=>item.case_id==="case-2"));
  editor.setResults({"case-2":5}); assert.match(node("case-result").textContent,/5/);
  node("load-aggregation").value="worst"; node("load-aggregation").emit("change");
  assert.equal(editor.aggregation,"worst"); assert.equal(node("case-weight").disabled,true);
  node("case-remove").emit("click"); assert.equal(editor.cases.length,1);
  assert.ok(editor.conditions.every(item=>item.case_id!=="case-2"));
  tools[1].emit("click"); canvas.emit("click",{node:{x:16,y:0}}); canvas.emit("click",{node:{x:16,y:8}});
  node("boundary-kind").value="displacement"; node("boundary-components").value="x"; node("boundary-kind").emit("change");
  node("boundary-ux").value=.03; node("boundary-uy").value=9;
  node("boundary-apply").emit("click");
  const motion=editor.conditions.at(-1); assert.equal(motion.kind,"displacement");
  assert.equal(motion.ux,.03); assert.equal(motion.uy,0); assert.equal(editor.validationError,"");
  node("boundary-kind").value="solid";node("boundary-kind").emit("change");
  assert.equal(editor.mode,"surface");
  assert.equal(tools[0].disabled,true);
  node("selection-x0").value="0";node("selection-y0").value="0";
  node("selection-x1").value="2";node("selection-y1").value="2";
  node("boundary-select-coordinates").emit("click");
  assert.equal(node("selection-count").textContent,"4 elements selected");
  node("boundary-apply").emit("click");
  assert.equal(editor.regions.length,1);assert.equal(editor.regions[0].kind,"solid");
  node("boundary-kind").value="void";node("boundary-kind").emit("change");
  node("boundary-apply").emit("click");
  assert.match(error,/overlap/);assert.equal(editor.regions.length,1);
  node("non-design-list").children[0].children[0].emit("click");
  node("selection-x1").value="3";node("boundary-select-coordinates").emit("click");
  node("boundary-apply").emit("click");
  assert.equal(editor.regions.length,1);
  assert.equal(node("non-design-list").children[0].children[0].children[1].textContent,"6 elements");
  running=true;editor.render();
  node("non-design-list").children[0].children[1].emit("click");
  assert.equal(editor.regions.length,1);
  running=false;editor.render();
  const persisted={...model,non_design_regions:structuredClone(editor.regions)};
  editor.loadModel(persisted);assert.deepEqual(editor.regions,persisted.non_design_regions);
  node("non-design-list").children[0].children[1].emit("click");
  assert.equal(editor.regions.length,0);
  delete globalThis.document;
  delete globalThis.window;
});

import assert from "node:assert/strict";
import test from "node:test";

class Element {
  constructor(text="",dataset={},controls="") {this.textContent=text;this.dataset=dataset;this.attrs={"aria-controls":controls};this.events=new Map();this.hidden=false;}
  setAttribute(name,value){this.attrs[name]=value;}
  getAttribute(name){return this.attrs[name];}
  addEventListener(name,callback){this.events.set(name,[...(this.events.get(name)||[]),callback]);}
  emit(name,key){for(const fn of this.events.get(name)||[]) fn({key,preventDefault(){}});}
  focus(){Element.focused=this;}
}

test("Regression dropdown routes both views and supports keyboard, history, and legacy links",async()=>{
  const names=["document","window","location","history","CustomEvent"];
  const previous=new Map(names.map(name=>[name,Object.getOwnPropertyDescriptor(globalThis,name)]));
  const optimisation=new Element("Optimisation",{workspace:"cantilever-2d"},"workspace-cantilever");
  const ml=new Element("Machine learning",{workspace:"ml"},"workspace-ml");
  const regression=new Element("Regression ▾",{method:"regression"},"workspace-regression");
  const rom=new Element("Reduced models",{method:"rom"},"workspace-rom-training");
  const compliance=new Element("Compliance",{regressionView:"compliance"});
  const topology=new Element("Topology",{regressionView:"topology"});
  const menu=new Element();menu.hidden=true;menu.querySelectorAll=()=>[compliance,topology];
  const nodes=new Map([["tab-regression",regression],["regression-menu",menu],
    ...["workspace-cantilever","workspace-ml","workspace-regression","workspace-rom-training"].map(id=>[id,new Element()])]);
  const documentEvents=new Map(),windowEvents=new Map(),viewEvents=[];
  try {
    globalThis.location={hash:"#ml/regression/topology"};
    globalThis.history={pushState(_a,_b,hash){location.hash=hash;}};
    globalThis.window={addEventListener:(name,fn)=>windowEvents.set(name,fn)};
    globalThis.CustomEvent=class {constructor(type,{detail}){this.type=type;this.detail=detail;}};
    globalThis.document={getElementById:id=>nodes.get(id),
      querySelectorAll:selector=>selector.startsWith(".workspace-tabs") ? [optimisation,ml] : [regression,rom],
      addEventListener:(name,fn)=>documentEvents.set(name,fn),
      dispatchEvent:event=>{if(event.type==="regressionviewchange") viewEvents.push(event.detail);}};
    await import("../src/ml_opt/web/static/navigation.js");
    assert.match(document.title,/Regression · Topology/);
    assert.equal(nodes.get("workspace-ml").hidden,false);
    assert.equal(topology.attrs["aria-checked"],"true");
    regression.emit("click");assert.equal(menu.hidden,false);assert.equal(Element.focused,topology);
    topology.emit("keydown","ArrowUp");assert.equal(Element.focused,compliance);
    compliance.emit("click");assert.equal(location.hash,"#ml/regression/compliance");assert.equal(menu.hidden,true);
    assert.equal(Element.focused,regression);assert.equal(viewEvents.at(-1),"compliance");
    regression.emit("keydown","ArrowDown");assert.equal(menu.hidden,false);
    compliance.emit("keydown","Escape");assert.equal(menu.hidden,true);assert.equal(Element.focused,regression);
    regression.emit("keydown","ArrowRight");assert.equal(location.hash,"#ml/rom");assert.equal(Element.focused,rom);
    rom.emit("keydown","ArrowLeft");assert.equal(location.hash,"#ml/regression/compliance");
    regression.emit("click");topology.emit("click");
    optimisation.emit("click");ml.emit("click");assert.equal(location.hash,"#ml/regression/topology");
    regression.emit("click");documentEvents.get("click")({target:{closest:()=>null}});assert.equal(menu.hidden,true);
    location.hash="#ml/regression";windowEvents.get("hashchange")();assert.equal(viewEvents.at(-1),"compliance");
    location.hash="#rom-training";windowEvents.get("hashchange")();assert.equal(nodes.get("workspace-rom-training").hidden,false);
  } finally {
    for(const name of names){const descriptor=previous.get(name);if(descriptor)Object.defineProperty(globalThis,name,descriptor);else delete globalThis[name];}
  }
});

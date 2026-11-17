import assert from "node:assert/strict";
import test from "node:test";
import {editDensity,fixedElements,predictDesign,comparableReports} from "../src/ml_opt/web/static/design-models.js";
import {pickElement} from "../src/ml_opt/web/static/viewer.js";

test("density edits preserve fixed regions and portable models use layout rather than volume",()=>{
  const config={nx:3,ny:2,non_design_regions:[{kind:"solid",selection:{start:[0,0],end:[1/3,1]}}]};
  assert.deepEqual([...fixedElements(config)],[[0,1],[3,1]]);
  const density=[1,.2,.8,1,.5,.5];
  assert.deepEqual(editDensity(density,config,2,.1),[1,.2,.1,1,.5,.5]);
  assert.deepEqual(density,[1,.2,.8,1,.5,.5]);
  assert.throws(()=>editDensity(density,config,0,0),/fixed/);
  assert.throws(()=>editDensity(density,config,6,.5),/element/);
  assert.throws(()=>editDensity(density,config,1,NaN),/between/);
  const model={kind:"neural_network",target:"compliance",input_kind:"density",input_indices:[1,2],input_offset:.5,input_scale:.5,
    target_offset:2,target_scale:1,target_transform:"log",output_activation:"linear",
    weights:{hidden_weight:[[1,2]],hidden_bias:[0],output_weight:[[1]],output_bias:[0]}};
  assert.ok(predictDesign(model,density)>0);
  assert.notEqual(predictDesign(model,density),predictDesign(model,[1,.8,.2,1,.5,.5]));
  assert.throws(()=>predictDesign(model,[1,-.2,1]),/Invalid/);
  const report={train_indices:[0,3],validation_indices:[1],test_indices:[2]};
  assert.ok(comparableReports(report,{...report,train_indices:[3,0]}));
  assert.ok(!comparableReports(report,{...report,test_indices:[1]}));
});

test("element picking uses density row order and handles mesh boundaries",()=>{
  const old=globalThis.window;globalThis.window={devicePixelRatio:1};
  try {
    const canvas={width:720,height:300,getBoundingClientRect:()=>({width:720,height:300,left:0,top:0}),getContext:()=>({setTransform(){}})};
    const config={nx:3,ny:2,length:3,height:1};
    // scale=min(600/3,218/1)=200; left=60, top=47.
    assert.equal(pickElement(canvas,config,{clientX:61,clientY:246}),0);
    assert.equal(pickElement(canvas,config,{clientX:61,clientY:47}),3);
    assert.equal(pickElement(canvas,config,{clientX:659,clientY:48}),5);
    assert.equal(pickElement(canvas,config,{clientX:660,clientY:47}),null);
    assert.equal(pickElement(canvas,config,{clientX:60,clientY:247}),null);
  } finally {globalThis.window=old;}
});

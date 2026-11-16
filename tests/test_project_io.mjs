import assert from "node:assert/strict";
import test from "node:test";
import {parseProject,serialiseProject} from "../src/ml_opt/web/static/project-io.js";

const model={nx:2,ny:1,max_iterations:5,robust:false,non_design_regions:[{kind:"solid",selection:{kind:"surface",start:[0,0],end:[.5,1]}}],boundary_conditions:[{kind:"restrain_y",selection:{kind:"point",start:[1,0],end:null}}],load_cases:[{id:"a",name:"One",weight:1},{id:"b",name:"Two",weight:2}],load_aggregation:"worst"};
const result={density:[1,0],fields:null,history:[{iteration:0,compliance:4,volume_fraction:.5,elapsed_seconds:1},{iteration:2,compliance:2,volume_fraction:.5,elapsed_seconds:3}],stopping_tolerances:{design_tolerance:.005}};

test("project files preserve supports, load cases, settings, and the latest snapshot",()=>{
  const restored=parseProject(serialiseProject(model,result));
  assert.deepEqual(restored,{model,result});
  assert.equal(parseProject(serialiseProject(model,null)).result,null);
});

test("inconsistent and corrupt snapshots are rejected before restoration",()=>{
  for(const bad of [{...result,density:[1]},{...result,density:[-1,2]},
    {...result,history:[{iteration:0,compliance:Infinity,volume_fraction:.5}]},
    {...result,history:[...result.history,{iteration:1,compliance:1,volume_fraction:.5}]},
    {...result,stopping_tolerances:{design_tolerance:-1}}]) {
    assert.throws(()=>parseProject(serialiseProject(model,bad)));
  }
  assert.throws(()=>parseProject('{"format":"other","version":1}'));
  assert.throws(()=>parseProject(serialiseProject({...model,nx:0},null)));
});

test("robust snapshots must include matching fields and field metrics",()=>{
  const robust={...model,robust:true};
  assert.throws(()=>parseProject(serialiseProject(robust,result)));
  const row={...result.history[0],field_compliances:{eroded:4,nominal:3,dilated:2},field_volumes:{eroded:.6,nominal:.65,dilated:.7}};
  const snapshot={...result,density:[1,.3],fields:{eroded:[1,.2],nominal:[1,.3],dilated:[1,.4]},history:[row]};
  assert.deepEqual(parseProject(serialiseProject(robust,snapshot)).result,snapshot);
});

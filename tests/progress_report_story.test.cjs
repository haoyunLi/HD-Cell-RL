const test=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const path=require('node:path');
const root=path.resolve(__dirname,'..');
const M=require(path.join(root,'web/progress-report/story-math.js'));
const d=M.makeDemo();
const close=(a,b,tol=1e-10)=>assert.ok(Math.abs(a-b)<tol,`${a} != ${b}`);
const context={window:{}};
vm.runInNewContext(fs.readFileSync(path.join(root,'web/progress-report/assets/story-data.js'),'utf8'),context);
const data=context.window.PROGRESS_STORY_DATA;

test('schematic includes every nucleus within MaxDis, including five for query b',()=>{
  assert.equal(d.bins[d.query].candidates.length,5);
  d.bins.forEach(b=>assert.deepEqual(b.candidates,d.cells.map((_,i)=>i).filter(i=>Math.hypot(d.cells[i].x-b.x,d.cells[i].y-b.y)<=20)));
});
test('soft ownership normalizes and nuclear rows remain exactly one-hot',()=>{
  [d.spatial,d.low,d.high,d.r].forEach(r=>r.forEach((row,i)=>{
    close(row.reduce((a,b)=>a+b,0),1);
    row.forEach(p=>assert.ok(Number.isFinite(p)&&p>=0&&p<=1));
    if(d.bins[i].lock>=0)assert.deepEqual(row,d.bins[i].candidates.map(c=>c===d.bins[i].lock?1:0));
  }));
});
test('spatial-only E-step reproduces spatial prior',()=>{
  const r=M.eStep(d.bins,d.cells,d.q0,{beta:0});
  r.forEach((row,i)=>row.forEach((v,j)=>close(v,d.spatial[i][j])));
});
test('B expression overcomes A distance, more strongly at high confidence',()=>{
  const i=d.query; assert.ok(d.spatial[i][0]>d.spatial[i][1]);
  assert.ok(d.high[i][1]>d.high[i][0]);assert.ok(d.high[i][1]>d.low[i][1]);
});
test('E/M is batch order independent; M does not accumulate previous iterations',()=>{
  const bins=d.bins.slice().reverse();const r=M.eStep(bins,d.cells,d.q0);
  const original=M.eStep(d.bins,d.cells,d.q0);
  r.slice().reverse().forEach((row,i)=>row.forEach((v,j)=>close(v,original[i][j])));
  const q=M.mStep(bins,d.cells,r),q2=M.mStep(d.bins,d.cells,original);
  q.forEach((row,i)=>row.forEach((v,j)=>close(v,q2[i][j])));
  assert.deepEqual(M.mStep(d.bins,d.cells,d.r),M.mStep(d.bins,d.cells,d.r));
});
test('M-step uses fractional evidence, and never changes nuclear profiles',()=>{
  const before=JSON.stringify(d.cells);const b={candidates:[0,1],conf:.8,ll:[-1,-3,-2]};
  const q=M.mStep([b],d.cells,[[.8,.2]]);
  [.8,.2].forEach((w,i)=>M.softmax(b.ll.map(v=>v*.8*w)).forEach((v,j)=>close(v,q[i][j])));
  assert.equal(JSON.stringify(d.cells),before);
});
test('normalized entropy has correct endpoints',()=>{
  close(M.entropy([.5,.5]),1);close(M.entropy([1/3,1/3,1/3]),1);
  assert.equal(M.entropy([1]),0);assert.equal(M.entropy([1,0]),0);assert.ok(M.entropy([.99,.01])<.1);
});
test('historical weighted deltas reconcile and both GT outcomes are genuine',()=>{
  assert.equal(data.reward_cases.length,2);
  data.reward_cases.forEach(c=>{
    close(Object.values(c.components).reduce((a,b)=>a+b,0),c.reward);
    if(c.outcome==='corrected'){assert.notEqual(c.old_gt,c.gt_owner);assert.equal(c.new_gt,c.gt_owner);}
    else {assert.equal(c.old_gt,c.gt_owner);assert.notEqual(c.new_gt,c.gt_owner);}
  });
});
test('all contexts preserve loaded bin universe and nuclear shift audit',()=>{
  assert.equal(data.patches.length,4);assert.equal(data.patches.reduce((s,p)=>s+p.bins.length,0),10827);
  const expected=[[2627,14,0],[2581,267,46],[2533,731,94]];
  for(let shift=0;shift<3;shift++){
    let valid=0,bad=0,lost=0;
    data.patches.forEach(p=>{const lookup=new Map(p.bins.map(b=>[b[0]+','+b[1],b]));
      p.bins.filter(b=>b[4]>=0).forEach(b=>{const dest=lookup.get(b[0]+','+(b[1]+shift));if(!dest){lost++;return;}valid++;if(dest[3]!==b[4])bad++;});
    });
    assert.deepEqual([valid,bad,lost],expected[shift]);
  }
});
test('support BA matches the saved realization, not the three-seed average',()=>{
  const bins=data.patches.flatMap(p=>p.bins),inside=bins.filter(b=>b[2]>0),outside=bins.filter(b=>b[2]<=0);
  [5,6,7].forEach((field,i)=>{const ba=(inside.filter(b=>b[field]>0).length/inside.length+outside.filter(b=>b[field]===0).length/outside.length)/2;
    close(ba,data.support_realization[['source_high_depth','depth_uniform','real'][i]]);
  });
});
test('profile chart contains actual sorted samples; presentation sources contain no internal paths',()=>{
  assert.equal(data.profile.synthetic.length,60);
  for(const name of ['raw','synthetic']){const a=data.profile[name];assert.ok(a.length>1);a.forEach((v,i)=>assert.ok(Number.isInteger(v)&&v>=0&&(i===0||v>=a[i-1])));}
  for(const name of ['index.html','story.js','story-math.js','assets/story-data.js']){
    assert.doesNotMatch(fs.readFileSync(path.join(root,'web/progress-report',name),'utf8'),/\/projects\/|\/taiga\//);
  }
});

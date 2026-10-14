/* Deterministic educational example, not the production EM implementation. */
(function (root) {
  'use strict';
  const softmax = xs => { const m = Math.max(...xs); const e = xs.map(x => Math.exp(x - m)); const s = e.reduce((a,b)=>a+b,0); return e.map(x=>x/s); };
  const dot = (a,b) => a.reduce((s,x,i)=>s+x*b[i],0);
  const entropy = r => r.length <= 1 ? 0 : Math.max(0,Math.min(1,-r.reduce((s,p)=>s+(p ? p*Math.log(p):0),0)/Math.log(r.length)));
  function eStep(bins, cells, q, {beta=1.5, confidence=null, damping=1, previous=null}={}) {
    return bins.map((b,bi)=>{
      if (b.lock >= 0) return b.candidates.map(c=>c===b.lock?1:0);
      const scores=b.candidates.map((c,j)=>Math.log(b.spatial[j])+beta*(confidence ?? b.conf)*(.5*dot(q[c],b.ll)+.5*dot(b.expression,cells[c].profile.map(Math.log))));
      const raw=softmax(scores);
      return raw.map((p,j)=>previous ? damping*p+(1-damping)*previous[bi][j] : p);
    });
  }
  function mStep(bins, cells, r) {
    const scores=cells.map(()=>[0,0,0]); // Uniform prior; reset evidence each cycle.
    bins.forEach((b,bi)=>b.candidates.forEach((c,j)=>b.ll.forEach((v,k)=>{scores[c][k]+=r[bi][j]*b.conf*v;})));
    return scores.map(softmax);
  }
  function makeDemo() {
    const cells=[{id:'A',x:10,y:12,color:'#168d88',profile:[.84,.10,.06]},
      {id:'B',x:22,y:12,color:'#b62569',profile:[.07,.87,.06]},
      {id:'C',x:8,y:24,color:'#526e9a',profile:[.07,.13,.80]},
      {id:'D',x:24,y:26,color:'#a16627',profile:[.20,.12,.68]},
      {id:'E',x:2,y:24,color:'#725991',profile:[.18,.16,.66]}];
    const theta=[[.88,.06,.06],[.06,.88,.06],[.06,.06,.88]];
    const bins=[];
    for(let y=0;y<=32;y+=2) for(let x=0;x<=32;x+=2) {
      const d=cells.map(c=>Math.hypot(c.x-x,c.y-y));
      const candidates=cells.map((_,i)=>i).filter(i=>d[i]<=20);
      if(!candidates.length) continue;
      const lock=cells.findIndex(c=>c.x===x&&c.y===y);
      const nearest=d.indexOf(Math.min(...d));
      const expression=(lock>=0?cells[lock].profile:(x>=14&&y<20?[.08,.86,.06]:cells[nearest].profile)).slice();
      const ll=theta.map(t=>dot(expression,t.map(Math.log)));
      bins.push({x,y,lock,candidates,spatial:softmax(candidates.map(c=>-((d[c]/20)**2))),expression,ll,conf:.8});
    }
    const q0=cells.map((_,c)=>softmax(bins.filter(b=>b.lock===c).reduce((s,b)=>s.map((v,k)=>v+b.conf*b.ll[k]),[0,0,0])));
    const spatial=bins.map(b=>b.lock>=0?b.candidates.map(c=>c===b.lock?1:0):b.spatial);
    const r=eStep(bins,cells,q0,{previous:spatial,damping:.5});
    return {cells,bins,q0,spatial,r,q1:mStep(bins,cells,r),query:bins.findIndex(b=>b.x===14&&b.y===12),
      low:eStep(bins,cells,q0,{confidence:.08}),high:eStep(bins,cells,q0,{confidence:.8})};
  }
  const api={softmax,entropy,eStep,mStep,makeDemo};
  if(typeof module!=='undefined'&&module.exports) module.exports=api;
  else root.ProgressStoryMath=api;
})(typeof window==='undefined'?globalThis:window);

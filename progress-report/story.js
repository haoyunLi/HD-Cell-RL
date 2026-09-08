/* Editable, source-backed slide scenes. No inference runs in this presentation. */
(() => {
  'use strict';
  const M=window.ProgressStoryMath, D=window.PROGRESS_STORY_DATA, demo=M.makeDemo();
  const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const pct=x=>(100*x).toFixed(1)+'%';
  const colors=demo.cells.map(c=>c.color);
  const steps={
    simulation:['Nuclear anchors','Simulated cell masks','Place RNA','Measure bins','Hide truth'],
    em:['Distance','Expression','All bins together','Update cell types'],
    replace:['Current state','Policy choice','REPLACE','Global delta','Reveal GT'],
    support:['Observe counts','Overlay pseudo coverage','Thin counts'],
    shift:['0 µm','2 µm','4 µm'],
  };
  const descriptions={
    simulation:[
      ['Start from what the image reveals','The same labelled nuclei A–E stay in place throughout this demonstration. Real H&E provides nuclear observations, not known whole-cell boundaries.'],
      ['Create a testable, simulated answer','Cell regions grow around nuclear anchors. These are simulator-defined masks, not verified real cell boundaries. A square can cross adjacent regions.'],
      ['Give each source cell expression','The schematic marks indicate compartment-biased RNA placement. They are illustrative samples, not literal UMI counts or inferred RNA motion.'],
      ['The assay hides the source labels','Aggregate molecules into fixed 2 µm squares. The inference input is one expression vector per physical barcode, not separate observations for each candidate cell.'],
      ['Keep the answer key out of inference','Inference receives nuclei + bin counts + an independent scRNA reference. Source ownership is opened only for evaluation. Generator and reference donors are disjoint.'],
    ],
    em:[
      ['A is closer to bin b','All five shown nuclei are valid candidates within the existing 20 µm MaxDis. Spatial evidence prefers A; there is no fixed candidate-count limit.'],
      ['Expression can favor B','This bin has a B-like expression pattern. Compare low and high expression confidence: stronger evidence changes ownership more. The bin does not physically move.'],
      ['One E-step uses one fixed set of cell posteriors','Every non-nuclear bin is updated from the same q. Each small stacked square is a soft ownership distribution; nuclear anchor rows remain one-hot. Damping blends old and new probabilities.'],
      ['Only then update all cell-type posteriors','The completed responsibility matrix weights the M-step evidence. Cell-specific nuclear profiles stay fixed in the current baseline. Cell-type probability is not physical-cell ownership.'],
    ],
    replace:[
      ['One physical barcode, one current owner','The transfer drawing is schematic. Cell IDs, action ranking and weighted reward changes below come from a saved August rollout, not a new experiment.'],
      ['PPO chooses among legal actions','The policy makes a discrete choice; it does not necessarily choose the largest immediate global delta. GT is not a policy input.'],
      ['Remove from A and add to B','The same physical barcode changes owner. Both affected cells are rebuilt: expression, type posterior, geometry, neighborhood and shape.'],
      ['The reward is the whole-patch change','Bars show already-weighted contributions to J_after − J_before. A favorable expression or shape change can offset other penalties. Numerical residuals are retained.'],
      ['Now open the evaluation answer key','Positive objective change is not proof of a correct assignment. Compare one corrected and one damaged historical action. Neither case establishes performance on the latest data.'],
    ],
    support:[
      ['Can a one-bit image reveal the simulator?','Black means total counts > 0. These are actual saved matrices on the same barcode grid, not generated microscopy. Inspect any of the four contexts.'],
      ['Overlay the pseudo coverage boundary','The magenta outline is the union of bins with any pseudo cell coverage, not real cell GT and not individual cell identities. Pseudo nonzero support follows it closely.'],
      ['Thinning leaves the empty regions empty','Show the saved depth-only control, seed 20260907. This single realization differs slightly from the three-seed summary.'],
    ],
    shift:[
      ['Keep pseudo GT fixed','Baseline nuclear-bin locations are overlaid on the pseudo coverage. Dark marks retain the same dominant owner; orange crosses do not.'],
      ['Shift only observed nuclear bins','Move observations one array column (2 µm). Some seed bins now lie over a different pseudo dominant owner. This is a sensitivity control, not a measured registration error.'],
      ['Larger shifts expose the anchor assumption','Move observations two array columns (4 µm). Cropped-out destinations are counted separately. Realistic noise must be estimated independently, not tuned to lower tool scores.'],
    ],
  };

  function schematicMolecules() {
    const points=[];
    demo.cells.forEach((c,ci)=>{for(let n=0;n<21;n++){
      const angle=n*2.399+ci,radius=n<5?Math.sqrt(n+1)*.55:(n<14?2+((n*7)%11)/3:6.6);
      const x=c.x+Math.cos(angle)*radius,y=c.y+Math.sin(angle)*radius;
      const nearest=demo.cells.map(p=>Math.hypot(p.x-x,p.y-y));
      if(x>=0&&x<=32&&y>=0&&y<=32&&nearest.indexOf(Math.min(...nearest))===ci) points.push({x,y,cell:ci});
    }});
    return points;
  }
  function svgScene(mode='seeds', r=null) {
    const unit=11.5, off=44, xy=v=>off+v*unit;
    let s='<svg class="cell-scene" viewBox="0 0 480 460" role="img" aria-label="Schematic fixed nuclei A to E and a selected boundary bin b">';
    s+='<rect x="30" y="30" width="407" height="407" fill="#fff" stroke="#dfe3e8"/>';
    for(let a=0;a<=34;a+=2) s+=`<path d="M${xy(a)-unit} 30V437 M30 ${xy(a)-unit}H437" stroke="#e9edf0" stroke-width="1"/>`;
    if(mode==='masks'||mode==='rna') {
      demo.bins.forEach(b=>{
        const distances=demo.cells.map(c=>Math.hypot(c.x-b.x,c.y-b.y)); const c=distances.indexOf(Math.min(...distances));
        if(distances[c]<8) s+=`<rect x="${xy(b.x)-unit}" y="${xy(b.y)-unit}" width="23" height="23" fill="${colors[c]}" opacity=".16"/>`;
      });
      if(mode==='rna') schematicMolecules().forEach(p=>{s+=`<circle cx="${xy(p.x)}" cy="${xy(p.y)}" r="3" fill="${colors[p.cell]}"/>`;});
    }
    if(mode==='bins'||mode==='blind') {
      const counts=new Map();schematicMolecules().forEach(p=>{const key=(2*Math.round(p.x/2))+','+(2*Math.round(p.y/2));counts.set(key,(counts.get(key)||0)+1);});
      demo.bins.forEach(b=>{const n=counts.get(b.x+','+b.y)||0;if(n)s+=`<rect x="${xy(b.x)-unit+1}" y="${xy(b.y)-unit+1}" width="21" height="21" fill="#6b7e8e" opacity="${Math.min(.9,.15+.12*n)}"/>`;});
    }
    if(r) demo.bins.forEach((b,i)=>{
      let x=xy(b.x)-unit+1;
      b.candidates.forEach((c,j)=>{const w=21*r[i][j]; if(w>.01) s+=`<rect x="${x}" y="${xy(b.y)-unit+1}" width="${w}" height="21" fill="${colors[c]}" opacity=".8"/>`; x+=w;});
    });
    demo.cells.forEach(c=>{s+=`<circle cx="${xy(c.x)}" cy="${xy(c.y)}" r="15" fill="${c.color}" stroke="white" stroke-width="2"/><text x="${xy(c.x)}" y="${xy(c.y)+6}" text-anchor="middle" font-size="18" fill="white" font-weight="700">${c.id}</text>`;});
    const b=demo.bins[demo.query]; s+=`<rect x="${xy(b.x)-unit}" y="${xy(b.y)-unit}" width="23" height="23" fill="white" stroke="#101b2d" stroke-width="2"/><text x="${xy(b.x)}" y="${xy(b.y)+6}" text-anchor="middle" font-size="18">b</text>`;
    return s+'</svg>';
  }
  function ownerBars(r, title) {
    const b=demo.bins[demo.query];
    return `<div class="owner-bars"><h3>${title}</h3>${b.candidates.map((c,j)=>`<div class="owner-row"><b>${demo.cells[c].id}</b><span><i style="width:${r[j]*100}%;background:${colors[c]}"></i></span><strong>${pct(r[j])}</strong></div>`).join('')}</div>`;
  }
  function qTable() {
    return `<div class="type-posteriors"><h3>Cell-type posterior q: before → after</h3><p>Type 1 / Type 2 / Type 3 · illustrative reference</p>${demo.cells.map((c,i)=>`<div><b style="color:${c.color}">${c.id}</b><span>${demo.q0[i].map(pct).join(' / ')}</span><b>→</b><span>${demo.q1[i].map(pct).join(' / ')}</span></div>`).join('')}<small>Nuclear profile: unchanged. Scores are recomputed, not repeatedly accumulated.</small></div>`;
  }
  function setup(host) {
    const kind=host.dataset.story, names=steps[kind];
    host.innerHTML=`<div class="story-frame"><div class="story-visual"></div><div class="story-reading"></div></div><div class="story-stepper" role="group" aria-label="Demonstration steps"><div class="story-progress"></div><button type="button" data-command="replay">Replay</button><button type="button" data-command="back">Back</button><button type="button" data-command="next">Next →</button></div><p class="story-status" aria-live="polite"></p><div class="story-fragments" aria-hidden="true">${names.slice(1).map((_,i)=>`<span class="fragment" data-fragment-index="${i}"></span>`).join('')}</div>`;
    const params=new URLSearchParams(location.search);
    host._case=Math.max(0,D.reward_cases.findIndex(c=>c.outcome===params.get('case')));
    host._patch=Math.max(0,D.patches.findIndex(p=>p.patch_id===params.get('patch')));
    host.addEventListener('click',e=>{
      const control=e.target.closest('button'); if(!control) return;
      if(control.dataset.case!==undefined){host._case=Number(control.dataset.case);render(host);return;}
      if(control.dataset.step!==undefined){go(host,Number(control.dataset.step));return;}
      if(control.dataset.command==='replay') go(host,0);
      if(control.dataset.command==='back') go(host,Math.max(0,step(host)-1));
      if(control.dataset.command==='next') step(host)<names.length-1?go(host,step(host)+1):Reveal.next();
    });
    host.addEventListener('change',e=>{if(e.target.matches('[data-patch-select]')){host._patch=Number(e.target.value);render(host);}});
  }
  function step(host) {
    if(location.search.includes('print-pdf')) return steps[host.dataset.story].length-1;
    return Math.max(0,Math.min(steps[host.dataset.story].length-1,host.querySelectorAll('.fragment.visible').length));
  }
  function go(host,n){const sections=[...document.querySelectorAll('.slides > section')];Reveal.slide(sections.indexOf(host.closest('section')),0,n-1);render(host);}
  function patchSelect(host) {
    return `<label class="patch-select">Saved context <select data-patch-select>${D.patches.map((p,i)=>`<option value="${i}" ${i===host._patch?'selected':''}>${p.patch_id} · ${p.partition}</option>`).join('')}</select></label>`;
  }
  function drawMap(canvas,p,mode,outline=false,shift=null) {
    const u=7,pad=8;canvas.width=p.cols*u+2*pad;canvas.height=p.rows*u+2*pad+26;
    const ctx=canvas.getContext('2d');ctx.fillStyle='#e7ebef';ctx.fillRect(0,0,canvas.width,canvas.height-26);
    const lookup=new Map(p.bins.map(b=>[b[0]+','+b[1],b]));
    const countIndex={source_high_depth:5,depth_uniform:6,real:7}[mode];
    p.bins.forEach(b=>{ctx.fillStyle=mode==='coverage'?(b[2]>0?'#c7deda':'#fff'):(b[countIndex]>0?'#263443':'#fff');ctx.fillRect(pad+b[1]*u,pad+b[0]*u,u,u);});
    if(outline){ctx.strokeStyle='#b62569';ctx.lineWidth=1.2;ctx.beginPath();p.bins.filter(b=>b[2]>0).forEach(b=>{
      const x=pad+b[1]*u,y=pad+b[0]*u;
      for(const [dy,dx,edge] of [[-1,0,0],[0,1,1],[1,0,2],[0,-1,3]]){
        if((lookup.get((b[0]+dy)+','+(b[1]+dx))?.[2]??0)>0) continue;
        if(edge===0){ctx.moveTo(x,y);ctx.lineTo(x+u,y);}if(edge===1){ctx.moveTo(x+u,y);ctx.lineTo(x+u,y+u);}
        if(edge===2){ctx.moveTo(x,y+u);ctx.lineTo(x+u,y+u);}if(edge===3){ctx.moveTo(x,y);ctx.lineTo(x,y+u);}
      }
    });ctx.stroke();}
    if(shift!==null){let bad=0,valid=0,lost=0; p.bins.filter(b=>b[4]>=0).forEach(b=>{
      const dest=lookup.get(b[0]+','+(b[1]+shift));if(!dest){lost++;return;}valid++;
      const wrong=dest[3]!==b[4];if(wrong)bad++;
      const x=pad+dest[1]*u,y=pad+dest[0]*u;ctx.fillStyle=wrong?'#c36519':'#101b2d';ctx.fillRect(x+1,y+1,u-2,u-2);
      if(wrong){ctx.strokeStyle='#fff';ctx.lineWidth=.8;ctx.beginPath();ctx.moveTo(x+1,y+1);ctx.lineTo(x+u-1,y+u-1);ctx.moveTo(x+u-1,y+1);ctx.lineTo(x+1,y+u-1);ctx.stroke();}
    });canvas.dataset.shiftSummary=JSON.stringify({bad,valid,lost});}
    ctx.fillStyle='#fff';ctx.fillRect(0,canvas.height-26,canvas.width,26);ctx.fillStyle='#101b2d';ctx.fillRect(pad,canvas.height-17,5*u,2);ctx.font='12px Arial';ctx.fillText('10 µm · barcode-array axes',pad+5*u+8,canvas.height-12);
  }
  function render(host) {
    const kind=host.dataset.story,n=step(host),visual=host.querySelector('.story-visual'),reading=host.querySelector('.story-reading');
    host.dataset.state=String(n);
    const [title,copy]=descriptions[kind][n];
    reading.innerHTML=`<h3 class="step-title">${title}</h3><p class="step-copy">${copy}</p>`;
    if(kind==='simulation') {
      visual.innerHTML=`<p class="evidence-tag">Schematic · same cells A–E</p>${svgScene(['seeds','masks','rna','bins','blind'][n])}<p class="visual-key">Nuclei stay fixed · whole-cell masks are simulated</p>`;
      reading.innerHTML+=`<div class="truth-lanes"><div><b>INFERENCE INPUT</b><span>Nuclei + bin counts + scRNA reference</span></div><div class="${n===4?'truth-open':'truth-hidden'}"><b>GT: EVALUATION ONLY</b><span>Source cell IDs and fractional contributions</span></div></div><p class="mini-claim">Generate → hide labels → infer → evaluate</p>`;
    }
    if(kind==='em') {
      visual.innerHTML=`<p class="evidence-tag">Schematic · not a measured EM rollout</p>${svgScene('seeds',n>=2?demo.r:null)}<p class="visual-key">A–E = physical cells · colored strips = soft ownership<br />Nuclear anchors remain exact one-hot</p>`;
      if(n===0) reading.innerHTML+=ownerBars(demo.spatial[demo.query],'Distance-only ownership');
      if(n===1) reading.innerHTML+=ownerBars(demo.low[demo.query],'Low expression confidence')+ownerBars(demo.high[demo.query],'High expression confidence');
      if(n===2) reading.innerHTML+=ownerBars(demo.r[demo.query],'After a complete, damped E-step')+`<p class="mini-claim">q fixed for all ${demo.bins.length} bins → complete R</p><p>Normalized entropy of b: <b>${M.entropy(demo.r[demo.query]).toFixed(3)}</b>. Entropy alone gates ambiguity. Later, top-1 initializes a separate discrete RL state.</p>`;
      if(n===3) reading.innerHTML+=qTable()+`<p class="print-summary">Key frames for b: distance A ${pct(demo.spatial[demo.query][0])} / B ${pct(demo.spatial[demo.query][1])}; strong expression A ${pct(demo.high[demo.query][0])} / B ${pct(demo.high[demo.query][1])}; full E-step, then M-step.</p>`;
    }
    if(kind==='replace') {
      const c=D.reward_cases[host._case],moved=n>=2,showGT=n>=4;
      visual.innerHTML=`<div class="case-select" role="group" aria-label="Historical action">${D.reward_cases.map((x,i)=>`<button type="button" data-case="${i}" aria-pressed="${i===host._case}">Case ${i+1}${showGT?' · '+x.outcome:''}</button>`).join('')}</div><p class="evidence-tag">Schematic transfer · actual historical values</p><div class="transfer-pair"><div class="transfer-cell owner-a"><b>A</b><span>cell ${c.old_cell_id}</span><i class="transfer-bin ${moved?'':'holds-bin'}">b</i><small>${moved?'− bin':'current owner'}</small></div><div class="transfer-arrow">${moved?'→':'?'}</div><div class="transfer-cell owner-b"><b>B</b><span>cell ${c.new_cell_id}</span><i class="transfer-bin ${moved?'holds-bin':''}">b</i><small>${moved?'+ bin':'candidate owner'}</small></div></div><p class="case-id">${esc(c.patch_id)} · action ${c.step}<br />${esc(c.barcode)}</p><div class="gt-reveal ${showGT?c.outcome:''}">${showGT?`GT result: <b>${c.outcome}</b><br />Matched source ${c.old_gt} → target ${c.new_gt}; true owner ${c.gt_owner}`:'GT hidden from policy and reward'}</div>`;
      if(n<3) reading.innerHTML+=`<div class="policy-facts"><p><b>Chosen action:</b> ${c.old_cell_id} → ${c.new_cell_id}</p><p><b>Immediate reward rank:</b> ${c.rank} / ${c.available} legal actions</p><p>${n===0?'State includes ownership, expression, type posterior, geometry, neighbors and shape.':n===1?'Design rule: nuclear bins locked; high-entropy non-nuclear bins eligible; STOP is available.':'State → policy → action → rebuild → reward → next state'}</p><p class="mini-claim">PPO learns from reward across repeated trajectories; GT is evaluation-only.</p></div>`;
      else {
        const labels=['w1 expression confidence','w5 expression compatibility','w2 distance','w3 competing nucleus','w4 neighbor support','shape prior','rounding residual','numeric residual'];
        const entries=Object.entries(c.components),max=Math.max(...entries.map(([,v])=>Math.abs(v)))*1e6;
        reading.innerHTML+=`<div class="delta-bars"><h4>Weighted global-delta contributions × 10⁻⁶</h4>${entries.map(([key,v],i)=>`<div class="delta-row"><span>${labels[i]}</span><div><i style="left:${v<0?50-Math.abs(v)*1e6/max*49:50}%;width:${Math.abs(v)*1e6/max*49}%;background:${v<0?'#b55d22':'#168d88'}"></i></div><b>${Math.abs(v*1e6)<.001?(v*1e6).toExponential(1):(v*1e6).toFixed(3)}</b></div>`).join('')}<p class="delta-total">Total reward: <b>${(c.reward*1e6).toFixed(3)} × 10⁻⁶</b></p></div>`;
      }
    }
    if(kind==='support') {
      const p=D.patches[host._patch],mode=n===2?'depth_uniform':'source_high_depth';
      visual.innerHTML=patchSelect(host)+`<div class="support-maps"><figure><b>Pseudo coverage</b><canvas data-map="coverage" role="img" aria-label="Union of loaded bins overlapping pseudo masks"></canvas></figure><figure><b>${n===2?'Depth-only pseudo':'High-depth pseudo'}</b><canvas data-map="${mode}" role="img" aria-label="Saved pseudo counts greater than zero"></canvas></figure><figure><b>Real HD</b><canvas data-map="real" role="img" aria-label="Saved real HD counts greater than zero"></canvas></figure></div><p class="visual-key">Black: counts &gt; 0 · magenta: pseudo coverage edge · gray: bin not loaded</p>`;
      visual.querySelectorAll('canvas').forEach(c=>drawMap(c,p,c.dataset.map,n>=1&&c.dataset.map!=='coverage'));
      reading.innerHTML+=`<div class="support-readout"><b>${p.n_bins} loaded bins</b><p>Displayed context only:</p><p>${n===2?'Depth-only':'High-depth'} BA <strong>${p.balanced_accuracy[mode].toFixed(4)}</strong></p><p>Real-HD BA <strong>${p.balanced_accuracy.real.toFixed(4)}</strong></p></div><p class="mini-claim">Support BA is not cell IoU.</p><p>Four-context summary: high-depth <b>0.9998</b>; depth-only <b>0.9990</b> (3-seed mean); real <b>0.5001</b>.</p>`;
    }
    if(kind==='shift') {
      const p=D.patches[host._patch];
      visual.innerHTML=patchSelect(host)+`<div class="shift-map"><canvas role="img" aria-label="Observed nuclear bins shifted over fixed pseudo coverage"></canvas></div><p class="visual-key">Dark: same dominant owner · orange ×: different owner<br />Pseudo coverage remains fixed; observations move right.</p>`;
      const canvas=visual.querySelector('canvas');drawMap(canvas,p,'coverage',false,n);const stats=JSON.parse(canvas.dataset.shiftSummary);
      reading.innerHTML+=`<div class="support-readout"><b>Displayed context · ${2*n} µm imposed shift</b><p>${stats.bad} / ${stats.valid} valid seeds disagree (${pct(stats.bad/stats.valid)})</p><p>${stats.lost} destinations not loaded</p></div><p class="mini-claim">All four contexts: ${['0.53%','10.34%','28.86%'][n]} disagree.</p><p>These are barcode-grid shifts, not a measured H&amp;E registration error. Original masks and shape prior are unchanged.</p>`;
    }
    if(kind==='replace'&&n===4)reading.innerHTML+=`<p class="print-summary">Both saved cases: ${D.reward_cases.map(c=>`${c.outcome}, reward +${(c.reward*1e6).toFixed(3)} × 10⁻⁶`).join('; ')}. Positive reward does not guarantee correct ownership.</p>`;
    host.querySelector('.story-progress').innerHTML=steps[kind].map((label,i)=>`<button type="button" data-step="${i}" aria-current="${i===n?'step':'false'}"><b>${i+1}</b><span>${label}</span></button>`).join('');
    host.querySelector('[data-command="back"]').disabled=n===0;
    host.querySelector('[data-command="next"]').textContent=n===steps[kind].length-1?'Next slide →':'Next →';
    host.querySelector('.story-status').textContent=`Step ${n+1} / ${steps[kind].length} · ${steps[kind][n]}${kind==='support'||kind==='shift'?' · '+D.patches[host._patch].patch_id:kind==='replace'?' · Case '+(host._case+1):''} · arrow keys or buttons`;
    if(Reveal.isReady()&&Reveal.getCurrentSlide()===host.closest('section')&&!location.search.includes('print-pdf')){
      const url=new URL(location.href);url.searchParams.set('slide',String(Reveal.getIndices().h));url.searchParams.set('fragment',String(n-1));
      if(kind==='support'||kind==='shift')url.searchParams.set('patch',D.patches[host._patch].patch_id);
      if(kind==='replace')url.searchParams.set('case',D.reward_cases[host._case].outcome);
      history.replaceState(null,'',url);
    }
  }

  function resultChart() {
    const host=document.querySelector('[data-result-chart]');if(!host) return;
    const methods=[['em','Frozen-profile EM'],['bin2cell','Bin2Cell*'],['stcs','STCS'],['distance_only','Distance only'],['nucleus_only','Nucleus only']];
    // Resolve the saved EM name explicitly; fail visibly if the source changes.
    const emName=D.comparison.find(r=>!['bin2cell','stcs','distance_only','nucleus_only'].includes(r.method))?.method;
    methods[0][0]=emName;
    let svg='<svg viewBox="0 0 1400 440" role="img" aria-label="Paired conditions on a common zero-to-one mean core-cell IoU scale">';
    const conditions=[['high_depth','High depth'],['capture_matched_ambient','Capture matched + tissue RNA']];
    conditions[0][0]=D.comparison.find(r=>r.condition!=='capture_matched_ambient').condition;
    for(let ci=0;ci<2;ci++){
      const start=340+ci*540;svg+=`<text x="${start}" y="30" font-size="25" font-weight="700">${conditions[ci][1]}</text>`;
      for(let t=0;t<=4;t++){const x=start+t*115;svg+=`<path d="M${x} 65V380" stroke="#dfe3e8"/><text x="${x}" y="415" text-anchor="middle" font-size="20">${t/4}</text>`;}
      methods.forEach(([key,label],i)=>{const r=D.comparison.find(r=>r.condition===conditions[ci][0]&&r.method===key);if(!r)throw new Error('Missing result '+key);const y=100+i*62;
        if(ci===0)svg+=`<text x="5" y="${y+8}" font-size="27" fill="${i===0?'#b62569':'#101b2d'}">${label}</text>`;
        const x=start+r.mean_pred_iou*460;svg+=`<circle cx="${x}" cy="${y}" r="9" fill="${i===0?'#b62569':'#168d88'}"/><text x="${x+15}" y="${y+7}" font-size="23">${r.mean_pred_iou.toFixed(3)}</text>`;
      });
    }
    host.innerHTML=svg+'</svg>';
  }
  function profileChart(){
    const host=document.querySelector('[data-profile-chart]');if(!host)return;
    const p=D.profile,series=[['Generator scRNA',p.raw,'#168d88'],['scDesign3',p.synthetic,'#b62569']];
    const max=Math.max(...p.raw,...p.synthetic),maxLog=Math.ceil(Math.log10(max+1));
    let s='<svg viewBox="0 0 1330 370" role="img" aria-label="Actual Cycling T total UMI empirical cumulative distributions on a common logarithmic x axis">';
    for(let t=0;t<=maxLog;t++){const value=t===0?0:10**t,x=110+Math.log10(value+1)/maxLog*800;s+=`<path d="M${x} 28V292" stroke="#e0e5e9"/><text x="${x}" y="322" text-anchor="middle" font-size="21">${value.toLocaleString()}</text>`;}
    for(let t=0;t<=4;t++){const y=292-t/4*264;s+=`<path d="M110 ${y}H910" stroke="#e0e5e9"/><text x="90" y="${y+7}" text-anchor="end" font-size="20">${t*25}%</text>`;}
    series.forEach(([label,vals,color],i)=>{let path='M110 292';vals.forEach((v,j)=>{const x=110+Math.log10(v+1)/maxLog*800,y=292-(j+1)/vals.length*264;path+=`H${x}V${y}`;});path+='H910';s+=`<path d="${path}" stroke="${color}" stroke-width="3" fill="none"/><text x="950" y="${95+i*70}" fill="${color}" font-size="27">${label}</text><text x="950" y="${123+i*70}" font-size="21">n = ${vals.length} cells</text>`;});
    s+='<text x="480" y="362" text-anchor="middle" font-size="23">Total UMI per cell · log10(1 + UMI) spacing</text><text x="12" y="20" font-size="19">Fraction of cells</text></svg>';
    host.innerHTML=s;
  }

  const hosts=[...document.querySelectorAll('[data-story]')];hosts.forEach(setup);
  function refresh(){
    const current=Reveal.getCurrentSlide();const host=current?.querySelector('[data-story]');if(host)render(host);
    const small=innerWidth<=600||(innerHeight<=500&&innerWidth<=1000);
    document.body.classList.toggle('story-reader',!!host&&small&&!location.search.includes('print-pdf'));
    if(Reveal.isReady()&&!location.search.includes('print-pdf')){
      const url=new URL(location.href);
      if(url.searchParams.has('slide')){url.searchParams.set('slide',String(Reveal.getIndices().h));url.searchParams.set('fragment',String(Reveal.getIndices().f??-1));history.replaceState(null,'',url);}
    }
  }
  Reveal.on('ready',()=>{Reveal.sync();hosts.forEach(render);resultChart();profileChart();refresh();});
  Reveal.on('slidechanged',refresh);Reveal.on('fragmentshown',refresh);Reveal.on('fragmenthidden',refresh);
  window.addEventListener('resize',refresh);
  window.ProgressStory={demo,drawMap,render,step};
})();

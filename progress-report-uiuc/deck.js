/* A fixed, source-backed research talk. All claims and speaker notes live in slides.json. */
const A = "assets/";
const BLUE = "#13294b", ORANGE = "#ff5f05", CYAN = "#00bfd2", PINK = "#dc3287", WHITE = "#ffffff";
const esc = s => String(s).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"})[c]);
const titleHtml = s => esc(s.title).replace(esc(s.key), `<span class="key">${esc(s.key)}</span>`);
const circle = (x,y,r,fill=BLUE) => `<circle cx="${x}" cy="${y}" r="${r}" fill="${fill}"/>`;
const line = (x1,y1,x2,y2,color="#859bb4",w=3) => `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="${w}"/>`;
const text = (x,y,t,size=30,weight=600,fill=BLUE) => `<text x="${x}" y="${y}" style="fill:${fill}" font-size="${size}" font-weight="${weight}">${esc(t)}</text>`;
const svg = (parts, label="Diagram") => `<svg class="svg-scene" viewBox="0 0 1430 500" role="img" aria-label="${esc(label)}">${parts.join("")}</svg>`;
const flow = (labels, icons, subtitles=[]) => `<div class="flow">${labels.map((v,i)=>`${i?'<div class="flow-arrow">→</div>':''}<div class="flow-item ${i===labels.length-1?'accent':''}"><div class="flow-icon" style="font-size:${String(icons[i]||i+1).length>2?34:62}px">${esc(icons[i]||String(i+1))}</div><span class="flow-title">${esc(v)}</span>${subtitles[i]?`<span class="flow-sub">${esc(subtitles[i])}</span>`:''}</div>`).join("")}</div>`;
const image = (file, caption, alt) => `<div class="photo-frame"><img src="${A+file}" alt="${esc(alt||caption)}"><span class="photo-caption">${esc(caption)}</span></div>`;
const cellStage = () => `<div class="cell-stage"><div class="grid"></div><div class="cell a"></div><div class="cell b"></div><div class="nucleus a"></div><div class="nucleus b"></div><div class="selected-bin"></div><span class="label a">cell A</span><span class="label b">cell B</span><span class="label bin">one shared bin</span></div>`;
const side = (heading,paras) => `<div class="side-copy"><h3>${esc(heading)}</h3>${paras.map(p=>`<p>${esc(p)}</p>`).join("")}</div>`;

function candidateDiagram(){
  const pts=[[195,130,"A"],[452,139,"B"],[226,350,"C"],[456,348,"D"]], c=[330,244];
  const a=[`<defs><pattern id="candgrid" width="50" height="50" patternUnits="userSpaceOnUse"><path d="M50 0H0V50" fill="none" stroke="#d6e0e9" stroke-width="1"/></pattern></defs>`,`<rect x="30" y="12" width="630" height="458" fill="url(#candgrid)"/>`,`<rect x="309" y="223" width="42" height="42" fill="${ORANGE}"/>`,text(365,257,"one 2 µm bin",26,700,ORANGE)];
  a.push(`<g class="fragment fade-in" data-fragment-index="0">`,`<circle cx="330" cy="244" r="195" fill="none" stroke="#8fa2b7" stroke-width="3" stroke-dasharray="10 10"/>`,text(430,75,"20 µm MaxDis",23,500));
  pts.forEach(([x,y,n])=>a.push(line(x,y,...c),circle(x,y,19),text(x+23,y-19,n,30,700)));
  a.push(circle(685,385,19,"#b3c0ce"),text(710,394,"E  outside",27,600,"#637990"),`</g>`);
  a.push(`<g class="fragment fade-in" data-fragment-index="1">`,text(865,55,"Illustrative soft ownership",30,700));
  [["A",.52],["B",.40],["C",.05],["D",.03]].forEach(([n,v],i)=>{let y=125+i*80;a.push(text(865,y+7,n,28,700),`<rect x="924" y="${y-20}" width="333" height="29" fill="#e6edf2"/>`,`<rect x="924" y="${y-20}" width="${333*v}" height="29" fill="${i===0?ORANGE:BLUE}"/>`,text(1280,y+6,v.toFixed(2),25,500));});
  a.push(line(35,455,1390,455,"#cad6e2",2),text(40,492,"Every valid candidate competes",25,700),text(750,492,"Nuclear seeds stay locked",25,700,ORANGE),`</g>`);
  return svg(a,"One bin and four eligible nuclei inside 20 micrometres; a fifth nucleus is outside");
}

function entropyExample(title, values, entropy, decision, uncertain=false){
  return `<div class="entropy-example ${uncertain?'uncertain':''}"><h3>${title}</h3><div class="entropy-bars">${values.map((v,i)=>`<div class="entropy-bar"><b>${"ABCD"[i]}</b><span><i style="width:${v*100}%"></i></span><em>${v.toFixed(2)}</em></div>`).join("")}</div><strong>Normalized entropy ${entropy}</strong><p>${decision}</p></div>`;
}

function replaceDiagram(){
  const a=[];
  [[35,CYAN,"BEFORE · EM owner A"],[770,PINK,"AFTER · RL owner B"]].forEach(([x,owner,title])=>{
    if(x===770)a.push(`<g class="fragment fade-in" data-fragment-index="0">`);
    a.push(text(x+8,54,title,30,700));
    a.push(`<ellipse cx="${x+220}" cy="257" rx="155" ry="161" fill="#ecf9fa" stroke="${CYAN}" stroke-width="4"/>`);
    a.push(`<ellipse cx="${x+423}" cy="264" rx="157" ry="151" fill="#fdf1f7" stroke="${PINK}" stroke-width="4"/>`);
    a.push(circle(x+210,255,23),circle(x+441,264,23));
    a.push(`<rect x="${x+310}" y="232" width="48" height="48" fill="${owner}" stroke="${owner}" stroke-width="3"/>`);
    a.push(text(x+92,439,"nucleus A",24,600),text(x+412,439,"nucleus B",24,600));
    if(x===770)a.push(`</g>`);
  });
  a.push(`<g class="fragment fade-in" data-fragment-index="0">`,`<path d="M704 245h45m-16-16 16 16-16 16" fill="none" stroke="${ORANGE}" stroke-width="6"/>`,text(625,210,"REPLACE",24,700,ORANGE),`</g>`);
  a.push(text(412,485,"The barcode stays in place; only its owner changes. Nuclear seeds stay fixed.",23,600));
  return svg(a,"One fixed bin changes discrete owner from cell A to cell B while both nuclei remain fixed");
}

function downstreamDiagram(){
  const a=[text(26,42,"MEASURED BARCODES",27,700),text(573,42,"RECONSTRUCTED CELLS",27,700),text(1050,42,"BIOLOGICAL QUESTIONS",27,700)];
  for(let row=0;row<5;row++)for(let col=0;col<7;col++){
    const x=34+col*55,y=92+row*55,fill=col<3?"#c8f0f3":col>3?"#f7d2e3":"#f4e7c2";
    a.push(`<rect x="${x}" y="${y}" width="53" height="53" fill="${fill}" stroke="#fff" stroke-width="2"/>`);
    if((row+col)%3===0)a.push(circle(x+27,y+26,5,BLUE));
  }
  a.push(text(35,431,"Squares split cells and mix edges",23,600,"#52657b"));
  a.push(`<path d="M442 226h76m-18-18 18 18-18 18" fill="none" stroke="${ORANGE}" stroke-width="6"/>`);
  a.push(`<path d="M608 129c-53 19-61 180 3 226 62 42 167 2 157-88-8-68-52-156-160-138z" fill="#def6f7" stroke="${CYAN}" stroke-width="4"/>`);
  a.push(`<path d="M820 137c-48 21-46 158 7 208 49 47 145 10 144-79-1-73-50-168-151-129z" fill="#fae2ed" stroke="${PINK}" stroke-width="4"/>`);
  a.push(circle(677,239,20),circle(875,239,20),text(643,402,"cell A",24,700),text(844,402,"cell B",24,700));
  [0,1,2,3].forEach((i)=>{a.push(`<rect x="${607+i*43}" y="${348-i*15}" width="26" height="${i*15+28}" fill="${CYAN}"/>`);a.push(`<rect x="${819+i*38}" y="${354-(3-i)*14}" width="24" height="${(3-i)*14+22}" fill="${PINK}"/>`)});
  a.push(text(618,464,"one RNA profile per cell",23,600,"#52657b"));
  a.push(`<path d="M979 226h50m-18-18 18 18-18 18" fill="none" stroke="${ORANGE}" stroke-width="6"/>`);
  a.push(`<rect x="1050" y="115" width="336" height="114" fill="#eef3f7"/>`,text(1073,162,"Cell annotation",28,700),text(1073,199,"What kind of cell?",23,400,"#52657b"));
  a.push(`<rect x="1050" y="250" width="336" height="174" fill="#eef3f7"/>`,text(1073,301,"Cell–cell",28,700),text(1073,340,"communication",28,700),text(1073,395,"Who may interact?",22,400,"#52657b"));
  return svg(a,"Spatial bin counts are grouped into one profile for each physical cell, enabling annotation and cell-cell communication");
}

function challengeDiagram(){
  const a=[text(24,49,"WHAT WE CAN SEE",27,700),text(800,49,"WHAT WE CANNOT SEE",27,700)];
  a.push(`<path d="M157 97c-87 28-120 223-17 316 90 81 240 9 212-110-22-104-56-241-195-206z" fill="#e5f8fa" stroke="${CYAN}" stroke-width="4"/>`);
  a.push(`<path d="M414 104c-94 11-130 186-84 284 55 116 238 58 252-59 17-132-46-240-168-225z" fill="#fae3ed" stroke="${PINK}" stroke-width="4"/>`);
  a.push(circle(208,252,22),circle(468,270,22),`<rect x="319" y="236" width="51" height="51" fill="${ORANGE}"/>`,text(195,477,"nucleus A",22,600),text(431,477,"nucleus B",22,600));
  a.push(`<path d="M604 260h88m-18-18 18 18-18 18" fill="none" stroke="${ORANGE}" stroke-width="6"/>`);
  a.push(`<rect x="756" y="106" width="590" height="296" fill="#fff4ee"/>`,text(797,164,"The orange barcode",30,700),text(797,213,"has measured RNA",30,700),text(797,295,"but no measured",30,700),text(797,344,"source-cell label",30,700,ORANGE));
  a.push(text(786,461,"That missing label is the ownership problem.",25,600,"#52657b"));
  return svg(a,"Two nuclei and one boundary barcode are visible; the cell of origin of the barcode's RNA is not observed");
}

function priorToolsDiagram(){
  const rows=[
    ["Bin2Cell","nuclei + image / GEX","grow cell regions",["growth boundary","is inferred"]],
    ["STCS","nuclei + distance + RNA","weigh candidate cells",["nuclear RNA may not","represent whole cell"]],
    ["Distance-only","nuclei + spatial distance","choose nearby owners",["ignores expression","evidence"]]
  ];
  const a=[text(30,42,"EXISTING APPROACH",24,700),text(384,42,"EVIDENCE USED",24,700),text(804,42,"WHAT IT DOES",24,700),text(1117,42,"LIMITATION",24,700)];
  rows.forEach(([name,evidence,act,lim],i)=>{let y=83+i*134;a.push(`<rect x="20" y="${y}" width="1390" height="114" fill="${i%2?'#f7f9fb':'#eef3f7'}"/>`,text(40,y+48,name,27,700),text(402,y+48,evidence,24,600),text(814,y+48,act,23,600),text(1117,y+36,lim[0],20,600,ORANGE),text(1117,y+68,lim[1],20,600,ORANGE));});
  a.push(text(40,486,"All are useful controls; none directly measures which physical cell supplied boundary RNA.",22,600,"#52657b"));
  return svg(a,"Bin2Cell grows nuclear labels using image and gene expression, STCS weighs candidate cells with distance and nucleus-derived RNA, and distance-only uses proximity; each still infers boundary ownership");
}

function approachDiagram(){
  const a=[text(31,44,"NUCLEAR ANCHORS",25,700),text(435,44,"EM SOFT OWNERSHIP",25,700),text(842,44,"ENTROPY-GATED RL",25,700),text(1190,44,"FINAL CELLS",25,700)];
  a.push(`<rect x="25" y="80" width="338" height="319" fill="#eef3f7"/>`);
  a.push(`<ellipse cx="132" cy="233" rx="67" ry="101" fill="#dff6f7" stroke="${CYAN}" stroke-width="4"/>`,`<ellipse cx="256" cy="238" rx="67" ry="101" fill="#fae3ed" stroke="${PINK}" stroke-width="4"/>`,circle(132,233,20),circle(256,238,20),`<rect x="182" y="216" width="34" height="34" fill="${ORANGE}"/>`,text(74,446,"Seeds stay locked",23,700));
  a.push(`<path d="M378 239h40m-15-15 15 15-15 15" fill="none" stroke="${ORANGE}" stroke-width="5"/>`);
  a.push(`<rect x="434" y="80" width="354" height="319" fill="#fff4ee"/>`,text(469,130,"one contested bin",23,700),text(469,192,"cell A     0.52",26,700),text(469,255,"cell B     0.40",26,700),text(469,319,"other      0.08",26,700),text(470,445,"Keep uncertainty",23,700));
  a.push(`<path d="M800 239h34m-13-13 13 13-13 13" fill="none" stroke="${ORANGE}" stroke-width="5"/>`);
  a.push(`<rect x="842" y="80" width="308" height="319" fill="#eef3f7"/>`,text(874,147,"high entropy",25,700),text(874,218,"REPLACE",28,700,ORANGE),text(874,274,"A → B?",28,700),text(874,446,"Test global score",23,700));
  a.push(`<path d="M1161 239h22m-9-9 9 9-9 9" fill="none" stroke="${ORANGE}" stroke-width="5"/>`);
  a.push(`<rect x="1187" y="80" width="218" height="319" fill="#eef3f7"/>`,text(1210,164,"barcode b",22,700),text(1210,229,"one owner",24,700),text(1210,292,"one cell ID",22,600),text(1210,445,"Discrete output",23,700));
  return svg(a,"Locked nuclear seeds anchor physical cells; EM gives each boundary bin soft owner probabilities; normalized entropy selects uncertain bins for RL REPLACE; final barcodes have one owner");
}

function typingDiagram(){
  const a=[text(30,46,"ONE SELECTED 8 µm PARENT BIN",26,700)];
  for(let row=0;row<4;row++)for(let col=0;col<4;col++)a.push(`<rect x="${55+col*95}" y="${80+row*95}" width="93" height="93" fill="${row===1&&col===1?'#ffeadb':'#f1f5f8'}" stroke="${row===1&&col===1?ORANGE:'#bdcbd8'}" stroke-width="${row===1&&col===1?4:2}"/>`);
  a.push(circle(194,220,23,BLUE),text(66,492,"selected nucleus inside parent bin",22,600,"#52657b"));
  a.push(`<path d="M490 260h90m-18-18 18 18-18 18" fill="none" stroke="${ORANGE}" stroke-width="6"/>`);
  a.push(text(635,46,"OBSERVED TYPE POSTERIOR",26,700));
  [["TA",.695],["Goblet",.285],["Immature Goblet",.020]].forEach(([name,p],i)=>{let y=103+i*105;a.push(text(635,y+31,name,25,700),`<rect x="875" y="${y+4}" width="335" height="36" fill="#e7edf2"/>`,`<rect x="875" y="${y+4}" width="${335*p}" height="36" fill="${i?BLUE:ORANGE}"/>`,text(1230,y+33,p.toFixed(3),24,600));});
  a.push(`<rect x="650" y="422" width="645" height="55" fill="#fff1e8"/>`,text(671,460,"Highest mean posterior → nucleus annotated TA",25,700,ORANGE));
  return svg(a,"An 8 micrometre parent bin containing a nucleus has type posterior TA 0.695, Goblet 0.285, Immature Goblet 0.020; TA is selected");
}

function aggregateDiagram(){
  const a=[text(18,44,"PLACE MOLECULES",27,700),text(489,44,"COUNT BY 2 µm BARCODE",27,700),text(1068,44,"KEEP SOURCES DISTINCT",25,700)];
  a.push(`<path d="M121 91c-80 24-107 233-15 330 83 90 304 27 323-94 15-105-143-282-308-236z" fill="#e8f7f8" stroke="${CYAN}" stroke-width="4"/>`,circle(250,243,29,BLUE));
  [[160,159,ORANGE],[191,203,ORANGE],[300,176,ORANGE],[330,269,ORANGE],[152,306,PINK],[323,343,PINK],[110,241,BLUE],[383,219,BLUE]].forEach(([x,y,c])=>a.push(circle(x,y,8,c)));
  a.push(`<path d="M445 257h47m-17-17 17 17-17 17" fill="none" stroke="${ORANGE}" stroke-width="5"/>`);
  for(let row=0;row<4;row++)for(let col=0;col<5;col++)a.push(`<rect x="${540+col*82}" y="${99+row*82}" width="80" height="80" fill="${row===1&&col===2?'#fff0e5':'#f0f5f8'}" stroke="${row===1&&col===2?ORANGE:'#bacbd8'}" stroke-width="${row===1&&col===2?4:1}"/>`);
  [[565,130],[702,192],[743,221],[764,238],[821,343],[630,366]].forEach(([x,y])=>a.push(circle(x,y,7,BLUE)));
  a.push(`<path d="M979 257h56m-18-18 18 18-18 18" fill="none" stroke="${ORANGE}" stroke-width="5"/>`);
  a.push(`<rect x="1060" y="91" width="333" height="194" fill="#eef3f7"/>`,text(1080,135,"Barcode b",25,700),text(1080,181,"Gene A · 3",25,500),text(1080,225,"Gene B · 2",25,500),text(1080,265,"…",25,500));
  a.push(text(1060,335,"Modeled cell RNA",22,700,CYAN),text(1060,374,"Unmodeled biology",22,700,PINK),text(1060,413,"Technical background",22,700,BLUE));
  return svg(a,"Molecules placed in a simulated cell are counted inside physical 2 micrometre barcodes; modeled RNA, unmodeled biology, and technical background remain distinct");
}

function emDiagram(){
  const a=[text(17,45,"CURRENT CELL BELIEFS",25,700)];
  a.push(`<rect x="17" y="91" width="370" height="317" fill="#eef3f7"/>`,text(43,137,"Cell A  TA-like",25,700),text(43,190,"Cell B  Goblet-like",25,700),text(43,261,"Nuclear bins stay",23,600),text(43,298,"with their cells",23,600),`<rect x="43" y="339" width="220" height="39" fill="${BLUE}"/>`,text(64,367,"LOCKED  1.00",20,700,WHITE));
  a.push(`<g class="fragment fade-in" data-fragment-index="0">`,text(470,45,"E-STEP  ·  ALL BINS",25,700),`<path d="M391 242h55m-17-17 17 17-17 17" fill="none" stroke="${ORANGE}" stroke-width="5"/>`);
  a.push(`<rect x="470" y="91" width="490" height="317" fill="#fff4ee"/>`,text(493,137,"Boundary bin b",24,700),text(493,178,"A  0.52",22,700),`<rect x="644" y="157" width="250" height="24" fill="#f5dfd3"/>`,`<rect x="644" y="157" width="130" height="24" fill="${ORANGE}"/>`,text(493,228,"B  0.40",22,700),`<rect x="644" y="207" width="250" height="24" fill="#f5dfd3"/>`,`<rect x="644" y="207" width="100" height="24" fill="${BLUE}"/>`,text(493,278,"other  0.08",22,700),text(493,345,"Every bin uses the same",22,600),text(493,378,"cell beliefs in this step",22,600));
  a.push(`</g>`,`<g class="fragment fade-in" data-fragment-index="1">`,text(1020,45,"M-STEP  ·  ALL CELLS",25,700),`<path d="M969 242h42m-15-15 15 15-15 15" fill="none" stroke="${ORANGE}" stroke-width="5"/>`);
  a.push(`<rect x="1033" y="91" width="380" height="317" fill="#eef3f7"/>`,text(1057,139,"Cell A uses 0.52",23,700),text(1057,178,"of bin b's evidence",23,600),text(1057,243,"Cell B uses 0.40",23,700),text(1057,282,"of bin b's evidence",23,600),text(1057,354,"Then update type beliefs",22,700,ORANGE));
  a.push(`</g>`,`<g class="fragment fade-in" data-fragment-index="2">`,`<path d="M1230 435H183m0 0v-17m0 17 17-11" fill="none" stroke="${BLUE}" stroke-width="3"/>`,text(508,472,"Repeat until assignments stabilize",23,600,"#52657b"),`</g>`);
  return svg(a,"Generalized EM uses one cell state for every bin, assigns soft owners in a batch E-step, then updates all cells from those responsibilities in an M-step; nuclear seeds remain locked");
}

function inputOutputDiagram(){
  return `<div class="io-layout"><div class="io-inputs"><h3>INPUTS</h3><div class="io-photo"><img src="${A}nuclear_segmentation_cell_roi.png" alt="H&E with detected nuclear contours"><b>H&E + physical nuclei</b></div><div class="io-counts"><div class="mini-count-grid">${[0,1,2,1,3,0,2,4,1,0,1,3].map(n=>`<i data-n="${n}">${n}</i>`).join("")}</div><b>2 µm gene counts</b></div><div class="io-ref"><span>TA</span><i style="width:75%"></i><span>Goblet</span><i style="width:43%"></i><b>annotated scRNA reference</b></div></div><div class="io-process"><span>→</span><strong>HD-Cell-RL</strong><small>inference never sees<br>the GT cell mask</small><span>→</span></div><div class="io-outputs"><h3>OUTPUTS</h3><div class="io-soft"><b>During EM</b><span>bin b → A 0.52</span><span>bin b → B 0.40</span><span>other cells 0.08</span><small>soft ownership shows uncertainty</small></div><div class="io-hard"><b>Final assignment</b><div class="io-mini-cells"><i></i><i></i><em>b → cell A</em></div><small>one barcode, one cell owner</small></div></div></div>`;
}

function visiumDiagram(){
  const grid=Array.from({length:16},(_,i)=>`<i class="${i===5?'picked':''}"></i>`).join("");
  return `<div class="visium-layout"><div class="visium-tissue"><img src="${A}visium-microscopic-grid.png" alt="Project H&E tissue field with aligned barcode-grid overlay"><span class="visium-bin fragment fade-in" data-fragment-index="0"></span><b>H&E tissue + spatial barcodes</b></div><span class="visium-connector fragment fade-in" data-fragment-index="1" aria-hidden="true">→</span><div class="visium-zoom fragment fade-in" data-fragment-index="1"><h3>Zoom into the measurement</h3><div class="visium-zoom-grid">${grid}</div><strong>one 2 × 2 µm barcode</strong><p>known position in the tissue</p><div class="visium-gene-bars fragment fade-in" data-fragment-index="2"><span>Gene A</span><i style="width:72%"></i><span>Gene B</span><i style="width:36%"></i><span>Gene C</span><i style="width:16%"></i></div><small class="fragment fade-in" data-fragment-index="2">one gene-count vector per barcode · bars illustrative</small></div></div>`;
}

function omicsContextDiagram(){
  return `<div class="omics-published"><div class="omics-source-crop"><img src="assets/jeon-2023-figure-1.png" alt="Published comparison of bulk RNA-seq, single-cell RNA-seq, and spatial transcriptomics, showing the measurement level of each"></div><div class="omics-takeaways"><div><strong>Bulk RNA-seq</strong><span>Mixed tissue profile; individual cells are lost.</span></div><div><strong>Single-cell RNA-seq</strong><span>Cell profiles; original locations are lost.</span></div><div><strong>Spatial transcriptomics</strong><span>Expression has coordinates; HD bins still need cell owners.</span></div></div></div>`;
}

function answerKeyDiagram(){
  const a=[text(35,43,"KNOWN DURING GENERATION",25,700),text(600,43,"GIVEN TO THE METHOD",25,700),text(1090,43,"REVEALED FOR SCORING",25,700)];
  a.push(`<path d="M85 124c-60 19-73 198 1 259 66 57 166-8 153-91-10-68-67-190-154-168z" fill="#def6f7" stroke="${CYAN}" stroke-width="4"/>`,circle(160,237,17));
  a.push(`<path d="M286 149c-65 13-79 171-18 236 57 62 162 8 160-85-2-80-54-169-142-151z" fill="#fae2ed" stroke="${PINK}" stroke-width="4"/>`,circle(349,244,17));
  [[215,220,CYAN],[247,235,CYAN],[263,219,PINK],[288,245,PINK]].forEach(([x,y,c])=>a.push(circle(x,y,8,c)));
  a.push(`<rect x="239" y="199" width="72" height="72" fill="none" stroke="${ORANGE}" stroke-width="5"/>`,text(91,432,"source: A or B",23,700,ORANGE));
  a.push(`<path d="M451 246h95m-18-18 18 18-18 18" fill="none" stroke="${ORANGE}" stroke-width="5"/>`);
  a.push(`<rect x="596" y="128" width="365" height="262" fill="#eef3f7"/>`,text(627,184,"barcode b",30,700),text(627,241,"Gene 1 · 3 counts",25,500),text(627,290,"Gene 2 · 2 counts",25,500),text(627,353,"source labels hidden",23,700,ORANGE));
  a.push(`<path d="M974 246h72m-18-18 18 18-18 18" fill="none" stroke="${ORANGE}" stroke-width="5"/>`);
  a.push(`<rect x="1090" y="128" width="288" height="262" fill="#fff4ee"/>`,text(1115,185,"answer key",29,700),text(1115,245,"RNA from A",24,700,CYAN),text(1115,297,"RNA from B",24,700,PINK),text(1115,352,"evaluation only",22,700,ORANGE));
  return svg(a,"The simulator records which cell supplied the RNA, hides those labels when giving the barcode counts to a method, then reveals them only for scoring");
}

function scene(s){
  switch(s.kind){
    case "omics-context": return omicsContextDiagram();
    case "visium": return visiumDiagram();
    case "bin-cell": return `<div class="split wide-left">${cellStage()}${side("The measurement is a square",["A cell crosses many squares.","A square near a boundary can intersect two cells.","We therefore need to infer ownership before forming cell profiles."])}</div>`;
    case "downstream": return downstreamDiagram();
    case "scales": return `<div class="scale-crops">${[["2 µm","Fine grid; one cell spans many bins",4.48],["8 µm","More RNA per square; edge still mixed",17.92],["16 µm","Even larger squares cross cell outlines",35.84]].map(([a,b,step])=>`<figure class="scale-crop"><div class="scale-image"><img src="${A}expanded_cell_mask_cell_roi.png" alt="The same H&E crop with simulated cyan cell boundaries"><div class="scale-overlay" style="background-size:${step}px ${step}px"></div></div><figcaption><b>${a}</b><span>${b}</span></figcaption></figure>`).join("")}</div>`;
    case "challenge": return challengeDiagram();
    case "prior-tools": return priorToolsDiagram();
    case "question": return `<div class="question-scene"><div class="question-photo"><img src="${A}nuclear_segmentation_cell_roi.png" alt="Two-dimensional H&E field with nuclear contours"><span class="question-target"></span><b>Observed nuclei + 2 µm barcode</b></div><div class="question-choice"><span class="question-mark">?</span><h3>Who produced the RNA<br>in this barcode?</h3><div class="choice-row"><span>nearby cell A</span><span>nearby cell B</span></div><p>The nucleus is an anchor, not a measured membrane.</p></div></div>`;
    case "input-output": return inputOutputDiagram();
    case "approach": return approachDiagram();
    case "pipeline": return `<div class="overview-scene"><div class="overview-image"><img src="${A}simulation-overview.jpg" alt="Illustrative tissue sequence from nuclear outlines to type labels, grown regions, and RNA compartments"></div><div class="overview-captions">${["Start with real nuclei","Assign cell types","Grow plausible regions","Define RNA compartments"].map(t=>`<span>${esc(t)}</span>`).join("")}</div><div class="overview-truth">The generator saves RNA source labels as an <strong>answer key</strong>; inference receives only bin counts.</div></div>`;
    case "answer-key": return answerKeyDiagram();
    case "nuclei-photo": return `<div class="split wide-left">${image("nuclear_segmentation_cell_roi.png","Detected nuclei in a real H&E crop","Project H&E close-up with saved magenta nuclear contours")}${side("A physical anchor",["Find nuclear instances in the tissue image.","Keep confident nuclear bins tied to their cell.","Use these anchors for the simulation and assignment stages."])}</div>`;
    case "typing": return typingDiagram();
    case "cell-size": return `<div class="split"><div style="display:flex;align-items:center;justify-content:space-evenly;height:100%"><div style="text-align:center"><div style="width:170px;height:170px;border:5px solid ${CYAN};border-radius:50%;display:grid;place-items:center;margin:auto"><span style="width:75px;height:70px;border-radius:50%;background:${PINK}"></span></div><b style="display:block;font-size:27px;margin-top:20px">lymphocyte</b></div><div style="text-align:center"><div style="width:340px;height:290px;border:5px solid ${CYAN};border-radius:50% 42% 54% 46%;display:grid;place-items:center;margin:auto"><span style="width:75px;height:70px;border-radius:50%;background:${PINK}"></span></div><b style="display:block;font-size:27px;margin-top:20px">fibroblast</b></div></div>${side("One nucleus, different whole-cell areas",["Cell type gives a target nucleus-to-cell area ratio.","Nearby nuclei and a maximum growth distance constrain the final shape.","These are generator priors, not measured membranes."])}</div>`;
    case "mask-photo": return `<div class="mask-reveal"><div class="mask-image"><img src="${A}nuclear_segmentation_cell_roi.png" alt="H&E crop with magenta nuclear contours"><img class="fragment fade-in" data-fragment-index="0" src="${A}expanded_cell_mask_cell_roi.png" alt="Same H&E crop with generated cyan cell boundaries"><span>Same H&E region</span></div><div class="mask-explainer"><div class="mask-step"><b>1</b><p>Real H&E gives us the <strong>nuclear anchors</strong>.</p></div><div class="mask-step fragment fade-in" data-fragment-index="0"><b>2</b><p>The simulator grows <strong>plausible cell regions</strong> around them.</p></div><small>Cyan boundaries are generated, not observed membranes.</small></div></div>`;
    case "fractional": return `<div class="split wide-left">${cellStage()}${side("One barcode, two sources",["The highlighted 2 µm square crosses a boundary.","Illustrative truth: 65% of its RNA from A, 35% from B.","The final method will still return one discrete owner."])}</div>`;
    case "sc-profiles": return svg([
      `<rect x="70" y="185" width="320" height="122" fill="#eef3f7" stroke="${BLUE}" stroke-width="3"/>`,text(103,231,"annotated CRC",30,700),text(103,271,"scRNA cells",30,700),
      line(390,245,650,245,BLUE,3),line(650,105,650,385,BLUE,3),line(650,105,820,105,BLUE,3),line(650,385,820,385,BLUE,3),
      `<rect x="820" y="45" width="500" height="130" fill="#eef3f7" stroke="${BLUE}" stroke-width="3"/>`,text(858,99,"generator donors",31,700),text(858,141,"source cell profiles",23,400),
      `<rect x="820" y="325" width="500" height="130" fill="#fff4ee" stroke="${ORANGE}" stroke-width="3"/>`,text(858,378,"reference donors",31,700),text(858,420,"EM type prior",23,400),
      text(435,209,"split by donor",23,600,ORANGE),text(840,251,"no shared donor cells",25,700,ORANGE)
    ],"Annotated CRC single-cell data branch into independent generator and reference donor groups");
    case "compartments": return `<div class="compartment-row">${[["nuclear","Nuclear","toward the center"],["cytoplasm","Cytoplasmic","through the interior"],["membrane","Membrane-associated","toward the edge"]].map(([cls,t,sub])=>`<div class="compartment"><div class="shape ${cls}"><i class="dot"></i></div><b>${t}</b><span>${sub}</span></div>`).join("")}</div>`;
    case "aggregate": return aggregateDiagram();
    case "tracks": return `<div class="track-comparison"><article><div class="track-visual track-pseudo"><img src="${A}expanded_cell_mask_cell_roi.png" alt="H&E with generated pseudo-cell boundaries"></div><h3>Colorectal pseudo HD</h3><p><b>Known source cells</b> let us score exact ownership.<br>Risk: generator assumptions.</p></article><article><div class="track-visual track-xenium"><i class="xcell a"></i><i class="xcell b"></i>${[[20,28],[31,41],[47,22],[54,56],[62,39],[78,61],[38,70]].map(([x,y])=>`<i class="xdot" style="left:${x}%;top:${y}%"></i>`).join("")}</div><h3>Xenium-derived 2 µm bins</h3><p><b>Observed transcript positions</b> are rebinned.<br>Risk: this is not HD capture.</p></article><article><div class="track-visual track-paired"><span>measured HD<br><i class="hd-mini-grid"></i></span><em>↔</em><span>matched Xenium<br><i class="xe-mini-cell"></i></span></div><h3>Paired HD + Xenium</h3><p><b>Measured HD expression</b> tests transfer.<br>Risk: alignment and assay gap.</p></article></div>`;
    case "candidate": return candidateDiagram();
    case "em-cycle": return emDiagram();
    case "entropy": return `<div class="entropy-comparison">${entropyExample("One clear owner",[.95,.03,.01,.01],"0.18","Keep EM owner A")}<div class="fragment fade-in" data-fragment-index="0">${entropyExample("Two plausible owners",[.52,.40,.05,.03],"0.69","RL may test A → B",true)}</div><p class="entropy-footnote">Same four candidate cells in both examples · one-candidate bins have zero entropy · only entropy gates RL eligibility</p></div>`;
    case "replace": return replaceDiagram();
    case "reward": return `<div class="split"><div class="badge-row" style="display:grid;grid-template-columns:1fr 1fr;gap:24px"><div><strong>Expression</strong><p>Does the new owner better fit cell evidence?</p></div><div><strong>Geometry</strong><p>Is the bin plausible for its nucleus?</p></div><div><strong>Local context</strong><p>Do neighbors and overlap improve?</p></div><div><strong>Shape</strong><p>Does the cell remain plausible?</p></div></div>${side("Global delta",["Score the whole patch before and after one REPLACE.","Reward = objective after − objective before.","EM probability is not added as a reward term."])}</div>`;
    case "training": return `<div class="patch-grid">${[["P1",66],["P2",29],["P3",45],["P4",50]].map(([n,c])=>`<div class="patch"><div class="core"></div><span class="patch-count">${c} context cells</span><b>${n}</b></div>`).join("")}<div class="scene-caption" style="bottom:-2px">Outer patch 64 × 64 µm · scored core 48 × 48 µm · loaded context 104 × 104 µm</div></div>`;
    case "methods": return `<div class="method-list">${[["Nucleus-only","Keep the shared nuclear seeds."],["Bin2Cell","Expand supplied nuclear labels."],["STCS","Use distance and nucleus-derived expression."],["Distance-only EM","Same EM candidate graph, no expression term."],["Current EM","Add reference-aware expression compatibility."]].map(([n,d])=>`<div><b>${n}</b><span>${d}</span></div>`).join("")}</div>`;
    case "result": return `<div class="result-chart">${s.labels.map((n,i)=>`<div class="result-row"><span class="dataset">${esc(n)}</span><div><span class="metric">distance-only EM</span><div class="bar-track"><div class="bar" style="width:${s.values[i][0]*100}%"></div></div></div><span class="score">${s.values[i][0].toFixed(3)}</span><div><span class="metric">expression-aware EM</span><div class="bar-track"><div class="bar em" style="width:${s.values[i][1]*100}%"></div></div></div><span class="score">${s.values[i][1].toFixed(3)}</span></div>`).join("")}<div class="result-legend"><span><i></i>distance-only</span><span><i class="em"></i>current EM</span><span>Four-patch mean cell IoU; fixed settings; diagnostic only</span></div></div>`;
    case "limits": return `<div class="badge-row">${[["01","Simulated CRC","Known owners, but model assumptions shape the task."],["02","Xenium-derived","Observed transcripts, but different capture and segmentation."],["03","Paired HD / Xenium","Measured HD, but alignment and assay differences remain."]].map(([n,t,p])=>`<div><span class="large">${n}</span><strong>${t}</strong><p>${p}</p></div>`).join("")}</div>`;
    case "close": return `<div class="end-block"><div class="end-step"><strong>Match reality</strong><p>Depth, cell variation, unmodeled RNA, and background.</p></div><div class="end-arrow">→</div><div class="end-step"><strong>Freeze and test</strong><p>Compare methods on fresh regions and sections.</p></div><div class="end-arrow">→</div><div class="end-step"><strong>Then ask about RL</strong><p>Require a reliable net gain before training PPO again.</p></div></div>`;
    default: return `<p>${esc(s.lead||"")}</p>`;
  }
}

function renderSlide(s,i){
  if(s.kind==="cover") return `<section class="slide cover" aria-label="${esc(s.title)}"><h1>From <span class="accent">2 µm</span> bins<br>to physical cells</h1><div class="cover-photo"><img src="${A}visium-microscopic-grid.png" alt="H&E field with a 2 µm measurement grid"></div><span class="cover-credit">HD-Cell-RL · Research progress report · September 2026</span><aside class="notes">${esc(s.note)}</aside></section>`;
  const cls=s.title.length>62?"title-long":s.title.length<42?"title-short":"";
  return `<section class="slide ${cls}" data-kind="${esc(s.kind)}" aria-label="${esc(s.title)}"><span class="top-track">${esc(s.section)}</span><h2>${titleHtml(s)}</h2><p class="lead">${esc(s.lead)}</p><div class="scene">${scene(s)}</div><p class="source">Source · ${esc(s.source)}</p><aside class="notes">${esc(s.note)}</aside></section>`;
}

document.getElementById("slides").innerHTML=window.HD_SLIDES.map(renderSlide).join("");
const query=new URLSearchParams(window.location.search);
const slide=Number.parseInt(query.get("slide"),10);
const fragment=Number.parseInt(query.get("fragment"),10);
if(Number.isInteger(slide)&&slide>=0)window.location.hash=Number.isInteger(fragment)&&fragment>=0?`/${slide}/0/${fragment}`:`/${slide}`;
Reveal.initialize({hash:true,history:true,controls:true,controlsTutorial:false,progress:true,center:false,width:1600,height:900,margin:0,minScale:.2,maxScale:1.5,transition:"fade",transitionSpeed:"fast",slideNumber:"c/t",pdfSeparateFragments:false});

#!/usr/bin/env python3
"""Inspect source profiles and save a rerunnable, executed diagnostic notebook."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import sparse

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from hd_cell_rl.pseudo_count_diagnostics import expected_genes,rarefy_exact,row_features
from hd_cell_rl.experiment_protocol import write_experiment_manifest
from scripts.run_pseudo_realism_experiment import load_h5
from scripts.build_source_separated_pseudo import align_genes
from scripts.run_regional_depth_calibration import prepare_inputs


def source_audit(out,audit,bench,validation):
    data=prepare_inputs(audit,bench,None);bins=data['bins'];genes=data['genes']
    sy,sbar,sg=load_h5(bench/'simulation/scdesign3_generator_disjoint/scdesign3_synthetic_counts.h5')
    lookup={b:i for i,b in enumerate(sbar)};sy=align_genes(sy[[lookup[b] for b in data['assignment'].sc_cell_barcode]],sg,genes)
    raw,rbar,rg=load_h5(validation/'data/single_cell/colorectal/CRC_GSE166555_raw_tumor.h5')
    meta=pd.read_csv(bench/'simulation/scdesign3_generator_disjoint/model_reference_cell_metadata.tsv',sep='\t')
    lookup={b:i for i,b in enumerate(rbar)};raw=align_genes(raw[[lookup[b] for b in meta.Cell]],rg,genes)
    rows=[]
    for name,x,types in [('generator_scRNA',raw,meta.cell_type.to_numpy()),('scdesign3',sy,data['types'])]:
        totals=np.asarray(x.sum(1)).ravel()
        for budget in [250,500,1000,2000]:
            eligible=totals>=budget;target=np.minimum(totals,budget)
            richness=expected_genes(x,target)
            for t in sorted(set(types)):
                keep=(types==t)&eligible
                if not keep.any():continue
                rows.append(dict(source=name,cell_type=t,budget=budget,n_total=int(np.sum(types==t)),n_eligible=int(keep.sum()),
                    expected_genes_mean=float(richness[keep].mean()),expected_genes_median=float(np.median(richness[keep])),
                    expected_genes_q25=float(np.quantile(richness[keep],.25)),expected_genes_q75=float(np.quantile(richness[keep],.75))))
    pd.DataFrame(rows).to_csv(out/'subdiagnostic/source_profile_richness.csv',index=False)
    genes_array=np.asarray(genes);total=np.asarray(sy.sum(1)).ravel();j=genes.index('IGKC')
    ig=np.asarray(sy[:,j].toarray()).ravel()
    pd.DataFrame({'cell_id':data['cells'],'cell_type':data['types'],'patch_id':data['cell_patch'],
        'scdesign_counts':total,'IGKC_counts':ig,'IGKC_fraction':ig/np.maximum(total,1)}).to_csv(out/'subdiagnostic/scdesign_IGKC_sources.csv',index=False)
    top=[]
    for name,path in [('real',audit/'matrices/real.npz'),('source_high_depth',audit/'matrices/source_high_depth.npz')]:
        x=sparse.load_npz(path)
        for p in sorted(bins.patch_id.unique()):
            value=np.asarray(x[bins.patch_id.eq(p).to_numpy()].sum(0)).ravel();indices=np.argsort(-value,kind='stable')[:30]
            for rank,i in enumerate(indices,1):top.append(dict(source=name,patch_id=p,rank=rank,gene=genes_array[i],counts=int(value[i]),fraction=float(value[i]/value.sum())))
    pd.DataFrame(top).to_csv(out/'subdiagnostic/top_genes.csv',index=False)
    return dict(scdesign=bench/'simulation/scdesign3_generator_disjoint/scdesign3_synthetic_counts.h5',
        generator_scRNA=validation/'data/single_cell/colorectal/CRC_GSE166555_raw_tumor.h5',
        generator_metadata=bench/'simulation/scdesign3_generator_disjoint/model_reference_cell_metadata.tsv')


def notebook(out):
    import nbformat
    from nbclient import NotebookClient
    md=nbformat.v4.new_markdown_cell;code=nbformat.v4.new_code_cell
    cells=[md('# Pseudo-HD at matched counts\n\n## tl;dr\n\nAt identical observed counts, original pseudo has roughly matched richness at 2 µm but lower richness in three of four patches at 16 µm and core-cell-region scales. Marker enrichment is patch-specific, and is not equivalent to stronger reference-type separation. See the exact tables below.'),
        md('## Context & Methods\n\nSame physical bins, complete 8/16 µm blocks and binary core-cell-region proxies. Each pair is downsampled without replacement to min(real, pseudo) counts; nothing is upsampled. Reference metrics are rematched separately on the common reference panel.\n\n### Key Assumptions\n\nAll four patches are previously inspected development regions from one section. These conditional controls use real counts and are NOT an independent benchmark. Three seeds are rarefaction repeats of existing matrices, not three generated datasets. Core masks are synthetic proxies, not real-cell GT. Zero-support mismatches are reported separately; different variants can have different retained depths/support. Source-profile comparisons are same-type cohorts, not matched real physical cells.'),
        md('## Data\n\nLocal CSVs are sufficient to rerun this notebook. The experiment and delivery manifests record original sources and source-code snapshots. Sources: matched_metrics.csv, support_and_depth.csv, spatial_metrics.csv and subdiagnostic/source_profile_richness.csv.'),
        code("from pathlib import Path\nimport pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\np=Path('.')\nm=pd.read_csv(p/'per_patch_comparison.csv')\ns=pd.read_csv(p/'support_and_depth.csv')\nz=pd.read_csv(p/'spatial_metrics.csv')\nplt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})\ncolors={'source_high_depth':'#246A9D','regional_uniform_budget':'#B76B1E','regional_geometry_mean':'#727A35','real_half_depth_null':'#555555'}\nnames={'source_high_depth':'Original high depth','regional_uniform_budget':'Source-separated','regional_geometry_mean':'Regional geometry','real_half_depth_null':'Real half-depth null'}\npatches=sorted(m.patch_id.unique())\nassert s.matched_exact_counts.all()\ns.groupby('variant')[['n_total','n_both_positive']].mean().round(1)"),
        md('## Results\n\n### 1. Gene richness after count matching\n\nRatios compare analytic expected detected genes under exact rarefaction, not original raw genes. 1 means matched richness. Each point is one patch; lines only connect the displayed patch categories.'),
        code("fig,axes=plt.subplots(2,2,figsize=(12,8),sharey=True)\nfor ax,scale in zip(axes.ravel(),[2,8,16,0]):\n for variant,color in colors.items():\n  a=m[(m.variant==variant)&(m.panel=='all_genes')&(m.scale_um==scale)].set_index('patch_id').reindex(patches)\n  ax.plot(range(4),a.expected_genes_ratio,marker='o',label=names[variant],color=color,linestyle='--' if variant=='real_half_depth_null' else '-')\n ax.axhline(1,color='#333333',lw=1)\n ax.set_title('Core-cell regions' if scale==0 else str(scale)+' µm complete bins/blocks')\n ax.set_xticks(range(4),[q.replace('patch_','') for q in patches]);ax.set_ylabel('Pseudo / real expected genes')\n ax.set_ylim(.75,1.16);ax.grid(axis='y',alpha=.2)\nhandles,labels=axes[0,0].get_legend_handles_labels()\nfig.legend(handles,labels,loc='upper center',ncol=2,frameon=False)\nfig.tight_layout(rect=[0,0,1,.90]);fig.savefig(p/'subdiagnostic/richness.png',dpi=160);plt.show()"),
        md('The original pseudo excess in raw detected genes does not persist after count matching. At larger regions, the remaining deficit suggests a composition/diversity mismatch. This does not identify a unique cause; zero-only pseudo support and patch composition still matter.'),
        code("a=m[(m.panel=='all_genes')&(m.variant=='source_high_depth')]\na.pivot(index='patch_id',columns='scale_um',values='expected_genes_ratio').round(3)"),
        md('### 2. Marker enrichment versus type separation\n\nBoth metrics use only common reference genes with independently rematched counts, and regions with at least 20 matched counts. Marker score is the maximum share in any fixed reference-derived marker set. LL gap is top1-minus-top2 normalized bin log likelihood, not an EM posterior or segmentation accuracy. Differences are pseudo minus real.'),
        code("fig,axes=plt.subplots(1,2,figsize=(12,4.8))\nfor ax,key,title in zip(axes,['type_ll_gap','max_marker_fraction'],['Reference LL gap at 16 µm','Maximum marker-set share at 16 µm']):\n for variant,color in colors.items():\n  a=m[(m.variant==variant)&(m.panel=='reference_genes')&(m.scale_um==16)].set_index('patch_id').reindex(patches)\n  ax.plot(range(4),a['sim_'+key]-a['real_'+key],marker='o',label=names[variant],color=color,linestyle='--' if variant=='real_half_depth_null' else '-')\n ax.axhline(0,color='#333333',lw=1);ax.set_title(title);ax.set_ylabel('Pseudo minus real')\n ax.set_xticks(range(4),[q.replace('patch_','') for q in patches]);ax.grid(axis='y',alpha=.2)\nhandles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=2,frameon=False)\nfig.tight_layout(rect=[0,0,1,.84]);fig.savefig(p/'subdiagnostic/reference_signal.png',dpi=160);plt.show()"),
        md('Marker enrichment changes direction across patches and does not track LL-gap changes consistently. Avoid describing all pseudo cell types as uniformly too easy to distinguish; named gene sets are diagnostic proxies, not biological validation.'),
        md('### 3. Expression outside the modeled masks\n\nFraction of all observed patch counts outside synthetic masks, before diagnostic rarefaction. Real signal outside these masks is unattributed tissue expression, not proven technical ambient.'),
        code("fig,ax=plt.subplots(figsize=(10,4.5));xx=np.arange(4);width=.25\nfor shift,variant,color in [(-1,'real','#333333'),(0,'source_high_depth',colors['source_high_depth']),(1,'regional_uniform_budget',colors['regional_uniform_budget'])]:\n a=z[(z.variant==variant)&(z.component=='observed')&(z.region=='mask_complement')].drop_duplicates('patch_id').set_index('patch_id').reindex(patches)\n ax.bar(xx+shift*width,a.fraction_counts,width,label='Real HD' if variant=='real' else names[variant],color=color)\nax.set_xticks(xx,[q.replace('patch_','') for q in patches]);ax.set_ylabel('Fraction of patch counts');ax.set_ylim(0,.55);ax.legend(frameon=False,ncol=3);ax.grid(axis='y',alpha=.2)\nfig.tight_layout();fig.savefig(p/'subdiagnostic/mask_complement.png',dpi=160);plt.show()"),
        md('Original high-depth pseudo has no mask-complement RNA, while the real data has substantial expression there. The source-separated model removes that hard support shortcut but does not reproduce the regional proportions everywhere.'),
        md('### 4. Same-type generator profiles at fixed count budgets\n\nCells whose original panel counts are below the budget are excluded, never upsampled. Eligibility is reported because it can differ between raw scRNA and scDesign3. These source-level comparisons are not real-HD-cell validation.'),
        code("profiles=pd.read_csv(p/'subdiagnostic/source_profile_richness.csv')\nprofiles[profiles.budget==1000][['source','cell_type','n_total','n_eligible','expected_genes_mean','expected_genes_q25','expected_genes_q75']].round(2)"),
        md('### 5. Trace the IGKC peak\n\nA synthetic Cycling B cell can contribute a large IGKC peak in a mostly fibroblast context; a peak alone does not establish a scDesign3 bug. This table traces the supplied source profiles, not real-cell expression.'),
        code("src=pd.read_csv(p/'subdiagnostic/scdesign_IGKC_sources.csv')\nsrc[src.patch_id=='patch_0015069'].sort_values('IGKC_counts',ascending=False).head(6).round(4)"),
        md('## Takeaways\n\nDepth inflation, gene-composition mismatch and spatial support mismatch are distinct. Do not globally sharpen or flatten all markers based on these results. Prioritize calibrated source budgets/mixtures and gene-spectrum transfer, with patch-specific checks and independent future validation. Existing generation, EM/RL, shape priors and old runs were not changed.')]
    for i,c in enumerate(cells):c['id']=f'count-diagnostic-{i:02d}'
    nb=nbformat.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
    client=NotebookClient(nb,timeout=180,kernel_name='python3',resources={'metadata':{'path':str(out)}})
    client.km=client.create_kernel_manager();client.km.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
    client.execute();nbformat.validate(nb);nbformat.write(nb,out/'analysis.ipynb')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--skip-source-audit',action='store_true')
    args=p.parse_args();out=args.output_dir.resolve()
    audit=ROOT/'runs/benchmarks/pseudo_realism_20260908T012500Z';bench=ROOT/'runs/benchmarks/em_donor_disjoint_unseen_patches_20260903T192649Z'
    validation=ROOT.parent/'Bin2Cell_Validation'
    inputs={} if args.skip_source_audit else source_audit(out,audit,bench,validation)
    notebook(out)
    if not (out/'delivery/experiment_manifest.json').exists():
        inputs.update({name:out/name for name in ['experiment_manifest.json','matched_metrics.csv','support_and_depth.csv','spatial_metrics.csv',
            'subdiagnostic/source_profile_richness.csv','subdiagnostic/scdesign_IGKC_sources.csv','subdiagnostic/top_genes.csv','analysis.ipynb']})
        regional=ROOT/'runs/benchmarks/regional_depth_blocked_20260909/experiment'
        for path in regional.glob('realizations/leave_out_*/*/*biological.npz'):inputs[str(path.relative_to(regional))]=path
        for path in regional.glob('realizations/leave_out_*/*/modeled_cell.npz'):inputs[str(path.relative_to(regional))]=path
        write_experiment_manifest(out/'delivery',config={'diagnostic':'source-profile audit plus notebook','no_training':True},inputs=inputs,
            protocol={'source_snapshot':'after analytical code completion; original execution snapshot kept separately',
                      'source_comparison':'type-matched source cohorts; count-budget eligibility reported; not real-cell GT'})
    print('NOTEBOOK COMPLETE',out/'analysis.ipynb',flush=True)


if __name__=='__main__':main()

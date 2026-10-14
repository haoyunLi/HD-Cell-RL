#!/usr/bin/env python3
"""Run data-only controls against real HD; no EM or policy is fitted.

Matrices use a compact bin x gene axis with stable physical barcodes. The first
two sorted patch IDs are calibration, the remaining two are audit patches. They
have been used in earlier experiments, so this is diagnostic, not a new test set.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import io, sparse
from scipy.optimize import nnls
from scipy.spatial import cKDTree
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp, spearmanr, wasserstein_distance

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hd_cell_rl.pseudo_realism import (
    aggregate_bins, gene_reweight, move_counts, soften_compartment_weights, thin_counts,
)
from scripts.build_colorectal_pseudo_v3_dataset import (
    calibrate_capture_probabilities, read_feature_names, smoothed_capture_field,
)
from scripts.build_patch_real_background_stress_dataset import read_selected_real_matrix


def vec(x):
    return np.asarray(x).ravel()


def decode(x):
    return np.array([v.decode() if isinstance(v, bytes) else str(v) for v in x])


def save_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False, default=str) + '\n')


def load_h5(path):
    with h5py.File(path) as f:
        m = f['matrix']
        shape = tuple(m['shape'][:])
        x = sparse.csc_matrix((m['data'][:], m['indices'][:], m['indptr'][:]), shape=shape)
        return x.T.tocsr(), decode(m['barcodes'][:]), decode(m['features/name'][:])


def corr(a, b):
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(spearmanr(a, b).statistic)


def place_profiles(profiles, cells, groups, local_by_global, n_bins, membrane_flags,
                   p0, target_nuclear, membrane_strength, uniform, seed, source_module,
                   *, capture_probability=None, capture_diagnostics=None):
    """Use existing compartment geometry, preserving full-cell counts before cropping."""
    rng = np.random.default_rng(seed)
    probability=None if capture_probability is None else np.asarray(capture_probability,float)
    if probability is not None and (probability.shape!=(n_bins,) or not np.isfinite(probability).all() or
                                    np.any((probability<0)|(probability>1))):
        raise ValueError('capture probabilities must align to bins and lie in [0,1]')
    if probability is not None and np.all(probability==1):probability=None
    totals = vec(profiles.sum(0))
    factors = rng.lognormal(0, 0.2, len(cells))
    base = profiles.multiply(factors[:, None]).multiply(p0[None, :])
    alpha = target_nuclear * profiles.sum() / base.sum()
    # Solve clipping-aware target using the same fixed nuclear factors.
    x = profiles.tocoo()
    for _ in range(60):
        achieved = np.dot(x.data, np.clip(alpha * factors[x.row] * p0[x.col], 1e-8, 1-1e-8))
        alpha *= target_nuclear * profiles.sum() / achieved
    rows, cols, data = [], [], []
    contribution_rows, contribution_cols, contribution_values = [], [], []
    outside = 0
    capture_rejected = 0
    for i, cell in enumerate(cells):
        group = groups[int(cell)]
        bins = group['barcode_index'].to_numpy(dtype=np.int64)
        mapped = np.array([local_by_global.get(int(v), -1) for v in bins])
        coverage = group.cell_coverage_fraction.to_numpy()
        nw, cw, mw = soften_compartment_weights(
            group.nuclear_fraction, group.cytoplasm_fraction, group.membrane_fraction,
            membrane_strength,
        )
        nw = source_module.nuclear_kernel(group)[0]
        s, e = profiles.indptr[i:i+2]
        genes, counts = profiles.indices[s:e], profiles.data[s:e].astype(np.int64)
        nuclear = rng.binomial(counts, np.clip(alpha * factors[i] * p0[genes], 1e-8, 1-1e-8))
        membrane = np.where(membrane_flags[genes], counts - nuclear, 0)
        cytoplasm = counts - nuclear - membrane
        batches = [(counts, coverage/coverage.sum())] if uniform else [(nuclear, nw), (cytoplasm, cw), (membrane, mw)]
        for amounts, weights in batches:
            molecule_genes = np.repeat(genes, amounts)
            destinations = rng.choice(len(bins), len(molecule_genes), p=weights)
            local = mapped[destinations]
            valid = local >= 0
            outside += int((~valid).sum())
            if probability is not None:
                accepted=rng.random(int(valid.sum()))<probability[local[valid]]
                capture_rejected+=int((~accepted).sum())
                valid[valid]=accepted
            rows.append(local[valid]); cols.append(molecule_genes[valid])
            data.append(np.ones(valid.sum(), dtype=np.int32))
            contribution_rows.append(np.full(valid.sum(), i, dtype=np.int64))
            contribution_cols.append(local[valid]); contribution_values.append(np.ones(valid.sum(), dtype=np.int32))
    matrix = sparse.csr_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
                               shape=(n_bins, profiles.shape[1]))
    contribution = sparse.csr_matrix((np.concatenate(contribution_values),
                                     (np.concatenate(contribution_rows), np.concatenate(contribution_cols))),
                                    shape=(len(cells), n_bins))
    if matrix.sum() + outside + capture_rejected != profiles.sum():
        raise AssertionError('localization count conservation failed')
    if capture_diagnostics is not None:
        capture_diagnostics.update(proposed_counts=int(profiles.sum()),outside_crop_counts=outside,
                                   capture_rejected_counts=capture_rejected,captured_counts=int(matrix.sum()))
    return matrix, contribution


def measure(matrix, real, grid, patch, background, owner, nuclear, core_cells, centers,
            coordinates, reference, reference_genes, label, seed):
    rows = []
    for part, keep in [('calibration', patch < 2), ('audit', patch >= 2)]:
        for width in (1, 4, 8):
            a, keys = aggregate_bins(matrix[keep], grid[keep], patch[keep], width)
            b, _ = aggregate_bins(real[keep], grid[keep], patch[keep], width)
            if a.shape[0] == 0:
                continue
            total, rt = vec(a.sum(1)), vec(b.sum(1))
            genes, rg = np.diff(a.indptr), np.diff(b.indptr)
            p, q = vec(a.sum(0)) + 1e-8, vec(b.sum(0)) + 1e-8
            gene_mean, real_mean = vec(a.mean(0)), vec(b.mean(0))
            gene_var = vec(a.power(2).mean(0)) - gene_mean ** 2
            real_var = vec(b.power(2).mean(0)) - real_mean ** 2
            row = dict(variant=label, seed=seed, partition=part, scale_um=2*width,
                       n_blocks=len(total), total_umis=int(total.sum()), mean_umi=float(total.mean()),
                       median_umi=float(np.median(total)), p99_umi=float(np.quantile(total,.99)),
                       zero_bin_fraction=float(np.mean(total==0)), mean_genes=float(genes.mean()),
                       depth_ks=float(ks_2samp(total,rt).statistic), genes_ks=float(ks_2samp(genes,rg).statistic),
                       log_depth_wasserstein=float(wasserstein_distance(np.log1p(total),np.log1p(rt))),
                       gene_js=float(jensenshannon(p,q)), gene_total_spearman=corr(p,q),
                       mean_variance_spearman=corr(gene_var,real_var),
                       mean_ratio_to_real=float(total.sum()/max(rt.sum(),1)))
            # Adjacent-block cosine retains the absolute grid and cannot connect patches.
            lookup={tuple(k):i for i,k in enumerate(keys)}
            pairs=[(i,lookup[tuple(k+[0,0,1])]) for i,k in enumerate(keys) if tuple(k+[0,0,1]) in lookup]
            if pairs:
                ii,jj=np.asarray(pairs).T
                for prefix,x in [('sim',a),('real',b)]:
                    norms=np.sqrt(vec(x.power(2).sum(1)))
                    cosine=vec(x[ii].multiply(x[jj]).sum(1))/np.maximum(norms[ii]*norms[jj],1e-20)
                    row[prefix+'_neighbor_cosine']=float(cosine.mean())
                    row[prefix+'_neighbor_depth_corr']=corr(vec(x.sum(1))[ii],vec(x.sum(1))[jj])
            if width==8:
                # A reference-span diagnostic, not the RESCUE algorithm or an ambient estimator.
                z=a[:,reference_genes].toarray().astype(float)
                residual=[]
                for v in z:
                    if v.sum()>0:
                        y=v/v.sum()
                        fitted=reference.T@nnls(reference.T,y)[0]
                        residual.append(np.maximum(y-fitted,0).sum())
                row['reference_positive_residual_fraction']=float(np.mean(residual)) if residual else np.nan
            rows.append(row)
    totals=vec(matrix.sum(1))
    # The support-only baseline never sees GT when assigning: it uses positive counts and nuclei.
    nearest=cKDTree(centers).query(coordinates)[1]
    predicted=np.where(totals>0,core_cells[nearest],-1)
    cell_rows=[]
    for cell in core_cells:
        gt=owner==cell; pred=predicted==cell
        if not gt.any():continue
        union=(gt|pred).sum()
        cell_rows.append(dict(variant=label,seed=seed,cell_id=int(cell),
                              support_nearest_iou=float((gt&pred).sum()/max(union,1))))
    diag=dict(variant=label,seed=seed,total=int(totals.sum()),
              fraction_counts_outside_pseudo_masks=float(totals[background].sum()/max(totals.sum(),1)),
              background_positive_fraction=float(np.mean(totals[background]>0)),
              cell_covered_positive_fraction=float(np.mean(totals[~background]>0)),
              nuclear_umi_fraction=float(totals[nuclear].sum()/max(totals.sum(),1)),
              support_detects_gt_coverage_balanced_accuracy=float((np.mean(totals[~background]>0)+np.mean(totals[background]==0))/2))
    return rows,diag,cell_rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark-dir',type=Path,required=True)
    parser.add_argument('--validation-repo',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--seeds',type=int,nargs='+',default=[20260907,20260908,20260909])
    args=parser.parse_args()
    out=args.output_dir.resolve(); out.mkdir(parents=True,exist_ok=False)
    for name in ['subdiagnostic','matrices','comparison']: (out/name).mkdir()
    bench=args.benchmark_dir.resolve(); validation=args.validation_repo.resolve()
    save_json(out/'protocol.json',dict(benchmark=str(bench),seeds=args.seeds,
              no_inference_training=True,reference='donor-disjoint scRNA',
              split='first two sorted patch IDs calibration; last two audit; previously examined geometry',
              units='integer UMI-like probe counts; gene x physical barcode axes verified',
              ambient_warning='mask-complement real expression is structured unassigned tissue, not known ambient'))
    selected=pd.read_parquet(bench/'inputs/selected_context_bins.parquet').sort_values('matrix_col_index').reset_index(drop=True)
    glob=selected.matrix_col_index.to_numpy(np.int64); coords=selected[['x_um','y_um']].to_numpy()
    grid=selected.barcode.str.extract(r's_002um_(\d+)_(\d+)').astype(int).to_numpy()
    patches=pd.read_csv(bench/'selection/test_patches_index.csv').sort_values('patch_id').reset_index(drop=True)
    patch=np.full(len(glob),-1,dtype=int)
    for i,row in patches.iterrows():
        keep=(coords[:,0]>=row.context_x_min)&(coords[:,0]<=row.context_x_max)&(coords[:,1]>=row.context_y_min)&(coords[:,1]<=row.context_y_max)
        if np.any(patch[keep]>=0):raise ValueError('overlapping contexts')
        patch[keep]=i
    if np.any(patch<0):raise ValueError('unmapped context bins')
    selected['patch_id']=patches.patch_id.to_numpy()[patch]
    selected['partition']=np.where(patch<2,'calibration','audit')
    selected.to_parquet(out/'barcodes.parquet',index=False)
    source_dir=bench/'datasets/source_pseudo_hd_generator_disjoint'
    genes=read_feature_names(source_dir/'filtered_feature_bc_matrix')
    pd.Series(genes,name='gene').to_csv(out/'genes.csv',index=False)
    pseudo_full=io.mmread(source_dir/'filtered_feature_bc_matrix/matrix.mtx.gz').tocsc()
    shape=pseudo_full.shape; source=pseudo_full[:,glob].T.tocsr(); del pseudo_full
    real_path=validation/'data/visium_hd/colorectal/output/binned_outputs/square_002um/filtered_feature_bc_matrix.h5'
    real=read_selected_real_matrix(real_h5_path=real_path,selected_columns=glob,
         expected_shape=shape,expected_barcodes=selected.barcode.to_numpy(),expected_feature_names=genes).tocsc()[:,glob].T.tocsr()
    legacy=io.mmread(bench/'datasets/capture_matched_structured_ambient/filtered_feature_bc_matrix/matrix.mtx.gz').tocsc()[:,glob].T.tocsr()
    sparse.save_npz(out/'matrices/real.npz',real)
    sparse.save_npz(out/'matrices/source_high_depth.npz',source)
    gt=pd.read_parquet(bench/'inputs/fractional_gt_contributing_cells.parquet')
    lookup={int(v):i for i,v in enumerate(glob)}
    local=gt[gt.barcode_index.isin(glob)].copy()
    coverage=local.groupby('barcode_index').cell_coverage_fraction.sum().reindex(glob,fill_value=0).to_numpy()
    background=coverage<=0
    nuclear=local.groupby('barcode_index').nuclear_fraction.sum().reindex(glob,fill_value=0).to_numpy()>0
    dominant=local.sort_values(['barcode_index','cell_coverage_fraction','cell_id'],ascending=[True,False,True]).drop_duplicates('barcode_index').set_index('barcode_index').cell_id
    owner=dominant.reindex(glob,fill_value=-1).to_numpy()
    cellmeta=pd.read_csv(bench/'inputs/ground_truth_cells.csv.gz').sort_values('cell_id')
    allcells=cellmeta.cell_id.to_numpy(np.int64)
    nuc=pd.read_parquet(bench/'episode_context/nuclei_patch_context.parquet')
    core=pd.read_csv(bench/'selection/source_eval/per_episode.csv').cell_id.to_numpy(np.int64)
    centers=nuc.set_index('cell_id').loc[core,['center_x_um','center_y_um']].to_numpy()
    # Include context nuclei in the diagnostic owner search, then score core cells only below.
    diagnostic_cells=nuc.cell_id.to_numpy(np.int64)
    diagnostic_centers=nuc[['center_x_um','center_y_um']].to_numpy()
    save_json(out/'subdiagnostic/geometry.json',dict(n_bins=len(glob),n_core_cells=len(core),
              n_context_nuclei=len(nuc),n_contributing_cells=len(cellmeta),
              multi_owner_bin_fraction=float((local.groupby('barcode_index').cell_id.nunique()>1).mean()),
              area_coverage_fraction=float(coverage.mean()),
              fraction_real_counts_outside_pseudo_masks=float(real[background].sum()/real.sum()),
              fraction_cells_with_area_target_reached=float(cellmeta.target_reached.mean())))
    # Type profiles are fitted only on calibration-selected genes for span diagnostics.
    ref=np.load(bench/'reference/reference_counts_donor_disjoint.npz',allow_pickle=True)
    refgenes=decode(ref['genes']); rmap={g:i for i,g in enumerate(refgenes)}
    rg=np.array([i for i,g in enumerate(genes) if g in rmap])
    rg=rg[np.argsort(vec(real[patch<2].sum(0))[rg])[-500:]]
    reference=ref['reference_counts'][:,[rmap[genes[i]] for i in rg]].astype(float)+1e-8
    reference/=reference.sum(1,keepdims=True)
    # Weakly shrunk gene-specific ratios; they confound biology/platform and are a diagnostic.
    a=vec(source[patch<2].sum(0)); b=vec(real[patch<2].sum(0))
    gene_eff=np.clip(((b+5)/(b.sum()+5*len(b)))/((a+5)/(a.sum()+5*len(a))),.25,4.)**.5
    pd.DataFrame({'gene':genes,'calibration_source_counts':a,'calibration_real_counts':b,
                  'shrunk_ratio':gene_eff}).to_csv(out/'subdiagnostic/gene_platform_factors.csv',index=False)
    realtot=vec(real.sum(1)); sourcetot=vec(source.sum(1))
    field=smoothed_capture_field(coords,realtot,sigma_um=16,n_neighbors=128)
    legacy_cell_target=int(real[~background].sum())
    def field_p(x,target):return calibrate_capture_probabilities(vec(x.sum(1)),field,int(target))
    rows=[]; diagnostics=[]; cellrows=[]
    def record(name,x,seed,extra=None):
        m,d,c=measure(x,real,grid,patch,background,owner,nuclear,diagnostic_cells,
                      diagnostic_centers,coords,reference,rg,name,seed)
        c=[v for v in c if v['cell_id'] in set(core)]
        rows.extend(m); d.update(extra or {}); diagnostics.append(d);cellrows.extend(c)
        pd.DataFrame(rows).to_csv(out/'comparison/metrics.csv',index=False)
        pd.DataFrame(diagnostics).to_csv(out/'comparison/support_and_compartments.csv',index=False)
        pd.DataFrame(cellrows).to_csv(out/'subdiagnostic/support_nearest_per_cell.csv',index=False)
        if seed in (0,args.seeds[0]):sparse.save_npz(out/f'matrices/{name}.npz',x)
        audit=[v for v in m if v['partition']=='audit' and v['scale_um']==2][0]
        print(f'{name} seed={seed} audit depth_KS={audit["depth_ks"]:.3f} gene_JS={audit["gene_js"]:.3f} support_BA={d["support_detects_gt_coverage_balanced_accuracy"]:.3f}',flush=True)
    record('real',real,0);record('source_high_depth',source,0);record('legacy_capture_structured',legacy,0)
    for seed in args.seeds:
        rng=np.random.default_rng(seed)
        uniform=thin_counts(source,min(real.sum()/source.sum(),1),rng)
        record('depth_uniform',uniform,seed)
        capture=thin_counts(source,field_p(source,legacy_cell_target),np.random.default_rng(seed+1))
        record('thinning_only_legacy_target',capture,seed)
        record('depth_field_total',thin_counts(source,field_p(source,real.sum()),rng),seed)
        pp=np.zeros(len(glob))
        for p in range(4):
            keep=patch==p;pp[keep]=min(real[keep].sum()/source[keep].sum(),1)
        record('depth_per_patch',thin_counts(source,pp,rng),seed)
        pp=np.clip(realtot/np.maximum(sourcetot,1),0,1)
        record('depth_per_bin_oracle_control',thin_counts(source,pp,rng),seed)
        unassigned=real.multiply(background[:,None]).tocsr()
        transported,rejection=move_counts(unassigned,grid,patch,6,1,np.random.default_rng(seed+2))
        record('unassigned_only_added_high_depth',source+transported,seed,{'transport_rejected_fraction':rejection})
        record('capture_plus_unassigned_transport',capture+transported,seed,{'transport_rejected_fraction':rejection})
        for frac in (.1,.25):
            # Counts from real tissue are replay controls with unknown ownership, not synthetic ambient GT.
            mixture=thin_counts(source,(1-frac)*real.sum()/source.sum(),rng)+thin_counts(real,frac,rng)
            record(f'real_tissue_replay_{frac:g}',mixture,seed,{'real_component_ownership_known':False})
        for sigma,frac in [(1,.1),(1,.3),(2,.1),(4,.1)]:
            moved,rejection=move_counts(uniform,grid,patch,sigma,frac,rng)
            record(f'local_transport_{sigma*2}um_{frac:g}',moved,seed,{'transport_rejected_fraction':rejection})
        record('gene_platform_depth_matched',gene_reweight(uniform,gene_eff,rng),seed)
    # Reuse source localization code, changing only requested experimental controls.
    module_dir=validation/'scripts/simulation';sys.path.insert(0,str(module_dir))
    spec=importlib.util.spec_from_file_location('source_localization',module_dir/'create_colorectal_pseudo_hd.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    profiles,bars,pgenes=load_h5(bench/'simulation/scdesign3_generator_disjoint/scdesign3_synthetic_counts.h5')
    assign=pd.read_csv(bench/'assignment/ground_truth_cell_to_synthetic_scdesign3.csv').set_index('cell_id').loc[allcells]
    profiles=profiles[[{v:i for i,v in enumerate(bars)}[v] for v in assign.sc_cell_barcode]].tocsr()
    p0,_,_=mod.localization_probabilities(pgenes,profiles.data,profiles.indices,
          validation/'references/gene_annotation/gencode_v38.gtf')
    pmap={g:i for i,g in enumerate(pgenes)}
    gene_mapping=sparse.csr_matrix((np.ones(sum(g in pmap for g in genes)),
             ([pmap[g] for g in genes if g in pmap],[i for i,g in enumerate(genes) if g in pmap])),shape=(len(pgenes),len(genes)))
    profiles=(profiles@gene_mapping).astype(np.int64).tocsr()
    p0=np.array([p0[pmap[g]] if g in pmap else .02 for g in genes])
    membrane_genes=mod.load_membrane_genes(validation/'references/gene_localization/human_membrane_secretory_genes_uniprot.txt')
    flags=np.array([g in membrane_genes for g in genes])
    groups={int(c):g for c,g in gt.groupby('cell_id')}
    types=assign.cell_type.to_numpy()
    type_means=np.stack([vec(profiles[types==t].mean(0)) for t in types])
    means=type_means/np.maximum(type_means.sum(1,keepdims=True),1)
    # Test loss of unique synthetic cell fingerprints separately from compartment localization.
    for seed in args.seeds:
        rng=np.random.default_rng(seed)
        for name,mem_strength,nuc_target,uniform_place,pooled in [
            ('regenerated_current',1,.25,False,False),
            ('membrane_relaxed_025',.25,.25,False,False),
            ('membrane_no_outer_bias',0,.25,False,False),
            ('nuclear_target_010',.25,.10,False,False),
            ('nuclear_target_040',.25,.40,False,False),
            ('uniform_cell_localization',0,.25,True,False),
            ('pooled_type_profiles',.25,.25,False,True)]:
            used=profiles
            if pooled:
                used=sparse.csr_matrix(np.stack([rng.multinomial(int(n),p) for n,p in zip(vec(profiles.sum(1)),means)]))
            x,contribution=place_profiles(used,allcells,groups,lookup,len(glob),flags,p0,nuc_target,
                                          mem_strength,uniform_place,seed,mod)
            x=thin_counts(x,min(real.sum()/x.sum(),1),np.random.default_rng(seed+91))
            record(name,x,seed)
            if seed==args.seeds[0]:
                sparse.save_npz(out/f'subdiagnostic/{name}.source_cell_bin_counts.npz',contribution)
    # Seed observation shifts change only the observed nuclear mask, not simulated ownership.
    nuclear_owners=local[local.nuclear_fraction>0].sort_values(['barcode_index','nuclear_fraction'],ascending=[True,False]).drop_duplicates('barcode_index')
    ni=np.array([lookup[int(v)] for v in nuclear_owners.barcode_index]);nc=nuclear_owners.cell_id.to_numpy()
    grid_lookup={(int(p),int(g[0]),int(g[1])):i for i,(p,g) in enumerate(zip(patch,grid))}
    seed_rows=[]
    for shift in (0,1,2):
        dest=np.array([grid_lookup.get((int(patch[i]),int(grid[i,0]),int(grid[i,1]+shift)),-1) for i in ni])
        valid=dest>=0
        seed_rows.append(dict(shift_um=2*shift,n_seed_bins=int(valid.sum()),
                             fraction_shifted_seed_bins_with_different_dominant_owner=float(np.mean(owner[dest[valid]]!=nc[valid])),
                             dropped_at_context_edge=int((~valid).sum())))
    pd.DataFrame(seed_rows).to_csv(out/'comparison/nuclear_observation_shift.csv',index=False)
    # Gene-matched raw-scRNA versus synthetic cell depth and reference fingerprints.
    raw,rawbars,rawgenes=load_h5(validation/'data/single_cell/colorectal/CRC_GSE166555_raw_tumor.h5')
    meta=pd.read_csv(bench/'simulation/scdesign3_generator_disjoint/model_reference_cell_metadata.tsv',sep='\t')
    rawmap={v:i for i,v in enumerate(rawbars)}
    raw=raw[[rawmap[v] for v in meta.Cell]]
    rawg={v:i for i,v in enumerate(rawgenes)}
    shared=[g for g in genes if g in rawg]
    raw=raw[:,[rawg[g] for g in shared]]
    syn=profiles[:,[genes.index(g) for g in shared]]
    profile_rows=[]
    for typ in np.unique(types):
        for name,x in [('raw_generator_scRNA',raw[meta.cell_type.to_numpy()==typ]),('scdesign3',syn[types==typ])]:
            totals=vec(x.sum(1)); detected=np.diff(x.indptr)
            profile_rows.append(dict(cell_type=typ,source=name,n_cells=x.shape[0],mean_umi=totals.mean(),
                                     median_umi=np.median(totals),umi_cv=totals.std()/max(totals.mean(),1),
                                     median_genes=np.median(detected)))
    pd.DataFrame(profile_rows).to_csv(out/'comparison/source_profile_depth.csv',index=False)
    save_json(out/'summary.json',dict(completed=True,n_configurations=len({d['variant'] for d in diagnostics}),
               n_realizations=len(diagnostics),n_seeds=len(args.seeds),n_genes=len(genes),n_bins=len(glob),
               no_em_or_ppo_training=True,old_outputs_overwritten=False))
    print('COMPLETE',out,flush=True)


if __name__=='__main__':main()

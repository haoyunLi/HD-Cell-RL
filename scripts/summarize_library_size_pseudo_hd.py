#!/usr/bin/env python3
"""Combine the paired library-size pseudo-HD benchmark without tuning methods."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import pandas as pd
from scipy import sparse

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from hd_cell_rl.experiment_protocol import paired_cluster_mean_ci


def table(x):
    return '\n'.join(['| '+' | '.join(x.columns)+' |','| '+' | '.join(['---']*len(x.columns))+' |']+
        ['| '+' | '.join(f'{v:.4f}' if isinstance(v,(float,np.floating)) else str(v) for v in row)+' |'
         for row in x.itertuples(index=False,name=None)])


def owner_target_diagnostics(root):
    """Evaluation only: dominant-area GT versus largest modeled-RNA contributor."""
    from scripts.run_source_separated_method_comparison import GT, BENCH
    bins=pd.read_parquet(root/'barcodes.parquet')
    gt=pd.read_csv(GT,usecols=['barcode','cell_id']).set_index('barcode').cell_id
    if not gt.index.is_unique:raise ValueError('GT barcode owners are not unique')
    truth=bins.barcode.map(gt)
    core=set(pd.read_csv(BENCH/'selection/source_eval/per_episode.csv').cell_id.astype(int))
    rows=[]
    for model in ['A_current','B_library_offset']:
        identities=pd.read_csv(root/model/'profile_assignment.csv').cell_id.to_numpy(int)
        for patch,group in bins.groupby('patch_id'):
            matrix=sparse.load_npz(root/model/'regional/realizations'/('leave_out_'+patch)/'uniform_budget/modeled_source_bin_counts.npz').tocsc()
            if matrix.shape!=(len(identities),len(bins)):raise ValueError('source contribution axes changed')
            n=tied=positive=mixed=disagree=0
            for index in group.index:
                owner=truth.iloc[index]
                if pd.isna(owner) or int(owner) not in core:continue
                n+=1;s,e=matrix.indptr[index:index+2];counts=matrix.data[s:e];sources=matrix.indices[s:e]
                keep=counts>0;counts=counts[keep];sources=sources[keep]
                if not len(counts):continue
                positive+=1;mixed+=int(len(counts)>1)
                winners=sources[counts==counts.max()]
                if len(winners)!=1:tied+=1;continue
                disagree+=int(identities[winners[0]]!=int(owner))
            rows.append(dict(generator=model,patch_id=patch,n_core_gt_bins=n,n_positive_modeled_bins=positive,
                n_multiple_modeled_sources=mixed,n_tied_rna_top=tied,n_unique_rna_top=positive-tied,
                n_rna_top_differs_from_gt=disagree,fraction_disagree=disagree/max(positive-tied,1)))
    result=pd.DataFrame(rows);result.to_csv(root/'subdiagnostic/ownership_target_diagnostics.csv',index=False)
    total=result.groupby('generator')[['n_unique_rna_top','n_rna_top_differs_from_gt']].sum().reset_index()
    total['fraction_disagree']=total.n_rna_top_differs_from_gt/total.n_unique_rna_top
    return total


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run_dir',type=Path)
    root=parser.parse_args().run_dir.resolve();methods=['A_current','B_library_offset']
    summaries=[];per_cells=[];per_patch=[];uptakes=[]
    all_checks=[]
    for model in methods:
        path=root/model/'benchmark'
        summary=pd.read_csv(path/'method_comparison.csv')
        if len(summary)!=10 or not summary.n_cells.eq(43).all():raise ValueError('incomplete five-method paired comparison')
        summaries.append(summary.assign(generator=model))
        per_cells.append(pd.read_csv(path/'subdiagnostic/per_cell_comparison.csv',dtype={'cell_id':str}).assign(generator=model))
        per_patch.append(pd.read_csv(path/'per_patch_comparison.csv').assign(generator=model))
        uptakes.append(pd.read_csv(path/'unmodeled_uptake.csv').assign(generator=model))
        checks=json.loads((path/'subdiagnostic/evaluation_checks.json').read_text())
        all_checks += [dict(generator=model,**c) for c in checks]
        if len(checks)!=10 or not all(c['exported_barcodes_unique'] and c['retained_nuclear_owners_exact'] and c['core_cells_exact'] for c in checks):
            raise ValueError('ownership, seed, or evaluation scope checks failed')
    a=pd.read_csv(root/methods[0]/'profile_assignment.csv');b=pd.read_csv(root/methods[1]/'profile_assignment.csv')
    pd.testing.assert_frame_equal(a[['cell_id','cell_type','sc_cell_barcode']],b[['cell_id','cell_type','sc_cell_barcode']])
    seeds=[pd.read_csv(root/m/'benchmark/common_confident_nuclear_bins.csv.gz',dtype=str) for m in methods]
    pd.testing.assert_frame_equal(seeds[0],seeds[1])
    for fold in (root/methods[0]/'regional/realizations').iterdir():
        filename=Path('regional/realizations')/fold.name/'uniform_budget/unmodeled_biological.npz'
        if (sparse.load_npz(root/methods[0]/filename)-sparse.load_npz(root/methods[1]/filename)).nnz:
            raise ValueError('unmodeled counts differ across A/B')
    summary=pd.concat(summaries,ignore_index=True);summary.to_csv(root/'method_comparison.csv',index=False)
    cells=pd.concat(per_cells,ignore_index=True);cells.to_csv(root/'subdiagnostic/per_cell_comparison.csv',index=False)
    if cells.gene_common_count.nunique(dropna=False)!=1:raise ValueError('expression evaluation gene panels differ')
    pd.DataFrame(all_checks).to_csv(root/'subdiagnostic/all_method_checks.csv',index=False)
    pd.concat(per_patch,ignore_index=True).to_csv(root/'per_patch_comparison.csv',index=False)
    pd.concat(uptakes,ignore_index=True).to_csv(root/'unmodeled_uptake.csv',index=False)
    rows=[]
    for (condition,method),group in cells.groupby(['condition','method']):
        a=group[group.generator==methods[0]].set_index('cell_id')
        b=group[group.generator==methods[1]].set_index('cell_id').loc[a.index]
        if not a.patch_id.equals(b.patch_id):raise ValueError('patch pairing changed')
        for metric in ['pred_iou','pred_fractional_iou','gene_spearman_r','pred_precision','pred_recall']:
            delta=b[metric].to_numpy()-a[metric].to_numpy()
            if not np.isfinite(delta).all():raise ValueError('nonfinite paired metric')
            lo,hi=paired_cluster_mean_ci(delta,a.patch_id.to_numpy(),rng=np.random.default_rng(20260916))
            rows.append(dict(condition=condition,method=method,metric=metric,delta_B_minus_A=float(delta.mean()),
                             patch_bootstrap_low=lo,patch_bootstrap_high=hi,n_patches=a.patch_id.nunique()))
    pd.DataFrame(rows).to_csv(root/'paired_generator_effect.csv',index=False)
    panels=[]
    for condition in ['modeled_only','source_separated']:
        z=summary[summary.condition==condition]
        panel=z.pivot(index='method',columns='generator',values=['mean_pred_iou','mean_gene_spearman_r'])
        panel.columns=['A IoU','B IoU','A gene Spearman','B gene Spearman'];panel=panel.reset_index()
        panels.append('### '+condition+'\n\n'+table(panel))
    realism=pd.read_csv(root/'multiscale_realism.csv')
    metrics=realism[(realism.scale_um==2)&(realism.condition=='source_separated')][
        ['variant','patch_id','counts_mean','real_counts_mean','zero_fraction','real_zero_fraction']]
    acceptance=json.loads((root/'realism_acceptance.json').read_text())
    flags=json.loads((root/'subdiagnostic/capture_checks.json').read_text())
    for fold in {r['fold'] for r in flags}:
        values=[r['probability'] for r in flags if r['fold']==fold]
        if len(values)!=2 or values[0]!=values[1]:raise ValueError('A/B capture probabilities differ')
    check=dict(complete=True,n_method_evaluations=20,n_core_cells=43,n_patches=4,
        type_matched_source_ids=True,identical_confident_seeds=True,identical_unmodeled_counts=True,
        identical_capture_probabilities=True,unique_ownership=True,
        realism_checks_passed=sum(r['passed'] for r in acceptance['checks']),
        realism_checks_total=len(acceptance['checks']))
    (root/'subdiagnostic/paired_checks.json').write_text(json.dumps(check,indent=2)+'\n')
    variation=[]
    for model in methods:
        for path in (root/model/'regional/realizations').glob('*/uniform_budget/modeled_sources.csv'):
            frame=pd.read_csv(path)
            variation.append(dict(generator=model,fold=path.parents[1].name,
                source_counts_cv=frame.source_counts.std()/frame.source_counts.mean(),
                captured_counts_cv=frame.captured_in_crop_counts.std()/frame.captured_in_crop_counts.mean()))
    pd.DataFrame(variation).to_csv(root/'subdiagnostic/source_to_capture_variation.csv',index=False)
    owner_targets=owner_target_diagnostics(root)
    em=summary[(summary.condition=='modeled_only')&(summary.method=='frozen_em')].set_index('generator')
    report=f'''# Library-size 模式的 pseudo-HD 算法比较

已完成 A/B × modeled-only/source-separated × 五种方法，共 20 份评价。
评价使用相同 4 个已检查过的开发 patches、43 个 core cells；233 个物理细胞提供表达来源，context cells 参与竞争。
这是一份单一 source seed 的开发实验，不是独立 donor 或新 test-set 验证。
矩阵保留全局 barcode 轴以兼容既有代码，只有选区 10,827 个 bins 被生成，选区外为空；不能作为完整切片使用。

## 如何生成

- A 使用原 scDesign3 模型，B 使用 library-size offset 模型。按原 cell type 配对抽取相同 source ID；seed=20260916，无重复使用 source cell。
- 保留源 counts，通过现有核/胞质/膜定位核分配到 2 µm bins。核比例 0.25、膜设置及 cell 几何未改。
- 不再逐 cell 重采样到指定 counts。每个 leave-out fold 仅用另外三个 patch 的旧 modeled counts 预算确定 A 的捕获概率，B 使用完全相同的概率；范围 {min(r['probability'] for r in flags):.4f}–{max(r['probability'] for r in flags):.4f}。
- modeled-only 为捕获后的 modeled-cell RNA；source-separated 另加完全相同的已保存 unmodeled biological RNA。技术背景为零。
- 新生成 counts、数据集、episode 与结果均单独保存。原生成器、旧训练/evolution、默认数据未替换。

## 冻结的算法比较

EM：alpha=1、beta=1.5、kappa=2、profile_update_damping=0、background disabled；distance-only 的 beta=0。
Bin2Cell/STCS 沿用之前的参数和共同 confident nuclear seeds。没有调 weights、重新校准 entropy、训练 PPO 或运行 SMURF。
Gene Spearman 对比各自正确的 generating profile，非 realized captured-count GT。所有方法共享 {int(cells.gene_common_count.iloc[0]):,} 个评价基因。

{(chr(10)*2).join(panels)}

## 本轮观察

在 modeled-only 中，冻结 EM 的 IoU 从 {em.loc['A_current','mean_pred_iou']:.4f} 降至 {em.loc['B_library_offset','mean_pred_iou']:.4f}。
Precision 从 {em.loc['A_current','mean_pred_precision']:.4f} 到 {em.loc['B_library_offset','mean_pred_precision']:.4f}，recall 从 {em.loc['A_current','mean_pred_recall']:.4f} 到 {em.loc['B_library_offset','mean_pred_recall']:.4f}。
因此不能只用“错误扩张”解释结果；GT bins 未被归回原 cell 的情况也增加了。具体机制仍需按 cell 的核表达、counts 和错误 bin 分析，不能由总体均值确定。
本轮新 B 数据上，EM 的 IoU 未超过 Bin2Cell 或 distance-only。没有据此调整参数，也不能据此宣称这些方法在所有真实 HD 数据上的排序。

## 数据现实性仍有限

现实性验收通过 {check['realism_checks_passed']}/{check['realism_checks_total']} 个 condition/patch 组合。算法分数不作为现实性验收目标。
以下是 source-separated 的 2 µm 深度和空 bin 比例；2/8/16 µm 及 cell-region 完整指标见 multiscale_realism.csv。

{table(metrics)}

在 core GT bins 中，最大 modeled-RNA 来源与当前 dominant-owner GT 不同的比例如下。只统计有 modeled counts 且 RNA 第一名不并列的 bins；不包含 unmodeled RNA。这是评价目标差异诊断，不代表算法预测正确或错误，也不证明 EM 下降的因果来源。

{table(owner_targets)}

- 未建模 RNA 沿用原压力测试预算，未随新模式再调参。因此其相对占比可能随 modeled counts 改变，并非固定总 counts 的混合实验。
- 新模式仍有 Cycling B 过度离散和参考注释差异；本轮未更改标签。
- 尚未建模 source library size 与物理 cell 面积的相关性。区域深度也仍与真实数据不匹配。
- 这里的捕获概率是计数抽样参数，不是测量或识别出的真实技术捕获效率。
- 与 9 月 14 日历史结果相比，capture 流程也改变了，不能把差异全归因于 library size。应以本轮配对 A/B 为主。
- 四个 patch 的 bootstrap 仅作开发诊断，不能当作独立 donor 的泛化区间。
- Bin2Cell 的现有 TensorFlow 环境自动回退到 CPU；可选可视化因缺少 scaled-image key 警告跳过。赋值及数值评价需通过下述检查才会生成此报告。

## 输出与验证

- [总表](method_comparison.csv)、[逐 patch](per_patch_comparison.csv)、[A/B 配对差值](paired_generator_effect.csv)。
- [数据现实性](multiscale_realism.csv)、[来源 uptake](unmodeled_uptake.csv)。
- [配对检查](subdiagnostic/paired_checks.json)：相同 source IDs、核种子、捕获概率和 unmodeled counts；20 份结果均核对 unique ownership 与 core-cell 范围。
- [逐方法检查](subdiagnostic/all_method_checks.csv) 同时记录原生过滤器丢弃的 core nuclear bins；不事后补回以提高分数。
- A_current/benchmark 与 B_library_offset/benchmark 各自保留数据、配置、命令、逐 cell 评价、核种子缺失检查及源码指纹。
'''
    (root/'report.md').write_text(report)
    print(report)


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Package reviewed regional-depth evidence for the shared HTML report renderer."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import sys

import numpy as np
import pandas as pd
from scipy import sparse

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.build_source_separated_pseudo import dump
from hd_cell_rl.experiment_protocol import write_experiment_manifest


def records(frame):
    return json.loads(frame.to_json(orient='records'))


def summarize(out):
    verdict=json.loads((out/'acceptance.json').read_text())
    if not verdict['complete']:raise ValueError('experiment is not complete')
    metrics=pd.read_csv(out/'multiscale_metrics.csv');pred=pd.read_csv(out/'depth_predictions.csv')
    fits=json.loads((out/'fitted_models.json').read_text());features=pd.read_csv(out/'block_features_and_depth.csv')
    diagnostics=json.loads((out/'capture_diagnostics.json').read_text())
    checks=pd.DataFrame([dict(**c,n_failed=len(c['failed_checks'])) for c in verdict['checks']])
    checks['generated']=checks.failed_checks.map(lambda x:'unsafe_depth_extrapolation' not in x)
    checks['failed_checks']=checks.failed_checks.map(lambda x:'; '.join(x))
    checks.to_csv(out/'acceptance_table.csv',index=False)
    summary_rows=[]
    for (fold,patch,variant),g in checks.groupby(['fold','patch_id','variant']):
        m=metrics[(metrics.fold==fold)&(metrics.patch_id==patch)&(metrics.variant==variant)]
        row=dict(fold=fold,patch_id=patch,variant=variant,n_seeds=len(g),generated=int(g.generated.sum()),
                 passed=int(g.passed.sum()),mean_failed_checks=float(g.n_failed.mean()))
        for grain,scale,key in [('spatial_block',2,'bin2'),('spatial_block',8,'bin8'),
                                ('spatial_block',16,'bin16'),('binary_cell_region',0,'cell_region')]:
            z=m[(m.grain==grain)&(m.scale_um==scale)]
            for col in ['counts_mean_ratio','genes_mean_ratio','counts_ks','genes_ks','zero_fraction',
                        'real_zero_fraction','counts_cv','real_counts_cv','counts_median','real_counts_median',
                        'genes_median','real_genes_median','sim_neighbor_depth_corr','real_neighbor_depth_corr']:
                row[key+'_'+col]=float(z[col].mean()) if len(z) and col in z else np.nan
        summary_rows.append(row)
    summary=pd.DataFrame(summary_rows);summary.to_csv(out/'per_patch_summary.csv',index=False)
    support=[]
    for f in fits:
        if f['kind']!='composition':continue
        model=f['model'];cal=features[features.patch_id.isin(f['calibration_patches'])&features.n_bins.eq(64)]
        for p in f['audit_patches']:
            audit=features[features.patch_id.eq(p)&features.n_bins.eq(64)]
            for name,beta,mu,sd in zip(model['feature_names'],model['coefficient'],model['feature_mean'],model['feature_scale']):
                lo,hi=cal[name].min(),cal[name].max()
                support.append(dict(fold=f['fold'],patch_id=p,feature=name,cal_min=float(lo),cal_max=float(hi),
                    audit_min=float(audit[name].min()),audit_max=float(audit[name].max()),
                    fraction_outside_cal_range=float(((audit[name]<lo-1e-10)|(audit[name]>hi+1e-10)).mean()),
                    coefficient=float(beta),mean_log_depth_contribution=float((audit[name].mean()-mu)/sd*beta)))
    support=pd.DataFrame(support);support.to_csv(out/'subdiagnostic/feature_support.csv',index=False)
    conservation=[]
    for file in sorted((out/'realizations').glob('*/*/observed.npz')):
        d=file.parent;observed=sparse.load_npz(file)
        modeled=sparse.load_npz(d/'modeled_cell.npz');unknown=sparse.load_npz(d/'unmodeled_biological.npz')
        technical=sparse.load_npz(d/'technical_background.npz')
        mt=sparse.load_npz(d/'modeled_source_bin_counts.npz');ut=sparse.load_npz(d/'unmodeled_source_bin_counts.npz')
        passed=((observed-modeled-unknown-technical).nnz==0 and technical.nnz==0 and
                np.array_equal(np.asarray(mt.sum(0)).ravel(),np.asarray(modeled.sum(1)).ravel()) and
                np.array_equal(np.asarray(ut.sum(0)).ravel(),np.asarray(unknown.sum(1)).ravel()))
        if not passed:raise AssertionError('source trace mismatch: '+str(d))
        conservation.append(dict(fold=d.parent.name,variant=d.name,passed=True,
            modeled_counts=int(modeled.sum()),unmodeled_counts=int(unknown.sum()),observed_counts=int(observed.sum())))
    dump(out/'subdiagnostic/source_conservation.json',conservation)
    failures=Counter(check for c in verdict['checks'] for check in c['failed_checks'])
    lopo=pred[pred.fold.str.startswith('leave_out')&pred.partition.eq('audit')].copy()
    depth_pass={kind:int(((g.predicted_to_real_ratio>=.8)&(g.predicted_to_real_ratio<=1.25)).sum()) for kind,g in lopo.groupby('kind')}
    result=dict(complete=True,full_acceptance_passes=int(checks.passed.sum()),patch_variant_seed_checks=len(checks),
        generated_patch_checks=int(checks.generated.sum()),rejected_patch_checks=int((~checks.generated).sum()),
        attempted_realizations=len(diagnostics),generated_realizations=sum('modeled' in d for d in diagnostics),
        retained_first_seed_realizations=len(conservation),source_conservation_passed=True,
        independent_sections=1,existing_patches=4,lopo_predictive_mean_ratio_passes_out_of_4=depth_pass,
        failure_counts=dict(failures),model_promoted=False,EM_RL_changed=False,
        caveat='seeds and repeated split views are not independent tissues; no technical efficiency identified')
    dump(out/'summary.json',result)
    return result,pred,summary,checks,support


def build_artifact(out,result,pred,summary,checks,support):
    title='区域深度与空间分块校准：开发实验报告'
    now=datetime.now(timezone.utc).isoformat();audit=pred[pred.partition.eq('audit')]
    lopo=audit[audit.fold.str.startswith('leave_out')].copy()
    chart=[]
    for p,g in lopo.groupby('patch_id'):
        row=g.iloc[0];chart.append(dict(patch=p.replace('patch_',''),model='真实 HD',counts_per_bin=float(row.real_mean_counts)))
        for _,r in g[g.kind.isin(['global','geometry'])].iterrows():
            chart.append(dict(patch=p.replace('patch_',''),model={'global':'全局均值','geometry':'几何条件均值'}[r.kind],counts_per_bin=float(r.predicted_mean_counts)))
    sources=[dict(id='predictions',label='冻结参数后的逐 patch 深度预测',path='depth_predictions.csv',
                  query={'description':'Counts per unique physical 2 um bin; each leave-one-patch-out fit excludes its evaluation patch.'}),
             dict(id='generated',label='三随机种子生成后的逐 patch 多尺度验收',path='per_patch_summary.csv',
                  query={'description':'Seed-averaged metric ratios, not pooled tissue replicates; unsafe extrapolations remain missing and failed.'}),
             dict(id='checks',label='预先冻结阈值的完整验收记录',path='acceptance_table.csv'),
             dict(id='support',label='完整 16 um blocks 的特征支持范围',path='subdiagnostic/feature_support.csv')]
    blocks=[];tables=[];datasets={};connection=sqlite3.connect(':memory:')
    def reviewed_query(id,frame,source):
        # The shared renderer requires executed SQL provenance. Project the
        # reviewed local tables through SQLite; do not pretend a warehouse was queried.
        frame.to_sql(id,connection,index=False,if_exists='fail')
        sql='SELECT * FROM "'+id+'"'
        datasets[id]=records(pd.read_sql_query(sql,connection))
        original=next(s for s in sources if s['id']==source)
        query_source=dict(original,id=id+'_source',query={'engine':'sqlite','sql':sql,
            'description':'Executed in-memory SQLite projection of the reviewed local CSV-derived table; no external warehouse.'})
        sources.append(query_source)
        return query_source['id']
    def text(section,body):blocks.append(dict(id=section,type='markdown',body=body))
    def table(id,title,frame,columns,source):
        query_source=reviewed_query(id,frame,source)
        tables.append(dict(id=id,title=title,dataset=id,sourceId=query_source,
            defaultSort={'field':columns[0][0],'direction':'asc'},
            columns=[dict(field=f,label=l,type='text' if frame[f].dtype=='object' else 'number') for f,l in columns]))
        blocks.append(dict(id=id+'_block',type='table',tableId=id))
    text('title','# '+title)
    text('summary',f"## 结论\n\n本轮完成 {result['attempted_realizations']} 个生成尝试，{result['generated_realizations']} 个实际生成，其余因极端外推被拒绝。按 patch × 设置 × seed 记录的 {result['patch_variant_seed_checks']} 项联合验收中，{result['full_acceptance_passes']} 项通过；其中 {result['rejected_patch_checks']} 项为未生成的外推失败，不是测得的质量指标。\n\n区域建模还没有把 pseudo 变成通过验收的真实 HD 替代物。不能据此推广新的生成器，也不能把残余深度变化解释为已测得的 capture efficiency。本轮不选择 EM/RL weights、不训练正式 PPO、不修改 shape prior 或覆盖旧数据。")
    text('definitions','## 读表前的定义\n\nCounts / UMI 指统一 18,132-gene panel 上的观测计数；2 µm 单位是唯一物理 barcode。预测深度是模型给出的区域条件均值，生成深度则经过 RNA 来源、空间分布和随机采样，二者分开评价。比值均为 simulated 或 predicted 除以 real，1 表示均值一致，但不保证分布一致。\n\nCell-region 只是在同一组 synthetic masks 上分别聚合真实和模拟表达，并不是真实细胞边界 GT；binary regions 可共用混合 bin，fractional regions 按覆盖率加权。三个随机种子不是三个组织重复。')
    text('predictive','## 留一 patch 的条件均值预测\n\n下图比较独立留出的每个开发区域，四个 patch 来自同一个 section。全局与几何模型的深度误差仍明显；composition 模型因外推幅度过大单独列入下表，避免压扁其余比较。几何模型没有使用测试 patch 的 counts 来缩放。')
    chart_source=reviewed_query('depth_chart',pd.DataFrame(chart),'predictions')
    charts=[dict(id='depth',title='留一 patch 的真实与预测深度',type='bar',dataset='depth_chart',sourceId=chart_source,
        encodings={'x':{'field':'patch','type':'nominal','label':'Patch'},
                   'y':{'field':'counts_per_bin','type':'quantitative','label':'Counts per 2 µm bin'},
                   'color':{'field':'model','type':'nominal','label':'模型'}},
        options={'grouping':'grouped','showLegend':True})]
    blocks.append(dict(id='depth_plot',type='chart',chartId='depth'))
    table('predictions_table','全部 audit 条件均值：原 2+2 与留一 patch',audit,
          [('fold','Split'),('patch_id','Patch'),('kind','模型'),('real_mean_counts','真实 counts/bin'),
           ('predicted_mean_counts','预测 counts/bin'),('predicted_to_real_ratio','预测/真实'),
           ('n_complete_blocks','完整 16 µm blocks')],'predictions')
    text('extrapolation','## 失败机制：组成超出校准范围\n\n组成模型使用带正则的 log-link 条件均值。Stem-rich 留出区域的类型密度超出了校准范围，指数映射放大了误差。下表显示组成变量的训练范围、留出范围和对 log-depth 的贡献；这是该模型的外推诊断，不是证明 Stem 细胞或技术 capture 本身异常。\n\n生成器对需要超过 64 倍潜在计数 reservoir 的设置明确拒绝，并记录为失败，没有根据验证区真实深度裁剪。这个上限是计算保护，不是生物捕获率。')
    stem=support[support.feature.eq('type_density:Stem')]
    table('stem_support','Stem 核密度的校准与留出支持范围',stem,
          [('fold','Split'),('patch_id','Patch'),('cal_max','校准最大值'),('audit_max','留出最大值'),
           ('fraction_outside_cal_range','超范围 blocks 比例'),('mean_log_depth_contribution','平均 log-depth 贡献')],'support')
    text('generation','## 生成后的多尺度结果\n\n以下每行是同一 split、patch、设置的三个 seed；缺失指标表示因外推未生成，不能按零误差处理。这里的均值比之外，还联合检查 KS 分布差异、零计数比例、邻接深度相关和 cell-region CV。表格按 patch 保留差异，没有把四个区域混成一个好看的平均值。\n\nuniform_budget 保留原来源模型的空间分布，并非把每个 bin 设成相同 counts。其他设置使用平滑条件均值、可选的独立空间场和 patch 层随机偏移；这些场不复制真实逐 bin 表达图。')
    table('generated_table','逐 patch 生成与联合验收',summary,
          [('fold','Split'),('patch_id','Patch'),('variant','设置'),('generated','已生成 seeds'),('passed','通过 seeds'),
           ('bin2_counts_mean_ratio','2 µm counts 比'),('bin8_genes_mean_ratio','8 µm genes 比'),
           ('bin16_genes_mean_ratio','16 µm genes 比'),('cell_region_counts_mean_ratio','Cell-region counts 比'),
           ('cell_region_genes_mean_ratio','Cell-region genes 比')],'generated')
    text('checks_note','## 验收失败项\n\nCounts/genes 均值比须在 0.8–1.25，KS ≤ 0.2，零率绝对误差 ≤ 0.1，邻接 Spearman 差 ≤ 0.2，binary cell-region counts CV 比在 0.67–1.5。沿用前轮阈值，没有看到本轮结果后放宽。原 2+2 和留一 patch 复用同一数据，不合并计算独立样本显著性。')
    table('checks_table','全部验收记录',checks,[('fold','Split'),('patch_id','Patch'),('variant','设置'),
          ('seed','Seed'),('generated','已生成'),('passed','通过'),('failed_checks','失败项')],'checks')
    text('methods','## 方法与来源隔离\n\n使用 10,827 个唯一 bins、4 个既有开发 contexts；43 个 core cell-regions 用于区域聚合诊断，所有 context bins 用于空间深度诊断。原 2+2 加四次 leave-one-patch-out，只有校准区完整 16 µm blocks 用来估计均值模型、标准化、空间残差尺度。\n\n核密度 Gaussian bandwidth 16 µm，模型预测向 bins 平滑 8 µm，ridge 0.1；残差带宽在 4/8/16/32 µm 中按校准残差相关选择。未建模来源中心从 masks 外空间支撑抽取，类型参考邻近已建模细胞，表达来自 generator-side scRNA 而非真实 bin replay；σ=8 µm 的来源 kernel 是假设。Technical background 固定为零，不等于真实组织没有技术背景。\n\n已建模细胞保留现有核/胞质/膜定位。校准区 cell-region counts 密度给出随机来源 library budgets；区域 multiplier 通过潜在 reservoir 的 binomial thinning 同时作用于两类来源。来源贡献可追溯，并检查精确整数计数守恒。')
    text('limitations','## 局限与稳健性\n\n这些区域之前已经看过，属于开发证据，不是全新 held-out donor。类型标签可能源自真实表达 annotation，因此 composition 对照不能称为纯形态学或完全独立标签预测。空间特征仅基于已加载的核 context；未建模核与边缘截断可能影响密度估计。\n\n残差场参数在两到三个校准 patch 内拟合，不能可靠识别技术与生物来源，也可能低估不确定性。独立随机场可用于分布模拟，但不负责预测真实热点的位置。主验收沿用旧多尺度实现：相同裁剪支撑的边缘不完整 blocks 也参与，邻接相关使用水平相邻对；这可能影响空间相关诊断。核定位、胞质 kernel、scRNA 基因分布、细胞间变异和真实 section 的支持覆盖都仍可能造成不匹配。')
    text('next','## 下一步与未解决问题\n\n先补足跨区域类型密度的校准支持，尤其是 Stem-rich 区域；新数据应另留最终验证 section，不能再把当前四个区域称为从未查看的 test。可测试带收缩的非负可加 RNA 来源强度，作为对当前指数外推机制的明确对照，而非继续调 EM/RL weights。\n\n在固定 counts 条件下单独检查 detected genes、gene abundance 和 cell-region CV，区分单纯深度问题与 scRNA-to-HD 表达分布问题。随机空间场是否改善这些分布，必须按 patch 对照报告，不能以某一次种子改善或工具 IoU 降低作为通过标准。本轮没有生成器被自动晋升为正式 benchmark 输入。')
    artifact={'surface':'report','manifest':{'version':1,'surface':'report','title':title,'generatedAt':now,
        'sources':sources,'charts':charts,'tables':tables,'blocks':blocks},
        'snapshot':{'version':1,'status':'ready','generatedAt':now,'datasets':datasets},'sources':sources}
    dump(out/'artifact.json',artifact)
    connection.close()
    # A compact portable analytical companion, requiring no access to original raw H5.
    cells=[{'cell_type':'markdown','metadata':{},'source':['# Regional depth audit companion\n',
        'Open with this notebook directory as working directory. Reuses saved reviewed CSVs; no generator or inference training.\n']},
        {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],
         'source':['import json\n','from pathlib import Path\n','import pandas as pd\n',
                   "p = Path('.')\n","summary = json.loads((p/'summary.json').read_text())\n","summary\n"]},
        {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],
         'source':["pred = pd.read_csv(p/'depth_predictions.csv')\n",
                   "pred.query(\"partition == 'audit'\")[['fold','patch_id','kind','predicted_to_real_ratio']]\n"]},
        {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],
         'source':["checks = pd.read_csv(p/'acceptance_table.csv')\n",
                   "assert len(checks) == summary['patch_variant_seed_checks']\n",
                   "checks.groupby(['variant'])[['generated','passed']].sum()\n"]},
        {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],
         'source':["metrics = pd.read_csv(p/'multiscale_metrics.csv')\n",
                   "metrics.groupby(['fold','patch_id','variant','grain','scale_um'])[['counts_mean_ratio','genes_mean_ratio','counts_ks','genes_ks']].mean()\n"]},
        {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],
         'source':["support = pd.read_csv(p/'subdiagnostic/feature_support.csv')\n",
                   "support.query(\"feature == 'type_density:Stem'\")\n"]}]
    import nbformat
    def md(body):return nbformat.v4.new_markdown_cell(body)
    notebook_cells=[md('# Regional depth audit companion'),
        md(f"## tl;dr\n\n{result['full_acceptance_passes']} of {result['patch_variant_seed_checks']} patch-setting-seed checks passed. "
           f"{result['rejected_patch_checks']} checks were rejected before generation for unsafe extrapolation. No model is promoted."),
        md('## Context & Methods\n\nFour previously inspected contexts from one colorectal section. Original 2+2 and four leave-one-patch-out views overlap. '
           'Seeds measure simulation variation, not independent tissues.\n\n### Key Assumptions\n\n'
           'Conditional type labels may be expression-derived. Cell-region masks are synthetic geometry, not real-cell GT. '
           'The reservoir is a sampling device, not identified capture efficiency.\n\n'
           'Use this notebook directory as working directory. Requires pandas, nbformat and an IPython kernel; only saved CSV/JSON inputs are loaded.'),
        md('## Data\n\nThe local experiment manifest records input fingerprints, source bytes, configuration and splits.'),
        nbformat.from_dict(cells[1]),md('## Results\n\n### 1. Frozen mean predictions'),nbformat.from_dict(cells[2]),
        md('### 2. Full acceptance and extrapolation checks'),nbformat.from_dict(cells[3]),
        md('### 3. Per-patch multiscale metric ratios'),nbformat.from_dict(cells[4]),
        md('### 4. Stem density extrapolation'),nbformat.from_dict(cells[5]),
        md('## Takeaways\n\nRegionalizing the depth model is not sufficient to pass the declared realism criteria. '
           'Keep EM/RL fixed; inspect calibration support and count-conditioned gene diversity before promoting a new generator. '
           'A failed generative hypothesis is not evidence that technical capture alone caused the observed variation.')]
    for i,c in enumerate(notebook_cells):c['id']=f'regional-{i:02d}'
    notebook=nbformat.v4.new_notebook(cells=notebook_cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
    nbformat.validate(notebook);nbformat.write(notebook,out/'analysis.ipynb')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--experiment-dir',type=Path,required=True)
    p.add_argument('--execute-notebook',action='store_true')
    p.add_argument('--archive-delivery',action='store_true',help='Archive final source and existing report without regenerating it.')
    args=p.parse_args();out=args.experiment_dir.resolve()
    if args.archive_delivery:
        manifest=json.loads((out/'experiment_manifest.json').read_text())
        inputs={name:out/name for name in ['experiment_manifest.json','summary.json','depth_predictions.csv',
            'multiscale_metrics.csv','cell_region_metrics.csv','acceptance_table.csv','artifact.json','report.html',
            'analysis.ipynb','subdiagnostic/validation.md','subdiagnostic/source_conservation.json',
            'subdiagnostic/create_colorectal_pseudo_hd.py','subdiagnostic/gene_localization.py']}
        path=write_experiment_manifest(out.parent/'delivery',config=manifest['resolved_config'],inputs=inputs,
            protocol={'role':'final source and analytical artifacts; original executed source remains in experiment archive',
                'model_promoted':False,'external_localization_source':'fingerprinted copies in experiment/subdiagnostic',
                'browser_QA':'structural only; no installed Chromium; no browser downloaded',
                'no_formal_EM_RL_training':True,'no_new_test_regions':True})
        print(path);return
    result,pred,summary,checks,support=summarize(out);build_artifact(out,result,pred,summary,checks,support)
    if args.execute_notebook:
        import nbformat
        from nbclient import NotebookClient
        notebook=nbformat.read(out/'analysis.ipynb',as_version=4)
        client=NotebookClient(notebook,timeout=120,kernel_name='python3',resources={'metadata':{'path':str(out)}})
        client.km=client.create_kernel_manager()
        client.km.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
        client.execute();nbformat.validate(notebook);nbformat.write(notebook,out/'analysis.ipynb')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()

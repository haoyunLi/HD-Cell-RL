#!/usr/bin/env python3
"""Package measured remediation results and a recoverable source snapshot."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from hd_cell_rl.experiment_protocol import write_experiment_manifest
from hd_cell_rl.patch_assignment import validate_unique_ownership


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--verified-cpu-tests',type=int,required=True)
    args=p.parse_args();out=args.run_dir
    if (out/'report.md').exists():raise FileExistsError('do not overwrite a completed report')
    em=out/'em_context_regression_final';pseudo=out/'source_separated_pseudo';stats=out/'cluster_statistics'
    assignments=pd.read_csv(em/'assignments.csv',dtype={'cell_id':str,'barcode':str})
    validate_unique_ownership(assignments.to_dict('records'))
    regression=pd.read_csv(em/'frozen_code_regression.csv');sensitivity=pd.read_csv(em/'context_sensitivity.csv')
    reliability=pd.read_csv(em/'cell_reliability.csv')
    acceptance=json.loads((pseudo/'acceptance.json').read_text())
    metrics=pd.read_csv(pseudo/'multiscale_metrics.csv')
    example=metrics[(metrics.variant=='components_sigma8')&(metrics.seed==20260909)]
    means=example[example.scale_um==2].set_index('partition')
    summary={'cpu_tests_passed':args.verified_cpu_tests,'formal_PPO_started':False,'reward_formula_changed':False,
        'shape_prior_changed':False,'old_runs_overwritten':False,'fresh_test_regions_consumed':False,
        'n_unique_em_assignment_barcodes':len(assignments),'frozen_changed_cell_exports':int((regression.changed_bins>0).sum()),
        'frozen_per_cell_symmetric_difference_sum':int(regression.changed_bins.sum()),
        'frozen_old_new_assignment_iou':float(regression.old_new_assignment_iou.mean()),
        'normalization_only_max_common_owner_change':float(sensitivity.normalization_only_hard_owner_changed_fraction.max()),
        'common_cell_nuclear_counts_unchanged':bool(np.allclose(reliability.nuclear_umi_before,reliability.nuclear_umi_after)),
        'n_generator_repeats':len(acceptance['checks']),'n_generator_repeats_passed':sum(x['passed'] for x in acceptance['checks']),
        'generator_accepted_for_production':False,
        'example_calibration_mean_counts':float(means.loc['calibration','counts_mean']),
        'real_calibration_mean_counts':float(means.loc['calibration','real_counts_mean']),
        'example_audit_mean_counts':float(means.loc['audit','counts_mean']),
        'real_audit_mean_counts':float(means.loc['audit','real_counts_mean'])}
    report=f'''# 项目修复与生成器实验

## 结论

工程改动已通过 {args.verified_cpu_tests} 项 CPU 测试。生成器完成 7 组 × 3 个 seed，
但没有一组通过完整多尺度验收。没有替换正式数据，没有启动 PPO，没有改 reward 或 shape prior。

## 已完成

1. EM-only、distance-only 和 checkpoint 评估共用跨 patch ownership 合并；
   检查唯一 barcode、核锁定和输入顺序，保存冲突，不用字典静默覆盖。
2. 新增明确命名的冻结 EM 配置，并归档可恢复的源码与完整实验 manifest。
3. 当前代码下冻结参数回归和 context/prior 敏感性实验。
4. 五方法历史结果改用完整 patch 配对 bootstrap；旧报告保留。
5. 可选的独立 validation checkpoint 选择与正式训练准入检查；旧 development 行为保留。
6. 三来源生成器及 counts/genes/多尺度验收流程，完整保存 modeled、unmodeled、technical 三类贡献。

## 回归与 context

原始四个 patch 在冻结的 15 次迭代上限内均收敛。{summary['frozen_changed_cell_exports']}/43 个 cell 导出有变化，
每 cell barcode 对称差合计 {summary['frozen_per_cell_symmetric_difference_sum']}；转移可能在两个 cell 各计一次。
新旧 assignment 的平均 per-cell IoU 为 {summary['frozen_old_new_assignment_iou']:.6f}，这是版本一致性，不是 GT 准确度。
最终 {len(assignments)} 个输出 barcode 无重复。

共同 cell 的核 counts 保持不变；仅改变 patch 归一化尺度，单个条件最多
{100*summary['normalization_only_max_common_owner_change']:.3f}% 共同 bins 改变 top1。
没有直接改生产 prior。缩小 outer 区域会截掉一部分可导出的 cell bins，
因此 `merged_context_sensitivity.csv` 单独报告 support 变化，不能把全部差异归因于 prior。
个别缩小 context 的控制在 15 次内未收敛，不能视为稳定固定点。

## Pseudo 数据

`components_sigma8`（预先声明的 8 µm 空间核，seed 20260909）在 calibration 上
匹配到 {summary['example_calibration_mean_counts']:.3f} counts/bin，真实值为 {summary['real_calibration_mean_counts']:.3f}；
但 audit 上只有 {summary['example_audit_mean_counts']:.3f}，真实值为 {summary['real_audit_mean_counts']:.3f}。
该例不是选出的“最佳公式”。它说明填补 mask 外表达、匹配 calibration 总量，仍不能保证跨区域真实性。

新模型没有复制真实 per-bin expression。未建模来源由附近 cell-type 组成、generator-side scRNA profile
和独立空间核生成，可以进入已占用 bins；技术背景独立记录，1% 仅为假设控制。
真实 mask 外表达不是已知的纯 ambient，也不能证明全部来自未建模生物来源。

目前的 cell-region 指标用同一组模拟 mask 聚合 real/pseudo；它们不是真实 cell GT。
二值区域可能共用 mixed bins，fractional 区域指标和源 counts 分开存储。
仅用这两个 calibration 区域的总体预算仍未描述好区域间深度变化。

## 文件入口

- [工程协议与全部改动说明](../../../docs/PROJECT_REMEDIATION_20260909.md)
- [EM context 与归一化对照](em_context_regression_final/context_sensitivity.csv)
- [EM 合并后的结果](em_context_regression_final/merged_context_sensitivity.csv)
- [Patch bootstrap 报告](cluster_statistics/report.md)
- [生成器报告](source_separated_pseudo/report.md)
- [多尺度指标](source_separated_pseudo/multiscale_metrics.csv)
- [三来源 counts](source_separated_pseudo/component_counts.csv)
- [数据验收失败项](source_separated_pseudo/acceptance.json)

## 下一步

先研究区域级 capture 异质性，采用空间分块 calibration 检查；继续联合验收 counts、genes、
空间相关和 cell-region 变异，不以工具分数下降为目标。不消耗新 held-out 区域，不校准 entropy，不训练 PPO。
源码快照保存当前工作树源码（含未跟踪文件），不是 Git 提交，也不复制大数据文件。
各子实验保留各自源码快照；根 manifest 是本次交付版本。
'''
    (out/'report.md').write_text(report)
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    write_experiment_manifest(out,config={'protocol_document':'docs/PROJECT_REMEDIATION_20260909.md'},
        inputs={'em_config':ROOT/'configs/em_assignment.frozen_baseline.yaml',
            'generator_config':ROOT/'configs/pseudo_components.calibration.yaml',
            'em_manifest':em/'experiment_manifest.json','generator_manifest':pseudo/'experiment_manifest.json',
            'cluster_statistics':stats/'summary.json','acceptance':pseudo/'acceptance.json'},
        protocol=summary)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()

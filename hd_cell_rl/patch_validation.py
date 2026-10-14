"""Opt-in validation selection and fail-closed formal-training admission.

GT is consulted after greedy policy actions, only on declared validation data.
No reward term, observation feature, or policy update uses these labels.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .patch_assignment import PatchOwnershipMerger, patch_assignments_for_core_cells
from .patch_dataset import PatchDataset
from .patch_rollout import collect_patch_trajectories_batched
from .provenance import file_fingerprint


def validate_split_indexes(train, validation):
    """Reject overlapping IDs, context cells and rectangles before validation."""
    if train.empty or validation.empty:
        raise ValueError('training and validation indexes must be nonempty')
    if set(train.patch_id.astype(str)) & set(validation.patch_id.astype(str)):
        raise ValueError('training/validation patch IDs overlap')
    def cells(frame):
        return {str(c) for column in ['patch_cell_ids', 'core_cell_ids', 'margin_cell_ids']
                if column in frame for v in frame[column] for c in json.loads(v)}
    if cells(train) & cells(validation):
        raise ValueError('training/validation context cells overlap')
    prefix = 'context' if 'context_x_min' in train and 'context_x_min' in validation else 'outer'
    columns = [f'{prefix}_{axis}_{bound}' for axis in ['x','y'] for bound in ['min','max']]
    if not set(columns).issubset(train) or not set(columns).issubset(validation):
        raise ValueError('split validation requires explicit context/outer rectangles')
    for a in train[columns].to_numpy(float):
        for b in validation[columns].to_numpy(float):
            if not np.isfinite(np.r_[a,b]).all():
                raise ValueError('non-finite split bounds')
            if any(v[0] > v[1] or v[2] > v[3] for v in [a, b]):
                raise ValueError('inverted split bounds')
            if a[0] <= b[1] and b[0] <= a[1] and a[2] <= b[3] and b[2] <= a[3]:
                raise ValueError('training/validation context rectangles overlap')


def training_evidence_identity(config, settings):
    """Bind admission to resolved formulas and data, not filenames alone."""
    from dataclasses import asdict
    from .provenance import source_provenance
    data = {'base_config': config.to_serializable_dict(), 'patch_settings': asdict(settings),
            'inputs': {k: file_fingerprint(v) for k,v in {
                'episodes': config.episodes_index_path, 'reference': config.reference_path,
                'nuclei': config.nuclei_path, 'patches': settings.patches_index_path}.items()}}
    data['source'] = source_provenance()
    return hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()


def require_training_admission(config, settings, run, validation):
    if str(run.get('purpose', 'development')) != 'formal':
        return
    if not validation.get('enabled', False):
        raise ValueError('formal PPO requires independent validation checkpoint selection')
    path = run.get('admission_manifest')
    if not path:
        raise ValueError('formal PPO paused: missing data-realism/objective admission evidence')
    evidence = json.loads(Path(path).read_text())
    if evidence.get('training_evidence_identity') != training_evidence_identity(config, settings):
        raise ValueError('formal PPO admission is stale or belongs to another data/formula protocol')
    if evidence.get('data_realism_passed') is not True:
        raise ValueError('formal PPO paused: data realism has not passed')
    realism_path = Path(evidence['data_realism_report'])
    if hashlib.sha256(realism_path.read_bytes()).hexdigest() != evidence.get('data_realism_sha256'):
        raise ValueError('data realism evidence changed')
    if json.loads(realism_path.read_text()).get('passed') is not True:
        raise ValueError('formal PPO paused: data realism report failed')
    objective_path = Path(evidence['objective_per_patch_csv'])
    if hashlib.sha256(objective_path.read_bytes()).hexdigest() != evidence.get('objective_per_patch_sha256'):
        raise ValueError('objective validation evidence changed')
    d = pd.read_csv(objective_path)
    required = {'patch_id', 'initial_iou', 'final_iou'}
    if not required.issubset(d) or d.patch_id.duplicated().any() or len(d) < 2:
        raise ValueError('objective evidence needs at least two unique validation patches')
    expected = set(pd.read_csv(validation['patches_index_path']).patch_id.astype(str))
    if set(d.patch_id.astype(str)) != expected:
        raise ValueError('objective evidence must use the declared validation patches')
    delta = d.final_iou.to_numpy(float) - d.initial_iou.to_numpy(float)
    if not d[['initial_iou', 'final_iou']].apply(lambda s: s.between(0, 1)).all().all():
        raise ValueError('invalid objective IoU evidence')
    if not np.isfinite(delta).all() or not (delta.mean() > 0 and np.median(delta) > 0 and np.mean(delta > 0) > .5):
        raise ValueError('formal PPO paused: exact-objective validation failed improvement gates')


class PatchValidationEvaluator:
    def __init__(self, *, config, settings, validation):
        if validation.get('role') != 'validation':
            raise ValueError('checkpoint selection must explicitly declare role: validation, never test')
        self.every = int(validation.get('every_updates', 10))
        if self.every < 1:
            raise ValueError('validation.every_updates must be positive')
        self.seed = int(validation.get('seed', 20260909))
        frame = pd.read_csv(validation['patches_index_path']).sort_values('patch_id')
        validate_split_indexes(pd.read_csv(settings.patches_index_path), frame)
        vsettings = replace(settings, patches_index_path=Path(validation['patches_index_path']),
            cache_patch_contexts=False, em_assignment=replace(settings.em_assignment, save_artifacts=False, debug_patch_ids=()))
        dataset = PatchDataset(base_config=config, settings=vsettings, rng=np.random.default_rng(self.seed))
        try:
            self.contexts = [dataset.load_patch_context(r) for r in frame.itertuples(index=False)]
            # Resolved MaxDis may extend beyond legacy index rectangles. Check
            # the actually loaded validation cells against the train halo too.
            actual_validation_ids = {str(c.cell_id) for ctx in self.contexts if ctx is not None for c in ctx.cells}
            for row in pd.read_csv(settings.patches_index_path).itertuples(index=False):
                expansion = dataset._resolve_candidate_max_distance(row) if settings.em_assignment.enabled else 0.
                resolved = dataset._cell_dataset.candidate_cell_ids_for_expanded_bounds(
                    x_min=row.outer_x_min, x_max=row.outer_x_max, y_min=row.outer_y_min,
                    y_max=row.outer_y_max, expansion_um=expansion)
                if actual_validation_ids & set(map(str,resolved)):
                    raise ValueError('resolved train MaxDis context overlaps validation cells')
        finally:
            dataset.close()
        if any(c is None for c in self.contexts):
            raise ValueError('validation context missing; do not silently change the scored subset')
        truth = pd.read_csv(validation['gt_cell_bins_path'], dtype={'cell_id':str, 'barcode':str})
        self.truth = {c:set(g.barcode) for c,g in truth.groupby('cell_id')}
        self.cell_ids = sorted({str(c) for ctx in self.contexts for c in ctx.core_cell_ids})
        if not set(self.cell_ids).issubset(self.truth):
            raise ValueError('validation GT must use the same physical cell IDs and cover every core cell')
        self.backend = settings.rollout_backend

    def evaluate(self, model, device):
        import torch
        from preprocessing.ppo_eval_metrics import compute_spatial_overlap_metrics
        was_training = model.training
        try:
            model.eval()
            devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
            with torch.random.fork_rng(devices=devices):
                torch.random.default_generator.manual_seed(self.seed)
                if devices:
                    torch.cuda.default_generators[devices[0]].manual_seed(self.seed)
                trajectories, _ = collect_patch_trajectories_batched(contexts=self.contexts,
                    model=model, device=device, rng=np.random.default_rng(self.seed),
                    policy_mode='greedy', rollout_backend=self.backend)
        finally:
            model.train(was_training)
        merger = PatchOwnershipMerger()
        for ctx, trajectory in zip(self.contexts, trajectories, strict=True):
            groups = patch_assignments_for_core_cells(context=ctx, final_masks=trajectory.final_masks)
            merger.add_patch(context=ctx, rows=[r for rr in groups.values() for r in rr],
                             score=trajectory.patch_score, target_cell_ids=ctx.core_cell_ids)
        rows, conflicts = merger.finalize()
        pred = {c:set() for c in self.cell_ids}
        for row in rows:
            pred[str(row['cell_id'])].add(str(row['barcode']))
        scores = [compute_spatial_overlap_metrics(pred[c], self.truth[c])['iou'] for c in self.cell_ids]
        return {'mean_core_cell_iou': float(np.mean(scores)), 'n_cells': len(scores),
                'n_patches': len(self.contexts), 'n_merge_conflicts': len(conflicts),
                'selection_basis': 'validation_core_cell_iou_post_update'}

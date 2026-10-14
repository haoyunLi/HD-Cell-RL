from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import yaml

from hd_cell_rl.patch_assignment import PatchOwnershipMerger, validate_unique_ownership
from hd_cell_rl.experiment_protocol import paired_cluster_mean_ci
from hd_cell_rl.em_types import EMAssignmentConfig
from scripts.run_em_weight_sweep import _evaluate_hard_assignments


def context(patch, seeds=()):
    return SimpleNamespace(patch_id=patch, cells=tuple(SimpleNamespace(cell_id=c,
        candidate_bin_ids=(b,), initial_membership_mask=np.array([1])) for b, c in seeds))


class ProjectProtocolTests(unittest.TestCase):
    def test_all_modes_share_unique_order_independent_export(self):
        for scores in [(0., 0.), (1., 2.)]:
            outputs = []
            for reverse in [False, True]:
                m = PatchOwnershipMerger()
                claims = [(context('p1'), [{'barcode':'b', 'cell_id':'A'}], scores[0]),
                          (context('p2'), [{'barcode':'b', 'cell_id':'B'}], scores[1])]
                for ctx, rows, score in claims[::(-1 if reverse else 1)]:
                    m.add_patch(context=ctx, rows=rows, score=score)
                outputs.append(m.finalize())
            self.assertEqual(outputs[0], outputs[1])
            self.assertEqual(len(outputs[0][0]), 1)
            self.assertEqual(len(outputs[0][1]), 2)

    def test_same_cell_patch_tie_uses_stable_id(self):
        results = []
        for patches in [('p2','p1'), ('p1','p2')]:
            m = PatchOwnershipMerger()
            for p in patches:
                m.add_patch(context=context(p), rows=[{'barcode':p, 'cell_id':'A'}])
            results.append(m.finalize()[0])
        self.assertEqual(results[0], results[1])
        self.assertEqual(results[0][0]['barcode'], 'p1')

    def test_empty_selected_snapshot_does_not_keep_stale_ownership(self):
        m = PatchOwnershipMerger()
        m.add_patch(context=context('p2'), rows=[{'barcode':'b','cell_id':'A'}])
        m.add_patch(context=context('p1'), rows=[], target_cell_ids=['A'])
        self.assertEqual(m.finalize()[0], [])

    def test_validation_split_checks_cross_core_margin_and_bounds(self):
        import pandas as pd
        from hd_cell_rl.patch_validation import validate_split_indexes
        def frame(p, core, margin, x):
            return pd.DataFrame([dict(patch_id=p, core_cell_ids=core, margin_cell_ids=margin,
                context_x_min=x, context_x_max=x+1, context_y_min=0, context_y_max=1)])
        train = frame('a', '["A"]', '[]', 0)
        with self.assertRaisesRegex(ValueError, 'cells overlap'):
            validate_split_indexes(train, frame('b','["B"]','["A"]',10))
        with self.assertRaisesRegex(ValueError, 'rectangles overlap'):
            validate_split_indexes(train, frame('b','["B"]','[]',0))
        validate_split_indexes(train, frame('b','["B"]','[]',10))

    def test_formal_training_fails_closed(self):
        from hd_cell_rl.patch_validation import require_training_admission
        with self.assertRaisesRegex(ValueError, 'independent validation'):
            require_training_admission(None, None, {'purpose':'formal'}, {})
        with self.assertRaisesRegex(ValueError, 'missing data-realism'):
            require_training_admission(None, None, {'purpose':'formal'}, {'enabled':True})
        require_training_admission(None, None, {}, {})

    def test_source_snapshot_is_recoverable_and_never_overwritten(self):
        import hashlib, json, tarfile, tempfile
        from hd_cell_rl.experiment_protocol import write_experiment_manifest
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            path=write_experiment_manifest(tmp,config={'frozen':True},inputs={},protocol={'role':'test'})
            manifest=json.loads(path.read_text())
            with tarfile.open(Path(tmp)/manifest['source_snapshot']) as archive:
                data=archive.extractfile('hd_cell_rl/em_assignment.py').read()
                self.assertEqual(data,(root/'hd_cell_rl/em_assignment.py').read_bytes())
                self.assertEqual(hashlib.sha256(data).hexdigest(),manifest['archived_source_sha256']['hd_cell_rl/em_assignment.py'])
            with self.assertRaises(FileExistsError):
                write_experiment_manifest(tmp,config={},inputs={},protocol={})

    def test_selected_nuclear_seed_cannot_silently_disappear(self):
        m=PatchOwnershipMerger()
        m.add_patch(context=context('p', [('seed','A')]),rows=[],target_cell_ids=['A'])
        with self.assertRaisesRegex(ValueError,'seeds missing'):
            m.finalize()

    def test_invalid_nuclear_claim_fails_independent_of_order(self):
        for reverse in [False,True]:
            m=PatchOwnershipMerger()
            claims=[(context('p1',[('b','A')]),[]),(context('p2'),[{'barcode':'b','cell_id':'B'}])]
            for ctx,rows in claims[::(-1 if reverse else 1)]:m.add_patch(context=ctx,rows=rows)
            with self.assertRaisesRegex(ValueError,'nuclear ownership changed'):m.finalize()

    def test_validation_selects_post_update_weights_and_restores_policy_state(self):
        import json, tempfile, torch
        from unittest.mock import patch
        from tests.test_ppo_training import PPOTrainingTests
        from tests.test_em_assignment import _episode
        from hd_cell_rl.patch_types import PatchContext, PatchBounds
        import scripts.run_patch_training as runner
        cell=_episode('A',(0,0),['seed','free'],[(0,0),(2,0)],ll=[[-.1,-2.],[-.2,-1.]],seed=[1,0])
        ctx=PatchContext(patch_id='fixture',cells=(cell,),core_cell_ids=('A',),margin_cell_ids=(),
            outer_bounds=PatchBounds(-1,3,-1,1),core_bounds=PatchBounds(-1,3,-1,1),max_steps=2)
        evaluated=[]
        def evaluate(model, device):
            evaluated.append({k:v.clone() for k,v in model.state_dict().items()})
            from hd_cell_rl.patch_validation import PatchValidationEvaluator
            evaluator=object.__new__(PatchValidationEvaluator)
            evaluator.contexts=[ctx];evaluator.truth={'A':{'seed','free'}}
            evaluator.cell_ids=['A'];evaluator.backend='legacy_cpu';evaluator.seed=12
            state=torch.random.get_rng_state().clone();was_training=model.training
            result=evaluator.evaluate(model,device)
            torch.testing.assert_close(torch.random.get_rng_state(),state)
            self.assertEqual(model.training,was_training)
            self.assertEqual(result['n_cells'],1)
            return {'mean_core_cell_iou':.8 if len(evaluated)==1 else .7}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);base=PPOTrainingTests()._write_minimal_config(root,batch_cells=1,group_relative_enabled=False)
            cfg={'base_ppo_config':str(base),'run':{'name':'validation_fixture','device':'cpu','max_updates':2,'batch_patches':1},
                 'patch_training':{'patches_index_path':str(root/'patches.csv'),'rollout_backend':'legacy_cpu'},
                 'validation':{'enabled':True,'role':'validation'}}
            file=root/'patch.yaml';file.write_text(yaml.safe_dump(cfg))
            with patch.object(runner,'_parse_args',return_value=SimpleNamespace(config=str(file))), \
                 patch.object(runner,'PatchValidationEvaluator',return_value=SimpleNamespace(every=1,evaluate=evaluate)), \
                 patch.object(runner,'PatchDataset',return_value=SimpleNamespace(n_patches=1,close=lambda:None)), \
                 patch.object(runner,'collect_patch_contexts',return_value=[ctx]):
                runner.main()
            run=next((root/'runs').iterdir())
            best=torch.load(run/'checkpoints/best_model.pt',weights_only=True)
            final=torch.load(run/'checkpoints/final_model.pt',weights_only=True)
            self.assertEqual(best['update_index'],1);self.assertEqual(final['update_index'],2)
            self.assertEqual(best['selection_basis'],'validation_core_cell_iou_post_update')
            for key,value in best['model_state_dict'].items():torch.testing.assert_close(value,evaluated[0][key])
            self.assertEqual(json.loads((run/'summary.json').read_text())['best_validation_core_cell_iou'],.8)

    def test_seeds_conflicts_and_duplicate_rows_rejected(self):
        m = PatchOwnershipMerger()
        m.add_patch(context=context('p1', [('b','A')]), rows=[{'barcode':'b','cell_id':'A'}])
        m.add_patch(context=context('p2'), rows=[{'barcode':'b','cell_id':'B'}])
        with self.assertRaises(ValueError):
            m.finalize()
        with self.assertRaises(ValueError):
            validate_unique_ownership([{'barcode':'b','cell_id':'A'}]*2)
        with self.assertRaises(ValueError):
            _evaluate_hard_assignments(assignment_rows=[{'barcode':'b','cell_id':'A'}]*2,
                gt=None, target_cell_ids=('A',))

    def test_cluster_resampling_does_not_treat_cells_as_independent(self):
        lo, hi = paired_cluster_mean_ci(np.r_[np.zeros(100), np.ones(100)],
            np.array(['a']*100+['b']*100), replicates=2000, rng=np.random.default_rng(7))
        self.assertEqual((lo,hi), (0.,1.))
        self.assertTrue(np.isnan(paired_cluster_mean_ci([1,2], ['a','a'], rng=np.random.default_rng(7))[0]))
        with self.assertRaises(ValueError):
            paired_cluster_mean_ci([1,2], ['a',None], rng=np.random.default_rng(7))

    def test_named_frozen_baseline(self):
        p = Path(__file__).resolve().parents[1] / 'configs/em_assignment.frozen_baseline.yaml'
        c = EMAssignmentConfig.from_mapping(yaml.safe_load(p.read_text())['em_assignment'])
        self.assertEqual((c.expression_weight, c.cell_profile_relative_prior_strength, c.cell_profile_update_damping), (1.5,2.,0.))
        self.assertFalse(c.background_enabled)


if __name__ == '__main__':
    unittest.main()

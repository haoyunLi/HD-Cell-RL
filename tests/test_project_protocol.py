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

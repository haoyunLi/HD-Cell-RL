import json
import unittest

import pandas as pd

from scripts.prepare_real_scrna_patch_comparison import select_patches


class ExpandedPatchSelectionTests(unittest.TestCase):
    def fixture(self):
        return pd.DataFrame([
            dict(patch_id=f'p{i:03}', center_x_um=i*300., center_y_um=0.,
                 n_core_cells=6, core_cell_ids=json.dumps([str(i*10+j) for j in range(6)]))
            for i in range(30)])

    def test_sixteen_preserve_four_exclude_known_and_reproducible(self):
        full=self.fixture();original=full.iloc[:4]
        a=select_patches(full,original,excluded_ids=['p004','p005'])
        b=select_patches(full.iloc[::-1],original,excluded_ids=['p004','p005'])
        pd.testing.assert_frame_equal(a,b)
        self.assertEqual(len(a),16)
        self.assertEqual(a.iloc[:4].patch_id.tolist(),original.patch_id.tolist())
        self.assertEqual(a.cohort.value_counts().to_dict(),{'additional12':12,'original4':4})
        self.assertFalse(a.patch_id.isin(['p004','p005']).any())

    def test_nearby_patch_excluded(self):
        full=self.fixture();full.loc[4,'center_x_um']=10.
        selected=select_patches(full,full.iloc[:4])
        self.assertNotIn('p004',selected.patch_id.tolist())

    def test_not_enough_contexts_raises(self):
        full=self.fixture().iloc[:10]
        with self.assertRaisesRegex(ValueError,'insufficient'):
            select_patches(full,full.iloc[:4])


if __name__=='__main__':unittest.main()

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import unittest

spec = importlib.util.spec_from_file_location(
    "scdesign_ab_summary", Path(__file__).resolve().parents[1] / "scripts/summarize_scdesign3_library_size_ab.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TestSummary(unittest.TestCase):
    def test_distribution_and_identity_comparison(self):
        result = module.distribution([1, 2, 3])
        self.assertEqual(result["mean"], 2)
        self.assertEqual(result["variance"], 1)
        self.assertEqual(result["cv"], .5)
        self.assertEqual(result["p50"], 2)
        comparison = module.compare([1, 2, 3], [1, 2, 3])
        self.assertEqual(comparison["cv_ratio"], 1)
        self.assertEqual(comparison["ks"], 0)
        self.assertEqual(comparison["log1p_wasserstein"], 0)


    def test_invalid_distribution(self):
        for values in [[1], [1, np.nan], [-1, 2]]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                module.distribution(values)


    def test_validate_cells(self):
        frame = pd.DataFrame({"cell_id": ["a", "b"], "counts": [4, 5], "detected_genes": [2, 3]})
        module.validate_cells(frame, 2, 4)
        for bad in [frame.assign(cell_id="a"), frame.assign(counts=1), frame.assign(detected_genes=5),
                    frame.assign(counts=1.5), frame.assign(counts=np.inf)]:
            with self.assertRaises(ValueError):
                module.validate_cells(bad, 2, 4)


    def test_constant_correlation_is_undefined(self):
        self.assertTrue(np.isnan(module.corr([1, 1], [2, 3])))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import math
import unittest

import pandas as pd

from preprocessing.build_nuclei_table_from_nuclear_bins import build_nuclei_table


class BuildNucleiTableFromNuclearBinsTests(unittest.TestCase):
    def test_fraction_weighted_center_and_equivalent_radius(self) -> None:
        bins = pd.DataFrame(
            {
                "cell_id": [1, 1, 2],
                "x_um": [0.0, 2.0, 10.0],
                "y_um": [0.0, 0.0, 4.0],
                "pxl_row_in_fullres": [0.0, 0.0, 4.0],
                "pxl_col_in_fullres": [0.0, 2.0, 10.0],
                "nuclear_fraction": [1.0, 0.5, 1.0],
            }
        )
        result = build_nuclei_table(
            bins,
            bin_size_um=2.0,
            microns_per_pixel=1.0,
        )
        first = result.loc[result["cell_id"] == 1].iloc[0]
        self.assertAlmostEqual(float(first["center_x_um"]), 2.0 / 3.0)
        self.assertAlmostEqual(float(first["center_y_um"]), 0.0)
        self.assertAlmostEqual(float(first["nuclear_area_um2"]), 6.0)
        self.assertAlmostEqual(float(first["radius_um"]), math.sqrt(6.0 / math.pi))
        self.assertEqual(int(first["nuclear_bin_count"]), 2)


if __name__ == "__main__":
    unittest.main()

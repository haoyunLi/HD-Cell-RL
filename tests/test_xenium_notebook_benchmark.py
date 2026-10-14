"""Data-interface regression checks for the Xenium notebook benchmark."""

import numpy as np
import pandas as pd
from shapely import Polygon, contains_xy

from scripts.build_xenium_notebook_patch_benchmark import _barcode, _notebook_barcode, _polygons, _raster_owners


def test_notebook_grid_has_reversible_10x_barcode_adapter():
    row = np.array([5, 500], dtype=np.int64)
    col = np.array([12, 3150], dtype=np.int64)
    assert _notebook_barcode(row, col).tolist() == ["2um_00005_00012", "2um_00500_03150"]
    assert _barcode(row, col).tolist() == ["s_002um_00005_00012-1", "s_002um_00500_03150-1"]
    for x, r, c in ((9.01, 5, 12), (999.99, 500, 3150)):
        assert int(np.ceil(x / 2)) == r
        assert int(np.ceil((c * 2 - 1) / 2)) == c


def test_raster_owner_drops_overlap_instead_of_using_iteration_order():
    first = Polygon([(0, 0), (3, 0), (3, 3), (0, 3)])
    second = Polygon([(2, 0), (5, 0), (5, 3), (2, 3)])
    x = np.array([1.0, 2.5, 4.0])
    y = np.array([1.0, 1.0, 1.0])
    owners, collisions = _raster_owners({"a": first, "b": second}, x, y, {"a": 1, "b": 2})
    assert owners.tolist() == [1, 0, 2]
    assert collisions == 1

def test_prime_5k_multiple_nuclear_rings_are_one_physical_cell(tmp_path):
    rings = []
    for label, x0 in ((1, 0), (2, 4)):
        rings.extend({"cell_id": "multi", "label_id": label, "vertex_x": x0 + dx,
                      "vertex_y": dy} for dx, dy in ((0, 0), (2, 0), (2, 2), (0, 2)))
    path = tmp_path / "nuclei.parquet"
    pd.DataFrame(rings).to_parquet(path, index=False)
    polygons = _polygons(path, {"multi"})
    assert set(polygons) == {"multi"}
    assert contains_xy(polygons["multi"], np.array([1., 5., 3.]),
                       np.array([1., 1., 1.])).tolist() == [True, True, False]

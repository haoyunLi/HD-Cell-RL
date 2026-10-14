import pandas as pd
import pytest

from scripts.summarize_gt_area_scale_experiment import boundary, check_paired_sources, f1, outer_ring


def test_ring_and_boundary_for_single_bin():
    cell = {(5, 7)}
    assert boundary(cell) == cell
    assert len(outer_ring(cell)) == 8
    assert cell.isdisjoint(outer_ring(cell))


def test_boundary_f1():
    assert f1({1, 2}, {2, 3}) == 0.5
    assert f1(set(), {1}) == 0.0


def test_paired_source_identity_check():
    base = pd.DataFrame({"cell_id": ["1"], "cell_type": ["T"],
                         "sc_cell_barcode": ["source-1"], "source_donor": ["donor-1"],
                         "x_um": [10.0], "y_um": [20.0]})
    shifted = base.copy()
    shifted.loc[0, "x_um"] = 11.0
    check_paired_sources({1.0: (base, "reference"), 0.8: (base.copy(), "reference")})
    with pytest.raises(ValueError, match="nuclear centers"):
        check_paired_sources({1.0: (base, "reference"), 0.8: (shifted, "reference")})
    with pytest.raises(ValueError, match="reference"):
        check_paired_sources({1.0: (base, "reference"), 0.8: (base, "different")})

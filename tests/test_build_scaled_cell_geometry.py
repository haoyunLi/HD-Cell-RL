import numpy as np
import pytest

from scripts.build_scaled_cell_geometry import scaled_target_areas


def test_scales_only_non_nuclear_area():
    nuclear = np.array([10, 20, 5], dtype=np.int64)
    target = np.array([20, 30, 5], dtype=np.int64)
    assert scaled_target_areas(nuclear, target, 0.6).tolist() == [16, 26, 5]
    assert scaled_target_areas(nuclear, target, 1.0).tolist() == target.tolist()
    assert scaled_target_areas(nuclear, target, 1.4).tolist() == [24, 34, 5]


@pytest.mark.parametrize("scale", [0, -1, float("nan"), float("inf")])
def test_rejects_invalid_scale(scale):
    with pytest.raises(ValueError):
        scaled_target_areas(np.array([10]), np.array([20]), scale)


def test_rejects_target_smaller_than_nucleus():
    with pytest.raises(ValueError):
        scaled_target_areas(np.array([10]), np.array([9]), 0.8)

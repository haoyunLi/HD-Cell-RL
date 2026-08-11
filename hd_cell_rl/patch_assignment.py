"""Convert patch final masks into bin assignment rows."""

from __future__ import annotations

from typing import Any

import numpy as np

from .patch_types import PatchContext


def patch_assignments_for_core_cells(
    *,
    context: PatchContext,
    final_masks: dict[str, np.ndarray],
) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    ctx_by_cell = {ctx.cell_id: ctx for ctx in context.cells}
    for cell_id in context.core_cell_ids:
        ctx = ctx_by_cell.get(cell_id)
        mask = final_masks.get(cell_id)
        if ctx is None or mask is None:
            continue
        rows: list[dict[str, Any]] = []
        nuclear = np.asarray(ctx.initial_membership_mask, dtype=np.uint8) > 0
        xy = np.asarray(ctx.candidate_bin_xy_um, dtype=np.float64)
        for idx in np.flatnonzero(np.asarray(mask, dtype=np.uint8) > 0).tolist():
            barcode = str(ctx.candidate_bin_ids[int(idx)])
            row_col = _parse_square_barcode(barcode)
            array_row = row_col[0] if row_col is not None else int(round(float(xy[idx, 1]) / 2.0))
            array_col = row_col[1] if row_col is not None else int(round(float(xy[idx, 0]) / 2.0))
            rows.append(
                {
                    "barcode": barcode,
                    "cell_id": str(cell_id),
                    "array_row": int(array_row),
                    "array_col": int(array_col),
                    "x_um": float(xy[idx, 0]),
                    "y_um": float(xy[idx, 1]),
                    "is_nuclear": bool(nuclear[idx]),
                    "assignment_source": "patch_multi_cell_rl",
                    "patch_id": str(context.patch_id),
                }
            )
        out[str(cell_id)] = rows
    return out

def _parse_square_barcode(barcode: str) -> tuple[int, int] | None:
    parts = str(barcode).replace("-1", "").split("_")
    if len(parts) < 4:
        return None
    try:
        return int(parts[-2]), int(parts[-1])
    except ValueError:
        return None

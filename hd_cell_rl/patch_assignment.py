"""Convert patch final masks into bin assignment rows."""

from __future__ import annotations

from typing import Any

import numpy as np


def validate_unique_ownership(rows, *, nuclear_owners=None):
    """Reject duplicate physical observations before computing any metric."""
    seen = set()
    for row in rows:
        barcode, cell = str(row['barcode']), str(row['cell_id'])
        if barcode in seen:
            raise ValueError(f'duplicate physical barcode in assignments: {barcode}')
        seen.add(barcode)
        owner = (nuclear_owners or {}).get(barcode)
        if owner is not None and str(owner) != cell:
            raise ValueError(f'nuclear ownership changed for {barcode}: {owner} -> {cell}')


class PatchOwnershipMerger:
    """Shared deterministic export protocol for EM, distance and RL runners.

    Choose one snapshot per cell, then reconcile physical barcodes. EM-only
    runners use score zero: stable patch IDs break ties, never GT or entropy.
    RL retains the historical patch-score rule (not calibrated confidence).
    """

    def __init__(self):
        self.best_rows_by_cell = {}
        self._patch_by_cell = {}
        self.nuclear_owners = {}
        self.patch_choices = []
        self._claims = {}

    def add_patch(self, *, context, rows, score=0.0, target_cell_ids=None):
        score = float(score)
        if not np.isfinite(score):
            raise ValueError('cannot merge non-finite patch scores')
        validate_unique_ownership(rows)
        for cell in context.cells:
            for i in np.flatnonzero(np.asarray(cell.initial_membership_mask) > 0):
                barcode, owner = str(cell.candidate_bin_ids[i]), str(cell.cell_id)
                previous = self.nuclear_owners.get(barcode)
                if previous is not None and previous != owner:
                    raise ValueError(f'inconsistent nuclear owner across patches: {barcode}')
                self.nuclear_owners[barcode] = owner
        groups = {str(c): [] for c in (target_cell_ids or ())}
        for row in rows:
            self._claims.setdefault(str(row['barcode']), set()).add(str(row['cell_id']))
            if str(row.get('patch_id', context.patch_id)) != str(context.patch_id):
                raise ValueError('assignment source patch does not match context')
            groups.setdefault(str(row['cell_id']), []).append(dict(row, patch_id=str(context.patch_id)))
        for cell_id, values in groups.items():
            values.sort(key=lambda r: str(r['barcode']))
            patch_id = str(context.patch_id)
            current = self.best_rows_by_cell.get(cell_id)
            self.patch_choices.append({'cell_id': cell_id, 'patch_id': patch_id,
                                       'source_patch_score': score, 'n_bins': len(values)})
            if current is None or (-score, patch_id) < (-current[0], self._patch_by_cell[cell_id]):
                self.best_rows_by_cell[cell_id] = (score, values)
                self._patch_by_cell[cell_id] = patch_id
            elif (-score, patch_id) == (-current[0], self._patch_by_cell[cell_id]):
                if {str(r['barcode']) for r in current[1]} != {str(r['barcode']) for r in values}:
                    raise ValueError(f'inconsistent repeated cell/patch snapshot: {cell_id}/{patch_id}')

    def finalize(self):
        # Check the complete seed map only after all patches are seen. Otherwise
        # an invalid claim can fail or be dropped depending on input order.
        for barcode, owners in self._claims.items():
            locked = self.nuclear_owners.get(barcode)
            if locked is not None and owners != {locked}:
                raise ValueError(f'nuclear ownership changed across patch exports: {barcode}')
        rows, conflicts = reconcile_patch_owners(self.best_rows_by_cell, nuclear_owners=self.nuclear_owners)
        validate_unique_ownership(rows, nuclear_owners=self.nuclear_owners)
        present = {str(r['barcode']) for r in rows}
        missing = [b for b, c in self.nuclear_owners.items() if c in self.best_rows_by_cell and b not in present]
        if missing:
            raise ValueError(f'selected-cell nuclear seeds missing from merged output: {missing[:5]}')
        return rows, conflicts


def reconcile_patch_owners(
    rows_by_cell: dict[str, tuple[float, list[dict[str, Any]]]],
    *,
    nuclear_owners: dict[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Resolve overlapping patch exports without GT or fractional RL membership.

    Known nuclear owners take precedence. Other conflicts use the existing
    source patch score, then stable patch/cell IDs. Scores are not calibrated
    across patches; this is deterministic stitching, not new inference.
    """
    candidates: dict[str, list[tuple[float, dict[str, Any]]]] = {}
    for score, rows in rows_by_cell.values():
        if not np.isfinite(score):
            raise ValueError("cannot merge non-finite patch scores")
        for row in rows:
            candidates.setdefault(str(row["barcode"]), []).append((float(score), row))
    output, conflicts = [], []
    for barcode in sorted(candidates):
        claims = candidates[barcode]
        locked = nuclear_owners.get(barcode)
        eligible = [(score, row) for score, row in claims if locked is None or str(row["cell_id"]) == locked]
        eligible.sort(key=lambda item: (-item[0], str(item[1].get("patch_id", "")), str(item[1]["cell_id"])))
        winner = None if not eligible else eligible[0][1]
        if winner is not None:
            output.append(dict(winner))
        if len({str(row["cell_id"]) for _, row in claims}) > 1 or len(eligible) != len(claims):
            for score, row in sorted(claims, key=lambda item: (-item[0], str(item[1].get('patch_id', '')), str(item[1]['cell_id']))):
                conflicts.append({"barcode": barcode, "candidate_cell_id": str(row["cell_id"]),
                                  "source_patch_id": str(row.get("patch_id", "")), "source_patch_score": score,
                                  "nuclear_owner": locked, "final_owner": None if winner is None else str(winner["cell_id"]),
                                  "rule": "nuclear_then_patch_score_then_stable_ids"})
    if len({row["barcode"] for row in output}) != len(output):
        raise AssertionError("merged assignments violate unique barcode ownership")
    return output, conflicts

from .patch_types import PatchContext


def patch_assignments_for_core_cells(
    *,
    context: PatchContext,
    final_masks: dict[str, np.ndarray],
) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    ctx_by_cell = {ctx.cell_id: ctx for ctx in context.cells}
    em_result = context.em_assignment
    em_bin_index = {} if em_result is None else em_result.barcode_to_index()
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
            row = {
                    "barcode": barcode,
                    "cell_id": str(cell_id),
                    "array_row": int(array_row),
                    "array_col": int(array_col),
                    "x_um": float(xy[idx, 0]),
                    "y_um": float(xy[idx, 1]),
                    "is_nuclear": bool(nuclear[idx]),
                    "assignment_source": (
                        "patch_em_plus_rl"
                        if em_result is not None
                        else "patch_multi_cell_rl"
                    ),
                    "patch_id": str(context.patch_id),
                }
            em_idx = em_bin_index.get(barcode)
            if em_result is not None and em_idx is not None:
                row.update(
                    {
                        "em_top1_cell_id": str(
                            em_result.cell_ids[
                                int(em_result.top1_cell_index[int(em_idx)])
                            ]
                        ),
                        "em_top1_probability": float(
                            em_result.top1_probability[int(em_idx)]
                        ),
                        "em_top2_probability": float(
                            em_result.top2_probability[int(em_idx)]
                        ),
                        "em_margin": float(em_result.margin[int(em_idx)]),
                        "em_normalized_entropy": float(
                            em_result.normalized_entropy[int(em_idx)]
                        ),
                        "em_is_ambiguous": bool(
                            em_result.is_ambiguous[int(em_idx)]
                        ),
                        "em_is_nuclear_locked": bool(
                            em_result.is_nuclear_locked[int(em_idx)]
                        ),
                    }
                )
            rows.append(row)
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

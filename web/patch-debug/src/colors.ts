import type { OverlapCategory, PatchPayload } from "./types";

const OWNER_PALETTE = [
  "#2fa89a",
  "#de8f32",
  "#5f83d6",
  "#d56873",
  "#82a955",
  "#a777c0",
  "#4fa8c7",
  "#c4a33d",
  "#668f73",
  "#d07c51",
  "#887fc9",
  "#4b9b67",
  "#c45b9a",
  "#8fa2b7",
  "#c783a7",
  "#6ea4a0",
  "#d5a16b",
  "#758bc1",
  "#9aaf5f",
  "#c76b5b",
];

export const OVERLAP_COLORS: Record<OverlapCategory, string> = {
  correct_owner: "#2fa89a",
  wrong_owner: "#f0b45c",
  unmatched_owner: "#b78ad6",
  pred_only: "#ff7b71",
  gt_only: "#76a9ff",
  unscored: "#303a36",
};

export function ownerColorMap(ownerIds: string[]): Map<string, string> {
  const sorted = [...new Set(ownerIds)].sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
  return new Map(sorted.map((owner, index) => [owner, OWNER_PALETTE[index % OWNER_PALETTE.length]]));
}

export function patchOwnerColorMap(patch: PatchPayload | null): Map<string, string> {
  if (patch === null) {
    return new Map();
  }
  const allOwners = [
    ...new Set(
      patch.bins
        .flatMap((bin) => [
          bin.predicted_owner_cell_id,
          bin.trajectory_final_owner_cell_id ?? null,
          bin.gt_owner_cell_id,
        ])
        .filter((owner): owner is string => owner !== null),
    ),
    ...(patch.trajectory?.initial_owners.map((row) => row.cell_id) ?? []),
    ...(patch.trajectory?.final_owners.map((row) => row.cell_id) ?? []),
    ...(patch.trajectory?.steps.flatMap((step) =>
      step.actions.flatMap((action) => [action.cell_id, action.old_cell_id].filter((value): value is string => value !== null))
    ) ?? []),
  ].sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
  const colors = ownerColorMap(allOwners);
  for (const cell of patch.cells) {
    if (cell.matched_gt_cell_id !== null && colors.has(cell.cell_id)) {
      colors.set(cell.matched_gt_cell_id, colors.get(cell.cell_id)!);
    }
  }
  return colors;
}

export function ownerColor(owner: string | null, colors: Map<string, string>): string {
  if (owner === null) {
    return "#303a36";
  }
  return colors.get(owner) ?? "#87958e";
}

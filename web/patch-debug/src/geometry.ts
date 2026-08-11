import type { PatchBin } from "./types";

export type Segment = {
  key: string;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  owner: string;
};

export function buildOwnerOutlineSegments(
  bins: PatchBin[],
  ownerField: "predicted_owner_cell_id" | "gt_owner_cell_id",
  binSizeUm: number,
): Segment[] {
  const owners = new Map<string, string>();
  for (const bin of bins) {
    const owner = bin[ownerField];
    if (owner !== null) {
      owners.set(bin.barcode, owner);
    }
  }
  return buildOwnerOutlineSegmentsForOwners(bins, owners, binSizeUm);
}

export function buildOwnerOutlineSegmentsForOwners(
  bins: PatchBin[],
  owners: Map<string, string>,
  binSizeUm: number,
): Segment[] {
  const byOwner = new Map<string, PatchBin[]>();
  for (const bin of bins) {
    const owner = owners.get(bin.barcode);
    if (owner === undefined) {
      continue;
    }
    const group = byOwner.get(owner) ?? [];
    group.push(bin);
    byOwner.set(owner, group);
  }

  const segments: Segment[] = [];
  const half = binSizeUm / 2;
  for (const [owner, ownerBins] of byOwner) {
    const occupied = new Set(ownerBins.map((bin) => `${bin.array_row}:${bin.array_col}`));
    for (const bin of ownerBins) {
      const row = bin.array_row;
      const col = bin.array_col;
      const edges = [
        {
          neighbor: `${row}:${col - 1}`,
          x1: bin.x_um - half,
          y1: bin.y_um - half,
          x2: bin.x_um - half,
          y2: bin.y_um + half,
          side: "l",
        },
        {
          neighbor: `${row}:${col + 1}`,
          x1: bin.x_um + half,
          y1: bin.y_um - half,
          x2: bin.x_um + half,
          y2: bin.y_um + half,
          side: "r",
        },
        {
          neighbor: `${row - 1}:${col}`,
          x1: bin.x_um - half,
          y1: bin.y_um - half,
          x2: bin.x_um + half,
          y2: bin.y_um - half,
          side: "t",
        },
        {
          neighbor: `${row + 1}:${col}`,
          x1: bin.x_um - half,
          y1: bin.y_um + half,
          x2: bin.x_um + half,
          y2: bin.y_um + half,
          side: "b",
        },
      ];
      for (const edge of edges) {
        if (!occupied.has(edge.neighbor)) {
          segments.push({
            key: `${owner}:${row}:${col}:${edge.side}`,
            x1: edge.x1,
            y1: edge.y1,
            x2: edge.x2,
            y2: edge.y2,
            owner,
          });
        }
      }
    }
  }
  return segments;
}

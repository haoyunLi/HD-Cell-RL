import { ArrowDownUp, History, Search } from "lucide-react";
import { useMemo, useState } from "react";

import type { PatchManifestEntry } from "../types";

type SortMode = "lowest_iou" | "highest_reward" | "patch_id";

type Props = {
  patches: PatchManifestEntry[];
  selectedPatchId: string | null;
  onSelectPatch: (patchId: string) => void;
};

export function PatchSidebar({ patches, selectedPatchId, onSelectPatch }: Props) {
  const [query, setQuery] = useState("");
  const [sortMode, setSortMode] = useState<SortMode>("lowest_iou");
  const visiblePatches = useMemo(() => {
    const normalized = query.trim().toLowerCase();
    const filtered = normalized
      ? patches.filter((patch) => patch.patch_id.toLowerCase().includes(normalized))
      : [...patches];
    filtered.sort((left, right) => comparePatch(left, right, sortMode));
    return filtered;
  }, [patches, query, sortMode]);

  return (
    <aside className="patch-sidebar" aria-label="Patch selection">
      <div className="sidebar-heading">
        <div>
          <h2>Patches</h2>
          <span>{patches.length} evaluated</span>
        </div>
        <History size={15} aria-hidden="true" />
      </div>

      <label className="search-field">
        <Search size={15} aria-hidden="true" />
        <input
          type="search"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Find patch"
          aria-label="Find patch"
        />
      </label>

      <label className="sort-field">
        <ArrowDownUp size={13} aria-hidden="true" />
        <span className="sr-only">Sort patches</span>
        <select value={sortMode} onChange={(event) => setSortMode(event.target.value as SortMode)}>
          <option value="lowest_iou">Lowest owner IoU</option>
          <option value="highest_reward">Highest reward</option>
          <option value="patch_id">Patch ID</option>
        </select>
      </label>

      <div className="patch-list">
        {visiblePatches.map((patch, index) => {
          const ownerIou = patch.metrics.owner_micro_iou;
          return (
            <button
              className={patch.patch_id === selectedPatchId ? "patch-row selected" : "patch-row"}
              key={patch.patch_id}
              onClick={() => onSelectPatch(patch.patch_id)}
              type="button"
            >
              <span className="patch-index">{String(index + 1).padStart(2, "0")}</span>
              <span className="patch-row-body">
                <span className="patch-row-heading">
                  <span className="patch-row-title">{shortPatchId(patch.patch_id)}</span>
                  {patch.trajectory_available ? (
                    <History size={12} aria-label="Trajectory available" />
                  ) : null}
                </span>
                <span className="patch-quality">
                  <i style={{ width: `${Math.max(2, (ownerIou ?? 0) * 100)}%` }} />
                </span>
                <span className="patch-row-counts">
                  {patch.n_core_cells} cells · {patch.n_predicted_bins}/{patch.n_gt_bins} bins
                </span>
              </span>
              <span className={metricTone(ownerIou)}>
                <strong>{formatPercent(ownerIou)}</strong>
                <small>owner IoU</small>
              </span>
            </button>
          );
        })}
        {visiblePatches.length === 0 ? <p className="empty-list">No matching patches.</p> : null}
      </div>
    </aside>
  );
}

function comparePatch(left: PatchManifestEntry, right: PatchManifestEntry, mode: SortMode): number {
  if (mode === "patch_id") {
    return left.patch_id.localeCompare(right.patch_id, undefined, { numeric: true });
  }
  if (mode === "highest_reward") {
    return (
      nullableNumber(right.total_reward, Number.NEGATIVE_INFINITY) -
      nullableNumber(left.total_reward, Number.NEGATIVE_INFINITY)
    );
  }
  return (
    nullableNumber(left.metrics.owner_micro_iou, Number.POSITIVE_INFINITY) -
    nullableNumber(right.metrics.owner_micro_iou, Number.POSITIVE_INFINITY)
  );
}

function nullableNumber(value: number | null, fallback: number): number {
  return value === null || !Number.isFinite(value) ? fallback : value;
}

function shortPatchId(patchId: string): string {
  return patchId.replace(/^patch_/, "Patch ");
}

function formatPercent(value: number | null): string {
  return value === null ? "n/a" : `${Math.round(value * 100)}%`;
}

function metricTone(value: number | null): string {
  if (value === null) {
    return "patch-row-score";
  }
  if (value < 0.35) {
    return "patch-row-score low";
  }
  if (value >= 0.6) {
    return "patch-row-score high";
  }
  return "patch-row-score";
}

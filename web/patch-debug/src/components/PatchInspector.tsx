import {
  ArrowDown,
  ArrowRight,
  ArrowUp,
  Ban,
  Crosshair,
  Plus,
  RefreshCw,
  X,
} from "lucide-react";
import { useMemo, useState } from "react";

import { ownerColor } from "../colors";
import { dynamicOverlapCategory } from "../trajectory";
import type {
  PatchBin,
  PatchCell,
  PatchPayload,
  PatchTrajectory,
  PatchTrajectoryAction,
  PatchTrajectoryStep,
} from "../types";

type SortKey = "patch_iou" | "current_bins" | "cell_id";

type Props = {
  patch: PatchPayload;
  trajectory: PatchTrajectory;
  currentStep: PatchTrajectoryStep | null;
  currentOwners: Map<string, string>;
  ownerColors: Map<string, string>;
  stepIndex: number;
  selectedCellId: string | null;
  selectedBinBarcode: string | null;
  onSelectCell: (cellId: string | null) => void;
  onSelectBin: (barcode: string | null, ownerId?: string | null) => void;
};

export function PatchInspector({
  patch,
  trajectory,
  currentStep,
  currentOwners,
  ownerColors,
  stepIndex,
  selectedCellId,
  selectedBinBarcode,
  onSelectCell,
  onSelectBin,
}: Props) {
  const [sortKey, setSortKey] = useState<SortKey>("patch_iou");
  const [ascending, setAscending] = useState(true);
  const currentBinsByOwner = useMemo(() => countOwners(currentOwners), [currentOwners]);
  const cells = useMemo(() => {
    const sorted = [...patch.cells];
    sorted.sort((left, right) =>
      compareCell(left, right, sortKey, ascending, currentBinsByOwner),
    );
    return sorted;
  }, [ascending, currentBinsByOwner, patch.cells, sortKey]);
  const selectedCell = patch.cells.find((cell) => cell.cell_id === selectedCellId) ?? null;
  const selectedBin =
    selectedBinBarcode === null
      ? null
      : (patch.bins.find((bin) => bin.barcode === selectedBinBarcode) ?? null);
  const selectedAction =
    selectedBin === null
      ? null
      : (currentStep?.actions.find((action) => action.barcode === selectedBin.barcode) ?? null);
  const matchedGtByOwner = useMemo(
    () =>
      new Map(
        patch.cells
          .filter((cell) => cell.matched_gt_cell_id !== null)
          .map((cell) => [cell.cell_id, cell.matched_gt_cell_id!]),
      ),
    [patch.cells],
  );

  const updateSort = (nextKey: SortKey) => {
    if (nextKey === sortKey) {
      setAscending((value) => !value);
    } else {
      setSortKey(nextKey);
      setAscending(nextKey !== "current_bins");
    }
  };

  return (
    <aside className="patch-inspector" aria-label="Patch assignment inspector">
      <StepSection
        trajectory={trajectory}
        currentStep={currentStep}
        stepIndex={stepIndex}
        ownerColors={ownerColors}
        onSelectAction={(action) => onSelectBin(action.barcode, action.cell_id)}
      />

      {selectedBin !== null ? (
        <BinSection
          bin={selectedBin}
          action={selectedAction}
          currentOwner={currentOwners.get(selectedBin.barcode) ?? null}
          ownerColors={ownerColors}
          category={dynamicOverlapCategory(
            selectedBin,
            currentOwners.get(selectedBin.barcode) ?? null,
            matchedGtByOwner,
          )}
          onClear={() => onSelectBin(null)}
        />
      ) : (
        <section className="bin-section bin-empty">
          <Crosshair size={17} aria-hidden="true" />
          <div>
            <h2>No bin selected</h2>
            <span>Current step actions remain highlighted on the patch.</span>
          </div>
        </section>
      )}

      {selectedCell !== null ? (
        <SelectedCellSection
          cell={selectedCell}
          currentBins={currentBinsByOwner.get(selectedCell.cell_id) ?? 0}
          color={ownerColor(selectedCell.cell_id, ownerColors)}
          onClear={() => onSelectCell(null)}
        />
      ) : null}

      <PatchQualitySection patch={patch} />

      <section className="cell-table-section">
        <div className="inspector-heading">
          <div>
            <h2>Cells in patch</h2>
            <span>{patch.cells.length} core cells · current ownership</span>
          </div>
        </div>
        <div className="cell-table-wrap">
          <table className="cell-table">
            <thead>
              <tr>
                <SortableHeader
                  label="Cell"
                  active={sortKey === "cell_id"}
                  ascending={ascending}
                  onClick={() => updateSort("cell_id")}
                />
                <SortableHeader
                  label="IoU"
                  active={sortKey === "patch_iou"}
                  ascending={ascending}
                  onClick={() => updateSort("patch_iou")}
                />
                <SortableHeader
                  label="Now / GT"
                  active={sortKey === "current_bins"}
                  ascending={ascending}
                  onClick={() => updateSort("current_bins")}
                />
              </tr>
            </thead>
            <tbody>
              {cells.map((cell) => {
                const currentBins = currentBinsByOwner.get(cell.cell_id) ?? 0;
                return (
                  <tr
                    key={cell.cell_id}
                    className={cell.cell_id === selectedCellId ? "selected" : ""}
                    onClick={() => onSelectCell(cell.cell_id)}
                  >
                    <td>
                      <span className="cell-identity">
                        <i style={{ background: ownerColor(cell.cell_id, ownerColors) }} />
                        <strong>{cell.cell_id}</strong>
                      </span>
                      <span>{cell.gt_cell_type ?? cell.matched_gt_cell_id ?? "GT unmatched"}</span>
                    </td>
                    <td className={iouClass(cell.patch_iou)}>{formatMetric(cell.patch_iou)}</td>
                    <td>{currentBins} / {cell.gt_bins}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>
    </aside>
  );
}

function StepSection({
  trajectory,
  currentStep,
  stepIndex,
  ownerColors,
  onSelectAction,
}: {
  trajectory: PatchTrajectory;
  currentStep: PatchTrajectoryStep | null;
  stepIndex: number;
  ownerColors: Map<string, string>;
  onSelectAction: (action: PatchTrajectoryAction) => void;
}) {
  if (!trajectory.available) {
    return (
      <section className="step-section final-only-step">
        <div className="inspector-heading">
          <div><h2>Final state</h2><span>No exact action trace</span></div>
        </div>
        <div className="step-empty">
          <Ban size={16} aria-hidden="true" />
          <span>This evaluation contains only its saved assignment.</span>
        </div>
      </section>
    );
  }

  const initial = currentStep === null;
  const score = initial ? trajectory.initial_patch_score : currentStep.patch_score_after;
  const owned = initial ? trajectory.initial_owned_target_count : currentStep.owned_target_count_after;
  const target = initial ? trajectory.target_count : currentStep.target_count;
  const appliedCount = initial ? 0 : currentStep.actions.filter((action) => action.applied).length;
  const rejectedCount = initial ? 0 : currentStep.actions.length - appliedCount;

  return (
    <section className="step-section">
      <div className="step-hero">
        <div>
          <span className="step-label">{initial ? "Initial state" : `Step ${stepIndex}`}</span>
          <h2>{initial ? "Seed ownership" : stepHeadline(currentStep)}</h2>
        </div>
        {!initial ? (
          <strong className={rewardTone(currentStep.reward)}>{signedNumber(currentStep.reward)}</strong>
        ) : null}
      </div>

      <dl className="step-readout">
        <Readout label="Cumulative" value={initial ? "0.000" : formatNumber(currentStep.cumulative_reward)} />
        <Readout label="Patch score" value={formatNumber(score)} />
        <Readout label="Targets" value={formatCount(owned, target)} />
        <Readout label="Applied" value={String(appliedCount)} />
        <Readout label="Rollback" value={String(rejectedCount)} />
        <Readout label="No-op cells" value={initial ? "—" : String(currentStep.n_noop_actions)} />
      </dl>

      {!initial ? (
        <div className="action-event-list">
          {currentStep.actions.length === 0 ? (
            <div className="step-empty compact"><Ban size={14} aria-hidden="true" /><span>No owner change.</span></div>
          ) : (
            currentStep.actions.map((action, index) => (
              <ActionEventRow
                key={`${action.barcode}:${action.cell_id}:${index}`}
                action={action}
                ownerColors={ownerColors}
                onClick={() => onSelectAction(action)}
              />
            ))
          )}
        </div>
      ) : null}
    </section>
  );
}

function BinSection({
  bin,
  action,
  currentOwner,
  ownerColors,
  category,
  onClear,
}: {
  bin: PatchBin;
  action: PatchTrajectoryAction | null;
  currentOwner: string | null;
  ownerColors: Map<string, string>;
  category: string;
  onClear: () => void;
}) {
  return (
    <section className="bin-section">
      <div className="inspector-heading">
        <div>
          <span className="section-label">Selected bin</span>
          <h2>{bin.barcode}</h2>
        </div>
        <button className="icon-button dark" type="button" onClick={onClear} title="Clear bin selection">
          <X size={15} aria-hidden="true" />
          <span className="sr-only">Clear bin selection</span>
        </button>
      </div>

      {action !== null ? (
        <div className={`bin-action-summary ${action.applied ? action.type : "rollback"}`}>
          <span>{action.type === "replace" ? <RefreshCw size={15} /> : <Plus size={15} />}</span>
          <div>
            <strong>{action.applied ? action.type.toUpperCase() : "ROLLBACK"}</strong>
            <small>{action.applied ? "owner change applied" : "proposal not applied"}</small>
          </div>
        </div>
      ) : null}

      {action?.type === "replace" && action.old_cell_id !== null ? (
        <div className="owner-transfer-readout">
          <OwnerLabel owner={action.old_cell_id} ownerColors={ownerColors} />
          <ArrowRight size={15} aria-hidden="true" />
          <OwnerLabel owner={action.cell_id} ownerColors={ownerColors} />
        </div>
      ) : null}

      <dl className="bin-readout">
        <Readout label="Current owner" value={currentOwner ?? "unassigned"} swatch={ownerColor(currentOwner, ownerColors)} />
        <Readout label="GT owner" value={bin.gt_owner_cell_id ?? "none"} swatch={ownerColor(bin.gt_owner_cell_id, ownerColors)} />
        <Readout label="Overlap" value={category.replaceAll("_", " ")} />
        <Readout label="Grid" value={`${bin.array_row}:${bin.array_col}`} />
        <Readout label="x / y" value={`${bin.x_um.toFixed(1)} / ${bin.y_um.toFixed(1)} um`} />
        <Readout label="Core" value={bin.inside_core ? "inside" : "margin"} />
      </dl>
    </section>
  );
}

function SelectedCellSection({
  cell,
  currentBins,
  color,
  onClear,
}: {
  cell: PatchCell;
  currentBins: number;
  color: string;
  onClear: () => void;
}) {
  return (
    <section className="selected-cell-section">
      <div className="inspector-heading">
        <div>
          <span className="cell-heading"><i style={{ background: color }} />Cell {cell.cell_id}</span>
          <h2>{cell.gt_cell_type ?? `GT ${cell.matched_gt_cell_id ?? "unmatched"}`}</h2>
        </div>
        <button className="icon-button" type="button" onClick={onClear} title="Clear cell selection">
          <X size={15} aria-hidden="true" />
          <span className="sr-only">Clear cell selection</span>
        </button>
      </div>
      <dl className="cell-readout">
        <Readout label="Current / GT bins" value={`${currentBins} / ${cell.gt_bins}`} />
        <Readout label="Patch IoU" value={formatMetric(cell.patch_iou)} />
        <Readout label="Precision / recall" value={`${formatMetric(cell.patch_precision)} / ${formatMetric(cell.patch_recall)}`} />
      </dl>
    </section>
  );
}

function PatchQualitySection({ patch }: { patch: PatchPayload }) {
  const categories = [
    ["correct", patch.metrics.correct_owner_bins],
    ["wrong", patch.metrics.wrong_owner_bins],
    ["unmatched", patch.metrics.unmatched_owner_bins],
    ["pred-only", patch.metrics.pred_only_bins],
    ["gt-only", patch.metrics.gt_only_bins],
  ] as const;
  const total = categories.reduce((sum, [, value]) => sum + value, 0) || 1;
  return (
    <section className="metric-section">
      <div className="inspector-heading">
        <div><h2>Patch quality</h2><span>Final assignment against matched GT</span></div>
      </div>
      <div className="quality-metrics">
        <QualityMetric label="Foreground IoU" value={patch.metrics.foreground_iou} />
        <QualityMetric label="Owner micro IoU" value={patch.metrics.owner_micro_iou} />
        <QualityMetric label="Owner accuracy" value={patch.metrics.owner_accuracy} />
        <QualityMetric label="Macro cell IoU" value={patch.metrics.macro_cell_iou} />
      </div>
      <div className="ownership-composition" aria-label="Ownership outcome counts">
        <div className="composition-bar" aria-hidden="true">
          {categories.map(([key, value]) => (
            <i key={key} className={key} style={{ width: `${(value / total) * 100}%` }} />
          ))}
        </div>
        <div className="composition-key">
          {categories.map(([key, value]) => (
            <span key={key}><i className={key} />{key.replace("-", " ")} <strong>{value}</strong></span>
          ))}
        </div>
      </div>
    </section>
  );
}

function QualityMetric({ label, value }: { label: string; value: number | null }) {
  const width = value === null ? 0 : Math.max(0, Math.min(1, value)) * 100;
  return (
    <div>
      <span>{label}</span>
      <strong>{formatMetric(value)}</strong>
      <i aria-hidden="true"><b style={{ width: `${width}%` }} /></i>
    </div>
  );
}

function ActionEventRow({
  action,
  ownerColors,
  onClick,
}: {
  action: PatchTrajectoryAction;
  ownerColors: Map<string, string>;
  onClick: () => void;
}) {
  const replace = action.type === "replace";
  return (
    <button className={`action-event ${action.applied ? action.type : "rollback"}`} type="button" onClick={onClick}>
      <span className="action-event-icon" aria-hidden="true">{replace ? <RefreshCw size={13} /> : <Plus size={14} />}</span>
      <span className="action-event-main">
        <span><strong>{action.applied ? action.type.toUpperCase() : "ROLLBACK"}</strong><code>{action.barcode}</code></span>
        <span className="action-owner-flow">
          {replace && action.old_cell_id !== null ? <OwnerLabel owner={action.old_cell_id} ownerColors={ownerColors} compact /> : null}
          {replace && action.old_cell_id !== null ? <ArrowRight size={12} /> : null}
          <OwnerLabel owner={action.cell_id} ownerColors={ownerColors} compact />
        </span>
      </span>
      <ArrowRight size={14} className="action-open" aria-hidden="true" />
    </button>
  );
}

function OwnerLabel({ owner, ownerColors, compact = false }: { owner: string; ownerColors: Map<string, string>; compact?: boolean }) {
  return <span className={compact ? "owner-label compact" : "owner-label"}><i style={{ background: ownerColor(owner, ownerColors) }} />{owner}</span>;
}

function Readout({ label, value, swatch }: { label: string; value: string; swatch?: string }) {
  return <div><dt>{label}</dt><dd>{swatch ? <i style={{ background: swatch }} /> : null}{value}</dd></div>;
}

function SortableHeader({
  label,
  active,
  ascending,
  onClick,
}: {
  label: string;
  active: boolean;
  ascending: boolean;
  onClick: () => void;
}) {
  return (
    <th>
      <button type="button" className="table-sort" onClick={onClick}>
        {label}{active ? ascending ? <ArrowUp size={12} /> : <ArrowDown size={12} /> : null}
      </button>
    </th>
  );
}

function countOwners(owners: Map<string, string>): Map<string, number> {
  const counts = new Map<string, number>();
  for (const owner of owners.values()) {
    counts.set(owner, (counts.get(owner) ?? 0) + 1);
  }
  return counts;
}

function compareCell(
  left: PatchCell,
  right: PatchCell,
  key: SortKey,
  ascending: boolean,
  currentBinsByOwner: Map<string, number>,
): number {
  let result: number;
  if (key === "cell_id") {
    result = left.cell_id.localeCompare(right.cell_id, undefined, { numeric: true });
  } else if (key === "current_bins") {
    result = (currentBinsByOwner.get(left.cell_id) ?? 0) - (currentBinsByOwner.get(right.cell_id) ?? 0);
  } else {
    result = nullableMetric(left.patch_iou) - nullableMetric(right.patch_iou);
  }
  return ascending ? result : -result;
}

function stepHeadline(step: PatchTrajectoryStep): string {
  const applied = step.actions.filter((action) => action.applied);
  const adds = applied.filter((action) => action.type === "add").length;
  const replaces = applied.length - adds;
  if (applied.length === 0) {
    return step.outcome?.replaceAll("_", " ") ?? "No owner change";
  }
  const parts = [];
  if (adds > 0) parts.push(`${adds} ADD`);
  if (replaces > 0) parts.push(`${replaces} REPLACE`);
  return parts.join(" · ");
}

function formatCount(value: number | null, total: number | null): string {
  if (value === null && total === null) return "n/a";
  return `${value ?? "?"} / ${total ?? "?"}`;
}

function nullableMetric(value: number | null): number {
  return value === null ? Number.POSITIVE_INFINITY : value;
}

function formatMetric(value: number | null): string {
  return value === null ? "n/a" : value.toFixed(3);
}

function formatNumber(value: number | null): string {
  if (value === null) return "n/a";
  return Math.abs(value) >= 1000 ? value.toExponential(2) : value.toFixed(3);
}

function signedNumber(value: number): string {
  return `${value > 0 ? "+" : ""}${formatNumber(value)}`;
}

function rewardTone(value: number): string {
  if (value > 0) return "positive";
  if (value < 0) return "negative";
  return "neutral";
}

function iouClass(value: number | null): string {
  if (value === null) return "";
  if (value < 0.35) return "metric-low";
  if (value >= 0.6) return "metric-high";
  return "";
}

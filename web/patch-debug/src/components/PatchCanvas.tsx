import {
  CircleHelp,
  Layers3,
  Minus,
  Move,
  Plus,
  RotateCcw,
  ScanSearch,
} from "lucide-react";
import {
  type CSSProperties,
  type PointerEvent,
  type WheelEvent,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import { OVERLAP_COLORS, ownerColor } from "../colors";
import {
  buildOwnerOutlineSegments,
  buildOwnerOutlineSegmentsForOwners,
} from "../geometry";
import { dynamicOverlapCategory } from "../trajectory";
import type {
  ActionFocusMode,
  Bounds,
  DebugJumpKind,
  OverlapCategory,
  PatchBin,
  PatchCell,
  PatchPayload,
  PatchTrajectory,
  PatchTrajectoryAction,
  PatchTrajectoryStep,
  StepStateMode,
  ViewMode,
} from "../types";
import { TrajectoryControls } from "./TrajectoryControls";

type ViewBox = { x: number; y: number; width: number; height: number };

type Props = {
  patch: PatchPayload;
  mode: ViewMode;
  ownerColors: Map<string, string>;
  selectedCell: PatchCell | null;
  selectedBinBarcode: string | null;
  currentOwners: Map<string, string>;
  trajectory: PatchTrajectory;
  currentStep: PatchTrajectoryStep | null;
  actionFocusMode: ActionFocusMode;
  stepStateMode: StepStateMode;
  availableDebugJumps: Set<DebugJumpKind>;
  stepIndex: number;
  playing: boolean;
  playbackRate: number;
  showPredictedOutlines: boolean;
  showGtOutlines: boolean;
  showNuclei: boolean;
  showCoreBounds: boolean;
  onSelectCell: (cellId: string | null) => void;
  onSelectBin: (barcode: string | null, ownerId?: string | null) => void;
  onActionFocusModeChange: (mode: ActionFocusMode) => void;
  onStepStateModeChange: (mode: StepStateMode) => void;
  onDebugJump: (kind: DebugJumpKind) => void;
  onStepChange: (stepIndex: number) => void;
  onPlayingChange: (playing: boolean) => void;
  onPlaybackRateChange: (rate: number) => void;
  onShowPredictedOutlinesChange: (visible: boolean) => void;
  onShowGtOutlinesChange: (visible: boolean) => void;
  onShowNucleiChange: (visible: boolean) => void;
  onShowCoreBoundsChange: (visible: boolean) => void;
};

type HoveredBin = {
  bin: PatchBin;
  owner: string | null;
  category: OverlapCategory;
  action: PatchTrajectoryAction | null;
  clientX: number;
  clientY: number;
};

export function PatchCanvas({
  patch,
  mode,
  ownerColors,
  selectedCell,
  selectedBinBarcode,
  currentOwners,
  trajectory,
  currentStep,
  actionFocusMode,
  stepStateMode,
  availableDebugJumps,
  stepIndex,
  playing,
  playbackRate,
  showPredictedOutlines,
  showGtOutlines,
  showNuclei,
  showCoreBounds,
  onSelectCell,
  onSelectBin,
  onActionFocusModeChange,
  onStepStateModeChange,
  onDebugJump,
  onStepChange,
  onPlayingChange,
  onPlaybackRateChange,
  onShowPredictedOutlinesChange,
  onShowGtOutlinesChange,
  onShowNucleiChange,
  onShowCoreBoundsChange,
}: Props) {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const resetBox = useMemo(() => boundsToViewBox(patch.outer_bounds, patch.bin_size_um), [patch]);
  const [viewBox, setViewBox] = useState<ViewBox>(resetBox);
  const [dragStart, setDragStart] = useState<{
    clientX: number;
    clientY: number;
    viewBox: ViewBox;
  } | null>(null);
  const [hovered, setHovered] = useState<HoveredBin | null>(null);
  const [layersOpen, setLayersOpen] = useState(false);

  useEffect(() => {
    setViewBox(resetBox);
    setHovered(null);
    setLayersOpen(false);
  }, [resetBox, patch.patch_id]);

  const matchedGtByOwner = useMemo(
    () =>
      new Map(
        patch.cells
          .filter((cell) => cell.matched_gt_cell_id !== null)
          .map((cell) => [cell.cell_id, cell.matched_gt_cell_id!]),
      ),
    [patch.cells],
  );
  const predictedSegments = useMemo(
    () => buildOwnerOutlineSegmentsForOwners(patch.bins, currentOwners, patch.bin_size_um),
    [currentOwners, patch],
  );
  const gtSegments = useMemo(
    () => buildOwnerOutlineSegments(patch.bins, "gt_owner_cell_id", patch.bin_size_um),
    [patch],
  );
  const currentActions = currentStep?.actions ?? EMPTY_ACTIONS;
  const actionsByFocus = useMemo(
    () => ({
      selected:
        selectedBinBarcode === null
          ? EMPTY_ACTIONS
          : currentActions.filter((action) => action.barcode === selectedBinBarcode),
      cell:
        selectedCell === null
          ? EMPTY_ACTIONS
          : currentActions.filter(
              (action) =>
                action.cell_id === selectedCell.cell_id ||
                action.old_cell_id === selectedCell.cell_id,
            ),
      step: currentActions,
    }),
    [currentActions, selectedBinBarcode, selectedCell],
  );
  const visibleActions = actionsByFocus[actionFocusMode];
  const actionCounts = {
    selected: actionsByFocus.selected.length,
    cell: actionsByFocus.cell.length,
    step: actionsByFocus.step.length,
  };
  const actionByBarcode = useMemo(
    () => new Map(currentActions.map((action) => [action.barcode, action])),
    [currentActions],
  );
  const visibleActionByBarcode = useMemo(
    () => new Map(visibleActions.map((action) => [action.barcode, action])),
    [visibleActions],
  );
  const changedBarcodes = useMemo(
    () => new Set(currentActions.map((action) => action.barcode)),
    [currentActions],
  );
  const focusedCellIds = useMemo(
    () =>
      new Set(
        visibleActions.flatMap((action) =>
          action.old_cell_id === null
            ? [action.cell_id]
            : [action.cell_id, action.old_cell_id],
        ),
      ),
    [visibleActions],
  );
  const binsByBarcode = useMemo(
    () => new Map(patch.bins.map((bin) => [bin.barcode, bin])),
    [patch.bins],
  );
  const cellsById = useMemo(
    () => new Map(patch.cells.map((cell) => [cell.cell_id, cell])),
    [patch.cells],
  );
  const selectedBin =
    selectedBinBarcode === null ? null : (binsByBarcode.get(selectedBinBarcode) ?? null);

  const handleWheel = (event: WheelEvent<SVGSVGElement>) => {
    event.preventDefault();
    const svg = svgRef.current;
    if (svg === null) {
      return;
    }
    const rect = svg.getBoundingClientRect();
    const cursorX = viewBox.x + ((event.clientX - rect.left) / rect.width) * viewBox.width;
    const cursorY = viewBox.y + ((event.clientY - rect.top) / rect.height) * viewBox.height;
    const scale = event.deltaY < 0 ? 0.86 : 1.16;
    zoomAroundPoint(scale, cursorX, cursorY);
  };

  const zoomAroundPoint = (scale: number, centerX?: number, centerY?: number) => {
    const focusX = centerX ?? viewBox.x + viewBox.width / 2;
    const focusY = centerY ?? viewBox.y + viewBox.height / 2;
    const nextWidth = clamp(viewBox.width * scale, resetBox.width * 0.08, resetBox.width * 3);
    const nextHeight = nextWidth * (viewBox.height / viewBox.width);
    const relativeX = (focusX - viewBox.x) / viewBox.width;
    const relativeY = (focusY - viewBox.y) / viewBox.height;
    setViewBox({
      x: focusX - relativeX * nextWidth,
      y: focusY - relativeY * nextHeight,
      width: nextWidth,
      height: nextHeight,
    });
  };

  const handlePointerDown = (event: PointerEvent<SVGSVGElement>) => {
    if (event.button !== 0) {
      return;
    }
    event.currentTarget.setPointerCapture(event.pointerId);
    setDragStart({ clientX: event.clientX, clientY: event.clientY, viewBox });
  };

  const handlePointerMove = (event: PointerEvent<SVGSVGElement>) => {
    if (dragStart === null || svgRef.current === null) {
      return;
    }
    const rect = svgRef.current.getBoundingClientRect();
    const dx = ((event.clientX - dragStart.clientX) / rect.width) * dragStart.viewBox.width;
    const dy = ((event.clientY - dragStart.clientY) / rect.height) * dragStart.viewBox.height;
    setViewBox({
      ...dragStart.viewBox,
      x: dragStart.viewBox.x - dx,
      y: dragStart.viewBox.y - dy,
    });
  };

  const stopDragging = () => setDragStart(null);
  const half = patch.bin_size_um / 2;
  const widthUm = patch.outer_bounds.x_max - patch.outer_bounds.x_min;
  const heightUm = patch.outer_bounds.y_max - patch.outer_bounds.y_min;
  const zoom = resetBox.width / viewBox.width;
  const scaleBarUm = scaleBarLength(viewBox.width);
  const scaleBarWidth = Math.min(44, Math.max(12, (scaleBarUm / viewBox.width) * 100));

  return (
    <section className={trajectory.available ? "canvas-region has-debug-toolbar" : "canvas-region"}>
      <div className="canvas-toolbar">
        <div className="canvas-title">
          <h2>{patch.patch_id}</h2>
          <div className="canvas-meta">
            <span className="bin-size-spec">Bin {patch.bin_size_um} um x {patch.bin_size_um} um</span>
            <span>{formatCompact(widthUm)} x {formatCompact(heightUm)} um patch</span>
            <span>{patch.counts.core_cells} core cells</span>
            <span>{patch.counts.display_bins} visible bins</span>
          </div>
        </div>
        <div className="canvas-kpis" aria-label="Patch summary">
          <span><small>Owner IoU</small>{formatMetric(patch.metrics.owner_micro_iou)}</span>
          <span><small>Foreground</small>{formatMetric(patch.metrics.foreground_iou)}</span>
          <span><small>Reward</small>{formatNumber(patch.total_reward)}</span>
        </div>
        <div className="map-tools" aria-label="Map controls">
          <button className="icon-button" type="button" onClick={() => zoomAroundPoint(1.22)} title="Zoom out">
            <Minus size={15} aria-hidden="true" />
            <span className="sr-only">Zoom out</span>
          </button>
          <span className="zoom-readout">{zoom.toFixed(1)}x</span>
          <button className="icon-button" type="button" onClick={() => zoomAroundPoint(0.82)} title="Zoom in">
            <Plus size={15} aria-hidden="true" />
            <span className="sr-only">Zoom in</span>
          </button>
          <button className="icon-button" type="button" onClick={() => setViewBox(resetBox)} title="Reset view">
            <RotateCcw size={15} aria-hidden="true" />
            <span className="sr-only">Reset view</span>
          </button>
          <button
            className={layersOpen ? "icon-button active" : "icon-button"}
            type="button"
            aria-expanded={layersOpen}
            onClick={() => setLayersOpen((value) => !value)}
            title="Layers"
          >
            <Layers3 size={15} aria-hidden="true" />
            <span className="sr-only">Layers</span>
          </button>
        </div>
      </div>

      {trajectory.available ? (
        <DebugToolbar
          actionFocusMode={actionFocusMode}
          stepStateMode={stepStateMode}
          hasStep={currentStep !== null}
          hasSelectedCell={selectedCell !== null}
          hasSelectedBin={selectedBinBarcode !== null}
          actionCounts={actionCounts}
          availableDebugJumps={availableDebugJumps}
          onActionFocusModeChange={onActionFocusModeChange}
          onStepStateModeChange={onStepStateModeChange}
          onDebugJump={onDebugJump}
        />
      ) : null}

      <div className={dragStart === null ? "patch-canvas" : "patch-canvas dragging"}>
        <svg
          ref={svgRef}
          viewBox={`${viewBox.x} ${viewBox.y} ${viewBox.width} ${viewBox.height}`}
          preserveAspectRatio="xMidYMid meet"
          role="img"
          aria-label={`Spatial assignment map for ${patch.patch_id}`}
          onWheel={handleWheel}
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={stopDragging}
          onPointerCancel={stopDragging}
          onPointerLeave={() => {
            stopDragging();
            setHovered(null);
          }}
          onClick={() => {
            onSelectBin(null);
            onSelectCell(null);
          }}
        >
          <defs>
            <marker id="trace-add-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
              <path d="M0.8 0.8L9.2 5L0.8 9.2Z" fill="context-stroke" stroke="#07100d" strokeWidth="1.4" paintOrder="stroke" />
            </marker>
            <filter id="selected-bin-glow" x="-80%" y="-80%" width="260%" height="260%">
              <feGaussianBlur stdDeviation="0.6" result="blur" />
              <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
            </filter>
          </defs>
          <rect
            x={patch.outer_bounds.x_min}
            y={patch.outer_bounds.y_min}
            width={widthUm}
            height={heightUm}
            fill="#0d1210"
          />

          {patch.bins.map((bin) => {
            const currentOwner = currentOwners.get(bin.barcode) ?? null;
            const category = dynamicOverlapCategory(bin, currentOwner, matchedGtByOwner);
            const inSelectedCell = isSelectedBin(bin, currentOwner, selectedCell);
            const focusMuted = selectedCell !== null && !inSelectedCell;
            const changeMuted =
              stepStateMode === "changes" &&
              currentStep !== null &&
              !changedBarcodes.has(bin.barcode);
            const action = actionByBarcode.get(bin.barcode) ?? null;
            const visibleAction = visibleActionByBarcode.get(bin.barcode) ?? null;
            const selected = selectedBinBarcode === bin.barcode;
            const fillOpacity =
              mode === "gt" && bin.gt_owner_cell_id === null
                ? 0.08
                : currentOwner === null && bin.gt_owner_cell_id === null
                  ? 0.34
                  : 0.9;
            return (
              <rect
                key={bin.barcode}
                className={`bin-tile${selected ? " selected" : ""}${changeMuted ? " change-muted" : ""}${visibleAction === null ? "" : ` action-${visibleAction.applied ? visibleAction.type : "rollback"}`}`}
                x={bin.x_um - half}
                y={bin.y_um - half}
                width={patch.bin_size_um}
                height={patch.bin_size_um}
                fill={binFill(bin, currentOwner, category, mode, ownerColors)}
                fillOpacity={changeMuted ? 0.12 : focusMuted ? 0.3 : fillOpacity}
                stroke={bin.owner_conflict ? "#f0b45c" : "rgba(235,242,238,0.12)"}
                strokeWidth={bin.owner_conflict ? 0.5 : 0.08}
                vectorEffect="non-scaling-stroke"
                shapeRendering="crispEdges"
                onPointerEnter={(event) =>
                  setHovered({
                    bin,
                    owner: currentOwner,
                    category,
                    action,
                    clientX: event.clientX,
                    clientY: event.clientY,
                  })
                }
                onPointerMove={(event) =>
                  setHovered({
                    bin,
                    owner: currentOwner,
                    category,
                    action,
                    clientX: event.clientX,
                    clientY: event.clientY,
                  })
                }
                onPointerDown={(event) => event.stopPropagation()}
                onClick={(event) => {
                  event.stopPropagation();
                  onSelectBin(bin.barcode, cellIdForBin(bin, currentOwner, patch.cells));
                }}
              />
            );
          })}

          {showPredictedOutlines
            ? predictedSegments.map((segment) => (
                <line
                  key={`pred:${segment.key}`}
                  x1={segment.x1}
                  y1={segment.y1}
                  x2={segment.x2}
                  y2={segment.y2}
                  stroke={ownerColor(segment.owner, ownerColors)}
                  strokeOpacity={
                    stepStateMode === "changes"
                      ? focusedCellIds.has(segment.owner)
                        ? 0.9
                        : 0.14
                      : selectedCell === null || segment.owner === selectedCell.cell_id
                        ? 1
                        : 0.3
                  }
                  strokeWidth={0.72}
                  vectorEffect="non-scaling-stroke"
                  pointerEvents="none"
                />
              ))
            : null}

          {showGtOutlines
            ? gtSegments.flatMap((segment) => {
                const opacity =
                  stepStateMode === "changes"
                    ? 0.18
                    : selectedCell === null || segment.owner === selectedCell.matched_gt_cell_id
                      ? 1
                      : 0.3;
                return [
                  <line
                    key={`gt-halo:${segment.key}`}
                    x1={segment.x1}
                    y1={segment.y1}
                    x2={segment.x2}
                    y2={segment.y2}
                    stroke="#08100d"
                    strokeOpacity={opacity}
                    strokeWidth={4.2}
                    strokeLinecap="square"
                    vectorEffect="non-scaling-stroke"
                    pointerEvents="none"
                  />,
                  <line
                    key={`gt:${segment.key}`}
                    x1={segment.x1}
                    y1={segment.y1}
                    x2={segment.x2}
                    y2={segment.y2}
                    stroke="#76a9ff"
                    strokeOpacity={opacity}
                    strokeWidth={1.65}
                    strokeLinecap="square"
                    vectorEffect="non-scaling-stroke"
                    pointerEvents="none"
                  />,
                ];
              })
            : null}

          {showNuclei
            ? patch.cells.map((cell) =>
                cell.nucleus_center_xy_um === null ? null : (
                  <g
                    key={`nucleus:${cell.cell_id}`}
                    className="nucleus-marker"
                    opacity={
                      stepStateMode === "changes"
                        ? focusedCellIds.has(cell.cell_id)
                          ? 1
                          : 0.2
                        : selectedCell === null || selectedCell.cell_id === cell.cell_id
                          ? 1
                          : 0.3
                    }
                    pointerEvents="none"
                  >
                    <circle
                      cx={cell.nucleus_center_xy_um[0]}
                      cy={cell.nucleus_center_xy_um[1]}
                      r={1.05}
                      fill="#f1f5f3"
                      stroke="#0a0f0d"
                      strokeWidth={1.2}
                      vectorEffect="non-scaling-stroke"
                    />
                    <circle
                      cx={cell.nucleus_center_xy_um[0]}
                      cy={cell.nucleus_center_xy_um[1]}
                      r={0.3}
                      fill={ownerColor(cell.cell_id, ownerColors)}
                    />
                  </g>
                ),
              )
            : null}

          <g key={`trace:${stepIndex}`} className="trace-layer">
            {visibleActions.map((action, index) => {
              const bin = binsByBarcode.get(action.barcode);
              if (bin === undefined) {
                return null;
              }
              return (
                <ActionHighlight
                  key={`${action.barcode}:${action.cell_id}:${index}`}
                  action={action}
                  bin={bin}
                  binSizeUm={patch.bin_size_um}
                  ownerColors={ownerColors}
                  cellsById={cellsById}
                  emphasized
                  actionIndex={index}
                />
              );
            })}
          </g>

          {selectedBin !== null ? (
            <SelectedBinMarker bin={selectedBin} binSizeUm={patch.bin_size_um} />
          ) : null}

          {showCoreBounds ? (
            <rect
              x={patch.core_bounds.x_min}
              y={patch.core_bounds.y_min}
              width={patch.core_bounds.x_max - patch.core_bounds.x_min}
              height={patch.core_bounds.y_max - patch.core_bounds.y_min}
              fill="none"
              stroke="#76a9ff"
              strokeOpacity={0.75}
              strokeWidth={0.75}
              strokeDasharray="3 2"
              vectorEffect="non-scaling-stroke"
              pointerEvents="none"
            />
          ) : null}
        </svg>

        <div className="canvas-step-status">
          <strong>
            {trajectory.available
              ? stepIndex === 0
                ? "Initial"
                : `Step ${stepIndex} · ${stepStateMode}`
              : "Final"}
          </strong>
          <span>{currentStep?.outcome?.replaceAll("_", " ") ?? (trajectory.available ? "seed state" : "saved assignment")}</span>
        </div>
        <div className="canvas-pan-status"><Move size={13} aria-hidden="true" /></div>
        <CanvasLegend mode={mode} />
        <div className="scale-bar" aria-label={`Physical scale: ${scaleBarUm} micrometers`}>
          <i style={{ width: `${scaleBarWidth}%` }} />
          <span>Scale {scaleBarUm} um</span>
        </div>
        {layersOpen ? (
          <LayerPanel
            showPredictedOutlines={showPredictedOutlines}
            showGtOutlines={showGtOutlines}
            showNuclei={showNuclei}
            showCoreBounds={showCoreBounds}
            onShowPredictedOutlinesChange={onShowPredictedOutlinesChange}
            onShowGtOutlinesChange={onShowGtOutlinesChange}
            onShowNucleiChange={onShowNucleiChange}
            onShowCoreBoundsChange={onShowCoreBoundsChange}
          />
        ) : null}
        {hovered !== null ? <BinTooltip hovered={hovered} ownerColors={ownerColors} /> : null}
      </div>

      <TrajectoryControls
        trajectory={trajectory}
        stepIndex={stepIndex}
        playing={playing}
        playbackRate={playbackRate}
        onStepChange={onStepChange}
        onPlayingChange={onPlayingChange}
        onPlaybackRateChange={onPlaybackRateChange}
      />
    </section>
  );
}

const EMPTY_ACTIONS: PatchTrajectoryAction[] = [];

const DEBUG_JUMP_OPTIONS: Array<{ value: DebugJumpKind; label: string }> = [
  { value: "largest_reward_drop", label: "Largest reward drop" },
  { value: "largest_reward_gain", label: "Largest reward gain" },
  { value: "first_rollback", label: "First rollback" },
  { value: "first_wrong_replace", label: "First wrong replacement" },
  { value: "most_actions", label: "Most actions in one step" },
  { value: "most_overgrown_cell", label: "Most overgrown cell" },
  { value: "most_undersegmented_cell", label: "Most undersegmented cell" },
  { value: "lowest_iou_cell", label: "Lowest-IoU cell" },
];

function DebugToolbar({
  actionFocusMode,
  stepStateMode,
  hasStep,
  hasSelectedCell,
  hasSelectedBin,
  actionCounts,
  availableDebugJumps,
  onActionFocusModeChange,
  onStepStateModeChange,
  onDebugJump,
}: {
  actionFocusMode: ActionFocusMode;
  stepStateMode: StepStateMode;
  hasStep: boolean;
  hasSelectedCell: boolean;
  hasSelectedBin: boolean;
  actionCounts: Record<ActionFocusMode, number>;
  availableDebugJumps: Set<DebugJumpKind>;
  onActionFocusModeChange: (mode: ActionFocusMode) => void;
  onStepStateModeChange: (mode: StepStateMode) => void;
  onDebugJump: (kind: DebugJumpKind) => void;
}) {
  const [helpOpen, setHelpOpen] = useState(false);
  return (
    <div className="debug-toolbar" aria-label="Trajectory inspection controls">
      <div className="debug-control-group">
        <span>State</span>
        <div className="debug-segmented">
          {(["before", "after", "changes"] as const).map((mode) => (
            <button
              key={mode}
              type="button"
              className={stepStateMode === mode ? "active" : ""}
              aria-pressed={stepStateMode === mode}
              title={STATE_MODE_HELP[mode]}
              disabled={!hasStep && mode !== "after"}
              onClick={() => onStepStateModeChange(mode)}
            >
              {mode[0].toUpperCase() + mode.slice(1)}
            </button>
          ))}
        </div>
      </div>

      <div className="debug-control-group">
        <span>Focus</span>
        <div className="debug-segmented action-focus-segmented">
          {(["selected", "cell", "step"] as const).map((mode) => (
            <button
              key={mode}
              type="button"
              className={actionFocusMode === mode ? "active" : ""}
              aria-pressed={actionFocusMode === mode}
              title={ACTION_FOCUS_HELP[mode]}
              disabled={
                !hasStep ||
                (mode === "selected" && !hasSelectedBin) ||
                (mode === "cell" && !hasSelectedCell)
              }
              onClick={() => onActionFocusModeChange(mode)}
            >
              <span>{ACTION_FOCUS_LABELS[mode]}</span>
              <b>{actionCounts[mode]}</b>
            </button>
          ))}
        </div>
      </div>

      <div className="debug-toolbar-tools">
        <label
          className="debug-jump-select"
          title="Jump directly to a reward, rollback, replacement, or cell-quality issue"
        >
          <ScanSearch size={15} aria-hidden="true" />
          <span className="sr-only">Jump to debug event</span>
          <select
            value=""
            aria-label="Jump to debug event"
            onChange={(event) => onDebugJump(event.target.value as DebugJumpKind)}
          >
            <option value="">Jump to issue...</option>
            {DEBUG_JUMP_OPTIONS.map((option) => (
              <option
                key={option.value}
                value={option.value}
                disabled={!availableDebugJumps.has(option.value)}
              >
                {option.label}
              </option>
            ))}
          </select>
        </label>
        <button
          className={helpOpen ? "debug-help-button active" : "debug-help-button"}
          type="button"
          aria-label="Explain trajectory controls"
          aria-expanded={helpOpen}
          aria-controls="trajectory-control-help"
          title="Explain these controls"
          onClick={() => setHelpOpen((value) => !value)}
        >
          <CircleHelp size={16} aria-hidden="true" />
        </button>
      </div>

      {helpOpen ? <TrajectoryControlHelp /> : null}
    </div>
  );
}

const STATE_MODE_HELP: Record<StepStateMode, string> = {
  before: "Show ownership immediately before the current step",
  after: "Show ownership after applied actions in the current step",
  changes: "Show the after state while dimming bins unchanged by the current step",
};

const ACTION_FOCUS_LABELS: Record<ActionFocusMode, string> = {
  selected: "Bin",
  cell: "Cell",
  step: "Step",
};

const ACTION_FOCUS_HELP: Record<ActionFocusMode, string> = {
  selected: "Show only the current step action for the selected bin",
  cell: "Show current step actions where the selected cell gains or loses a bin",
  step: "Show every action in the current patch step",
};

function TrajectoryControlHelp() {
  return (
    <aside
      id="trajectory-control-help"
      className="debug-help-popover"
      aria-label="Trajectory control guide"
    >
      <section>
        <strong>Map state</strong>
        <dl>
          <div><dt>Before</dt><dd>Ownership before this step runs.</dd></div>
          <div><dt>After</dt><dd>Ownership after applied actions.</dd></div>
          <div><dt>Changes</dt><dd>After state with unchanged bins dimmed.</dd></div>
        </dl>
      </section>
      <section>
        <strong>Action focus</strong>
        <dl>
          <div><dt>Bin</dt><dd>Only the selected bin's action.</dd></div>
          <div><dt>Cell</dt><dd>Actions where the selected cell gains or loses a bin.</dd></div>
          <div><dt>Step</dt><dd>All actions in the current step.</dd></div>
        </dl>
      </section>
      <p>The number beside each focus button is the number of action paths currently available in that scope.</p>
    </aside>
  );
}

function ActionHighlight({
  action,
  bin,
  binSizeUm,
  ownerColors,
  cellsById,
  emphasized,
  actionIndex,
}: {
  action: PatchTrajectoryAction;
  bin: PatchBin;
  binSizeUm: number;
  ownerColors: Map<string, string>;
  cellsById: Map<string, PatchCell>;
  emphasized: boolean;
  actionIndex: number;
}) {
  const inset = binSizeUm * 0.2;
  const newCenter = cellsById.get(action.cell_id)?.nucleus_center_xy_um ?? null;
  const oldCenter =
    action.old_cell_id === null
      ? null
      : (cellsById.get(action.old_cell_id)?.nucleus_center_xy_um ?? null);
  const tone = action.applied
    ? action.type === "replace"
      ? "replace"
      : "add"
    : "rejected";
  const targetColor = action.applied ? ownerColor(action.cell_id, ownerColors) : "#ff7b71";

  return (
    <g
      className={emphasized ? "trace-action-highlight" : "trace-action-highlight muted"}
      style={{ "--trace-delay": `${Math.min(actionIndex, 20) * 24}ms` } as CSSProperties}
      pointerEvents="none"
    >
      {action.type === "replace" && oldCenter !== null ? (
        <>
          <line
            className="owner-transfer-halo old"
            x1={oldCenter[0]}
            y1={oldCenter[1]}
            x2={bin.x_um}
            y2={bin.y_um}
            vectorEffect="non-scaling-stroke"
          />
          <line
            className="owner-transfer old"
            x1={oldCenter[0]}
            y1={oldCenter[1]}
            x2={bin.x_um}
            y2={bin.y_um}
            stroke={ownerColor(action.old_cell_id, ownerColors)}
            vectorEffect="non-scaling-stroke"
          />
        </>
      ) : null}
      {newCenter !== null ? (
        <>
          <line
            className="owner-transfer-halo new"
            x1={newCenter[0]}
            y1={newCenter[1]}
            x2={bin.x_um}
            y2={bin.y_um}
            vectorEffect="non-scaling-stroke"
          />
          <line
            className={action.applied ? "owner-transfer new" : "owner-transfer rejected"}
            x1={newCenter[0]}
            y1={newCenter[1]}
            x2={bin.x_um}
            y2={bin.y_um}
            stroke={targetColor}
            markerEnd="url(#trace-add-arrow)"
            vectorEffect="non-scaling-stroke"
          />
          <circle
            className={`trace-origin-pulse ${tone}`}
            cx={newCenter[0]}
            cy={newCenter[1]}
            r={binSizeUm * 0.72}
            fill="none"
            stroke={targetColor}
            vectorEffect="non-scaling-stroke"
          />
        </>
      ) : null}
      {action.type === "replace" && action.old_cell_id !== null ? (
        <rect
          className="trace-old-owner"
          x={bin.x_um - binSizeUm / 2}
          y={bin.y_um - binSizeUm / 2}
          width={binSizeUm}
          height={binSizeUm}
          fill={ownerColor(action.old_cell_id, ownerColors)}
        />
      ) : null}
      <rect
        className={`trace-action-ring ${tone}`}
        x={bin.x_um - binSizeUm / 2 - inset}
        y={bin.y_um - binSizeUm / 2 - inset}
        width={binSizeUm + inset * 2}
        height={binSizeUm + inset * 2}
        fill="none"
        vectorEffect="non-scaling-stroke"
      />
      <circle
        className={`trace-destination-pulse ${tone}`}
        cx={bin.x_um}
        cy={bin.y_um}
        r={binSizeUm * 0.88}
        fill="none"
        vectorEffect="non-scaling-stroke"
      />
    </g>
  );
}

function SelectedBinMarker({ bin, binSizeUm }: { bin: PatchBin; binSizeUm: number }) {
  const reach = binSizeUm * 1.1;
  return (
    <g className="selected-bin-marker" pointerEvents="none" filter="url(#selected-bin-glow)">
      <rect
        x={bin.x_um - binSizeUm / 2}
        y={bin.y_um - binSizeUm / 2}
        width={binSizeUm}
        height={binSizeUm}
        fill="none"
        vectorEffect="non-scaling-stroke"
      />
      <line x1={bin.x_um - reach} x2={bin.x_um + reach} y1={bin.y_um} y2={bin.y_um} vectorEffect="non-scaling-stroke" />
      <line x1={bin.x_um} x2={bin.x_um} y1={bin.y_um - reach} y2={bin.y_um + reach} vectorEffect="non-scaling-stroke" />
    </g>
  );
}

function BinTooltip({ hovered, ownerColors }: { hovered: HoveredBin; ownerColors: Map<string, string> }) {
  const { bin, owner, category, action, clientX, clientY } = hovered;
  return (
    <div className="bin-tooltip" style={{ left: clientX + 14, top: clientY + 14 }}>
      <div className="tooltip-title">
        <i style={{ background: ownerColor(owner, ownerColors) }} />
        <strong>{bin.barcode}</strong>
      </div>
      <span>Current owner <b>{owner ?? "unassigned"}</b></span>
      <span>GT owner <b>{bin.gt_owner_cell_id ?? "none"}</b></span>
      <span>{category.replaceAll("_", " ")}</span>
      <span>row {bin.array_row} · col {bin.array_col}</span>
      {action !== null ? (
        <span className={`tooltip-action ${action.applied ? action.type : "rollback"}`}>
          {action.applied ? action.type.toUpperCase() : "ROLLBACK"} · cell {action.cell_id}
        </span>
      ) : null}
    </div>
  );
}

function CanvasLegend({ mode }: { mode: ViewMode }) {
  if (mode === "overlap") {
    const entries = [
      ["correct_owner", "Correct"],
      ["wrong_owner", "Wrong owner"],
      ["unmatched_owner", "Unmatched"],
      ["pred_only", "Pred only"],
      ["gt_only", "GT only"],
    ] as const;
    return (
      <div className="canvas-legend">
        {entries.map(([key, label]) => (
          <span key={key}><i style={{ background: OVERLAP_COLORS[key] }} />{label}</span>
        ))}
      </div>
    );
  }
  return (
    <div className="canvas-legend">
      <span><i className="legend-owner" />Assigned</span>
      <span><i className="legend-unassigned" />Unassigned</span>
      <span><i className="legend-add" />ADD</span>
      <span><i className="legend-replace" />REPLACE</span>
      <span><i className="legend-rollback" />Rollback</span>
      <span><i className="legend-transfer" />Cell → bin</span>
      <span><i className="legend-gt-outline" />GT outline</span>
    </div>
  );
}

function LayerPanel({
  showPredictedOutlines,
  showGtOutlines,
  showNuclei,
  showCoreBounds,
  onShowPredictedOutlinesChange,
  onShowGtOutlinesChange,
  onShowNucleiChange,
  onShowCoreBoundsChange,
}: {
  showPredictedOutlines: boolean;
  showGtOutlines: boolean;
  showNuclei: boolean;
  showCoreBounds: boolean;
  onShowPredictedOutlinesChange: (visible: boolean) => void;
  onShowGtOutlinesChange: (visible: boolean) => void;
  onShowNucleiChange: (visible: boolean) => void;
  onShowCoreBoundsChange: (visible: boolean) => void;
}) {
  return (
    <div className="layer-panel">
      <strong>Layers</strong>
      <LayerToggle label="Owner outlines" checked={showPredictedOutlines} onChange={onShowPredictedOutlinesChange} />
      <LayerToggle label="GT outlines" checked={showGtOutlines} onChange={onShowGtOutlinesChange} />
      <LayerToggle label="Nuclei" checked={showNuclei} onChange={onShowNucleiChange} />
      <LayerToggle label="Core bounds" checked={showCoreBounds} onChange={onShowCoreBoundsChange} />
    </div>
  );
}

function LayerToggle({ label, checked, onChange }: { label: string; checked: boolean; onChange: (checked: boolean) => void }) {
  return (
    <label className="layer-toggle">
      <input type="checkbox" checked={checked} onChange={(event) => onChange(event.target.checked)} />
      <span aria-hidden="true" />
      {label}
    </label>
  );
}

function binFill(
  bin: PatchBin,
  currentOwner: string | null,
  category: OverlapCategory,
  mode: ViewMode,
  ownerColors: Map<string, string>,
): string {
  if (mode === "overlap") {
    return OVERLAP_COLORS[category];
  }
  if (mode === "gt") {
    return ownerColor(bin.gt_owner_cell_id, ownerColors);
  }
  return ownerColor(currentOwner, ownerColors);
}

function isSelectedBin(bin: PatchBin, currentOwner: string | null, selectedCell: PatchCell | null): boolean {
  if (selectedCell === null) {
    return true;
  }
  return (
    currentOwner === selectedCell.cell_id ||
    (selectedCell.matched_gt_cell_id !== null && bin.gt_owner_cell_id === selectedCell.matched_gt_cell_id)
  );
}

function cellIdForBin(bin: PatchBin, currentOwner: string | null, cells: PatchCell[]): string | null {
  if (currentOwner !== null && cells.some((cell) => cell.cell_id === currentOwner)) {
    return currentOwner;
  }
  return cells.find((cell) => cell.matched_gt_cell_id === bin.gt_owner_cell_id)?.cell_id ?? null;
}

function boundsToViewBox(bounds: Bounds, binSizeUm: number): ViewBox {
  const pad = Math.max(binSizeUm, 1);
  return {
    x: bounds.x_min - pad,
    y: bounds.y_min - pad,
    width: bounds.x_max - bounds.x_min + pad * 2,
    height: bounds.y_max - bounds.y_min + pad * 2,
  };
}

function scaleBarLength(viewWidth: number): number {
  const ideal = viewWidth * 0.18;
  const choices = [1, 2, 5, 10, 20, 50, 100, 200];
  let selected = choices[0];
  for (const choice of choices) {
    if (choice <= ideal) {
      selected = choice;
    }
  }
  return selected;
}

function formatNumber(value: number | null): string {
  if (value === null) {
    return "n/a";
  }
  return Math.abs(value) >= 1000 ? value.toExponential(2) : value.toFixed(3);
}

function formatMetric(value: number | null): string {
  return value === null ? "n/a" : value.toFixed(3);
}

function formatCompact(value: number): string {
  return Number.isInteger(value) ? String(value) : value.toFixed(1);
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(Math.max(value, minimum), maximum);
}

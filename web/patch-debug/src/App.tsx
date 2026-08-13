import { AlertCircle, Activity, LoaderCircle } from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import { loadManifest, loadPatch, manifestUrlFromLocation } from "./api";
import { patchOwnerColorMap } from "./colors";
import { PatchCanvas } from "./components/PatchCanvas";
import { PatchInspector } from "./components/PatchInspector";
import { PatchSidebar } from "./components/PatchSidebar";
import { ownersAtStep, patchTrajectory, trajectoryStepAt } from "./trajectory";
import type {
  ActionFocusMode,
  DebugJumpKind,
  PatchCell,
  PatchManifest,
  PatchPayload,
  PatchTrajectory,
  PatchTrajectoryAction,
  StepStateMode,
  ViewMode,
} from "./types";

const MANIFEST_URL = manifestUrlFromLocation();

export default function App() {
  const [manifest, setManifest] = useState<PatchManifest | null>(null);
  const [patch, setPatch] = useState<PatchPayload | null>(null);
  const [selectedPatchId, setSelectedPatchId] = useState<string | null>(null);
  const [selectedCellId, setSelectedCellId] = useState<string | null>(null);
  const [selectedBinBarcode, setSelectedBinBarcode] = useState<string | null>(null);
  const [mode, setMode] = useState<ViewMode>("assignment");
  const [showPredictedOutlines, setShowPredictedOutlines] = useState(true);
  const [showGtOutlines, setShowGtOutlines] = useState(true);
  const [showNuclei, setShowNuclei] = useState(true);
  const [showCoreBounds, setShowCoreBounds] = useState(false);
  const [actionFocusMode, setActionFocusMode] = useState<ActionFocusMode>("selected");
  const [stepStateMode, setStepStateMode] = useState<StepStateMode>("after");
  const [stepIndex, setStepIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [playbackRate, setPlaybackRate] = useState(1);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    loadManifest(MANIFEST_URL)
      .then((result) => {
        if (cancelled) {
          return;
        }
        setManifest(result);
        setSelectedPatchId(result.patches[0]?.patch_id ?? null);
        setError(null);
      })
      .catch((cause: unknown) => {
        if (!cancelled) {
          setError(cause instanceof Error ? cause.message : String(cause));
        }
      })
      .finally(() => {
        if (!cancelled) {
          setLoading(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    const entry = manifest?.patches.find((item) => item.patch_id === selectedPatchId);
    if (entry === undefined) {
      return;
    }
    let cancelled = false;
    setLoading(true);
    loadPatch(MANIFEST_URL, entry)
      .then((result) => {
        if (!cancelled) {
          setPatch(result);
          setSelectedCellId(null);
          setSelectedBinBarcode(null);
          setStepIndex(0);
          setPlaying(false);
          setActionFocusMode("selected");
          setStepStateMode("after");
          setError(null);
        }
      })
      .catch((cause: unknown) => {
        if (!cancelled) {
          setError(cause instanceof Error ? cause.message : String(cause));
        }
      })
      .finally(() => {
        if (!cancelled) {
          setLoading(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [manifest, selectedPatchId]);

  const ownerColors = useMemo(() => patchOwnerColorMap(patch), [patch]);
  const selectedCell = patch?.cells.find((cell) => cell.cell_id === selectedCellId) ?? null;
  const trajectory = useMemo(() => (patch === null ? null : patchTrajectory(patch)), [patch]);
  const afterOwners = useMemo(
    () => (patch === null ? new Map<string, string>() : ownersAtStep(patch, stepIndex)),
    [patch, stepIndex],
  );
  const beforeOwners = useMemo(
    () =>
      patch === null
        ? new Map<string, string>()
        : ownersAtStep(patch, Math.max(0, stepIndex - 1)),
    [patch, stepIndex],
  );
  const displayedOwners = stepStateMode === "before" ? beforeOwners : afterOwners;
  const currentStep = useMemo(
    () => (patch === null ? null : trajectoryStepAt(patch, stepIndex)),
    [patch, stepIndex],
  );

  useEffect(() => {
    if (stepIndex === 0 || currentStep === null) {
      setSelectedBinBarcode(null);
      setSelectedCellId(null);
      return;
    }
    const action =
      currentStep.actions.find((candidate) => candidate.barcode === selectedBinBarcode) ??
      currentStep.actions.find((candidate) => candidate.applied) ??
      currentStep.actions[0] ??
      null;
    if (action !== null) {
      setSelectedBinBarcode(action.barcode);
      setSelectedCellId(action.cell_id);
    } else {
      setSelectedBinBarcode(null);
      setSelectedCellId(null);
    }
  }, [currentStep, stepIndex]);

  const debugJumpTargets = useMemo(
    () => (patch === null || trajectory === null ? new Map() : buildDebugJumpTargets(patch, trajectory)),
    [patch, trajectory],
  );

  useEffect(() => {
    if (!playing || trajectory === null || !trajectory.available) {
      return;
    }
    if (stepIndex >= trajectory.steps.length) {
      setPlaying(false);
      return;
    }
    const timer = window.setTimeout(
      () => setStepIndex((value) => Math.min(trajectory.steps.length, value + 1)),
      Math.max(70, 520 / playbackRate),
    );
    return () => window.clearTimeout(timer);
  }, [playbackRate, playing, stepIndex, trajectory]);

  const handlePlayingChange = (nextPlaying: boolean) => {
    if (
      nextPlaying &&
      trajectory !== null &&
      trajectory.available &&
      stepIndex >= trajectory.steps.length
    ) {
      setStepIndex(0);
    }
    setPlaying(nextPlaying);
  };

  const handleSelectCell = (cellId: string | null) => {
    setSelectedCellId(cellId);
    if (cellId === null) {
      setSelectedBinBarcode(null);
    }
  };

  const handleSelectBin = (barcode: string | null, ownerId?: string | null) => {
    setSelectedBinBarcode(barcode);
    if (ownerId !== undefined) {
      setSelectedCellId(ownerId);
    }
  };

  const handleDebugJump = (kind: DebugJumpKind) => {
    const target = debugJumpTargets.get(kind);
    if (target === undefined) {
      return;
    }
    setPlaying(false);
    setStepIndex(target.stepIndex);
    setSelectedCellId(target.cellId ?? null);
    setSelectedBinBarcode(target.barcode ?? null);
    setStepStateMode("changes");
    setActionFocusMode(target.barcode !== undefined ? "selected" : "cell");
  };

  if (error !== null && manifest === null) {
    return <FullPageError message={error} />;
  }

  const exactTraceCount =
    manifest?.patches.filter((item) => item.trajectory_available).length ?? 0;

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="product-title">
          <span className="product-mark" aria-hidden="true">
            <span />
            <span />
            <span />
            <span />
          </span>
          <div>
            <h1>HD Cell RL</h1>
            <p>Patch assignment debugger · {selectedPatchId ?? "Loading patch"}</p>
          </div>
        </div>

        <nav className="view-tabs" aria-label="Spatial view">
          {(["assignment", "overlap", "gt"] as const).map((value) => (
            <button
              key={value}
              type="button"
              className={mode === value ? "active" : ""}
              aria-current={mode === value ? "page" : undefined}
              onClick={() => setMode(value)}
            >
              {value === "assignment"
                ? "Assignment"
                : value === "overlap"
                  ? "GT overlap"
                  : "GT owners"}
            </button>
          ))}
        </nav>

        <div className="header-status">
          <span className="trace-status">
            <Activity size={14} aria-hidden="true" />
            {exactTraceCount > 0
              ? `${exactTraceCount} exact ${exactTraceCount === 1 ? "trace" : "traces"}`
              : "Final states"}
          </span>
          <span>{manifest?.n_patches ?? 0} patches</span>
          <span>{manifest?.schema_version ? `schema ${manifest.schema_version}` : "loading"}</span>
        </div>
      </header>

      <main className="workspace">
        <PatchSidebar
          patches={manifest?.patches ?? []}
          selectedPatchId={selectedPatchId}
          onSelectPatch={setSelectedPatchId}
        />
        {patch === null ? (
          <LoadingState />
        ) : (
          <PatchCanvas
            patch={patch}
            mode={mode}
            ownerColors={ownerColors}
            selectedCell={selectedCell}
            selectedBinBarcode={selectedBinBarcode}
            currentOwners={displayedOwners}
            trajectory={trajectory ?? patchTrajectory(patch)}
            currentStep={currentStep}
            actionFocusMode={actionFocusMode}
            stepStateMode={stepStateMode}
            availableDebugJumps={new Set(debugJumpTargets.keys())}
            stepIndex={stepIndex}
            playing={playing}
            playbackRate={playbackRate}
            showPredictedOutlines={showPredictedOutlines}
            showGtOutlines={showGtOutlines}
            showNuclei={showNuclei}
            showCoreBounds={showCoreBounds}
            onSelectCell={handleSelectCell}
            onSelectBin={handleSelectBin}
            onActionFocusModeChange={setActionFocusMode}
            onStepStateModeChange={setStepStateMode}
            onDebugJump={handleDebugJump}
            onStepChange={setStepIndex}
            onPlayingChange={handlePlayingChange}
            onPlaybackRateChange={setPlaybackRate}
            onShowPredictedOutlinesChange={setShowPredictedOutlines}
            onShowGtOutlinesChange={setShowGtOutlines}
            onShowNucleiChange={setShowNuclei}
            onShowCoreBoundsChange={setShowCoreBounds}
          />
        )}
        {patch === null ? (
          <div className="inspector-placeholder" />
        ) : (
          <PatchInspector
            patch={patch}
            trajectory={trajectory ?? patchTrajectory(patch)}
            currentStep={currentStep}
            currentOwners={displayedOwners}
            ownerColors={ownerColors}
            stepIndex={stepIndex}
            selectedCellId={selectedCellId}
            selectedBinBarcode={selectedBinBarcode}
            onSelectCell={handleSelectCell}
            onSelectBin={handleSelectBin}
          />
        )}
      </main>

      {loading && patch !== null ? (
        <div className="loading-indicator" role="status">
          <LoaderCircle size={15} className="spin" />
          Loading patch
        </div>
      ) : null}
      {error !== null && manifest !== null ? (
        <div className="error-toast" role="alert">
          <AlertCircle size={16} />
          {error}
        </div>
      ) : null}
    </div>
  );
}

type DebugJumpTarget = {
  stepIndex: number;
  cellId?: string;
  barcode?: string;
};

function buildDebugJumpTargets(
  patch: PatchPayload,
  trajectory: PatchTrajectory,
): Map<DebugJumpKind, DebugJumpTarget> {
  const targets = new Map<DebugJumpKind, DebugJumpTarget>();
  if (!trajectory.available || trajectory.steps.length === 0) {
    return targets;
  }

  const targetForAction = (stepOffset: number, action?: PatchTrajectoryAction) => ({
    stepIndex: stepOffset + 1,
    ...(action === undefined ? {} : { cellId: action.cell_id, barcode: action.barcode }),
  });
  const minimum = trajectory.steps.reduce(
    (best, step, index) => (step.reward < best.step.reward ? { step, index } : best),
    { step: trajectory.steps[0], index: 0 },
  );
  if (minimum.step.reward < 0) {
    targets.set(
      "largest_reward_drop",
      targetForAction(
        minimum.index,
        minimum.step.actions.find((action) => action.applied) ?? minimum.step.actions[0],
      ),
    );
  }

  const maximum = trajectory.steps.reduce(
    (best, step, index) => (step.reward > best.step.reward ? { step, index } : best),
    { step: trajectory.steps[0], index: 0 },
  );
  if (maximum.step.reward > 0) {
    targets.set(
      "largest_reward_gain",
      targetForAction(
        maximum.index,
        maximum.step.actions.find((action) => action.applied) ?? maximum.step.actions[0],
      ),
    );
  }

  const rollbackIndex = trajectory.steps.findIndex(
    (step) => step.actions.some((action) => !action.applied) || step.outcome?.includes("rollback"),
  );
  if (rollbackIndex >= 0) {
    const step = trajectory.steps[rollbackIndex];
    targets.set(
      "first_rollback",
      targetForAction(rollbackIndex, step.actions.find((action) => !action.applied) ?? step.actions[0]),
    );
  }

  const binsByBarcode = new Map(patch.bins.map((bin) => [bin.barcode, bin]));
  const matchedGtByCell = new Map(
    patch.cells.map((cell) => [cell.cell_id, cell.matched_gt_cell_id]),
  );
  for (let index = 0; index < trajectory.steps.length; index += 1) {
    const wrongReplace = trajectory.steps[index].actions.find((action) => {
      if (!action.applied || action.type !== "replace") {
        return false;
      }
      const gtOwner = binsByBarcode.get(action.barcode)?.gt_owner_cell_id ?? null;
      return gtOwner !== null && matchedGtByCell.get(action.cell_id) !== gtOwner;
    });
    if (wrongReplace !== undefined) {
      targets.set("first_wrong_replace", targetForAction(index, wrongReplace));
      break;
    }
  }

  const busiest = trajectory.steps.reduce(
    (best, step, index) =>
      step.actions.length > best.step.actions.length ? { step, index } : best,
    { step: trajectory.steps[0], index: 0 },
  );
  if (busiest.step.actions.length > 0) {
    targets.set(
      "most_actions",
      targetForAction(
        busiest.index,
        busiest.step.actions.find((action) => action.applied) ?? busiest.step.actions[0],
      ),
    );
  }

  const overgrown = maxCellBy(patch.cells, (cell) => cell.predicted_bins - cell.gt_bins, true);
  const undersegmented = maxCellBy(patch.cells, (cell) => cell.gt_bins - cell.predicted_bins, true);
  const lowestIou = maxCellBy(
    patch.cells,
    (cell) => (cell.patch_iou === null ? Number.NEGATIVE_INFINITY : -cell.patch_iou),
    false,
  );
  setCellJumpTarget(targets, "most_overgrown_cell", overgrown, trajectory);
  setCellJumpTarget(targets, "most_undersegmented_cell", undersegmented, trajectory);
  setCellJumpTarget(targets, "lowest_iou_cell", lowestIou, trajectory);
  return targets;
}

function maxCellBy(
  cells: PatchCell[],
  score: (cell: PatchCell) => number,
  requirePositive: boolean,
): PatchCell | null {
  let best: PatchCell | null = null;
  let bestScore = Number.NEGATIVE_INFINITY;
  for (const cell of cells) {
    const value = score(cell);
    if (Number.isFinite(value) && value > bestScore) {
      best = cell;
      bestScore = value;
    }
  }
  return best !== null && (!requirePositive || bestScore > 0) ? best : null;
}

function setCellJumpTarget(
  targets: Map<DebugJumpKind, DebugJumpTarget>,
  kind: DebugJumpKind,
  cell: PatchCell | null,
  trajectory: PatchTrajectory,
) {
  if (cell === null) {
    return;
  }
  for (let index = trajectory.steps.length - 1; index >= 0; index -= 1) {
    const action = trajectory.steps[index].actions.find(
      (candidate) =>
        candidate.cell_id === cell.cell_id || candidate.old_cell_id === cell.cell_id,
    );
    if (action !== undefined) {
      targets.set(kind, { stepIndex: index + 1, cellId: cell.cell_id, barcode: action.barcode });
      return;
    }
  }
  targets.set(kind, { stepIndex: trajectory.steps.length, cellId: cell.cell_id });
}

function LoadingState() {
  return (
    <div className="loading-state">
      <LoaderCircle size={22} className="spin" />
      <span>Loading patch data</span>
    </div>
  );
}

function FullPageError({ message }: { message: string }) {
  return (
    <main className="full-page-error">
      <AlertCircle size={28} />
      <h1>Patch data could not be loaded</h1>
      <p>{message}</p>
      <code>{MANIFEST_URL}</code>
    </main>
  );
}

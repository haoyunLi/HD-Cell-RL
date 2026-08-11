import { AlertCircle, Activity, LoaderCircle } from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import { loadManifest, loadPatch, manifestUrlFromLocation } from "./api";
import { patchOwnerColorMap } from "./colors";
import { PatchCanvas } from "./components/PatchCanvas";
import { PatchInspector } from "./components/PatchInspector";
import { PatchSidebar } from "./components/PatchSidebar";
import { ownersAtStep, patchTrajectory, trajectoryStepAt } from "./trajectory";
import type { PatchManifest, PatchPayload, ViewMode } from "./types";

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
  const currentOwners = useMemo(
    () => (patch === null ? new Map<string, string>() : ownersAtStep(patch, stepIndex)),
    [patch, stepIndex],
  );
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
      currentStep.actions.find((candidate) => candidate.applied) ?? currentStep.actions[0] ?? null;
    if (action !== null) {
      setSelectedBinBarcode(action.barcode);
      setSelectedCellId(action.cell_id);
    } else {
      setSelectedBinBarcode(null);
      setSelectedCellId(null);
    }
  }, [currentStep, stepIndex]);

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
            <h1>Patch assignment</h1>
            <p>{selectedPatchId ?? "Loading patch"}</p>
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
            currentOwners={currentOwners}
            trajectory={trajectory ?? patchTrajectory(patch)}
            currentStep={currentStep}
            stepIndex={stepIndex}
            playing={playing}
            playbackRate={playbackRate}
            showPredictedOutlines={showPredictedOutlines}
            showGtOutlines={showGtOutlines}
            showNuclei={showNuclei}
            showCoreBounds={showCoreBounds}
            onSelectCell={handleSelectCell}
            onSelectBin={handleSelectBin}
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
            currentOwners={currentOwners}
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

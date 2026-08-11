import {
  ChevronLeft,
  ChevronRight,
  History,
  Pause,
  Play,
  SkipBack,
  SkipForward,
} from "lucide-react";
import { useMemo } from "react";
import type { ReactNode } from "react";

import type { PatchTrajectory, PatchTrajectoryStep } from "../types";

type Props = {
  trajectory: PatchTrajectory;
  stepIndex: number;
  playing: boolean;
  playbackRate: number;
  onStepChange: (stepIndex: number) => void;
  onPlayingChange: (playing: boolean) => void;
  onPlaybackRateChange: (rate: number) => void;
};

type TimelinePaths = {
  add: string;
  replace: string;
  rejected: string;
  positiveReward: string;
  negativeReward: string;
};

const TIMELINE_WIDTH = 1000;
const REWARD_BASELINE = 68;
const REWARD_HEIGHT = 24;

export function TrajectoryControls({
  trajectory,
  stepIndex,
  playing,
  playbackRate,
  onStepChange,
  onPlayingChange,
  onPlaybackRateChange,
}: Props) {
  const paths = useMemo(() => buildTimelinePaths(trajectory.steps), [trajectory.steps]);

  if (!trajectory.available) {
    return (
      <div className="trajectory-unavailable" role="status">
        <History size={17} aria-hidden="true" />
        <div>
          <strong>Final state only</strong>
          <span>No rollout trajectory was captured for this evaluation.</span>
        </div>
      </div>
    );
  }

  const maxStep = trajectory.steps.length;
  const currentStep = stepIndex > 0 ? trajectory.steps[stepIndex - 1] : null;
  const owned =
    currentStep?.owned_target_count_after ?? trajectory.initial_owned_target_count ?? 0;
  const target = currentStep?.target_count ?? trajectory.target_count ?? 0;
  const fillRatio = target > 0 ? Math.min(1, owned / target) : 0;
  const actionCounts = countStepActions(currentStep);
  const playheadX = maxStep > 0 ? (stepIndex / maxStep) * TIMELINE_WIDTH : 0;

  return (
    <section className="trajectory-panel" aria-label="Rollout trajectory">
      <div className="trajectory-heading">
        <div>
          <h3>Trajectory</h3>
          <span>{currentStep?.phase ?? (stepIndex === 0 ? "initial seeds" : "rollout")}</span>
        </div>
        <div className="trajectory-current-event" aria-live="polite">
          <strong>{stepIndex === 0 ? "Initial" : `Step ${stepIndex}`}</strong>
          <span>{currentEventLabel(currentStep, actionCounts)}</span>
        </div>
        <span className="trajectory-step-count">
          {stepIndex} / {maxStep}
        </span>
      </div>

      <div className="target-progress">
        <span>Targets</span>
        <div className="target-progress-track" aria-hidden="true">
          <i style={{ width: `${fillRatio * 100}%` }} />
        </div>
        <strong>
          {owned} / {target || "?"}
        </strong>
      </div>

      <div className="timeline-layout">
        <div className="timeline-axis-labels" aria-hidden="true">
          <span>Events</span>
          <span>Reward</span>
        </div>
        <div className="timeline-chart">
          <svg viewBox={`0 0 ${TIMELINE_WIDTH} 96`} preserveAspectRatio="none" aria-hidden="true">
            <line className="timeline-guide" x1="0" x2={TIMELINE_WIDTH} y1="34" y2="34" />
            <line
              className="timeline-guide reward"
              x1="0"
              x2={TIMELINE_WIDTH}
              y1={REWARD_BASELINE}
              y2={REWARD_BASELINE}
            />
            <path className="timeline-event add" d={paths.add} />
            <path className="timeline-event replace" d={paths.replace} />
            <path className="timeline-event rejected" d={paths.rejected} />
            <path className="timeline-reward positive" d={paths.positiveReward} />
            <path className="timeline-reward negative" d={paths.negativeReward} />
            <line
              className="timeline-playhead"
              x1={playheadX}
              x2={playheadX}
              y1="3"
              y2="94"
            />
            <circle className="timeline-playhead-handle" cx={playheadX} cy="4" r="3.7" />
          </svg>
          <input
            className="timeline-scrubber"
            type="range"
            min={0}
            max={maxStep}
            step={1}
            value={stepIndex}
            aria-label="Trajectory step"
            aria-valuetext={stepIndex === 0 ? "Initial seeds" : `Step ${stepIndex}`}
            onChange={(event) => onStepChange(Number(event.target.value))}
          />
          <div className="timeline-ticks" aria-hidden="true">
            <span>0</span>
            <span>{Math.round(maxStep / 2)}</span>
            <span>{maxStep}</span>
          </div>
        </div>
        <div className="timeline-legend" aria-label="Trajectory event legend">
          <span><i className="add" />ADD</span>
          <span><i className="replace" />REPLACE</span>
          <span><i className="rejected" />ROLLBACK</span>
        </div>
      </div>

      <div className="trajectory-transport">
        <div className="transport-controls" aria-label="Trajectory playback">
          <TransportButton label="First step" disabled={stepIndex === 0} onClick={() => onStepChange(0)}>
            <SkipBack size={16} />
          </TransportButton>
          <TransportButton
            label="Previous step"
            disabled={stepIndex === 0}
            onClick={() => onStepChange(Math.max(0, stepIndex - 1))}
          >
            <ChevronLeft size={17} />
          </TransportButton>
          <TransportButton
            label={playing ? "Pause" : "Play"}
            disabled={maxStep === 0}
            primary
            onClick={() => onPlayingChange(!playing)}
          >
            {playing ? <Pause size={16} /> : <Play size={16} />}
          </TransportButton>
          <TransportButton
            label="Next step"
            disabled={stepIndex >= maxStep}
            onClick={() => onStepChange(Math.min(maxStep, stepIndex + 1))}
          >
            <ChevronRight size={17} />
          </TransportButton>
          <TransportButton
            label="Last step"
            disabled={stepIndex >= maxStep}
            onClick={() => onStepChange(maxStep)}
          >
            <SkipForward size={16} />
          </TransportButton>
        </div>
        <div className="transport-readout">
          <span>Step reward</span>
          <strong className={rewardTone(currentStep?.reward ?? null)}>
            {currentStep === null ? "—" : formatNumber(currentStep.reward)}
          </strong>
        </div>
        <label className="playback-rate">
          <span>Speed</span>
          <select
            value={playbackRate}
            title="Playback rate"
            onChange={(event) => onPlaybackRateChange(Number(event.target.value))}
          >
            <option value={0.5}>0.5x</option>
            <option value={1}>1x</option>
            <option value={2}>2x</option>
            <option value={4}>4x</option>
          </select>
        </label>
      </div>
    </section>
  );
}

function TransportButton({
  label,
  disabled,
  primary = false,
  onClick,
  children,
}: {
  label: string;
  disabled: boolean;
  primary?: boolean;
  onClick: () => void;
  children: ReactNode;
}) {
  return (
    <button
      className={primary ? "transport-button primary" : "transport-button"}
      type="button"
      disabled={disabled}
      title={label}
      aria-label={label}
      onClick={onClick}
    >
      {children}
    </button>
  );
}

function buildTimelinePaths(steps: PatchTrajectoryStep[]): TimelinePaths {
  const paths: TimelinePaths = {
    add: "",
    replace: "",
    rejected: "",
    positiveReward: "",
    negativeReward: "",
  };
  if (steps.length === 0) {
    return paths;
  }
  let maximumReward = 0;
  for (const step of steps) {
    maximumReward = Math.max(maximumReward, Math.abs(step.reward));
  }
  maximumReward = maximumReward || 1;

  steps.forEach((step, index) => {
    const x = ((index + 0.5) / steps.length) * TIMELINE_WIDTH;
    const flags = stepEventFlags(step);
    if (flags.add) {
      paths.add += `M${x.toFixed(2)} 8V18`;
    }
    if (flags.replace) {
      paths.replace += `M${x.toFixed(2)} 19V29`;
    }
    if (flags.rejected) {
      paths.rejected += `M${x.toFixed(2)} 30V40`;
    }
    const magnitude = Math.min(1, Math.abs(step.reward) / maximumReward) * REWARD_HEIGHT;
    const y = step.reward >= 0 ? REWARD_BASELINE - magnitude : REWARD_BASELINE + magnitude;
    const segment = `M${x.toFixed(2)} ${REWARD_BASELINE}V${y.toFixed(2)}`;
    if (step.reward >= 0) {
      paths.positiveReward += segment;
    } else {
      paths.negativeReward += segment;
    }
  });
  return paths;
}

function stepEventFlags(step: PatchTrajectoryStep) {
  return {
    add: step.actions.some((action) => action.applied && action.type === "add"),
    replace: step.actions.some((action) => action.applied && action.type === "replace"),
    rejected:
      step.outcome === "rollback" || step.actions.some((action) => !action.applied),
  };
}

function countStepActions(step: PatchTrajectoryStep | null) {
  if (step === null) {
    return { add: 0, replace: 0, rejected: 0 };
  }
  let add = 0;
  let replace = 0;
  let rejected = 0;
  for (const action of step.actions) {
    if (!action.applied) {
      rejected += 1;
    } else if (action.type === "replace") {
      replace += 1;
    } else {
      add += 1;
    }
  }
  return { add, replace, rejected };
}

function currentEventLabel(
  step: PatchTrajectoryStep | null,
  counts: { add: number; replace: number; rejected: number },
): string {
  if (step === null) {
    return "seed ownership";
  }
  const parts = [];
  if (counts.add > 0) {
    parts.push(`${counts.add} ADD`);
  }
  if (counts.replace > 0) {
    parts.push(`${counts.replace} REPLACE`);
  }
  if (counts.rejected > 0) {
    parts.push(`${counts.rejected} rollback`);
  }
  return parts.length > 0 ? parts.join(" · ") : (step.outcome?.replaceAll("_", " ") ?? "no owner change");
}

function rewardTone(value: number | null): string {
  if (value === null || value === 0) {
    return "";
  }
  return value > 0 ? "positive" : "negative";
}

function formatNumber(value: number): string {
  return Math.abs(value) >= 1000 ? value.toExponential(2) : value.toFixed(3);
}

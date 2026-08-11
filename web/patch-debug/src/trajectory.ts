import type {
  OverlapCategory,
  PatchBin,
  PatchPayload,
  PatchTrajectory,
  PatchTrajectoryStep,
} from "./types";

const FINAL_ONLY_TRAJECTORY: PatchTrajectory = {
  available: false,
  capture_status: "final_only",
  initial_patch_score: null,
  initial_raw_patch_score: null,
  initial_owned_target_count: null,
  target_count: null,
  initial_owners: [],
  final_owners: [],
  steps: [],
};

type OwnerReplayCache = {
  stepIndex: number;
  owners: Map<string, string>;
};

const OWNER_REPLAY_CACHE = new WeakMap<PatchPayload, OwnerReplayCache>();

export function patchTrajectory(patch: PatchPayload): PatchTrajectory {
  return patch.trajectory ?? FINAL_ONLY_TRAJECTORY;
}

export function ownersAtStep(patch: PatchPayload, stepIndex: number): Map<string, string> {
  const trajectory = patchTrajectory(patch);
  if (!trajectory.available) {
    return new Map(
      patch.bins
        .filter((bin) => bin.predicted_owner_cell_id !== null)
        .map((bin) => [bin.barcode, bin.predicted_owner_cell_id!]),
    );
  }

  const clampedStep = Math.min(Math.max(0, stepIndex), trajectory.steps.length);
  let cache = OWNER_REPLAY_CACHE.get(patch);
  if (cache === undefined || clampedStep < cache.stepIndex) {
    cache = {
      stepIndex: 0,
      owners: new Map(trajectory.initial_owners.map((row) => [row.barcode, row.cell_id])),
    };
    OWNER_REPLAY_CACHE.set(patch, cache);
  }
  for (let index = cache.stepIndex; index < clampedStep; index += 1) {
    for (const action of trajectory.steps[index].actions) {
      if (action.applied) {
        cache.owners.set(action.barcode, action.cell_id);
      }
    }
  }
  cache.stepIndex = clampedStep;
  return new Map(cache.owners);
}

export function trajectoryStepAt(
  patch: PatchPayload,
  stepIndex: number,
): PatchTrajectoryStep | null {
  const trajectory = patchTrajectory(patch);
  if (!trajectory.available || stepIndex <= 0 || stepIndex > trajectory.steps.length) {
    return null;
  }
  return trajectory.steps[stepIndex - 1];
}

export function dynamicOverlapCategory(
  bin: PatchBin,
  owner: string | null,
  matchedGtByOwner: Map<string, string>,
): OverlapCategory {
  const gtOwner = bin.gt_owner_cell_id;
  if (owner === null && gtOwner === null) {
    return "unscored";
  }
  if (owner === null) {
    return "gt_only";
  }
  if (gtOwner === null) {
    return "pred_only";
  }
  const matchedGt = matchedGtByOwner.get(owner);
  if (matchedGt === undefined) {
    return "unmatched_owner";
  }
  return matchedGt === gtOwner ? "correct_owner" : "wrong_owner";
}

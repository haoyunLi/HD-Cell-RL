export type Bounds = {
  x_min: number;
  x_max: number;
  y_min: number;
  y_max: number;
};

export type PatchMetrics = {
  foreground_iou: number | null;
  foreground_precision: number | null;
  foreground_recall: number | null;
  owner_accuracy: number | null;
  owner_micro_iou: number | null;
  macro_cell_iou: number | null;
  correct_owner_bins: number;
  wrong_owner_bins: number;
  unmatched_owner_bins: number;
  pred_only_bins: number;
  gt_only_bins: number;
  foreground_intersection_bins: number;
  foreground_union_bins: number;
};

export type PatchManifestEntry = {
  patch_id: string;
  file: string;
  plot: string | null;
  patch_score: number | null;
  total_reward: number | null;
  n_steps: number | null;
  n_core_cells: number;
  n_predicted_bins: number;
  n_gt_bins: number;
  trajectory_available?: boolean;
  metrics: PatchMetrics;
};

export type PatchManifest = {
  schema_version: string;
  generated_at_utc: string;
  source_eval_run_dir: string | null;
  bin_size_um: number;
  n_patches: number;
  patches: PatchManifestEntry[];
};

export type OverlapCategory =
  | "correct_owner"
  | "wrong_owner"
  | "unmatched_owner"
  | "pred_only"
  | "gt_only"
  | "unscored";

export type PatchBin = {
  barcode: string;
  array_row: number;
  array_col: number;
  x_um: number;
  y_um: number;
  predicted_owner_cell_id: string | null;
  predicted_owner_cell_ids: string[];
  predicted_matched_gt_cell_id: string | null;
  trajectory_final_owner_cell_id?: string | null;
  gt_owner_cell_id: string | null;
  gt_cell_type: string | null;
  gt_is_nuclear: boolean;
  owner_conflict: boolean;
  overlap_category: OverlapCategory;
  trace_only?: boolean;
  inside_core: boolean;
};

export type PatchTrajectoryAction = {
  type: "add" | "replace";
  cell_id: string;
  old_cell_id: string | null;
  barcode: string;
  applied: boolean;
};

export type PatchTrajectoryStep = {
  step_index: number;
  reward: number;
  cumulative_reward: number;
  patch_score_after: number | null;
  raw_patch_score_after: number | null;
  owned_target_count_after: number | null;
  target_count: number | null;
  phase: string | null;
  outcome: string | null;
  done: boolean;
  n_local_actions: number;
  n_noop_actions: number;
  actions: PatchTrajectoryAction[];
};

export type PatchTrajectory = {
  available: boolean;
  capture_status: "exact" | "final_only";
  initial_patch_score: number | null;
  initial_raw_patch_score: number | null;
  initial_owned_target_count: number | null;
  target_count: number | null;
  initial_owners: Array<{ barcode: string; cell_id: string }>;
  final_owners: Array<{ barcode: string; cell_id: string }>;
  steps: PatchTrajectoryStep[];
};

export type PatchCell = {
  cell_id: string;
  matched_gt_cell_id: string | null;
  gt_cell_type: string | null;
  nucleus_center_xy_um: [number, number] | null;
  predicted_bins: number;
  gt_bins: number;
  intersection: number;
  union: number;
  patch_iou: number | null;
  patch_dice: number | null;
  patch_precision: number | null;
  patch_recall: number | null;
  eval_iou: number | null;
  eval_dice: number | null;
  eval_precision: number | null;
  eval_recall: number | null;
  gene_spearman_r: number | null;
};

export type PatchPayload = {
  schema_version: string;
  patch_id: string;
  bin_size_um: number;
  outer_bounds: Bounds;
  core_bounds: Bounds;
  patch_score: number | null;
  total_reward: number | null;
  n_steps: number | null;
  metrics: PatchMetrics;
  counts: {
    core_cells: number;
    margin_cells: number;
    predicted_bins: number;
    gt_bins: number;
    display_bins: number;
    trace_only_bins?: number;
    owner_conflicts: number;
  };
  core_cell_ids: string[];
  margin_cell_ids: string[];
  rollout_metrics: Record<string, string | number | boolean | null>;
  trajectory?: PatchTrajectory;
  cells: PatchCell[];
  bins: PatchBin[];
};

export type ViewMode = "assignment" | "overlap" | "gt";

export type ActionFocusMode = "selected" | "cell" | "step";

export type StepStateMode = "before" | "after" | "changes";

export type DebugJumpKind =
  | "largest_reward_drop"
  | "largest_reward_gain"
  | "first_rollback"
  | "first_wrong_replace"
  | "most_actions"
  | "most_overgrown_cell"
  | "most_undersegmented_cell"
  | "lowest_iou_cell";

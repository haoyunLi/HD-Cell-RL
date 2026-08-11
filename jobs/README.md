# Slurm Jobs

Submit jobs from the repository root. Scheduler stdout and stderr are written to `logs/slurm/<job-name>-<job-id>.out` and `.err`.

## Preprocessing

`jobs/preprocessing/` contains image cropping, segmentation, bin mapping, reference construction, shape-prior construction, and episode builders.

- `run_crop_visium_hd.sbatch`
- `run_cellpose_sam.sbatch`
- `run_pixel_to_square_002um_bins.sbatch`
- `run_pseudo_pixel_to_square_002um_bins.sbatch`
- `run_merge_square_002um_expression_with_nuclear.sbatch`
- `run_build_nuclei_table.sbatch`
- `run_build_reference_counts_sct.sbatch`
- `run_build_reference_shape_features_gt.sbatch`
- `run_plot_shape_reference_tsne.sbatch`
- `run_build_shape_prior_model.sbatch`
- `run_episode_build.sbatch`
- `run_build_spatial_patch_episodes.sbatch`

## Training

`jobs/training/` contains cell-level and patch-level PPO/GRPO training launchers.

- `run_ppo_training_cpu.sbatch`
- `run_ppo_training_gpu.sbatch`
- `run_ppo_training_grpo_cpu.sbatch`
- `run_ppo_training_full_grpo_cpu.sbatch`
- `run_ppo_training_patch_full_grpo_cpu.sbatch`
- `run_ppo_training_patch_overfit4_cpu.sbatch`

## Evaluation

`jobs/evaluation/` contains checkpoint evaluation and fixed-baseline evaluation launchers.

- `run_evaluate_ppo_checkpoint.sbatch`
- `run_evaluate_patch_checkpoint.sbatch`
- `run_evaluate_patch_overfit4.sbatch`
- `run_method_ppo_eval_only.sbatch`
- `run_multicell_greedy_eval.sbatch`

## Baselines

`jobs/baselines/` contains Bin2Cell, SMURF, and STCS setup/run launchers.

- `run_bin2cell.sbatch`
- `run_smurf.sbatch`
- `setup_stcs_env.sbatch`
- `run_stcs.sbatch`

## Apps

`jobs/apps/` contains long-running interactive tools.

- `run_ppo_debug_app.sbatch`

The full execution order and common overrides remain in the root `README.md`.

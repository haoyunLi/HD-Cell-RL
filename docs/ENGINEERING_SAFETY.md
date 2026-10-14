# Correctness and engineering safeguards

This change addresses implementation defects and reproducibility. It does not change pseudo-data generation, scientific split selection, shape reference sources, reward formulas or reward weights. Existing runs and evolution exports are not rewritten.

## Correctness

- EM convergence now also requires `em_assignment.convergence.max_q_change <= 0.001` by default. Responsibility stability alone can occur before the first informative posterior update. This can change newly computed results; old artifacts retain their original meaning.
- Confident nuclear seeds are not REPLACE targets, including when EM is disabled. Previously some no-EM paths could replace them. This is an intentional correctness fix, not bit-for-bit reproduction of that bug.
- Inference and evaluation share the same confident-nuclear filter: annotated and not ambiguous. Legacy metadata without an ambiguity column is supported with a warning.
- Cross-patch checkpoint evaluation exports one owner per physical barcode. Nuclear owners take precedence; otherwise the source patch score and stable IDs break conflicts. No GT is used. This is a stitching heuristic, not a calibrated cross-patch probability. Conflicts are recorded under `subdiagnostic/merge_conflicts.csv` in the packaged evaluation source.
- A matrix directory always means its MTX/TSV contents. To use H5, pass the H5 path explicitly. Old sibling H5 files are not deleted or silently substituted. Local CSC caches validate source fingerprints and matrix dimensions before reuse/publication.

## Checkpoints and provenance

`best_model.pt` records the pre-update policy that generated the scored rollout. `final_model.pt` is always saved after the last successful update. The selection basis is recorded: a trailing training-reward mean, **not held-out validation**. Changing model-selection methodology is deferred with the scientific experiment work.

Checkpoint writes use a temporary sibling file and atomic replacement. A failed write does not destroy the previous complete checkpoint. Temporary files from failed writes are removed.

New checkpoints record feature order and environment semantics (agent mode, training unit, reward backend and rollout backend). Main evaluation loaders reject mismatches, including same-shaped tensors with different meanings. Legacy checkpoints remain loadable with an explicit warning. Cross-mode warm-start transfer for new checkpoints requires `run.allow_semantic_warm_start: true`; this loads matching tensors and is not exact training resumption. Feature dimensions are unchanged.

New run metadata records dependency versions, tracked-diff hash, hashes of tracked and untracked source/config files, and key input fingerprints. Files up to 8 MiB receive a full SHA256. Larger input fingerprints use the first/last 8 MiB plus size and timestamp, explicitly labelled as sampled—not a full integrity proof. Hashes identify code; they cannot recover unpublished source contents. Commit or archive the code before a formal run.

## Memory and naming

The H5 cell-profile path reads requested unique bins directly into CSR and avoids the previous per-cell dense bins × genes buffer. Legacy NPZ artifacts containing dense expression remain supported; their embedded arrays still load densely.

Patch contexts use a bounded LRU when caching is enabled. `patch_training.patch_context_cache_size` defaults to 128; zero disables retention. This is a context-count limit, not a byte limit. Four-patch overfit runs retain the same reuse behavior.

Global-delta objectives include both core and margin cells. New normalization names `mean_patch_cells`, `sqrt_patch_cells`, and `sum_patch_cells` describe this scope. Their existing `*_core_cells` aliases keep exactly the same values. Scientific IoU evaluation still follows its existing core-cell selection. This change does not alter either objective or metric formulas.

## CPU verification

Run from the repository root with the chosen Python environment:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
python -B -m unittest tests.test_engineering_safety tests.test_em_assignment \
  tests.test_patch_training tests.test_ppo_training tests.test_episode_artifacts \
  tests.test_ppo_eval_metrics tests.test_evaluate_patch_checkpoint
```

Tests use temporary directories, synthetic counts and CPU tensors. They do not create a formal training run, alter production matrices or publish a website. The GitHub Actions CPU workflow runs this suite on future pushes/PRs; adding the workflow does not imply that a remote CI run has already passed.

Pages publication from the overfit evaluation wrapper defaults to off. Set `PUBLISH_GITHUB_PAGES=true` only when deliberately publishing an evaluation.

## Verification completed (2026-09-08)

- Full local CPU suite: `python -B -m unittest discover -s tests` — **197 tests passed**.
- The patch-runner integration test performed one synthetic update in a temporary directory. It verified separate pre-update best and post-update final checkpoints, provenance output, and preservation of a previous-run marker. The fixture was removed on completion.
- Python syntax: 151 source/test files parsed. All job wrappers passed `bash -n`; the CI YAML parsed; `git diff --check` passed.
- Patch training, patch evaluation and EM sweep `--help` entry points succeeded.
- No production training/evaluation, pseudo-data regeneration, GitHub push or Pages publication was performed. Publication tests used temporary local Git repositories only. A fresh dependency installation and remote CI execution have not been tested.

## Files in this change

This list identifies files touched by this engineering pass, not all pre-existing workspace changes.

- EM: `hd_cell_rl/em_types.py`, `hd_cell_rl/em_assignment.py`.
- Ownership and seeds: `hd_cell_rl/nuclear_seeds.py`, `hd_cell_rl/ppo_dataset.py`, `hd_cell_rl/patch_env_torch.py`, `hd_cell_rl/patch_assignment.py`, `preprocessing/ppo_eval_metrics.py`, `scripts/evaluate_patch_checkpoint.py`.
- Loading and caching: `hd_cell_rl/matrix_io.py`, `hd_cell_rl/episode_artifacts.py`, `hd_cell_rl/patch_types.py`, `hd_cell_rl/patch_dataset.py`.
- Checkpoints and provenance: `hd_cell_rl/ppo_checkpoint.py`, `hd_cell_rl/ppo_run_io.py`, `hd_cell_rl/ppo_training.py`, `hd_cell_rl/provenance.py`, `scripts/run_patch_training.py`.
- Verification: `tests/test_engineering_safety.py`, `tests/test_patch_training.py`, `.github/workflows/cpu-tests.yml`.
- Runtime and documentation: `requirements-core.txt`, `requirements.txt`, `jobs/evaluation/run_evaluate_patch_overfit4.sbatch`, `README.md`, this document.

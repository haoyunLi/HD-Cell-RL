# Project Structure

This repository separates implementation, launchers, inputs, and generated artifacts. Keep new files inside the existing ownership boundary instead of adding them at the repository root.

## Directory Map

| Path | Purpose | Persistence |
| --- | --- | --- |
| `hd_cell_rl/` | Core RL environments, rewards, rollout, PPO/GRPO, and patch assignment code | Source-controlled |
| `configs/` | YAML configuration templates for episode building and training | Source-controlled |
| `scripts/` | User-facing Python command-line entry points | Source-controlled |
| `preprocessing/` | Data preparation, evaluation formatting, and plotting implementations | Source-controlled |
| `jobs/` | Slurm launchers, grouped by workflow stage | Source-controlled |
| `web/` | Browser applications, currently the React patch debugger | Source-controlled except build dependencies |
| `tests/` | Unit and regression tests | Source-controlled |
| `examples/` | Small runnable examples | Source-controlled |
| `docs/` | Architecture, operating notes, and archived handoffs | Source-controlled when generally useful |
| `runs/` | Timestamped training and evaluation runs, checkpoints, and run-local logs | Generated; do not edit old runs |
| `workspace_outputs/` | Reusable preprocessing outputs and baseline-method results | Generated |
| `logs/` | Central Slurm output plus archived legacy logs | Generated except documentation/placeholders |
| `external/` | Third-party source checkouts such as STCS | Local dependency |
| `cache/` | Local computed/download cache | Generated |

## Local Inputs

These directories remain at the repository root because current configs and saved run metadata refer to their existing paths:

| Path | Contents |
| --- | --- |
| `Human_Colorectal/` | Original Visium HD input and vendor output |
| `colorectal_pseudo_visium_hd_output_full_0.25/` | Pseudo-HD matrix, spatial files, and simulated assignments |
| `colorectal_sc_data/` | Colorectal scRNA reference matrices and metadata |

Do not rename these inputs in place. New workflows should expose input paths through config or environment variables so a later data migration does not require algorithm changes.

## Local Environments

`Bin2Cell_Validation/`, `bin2cell_env/`, `cellpose/`, `smurf/`, and `stcs_env/` are local Python/Conda environments. They remain at the root because installed launchers can contain absolute prefixes; moving them can silently break activation and executable shebangs.

For a clean future setup, create new environments outside the repository and point jobs to them explicitly. Do not add another environment directory at the root.

## Placement Rules

1. Put reusable Python logic in `hd_cell_rl/` or the relevant `preprocessing/` module.
2. Put CLI entry points in `scripts/`; do not add executable Python files at the root.
3. Put Slurm launchers in the matching `jobs/` category documented in `jobs/README.md`.
4. Put one experiment's outputs under `runs/<run_name>_<timestamp>/`.
5. Put reusable intermediate data under `workspace_outputs/<dataset>/intermediate/`.
6. Put generated plots inside their run or dataset output directory, not in a root `figures/` directory.
7. Let Slurm write to `logs/slurm/`; do not write `.out`, `.err`, or `.log` files into the root or `jobs/`.
8. Put long-lived project notes in `docs/` and superseded context in `docs/archive/`.

## Path-Sensitive Areas

- Historical files under `runs/` can contain absolute paths to configs, checkpoints, inputs, and episode indexes.
- Local environments can contain absolute installation prefixes.
- The Slurm launchers currently use the fixed project root `/taiga/illinois/vetmed/cb/kwang222/Haoyun_Li/RL`.

Moving any of these requires an explicit migration and validation pass. They were intentionally left unchanged by the directory cleanup.

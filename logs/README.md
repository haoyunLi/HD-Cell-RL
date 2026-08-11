# Logs

- `slurm/`: output from new Slurm submissions, named `<job-name>-<job-id>.out` and `.err`.
- `archive/root/`: legacy logs that previously lived in the repository root.
- `archive/jobs/`: legacy logs that previously lived beside the `.sbatch` files.

The archive preserves old diagnostics but is ignored by Git. New launchers should use the central `logs/slurm/` location.

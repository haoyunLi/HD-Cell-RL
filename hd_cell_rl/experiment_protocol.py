"""Shared statistical and provenance contracts for new, non-overwriting runs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

import numpy as np


def paired_cluster_mean_ci(values, clusters, *, replicates=10000, rng):
    """Resample whole patches; estimate the pooled, cell-weighted mean delta.

    Cell pairs stay together. Fewer than two observed clusters cannot identify
    between-patch uncertainty, so the interval is undefined rather than zero.
    """
    values = np.asarray(values, dtype=float)
    clusters = np.asarray(clusters)
    if values.ndim != 1 or clusters.shape != values.shape or replicates <= 0:
        raise ValueError('paired values/clusters must be aligned vectors; replicates must be positive')
    if any(v is None or str(v) in {'nan', '', 'None'} for v in clusters):
        raise ValueError('missing cluster identity')
    keep = np.isfinite(values)
    values, clusters = values[keep], clusters[keep].astype(str)
    names = np.unique(clusters)
    if len(names) < 2:
        return np.nan, np.nan
    totals = np.array([np.sum(np.sort(values[clusters == c])) for c in names])
    sizes = np.array([np.sum(clusters == c) for c in names])
    draw = rng.integers(0, len(names), size=(int(replicates), len(names)))
    means = totals[draw].sum(1) / sizes[draw].sum(1)
    low, high = np.quantile(means, [.025, .975])
    return float(low), float(high)


def write_experiment_manifest(output_dir, *, config, inputs, protocol, source_root=None):
    """Archive recoverable source bytes; input fingerprints are not data copies.

    Only project source/config/tests/docs are archived. No datasets, credentials,
    model checkpoints or Git internals are copied. Refuse existing manifests.
    """
    from .provenance import source_provenance, file_fingerprint
    root = Path(source_root or Path(__file__).resolve().parents[1]).resolve()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / 'experiment_manifest.json'
    archive = out / 'source_snapshot.tar.gz'
    if manifest_path.exists() or archive.exists():
        raise FileExistsError('refusing to overwrite experiment provenance')
    paths = subprocess.check_output(['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'], cwd=root)
    allowed_roots = {'hd_cell_rl', 'scripts', 'preprocessing', 'configs', 'tests', 'docs', '.github', 'examples', 'jobs'}
    allowed_suffixes = {'.py', '.yaml', '.yml', '.toml', '.md', '.txt', '.cjs', '.sbatch'}
    source_hashes = {}
    with tarfile.open(archive, 'x:gz') as tar:
        for raw in sorted(set(paths.split(b'\0'))):
            if not raw:
                continue
            rel = Path(raw.decode())
            if rel.parts[0] not in allowed_roots and rel.name not in {'README.md', 'requirements.txt', 'requirements-core.txt'}:
                continue
            path = root / rel
            if path.is_symlink() or not path.is_file() or path.suffix not in allowed_suffixes:
                continue
            source_hashes[str(rel)] = hashlib.sha256(path.read_bytes()).hexdigest()
            tar.add(path, arcname=str(rel), recursive=False)
    payload = {'schema_version': 1, 'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
               'source': source_provenance(), 'source_snapshot': archive.name,
               'source_snapshot_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
               'archived_source_sha256': source_hashes, 'resolved_config': config,
               'inputs': {name: file_fingerprint(path) for name, path in inputs.items()},
               'evaluation_protocol': protocol}
    with manifest_path.open('x') as handle:
        json.dump(payload, handle, indent=2, allow_nan=False, default=str)
        handle.write('\n')
    return manifest_path

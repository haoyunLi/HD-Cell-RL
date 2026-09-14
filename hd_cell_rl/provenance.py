"""Record source identity, including uncommitted code, without copying datasets."""
from __future__ import annotations

import hashlib
from importlib import metadata
from pathlib import Path
import subprocess


def file_fingerprint(path: str | Path) -> dict:
    """Full SHA256 for small files; explicitly labelled samples for large inputs."""
    path = Path(path).resolve()
    result = {"path": str(path)}
    if not path.is_file():
        result["status"] = "directory" if path.is_dir() else "missing"
        return result
    stat = path.stat()
    digest = hashlib.sha256()
    limit = 8 * 1024 * 1024
    with path.open("rb") as handle:
        digest.update(handle.read(limit))
        if stat.st_size > limit:
            handle.seek(max(limit, stat.st_size - limit))
            digest.update(handle.read(limit))
    result.update(size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
                  hash_mode="full" if stat.st_size <= limit else "first_last_8MiB",
                  sha256=digest.hexdigest())
    return result


def source_provenance() -> dict:
    root = Path(__file__).resolve().parents[1]
    result = {"source_root": str(root), "dependencies": {}}
    for package in ("numpy", "scipy", "pandas", "torch", "h5py", "pyarrow", "PyYAML"):
        try:
            result["dependencies"][package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            result["dependencies"][package] = None
    try:
        diff = subprocess.run(["git", "diff", "HEAD", "--binary"], cwd=root, check=True, capture_output=True).stdout
        paths = subprocess.run(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=root, check=True, capture_output=True).stdout
        hashes = {}
        for raw in paths.split(b"\0"):
            if not raw:
                continue
            relative = raw.decode("utf-8", errors="surrogateescape")
            path = root / relative
            if path.suffix in {".py", ".yaml", ".yml", ".toml", ".sbatch"} and path.is_file():
                hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        result.update(dirty_diff_sha256=hashlib.sha256(diff).hexdigest(),
                      has_tracked_changes=bool(diff), source_sha256=hashes)
    except (OSError, subprocess.CalledProcessError) as exc:
        result["git_error"] = str(exc)
    return result

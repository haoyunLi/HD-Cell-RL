#!/usr/bin/env python
"""Publish one patch-debug evaluation as a single-commit GitHub Pages snapshot."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tempfile
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
PAGES_BRANCH = "gh-pages"
DEFAULT_EVAL_GLOB = "human_colorectal_patch_overfit4_eval_*"


class PublishError(RuntimeError):
    """Raised when a Pages snapshot is incomplete or unsafe to publish."""


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-run-dir",
        type=Path,
        default=None,
        help="Completed evaluation run. Defaults to the newest run with patch_debug/manifest.json.",
    )
    parser.add_argument("--eval-root", type=Path, default=REPO_ROOT / "runs")
    parser.add_argument("--eval-glob", default=DEFAULT_EVAL_GLOB)
    parser.add_argument("--frontend-dir", type=Path, default=REPO_ROOT / "web" / "patch-debug")
    parser.add_argument("--remote", default="origin")
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=None,
        help="Also copy the prepared site here for inspection; the destination must not exist.",
    )
    parser.add_argument("--skip-build", action="store_true", help="Reuse the existing frontend dist directory.")
    parser.add_argument(
        "--push",
        action="store_true",
        help=f"Force-push the new single root commit to the protected {PAGES_BRANCH} branch.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    eval_run_dir = _resolve_eval_run(
        eval_run_dir=args.eval_run_dir,
        eval_root=args.eval_root,
        eval_glob=str(args.eval_glob),
    )
    patch_debug_dir = eval_run_dir / "patch_debug"
    manifest = _load_and_validate_manifest(patch_debug_dir)

    frontend_dir = args.frontend_dir.expanduser().resolve()
    if not args.skip_build:
        _run(("npm", "run", "build"), cwd=frontend_dir)
    frontend_dist = frontend_dir / "dist"
    _validate_frontend_dist(frontend_dist)

    with tempfile.TemporaryDirectory(prefix="patch-debug-pages-") as temp_dir:
        site_dir = Path(temp_dir) / "site"
        _build_snapshot(
            frontend_dist=frontend_dist,
            patch_debug_dir=patch_debug_dir,
            eval_run_dir=eval_run_dir,
            manifest=manifest,
            site_dir=site_dir,
        )
        if args.snapshot_dir is not None:
            snapshot_dir = args.snapshot_dir.expanduser().resolve()
            if snapshot_dir.exists():
                raise PublishError(f"snapshot destination already exists: {snapshot_dir}")
            shutil.copytree(site_dir, snapshot_dir)
            print(f"Snapshot copied: {snapshot_dir}")

        print(f"Prepared Pages snapshot: {eval_run_dir}")
        if not args.push:
            print("Push skipped; pass --push to replace the remote gh-pages snapshot.")
            return

        commit = _create_orphan_commit(site_dir=site_dir, eval_run_name=eval_run_dir.name)
        remote_url = _remote_url(repo_root=REPO_ROOT, remote=str(args.remote))
        _push_pages_commit(site_dir=site_dir, remote_url=remote_url)
        print(f"Published {eval_run_dir.name} as {PAGES_BRANCH} commit {commit}")


def _resolve_eval_run(*, eval_run_dir: Path | None, eval_root: Path, eval_glob: str) -> Path:
    if eval_run_dir is not None:
        candidate = eval_run_dir.expanduser().resolve()
        if not candidate.is_dir():
            raise PublishError(f"evaluation run directory not found: {candidate}")
        return candidate

    root = eval_root.expanduser().resolve()
    candidates = [
        path.resolve()
        for path in root.glob(eval_glob)
        if path.is_dir() and (path / "patch_debug" / "manifest.json").is_file()
    ]
    if not candidates:
        raise PublishError(f"no completed patch-debug evaluations match {root / eval_glob}")
    return max(candidates, key=lambda path: path.name)


def _load_and_validate_manifest(patch_debug_dir: Path) -> dict[str, Any]:
    bundle_root = patch_debug_dir.expanduser().resolve()
    manifest_path = bundle_root / "manifest.json"
    if not manifest_path.is_file():
        raise PublishError(f"patch debug manifest not found: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PublishError(f"invalid patch debug manifest: {manifest_path}") from exc
    if not isinstance(manifest, dict):
        raise PublishError("patch debug manifest must be a JSON object")

    patches = manifest.get("patches")
    if not isinstance(patches, list) or not patches:
        raise PublishError("patch debug manifest contains no patches")
    try:
        n_patches = int(manifest.get("n_patches", -1))
    except (TypeError, ValueError) as exc:
        raise PublishError("patch debug manifest n_patches must be an integer") from exc
    if n_patches != len(patches):
        raise PublishError("patch debug manifest n_patches does not match patches")

    patch_ids: set[str] = set()
    for index, entry in enumerate(patches):
        if not isinstance(entry, dict):
            raise PublishError(f"patches[{index}] must be a JSON object")
        patch_id = str(entry.get("patch_id", "")).strip()
        if not patch_id or patch_id in patch_ids:
            raise PublishError(f"patches[{index}] has a missing or duplicate patch_id")
        patch_ids.add(patch_id)
        _validate_bundle_json(bundle_root, entry.get("file"), field=f"patches[{index}].file")
        plot = entry.get("plot")
        if plot is not None:
            _validate_bundle_file(bundle_root, plot, field=f"patches[{index}].plot")
    return manifest


def _validate_bundle_json(bundle_root: Path, raw_path: Any, *, field: str) -> None:
    path = _validate_bundle_file(bundle_root, raw_path, field=field)
    try:
        json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PublishError(f"{field} is not valid JSON: {path}") from exc


def _validate_bundle_file(bundle_root: Path, raw_path: Any, *, field: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise PublishError(f"{field} must be a non-empty relative path")
    relative = PurePosixPath(raw_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise PublishError(f"{field} must stay inside patch_debug: {raw_path}")
    resolved = (bundle_root / Path(*relative.parts)).resolve()
    if not resolved.is_relative_to(bundle_root) or not resolved.is_file():
        raise PublishError(f"{field} does not reference a bundle file: {raw_path}")
    if resolved.is_symlink():
        raise PublishError(f"{field} must not be a symbolic link: {raw_path}")
    return resolved


def _validate_frontend_dist(frontend_dist: Path) -> None:
    dist = frontend_dist.expanduser().resolve()
    index_path = dist / "index.html"
    if not index_path.is_file():
        raise PublishError(f"frontend build not found: {index_path}")
    index_html = index_path.read_text(encoding="utf-8")
    if 'src="/assets/' in index_html or 'href="/assets/' in index_html:
        raise PublishError("frontend build uses root-absolute assets and will fail on project GitHub Pages")
    _reject_symlinks(dist)


def _build_snapshot(
    *,
    frontend_dist: Path,
    patch_debug_dir: Path,
    eval_run_dir: Path,
    manifest: dict[str, Any],
    site_dir: Path,
) -> None:
    _reject_symlinks(patch_debug_dir)
    shutil.copytree(frontend_dist, site_dir)
    published_bundle = site_dir / "patch_debug"
    shutil.copytree(patch_debug_dir, published_bundle)
    sanitized_manifest = _sanitize_manifest(manifest, eval_run_name=eval_run_dir.name)
    (published_bundle / "manifest.json").write_text(
        json.dumps(sanitized_manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    (site_dir / ".nojekyll").touch()
    _reject_internal_json_paths(site_dir)


def _sanitize_manifest(manifest: dict[str, Any], *, eval_run_name: str) -> dict[str, Any]:
    sanitized = deepcopy(manifest)
    for key, value in list(sanitized.items()):
        if not str(key).startswith("source_") or not isinstance(value, str):
            continue
        sanitized[key] = eval_run_name if key == "source_eval_run_dir" else Path(value).name
    return sanitized


def _reject_symlinks(root: Path) -> None:
    if root.is_symlink():
        raise PublishError(f"publication source must not be a symbolic link: {root}")
    for path in root.rglob("*"):
        if path.is_symlink():
            raise PublishError(f"publication source contains a symbolic link: {path}")


def _reject_internal_json_paths(site_dir: Path) -> None:
    forbidden = ("/taiga/", "/u/haoyunli/")
    for path in site_dir.rglob("*.json"):
        text = path.read_text(encoding="utf-8")
        if any(prefix in text for prefix in forbidden):
            raise PublishError(f"publication JSON contains an internal absolute path: {path}")


def _create_orphan_commit(*, site_dir: Path, eval_run_name: str) -> str:
    _run(("git", "init", "--quiet", f"--initial-branch={PAGES_BRANCH}"), cwd=site_dir)
    _run(("git", "add", "--all"), cwd=site_dir)
    env = os.environ.copy()
    env.update(
        {
            "GIT_AUTHOR_NAME": "Patch Debug Publisher",
            "GIT_AUTHOR_EMAIL": "patch-debug@users.noreply.github.com",
            "GIT_COMMITTER_NAME": "Patch Debug Publisher",
            "GIT_COMMITTER_EMAIL": "patch-debug@users.noreply.github.com",
        }
    )
    _run(("git", "commit", "--quiet", "-m", f"deploy: {eval_run_name}"), cwd=site_dir, env=env)
    count = _capture(("git", "rev-list", "--count", "HEAD"), cwd=site_dir)
    if count != "1":
        raise PublishError(f"refusing to publish gh-pages history with {count} commits")
    parent_line = _capture(("git", "rev-list", "--parents", "-n", "1", "HEAD"), cwd=site_dir)
    if len(parent_line.split()) != 1:
        raise PublishError("refusing to publish a gh-pages commit with a parent")
    return _capture(("git", "rev-parse", "HEAD"), cwd=site_dir)


def _remote_url(*, repo_root: Path, remote: str) -> str:
    if not remote or remote.startswith("-"):
        raise PublishError(f"invalid git remote name: {remote!r}")
    return _capture(("git", "remote", "get-url", remote), cwd=repo_root)


def _push_pages_commit(*, site_dir: Path, remote_url: str) -> None:
    if not remote_url:
        raise PublishError("git remote URL is empty")
    _run(
        ("git", "push", "--force", remote_url, f"HEAD:refs/heads/{PAGES_BRANCH}"),
        cwd=site_dir,
    )


def _run(command: Sequence[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    try:
        subprocess.run(list(command), cwd=cwd, env=env, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PublishError(f"command failed: {' '.join(command)}") from exc


def _capture(command: Sequence[str], *, cwd: Path) -> str:
    try:
        result = subprocess.run(list(command), cwd=cwd, check=True, text=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PublishError(f"command failed: {' '.join(command)}") from exc
    return result.stdout.strip()


if __name__ == "__main__":
    main()

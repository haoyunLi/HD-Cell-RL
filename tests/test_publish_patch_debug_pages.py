from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from scripts import publish_patch_debug_pages as pages


class PublishPatchDebugPagesTest(unittest.TestCase):
    def test_builds_sanitized_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eval_run, manifest = _write_eval_run(root, "eval_20260810T213100Z")
            frontend_dist = _write_frontend_dist(root)
            site_dir = root / "site"

            pages._build_snapshot(
                frontend_dist=frontend_dist,
                patch_debug_dir=eval_run / "patch_debug",
                eval_run_dir=eval_run,
                manifest=manifest,
                site_dir=site_dir,
            )

            published = json.loads((site_dir / "patch_debug" / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(published["source_eval_run_dir"], eval_run.name)
            self.assertEqual(published["source_patch_index"], "patches.csv")
            self.assertTrue((site_dir / "index.html").is_file())
            self.assertTrue((site_dir / "assets" / "app.js").is_file())
            self.assertTrue((site_dir / "patch_debug" / "patches" / "p1.json").is_file())
            self.assertTrue((site_dir / ".nojekyll").is_file())
            self.assertNotIn("/taiga/", (site_dir / "patch_debug" / "manifest.json").read_text(encoding="utf-8"))

    def test_selects_newest_completed_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_eval_run(root, "eval_20260810T213000Z")
            expected, _ = _write_eval_run(root, "eval_20260810T213100Z")
            incomplete = root / "eval_incomplete"
            incomplete.mkdir()

            selected = pages._resolve_eval_run(
                eval_run_dir=None,
                eval_root=root,
                eval_glob="eval_*",
            )

            self.assertEqual(selected, expected.resolve())

    def test_rejects_manifest_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eval_run, _ = _write_eval_run(root, "eval_bad")
            manifest_path = eval_run / "patch_debug" / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["patches"][0]["file"] = "../outside.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(pages.PublishError, "must stay inside patch_debug"):
                pages._load_and_validate_manifest(eval_run / "patch_debug")

    def test_orphan_commit_has_no_parent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            site_dir = Path(tmp) / "site"
            site_dir.mkdir()
            (site_dir / "index.html").write_text("ready\n", encoding="utf-8")

            commit = pages._create_orphan_commit(site_dir=site_dir, eval_run_name="eval_test")

            count = subprocess.check_output(
                ["git", "rev-list", "--count", "HEAD"],
                cwd=site_dir,
                text=True,
            ).strip()
            parents = subprocess.check_output(
                ["git", "rev-list", "--parents", "-n", "1", "HEAD"],
                cwd=site_dir,
                text=True,
            ).split()
            branch = subprocess.check_output(
                ["git", "branch", "--show-current"],
                cwd=site_dir,
                text=True,
            ).strip()
            self.assertEqual(count, "1")
            self.assertEqual(parents, [commit])
            self.assertEqual(branch, "gh-pages")

    def test_force_push_replaces_reachable_pages_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            remote = root / "remote.git"
            subprocess.run(["git", "init", "--bare", "--quiet", str(remote)], check=True)

            first_site = _write_site(root / "first", "first")
            first_commit = pages._create_orphan_commit(site_dir=first_site, eval_run_name="eval_first")
            pages._push_pages_commit(site_dir=first_site, remote_url=str(remote))

            second_site = _write_site(root / "second", "second")
            second_commit = pages._create_orphan_commit(site_dir=second_site, eval_run_name="eval_second")
            pages._push_pages_commit(site_dir=second_site, remote_url=str(remote))

            reachable = subprocess.check_output(
                ["git", f"--git-dir={remote}", "rev-list", "gh-pages"],
                text=True,
            ).split()
            self.assertEqual(reachable, [second_commit])
            self.assertNotIn(first_commit, reachable)


def _write_eval_run(root: Path, name: str) -> tuple[Path, dict[str, object]]:
    eval_run = root / name
    patch_debug = eval_run / "patch_debug"
    (patch_debug / "patches").mkdir(parents=True)
    (patch_debug / "plots").mkdir()
    (patch_debug / "patches" / "p1.json").write_text('{"patch_id": "p1"}\n', encoding="utf-8")
    (patch_debug / "plots" / "p1.png").write_bytes(b"png")
    manifest: dict[str, object] = {
        "schema_version": "1.1",
        "source_eval_run_dir": f"/taiga/private/{name}",
        "source_patch_index": "/taiga/private/patches.csv",
        "n_patches": 1,
        "patches": [
            {
                "patch_id": "p1",
                "file": "patches/p1.json",
                "plot": "plots/p1.png",
            }
        ],
    }
    (patch_debug / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return eval_run, manifest


def _write_frontend_dist(root: Path) -> Path:
    dist = root / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text('<script src="./assets/app.js"></script>\n', encoding="utf-8")
    (dist / "assets" / "app.js").write_text("console.log('ready');\n", encoding="utf-8")
    return dist


def _write_site(path: Path, content: str) -> Path:
    path.mkdir()
    (path / "index.html").write_text(f"{content}\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    unittest.main()

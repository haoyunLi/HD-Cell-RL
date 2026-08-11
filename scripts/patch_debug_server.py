#!/usr/bin/env python
"""Serve the React patch debug UI and one evaluation run's patch data."""

from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import sys
from urllib.parse import unquote, urlparse


REPO_ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-run-dir", required=True)
    parser.add_argument("--frontend-dist", default=str(REPO_ROOT / "web" / "patch-debug" / "dist"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8501)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    eval_run_dir = Path(args.eval_run_dir).expanduser().resolve()
    patch_debug_dir = eval_run_dir / "patch_debug"
    frontend_dist = Path(args.frontend_dist).expanduser().resolve()
    manifest_path = patch_debug_dir / "manifest.json"
    index_path = frontend_dist / "index.html"
    if not manifest_path.exists():
        raise FileNotFoundError(f"patch debug manifest not found: {manifest_path}")
    if not index_path.exists():
        raise FileNotFoundError(f"React frontend build not found: {index_path}")
    if int(args.port) <= 0 or int(args.port) > 65535:
        raise ValueError("--port must be in [1, 65535]")

    handler = _build_handler(frontend_dist=frontend_dist, patch_debug_dir=patch_debug_dir)
    server = ThreadingHTTPServer((str(args.host), int(args.port)), handler)
    print(f"Patch debug app: http://{args.host}:{args.port}", flush=True)
    print(f"Evaluation run: {eval_run_dir}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def _build_handler(*, frontend_dist: Path, patch_debug_dir: Path) -> type[BaseHTTPRequestHandler]:
    class PatchDebugRequestHandler(BaseHTTPRequestHandler):
        server_version = "PatchDebugHTTP/1.0"

        def do_GET(self) -> None:
            self._serve(head_only=False)

        def do_HEAD(self) -> None:
            self._serve(head_only=True)

        def _serve(self, *, head_only: bool) -> None:
            request_path = unquote(urlparse(self.path).path)
            if request_path == "/healthz":
                self._send_json({"status": "ok"}, head_only=head_only)
                return
            if request_path.startswith("/patch_debug/"):
                relative = request_path.removeprefix("/patch_debug/")
                try:
                    target = _safe_resolve(patch_debug_dir, relative)
                except ValueError:
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                self._send_file(target, head_only=head_only, no_store=target.suffix == ".json")
                return
            if request_path == "/patch_debug":
                self.send_response(HTTPStatus.PERMANENT_REDIRECT)
                self.send_header("Location", "/")
                self.end_headers()
                return

            relative = request_path.lstrip("/")
            try:
                target = _safe_resolve(frontend_dist, relative)
            except ValueError:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            if request_path == "/" or not target.is_file():
                target = frontend_dist / "index.html"
            self._send_file(target, head_only=head_only, no_store=target.name == "index.html")

        def _send_json(self, payload: dict[str, object], *, head_only: bool) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if not head_only:
                self.wfile.write(body)

        def _send_file(self, path: Path, *, head_only: bool, no_store: bool) -> None:
            if not path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            content_type, _ = mimetypes.guess_type(path.name)
            body = path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type or "application/octet-stream")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store" if no_store else "public, max-age=3600")
            self.end_headers()
            if not head_only:
                self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            sys.stdout.write(f"{self.address_string()} - {format % args}\n")
            sys.stdout.flush()

    return PatchDebugRequestHandler


def _safe_resolve(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError("requested path escapes the server root")
    return candidate


if __name__ == "__main__":
    main()

"""Build MkDocs documentation locally."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _ensure_mkdocs() -> None:
    if shutil.which("mkdocs") is None:
        msg = (
            "mkdocs is not available. Install docs dependencies with "
            "`pip install -e '.[docs]'`."
        )
        raise RuntimeError(msg)


def _build(args: argparse.Namespace) -> None:
    root = _repo_root()
    config_path = _resolve_path(root, args.config)
    site_dir = _resolve_path(root, args.site_dir) if args.site_dir else None

    cmd = ["mkdocs", "build", "-f", str(config_path)]
    if args.clean:
        cmd.append("--clean")
    if args.strict:
        cmd.append("--strict")
    if site_dir is not None:
        cmd.extend(["-d", str(site_dir)])

    subprocess.run(cmd, cwd=root, check=True)


def _serve(args: argparse.Namespace) -> None:
    root = _repo_root()
    config_path = _resolve_path(root, args.config)

    cmd = ["mkdocs", "serve", "-f", str(config_path)]
    if args.dev_addr:
        cmd.extend(["--dev-addr", args.dev_addr])

    subprocess.run(cmd, cwd=root, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build or serve MkDocs documentation.")
    parser.add_argument(
        "--config",
        default="docs/mkdocs.yml",
        help="Path to mkdocs.yml (default: docs/mkdocs.yml).",
    )
    parser.add_argument(
        "--site-dir",
        default="site",
        help="Output directory for built site (default: site/).",
    )
    parser.add_argument("--clean", action="store_true", help="Clean build directory first.")
    parser.add_argument("--strict", action="store_true", help="Enable strict mode.")
    parser.add_argument("--serve", action="store_true", help="Serve docs with live reload.")
    parser.add_argument(
        "--dev-addr",
        default="",
        help="Address for mkdocs serve, e.g. 127.0.0.1:8000.",
    )
    args = parser.parse_args()

    try:
        _ensure_mkdocs()
        if args.serve:
            _serve(args)
        else:
            _build(args)
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

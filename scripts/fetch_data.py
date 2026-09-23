#!/usr/bin/env python3
"""Reproduce every download listed in scripts/datasets.json and verify its SHA-256.

Idempotent: a file whose SHA-256 already matches the manifest is skipped. Uses only the
standard library so it can run before `uv sync` (the venv does not need to exist yet).

Usage:
    python scripts/fetch_data.py                # fetch/verify everything
    python scripts/fetch_data.py --dataset banc-626   # only one dataset
    python scripts/fetch_data.py --check-only    # verify existing files, download nothing
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / "scripts" / "datasets.json"
CHUNK_SIZE = 8 * 1024 * 1024  # 8 MiB


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  downloading {url}")
    with urllib.request.urlopen(url, timeout=60) as response, tmp.open("wb") as fh:
        while chunk := response.read(CHUNK_SIZE):
            fh.write(chunk)
    tmp.rename(dest)


def process_entry(entry: dict, *, check_only: bool) -> bool:
    """Returns True if the file is present and verified after this call."""
    dest = REPO_ROOT / entry["path"]
    expected_sha256 = entry["sha256"]
    expected_size = entry["size_bytes"]

    if dest.exists() and dest.stat().st_size == expected_size:
        actual_sha256 = sha256_of(dest)
        if actual_sha256 == expected_sha256:
            print(f"OK    {entry['path']}")
            return True
        print(f"STALE {entry['path']}: sha256 {actual_sha256} != {expected_sha256}")
    elif dest.exists():
        print(f"STALE {entry['path']}: size {dest.stat().st_size} != {expected_size}")

    if check_only:
        print(f"MISS  {entry['path']} (not fetched, --check-only)")
        return False

    download(entry["url"], dest)
    actual_size = dest.stat().st_size
    actual_sha256 = sha256_of(dest)
    if actual_size != expected_size or actual_sha256 != expected_sha256:
        print(
            f"FAIL  {entry['path']}: got size={actual_size} sha256={actual_sha256}, "
            f"expected size={expected_size} sha256={expected_sha256}"
        )
        return False
    print(f"FETCHED {entry['path']}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", help="only process entries whose 'dataset' field matches")
    parser.add_argument(
        "--check-only", action="store_true", help="verify existing files, do not download"
    )
    args = parser.parse_args()

    manifest = json.loads(MANIFEST_PATH.read_text())
    entries = manifest["files"]
    if args.dataset:
        entries = [e for e in entries if e["dataset"] == args.dataset]
        if not entries:
            print(f"no manifest entries for dataset={args.dataset!r}", file=sys.stderr)
            return 2

    ok = True
    for entry in entries:
        ok &= process_entry(entry, check_only=args.check_only)

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

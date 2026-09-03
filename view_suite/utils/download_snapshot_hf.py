#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fire-based CLI to snapshot a Hugging Face repo into a directory.

The companion to download_targz_hf: use that one when the payload is archives to
unpack, this one when the repo *is* the directory layout you want on disk (e.g.
the Habitat-GS splat scenes, which are served in place).
"""

import os

# Must be set BEFORE importing huggingface_hub -- both are frozen into module
# constants at import time, so setting them afterwards has no effect.
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")

from pathlib import Path

import fire
from huggingface_hub import snapshot_download


def cli(
    repo: str,
    out: str,
    allow: str | None = None,
    repo_type: str = "dataset",
    revision: str = "main",
    token: str | None = None,
    max_workers: int = 8,
) -> None:
    """
    Download a repo snapshot into `out`.

    Args:
      repo: HF repo_id, e.g. "RukawaY/gs_scenes"
      out: Directory to populate
      allow: Comma-separated allow_patterns, e.g. "train/**,val/**". All files if unset.
      repo_type: "dataset" | "model" | "space" (default: dataset)
      revision: Git branch/tag/sha (default: main)
      token: HF access token (defaults to $HF_TOKEN if not provided)
      max_workers: Parallel file downloads
    """
    out_dir = Path(out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    patterns = [p.strip() for p in allow.split(",") if p.strip()] if allow else None

    print(f"Downloading {repo} -> {out_dir}")
    if patterns:
        print(f"  allow_patterns: {patterns}")
    path = snapshot_download(
        repo,
        repo_type=repo_type,
        revision=revision,
        local_dir=str(out_dir),
        allow_patterns=patterns,
        max_workers=max_workers,
        token=token or os.environ.get("HF_TOKEN"),
    )
    print(f"Done. Contents are under: {path}")


if __name__ == "__main__":
    fire.Fire(cli)

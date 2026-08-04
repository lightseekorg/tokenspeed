"""Resolve native MSA sources for runtime JIT compilation."""

from pathlib import Path


def msa_source_dir() -> Path:
    """Return the MSA sources from the unified CUDA source tree."""
    return Path(__file__).resolve().parents[5] / "csrc" / "cuda" / "msa"


__all__ = ["msa_source_dir"]

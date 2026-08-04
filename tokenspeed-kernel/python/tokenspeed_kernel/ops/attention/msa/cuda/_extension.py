"""Load setup-built MSA PyTorch CUDA extensions."""

from __future__ import annotations

import importlib.util
import sys
import sysconfig
import threading
from functools import cache
from pathlib import Path

import torch  # noqa: F401 - load PyTorch shared libraries before the extension

_LOAD_LOCK = threading.Lock()


@cache
def load_extension(name: str):
    """Load an AOT pybind extension from the MSA CUDA implementation."""
    with _LOAD_LOCK:
        suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
        path = Path(__file__).resolve().parent / "objs" / name / f"{name}{suffix}"
        if not path.is_file():
            raise RuntimeError(
                f"tokenspeed_kernel MSA CUDA extension not found at {path}. "
                "Reinstall tokenspeed-kernel with the CUDA backend."
            )

        qualified_name = f"{__package__}.{name}"
        spec = importlib.util.spec_from_file_location(qualified_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot create an import specification for {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[qualified_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(qualified_name, None)
            raise
        return module

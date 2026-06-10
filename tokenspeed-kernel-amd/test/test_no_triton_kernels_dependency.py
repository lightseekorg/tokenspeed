from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_fused_mxfp_gfx950_imports_without_triton_kernels():
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    package_path = str(root / "python")
    env["PYTHONPATH"] = os.pathsep.join(
        [package_path, env["PYTHONPATH"]] if env.get("PYTHONPATH") else [package_path]
    )
    script = r"""
import builtins

real_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name == "triton_kernels" or name.startswith("triton_kernels."):
        raise AssertionError(f"unexpected import: {name}")
    return real_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
from tokenspeed_kernel_amd.ops.moe import fused_mxfp_gfx950

assert fused_mxfp_gfx950.RaggedTensorMetadata.block_sizes()
"""
    subprocess.run([sys.executable, "-c", script], env=env, check=True)

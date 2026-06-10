from __future__ import annotations

import ast
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


def _python_files():
    root = Path(__file__).resolve().parents[1]
    return sorted(path for path in root.rglob("*.py") if ".venv" not in path.parts)


def _is_tokenspeed_kernel_import(module: str | None) -> bool:
    return module == "tokenspeed_kernel" or (
        module is not None and module.startswith("tokenspeed_kernel.")
    )


def test_no_tokenspeed_kernel_imports_in_amd_package():
    violations = []
    for path in _python_files():
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if _is_tokenspeed_kernel_import(alias.name):
                        violations.append(f"{path}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if _is_tokenspeed_kernel_import(node.module):
                    violations.append(f"{path}: from {node.module} import ...")

    assert violations == []

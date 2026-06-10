from __future__ import annotations

import ast
from pathlib import Path


def _python_files():
    root = Path(__file__).resolve().parents[1]
    return sorted(path for path in root.rglob("*.py") if ".venv" not in path.parts)


def _is_module_import(module: str | None, module_root: str) -> bool:
    return module == module_root or (
        module is not None and module.startswith(f"{module_root}.")
    )


def _find_imports(module_root: str):
    violations = []
    for path in _python_files():
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if _is_module_import(alias.name, module_root):
                        violations.append(f"{path}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if _is_module_import(node.module, module_root):
                    violations.append(f"{path}: from {node.module} import ...")
    return violations


def test_no_triton_kernels_imports_in_amd_package():
    assert _find_imports("triton_kernels") == []


def test_no_tokenspeed_kernel_imports_in_amd_package():
    assert _find_imports("tokenspeed_kernel") == []

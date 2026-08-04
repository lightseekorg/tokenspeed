# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Persistent AOT cache for CuTe DSL compiled kernels.

Saves compiled TVM FFI kernels as .o files on first compile,
loads them on subsequent runs to skip JIT compilation.

Environment variables:
    MM_SPARSE_ATTN_AOT_CACHE: Override cache directory
        (default: ~/.cache/minfer/mm_sparse_attn)
    MM_SPARSE_ATTN_AOT_DISABLE=1: Disable AOT cache entirely
"""

import hashlib
import os
import time

import cutlass.cute as cute

_AOT_CACHE_DIR = os.environ.get(
    "MM_SPARSE_ATTN_AOT_CACHE",
    os.path.expanduser("~/.cache/minfer/mm_sparse_attn"),
)
_AOT_DISABLE = os.environ.get("MM_SPARSE_ATTN_AOT_DISABLE", "0") == "1"

_loaded_modules: dict[str, object] = {}


def _key_to_path(key: tuple) -> str:
    h = hashlib.sha256(repr(key).encode()).hexdigest()[:16]
    name = str(key[0]).replace("/", "_")
    return os.path.join(_AOT_CACHE_DIR, f"{name}_{h}")


def try_load_aot(key: tuple):
    if _AOT_DISABLE:
        return None
    obj_path = _key_to_path(key) + ".o"
    if not os.path.isfile(obj_path):
        return None
    func_name = str(key[0])
    try:
        if obj_path not in _loaded_modules:
            _loaded_modules[obj_path] = cute.runtime.load_module(
                obj_path, enable_tvm_ffi=True
            )
        return getattr(_loaded_modules[obj_path], func_name)
    except Exception as e:
        print(f"[aot_cache] Failed to load {obj_path}: {e}")
        return None


def save_aot(key: tuple, compiled) -> None:
    if _AOT_DISABLE:
        return
    if not hasattr(compiled, "export_to_c"):
        return
    obj_path = _key_to_path(key) + ".o"
    os.makedirs(_AOT_CACHE_DIR, exist_ok=True)
    tmp_path = obj_path + f".tmp.{os.getpid()}"
    func_name = str(key[0])
    try:
        t0 = time.time()
        compiled.export_to_c(tmp_path, function_name=func_name)
        os.replace(tmp_path, obj_path)
        dt = time.time() - t0
        print(f"[aot_cache] Saved {func_name} -> {obj_path} ({dt:.1f}s)")
    except Exception as e:
        print(f"[aot_cache] Failed to save {func_name}: {e}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

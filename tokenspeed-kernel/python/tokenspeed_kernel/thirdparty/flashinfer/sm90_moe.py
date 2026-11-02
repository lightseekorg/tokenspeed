# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Private, source-versioned SM90 block-FP8 MoE modules for FlashInfer 0.7.2.post1."""

from __future__ import annotations

import ast
import functools
import hashlib
import json
import os
import re
import shutil
import threading
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Literal

from filelock import FileLock

MoEMode = Literal["default", "expected_m", "optimized"]
_PATCH = Path(__file__).with_name("sm90_moe.patch")
_HUNK = re.compile(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_ENTRYPOINT_LOCK = threading.Lock()


def _replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise RuntimeError(
            "Unsupported FlashInfer source: expected exactly one patch site"
        )
    return source.replace(old, new, 1)


def _apply_patch(source: str, diff: str) -> str:
    """Apply a single-file patch with exact positions and context, without fuzz."""
    source_lines = source.splitlines(keepends=True)
    diff_lines = diff.splitlines(keepends=True)
    result: list[str] = []
    cursor = 0
    i = 0
    applied = False
    while i < len(diff_lines):
        match = _HUNK.match(diff_lines[i])
        if match is None:
            i += 1
            continue
        old_count = int(match[2]) if match[2] is not None else 1
        new_count = int(match[4]) if match[4] is not None else 1
        start = int(match[1]) - (1 if old_count else 0)
        old: list[str] = []
        new: list[str] = []
        i += 1
        while i < len(diff_lines) and diff_lines[i][:1] in (" ", "+", "-"):
            line = diff_lines[i]
            if line[0] != "+":
                old.append(line[1:])
            if line[0] != "-":
                new.append(line[1:])
            i += 1
        if (
            len(old) != old_count
            or len(new) != new_count
            or start < cursor
            or source_lines[start : start + old_count] != old
        ):
            raise RuntimeError(
                "Unsupported FlashInfer source: SM90 MoE patch context differs"
            )
        result.extend(source_lines[cursor:start])
        result.extend(new)
        cursor = start + old_count
        applied = True
    if not applied:
        raise RuntimeError("Empty SM90 MoE patch")
    result.extend(source_lines[cursor:])
    return "".join(result)


def _patched_files(package: Path, csrc: Path) -> dict[str, str]:
    """Check every hunk before creating a private source tree."""
    patched = {}
    for section in _PATCH.read_text().split("diff --git ")[1:]:
        names = section.splitlines()[0].split()
        old, new = names
        relative = old.removeprefix("a/")
        if new != f"b/{relative}" or ".." in Path(relative).parts:
            raise RuntimeError("Invalid path in SM90 MoE patch")
        if relative.startswith("csrc/"):
            source = csrc / relative.removeprefix("csrc/")
        elif relative == "flashinfer/fused_moe/core.py":
            source = package / "fused_moe/core.py"
        else:
            raise RuntimeError(f"Unexpected SM90 MoE patch target: {relative}")
        data = source.read_bytes()
        index = re.search(r"^index ([0-9a-f]{40})\.\.", section, re.MULTILINE)
        blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        if index is None or index[1] != blob:
            raise RuntimeError(
                f"Unsupported FlashInfer source for SM90 MoE: {relative}"
            )
        patched[relative] = _apply_patch(data.decode(), section)
    if not patched:
        raise RuntimeError("Missing SM90 MoE patch")
    return patched


def _source_tree(
    package: Path,
    csrc: Path,
    cache: Path,
    max_num_stages_fc1: int,
    max_num_stages_fc2: int,
) -> tuple[Path, str]:
    patched = _patched_files(package, csrc)
    digest = hashlib.sha256()
    digest.update(Path(__file__).read_bytes())
    digest.update(f"max_num_stages_fc1={max_num_stages_fc1}".encode())
    digest.update(f"max_num_stages_fc2={max_num_stages_fc2}".encode())
    for name, text in sorted(patched.items()):
        digest.update(name.encode())
        digest.update(text.encode())
    tag = f"tokenspeed_sm90_moe_{digest.hexdigest()[:16]}"
    compiler = "csrc/nv_internal/tensorrt_llm/deep_gemm/compiler.cuh"
    # DeepGEMM has a second cache, outside FlashInfer's JIT cache.
    patched[compiler] = _replace_once(
        patched[compiler],
        "std::filesystem::path path = getCacheDir();",
        f'std::filesystem::path path = getCacheDir() / "{tag}";\n'
        "#ifdef FLASHINFER_SM90_GROUPED_MOE_OPTIMIZATIONS\n"
        '    path /= "optimized";\n'
        "#else\n"
        '    path /= "expected_m";\n'
        "#endif",
    )
    directory = cache / tag
    # FlashInfer's inline include-directory storage is one process-wide object
    # shared across DSOs. Bind the private JIT compiler to this immutable source
    # tree, so loading another MoE module cannot redirect cold compilation to
    # its headers, and rename the private storage, so loading this module
    # cannot redirect the installed module to these patched headers either.
    include_dir = json.dumps(str(directory / "csrc/nv_internal/tensorrt_llm"))
    patched[compiler] = _replace_once(
        patched[compiler],
        "for (auto const& dir : getJitIncludeDirs()) {",
        "for (auto const& dir : std::vector<std::filesystem::path>{"
        + include_dir
        + "}) {",
    )
    patched[compiler] = re.sub(
        r"\b(get|set)JitIncludeDirs\b", rf"\1JitIncludeDirs_{tag}", patched[compiler]
    )
    directory.mkdir(parents=True, exist_ok=True)
    with FileLock(str(directory / "source.lock")):
        marker = directory / "ready"
        if marker.exists():
            if marker.read_text() != tag or any(
                not (directory / name).is_file()
                or (directory / name).read_text() != text
                for name, text in patched.items()
            ):
                raise RuntimeError(f"SM90 MoE source-cache mismatch: {directory}")
        else:
            # Copy the source tree so quoted includes resolve to patched siblings.
            # Only one copy is needed across ranks and both private module modes.
            shutil.copytree(csrc, directory / "csrc", dirs_exist_ok=True)
            for name, text in patched.items():
                target = directory / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(text)
            marker.write_text(tag)
    return directory, tag


def _register(register: Callable, namespace: str, name: str, *args, **kwargs):
    owner, separator, operation = name.partition("::")
    if owner != "flashinfer" or not separator:
        raise RuntimeError(f"Unexpected FlashInfer operator: {name}")
    return register(f"{namespace}::{operation}", *args, **kwargs)


def _private_spec(
    directory: Path,
    tag: str,
    optimized: bool,
    use_fast_build: bool,
    max_num_stages_fc1: int,
    max_num_stages_fc2: int,
):
    from flashinfer.jit import env as jit_env
    from flashinfer.jit.fused_moe import gen_cutlass_fused_moe_sm90_module

    spec = gen_cutlass_fused_moe_sm90_module(use_fast_build)

    def remap(path):
        path = Path(path)
        if path.is_relative_to(jit_env.FLASHINFER_CSRC_DIR):
            return directory / "csrc" / path.relative_to(jit_env.FLASHINFER_CSRC_DIR)
        return path

    flags = list(spec.extra_cuda_cflags or [])
    flags.append(f"-DFLASHINFER_SM90_MOE_MAX_STAGES_FC1={max_num_stages_fc1}")
    flags.append(f"-DFLASHINFER_SM90_MOE_MAX_STAGES_FC2={max_num_stages_fc2}")
    if optimized:
        flags.append("-DFLASHINFER_SM90_GROUPED_MOE_OPTIMIZATIONS=1")
        flags.append("-DFLASHINFER_SM90_MOE_FUSE_SWIGLU_QUANT=1")
    mode = "optimized" if optimized else "expected_m"
    return replace(
        spec,
        name=f"{tag}_{spec.name}_{mode}",
        sources=[remap(path) for path in spec.sources],
        extra_include_dirs=[directory / "csrc"]
        + [remap(path) for path in spec.extra_include_dirs or []],
        extra_cuda_cflags=flags,
    )


@functools.cache
def _private_entrypoint(
    optimized: bool, max_num_stages_fc1: int, max_num_stages_fc2: int
) -> Callable:
    import flashinfer
    from flashinfer.fused_moe import core
    from flashinfer.jit import env as jit_env
    from flashinfer.jit.cpp_ext import is_cuda_version_at_least

    if flashinfer.__version__ != "0.7.2.post1":
        raise RuntimeError(
            f"SM90 MoE patch requires FlashInfer 0.7.2.post1, got {flashinfer.__version__}; "
            "use mode='default' for the installed implementation"
        )
    if os.environ.get("FLASHINFER_DISABLE_JIT"):
        raise RuntimeError("Private SM90 MoE modes require FlashInfer JIT compilation")
    if not is_cuda_version_at_least("12.8"):
        raise RuntimeError(
            "Private SM90 block-FP8 MoE modes require CUDA 12.8 or newer"
        )
    package = Path(flashinfer.__file__).parent
    directory, tag = _source_tree(
        package,
        jit_env.FLASHINFER_CSRC_DIR,
        jit_env.FLASHINFER_GEN_SRC_DIR,
        max_num_stages_fc1,
        max_num_stages_fc2,
    )
    namespace = dict(vars(core))
    mode = "optimized" if optimized else "expected_m"
    operator_namespace = f"{tag}_{mode}"
    for name in ("register_custom_op", "register_fake_op"):
        namespace[name] = functools.partial(
            _register, namespace[name], operator_namespace
        )

    def generator(use_fast_build, *, enable_sm90_moe_optimizations):
        if enable_sm90_moe_optimizations != optimized:
            raise ValueError("SM90 MoE mode was fixed during initialization")
        return _private_spec(
            directory,
            tag,
            optimized,
            use_fast_build,
            max_num_stages_fc1,
            max_num_stages_fc2,
        )

    namespace["gen_cutlass_fused_moe_sm90_module"] = generator
    namespace["_tokenspeed_csrc_dir"] = directory / "csrc"
    source = (directory / "flashinfer/fused_moe/core.py").read_text()
    source = _replace_once(
        source,
        ') + (("sm90_grouped_optimized",) if enable_sm90_moe_optimizations else ())',
        ") + (_tokenspeed_tuning_namespace,)",
    )
    namespace["_tokenspeed_tuning_namespace"] = operator_namespace
    # Execute only the patched entry point and its factory. The rest of core.py
    # remains the upstream module, including quantization and routing helpers.
    names = {
        "_validate_cutlass_moe_expected_m",
        "get_cutlass_fused_moe_module",
        "cutlass_fused_moe",
    }
    parsed = ast.parse(source)
    functions = [
        node
        for node in parsed.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    if {node.name for node in functions} != names:
        raise RuntimeError("Unsupported FlashInfer CUTLASS MoE entry points")

    # The factory passes include paths to DeepGEMM at run time as well as nvcc.
    class IncludePath(ast.NodeTransformer):
        def visit_Attribute(self, node):
            if (
                isinstance(node.value, ast.Name)
                and node.value.id == "jit_env"
                and node.attr == "FLASHINFER_CSRC_DIR"
            ):
                return ast.copy_location(
                    ast.Name(id="_tokenspeed_csrc_dir", ctx=ast.Load()), node
                )
            return self.generic_visit(node)

    tree = ast.fix_missing_locations(
        IncludePath().visit(
            ast.Module(
                body=[
                    node
                    for node in parsed.body
                    if isinstance(node, ast.ImportFrom) and node.module == "__future__"
                ]
                + functions,
                type_ignores=[],
            )
        )
    )
    exec(
        compile(
            tree,
            str(directory / "flashinfer/fused_moe/core.py"),
            "exec",
            dont_inherit=True,
        ),
        namespace,
    )
    # Materialize the module during initialization, before any graph capture.
    factory = namespace["get_cutlass_fused_moe_module"]

    def sm90_module(backend, *args, **kwargs):
        if backend != "90":
            raise ValueError("Private SM90 MoE modes require the SM90 backend")
        return factory(backend, *args, **kwargs)

    namespace["get_cutlass_fused_moe_module"] = sm90_module
    sm90_module("90", enable_sm90_moe_optimizations=optimized)
    return namespace["cutlass_fused_moe"]


def make_cutlass_fused_moe(
    *, mode: MoEMode, max_num_stages_fc1: int = -1, max_num_stages_fc2: int = -1
) -> Callable:
    """Bind an isolated FlashInfer MoE implementation before warmup/capture.

    Args:
        mode: ``default`` returns the installed FlashInfer entry point unchanged.
            ``expected_m`` adds a per-call hint to the original SM90 kernels.
            ``optimized`` additionally enables prefix and tile-to-expert lookup.
        max_num_stages_fc1: Upper bound on FC1 grouped GEMM pipeline stages in either
            private mode. ``-1`` preserves upstream selection. Positive bounds
            must be at least four, the backend's minimum candidate depth.
            Fixed during initialization and independent of the FC2 bound.
        max_num_stages_fc2: Independent FC2 upper bound, with the same
            semantics and default as ``max_num_stages_fc1``.

    Returns:
        A callable with FlashInfer's ``cutlass_fused_moe`` arguments and return
        values. Private modes additionally accept ``expected_m`` (positive
        uint32 or None). The caller owns the estimate and its routing assumptions.
        The mode is fixed for the callable's lifetime; create a new callable and
        recapture graphs to change it. Unsupported private builds raise at init.
    """
    if mode not in ("default", "expected_m", "optimized"):
        raise ValueError(f"Unknown SM90 MoE mode: {mode!r}")
    for name, value in (
        ("max_num_stages_fc1", max_num_stages_fc1),
        ("max_num_stages_fc2", max_num_stages_fc2),
    ):
        if type(value) is not int or not (value == -1 or 4 <= value <= 2**31 - 1):
            raise ValueError(f"{name} must be -1 or an integer from 4 to 2**31 - 1")
        if mode == "default" and value != -1:
            raise ValueError(f"{name} requires a private SM90 MoE mode")
    from flashinfer import cutlass_fused_moe

    if mode == "default":
        return cutlass_fused_moe
    import torch

    if torch.cuda.get_device_capability() != (9, 0):
        raise ValueError("Private SM90 MoE modes require a Hopper device")
    optimized = mode == "optimized"
    # functools.cache alone can execute concurrent cold misses twice; custom-op
    # registration must happen once per private namespace.
    with _ENTRYPOINT_LOCK:
        entrypoint = _private_entrypoint(
            optimized, max_num_stages_fc1, max_num_stages_fc2
        )

    def apply(*args, **kwargs):
        if any(
            name in kwargs
            for name in (
                "enable_sm90_moe_optimizations",
                "max_num_stages_fc1",
                "max_num_stages_fc2",
            )
        ):
            raise TypeError(
                "Choose the SM90 MoE mode and stage bounds during initialization"
            )
        return entrypoint(*args, enable_sm90_moe_optimizations=optimized, **kwargs)

    return apply

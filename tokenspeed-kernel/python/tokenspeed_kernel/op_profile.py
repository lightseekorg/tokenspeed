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

"""Host metadata collection and offline costs for registered ops and layers."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import re
import shutil
import stat
import sys
import time
from contextlib import AbstractContextManager, contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field, fields, is_dataclass, replace
from functools import lru_cache, wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterator, Protocol
from urllib.parse import quote

import torch

if TYPE_CHECKING:
    from tokenspeed_kernel.profile_costs import CollectiveTraffic
    from tokenspeed_kernel.profile_nvtx import NvtxTimeline
    from tokenspeed_kernel.registry import KernelSpec


class ProfileObserver(Protocol):
    """Capture scopes and bounded integer metadata without choosing execution."""

    def record_func(
        self, label: str, *, display_name: str
    ) -> AbstractContextManager: ...

    def snapshot_integer_values(
        self, source: torch.Tensor, target: dict, name: str
    ) -> None: ...

    def snapshot_integer_selection(
        self, producer: Callable, target: dict, inputs: dict, name: str
    ) -> None: ...


@dataclass(frozen=True)
class TensorDescription:
    shape: tuple[int, ...]
    dtype: str
    stride: tuple[int, ...]
    device: str
    logical_bytes: int
    storage_offset: int | None = None
    pointer_mod256: int | None = None


def _describe(value: Any, path: str, result: dict[str, TensorDescription]) -> None:
    if isinstance(value, torch.Tensor):
        result[path] = TensorDescription(
            tuple(value.shape),
            str(value.dtype),
            tuple(value.stride()),
            str(value.device),
            value.numel() * value.element_size(),
            value.storage_offset(),
            value.data_ptr() % 256,
        )
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _describe(item, f"{path}[{index}]", result)
    elif isinstance(value, dict):
        for key, item in value.items():
            _describe(item, f"{path}[{key!r}]", result)
    elif is_dataclass(value) and not isinstance(value, type):
        for item in fields(value):
            _describe(getattr(value, item.name), f"{path}.{item.name}", result)


def _record_torch_execution_context(arguments: dict[str, Any], metadata: dict) -> None:
    """Read host-owned math settings during profiling; never query a device."""
    matmul = torch.backends.cuda.matmul
    for name in (
        "allow_tf32",
        "allow_fp16_reduced_precision_reduction",
        "allow_bf16_reduced_precision_reduction",
        "allow_fp16_accumulation",
    ):
        metadata[f"torch_matmul_{name}"] = getattr(matmul, name, None)
    metadata.update(
        torch_version=str(torch.__version__),
        torch_cuda_version=torch.version.cuda,
        torch_git_version=torch.version.git_version,
        torch_float32_matmul_precision=torch.get_float32_matmul_precision(),
        torch_preferred_blas_library=str(torch.backends.cuda.preferred_blas_library()),
        profile_argument_names=tuple(arguments),
        profile_none_arguments=tuple(
            name for name, value in arguments.items() if value is None
        ),
        profile_tensor_arguments=tuple(
            name for name, value in arguments.items() if isinstance(value, torch.Tensor)
        ),
    )
    for name, value in arguments.items():
        if type(value) in (bool, int, float, str) or isinstance(value, torch.dtype):
            metadata[f"profile_argument_{name}"] = (
                str(value) if isinstance(value, torch.dtype) else value
            )


_FP8_SOURCE_FILES = {
    "deep_gemm": (
        "__init__.py",
        "csrc/apis/gemm.hpp",
        "csrc/jit_kernels/impls/sm90_fp8_gemm_1d2d.hpp",
        "csrc/runtime/jit.hpp",
        "include/cute/arch/mma_sm90_gmma.hpp",
        "include/deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh",
        "include/deep_gemm/mma/sm90.cuh",
    ),
    "flashinfer": (
        "data/csrc/fp8_blockscale_gemm_sm90_binding.cu",
        "data/csrc/fused_moe/cutlass_backend/cutlass_fused_moe_instantiation.cu",
        "data/csrc/fused_moe/cutlass_backend/cutlass_fused_moe_kernels.cuh",
        "data/csrc/fused_moe/cutlass_backend/flashinfer_cutlass_fused_moe_binding.cu",
        "data/csrc/nv_internal/tensorrt_llm/deep_gemm/compiler.cuh",
        "data/csrc/nv_internal/tensorrt_llm/deep_gemm/fp8_gemm_impl.cuh",
        "data/csrc/nv_internal/tensorrt_llm/deep_gemm/mma_utils.cuh",
        "data/csrc/nv_internal/tensorrt_llm/deep_gemm/runtime.cuh",
        "data/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/fp8_blockscale_gemm.cu",
        "data/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/fp8_blockscale_gemm.h",
        "data/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/fp8_blockscale_gemm_kernel.cuh",
        "data/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/fp8_blockscale_mma_utils.cuh",
        "data/cutlass/include/cute/arch/mma_sm90_gmma.hpp",
        "fused_moe/core.py",
        "gemm/gemm_base.py",
        "jit/env.py",
        "jit/fused_moe.py",
        "jit/gemm/fp8_blockscale.py",
    ),
    "tokenspeed_kernel": ("ops/moe/flashinfer.py",),
}


def _profile_fingerprint(
    path: Path, expected_inode=None, *, max_bytes: int | None = None
) -> dict[str, Any]:
    item = {"path": str(path), "sha256": None, "unavailable_reason": None}
    try:
        flags = os.O_RDONLY | (os.O_NOFOLLOW if max_bytes is not None else 0)
        with os.fdopen(os.open(path, flags), "rb") as stream:
            before = os.fstat(stream.fileno())
            if max_bytes is not None and before.st_size > max_bytes:
                raise ValueError("file grew beyond fingerprint byte budget")
            if expected_inode is not None and before.st_ino != expected_inode:
                raise ValueError("mapped inode differs from opened file")
            digest = hashlib.sha256()
            count = 0
            while block := stream.read(
                1024 * 1024
                if max_bytes is None
                else min(1024 * 1024, max_bytes - count + 1)
            ):
                count += len(block)
                if max_bytes is not None and count > max_bytes:
                    raise ValueError("file grew beyond fingerprint byte budget")
                digest.update(block)
            after = os.fstat(stream.fileno())
        if _profile_file_identity(before) != _profile_file_identity(after):
            raise ValueError("file changed while fingerprinting")
        item.update(
            sha256=digest.hexdigest(),
            bytes=after.st_size,
            inode=after.st_ino,
            device=after.st_dev,
            mtime_ns=after.st_mtime_ns,
            ctime_ns=after.st_ctime_ns,
        )
    except (OSError, ValueError) as error:
        item["unavailable_reason"] = f"{type(error).__name__}: {error}"
    return item


def _profile_file_identity(value):
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _profile_cache_snapshot(directory: Path, fingerprint) -> dict[str, Any]:
    """Bound the whole walk before hashing, then recheck every identity/stat."""
    result = dict(
        root=str(directory), complete=False, candidates=[], unavailable_reason=None
    )

    def census():
        rows = {}
        directories = 0
        pending = [directory]
        while pending:
            path = pending.pop()
            value = path.lstat()
            if stat.S_ISLNK(value.st_mode):
                raise OSError("symlink cache root, subtree or candidate")
            if not (stat.S_ISREG(value.st_mode) or stat.S_ISDIR(value.st_mode)):
                raise OSError("nonregular cache entry")
            rows[str(path)] = _profile_file_identity(value)
            if len(rows) > 8192:
                raise OSError("cache entry budget exhausted")
            if stat.S_ISDIR(value.st_mode):
                directories += 1
                if directories > 512:
                    raise OSError("cache directory budget exhausted")
                with os.scandir(path) as entries:
                    for entry in entries:
                        pending.append(Path(entry.path))
                        if len(pending) + len(rows) > 8192:
                            raise OSError("cache entry budget exhausted")
        return rows

    try:
        if any(path.is_symlink() for path in (directory, *directory.parents)):
            raise OSError("symlinked cache root ancestry")
        first = census()
        relevant = [
            identity[3]
            for name, identity in first.items()
            if stat.S_ISREG(identity[2])
            and (
                Path(name).suffix in (".cubin", ".ptx", ".cu", ".json")
                or Path(name).name == ".committed"
            )
        ]
        if (
            any(size > 128 * 1024 * 1024 for size in relevant)
            or sum(relevant) > 512 * 1024 * 1024
        ):
            raise OSError("cache per-file/total byte budget exhausted before hashing")
        for name, identity in sorted(first.items()):
            path = Path(name)
            if not stat.S_ISREG(identity[2]):
                continue
            if (
                path.suffix not in (".cubin", ".ptx", ".cu", ".json")
                and path.name != ".committed"
            ):
                continue
            item = dict(
                relative_path=str(path.relative_to(directory)), **fingerprint(path)
            )
            result["candidates"].append(item)
            if item["unavailable_reason"]:
                raise OSError(item["unavailable_reason"])
        if first != census():
            raise OSError("cache candidate identity/stat set changed during snapshot")
        result["complete"] = True
    except (OSError, ValueError) as error:
        result["unavailable_reason"] = f"{type(error).__name__}: {error}"
    return result


def _profile_process_identity() -> dict:
    """Record this worker's raw PID/clock domains; never infer a birthday."""
    result = dict(process_id=os.getpid(), parent_process_id=os.getppid())
    before = time.time_ns()
    result["monotonic_ns"] = time.monotonic_ns()
    result["realtime_before_ns"] = before
    result["realtime_after_ns"] = time.time_ns()
    try:
        raw = Path("/proc/self/stat").read_text()
        fields = raw.rsplit(") ", 1)[1].split()
        result.update(
            stat_process_id=int(raw.split(" (", 1)[0]),
            start_ticks=int(fields[19]),
            clock_ticks_per_second=os.sysconf("SC_CLK_TCK"),
            boottime_ns=time.clock_gettime_ns(time.CLOCK_BOOTTIME),
            boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            time_namespace=os.readlink("/proc/self/ns/time"),
            pid_namespace=os.readlink("/proc/self/ns/pid"),
            namespace_pids=[
                int(value)
                for line in Path("/proc/self/status").read_text().splitlines()
                if line.startswith("NSpid:")
                for value in line.split()[1:]
            ],
        )
        ancestors, parent = [], os.getppid()
        while parent > 1 and parent not in ancestors and len(ancestors) < 64:
            ancestors.append(parent)
            fields = Path(f"/proc/{parent}/stat").read_text().rsplit(") ", 1)[1].split()
            parent = int(fields[1])
        result["ancestor_process_ids"] = ancestors
        result["unavailable_reason"] = None
    except (OSError, ValueError, IndexError, AttributeError) as error:
        result["unavailable_reason"] = f"{type(error).__name__}: {error}"
    return result


# Shared data-only timeline schema. Runtime defaults and offline epoch checks use
# these same values; an observer never imports the serving runtime to expand it.
_PROFILE_CONFIG_REQUIRED = frozenset(
    {
        "run_id",
        "output_dir",
        "hardware",
        "layers",
        "start_step",
        "num_steps",
        "max_scopes",
        "hbm_policy",
    }
)
_PROFILE_CONFIG_DEFAULTS = dict(
    cuda_runtime_library=None,
    cupti_library=None,
    max_metadata_bytes=1 << 20,
    metadata_slots=8,
    ranks=None,
    startup_observation=None,
)


def _profile_expand_timeline_config(config: dict) -> dict:
    if isinstance(config, dict) and {"library", "nvtx_library"} & config.keys():
        raise ValueError(
            "Remove the legacy library or nvtx_library field; "
            "eager native profiling uses NVIDIA nvtx==0.2.17"
        )
    if (
        not isinstance(config, dict)
        or _PROFILE_CONFIG_REQUIRED - config.keys()
        or config.keys() - _PROFILE_CONFIG_REQUIRED - _PROFILE_CONFIG_DEFAULTS.keys()
    ):
        raise ValueError("Missing or unknown timeline config field")
    return {**_PROFILE_CONFIG_DEFAULTS, **config}


def _profile_config_for_observation(observation: dict, *, run_id: str) -> dict:
    body = _profile_validate_startup_observation(observation, run_id=run_id)
    expected = _profile_expand_timeline_config(body["timeline_config"])
    expected["startup_observation"] = observation
    return expected


def _profile_observed_configs(observation: dict, *, run_id: str) -> tuple[dict, ...]:
    """Reconstruct configs for verification of sealed current and old reports."""
    body = _profile_validate_startup_observation(observation, run_id=run_id)
    config = dict(body["timeline_config"])
    # The former live config always expanded the library key, even if unset.
    bridge_specified = "library" in config
    bridge = config.pop("library", None)
    current = _profile_expand_timeline_config(config)
    current["startup_observation"] = observation
    legacy = {**current, "library": bridge}
    return (legacy,) if bridge_specified else (current, legacy)


def _profile_validate_startup_observation(
    observation: dict, *, run_id: str, timeline_config: dict | None = None
) -> dict:
    """Validate data-only before-Engine observation identity, without imports."""
    if not isinstance(observation, dict) or not isinstance(
        observation.get("body"), dict
    ):
        raise ValueError("before-Engine startup observation unavailable")
    body = observation["body"]
    digest = hashlib.sha256(
        json.dumps(
            body, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    case_sha = body.get("case_sha256")
    repetition = body.get("repetition")
    config = body.get("timeline_config")
    source = body.get("driver_source", {})
    if (
        observation.get("sha256") != digest
        or body.get("schema") != "tokenspeed.before_engine_observation/v1"
        or body.get("run_id") != run_id
        or body.get("arm") != "profiled"
        or type(repetition) is not int
        or repetition < 0
        or not isinstance(case_sha, str)
        or re.fullmatch(r"[0-9a-f]{64}", case_sha) is None
        or run_id != f"{case_sha[:16]}-{repetition}"
        or not isinstance(config, dict)
        or config.get("run_id") != run_id
        or "startup_observation" in config
        or (timeline_config is not None and config != timeline_config)
        or body.get("timeline_config_sha256")
        != hashlib.sha256(
            json.dumps(
                config, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        ).hexdigest()
        or any(
            type(body.get(key)) is not int or body[key] <= 0
            for key in (
                "process_id",
                "parent_process_id",
                "realtime_before_ns",
                "realtime_after_ns",
                "monotonic_ns",
            )
        )
        or body["realtime_before_ns"] > body["realtime_after_ns"]
        or not isinstance(source.get("path"), str)
        or not isinstance(source.get("sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", source["sha256"]) is None
        or type(source.get("bytes")) is not int
        or source["bytes"] <= 0
    ):
        raise ValueError(
            "before-Engine startup run/case/config/source/hash identity mismatch"
        )
    return body


def _profile_include_search_roots(
    roots: dict, configured: dict, environment: dict, cwd: str
) -> list[dict]:
    """Represent every actual configured/environment search segment in order."""
    result = []

    def add(value, origin):
        if not isinstance(value, str):
            raise ValueError("include root is not a path string")
        path = Path(value) if value else Path(cwd)
        if not path.is_absolute():
            path = Path(cwd) / path
        result.append(dict(root=str(path), origin=origin))

    if roots.get("deep_gemm"):
        add(str(Path(roots["deep_gemm"]) / "include"), "deep_gemm.include")
    if roots.get("flashinfer"):
        values = configured.get("CUTLASS_INCLUDE_DIRS")
        if not isinstance(values, list):
            raise ValueError("actual CUTLASS include search unavailable")
        for index, value in enumerate(values):
            add(value, f"CUTLASS_INCLUDE_DIRS[{index}]")
    for key in ("CPATH", "CPLUS_INCLUDE_PATH", "C_INCLUDE_PATH"):
        if key in environment:
            if not isinstance(environment[key], str):
                raise ValueError("include environment must contain strings")
            for index, value in enumerate(environment[key].split(os.pathsep)):
                add(value, f"{key}[{index}]")
    return result


def _profile_include_candidate(entry: dict, fingerprint) -> dict:
    """Probe one exact path, retaining ENOENT separately from unreadable input."""
    directory = Path(entry["root"])
    path = directory / "cute/arch/mma_sm90_gmma.hpp"
    item = dict(entry, path=str(path))
    present = False
    try:
        ancestry = {}
        for parent in (path.parent, *path.parent.parents):
            try:
                value = parent.lstat()
            except FileNotFoundError:
                continue
            if stat.S_ISLNK(value.st_mode) or not stat.S_ISDIR(value.st_mode):
                raise OSError("untrusted include root ancestry")
            ancestry[str(parent)] = _profile_file_identity(value)
        value = path.lstat()
        if not stat.S_ISREG(value.st_mode):
            raise OSError("nonregular or symlink include candidate")
        present = True
        item.update(fingerprint(path))
        if any(
            _profile_file_identity(Path(parent).lstat()) != identity
            for parent, identity in ancestry.items()
        ):
            raise OSError("include root ancestry changed during probe")
        item["presence"] = (
            "present_regular_file" if not item["unavailable_reason"] else "unavailable"
        )
    except FileNotFoundError as error:
        if present:
            item.update(
                presence="unavailable",
                sha256=None,
                unavailable_reason="include candidate/root disappeared during probe",
            )
        else:
            item.update(
                presence="proved_absent_ENOENT",
                sha256=None,
                unavailable_reason=None,
                exception_type=type(error).__name__,
                errno=error.errno,
                exception_path=str(path),
            )
    except (OSError, ValueError) as error:
        item.update(
            presence="unavailable",
            sha256=None,
            unavailable_reason=f"{type(error).__name__}: {error}",
        )
    return item


def profile_fp8_source_provenance(
    *, phase: str, libraries: list | None = None, source_context: dict | None = None
) -> dict[str, Any]:
    """CPU-only loaded API/source/compiler/cache snapshot outside timed work.

    Existing blobs and current generation require different provider predicates.
    This is never described as a CUDA module or runtimeCache memory inventory.
    """
    result = dict(
        schema="tokenspeed.fp8_source_provenance/v1",
        process_id=os.getpid(),
        user_home=str(Path.home()),
        started_unix_ns=time.time_ns(),
        collection_phase=phase,
        modules=[],
        loaded_extensions=[],
        source_files=[],
        include_candidates=[],
        caches=[],
        compilers={},
        configured_roots={},
        runtime_cache_inventory="not_observed",
        source_context=source_context,
        local_process_identity=_profile_process_identity(),
        working_directory=str(Path.cwd()),
    )
    budget = [512 * 1024 * 1024]

    def fingerprint(path):
        try:
            value = path.lstat()
            if (
                not stat.S_ISREG(value.st_mode)
                or value.st_size > 128 * 1024 * 1024
                or value.st_size > budget[0]
            ):
                raise OSError(
                    "semantic provenance file/symlink/byte budget unavailable"
                )
            budget[0] -= value.st_size
            item = _profile_fingerprint(path, max_bytes=value.st_size)
            if path.name == "meta.json" and not item["unavailable_reason"]:
                if value.st_size > 1024 * 1024 or value.st_size > budget[0]:
                    raise OSError("compile metadata byte budget exhausted")
                budget[0] -= value.st_size
                with os.fdopen(
                    os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb"
                ) as stream:
                    payload = stream.read(value.st_size + 1)
                if len(payload) != value.st_size:
                    raise OSError("compile metadata changed size")
                item["metadata"] = json.loads(payload)
            return item
        except (OSError, ValueError) as error:
            return dict(
                path=str(path),
                sha256=None,
                unavailable_reason=f"{type(error).__name__}: {error}",
            )

    roots = {}
    jit_environment = sys.modules.get("flashinfer.jit.env")
    jit_values = {} if jit_environment is None else vars(jit_environment)
    for name, relatives in _FP8_SOURCE_FILES.items():
        module = sys.modules.get(name)
        path = None if module is None else vars(module).get("__file__")
        if not isinstance(path, str):
            continue
        root = Path(path).resolve().parent
        roots[name] = root
        result["modules"].append(
            dict(name=name, root=str(root), **fingerprint(Path(path)))
        )
        if name == "flashinfer":
            for key in (
                "FLASHINFER_CSRC_DIR",
                "FLASHINFER_INCLUDE_DIR",
                "CUTLASS_INCLUDE_DIRS",
            ):
                configured = jit_values.get(key)
                result["configured_roots"][key] = (
                    [str(value) for value in configured]
                    if isinstance(configured, (tuple, list))
                    else (
                        str(configured) if isinstance(configured, (str, Path)) else None
                    )
                )
        for relative in relatives:
            source_path = root / relative
            if name == "flashinfer" and relative.startswith("data/csrc/"):
                configured = result["configured_roots"]["FLASHINFER_CSRC_DIR"]
                source_path = Path(
                    configured or root / "data/csrc"
                ) / relative.removeprefix("data/csrc/")
            loaded_name = (
                f"{name}.{relative[:-3].replace('/', '.')}"
                if relative.endswith(".py")
                else None
            )
            loaded_module = sys.modules.get(loaded_name)
            loaded_path = (
                None if loaded_module is None else vars(loaded_module).get("__file__")
            )
            if isinstance(loaded_path, str):
                source_path = Path(loaded_path)
            result["source_files"].append(
                dict(module=name, relative_path=relative, **fingerprint(source_path))
            )
    # Reuse mapped API fingerprints from the existing final library manifest.
    result["loaded_extensions"] = [
        item
        for item in (libraries or [])
        if "/deep_gemm/_C." in item["path"]
        or Path(item["path"]).name.startswith(
            ("fp8_blockscale_gemm_90.so", "fused_moe_90.so")
        )
    ]
    communication = sys.modules.get("tokenspeed_kernel.profile_communication")
    binding = None if communication is None else vars(communication).get("_binding")
    peers = [] if binding is None else binding.get("rank_evidence", [])
    result["run_id"] = None if binding is None else binding.get("run_id")
    result["rank"] = None if binding is None else binding.get("rank")
    result["workers"] = [
        dict(
            process_id=peer.get("pid"),
            rank=peer.get("rank"),
            process_identity=peer.get("process_identity"),
        )
        for peer in peers
    ] or [
        dict(
            process_id=os.getpid(),
            rank=result["rank"],
            process_identity=result["local_process_identity"],
        )
    ]
    observation = (source_context or {}).get("startup_observation")
    if isinstance(observation, dict):
        driver = observation.get("body", {}).get("driver_source", {})
        if isinstance(driver.get("path"), str):
            result["startup_driver_source"] = fingerprint(Path(driver["path"]))
    result["environment"] = {
        key: value
        for key, value in os.environ.items()
        if key.startswith(("DG_", "TRTLLM_DG_"))
        or key.endswith("FLAGS")
        or key
        in (
            "CPATH",
            "CPLUS_INCLUDE_PATH",
            "C_INCLUDE_PATH",
            "CUDA_HOME",
            "PATH",
            "FLASHINFER_WORKSPACE_BASE",
            "FLASHINFER_JIT_DIR",
        )
    }
    for name in ("deep_gemm", "trtllm"):
        variable = (
            "DG_JIT_NVCC_COMPILER" if name == "deep_gemm" else "TRTLLM_DG_NVCC_COMPILER"
        )
        requested = os.environ.get(variable)
        basis = variable
        if not requested:
            home = os.environ.get("CUDA_HOME")
            requested = str(Path(home) / "bin/nvcc") if home else "nvcc"
            basis = "CUDA_HOME" if home else "PATH"
        resolved = shutil.which(requested)
        result["compilers"][name] = dict(
            selection=basis,
            requested=requested,
            **(
                fingerprint(Path(resolved).resolve())
                if resolved
                else dict(
                    path=None,
                    sha256=None,
                    unavailable_reason="compiler resolution unavailable",
                )
            ),
        )
    try:
        search = _profile_include_search_roots(
            roots,
            result["configured_roots"],
            result["environment"],
            result["working_directory"],
        )
        result["include_search"] = dict(
            schema="tokenspeed.gmma_include_search/v1", entries=search, complete=True
        )
        for entry in search:
            result["include_candidates"].append(
                _profile_include_candidate(entry, fingerprint)
            )
    except (OSError, ValueError) as error:
        result["include_search"] = dict(
            schema="tokenspeed.gmma_include_search/v1",
            complete=False,
            unavailable_reason=f"{type(error).__name__}: {error}",
        )
    for name, module, variable, default in (
        ("deep_gemm", "deep_gemm", "DG_JIT_CACHE_DIR", "~/.dj"),
        ("trtllm", "flashinfer", "TRTLLM_DG_CACHE_DIR", "~/.tensorrt_llm"),
    ):
        if module in roots:
            directory = Path(os.environ.get(variable, default)).expanduser() / "cache"
            result["caches"].append(
                dict(name=name, **_profile_cache_snapshot(directory, fingerprint))
            )
    result["finished_unix_ns"] = time.time_ns()
    result["remaining_byte_budget"] = budget[0]
    return result


def loaded_library_manifest(
    *,
    source_context: dict | None = None,
    collection_phase: str = "collector_final_flush",
) -> dict[str, Any]:
    """Fingerprint mapped selected-library candidates only after collection.

    The caller owns the run/window association. No CUDA APIs, weights, loader
    changes or imports from a trace. Unreadable/replaced mappings stay unknown.
    """
    result = {
        "schema": "tokenspeed.loaded_libraries/v1",
        "process_id": os.getpid(),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": torch.version.cuda,
        "libraries": [],
        "source_files": [],
        "unavailable_reason": None,
    }
    try:
        mappings = Path("/proc/self/maps").read_text()
    except OSError as error:
        result["unavailable_reason"] = (
            f"mapped libraries unavailable: {type(error).__name__}"
        )
        return result
    result["maps_sha256"] = hashlib.sha256(mappings.encode()).hexdigest()
    selected = {}
    for line in mappings.splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or not fields[5].startswith("/"):
            continue
        name = Path(fields[5]).name
        if (
            name.startswith(
                (
                    "libtorch_cuda.so",
                    "libcublas.so",
                    "libcublasLt.so",
                    "silu_and_mul.so",
                    "fp8_blockscale_gemm_90.so",
                    "fused_moe_90.so",
                )
            )
            or "/deep_gemm/_C." in fields[5]
        ):
            selected[fields[5]] = int(fields[4])

    if collection_phase == "collector_start":
        selected = {
            path: inode
            for path, inode in selected.items()
            if "/deep_gemm/_C." in path
            or Path(path).name.startswith(
                ("fp8_blockscale_gemm_90.so", "fused_moe_90.so")
            )
        }
    api_byte_budget = 256 * 1024 * 1024
    for path, inode in sorted(selected.items()):
        source = Path(path)
        limit = None
        if "/deep_gemm/_C." in path or source.name.startswith(
            ("fp8_blockscale_gemm_90.so", "fused_moe_90.so")
        ):
            try:
                value = source.lstat()
                if not stat.S_ISREG(value.st_mode) or value.st_size > min(
                    128 * 1024 * 1024, api_byte_budget
                ):
                    raise OSError(
                        "mapped semantic API object exceeds file/symlink/byte budget"
                    )
                api_byte_budget -= value.st_size
                limit = value.st_size
            except OSError as error:
                result["libraries"].append(
                    dict(path=path, sha256=None, unavailable_reason=str(error))
                )
                continue
        result["libraries"].append(_profile_fingerprint(source, inode, max_bytes=limit))
    sources = [Path(torch.__file__).parent / "include/torch/headeronly/util/BFloat16.h"]
    flashinfer = sys.modules.get("flashinfer")
    if flashinfer is not None and isinstance(
        getattr(flashinfer, "__file__", None), str
    ):
        root = Path(flashinfer.__file__).parent
        sources.extend(
            root / name
            for name in (
                "activation.py",
                "jit/activation.py",
                "data/include/flashinfer/activation.cuh",
            )
        )
    result["source_files"] = [_profile_fingerprint(path) for path in sources]
    result["semantic_provenance"] = profile_fp8_source_provenance(
        phase=collection_phase,
        libraries=result["libraries"],
        source_context=source_context,
    )
    return result


@dataclass(frozen=True)
class OpProfile:
    """Opt in a function; scalar_args retains named host values for cost rules.

    Values must be JSON scalars or tuples of scalars, never device tensors.
    Multi-valued traits automatically capture same-named host arguments.
    Explicit scalar/tensor/integer paths may traverse declared object storage,
    dictionaries and NamedTuple fields; properties are never evaluated.
    integer_tensor_args snapshots contiguous 1-D/2-D int32/int64 metadata up to
    integer_tensor_max_elements per argument. Matrices use a flat alias view
    without copying. The active observer also enforces its total byte budget.
    The active observer owns snapshot ordering: CUDA graphs retain D2D copies
    per replay, never capture-time CPU values. Without an observer only CPU
    integers are read. tensor_args records descriptions of explicit paths.
    metadata_preparer is an implementation-owned callback receiving bound
    arguments. It runs with metadata recording active, before the Op timing
    range and before numerical execution, so selection/copy overhead is excluded
    from eager Op GPU projections. It adds no arguments to the profiled function.
    cost_rule=None records metadata only.
    """

    cost_rule: str | None
    scalar_args: tuple[str, ...] = ()
    integer_tensor_args: tuple[str, ...] = ()
    tensor_args: tuple[str, ...] = ()
    metadata_preparer: Callable[[dict[str, Any]], None] | None = None
    integer_tensor_max_elements: int = 4096

    def __post_init__(self):
        if (
            type(self.integer_tensor_max_elements) is not int
            or self.integer_tensor_max_elements < 1
        ):
            raise ValueError("integer_tensor_max_elements must be a positive integer")


@dataclass(frozen=True)
class OpRecord:
    iter_id: int
    layer_id: int | None
    occurrence: int | None
    inner_id: int
    op_name: str
    implementation: str
    cost_rule: str | None
    inputs: dict[str, TensorDescription]
    outputs: dict[str, TensorDescription]
    cpu_start_ns: int
    cpu_duration_ns: int
    error: str | None
    trace_label: str
    autocast_enabled: bool = False
    scalar_args: dict[str, Any] = field(default_factory=dict)
    tensor_values: dict[str, tuple[int, ...] | None] = field(default_factory=dict)
    static_traits: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LayerRecord:
    iter_id: int
    layer_id: int | None
    occurrence: int
    cpu_start_ns: int
    cpu_duration_ns: int
    error: str | None
    trace_label: str


@dataclass(frozen=True)
class _Context:
    session: ProfileSession
    iter_id: int
    layer_id: int | None
    occurrence: int | None
    metadata: dict[str, Any] | None = None


_current: ContextVar[_Context | None] = ContextVar("op_profile_context", default=None)
_recording_inputs: ContextVar[dict | None] = ContextVar("profile_inputs", default=None)
_recording_values: ContextVar[dict | None] = ContextVar("profile_values", default=None)

_recording_op: ContextVar[dict[str, Any] | None] = ContextVar(
    "op_profile_recording_op", default=None
)


def current_op_metadata() -> dict[str, Any] | None:
    """Return the active op's host metadata, or None when not collecting.

    Existing dispatchers may record their actual branch/group here. Write only
    JSON scalars or tuples of scalars; never read device tensors for metadata.
    """
    return _recording_op.get()


def profile_session_active() -> bool:
    """Whether this thread/context is inside a collected iteration window."""
    return _current.get() is not None


@lru_cache(maxsize=128)
def _profile_source_digest(path: str, modified_ns: int, size: int) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _profile_source(implementation: Callable) -> tuple[str | None, str | None]:
    try:
        path = inspect.getsourcefile(implementation)
        if path is not None:
            status = Path(path).stat()
            return path, _profile_source_digest(
                path, status.st_mtime_ns, status.st_size
            )
    except (OSError, TypeError):
        pass
    return None, None


@dataclass(frozen=True)
class _ImplementationEngineContract:
    implementation: Callable
    compute_type: str
    contract_id: str


def profile_engine_contract(*, compute_type: str, contract_id: str) -> Callable:
    """Declare an implementation-owned compute contract without wrapping it.

    This is a source assertion about the declared implementation, not engine
    detection. A replacement must supply its own reviewed declaration; copying
    function attributes with functools.wraps does not transfer the contract.
    """
    if (
        not isinstance(compute_type, str)
        or not compute_type.endswith(("_simt", "_tensor"))
        or not isinstance(contract_id, str)
        or not contract_id
    ):
        raise ValueError("Invalid implementation engine contract")

    def declare(implementation: Callable) -> Callable:
        implementation._tokenspeed_profile_engine_contract = (
            _ImplementationEngineContract(implementation, compute_type, contract_id)
        )
        return implementation

    return declare


def _implementation_engine_contract(implementation: Callable) -> tuple[str, str] | None:
    """Resolve a declaration only when it belongs to this exact callable."""
    contract = getattr(implementation, "_tokenspeed_profile_engine_contract", None)
    if (
        not isinstance(contract, _ImplementationEngineContract)
        or contract.implementation is not implementation
        or not isinstance(contract.compute_type, str)
        or not contract.compute_type.endswith(("_simt", "_tensor"))
        or not isinstance(contract.contract_id, str)
        or not contract.contract_id
    ):
        return None
    return contract.compute_type, contract.contract_id


def record_selected_kernel(
    name: str, solution: str, *, implementation: Callable
) -> None:
    """Attach the actual leaf and its implementation-owned proof to this op.

    Ordered tuples retain every dispatch, including repeated leaves. A single
    engine summary is available only for one leaf with a reviewed declaration;
    multiple leaves need a separate FLOP partition and remain unnormalized.
    Reserved selected_* fields never replace a caller's compute_type argument.
    Outside collection this does nothing and never changes execution selection.
    """
    metadata = current_op_metadata()
    if metadata is None:
        return
    if (
        not isinstance(name, str)
        or not name
        or not isinstance(solution, str)
        or not solution
    ):
        raise ValueError("Selected kernel identity requires name and solution")
    contract = _implementation_engine_contract(implementation)
    compute_type, engine_contract_id = (
        contract if contract is not None else (None, None)
    )
    source_path, source_sha256 = _profile_source(implementation)
    for key, value in (
        ("selected_kernel_names", name),
        ("selected_solutions", solution),
        ("selected_compute_types", compute_type),
        ("selected_engine_contract_ids", engine_contract_id),
        ("selected_source_paths", source_path),
        ("selected_source_sha256s", source_sha256),
    ):
        metadata[key] = (*metadata.get(key, ()), value)
    single = len(metadata["selected_kernel_names"]) == 1
    metadata["selected_kernel_name"] = name if single else None
    metadata["selected_solution"] = solution if single else None
    metadata["selected_compute_type"] = compute_type if single else None
    metadata["selected_engine_contract_id"] = engine_contract_id if single else None


class ProfileSession:
    """Collect host metadata within explicit, synchronous iteration windows.

    Args:
        run_id: Artifact namespace.
        model_id: Model identity, including target/draft role when applicable.
        rank: Global process rank.
        backend: Explicit scope backend: none, nvtx or torch. Collectors are external.
    """

    def __init__(self, *, run_id: str, model_id: str, rank: int, backend: str) -> None:
        if backend not in {"none", "nvtx", "torch"}:
            raise ValueError(f"Unsupported recording backend: {backend}")
        self.backend = backend
        self.run_id = run_id
        self.model_id = model_id
        self.rank = rank
        self.ops: list[OpRecord] = []
        self.layers: list[LayerRecord] = []
        self._identities: dict[int, tuple[object, int]] = {}
        self._iterations: set[int] = set()
        self._active = False
        self._counts: dict[tuple[int, int | None], int] = {}
        self._occurrences: dict[tuple[int, int], int] = {}
        self._timeline: ProfileObserver | None = None

    @staticmethod
    def current() -> ProfileSession | None:
        """The active session in this execution context, or None."""
        context = _current.get()
        return context.session if context is not None else None

    def record_func(self, label: str, *, display_name: str | None = None):
        """Annotate an identity; native NVTX may display a shorter name.

        Omitting display_name displays the full identity. Collector lifetime
        stays external; this method does not measure GPU time.
        """
        if self._timeline is not None:
            return self._timeline.record_func(
                label, display_name=label if display_name is None else display_name
            )
        if self.backend == "nvtx":
            return torch.cuda.nvtx.range(label)
        if self.backend == "torch":
            return torch.profiler.record_function(label)
        return nullcontext()

    def trace_label(
        self,
        *,
        level: str,
        name: str,
        iter_id: int,
        layer_id: int | None,
        occurrence: int | None,
        inner_id: int | None,
    ) -> str:
        """Return a stable serialized identity shared by metadata and trace events."""
        # Kineto rewrites quotes in event names; encode names so both backends
        # preserve exactly the identity stored in our metadata.
        identity = "ts.profile/" + "/".join(
            quote(str(value), safe="")
            for value in (
                self.run_id,
                self.model_id,
                self.rank,
                level,
                iter_id,
                layer_id,
                occurrence,
                inner_id,
            )
        )
        readable_name = quote(name, safe="._")
        scope = f"layer={layer_id}"
        scope += f" | op={readable_name}" if level == "op" else f" | {readable_name}"
        call = inner_id if level == "op" else occurrence
        return f"{scope} | rank={self.rank} iter={iter_id} call={call} | {identity}"

    def bind_layer(self, instance: object, *, layer_id: int | None) -> None:
        """Bind an instance to its declared global layer number; return None."""
        if self._active:
            raise RuntimeError("Bind layer identities before recording")
        if layer_id is not None and layer_id < 0:
            raise ValueError("layer_id must be nonnegative")
        self._identities[id(instance)] = (instance, layer_id)

    def is_bound(self, instance: object) -> bool:
        """Whether this session has an explicit identity for the instance."""
        return id(instance) in self._identities

    @contextmanager
    def observe(self, observer: ProfileObserver) -> Iterator[None]:
        """Attach one scope/metadata observer outside iteration windows."""
        if self._timeline is not None or self._active:
            raise RuntimeError(
                "Nested observers or attachment during an iteration are unsupported"
            )
        self._timeline = observer
        try:
            yield
        finally:
            if self._active:
                raise RuntimeError("Exit iteration before detaching its observer")
            self._timeline = None

    def observed_by(self, observer: ProfileObserver) -> bool:
        """Whether this exact observer owns the session's current collection."""
        return self._timeline is observer

    def snapshot_integer_values(
        self, source: torch.Tensor, target: dict, name: str
    ) -> None:
        if self._timeline is not None:
            self._timeline.snapshot_integer_values(source, target, name)
        elif source.device.type == "cpu":
            target[name] = tuple(source.tolist())

    @contextmanager
    def layer(
        self, instance: object, *, name: str, display_name: str
    ) -> Iterator[None]:
        """Record a bound module/method scope using the active iteration."""
        context = _current.get()
        if context is None or context.session is not self:
            yield
            return
        identity = self._identities.get(id(instance))
        if identity is None:
            raise ValueError("Bind the layer instance before recording")
        layer_id = identity[1]
        key = (context.iter_id, layer_id)
        occurrence = self._occurrences.get(key, 0)
        self._occurrences[key] = occurrence + 1
        token = _current.set(
            _Context(self, context.iter_id, layer_id, occurrence, context.metadata)
        )
        label = self.trace_label(
            level="layer",
            name=name,
            iter_id=context.iter_id,
            layer_id=layer_id,
            occurrence=occurrence,
            inner_id=None,
        )
        start = time.perf_counter_ns()
        error = None
        try:
            with self.record_func(label, display_name=display_name):
                yield
        except BaseException as exc:
            error = type(exc).__name__
            raise
        finally:
            _current.reset(token)
            self.layers.append(
                LayerRecord(
                    context.iter_id,
                    layer_id,
                    occurrence,
                    start,
                    time.perf_counter_ns() - start,
                    error,
                    label,
                )
            )

    @contextmanager
    def watch_module(
        self, module: torch.nn.Module, *, layer_id: int | None, name: str
    ) -> Iterator[None]:
        """Observe a module through standard PyTorch hooks, preserving its forward.

        Hooks are active only for this session's iterations and removed on exit;
        captured graph replay contains no Python module hooks.
        """
        self.bind_layer(module, layer_id=layer_id)
        if getattr(module.forward, "_tokenspeed_profiled_layer", False):
            yield
            return
        stack = []

        def before(instance, args):
            context = _current.get()
            if context is not None and context.session is self:
                scope = self.layer(instance, name=name, display_name=name)
                scope.__enter__()
                stack.append(scope)

        def after(instance, args, output):
            if stack:
                stack.pop().__exit__(*sys.exc_info())

        first = module.register_forward_pre_hook(before)
        last = module.register_forward_hook(after, always_call=True)
        try:
            yield
        finally:
            first.remove()
            last.remove()
            if stack:
                raise RuntimeError("Unclosed module profiling scopes")

    @contextmanager
    def iteration(
        self, *, iter_id: int, metadata: dict[str, Any] | None = None
    ) -> Iterator[None]:
        """Record one eager iteration; IDs must be unique within this session."""
        if self._active or _current.get() is not None:
            raise RuntimeError("Nested or concurrent session windows are unsupported")
        if iter_id in self._iterations:
            raise ValueError(f"Iteration {iter_id} was already recorded")
        self._iterations.add(iter_id)
        self._active = True
        token = _current.set(_Context(self, iter_id, None, None, metadata))
        try:
            yield
        finally:
            _current.reset(token)
            self._active = False

    def report(self, *, costs: CostRegistry) -> dict[str, Any]:
        """Return collected metadata and costs without retaining tensor storage."""
        return {
            "schema_version": 1,
            "run_id": self.run_id,
            "model_id": self.model_id,
            "rank": self.rank,
            "timing_kind": "cpu_scope",
            "backend": self.backend,
            "execution_mode": "eager",
            "ops": [asdict(record) for record in self.ops],
            "layers": [asdict(record) for record in self.layers],
            "analysis": costs.analyze(self),
        }

    def export(self, path: str | Path, *, costs: CostRegistry) -> None:
        """Write metadata and offline op/layer costs to JSON; return None."""
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.report(costs=costs), indent=2) + "\n")


def profiled(*, level: str) -> Callable:
    """Decorate a synchronous layer method, preserving its result and signature.

    Args:
        level: Must be 'layer' in this initial implementation.
    Returns:
        A decorator using the active session's explicit instance identity map.
    """
    if level != "layer":
        raise ValueError("Only layer profiling is implemented")

    def decorate(fn: Callable) -> Callable:
        if inspect.iscoroutinefunction(fn) or inspect.isgeneratorfunction(fn):
            raise TypeError("Profiling requires a synchronous function")

        @wraps(fn)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            context = _current.get()
            if context is None:
                return fn(*args, **kwargs)
            session = context.session
            instance = args[0] if args else kwargs.get("self")
            with session.layer(
                instance,
                name=f"{type(instance).__name__}.{fn.__name__}",
                display_name=f"L{session._identities.get(id(instance), (None, None))[1]}",
            ):
                return fn(*args, **kwargs)

        wrapped._tokenspeed_profiled_layer = True

        return wrapped

    return decorate


def _profile_argument(arguments: dict[str, Any], path: str) -> Any:
    """Resolve declared storage fields; never execute a property for metadata."""
    pieces = path.split(".")
    value = arguments.get(pieces[0])
    for piece in pieces[1:]:
        if value is None or not piece or piece.startswith("__"):
            return None
        if isinstance(value, dict):
            value = value.get(piece)
        elif isinstance(value, tuple) and piece in getattr(type(value), "_fields", ()):
            value = value[type(value)._fields.index(piece)]
        elif isinstance(value, torch.nn.Module):
            fields = vars(value)
            if piece in fields:
                value = fields[piece]
            else:
                value = next(
                    (
                        fields[registry][piece]
                        for registry in ("_parameters", "_buffers", "_modules")
                        if piece in fields.get(registry, {})
                    ),
                    None,
                )
        else:
            try:
                value = vars(value).get(piece)
            except TypeError:
                return None
    return value


def wrap_op_for_profiling(
    fn: Callable, *, spec: KernelSpec, profile: OpProfile
) -> Callable:
    """Collect an existing synchronous function without changing its dispatch.

    Kernel registration uses this same wrapper. Direct bindings use spec only
    for profiling identity and do not add entries to the kernel registry.
    """
    if (
        not inspect.isfunction(fn)
        or inspect.iscoroutinefunction(fn)
        or inspect.isgeneratorfunction(fn)
    ):
        raise TypeError(
            "OpProfile currently supports synchronous registered functions only"
        )
    signature = inspect.signature(fn)
    scalar_names = profile.scalar_args + tuple(
        name
        for name, values in spec.traits.items()
        if len(values) > 1
        and name in signature.parameters
        and name not in profile.scalar_args + profile.integer_tensor_args
    )
    metadata_args = scalar_names + profile.integer_tensor_args + profile.tensor_args
    if any(name.split(".", 1)[0] not in signature.parameters for name in metadata_args):
        raise ValueError("Profile metadata arguments must name function parameters")
    if len(metadata_args) != len(set(metadata_args)):
        raise ValueError("Profile metadata arguments must be distinct")
    op_name = f"{spec.family}.{spec.mode}"
    # A singleton trait fixes the format; multiple values only describe support.
    static_traits = {}
    for name, values in spec.traits.items():
        if len(values) == 1:
            (value,) = values
            if type(value) in (type(None), bool, int, float, str):
                static_traits[name] = value

    @wraps(fn)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        context = _current.get()
        if context is None or _recording_op.get() is not None:
            return fn(*args, **kwargs)
        session = context.session
        inputs: dict[str, TensorDescription] = {}
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        for key, value in bound.arguments.items():
            _describe(value, key, inputs)
        for name in profile.tensor_args:
            _describe(_profile_argument(bound.arguments, name), name, inputs)
        scalar_args = dict(context.metadata or {})
        source_path, source_sha256 = _profile_source(fn)
        scalar_args.update(
            profile_source_path=source_path,
            profile_source_sha256=source_sha256,
            profile_rank=session.rank,
        )
        _record_torch_execution_context(bound.arguments, scalar_args)
        tensor_values = {}
        if metadata_args:
            for name in scalar_names:
                value = _profile_argument(bound.arguments, name)
                values = value if isinstance(value, tuple) else (value,)
                if any(
                    type(item) not in (type(None), bool, int, float, str)
                    for item in values
                ):
                    if name in profile.scalar_args:
                        raise TypeError(
                            f"Profile argument {name} must contain host scalars"
                        )
                    continue
                scalar_args[name] = value
            for name in profile.integer_tensor_args:
                value = _profile_argument(bound.arguments, name)
                _describe(value, name, inputs)
                tensor_values[name] = None
                # These are bounded metadata vectors, never activations or
                # weights. Unsupported arguments leave a missing-cost reason.
                if (
                    isinstance(value, torch.Tensor)
                    and 1 <= value.ndim <= 2
                    and value.dtype in (torch.int32, torch.int64)
                    and value.is_contiguous()
                    and value.numel() <= profile.integer_tensor_max_elements
                ):
                    session.snapshot_integer_values(value.view(-1), tensor_values, name)

        counter_key = (context.iter_id, context.layer_id)
        inner_id = session._counts.get(counter_key, 0)
        session._counts[counter_key] = inner_id + 1
        outputs: dict[str, TensorDescription] = {}
        autocast_enabled = torch.is_autocast_enabled(
            "cuda"
        ) or torch.is_autocast_enabled("cpu")
        error = None
        start = time.perf_counter_ns()
        label = session.trace_label(
            level="op",
            name=op_name,
            iter_id=context.iter_id,
            layer_id=context.layer_id,
            occurrence=context.occurrence,
            inner_id=inner_id,
        )
        token = _recording_op.set(scalar_args)
        input_token = _recording_inputs.set(inputs)
        values_token = _recording_values.set(tensor_values)
        if spec.solution:
            record_selected_kernel(spec.name, spec.solution, implementation=fn)
        try:
            if profile.metadata_preparer is not None:
                profile.metadata_preparer(bound.arguments)
            start = time.perf_counter_ns()
            with session.record_func(label, display_name=op_name):
                result = fn(*args, **kwargs)
        except BaseException as exc:
            error = type(exc).__name__
            raise
        else:
            _describe(result, "return", outputs)
            input_tensors = {}

            def tensors(value, path, target):
                if isinstance(value, torch.Tensor):
                    target[path] = value
                elif isinstance(value, (tuple, list)):
                    for index, item in enumerate(value):
                        tensors(item, f"{path}[{index}]", target)

            for name, value in bound.arguments.items():
                tensors(value, name, input_tensors)
            output_tensors = {}
            tensors(result, "return", output_tensors)
            scalar_args["output_aliases"] = tuple(
                f"{out_name}={in_name}"
                for out_name, out_tensor in output_tensors.items()
                for in_name, in_tensor in input_tensors.items()
                if out_tensor.device == in_tensor.device
                and out_tensor.untyped_storage().data_ptr()
                == in_tensor.untyped_storage().data_ptr()
                and out_tensor.numel()
                and in_tensor.numel()
            )
            return result
        finally:
            _recording_op.reset(token)
            _recording_inputs.reset(input_token)
            _recording_values.reset(values_token)
            session.ops.append(
                OpRecord(
                    iter_id=context.iter_id,
                    layer_id=context.layer_id,
                    occurrence=context.occurrence,
                    inner_id=inner_id,
                    op_name=op_name,
                    implementation=spec.name,
                    cost_rule=profile.cost_rule,
                    inputs=inputs,
                    outputs=outputs,
                    cpu_start_ns=start,
                    cpu_duration_ns=time.perf_counter_ns() - start,
                    error=error,
                    trace_label=label,
                    autocast_enabled=autocast_enabled,
                    scalar_args=scalar_args,
                    tensor_values=tensor_values,
                    static_traits=static_traits,
                )
            )

    return wrapped


@dataclass(frozen=True)
class ComputeComponent:
    """A disjoint useful-FLOP partition; None preserves an unresolved engine."""

    useful_flops: int
    compute_type: str | None
    contract_id: str | None = None
    missing_reason: str | None = None

    def __post_init__(self) -> None:
        if type(self.useful_flops) is not int or self.useful_flops < 0:
            raise ValueError("Component FLOPs must be nonnegative integers")
        if self.compute_type is not None and (
            not isinstance(self.compute_type, str)
            or not self.compute_type.endswith(("_simt", "_tensor"))
        ):
            raise ValueError("Component type must identify precision and engine")


def record_integer_selection(name: str, producer: Callable[[], torch.Tensor]) -> None:
    """Capture a bounded device selection through the existing observer slab.

    The observer owns the producer's CUDA-node difference and snapshot lifetime.
    No observer means unavailable; there is no synchronous device read fallback.
    """
    context = _current.get()
    values, inputs = _recording_values.get(), _recording_inputs.get()
    if context is None or values is None or inputs is None:
        return
    values[name] = None
    observer = context.session._timeline
    if observer is not None:
        observer.snapshot_integer_selection(producer, values, inputs, name)


def profile_function(*, family: str, name: str, profile: OpProfile) -> Callable:
    """Observe a source facade without registering or changing its execution."""
    from tokenspeed_kernel.registry import KernelSpec

    def decorate(fn: Callable) -> Callable:
        return wrap_op_for_profiling(
            fn,
            spec=KernelSpec(name=fn.__name__, family=family, mode=name),
            profile=profile,
        )

    return decorate


def record_compiled_kernel(compiled: Any, *, contract_id: str) -> None:
    """Retain the actual launch-returned Triton artifact during collection only.

    No device queries or synchronization. PTX is evidence about this compiled
    launch, never a filename/cache candidate. Providers apply source/dtype guards.
    """
    metadata = current_op_metadata()
    if metadata is None or compiled is None:
        return
    asm = getattr(compiled, "asm", {})
    ptx, ttir = asm.get("ptx", ""), asm.get("ttir", "")
    if not isinstance(ptx, str) or not isinstance(ttir, str):
        return
    metadata["compiled_contract_id"] = contract_id
    metadata["compiled_kernel_hash"] = str(getattr(compiled, "hash", ""))
    for key, value in (("ptx", ptx), ("ttir", ttir)):
        metadata[f"compiled_{key}_sha256"] = hashlib.sha256(value.encode()).hexdigest()
    cubin = asm.get("cubin")
    if isinstance(cubin, bytes):
        metadata["compiled_cubin_sha256"] = hashlib.sha256(cubin).hexdigest()
    source = getattr(getattr(getattr(compiled, "src", None), "fn", None), "src", None)
    if isinstance(source, str):
        metadata["compiled_source_sha256"] = hashlib.sha256(source.encode()).hexdigest()
    target = re.search(r"\.target\s+sm_([0-9]+[a-z]?)", ptx)
    metadata["compiled_target"] = target.group(1) if target else None
    metadata["compiled_dot_tf32"] = bool(
        re.search(r"mma\.sync\.aligned\.[^;\n]*\.f32\.tf32\.tf32\.f32", ptx)
        and re.search(r"inputPrecision\s*=\s*tf32(?:[>,\s]|$)", ttir)
        and not re.search(r"tf32x[36]|bf16x[369]", ttir)
    )
    metadata["compiled_bf16_mul_fp32_add"] = bool(
        "mul.bf16x2" in ptx
        and "add.f32" in ptx
        and "fma.rn.bf16" not in ptx
        and "fma.rn.f32" not in ptx
    )


def record_torch_bf16_add(
    output: torch.Tensor, shared: torch.Tensor, result: torch.Tensor
) -> None:
    """Record the source branch; the offline binder owns artifact proof."""
    metadata = current_op_metadata()
    if metadata is not None:
        metadata["pytorch_add_source_branch"] = "torch.Tensor.__add__:alpha1"


@dataclass(frozen=True)
class OpCost:
    useful_flops: int
    executed_flops: int | None
    logical_read_bytes: int
    logical_write_bytes: int
    estimated_hbm_bytes: int | None
    rule: str
    assumptions: tuple[str, ...]
    # Cost-rule normalization key, not a measurement of executed instructions.
    compute_type: str | None = None
    communication: CollectiveTraffic | None = None
    # Omission keeps the legacy whole-op compute_type contract.
    compute_components: tuple[ComputeComponent, ...] | None = None
    host_mapped_read_bytes: int | None = None

    def __post_init__(self) -> None:
        if self.compute_components is not None:
            if any(
                not isinstance(item, ComputeComponent)
                for item in self.compute_components
            ):
                raise ValueError("Expected ComputeComponent entries")
            if (
                sum(item.useful_flops for item in self.compute_components)
                != self.useful_flops
            ):
                raise ValueError(
                    "Compute components must exactly partition useful FLOPs"
                )


class CostRegistry:
    """Explicit offline rule registry; missing costs remain unavailable."""

    def __init__(self) -> None:
        self._rules: dict[str, Callable[[OpRecord], OpCost]] = {}
        self._engine_resolver: (
            Callable[[OpRecord, OpCost, dict], dict | None] | None
        ) = None

    def register(self, name: str, rule: Callable[[OpRecord], OpCost]) -> None:
        """Register a named rule; reject duplicate names."""
        if name in self._rules:
            raise ValueError(f"Duplicate cost rule: {name}")
        self._rules[name] = rule

    def register_engine_resolver(
        self, resolver: Callable[[OpRecord, OpCost, dict], dict | None]
    ) -> None:
        """Install one trusted offline provider; captures never choose code."""
        if self._engine_resolver is not None:
            raise ValueError("Selected-engine resolver already registered")
        self._engine_resolver = resolver

    def analyze_recorded(
        self, report: dict, *, engine_context: dict[str, dict] | None = None
    ) -> dict:
        """Apply offline rules to serialized OpRecords with their actual metadata."""
        session = ProfileSession(
            run_id=report["run_id"],
            model_id=report["model_id"],
            rank=report["rank"],
            backend="none",
        )
        for item in report["ops"]:
            op = dict(item)
            for key in ("inputs", "outputs"):
                op[key] = {
                    name: TensorDescription(
                        **{
                            **value,
                            "shape": tuple(value["shape"]),
                            "stride": tuple(value["stride"]),
                        }
                    )
                    for name, value in op[key].items()
                }
            # ranks is a declared collective tuple, serialized as a JSON array.
            # Do not recursively reinterpret arbitrary host metadata.
            if isinstance(op.get("scalar_args", {}).get("ranks"), list):
                op["scalar_args"] = {
                    **op["scalar_args"],
                    "ranks": tuple(op["scalar_args"]["ranks"]),
                }
            session.ops.append(OpRecord(**op))
        session.layers = [LayerRecord(**item) for item in report["layers"]]
        return self.analyze(session, engine_context=engine_context)

    def analyze(
        self, session: ProfileSession, *, engine_context: dict[str, dict] | None = None
    ) -> dict[str, Any]:
        """Return per-op costs and per-layer sums with explicit coverage.

        CPU scopes cannot provide device MFU/MBU denominators. These metrics
        remain unavailable without device timing and matching hardware peaks.
        """
        rows = []
        groups: dict[tuple[int, int | None], list[dict[str, Any]]] = {}
        for op in session.ops:
            rule = self._rules.get(op.cost_rule) if op.cost_rule is not None else None
            cost = None
            reason = None
            engine_proof = None
            if op.error is not None:
                reason = f"op failed: {op.error}"
            elif rule is None:
                reason = "cost rule unavailable"
            else:
                try:
                    cost = rule(op)
                    context = (engine_context or {}).get(op.trace_label)
                    eager = (
                        context is not None
                        and context.get("attribution_mode")
                        == "eager_nvtx_gpu_projection"
                    )
                    proof = None
                    if self._engine_resolver is not None and context is not None:
                        proof = self._engine_resolver(op, cost, context)
                    if (
                        eager
                        and cost.useful_flops
                        and (
                            proof is None
                            or "unavailable_reason" not in proof
                            and (
                                not context.get("complete_join")
                                or proof.get("status") != "bound"
                            )
                        )
                    ):
                        proof = {
                            "status": "unavailable",
                            "attribution_mode": context["attribution_mode"],
                            "scope_identity": context["scope_identity"],
                            "unavailable_reason": context.get("incomplete_join_reason")
                            or "eager native compute engine proof unavailable",
                        }
                    engine_proof = proof
                    if proof is not None and "unavailable_reason" in proof:
                        cost = replace(
                            cost,
                            compute_type=None,
                            compute_components=(
                                ComputeComponent(
                                    cost.useful_flops,
                                    None,
                                    missing_reason=proof["unavailable_reason"],
                                ),
                            ),
                        )
                    elif proof is not None:
                        components = tuple(
                            ComputeComponent(**item) for item in proof["components"]
                        )
                        if eager and any(
                            item.useful_flops
                            and item.compute_type is not None
                            and not item.contract_id
                            for item in components
                        ):
                            raise ValueError(
                                "Eager engine proof needs a reviewed component contract ID"
                            )
                        known = {}
                        for item in cost.compute_components or ():
                            if item.compute_type is not None:
                                known[item.compute_type] = (
                                    known.get(item.compute_type, 0) + item.useful_flops
                                )
                        if (
                            cost.compute_components is None
                            and cost.compute_type is not None
                        ):
                            known[cost.compute_type] = (
                                known.get(cost.compute_type, 0) + cost.useful_flops
                            )
                        proposed = {}
                        for item in components:
                            proposed[item.compute_type] = (
                                proposed.get(item.compute_type, 0) + item.useful_flops
                            )
                        if any(
                            proposed.get(key, 0) < amount
                            for key, amount in known.items()
                        ):
                            raise ValueError(
                                "Selected-engine evidence conflicts with the existing source contract"
                            )
                        cost = replace(
                            cost, compute_type=None, compute_components=components
                        )
                except ValueError as exc:
                    reason = str(exc)
                    if engine_proof is not None:
                        engine_proof = {
                            "status": "rejected",
                            "unavailable_reason": reason,
                            "rejected_candidate": engine_proof,
                        }
                    if cost is not None:
                        cost = replace(
                            cost,
                            compute_type=None,
                            compute_components=(
                                ComputeComponent(
                                    cost.useful_flops,
                                    None,
                                    missing_reason=f"rejected engine proof: {exc}",
                                ),
                            ),
                        )
            row = {
                "iter_id": op.iter_id,
                "layer_id": op.layer_id,
                "inner_id": op.inner_id,
                "cost": asdict(cost) if cost is not None else None,
                "missing_reason": reason,
                "engine_proof": engine_proof,
            }
            rows.append(row)
            groups.setdefault((op.iter_id, op.layer_id), []).append(row)
        layers = []
        for (iter_id, layer_id), entries in groups.items():
            covered = [entry["cost"] for entry in entries if entry["cost"] is not None]
            sums = {
                field: (
                    sum(cost[field] for cost in covered)
                    if all(cost[field] is not None for cost in covered)
                    else None
                )
                for field in (
                    "useful_flops",
                    "executed_flops",
                    "logical_read_bytes",
                    "logical_write_bytes",
                )
            }
            layers.append(
                {
                    "iter_id": iter_id,
                    "layer_id": layer_id,
                    "op_count": len(entries),
                    "covered_op_count": len(covered),
                    "covered_cost": sums,
                    "complete_cost": sums if len(covered) == len(entries) else None,
                    "mfu": None,
                    "estimated_mbu": None,
                    "utilization_missing_reason": "device timing and matched hardware peaks unavailable",
                }
            )
        return {
            "coverage_scope": "opted_in_registered_calls",
            "ops": rows,
            "layers": layers,
        }

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

"""
tokenspeed_kernel build script.

Compiles CUDA sources into TVM-FFI shared libraries and PyTorch pybind
extensions. On systems without an NVIDIA CUDA build target, the build is
skipped and the package installs as a pure-Python stub.
"""

import ctypes
import importlib
import os
import shutil
import site
import subprocess
import sys
import sysconfig
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from setuptools import Command, Distribution, find_packages, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop
from setuptools.command.editable_wheel import editable_wheel

ROOT = Path(__file__).resolve().parent
REQUIREMENTS_DIR = ROOT / "requirements"
BASE_VERSION = "0.1.3"
BACKEND_ENV = "TOKENSPEED_KERNEL_BACKEND"
VALID_BACKENDS = {"cuda", "rocm"}
DEFAULT_CUDA_ARCHS = ("100a", "103a")

# CUDA kernel sources live outside the Python package and are included in sdists.
CUDA_CSRC_DIR = ROOT / "csrc" / "cuda"

CUDA_HOME = os.environ.get("CUDA_HOME", "/usr/local/cuda")
NVCC = os.environ.get("FLASHINFER_NVCC", f"{CUDA_HOME}/bin/nvcc")
CXX = os.environ.get("CXX", "g++")


def _version_date() -> str:
    override = os.environ.get("TOKENSPEED_KERNEL_VERSION_DATE")
    if override:
        return override

    source_date_epoch = os.environ.get("SOURCE_DATE_EPOCH")
    if source_date_epoch:
        return datetime.fromtimestamp(int(source_date_epoch), tz=timezone.utc).strftime(
            "%Y%m%d"
        )

    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _git_sha() -> str:
    override = os.environ.get("TOKENSPEED_KERNEL_GIT_SHA") or os.environ.get(
        "GIT_COMMIT"
    )
    if override:
        return override[:8].ljust(8, "0")

    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short=8", "HEAD"],
                cwd=ROOT,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            .strip()[:8]
            .ljust(8, "0")
        )
    except (OSError, subprocess.CalledProcessError):
        return "00000000"


def _git_branch() -> str:
    for env_name in (
        "TOKENSPEED_KERNEL_GIT_BRANCH",
        "GITHUB_REF_NAME",
    ):
        branch = os.environ.get(env_name)
        if branch:
            return branch.removeprefix("refs/heads/")

    github_ref = os.environ.get("GITHUB_REF")
    if github_ref:
        return github_ref.removeprefix("refs/heads/")

    try:
        return subprocess.check_output(
            ["git", "branch", "--show-current"],
            cwd=ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _package_version() -> str:
    if _git_branch().startswith("release/"):
        return BASE_VERSION

    return f"{BASE_VERSION}.dev{_version_date()}+git{_git_sha()}"


def _is_cuda_platform() -> bool:
    def toolkit_available() -> bool:
        if shutil.which(NVCC) is not None:
            return True
        cuda_home = Path(CUDA_HOME)
        return (cuda_home / "bin" / "nvcc").exists()

    for lib_name in ("libcuda.so.1", "libcuda.so"):
        try:
            libcuda = ctypes.CDLL(lib_name)
            break
        except OSError:
            pass
    else:
        return toolkit_available()

    try:
        if libcuda.cuInit(0) != 0:
            return toolkit_available()
        count = ctypes.c_int()
        if libcuda.cuDeviceGetCount(ctypes.byref(count)) != 0:
            return toolkit_available()
        if count.value > 0:
            return True
    except AttributeError:
        pass

    return toolkit_available()


def _is_rocm_platform() -> bool:
    rocm_env_names = (
        "ROCM_HOME",
        "ROCM_PATH",
        "ROCM_VERSION",
        "HIP_PATH",
        "HIP_PLATFORM",
    )
    if any(os.environ.get(name) for name in rocm_env_names):
        return True
    if shutil.which("hipcc") is not None:
        return True
    if Path("/dev/kfd").exists():
        return True
    return Path("/opt/rocm").exists()


def _selected_backend() -> str:
    override = os.environ.get(BACKEND_ENV, "").strip().lower()
    if override:
        if override not in VALID_BACKENDS:
            valid = ", ".join(sorted(VALID_BACKENDS))
            raise RuntimeError(f"{BACKEND_ENV} must be one of: {valid}")
        return override

    if _is_cuda_platform():
        return "cuda"
    if _is_rocm_platform():
        return "rocm"

    raise RuntimeError(
        "Unable to detect CUDA or ROCm for tokenspeed_kernel dependencies. "
        f"Set {BACKEND_ENV}=cuda or {BACKEND_ENV}=rocm."
    )


def _read_requirements(path: Path, seen=None) -> list[str]:
    seen = seen or set()
    path = path.resolve()
    if path in seen:
        return []
    seen.add(path)

    requirements = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        include = None
        if line.startswith("-r ") or line.startswith("--requirement "):
            include = line.split(maxsplit=1)[1]
        elif line.startswith("-r") and len(line) > 2:
            include = line[2:].strip()
        elif line.startswith("--requirement="):
            include = line.split("=", maxsplit=1)[1].strip()
        if include:
            requirements.extend(_read_requirements(path.parent / include, seen))
            continue
        if line.startswith("-"):
            # Installer options such as --extra-index-url are not valid
            # project dependency metadata.
            continue
        requirements.append(line)
    return requirements


def _selected_install_requires() -> list[str]:
    backend = _selected_backend()
    requirements = _read_requirements(REQUIREMENTS_DIR / f"{backend}.txt")
    requirements.extend(
        _read_requirements(REQUIREMENTS_DIR / f"{backend}-thirdparty.txt")
    )

    deduped = []
    seen = set()
    for requirement in requirements:
        if requirement not in seen:
            deduped.append(requirement)
            seen.add(requirement)
    return deduped


def _pip_verbose_args(verbose) -> list[str]:
    try:
        level = int(verbose)
    except (TypeError, ValueError):
        level = 1 if verbose else 0
    return ["-" + ("v" * min(level, 3))] if level > 0 else []


def _refresh_python_install_paths() -> None:
    """Expose packages installed by subprocess pip to this build process."""
    candidates = []
    for paths in (site.getsitepackages(), site.getusersitepackages()):
        if isinstance(paths, str):
            candidates.append(paths)
        else:
            candidates.extend(paths)

    for path in candidates:
        if path and Path(path).exists():
            site.addsitedir(str(path))

    importlib.invalidate_caches()


def _install_backend_build_requirements(verbose=False) -> None:
    backend = _selected_backend()
    print(f"Installing {backend} build requirements before native build")
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-r",
            str(REQUIREMENTS_DIR / f"{backend}.txt"),
            "--no-build-isolation",
        ]
        + _pip_verbose_args(verbose)
    )

    # The same setup.py process imports build deps immediately after pip adds
    # them. If pip created user site-packages during this run, that path was not
    # present when Python started, so add site paths before resolving headers.
    _refresh_python_install_paths()


def _ensure_cuda_compiler() -> None:
    if shutil.which(NVCC) is None:
        raise RuntimeError(f"CUDA backend selected but nvcc was not found: {NVCC}")


# Kernel groups: (name, sources, output package, extra ldflags, extra cflags).
# Each group produces <output package>/objs/<name>/<name>.so.
KERNEL_GROUPS = [
    (
        "rope",
        [
            CUDA_CSRC_DIR / "rope.cu",
            CUDA_CSRC_DIR / "flashinfer_rope_binding.cu",
        ],
        "tokenspeed_kernel.ops.embedding.cuda",
        [],
        [],
    ),
    (
        "deepseek_v4_attention",
        [
            CUDA_CSRC_DIR / "deepseek_v4_attention.cu",
            CUDA_CSRC_DIR / "deepseek_v4_topk.cu",
            CUDA_CSRC_DIR / "deepseek_v4_attention_binding.cu",
        ],
        "tokenspeed_kernel.ops.model.deepseek_v4.cuda",
        [],
        [],
    ),
    (
        "minimax_m3_fused",
        [
            CUDA_CSRC_DIR / "fused_minimax_m3_qknorm_rope_kv_insert.cu",
        ],
        "tokenspeed_kernel.ops.model.minimax_m3.cuda",
        [],
        [],
    ),
    (
        "dsv3_gemm",
        [
            CUDA_CSRC_DIR / "dsv3_router_gemm_float_out.cu",
            CUDA_CSRC_DIR / "dsv3_router_gemm.cu",
            CUDA_CSRC_DIR / "dsv3_router_gemm_binding.cu",
        ],
        "tokenspeed_kernel.ops.gemm.fp16.cuda",
        ["-lcublas", "-lcublasLt"],
        [],
    ),
    (
        "marlin",
        [
            CUDA_CSRC_DIR / "gptq_marlin_repack.cu",
            CUDA_CSRC_DIR / "flashinfer_marlin_binding.cu",
        ],
        "tokenspeed_kernel.ops.quantization.cuda",
        [],
        [],
    ),
    (
        "routing",
        [
            CUDA_CSRC_DIR / "routing_flash.cu",
        ],
        "tokenspeed_kernel.ops.moe.routing.cuda",
        [],
        [],
    ),
    (
        "sampling_chain",
        [
            CUDA_CSRC_DIR / "sampling_chain.cu",
            CUDA_CSRC_DIR / "flashinfer_sampling_chain_binding.cu",
        ],
        "tokenspeed_kernel.ops.sampling.cuda",
        [],
        [],
    ),
    (
        "fused_topk_topp",
        [
            CUDA_CSRC_DIR / "fused_topk_topp" / "fused_topk_topp.cu",
            CUDA_CSRC_DIR / "fused_topk_topp" / "fused_topk_topp_binding.cu",
        ],
        "tokenspeed_kernel.ops.sampling.cuda",
        [],
        # --expt-extended-lambda is required by air_topk_stable.cuh's CUB usage.
        ["--expt-extended-lambda"],
    ),
    (
        "rmsnorm_fused_parallel",
        [
            CUDA_CSRC_DIR / "rmsnorm_fused_parallel.cu",
            CUDA_CSRC_DIR / "flashinfer_rmsnorm_fused_parallel_binding.cu",
        ],
        "tokenspeed_kernel.ops.layernorm.cuda",
        [],
        [],
    ),
    (
        "merge_state",
        [
            CUDA_CSRC_DIR / "merge_state.cu",
        ],
        "tokenspeed_kernel.ops.attention.merge_state.cuda",
        [],
        [],
    ),
    (
        "flashinfer_softmax",
        [
            CUDA_CSRC_DIR / "flashinfer_softmax.cu",
        ],
        "tokenspeed_kernel.ops.sampling.flashinfer",
        [],
        [],
    ),
    (
        "silu_fuse_block_quant",
        [
            CUDA_CSRC_DIR / "silu_and_mul_fuse_block_quant.cu",
            CUDA_CSRC_DIR / "silu_and_mul_fuse_block_quant_ep.cu",
        ],
        "tokenspeed_kernel.ops.activation.cuda",
        [],
        [],
    ),
    (
        "silu_fuse_nvfp4_quant",
        [
            CUDA_CSRC_DIR / "silu_and_mul_fuse_nvfp4_quant.cu",
        ],
        "tokenspeed_kernel.ops.activation.cuda",
        [],
        [],
    ),
    (
        "moe_finalize_fuse_shared",
        [
            CUDA_CSRC_DIR / "moe_finalize_fuse_shared.cu",
        ],
        "tokenspeed_kernel.ops.moe.finalize.cuda",
        [],
        [],
    ),
    (
        "kvcacheio",
        [
            CUDA_CSRC_DIR / "kvcacheio_transfer.cu",
            CUDA_CSRC_DIR / "flashinfer_kvcacheio_binding.cu",
        ],
        "tokenspeed_kernel.ops.kvcache.cuda",
        [],
        [],
    ),
    (
        "lm_head_gemm",
        [
            CUDA_CSRC_DIR / "lm_head_gemm.cu",
            CUDA_CSRC_DIR / "lm_head_gemm_binding.cu",
        ],
        "tokenspeed_kernel.ops.gemm.fp16.cuda",
        [],
        [],
    ),
    (
        "trtllm_comm",
        [
            CUDA_CSRC_DIR / "trtllm_allreduce.cu",
            CUDA_CSRC_DIR / "trtllm_allreduce_fusion.cu",
            CUDA_CSRC_DIR / "trtllm_mnnvl_allreduce_fusion.cu",
            CUDA_CSRC_DIR / "trtllm_reducescatter_fusion.cu",
            CUDA_CSRC_DIR / "trtllm_allgather_fusion.cu",
            CUDA_CSRC_DIR / "minimax_reduce_rms.cu",
        ],
        "tokenspeed_kernel.ops.communication.trtllm",
        [],
        [],
    ),
    (
        "attn_res",
        [
            CUDA_CSRC_DIR / "attn_res" / "attn_res_fwd_tma.cu",
            CUDA_CSRC_DIR / "attn_res_binding.cu",
        ],
        "tokenspeed_kernel.ops.model.kimi_k3.attn_res.cuda",
        [],
        [],
    ),
]

# PyTorch pybind extensions are built separately from the TVM-FFI groups.
PYTORCH_CUDA_EXTENSION_GROUPS = [
    (
        "sparse_build_k2q_csr_ext",
        CUDA_CSRC_DIR / "msa" / "build_k2q_csr.cu",
        "tokenspeed_kernel.ops.attention.msa.cuda",
    ),
    (
        "sparse_decode_schedule_ext",
        CUDA_CSRC_DIR / "msa" / "build_decode_schedule.cu",
        "tokenspeed_kernel.ops.attention.msa.cuda",
    ),
]


def _package_dir(package: str) -> Path:
    return ROOT.joinpath(*package.split("."))


def _kernel_output_path(name: str, output_package: str) -> Path:
    return _package_dir(output_package) / "objs" / name / f"{name}.so"


def _python_extension_output_path(name: str, output_package: str) -> Path:
    suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    return _package_dir(output_package) / "objs" / name / f"{name}{suffix}"


def _validate_kernel_groups(kernel_groups) -> None:
    missing_sources = [
        source
        for _name, sources, _package, _ldflags, _cflags in kernel_groups
        for source in sources
        if not source.is_file()
    ]
    if missing_sources:
        missing = "\n".join(f"  - {path}" for path in missing_sources)
        raise FileNotFoundError(f"CUDA kernel sources are missing:\n{missing}")

    missing_packages = sorted(
        {
            package
            for _name, _sources, package, _ldflags, _cflags in kernel_groups
            if not _package_dir(package).is_dir()
        }
    )
    if missing_packages:
        missing = "\n".join(f"  - {package}" for package in missing_packages)
        raise FileNotFoundError(f"CUDA output packages are missing:\n{missing}")


def _validate_pytorch_extension_groups(extension_groups) -> None:
    missing_sources = [
        source for _name, source, _package in extension_groups if not source.is_file()
    ]
    if missing_sources:
        missing = "\n".join(f"  - {path}" for path in missing_sources)
        raise FileNotFoundError(
            f"PyTorch CUDA extension sources are missing:\n{missing}"
        )

    missing_packages = sorted(
        {
            package
            for _name, _source, package in extension_groups
            if not _package_dir(package).is_dir()
        }
    )
    if missing_packages:
        missing = "\n".join(f"  - {package}" for package in missing_packages)
        raise FileNotFoundError(f"PyTorch CUDA output packages are missing:\n{missing}")


class CudaKernelBuilder:
    def __init__(self, kernel_groups, verbose: bool, pytorch_extension_groups=()):
        self.kernel_groups = kernel_groups
        self.pytorch_extension_groups = pytorch_extension_groups
        self.verbose = verbose

    # Target GPU architectures: detect from the CUDA driver or use env var override.
    # FLASHINFER_CUDA_ARCH_LIST is accepted for compatibility, but TokenSpeed
    # docs prefer TOKENSPEED_CUDA_ARCH=100 on GB200.
    def _normalize_cuda_arch(self, arch):
        has_suffix = arch.endswith("a")
        arch_clean = arch.rstrip("a")
        if "." in arch_clean:
            major_s, minor_s = arch_clean.split(".", 1)
            major = int(major_s)
            minor = int(minor_s)
        else:
            major = int(arch_clean[:-1])
            minor = int(arch_clean[-1])
        suffix = "a" if has_suffix or major >= 9 else ""
        return f"{major}{minor}{suffix}"

    def _detect_cuda_archs(self):
        archs = set()

        arch_list = os.environ.get("FLASHINFER_CUDA_ARCH_LIST", "")
        if arch_list:
            for arch in arch_list.split():
                archs.add(self._normalize_cuda_arch(arch))
            return archs

        direct = os.environ.get("TOKENSPEED_CUDA_ARCH", "")
        if direct:
            archs.add(self._normalize_cuda_arch(direct))
            return archs

        if not archs:
            archs.update(DEFAULT_CUDA_ARCHS)
        return archs

    def _site_paths(self):
        paths = []
        try:
            paths.extend(site.getsitepackages())
        except Exception:
            pass
        paths.extend(sys.path)

        seen = set()
        for raw_path in paths:
            if not raw_path:
                continue
            path = Path(raw_path).expanduser()
            path_str = str(path)
            if path.exists() and path_str not in seen:
                seen.add(path_str)
                yield path

    def _read_cuda_header_version(self, include_dir: Path):
        header = include_dir / "cuda_runtime_api.h"
        if not header.exists():
            return None

        try:
            for line in header.read_text(
                encoding="utf-8", errors="ignore"
            ).splitlines():
                if line.startswith("#define CUDART_VERSION"):
                    parts = line.split()
                    if len(parts) >= 3:
                        version = int(parts[2])
                        return version // 1000, (version % 1000) // 10
        except (OSError, ValueError):
            return None

        return None

    def _nvcc_toolkit_version(self):
        try:
            output = subprocess.check_output(
                [NVCC, "--version"],
                stderr=subprocess.STDOUT,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return None

        marker = "release "
        for line in output.splitlines():
            if marker not in line:
                continue
            version_text = line.split(marker, 1)[1].split(",", 1)[0].strip()
            parts = version_text.split(".")
            if len(parts) >= 2:
                try:
                    return int(parts[0]), int(parts[1])
                except ValueError:
                    return None
        return None

    def _cuda_toolkit_roots(self):
        roots = [Path(CUDA_HOME)]

        seen = set()
        for root in roots:
            root_str = str(root)
            if root.exists() and root_str not in seen:
                seen.add(root_str)
                yield root

    def _resolve_include_dirs(self):
        dirs = [str(CUDA_CSRC_DIR / "include"), str(CUDA_CSRC_DIR)]
        seen = set(dirs)

        def _add_dir(path: Path) -> None:
            path_str = str(path)
            if path.exists() and path_str not in seen:
                dirs.append(path_str)
                seen.add(path_str)

        def _is_complete_cuda_include(path: Path) -> bool:
            return all(
                (path / header).exists() for header in ("cuda_runtime.h", "cublas_v2.h")
            )

        found_toolkit_headers = False
        for cuda_root in self._cuda_toolkit_roots():
            cuda_include = cuda_root / "include"
            if not _is_complete_cuda_include(cuda_include):
                continue
            _add_dir(cuda_include)
            if (cuda_include / "cccl").exists():
                _add_dir(cuda_include / "cccl")
            found_toolkit_headers = True
            break

        # Do not mix wheel CUDA headers with an available toolkit.
        if not found_toolkit_headers:
            nvcc_version = self._nvcc_toolkit_version()
            found_wheel_headers = False
            for base_path in self._site_paths():
                for candidate in sorted(
                    base_path.glob("nvidia/cu*/include"), reverse=True
                ):
                    if not _is_complete_cuda_include(candidate):
                        continue
                    # The nvidia-cuda-runtime wheels may lag the nvcc minor
                    # version. Mixing them trips CCCL's toolkit compatibility
                    # check, so only use matching fallback headers.
                    header_version = self._read_cuda_header_version(candidate)
                    if (
                        nvcc_version
                        and header_version
                        and header_version != nvcc_version
                    ):
                        if self.verbose:
                            print(
                                "Skipping CUDA include with mismatched toolkit "
                                f"version: {candidate} "
                                f"({header_version[0]}.{header_version[1]} != nvcc "
                                f"{nvcc_version[0]}.{nvcc_version[1]})"
                            )
                        continue
                    _add_dir(candidate)
                    if (candidate / "cccl").exists():
                        _add_dir(candidate / "cccl")
                    found_wheel_headers = True
                    break
                if found_wheel_headers:
                    break

        try:
            tvm_ffi = importlib.import_module("tvm_ffi")
            _add_dir(Path(tvm_ffi.__file__).parent / "include")
        except ImportError:
            pass

        # flashinfer bundles TRT-LLM internal FP4 helpers
        # (tensorrt_llm/kernels/quantization_utils.cuh: cvt_warp_fp16_to_fp4,
        # silu_and_mul, cvt_quant_to_fp4_get_sf_out_offset). Expose them so
        # our own fused silu+mul+nvfp4 kernel can reuse them.
        try:
            flashinfer = importlib.import_module("flashinfer")
            fi_root = Path(flashinfer.__file__).parent / "data"
            for sub in (
                fi_root / "csrc" / "nv_internal",
                fi_root / "csrc" / "nv_internal" / "include",
                fi_root / "include",
                fi_root / "cutlass" / "include",
            ):
                _add_dir(sub)
            spdlog = fi_root / "spdlog" / "include"
            if (spdlog / "spdlog" / "spdlog.h").exists():
                _add_dir(spdlog)
                return dirs
        except ImportError:
            pass
        if (Path("/usr/include") / "spdlog" / "spdlog.h").exists():
            _add_dir(Path("/usr/include"))

        return dirs

    def _resolve_cuda_lib_flags(self):
        cuda_home = Path(CUDA_HOME)
        lib_candidates = []
        for cuda_root in self._cuda_toolkit_roots():
            lib_candidates.extend([cuda_root / "lib64", cuda_root / "lib"])
        for base in self._site_paths():
            lib_candidates.extend(
                sorted(Path(base).glob("nvidia/cu*/lib"), reverse=True)
            )

        seen_lib_dirs = set()
        unique_lib_candidates = []
        for candidate in lib_candidates:
            candidate_str = str(candidate)
            if candidate.exists() and candidate_str not in seen_lib_dirs:
                unique_lib_candidates.append(candidate)
                seen_lib_dirs.add(candidate_str)
        lib_candidates = unique_lib_candidates
        self._cuda_library_dirs = lib_candidates

        cuda_lib_dir = lib_candidates[0] if lib_candidates else cuda_home / "lib64"
        for candidate in lib_candidates:
            if (candidate / "libcudart.so").exists() or list(
                candidate.glob("libcudart.so.*")
            ):
                cuda_lib_dir = candidate
                break

        flags = [f"-L{lib_dir}" for lib_dir in lib_candidates] or [f"-L{cuda_lib_dir}"]
        cuda_stubs_dir = cuda_lib_dir / "stubs"
        if cuda_stubs_dir.exists():
            flags.append(f"-L{cuda_stubs_dir}")

        cudart_so = cuda_lib_dir / "libcudart.so"
        cudart_versioned = sorted(cuda_lib_dir.glob("libcudart.so.*"))
        if cudart_so.exists():
            flags.append("-lcudart")
        elif cudart_versioned:
            flags.append(f"-l:{cudart_versioned[-1].name}")
        else:
            flags.append("-lcudart")

        flags.append("-lcuda")
        return flags

    def _resolve_library_ldflag(self, ldflag):
        if not ldflag.startswith("-l") or ldflag.startswith("-l:"):
            return ldflag

        lib_name = ldflag[2:]
        for lib_dir in getattr(self, "_cuda_library_dirs", []):
            if (lib_dir / f"lib{lib_name}.so").exists():
                return ldflag
            versioned = sorted(lib_dir.glob(f"lib{lib_name}.so.*"))
            if versioned:
                return f"-l:{versioned[-1].name}"
        return ldflag

    def _prepare_cuda_toolchain_env(self):
        path = os.environ.get("PATH", "")
        path_entries = [entry for entry in path.split(os.pathsep) if entry]
        candidates = [Path(NVCC).resolve().parent]

        for cuda_root in self._cuda_toolkit_roots():
            candidates.append(cuda_root / "bin")
            candidates.append(cuda_root / "nvvm" / "bin")

        for base in self._site_paths():
            for cuda_root in sorted(Path(base).glob("nvidia/cu*"), reverse=True):
                candidates.append(cuda_root / "bin")
                candidates.append(cuda_root / "nvvm" / "bin")

        for candidate in reversed(candidates):
            candidate_str = str(candidate)
            if candidate.exists() and candidate_str not in path_entries:
                path_entries.insert(0, candidate_str)
        if path_entries:
            os.environ["PATH"] = os.pathsep.join(path_entries)

    def _compile_one(self, src, obj, nvcc_flags, include_dirs, extra_cflags=()):
        include_flags = [f"-I{d}" for d in include_dirs]
        cmd = (
            [NVCC]
            + nvcc_flags
            + list(extra_cflags)
            + include_flags
            + ["-c", str(src), "-o", str(obj)]
        )
        subprocess.check_call(cmd)
        return obj

    def _build_pytorch_extensions(self, nvcc_flags, include_dirs, ldflags):
        _validate_pytorch_extension_groups(self.pytorch_extension_groups)
        if not self.pytorch_extension_groups:
            return

        # Build requirements are installed by BuildNative before this method
        # runs, so resolve PyTorch dynamically rather than at setup import time.
        torch = importlib.import_module("torch")
        cpp_extension = importlib.import_module("torch.utils.cpp_extension")
        extension_include_dirs = list(include_dirs)
        for include_dir in [
            *cpp_extension.include_paths(device_type="cuda"),
            sysconfig.get_paths()["include"],
        ]:
            if include_dir not in extension_include_dirs:
                extension_include_dirs.append(include_dir)

        torch_library_dirs = cpp_extension.library_paths(device_type="cuda")
        torch_ldflags = [f"-L{path}" for path in torch_library_dirs] + [
            "-Wl,--no-as-needed",
            "-lc10",
            "-ltorch",
            "-ltorch_cpu",
            "-ltorch_python",
            "-lc10_cuda",
            "-ltorch_cuda",
        ]
        abi = int(torch._C._GLIBCXX_USE_CXX11_ABI)

        stale_groups = []
        for name, source, output_package in self.pytorch_extension_groups:
            output = _python_extension_output_path(name, output_package)
            output.parent.mkdir(parents=True, exist_ok=True)
            if output.exists() and all(
                output.stat().st_mtime > dependency.stat().st_mtime
                for dependency in (source, Path(__file__))
            ):
                continue
            stale_groups.append((name, source, output))

        print(
            f"Building {len(stale_groups)}/{len(self.pytorch_extension_groups)} "
            "PyTorch CUDA extension(s)..."
        )
        for name, source, output in stale_groups:
            obj = output.parent / f"{source.stem}.o"
            extension_flags = [
                "-lineinfo",
                "-DTORCH_API_INCLUDE_EXTENSION_H",
                f"-DTORCH_EXTENSION_NAME={name}",
                f"-D_GLIBCXX_USE_CXX11_ABI={abi}",
            ]
            self._compile_one(
                source,
                obj,
                nvcc_flags,
                extension_include_dirs,
                extension_flags,
            )
            subprocess.check_call(
                [CXX, str(obj)] + ldflags + torch_ldflags + ["-o", str(output)]
            )

    def run(self):
        _validate_kernel_groups(self.kernel_groups)
        self._prepare_cuda_toolchain_env()
        max_jobs = int(os.environ.get("MAX_JOBS", min(os.cpu_count() or 1, 16)))
        total_sources = sum(len(entry[1]) for entry in self.kernel_groups)

        archs = self._detect_cuda_archs()
        gencode_flags = [
            f"-gencode=arch=compute_{a},code=sm_{a}" for a in sorted(archs)
        ]
        nvcc_flags = [
            "-std=c++17",
            "-O3",
            "-DNDEBUG",
            "-use_fast_math",
            "--expt-relaxed-constexpr",
            "--compiler-options=-fPIC",
            "-DFLASHINFER_ENABLE_BF16",
            "-DFLASHINFER_ENABLE_F16",
            "-DENABLE_BF16",
            "-DENABLE_FP8",
        ] + gencode_flags
        include_dirs = self._resolve_include_dirs()
        ldflags = ["-shared"] + self._resolve_cuda_lib_flags()
        self._build_pytorch_extensions(nvcc_flags, include_dirs, ldflags)

        build_dependencies = [
            path
            for path in CUDA_CSRC_DIR.rglob("*")
            if path.is_file() and "msa" not in path.relative_to(CUDA_CSRC_DIR).parts
        ]
        build_dependencies.append(Path(__file__))

        stale_groups = []
        skipped_groups = 0
        for (
            name,
            sources,
            output_package,
            extra_ldflags,
            extra_cflags,
        ) in self.kernel_groups:
            so_path = _kernel_output_path(name, output_package)
            out_dir = so_path.parent
            out_dir.mkdir(parents=True, exist_ok=True)
            if so_path.exists() and all(
                so_path.stat().st_mtime > dependency.stat().st_mtime
                for dependency in build_dependencies
            ):
                skipped_groups += 1
                continue
            stale_groups.append((name, sources, extra_ldflags, extra_cflags, so_path))

        stale_sources = sum(len(srcs) for _, srcs, _, _, _ in stale_groups)
        print(
            f"Building {len(stale_groups)}/{len(self.kernel_groups)} kernel group(s) "
            f"({stale_sources}/{total_sources} files, {max_jobs} parallel jobs)..."
        )
        if skipped_groups and self.verbose:
            print(f"Skipped {skipped_groups} up-to-date kernel group(s)")

        if not stale_groups:
            return

        with ThreadPoolExecutor(max_workers=max_jobs) as executor:
            group_meta = []
            futures = []
            for name, sources, extra_ldflags, extra_cflags, so_path in stale_groups:
                out_dir = so_path.parent
                objects = []
                for src in sources:
                    obj = out_dir / (src.stem + ".o")
                    objects.append(obj)
                    futures.append(
                        executor.submit(
                            self._compile_one,
                            str(src),
                            str(obj),
                            nvcc_flags,
                            include_dirs,
                            extra_cflags,
                        )
                    )
                group_meta.append((name, objects, extra_ldflags, so_path))

            for future in as_completed(futures):
                future.result()

        for name, objects, extra_ldflags, so_path in group_meta:
            extra_ldflags = [
                self._resolve_library_ldflag(ldflag) for ldflag in (extra_ldflags or [])
            ]
            link_cmd = (
                [CXX]
                + [str(o) for o in objects]
                + ldflags
                + extra_ldflags
                + ["-o", str(so_path)]
            )
            subprocess.check_call(link_cmd)


class BuildKernels(build_ext):
    """Compile CUDA kernels into .so files for the CUDA backend."""

    def run(self):
        if _selected_backend() != "cuda":
            print(
                f"CUDA backend not selected; skipping CUDA kernel build. "
                f"{self.distribution.get_name()}"
            )
            return

        _ensure_cuda_compiler()
        verbose = bool(getattr(self, "verbose", False))
        CudaKernelBuilder(
            KERNEL_GROUPS,
            verbose=verbose,
            pytorch_extension_groups=PYTORCH_CUDA_EXTENSION_GROUPS,
        ).run()


class BuildNative(Command):
    description = "Build CUDA kernels"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        backend = _selected_backend()
        _install_backend_build_requirements(getattr(self, "verbose", False))
        if backend != "cuda":
            print("CUDA backend not selected; skipping CUDA kernel build")
            return

        self.run_command("build_ext")


class EditableWheelWithBuild(editable_wheel):
    """Ensure kernels are built during `pip install -e .` (PEP 660)."""

    def run(self):
        self.run_command("build_native")
        super().run()


class DevelopWithBuild(develop):
    """Ensure kernels are built during `setup.py develop`."""

    def run(self):
        self.run_command("build_native")
        super().run()


class BuildPyWithBuild(build_py):
    """Ensure kernels are built for regular installs."""

    def run(self):
        self.run_command("build_native")
        package_build_dir = Path(self.build_lib) / "tokenspeed_kernel"
        if package_build_dir.exists():
            shutil.rmtree(package_build_dir)
        super().run()


class BinaryDistribution(Distribution):
    """Mark CUDA wheels as platform-specific distributions."""

    def has_ext_modules(self):
        return _selected_backend() == "cuda"


CUDA_OUTPUT_PACKAGES = sorted(
    {entry[2] for entry in KERNEL_GROUPS}
    | {entry[2] for entry in PYTORCH_CUDA_EXTENSION_GROUPS}
)
PACKAGE_DATA = {
    # Pre-swept flashinfer MoE tactic tables.
    "tokenspeed_kernel.ops.model.kimi_k3": ["tactics/*.json"],
    # MSA runtime documentation and optional CuTeDSL requirements.
    "tokenspeed_kernel.ops.attention.msa": ["README.md"],
    "tokenspeed_kernel.ops.attention.msa.cute_dsl": [
        "README.md",
        "requirements.txt",
    ],
    **(
        {package: ["objs/**/*.so"] for package in CUDA_OUTPUT_PACKAGES}
        if _selected_backend() == "cuda"
        else {}
    ),
}


setup(
    name="tokenspeed_kernel",
    version=_package_version(),
    install_requires=_selected_install_requires(),
    packages=find_packages(),
    package_data=PACKAGE_DATA,
    distclass=BinaryDistribution,
    cmdclass={
        "build_native": BuildNative,
        "build_ext": BuildKernels,
        "build_py": BuildPyWithBuild,
        "editable_wheel": EditableWheelWithBuild,
        "develop": DevelopWithBuild,
    },
)

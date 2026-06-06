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

Compiles .cu files into shared libraries (.so) loaded via tvm_ffi.load_module().
On systems without an NVIDIA CUDA build target, the build is skipped and the
package installs as a pure-Python stub.
"""

import importlib
import os
import shutil
import site
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from setuptools import Command, find_packages, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop
from setuptools.command.editable_wheel import editable_wheel

ROOT = Path(__file__).resolve().parent
REQUIREMENTS_DIR = ROOT / "requirements"
THIRDPARTY_DIR = ROOT / "tokenspeed_kernel_nvidia" / "thirdparty"
BASE_VERSION = "0.1.0"
PACKAGE_ENV = "TOKENSPEED_KERNEL_PACKAGE"
VALID_PACKAGE_MODES = {"core", "nvidia", "amd"}
DIST_NAMES = {
    "core": "tokenspeed-kernel",
    "nvidia": "tokenspeed-kernel-nvidia",
    "amd": "tokenspeed-kernel-amd",
}
PACKAGE_BACKENDS = {
    "core": None,
    "nvidia": "cuda",
    "amd": "rocm",
}

# CUDA kernels source and output directories
CUDA_CSRC_DIR = THIRDPARTY_DIR / "cuda" / "csrc"
CUDA_OBJS_DIR = THIRDPARTY_DIR / "cuda" / "objs"

# JIT kernels source directory (no pre-compilation, just need sources available)
JIT_CSRC_DIR = THIRDPARTY_DIR / "jit_kernel" / "csrc"

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


def _package_mode() -> str:
    mode = os.environ.get(PACKAGE_ENV, "core").strip().lower()
    if mode not in VALID_PACKAGE_MODES:
        valid = ", ".join(sorted(VALID_PACKAGE_MODES))
        raise RuntimeError(f"{PACKAGE_ENV} must be one of: {valid}")
    return mode


def _distribution_name() -> str:
    return DIST_NAMES[_package_mode()]


def _vendor_distribution_requirement(mode: str, version: str) -> str:
    return f"{DIST_NAMES[mode]}=={version}"


def _selected_packages() -> list[str]:
    prefix = {
        "core": "tokenspeed_kernel",
        "nvidia": "tokenspeed_kernel_nvidia",
        "amd": "tokenspeed_kernel_amd",
    }[_package_mode()]
    return [
        package
        for package in find_packages()
        if package == prefix or package.startswith(prefix + ".")
    ]


def _package_backend() -> str | None:
    return PACKAGE_BACKENDS[_package_mode()]


def _read_requirements(
    path: Path, seen=None, *, include_nested: bool = True
) -> list[str]:
    seen = seen or set()
    path = path.resolve()
    if path in seen:
        return []
    seen.add(path)

    requirements = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("--"):
            continue
        if line.startswith("-r ") or line.startswith("--requirement "):
            if include_nested:
                include = line.split(maxsplit=1)[1]
                requirements.extend(_read_requirements(path.parent / include, seen))
            continue
        requirements.append(line)
    return requirements


def _dedupe_requirements(requirements: list[str]) -> list[str]:
    deduped = []
    seen = set()
    for requirement in requirements:
        if requirement not in seen:
            deduped.append(requirement)
            seen.add(requirement)
    return deduped


def _vendor_requirements(backend: str) -> list[str]:
    requirements = []
    requirements.extend(
        _read_requirements(REQUIREMENTS_DIR / f"{backend}.txt", include_nested=False)
    )
    requirements.extend(
        _read_requirements(REQUIREMENTS_DIR / f"{backend}-thirdparty.txt")
    )
    return requirements


def _selected_install_requires() -> list[str]:
    backend = _package_backend()
    if backend is None:
        requirements = _read_requirements(REQUIREMENTS_DIR / "common.txt")
    else:
        requirements = _vendor_requirements(backend)
    return _dedupe_requirements(requirements)


def _selected_extras_require() -> dict[str, list[str]]:
    if _package_mode() != "core":
        return {}

    version = _package_version()
    nvidia = [_vendor_distribution_requirement("nvidia", version)]
    amd = [_vendor_distribution_requirement("amd", version)]
    return {
        "nvidia": nvidia,
        "amd": amd,
        "all": nvidia + amd,
    }


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


def _install_backend_build_requirements(backend: str, verbose=False) -> None:
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


def _cuda_compiler_available() -> bool:
    return shutil.which(NVCC) is not None or Path(NVCC).exists()


# Kernel groups: each entry produces one .so file.
# Format: (name, [source_files], extra_ldflags) or
#         (name, [source_files], extra_ldflags, extra_cflags)
# The 4-tuple form lets a kernel append nvcc flags on top of the global set —
# e.g., fused_topk_topp needs ``--expt-extended-lambda`` for CUB lambdas.
KERNEL_GROUPS = [
    (
        "rope",
        [
            CUDA_CSRC_DIR / "rope.cu",
            CUDA_CSRC_DIR / "flashinfer_rope_binding.cu",
        ],
        [],
    ),
    (
        "deepseek_v4_attention",
        [
            CUDA_CSRC_DIR / "deepseek_v4_attention.cu",
            CUDA_CSRC_DIR / "deepseek_v4_topk.cu",
            CUDA_CSRC_DIR / "deepseek_v4_attention_binding.cu",
        ],
        [],
    ),
    (
        "dsv3_gemm",
        [
            CUDA_CSRC_DIR / "dsv3_router_gemm_float_out.cu",
            CUDA_CSRC_DIR / "dsv3_router_gemm.cu",
            CUDA_CSRC_DIR / "dsv3_router_gemm_binding.cu",
        ],
        ["-lcublas", "-lcublasLt"],
    ),
    (
        "fp32_router_gemm",
        [
            CUDA_CSRC_DIR / "fp32_router_gemm.cu",
            CUDA_CSRC_DIR / "fp32_router_gemm_entry.cu",
            CUDA_CSRC_DIR / "fp32_router_gemm_binding.cu",
        ],
        ["-lcublas", "-lcublasLt"],
    ),
    (
        "marlin",
        [
            CUDA_CSRC_DIR / "gptq_marlin_repack.cu",
            CUDA_CSRC_DIR / "flashinfer_marlin_binding.cu",
        ],
        [],
    ),
    (
        "routing",
        [
            CUDA_CSRC_DIR / "routing_flash.cu",
        ],
        [],
    ),
    (
        "sampling_chain",
        [
            CUDA_CSRC_DIR / "sampling_chain.cu",
            CUDA_CSRC_DIR / "flashinfer_sampling_chain_binding.cu",
        ],
        [],
    ),
    (
        "fused_topk_topp",
        [
            CUDA_CSRC_DIR / "fused_topk_topp" / "fused_topk_topp.cu",
            CUDA_CSRC_DIR / "fused_topk_topp" / "fused_topk_topp_binding.cu",
        ],
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
        [],
    ),
    (
        "merge_state",
        [
            CUDA_CSRC_DIR / "merge_state.cu",
        ],
        [],
    ),
    (
        "flashinfer_softmax",
        [
            CUDA_CSRC_DIR / "flashinfer_softmax.cu",
        ],
        [],
    ),
    (
        "silu_fuse_block_quant",
        [
            CUDA_CSRC_DIR / "silu_and_mul_fuse_block_quant.cu",
            CUDA_CSRC_DIR / "silu_and_mul_fuse_block_quant_ep.cu",
        ],
        [],
    ),
    (
        "silu_fuse_nvfp4_quant",
        [
            CUDA_CSRC_DIR / "silu_and_mul_fuse_nvfp4_quant.cu",
        ],
        [],
    ),
    (
        "moe_finalize_fuse_shared",
        [
            CUDA_CSRC_DIR / "moe_finalize_fuse_shared.cu",
        ],
        [],
    ),
    (
        "kvcacheio",
        [
            CUDA_CSRC_DIR / "kvcacheio_transfer.cu",
            CUDA_CSRC_DIR / "flashinfer_kvcacheio_binding.cu",
        ],
        [],
    ),
    (
        "lm_head_gemm",
        [
            CUDA_CSRC_DIR / "lm_head_gemm.cu",
            CUDA_CSRC_DIR / "lm_head_gemm_binding.cu",
        ],
        [],
    ),
    (
        "trtllm_comm",
        [
            CUDA_CSRC_DIR / "trtllm_allreduce.cu",
            CUDA_CSRC_DIR / "trtllm_allreduce_fusion.cu",
            CUDA_CSRC_DIR / "trtllm_reducescatter_fusion.cu",
            CUDA_CSRC_DIR / "trtllm_allgather_fusion.cu",
            CUDA_CSRC_DIR / "minimax_reduce_rms.cu",
        ],
        [],
    ),
]


class CudaKernelBuilder:
    def __init__(self, kernel_groups, verbose: bool):
        self.kernel_groups = kernel_groups
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
            archs.add("100a")
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

        for cuda_root in self._cuda_toolkit_roots():
            cuda_include = cuda_root / "include"
            if (cuda_include / "cuda_runtime.h").exists():
                _add_dir(cuda_include)
            if (cuda_include / "cccl").exists():
                _add_dir(cuda_include / "cccl")

        for base_path in self._site_paths():
            for candidate in sorted(base_path.glob("nvidia/cu*/include"), reverse=True):
                if (candidate / "cuda_runtime.h").exists():
                    _add_dir(candidate)
                if (candidate / "cccl").exists():
                    _add_dir(candidate / "cccl")

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

    def run(self):
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

        # Ensure output directory exists
        CUDA_OBJS_DIR.mkdir(parents=True, exist_ok=True)

        stale_groups = []
        skipped_groups = 0
        for entry in self.kernel_groups:
            name, sources, extra_ldflags = entry[0], entry[1], entry[2]
            extra_cflags = entry[3] if len(entry) > 3 else []
            out_dir = CUDA_OBJS_DIR / name
            out_dir.mkdir(parents=True, exist_ok=True)
            so_path = out_dir / f"{name}.so"
            if so_path.exists() and all(
                so_path.stat().st_mtime > src.stat().st_mtime for src in sources
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
    """Compile CUDA kernels into .so files for the NVIDIA package."""

    def run(self):
        if _package_mode() != "nvidia":
            print(
                f"NVIDIA package not selected; skipping CUDA kernel build. "
                f"{self.distribution.get_name()}"
            )
            return
        if not _cuda_compiler_available():
            print(f"nvcc not found ({NVCC}); skipping CUDA kernel build")
            return

        verbose = bool(getattr(self, "verbose", False))
        CudaKernelBuilder(KERNEL_GROUPS, verbose=verbose).run()


class BuildNative(Command):
    description = "Build CUDA kernels"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        if _package_mode() != "nvidia":
            print("NVIDIA package not selected; skipping native CUDA build")
            return
        if not _cuda_compiler_available():
            print(f"nvcc not found ({NVCC}); skipping native CUDA build")
            return

        _install_backend_build_requirements("cuda", getattr(self, "verbose", False))
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
    """Ensure kernels and vendored deps are built for regular installs."""

    def run(self):
        self.run_command("build_native")
        super().run()


setup(
    name=_distribution_name(),
    version=_package_version(),
    install_requires=_selected_install_requires(),
    extras_require=_selected_extras_require(),
    packages=_selected_packages(),
    package_data={
        "tokenspeed_kernel_nvidia.thirdparty.cuda": ["objs/**/*.so"],
    },
    license_files=["LICENSE", "THIRDPARTYNOTICES"],
    cmdclass={
        "build_native": BuildNative,
        "build_ext": BuildKernels,
        "build_py": BuildPyWithBuild,
        "editable_wheel": EditableWheelWithBuild,
        "develop": DevelopWithBuild,
    },
)

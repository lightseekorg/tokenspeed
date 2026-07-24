# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Per-variant lazy JIT compilation for FMHA varlen kernels.

Each FMHA variant (dtype x tile x sparse x page x split_kv x pack_factor) is compiled
independently on first use and cached to ~/.cache/minfer/fmha_sm100/.

To recompile after kernel changes: scripts/clear_fmha_cache.sh
"""

import fcntl
import itertools
import logging
import os
import shutil
import subprocess
import threading
from pathlib import Path

import jinja2

logger = logging.getLogger(__name__)


def _compute_cache_base():
    explicit = os.environ.get("MINFER_FMHA_CACHE_DIR")
    if explicit:
        return Path(explicit)
    base = Path(os.path.expanduser("~/.cache/minfer/fmha_sm100"))
    # Different build configs get separate cache dirs to avoid conflicts
    suffix = ""
    if os.environ.get("GPU_TRACE") is not None:
        suffix += "_gpu_trace"
    if os.environ.get("SM_TIMING") is not None:
        suffix += "_sm_timing"
    if os.environ.get("FMHA_GMEM_CHECK") is not None:
        suffix += "_gmem_check"
    if suffix:
        base = base.parent / (base.name + suffix)
    return base


CACHE_BASE = _compute_cache_base()


def _acquire_file_lock(lock_path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o666)
    fcntl.flock(fd, fcntl.LOCK_EX)
    return fd


def _release_file_lock(fd):
    fcntl.flock(fd, fcntl.LOCK_UN)
    os.close(fd)


# Kernel sources and CUTLASS headers are shipped inside the package directory
# so that JIT compilation works from both editable and wheel installs.
_PACKAGE_DIR = Path(__file__).resolve().parent
_FMHA_VARLEN_DIR = _PACKAGE_DIR / "csrc"


# TokenSpeed patch: the upstream cutlass/ submodule is not vendored; resolve
# CUTLASS headers from the env override, a package-local checkout, or the
# flashinfer wheel's bundled tree (see thirdparty/msa/README.md).
def _find_cutlass_dir():
    explicit = os.environ.get("TOKENSPEED_MSA_CUTLASS_DIR")
    if explicit:
        return Path(explicit)
    local = _PACKAGE_DIR / "cutlass"
    if (local / "include" / "cutlass").is_dir():
        return local
    try:
        import flashinfer

        bundled = Path(flashinfer.__file__).resolve().parent / "data" / "cutlass"
        if (bundled / "include" / "cutlass").is_dir():
            return bundled
    except ImportError:
        pass
    raise RuntimeError(
        "Cannot find CUTLASS headers for the MSA FMHA JIT. Set "
        "TOKENSPEED_MSA_CUTLASS_DIR to a CUTLASS checkout, or install "
        "flashinfer (its wheel bundles the headers)."
    )


_PACK_FACTORS = [1, 2, 4, 6, 8, 16]
# _PACK_FACTORS = [1, 6]

# DLPack dtype codes (must match tvm_ffi_utils.h encode_dlpack_dtype)
_BFLOAT16_CODE = (4 << 16) | (16 << 8) | 1  # 266241
_FLOAT8_E4M3FN_CODE = (12 << 16) | (8 << 8) | 1  # 788481

_FMHA_SM100_DISPATCH = [
    (
        "int64_t dtype_code",
        [
            (
                _BFLOAT16_CODE,
                {"dtype_in": "nv_bfloat16", "cutlass_dtype_out": "cutlass::bfloat16_t"},
            ),
            (
                _FLOAT8_E4M3FN_CODE,
                {
                    "dtype_in": "__nv_fp8_e4m3",
                    "cutlass_dtype_out": "cutlass::bfloat16_t",
                },
            ),
        ],
    ),
    (
        "int qo_tile_size",
        [
            (128, {"tile_q": "_128", "tile_kv": "_256", "thread_shape": "_1, _2, _1"}),
            (256, {"tile_q": "_256", "tile_kv": "_128", "thread_shape": "_2, _1, _1"}),
        ],
    ),
    (
        "bool single_wg",
        [
            ("true", {"single_wg": "true"}),
            ("false", {"single_wg": "false"}),
        ],
    ),
    (
        "int sparse_mode",
        [
            (0, {"sparse_mode": "Sparse"}),
            (1, {"sparse_mode": "Full"}),
            (2, {"sparse_mode": "OnlyScore"}),
            (None, {"sparse_mode": "Off"}),
        ],
    ),
    (
        "int page_size",
        [
            (-1, {"page_size": -1}),
            (128, {"page_size": 128}),
            # (256, {"page_size": 256}),
        ],
    ),
    (
        "bool split_kv",
        [
            ("false", {"is_split_kv": "false"}),
            ("true", {"is_split_kv": "true"}),
        ],
    ),
    ("int pack_factor", [(i, {"pack_factor": i}) for i in _PACK_FACTORS]),
]

_FMHA_SM100_IMPOSSIBLE = lambda p: (
    (p.get("tile_q") == "_256" and p.get("single_wg") == "true")
    or (p.get("tile_q") == "_256" and p.get("is_split_kv") == "true")
    or (p.get("page_size") == -1 and p.get("sparse_mode") == "Sparse")
    or (p.get("pack_factor", 1) > 1 and p.get("tile_q") == "_256")
)


def _dlpack_dtype_code(torch_dtype):
    """Encode a torch dtype as a DLPack int64 code."""
    import torch

    _map = {
        torch.float16: (2 << 16) | (16 << 8) | 1,
        torch.bfloat16: (4 << 16) | (16 << 8) | 1,
        torch.float32: (2 << 16) | (32 << 8) | 1,
        torch.float8_e4m3fn: (12 << 16) | (8 << 8) | 1,
        torch.float8_e5m2: (13 << 16) | (8 << 8) | 1,
    }
    return _map[torch_dtype]


def _variant_key_from_runtime(
    dtype_code, qo_tile_size, single_wg, sparse_mode, page_size, split_kv, pack_factor
):
    """Compute a variant key string from runtime parameters."""
    dims = _FMHA_SM100_DISPATCH

    def _match_idx(dim_values, runtime_val):
        # Convert Python bool to string to match dispatch table ("true"/"false")
        if isinstance(runtime_val, bool):
            runtime_val = "true" if runtime_val else "false"
        for idx, (match_val, _) in enumerate(dim_values):
            if match_val is None:
                return idx
            if match_val == runtime_val:
                return idx
        return len(dim_values) - 1

    runtime_vals = [
        dtype_code,
        qo_tile_size,
        single_wg,
        sparse_mode,
        page_size,
        split_kv,
        pack_factor,
    ]
    indices = []
    params = {}
    for (_, dim_values), rv in zip(dims, runtime_vals):
        idx = _match_idx(dim_values, rv)
        indices.append(idx)
        _, tparams = dim_values[idx]
        params.update(tparams)

    if _FMHA_SM100_IMPOSSIBLE(params):
        raise ValueError(f"Impossible FMHA variant combination: {params}")

    func_name = "fmha_sm100_" + "_".join(str(i) for i in indices)
    variant_name = "_".join(str(i) for i in indices)
    params["func_name"] = func_name
    params["variant_name"] = variant_name
    return variant_name, params


def _get_tvm_ffi_include():
    """Find TVM-FFI include directory."""
    try:
        import tvm_ffi

        tvm_dir = Path(tvm_ffi.__path__[0])
        inc = tvm_dir / "include"
        if inc.exists():
            return str(inc)
        inc2 = tvm_dir.parent / "include"
        if inc2.exists():
            return str(inc2)
    except ImportError:
        pass
    raise RuntimeError("Cannot find TVM-FFI include directory; install apache-tvm-ffi")


def _get_cuda_home():
    """Find CUDA toolkit root."""
    if "CUDA_HOME" in os.environ:
        return os.environ["CUDA_HOME"]
    nvcc = shutil.which("nvcc")
    if nvcc:
        return str(Path(nvcc).resolve().parent.parent)
    for p in ["/usr/local/cuda", "/opt/cuda"]:
        if os.path.isdir(p):
            return p
    raise RuntimeError("Cannot find CUDA toolkit. Set CUDA_HOME.")


_ALL_VARIANTS_SO = CACHE_BASE / "_all_variants" / "all_variants.so"


def _get_nvcc_flags(cache_dir, fmha=True):
    tvm_include = _get_tvm_ffi_include()
    fmha_include = str(_FMHA_VARLEN_DIR / "include")
    cutlass_dir = _find_cutlass_dir()
    cutlass_include = str(cutlass_dir / "include")
    cutlass_util_include = str(cutlass_dir / "tools" / "util" / "include")
    nvcc_flags = [
        "-O3",
        "-std=c++20",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "-gencode=arch=compute_100a,code=sm_100a",
        "-gencode=arch=compute_103a,code=sm_103a",
        "-static-global-template-stub=false",
        "-DFLASHINFER_ENABLE_BF16",
        "-DFLASHINFER_ENABLE_FP8_E4M3",
        "-DFLASHINFER_ENABLE_FP8_E5M2",
        "-DFLASHINFER_ENABLE_FP8_E8M0",
        "-DFLASHINFER_ENABLE_FP4_E2M1",
        "-DFLASHINFER_ENABLE_F16",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-Xcudafe --diag_suppress=2908",
        f"-I{fmha_include}",
        f"-I{cutlass_include}",
        f"-I{cutlass_util_include}",
        f"-I{tvm_include}",
        f"-I{cache_dir}",
        "-use_fast_math",
        "-DNDEBUG",
        "-Xptxas",
        "-O1" if fmha else "-O3",
        "-Xcompiler",
        "-fPIC",
    ]
    if os.environ.get("GPU_TRACE") is not None:
        nvcc_flags.append("-DGPU_TRACE_ENABLED")
    if os.environ.get("SM_TIMING") is not None:
        nvcc_flags.append("-DSM_TIMING_ENABLED")
    if os.environ.get("FMHA_GMEM_CHECK") is not None:
        nvcc_flags.append("-DFMHA_GMEM_BOUNDS_CHECK")
    return " ".join(nvcc_flags)


class _VariantWrapper:
    """Wraps a TVM-FFI function to look like a module with .run()."""

    def __init__(self, fn):
        self._fn = fn

    def run(self, *args):
        return self._fn(*args)


class FMHAVariantManager:
    """Manages FMHA variant kernels. Loads from all_variants.so or per-variant .so."""

    def __init__(self):
        self._loaded = {}
        self._lock = threading.Lock()
        self._all_module = None
        self._all_module_checked = False
        self._inst_template = None
        self._run_template = None

    def _load_templates(self):
        if self._inst_template is None:
            with open(_FMHA_VARLEN_DIR / "fmha_sm100_inst.jinja") as f:
                self._inst_template = jinja2.Template(f.read())
            with open(_FMHA_VARLEN_DIR / "fmha_sm100_variant_run.cu.jinja") as f:
                self._run_template = jinja2.Template(f.read())

    def _ensure_all_module(self):
        if not self._all_module_checked:
            self._all_module_checked = True
            if _ALL_VARIANTS_SO.exists():
                import tvm_ffi

                self._all_module = tvm_ffi.load_module(str(_ALL_VARIANTS_SO))

    def get_variant(
        self,
        dtype_code,
        qo_tile_size,
        single_wg,
        sparse_mode,
        page_size,
        split_kv,
        pack_factor,
    ):
        variant_name, params = _variant_key_from_runtime(
            dtype_code,
            qo_tile_size,
            single_wg,
            sparse_mode,
            page_size,
            split_kv,
            pack_factor,
        )

        cached = self._loaded.get(variant_name)
        if cached is not None:
            return cached

        with self._lock:
            cached = self._loaded.get(variant_name)
            if cached is not None:
                return cached

            fn_name = f"run_{variant_name}"

            # Try all_variants.so
            self._ensure_all_module()
            if self._all_module is not None:
                try:
                    fn = getattr(self._all_module, fn_name)
                    self._loaded[variant_name] = _VariantWrapper(fn)
                    return self._loaded[variant_name]
                except AttributeError:
                    pass

            # Try per-variant .so
            cache_dir = CACHE_BASE / variant_name
            so_path = cache_dir / f"{variant_name}.so"
            lock_fd = _acquire_file_lock(cache_dir / ".compile.lock")
            try:
                if not so_path.exists():
                    self._compile_only(variant_name, params)

                import tvm_ffi

                module = tvm_ffi.load_module(str(so_path))
                fn = getattr(module, fn_name)
            finally:
                _release_file_lock(lock_fd)
            self._loaded[variant_name] = _VariantWrapper(fn)
            return self._loaded[variant_name]

    def _compile_only(self, variant_name, params):
        """Compile a variant without loading. Safe to call from subprocesses."""
        cache_dir = CACHE_BASE / variant_name
        so_path = cache_dir / f"{variant_name}.so"

        if so_path.exists():
            return

        logger.info(f"JIT compiling FMHA variant: {variant_name}")
        cache_dir.mkdir(parents=True, exist_ok=True)

        self._load_templates()

        inst_cu = cache_dir / f"fmha_sm100_inst_{variant_name}.cu"
        run_cu = cache_dir / f"fmha_sm100_run_{variant_name}.cu"

        inst_cu.write_text(self._inst_template.render(**params))
        run_cu.write_text(self._run_template.render(**params))

        for name in ["fmha_sm100_params.h", "tvm_ffi_utils.h", "gmem_bounds_check.h"]:
            src = _FMHA_VARLEN_DIR / name
            dst = cache_dir / name
            if not dst.exists() or dst.read_text() != src.read_text():
                shutil.copy2(src, dst)

        self._write_ninja(cache_dir, variant_name, inst_cu, run_cu)
        self._run_ninja(cache_dir)

    def _write_ninja(self, cache_dir, variant_name, inst_cu, run_cu):
        cuda_home = _get_cuda_home()
        nvcc = os.path.join(cuda_home, "bin", "nvcc")

        nvcc_flags = _get_nvcc_flags(cache_dir)

        so_path = cache_dir / f"{variant_name}.so"
        inst_obj = cache_dir / f"inst_{variant_name}.o"
        run_obj = cache_dir / f"run_{variant_name}.o"

        ninja_content = f"""ninja_required_version = 1.5

nvcc = {nvcc}
nvcc_flags = {nvcc_flags}

rule nvcc_compile
  command = $nvcc $nvcc_flags -c $in -o $out
  description = Compiling $in

rule nvcc_link
  command = $nvcc -shared $in -o $out -lcuda
  description = Linking $out

build {inst_obj}: nvcc_compile {inst_cu}
build {run_obj}: nvcc_compile {run_cu}
build {so_path}: nvcc_link {inst_obj} {run_obj}
"""
        (cache_dir / "build.ninja").write_text(ninja_content)

    def _run_ninja(self, cache_dir):
        result = subprocess.run(
            ["ninja", "-j1"],
            cwd=str(cache_dir),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"FMHA variant compilation failed:\n"
                f"stdout: {result.stdout}\n"
                f"stderr: {result.stderr}"
            )


_variant_manager = FMHAVariantManager()


def get_fmha_variant(
    dtype_code, qo_tile_size, single_wg, sparse_mode, page_size, split_kv, pack_factor
):
    """Get a compiled FMHA variant module. Thread-safe, lazy compilation."""
    return _variant_manager.get_variant(
        dtype_code,
        qo_tile_size,
        single_wg,
        sparse_mode,
        page_size,
        split_kv,
        pack_factor,
    )


# ============================================================================
# Plan kernel JIT (also needs sm_100a, can't be statically compiled)
# ============================================================================

_plan_module = None
_plan_lock = threading.Lock()


def _do_compile_plan():
    """Generate and compile plan kernel. No TVM loading."""
    cache_dir = CACHE_BASE / "plan"
    so_path = cache_dir / "fmha_sm100_plan.so"

    if so_path.exists():
        return

    logger.info("JIT compiling FMHA plan module")
    cache_dir.mkdir(parents=True, exist_ok=True)

    plan_cu = _FMHA_VARLEN_DIR / "fmha_sm100_plan.cu"
    shutil.copy2(plan_cu, cache_dir / "fmha_sm100_plan.cu")

    cuda_home = _get_cuda_home()
    nvcc = os.path.join(cuda_home, "bin", "nvcc")

    obj = cache_dir / "fmha_sm100_plan.o"

    nvcc_flags = _get_nvcc_flags(cache_dir, False)

    ninja_content = f"""ninja_required_version = 1.5

nvcc = {nvcc}
nvcc_flags = {nvcc_flags}

rule nvcc_compile
  command = $nvcc $nvcc_flags -c $in -o $out
  description = Compiling $in

rule nvcc_link
  command = $nvcc -shared $in -o $out -lcuda
  description = Linking $out

build {obj}: nvcc_compile {cache_dir / "fmha_sm100_plan.cu"}
build {so_path}: nvcc_link {obj}
"""
    (cache_dir / "build.ninja").write_text(ninja_content)

    result = subprocess.run(
        ["ninja", "-j1"],
        cwd=str(cache_dir),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"FMHA plan compilation failed:\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )


def _compile_plan_only():
    """Compile plan kernel without loading. Safe for subprocesses."""
    _do_compile_plan()


def _compile_plan_module():
    """Compile and load plan kernel."""
    lock_fd = _acquire_file_lock(CACHE_BASE / "plan.lock")
    try:
        _do_compile_plan()
        import tvm_ffi

        so_path = CACHE_BASE / "plan" / "fmha_sm100_plan.so"
        return tvm_ffi.load_module(str(so_path))
    finally:
        _release_file_lock(lock_fd)


def get_plan_fn():
    """Get the plan function. JIT compiles on first call."""
    global _plan_module
    if _plan_module is not None:
        return _plan_module

    with _plan_lock:
        if _plan_module is not None:
            return _plan_module
        _plan_module = _compile_plan_module()
        return _plan_module


# ============================================================================
# Sparse TopK Select kernel JIT
# ============================================================================

_sparse_topk_module = None
_sparse_topk_lock = threading.Lock()


def _do_compile_sparse_topk():
    cache_dir = CACHE_BASE / "sparse_topk"
    so_path = cache_dir / "sparse_topk_select.so"

    tracked_sources = [
        _FMHA_VARLEN_DIR / "sparse_topk_select.cu",
        _FMHA_VARLEN_DIR / "include" / "sparse_topk_select.cuh",
        _FMHA_VARLEN_DIR / "tvm_ffi_utils.h",
    ]
    needs_rebuild = not so_path.exists()
    for src in tracked_sources:
        dst = cache_dir / src.name
        if not dst.exists() or dst.read_text() != src.read_text():
            needs_rebuild = True

    if not needs_rebuild:
        return

    logger.info("JIT compiling sparse_topk_select module")
    cache_dir.mkdir(parents=True, exist_ok=True)

    src_cu = _FMHA_VARLEN_DIR / "sparse_topk_select.cu"
    shutil.copy2(src_cu, cache_dir / "sparse_topk_select.cu")
    shutil.copy2(
        _FMHA_VARLEN_DIR / "include" / "sparse_topk_select.cuh",
        cache_dir / "sparse_topk_select.cuh",
    )
    shutil.copy2(_FMHA_VARLEN_DIR / "tvm_ffi_utils.h", cache_dir / "tvm_ffi_utils.h")

    cuda_home = _get_cuda_home()
    nvcc = os.path.join(cuda_home, "bin", "nvcc")

    obj = cache_dir / "sparse_topk_select.o"
    cached_cu = cache_dir / "sparse_topk_select.cu"
    cached_cuh = cache_dir / "sparse_topk_select.cuh"
    cached_ffi_header = cache_dir / "tvm_ffi_utils.h"

    nvcc_flags = _get_nvcc_flags(cache_dir, False)

    ninja_content = f"""ninja_required_version = 1.5

nvcc = {nvcc}
nvcc_flags = {nvcc_flags}

rule nvcc_compile
  command = $nvcc $nvcc_flags -c $in -o $out
  description = Compiling $in

rule nvcc_link
  command = $nvcc -shared $in -o $out -lcuda
  description = Linking $out

build {obj}: nvcc_compile {cached_cu} | {cached_cuh} {cached_ffi_header}
build {so_path}: nvcc_link {obj}
"""
    (cache_dir / "build.ninja").write_text(ninja_content)

    result = subprocess.run(
        ["ninja", "-j1"],
        cwd=str(cache_dir),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"sparse_topk_select compilation failed:\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )


def get_sparse_topk_module():
    """Get the sparse_topk_select module. JIT compiles on first call."""
    global _sparse_topk_module
    if _sparse_topk_module is not None:
        return _sparse_topk_module

    with _sparse_topk_lock:
        if _sparse_topk_module is not None:
            return _sparse_topk_module
        lock_fd = _acquire_file_lock(CACHE_BASE / "sparse_topk.lock")
        try:
            _do_compile_sparse_topk()
            import tvm_ffi

            so_path = CACHE_BASE / "sparse_topk" / "sparse_topk_select.so"
            _sparse_topk_module = tvm_ffi.load_module(str(so_path))
            _sparse_topk_module.sparse_topk_select_init()
        finally:
            _release_file_lock(lock_fd)
        return _sparse_topk_module


# ============================================================================
# Split-KV reduction kernel JIT
# ============================================================================

_reduction_module = None
_reduction_lock = threading.Lock()


def _do_compile_reduction():
    cache_dir = CACHE_BASE / "reduction"
    so_path = cache_dir / "fmha_sm100_reduction.so"

    if so_path.exists():
        return

    logger.info("JIT compiling fmha_sm100_reduction module")
    cache_dir.mkdir(parents=True, exist_ok=True)

    src_cu = _FMHA_VARLEN_DIR / "fmha_sm100_reduction.cu"
    shutil.copy2(src_cu, cache_dir / "fmha_sm100_reduction.cu")
    for name in ["gmem_bounds_check.h"]:
        src = _FMHA_VARLEN_DIR / name
        dst = cache_dir / name
        if src.exists() and (not dst.exists() or dst.read_text() != src.read_text()):
            shutil.copy2(src, dst)

    cuda_home = _get_cuda_home()
    nvcc = os.path.join(cuda_home, "bin", "nvcc")

    obj = cache_dir / "fmha_sm100_reduction.o"

    nvcc_flags = _get_nvcc_flags(cache_dir, False)

    ninja_content = f"""ninja_required_version = 1.5

nvcc = {nvcc}
nvcc_flags = {nvcc_flags}

rule nvcc_compile
  command = $nvcc $nvcc_flags -c $in -o $out
  description = Compiling $in

rule nvcc_link
  command = $nvcc -shared $in -o $out -lcuda
  description = Linking $out

build {obj}: nvcc_compile {cache_dir / "fmha_sm100_reduction.cu"}
build {so_path}: nvcc_link {obj}
"""
    (cache_dir / "build.ninja").write_text(ninja_content)

    result = subprocess.run(
        ["ninja", "-j1"],
        cwd=str(cache_dir),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"fmha_sm100_reduction compilation failed:\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )


def get_reduction_module():
    """Get the split-KV reduction module. JIT compiles on first call."""
    global _reduction_module
    if _reduction_module is not None:
        return _reduction_module

    with _reduction_lock:
        if _reduction_module is not None:
            return _reduction_module
        lock_fd = _acquire_file_lock(CACHE_BASE / "reduction.lock")
        try:
            _do_compile_reduction()
            import tvm_ffi

            so_path = CACHE_BASE / "reduction" / "fmha_sm100_reduction.so"
            _reduction_module = tvm_ffi.load_module(str(so_path))
        finally:
            _release_file_lock(lock_fd)
        return _reduction_module

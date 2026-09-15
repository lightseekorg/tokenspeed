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

"""Read compiled up-projection resources outside CUDA Graph capture.

Compiler geometry is not a measured occupancy or overlap claim. Driver
attributes describe a separately loaded copy of the exact compiled CUBIN;
the executable bound to the benchmark plan is never modified or launched.
"""

import hashlib
import re

_KERNEL_ATTRIBUTES = (
    "mma_tiler",
    "cta_tile_shape_mnk",
    "cluster_shape_mn",
    "epi_tile",
    "num_acc_stage",
    "num_ab_stage",
    "num_c_stage",
    "num_tmem_alloc_cols",
    "smem_capacity",
    "threads_per_cta",
    "occupancy",
    "use_2cta_instrs",
    "enable_pdl",
    "store_kind",
    "preadded_owner",
    "addends_before_gemm",
    "shared_leading_dim",
    "release_acc_early",
    "acquire_before_store",
    "prefetch_acc_tile",
    "requested_epilogue_tile",
    "addend_cache_policy",
    "paired_addend_loads",
    "round_acc_before_add",
    "addend_stages",
    "addend_smem_bytes",
    "fused_shared_rs",
    "fused_reduce_vectors",
    "experimental_fused_n64",
    "n64_a_smem_bytes",
    "n64_b_smem_bytes",
    "n64_c_smem_bytes",
    "n64_smem_budget_bytes",
)


def _json_value(value):
    # Exact built-in types only: DSL subclasses can override conversion,
    # iteration and even attribute access using an expired MLIR context.
    value_type = type(value)
    if value is None or value_type in (str, int, float, bool):
        return value
    if value_type in (tuple, list):
        return [_json_value(item) for item in value]
    # In particular, never str/repr, iterate or inspect an unknown DSL value.
    # CuTe layout objects can outlive the MLIR context used by compilation;
    # printing one can segfault natively rather than raise a Python exception.
    return {
        "status": "unavailable",
        "python_type": type.__getattribute__(value_type, "__name__"),
        "reason": "non-builtin compiler value not inspected after compilation",
    }


def _cuda_result(result):
    if int(result[0]) != 0:
        raise RuntimeError(f"CUDA driver resource query failed: {result}")
    return result[1] if len(result) == 2 else result[1:]


def _driver_resources(ptx, cubin):
    import cuda.bindings.driver as cuda

    symbols = re.findall(r"\.entry\s+([A-Za-z0-9_$]+)", ptx)
    if len(symbols) != 1:
        raise ValueError(f"expected one compiled GEMM CUDA function, got {symbols}")
    module = _cuda_result(cuda.cuModuleLoadData(cubin))
    try:
        function = _cuda_result(cuda.cuModuleGetFunction(module, symbols[0].encode()))
        result = {"status": "available", "symbol": symbols[0]}
        attributes = (
            ("registers_per_thread", "CU_FUNC_ATTRIBUTE_NUM_REGS"),
            ("local_bytes_per_thread", "CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES"),
            ("static_shared_bytes", "CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES"),
            ("max_threads_per_block", "CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK"),
            ("binary_version", "CU_FUNC_ATTRIBUTE_BINARY_VERSION"),
            ("ptx_version", "CU_FUNC_ATTRIBUTE_PTX_VERSION"),
        )
        for key, name in attributes:
            enum = getattr(cuda.CUfunction_attribute, name)
            result[key] = int(_cuda_result(cuda.cuFuncGetAttribute(enum, function)))
        result["occupancy_status"] = "not_queried"
        result["occupancy_note"] = (
            "Actual launch dynamic-shared bytes and cluster launch attributes "
            "are required; a zero-dynamic-shared CTA query is not the occupancy "
            "of this clustered GEMM. Use recorded launch resources instead."
        )
        return result
    finally:
        _cuda_result(cuda.cuModuleUnload(module))


def collect_plan_resources(plan):
    """Return JSON-safe static geometry and available compiled driver metadata.

    Args:
        plan: A prepared BoundSymmetricUpProjection with ``kernel``,
            ``compiled`` and ``max_active_clusters`` attributes. Call on the
            plan's CUDA device, outside graph capture and benchmark timing.

    Returns:
        A dictionary containing compiler attributes, available PTX/CUBIN
        hashes and driver resource attributes. Only exact Python primitives
        and nested built-in tuple/list values are read as static metadata;
        symbolic compiler values are never formatted or introspected.
        Missing artifacts or failed
        optional driver queries are explicit ``unavailable`` records, never
        fabricated zeros. This function does not launch kernels, change the
        plan, allocate tensor buffers or perform rank collectives.
    """
    kernel = getattr(plan, "kernel", None)
    result = {
        "static": {
            key: _json_value(getattr(kernel, key))
            for key in _KERNEL_ATTRIBUTES
            if kernel is not None and hasattr(kernel, key)
        },
        "max_active_clusters": _json_value(getattr(plan, "max_active_clusters", None)),
        "driver": {"status": "unavailable", "reason": "compiled artifacts absent"},
        "artifacts": {},
    }
    try:
        compiled = getattr(plan, "compiled", None)
        ptx = getattr(compiled, "__ptx__", None)
        cubin = getattr(compiled, "__cubin__", None)
        if type(ptx) is bytes:
            ptx = ptx.decode("utf-8")
        if ptx is not None and type(ptx) is not str:
            result["artifacts"]["ptx_unavailable"] = _json_value(ptx)
            ptx = None
        if cubin is not None and type(cubin) not in (bytes, bytearray):
            result["artifacts"]["cubin_unavailable"] = _json_value(cubin)
            cubin = None
        result["artifacts"]["ptx_available"] = bool(ptx)
        result["artifacts"]["cubin_available"] = bool(cubin)
        if ptx:
            result["artifacts"]["ptx_sha256"] = hashlib.sha256(ptx.encode()).hexdigest()
        if cubin:
            result["artifacts"]["cubin_sha256"] = hashlib.sha256(cubin).hexdigest()
        if ptx and cubin:
            result["driver"] = _driver_resources(ptx, cubin)
    except Exception as exc:
        result["driver"] = {
            "status": "unavailable",
            "reason": f"{type(exc).__name__}: {exc}",
        }
    return result

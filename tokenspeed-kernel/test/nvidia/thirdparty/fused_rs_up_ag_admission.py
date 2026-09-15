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

"""Pre-launch checks of the fixed two-CTA fused producer; no timed work."""

import re

CAPS = (38, 56, 64, 76)


def _entry(ptx):
    entries = re.findall(r"\.entry\s+([A-Za-z0-9_$]+)", ptx)
    if len(entries) != 1:
        raise ValueError("one actual compiled producer symbol is required")
    return entries[0]


def cap_smem_precheck(resources, ptx, cap):
    """Validate actual built-in compiler/driver fields and emitted buffer layout."""
    if type(cap) is not int or cap not in CAPS:
        raise ValueError("cluster cap must be the explicit integer 38,56,64 or76")
    static, driver = resources["static"], resources["driver"]
    expected = {
        "mma_tiler": [256, 128, 64],
        "cta_tile_shape_mnk": [128, 128, 64],
        "cluster_shape_mn": [2, 1],
        "requested_epilogue_tile": [128, 64],
        "threads_per_cta": 192,
        "num_ab_stage": 6,
        "num_c_stage": 3,
        "num_acc_stage": 2,
        "num_tmem_alloc_cols": 256,
        "addend_stages": 1,
        "addend_smem_bytes": 32768,
        "fused_shared_rs": True,
        "fused_reduce_vectors": 4,
        "experimental_fused_n64": False,
        "enable_pdl": False,
        "use_2cta_instrs": True,
        "store_kind": "tma",
    }
    if any(static.get(key) != value for key, value in expected.items()):
        raise ValueError(
            "actual compiled cap producer differs from fixed N128/group4/AB6/C3"
        )
    if resources["max_active_clusters"] != cap or driver.get("status") != "available":
        raise ValueError("actual cap specialization or CUBIN metadata is unavailable")
    if (
        driver["max_threads_per_block"] < 192
        or driver["static_shared_bytes"] != 0
        or driver["local_bytes_per_thread"] != 0
    ):
        raise ValueError(
            "cap producer requires 192 threads, dynamic-only SMEM and no local spills"
        )
    if (
        "griddepcontrol." in ptx
        or ptx.count("multimem.ld_reduce.relaxed.sys.global.add.acc::f32.v4.bf16x2") < 4
    ):
        raise ValueError("cap producer PTX must retain group4 reductions and PDL-off")
    # Read emitted TMA destination expressions and their actual stage shifts.
    # This avoids touching CuTe layout objects after their MLIR context expires.
    tma = list(
        re.finditer(
            r"cp\.async\.bulk\.tensor\.2d\.shared::cluster\.global[^;\n]*?\[\s*(%r\d+)\s*\+\s*(\d+)\s*\]",
            ptx,
        )
    )
    if len(tma) != 2:
        raise ValueError("fixed cap PTX requires two actual operand TMA instructions")
    bases, strides = [], []
    for instruction in tma:
        destination, offset = instruction[1], int(instruction[2])
        additions = list(
            re.finditer(
                r"add\.s32\s+"
                + re.escape(destination)
                + r"\s*,\s*(%r\d+)\s*,\s*(%r\d+)\s*;",
                ptx[: instruction.start()],
            )
        )
        if not additions:
            raise ValueError(
                "operand TMA address lacks the expected actual stage addition"
            )
        addition = additions[-1]
        shifts = list(
            re.finditer(
                r"shl\.b32\s+"
                + re.escape(addition[2])
                + r"\s*,\s*%r\d+\s*,\s*(\d+)\s*;",
                ptx[: addition.start()],
            )
        )
        if not shifts:
            raise ValueError("operand TMA address lacks an actual byte-stride shift")
        bases.append(offset)
        strides.append(1 << int(shifts[-1][1]))
    stores = re.findall(
        r"cp\.async\.bulk\.tensor\.2d\.global\.shared::cta[^;\n]*,\s*\[\s*%r\d+\s*\+\s*(\d+)\s*\]\s*,",
        ptx,
    )
    if bases != [256, 98560] or strides != [16384, 8192] or stores != ["147712"]:
        raise ValueError(
            "actual cap TMA offsets/strides differ from the audited N128 layout"
        )
    ab_bytes = static["num_ab_stage"] * sum(strides)
    c_bytes = static["num_c_stage"] * 128 * 64 * 2
    shared_plane, residual_plane = (
        bases[0] + ab_bytes + c_bytes,
        bases[0] + ab_bytes + c_bytes + 16384,
    )
    for offset in (shared_plane, residual_plane):
        if not re.search(
            r"add\.s32\s+%r\d+\s*,\s*%r\d+\s*,\s*" + str(offset) + r"\s*;", ptx
        ):
            raise ValueError("actual PTX is missing a required addend-plane base")
    payload = ab_bytes + c_bytes + static["addend_smem_bytes"]
    metadata = 16 * static["num_ab_stage"] + 16 * static["num_acc_stage"] + 12
    budget = payload + 1024
    if (
        metadata + 5 * 127 > 1024
        or type(static["smem_capacity"]) is not int
        or budget > static["smem_capacity"]
    ):
        raise ValueError(
            "fixed allocation reserve cannot cover metadata/alignment within device SMEM"
        )
    return {
        "passed": True,
        "ptx_entry": _entry(ptx),
        "ab_stage_bytes": strides,
        "operand_bases": bases,
        "c_base": 147712,
        "addend_bases": [shared_plane, residual_plane],
        "ab_bytes": ab_bytes,
        "c_bytes": c_bytes,
        "addend_bytes": 32768,
        "metadata_alignment_reserve_bytes": 1024,
        "dynamic_smem_upper_bound_bytes": budget,
        "actual_launch_dynamic_smem_observed": False,
        "registers_per_thread": driver["registers_per_thread"],
    }


def _query_cluster_admission(cubin, symbol, cap, budget):
    import cuda.bindings.driver as cuda
    from up_projection_resources import _cuda_result

    module = _cuda_result(cuda.cuModuleLoadData(cubin))
    try:
        function = _cuda_result(cuda.cuModuleGetFunction(module, symbol.encode()))
        _cuda_result(
            cuda.cuFuncSetAttribute(
                function,
                cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                budget,
            )
        )
        attribute = cuda.CUlaunchAttribute()
        attribute.id = cuda.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
        (
            attribute.value.clusterDim.x,
            attribute.value.clusterDim.y,
            attribute.value.clusterDim.z,
        ) = (2, 1, 1)
        config = cuda.CUlaunchConfig()
        config.gridDimX, config.gridDimY, config.gridDimZ = 2, 1, cap
        config.blockDimX, config.blockDimY, config.blockDimZ = 192, 1, 1
        config.sharedMemBytes = budget
        config.attrs, config.numAttrs = [attribute], 1
        active = int(_cuda_result(cuda.cuOccupancyMaxActiveClusters(function, config)))
        if active < cap:
            raise ValueError(
                "actual CUBIN cannot admit the requested complete two-CTA cluster population"
            )
        return {
            "active_clusters_at_smem_upper_bound": active,
            "query_grid": [2, 1, cap],
            "query_block": [192, 1, 1],
            "query_cluster": [2, 1, 1],
            "dynamic_smem_upper_bound_bytes": budget,
            "actual_launch_dynamic_smem_observed": False,
        }
    finally:
        _cuda_result(cuda.cuModuleUnload(module))

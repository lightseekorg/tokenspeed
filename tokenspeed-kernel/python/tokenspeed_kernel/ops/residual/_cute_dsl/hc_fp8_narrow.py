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

"""Register native single-launch HC mixing with prequantized FP8 operands."""

from __future__ import annotations

import threading

import cutlass
import torch
from cuda.bindings import driver
from cuda.bindings.driver import CUstream
from cutlass.cute import experimental as cute_ext
from tokenspeed_kernel.ops.residual.cute_dsl_fp8_fused import (
    Plan,
    Workspace,
    _checked,
    _view,
)
from tokenspeed_kernel.platform import pdl_enabled
from tokenspeed_kernel.thirdparty.cute_dsl._fp8_mixer.narrow import (
    Fp8FusedGatedResidualKernel,
)

_PLANS = {}
_WORKSPACES = {}
_LOCK = threading.RLock()
# Phase ordinals must never reuse scratch initialized with round generations.
_EPOCH_PROTOCOL_VERSION = 8


def _select_entry_down_prefill(g, h, r, token_bucket):
    """Choose actual first-ring issue during independent entry setup."""
    return (g, h, r) == (4, 2560, 320)


def _select_residual_issue(g, h, r, p, token_bucket):
    if (g, h, r, p, token_bucket) == (4, 2560, 320, 324, 16):
        return "register_before_mma"
    if (g, h, r) == (4, 2560, 320):
        if token_bucket == 128:
            return "down_entry"
        if token_bucket in (32, 64):
            return "handoff"
    return "up"


def _select_residual_transfer(g, h, r, token_bucket):
    return "tma" if (g, h, r, token_bucket) == (4, 2560, 320, 128) else "lane"


def _select_up_retirement(g, h, r, token_bucket, narrow_b300):
    if narrow_b300 and token_bucket in (32, 64):
        return "skip_final_and_tail_join"
    if (g, h, r) == (4, 2560, 320):
        return "skip_final"
    return "native"


def _select_up_hidden_columns(g, h, r, p, token_bucket):
    if (g, h, r, p) == (4, 2560, 320, 324):
        if token_bucket == 16:
            return 2
        if token_bucket in (32, 64):
            return 4
    return 1


def _select_up_gate_exchange(narrow_b300, token_bucket):
    return "row_pairs_scalar" if narrow_b300 and token_bucket == 64 else "native"


def _select_quant_copy_warp(narrow_b300, token_bucket):
    return 5 if narrow_b300 and token_bucket == 32 else -1


def _select_warp_owned_gates(narrow_b300, token_bucket):
    return narrow_b300 and token_bucket == 32


def _select_up_scale_issue(g, h, r, token_bucket):
    return "up_async" if (g, h, r, token_bucket) == (4, 2560, 320, 128) else "native"


def _select_quant_down_overlap(narrow_b300, token_bucket):
    return "early_scalar" if narrow_b300 and token_bucket == 16 else "native"


def _select_quant_down_poll(narrow_b300, token_bucket):
    return "acquire_fence" if narrow_b300 and token_bucket == 16 else "native"


def _select_aligned_partial_rows(narrow_b300, token_bucket):
    return narrow_b300 and token_bucket == 16


def _allocate_partials(kernel, rows, projection_rows, device):
    if kernel.aligned_partial_rows:
        # The 16-token aligned planes use this buffer as raw storage.
        shape = (rows, kernel.splits, projection_rows)
        return torch.empty(shape, device=device, dtype=torch.float32).transpose(0, 1)
    shape = (kernel.splits, rows, projection_rows)
    return torch.empty(shape, device=device, dtype=torch.float32)


def _select_quant_activation_store(narrow_b300, token_bucket):
    return "pack_max" if narrow_b300 and token_bucket in (16, 32, 64) else "native"


def _select_up_activation_prefix(narrow_b300, token_bucket):
    return (3, True) if narrow_b300 and token_bucket == 64 else (0, False)


def _select_activation_retry_ns(g, h, r, token_bucket):
    return 64 if (g, h, r, token_bucket) == (4, 2560, 320, 16) else 0


def _select_stage_handoff(g, h, r, p, token_bucket, async_phase_arrivals):
    if (g, h, r) == (4, 2560, 320):
        if token_bucket == 16:
            if p == r + g and async_phase_arrivals == "both":
                return "row_async_bulk", True
            return "grid", True
        if token_bucket == 32 and p == r + g and async_phase_arrivals == "down":
            return "row_tags_quant", False
        if token_bucket == 64:
            return "warp", False
    return "grid", False


def _select_activation_poll_group(narrow_b300, token_bucket):
    return 0


def _select_down_input_refill_hint(narrow_b300, token_bucket):
    return (3, 4) if narrow_b300 and token_bucket == 256 else ()


def _select_up_activation_transport(g, h, r, token_bucket):
    if (g, h, r) == (4, 2560, 320):
        if token_bucket == 16:
            return "packed_tma"
        if token_bucket == 64:
            return "bulk"
    return "native"


def _select_down_tactic(g, h, r, token_bucket):
    """Choose startup-static (projection rows, splits, K width, stages)."""
    if (g, h, r) == (4, 2560, 320) and token_bucket in (64, 128):
        return 64, 8, 640, 2
    if (g, h, r, token_bucket) == (4, 2560, 320, 256):
        return 64, 4, 512, 3
    if r == 1024:
        return (
            64,
            8,
            512 if token_bucket == 16 else 256,
            2 if token_bucket == 256 else 3,
        )
    if token_bucket <= 128:
        return 64, 16, g * h // 32, 2
    return 64, 16, 128, 4


def _select_up_resident_tactic(g, h, r, token_bucket):
    if (g, h, r, token_bucket) == (4, 2560, 320, 128):
        return 64, 5
    k = 128 if r == 1024 else 64 if token_bucket == 128 else 320
    return k, min(3, r // k)


def _select_down_token_tile(g, h, r, token_bucket):
    if (g, h, r) == (4, 2560, 320):
        return {64: 32, 128: 64, 256: 64}.get(token_bucket, token_bucket)
    return token_bucket


def _select_omit_up_empty(g, h, r, token_bucket):
    return (g, h, r) == (4, 2560, 320)


def _select_cta_reader_handoff(r, token_bucket):
    return r == 320 and token_bucket <= 128


def _select_partial_geometry(r, token_bucket):
    """Select fixed workspace geometry for (Down stores, ordered merge loads)."""
    if r == 320 and token_bucket == 16:
        return False, True
    if r == 320 and token_bucket == 32:
        return True, True
    return False, False


def _select_native_warp_max(device):
    # Float32 redux is available on the supported SM100 family; keep the
    # shuffle implementation for other architectures.
    major, _ = torch.cuda.get_device_capability(device)
    return major == 10


def _select_quant_max_join(g, h, r, token_bucket):
    if (g, h, r) == (4, 2560, 320):
        if token_bucket in (16, 32, 128):
            return "single_full"
        if token_bucket == 64:
            return "single_active"
    return "double"


def _select_quant_warp_handoff(g, h, r, token_bucket):
    return (g, h, r, token_bucket) == (4, 2560, 320, 64)


def _select_down_release(g, h, r, token_bucket):
    if (g, h, r, token_bucket) == (4, 2560, 320, 16):
        return "early_arrive"
    return "native"


def _select_down_writer_release(narrow_b300, token_bucket):
    return 1, narrow_b300 and token_bucket == 16, False


def _select_async_phase_arrivals(device, g, h, r, token_bucket):
    if (g, h, r) == (4, 2560, 320) and torch.cuda.get_device_capability(device)[
        0
    ] == 10:
        if token_bucket == 16:
            return "both"
        if token_bucket == 32:
            return "down"
    return "native"


def _select_omit_down_empty(g, h, r, token_bucket):
    return (g, h, r) == (4, 2560, 320) and token_bucket in (16, 32, 64)


def _select_up_register_rows(g, h, r, token_bucket):
    return (g, h, r, token_bucket) == (4, 2560, 320, 256)


def _select_entry_transport(g, h, r, token_bucket):
    """Choose entry copy transport and compiler minimum occupancy."""
    return ("deferred", 1) if (g, h, r) == (4, 2560, 320) else ("native", 0)


def _select_private_epoch_order(g, h, r, token_bucket):
    transport, _ = _select_entry_transport(g, h, r, token_bucket)
    return "coherent" if transport == "deferred" else "store" if r == 320 else "ordered"


def _select_up_cluster(g, h, r, token_bucket):
    if (g, h, r, token_bucket) == (4, 2560, 320, 32):
        return 1, "private_deferred"
    return 1, "native"


def _activation_workspace(rows, width, device, bundle_alpha):
    if bundle_alpha:
        codes_bytes = rows * width
        buffer = torch.empty(codes_bytes + rows * 4, device=device, dtype=torch.uint8)
        activation = buffer[:codes_bytes].view(torch.float8_e4m3fn).view(rows, width)
        alpha = buffer[codes_bytes:].view(torch.float32)
        return activation, alpha
    return (
        torch.empty((rows, width), device=device, dtype=torch.float8_e4m3fn),
        torch.empty((rows,), device=device, dtype=torch.float32),
    )


def _allocate_activation(kernel, device):
    if kernel.stamped_activation:
        records = torch.zeros(
            (kernel.n, (kernel.activation_payload_records // kernel.n + 1) * 16),
            dtype=torch.uint8,
            device=device,
        ).view(torch.float8_e4m3fn)
        alpha = torch.empty((kernel.n,), dtype=torch.float32, device=device)
        return records, alpha
    return _activation_workspace(
        kernel.n,
        kernel.r,
        device,
        kernel.bundle_alpha or kernel.up_cluster == "private_deferred",
    )


def _query_cluster_resources(
    compiled, operands, function, threads, cluster_size, device
):
    """Capture without replay and query the actual cooperative cluster launch."""
    with torch.cuda.device(device):
        probe_stream = _checked(
            driver.cuStreamCreate(driver.CUstream_flags.CU_STREAM_NON_BLOCKING)
        )
        graph = None
        try:
            _checked(
                driver.cuStreamBeginCapture(
                    probe_stream,
                    driver.CUstreamCaptureMode.CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
                )
            )
            try:
                compiled(*operands, cutlass.Int32(cluster_size), probe_stream)
            finally:
                graph = _checked(driver.cuStreamEndCapture(probe_stream))
            _, count = _checked(driver.cuGraphGetNodes(graph, 0))
            nodes, _ = _checked(driver.cuGraphGetNodes(graph, count))
            kernels = [
                node
                for node in nodes
                if _checked(driver.cuGraphNodeGetType(node))
                == driver.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL
            ]
            if len(kernels) != 1:
                raise RuntimeError("Expected one cooperative cluster compute node")
            params = _checked(driver.cuGraphKernelNodeGetParams(kernels[0]))
            if int(params.func) != int(function) or params.blockDimX != threads:
                raise RuntimeError(
                    "Captured cluster function or threads differ from the compiled geometry"
                )
            cluster_value = _checked(
                driver.cuGraphKernelNodeGetAttribute(
                    kernels[0],
                    driver.CUkernelNodeAttrID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION,
                )
            )
            cooperative_value = _checked(
                driver.cuGraphKernelNodeGetAttribute(
                    kernels[0],
                    driver.CUkernelNodeAttrID.CU_LAUNCH_ATTRIBUTE_COOPERATIVE,
                )
            )
            if (
                cluster_value.clusterDim.x,
                cluster_value.clusterDim.y,
                cluster_value.clusterDim.z,
            ) != (cluster_size, 1, 1) or cooperative_value.cooperative != 1:
                raise RuntimeError(
                    "Captured cooperative cluster attributes differ from the plan"
                )
            cluster = driver.CUlaunchAttribute()
            cluster.id = (
                driver.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
            )
            (
                cluster.value.clusterDim.x,
                cluster.value.clusterDim.y,
                cluster.value.clusterDim.z,
            ) = (cluster_size, 1, 1)
            cooperative = driver.CUlaunchAttribute()
            cooperative.id = driver.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_COOPERATIVE
            cooperative.value.cooperative = 1
            config = driver.CUlaunchConfig()
            config.gridDimX, config.gridDimY, config.gridDimZ = (
                params.gridDimX,
                params.gridDimY,
                params.gridDimZ,
            )
            config.blockDimX, config.blockDimY, config.blockDimZ = (
                params.blockDimX,
                params.blockDimY,
                params.blockDimZ,
            )
            config.sharedMemBytes = params.sharedMemBytes
            config.hStream = probe_stream
            config.attrs = [cluster, cooperative]
            config.numAttrs = 2
            clusters = _checked(driver.cuOccupancyMaxActiveClusters(function, config))
            blocks = _checked(
                driver.cuOccupancyMaxActiveBlocksPerMultiprocessor(
                    function, threads, params.sharedMemBytes
                )
            )
            return params.sharedMemBytes, blocks, clusters
        finally:
            if graph is not None:
                _checked(driver.cuGraphDestroy(graph))
            _checked(driver.cuStreamDestroy(probe_stream))


def _execute(
    x,
    xs,
    down,
    ds,
    up,
    us,
    *,
    g,
    h,
    r,
    p,
    output_dtype,
    projection_scale,
    weights_independent,
    token_bucket,
):
    rows = x.shape[0]
    out = torch.empty((rows, h), device=x.device, dtype=output_dtype)
    inject = (
        torch.empty((rows, g), device=x.device, dtype=output_dtype) if p > r else None
    )
    if rows == 0:
        return out, inject
    stream = CUstream(int(torch.cuda.current_stream(x.device).cuda_stream))
    down_tactic = _select_down_tactic(g, h, r, token_bucket)
    if (g, h, r, p, token_bucket) == (
        4,
        2560,
        320,
        324,
        16,
    ) and torch.cuda.get_device_capability(x.device) == (10, 3):
        # Fit 120 shorter Down tasks and 16 dedicated quantization CTAs in one resident grid.
        down_tactic = 64, 20, 256, 2
    down_token_tile = _select_down_token_tile(g, h, r, token_bucket)
    up_tactic = _select_up_resident_tactic(g, h, r, token_bucket)
    use_cta_reader_handoff = _select_cta_reader_handoff(r, token_bucket)
    partial_geometry = _select_partial_geometry(r, token_bucket)
    if (g, h, r, p, token_bucket) == (4, 2560, 320, 324, 64):
        partial_geometry = True, True
    quant_max_join = _select_quant_max_join(g, h, r, token_bucket)
    quant_warp_handoff = _select_quant_warp_handoff(g, h, r, token_bucket)
    down_release = _select_down_release(g, h, r, token_bucket)
    async_phase_arrivals = _select_async_phase_arrivals(x.device, g, h, r, token_bucket)
    omit_down_empty = _select_omit_down_empty(g, h, r, token_bucket)
    up_register_rows = _select_up_register_rows(g, h, r, token_bucket)
    omit_up_empty = _select_omit_up_empty(g, h, r, token_bucket)
    use_native_warp_max = _select_native_warp_max(x.device)
    private_epoch_order = _select_private_epoch_order(g, h, r, token_bucket)
    entry_transport, minimum_blocks_per_sm = _select_entry_transport(
        g, h, r, token_bucket
    )
    entry_down_prefill = _select_entry_down_prefill(g, h, r, token_bucket)
    entry_down_input_ring_first = (g, h, r) == (4, 2560, 320) and token_bucket <= 64
    evict_down_weights_last = (g, h, r) == (4, 2560, 320)
    residual_issue = _select_residual_issue(g, h, r, p, token_bucket)
    residual_transfer = _select_residual_transfer(g, h, r, token_bucket)
    if residual_transfer == "tma" and rows >= 1 << 31:
        raise ValueError("Residual TMA coordinates must fit signed int32")
    up_hidden_columns = _select_up_hidden_columns(g, h, r, p, token_bucket)
    up_scale_issue = _select_up_scale_issue(g, h, r, token_bucket)
    activation_handoff, dedicated_quant_ctas = _select_stage_handoff(
        g, h, r, p, token_bucket, async_phase_arrivals
    )
    narrow_b300 = (g, h, r, p) == (
        4,
        2560,
        320,
        324,
    ) and torch.cuda.get_device_capability(x.device) == (10, 3)
    down_writer_release = _select_down_writer_release(narrow_b300, token_bucket)
    up_gate_exchange = _select_up_gate_exchange(narrow_b300, token_bucket)
    warp_owned_gates = _select_warp_owned_gates(narrow_b300, token_bucket)
    quant_copy_warp = _select_quant_copy_warp(narrow_b300, token_bucket)
    up_retirement = _select_up_retirement(g, h, r, token_bucket, narrow_b300)
    quant_down_overlap = _select_quant_down_overlap(narrow_b300, token_bucket)
    quant_down_poll = _select_quant_down_poll(narrow_b300, token_bucket)
    quant_activation_store = _select_quant_activation_store(narrow_b300, token_bucket)
    up_activation_prefix = _select_up_activation_prefix(narrow_b300, token_bucket)
    stamped_activation = narrow_b300 and token_bucket == 16
    aligned_partial_rows = _select_aligned_partial_rows(narrow_b300, token_bucket)
    pad_residual_rows = narrow_b300 and token_bucket == 64
    paired_up_sigmoid = narrow_b300 and token_bucket == 32
    activation_poll_group = _select_activation_poll_group(narrow_b300, token_bucket)
    input_refill_hint = _select_down_input_refill_hint(narrow_b300, token_bucket)
    split_activation_counter = narrow_b300 and token_bucket == 64
    prefetch_up_weights_on_down_retire = narrow_b300 and token_bucket == 64
    up_activation_transport = _select_up_activation_transport(g, h, r, token_bucket)
    activation_retry_ns = _select_activation_retry_ns(g, h, r, token_bucket)
    cluster_size, up_cluster = _select_up_cluster(g, h, r, token_bucket)
    if activation_handoff == "row_tags_quant":
        up_cluster = "bundle_deferred"
    if up_cluster != "native" or activation_handoff == "row_async_bulk":
        up_activation_transport = "bulk"
    key = (
        x.device.index,
        g,
        h,
        r,
        p,
        output_dtype,
        projection_scale,
        weights_independent,
        ds.numel() > 1,
        us.numel() > 1,
        pdl_enabled(),
        down_tactic,
        down_token_tile,
        up_tactic,
        use_cta_reader_handoff,
        partial_geometry,
        quant_max_join,
        quant_warp_handoff,
        down_release,
        down_writer_release,
        async_phase_arrivals,
        omit_down_empty,
        up_register_rows,
        omit_up_empty,
        use_native_warp_max,
        private_epoch_order,
        entry_transport,
        minimum_blocks_per_sm,
        entry_down_prefill,
        entry_down_input_ring_first,
        evict_down_weights_last,
        residual_issue,
        residual_transfer,
        pad_residual_rows,
        up_hidden_columns,
        up_gate_exchange,
        warp_owned_gates,
        quant_copy_warp,
        paired_up_sigmoid,
        up_scale_issue,
        prefetch_up_weights_on_down_retire,
        up_retirement,
        activation_handoff,
        split_activation_counter,
        dedicated_quant_ctas,
        quant_down_overlap,
        quant_down_poll,
        quant_activation_store,
        up_activation_prefix,
        stamped_activation,
        aligned_partial_rows,
        up_activation_transport,
        activation_retry_ns,
        activation_poll_group,
        input_refill_hint,
        cluster_size,
        up_cluster,
        _EPOCH_PROTOCOL_VERSION,
        token_bucket,
    )
    stream_key = key + (int(stream),)
    with _LOCK:
        plan = _PLANS.get(key)
        workspace = _WORKSPACES.get(stream_key)
        if plan is None or workspace is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "warm the prequantized FP8 bucket on this stream before capture"
                )
            kernel = Fp8FusedGatedResidualKernel(
                g=g,
                h=h,
                r=r,
                p=p,
                token_bucket=token_bucket,
                projection_scale=projection_scale,
                use_pdl=pdl_enabled(),
                weights_independent=weights_independent,
                down_per_channel=ds.numel() > 1,
                up_per_channel=us.numel() > 1,
                down_tactic=down_tactic,
                down_token_tile=down_token_tile,
                up_tactic=up_tactic,
                use_cta_reader_handoff=use_cta_reader_handoff,
                static_partial_store=partial_geometry[0],
                static_partial_stride=partial_geometry[1],
                quant_max_join=quant_max_join,
                quant_warp_handoff=quant_warp_handoff,
                down_release=down_release,
                down_writer_release=down_writer_release,
                async_phase_arrivals=async_phase_arrivals,
                omit_down_empty=omit_down_empty,
                up_register_rows=up_register_rows,
                omit_up_empty=omit_up_empty,
                use_native_warp_max=use_native_warp_max,
                private_epoch_order=private_epoch_order,
                entry_transport=entry_transport,
                minimum_blocks_per_sm=minimum_blocks_per_sm,
                entry_down_prefill=entry_down_prefill,
                entry_down_input_ring_first=entry_down_input_ring_first,
                evict_down_weights_last=evict_down_weights_last,
                residual_issue=residual_issue,
                residual_transfer=residual_transfer,
                pad_residual_rows=pad_residual_rows,
                up_hidden_columns=up_hidden_columns,
                up_gate_exchange=up_gate_exchange,
                warp_owned_gates=warp_owned_gates,
                quant_copy_warp=quant_copy_warp,
                paired_up_sigmoid=paired_up_sigmoid,
                up_scale_issue=up_scale_issue,
                prefetch_up_weights_on_down_retire=prefetch_up_weights_on_down_retire,
                up_retirement=up_retirement,
                activation_handoff=activation_handoff,
                split_activation_counter=split_activation_counter,
                dedicated_quant_ctas=dedicated_quant_ctas,
                quant_down_overlap=quant_down_overlap,
                quant_down_poll=quant_down_poll,
                quant_activation_store=quant_activation_store,
                up_activation_prefix=up_activation_prefix,
                stamped_activation=stamped_activation,
                aligned_partial_rows=aligned_partial_rows,
                up_activation_transport=up_activation_transport,
                activation_retry_ns=activation_retry_ns,
                activation_poll_group=activation_poll_group,
                input_refill_hint=input_refill_hint,
                cluster_size=cluster_size,
                up_cluster=up_cluster,
            )
            if plan is None:
                # Compile with one safe cooperative block, then query real capacity.
                scratch = Workspace(
                    _allocate_partials(kernel, token_bucket, p, x.device),
                    *_allocate_activation(kernel, x.device),
                    torch.zeros(
                        (cluster_size + 1 + kernel.extra_epoch_slots,),
                        device=x.device,
                        dtype=torch.int64,
                    ),
                )
                values = (
                    x,
                    xs.reshape(-1),
                    down,
                    ds.reshape(-1),
                    up,
                    us.reshape(-1),
                    scratch.partials,
                    scratch.activation,
                    scratch.alpha,
                    scratch.epochs,
                    out,
                    out if inject is None else inject,
                )
                operands = tuple(_view(v) for v in values)
                compiled = cute_ext.compile(
                    kernel, *operands, cutlass.Int32(cluster_size), stream
                )
                compiled.to(device=x.device.index)
                library = driver.CUlibrary(int(compiled.jit_module.cuda_library[0]))
                handle = _checked(driver.cuLibraryEnumerateKernels(1, library))[0]
                function = _checked(driver.cuKernelGetFunction(handle))
                query_shared = kernel.shared_bytes
                active_clusters = None
                if cluster_size > 1:
                    query_shared, blocks, active_clusters = _query_cluster_resources(
                        compiled,
                        operands,
                        function,
                        kernel.threads,
                        cluster_size,
                        x.device,
                    )
                else:
                    blocks = _checked(
                        driver.cuOccupancyMaxActiveBlocksPerMultiprocessor(
                            function,
                            kernel.threads,
                            query_shared,
                        )
                    )
                properties = torch.cuda.get_device_properties(x.device)
                capacity = blocks * properties.multi_processor_count
                if active_clusters is not None:
                    capacity = min(capacity, active_clusters * cluster_size)
                if capacity < cluster_size:
                    raise RuntimeError(
                        "prequantized FP8 cooperative kernel has no resident capacity"
                    )
                desired_grid = max(kernel.down_tasks, kernel.up_tasks)
                if dedicated_quant_ctas:
                    desired_grid += kernel.quant_groups
                desired_clusters = (desired_grid + cluster_size - 1) // cluster_size
                grid = min(capacity // cluster_size, desired_clusters) * cluster_size
                resources = {
                    "registers": _checked(
                        driver.cuFuncGetAttribute(
                            driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NUM_REGS,
                            function,
                        )
                    ),
                    "local_bytes": _checked(
                        driver.cuFuncGetAttribute(
                            driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
                            function,
                        )
                    ),
                    "query_shared_bytes": query_shared,
                    "cluster_size": cluster_size,
                    "active_clusters": active_clusters,
                    "resident_capacity": capacity,
                    "blocks_per_sm": blocks,
                    "sm_count": properties.multi_processor_count,
                    "grid": grid,
                    "threads": kernel.threads,
                }
                plan = Plan(compiled, grid, kernel, resources)
                _PLANS[key] = plan
            with torch.inference_mode(False):
                workspace = Workspace(
                    _allocate_partials(plan.kernel, token_bucket, p, x.device),
                    *_allocate_activation(plan.kernel, x.device),
                    torch.zeros(
                        (plan.grid + 1 + plan.kernel.extra_epoch_slots,),
                        device=x.device,
                        dtype=torch.int64,
                    ),
                )
            _WORKSPACES[stream_key] = workspace
    values = (
        x,
        xs.reshape(-1),
        down,
        ds.reshape(-1),
        up,
        us.reshape(-1),
        workspace.partials,
        workspace.activation,
        workspace.alpha,
        workspace.epochs,
        out,
        out if inject is None else inject,
    )
    operands = tuple(_view(v) for v in values)
    plan.compiled(*operands, cutlass.Int32(plan.grid), stream)
    return out, inject

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

"""Prepared-plan adapter for the packaged CuteDSL KDA prefill kernel.

The ops wrapper owns architecture admission and imports the package's public
entry points directly. This adapter only bridges a device-built chunk plan to
the native launch ABI; it does not load kernels or probe package installation.
Native imports stay inside the functions so importing the adapter is safe on
platforms that do not use CuteDSL KDA.
"""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def cutedsl_kda_supports_prepared_plan() -> bool:
    """Whether the pinned native wrapper exposes the token-major launch ABI.

    Keep this compatibility adapter at the optional dependency boundary.
    Unknown payload layouts retain the public wrapper's planning path.
    """
    from tokenspeed_cutedsl_kda import kda_host, kda_wrapper

    metadata = getattr(kda_wrapper, "_meta", {})
    return (
        metadata.get("layout") == "token_major"
        and metadata.get("dtype") == "BFloat16"
        and metadata.get("gate_dtype") == "Float32"
        and metadata.get("state_dtype") == "Float32"
        and metadata.get("safe_gate") is True
        and metadata.get("has_state_in") is True
        and metadata.get("has_state_out") is True
        and metadata.get("mode") in ("auto", "engine", "decomp")
        and tuple(getattr(kda_host, name, None) for name in ("BT", "DK", "DV"))
        == (16, 128, 128)
        and callable(getattr(kda_wrapper, "_kda_fwd", None))
        and all(
            callable(getattr(kda_host, name, None))
            for name in (
                "_route_for_workspace",
                "_device_sm_count",
                "_engine_dummies",
                "_decomp_ws_bytes",
                "_partition_workspace",
                "_k1_grid",
            )
        )
    )


def cutedsl_kda_forward_with_prepared_plan(
    q,
    k,
    v,
    gate,
    a_log,
    dt_bias,
    beta,
    boundaries,
    state,
    *,
    token_capacity: int,
    cu_chunks,
    chunk_to_seq,
    scale: float,
):
    """Launch the same native scan with an explicit device-built chunk plan.

    The caller provides Q/K/V/beta in the native token-major layout, FP32 gate
    and live boundaries. cu_chunks is an int32
    prefix sum per sequence; chunk_to_seq has ceil(token_capacity / 16) + N - 1
    entries. Extra chunks map to the last sequence and are rejected by native
    live-bound checks. The caller initializes these buffers on the consumer
    stream. This entry still asks the native host to choose a route, obtains
    or allocates workspace, and allocates output and final state. It is not
    a fully preallocated entry, and a prepared plan does not imply a captured
    CUDA graph. Scratch and outputs belong to this invocation (or its graph
    pool). Returns output and final state. No process-global planner is replaced.
    """
    import torch
    from tokenspeed_cutedsl_kda import kda_host as host
    from tokenspeed_cutedsl_kda import kda_wrapper as native

    if not cutedsl_kda_supports_prepared_plan():
        raise RuntimeError("CuteDSL KDA payload does not support prepared plans")
    sequences, heads = boundaries.numel() - 1, q.shape[2]
    max_chunks = (token_capacity + host.BT - 1) // host.BT
    total_chunks = max_chunks + sequences - 1
    if cu_chunks.numel() != sequences + 1 or chunk_to_seq.numel() != total_chunks:
        raise ValueError("KDA prepared plan extent differs from packed capacity")
    # Route selection still belongs to the native host. Its decision depends
    # on sequence/head occupancy, not these synthetic per-sequence lengths.
    bounds = tuple(i * token_capacity for i in range(sequences + 1))
    route = host._route_for_workspace(bounds, heads, q.device, native._MODE)
    sm_count = host._device_sm_count(q.device)
    if route == "engine":
        kd, qd, wt, qk, diag, cuc, cts = host._engine_dummies(heads, q.device, q.dtype)
        num_ctas, cpc = 1, 1
    else:
        workspace = torch.empty(
            host._decomp_ws_bytes(heads, total_chunks),
            dtype=torch.uint8,
            device=q.device,
        )
        regions = host._partition_workspace(workspace, heads, total_chunks)
        kd, qd, wt, qk, diag = (
            regions[name] for name in ("kd", "qd", "w", "qk", "diag")
        )
        cuc, cts = cu_chunks, chunk_to_seq
        num_ctas, cpc = host._k1_grid(total_chunks, heads)
    out, new_state = torch.empty_like(v), torch.empty_like(state)
    native._kda_fwd(
        q,
        k,
        v,
        gate,
        a_log,
        dt_bias,
        beta,
        boundaries,
        cuc,
        cts,
        kd,
        qd,
        wt,
        qk,
        diag,
        state,
        out,
        new_state,
        native.cd.CUstream(torch.cuda.current_stream(q.device).cuda_stream),
        1,
        sm_count,
        max_chunks,
        num_ctas,
        cpc,
        scale,
        None,
        None,
    )
    return out, new_state

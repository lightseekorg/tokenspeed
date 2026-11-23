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

import torch
from tokenspeed_kernel._triton import tl, triton


@triton.jit
def _to_fp8(
    P,
    S,
    G,
    C,
    Q,
    QS,
    IDS,
    W,
    H: tl.constexpr,
    capacity,
    E: tl.constexpr,
    COMPACT: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    block = tl.program_id(1)
    dest = row
    if COMPACT:
        experts = tl.arange(0, triton.next_power_of_2(E))
        counts = tl.load(C + experts, experts < E, 0)
        ends = tl.cumsum(counts)
        expert = tl.sum(((row >= ends) & (experts < E)).to(tl.int32), 0)
        start = tl.sum(tl.where(experts < expert, counts, 0), 0)
        valid = row < tl.sum(counts, 0)
        row = expert.to(tl.int64) * capacity + row - start
    else:
        expert = row // capacity
        valid = row % capacity < tl.load(C + expert)
    tile = tl.arange(0, 8)
    lane = tl.arange(0, 128)
    col = block * 1024 + tile[:, None] * 128 + lane[None, :]
    values = tl.full((8, 128), 0, tl.float32)
    if valid:
        byte = tl.load(P + row * (H // 2) + col // 2, col < H, 0).to(tl.int32)
        code = (byte >> ((col & 1) * 4)) & 15
        mag = code & 7
        value = tl.where(
            mag < 2,
            mag * 0.5,
            tl.where(
                mag < 4,
                1.0 + (mag - 2) * 0.5,
                tl.where(mag < 6, 2.0 + (mag - 4), tl.where(mag == 6, 4.0, 6.0)),
            ),
        )
        value = tl.where((code & 8) != 0, -value, value)
        sf = (
            tl.load(S + row * (H // 16) + col // 16, col < H, 0)
            .to(tl.float8e4nv, bitcast=True)
            .to(tl.float32)
        )
        g = tl.load(G + row)
        values = (value * sf) * g
    amax = tl.maximum(tl.max(tl.abs(values), 1), 1.0e-4)
    scale = amax / 448.0
    bits = scale.to(tl.int32, bitcast=True)
    exp = ((bits >> 23) & 255) + tl.where((bits & 0x7FFFFF) != 0, 1, 0)
    exp = tl.minimum(tl.maximum(exp, 1), 254)
    power = (exp << 23).to(tl.float32, bitcast=True)
    q = (values / power[:, None]).to(Q.dtype.element_ty)
    tl.store(Q + dest * H + col, q, col < H)
    shift = (tile & 3) * 8
    s0 = tl.sum(tl.where(tile < 4, exp << shift, 0), 0)
    s1 = tl.sum(tl.where(tile >= 4, exp << shift, 0), 0)
    tl.store(QS + dest * (H // 512) + block * 2, s0, block * 2 < H // 512)
    tl.store(QS + dest * (H // 512) + block * 2 + 1, s1, block * 2 + 1 < H // 512)
    if block == 0:
        tl.store(IDS + dest, tl.where(valid, expert, -1))
        tl.store(W + dest, tl.where(valid, 1.0, 0.0))


def to_fp8(payload, scales, global_scale, counts, *, max_compact_rows=None):
    e, cap, half = payload.shape
    h = half * 2
    rows = e * cap
    if max_compact_rows is not None:
        rows = min(rows, max_compact_rows)
    q = torch.empty((rows, h), device=payload.device, dtype=torch.float8_e4m3fn)
    sf = torch.empty((rows, h // 512), device=payload.device, dtype=torch.int32)
    ids = torch.empty((rows, 1), device=payload.device, dtype=torch.int32)
    weights = torch.empty((rows, 1), device=payload.device, dtype=torch.float32)
    _to_fp8[(rows, triton.cdiv(h, 1024))](
        payload,
        scales,
        global_scale,
        counts,
        q,
        sf,
        ids,
        weights,
        h,
        cap,
        e,
        max_compact_rows is not None,
        num_warps=4,
    )
    return q, sf, ids, weights


@triton.jit
def _restore_expert_rows(
    X, C, Y, H: tl.constexpr, E: tl.constexpr, capacity, B: tl.constexpr
):
    row = tl.program_id(0).to(tl.int64)
    experts = tl.arange(0, triton.next_power_of_2(E))
    counts = tl.load(C + experts, experts < E, 0)
    ends = tl.cumsum(counts)
    valid = row < tl.sum(counts, 0)
    if valid:
        expert = tl.sum(((row >= ends) & (experts < E)).to(tl.int32), 0)
        start = tl.sum(tl.where(experts < expert, counts, 0), 0)
        dest = expert.to(tl.int64) * capacity + row - start
        col = tl.program_id(1) * B + tl.arange(0, B)
        value = tl.load(X + row * H + col, col < H, 0)
        tl.store(Y + dest * H + col, value, col < H)


def restore_expert_rows(values, counts, capacity, *, zero_padding):
    """Scatter compact rows; unused slots are defined only with zero_padding.

    Low-latency combine reads only rows described by its dispatch handle.
    Other consumers must explicitly request deterministic zero padding.
    """
    experts = counts.numel()
    hidden = values.shape[1]
    allocate = torch.zeros if zero_padding else torch.empty
    output = allocate(
        (experts, capacity, hidden), device=values.device, dtype=values.dtype
    )
    _restore_expert_rows[(values.shape[0], triton.cdiv(hidden, 1024))](
        values, counts, output, hidden, experts, capacity, 1024, num_warps=4
    )
    return output


class LowLatencyWire1Buffer:
    """Serialized legacy DeepEP with NVFP4 dispatch and FP8 combine.

    Expert GEMMs receive unit routing weights; combine applies the source
    weights exactly once. Logical and layout capacities are shared across
    ranks, including idle ranks. The logical bound limits compact expert work;
    the layout capacity sets RDMA offsets and the restored output stride.
    """

    def __init__(self, group, hidden, experts, capacity, shared_owner=None):
        import os

        from deep_ep import Buffer

        if Buffer.wire_format_version < 1:
            raise RuntimeError("NVFP4/FP8 requires the DeepEP wire-format-v1 extension")
        if hidden % 512 or experts % group.size() or capacity < 1:
            raise ValueError("Invalid hidden size, expert partition or capacity")
        from tokenspeed_kernel.thirdparty.deepep_sm120_experiment.buffer import (
            configure_device_nic,
        )

        configure_device_nic("nvfp4_fp8")
        import logging

        logging.getLogger(__name__).warning(
            "Wire1 layout-clean-v1 device=%s environment=%s",
            torch.cuda.current_device(),
            {k: v for k, v in os.environ.items() if k.startswith("NVSHMEM")},
        )
        self.compact = os.environ.get("TOKENSPEED_SM120_DEEPEP_COMPACT", "0") == "1"
        logging.getLogger(__name__).warning(
            "Wire1 compact expert rows: %s", self.compact
        )
        self.expert_impl = os.environ.get(
            "TOKENSPEED_SM120_DEEPEP_EXPERT_IMPL", "routed"
        )
        if self.expert_impl not in ("routed", "web"):
            raise ValueError(
                f"Unsupported wire1 expert implementation: {self.expert_impl}"
            )
        self.web_workspaces = {}
        self.web_prepared = {}
        self.web_states = {}
        self.cached_clean = (
            os.environ.get("TOKENSPEED_SM120_DEEPEP_CACHED_CLEAN", "0") == "1"
        )
        if self.cached_clean:
            Buffer.clean_low_latency_buffer_cached  # Require the explicit dependency patch.
        self.layout_state = (
            shared_owner.layout_state
            if shared_owner is not None
            else torch.zeros(2, dtype=torch.int64, device=torch.cuda.current_device())
        )
        if shared_owner is not None and shared_owner.cached_clean != self.cached_clean:
            raise ValueError("Shared transport must use one cleanup mode")
        logging.getLogger(__name__).warning(
            "Wire1 expert_impl=%s cached_clean=%s", self.expert_impl, self.cached_clean
        )
        self.hidden = hidden
        self.experts = experts
        self.capacity = capacity
        self.world_size = group.size()
        self.rdma_bytes = Buffer.get_low_latency_rdma_size_hint(
            capacity, hidden, self.world_size, experts
        )
        if shared_owner is not None:
            if (
                hidden != shared_owner.hidden
                or self.world_size != shared_owner.world_size
                or self.rdma_bytes > shared_owner.rdma_bytes
                or experts > shared_owner.experts
            ):
                raise ValueError("Shared legacy transport is too small for this model")
            self.buffer = shared_owner.buffer
            return
        self.buffer = Buffer(
            group,
            0,
            Buffer.get_low_latency_rdma_size_hint(
                capacity, hidden, self.world_size, experts
            ),
            low_latency_mode=True,
            num_qps_per_rank=experts // self.world_size,
            allow_nvlink_for_low_latency_mode=False,
            allow_mnnvl=False,
        )

    def apply(self, x, state, ids, weights, *, capacity, layout_capacity):
        from tokenspeed_kernel.thirdparty.deepep_sm120_experiment.prequant_fused_moe_opt import (
            apply_sm12x_fused_moe,
        )

        if (
            not 0 < capacity <= layout_capacity <= self.capacity
            or x.shape[0] > capacity
        ):
            raise ValueError("Low-latency dispatch capacity exceeded")
        if x.ndim != 2 or x.shape[1] != self.hidden or x.dtype != torch.bfloat16:
            raise ValueError("Expected BF16 token-major hidden states")
        if ids.shape != weights.shape or ids.shape[0] != x.shape[0]:
            raise ValueError("Route shape does not match source rows")
        if state.w2.shape[0] * self.world_size != self.experts:
            raise ValueError("Expert state does not match communication partition")
        # Dispatch capacity changes the RDMA metadata offsets. Clean both
        # buffers with the library's rank barrier before reusing a new layout.
        # Capture this operation too: graph replay bypasses Python bookkeeping.
        if self.cached_clean:
            self.buffer.clean_low_latency_buffer_cached(
                layout_capacity, self.hidden, self.experts, self.layout_state
            )
        else:
            self.buffer.clean_low_latency_buffer(
                layout_capacity, self.hidden, self.experts
            )
        received, counts, handle, _, hook = self.buffer.low_latency_dispatch(
            x.contiguous(),
            ids.to(torch.int64).contiguous(),
            layout_capacity,
            self.experts,
            use_fp8=False,
            use_nvfp4=True,
            return_recv_hook=True,
        )
        hook()
        payload, scales, globals_ = received
        from tokenspeed_kernel.thirdparty.flashinfer.sm12x_nvfp4_moe import (
            Sm12xNvfp4MoEWeights,
            apply_sm12x_nvfp4,
        )

        if isinstance(state, Sm12xNvfp4MoEWeights):
            from tokenspeed_kernel.thirdparty.deepep_sm120_experiment.nvfp4_receive import (
                decode_wire_nvfp4,
            )

            rows, local_ids, unit_weights = decode_wire_nvfp4(
                payload,
                scales,
                globals_,
                counts,
                self.world_size * capacity * ids.shape[1],
            )
            values = apply_sm12x_nvfp4(rows, state, local_ids, unit_weights, False)
            values = restore_expert_rows(
                values, counts, self.world_size * layout_capacity, zero_padding=False
            )
        elif self.expert_impl == "web":
            from tokenspeed_kernel.thirdparty.deepep_sm120_experiment.web_w4a8.online import (
                PreparedAdaptiveExperts,
                SharedWorkspace,
            )

            receive_capacity = self.world_size * layout_capacity
            intermediate = state.w2.shape[2] * 2
            topk = ids.shape[1]
            workspace_key = (layout_capacity, capacity, intermediate, topk)
            if workspace_key not in self.web_workspaces:
                self.web_workspaces[workspace_key] = SharedWorkspace(
                    experts=self.experts // self.world_size,
                    capacity=receive_capacity,
                    compact_capacity=min(
                        self.experts // self.world_size * receive_capacity,
                        self.world_size * capacity * topk,
                    ),
                    hidden=self.hidden,
                    intermediate=intermediate,
                    device=x.device,
                )
            workspace = self.web_workspaces[workspace_key]
            key = (id(state), workspace_key)
            if key not in self.web_prepared:
                self.web_states[id(state)] = state
                # Existing state already has [up; gate] and MN-major scales.
                # Calling prepare_weight_views again would swap the gates twice.
                self.web_prepared[key] = PreparedAdaptiveExperts(
                    state.w13,
                    state.sf13,
                    state.w2,
                    state.sf2,
                    workspace,
                    limit=state.limit,
                    threshold=128,
                )
            values = self.web_prepared[key].forward_wire(
                payload,
                scales,
                globals_,
                counts,
                wire_dtype="nvfp4",
                out=workspace.output,
            )
        else:
            q, qs, local_ids, unit_weights = to_fp8(
                payload,
                scales.view(torch.uint8),
                globals_,
                counts,
                max_compact_rows=(
                    self.world_size * capacity * ids.shape[1] if self.compact else None
                ),
            )
            values = apply_sm12x_fused_moe(q, qs, state, local_ids, unit_weights, False)
            if self.compact:
                values = restore_expert_rows(
                    values,
                    counts,
                    self.world_size * layout_capacity,
                    zero_padding=False,
                )
            else:
                values = values.view(
                    self.experts // self.world_size,
                    self.world_size * layout_capacity,
                    self.hidden,
                )
        output, _, hook = self.buffer.low_latency_combine(
            values,
            ids.to(torch.int64).contiguous(),
            weights.float().contiguous(),
            handle,
            use_fp8=True,
            return_recv_hook=True,
        )
        hook()
        return output

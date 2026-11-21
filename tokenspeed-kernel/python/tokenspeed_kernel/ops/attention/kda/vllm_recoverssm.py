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

"""TokenSpeed adapters for the vendored vLLM KDA RecoverSSM kernels."""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.attention.kda import KdaRetainedHistory
from tokenspeed_kernel.platform import CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures
from tokenspeed_kernel.thirdparty.vllm_recoverssm.kda import (
    _publish_kda_history_kernel,
    kda_recoverssm_verify,
)

_DENSE_HALF_SIGNATURES = format_signatures(
    ("q", "k", "v"), "dense", {torch.float16, torch.bfloat16}
)


def _validate_retained_request(
    history: KdaRetainedHistory,
    checkpoint_state: torch.Tensor,
    state_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    spec_query_len: int,
    record_metadata: torch.Tensor | None,
    record_stamps: torch.Tensor | None,
) -> None:
    """Check request inputs against validated storage without reading GPU values."""
    _, heads, _, values = history.correction.shape
    if (
        checkpoint_state.ndim != 4
        or checkpoint_state.shape[1:]
        != (
            heads,
            values,
            history.key.shape[3],
        )
        or checkpoint_state.dtype not in (torch.float32, torch.bfloat16)
    ):
        raise ValueError(
            "retained FP32/BF16 checkpoint must match the history geometry"
        )
    if history.live_state_indices.shape != state_indices.shape:
        raise ValueError(
            "retained source and live indices must have one entry per request"
        )
    if query_start_loc.shape != (state_indices.numel() + 1,):
        raise ValueError("retained query offsets must contain one boundary per request")
    if any(
        x.dtype not in (torch.int32, torch.int64)
        for x in (state_indices, query_start_loc)
    ):
        raise ValueError("retained indices and query offsets must use integer storage")
    if any(
        x.device != checkpoint_state.device
        for x in (history.correction, state_indices, query_start_loc)
    ):
        raise ValueError("retained history must be on the checkpoint device")
    if spec_query_len <= 0:
        raise ValueError("retained speculative window must be positive")
    if (record_metadata is None) != (record_stamps is None):
        raise ValueError("record metadata and stamps must be supplied together")
    if record_metadata is not None:
        if (
            record_metadata.shape != (state_indices.numel(), 3)
            or record_stamps.ndim != 3
            or record_stamps.shape[2] != 2
            or record_stamps.shape[1] < spec_query_len
        ):
            raise ValueError(
                "retained stamps need independent [slots, capacity, 2] storage covering the window"
            )
        if any(
            x.dtype != torch.int64
            or not x.is_contiguous()
            or x.device != checkpoint_state.device
            for x in (record_metadata, record_stamps)
        ):
            raise ValueError(
                "record metadata must be contiguous INT64 on the checkpoint device"
            )


@register_kernel(
    "attention",
    "kda_recoverssm_verify",
    name="vllm_triton_kda_recoverssm_verify",
    solution="vllm_triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
    signatures=_DENSE_HALF_SIGNATURES,
    priority=Priority.REFERENCE,
    traits={
        "paged_state": frozenset({True}),
        "recurrent_layout": frozenset({"v_major"}),
    },
)
def vllm_triton_kda_recoverssm_verify(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    checkpoint_state: torch.Tensor,
    retained_history: KdaRetainedHistory,
    query_start_loc: torch.Tensor,
    state_indices: torch.Tensor,
    spec_query_len: int,
    lower_bound: float | None,
    out: torch.Tensor,
    record_metadata: torch.Tensor | None,
    record_stamps: torch.Tensor | None,
) -> torch.Tensor:
    """Verify packed candidates into private retained pages; return out."""
    _validate_retained_request(
        retained_history,
        checkpoint_state,
        state_indices,
        query_start_loc,
        spec_query_len,
        record_metadata,
        record_stamps,
    )
    return kda_recoverssm_verify(
        q,
        k,
        v,
        raw_g,
        raw_beta,
        A_log,
        dt_bias,
        lower_bound,
        checkpoint_state,
        query_start_loc,
        state_indices,
        spec_query_len,
        out=out,
        record_metadata=record_metadata,
        record_stamps=record_stamps,
        retained_history=retained_history,
    )


@register_kernel(
    "attention",
    "kda_recoverssm_commit",
    name="vllm_triton_kda_recoverssm_commit",
    solution="vllm_triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
    signatures=_DENSE_HALF_SIGNATURES,
    priority=Priority.REFERENCE,
    traits={
        "attention_only": frozenset({True}),
        "paged_state": frozenset({True}),
        "recurrent_layout": frozenset({"v_major"}),
    },
)
def vllm_triton_kda_recoverssm_commit(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    checkpoint_state: torch.Tensor,
    retained_history: KdaRetainedHistory,
    state_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    spec_query_len: int,
    record_metadata: torch.Tensor | None,
    record_stamps: torch.Tensor | None,
) -> None:
    """Validate accepted stamps and publish slot retained lengths without changing dense state."""
    del q, k, v
    batch = state_indices.shape[0]
    _validate_retained_request(
        retained_history,
        checkpoint_state,
        state_indices,
        query_start_loc,
        spec_query_len,
        record_metadata,
        record_stamps,
    )
    if (
        num_accepted_tokens.shape != (batch,)
        or num_accepted_tokens.dtype not in (torch.int32, torch.int64)
        or num_accepted_tokens.device != checkpoint_state.device
    ):
        raise ValueError(
            "retained accepted counts must be an integer vector on the checkpoint device"
        )
    if batch:
        _publish_kda_history_kernel[batch,](
            retained_history.length,
            state_indices,
            retained_history.live_state_indices,
            retained_history.record_slots,
            query_start_loc,
            num_accepted_tokens,
            record_metadata,
            record_stamps,
            retained_history.length.stride(0),
            state_indices.stride(0),
            retained_history.live_state_indices.stride(0),
            retained_history.record_slots.stride(0),
            retained_history.correction.shape[0],
            query_start_loc.stride(0),
            num_accepted_tokens.stride(0),
            record_stamps.shape[0] if record_stamps is not None else 0,
            checkpoint_state.shape[0],
            STAMP_CAPACITY=(record_stamps.shape[1] if record_stamps is not None else 1),
            CAPACITY=retained_history.correction.shape[2],
            SPEC_QUERY_LEN=spec_query_len,
            num_warps=1,
            debug=True,
        )
    return


__all__ = ["vllm_triton_kda_recoverssm_commit", "vllm_triton_kda_recoverssm_verify"]


@triton.jit(do_not_specialize=["batch", "generation"])
def _prepare_record_metadata_kernel(
    seq_lens, metadata, batch, generation, draft_token_num, BLOCK: tl.constexpr
):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = row < batch
    position = (
        tl.load(seq_lens + row, mask=live, other=0).to(tl.int64) - draft_token_num
    )
    tl.store(metadata + row * 3, row.to(tl.int64), mask=live)
    tl.store(metadata + row * 3 + 1, generation.to(tl.int64), mask=live)
    tl.store(metadata + row * 3 + 2, position, mask=live)


def prepare_recoverssm_metadata(
    seq_lens: torch.Tensor,
    metadata: torch.Tensor,
    *,
    batch: int,
    generation: int,
    draft_token_num: int,
) -> None:
    """Refresh the per-round slot, generation and checkpoint-position view.

    seq_lens holds each request's INT32 or INT64 cache length after the
    verify window; its checkpoint position is ``seq_lens - draft_token_num``.
    metadata is caller-owned, contiguous INT64 [max_batch, 3] storage. Only
    the first batch rows change. generation identifies this round's workspace
    ownership. Return None.
    """
    if seq_lens.dtype not in (torch.int32, torch.int64):
        raise ValueError("record metadata reads INT32 or INT64 cache lengths")
    if metadata.dtype != torch.int64:
        raise ValueError("record metadata requires INT64 storage")
    if seq_lens.ndim != 1 or seq_lens.stride(0) != 1:
        raise ValueError("record metadata reads dense cache lengths")
    if metadata.ndim != 2 or metadata.shape[1] != 3 or (not metadata.is_contiguous()):
        raise ValueError("record metadata must be contiguous [max_batch, 3]")
    if batch > min(seq_lens.numel(), metadata.shape[0]) or generation < 0:
        raise ValueError("invalid record metadata batch or generation")
    if batch == 0:
        return
    _prepare_record_metadata_kernel[triton.cdiv(batch, 128),](
        seq_lens, metadata, batch, generation, draft_token_num, BLOCK=128
    )

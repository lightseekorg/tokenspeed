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

"""Benchmark generators for Qwen4-Exp QSA sparse attention."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from tokenspeed_kernel.benchmark.graph import PreparedInvocation
from tokenspeed_kernel.benchmark.harness import (
    BenchmarkCaseError,
    BenchmarkRequest,
    BenchmarkStatus,
    PreparedBenchmark,
)
from tokenspeed_kernel.platform import PlatformInfo
from tokenspeed_kernel.registry import KernelRegistry, load_builtin_kernels
from tokenspeed_kernel.selection import NoKernelFoundError

__all__ = ["prepare_qsa_sparse_attention", "qsa_selected_slots"]


@dataclass(frozen=True)
class _QsaModelProfile:
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    block_size: int
    block_topk: int
    page_size: int


# Per-rank QSA geometry: Qwen3.8-Flash-Next has 24 query / 2 KV heads of 256
# dims split over TP2, and selects 2051 KV slots per query row.
_MODEL_PROFILES = {
    "qwen38_flash_tp2": _QsaModelProfile(
        num_q_heads=12,
        num_kv_heads=1,
        head_dim=256,
        block_size=4,
        block_topk=512,
        page_size=64,
    ),
}
_KV_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float8_e4m3fn": torch.float8_e4m3fn,
}
# Slot 0 may hold NaN padding and slot 1 is the finite masked-read target, so
# request KV starts at slot 2.
_FIRST_REQUEST_SLOT = 2


def _invalid(message: str) -> BenchmarkCaseError:
    return BenchmarkCaseError(BenchmarkStatus.INVALID_CASE, message)


def _positive(parameters: dict[str, Any], name: str) -> int:
    value = parameters[name]
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise _invalid(f"QSA {name} must be a positive integer")
    return value


def qsa_selected_slots(
    positions: torch.Tensor,
    request_index: torch.Tensor,
    region_slots: int,
    width: int,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    """Build ``[rows, width]`` selected slots for query rows at ``positions``.

    Request ``r`` owns the contiguous slots ``[2 + r * region_slots, ...)``.
    A row at position ``p`` (1-based context length) selects every one of its
    ``p`` slots while ``p <= width``, otherwise a random ``width`` subset of
    them, in increasing slot order. Unused columns are ``-1``.
    """

    rows = positions.numel()
    context = int(positions.max())
    scores = torch.rand((rows, context), device=positions.device, generator=generator)
    offsets = torch.arange(context, device=positions.device)
    visible = offsets.unsqueeze(0) < positions.unsqueeze(1)
    scores = scores.masked_fill(~visible, -1.0)
    take = min(width, context)
    chosen_scores, chosen = scores.topk(take, dim=1)
    # Invalid picks sort after every valid offset, then become -1.
    chosen = chosen.masked_fill(chosen_scores < 0, context)
    chosen, _ = chosen.sort(dim=1)
    base = _FIRST_REQUEST_SLOT + request_index.unsqueeze(1) * region_slots
    slots = torch.where(chosen < context, chosen + base, -1)
    if take < width:
        pad = slots.new_full((rows, width - take), -1)
        slots = torch.cat((slots, pad), dim=1)
    return slots.to(torch.int32).contiguous()


def prepare_qsa_sparse_attention(
    request: BenchmarkRequest,
    platform: PlatformInfo,
) -> PreparedBenchmark:
    """Prepare one QSA sparse attention call over selected KV slots.

    ``phase="decode"`` runs ``batch`` requests of ``q_len`` query rows (1 for
    a draft step, the MTP width for verify) at a context of
    ``context_tokens``. ``phase="prefill"`` runs ``batch`` prompts of
    ``context_tokens`` rows each, every row attending to its own prefix.
    """

    _ = platform
    parameters = request.parameters
    if parameters.get("validation") is not None:
        raise _invalid("QSA benchmark correctness validation is not implemented")
    model_profile = parameters["model_profile"]
    if model_profile not in _MODEL_PROFILES:
        raise _invalid(
            f"Implemented QSA model_profile values: {', '.join(sorted(_MODEL_PROFILES))}"
        )
    profile = _MODEL_PROFILES[model_profile]
    if parameters["dtype"] != "bfloat16":
        raise _invalid("QSA query dtype must be bfloat16")
    kv_dtype_name = parameters["kv_dtype"]
    if kv_dtype_name not in _KV_DTYPES:
        raise _invalid(f"QSA kv_dtype must be one of: {', '.join(sorted(_KV_DTYPES))}")
    kv_dtype = _KV_DTYPES[kv_dtype_name]
    phase = parameters["phase"]
    batch = _positive(parameters, "batch")
    context = _positive(parameters, "context_tokens")
    if phase == "decode":
        q_len = _positive(parameters, "q_len")
        if q_len > context:
            raise _invalid("QSA q_len must not exceed context_tokens")
        positions = torch.arange(context - q_len + 1, context + 1, device="cuda")
        max_seqlen_q: int | None = q_len
    elif phase == "prefill":
        q_len = context
        positions = torch.arange(1, context + 1, device="cuda")
        max_seqlen_q = None
    else:
        raise _invalid("QSA phase must be decode or prefill")
    positions = positions.repeat(batch)
    request_index = torch.arange(batch, device="cuda").repeat_interleave(q_len)
    rows = batch * q_len

    generator = torch.Generator(device="cuda").manual_seed(request.seed)
    # Sample complete logical blocks; the physical-slot expansion adds the
    # causal tail. Every solution then attends to the same candidates.
    num_blocks = (context + profile.block_size - 1) // profile.block_size
    scores = torch.rand((rows, num_blocks), device="cuda", generator=generator)
    completed = positions // profile.block_size
    scores.masked_fill_(
        torch.arange(num_blocks, device="cuda")[None, :] >= completed[:, None], -1
    )
    chosen_scores, blocks = scores.topk(min(profile.block_topk, num_blocks), dim=1)
    blocks.masked_fill_(chosen_scores < 0, -1)
    if blocks.shape[1] < profile.block_topk:
        blocks = torch.cat(
            (blocks, blocks.new_full((rows, profile.block_topk - blocks.shape[1]), -1)),
            dim=1,
        )
    blocks = blocks.to(torch.int32)
    pages_per_request = (context + profile.page_size - 1) // profile.page_size
    block_table = torch.arange(
        1, 1 + batch * pages_per_request, device="cuda", dtype=torch.int32
    ).view(batch, pages_per_request)
    cache_slots = (1 + batch * pages_per_request) * profile.page_size
    q = torch.randn(
        (rows, profile.num_q_heads, profile.head_dim),
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    k_cache, v_cache = (
        torch.randn(
            (cache_slots, profile.num_kv_heads, profile.head_dim),
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        ).to(kv_dtype)
        for _ in range(2)
    )
    k_cache[: profile.page_size].zero_()
    v_cache[: profile.page_size].zero_()
    descale = 1.0 if kv_dtype == torch.float8_e4m3fn else None

    from tokenspeed_kernel.ops.attention.qsa import (
        QSASparseMetadata,
        QSAWorkspace,
        qsa_sparse_attention,
        select_qsa_sparse_attention_kernel,
    )
    from tokenspeed_kernel.ops.attention.qsa.triton import qwen4_exp_qsa_selected_slots

    sparse_metadata = QSASparseMetadata(
        selected_blocks=blocks,
        block_table=block_table,
        token_to_request=request_index.to(torch.int32),
        query_positions=positions - 1,
        query_start_loc=tuple(range(0, rows + 1, q_len)),
        page_size=profile.page_size,
        block_size=profile.block_size,
        workspace=QSAWorkspace(),
    )
    selected = qwen4_exp_qsa_selected_slots(
        blocks,
        completed.to(torch.int32),
        sparse_metadata.query_positions,
        sparse_metadata.token_to_request,
        block_table,
        profile.page_size,
        profile.block_size,
        profile.block_topk * profile.block_size,
        enable_pdl=False,
    )

    load_builtin_kernels()
    try:
        kernel = select_qsa_sparse_attention_kernel(
            q,
            k_cache,
            v_cache,
            selected,
            max_seqlen_q=max_seqlen_q,
            k_scale=descale,
            v_scale=descale,
            sparse_metadata=sparse_metadata,
            override=request.registration,
            solution=request.solution,
        )
    except NoKernelFoundError as error:
        raise BenchmarkCaseError(BenchmarkStatus.NOT_APPLICABLE, str(error)) from error
    spec = KernelRegistry.get().get_by_name(kernel.name)
    if spec is None:
        raise BenchmarkCaseError(
            BenchmarkStatus.REGISTRATION_MISSING,
            f"Selected registration {kernel.name!r} is not available",
        )

    def invoke() -> object:
        return qsa_sparse_attention(
            q,
            k_cache,
            v_cache,
            selected,
            scale=profile.head_dim**-0.5,
            max_seqlen_q=max_seqlen_q,
            metadata_capacity_rows=rows,
            k_scale=descale,
            v_scale=descale,
            override=request.registration,
            solution=request.solution,
            sparse_metadata=sparse_metadata,
        )

    return PreparedBenchmark(
        registration=spec,
        invocation=PreparedInvocation(invoke=invoke),
        parameters={
            "model_profile": model_profile,
            "phase": phase,
            "batch": batch,
            "q_len": q_len,
            "context_tokens": context,
            "dtype": "bfloat16",
            "kv_dtype": kv_dtype_name,
        },
        validation=None,
    )

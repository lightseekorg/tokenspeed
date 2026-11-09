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

"""CuTe DSL top-k for the DSA decode indexer.

``cute_dsl_decode_topk`` wraps the vendored TensorRT-LLM CuTe DSL radix runners
as a length-aware per-row top-k over a ``[num_rows, num_cols]`` indexer-logits
matrix, matching the ``deterministic_decode_topk(..., lengths=seq_lens,
q_len_per_req=next_n)`` contract it replaces (returned int32 column indices are
the "local offsets" that ``local_topk_to_global_slots`` maps to KV slots).
Cluster-first: falls back to the non-cluster runner when the problem exceeds
cluster capacity. NVIDIA Blackwell (sm_100+) only; gate on
:func:`has_cute_dsl_decode_topk`.

``cute_dsl_dsa_decode_topk`` is the registered fused ``dsa_decode_topk`` leaf
(README, "Fused decode top-k"): the paged FP32 logits and the exact top-k of a
decode step in two launches, selected only by ``solution="cute_dsl"``.
"""

from __future__ import annotations

import torch

# The package defines SLOT_ORDERS before importing its leaves.
from tokenspeed_kernel.ops.attention.dsa import SLOT_ORDERS
from tokenspeed_kernel.ops.attention.dsa._triton.topk import (
    quantize_index_query_weights,
)
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
    pdl_enabled,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

__all__ = [
    "cute_dsl_decode_topk",
    "has_cute_dsl_decode_topk",
]


def _ts_supported_arch() -> bool:
    """Gate: the CuTe DSL multi-CTA / cluster radix top-k needs NVIDIA sm_100+.

    Returns False if platform detection raises (e.g. CPU-only host) so callers
    fall back transparently.
    """
    try:
        p = current_platform()
    except Exception:
        return False
    if not p.is_nvidia:
        return False
    sm = p.arch_version.major * 10 + p.arch_version.minor
    return sm >= 100


_CUTE_DSL_TOPK_AVAILABLE = False
_ClusterRunner = None
_BaseRunner = None

if _ts_supported_arch():
    try:
        import cutlass  # noqa: F401  (import probe: kernels JIT via cutlass-dsl)
        import cutlass.cute  # noqa: F401
        from tokenspeed_kernel.thirdparty.cute_dsl.topk import (
            CuteDSLTopKDecodeSinglePassMultiCTAClusterRunner as _ClusterRunner,
        )
        from tokenspeed_kernel.thirdparty.cute_dsl.topk import (
            CuteDSLTopKDecodeSinglePassMultiCTARunner as _BaseRunner,
        )

        _CUTE_DSL_TOPK_AVAILABLE = True
    except ImportError:
        _CUTE_DSL_TOPK_AVAILABLE = False


def has_cute_dsl_decode_topk() -> bool:
    """Whether the CuTe DSL decode top-k kernel can run on this platform."""
    return _CUTE_DSL_TOPK_AVAILABLE


_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def cute_dsl_decode_topk(
    logits: torch.Tensor,
    seq_lens: torch.Tensor,
    topk: int,
    *,
    next_n: int = 1,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Length-aware per-row top-k over a dense indexer-logits matrix.

    Args:
        logits: 2-D row-major contiguous CUDA tensor ``[num_rows, num_cols]``
            (``num_rows = num_reqs * next_n``), fp16/bf16/fp32. No pre-masking
            needed; the causal window is derived in-kernel from seq_lens/next_n.
        seq_lens: Per-request candidate length, int32 ``[num_reqs]``. Row ``r``'s
            window is ``seq_lens[r // next_n] - next_n + (r % next_n) + 1`` cols.
        topk: Candidates to select per row (1..2048).
        next_n: Query rows per request (speculative ``q_len_per_req``); 1 for decode.
        out: Optional int32 ``[num_rows, topk]`` buffer, written in place and returned.

    Returns:
        int32 ``[num_rows, topk]`` of per-row selected column indices (unsorted).
        Rows with window < topk have surplus slots filled by the radix collection;
        track valid count separately (``min(window, topk)``), as with
        ``deterministic_decode_topk``.

    Raises:
        RuntimeError: if the kernel is unavailable on this platform.
    """
    if not _CUTE_DSL_TOPK_AVAILABLE:
        raise RuntimeError(
            "cute_dsl_decode_topk is unavailable on this platform "
            "(requires NVIDIA Blackwell sm_100+ with nvidia-cutlass-dsl)."
        )
    if logits.dim() != 2:
        raise ValueError(
            f"logits must be 2-D [num_rows, num_cols], got {tuple(logits.shape)}"
        )
    if logits.dtype not in _SUPPORTED_DTYPES:
        raise ValueError(
            f"logits dtype must be one of {_SUPPORTED_DTYPES}, got {logits.dtype}"
        )
    num_rows = logits.shape[0]
    topk = int(topk)
    next_n = int(next_n)

    logits = logits.contiguous()
    seq_lens = seq_lens.to(device=logits.device, dtype=torch.int32).contiguous()

    if out is not None:
        if out.shape != (num_rows, topk):
            raise ValueError(
                f"out must have shape {(num_rows, topk)}, got {tuple(out.shape)}"
            )
        if out.dtype != torch.int32:
            raise ValueError(f"out dtype must be int32, got {out.dtype}")
        out = out.contiguous()

    # Cluster-first. Re-zero the shared ``row_states`` scratch every call: the
    # kernels self-clean it, but that reuse races under tight CUDA-graph scheduling.
    _ClusterRunner._row_states_initialized = False
    indices, _ = _ClusterRunner.forward(
        logits,
        seq_lens,
        topk,
        next_n,
        return_val=False,
        output_indices=out,
    )
    if indices is None:
        # Exceeds cluster capacity: fall back to the non-cluster runner.
        _BaseRunner._row_states_initialized = False
        indices, _ = _BaseRunner.forward(
            logits,
            seq_lens,
            topk,
            next_n,
            return_val=False,
            output_indices=out,
        )
    return indices


# Fused decode top-k: paged FP32 logits and the exact top-k in two launches.

_fused_indexer = None
if _ts_supported_arch():
    try:
        from tokenspeed_kernel.thirdparty.cute_dsl.k3_decode.dsa import (
            indexer as _fused_indexer,
        )
    except ImportError:
        _fused_indexer = None

# Requests one fused decode round schedules (the scorer's tile prefix sum), 0 without it.
FUSED_DECODE_TOPK_MAX_REQUESTS = (
    0 if _fused_indexer is None else _fused_indexer.MAX_REQUESTS
)

# Index-K page: 64 keys of 128 FP8 bytes, then their 64 FP32 scales.
_FUSED_PAGE = 64
_FUSED_ROW_BYTES = 128 + 4
_FUSED_HEADS = 32
_FUSED_TILE = 128


def _fused_outputs(
    out: torch.Tensor | None,
    lens_out: torch.Tensor | None,
    *,
    tokens: int,
    topk: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The caller's slot rows and counts, or new ones.

    The slot rows are int32 ``[tokens, topk]`` with unit column stride and the
    counts int32 ``[tokens]``.
    """
    if out is None:
        out = torch.empty((tokens, topk), dtype=torch.int32, device=device)
    elif (
        out.shape != (tokens, topk)
        or out.dtype != torch.int32
        or out.device != device
        or out.stride(1) != 1
    ):
        raise ValueError(
            f"out must be int32 [{tokens}, {topk}] with unit column stride on "
            f"{device}, got {tuple(out.shape)} {out.dtype} {out.device} "
            f"strides {out.stride()}"
        )
    if lens_out is None:
        lens_out = torch.empty((tokens,), dtype=torch.int32, device=device)
    elif (
        lens_out.shape != (tokens,)
        or lens_out.dtype != torch.int32
        or lens_out.device != device
        or lens_out.stride(0) != 1
    ):
        raise ValueError(
            f"lens_out must be contiguous int32 [{tokens}] on {device}, got "
            f"{tuple(lens_out.shape)} {lens_out.dtype} {lens_out.device}"
        )
    return out, lens_out


def fused_decode_topk_scores(
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    index_k_cache: torch.Tensor,
    *,
    q_len_per_req: int,
    topk: int,
    out: torch.Tensor,
    lens_out: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score and select a decode step with the fused CuTe DSL kernels.

    Args:
        q_fp8: ``[tokens, 32, 128]`` contiguous E4M3 index queries.
        weights: ``[tokens, 32]`` contiguous FP32 head weights with the query
            scales and the score scale folded in (``quantize_index_query_weights``).
        seq_lens: int32 ``[requests]`` request lengths, unit stride
            (``requests = tokens / q_len_per_req``).
        block_table: int32 ``[requests, pages]`` page table, unit column stride.
        index_k_cache: contiguous packed uint8 index-K plane ``[slots, 132]``.
        q_len_per_req: Query rows per request (1-8).
        topk: Keys selected per row.
        out: int32 ``[tokens, topk]`` slot rows, unit column stride; written.
        lens_out: int32 ``[tokens]`` counts; written.

    Returns:
        ``(logits, tops)``: the request-compact FP32 logits scratch and its
        int32 8-key maxima (indexer.py describes the layout). Rows of requests
        no longer than ``topk`` are not scored.
    """
    tokens = q_fp8.shape[0]
    device = q_fp8.device
    width = block_table.shape[1]
    # Request-compact rows: at most every row the whole block table's keys.
    row_capacity = -(-width * _FUSED_PAGE // _FUSED_TILE) * _FUSED_TILE
    elements = tokens * row_capacity
    # The scratch's size is a 32-bit runtime extent; offsets into it are 64-bit.
    if elements >= 1 << 31:
        raise ValueError(
            f"fused DSA decode top-k scratch of {elements} logits exceeds 2**31 "
            f"(tokens={tokens}, block-table width={width})"
        )
    logits = torch.empty(elements, dtype=torch.float32, device=device)
    tops = torch.empty(elements // 8, dtype=torch.int32, device=device)
    score, select = _fused_indexer.compile_indexer(
        int(q_len_per_req), int(topk), current_platform().sm_count, pdl_enabled()
    )
    score(
        q_fp8.view(torch.uint8).view(tokens * _FUSED_HEADS, 128),
        weights.view(-1),
        index_k_cache,
        block_table,
        seq_lens,
        logits,
        tops,
    )
    select(logits, tops, seq_lens, block_table, out, lens_out)
    return logits, tops


if _fused_indexer is not None:

    @register_kernel(
        "attention",
        "dsa_decode_topk",
        name="cute_dsl_dsa_decode_topk",
        solution="cute_dsl",
        # Row-local and deterministic: ties go to the lowest key.
        features=frozenset({"batch_invariant"}),
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(10, 0),
            max_arch_version=ArchVersion(10, 3),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=frozenset(
            {
                format_signature(
                    q=dense_tensor_format(torch.bfloat16),
                    weights=dense_tensor_format(weights_dtype),
                )
                for weights_dtype in (torch.float32, torch.bfloat16)
            }
            | {
                # Queries quantized ahead: unit scales, folded into the weights.
                format_signature(
                    q=dense_tensor_format(torch.float8_e4m3fn),
                    weights=dense_tensor_format(torch.float32),
                )
            }
        ),
        traits={
            "q_len": frozenset({1, 2, 3, 4, 5, 6, 7, 8}),
            "index_heads": frozenset({_FUSED_HEADS}),
            "head_dim": frozenset({128}),
            "page_size": frozenset({_FUSED_PAGE}),
            "topk": frozenset({2048}),
            "index_k_format": frozenset({"fp8_scaled"}),
            "index_k_layout": frozenset({"packed"}),
            "slot_order": frozenset({"selection", "sorted"}),
        },
        # Opt-in: below every other leaf, so only solution="cute_dsl" selects it.
        priority=Priority.PORTABLE - 1,
    )
    def cute_dsl_dsa_decode_topk(
        q: torch.Tensor,
        weights: torch.Tensor,
        seq_lens: torch.Tensor,
        block_table: torch.Tensor,
        *,
        page_size: int,
        topk: int,
        softmax_scale: float,
        q_len_per_req: int,
        slot_order: str,
        index_k_cache: torch.Tensor | None,
        seq_lens_2d: torch.Tensor | None,
        plan: object | None,
        out: torch.Tensor | None = None,
        lens_out: torch.Tensor | None = None,
        batch_invariant: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fused DSA decode top-k: the paged logits and each row's exact top-k.

        The FP32 logits are bit-identical to DeepGEMM's paged MQA logits, with
        NaN and +inf scored -inf. The exact top-k of each row's causal window
        comes in ascending position order (ties at the threshold to the lowest
        positions), as global slots with a ``-1`` tail.

        Args:
            q: ``[tokens, 32, 128]`` BF16 index queries, unit last stride, or
                contiguous E4M3 ones with unit scales
                (``index_k_layernorm_fp8_scatter_and_quantize_query``).
            weights: ``[tokens, 32]`` FP32 or BF16 head weights, unit last
                stride; with E4M3 queries contiguous FP32 weights carrying
                each query's scale and the score scale.
            seq_lens: int32 ``[requests]`` request lengths; verify row ``j``
                sees ``seq_lens[req] - (q_len_per_req - 1) + j`` keys.
            block_table: int32 ``[requests, pages]`` page table of the index-K
                plane and the latent cache, unit column stride.
            page_size: Keys per page (64).
            topk: Keys selected per row (2048).
            softmax_scale: Score scale folded into the head weights; 1 with
                E4M3 queries, whose weights carry it.
            q_len_per_req: Query rows per request (1-8).
            slot_order: ``"selection"`` or ``"sorted"``; both emit ascending
                positions, which is this leaf's selection order.
            index_k_cache: Contiguous packed uint8 index-K plane ``[slots, 132]``.
            seq_lens_2d: The DeepGEMM scoring extent of other leaves; unused, the
                scorer schedules from ``seq_lens`` on the device.
            plan: The DeepGEMM schedule of other leaves; unused, as above.
            out: Optional int32 ``[tokens, topk]`` slots, unit column stride.
            lens_out: Optional contiguous int32 ``[tokens]`` counts.
            batch_invariant: Accepted for the facade; every selection is
                already a function of its own row.

        Returns:
            ``(slots, lens)``: ``[tokens, topk]`` global slots, ``-1`` past
            each row's count, and the counts ``min(window, topk)``.
        """
        del seq_lens_2d, plan, batch_invariant
        if slot_order not in SLOT_ORDERS:
            raise ValueError(
                f"slot_order must be one of {list(SLOT_ORDERS)}, got {slot_order!r}"
            )
        tokens = q.shape[0]
        q_len_per_req = int(q_len_per_req)
        if not 1 <= q_len_per_req <= 8 or tokens % q_len_per_req:
            raise ValueError(
                f"q_len_per_req={q_len_per_req} must be 1-8 and divide tokens={tokens}"
            )
        requests = tokens // q_len_per_req
        if requests > FUSED_DECODE_TOPK_MAX_REQUESTS:
            raise ValueError(
                f"the fused DSA decode top-k schedules at most "
                f"{FUSED_DECODE_TOPK_MAX_REQUESTS} requests, got {requests}"
            )
        if int(page_size) != _FUSED_PAGE or q.shape[1:] != (_FUSED_HEADS, 128):
            raise ValueError(
                "the fused DSA decode top-k takes [tokens, 32, 128] queries over "
                f"64-key pages, got {tuple(q.shape)} and page_size={page_size}"
            )
        if (
            index_k_cache is None
            or index_k_cache.dtype != torch.uint8
            or index_k_cache.dim() != 2
            or index_k_cache.shape[1] != _FUSED_ROW_BYTES
            or index_k_cache.shape[0] % _FUSED_PAGE
            or not index_k_cache.is_contiguous()
            or index_k_cache.data_ptr() % 16
        ):
            raise ValueError(
                "the fused DSA decode top-k reads a contiguous, 16-byte aligned "
                "packed uint8 index-K plane [slots, 132] of whole pages"
            )
        if (
            seq_lens.dtype != torch.int32
            or seq_lens.dim() != 1
            or seq_lens.shape[0] < requests
            or seq_lens.stride(0) != 1
        ):
            raise ValueError(
                f"seq_lens must be int32 with unit stride and {requests} entries"
            )
        if (
            block_table.dtype != torch.int32
            or block_table.dim() != 2
            or block_table.shape[0] < requests
            or block_table.shape[1] < 1
            or block_table.stride(1) != 1
        ):
            raise ValueError(
                "block_table must be int32 [requests, pages] with unit column stride"
            )
        prequantized = q.dtype == torch.float8_e4m3fn
        if prequantized and (
            softmax_scale != 1.0
            or weights.dtype != torch.float32
            or weights.shape != (tokens, _FUSED_HEADS)
            or not (q.is_contiguous() and weights.is_contiguous())
        ):
            raise ValueError(
                "E4M3 queries take unit scales: contiguous FP32 [tokens, 32] "
                "weights carrying the query scales and the score scale, and "
                f"softmax_scale 1, got softmax_scale={softmax_scale} and "
                f"{weights.dtype} weights {tuple(weights.shape)}"
            )
        out, lens_out = _fused_outputs(
            out, lens_out, tokens=tokens, topk=int(topk), device=q.device
        )
        if tokens == 0:
            return out, lens_out
        if prequantized:
            q_fp8, folded = q, weights
        else:
            q_fp8, folded = quantize_index_query_weights(q, weights, softmax_scale)
        fused_decode_topk_scores(
            q_fp8,
            folded,
            seq_lens[:requests],
            block_table[:requests],
            index_k_cache,
            q_len_per_req=q_len_per_req,
            topk=int(topk),
            out=out,
            lens_out=lens_out,
        )
        return out, lens_out

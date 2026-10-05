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

from __future__ import annotations

import math

import torch
from tokenspeed_kernel.platform import pdl_enabled
from tokenspeed_kernel.profiling import ShapeCapture, kernel_scope
from tokenspeed_kernel.registry import KernelRegistry
from tokenspeed_kernel.selection import NoKernelFoundError, select_kernel
from tokenspeed_kernel.signature import (
    MXFP8_BLOCK_SCALE,
    dense_tensor_format,
    format_signature,
    tensor_format,
)

AttentionResult = torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]


# One UE8M0 scale per 32 consecutive head_dim elements (MXFP8).
MXFP8_ATTENTION_BLOCK_SCALE = MXFP8_BLOCK_SCALE


def _attention_format_signature(**roles: torch.Tensor):
    return format_signature(
        **{role: dense_tensor_format(tensor.dtype) for role, tensor in roles.items()}
    )


def _mxfp8_attention_format_signature(**roles: torch.Tensor):
    return format_signature(
        **{
            role: tensor_format(
                "mxfp8", tensor.dtype, scale=MXFP8_ATTENTION_BLOCK_SCALE
            )
            for role, tensor in roles.items()
        }
    )


def _blockscaled_signature_and_scales(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    q_scale: torch.Tensor | None,
    k_scale: torch.Tensor | None,
    v_scale: torch.Tensor | None,
):
    """Pick dense vs MXFP8 signature and build the scale kwargs splat.

    q_scale selects the block-scaled path; k_scale/v_scale must accompany it.
    Returns (signature, scale_kwargs) for the paged-KV-cache entry points.
    """
    if q_scale is not None:
        assert (
            k_scale is not None and v_scale is not None
        ), "MXFP8 attention requires q_scale, k_scale, and v_scale together"
        signature = _mxfp8_attention_format_signature(
            q=q, k_cache=k_cache, v_cache=v_cache
        )
    else:
        signature = _attention_format_signature(q=q, k_cache=k_cache, v_cache=v_cache)
    return signature, dict(q_scale=q_scale, k_scale=k_scale, v_scale=v_scale)


LSE_LN = math.log2(math.e)

# The order a token's selected KV rows are reduced in: established by the
# top-k leaf (``dsa_decode_topk`` / ``dsa_prefill_topk`` ``slot_order``),
# which emits its selection in that order, and kept by the sparse core
# (``dsa_decode`` / ``dsa_prefill`` ``slot_order``), which reduces the slots
# as they arrive:
#
# ``"selection"``
#     The top-k leaf's own order (its segment layout, its tie order). Every
#     registered leaf and core serves this form; declaring the trait is
#     optional.
# ``"sorted"``
#     Ascending POSITION order of the selected rows, whatever the leaf's tie
#     order, so the reduction is one function of the selected set: invariant
#     across batch compositions, runs and engines. Only the top-k leaf knows a
#     slot's position, so only leaves declaring ``slot_order={"sorted", ...}``
#     serve it and receive the kwarg; a core declaring the trait promises to
#     reduce in the emitted order. Ascending *slot* order is not a canonical
#     order: a request's pages are allocated in arbitrary id order once pages
#     recycle, so the same positions map to differently ordered slots on
#     another engine (or the same engine later), and the reduction's bits
#     would follow the page placement.
#
# The default form first, as the host's ``DSA_SLOT_ORDERS`` lists them.
SLOT_ORDERS = ("selection", "sorted")

# Feature a ``dsa_prefill_topk`` leaf declares when its signature takes the
# ``candidate_lens_cpu`` keyword (the host mirror of each token's candidate
# count, from which it sizes its launches without a stream sync). The facade
# hands the mirror to exactly the leaves declaring it, by registry lookup;
# a ``*args, **kwargs`` wrapper that does not declare it never sees it.
CANDIDATE_LENS_CPU_FEATURE = "candidate_lens_cpu"

# Feature a ``dsa_prefill_topk`` leaf declares when its launcher takes index-K
# rows handed to it already in workspace-row order instead of resolving them
# from a plane through ``kv_workspace_slots``: the ``index_k_fp8`` +
# ``index_k_scale`` keywords for an ``fp8_scaled`` plane, ``index_k_bf16`` for
# a ``bf16`` one (the query-context-parallel history gather over page-sharded
# caches produces them). The facade REQUIRES it whenever such rows are passed
# and hands the keywords to declaring leaves only (``_index_k_rows_kwargs``),
# so a leaf that only reads planes is never selected for them -- not by
# ranking, not by an override -- and the failure is a ``NoKernelFoundError``
# at selection, not the leaf raising mid-forward. A host probes the selection
# at construction through :func:`select_dsa_prefill_topk_for_rows`.
INDEX_K_WORKSPACE_ROWS_FEATURE = "index_k_workspace_rows"

# Storage of an index-key plane, read off its dtype (README, "Index-K plane
# formats"): one layout per dtype, never guessed from a row width alone.
_INDEX_K_FP8_GROUP_SIZE = 128
_INDEX_K_SCALE_BYTES = 4


def _index_k_plane_traits(index_k_cache: torch.Tensor, head_dim: int) -> dict:
    """``index_k_format`` / ``index_k_layout`` selection traits of a plane.

    ``uint8`` planes hold FP8 keys with one fp32 scale per 128 elements
    (``"fp8_scaled"``): ``"packed"`` as ``[slots, head_dim + 4 *
    head_dim / 128]`` rows, otherwise ``"page_planar"``. ``bfloat16`` planes
    hold the keys unquantized (``"bf16"``) and are only ever ``"packed"``
    ``[slots, head_dim]``.
    """
    if index_k_cache.dtype == torch.uint8:
        row_bytes = (
            head_dim + head_dim // _INDEX_K_FP8_GROUP_SIZE * _INDEX_K_SCALE_BYTES
        )
        packed = index_k_cache.ndim == 2 and index_k_cache.shape[1] == row_bytes
        return {
            "index_k_format": "fp8_scaled",
            "index_k_layout": "packed" if packed else "page_planar",
        }
    if index_k_cache.dtype == torch.bfloat16:
        if index_k_cache.ndim != 2 or index_k_cache.shape[1] != head_dim:
            raise ValueError(
                "a bf16 index-K plane must be packed [slots, head_dim] = "
                f"[slots, {head_dim}], got {tuple(index_k_cache.shape)}"
            )
        return {"index_k_format": "bf16", "index_k_layout": "packed"}
    raise TypeError(
        f"index-K plane dtype {index_k_cache.dtype} has no registered format: "
        "uint8 holds FP8 keys with scales (fp8_scaled), bfloat16 holds the keys "
        "unquantized (bf16)"
    )


def _candidate_lens_cpu_kwargs(kernel, candidate_lens_cpu: torch.Tensor | None) -> dict:
    """The ``candidate_lens_cpu`` kwarg for a selected top-k leaf, or nothing.

    Only a leaf registered with ``CANDIDATE_LENS_CPU_FEATURE`` takes the
    keyword; the decision is the registration's, read here, never a signature
    probe on the forward path.
    """
    if candidate_lens_cpu is None:
        return {}
    spec = KernelRegistry.get().get_by_name(kernel.name)
    if spec is None or CANDIDATE_LENS_CPU_FEATURE not in spec.features:
        return {}
    return {"candidate_lens_cpu": candidate_lens_cpu}


# The index-K formats whose rows ``dsa_prefill_topk`` takes in workspace-row
# order, and the keywords carrying them (README, "Index-K plane formats").
_INDEX_K_ROW_KEYWORDS = {
    "fp8_scaled": ("index_k_fp8", "index_k_scale"),
    "bf16": ("index_k_bf16",),
}


def _index_k_rows_traits(index_k_format: str) -> dict:
    """``index_k_format`` / ``index_k_layout`` selection traits of index keys
    handed as rows in workspace-row order: the rows of a plane of that format,
    one per workspace row, so always ``"packed"``."""
    if index_k_format not in _INDEX_K_ROW_KEYWORDS:
        raise ValueError(
            "index keys in workspace-row order come in one of "
            f"{sorted(_INDEX_K_ROW_KEYWORDS)}, got {index_k_format!r}"
        )
    return {"index_k_format": index_k_format, "index_k_layout": "packed"}


def _check_index_k_rows(
    index_k_fp8: torch.Tensor | None,
    index_k_scale: torch.Tensor | None,
    index_k_bf16: torch.Tensor | None,
    *,
    head_dim: int,
    workspace_rows: int,
) -> None:
    """Refuse rows that are not one key per workspace row in their format.

    ``index_k_bf16`` is ``[workspace_rows, head_dim]`` bf16; ``index_k_fp8`` is
    ``[workspace_rows, head_dim]`` FP8 bytes (uint8 or float8_e4m3fn) with
    ``index_k_scale`` ``[workspace_rows, head_dim / 128]`` fp32.
    """
    if index_k_bf16 is not None:
        if index_k_bf16.dtype != torch.bfloat16 or index_k_bf16.shape != (
            workspace_rows,
            head_dim,
        ):
            raise ValueError(
                "index_k_bf16 holds bf16 rows [workspace_rows, head_dim] = "
                f"[{workspace_rows}, {head_dim}], got {index_k_bf16.dtype} "
                f"{tuple(index_k_bf16.shape)}"
            )
        return
    groups = head_dim // _INDEX_K_FP8_GROUP_SIZE
    if index_k_fp8.dtype not in (
        torch.uint8,
        torch.float8_e4m3fn,
    ) or index_k_fp8.shape != (workspace_rows, head_dim):
        raise ValueError(
            "index_k_fp8 holds FP8 rows [workspace_rows, head_dim] = "
            f"[{workspace_rows}, {head_dim}] as uint8 or float8_e4m3fn, got "
            f"{index_k_fp8.dtype} {tuple(index_k_fp8.shape)}"
        )
    if index_k_scale.dtype != torch.float32 or index_k_scale.shape != (
        workspace_rows,
        groups,
    ):
        raise ValueError(
            "index_k_scale holds fp32 scales [workspace_rows, head_dim / "
            f"{_INDEX_K_FP8_GROUP_SIZE}] = [{workspace_rows}, {groups}], got "
            f"{index_k_scale.dtype} {tuple(index_k_scale.shape)}"
        )


def _index_k_rows_kwargs(
    kernel,
    *,
    index_k_fp8: torch.Tensor | None,
    index_k_scale: torch.Tensor | None,
    index_k_bf16: torch.Tensor | None,
) -> dict:
    """The workspace-row keywords for a selected top-k leaf, or nothing.

    Only a leaf registered with ``INDEX_K_WORKSPACE_ROWS_FEATURE`` takes them;
    the decision is the registration's, read here, never a signature probe.
    Selection already required the feature for rows (ranking and overrides
    alike), so a leaf without it here is refused rather than handed a keyword
    its launcher does not take.
    """
    if index_k_fp8 is None and index_k_bf16 is None:
        return {}
    spec = KernelRegistry.get().get_by_name(kernel.name)
    if spec is None or INDEX_K_WORKSPACE_ROWS_FEATURE not in spec.features:
        raise ValueError(
            f"DSA kernel {kernel.name!r} does not declare the "
            f"{INDEX_K_WORKSPACE_ROWS_FEATURE!r} feature: it resolves index keys "
            "from a plane and cannot score rows in workspace-row order"
        )
    if index_k_bf16 is not None:
        return {"index_k_bf16": index_k_bf16}
    return {"index_k_fp8": index_k_fp8, "index_k_scale": index_k_scale}


def select_dsa_prefill_topk_for_rows(
    *,
    index_k_format: str,
    q_dtype: torch.dtype,
    weights_dtype: torch.dtype,
    index_heads: int,
    head_dim: int,
    topk: int,
    page_size: int | None,
    batch_invariant: bool,
    solution: str | None,
) -> str:
    """Select, without running it, the ``dsa_prefill_topk`` leaf for index keys
    handed as rows in workspace-row order of ``index_k_format``.

    The selection :func:`dsa_prefill_topk` makes for ``index_k_fp8`` +
    ``index_k_scale`` (``"fp8_scaled"``) or ``index_k_bf16`` (``"bf16"``):
    the rows' format and layout traits plus ``INDEX_K_WORKSPACE_ROWS_FEATURE``
    required. A host whose sharded prefill will hand such rows (the
    query-context-parallel history gather over page-sharded caches) calls this
    at construction so a platform without a declaring leaf fails at startup
    rather than in the first sharded prefill.

    Args:
        index_k_format: The plane format the rows are packed in, one of
            ``"fp8_scaled"`` or ``"bf16"``.
        q_dtype: dtype of the indexer query the leaf will score.
        weights_dtype: dtype of the per-token/head weights.
        index_heads: Indexer heads, the ``index_heads`` trait.
        head_dim: Key width, the ``head_dim`` trait.
        topk: Candidates selected per row, the ``topk`` trait.
        page_size: KV cache page size, the ``page_size`` trait (a leaf pinned
            to one is not selected without it), or ``None``.
        batch_invariant: Whether the ``batch_invariant`` feature is required
            too, as the forward will require it.
        solution: Restrict selection to a registered solution, as the forward
            will.

    Returns:
        The selected kernel's name.

    Raises:
        NoKernelFoundError: No registered leaf declares the feature for the
            format with these traits on this platform.
    """
    traits = {
        "index_heads": int(index_heads),
        "head_dim": int(head_dim),
        "page_size": None if page_size is None else int(page_size),
        "topk": int(topk),
        **_index_k_rows_traits(index_k_format),
    }
    required_features = {INDEX_K_WORKSPACE_ROWS_FEATURE}
    if batch_invariant:
        required_features.add("batch_invariant")
    return select_kernel(
        "attention",
        "dsa_prefill_topk",
        format_signature(
            q=dense_tensor_format(q_dtype), weights=dense_tensor_format(weights_dtype)
        ),
        traits=traits,
        features=frozenset(required_features),
        solution=solution,
    ).name


def _slot_order_kwargs(kernel, slot_order: str, *, role: str) -> dict:
    """The ``slot_order`` kwarg for a selected leaf or core, refusing what it cannot do.

    A kernel declaring the trait was already matched on it and takes the
    kwarg. A silent one serves ``"selection"`` implicitly -- a top-k leaf
    (``role="top-k"``) emits its own order, a core (``role="core"``) reduces
    the slots as they arrive -- and cannot promise ``"sorted"``.
    """
    if slot_order not in SLOT_ORDERS:
        raise ValueError(
            f"slot_order must be one of {list(SLOT_ORDERS)}, got {slot_order!r}"
        )
    spec = KernelRegistry.get().get_by_name(kernel.name)
    declared = None if spec is None else spec.traits.get("slot_order")
    if declared is None:
        if slot_order == "sorted":
            what = (
                "emits its selection in its own order"
                if role == "top-k"
                else "reduces the selected slots in the top-k leaf's order"
            )
            raise ValueError(
                f"DSA kernel {kernel.name!r} does not declare the slot_order "
                f"trait: it {what} and cannot promise slot_order='sorted'; "
                "select a kernel declaring the trait through solution="
            )
        return {}
    return {"slot_order": slot_order}


# ===-----------------------------------------------------------------------===#
# DSA Kernels
# ===-----------------------------------------------------------------------===#


def dsa_decode(
    q: torch.Tensor,
    kv_cache: torch.Tensor | None,
    sparse_kv_cache: torch.Tensor | None,
    topk_slots: torch.Tensor,
    topk_lens: torch.Tensor | None,
    max_seqlen_k: int,
    qk_nope_head_dim: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    softmax_scale: float,
    page_size: int,
    q_len_per_req: int = 1,
    logit_cap: float = 0.0,
    k_scale: float = 1.0,
    return_lse: bool = False,
    out: torch.Tensor | None = None,
    override: str | None = None,
    solution: str | None = None,
    kv_seq_lens: torch.Tensor | None = None,
    *,
    slot_order: str,
) -> AttentionResult:
    """Sparse DSA decode over selected global KV slots.

    Args:
        q: Absorbed MLA query with shape [tokens, heads, R + D_rope] or
            [batch, q_len, heads, R + D_rope].
        kv_cache: Regular compressed MLA KV cache, flat [slots, dim] or paged.
        sparse_kv_cache: Packed sparse DSA KV cache, flat [slots, row_bytes] or
            paged.
        topk_slots: Global KV slot ids with shape [tokens, topk]. Invalid
            entries are -1.
        topk_lens: Valid selected-slot count per token, or None when the
            implementation relies on -1 padding.
        max_seqlen_k: Maximum dense visible context length for this batch.
        qk_nope_head_dim: Original non-RoPE q/k dimension.
        kv_lora_rank: MLA latent rank and output head dimension.
        qk_rope_head_dim: RoPE q/k dimension.
        softmax_scale: Scale applied to attention logits.
        page_size: KV cache page size.
        q_len_per_req: Query rows per request.
        kv_seq_lens: Optional physical KV length for every query row. Sparse
            backends that gather direct global slots use this separately from
            ``topk_lens``, which describes the selected sparse width.
        logit_cap: Optional logit cap.
        k_scale: KV scale multiplier for FP8 backends.
        return_lse: Whether to return LSE in addition to output.
        out: Optional output buffer.
        override: Optional exact kernel override name.
        solution: Optional kernel solution to force through normal selection.
        slot_order: The order the selected slots are reduced in, one of
            ``SLOT_ORDERS``: ``"selection"`` (as the top-k leaf emitted them;
            every kernel) or ``"sorted"`` (the ascending position order the
            top-k leaf emitted under the same ``slot_order``; only cores
            declaring the ``slot_order`` trait, which receive it as a keyword
            and promise to reduce in the emitted order rather than by slot).
            Required keyword.

    Returns:
        Latent DSA attention output, or ``(out, lse)`` when ``return_lse=True``.
        Partials come back in the query dtype (each context shard is rounded
        before aggregation); only the LSE stays FP32 for the cross-shard merge.
    """
    if slot_order not in SLOT_ORDERS:
        raise ValueError(
            f"slot_order must be one of {list(SLOT_ORDERS)}, got {slot_order!r}"
        )
    if q.dim() == 4:
        batch_size, q_len, num_heads, head_dim = q.shape
        tokens = batch_size * q_len
    else:
        tokens, num_heads, head_dim = q.shape
        q_len = int(q_len_per_req)
        batch_size = tokens // q_len

    traits = {
        "q_len": int(q_len_per_req),
        "qk_nope_head_dim": int(qk_nope_head_dim),
        "kv_lora_rank": int(kv_lora_rank),
        "qk_rope_head_dim": int(qk_rope_head_dim),
        "page_size": int(page_size),
        "topk": int(topk_slots.shape[-1]),
        "has_kv_cache": kv_cache is not None,
        "has_sparse_kv_cache": sparse_kv_cache is not None,
        "logit_cap": logit_cap != 0.0,
        "return_lse": return_lse,
        "topk_layout": "global_slots",
        "slot_order": slot_order,
    }
    signature = _attention_format_signature(q=q)
    kernel = select_kernel(
        "attention",
        "dsa_decode",
        signature,
        traits=traits,
        solution=solution,
        override=override,
    )
    slot_order_kwargs = _slot_order_kwargs(kernel, slot_order, role="core")
    shape_params = {
        "batch_size": batch_size,
        "q_len": q_len,
        "tokens": tokens,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "topk": topk_slots.shape[-1],
        "page_size": int(page_size),
        "max_seqlen_k": int(max_seqlen_k),
    }
    ShapeCapture.get().record(
        "attention", "dsa_decode", kernel.name, q.dtype, shape_params
    )
    with kernel_scope(
        "attention", "dsa_decode", q.dtype, kernel_name=kernel.name, **shape_params
    ):
        return kernel(
            q=q,
            kv_cache=kv_cache,
            sparse_kv_cache=sparse_kv_cache,
            topk_slots=topk_slots,
            topk_lens=topk_lens,
            max_seqlen_k=max_seqlen_k,
            qk_nope_head_dim=qk_nope_head_dim,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            softmax_scale=softmax_scale,
            page_size=page_size,
            q_len_per_req=q_len_per_req,
            kv_seq_lens=kv_seq_lens,
            logit_cap=logit_cap,
            k_scale=k_scale,
            return_lse=return_lse,
            out=out,
            enable_pdl=pdl_enabled(),
            **slot_order_kwargs,
        )


def dsa_prefill(
    q: torch.Tensor,
    kv_cache: torch.Tensor | None,
    sparse_kv_cache: torch.Tensor | None,
    topk_slots: torch.Tensor,
    topk_lens: torch.Tensor,
    max_seqlen_k: int,
    qk_nope_head_dim: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    softmax_scale: float,
    page_size: int,
    logit_cap: float = 0.0,
    k_scale: float = 1.0,
    return_lse: bool = False,
    out: torch.Tensor | None = None,
    override: str | None = None,
    solution: str | None = None,
    kv_seq_lens: torch.Tensor | None = None,
    *,
    slot_order: str,
) -> AttentionResult:
    """Sparse DSA prefill over selected global KV slots.

    Args:
        q: Absorbed MLA query with shape [tokens, heads, R + D_rope] or
            [batch, q_len, heads, R + D_rope].
        kv_cache: Regular compressed MLA KV cache, flat [slots, dim] or paged.
        sparse_kv_cache: Packed sparse DSA KV cache, flat [slots, row_bytes] or
            paged.
        topk_slots: Global KV slot ids with shape [tokens, topk]. Invalid
            entries are -1.
        topk_lens: Valid selected-slot count per token.
        max_seqlen_k: Maximum dense visible context length for this batch.
        qk_nope_head_dim: Original non-RoPE q/k dimension.
        kv_lora_rank: MLA latent rank and output head dimension.
        qk_rope_head_dim: RoPE q/k dimension.
        softmax_scale: Scale applied to attention logits.
        page_size: KV cache page size.
        kv_seq_lens: Optional physical KV length for every query row. Sparse
            backends that gather direct global slots use this separately from
            ``topk_lens``, which describes the selected sparse width.
        logit_cap: Optional logit cap.
        k_scale: KV scale multiplier for FP8 backends.
        return_lse: Whether to return LSE in addition to output.
        out: Optional output buffer.
        override: Optional exact kernel override name.
        solution: Optional kernel solution to force through normal selection.
        slot_order: The order the selected slots are reduced in, as for
            :func:`dsa_decode`. Required keyword.

    Returns:
        Latent DSA attention output, or ``(out, lse)`` when ``return_lse=True``.
    """
    if slot_order not in SLOT_ORDERS:
        raise ValueError(
            f"slot_order must be one of {list(SLOT_ORDERS)}, got {slot_order!r}"
        )
    if q.dim() == 4:
        batch_size, q_len, num_heads, head_dim = q.shape
        tokens = batch_size * q_len
    else:
        tokens, num_heads, head_dim = q.shape
        q_len = 1
        batch_size = tokens

    traits = {
        "q_len": 1,
        "qk_nope_head_dim": int(qk_nope_head_dim),
        "kv_lora_rank": int(kv_lora_rank),
        "qk_rope_head_dim": int(qk_rope_head_dim),
        "page_size": int(page_size),
        "topk": int(topk_slots.shape[-1]),
        "has_kv_cache": kv_cache is not None,
        "has_sparse_kv_cache": sparse_kv_cache is not None,
        "logit_cap": logit_cap != 0.0,
        "return_lse": return_lse,
        "topk_layout": "global_slots",
        "slot_order": slot_order,
    }
    signature = _attention_format_signature(q=q)
    kernel = select_kernel(
        "attention",
        "dsa_prefill",
        signature,
        traits=traits,
        solution=solution,
        override=override,
    )
    slot_order_kwargs = _slot_order_kwargs(kernel, slot_order, role="core")
    shape_params = {
        "batch_size": batch_size,
        "q_len": q_len,
        "tokens": tokens,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "topk": topk_slots.shape[-1],
        "page_size": int(page_size),
        "max_seqlen_k": int(max_seqlen_k),
    }
    ShapeCapture.get().record(
        "attention", "dsa_prefill", kernel.name, q.dtype, shape_params
    )
    with kernel_scope(
        "attention", "dsa_prefill", q.dtype, kernel_name=kernel.name, **shape_params
    ):
        return kernel(
            q=q,
            kv_cache=kv_cache,
            sparse_kv_cache=sparse_kv_cache,
            topk_slots=topk_slots,
            topk_lens=topk_lens,
            max_seqlen_k=max_seqlen_k,
            qk_nope_head_dim=qk_nope_head_dim,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            softmax_scale=softmax_scale,
            page_size=page_size,
            q_len_per_req=1,
            kv_seq_lens=kv_seq_lens,
            logit_cap=logit_cap,
            k_scale=k_scale,
            return_lse=return_lse,
            out=out,
            enable_pdl=pdl_enabled(),
            **slot_order_kwargs,
        )


def dsa_index_candidates(
    q: torch.Tensor,
    weights: torch.Tensor,
    index_k_cache: torch.Tensor,
    local_page_table: torch.Tensor,
    query_requests: torch.Tensor,
    causal_lens: torch.Tensor,
    *,
    page_size: int,
    topk: int,
    softmax_scale: float,
    initial_tokens: int,
    local_tokens: int,
    solution: str | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score a query tile against owned Index-K pages and select local candidates.

    Query rows carry request IDs and global causal lengths. Page-table columns
    retain global request order; absent pages are -1. Windows are measured in
    global token positions. Returns global logical offsets and FP32 scores;
    invalid candidates use (-1, -inf), mandatory candidates use +inf scores.
    ``solution=None`` selects the best supported kernel, or force a solution.
    """
    if topk <= 0 or topk & (topk - 1):
        raise ValueError("Index candidate topk must be a positive power of two")
    if q.ndim != 3 or weights.shape != q.shape[:2]:
        raise ValueError("Index queries and per-head weights must match")
    query_requests = query_requests.contiguous()
    causal_lens = causal_lens.contiguous()
    if index_k_cache.dtype != torch.uint8:
        raise TypeError("DSA index candidates require packed FP8 Index-K storage")
    if initial_tokens < 0 or local_tokens < 0 or initial_tokens + local_tokens > topk:
        raise ValueError("Forced windows must fit in topk")
    if query_requests.shape != (q.shape[0],) or causal_lens.shape != (q.shape[0],):
        raise ValueError("Index candidate rows must match queries")
    if not local_page_table.shape[0] or not local_page_table.shape[1]:
        raise ValueError("Index candidates require a nonempty page table")
    if not q.shape[0]:
        return (
            torch.empty((0, topk), device=q.device, dtype=torch.int32),
            torch.empty((0, topk), device=q.device, dtype=torch.float32),
        )
    kernel = select_kernel(
        "attention",
        "dsa_index_candidates",
        _attention_format_signature(q=q, weights=weights),
        traits={
            "index_heads": q.shape[1],
            "head_dim": q.shape[2],
            "page_size": page_size,
            "index_k_layout": (
                "packed"
                if (
                    index_k_cache.ndim == 2
                    and index_k_cache.shape[1] == q.shape[-1] + q.shape[-1] // 128 * 4
                )
                else "page_planar"
            ),
        },
        solution=solution,
    )
    with kernel_scope(
        "attention", "dsa_index_candidates", q.dtype, kernel_name=kernel.name
    ):
        return kernel(
            q,
            weights,
            index_k_cache,
            local_page_table,
            query_requests,
            causal_lens,
            page_size=page_size,
            topk=topk,
            softmax_scale=softmax_scale,
            initial_tokens=initial_tokens,
            local_tokens=local_tokens,
        )


def dsa_prefill_topk(
    q: torch.Tensor,
    weights: torch.Tensor,
    kv_workspace_slots: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    *,
    topk: int,
    softmax_scale: float,
    batch_invariant: bool,
    index_k_cache: torch.Tensor | None = None,
    page_size: int | None = None,
    index_k_fp8: torch.Tensor | None = None,
    index_k_scale: torch.Tensor | None = None,
    index_k_bf16: torch.Tensor | None = None,
    q_scales: torch.Tensor | None = None,
    max_logits_bytes: int | None = None,
    candidate_lens_cpu: torch.Tensor | None = None,
    initial_tokens: int = 0,
    local_tokens: int = 0,
    out: torch.Tensor | None = None,
    lens_out: torch.Tensor | None = None,
    slot_order: str,
    override: str | None = None,
    solution: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute DSA prefill top-k over packed workspace rows.

    Args:
        q: BF16 or FP8 E4M3 indexer query with shape
            [tokens, index_heads, head_dim]. FP8 queries require q_scales.
        weights: Per-token/head weights with shape [tokens, index_heads],
            FP32 or raw BF16 (implementations upcast on the fly).
        kv_workspace_slots: Global KV slot for each workspace row, shape
            [workspace_rows].
        row_starts: Inclusive workspace-row start per query token, shape [tokens].
        row_ends: Exclusive workspace-row end per query token, shape [tokens].
        topk: Number of workspace candidates to select.
        softmax_scale: Score scale. Each candidate score is exactly
            ``softmax_scale * sum_h(weights[h] * relu(dot(dequant(q[h]), dequant(k))))``.
            BF16 queries are already in their compute representation.
        batch_invariant: Require the selection to be a function of each
            token's own row only: equal scores resolve toward the lowest
            candidate, whatever else the batch holds. Only implementations
            declaring the ``batch_invariant`` feature are eligible.
        index_k_cache: Index-K plane, used with kv_workspace_slots to resolve
            workspace rows inside the selected implementation. Its dtype
            selects the ``index_k_format`` trait: uint8 is FP8 with scales
            (``"fp8_scaled"``, packed or page-planar; page-planar caches may
            have a padded outer page stride), bfloat16 is the unquantized
            ``[slots, head_dim]`` plane (``"bf16"``, packed). The index keys
            come either as this plane or as rows in workspace-row order
            (below), exactly one of the two.
        page_size: KV cache page size for index_k_cache. Passed to selection
            as the ``page_size`` trait whenever given, rows or plane: a leaf
            pinned to a page size is not selected without it.
        index_k_fp8: FP8 index-K rows already in workspace-row order
            (``[workspace_rows, head_dim]`` uint8 or float8_e4m3fn, the
            gathered form of an ``fp8_scaled`` plane). Must be provided
            together with index_k_scale and instead of index_k_cache.
        index_k_scale: FP8 index-K scales already in workspace-row order
            (``[workspace_rows, head_dim / 128]`` fp32). Must be provided
            together with index_k_fp8.
        index_k_bf16: bf16 index-K rows already in workspace-row order
            (``[workspace_rows, head_dim]``, the gathered form of a ``bf16``
            plane), instead of index_k_cache and of the FP8 pair. Rows in
            workspace-row order, of either format, select only leaves
            declaring ``INDEX_K_WORKSPACE_ROWS_FEATURE`` for the matching
            ``index_k_format`` and reach only such a leaf, as the
            ``index_k_fp8`` + ``index_k_scale`` or ``index_k_bf16`` keyword
            (:func:`select_dsa_prefill_topk_for_rows` probes the selection).
        q_scales: Optional positive FP32 scale per token/head for FP8 queries,
            defining ``dequant(q[token, head]) = q[token, head].float() *
            q_scales[token, head]``.
        max_logits_bytes: Optional temporary logits memory cap.
        candidate_lens_cpu: Optional CPU mirror of ``row_ends - row_starts``,
            handed to every implementation registered with the
            ``candidate_lens_cpu`` feature (DeepGEMM sizes its chunk launches
            from it without synchronizing the CUDA stream); implementations
            without the feature never see it.
        out: Optional contiguous int32 output buffer on q's device with shape
            [tokens, topk].
        lens_out: Optional contiguous int32 output buffer on q's device with
            shape [tokens].
        slot_order: The order the leaf emits each row's selection in, one of
            ``SLOT_ORDERS``: ``"selection"`` (the leaf's own layout and tie
            order; every leaf) or ``"sorted"`` (ascending position order; only
            leaves declaring the ``slot_order`` trait, which receive it as a
            keyword -- the sparse core then reduces in that order). Required
            keyword.
        override: Optional exact kernel override name.
        solution: Optional kernel solution to force through normal selection.

    Implementations may accept a strided outer weight dimension, including the
    fused model projection view. q, kv_workspace_slots, row_starts, and row_ends
    must be contiguous on q's device. index_k_cache may be a contiguous packed
    slot matrix or a page-planar matrix with contiguous bytes within each page.
    kv_workspace_slots must be int64; row_starts and row_ends must be int32.

    Returns:
        Tuple of workspace row ids and valid counts. Returned indices are
        absolute row ids into kv_workspace_slots; invalid entries are -1.
    """
    if candidate_lens_cpu is not None and (
        candidate_lens_cpu.device.type != "cpu"
        or candidate_lens_cpu.shape != (q.shape[0],)
    ):
        raise ValueError(
            "candidate_lens_cpu must be a CPU tensor with shape "
            f"{(q.shape[0],)}, got device={candidate_lens_cpu.device}, "
            f"shape={tuple(candidate_lens_cpu.shape)}"
        )
    if out is not None and out.shape != (q.shape[0], int(topk)):
        raise ValueError(
            f"out must have shape {(q.shape[0], int(topk))}, got {tuple(out.shape)}"
        )
    if lens_out is not None and lens_out.shape != (q.shape[0],):
        raise ValueError(
            f"lens_out must have shape {(q.shape[0],)}, got {tuple(lens_out.shape)}"
        )
    traits = {
        "index_heads": q.shape[1],
        "head_dim": q.shape[-1],
        "page_size": None if page_size is None else int(page_size),
        "topk": int(topk),
    }
    if (index_k_fp8 is None) != (index_k_scale is None):
        raise ValueError(
            "index_k_fp8 and index_k_scale must be provided together for "
            "workspace-row input"
        )
    has_fp8_rows = index_k_fp8 is not None
    has_bf16_rows = index_k_bf16 is not None
    if has_fp8_rows and has_bf16_rows:
        raise ValueError(
            "index_k_fp8/index_k_scale and index_k_bf16 are the workspace rows "
            "of two index-K formats; pass the plane's one"
        )
    has_workspace_rows = has_fp8_rows or has_bf16_rows
    if has_workspace_rows and index_k_cache is not None:
        raise ValueError(
            "index_k_cache and workspace rows (index_k_fp8/index_k_scale or "
            "index_k_bf16) are two sources of index keys; pass one"
        )
    if index_k_cache is not None:
        traits.update(_index_k_plane_traits(index_k_cache, q.shape[-1]))
    elif has_workspace_rows:
        _check_index_k_rows(
            index_k_fp8,
            index_k_scale,
            index_k_bf16,
            head_dim=q.shape[-1],
            workspace_rows=kv_workspace_slots.numel(),
        )
        traits.update(_index_k_rows_traits("bf16" if has_bf16_rows else "fp8_scaled"))
    else:
        raise ValueError(
            "dsa_prefill_topk needs its index keys: a plane (index_k_cache) or "
            "rows in workspace-row order (index_k_fp8 + index_k_scale, or "
            "index_k_bf16)"
        )
    initial_tokens = int(initial_tokens)
    local_tokens = int(local_tokens)
    if initial_tokens < 0 or local_tokens < 0:
        raise ValueError("initial_tokens and local_tokens must be non-negative")
    if initial_tokens + local_tokens > int(topk):
        raise ValueError(
            "initial_tokens + local_tokens must not exceed topk; got "
            f"{initial_tokens} + {local_tokens} > {int(topk)}"
        )
    required_features = set()
    if initial_tokens or local_tokens:
        required_features.add("forced_initial_local")
    if batch_invariant:
        required_features.add("batch_invariant")
    if has_workspace_rows:
        required_features.add(INDEX_K_WORKSPACE_ROWS_FEATURE)
    signature = _attention_format_signature(q=q, weights=weights)
    kernel = select_kernel(
        "attention",
        "dsa_prefill_topk",
        signature,
        traits=traits,
        features=frozenset(required_features) if required_features else None,
        solution=solution,
        override=override,
    )
    candidate_lens_cpu_kwargs = _candidate_lens_cpu_kwargs(kernel, candidate_lens_cpu)
    index_k_rows_kwargs = _index_k_rows_kwargs(
        kernel,
        index_k_fp8=index_k_fp8,
        index_k_scale=index_k_scale,
        index_k_bf16=index_k_bf16,
    )
    slot_order_kwargs = _slot_order_kwargs(kernel, slot_order, role="top-k")
    shape_params = {
        "tokens": q.shape[0],
        "workspace_rows": kv_workspace_slots.numel(),
        "index_heads": q.shape[1],
        "head_dim": q.shape[-1],
        "topk": int(topk),
    }
    ShapeCapture.get().record(
        "attention", "dsa_prefill_topk", kernel.name, q.dtype, shape_params
    )
    with kernel_scope(
        "attention",
        "dsa_prefill_topk",
        q.dtype,
        kernel_name=kernel.name,
        **shape_params,
    ):
        kernel_kwargs = {
            "q": q,
            "weights": weights,
            "kv_workspace_slots": kv_workspace_slots,
            "row_starts": row_starts,
            "row_ends": row_ends,
            "topk": topk,
            "softmax_scale": softmax_scale,
            "index_k_cache": index_k_cache,
            "page_size": page_size,
            "max_logits_bytes": max_logits_bytes,
            "out": out,
            "lens_out": lens_out,
            **candidate_lens_cpu_kwargs,
            **index_k_rows_kwargs,
            **slot_order_kwargs,
        }
        if q_scales is not None:
            kernel_kwargs["q_scales"] = q_scales
        if initial_tokens or local_tokens:
            kernel_kwargs["initial_tokens"] = initial_tokens
            kernel_kwargs["local_tokens"] = local_tokens
        if batch_invariant:
            kernel_kwargs["batch_invariant"] = True
        return kernel(**kernel_kwargs)


def dsa_decode_topk(
    q: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    *,
    page_size: int,
    topk: int,
    softmax_scale: float,
    batch_invariant: bool,
    q_len_per_req: int = 1,
    topk_layout: str = "global_slots",
    block_table_base_offsets: torch.Tensor | None = None,
    index_k_cache: torch.Tensor | None = None,
    q_scales: torch.Tensor | None = None,
    seq_lens_2d: torch.Tensor | None = None,
    plan: object | None = None,
    initial_tokens: int = 0,
    local_tokens: int = 0,
    out: torch.Tensor | None = None,
    lens_out: torch.Tensor | None = None,
    slot_order: str,
    override: str | None = None,
    solution: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute DSA decode top-k over a paged KV cache.

    Args:
        q: BF16 or FP8 E4M3 indexer query with shape
            [tokens, index_heads, head_dim]. FP8 queries require q_scales.
        weights: Per-token/head weights with shape [tokens, index_heads],
            FP32 or raw BF16 (implementations upcast on the fly).
        seq_lens: Per-request full KV length, shape [num_reqs] (= tokens /
            q_len_per_req). Each query token's causal bound
            seq_lens[req] - (q_len_per_req - 1) + j is derived in-kernel.
        block_table: Paged KV block table with one row per request,
            shape [num_reqs, max_pages].
        page_size: Number of tokens per KV page.
        topk: Number of KV candidates to select.
        softmax_scale: Score scale. Each candidate score is exactly
            ``softmax_scale * sum_h(weights[h] * relu(dot(dequant(q[h]), dequant(k))))``.
            BF16 queries are already in their compute representation.
        batch_invariant: Require the selection to be a function of each
            token's own row only: equal scores resolve toward the lowest
            candidate, whatever else the batch holds. Only implementations
            declaring the ``batch_invariant`` feature are eligible.
        q_len_per_req: Query rows per request (spec-verify next_n). Plain
            decode uses 1, where per-request is equivalent to per-token.
        topk_layout: Return physical cache slots when ``global_slots`` or
            absolute logical row offsets when ``logical_offsets``.
        block_table_base_offsets: Optional compact-table base page per request.
            Used only with ``topk_layout="logical_offsets"``.
        index_k_cache: Index-K plane. Its dtype selects the ``index_k_format``
            trait: uint8 is FP8 with scales (``"fp8_scaled"``, packed or
            page-planar; page-planar caches may have a padded outer page
            stride), bfloat16 is the unquantized ``[slots, head_dim]`` plane
            (``"bf16"``, packed).
        q_scales: Optional positive FP32 scale per token/head for FP8 queries,
            defining ``dequant(q[token, head]) = q[token, head].float() *
            q_scales[token, head]``.
        seq_lens_2d: Optional per-token rows of the request's full KV length
            (``[tokens, 1]``, every row of a request carrying ``seq_lens[req]``),
            the scoring extent the ``plan`` was built from (:func:`dsa_plan`).
            It carries no per-row causal bound: a leaf derives that from
            ``seq_lens`` and ``q_len_per_req`` as under ``seq_lens`` above,
            never from these rows, so a verify or draft row never selects its
            window's later rows.
        plan: Optional opaque backend-specific plan.
        out: Optional contiguous int32 output buffer on q's device with shape
            [tokens, topk].
        lens_out: Optional contiguous int32 output buffer on q's device with
            shape [tokens].
        slot_order: The order the leaf emits each row's selection in, as for
            :func:`dsa_prefill_topk`. Required keyword.
        override: Optional exact kernel override name.
        solution: Optional kernel solution to force through normal selection.

    Implementations may accept a strided outer weight dimension, including the
    fused model projection view. q, seq_lens, and block_table must be contiguous
    on q's device. index_k_cache may be a contiguous packed slot matrix or a
    page-planar matrix with contiguous bytes within each page. seq_lens and
    block_table must be int32.

    Returns:
        Tuple of selected indices and valid counts. Indices are global KV slots
        or absolute logical offsets according to ``topk_layout``; invalid
        entries are -1.
    """
    if out is not None and out.shape != (q.shape[0], int(topk)):
        raise ValueError(
            f"out must have shape {(q.shape[0], int(topk))}, got {tuple(out.shape)}"
        )
    if topk_layout not in ("global_slots", "logical_offsets"):
        raise ValueError(
            "topk_layout must be 'global_slots' or 'logical_offsets', got "
            f"{topk_layout!r}"
        )
    if q_len_per_req < 1 or q.shape[0] % int(q_len_per_req) != 0:
        raise ValueError(
            f"q_len_per_req={q_len_per_req} must divide tokens={q.shape[0]}"
        )
    if block_table_base_offsets is not None and topk_layout != "logical_offsets":
        raise ValueError(
            "block_table_base_offsets requires topk_layout='logical_offsets'"
        )
    kernel_seq_lens = seq_lens
    if block_table_base_offsets is not None:
        num_reqs = q.shape[0] // int(q_len_per_req)
        if (
            block_table_base_offsets.ndim != 1
            or block_table_base_offsets.numel() < num_reqs
            or block_table_base_offsets.device != seq_lens.device
        ):
            raise ValueError(
                "block_table_base_offsets must have one entry per request on "
                "the same device as seq_lens"
            )
        kernel_seq_lens = (
            (
                seq_lens.to(torch.int64)
                - block_table_base_offsets[:num_reqs].to(torch.int64) * int(page_size)
            )
            .clamp(0, int(block_table.shape[1]) * int(page_size))
            .to(torch.int32)
        )
    if lens_out is not None and lens_out.shape != (q.shape[0],):
        raise ValueError(
            f"lens_out must have shape {(q.shape[0],)}, got {tuple(lens_out.shape)}"
        )
    traits = {
        "q_len": int(q_len_per_req),
        "index_heads": q.shape[1],
        "head_dim": q.shape[-1],
        "page_size": int(page_size),
        "topk": int(topk),
    }
    if index_k_cache is not None:
        traits.update(_index_k_plane_traits(index_k_cache, q.shape[-1]))
    initial_tokens = int(initial_tokens)
    local_tokens = int(local_tokens)
    if initial_tokens < 0 or local_tokens < 0:
        raise ValueError("initial_tokens and local_tokens must be non-negative")
    if initial_tokens + local_tokens > int(topk):
        raise ValueError(
            "initial_tokens + local_tokens must not exceed topk; got "
            f"{initial_tokens} + {local_tokens} > {int(topk)}"
        )
    required_features = set()
    if topk_layout == "logical_offsets":
        required_features.add("logical_offsets")
    if initial_tokens or local_tokens:
        required_features.add("forced_initial_local")
    if batch_invariant:
        required_features.add("batch_invariant")
    signature = _attention_format_signature(q=q, weights=weights)
    kernel = select_kernel(
        "attention",
        "dsa_decode_topk",
        signature,
        traits=traits,
        features=frozenset(required_features) if required_features else None,
        solution=solution,
        override=override,
    )
    slot_order_kwargs = _slot_order_kwargs(kernel, slot_order, role="top-k")
    shape_params = {
        "tokens": q.shape[0],
        "max_pages": block_table.shape[1],
        "index_heads": q.shape[1],
        "head_dim": q.shape[-1],
        "page_size": int(page_size),
        "topk": int(topk),
        "q_len_per_req": int(q_len_per_req),
    }
    ShapeCapture.get().record(
        "attention", "dsa_decode_topk", kernel.name, q.dtype, shape_params
    )
    with kernel_scope(
        "attention",
        "dsa_decode_topk",
        q.dtype,
        kernel_name=kernel.name,
        **shape_params,
    ):
        kernel_kwargs = {
            "q": q,
            "weights": weights,
            "seq_lens": kernel_seq_lens,
            "block_table": block_table,
            "page_size": page_size,
            "topk": topk,
            "softmax_scale": softmax_scale,
            "q_len_per_req": q_len_per_req,
            "index_k_cache": index_k_cache,
            "seq_lens_2d": seq_lens_2d,
            "plan": plan,
            "out": out,
            "lens_out": lens_out,
            **slot_order_kwargs,
        }
        if topk_layout == "logical_offsets":
            kernel_kwargs["topk_layout"] = topk_layout
            kernel_kwargs["block_table_base_offsets"] = block_table_base_offsets
        if q_scales is not None:
            kernel_kwargs["q_scales"] = q_scales
        if initial_tokens or local_tokens:
            kernel_kwargs["initial_tokens"] = initial_tokens
            kernel_kwargs["local_tokens"] = local_tokens
        if batch_invariant:
            kernel_kwargs["batch_invariant"] = True
        return kernel(**kernel_kwargs)


def dsa_plan(
    *,
    page_size: int,
    seq_lens_2d: torch.Tensor,
    out: object | None = None,
    override: str | None = None,
    solution: str | None = None,
) -> object | None:
    """Build or refresh an opaque plan for DSA decode top-k.

    Args:
        page_size: KV cache page size.
        seq_lens_2d: Prebuilt [num_reqs, next_n] context_lens (last column =
            full per-request KV length), built once per forward by the caller.
        out: Optional previously allocated plan object to refresh in place.
        override: Optional exact kernel override name.
        solution: Optional kernel solution to force through normal selection.

    Returns:
        Opaque backend-owned plan object, or None when no selected backend needs
        an explicit plan.
    """
    if seq_lens_2d.dtype != torch.int32:
        seq_lens_2d = seq_lens_2d.to(torch.int32)
    traits = {"page_size": int(page_size)}
    try:
        kernel = select_kernel(
            "attention",
            "dsa_plan",
            format_signature(),
            traits=traits,
            solution=solution,
            override=override,
        )
    except NoKernelFoundError:
        return None

    shape_params = {
        "batch_size": int(seq_lens_2d.shape[0]),
        "tokens": int(seq_lens_2d.numel()),
        "page_size": int(page_size),
    }
    ShapeCapture.get().record(
        "attention", "dsa_plan", kernel.name, seq_lens_2d.dtype, shape_params
    )
    with kernel_scope(
        "attention",
        "dsa_plan",
        seq_lens_2d.dtype,
        kernel_name=kernel.name,
        **shape_params,
    ):
        return kernel(
            seq_lens_2d=seq_lens_2d,
            page_size=page_size,
            out=out,
        )


# Backend registration (side-effect imports)
# isort: off
import tokenspeed_kernel.ops.attention.dsa.cuda  # noqa: E402,F401
import tokenspeed_kernel.ops.attention.dsa.cute_dsl  # noqa: E402,F401
import tokenspeed_kernel.ops.attention.dsa.deep_gemm  # noqa: E402,F401
import tokenspeed_kernel.ops.attention.dsa.flashinfer  # noqa: E402,F401
import tokenspeed_kernel.ops.attention.dsa.triton  # noqa: E402,F401
import tokenspeed_kernel.ops.attention.dsa.gluon  # noqa: E402,F401

# isort: on

__all__ = [
    "CANDIDATE_LENS_CPU_FEATURE",
    "INDEX_K_WORKSPACE_ROWS_FEATURE",
    "SLOT_ORDERS",
    "dsa_decode",
    "dsa_prefill",
    "dsa_prefill_topk",
    "dsa_decode_topk",
    "dsa_plan",
    "select_dsa_prefill_topk_for_rows",
]

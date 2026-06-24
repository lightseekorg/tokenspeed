"""MHA attention kernels for Ascend NPU via torch_npu."""

from __future__ import annotations

import math

import torch
import torch_npu  # noqa: F401

_CAUSAL_MASK_CACHE: torch.Tensor | None = None


def _get_causal_mask(device: torch.device) -> torch.Tensor:
    """Return a cached [2048, 2048] int8 upper-triangular causal mask."""
    global _CAUSAL_MASK_CACHE
    if _CAUSAL_MASK_CACHE is None or _CAUSAL_MASK_CACHE.device != device:
        _CAUSAL_MASK_CACHE = (
            torch.triu(torch.ones(2048, 2048), diagonal=1)
            .to(torch.int8)
            .to(device)
        )
    return _CAUSAL_MASK_CACHE


def torch_npu_mha_prefill(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_cpu: list[int],
    max_seqlen: int,
    window_left: int = -1,
    logit_cap: float = 0.0,
    sinks: torch.Tensor | None = None,
    return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
    """MHA prefill via npu_fused_infer_attention_score.

    Args:
        q: [total_q, num_q_heads, head_dim]
        k: [total_kv, num_kv_heads, head_dim]
        v: [total_kv, num_kv_heads, head_dim]
        cu_seqlens: [batch + 1]
        cu_seqlens_cpu: host-side cumulative sequence lengths
        max_seqlen: maximum sequence length
        window_left: sliding window size, -1 for full attention
        logit_cap: soft cap on logits (unused on Ascend)
        sinks: attention sinks (unused)
        return_lse: whether to return log-sum-exp
    """
    num_q_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    head_dim = q.shape[-1]
    scale = 1.0 / math.sqrt(head_dim)

    actual_seq_lengths_q = cu_seqlens_cpu[1:]
    actual_seq_lengths_kv = cu_seqlens_cpu[1:]

    fia_kwargs: dict = dict(
        block_table=None,
        input_layout="TND",
        block_size=0,
        actual_seq_lengths=actual_seq_lengths_q,
        actual_seq_lengths_kv=actual_seq_lengths_kv,
        num_key_value_heads=num_kv_heads,
        num_heads=num_q_heads,
        scale=scale,
        softmax_lse_flag=return_lse,
    )

    if window_left >= 0:
        fia_kwargs.update(
            atten_mask=_get_causal_mask(q.device),
            sparse_mode=4,
            pre_tokens=window_left,
            next_tokens=0,
        )
    else:
        fia_kwargs.update(
            atten_mask=_get_causal_mask(q.device),
            sparse_mode=3,
        )

    attn_output, softmax_lse = torch_npu.npu_fused_infer_attention_score(
        q, k, v, **fia_kwargs
    )
    if attn_output.dtype != q.dtype:
        attn_output = attn_output.to(q.dtype)

    if return_lse:
        return attn_output, softmax_lse
    return attn_output


# Thread-local to pass seq_lens from the last eager call into the
# capture path.  Set during warmup (eager) so the capture path can
# read it without calling .tolist() inside torch.npu.graph().
_last_eager_seq_lens_kv: list[int] | None = None


def torch_npu_mha_decode_with_kvcache(
    *,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    max_seqlen_k: int,
    max_seqlen_q: int = 1,
    window_left: int = -1,
    logit_cap: float = 0.0,
    sinks: torch.Tensor | None = None,
    return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
    """MHA decode with paged KV cache via npu_fused_infer_attention_score.

    When called inside an NPU graph capture, the FIA call is wrapped in
    ``graph_task_group_begin/end`` so that ``graph_task_update`` can
    refresh the tiling before each replay.  ``actual_seq_lengths_kv`` is
    a Python int list that gets baked into the capture; on replay the
    stale value is used by FIA.  The ``graph_task_update`` mechanism
    allows us to re-invoke FIA with the **current** seq_lens, which
    updates the tiling *and* the internal seq-length state for the
    captured task group.

    Args:
        q: [batch * max_seqlen_q, num_q_heads, head_dim]
        k_cache: [num_pages, page_size, num_kv_heads, head_dim]
        v_cache: [num_pages, page_size, num_kv_heads, head_dim]
        page_table: [batch, max_pages_per_seq]
        cache_seqlens: [batch]
        max_seqlen_k: maximum KV length
        max_seqlen_q: query tokens per request (1 for normal decode)
        window_left: sliding window size, -1 for full attention
        logit_cap: soft cap on logits (unused on Ascend)
        sinks: attention sinks (unused)
        return_lse: whether to return log-sum-exp
    """
    from tokenspeed.runtime.execution.cuda_graph_wrapper import (
        get_is_capture_mode,
    )
    from tokenspeed_kernel_ascend.execution import get_npu_graph_task_params

    global _last_eager_seq_lens_kv

    batch_size = cache_seqlens.shape[0] if hasattr(cache_seqlens, "shape") else len(cache_seqlens)
    num_q_heads = q.shape[1]
    num_kv_heads = k_cache.shape[2]
    head_dim = q.shape[-1]
    page_size = k_cache.shape[1]
    scale = 1.0 / math.sqrt(head_dim)

    actual_seq_lengths_q = list(
        range(max_seqlen_q, batch_size * max_seqlen_q + 1, max_seqlen_q)
    )

    # Reshape to Ascend's expected layout: [num_blocks, block_size, num_kv_heads * head_dim]
    num_blocks = k_cache.shape[0]
    k_for_fia = k_cache.view(num_blocks, page_size, -1)
    v_for_fia = v_cache.view(num_blocks, page_size, -1)

    is_capturing = get_is_capture_mode()

    if is_capturing:
        # --- NPU graph capture path ---
        # actual_seq_lengths_kv cannot use .tolist() inside torch.npu.graph()
        # (triggers D2H memcpy).  Use the value cached from the last eager
        # call (warmup) or a reasonable placeholder.
        seq_lengths_kv_list = _last_eager_seq_lens_kv or [1] * batch_size

        fia_kwargs: dict = dict(
            block_table=page_table,
            input_layout="TND",
            block_size=page_size,
            actual_seq_lengths=actual_seq_lengths_q,
            actual_seq_lengths_kv=seq_lengths_kv_list,
            num_key_value_heads=num_kv_heads,
            num_heads=num_q_heads,
            scale=scale,
            sparse_mode=0,
            softmax_lse_flag=return_lse,
        )
        if window_left >= 0:
            fia_kwargs.update(
                atten_mask=_get_causal_mask(q.device),
                sparse_mode=4,
                pre_tokens=window_left,
                next_tokens=0,
            )

        task_params = get_npu_graph_task_params()
        stream = torch_npu.npu.current_stream()

        # Pre-allocate output tensors for the .out() form
        num_tokens = q.shape[0]
        attn_output = torch.empty(
            (num_tokens, num_q_heads, head_dim),
            dtype=q.dtype,
            device=q.device,
        )
        softmax_lse = torch.empty(1, dtype=q.dtype, device=q.device)

        torch.npu.graph_task_group_begin(stream)

        torch_npu.npu_fused_infer_attention_score.out(
            query=q,
            key=k_for_fia,
            value=v_for_fia,
            out=[attn_output, softmax_lse],
            **fia_kwargs,
        )

        handle = torch.npu.graph_task_group_end(stream)

        # Record for replay-time task update
        task_params.record_decode_task(
            handle=handle,
            query=q,
            key=k_for_fia,
            value=v_for_fia,
            attn_output=attn_output,
            softmax_lse=softmax_lse,
            fia_kwargs=fia_kwargs,
        )

        if attn_output.dtype != q.dtype:
            attn_output = attn_output.to(q.dtype)

        if return_lse:
            return attn_output, softmax_lse
        return attn_output

    # --- Eager (non-capture) path ---
    seq_lengths_kv_list = (
        cache_seqlens if isinstance(cache_seqlens, list) else cache_seqlens.tolist()
    )

    # Cache the seq_lengths_kv for the next capture call.
    _last_eager_seq_lens_kv = seq_lengths_kv_list

    fia_kwargs: dict = dict(
        block_table=page_table,
        input_layout="TND",
        block_size=page_size,
        actual_seq_lengths=actual_seq_lengths_q,
        actual_seq_lengths_kv=seq_lengths_kv_list,
        num_key_value_heads=num_kv_heads,
        num_heads=num_q_heads,
        scale=scale,
        sparse_mode=0,
        softmax_lse_flag=return_lse,
    )
    if window_left >= 0:
        fia_kwargs.update(
            atten_mask=_get_causal_mask(q.device),
            sparse_mode=4,
            pre_tokens=window_left,
            next_tokens=0,
        )

    attn_output, softmax_lse = torch_npu.npu_fused_infer_attention_score(
        q, k_for_fia, v_for_fia, **fia_kwargs
    )
    if attn_output.dtype != q.dtype:
        attn_output = attn_output.to(q.dtype)

    if return_lse:
        return attn_output, softmax_lse
    return attn_output


def torch_npu_mha_extend_with_kvcache(
    *,
    q: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    is_causal: bool = True,
    window_left: int = -1,
    logit_cap: float = 0.0,
    sinks: torch.Tensor | None = None,
    return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
    """MHA extend with paged KV cache via npu_fused_infer_attention_score.

    Extend = prefill against a cached prefix: each request has Q_new new
    query tokens attending (causally) to cached_prefix + new tokens, which
    the caller has already written into the paged KV cache.  FIA's TND
    layout + block_table + per-batch cumulative q/kv lengths supports this
    directly.  Runs in the prefill phase (outside the decode graph capture),
    so ``.tolist()`` D2H is allowed and no task-group machinery is needed.

    Args:
        q: [total_q, num_q_heads, head_dim] — new query tokens.
        cu_seqlens_q: [batch + 1] cumulative new-query lengths.
        cu_seqlens_kv: [batch + 1] cumulative KV lengths (cached + new).
            Element values are cumulative (running sums), matching FIA's
            TND ``actual_seq_lengths_kv`` contract.
        k_cache: [num_pages, page_size, num_kv_heads, head_dim].
        v_cache: [num_pages, page_size, num_kv_heads, head_dim].
        page_table: [batch, max_pages_per_seq].
        cache_seqlens: [batch] visible KV length per request.
        max_seqlen_q: max new-query length (unused; derived from cu_seqlens).
        max_seqlen_k: max KV length (unused; derived from cu_seqlens).
        is_causal: whether new tokens form a causal suffix of the KV.
        window_left: sliding-window size, -1 for full attention.
        logit_cap: soft cap on logits (unused on Ascend).
        sinks: attention sinks (unused).
        return_lse: whether to return log-sum-exp.
    """
    num_q_heads = q.shape[1]
    num_kv_heads = k_cache.shape[2]
    head_dim = q.shape[-1]
    page_size = k_cache.shape[1]
    num_blocks = k_cache.shape[0]
    scale = 1.0 / math.sqrt(head_dim)

    # Q tokens are concatenated in the T dimension, so ``actual_seq_lengths``
    # (q) must be CUMULATIVE (cu_seqlens_q[1:]) to locate each request's
    # query span.  KV is paged via ``block_table`` (not concatenated), so
    # ``actual_seq_lengths_kv`` must be PER-REQUEST (each request's own KV
    # length = ``cache_seqlens``), NOT cumulative — passing cumulative KV
    # lengths makes FIA read the running-sum total for the last request and
    # deadlocks ACL (507015) under large concurrent extend batches.
    # Both are passed as device tensors to avoid a D2H ``.tolist()`` sync.
    actual_seq_lengths_q = cu_seqlens_q[1:]
    actual_seq_lengths_kv = cache_seqlens

    # Ascend FIA expected layout: [num_blocks, block_size, num_kv_heads * head_dim]
    k_for_fia = k_cache.view(num_blocks, page_size, -1)
    v_for_fia = v_cache.view(num_blocks, page_size, -1)

    fia_kwargs: dict = dict(
        block_table=page_table,
        input_layout="TND",
        block_size=page_size,
        actual_seq_lengths=actual_seq_lengths_q,
        actual_seq_lengths_kv=actual_seq_lengths_kv,
        num_key_value_heads=num_kv_heads,
        num_heads=num_q_heads,
        scale=scale,
        softmax_lse_flag=return_lse,
    )
    if window_left >= 0:
        fia_kwargs.update(
            atten_mask=_get_causal_mask(q.device),
            sparse_mode=4,
            pre_tokens=window_left,
            next_tokens=0,
        )
    else:
        fia_kwargs.update(
            atten_mask=_get_causal_mask(q.device),
            sparse_mode=3,
        )

    attn_output, softmax_lse = torch_npu.npu_fused_infer_attention_score(
        q, k_for_fia, v_for_fia, **fia_kwargs
    )
    if attn_output.dtype != q.dtype:
        attn_output = attn_output.to(q.dtype)

    if return_lse:
        return attn_output, softmax_lse
    return attn_output

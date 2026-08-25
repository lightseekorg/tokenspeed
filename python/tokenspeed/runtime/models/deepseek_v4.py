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

"""Inference-only DeepSeek V4 model skeleton.

This module intentionally registers only architecture pieces that map to the
DeepSeek V4 Flash checkpoint. The sparse MLA forward path still fails loudly
until the HCA/CSA cache kernels are wired into TokenSpeed.
"""

from __future__ import annotations

import gc
import re
from collections.abc import Iterable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from tokenspeed_kernel import (
    NoKernelFoundError,
    dsa_decode_topk,
    dsa_prefill_topk,
    dsv4_grouped_output_projection,
    dsv4_grouped_output_projection_plan,
    dsv4_grouped_output_projection_warmup_model,
    dsv4_indexer_cache_format,
    dsv4_indexer_decode_metadata_compute,
    dsv4_indexer_decode_topk,
    dsv4_indexer_prefill_topk,
)
from tokenspeed_kernel import dsv4_linear_fp32 as _kernel_dsv4_linear_fp32
from tokenspeed_kernel import (
    dsv4_mega_moe_apply,
    dsv4_mega_moe_plan,
    dsv4_mega_moe_process_weights,
    dsv4_mega_moe_warmup,
    dsv4_padded_heads,
    dsv4_plan,
)
from tokenspeed_kernel import dsv4_select_experts as _kernel_dsv4_select_experts
from tokenspeed_kernel import (
    dsv4_warmup,
)
from tokenspeed_kernel import mhc_fused_hc as fast_mhc_fused_hc
from tokenspeed_kernel import mhc_post as fast_mhc_post
from tokenspeed_kernel import mhc_pre as fast_mhc_pre
from torch import nn
from transformers import PretrainedConfig

from tokenspeed.runtime.distributed import Mapping
from tokenspeed.runtime.distributed.comm_manager import CommManager
from tokenspeed.runtime.distributed.pp_stage import PPStageState, pp_layer_window
from tokenspeed.runtime.distributed.process_group_manager import (
    process_group_manager as pg_manager,
)
from tokenspeed.runtime.execution.breakable_cuda_graph import (
    break_point,
    current_forward_ctx,
    slice_to_real_tokens,
)
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.cuda_graph_wrapper import get_is_capture_mode
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.activation import SiluAndMul
from tokenspeed.runtime.layers.attention.deepseek_v4.metadata import (
    DeepseekV4ForwardMetadata,
    DeepseekV4IndexerBatchMetadata,
    DeepseekV4IndexerDecodePlan,
    DeepseekV4IndexerPrefillChunkPlan,
    DeepseekV4IndexerPrefillMetadata,
    DeepseekV4SparseIndexerMetadata,
)
from tokenspeed.runtime.layers.attention.deepseek_v4_geometry import (
    DEEPSEEK_V4_MXFP4_BLOCK_SIZE,
    V4_KERNEL_BLOCK_ROWS,
    deepseek_v4_indexer_mxfp4_scale_dim,
    deepseek_v4_indexer_mxfp4_value_bytes,
    deepseek_v4_nope_dim,
    v4_compressed_kv_group_id,
)
from tokenspeed.runtime.layers.attention.deepseek_v4_ops import (
    deepseek_v4_csa_compress_kv_cache_insert,
    deepseek_v4_csa_indexer_cache_insert,
    deepseek_v4_hca_compress_kv_cache_insert,
    deepseek_v4_prepare_indexer_q,
    deepseek_v4_prepare_indexer_q_mxfp4,
    fused_qnorm_rope_kv_insert,
    save_deepseek_v4_compressor_state,
)
from tokenspeed.runtime.layers.attention.kv_cache.hybrid_deepseek_v4 import (
    _group_slot_mapping_from_raw,
    _mask_invalid_graph_tokens,
)
from tokenspeed.runtime.layers.layernorm import FusedRMSNorm, RMSNorm
from tokenspeed.runtime.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
    warmup_prepared_fp8_linears,
)
from tokenspeed.runtime.layers.moe import (
    ExpertCheckpointSchema,
    build_moe_checkpoint_loader,
)
from tokenspeed.runtime.layers.moe.expert import MoELayer
from tokenspeed.runtime.layers.moe.topk import (
    BypassedTopKOutput,
    StandardTopKOutput,
    TopK,
)
from tokenspeed.runtime.layers.moe.utils import RoutingMethodType, get_moe_backend
from tokenspeed.runtime.layers.quantization import Fp8Config, Mxfp4Config
from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig
from tokenspeed.runtime.layers.rotary_embedding import get_rope
from tokenspeed.runtime.layers.vocab_parallel_embedding import VocabParallelEmbedding
from tokenspeed.runtime.model_loader.weight_utils import default_weight_loader
from tokenspeed.runtime.models.base import BaseCausalLM
from tokenspeed.runtime.moe.expert_location import ModelConfigForExpertLocation
from tokenspeed.runtime.utils import (
    add_prefix,
    get_colorful_logger,
    set_weight_attrs,
)
from tokenspeed.runtime.utils.common import PPMissingLayer
from tokenspeed.runtime.utils.cuda_stream import StreamFork
from tokenspeed.runtime.utils.custom_ops import direct_register_custom_op
from tokenspeed.runtime.utils.env import global_server_args_dict
from tokenspeed.runtime.utils.nvtx import nvtx_range
from tokenspeed.runtime.utils.pdl import pdl_enabled

logger = get_colorful_logger(__name__)


def _deepseek_v4_metadata_matches_tokens(metadata, num_tokens: int) -> bool:
    return (
        metadata is not None
        and getattr(metadata, "token_to_req_indices", None) is not None
        and metadata.token_to_req_indices.numel() == num_tokens
    )


def _deepseek_v4_indexer_token_split(
    forward_mode: ForwardMode | None,
    metadata,
    total_tokens: int,
) -> tuple[int, int]:
    if forward_mode is not None and forward_mode.is_mixed():
        return int(metadata.num_prefill_tokens), metadata.decode_token_count()
    if forward_mode is not None and forward_mode.is_decode():
        return 0, int(total_tokens)
    return int(total_tokens), 0


def _deepseek_v4_forward_metadata(ctx: ForwardContext):
    metadata = getattr(ctx.attn_backend, "forward_metadata", None)
    forward_mode = getattr(ctx, "forward_mode", None)
    if forward_mode is not None and forward_mode.is_extend_or_mixed():
        return getattr(ctx.attn_backend, "forward_prefill_metadata", None) or metadata
    if forward_mode is not None and forward_mode.is_decode_or_idle():
        input_num_tokens = getattr(ctx, "input_num_tokens", None)
        decode_metadata = getattr(ctx.attn_backend, "forward_decode_metadata", None)
        if input_num_tokens is not None and _deepseek_v4_metadata_matches_tokens(
            decode_metadata,
            input_num_tokens,
        ):
            return decode_metadata
        prefill_metadata = getattr(ctx.attn_backend, "forward_prefill_metadata", None)
        if input_num_tokens is not None and _deepseek_v4_metadata_matches_tokens(
            prefill_metadata,
            input_num_tokens,
        ):
            return prefill_metadata
        return decode_metadata or metadata or prefill_metadata
    return metadata


def _dequant_fp8_weight(layer: nn.Module, shape: tuple[int, ...]) -> torch.Tensor:
    weight = layer.weight.view(*shape)
    scale = getattr(layer, "weight_scale_inv", None)
    if scale is None or weight.dtype != torch.float8_e4m3fn:
        return weight.float()

    block_n, block_k = getattr(layer.quant_config, "weight_block_size", (128, 128))
    if len(shape) == 2:
        out_dim, in_dim = shape
        scale = scale.view(
            (out_dim + block_n - 1) // block_n,
            (in_dim + block_k - 1) // block_k,
        )
        expanded_scale = (
            scale.float()
            .repeat_interleave(block_n, dim=0)
            .repeat_interleave(block_k, dim=1)
        )
        return weight.float() * expanded_scale[:out_dim, :in_dim]

    groups, out_dim, in_dim = shape
    scale = scale.view(
        groups,
        (out_dim + block_n - 1) // block_n,
        (in_dim + block_k - 1) // block_k,
    )
    expanded_scale = (
        scale.float()
        .repeat_interleave(block_n, dim=1)
        .repeat_interleave(block_k, dim=2)
    )
    return weight.float() * expanded_scale[:, :out_dim, :in_dim]


def _deepseek_v4_reorder_c4_ape_2604(ape: torch.Tensor) -> torch.Tensor:
    """Convert C4 overlap APE from checkpoint layout to runtime window layout."""

    if ape.dim() != 2 or ape.shape[0] != 4 or ape.shape[1] % 2 != 0:
        raise ValueError(f"expected C4 APE [4, even], got {tuple(ape.shape)}")
    older, newer = ape.chunk(2, dim=-1)
    return torch.cat([older, newer], dim=0).reshape_as(ape)


def mhc_pre(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return fast_mhc_pre(
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_eps,
        sinkhorn_iters,
    )


def mhc_post(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    return fast_mhc_post(hidden_states, residual, post, comb)


def mhc_fused_hc(
    x_prev: torch.Tensor,
    residual_prev: torch.Tensor,
    post_prev: torch.Tensor,
    comb_prev: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return fast_mhc_fused_hc(
        x_prev,
        residual_prev,
        post_prev,
        comb_prev,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_eps,
        sinkhorn_iters,
    )


def hc_head(
    hidden_states: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_norm_eps: float,
    hc_eps: float,
) -> torch.Tensor:
    shape, dtype = hidden_states.size(), hidden_states.dtype
    x = hidden_states.flatten(1).float()
    rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + rms_norm_eps)
    mixes = F.linear(x, hc_fn.float()) * rsqrt
    pre = torch.sigmoid(mixes * hc_scale.float() + hc_base.float()) + hc_eps
    y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=1)
    return y.to(dtype)


def pack_topk_as_router_logits(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """Encode preselected top-k weights for BYPASSED TokenSpeed MoE backends.

    MXFP4 backends currently build routing data from logits internally. Packing
    the normalized top-k weights as log-probabilities with very negative values
    elsewhere makes their TopK -> Softmax/Renormalize route recover the same
    selected ids and weights without changing the shared backend.
    """

    router_logits = torch.full(
        (topk_ids.shape[0], num_experts),
        -1e20,
        dtype=torch.float32,
        device=topk_weights.device,
    )
    safe_weights = topk_weights.clamp_min(torch.finfo(torch.float32).tiny)
    router_logits.scatter_(1, topk_ids.long(), safe_weights.log())
    return router_logits


@dataclass(frozen=True)
class _DeepseekV4IndexerPrefillChunk:
    token_start: int
    token_end: int
    req_start: int
    req_end: int
    query_start: int
    query_end: int
    skip_kv_gather: bool = False


def _deepseek_v4_indexer_prefill_max_logits_bytes(
    max_logits_bytes: int | None = None,
) -> int:
    if max_logits_bytes is not None:
        return max(1, int(max_logits_bytes))
    max_logits_mb = global_server_args_dict["deepseek_v4_indexer_prefill_max_logits_mb"]
    return max(1, int(max_logits_mb) * 1024 * 1024)


def _deepseek_v4_indexer_prefill_workspace_size(
    seq_lens_cpu: torch.Tensor,
    workspace_size: int | None = None,
) -> int:
    if workspace_size is not None:
        return max(1, int(workspace_size))
    context_len = global_server_args_dict.get("max_model_len")
    if isinstance(context_len, int) and context_len > 0:
        return context_len * 40
    max_seq_len = int(seq_lens_cpu.max().item()) if seq_lens_cpu.numel() else 1
    return max(1, max_seq_len) * 40


def _deepseek_v4_indexer_prefill_request_chunks(
    *,
    seq_lens_cpu: torch.Tensor,
    query_lens_cpu: torch.Tensor,
    compress_ratio: int,
    num_tokens: int,
    max_logits_bytes: int | None = None,
    workspace_size: int | None = None,
    request_offset: int = 0,
) -> list[_DeepseekV4IndexerPrefillChunk]:
    """Build request/query-slice sparse-indexer prefill chunks."""

    if num_tokens == 0:
        return []

    seq_lens = seq_lens_cpu.detach().cpu().to(torch.int64)
    query_lens = query_lens_cpu.detach().cpu().to(torch.int64)
    if seq_lens.numel() != query_lens.numel():
        return []

    query_lens_list = [max(0, int(x)) for x in query_lens.tolist()]
    if sum(query_lens_list) != num_tokens:
        return []

    compressed_seq_lens = torch.div(
        seq_lens,
        max(1, int(compress_ratio)),
        rounding_mode="floor",
    )
    compressed_seq_lens_list = [max(0, int(x)) for x in compressed_seq_lens.tolist()]
    workspace_rows = _deepseek_v4_indexer_prefill_workspace_size(
        seq_lens,
        workspace_size,
    )
    max_logits_elems = (
        _deepseek_v4_indexer_prefill_max_logits_bytes(max_logits_bytes) // 4
    )
    max_logits_elems = max(1, max_logits_elems)

    query_offsets = [0]
    for query_len in query_lens_list:
        query_offsets.append(query_offsets[-1] + query_len)

    chunks: list[_DeepseekV4IndexerPrefillChunk] = []
    n_reqs = len(query_lens_list)
    end = 0
    while end < n_reqs:
        start = end
        chunk_m = 0
        chunk_n = 0
        while end < n_reqs:
            q_len = query_lens_list[end]
            seq_len = compressed_seq_lens_list[end]
            new_m = chunk_m + q_len
            new_n = chunk_n + seq_len
            if new_n <= workspace_rows and new_m * new_n <= max_logits_elems:
                chunk_m = new_m
                chunk_n = new_n
                end += 1
            else:
                break

        if end == start:
            chunk_m = query_lens_list[end]
            chunk_n = compressed_seq_lens_list[end]
            end += 1

        if chunk_m <= 0:
            continue

        req_start = start + request_offset
        req_end = end + request_offset
        max_q = max(1, max_logits_elems // chunk_n) if chunk_n > 0 else chunk_m
        chunk_token_start = query_offsets[start]
        for query_start in range(0, chunk_m, max_q):
            query_end = min(query_start + max_q, chunk_m)
            chunks.append(
                _DeepseekV4IndexerPrefillChunk(
                    token_start=chunk_token_start + query_start,
                    token_end=chunk_token_start + query_end,
                    req_start=req_start,
                    req_end=req_end,
                    query_start=query_start,
                    query_end=query_end,
                    skip_kv_gather=query_start > 0,
                )
            )
    return chunks


def _deepseek_v4_indexer_decode_max_len(
    block_table: torch.Tensor,
    cache_block_size: int,
    compress_ratio: int,
) -> int:
    context_len = global_server_args_dict.get("max_model_len")
    if isinstance(context_len, int) and context_len > 0:
        return max(1, (context_len + compress_ratio - 1) // compress_ratio)
    return max(
        1,
        (block_table.shape[1] * cache_block_size + compress_ratio - 1)
        // compress_ratio,
    )


def _deepseek_v4_indexer_prefill_request_gather_plan(
    *,
    seq_lens_cpu: torch.Tensor,
    query_lens_cpu: torch.Tensor,
    block_table: torch.Tensor,
    cache_block_size: int,
    compress_ratio: int,
    req_start: int,
    req_end: int,
    query_start: int,
    query_end: int,
    block_table_base_offsets: torch.Tensor | None = None,
    build_slots: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    device = block_table.device
    num_rows = max(0, int(query_end) - int(query_start))
    if num_rows == 0 or req_end <= req_start:
        empty_i32 = torch.empty(0, dtype=torch.int32, device=device)
        empty_i64 = torch.empty(0, dtype=torch.int64, device=device)
        return empty_i64, empty_i32, empty_i32, empty_i32, 0

    seq_lens_list = (
        seq_lens_cpu.detach().cpu().to(torch.int64)[req_start:req_end].tolist()
    )
    query_lens_list = (
        query_lens_cpu.detach().cpu().to(torch.int64)[req_start:req_end].tolist()
    )
    if len(seq_lens_list) != len(query_lens_list):
        empty_i32 = torch.empty(0, dtype=torch.int32, device=device)
        empty_i64 = torch.empty(0, dtype=torch.int64, device=device)
        return empty_i64, empty_i32, empty_i32, empty_i32, 0

    ratio = max(1, int(compress_ratio))
    seq_lens_list = [max(0, int(x)) for x in seq_lens_list]
    query_lens_list = [max(0, int(x)) for x in query_lens_list]
    compressed_lens_list = [seq_len // ratio for seq_len in seq_lens_list]
    total_k = sum(compressed_lens_list)

    query_offsets: list[int] = [0]
    for query_len in query_lens_list:
        query_offsets.append(query_offsets[-1] + query_len)

    req_local_list: list[int] = []
    row_lens_list: list[int] = []
    req_local = 0
    last_req = max(0, len(query_lens_list) - 1)
    for row_offset in range(int(query_start), int(query_end)):
        while req_local < last_req and row_offset >= query_offsets[req_local + 1]:
            req_local += 1
        local_query_offset = row_offset - query_offsets[req_local]
        prefix_len = max(0, seq_lens_list[req_local] - query_lens_list[req_local])
        row_lens_list.append((prefix_len + local_query_offset + 1) // ratio)
        req_local_list.append(req_local)
    max_len = max(row_lens_list) if row_lens_list else 0

    compressed_lens = torch.tensor(
        compressed_lens_list,
        dtype=torch.int64,
        device=device,
    )
    request_ids = torch.arange(req_start, req_end, device=device, dtype=torch.int64)
    if block_table_base_offsets is None:
        request_base_pages = torch.zeros_like(request_ids)
    else:
        if block_table_base_offsets.ndim != 1:
            raise ValueError("block_table_base_offsets must be one-dimensional")
        request_base_pages = block_table_base_offsets.to(
            device=device, dtype=torch.int64
        )[request_ids]
    request_base_rows = request_base_pages * int(cache_block_size)
    request_capacity_ends = request_base_rows + int(block_table.shape[1]) * int(
        cache_block_size
    )
    request_starts = torch.minimum(request_base_rows, compressed_lens).clamp_min(0)
    request_ends = torch.minimum(request_capacity_ends, compressed_lens)
    request_ends = torch.maximum(request_ends, request_starts)

    cu_seq_lens = torch.empty(
        compressed_lens.numel() + 1,
        dtype=torch.int32,
        device=device,
    )
    cu_seq_lens[:1] = 0
    torch.cumsum(compressed_lens.to(torch.int32), dim=0, out=cu_seq_lens[1:])

    req_local_tensor = torch.tensor(req_local_list, dtype=torch.int64, device=device)
    row_lens = torch.tensor(row_lens_list, dtype=torch.int32, device=device)
    group_starts = cu_seq_lens[:-1][req_local_tensor]
    logical_starts = request_starts[req_local_tensor]
    logical_ends = torch.minimum(
        row_lens.to(torch.int64), request_ends[req_local_tensor]
    )
    logical_ends = torch.maximum(logical_ends, logical_starts)
    cu_start = group_starts + logical_starts.to(torch.int32)
    cu_end = group_starts + logical_ends.to(torch.int32)
    row_lens = (logical_ends - logical_starts).to(torch.int32)

    if total_k <= 0 or not build_slots:
        empty_i64 = torch.empty(0, dtype=torch.int64, device=device)
        return empty_i64, cu_start, cu_end, row_lens, max_len

    req_ids = torch.repeat_interleave(
        torch.arange(req_start, req_end, device=device, dtype=torch.int64),
        compressed_lens,
        output_size=total_k,
    )
    req_local_for_k = req_ids - int(req_start)
    group_bases = cu_seq_lens[:-1][req_local_for_k].to(torch.int64)
    local = torch.arange(total_k, device=device, dtype=torch.int64) - group_bases
    pages = torch.div(local, cache_block_size, rounding_mode="floor")
    page_offsets = local % cache_block_size
    base_pages = block_table_base_offsets
    if base_pages is None:
        base_pages_for_k = torch.zeros_like(req_ids)
    else:
        base_pages_for_k = base_pages.to(device=device, dtype=torch.int64)[req_ids]
    table_pages = pages - base_pages_for_k
    valid_pages = (table_pages >= 0) & (table_pages < block_table.shape[1])
    safe_pages = table_pages.clamp(0, max(int(block_table.shape[1]) - 1, 0))
    page_ids = block_table[req_ids, safe_pages.long()].to(torch.int64)
    valid_pages = valid_pages & (page_ids >= 0)
    slots = torch.where(
        valid_pages,
        page_ids * cache_block_size + page_offsets,
        torch.zeros_like(page_offsets),
    )
    return slots, cu_start, cu_end, row_lens, max_len


def _deepseek_v4_indexer_prefill_chunk_total_rows(
    *,
    seq_lens_cpu: torch.Tensor,
    compress_ratio: int,
    req_start: int,
    req_end: int,
) -> int:
    ratio = max(1, int(compress_ratio))
    seq_lens = seq_lens_cpu.detach().cpu().to(torch.int64)[req_start:req_end].tolist()
    return sum(max(0, int(seq_len)) // ratio for seq_len in seq_lens)


def _deepseek_v4_indexer_prefill_metadata(
    *,
    metadata: DeepseekV4ForwardMetadata,
    block_table: torch.Tensor,
    cache_block_size: int,
    compress_ratio: int,
    num_prefill_tokens: int,
    block_table_base_offsets: torch.Tensor | None = None,
) -> DeepseekV4IndexerPrefillMetadata:
    device = block_table.device
    if num_prefill_tokens <= 0:
        return DeepseekV4IndexerPrefillMetadata.empty(device)

    seq_lens_cpu = getattr(metadata, "seq_lens_cpu", None)
    query_lens_cpu = getattr(metadata, "query_lens_cpu", None)
    num_prefill_reqs = int(getattr(metadata, "num_prefill_reqs", 0) or 0)
    if seq_lens_cpu is None or query_lens_cpu is None or num_prefill_reqs <= 0:
        return DeepseekV4IndexerPrefillMetadata.empty(device)

    seq_lens_cpu = seq_lens_cpu[:num_prefill_reqs]
    query_lens_cpu = query_lens_cpu[:num_prefill_reqs]
    cache_key = (
        compress_ratio,
        cache_block_size,
        num_prefill_tokens,
        (
            0
            if block_table_base_offsets is None
            else int(block_table_base_offsets.data_ptr())
        ),
    )
    cache = metadata.indexer.prefill_plan_cache
    cached = cache.get(cache_key)
    if cached is not None and cached.slots.device == device:
        return cached

    chunks = _deepseek_v4_indexer_prefill_request_chunks(
        seq_lens_cpu=seq_lens_cpu,
        query_lens_cpu=query_lens_cpu,
        compress_ratio=compress_ratio,
        num_tokens=num_prefill_tokens,
    )
    if not chunks:
        out = DeepseekV4IndexerPrefillMetadata.empty(device)
        cache[cache_key] = out
        return out

    chunk_plans: list[DeepseekV4IndexerPrefillChunkPlan] = []
    slot_parts: list[torch.Tensor] = []
    cu_seq_lens_parts: list[torch.Tensor] = []
    cu_seqlen_k_start_parts: list[torch.Tensor] = []
    cu_seqlen_k_end_parts: list[torch.Tensor] = []
    seq_lens_k_parts: list[torch.Tensor] = []
    slot_offset = 0
    cu_seq_offset = 0
    row_offset = 0
    for chunk in chunks:
        slots, cu_seqlen_k_start, cu_seqlen_k_end, seq_lens_k, max_seqlen_k = (
            _deepseek_v4_indexer_prefill_request_gather_plan(
                seq_lens_cpu=seq_lens_cpu,
                query_lens_cpu=query_lens_cpu,
                block_table=block_table,
                cache_block_size=cache_block_size,
                compress_ratio=compress_ratio,
                req_start=chunk.req_start,
                req_end=chunk.req_end,
                query_start=chunk.query_start,
                query_end=chunk.query_end,
                block_table_base_offsets=block_table_base_offsets,
                build_slots=True,
            )
        )
        slot_count = _deepseek_v4_indexer_prefill_chunk_total_rows(
            seq_lens_cpu=seq_lens_cpu,
            compress_ratio=compress_ratio,
            req_start=chunk.req_start,
            req_end=chunk.req_end,
        )
        compressed_lens = torch.div(
            seq_lens_cpu[chunk.req_start : chunk.req_end].to(
                dtype=torch.int32,
                device=device,
            ),
            max(1, int(compress_ratio)),
            rounding_mode="floor",
        )
        cu_seq_lens = torch.empty(
            compressed_lens.numel() + 1,
            dtype=torch.int32,
            device=device,
        )
        cu_seq_lens[:1] = 0
        torch.cumsum(compressed_lens, dim=0, out=cu_seq_lens[1:])
        slot_end = slot_offset + slot_count
        cu_seq_end = cu_seq_offset + cu_seq_lens.numel()
        row_end = row_offset + seq_lens_k.numel()
        chunk_plans.append(
            DeepseekV4IndexerPrefillChunkPlan(
                token_start=chunk.token_start,
                token_end=chunk.token_end,
                request_start=chunk.req_start,
                request_end=chunk.req_end,
                slot_start=slot_offset,
                slot_end=slot_end,
                gather_row_start=row_offset,
                gather_row_end=row_end,
                max_seq_len_k=max_seqlen_k,
                cu_seq_lens_start=cu_seq_offset,
                cu_seq_lens_end=cu_seq_end,
                skip_kv_gather=chunk.skip_kv_gather,
            )
        )
        if slots.numel() > 0:
            slot_parts.append(slots)
        cu_seq_lens_parts.append(cu_seq_lens)
        cu_seqlen_k_start_parts.append(cu_seqlen_k_start)
        cu_seqlen_k_end_parts.append(cu_seqlen_k_end)
        seq_lens_k_parts.append(seq_lens_k)
        slot_offset = slot_end
        cu_seq_offset = cu_seq_end
        row_offset = row_end

    out = DeepseekV4IndexerPrefillMetadata(
        chunks=tuple(chunk_plans),
        chunk_specs=torch.tensor(
            [
                [
                    chunk.token_start,
                    chunk.token_end,
                    chunk.request_start,
                    chunk.request_end,
                    1 if chunk.skip_kv_gather else 0,
                ]
                for chunk in chunk_plans
            ],
            dtype=torch.int64,
            device="cpu",
        ),
        chunk_offsets=torch.tensor(
            [
                [
                    chunk.slot_start,
                    chunk.slot_end,
                    chunk.gather_row_start,
                    chunk.gather_row_end,
                    chunk.max_seq_len_k,
                    chunk.cu_seq_lens_start,
                    chunk.cu_seq_lens_end,
                ]
                for chunk in chunk_plans
            ],
            dtype=torch.int64,
            device="cpu",
        ),
        slots=(
            torch.cat(slot_parts, dim=0)
            if slot_parts
            else torch.empty(0, dtype=torch.int64, device=device)
        ),
        cu_seq_lens=(
            torch.cat(cu_seq_lens_parts, dim=0)
            if cu_seq_lens_parts
            else torch.empty(0, dtype=torch.int32, device=device)
        ),
        cu_seqlen_k_start=(
            torch.cat(cu_seqlen_k_start_parts, dim=0)
            if cu_seqlen_k_start_parts
            else torch.empty(0, dtype=torch.int32, device=device)
        ),
        cu_seqlen_k_end=(
            torch.cat(cu_seqlen_k_end_parts, dim=0)
            if cu_seqlen_k_end_parts
            else torch.empty(0, dtype=torch.int32, device=device)
        ),
        seq_lens_k=(
            torch.cat(seq_lens_k_parts, dim=0)
            if seq_lens_k_parts
            else torch.empty(0, dtype=torch.int32, device=device)
        ),
    )
    cache[cache_key] = out
    return out


def _deepseek_v4_indexer_decode_plan(
    *,
    positions: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    block_table: torch.Tensor,
    cache_block_size: int,
    compress_ratio: int,
    metadata: DeepseekV4ForwardMetadata | None = None,
    is_valid_token: torch.Tensor | None = None,
    block_table_base_offsets: torch.Tensor | None = None,
) -> DeepseekV4IndexerDecodePlan:
    num_tokens = positions.numel()
    key = (int(compress_ratio), int(cache_block_size), int(num_tokens))
    indexer_metadata = metadata.indexer if metadata is not None else None
    cache = None if indexer_metadata is None else indexer_metadata.decode_plan_cache
    refreshed_keys = (
        None
        if indexer_metadata is None
        else indexer_metadata.decode_plan_refreshed_keys
    )
    cached = cache.get(key) if cache is not None else None
    # Hot path: the attention metadata builder hook pre-builds the plan tensors
    # before per-layer parallel work so indexer layers only read cached buffers.
    if cached is not None and refreshed_keys is not None and key in refreshed_keys:
        return cached

    if num_tokens == 0:
        context_lens = torch.empty((0, 1), dtype=torch.int32, device=positions.device)
        block_tables = torch.empty(
            (0, 1),
            dtype=torch.int32,
            device=block_table.device,
        )
        plan = DeepseekV4IndexerDecodePlan(context_lens, block_tables, 0)
        if cache is not None:
            cache[key] = plan
        if refreshed_keys is not None:
            refreshed_keys.add(key)
        return plan

    rows = int(block_table.shape[0]) if block_table.ndim >= 1 else 0
    cols = int(block_table.shape[1]) if block_table.ndim >= 2 else 0
    max_len = _deepseek_v4_indexer_decode_max_len(
        block_table,
        cache_block_size,
        compress_ratio,
    )
    max_blocks = max(1, (max_len + cache_block_size - 1) // cache_block_size)

    expected_context_shape = (num_tokens, 1)
    expected_block_shape = (num_tokens, max_blocks)
    if (
        cached is None
        or cached.context_lens.shape != expected_context_shape
        or cached.context_lens.device != positions.device
        or cached.context_lens.dtype != torch.int32
        or cached.page_table.shape != expected_block_shape
        or cached.page_table.device != block_table.device
        or cached.page_table.dtype != torch.int32
    ):
        context_lens = torch.empty(
            expected_context_shape,
            dtype=torch.int32,
            device=positions.device,
        )
        block_tables = torch.empty(
            expected_block_shape,
            dtype=torch.int32,
            device=block_table.device,
        )
        plan = DeepseekV4IndexerDecodePlan(
            context_lens=context_lens,
            page_table=block_tables,
            max_context_len=max_len,
        )
        if cache is not None:
            cache[key] = plan
    else:
        plan = cached
        plan.max_context_len = max_len

    if rows <= 0 or cols <= 0:
        plan.context_lens.zero_()
        plan.page_table.zero_()
        plan.max_context_len = 0
    else:
        dsv4_indexer_decode_metadata_compute(
            positions=positions,
            token_to_req_indices=token_to_req_indices,
            block_table=block_table,
            cache_block_size=cache_block_size,
            compress_ratio=compress_ratio,
            max_blocks=max_blocks,
            out_context_lens=plan.context_lens,
            out_block_tables=plan.page_table,
            block_table_base_offsets=block_table_base_offsets,
        )
        if is_valid_token is None:
            is_valid_token = getattr(metadata, "is_valid_token", None)
        if is_valid_token is not None:
            valid = is_valid_token[:num_tokens].to(
                device=plan.context_lens.device,
                dtype=torch.bool,
            )
            with torch.inference_mode():
                plan.context_lens.masked_fill_(~valid.view(num_tokens, 1), 0)
                plan.page_table.masked_fill_(
                    ~valid.to(device=plan.page_table.device).view(num_tokens, 1),
                    0,
                )
    if refreshed_keys is not None:
        refreshed_keys.add(key)
    return plan


def _deepseek_v4_indexer_decode_schedule_metadata(
    *,
    positions: torch.Tensor,
    cache_block_size: int,
    compress_ratio: int,
    metadata: DeepseekV4ForwardMetadata | None,
    context_lens: torch.Tensor | None = None,
) -> torch.Tensor | None:
    if positions.numel() == 0:
        return None

    num_tokens = positions.numel()
    if context_lens is None:
        compressed_lens = torch.div(
            positions.to(torch.int64) + 1,
            compress_ratio,
            rounding_mode="floor",
        ).clamp_min(0)
        context_lens = compressed_lens.to(torch.int32).view(num_tokens, 1).contiguous()
    schedule_key = (compress_ratio, cache_block_size, num_tokens)
    indexer_metadata = metadata.indexer if metadata is not None else None
    schedule_cache = (
        None
        if indexer_metadata is None
        else indexer_metadata.decode_schedule_metadata_cache
    )
    schedule_metadata = (
        schedule_cache.get(schedule_key) if schedule_cache is not None else None
    )

    with nvtx_range("indexer_decode_schedule_metadata"):
        refreshed = dsv4_plan(
            seq_lens_2d=context_lens,
            page_size=cache_block_size,
        )
    if refreshed is None:
        return None
    if schedule_metadata is not None:
        if (
            schedule_metadata.shape == refreshed.shape
            and schedule_metadata.device == refreshed.device
            and schedule_metadata.dtype == refreshed.dtype
        ):
            with torch.inference_mode():
                schedule_metadata.copy_(refreshed)
            return schedule_metadata
        if schedule_cache is not None:
            schedule_cache[schedule_key] = refreshed
        return refreshed
    schedule_metadata = refreshed
    if schedule_cache is not None:
        schedule_cache[schedule_key] = schedule_metadata
    return schedule_metadata


def _deepseek_v4_sparse_attn_indexer_native(
    *,
    cache_2d: torch.Tensor,
    positions: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    query_lens_cpu: torch.Tensor,
    prefill_chunk_specs: torch.Tensor,
    prefill_chunk_offsets: torch.Tensor,
    prefill_slots: torch.Tensor,
    prefill_cu_seq_lens: torch.Tensor,
    prefill_cu_seqlen_k_start: torch.Tensor,
    prefill_cu_seqlen_k_end: torch.Tensor,
    prefill_seq_lens_k: torch.Tensor,
    packed_q_values: torch.Tensor,
    packed_q_scales: torch.Tensor,
    packed_weights: torch.Tensor,
    decode_schedule_metadata: torch.Tensor | None,
    decode_context_lens: torch.Tensor | None,
    decode_block_table: torch.Tensor | None,
    decode_max_context_len: int,
    topk_indices_buffer: torch.Tensor,
    prefill_gather_values_workspace: torch.Tensor,
    prefill_gather_scales_workspace: torch.Tensor,
    persistent_topk_workspace: torch.Tensor,
    cache_block_size: int,
    compress_ratio: int,
    topk_tokens: int,
    num_prefill_tokens: int,
    num_decode_tokens: int,
    use_fp4_cache: bool = True,
) -> torch.Tensor:
    total_tokens = positions.numel()
    topk_out = topk_indices_buffer[:total_tokens]
    topk_out.fill_(-1)
    if total_tokens == 0:
        return topk_out

    def fill_prefill() -> None:
        if num_prefill_tokens <= 0:
            return

        if prefill_chunk_specs.numel() == 0:
            raise RuntimeError(
                "DeepSeek V4 sparse indexer prefill requires prepared chunk metadata"
            )

        gather_cache_key = None
        gathered_k = None
        num_chunks = prefill_chunk_specs.shape[0]
        for chunk_idx in range(num_chunks):
            (
                token_start,
                token_end,
                req_start,
                req_end,
                skip_kv_gather_raw,
            ) = prefill_chunk_specs[chunk_idx].tolist()
            (
                slot_start,
                slot_end,
                row_start,
                row_end,
                max_seqlen_k,
                cu_seq_start,
                cu_seq_end,
            ) = prefill_chunk_offsets[chunk_idx].tolist()
            skip_kv_gather = bool(int(skip_kv_gather_raw))
            gather_rows = max(0, slot_end - slot_start)
            gather_workspace = None
            if (
                prefill_gather_values_workspace.numel() > 0
                and prefill_gather_scales_workspace.numel() > 0
                and gather_rows <= prefill_gather_values_workspace.shape[0]
                and gather_rows <= prefill_gather_scales_workspace.shape[0]
            ):
                gather_workspace = (
                    prefill_gather_values_workspace[:gather_rows],
                    prefill_gather_scales_workspace[:gather_rows],
                )
            with nvtx_range("indexer_topk_prefill"):
                key = (req_start, req_end)
                reuse_k = None
                if skip_kv_gather and gather_cache_key == key:
                    reuse_k = gathered_k
                if prefill_cu_seq_lens.numel() == 0 or cu_seq_end <= cu_seq_start:
                    raise RuntimeError(
                        "DeepSeek V4 sparse indexer prefill metadata is incomplete"
                    )
                topk, next_gathered_k = dsv4_indexer_prefill_topk(
                    index_q=(
                        packed_q_values[token_start:token_end],
                        packed_q_scales[token_start:token_end],
                    ),
                    weights=packed_weights[token_start:token_end],
                    index_k_cache=cache_2d,
                    block_table=block_table[req_start:req_end],
                    cu_seq_lens=prefill_cu_seq_lens[cu_seq_start:cu_seq_end],
                    cu_seqlen_k_start=prefill_cu_seqlen_k_start[row_start:row_end],
                    cu_seqlen_k_end=prefill_cu_seqlen_k_end[row_start:row_end],
                    seq_lens=prefill_seq_lens_k[row_start:row_end],
                    page_size=cache_block_size,
                    topk=topk_tokens,
                    max_seqlen_k=max_seqlen_k,
                    index_k_format="mxfp4" if use_fp4_cache else "fp8_scaled",
                    gathered_k=reuse_k,
                    gather_workspace=gather_workspace,
                    out=topk_out[token_start:token_end],
                )
                if next_gathered_k is not None:
                    gather_cache_key = key
                    gathered_k = next_gathered_k
            topk_out[token_start:token_end].copy_(topk)

    def fill_decode() -> None:
        if num_decode_tokens <= 0:
            return

        decode_start = num_prefill_tokens
        decode_end = decode_start + num_decode_tokens
        decode_out = topk_out[decode_start:decode_end]
        if (
            decode_schedule_metadata is None
            or decode_context_lens is None
            or decode_block_table is None
        ):
            raise RuntimeError("DeepSeek V4 decode indexer metadata is incomplete")
        with nvtx_range("indexer_topk_decode"):
            dsv4_indexer_decode_topk(
                index_q=(
                    packed_q_values[decode_start:decode_end],
                    packed_q_scales[decode_start:decode_end],
                ),
                weights=packed_weights[decode_start:decode_end],
                index_k_cache=cache_2d,
                context_lens=decode_context_lens,
                block_table=decode_block_table,
                page_size=cache_block_size,
                topk=topk_tokens,
                max_context_len=decode_max_context_len,
                plan=decode_schedule_metadata,
                index_k_format="mxfp4" if use_fp4_cache else "fp8_scaled",
                out=decode_out,
                persistent_topk_workspace=persistent_topk_workspace,
            )

    fill_prefill()
    fill_decode()
    return topk_out


def _deepseek_v4_sparse_attn_indexer_op(
    cache_2d: torch.Tensor,
    positions: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    query_lens_cpu: torch.Tensor,
    prefill_chunk_specs: torch.Tensor,
    prefill_chunk_offsets: torch.Tensor,
    prefill_slots: torch.Tensor,
    prefill_cu_seq_lens: torch.Tensor,
    prefill_cu_seqlen_k_start: torch.Tensor,
    prefill_cu_seqlen_k_end: torch.Tensor,
    prefill_seq_lens_k: torch.Tensor,
    packed_q_values: torch.Tensor,
    packed_q_scales: torch.Tensor,
    packed_weights: torch.Tensor,
    decode_schedule_metadata: torch.Tensor,
    decode_context_lens: torch.Tensor,
    decode_block_table: torch.Tensor,
    decode_max_context_len: int,
    topk_indices_buffer: torch.Tensor,
    prefill_gather_values_workspace: torch.Tensor,
    prefill_gather_scales_workspace: torch.Tensor,
    persistent_topk_workspace: torch.Tensor,
    cache_block_size: int,
    compress_ratio: int,
    topk_tokens: int,
    num_prefill_tokens: int,
    num_decode_tokens: int,
    use_fp4_cache: bool,
) -> torch.Tensor:
    schedule_metadata = (
        decode_schedule_metadata if decode_schedule_metadata.numel() > 0 else None
    )
    context_lens = decode_context_lens if decode_context_lens.numel() > 0 else None
    decode_blocks = decode_block_table if decode_block_table.numel() > 0 else None
    return _deepseek_v4_sparse_attn_indexer_native(
        cache_2d=cache_2d,
        positions=positions,
        token_to_req_indices=token_to_req_indices,
        block_table=block_table,
        seq_lens_cpu=seq_lens_cpu,
        query_lens_cpu=query_lens_cpu,
        prefill_chunk_specs=prefill_chunk_specs,
        prefill_chunk_offsets=prefill_chunk_offsets,
        prefill_slots=prefill_slots,
        prefill_cu_seq_lens=prefill_cu_seq_lens,
        prefill_cu_seqlen_k_start=prefill_cu_seqlen_k_start,
        prefill_cu_seqlen_k_end=prefill_cu_seqlen_k_end,
        prefill_seq_lens_k=prefill_seq_lens_k,
        packed_q_values=packed_q_values,
        packed_q_scales=packed_q_scales,
        packed_weights=packed_weights,
        decode_schedule_metadata=schedule_metadata,
        decode_context_lens=context_lens,
        decode_block_table=decode_blocks,
        decode_max_context_len=decode_max_context_len,
        topk_indices_buffer=topk_indices_buffer,
        prefill_gather_values_workspace=prefill_gather_values_workspace,
        prefill_gather_scales_workspace=prefill_gather_scales_workspace,
        persistent_topk_workspace=persistent_topk_workspace,
        cache_block_size=cache_block_size,
        compress_ratio=compress_ratio,
        topk_tokens=topk_tokens,
        num_prefill_tokens=num_prefill_tokens,
        num_decode_tokens=num_decode_tokens,
        use_fp4_cache=use_fp4_cache,
    )


def _deepseek_v4_sparse_attn_indexer_fake(
    cache_2d: torch.Tensor,
    positions: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    query_lens_cpu: torch.Tensor,
    prefill_chunk_specs: torch.Tensor,
    prefill_chunk_offsets: torch.Tensor,
    prefill_slots: torch.Tensor,
    prefill_cu_seq_lens: torch.Tensor,
    prefill_cu_seqlen_k_start: torch.Tensor,
    prefill_cu_seqlen_k_end: torch.Tensor,
    prefill_seq_lens_k: torch.Tensor,
    packed_q_values: torch.Tensor,
    packed_q_scales: torch.Tensor,
    packed_weights: torch.Tensor,
    decode_schedule_metadata: torch.Tensor,
    decode_context_lens: torch.Tensor,
    decode_block_table: torch.Tensor,
    decode_max_context_len: int,
    topk_indices_buffer: torch.Tensor,
    prefill_gather_values_workspace: torch.Tensor,
    prefill_gather_scales_workspace: torch.Tensor,
    persistent_topk_workspace: torch.Tensor,
    cache_block_size: int,
    compress_ratio: int,
    topk_tokens: int,
    num_prefill_tokens: int,
    num_decode_tokens: int,
    use_fp4_cache: bool,
) -> torch.Tensor:
    del (
        cache_2d,
        positions,
        token_to_req_indices,
        block_table,
        seq_lens_cpu,
        query_lens_cpu,
        prefill_chunk_specs,
        prefill_chunk_offsets,
        prefill_slots,
        prefill_cu_seq_lens,
        prefill_cu_seqlen_k_start,
        prefill_cu_seqlen_k_end,
        prefill_seq_lens_k,
        packed_q_values,
        packed_q_scales,
        packed_weights,
        decode_schedule_metadata,
        decode_context_lens,
        decode_block_table,
        decode_max_context_len,
        cache_block_size,
        prefill_gather_values_workspace,
        prefill_gather_scales_workspace,
        persistent_topk_workspace,
        compress_ratio,
        topk_tokens,
        num_prefill_tokens,
        num_decode_tokens,
        use_fp4_cache,
    )
    return topk_indices_buffer


direct_register_custom_op(
    op_name="deepseek_v4_sparse_attn_indexer",
    op_func=_deepseek_v4_sparse_attn_indexer_op,
    mutates_args=[
        "topk_indices_buffer",
        "prefill_gather_values_workspace",
        "prefill_gather_scales_workspace",
        "persistent_topk_workspace",
    ],
    fake_impl=_deepseek_v4_sparse_attn_indexer_fake,
)


def _deepseek_v4_sparse_attn_indexer(
    *,
    indexer_metadata: DeepseekV4SparseIndexerMetadata,
    indexer_cache: torch.Tensor,
    indexer_page_table: torch.Tensor,
    indexer_block_size: int,
    compress_ratio: int,
    packed_q_values: torch.Tensor,
    packed_q_scales: torch.Tensor,
    packed_weights: torch.Tensor,
    topk_indices_buffer: torch.Tensor,
    prefill_gather_values_workspace: torch.Tensor,
    prefill_gather_scales_workspace: torch.Tensor,
    persistent_topk_workspace: torch.Tensor,
    topk_tokens: int,
    use_fp4_cache: bool = True,
) -> torch.Tensor:
    batch_metadata = indexer_metadata.batch_metadata
    prefill_metadata = indexer_metadata.prefill_metadata
    if batch_metadata is None or prefill_metadata is None:
        raise RuntimeError("DeepSeek V4 sparse indexer metadata is not prepared")

    positions = batch_metadata.positions
    decode_plan = indexer_metadata.decode_plan
    decode_schedule_metadata = indexer_metadata.decode_schedule_metadata
    decode_context_lens = None if decode_plan is None else decode_plan.context_lens
    decode_block_table = None if decode_plan is None else decode_plan.page_table
    decode_max_context_len = 0 if decode_plan is None else decode_plan.max_context_len
    if decode_schedule_metadata is None:
        decode_schedule_metadata = torch.empty(
            0,
            dtype=torch.int32,
            device=positions.device,
        )
    if decode_context_lens is None:
        decode_context_lens = torch.empty(
            (0, 1),
            dtype=torch.int32,
            device=positions.device,
        )
    if decode_block_table is None:
        decode_block_table = torch.empty(
            (0, 1),
            dtype=indexer_page_table.dtype,
            device=indexer_page_table.device,
        )
    return torch.ops.tokenspeed.deepseek_v4_sparse_attn_indexer(
        indexer_cache,
        positions,
        batch_metadata.token_to_req_indices,
        indexer_page_table,
        batch_metadata.seq_lens_cpu,
        batch_metadata.query_lens_cpu,
        prefill_metadata.chunk_specs,
        prefill_metadata.chunk_offsets,
        prefill_metadata.slots,
        prefill_metadata.cu_seq_lens,
        prefill_metadata.cu_seqlen_k_start,
        prefill_metadata.cu_seqlen_k_end,
        prefill_metadata.seq_lens_k,
        packed_q_values,
        packed_q_scales,
        packed_weights,
        decode_schedule_metadata,
        decode_context_lens,
        decode_block_table,
        decode_max_context_len,
        topk_indices_buffer,
        prefill_gather_values_workspace,
        prefill_gather_scales_workspace,
        persistent_topk_workspace,
        indexer_block_size,
        compress_ratio,
        topk_tokens,
        batch_metadata.num_prefill_tokens,
        batch_metadata.num_decode_tokens,
        use_fp4_cache,
    )


def _deepseek_v4_mega_moe_max_num_tokens() -> int:
    override = int(
        global_server_args_dict.get("deepseek_v4_mega_moe_max_num_tokens", 0) or 0
    )
    if override > 0:
        return override

    candidates = [
        global_server_args_dict.get("chunked_prefill_size", 0),
        global_server_args_dict.get("prefill_graph_max_tokens", 0),
        global_server_args_dict.get("max_cudagraph_capture_size", 0),
        global_server_args_dict.get("max_num_seqs", 0),
    ]
    return max([int(value or 0) for value in candidates] + [1])


class _DeepseekV4TopKBuffer:
    def __init__(self, topk_tokens: int) -> None:
        self.topk_tokens = topk_tokens
        self.buffer: torch.Tensor | None = None

    def get(self, num_tokens: int, device: torch.device) -> torch.Tensor:
        rows = max(num_tokens, _deepseek_v4_mega_moe_max_num_tokens())
        needs_alloc = (
            self.buffer is None
            or self.buffer.device != device
            or self.buffer.shape[0] < rows
            or self.buffer.shape[1] != self.topk_tokens
        )
        if needs_alloc:
            if get_is_capture_mode():
                raise RuntimeError(
                    "DeepSeek V4 top-k buffer must be allocated before CUDA graph "
                    "capture"
                )
            self.buffer = torch.empty(
                (rows, self.topk_tokens),
                dtype=torch.int32,
                device=device,
            )
        return self.buffer[:num_tokens]


def _deepseek_v4_indexer_page_table(
    metadata: DeepseekV4ForwardMetadata,
    compress_ratio: int,
    indexer_block_size: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    page_table = metadata.cache.compressed_page_table(
        compress_ratio,
        indexer_block_size,
    )
    base_offsets = None
    if page_table is not metadata.cache.page_table:
        base_offsets = metadata.cache.block_table_base_offsets.get(
            v4_compressed_kv_group_id(compress_ratio)
        )
    return page_table, base_offsets


def _deepseek_v4_sanitize_swa_slot_mapping(
    slot_mapping: torch.Tensor,
    capacity: int,
    is_valid_token: torch.Tensor | None = None,
) -> torch.Tensor:
    # Fail closed: with no writable SWA capacity every slot is masked, so an
    # unchecked mapping can never reach the fused cache-insert kernels.
    if capacity <= 0:
        return torch.full_like(slot_mapping, -1)
    slot_mapping = _mask_invalid_graph_tokens(slot_mapping, is_valid_token)
    valid = (slot_mapping >= 0) & (slot_mapping < capacity)
    return torch.where(
        valid,
        slot_mapping,
        torch.full_like(slot_mapping, -1),
    )


def _deepseek_v4_swa_slot_mapping(
    ctx: ForwardContext,
    positions: torch.Tensor,
    out_cache_loc: torch.Tensor,
) -> torch.Tensor:
    """Build the SWA write-slot mapping, already sanitized for cache inserts.

    The returned mapping has invalid CUDA-graph tokens and out-of-capacity
    slots masked to -1, so per-layer SWA inserts can consume it directly.
    Sanitizing here keeps the mask/clamp elementwise chain at once per step;
    doing it per layer previously baked ~7 tiny kernels x 61 layers into the
    captured decode graph.
    """
    if positions.numel() == 0:
        return out_cache_loc
    metadata = _deepseek_v4_forward_metadata(ctx)
    if metadata is None:
        raise RuntimeError("DeepSeek V4 attention requires forward metadata")
    cache_metadata = metadata.cache
    token_to_req_indices = metadata.token_to_req_indices[: positions.numel()]
    if cache_metadata.swa_page_table is None:
        slot_mapping = out_cache_loc
    elif token_to_req_indices.numel() != positions.numel() and (
        token_to_req_indices.numel() <= 0
        or positions.numel() % token_to_req_indices.numel() != 0
    ):
        slot_mapping = out_cache_loc
    else:
        slot_mapping = _group_slot_mapping_from_raw(
            positions,
            token_to_req_indices,
            cache_metadata.swa_page_table,
            ctx.token_to_kv_pool.swa_block_size,
            base_offsets=cache_metadata.swa_base_logical_page,
        )
    is_valid_token = getattr(metadata, "is_valid_token", None)
    if is_valid_token is not None:
        is_valid_token = is_valid_token[: positions.numel()]
    # Attribute access is deliberately unguarded: a pool without
    # swa_capacity_slots must fail fast here rather than skip the bounds
    # check that protects the fused cache-insert kernels.
    return _deepseek_v4_sanitize_swa_slot_mapping(
        slot_mapping,
        ctx.token_to_kv_pool.swa_capacity_slots,
        is_valid_token,
    )


def _attention_use_fp4_indexer_cache(config: PretrainedConfig) -> bool:
    override = global_server_args_dict.get("attention_use_fp4_indexer_cache", None)
    attention_config = getattr(config, "attention_config", None)
    if isinstance(attention_config, dict):
        configured = attention_config.get("use_fp4_indexer_cache", None)
    else:
        configured = getattr(attention_config, "use_fp4_indexer_cache", None)
    requested = override if override is not None else configured
    return dsv4_indexer_cache_format(requested) == "mxfp4"


def deepseek_v4_rope_config(
    config: PretrainedConfig, compress_ratio: int
) -> tuple[float, dict | None]:
    """Return the per-layer DeepSeek V4 RoPE base and scaling config.

    DeepSeek V4 uses ordinary RoPE for SWA-only layers. Compressed layers
    use the checkpoint's separate `compress_rope_theta` together with YaRN.
    """

    if compress_ratio <= 1:
        return float(getattr(config, "rope_theta", 10000.0)), None

    rope_scaling = getattr(config, "rope_scaling", None)
    if rope_scaling is not None:
        rope_scaling = dict(rope_scaling)
        rope_scaling["rope_type"] = "deepseek_yarn"
        rope_scaling["mscale"] = 0
        rope_scaling["mscale_all_dim"] = 0
    return (
        float(
            getattr(
                config,
                "compress_rope_theta",
                getattr(config, "rope_theta", 10000.0),
            )
        ),
        rope_scaling,
    )


class DeepseekV4MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        mapping: Mapping,
        quant_config: QuantizationConfig | None,
        prefix: str,
        swiglu_limit: float | None = None,
        reduce_results: bool = False,
    ) -> None:
        super().__init__()
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}")
        tp = mapping.dense
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            tp_rank=tp.tp_rank,
            tp_size=tp.tp_size,
            tp_group=tp.tp_group,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            reduce_results=reduce_results,
            tp_rank=tp.tp_rank,
            tp_size=tp.tp_size,
            tp_group=tp.tp_group,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
        )
        self.swiglu_limit = swiglu_limit
        self.act_fn = SiluAndMul(swiglu_limit)
        self.reduce_results = reduce_results
        self.tp_rank = tp.tp_rank
        self.tp_size = tp.tp_size
        self.tp_group = tp.tp_group

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] == 0:
            return x.new_empty((0, self.down_proj.output_size))
        gate_up, _ = self.gate_up_proj(x)
        out, _ = self.down_proj.forward_with_activation(gate_up, self.act_fn)
        return out


def dsv4_select_experts(
    router_logits: torch.Tensor,
    top_k: int,
    renormalize: bool,
    correction_bias: torch.Tensor | None = None,
    hash_indices_table: torch.Tensor | None = None,
    input_ids: torch.Tensor | None = None,
    need_scores: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Use an accelerator router when available, otherwise run eager routing."""
    try:
        return _kernel_dsv4_select_experts(
            router_logits,
            top_k,
            renormalize,
            correction_bias,
            hash_indices_table,
            input_ids,
            need_scores,
        )
    except (NoKernelFoundError, AttributeError, RuntimeError):
        pass

    scores = torch.sqrt(F.softplus(router_logits.float()))
    if hash_indices_table is not None:
        if input_ids is None:
            raise ValueError("hash-routed DeepSeek V4 MoE requires input_ids")
        table = hash_indices_table.to(device=scores.device, dtype=torch.int64)
        ids = input_ids.reshape(-1).to(device=scores.device, dtype=torch.int64)
        topk_ids = table[ids]
    else:
        scores_for_choice = scores
        if correction_bias is not None:
            scores_for_choice = scores_for_choice + correction_bias.to(
                device=scores.device,
                dtype=scores.dtype,
            ).unsqueeze(0)
        topk_ids = torch.topk(
            scores_for_choice,
            k=top_k,
            dim=-1,
            sorted=True,
        ).indices

    topk_weights = scores.gather(1, topk_ids.long())
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(torch.finfo(topk_weights.dtype).tiny)
    return topk_weights.to(torch.float32), topk_ids.to(torch.int32), scores


def dsv4_linear_fp32(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    enable_pdl: bool = False,
) -> torch.Tensor:
    """Use the registered accelerator projection or an eager FP32 fallback."""
    try:
        return _kernel_dsv4_linear_fp32(
            hidden_states,
            weight,
            enable_pdl=enable_pdl,
        )
    except NoKernelFoundError:
        return F.linear(hidden_states.float(), weight.float())


class DeepseekV4MoEGate(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_index: int,
        hash_indices_dtype: torch.dtype = torch.int32,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(config.n_routed_experts, config.hidden_size)
        )
        self.is_hash_moe = layer_index < config.num_hash_layers
        if self.is_hash_moe:
            self.tid2eid = nn.Parameter(
                torch.empty(
                    config.vocab_size,
                    config.num_experts_per_tok,
                    dtype=hash_indices_dtype,
                ),
                requires_grad=False,
            )
            self.e_score_correction_bias = None
        elif getattr(config, "topk_method", None) == "noaux_tc":
            self.register_parameter("tid2eid", None)
            self.e_score_correction_bias = nn.Parameter(
                torch.empty(config.n_routed_experts, dtype=torch.float32),
                requires_grad=False,
            )
        else:
            self.register_parameter("tid2eid", None)
            self.e_score_correction_bias = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return dsv4_linear_fp32(
            hidden_states,
            self.weight,
            enable_pdl=pdl_enabled(),
        )


class DeepseekV4MegaMoEExperts(nn.Module):
    def __init__(
        self,
        *,
        num_experts: int,
        num_local_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        mapping: Mapping,
        prefix: str,
        swiglu_limit: float | None,
    ) -> None:
        super().__init__()
        self.prefix = prefix
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_num_tokens = _deepseek_v4_mega_moe_max_num_tokens()
        process_group = (
            None
            if mapping is None
            else pg_manager.get_device_process_group(mapping.moe.tp_ep_group)
        )
        self._plan = dsv4_mega_moe_plan(
            num_experts=num_experts,
            num_local_experts=num_local_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            max_num_tokens=self.max_num_tokens,
            process_group=process_group,
            activation_clamp=swiglu_limit,
        )
        self._processed_state: object | None = None

        weight_attrs = {"weight_loader": self.weight_loader}
        self.w13_weight = nn.Parameter(
            torch.zeros(
                num_local_experts,
                2 * intermediate_size,
                hidden_size // 2,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        set_weight_attrs(self.w13_weight, weight_attrs)

        self.w13_weight_scale = nn.Parameter(
            torch.zeros(
                num_local_experts,
                2 * intermediate_size,
                hidden_size // DEEPSEEK_V4_MXFP4_BLOCK_SIZE,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        set_weight_attrs(self.w13_weight_scale, weight_attrs)

        self.w2_weight = nn.Parameter(
            torch.zeros(
                num_local_experts,
                hidden_size,
                intermediate_size // 2,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        set_weight_attrs(self.w2_weight, weight_attrs)

        self.w2_weight_scale = nn.Parameter(
            torch.zeros(
                num_local_experts,
                hidden_size,
                intermediate_size // DEEPSEEK_V4_MXFP4_BLOCK_SIZE,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        set_weight_attrs(self.w2_weight_scale, weight_attrs)

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        shard_id: str,
        local_expert_id: int,
    ) -> None:
        expert_data = param.data[local_expert_id]
        if shard_id in ("w1", "w3"):
            if param is not self.w13_weight and param is not self.w13_weight_scale:
                raise ValueError(f"Unexpected MegaMoE w13 shard target: {shard_id}")
            shard_offset = 0 if shard_id == "w1" else self.intermediate_size
            expert_data = expert_data.narrow(0, shard_offset, self.intermediate_size)
        elif shard_id == "w2":
            if param is not self.w2_weight and param is not self.w2_weight_scale:
                raise ValueError(f"Unexpected MegaMoE w2 shard target: {shard_id}")
        else:
            raise ValueError(f"Unsupported DeepSeek V4 MegaMoE shard id: {shard_id}")

        if expert_data.dtype == torch.uint8 and loaded_weight.dtype == getattr(
            torch, "float8_e8m0fnu", None
        ):
            loaded_weight = loaded_weight.view(torch.uint8)
        if expert_data.shape != loaded_weight.shape:
            raise ValueError(
                f"DeepSeek V4 MegaMoE expert weight shape mismatch for "
                f"{self.prefix}: parameter shard {tuple(expert_data.shape)} "
                f"vs checkpoint {tuple(loaded_weight.shape)}"
            )
        expert_data.copy_(loaded_weight)

    def finalize_weights(self) -> None:
        if self._processed_state is not None:
            return
        self._processed_state = dsv4_mega_moe_process_weights(
            self._plan,
            self.w13_weight.data,
            self.w13_weight_scale.data,
            self.w2_weight.data,
            self.w2_weight_scale.data,
        )

        del self.w13_weight
        del self.w13_weight_scale
        del self.w2_weight
        del self.w2_weight_scale

    def warmup(self) -> None:
        if self._processed_state is not None:
            dsv4_mega_moe_warmup(self._plan, self._processed_state)

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        if self._processed_state is None:
            raise RuntimeError(
                "DeepseekV4MegaMoEExperts.finalize_weights() must run via "
                "post_load_weights() before forward()"
            )
        return dsv4_mega_moe_apply(
            self._plan,
            self._processed_state,
            hidden_states,
            topk_weights,
            topk_ids,
        )


def _deepseek_v4_routed_expert_quant_config(
    config: PretrainedConfig,
    quant_config: QuantizationConfig,
) -> tuple[QuantizationConfig, bool]:
    # ``quant_method=fp8`` describes the model-wide quantization, but DeepSeek
    # V4 can still serialize routed experts as packed MXFP4. Prefer the explicit
    # expert contract so packed FP4 tensors are never allocated as block FP8.
    is_block_fp8 = isinstance(quant_config, Fp8Config) and (
        getattr(config, "expert_dtype", None) != "fp4"
    )
    if is_block_fp8:
        return quant_config, True
    return (
        Mxfp4Config(
            ignored_layers=quant_config.ignored_layers,
            is_checkpoint_mxfp4_serialized=True,
        ),
        False,
    )


def _deepseek_v4_expert_scale_parameter_name(
    config: PretrainedConfig,
    *,
    use_mega_moe: bool,
) -> str:
    if use_mega_moe or getattr(config, "expert_dtype", None) == "fp4":
        return "weight_scale"
    return "weight_scale_inv"


class DeepseekV4MoE(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mapping: Mapping,
        quant_config: QuantizationConfig | None,
        layer_index: int,
        prefix: str,
        aux_stream: object | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.mapping = mapping
        self.layer_index = layer_index
        self.n_shared_experts = config.n_shared_experts
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)
        self.scoring_func = getattr(config, "scoring_func", "sqrtsoftplus")
        if self.scoring_func != "sqrtsoftplus":
            raise ValueError(
                f"Unsupported DeepSeek V4 MoE scoring: {self.scoring_func}"
            )
        self.stream_fork = StreamFork(aux_stream)

        self.use_mega_moe = get_moe_backend().is_mega_moe()
        if self.use_mega_moe:
            if mapping.moe.ep_size <= 1:
                raise ValueError("DeepSeek V4 MegaMoE requires expert parallelism.")
            if mapping.moe.tp_size != 1:
                raise ValueError("DeepSeek V4 MegaMoE does not support mixed TP/EP.")
            if global_server_args_dict.get("ep_num_redundant_experts", 0):
                raise ValueError(
                    "DeepSeek V4 MegaMoE does not support redundant EP experts."
                )
            if config.n_routed_experts % mapping.moe.ep_size != 0:
                raise ValueError(
                    "DeepSeek V4 MegaMoE requires n_routed_experts divisible by "
                    "EP size."
                )
        self.hash_indices_dtype = torch.int64 if self.use_mega_moe else torch.int32
        self.gate = DeepseekV4MoEGate(
            config,
            layer_index,
            hash_indices_dtype=self.hash_indices_dtype,
        )

        if config.n_shared_experts is not None:
            self.shared_experts = DeepseekV4MLP(
                config.hidden_size,
                config.moe_intermediate_size * config.n_shared_experts,
                config.hidden_act,
                mapping,
                quant_config,
                add_prefix("shared_experts", prefix),
                swiglu_limit=getattr(config, "swiglu_limit", None),
                reduce_results=False,
            )
        else:
            self.shared_experts = None

        if self.use_mega_moe:
            self.experts = DeepseekV4MegaMoEExperts(
                num_experts=config.n_routed_experts,
                num_local_experts=config.n_routed_experts // mapping.moe.ep_size,
                top_k=config.num_experts_per_tok,
                hidden_size=config.hidden_size,
                intermediate_size=config.moe_intermediate_size,
                mapping=mapping,
                prefix=add_prefix("experts", prefix),
                swiglu_limit=getattr(config, "swiglu_limit", None),
            )
            self.topk = None
        else:
            routed_quant_config, is_block_fp8 = _deepseek_v4_routed_expert_quant_config(
                config, quant_config
            )
            self.experts = MoELayer(
                top_k=config.num_experts_per_tok,
                num_experts=config.n_routed_experts
                + global_server_args_dict["ep_num_redundant_experts"],
                hidden_size=config.hidden_size,
                intermediate_size=config.moe_intermediate_size,
                quant_config=routed_quant_config,
                layer_index=layer_index,
                prefix=prefix,
                tp_rank=mapping.moe.tp_rank,
                tp_size=mapping.moe.tp_size,
                ep_rank=mapping.moe.ep_rank,
                ep_size=mapping.moe.ep_size,
                activation="swiglu",
                swiglu_limit=getattr(config, "swiglu_limit", None),
                with_bias=not is_block_fp8,
                routing_mode="precomputed_topk",
                routing_config={
                    "routed_scaling_factor": self.routed_scaling_factor,
                    "normalize_topk_weights": config.norm_topk_prob,
                    "correction_bias": self.gate.e_score_correction_bias,
                    "routing_method_type": RoutingMethodType.Renormalize,
                },
            )
            self.topk = TopK(
                top_k=config.num_experts_per_tok,
                renormalize=config.norm_topk_prob,
                correction_bias=self.gate.e_score_correction_bias,
                routed_scaling_factor=self.routed_scaling_factor,
                output_format=self.experts.topk_output_format,
            )

    def _select_experts(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        router_logits = self.gate(hidden_states)
        fmt = getattr(self.experts, "topk_output_format", None)
        need_scores = fmt is not None and not fmt.is_bypassed()
        return dsv4_select_experts(
            router_logits,
            self.config.num_experts_per_tok,
            self.config.norm_topk_prob,
            correction_bias=self.gate.e_score_correction_bias,
            hash_indices_table=self.gate.tid2eid,
            input_ids=input_ids,
            need_scores=need_scores,
        )

    def _make_topk_output(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        router_scores: torch.Tensor,
    ):
        if self.experts.topk_output_format.is_bypassed():
            router_logits = pack_topk_as_router_logits(
                topk_weights, topk_ids, self.config.n_routed_experts
            )
            return BypassedTopKOutput(
                hidden_states, router_logits, self.topk.topk_config
            )
        return StandardTopKOutput(topk_weights, topk_ids, router_scores)

    def _forward_shared_experts(
        self,
        hidden_states: torch.Tensor,
        ctx: ForwardContext | None = None,
        comm_manager: CommManager | None = None,
    ) -> torch.Tensor | None:
        if self.shared_experts is None:
            return None
        if comm_manager is not None:
            hidden_states = comm_manager.pre_dense_comm(hidden_states, ctx)
        if hidden_states.shape[0] == 0:
            return None
        with nvtx_range("moe_shared_experts"):
            shared = self.shared_experts(hidden_states)
        if comm_manager is not None:
            shared, _ = comm_manager.post_dense_comm(shared, None, ctx)
        return shared

    def forward_mega_moe(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        ctx: ForwardContext,
        comm_manager: CommManager,
    ) -> torch.Tensor:
        if hidden_states.shape[0] == 0:
            topk_weights = hidden_states.new_empty(
                (0, self.config.num_experts_per_tok), dtype=torch.float32
            )
            topk_ids = torch.empty(
                (0, self.config.num_experts_per_tok),
                device=hidden_states.device,
                dtype=torch.int64,
            )
        else:
            with nvtx_range("moe_select_experts"):
                topk_weights, topk_ids, _ = self._select_experts(
                    hidden_states, input_ids
                )

        shared = None
        with self.stream_fork.scope(enable=get_is_capture_mode()) as fork:
            with nvtx_range("moe_mega_experts"):
                if self.routed_scaling_factor != 1.0:
                    topk_weights = topk_weights * self.routed_scaling_factor
                routed = self.experts(
                    hidden_states,
                    topk_weights,
                    topk_ids,
                )
            with fork.branch():
                shared = self._forward_shared_experts(
                    hidden_states,
                    ctx=ctx,
                    comm_manager=comm_manager,
                )
        return routed + shared if shared is not None else routed

    def forward_normal(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        num_global_tokens: int,
        max_num_tokens_per_gpu: int,
    ) -> torch.Tensor:
        if hidden_states.shape[0] == 0:
            return hidden_states
        with nvtx_range("moe_select_experts"):
            topk_weights, topk_ids, router_scores = self._select_experts(
                hidden_states, input_ids
            )
        with nvtx_range("moe_make_topk_output"):
            topk_output = self._make_topk_output(
                hidden_states, topk_weights, topk_ids, router_scores
            )
        shared = None
        with self.stream_fork.scope(enable=get_is_capture_mode()) as fork:
            with nvtx_range("moe_experts"):
                routed = self.experts(
                    hidden_states=hidden_states,
                    topk_output=topk_output,
                    num_global_tokens=num_global_tokens,
                    max_num_tokens_per_gpu=max_num_tokens_per_gpu,
                )
                if self.routed_scaling_factor != 1.0:
                    routed *= self.routed_scaling_factor
            with fork.branch():
                shared = self._forward_shared_experts(hidden_states)
        return routed + shared if shared is not None else routed

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        num_global_tokens: int,
        max_num_tokens_per_gpu: int,
        ctx: ForwardContext | None = None,
        comm_manager: CommManager | None = None,
    ) -> torch.Tensor:
        if self.use_mega_moe:
            return self.forward_mega_moe(
                hidden_states,
                input_ids,
                ctx,
                comm_manager,
            )
        return self.forward_normal(
            hidden_states, input_ids, num_global_tokens, max_num_tokens_per_gpu
        )


class DeepseekV4Compressor(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        head_dim: int,
        compress_ratio: int,
        prefix: str,
    ) -> None:
        super().__init__()
        self.compress_ratio = compress_ratio
        self.head_dim = head_dim
        self.overlap = compress_ratio == 4
        self.coff = 2 if self.overlap else 1
        state_dtype = torch.float32
        self.ape = nn.Parameter(
            torch.empty(compress_ratio, self.coff * head_dim, dtype=state_dtype),
            requires_grad=False,
        )
        self._ape_reordered = False
        self.fused_wkv_wgate = MergedColumnParallelLinear(
            hidden_size,
            [self.coff * head_dim, self.coff * head_dim],
            bias=False,
            quant_config=None,
            prefix=add_prefix("fused_wkv_wgate", prefix),
        )
        self.norm = RMSNorm(head_dim, eps=config.rms_norm_eps)

    def process_weights_after_loading(self, module=None) -> None:
        del module
        if not self.overlap or self._ape_reordered:
            return
        with torch.no_grad():
            self.ape.data.copy_(_deepseek_v4_reorder_c4_ape_2604(self.ape.data))
        self._ape_reordered = True

    def compute_kv_score(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Pre-compute the input GEMM (fused_wkv_wgate @ hidden_states).

        Can be launched on an aux stream in parallel with the main Q/KV
        projection so that ``forward()`` only needs the lightweight state
        update + cache write phase.
        """
        weight_shape = (
            self.fused_wkv_wgate.output_size_per_partition,
            self.fused_wkv_wgate.input_size,
        )
        weight = self.fused_wkv_wgate.weight.view(*weight_shape)
        if weight.dtype == torch.float8_e4m3fn:
            weight = _dequant_fp8_weight(self.fused_wkv_wgate, weight_shape)
        return dsv4_linear_fp32(hidden_states, weight)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
        layer_index: int,
        cos_sin_cache: torch.Tensor,
        *,
        state_cache: torch.Tensor | None = None,
        state_block_table: torch.Tensor | None = None,
        state_block_size: int | None = None,
        state_base_logical_page: torch.Tensor | None = None,
        state_slot_mapping: torch.Tensor | None = None,
        write_compressed_cache: bool = True,
        compressor_slot_cache: dict | None = None,
        kv_score: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pool = ctx.token_to_kv_pool
        metadata = _deepseek_v4_forward_metadata(ctx)
        if metadata is None:
            raise RuntimeError("DeepSeek V4 compressor requires forward metadata")
        profile_prefix = (
            f"indexer_compressor_c{self.compress_ratio}"
            if not write_compressed_cache
            else f"compressor_c{self.compress_ratio}"
        )
        if kv_score is None:
            with nvtx_range(f"{profile_prefix}_dequant_weight"):
                weight_shape = (
                    self.fused_wkv_wgate.output_size_per_partition,
                    self.fused_wkv_wgate.input_size,
                )
                weight = self.fused_wkv_wgate.weight.view(*weight_shape)
                if weight.dtype == torch.float8_e4m3fn:
                    weight = _dequant_fp8_weight(self.fused_wkv_wgate, weight_shape)
            with nvtx_range(f"{profile_prefix}_matmul"):
                kv_score = dsv4_linear_fp32(hidden_states, weight)
        kv, score = kv_score.split([self.coff * self.head_dim] * 2, dim=-1)
        if state_cache is None:
            state_cache = pool.get_compressor_state_buffer(layer_index)
        cache_metadata = metadata.cache
        # state/compressed slot mappings depend only on (per-step state, ratio), so reuse
        # them across layers of the same ratio within a step. Attn-compressor path only:
        # the indexer-compressor passes an explicit state_block_table and is excluded.
        memo = compressor_slot_cache if state_block_table is None else None
        if state_block_table is None:
            state_block_table = cache_metadata.compressor_state_block_tables.get(
                self.compress_ratio
            )
            state_base_logical_page = (
                cache_metadata.compressor_state_base_logical_pages.get(
                    self.compress_ratio
                )
            )
        if state_block_size is None:
            state_block_size = (
                pool.get_compressor_state_block_size(layer_index)
                if state_block_table is not None
                else pool.state_block_size
            )
        valid_token = (
            metadata.is_valid_token[: positions.numel()]
            if getattr(metadata, "is_valid_token", None) is not None
            else None
        )
        if state_slot_mapping is None:
            state_hit = (
                memo.get(("state", self.compress_ratio)) if memo is not None else None
            )
            if state_hit is not None:
                state_slot_mapping, state_block_table = state_hit
            else:
                if state_block_table is None:
                    raise RuntimeError(
                        "DeepSeek V4 missing paged-cache block table for compressor "
                        f"state ratio={self.compress_ratio}"
                    )
                state_slot_mapping = _group_slot_mapping_from_raw(
                    positions,
                    metadata.token_to_req_indices[: positions.numel()],
                    state_block_table,
                    state_block_size,
                    base_offsets=state_base_logical_page,
                )
                state_slot_mapping = _mask_invalid_graph_tokens(
                    state_slot_mapping,
                    valid_token,
                )
                if memo is not None:
                    memo[("state", self.compress_ratio)] = (
                        state_slot_mapping,
                        state_block_table,
                    )
        with nvtx_range(f"{profile_prefix}_save_state"):
            save_deepseek_v4_compressor_state(
                kv=kv,
                score=score,
                ape=self.ape,
                state_cache=state_cache,
                slot_mapping=state_slot_mapping,
                positions=positions,
                block_size=state_block_size,
                compress_ratio=self.compress_ratio,
            )
        if not write_compressed_cache:
            return kv, score

        kv_cache_block_size = pool.get_compressed_block_size(layer_index)
        compressed_hit = (
            memo.get(("compressed", self.compress_ratio)) if memo is not None else None
        )
        if compressed_hit is not None:
            compressed_slots = compressed_hit
        else:
            with nvtx_range(f"{profile_prefix}_compressed_slot_mapping"):
                compressed_slots = cache_metadata.compressed_slot_mapping(
                    positions,
                    self.compress_ratio,
                    token_to_req_indices=metadata.token_to_req_indices[
                        : positions.numel()
                    ],
                    query_start_loc=metadata.query_start_loc,
                    seq_lens=metadata.seq_lens,
                    kv_cache_block_size=kv_cache_block_size,
                    use_decode_cache=(
                        ctx.forward_mode is not None and ctx.forward_mode.is_decode()
                    ),
                    is_valid_token=valid_token,
                )
            if memo is not None:
                memo[("compressed", self.compress_ratio)] = compressed_slots
        with nvtx_range(f"{profile_prefix}_cache_insert"):
            insert = (
                deepseek_v4_csa_compress_kv_cache_insert
                if self.compress_ratio == 4
                else deepseek_v4_hca_compress_kv_cache_insert
            )
            insert(
                state_cache=state_cache,
                token_to_req_indices=metadata.token_to_req_indices[: positions.numel()],
                positions=positions,
                compressor_slot_mapping=state_slot_mapping,
                block_table=state_block_table,
                block_table_base_offsets=state_base_logical_page,
                compressor_block_size=state_block_size,
                rms_norm_weight=self.norm.weight,
                rms_norm_eps=self.norm.variance_epsilon,
                cos_sin_cache=cos_sin_cache,
                kv_cache_2d=pool.get_compressed_kv_buffer_2d(layer_index),
                kv_slot_mapping=compressed_slots,
                kv_cache_block_size=kv_cache_block_size,
                compress_ratio=self.compress_ratio,
            )
        return kv, score


class DeepseekV4Indexer(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mapping: Mapping,
        quant_config: QuantizationConfig | None,
        prefix: str,
        compress_ratio: int,
        topk_buffer: _DeepseekV4TopKBuffer | None = None,
    ) -> None:
        super().__init__()
        self.use_fp4_cache = _attention_use_fp4_indexer_cache(config)
        self.wq_b = ReplicatedLinear(
            config.q_lora_rank,
            config.index_n_heads * config.index_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("wq_b", prefix),
        )
        self.weights_proj = ReplicatedLinear(
            config.hidden_size,
            config.index_n_heads,
            bias=False,
            quant_config=None,
            prefix=add_prefix("weights_proj", prefix),
        )
        self.compressor = DeepseekV4Compressor(
            config,
            config.hidden_size,
            config.index_head_dim,
            compress_ratio,
            add_prefix("compressor", prefix),
        )
        self.compress_ratio = compress_ratio
        self.n_head = int(config.index_n_heads)
        self.head_dim = int(config.index_head_dim)
        self.topk_tokens = int(config.index_topk)
        self.topk_buffer = topk_buffer
        self.softmax_scale = self.head_dim**-0.5
        value_bytes = deepseek_v4_indexer_mxfp4_value_bytes(self.head_dim)
        scale_bytes = deepseek_v4_indexer_mxfp4_scale_dim(self.head_dim)
        self.register_buffer(
            "_prefill_gather_values_workspace",
            torch.empty((0, value_bytes), dtype=torch.uint8),
            persistent=False,
        )
        self.register_buffer(
            "_prefill_gather_scales_workspace",
            torch.empty((0, scale_bytes), dtype=torch.uint8),
            persistent=False,
        )
        workspace_rows = 1024 * 1024 if self.topk_tokens in (512, 1024, 2048) else 0
        self.register_buffer(
            "_persistent_topk_workspace",
            torch.empty((workspace_rows,), dtype=torch.uint8),
            persistent=False,
        )

    def _prefill_gather_workspace(
        self,
        rows: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rows = max(0, int(rows))
        value_bytes = deepseek_v4_indexer_mxfp4_value_bytes(self.head_dim)
        scale_bytes = deepseek_v4_indexer_mxfp4_scale_dim(self.head_dim)
        if (
            self._prefill_gather_values_workspace.device != device
            or self._prefill_gather_values_workspace.shape[0] < rows
        ):
            self._prefill_gather_values_workspace = torch.empty(
                (rows, value_bytes),
                dtype=torch.uint8,
                device=device,
            )
        if (
            self._prefill_gather_scales_workspace.device != device
            or self._prefill_gather_scales_workspace.shape[0] < rows
        ):
            self._prefill_gather_scales_workspace = torch.empty(
                (rows, scale_bytes),
                dtype=torch.uint8,
                device=device,
            )
        return (
            self._prefill_gather_values_workspace[:rows],
            self._prefill_gather_scales_workspace[:rows],
        )

    def prepare_decode_metadata(
        self,
        *,
        positions: torch.Tensor,
        metadata: DeepseekV4ForwardMetadata,
        ctx: ForwardContext,
        indexer_block_size: int,
    ) -> None:
        if not self.use_fp4_cache:
            return
        forward_mode = ctx.forward_mode
        if forward_mode is not None and forward_mode.is_mixed():
            num_prefill_tokens = int(metadata.num_prefill_tokens)
            num_decode_tokens = metadata.decode_token_count()
        elif forward_mode is not None and forward_mode.is_decode():
            num_prefill_tokens = 0
            num_decode_tokens = positions.numel()
        else:
            return
        if num_decode_tokens <= 0:
            return

        decode_start = num_prefill_tokens
        decode_end = decode_start + num_decode_tokens
        decode_positions = positions[decode_start:decode_end]
        decode_valid_token = (
            metadata.is_valid_token[decode_start:decode_end]
            if getattr(metadata, "is_valid_token", None) is not None
            else None
        )
        indexer_page_table, indexer_page_table_base_offsets = (
            _deepseek_v4_indexer_page_table(
                metadata,
                self.compress_ratio,
                indexer_block_size,
            )
        )
        decode_plan = _deepseek_v4_indexer_decode_plan(
            positions=decode_positions,
            token_to_req_indices=metadata.token_to_req_indices[decode_start:decode_end],
            block_table=indexer_page_table,
            cache_block_size=indexer_block_size,
            compress_ratio=self.compress_ratio,
            metadata=metadata,
            is_valid_token=decode_valid_token,
            block_table_base_offsets=indexer_page_table_base_offsets,
        )
        _deepseek_v4_indexer_decode_schedule_metadata(
            positions=decode_positions,
            cache_block_size=indexer_block_size,
            compress_ratio=self.compress_ratio,
            metadata=metadata,
            context_lens=decode_plan.context_lens,
        )

    def _forward_sparse_indexer_portable(
        self,
        query: torch.Tensor,
        weights: torch.Tensor,
        indexer_cache: torch.Tensor,
        indexer_block_size: int,
        prefill_metadata: DeepseekV4IndexerPrefillMetadata,
        decode_plan: DeepseekV4IndexerDecodePlan | None,
        num_prefill_tokens: int,
        num_decode_tokens: int,
        token_to_req_indices: torch.Tensor,
        block_table_base_offsets: torch.Tensor | None,
        topk_out: torch.Tensor,
    ) -> torch.Tensor:
        topk_out.fill_(-1)
        max_logits_mb = int(
            global_server_args_dict.get(
                "deepseek_v4_indexer_prefill_max_logits_mb", 512
            )
            or 512
        )
        for chunk in prefill_metadata.chunks:
            token_slice = slice(chunk.token_start, chunk.token_end)
            row_slice = slice(chunk.gather_row_start, chunk.gather_row_end)
            slots = prefill_metadata.slots[chunk.slot_start : chunk.slot_end]
            row_starts = prefill_metadata.cu_seqlen_k_start[row_slice]
            row_ends = prefill_metadata.cu_seqlen_k_end[row_slice]
            selected, _ = dsa_prefill_topk(
                query[token_slice],
                weights[token_slice],
                slots,
                row_starts,
                row_ends,
                topk=self.topk_tokens,
                softmax_scale=self.softmax_scale,
                index_k_cache=indexer_cache,
                page_size=indexer_block_size,
                max_logits_bytes=max_logits_mb * 1024 * 1024,
                out=topk_out[token_slice],
            )
            selected_i64 = selected.to(torch.int64)
            row_starts_i64 = row_starts.to(torch.int64)
            if block_table_base_offsets is None:
                base_rows = torch.zeros_like(row_starts_i64)
            else:
                req_indices = token_to_req_indices[token_slice].to(torch.int64)
                safe_req_indices = req_indices.clamp(
                    0, block_table_base_offsets.shape[0] - 1
                )
                base_rows = block_table_base_offsets.to(
                    device=selected.device, dtype=torch.int64
                )[safe_req_indices] * int(indexer_block_size)
            logical = selected_i64 - row_starts_i64[:, None] + base_rows[:, None]
            valid = (selected_i64 >= row_starts_i64[:, None]) & (
                selected_i64 < row_ends.to(torch.int64)[:, None]
            )
            topk_out[token_slice].copy_(
                torch.where(valid, logical, torch.full_like(logical, -1)).to(
                    torch.int32
                )
            )

        if num_decode_tokens <= 0:
            return topk_out
        if decode_plan is None:
            raise RuntimeError("DeepSeek V4 decode indexer plan is missing")
        decode_start = num_prefill_tokens
        decode_end = decode_start + num_decode_tokens
        decode_slice = slice(decode_start, decode_end)
        seq_lens = decode_plan.context_lens[:, -1].to(torch.int32).contiguous()
        decode_base_offsets = None
        if block_table_base_offsets is not None:
            decode_req_indices = token_to_req_indices[decode_slice].to(torch.int64)
            safe_req_indices = decode_req_indices.clamp(
                0, block_table_base_offsets.shape[0] - 1
            )
            decode_base_offsets = block_table_base_offsets.to(
                device=decode_req_indices.device, dtype=torch.int32
            )[safe_req_indices]
            seq_lens = seq_lens + decode_base_offsets * int(indexer_block_size)
        dsa_decode_topk(
            query[decode_slice],
            weights[decode_slice],
            seq_lens,
            decode_plan.page_table,
            page_size=indexer_block_size,
            topk=self.topk_tokens,
            softmax_scale=self.softmax_scale,
            index_k_cache=indexer_cache,
            topk_layout="logical_offsets",
            block_table_base_offsets=decode_base_offsets,
            out=topk_out[decode_slice],
        )
        return topk_out

    def _forward_sparse_indexer_custom_op(
        self,
        *,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        positions: torch.Tensor,
        metadata: DeepseekV4ForwardMetadata,
        ctx: ForwardContext,
        indexer_cache: torch.Tensor,
        indexer_block_size: int,
        cos_sin_cache: torch.Tensor,
    ) -> torch.Tensor:
        forward_mode = ctx.forward_mode
        total_tokens = positions.numel()
        if total_tokens == 0:
            return torch.empty(
                (0, self.topk_tokens),
                device=positions.device,
                dtype=torch.int32,
            )
        num_prefill_tokens, num_decode_tokens = _deepseek_v4_indexer_token_split(
            forward_mode,
            metadata,
            total_tokens,
        )

        with nvtx_range("indexer_wq_b"):
            index_q, _ = self.wq_b(qr)
            index_q = index_q.view(-1, self.n_head, self.head_dim)
        with nvtx_range("indexer_weights_proj"):
            weights, _ = self.weights_proj(hidden_states)
        if not self.use_fp4_cache:
            with nvtx_range("indexer_prepare_portable"):
                q_values, packed_weights = deepseek_v4_prepare_indexer_q(
                    index_q=index_q,
                    positions=positions,
                    cos_sin_cache=cos_sin_cache,
                    weights=weights,
                    head_scale=self.n_head**-0.5,
                )
            packed_index_q = (
                q_values,
                q_values.new_empty((q_values.shape[0], 0), dtype=torch.float32),
            )
        else:
            with nvtx_range("indexer_prepare_mxfp4"):
                packed_index_q, packed_weights = deepseek_v4_prepare_indexer_q_mxfp4(
                    index_q=index_q,
                    positions=positions,
                    cos_sin_cache=cos_sin_cache,
                    weights=weights,
                    softmax_scale=self.softmax_scale,
                    head_scale=self.n_head**-0.5,
                )

        empty_cpu = torch.empty(0, dtype=torch.int32, device="cpu")
        seq_lens_cpu = (
            metadata.seq_lens_cpu[: metadata.num_prefill_reqs]
            if metadata.seq_lens_cpu is not None and num_prefill_tokens > 0
            else empty_cpu
        )
        query_lens_cpu = (
            metadata.query_lens_cpu[: metadata.num_prefill_reqs]
            if metadata.query_lens_cpu is not None and num_prefill_tokens > 0
            else empty_cpu
        )
        indexer_page_table, indexer_page_table_base_offsets = (
            _deepseek_v4_indexer_page_table(
                metadata,
                self.compress_ratio,
                indexer_block_size,
            )
        )
        prefill_metadata = _deepseek_v4_indexer_prefill_metadata(
            metadata=metadata,
            block_table=indexer_page_table,
            cache_block_size=indexer_block_size,
            compress_ratio=self.compress_ratio,
            num_prefill_tokens=num_prefill_tokens,
            block_table_base_offsets=(
                None if self.use_fp4_cache else indexer_page_table_base_offsets
            ),
        )

        decode_schedule_metadata = None
        decode_plan = None
        if num_decode_tokens > 0:
            decode_start = num_prefill_tokens
            decode_end = decode_start + num_decode_tokens
            decode_positions = positions[decode_start:decode_end]
            decode_valid_token = (
                metadata.is_valid_token[decode_start:decode_end]
                if getattr(metadata, "is_valid_token", None) is not None
                else None
            )
            decode_plan = _deepseek_v4_indexer_decode_plan(
                positions=decode_positions,
                token_to_req_indices=metadata.token_to_req_indices[
                    decode_start:decode_end
                ],
                block_table=indexer_page_table,
                cache_block_size=indexer_block_size,
                compress_ratio=self.compress_ratio,
                metadata=metadata,
                is_valid_token=decode_valid_token,
                block_table_base_offsets=indexer_page_table_base_offsets,
            )
            if self.use_fp4_cache:
                decode_schedule_metadata = (
                    _deepseek_v4_indexer_decode_schedule_metadata(
                        positions=decode_positions,
                        cache_block_size=indexer_block_size,
                        compress_ratio=self.compress_ratio,
                        metadata=metadata,
                        context_lens=decode_plan.context_lens,
                    )
                )
        indexer_metadata = DeepseekV4SparseIndexerMetadata(
            batch_metadata=DeepseekV4IndexerBatchMetadata(
                positions=positions,
                token_to_req_indices=metadata.token_to_req_indices[:total_tokens],
                seq_lens_cpu=seq_lens_cpu,
                query_lens_cpu=query_lens_cpu,
                num_prefill_tokens=num_prefill_tokens,
                num_decode_tokens=num_decode_tokens,
            ),
            prefill_metadata=prefill_metadata,
            decode_plan=decode_plan,
            decode_schedule_metadata=decode_schedule_metadata,
        )
        prefill_gather_values, prefill_gather_scales = self._prefill_gather_workspace(
            prefill_metadata.max_gather_rows(),
            positions.device,
        )

        topk_out = (
            self.topk_buffer.get(total_tokens, positions.device)
            if self.topk_buffer is not None
            else torch.empty(
                (total_tokens, self.topk_tokens),
                device=positions.device,
                dtype=torch.int32,
            )
        )[:total_tokens]
        if not self.use_fp4_cache:
            return self._forward_sparse_indexer_portable(
                query=packed_index_q[0],
                weights=packed_weights,
                indexer_cache=indexer_cache,
                indexer_block_size=indexer_block_size,
                prefill_metadata=prefill_metadata,
                decode_plan=decode_plan,
                num_prefill_tokens=num_prefill_tokens,
                num_decode_tokens=num_decode_tokens,
                token_to_req_indices=metadata.token_to_req_indices[:total_tokens],
                block_table_base_offsets=indexer_page_table_base_offsets,
                topk_out=topk_out,
            )
        return _deepseek_v4_sparse_attn_indexer(
            indexer_metadata=indexer_metadata,
            indexer_cache=indexer_cache,
            indexer_page_table=indexer_page_table,
            indexer_block_size=indexer_block_size,
            compress_ratio=self.compress_ratio,
            packed_q_values=packed_index_q[0],
            packed_q_scales=packed_index_q[1],
            packed_weights=packed_weights,
            topk_indices_buffer=topk_out,
            prefill_gather_values_workspace=prefill_gather_values,
            prefill_gather_scales_workspace=prefill_gather_scales,
            persistent_topk_workspace=self._persistent_topk_workspace,
            topk_tokens=self.topk_tokens,
            use_fp4_cache=self.use_fp4_cache,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        positions: torch.Tensor,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
        layer_index: int,
        cos_sin_cache: torch.Tensor,
        compressor_slot_cache: dict,
        indexer_compressor_kv_score: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pool = ctx.token_to_kv_pool
        metadata = _deepseek_v4_forward_metadata(ctx)
        if metadata is None:
            raise RuntimeError("DeepSeek V4 indexer requires forward metadata")
        cache_metadata = metadata.cache
        indexer_state = pool.get_indexer_state_buffer(layer_index)
        valid_token = (
            metadata.is_valid_token[: positions.numel()]
            if getattr(metadata, "is_valid_token", None) is not None
            else None
        )
        idx_hit = (
            compressor_slot_cache.get("indexer_state")
            if compressor_slot_cache is not None
            else None
        )
        if idx_hit is not None:
            (
                indexer_state_slot_mapping,
                indexer_state_block_table,
                indexer_state_block_size,
                indexer_state_base_logical_page,
            ) = idx_hit
        else:
            indexer_state_block_table = cache_metadata.indexer_state_block_table
            indexer_state_base_logical_page = (
                cache_metadata.indexer_state_base_logical_page
            )
            if indexer_state_block_table is None:
                raise RuntimeError(
                    "DeepSeek V4 missing paged-cache block table for indexer "
                    "compressor state"
                )
            indexer_state_block_size = pool.get_indexer_state_block_size(layer_index)
            indexer_state_slot_mapping = _group_slot_mapping_from_raw(
                positions,
                metadata.token_to_req_indices[: positions.numel()],
                indexer_state_block_table,
                indexer_state_block_size,
                base_offsets=indexer_state_base_logical_page,
            )
            indexer_state_slot_mapping = _mask_invalid_graph_tokens(
                indexer_state_slot_mapping,
                valid_token,
            )
            if compressor_slot_cache is not None:
                compressor_slot_cache["indexer_state"] = (
                    indexer_state_slot_mapping,
                    indexer_state_block_table,
                    indexer_state_block_size,
                    indexer_state_base_logical_page,
                )
        with nvtx_range("indexer_compressor_total"):
            self.compressor(
                hidden_states=hidden_states,
                positions=positions,
                ctx=ctx,
                out_cache_loc=out_cache_loc,
                layer_index=layer_index,
                cos_sin_cache=cos_sin_cache,
                state_cache=indexer_state,
                state_block_table=indexer_state_block_table,
                state_block_size=indexer_state_block_size,
                state_base_logical_page=indexer_state_base_logical_page,
                state_slot_mapping=indexer_state_slot_mapping,
                write_compressed_cache=False,
                kv_score=indexer_compressor_kv_score,
            )
        with nvtx_range("indexer_compressed_slot_mapping"):
            indexer_block_size = pool.get_indexer_block_size(layer_index)
            compressed_slots = cache_metadata.compressed_slot_mapping(
                positions,
                self.compress_ratio,
                token_to_req_indices=metadata.token_to_req_indices[: positions.numel()],
                query_start_loc=metadata.query_start_loc,
                seq_lens=metadata.seq_lens,
                kv_cache_block_size=indexer_block_size,
                use_decode_cache=(
                    ctx.forward_mode is not None and ctx.forward_mode.is_decode()
                ),
                is_valid_token=valid_token,
            )
        with nvtx_range("indexer_cache_insert"):
            deepseek_v4_csa_indexer_cache_insert(
                state_cache=indexer_state,
                token_to_req_indices=metadata.token_to_req_indices[: positions.numel()],
                positions=positions,
                compressor_slot_mapping=indexer_state_slot_mapping,
                block_table=indexer_state_block_table,
                block_table_base_offsets=indexer_state_base_logical_page,
                compressor_block_size=indexer_state_block_size,
                rms_norm_weight=self.compressor.norm.weight,
                rms_norm_eps=self.compressor.norm.variance_epsilon,
                cos_sin_cache=cos_sin_cache,
                kv_cache_2d=pool.get_indexer_kv_buffer_2d(layer_index),
                kv_slot_mapping=compressed_slots,
                kv_cache_block_size=indexer_block_size,
                use_fp4_cache=self.use_fp4_cache,
                compress_ratio=self.compress_ratio,
            )
        return self._forward_sparse_indexer_custom_op(
            hidden_states=hidden_states,
            qr=qr,
            positions=positions,
            metadata=metadata,
            ctx=ctx,
            indexer_cache=pool.get_indexer_kv_buffer_2d(layer_index),
            indexer_block_size=indexer_block_size,
            cos_sin_cache=cos_sin_cache,
        )


class DeepseekV4Attention(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mapping: Mapping,
        layer_index: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
        aux_stream: object | None = None,
        topk_buffer: _DeepseekV4TopKBuffer | None = None,
        cache_layer_index: int | None = None,
    ) -> None:
        super().__init__()
        # `layer_index` addresses checkpoint/config metadata; `cache_layer_index`
        # addresses this model's compact KV cache slot.
        self.layer_index = layer_index
        self.cache_layer_index = (
            layer_index if cache_layer_index is None else cache_layer_index
        )
        self.stream_fork = StreamFork(aux_stream)
        self.compress_ratio = max(1, int(config.compress_ratios[layer_index]))
        if self.compress_ratio <= 1:
            self.attention_kind = "swa"
        elif self.compress_ratio == 4:
            self.attention_kind = "csa"
        elif self.compress_ratio == 128:
            self.attention_kind = "hca"
        else:
            raise ValueError(
                f"Unsupported DeepSeek V4 compress_ratio={self.compress_ratio}; "
                "expected 1, 4, or 128."
            )
        self.num_heads = int(config.num_attention_heads)
        tp_rank = mapping.attn.tp_rank
        tp_size = mapping.attn.tp_size
        tp_group = mapping.attn.tp_group
        if self.num_heads % tp_size != 0:
            raise ValueError(
                f"num_attention_heads={self.num_heads} must be divisible "
                f"by attn_tp_size={tp_size}"
            )
        self.num_local_heads = self.num_heads // tp_size
        self.padded_heads = dsv4_padded_heads(self.num_local_heads)
        self.head_dim = int(config.head_dim)
        self.qk_rope_head_dim = int(config.qk_rope_head_dim)
        self.nope_head_dim = deepseek_v4_nope_dim(
            self.head_dim,
            self.qk_rope_head_dim,
        )
        self.swa_window = int(getattr(config, "sliding_window", 128))
        self.scale = self.head_dim**-0.5
        self.q_lora_rank = config.q_lora_rank
        self.o_lora_rank = config.o_lora_rank
        self.o_groups = config.o_groups
        self.num_local_groups = self.o_groups // tp_size
        num_local = self.num_local_heads
        self.attn_sink = nn.Parameter(
            torch.full((self.padded_heads,), -float("inf"), dtype=torch.float32),
            requires_grad=False,
        )
        set_weight_attrs(
            self.attn_sink,
            {
                "weight_loader": lambda param, loaded_weight: param.data.__setitem__(
                    slice(num_local),
                    loaded_weight[tp_rank * num_local : (tp_rank + 1) * num_local],
                ),
            },
        )
        rope_base, rope_scaling = deepseek_v4_rope_config(config, self.compress_ratio)
        self.rotary_emb = get_rope(
            self.qk_rope_head_dim,
            rotary_dim=self.qk_rope_head_dim,
            max_position=getattr(config, "max_position_embeddings", 8192),
            base=rope_base,
            rope_scaling=rope_scaling,
            is_neox_style=False,
        )
        self.indexer_rotary_emb = self.rotary_emb
        self.fused_wqa_wkv = MergedColumnParallelLinear(
            config.hidden_size,
            [self.q_lora_rank, self.head_dim],
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("fused_wqa_wkv", prefix),
        )
        self.q_norm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.wq_b = ColumnParallelLinear(
            self.q_lora_rank,
            self.num_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("wq_b", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
            tp_group=tp_group,
        )
        self.kv_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.fused_qkv_norm = FusedRMSNorm(self.q_norm, self.kv_norm)
        # Fused QKV-RMSNorm: fuses q_a/kv_a RMSNorm into one kernel instead of two
        # sequential RMSNorm + a kv.contiguous() copy. Validated via GSM8K.
        self.use_fused_qkv_rmsnorm = True
        self.wo_a = ColumnParallelLinear(
            self.num_heads * self.head_dim // self.o_groups,
            self.o_groups * self.o_lora_rank,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("wo_a", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
            tp_group=tp_group,
        )
        wo_a_quant_config = self.wo_a.quant_method.quant_config
        self._wo_a_output_projection_plan = dsv4_grouped_output_projection_plan(
            input_dtype=self.wo_a.orig_dtype,
            weight_dtype=self.wo_a.weight.dtype,
            weight_scale_dtype=self.wo_a.weight_scale_inv.dtype,
            num_groups=self.num_local_groups,
            heads_per_group=self.num_local_heads // self.num_local_groups,
            head_dim=self.head_dim,
            nope_dim=self.nope_head_dim,
            rope_dim=self.qk_rope_head_dim,
            output_dim=self.o_lora_rank,
            block_size=wo_a_quant_config.weight_block_size,
            scale_format=getattr(wo_a_quant_config, "scale_fmt", None),
        )
        self.wo_a._dsv4_grouped_output_projection_plan = (
            self._wo_a_output_projection_plan
        )
        self.wo_b = RowParallelLinear(
            self.o_groups * self.o_lora_rank,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("wo_b", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
            tp_group=tp_group,
        )
        if self.compress_ratio > 1:
            self.compressor = DeepseekV4Compressor(
                config,
                config.hidden_size,
                self.head_dim,
                self.compress_ratio,
                add_prefix("compressor", prefix),
            )
        else:
            self.compressor = None
        if self.compress_ratio == 4:
            self.indexer = DeepseekV4Indexer(
                config,
                mapping,
                quant_config,
                add_prefix("indexer", prefix),
                self.compress_ratio,
                topk_buffer=topk_buffer,
            )
        else:
            self.indexer = None

    def _project_q_kv(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        qr_kv, _ = self.fused_wqa_wkv(hidden_states)
        qr, kv = qr_kv.split([self.q_lora_rank, self.head_dim], dim=-1)
        if (
            self.use_fused_qkv_rmsnorm
            and qr.shape[0] > 0
            and self.fused_qkv_norm.supports(qr)
        ):
            qr_norm = torch.empty(
                qr.shape,
                dtype=qr.dtype,
                device=qr.device,
            )
            kv_norm = torch.empty(
                kv.shape,
                dtype=kv.dtype,
                device=kv.device,
            )
            self.fused_qkv_norm(
                input_q_a=qr,
                input_kv_a=kv,
                output_q_a=qr_norm,
                output_kv_a=kv_norm,
            )
            qr = qr_norm
            kv = kv_norm
        else:
            qr = self.q_norm(qr)
            kv = self.kv_norm(kv.contiguous())
        q, _ = self.wq_b(qr)
        q = q.view(-1, self.num_local_heads, self.head_dim)
        return q, kv, qr

    def _insert_swa_cache(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        swa_kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        block_size: int,
    ) -> None:
        # slot_mapping arrives pre-sanitized from _deepseek_v4_swa_slot_mapping
        # (invalid tokens and out-of-capacity slots masked to -1); sanitizing
        # here again would re-run the mask chain once per layer.
        if q.shape[0] == 0:
            return
        fused_qnorm_rope_kv_insert(
            q=q,
            kv=kv,
            swa_kv_cache_2d=swa_kv_cache.view(swa_kv_cache.shape[0], -1),
            slot_mapping=slot_mapping,
            positions=positions,
            cos_sin_cache=cos_sin_cache,
            rms_norm_eps=self.q_norm.variance_epsilon,
            block_size=block_size,
        )

    def _project_attention_output(
        self,
        attn_output: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
    ) -> torch.Tensor:
        z = dsv4_grouped_output_projection(
            self._wo_a_output_projection_plan,
            attn_output,
            positions,
            cos_sin_cache,
            self.wo_a.weight,
            self.wo_a.weight_scale_inv,
        )
        out, _ = self.wo_b(z.flatten(1))
        return out

    @break_point
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
        swa_slot_mapping: torch.Tensor | None = None,
        compressor_slot_cache: dict | None = None,
    ) -> torch.Tensor:
        """DSA attention, one COARSE breakable-graph break point.

        Per layer it does multiple paged-cache writes (SWA / compressor /
        indexer), a data/length-dependent indexer -> top-k stage, the FlashMLA
        sparse kernel, AND aux-stream forks -- none capturable into a CUDA
        graph. Under a prefill-graph capture the whole attention runs eager
        (reading the live ``ctx``) while the layer's norms + MoE stay graphed;
        direct call otherwise (see ``break_point``). Padding rows are handled
        by the existing ``metadata.is_valid_token`` masking, and the
        token-shaped inputs are sliced to the real count DSA kernels assert
        (see ``slice_to_real_tokens`` below). The cross-layer SWA slot mapping
        and compressor memo are shared via ctx -- replay-safe because ctx is
        rebound to the live forward (see ``DeepseekV4Model.forward``).
        """
        if hidden_states.shape[0] == 0:
            return hidden_states
        # Cross-layer compressor memo, created once per forward by the first layer.
        if compressor_slot_cache is None:
            compressor_slot_cache = ctx.dsa_compressor_slot_cache
            if compressor_slot_cache is None:
                compressor_slot_cache = ctx.dsa_compressor_slot_cache = {}
        profile_prefix = f"attn_{self.attention_kind}"
        cos_sin_cache = self.rotary_emb.cos_sin_cache
        if cos_sin_cache.dtype != torch.float32:
            cos_sin_cache = cos_sin_cache.float()
        pool = ctx.token_to_kv_pool
        metadata = ctx.attn_backend.forward_metadata
        if metadata is None:
            raise RuntimeError("DeepSeek V4 attention requires forward metadata")

        # Slice padded inputs to the real count the DSA kernels assert (see
        # docstring). Only under a breakable capture/replay (ambient ctx set):
        # eager forwards are never padded, and the MTP draft path reaches here
        # with metadata whose token_to_req_indices does NOT describe q's rows.
        token_to_req = getattr(metadata, "token_to_req_indices", None)
        if current_forward_ctx() is not None and token_to_req is not None:
            positions, hidden_states, out_cache_loc, swa_slot_mapping = (
                slice_to_real_tokens(
                    token_to_req.numel(),
                    positions,
                    hidden_states,
                    out_cache_loc,
                    swa_slot_mapping,
                )
            )

        # --- Phase 1: pre-compute input GEMMs in parallel ---
        # Q/KV projection on main stream; compressor GEMM(s) on aux stream.
        # After this phase, each compressor's forward() receives its
        # pre-computed kv_score and skips the internal GEMM, removing the
        # shared-weight dependency that prevents safe multi-stream overlap.
        compressor_kv_score: torch.Tensor | None = None
        indexer_compressor_kv_score: torch.Tensor | None = None
        with self.stream_fork.scope(
            enable=self.stream_fork.aux_stream is not None
        ) as fork:
            with nvtx_range(f"{profile_prefix}_project_q_kv"):
                q, kv, qr = self._project_q_kv(hidden_states)
            with fork.branch():
                if self.compressor is not None:
                    with nvtx_range(f"{profile_prefix}_compressor_gemm"):
                        compressor_kv_score = self.compressor.compute_kv_score(
                            hidden_states
                        )
                if self.indexer is not None:
                    with nvtx_range(f"{profile_prefix}_indexer_compressor_gemm"):
                        indexer_compressor_kv_score = (
                            self.indexer.compressor.compute_kv_score(hidden_states)
                        )

        if swa_slot_mapping is None:
            # Cross-layer SWA slot mapping (per-token, layer-invariant), first layer computes.
            swa_slot_mapping = ctx.dsa_swa_slot_mapping
            if swa_slot_mapping is None:
                swa_slot_mapping = _deepseek_v4_swa_slot_mapping(
                    ctx,
                    positions,
                    out_cache_loc,
                )
                ctx.dsa_swa_slot_mapping = swa_slot_mapping

        def insert_swa_cache() -> None:
            with nvtx_range(f"{profile_prefix}_insert_swa_cache"):
                self._insert_swa_cache(
                    q=q,
                    kv=kv,
                    swa_kv_cache=pool.get_swa_kv_buffer(self.cache_layer_index),
                    slot_mapping=swa_slot_mapping,
                    positions=positions,
                    cos_sin_cache=cos_sin_cache,
                    block_size=pool.swa_block_size,
                )

        def run_compressor() -> None:
            if self.compressor is None:
                return
            with nvtx_range(f"{profile_prefix}_compressor"):
                self.compressor(
                    hidden_states=hidden_states,
                    positions=positions,
                    ctx=ctx,
                    out_cache_loc=out_cache_loc,
                    layer_index=self.cache_layer_index,
                    cos_sin_cache=cos_sin_cache,
                    compressor_slot_cache=compressor_slot_cache,
                    kv_score=compressor_kv_score,
                )

        # --- Phase 2: state-update + cache-write overlap ---
        # With GEMMs already done, the remaining work per stream is lightweight
        # state updates and paged cache writes — safe to overlap.
        topk_indices = None
        if self.indexer is not None:
            if self.compressor is None:
                raise RuntimeError("compressor is required when indexer is enabled.")
            with nvtx_range(f"{profile_prefix}_indexer_prepare_decode_metadata"):
                self.indexer.prepare_decode_metadata(
                    positions=positions,
                    metadata=metadata,
                    ctx=ctx,
                    indexer_block_size=pool.get_indexer_block_size(
                        self.cache_layer_index
                    ),
                )

            with self.stream_fork.scope(
                enable=self.stream_fork.aux_stream is not None
            ) as fork:
                with nvtx_range(f"{profile_prefix}_indexer"):
                    topk_indices = self.indexer(
                        hidden_states=hidden_states,
                        qr=qr,
                        positions=positions,
                        ctx=ctx,
                        out_cache_loc=out_cache_loc,
                        layer_index=self.cache_layer_index,
                        cos_sin_cache=cos_sin_cache,
                        compressor_slot_cache=compressor_slot_cache,
                        indexer_compressor_kv_score=indexer_compressor_kv_score,
                    )
                with fork.branch():
                    insert_swa_cache()
                run_compressor()
        elif self.compressor is not None:
            with self.stream_fork.scope(
                enable=self.stream_fork.aux_stream is not None
            ) as fork:
                run_compressor()
                with fork.branch():
                    insert_swa_cache()
        else:
            insert_swa_cache()
        forward_mode = ctx.forward_mode
        if forward_mode is None:
            raise RuntimeError("DeepSeek V4 attention requires forward mode")
        if forward_mode.is_mixed():
            with nvtx_range(f"{profile_prefix}_mixed_backend"):
                attn_output = ctx.attn_backend.forward_deepseek_v4_mixed(
                    q=q,
                    positions=positions,
                    token_to_kv_pool=pool,
                    layer_id=self.cache_layer_index,
                    kind=self.attention_kind,
                    compress_ratio=self.compress_ratio,
                    num_local_heads=self.num_local_heads,
                    padded_heads=self.padded_heads,
                    head_dim=self.head_dim,
                    window_size=self.swa_window,
                    softmax_scale=self.scale,
                    attn_sink=self.attn_sink,
                    topk_indices=topk_indices,
                )
        elif forward_mode.is_decode():
            with nvtx_range(f"{profile_prefix}_decode_backend"):
                attn_output = ctx.attn_backend.forward_deepseek_v4_decode(
                    q=q,
                    positions=positions,
                    token_to_kv_pool=pool,
                    layer_id=self.cache_layer_index,
                    kind=self.attention_kind,
                    compress_ratio=self.compress_ratio,
                    num_local_heads=self.num_local_heads,
                    padded_heads=self.padded_heads,
                    head_dim=self.head_dim,
                    window_size=self.swa_window,
                    softmax_scale=self.scale,
                    attn_sink=self.attn_sink,
                    topk_indices=topk_indices,
                )
        elif forward_mode.is_extend_or_mixed():
            with nvtx_range(f"{profile_prefix}_prefill_backend"):
                attn_output = ctx.attn_backend.forward_deepseek_v4_prefill(
                    q=q,
                    positions=positions,
                    token_to_kv_pool=pool,
                    layer_id=self.cache_layer_index,
                    kind=self.attention_kind,
                    compress_ratio=self.compress_ratio,
                    num_local_heads=self.num_local_heads,
                    padded_heads=self.padded_heads,
                    head_dim=self.head_dim,
                    window_size=self.swa_window,
                    softmax_scale=self.scale,
                    attn_sink=self.attn_sink,
                    topk_indices=topk_indices,
                )
        else:
            raise RuntimeError(f"Unsupported DeepSeek V4 forward mode: {forward_mode}")
        with nvtx_range(f"{profile_prefix}_output_proj"):
            output = self._project_attention_output(
                attn_output,
                positions,
                cos_sin_cache,
            )
        return ctx.attn_backend.record_layer_cache_ready(output, forward_mode)


class DeepseekV4DecoderLayer(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        mapping: Mapping,
        quant_config: QuantizationConfig | None,
        prefix: str,
        aux_stream: object | None = None,
        topk_buffer: _DeepseekV4TopKBuffer | None = None,
        cache_layer_index: int | None = None,
    ) -> None:
        super().__init__()
        self.mapping = mapping
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        self.rms_norm_eps = config.rms_norm_eps
        self.hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        mix_hc = (2 + self.hc_mult) * self.hc_mult
        hc_dim = self.hc_mult * config.hidden_size
        self.attn = DeepseekV4Attention(
            config,
            mapping,
            layer_id,
            quant_config,
            add_prefix("attn", prefix),
            aux_stream=aux_stream,
            topk_buffer=topk_buffer,
            cache_layer_index=cache_layer_index,
        )
        self.ffn = DeepseekV4MoE(
            config,
            mapping,
            quant_config,
            layer_id,
            add_prefix("ffn", prefix),
            aux_stream=aux_stream,
        )
        self.comm_manager = CommManager(
            mapping=mapping,
            layer_id=layer_id,
            is_moe=True,
            prev_is_moe=True,
        )
        self.attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hc_attn_fn = nn.Parameter(
            torch.empty(mix_hc, hc_dim, dtype=torch.float32), requires_grad=False
        )
        self.hc_ffn_fn = nn.Parameter(
            torch.empty(mix_hc, hc_dim, dtype=torch.float32), requires_grad=False
        )
        self.hc_attn_base = nn.Parameter(
            torch.empty(mix_hc, dtype=torch.float32), requires_grad=False
        )
        self.hc_ffn_base = nn.Parameter(
            torch.empty(mix_hc, dtype=torch.float32), requires_grad=False
        )
        self.hc_attn_scale = nn.Parameter(
            torch.empty(3, dtype=torch.float32), requires_grad=False
        )
        self.hc_ffn_scale = nn.Parameter(
            torch.empty(3, dtype=torch.float32), requires_grad=False
        )

    def _pre_mlp_input_ids_comm(
        self, input_ids: torch.Tensor, ctx: ForwardContext
    ) -> torch.Tensor:
        if not self.mapping.moe.has_tp_ep:
            return input_ids
        if self.comm_manager.use_all_reduce(is_moe=True):
            return input_ids

        token_counts = self.comm_manager.moe_tp_ep_group_scattered_num_tokens(ctx)
        max_tokens = max(token_counts)
        padded = torch.empty(
            (max_tokens,), device=input_ids.device, dtype=input_ids.dtype
        )
        padded[: input_ids.shape[0]].copy_(input_ids)
        if input_ids.shape[0] < max_tokens:
            padded[input_ids.shape[0] :].zero_()

        gathered = [torch.empty_like(padded) for _ in token_counts]
        group = pg_manager.get_device_process_group(self.mapping.moe.tp_ep_group)
        torch.distributed.all_gather(gathered, padded, group=group)
        return torch.cat(
            [tokens[:count] for tokens, count in zip(gathered, token_counts)], dim=0
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
        input_ids: torch.Tensor,
        swa_slot_mapping: torch.Tensor | None = None,
        compressor_slot_cache: dict | None = None,
        hc_x_prev: torch.Tensor | None = None,
        hc_post_prev: torch.Tensor | None = None,
        hc_comb_prev: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if hc_x_prev is not None:
            with nvtx_range("hc_fused_attn_pre"):
                residual, hidden_states, post, comb = mhc_fused_hc(
                    hc_x_prev,
                    hidden_states,
                    hc_post_prev,
                    hc_comb_prev,
                    self.hc_attn_fn,
                    self.hc_attn_scale,
                    self.hc_attn_base,
                    self.rms_norm_eps,
                    self.hc_eps,
                    self.hc_sinkhorn_iters,
                )
        else:
            residual = hidden_states
            with nvtx_range("hc_attn_pre"):
                hidden_states, post, comb = mhc_pre(
                    hidden_states,
                    self.hc_attn_fn,
                    self.hc_attn_scale,
                    self.hc_attn_base,
                    self.rms_norm_eps,
                    self.hc_eps,
                    self.hc_sinkhorn_iters,
                )
        with nvtx_range("attn_norm"):
            hidden_states = self.attn_norm(hidden_states)

        with nvtx_range("attn_total"):
            hidden_states = self.attn(
                positions,
                hidden_states,
                ctx,
                out_cache_loc,
                swa_slot_mapping,
                compressor_slot_cache,
            )

        with nvtx_range("hc_fused_ffn_pre"):
            residual, hidden_states, post, comb = mhc_fused_hc(
                hidden_states,
                residual,
                post,
                comb,
                self.hc_ffn_fn,
                self.hc_ffn_scale,
                self.hc_ffn_base,
                self.rms_norm_eps,
                self.hc_eps,
                self.hc_sinkhorn_iters,
            )
        with nvtx_range("ffn_norm"):
            hidden_states = self.ffn_norm(hidden_states)

        ffn_input_ids = input_ids
        use_mega_moe = getattr(self.ffn, "use_mega_moe", False)
        if use_mega_moe:
            token_counts = self.comm_manager.moe_tp_ep_group_scattered_num_tokens(ctx)
            num_global_tokens = sum(token_counts)
            max_num_tokens_per_gpu = max(token_counts) if token_counts else 0
        else:
            token_counts = None
            with nvtx_range("pre_mlp_comm"):
                hidden_states = self.comm_manager.pre_mlp_comm(hidden_states, ctx)
            if self.ffn.gate.is_hash_moe:
                with nvtx_range("pre_mlp_input_ids_comm"):
                    ffn_input_ids = self._pre_mlp_input_ids_comm(input_ids, ctx)
            with nvtx_range("moe_get_num_tokens"):
                num_global_tokens, max_num_tokens_per_gpu = (
                    self.comm_manager.get_num_tokens(ctx)
                )
        with nvtx_range("ffn_total"):
            hidden_states = self.ffn(
                hidden_states,
                ffn_input_ids,
                num_global_tokens,
                max_num_tokens_per_gpu,
                ctx=ctx if use_mega_moe else None,
                comm_manager=self.comm_manager if use_mega_moe else None,
            )
        if not use_mega_moe:
            with nvtx_range("post_mlp_comm"):
                hidden_states, _ = self.comm_manager.post_mlp_comm(
                    hidden_states, None, ctx
                )
        # Defer ffn post_mapping to next layer's fused_hc
        return residual, hidden_states, post, comb


class DeepseekV4Model(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(
        self,
        config: PretrainedConfig,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.mapping = mapping
        self.hc_mult = config.hc_mult
        self.hc_eps = config.hc_eps
        self.rms_norm_eps = config.rms_norm_eps
        self.aux_stream = torch.cuda.Stream()
        self.topk_indices_buffer = _DeepseekV4TopKBuffer(int(config.index_topk))
        # Pipeline stage layer window: [pp_start_layer, pp_end_layer). Global
        # layer numbering everywhere; other stages' slots hold PPMissingLayer.
        self.pp_start_layer, self.pp_end_layer = pp_layer_window(
            config.num_hidden_layers, mapping
        )
        if mapping.is_first_pp_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                tp_rank=mapping.attn.tp_rank,
                tp_size=mapping.attn.tp_size,
                tp_group=mapping.attn.tp_group,
                prefix=add_prefix("embed_tokens", prefix),
            )
        else:
            self.embed_tokens = None
        self.layers = nn.ModuleList(
            [
                (
                    DeepseekV4DecoderLayer(
                        config,
                        layer_id,
                        mapping,
                        quant_config,
                        add_prefix(f"layers.{layer_id}", prefix),
                        aux_stream=self.aux_stream,
                        topk_buffer=self.topk_indices_buffer,
                    )
                    if self.pp_start_layer <= layer_id < self.pp_end_layer
                    else PPMissingLayer()
                )
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.norm = (
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if mapping.is_last_pp_rank
            else None
        )
        hc_dim = config.hc_mult * config.hidden_size
        self.hc_head_fn = nn.Parameter(
            torch.empty(config.hc_mult, hc_dim, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_head_base = nn.Parameter(
            torch.empty(config.hc_mult, dtype=torch.float32), requires_grad=False
        )
        self.hc_head_scale = nn.Parameter(
            torch.empty(1, dtype=torch.float32), requires_grad=False
        )
        self.dspark_layers_to_capture: tuple[int, ...] = ()

    def pp_stage_state_spec(
        self, num_tokens: int, device: torch.device
    ) -> list[tuple[str, tuple[int, ...], torch.dtype]]:
        """Wire spec (name, shape, dtype) of the inter-stage boundary bundle.

        Matches the PPStageState this model's forward returns on a non-last
        stage: the raw mHC thread state after the stage's final layer. Both
        the sender and the receiver derive this from config + token count, so
        no shape metadata crosses the wire.
        """
        del device
        hc = self.hc_mult
        hidden = self.config.hidden_size
        # The activation dtype. get_default_dtype() would be wrong here: the
        # model-dtype context only wraps weight loading, and the serving-time
        # default is float32 — a recv sized off it would expect twice the
        # bytes the sender ships and the NCCL P2P pair would spin forever.
        dtype = None
        if self.norm is not None:
            dtype = self.norm.weight.dtype
        else:
            for param in self.parameters():
                if param.is_floating_point() and param.dtype in (
                    torch.bfloat16,
                    torch.float16,
                ):
                    dtype = param.dtype
                    break
        if dtype is None:
            dtype = torch.bfloat16
        # Careful with the thread-state shapes: the model loop's variables are
        # crosswired relative to the layer's return order — the loop's
        # ``hidden_states`` carries the residual [t, hc, hidden] while
        # ``hc_x_prev`` carries the layer's FFN output [t, hidden].
        return [
            ("hidden_states", (num_tokens, hc, hidden), dtype),
            ("hc_x", (num_tokens, hidden), dtype),
            ("hc_post", (num_tokens, hc, 1), torch.float32),
            ("hc_comb", (num_tokens, hc, hc), torch.float32),
        ]

    def set_dspark_layers_to_capture(self, layer_ids: list[int]) -> None:
        normalized = tuple(int(layer_id) for layer_id in layer_ids)
        if not normalized:
            raise ValueError("DSpark requires at least one target capture layer.")
        if len(set(normalized)) != len(normalized):
            raise ValueError("DSpark target capture layer IDs must be unique.")
        if tuple(sorted(normalized)) != normalized:
            raise ValueError(
                "DSpark target capture layer IDs must be strictly increasing."
            )
        invalid = [
            layer_id
            for layer_id in normalized
            if layer_id < 0 or layer_id >= len(self.layers)
        ]
        if invalid:
            raise ValueError(
                "DSpark target capture layer IDs are out of range: "
                f"{invalid}; target has {len(self.layers)} layers."
            )
        self.dspark_layers_to_capture = normalized

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
        input_embeds: torch.Tensor | None = None,
        pp_inbound: PPStageState | None = None,
    ) -> tuple[torch.Tensor | PPStageState, list[torch.Tensor] | None]:
        # Layer-invariant DSA memos: computing them HERE (graphed scope) would bake
        # dummy addresses; the first attention break computes them LIVE onto ctx.
        ctx.dsa_swa_slot_mapping = None
        ctx.dsa_compressor_slot_cache = None
        if pp_inbound is not None:
            # Mid-pipeline stage: resume the mHC thread state received from
            # the upstream stage instead of embedding.
            hidden_states = pp_inbound.hidden_states
            hc_x_prev = pp_inbound.hc_x
            hc_post_prev = pp_inbound.hc_post
            hc_comb_prev = pp_inbound.hc_comb
        else:
            hidden_states = input_embeds
            if hidden_states is None:
                with nvtx_range("embed_tokens"):
                    hidden_states = self.embed_tokens(input_ids)
            with nvtx_range("hc_repeat"):
                hidden_states = hidden_states.unsqueeze(1).repeat(1, self.hc_mult, 1)
            hc_x_prev = None
            hc_post_prev = None
            hc_comb_prev = None
        dspark_captures = []
        dspark_capture_set = frozenset(self.dspark_layers_to_capture)
        for layer in self.layers[self.pp_start_layer : self.pp_end_layer]:
            hidden_states, hc_x_prev, hc_post_prev, hc_comb_prev = layer(
                positions,
                hidden_states,
                ctx,
                out_cache_loc,
                input_ids,
                None,  # swa_slot_mapping: first break computes + caches on ctx
                None,  # compressor_slot_cache: shared dict created on ctx
                hc_x_prev,
                hc_post_prev,
                hc_comb_prev,
            )
            if layer.layer_id in dspark_capture_set:
                resolved_hidden_states = mhc_post(
                    hc_x_prev,
                    hidden_states,
                    hc_post_prev,
                    hc_comb_prev,
                )
                dspark_captures.append(resolved_hidden_states.mean(dim=1))
        if not self.mapping.is_last_pp_rank:
            # Hand the raw mHC thread state to the next stage; it resumes the
            # layer loop exactly where this stage stopped (the deferred
            # post-mapping folds into the next layer's fused_hc downstream).
            return (
                PPStageState(
                    hidden_states=hidden_states,
                    hc_x=hc_x_prev,
                    hc_post=hc_post_prev,
                    hc_comb=hc_comb_prev,
                ),
                None,
            )
        with nvtx_range("hc_ffn_post_final"):
            hidden_states = mhc_post(
                hc_x_prev, hidden_states, hc_post_prev, hc_comb_prev
            )
        aux_hidden_states = None
        if (
            ctx.capture_hidden_mode is not None
            and ctx.capture_hidden_mode.need_capture()
        ):
            if self.dspark_layers_to_capture:
                if len(dspark_captures) != len(self.dspark_layers_to_capture):
                    raise RuntimeError(
                        "DSpark target hidden-state capture is incomplete: "
                        f"expected {self.dspark_layers_to_capture}, captured "
                        f"{len(dspark_captures)} tensors."
                    )
                aux_hidden_states = dspark_captures
            else:
                # V4 MTP consumes the pre-hc_head hypercompressed residual.
                aux_hidden_states = [hidden_states.flatten(1)]
        with nvtx_range("hc_head"):
            hidden_states = hc_head(
                hidden_states,
                self.hc_head_fn,
                self.hc_head_scale,
                self.hc_head_base,
                self.rms_norm_eps,
                self.hc_eps,
            )
        with nvtx_range("final_norm"):
            hidden_states = self.norm(hidden_states)
        return hidden_states, aux_hidden_states


class DeepseekV4ForCausalLM(BaseCausalLM):
    model_cls = DeepseekV4Model

    def set_dspark_layers_to_capture(self, layer_ids: list[int]) -> None:
        self.capture_aux_hidden_states = True
        self.model.set_dspark_layers_to_capture(layer_ids)

    def get_stacked_params_mapping(self):
        return [
            ("gate_up_proj", "w1", 0),
            ("gate_up_proj", "w3", 1),
            ("attn.fused_wqa_wkv", "attn.wq_a", 0),
            ("attn.fused_wqa_wkv", "attn.wkv", 1),
            ("compressor.fused_wkv_wgate", "compressor.wkv", 0),
            ("compressor.fused_wkv_wgate", "compressor.wgate", 1),
        ]

    def _map_weight_name(self, name: str) -> str:
        if name.startswith("layers."):
            name = "model." + name
        elif name.startswith("embed."):
            name = name.replace("embed.", "model.embed_tokens.", 1)
        elif name.startswith("norm."):
            name = "model." + name
        elif name.startswith("hc_head"):
            name = "model." + name
        elif name == "head.weight":
            name = "lm_head.weight"
        if ".shared_experts.w2" in name:
            name = name.replace(".shared_experts.w2", ".shared_experts.down_proj")
        if ".ffn.gate.bias" in name:
            name = name.replace(".ffn.gate.bias", ".ffn.gate.e_score_correction_bias")
        if re.search(r"\.experts\.\d+\.w[123]\.scale$", name):
            scale_name = _deepseek_v4_expert_scale_parameter_name(
                self.config,
                use_mega_moe=get_moe_backend().is_mega_moe(),
            )
            name = name.replace(".scale", f".{scale_name}")
        elif name.endswith(".scale"):
            name = name[:-6] + ".weight_scale_inv"
        return name

    @staticmethod
    def _block_quant_fp8_weight(
        weight: torch.Tensor, block: int = 128
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize a BF16 matrix to block-scaled FP8 (DeepSeek 128x128 layout)."""
        n, k = weight.shape
        w = weight.float()
        fmax = torch.finfo(torch.float8_e4m3fn).max
        pad_n = (n + block - 1) // block * block
        pad_k = (k + block - 1) // block * block
        padded = w.new_zeros(pad_n, pad_k)
        padded[:n, :k] = w
        blocks = padded.view(pad_n // block, block, pad_k // block, block)
        amax = blocks.abs().amax(dim=(1, 3), keepdim=True).clamp_min(1e-12)
        scale = amax / fmax
        qblocks = (blocks / scale).clamp(-fmax, fmax).to(torch.float8_e4m3fn)
        qweight = qblocks.reshape(pad_n, pad_k)[:n, :k].contiguous()
        scale_inv = scale.squeeze(-1).squeeze(1).contiguous()
        return qweight, scale_inv

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        stacked_params_mapping = self.get_stacked_params_mapping()
        params_dict = dict(self.named_parameters())
        moe_loader = build_moe_checkpoint_loader(
            params_dict=params_dict,
            expert_schema=ExpertCheckpointSchema(
                gate_proj_name="w1",
                down_proj_name="w2",
                up_proj_name="w3",
            ),
            num_experts=self.config.n_routed_experts,
            ep_rank=self.mapping.moe.ep_rank,
            ep_size=self.mapping.moe.ep_size,
        )
        pp_start = self.model.pp_start_layer
        pp_end = self.model.pp_end_layer
        pp_windowed = not (pp_start == 0 and pp_end == self.config.num_hidden_layers)

        def _pp_owns(name: str) -> bool:
            """Skip weights for layers another pipeline stage owns.

            The MoE loader raises on a matched-but-unloadable expert weight,
            so out-of-window layers must be dropped before it sees them.
            """
            if not name.startswith("model.layers."):
                return True
            layer_id = int(name.split(".", 3)[2])
            return pp_start <= layer_id < pp_end

        for raw_name, loaded_weight in weights:
            name = self._map_weight_name(raw_name)
            if name.startswith("mtp."):
                continue
            if pp_windowed and not _pp_owns(name):
                continue
            if (
                name.endswith("attn.wo_a.weight")
                and loaded_weight.dtype != torch.float8_e4m3fn
                and params_dict.get(name) is not None
                and params_dict[name].dtype == torch.float8_e4m3fn
            ):
                qweight, scale_inv = self._block_quant_fp8_weight(loaded_weight)
                scale_name = name.replace(".weight", ".weight_scale_inv")
                scale_param = params_dict[scale_name]
                scale_param.weight_loader(scale_param, scale_inv)
                loaded_weight = qweight
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name or ".experts." in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict.get(name)
                if param is None:
                    break
                param.weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if moe_loader.matches(name):
                    moe_loader.load(name, loaded_weight)
                    continue
                param = params_dict.get(name)
                if param is None:
                    continue
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
        del params_dict, moe_loader
        self.post_load_weights()
        self.warmup_kernels()

    def post_load_weights(self):
        for module in self.modules():
            if isinstance(module, DeepseekV4Compressor):
                module.process_weights_after_loading()
            elif isinstance(module, DeepseekV4MegaMoEExperts):
                module.finalize_weights()
            elif isinstance(module, MoELayer):
                module.process_weights_after_loading(module)

    def warmup_kernels(self) -> None:
        """Warm selected DSV4 kernels after model weights are loaded."""
        gc.collect()
        torch.cuda.empty_cache()
        for module in self.modules():
            if isinstance(module, DeepseekV4MegaMoEExperts):
                module.warmup()
        config = self.config
        tp_size = self.mapping.attn.tp_size if self.mapping else 1
        dsv4_warmup(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            head_dim=getattr(config, "head_dim", 128),
            hc_mult=getattr(config, "hc_mult", 0),
            kv_lora_rank=getattr(config, "kv_lora_rank", 0),
            index_n_heads=getattr(config, "index_n_heads", 0),
            index_head_dim=getattr(config, "index_head_dim", 0),
            indexer_cache_block_size=V4_KERNEL_BLOCK_ROWS,
            # With MTP/speculation the verify step flattens bs*num_draft_tokens
            # into the decode indexer's num_tokens, so the paged-logits metadata
            # kernel (JIT-keyed on the 32-aligned token count) hits larger
            # buckets than plain decode. Scale the warmup ceiling by the draft
            # factor so those buckets are covered (no-op when speculation is off).
            max_decode_tokens=max(
                int(global_server_args_dict.get("max_cudagraph_capture_size", 0) or 0),
                int(global_server_args_dict.get("max_num_seqs", 0) or 0),
                1,
            )
            * (
                int(global_server_args_dict.get("speculative_num_draft_tokens", 1) or 1)
                if global_server_args_dict.get("speculative_algorithm")
                else 1
            ),
            mxfp4_block_size=DEEPSEEK_V4_MXFP4_BLOCK_SIZE,
            tp_size=tp_size,
            # Prefill GEMM/prenorm M is capped per forward by chunked_prefill_size
            # (continuous batching), the same ceiling mega_moe warms to. Hardcoding
            # 8192 would leave M in (8192, chunked_prefill_size] to JIT inline.
            max_tokens=_deepseek_v4_mega_moe_max_num_tokens(),
            device=next(self.parameters()).device,
        )

    def post_quant_warmup(self) -> None:
        """Called by the weight loader after all quant process_weights_after_loading."""
        dsv4_grouped_output_projection_warmup_model(
            self, max_tokens=_deepseek_v4_mega_moe_max_num_tokens()
        )
        warmup_prepared_fp8_linears(
            self, max_tokens=_deepseek_v4_mega_moe_max_num_tokens()
        )

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=config.n_routed_experts,
            num_groups=getattr(config, "n_group", 0),
        )


EntryClass = [
    DeepseekV4ForCausalLM,
]

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

"""Inference-only GLM 5 model."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Any

import torch
from tokenspeed_kernel.ops.attention.triton.dsa_sparse_layout import (
    full_context_topk_to_global_slots,
    local_topk_to_global_slots,
)
from tokenspeed_kernel.ops.quantization import quantize_fp8_with_scale
from torch import nn
from transformers import PretrainedConfig

try:
    from tokenspeed_kernel.thirdparty import deep_gemm
except Exception:
    deep_gemm = None

try:
    from tokenspeed_kernel.ops.attention.flashinfer.dsa_topk import (
        deterministic_decode_topk,
        has_deterministic_decode_topk,
    )
except Exception:
    deterministic_decode_topk = None
    has_deterministic_decode_topk = None

from tokenspeed.runtime.configs.utils import get_rope_theta
from tokenspeed.runtime.distributed import Mapping
from tokenspeed.runtime.distributed.comm_manager import CommManager
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.layernorm import FusedRMSNorm, RMSNorm
from tokenspeed.runtime.layers.linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
)
from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig
from tokenspeed.runtime.layers.quantization.utils import block_dequant
from tokenspeed.runtime.layers.rotary_embedding import get_rope
from tokenspeed.runtime.layers.vocab_parallel_embedding import VocabParallelEmbedding
from tokenspeed.runtime.model_loader.weight_utils import default_weight_loader
from tokenspeed.runtime.models.deepseek_v3 import (
    DeepseekV3AttentionMLA,
    DeepseekV3DecoderLayer,
    DeepseekV3ForCausalLM,
    DeepseekV3MLP,
    DeepseekV3Model,
    DeepseekV3MoE,
    get_layer_id,
)
from tokenspeed.runtime.utils import add_prefix
from tokenspeed.runtime.utils.env import global_server_args_dict

_INDEXER_PREFILL_MAX_LOGITS_MB_ARG = "deepseek_v4_indexer_prefill_max_logits_mb"


@dataclass
class GlmDsaIndexerOutput:
    query: torch.Tensor
    key: torch.Tensor
    weights: torch.Tensor


@dataclass
class GlmDsaPrefillTopK:
    workspace_indices: torch.Tensor
    topk_lens: torch.Tensor
    block_tables: torch.Tensor
    seq_lens: torch.Tensor
    max_seq_len: int
    kv_workspace_slots: torch.Tensor


@dataclass
class GlmDsaDecodeTopK:
    topk_indices: torch.Tensor
    topk_lens: torch.Tensor


@dataclass(frozen=True)
class GlmDsaDecodeWindow:
    start: int
    end: int
    num_tokens: int
    num_reqs: int
    q_len_per_req: int


def _glm_dsa_is_decode_token_mode(forward_mode: ForwardMode | None) -> bool:
    return forward_mode is not None and (
        forward_mode.is_decode() or forward_mode.is_mixed()
    )


def _glm_dsa_is_pure_decode_token_mode(forward_mode: ForwardMode | None) -> bool:
    return forward_mode is not None and forward_mode.is_decode()


def _glm_dsa_skip_indexer_topk(config, layer_id: int | None) -> bool:
    if layer_id is None:
        return False
    indexer_types = getattr(config, "indexer_types", None)
    if indexer_types is not None and layer_id < len(indexer_types):
        return indexer_types[layer_id] in ("S", "shared")
    pattern = getattr(config, "index_topk_pattern", None)
    if pattern is not None and layer_id < len(pattern):
        return pattern[layer_id] in ("S", "shared")
    freq = int(getattr(config, "index_topk_freq", 1) or 1)
    if freq <= 1:
        return False
    offset = getattr(config, "index_skip_topk_offset", None)
    if offset is None:
        return max(layer_id - 1, 0) % freq != 0
    if offset <= 0:
        raise ValueError(
            "index_skip_topk_offset must be positive; offset <= 0 marks "
            "layer 0 as shared with no prior top-k to reuse"
        )
    return max(layer_id - offset + 1, 0) % freq != 0


def _build_prefill_kv_workspace_slots(
    *,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    page_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    local_offsets = torch.arange(
        int(max_seq_len),
        dtype=torch.int64,
        device=device,
    )
    page_offsets = torch.div(
        local_offsets,
        int(page_size),
        rounding_mode="floor",
    )
    block_offsets = local_offsets % int(page_size)
    pages = block_tables.to(device=device, dtype=torch.int64).index_select(
        1,
        page_offsets,
    )
    slots = pages * int(page_size) + block_offsets
    valid = local_offsets.unsqueeze(0) < seq_lens.to(
        device=device,
        dtype=torch.int64,
    ).unsqueeze(1)
    seq_cu = torch.zeros(
        seq_lens.numel() + 1,
        dtype=torch.int64,
        device=device,
    )
    torch.cumsum(
        seq_lens.to(device=device, dtype=torch.int64),
        dim=0,
        out=seq_cu[1:],
    )
    return slots[valid].contiguous(), seq_cu[:-1].contiguous()


def _glm_dsa_rope_scaling(
    rope_scaling: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not rope_scaling or "factor" not in rope_scaling:
        return None

    rope_scaling = dict(rope_scaling)
    rope_scaling["rope_type"] = "deepseek_yarn"
    return rope_scaling


def _glm_dsa_hadamard_rotate(x: torch.Tensor) -> torch.Tensor:
    try:
        from tokenspeed_kernel.thirdparty.fast_hadamard_transform import (
            hadamard_transform,
        )
    except Exception as exc:
        raise RuntimeError(
            "GLM DSA indexer requires fast_hadamard_transform. "
            "Build/install `tokenspeed-kernel/python` before serving GLM DSA."
        ) from exc

    shape = x.shape
    return hadamard_transform(
        x.to(torch.bfloat16).reshape(-1, shape[-1]).contiguous(),
        scale=shape[-1] ** -0.5,
    ).reshape(shape)


def _glm_dsa_hadamard_rotate_pair(
    query: torch.Tensor,
    key: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if query.shape[-1] != key.shape[-1]:
        raise ValueError(
            "GLM DSA paired Hadamard requires matching last dimensions; "
            f"got query={query.shape[-1]}, key={key.shape[-1]}"
        )
    try:
        from tokenspeed_kernel.thirdparty.fast_hadamard_transform import (
            hadamard_transform,
        )
    except Exception as exc:
        raise RuntimeError(
            "GLM DSA indexer requires fast_hadamard_transform. "
            "Build/install `tokenspeed-kernel/python` before serving GLM DSA."
        ) from exc

    query_shape = query.shape
    key_shape = key.shape
    head_dim = query_shape[-1]
    query_rows = query.numel() // head_dim
    key_rows = key.numel() // head_dim
    if query_rows == 0 and key_rows == 0:
        return query, key

    combined = torch.cat(
        (
            query.to(torch.bfloat16).reshape(query_rows, head_dim).contiguous(),
            key.to(torch.bfloat16).reshape(key_rows, head_dim).contiguous(),
        ),
        dim=0,
    )
    rotated = hadamard_transform(
        combined,
        scale=head_dim**-0.5,
    )
    return (
        rotated[:query_rows].reshape(query_shape),
        rotated[query_rows:].reshape(key_shape),
    )


class GlmDsaIndexer(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        q_lora_rank: int,
        qk_rope_head_dim: int,
        rope_theta: float,
        rope_scaling: dict[str, Any] | None,
        max_position_embeddings: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.index_topk = config.index_topk
        self.index_n_heads = config.index_n_heads
        self.index_head_dim = config.index_head_dim
        self.rope_head_dim = int(qk_rope_head_dim)
        self.softmax_scale = self.index_head_dim**-0.5

        if self.rope_head_dim <= 0 or self.rope_head_dim > self.index_head_dim:
            raise ValueError(
                "GLM DSA indexer requires 0 < qk_rope_head_dim <= index_head_dim; "
                f"got qk_rope_head_dim={self.rope_head_dim}, "
                f"index_head_dim={self.index_head_dim}"
            )

        self.wq_b = ReplicatedLinear(
            q_lora_rank,
            self.index_n_heads * self.index_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("wq_b", prefix),
        )
        self.wk = ReplicatedLinear(
            hidden_size,
            self.index_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("wk", prefix),
        )
        self.weights_proj = ReplicatedLinear(
            hidden_size,
            self.index_n_heads,
            bias=False,
            quant_config=None,
            prefix=add_prefix("weights_proj", prefix),
        )
        self.wk_weights_proj = MergedColumnParallelLinear(
            hidden_size,
            [self.index_head_dim, self.index_n_heads],
            bias=False,
            quant_config=None,
            prefix=add_prefix("wk_weights_proj", prefix),
        )
        self._wk_weights_proj_loaded = False
        self.k_norm = nn.LayerNorm(self.index_head_dim, eps=1e-6)

        rope_scaling = _glm_dsa_rope_scaling(rope_scaling)
        self.rotary_emb = get_rope(
            self.rope_head_dim,
            rotary_dim=self.rope_head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=not getattr(config, "indexer_rope_interleave", False),
        )
        if hasattr(self.rotary_emb, "forward_cuda"):
            self.rotary_emb.forward = self.rotary_emb.forward_cuda
        self._key_only_rope_query_workspace: torch.Tensor | None = None

    def set_wk_weights_proj_loaded(self, loaded: bool = True) -> None:
        self._wk_weights_proj_loaded = bool(loaded)

    def _get_key_only_rope_query_workspace(
        self,
        *,
        rows: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if (
            self._key_only_rope_query_workspace is None
            or self._key_only_rope_query_workspace.device != device
            or self._key_only_rope_query_workspace.dtype != dtype
            or self._key_only_rope_query_workspace.shape[0] < rows
        ):
            self._key_only_rope_query_workspace = torch.empty(
                (rows, 1, self.rope_head_dim),
                dtype=dtype,
                device=device,
            )
        return self._key_only_rope_query_workspace[:rows]

    def _compute_index_k_and_weights(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._wk_weights_proj_loaded:
            key_weights, _ = self.wk_weights_proj(hidden_states)
            return key_weights.split(
                [self.index_head_dim, self.index_n_heads],
                dim=-1,
            )

        index_k, _ = self.wk(hidden_states)
        weights, _ = self.weights_proj(hidden_states)
        return index_k, weights

    def _compute_index_k_only(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._wk_weights_proj_loaded:
            key_weights, _ = self.wk_weights_proj(hidden_states)
            return key_weights[..., : self.index_head_dim]

        index_k, _ = self.wk(hidden_states)
        return index_k

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_lora: torch.Tensor,
        positions: torch.Tensor,
    ) -> GlmDsaIndexerOutput:
        index_q, _ = self.wq_b(q_lora)
        index_q = index_q.view(-1, self.index_n_heads, self.index_head_dim)
        index_k, weights = self._compute_index_k_and_weights(hidden_states)
        index_k = self.k_norm(index_k)

        if positions.numel() > 0:
            q_rope, k_rope = self.rotary_emb(
                positions,
                index_q[..., : self.rope_head_dim],
                index_k[:, None, : self.rope_head_dim],
            )
            index_q[..., : self.rope_head_dim] = q_rope
            index_k[:, : self.rope_head_dim] = k_rope.squeeze(1)

        index_q, index_k = _glm_dsa_hadamard_rotate_pair(index_q, index_k)
        return GlmDsaIndexerOutput(
            query=index_q,
            key=index_k,
            weights=weights.float() * (self.index_n_heads**-0.5),
        )

    def forward_key_only(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        index_k = self._compute_index_k_only(hidden_states)
        index_k = self.k_norm(index_k)

        if positions.numel() > 0:
            dummy_q = self._get_key_only_rope_query_workspace(
                rows=index_k.shape[0],
                device=index_k.device,
                dtype=index_k.dtype,
            )
            _, k_rope = self.rotary_emb(
                positions,
                dummy_q,
                index_k[:, None, : self.rope_head_dim],
            )
            index_k[:, : self.rope_head_dim] = k_rope.squeeze(1)

        return _glm_dsa_hadamard_rotate(index_k)


class GlmMoeDsaAttention(DeepseekV3AttentionMLA):
    _MLA_KERNEL_BACKENDS = ("trtllm_mla", "tokenspeed_mla", "dsa")
    _RAGGED_PREFILL_BACKENDS = ("trtllm_mla", "tokenspeed_mla", "dsa")
    rope_is_neox_style = False

    def __init__(
        self,
        config: PretrainedConfig,
        mapping: Mapping,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        rope_theta: float = 10000,
        rope_scaling: dict[str, Any] | None = None,
        max_position_embeddings: int = 8192,
        quant_config: QuantizationConfig | None = None,
        layer_id=None,
        prefix: str = "",
        reduce_attn_results=True,
        alt_stream: torch.cuda.Stream | None = None,
        skip_rope: bool = False,
        is_nextn: bool = False,
    ) -> None:
        rope_scaling = _glm_dsa_rope_scaling(rope_scaling)
        super().__init__(
            config=config,
            mapping=mapping,
            hidden_size=hidden_size,
            num_heads=num_heads,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            layer_id=layer_id,
            prefix=prefix,
            reduce_attn_results=reduce_attn_results,
            alt_stream=alt_stream,
            skip_rope=skip_rope,
        )
        if q_lora_rank is None:
            raise ValueError("GLM DSA requires q_lora_rank.")
        # Let process_weights choose DeepGEMM only after it has transformed
        # FP8 block scales into the layout that kernel expects.
        self.q_a_layernorm = RMSNorm(q_lora_rank, eps=1e-6)
        self.kv_a_layernorm = RMSNorm(kv_lora_rank, eps=1e-6)
        self.fused_qk_layernorm = FusedRMSNorm(
            self.q_a_layernorm,
            self.kv_a_layernorm,
        )
        self.index_topk = config.index_topk
        self.is_nextn = is_nextn
        # NextN/MTP has its own indexer weights but may reuse the previous
        # draft iteration's top-k. Shared target layers do not have usable
        # indexer weights and must consume the context-carried top-k.
        self.skip_indexer_topk = (
            True if is_nextn else _glm_dsa_skip_indexer_topk(config, layer_id)
        )
        if self.skip_indexer_topk and not self.is_nextn:
            self.indexer = None
        else:
            self.indexer = GlmDsaIndexer(
                config=config,
                hidden_size=hidden_size,
                q_lora_rank=q_lora_rank,
                qk_rope_head_dim=qk_rope_head_dim,
                rope_theta=rope_theta,
                rope_scaling=rope_scaling,
                max_position_embeddings=max_position_embeddings,
                quant_config=quant_config,
                prefix=add_prefix("indexer", prefix),
            )
        self._decode_topk_indices_buffer: torch.Tensor | None = None
        self._decode_local_topk_offsets_buffer: torch.Tensor | None = None
        self._decode_topk_lens_buffer: torch.Tensor | None = None
        self._decode_topk_arange_buffer: torch.Tensor | None = None

    def _get_decode_topk_workspace(
        self,
        attr_name: str,
        rows: int,
        cols: int,
        device: torch.device,
        fill_value: int | None = -1,
    ) -> torch.Tensor:
        buffer = getattr(self, attr_name, None)
        if (
            buffer is None
            or buffer.device != device
            or buffer.shape[0] < rows
            or buffer.shape[1] != cols
        ):
            # A captured CUDA graph may still reference the old buffer; keep
            # it alive so a regrow never frees memory a graph replays into.
            if buffer is not None:
                self._retire_decode_workspace(buffer)
            buffer = torch.empty(
                (rows, cols),
                dtype=torch.int32,
                device=device,
            )
            setattr(self, attr_name, buffer)
        workspace = buffer[:rows]
        if fill_value is not None:
            workspace.fill_(fill_value)
        return workspace

    def _get_decode_topk_lens_workspace(
        self,
        rows: int,
        device: torch.device,
    ) -> torch.Tensor:
        buffer = getattr(self, "_decode_topk_lens_buffer", None)
        if buffer is None or buffer.device != device or buffer.numel() < rows:
            if buffer is not None:
                self._retire_decode_workspace(buffer)
            buffer = torch.empty(
                (rows,),
                dtype=torch.int32,
                device=device,
            )
            self._decode_topk_lens_buffer = buffer
        workspace = buffer[:rows]
        workspace.fill_(0)
        return workspace

    @staticmethod
    def _resolve_decode_q_len(
        ctx: ForwardContext,
        num_decode_tokens: int,
        num_decode_reqs: int,
    ) -> int:
        """Per-request query rows, derived from the actual batch shape.

        Spec-verify and the draft first step can both feed multiple query rows
        per request, while the draft model's later decode steps feed one row.
        The draft attention backend inherits the target verify width from the
        shared config, so trust the actual input row count instead of backend
        metadata.
        """
        if num_decode_reqs > 0 and num_decode_tokens > 0:
            q_len, rem = divmod(int(num_decode_tokens), int(num_decode_reqs))
            if rem == 0 and q_len > 0:
                return q_len
        return 1

    @staticmethod
    def _resolve_num_decode_tokens(
        ctx: ForwardContext,
        *,
        total_tokens: int,
        num_decode_reqs: int,
    ) -> int:
        if num_decode_reqs <= 0 or total_tokens <= 0:
            return 0
        spec_width = int(getattr(ctx.attn_backend, "spec_num_tokens", 1) or 1)
        expected_decode_tokens = num_decode_reqs * spec_width
        return min(int(total_tokens), int(expected_decode_tokens))

    @staticmethod
    def _resolve_decode_req_count(
        ctx: ForwardContext,
        metadata: Any,
    ) -> int:
        num_extends = int(getattr(metadata, "num_extends", 0) or 0)
        limits = [max(0, int(ctx.bs) - int(ctx.num_extends))]

        seq_lens = getattr(metadata, "seq_lens_k", None)
        if seq_lens is not None:
            limits.append(max(0, int(seq_lens.shape[0]) - num_extends))

        block_tables = getattr(metadata, "block_kv_indices", None)
        if block_tables is not None:
            limits.append(max(0, int(block_tables.shape[0]) - num_extends))

        return min(limits)

    @staticmethod
    def _resolve_decode_window(
        ctx: ForwardContext,
        metadata: Any,
        *,
        total_tokens: int,
    ) -> GlmDsaDecodeWindow:
        num_decode_reqs = GlmMoeDsaAttention._resolve_decode_req_count(ctx, metadata)
        num_decode_tokens = GlmMoeDsaAttention._resolve_num_decode_tokens(
            ctx,
            total_tokens=total_tokens,
            num_decode_reqs=num_decode_reqs,
        )
        if total_tokens < num_decode_tokens:
            raise RuntimeError(
                "GLM DSA decode token split is invalid: "
                f"tokens={total_tokens}, decode_tokens={num_decode_tokens}"
            )
        q_len_per_req = GlmMoeDsaAttention._resolve_decode_q_len(
            ctx, num_decode_tokens, num_decode_reqs
        )
        decode_start = int(total_tokens) - int(num_decode_tokens)
        return GlmDsaDecodeWindow(
            start=decode_start,
            end=decode_start + int(num_decode_tokens),
            num_tokens=int(num_decode_tokens),
            num_reqs=int(num_decode_reqs),
            q_len_per_req=int(q_len_per_req),
        )

    @staticmethod
    def _slice_decode_topk(
        decode_topk: GlmDsaDecodeTopK,
        start: int,
        end: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return decode_topk.topk_indices[start:end], decode_topk.topk_lens[start:end]

    @staticmethod
    def _decode_topk_covers_window(
        decode_topk: GlmDsaDecodeTopK | None,
        start: int,
        end: int,
    ) -> bool:
        return (
            decode_topk is not None
            and decode_topk.topk_indices.shape[0] >= end
            and decode_topk.topk_lens.shape[0] >= end
            and start <= end
        )

    def _retire_decode_workspace(self, buffer: torch.Tensor) -> None:
        retired = getattr(self, "_retired_decode_workspaces", None)
        if retired is None:
            retired = []
            self._retired_decode_workspaces = retired
        retired.append(buffer)

    def _get_decode_topk_arange(self, topk: int, device: torch.device) -> torch.Tensor:
        buffer = getattr(self, "_decode_topk_arange_buffer", None)
        if buffer is None or buffer.device != device or buffer.numel() < topk:
            if buffer is not None:
                self._retire_decode_workspace(buffer)
            buffer = torch.arange(topk, dtype=torch.int32, device=device)
            self._decode_topk_arange_buffer = buffer
        return buffer[:topk]

    @staticmethod
    def _expand_decode_seq_lens_per_token(
        seq_lens: torch.Tensor,
        q_len_per_req: int,
        *,
        draft_catchup: bool = False,
    ) -> torch.Tensor:
        """Per-token visible KV lengths for multi-query (spec-verify) decode.

        ``seq_lens`` holds the FULL per-request context: during target verify
        the draft tokens are already written to the KV cache and counted, so
        token ``j`` of a request may only see
        ``seq_lens - q_len_per_req + j + 1`` positions. With
        ``q_len_per_req == 1`` this is ``seq_lens`` itself (plain decode).

        The draft model's first catch-up step is the opposite: it writes its KV
        one token at a time after target verify, so visible lengths advance from
        ``seq_lens`` through ``seq_lens + q_len_per_req - 1``.
        """
        if q_len_per_req == 1:
            return seq_lens
        if draft_catchup:
            offsets = torch.arange(
                q_len_per_req, device=seq_lens.device, dtype=seq_lens.dtype
            )
        else:
            offsets = torch.arange(
                1 - q_len_per_req, 1, device=seq_lens.device, dtype=seq_lens.dtype
            )
        # Padded graph rows can be shorter than q_len; real rows are unaffected.
        return (seq_lens.view(-1, 1) + offsets).clamp_min_(0).reshape(-1)

    @staticmethod
    def _check_decode_q_len_per_req(q_len_per_req: int) -> None:
        # Multi-step MTP verify runs num_draft_tokens query rows per request.
        # DeepGEMM paged MQA logits (our fork) and FlashMLA sparse decode are
        # both verified bit-exact against batch expansion up to next_n = 6,
        # which covers --speculative-num-steps 5 (5 draft + 1 bonus).
        if not 1 <= q_len_per_req <= 6:
            raise NotImplementedError(
                "GLM DSA sparse decode supports 1-6 query tokens per request "
                f"(verified next_n <= 6), got {q_len_per_req}."
            )

    def _write_decode_topk_offsets(
        self,
        *,
        logits: torch.Tensor,
        local_topk_offsets: torch.Tensor,
        topk: int,
    ) -> None:
        if (
            deterministic_decode_topk is None
            or has_deterministic_decode_topk is None
            or not has_deterministic_decode_topk()
        ):
            raise RuntimeError(
                "GLM DSA decode top-k requires flashinfer deterministic top-k."
            )

        deterministic_decode_topk(logits, local_topk_offsets, topk)

    @staticmethod
    def _tensor_version_or_none(tensor: torch.Tensor) -> int | None:
        try:
            return int(getattr(tensor, "_version", 0))
        except RuntimeError as exc:
            if "Inference tensors do not track version counter" in str(exc):
                return None
            raise

    @staticmethod
    def _decode_seq_lens_fit_topk(
        metadata: Any,
        *,
        seq_lens: torch.Tensor,
        block_tables: torch.Tensor | None = None,
        page_size: int | None = None,
        topk: int,
        num_extends: int,
        num_decode_reqs: int,
        capturing: bool,
    ) -> bool:
        if seq_lens.numel() == 0:
            return False
        if capturing:
            if block_tables is None or page_size is None:
                return False
            max_static_context = int(block_tables.shape[1]) * int(page_size)
            return max_static_context <= int(topk)
        if seq_lens.is_cuda:
            # Avoid synchronizing the decode hot path for a tiny-context check.
            if block_tables is None or page_size is None:
                return False
            max_static_context = int(block_tables.shape[1]) * int(page_size)
            return max_static_context <= int(topk)

        # seq_lens may alias a scheduler buffer updated in place.
        seq_lens_version = GlmMoeDsaAttention._tensor_version_or_none(seq_lens)
        if seq_lens_version is None:
            return bool((seq_lens <= topk).all().item())
        cache_key = (
            int(topk),
            int(num_extends),
            int(num_decode_reqs),
            int(seq_lens.numel()),
            int(seq_lens.data_ptr()),
            seq_lens_version,
        )
        cache = getattr(metadata, "_dsa_seq_lens_fit_topk_cache", None)
        if isinstance(cache, tuple) and len(cache) == 2 and cache[0] == cache_key:
            return bool(cache[1])

        fits = bool((seq_lens <= topk).all().item())
        setattr(metadata, "_dsa_seq_lens_fit_topk_cache", (cache_key, fits))
        return fits

    def _compute_decode_full_context_topk_indices(
        self,
        *,
        ctx: ForwardContext,
        seq_lens: torch.Tensor,
        block_tables: torch.Tensor,
        decode_start: int,
        num_tokens: int,
        num_decode_tokens: int,
        topk: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        topk_indices = self._get_decode_topk_workspace(
            "_decode_topk_indices_buffer",
            num_tokens,
            topk,
            device,
            fill_value=None,
        )
        topk_slice = topk_indices[decode_start : decode_start + num_decode_tokens]
        topk_lens = self._get_decode_topk_lens_workspace(
            num_tokens,
            device,
        )
        topk_lens_slice = topk_lens[decode_start : decode_start + num_decode_tokens]
        full_context_topk_to_global_slots(
            seq_lens=seq_lens.to(device=device, dtype=torch.int32),
            block_table=block_tables,
            block_size=ctx.token_to_kv_pool.page_size,
            topk=topk,
            out=topk_slice,
            lens_out=topk_lens_slice,
        )
        return topk_indices, topk_lens

    def _try_compute_decode_full_context_topk_indices(
        self,
        ctx: ForwardContext,
        *,
        num_tokens: int,
        device: torch.device,
    ) -> GlmDsaDecodeTopK | None:
        if not _glm_dsa_is_decode_token_mode(ctx.forward_mode):
            return None

        metadata = getattr(ctx.attn_backend, "forward_decode_metadata", None)
        if metadata is None or metadata.block_kv_indices is None:
            return None
        decode_window = self._resolve_decode_window(
            ctx, metadata, total_tokens=num_tokens
        )
        if decode_window.num_reqs <= 0 or decode_window.num_tokens == 0:
            return None
        self._check_decode_q_len_per_req(decode_window.q_len_per_req)
        capturing = (
            torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
        )

        num_extends = int(metadata.num_extends or 0)
        seq_lens = metadata.seq_lens_k[
            num_extends : num_extends + decode_window.num_reqs
        ]
        if seq_lens.numel() == 0:
            return None

        block_tables = metadata.block_kv_indices[
            num_extends : num_extends + decode_window.num_reqs
        ]
        draft_catchup = bool(
            getattr(ctx.attn_backend, "is_draft", False)
            and ctx.forward_mode is not None
            and ctx.forward_mode.is_decode()
            and decode_window.q_len_per_req > 1
        )
        seq_lens_per_token = self._expand_decode_seq_lens_per_token(
            seq_lens,
            decode_window.q_len_per_req,
            draft_catchup=draft_catchup,
        )
        block_tables_per_token = (
            block_tables
            if decode_window.q_len_per_req == 1
            else block_tables.repeat_interleave(decode_window.q_len_per_req, dim=0)
        )
        topk = self.index_topk
        if not self._decode_seq_lens_fit_topk(
            metadata,
            seq_lens=seq_lens_per_token,
            block_tables=block_tables_per_token,
            page_size=ctx.token_to_kv_pool.page_size,
            topk=topk,
            num_extends=num_extends,
            num_decode_reqs=decode_window.num_reqs,
            capturing=capturing,
        ):
            return None

        cache_key = None
        if (
            _glm_dsa_is_pure_decode_token_mode(ctx.forward_mode)
            and num_extends == 0
            and num_tokens == decode_window.num_tokens
        ):
            seq_lens_version = self._tensor_version_or_none(seq_lens)
            block_tables_version = self._tensor_version_or_none(block_tables)
            if seq_lens_version is not None and block_tables_version is not None:
                cache_key = (
                    int(num_tokens),
                    int(decode_window.num_tokens),
                    int(decode_window.num_reqs),
                    int(ctx.attn_backend.spec_num_tokens),
                    int(topk),
                    int(ctx.token_to_kv_pool.page_size),
                    int(seq_lens.data_ptr()),
                    seq_lens_version,
                    int(block_tables.data_ptr()),
                    block_tables_version,
                    str(device),
                )
                cache = getattr(metadata, "_dsa_full_context_topk_cache", None)
                if (
                    isinstance(cache, tuple)
                    and len(cache) == 2
                    and cache[0] == cache_key
                ):
                    return cache[1]

        full_topk_indices, full_topk_lens = (
            self._compute_decode_full_context_topk_indices(
                ctx=ctx,
                seq_lens=seq_lens_per_token,
                block_tables=block_tables_per_token,
                decode_start=decode_window.start,
                num_tokens=num_tokens,
                num_decode_tokens=decode_window.num_tokens,
                topk=topk,
                device=device,
            )
        )
        full_topk = GlmDsaDecodeTopK(
            topk_indices=full_topk_indices,
            topk_lens=full_topk_lens,
        )
        if cache_key is not None:
            setattr(
                metadata,
                "_dsa_full_context_topk_cache",
                (cache_key, full_topk),
            )
        return full_topk

    def _compute_decode_topk_indices(
        self,
        indexer_output: GlmDsaIndexerOutput,
        ctx: ForwardContext,
    ) -> GlmDsaDecodeTopK | None:
        if not _glm_dsa_is_decode_token_mode(ctx.forward_mode):
            return None

        metadata = getattr(ctx.attn_backend, "forward_decode_metadata", None)
        if metadata is None or metadata.block_kv_indices is None:
            return None
        num_tokens = indexer_output.query.shape[0]
        decode_window = self._resolve_decode_window(
            ctx, metadata, total_tokens=num_tokens
        )
        if decode_window.num_reqs <= 0 or num_tokens == 0:
            return None
        self._check_decode_q_len_per_req(decode_window.q_len_per_req)

        num_extends = int(metadata.num_extends or 0)
        seq_lens = metadata.seq_lens_k[
            num_extends : num_extends + decode_window.num_reqs
        ]
        if seq_lens.numel() == 0:
            return None

        block_tables = metadata.block_kv_indices[
            num_extends : num_extends + decode_window.num_reqs
        ]
        draft_catchup = bool(
            getattr(ctx.attn_backend, "is_draft", False)
            and ctx.forward_mode is not None
            and ctx.forward_mode.is_decode()
            and decode_window.q_len_per_req > 1
        )
        seq_lens_per_token = self._expand_decode_seq_lens_per_token(
            seq_lens,
            decode_window.q_len_per_req,
            draft_catchup=draft_catchup,
        )
        block_tables_per_token = (
            block_tables
            if decode_window.q_len_per_req == 1
            else block_tables.repeat_interleave(decode_window.q_len_per_req, dim=0)
        )
        topk = self.index_topk
        full_topk = self._try_compute_decode_full_context_topk_indices(
            ctx,
            num_tokens=num_tokens,
            device=indexer_output.query.device,
        )
        if full_topk is not None:
            return full_topk

        topk_indices, _, topk_lens = self._compute_decode_topk_indices_deepgemm(
            indexer_output=indexer_output,
            ctx=ctx,
            seq_lens_per_token=seq_lens_per_token,
            block_tables=block_tables,
            block_tables_per_token=block_tables_per_token,
            q_len_per_req=decode_window.q_len_per_req,
            decode_start=decode_window.start,
            num_tokens=num_tokens,
            num_decode_tokens=decode_window.num_tokens,
            topk=topk,
        )
        return GlmDsaDecodeTopK(
            topk_indices=topk_indices,
            topk_lens=topk_lens,
        )

    def _compute_decode_topk_indices_deepgemm(
        self,
        *,
        indexer_output: GlmDsaIndexerOutput,
        ctx: ForwardContext,
        seq_lens_per_token: torch.Tensor,
        block_tables: torch.Tensor,
        block_tables_per_token: torch.Tensor,
        q_len_per_req: int,
        decode_start: int,
        num_tokens: int,
        num_decode_tokens: int,
        topk: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if deep_gemm is None:
            raise RuntimeError("GLM DSA decode top-k requires DeepGEMM.")
        if topk not in (512, 1024, 2048):
            raise RuntimeError(f"GLM DSA decode top-k does not support topk={topk}.")
        if (
            not hasattr(ctx.token_to_kv_pool, "has_index_k_with_scale_buffer")
            or not ctx.token_to_kv_pool.has_index_k_with_scale_buffer()
        ):
            raise RuntimeError(
                "GLM DSA decode top-k requires FP8 index-K cache with scales."
            )
        page_size = ctx.token_to_kv_pool.page_size
        if page_size != 64 or self.indexer.index_head_dim != 128:
            raise RuntimeError(
                "GLM DSA decode top-k requires page_size=64 and index_head_dim=128; "
                f"got page_size={page_size}, index_head_dim={self.indexer.index_head_dim}."
            )

        q = indexer_output.query[
            decode_start : decode_start + num_decode_tokens
        ].contiguous()
        q_2d = q.view(-1, self.indexer.index_head_dim)
        q_fp8, q_scale = quantize_fp8_with_scale(
            q_2d,
            granularity="token_group",
            group_size=128,
            scale_encoding="float32",
        )
        q_fp8 = q_fp8.view_as(q)
        q_scale = q_scale.view(num_decode_tokens, self.indexer.index_n_heads, 1)
        weights = (
            indexer_output.weights[decode_start : decode_start + num_decode_tokens]
            .float()
            .unsqueeze(-1)
            * q_scale
            * self.indexer.softmax_scale
        ).squeeze(-1)

        seq_lens_2d = (
            seq_lens_per_token.to(torch.int32).view(-1, q_len_per_req).contiguous()
        )
        block_tables = block_tables.to(torch.int32).contiguous()
        max_seq_len = int(block_tables.shape[1]) * page_size
        if max_seq_len < topk:
            raise RuntimeError(
                "GLM DSA decode top-k requires block table capacity >= topk; "
                f"got capacity={max_seq_len}, topk={topk}."
            )
        decode_metadata = getattr(ctx.attn_backend, "forward_decode_metadata", None)
        schedule_metadata = (
            None
            if decode_metadata is None
            else getattr(
                decode_metadata,
                "_dsa_paged_mqa_schedule_metadata",
                None,
            )
        )
        if schedule_metadata is None:
            schedule_metadata = deep_gemm.get_paged_mqa_logits_metadata(
                seq_lens_2d,
                page_size,
                deep_gemm.get_num_sms(),
            )
            if decode_metadata is not None:
                setattr(
                    decode_metadata,
                    "_dsa_paged_mqa_schedule_metadata",
                    schedule_metadata,
                )
                setattr(
                    decode_metadata,
                    "_dsa_paged_mqa_schedule_q_len",
                    q_len_per_req,
                )
        index_k_with_scale_cache = ctx.token_to_kv_pool.get_index_k_with_scale_buffer(
            self.attn_mqa.layer_id
        )
        kv_cache = index_k_with_scale_cache.view(
            -1,
            page_size,
            1,
            index_k_with_scale_cache.shape[-1],
        )
        logits = deep_gemm.fp8_paged_mqa_logits(
            q_fp8.view(
                -1,
                q_len_per_req,
                self.indexer.index_n_heads,
                self.indexer.index_head_dim,
            ),
            kv_cache,
            weights.contiguous(),
            seq_lens_2d,
            block_tables,
            schedule_metadata,
            max_seq_len,
            clean_logits=False,
        )
        logits.nan_to_num_(
            nan=float("-inf"),
            posinf=float("-inf"),
            neginf=float("-inf"),
        )
        col_ids = self._get_decode_topk_arange(logits.shape[1], q.device)
        logits.masked_fill_(
            col_ids.view(1, -1) >= seq_lens_2d.view(-1, 1),
            float("-inf"),
        )
        local_topk_offsets = self._get_decode_topk_workspace(
            "_decode_local_topk_offsets_buffer",
            num_decode_tokens,
            topk,
            q.device,
        )
        self._write_decode_topk_offsets(
            logits=logits,
            local_topk_offsets=local_topk_offsets,
            topk=topk,
        )
        topk_indices = self._get_decode_topk_workspace(
            "_decode_topk_indices_buffer",
            num_tokens,
            topk,
            q.device,
            fill_value=None,
        )
        topk_slice = topk_indices[decode_start : decode_start + num_decode_tokens]
        topk_lens = self._get_decode_topk_lens_workspace(
            num_tokens,
            q.device,
        )
        topk_lens_slice = topk_lens[decode_start : decode_start + num_decode_tokens]
        local_topk_to_global_slots(
            local_topk_offsets=local_topk_offsets,
            block_table=block_tables_per_token,
            block_size=page_size,
            seq_lens=seq_lens_per_token,
            out=topk_slice,
            lens_out=topk_lens_slice,
        )
        return topk_indices, local_topk_offsets, topk_lens

    def _compute_prefill_topk_indices(
        self,
        indexer_output: GlmDsaIndexerOutput,
        ctx: ForwardContext,
        num_prefill_tokens: int,
    ) -> GlmDsaPrefillTopK | None:
        if ctx.forward_mode is None or not ctx.forward_mode.is_extend_or_mixed():
            return None
        if ctx.num_extends <= 0 or num_prefill_tokens <= 0:
            return None

        chunk_meta = getattr(ctx.attn_backend, "chunked_prefill_metadata", None)
        if chunk_meta is None:
            return None
        prefix_lens = chunk_meta.extend_prefix_lens[: ctx.num_extends].to(torch.int32)
        extend_lens = chunk_meta.extend_seq_lens[: ctx.num_extends].to(torch.int32)
        seq_lens = prefix_lens + extend_lens
        if seq_lens.numel() == 0:
            return None
        if int(extend_lens.sum().item()) != num_prefill_tokens:
            raise RuntimeError(
                "GLM DSA prefill token count mismatch: "
                f"metadata={int(extend_lens.sum().item())}, "
                f"tokens={num_prefill_tokens}"
            )
        if ctx.req_to_page is None:
            raise RuntimeError("GLM DSA sparse prefill requires req_to_page metadata")

        topk = self.index_topk
        page_size = ctx.token_to_kv_pool.page_size
        max_seq_len = int(seq_lens.max().item())
        max_pages = (max_seq_len + page_size - 1) // page_size
        block_tables_snapshot = getattr(ctx.attn_backend, "_prefill_block_tables", None)
        if block_tables_snapshot is None:
            req_pool_indices = chunk_meta.req_pool_indices[: ctx.num_extends].long()
            block_tables_snapshot = ctx.req_to_page[req_pool_indices]
        block_tables = block_tables_snapshot[:, :max_pages].to(
            device=indexer_output.query.device,
            dtype=torch.int32,
        )
        kv_workspace_slots, kv_workspace_bases = _build_prefill_kv_workspace_slots(
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_seq_len=max_seq_len,
            page_size=page_size,
            device=indexer_output.query.device,
        )
        return self._compute_prefill_topk_indices_deepgemm(
            indexer_output=indexer_output,
            ctx=ctx,
            prefix_lens=prefix_lens,
            extend_lens=extend_lens,
            seq_lens=seq_lens,
            block_tables=block_tables,
            kv_workspace_slots=kv_workspace_slots,
            kv_workspace_bases=kv_workspace_bases,
            max_seq_len=max_seq_len,
            num_prefill_tokens=num_prefill_tokens,
            topk=topk,
        )

    def _compute_prefill_topk_indices_deepgemm(
        self,
        *,
        indexer_output: GlmDsaIndexerOutput,
        ctx: ForwardContext,
        prefix_lens: torch.Tensor,
        extend_lens: torch.Tensor,
        seq_lens: torch.Tensor,
        block_tables: torch.Tensor,
        kv_workspace_slots: torch.Tensor,
        kv_workspace_bases: torch.Tensor,
        max_seq_len: int,
        num_prefill_tokens: int,
        topk: int,
    ) -> GlmDsaPrefillTopK:
        if deep_gemm is None:
            raise RuntimeError("GLM DSA prefill top-k requires DeepGEMM.")
        if topk not in (512, 1024, 2048):
            raise RuntimeError(f"GLM DSA prefill top-k does not support topk={topk}.")
        if (
            not hasattr(ctx.token_to_kv_pool, "has_index_k_with_scale_buffer")
            or not ctx.token_to_kv_pool.has_index_k_with_scale_buffer()
        ):
            raise RuntimeError(
                "GLM DSA prefill top-k requires FP8 index-K cache with scales."
            )
        page_size = ctx.token_to_kv_pool.page_size
        if page_size != 64 or self.indexer.index_head_dim != 128:
            raise RuntimeError(
                "GLM DSA prefill top-k requires page_size=64 and index_head_dim=128; "
                f"got page_size={page_size}, index_head_dim={self.indexer.index_head_dim}."
            )
        if num_prefill_tokens <= 0 or seq_lens.numel() == 0:
            raise RuntimeError("GLM DSA prefill top-k requires at least one token.")

        seq_len_sum = int(seq_lens.sum().item())
        max_logits_mb = int(global_server_args_dict[_INDEXER_PREFILL_MAX_LOGITS_MB_ARG])
        q = indexer_output.query[:num_prefill_tokens].contiguous()
        q_2d = q.view(-1, self.indexer.index_head_dim)
        q_fp8, q_scale = quantize_fp8_with_scale(
            q_2d,
            granularity="token_group",
            group_size=128,
            scale_encoding="float32",
        )
        q_fp8 = q_fp8.view_as(q)
        q_scale = q_scale.view(num_prefill_tokens, self.indexer.index_n_heads, 1)
        weights = (
            indexer_output.weights[:num_prefill_tokens].float().unsqueeze(-1)
            * q_scale
            * self.indexer.softmax_scale
        ).squeeze(-1)

        k_fp8, k_scale = ctx.token_to_kv_pool.gather_index_k_with_scale(
            self.attn_mqa.layer_id,
            kv_workspace_slots,
        )
        k_fp8 = k_fp8.view(q_fp8.dtype)
        kv_fp8 = (k_fp8.contiguous(), k_scale.squeeze(-1).contiguous())

        req_ids = torch.arange(
            seq_lens.numel(),
            dtype=torch.int64,
            device=q.device,
        )
        token_req = torch.repeat_interleave(req_ids, extend_lens.to(torch.int64))
        extend_cu = torch.zeros(
            extend_lens.numel() + 1,
            dtype=torch.int64,
            device=q.device,
        )
        torch.cumsum(extend_lens.to(torch.int64), dim=0, out=extend_cu[1:])
        token_offsets = torch.arange(
            num_prefill_tokens, dtype=torch.int64, device=q.device
        ) - extend_cu.index_select(0, token_req)
        causal_lens = (
            prefix_lens.to(torch.int64).index_select(0, token_req) + token_offsets + 1
        )
        seq_cu = torch.zeros(
            seq_lens.numel() + 1,
            dtype=torch.int64,
            device=q.device,
        )
        torch.cumsum(seq_lens.to(torch.int64), dim=0, out=seq_cu[1:])
        row_starts = seq_cu.index_select(0, token_req)
        row_ends = row_starts + causal_lens

        workspace_indices = torch.full(
            (num_prefill_tokens, topk),
            -1,
            dtype=torch.int32,
            device=q.device,
        )
        trtllm_ops = getattr(torch.ops, "trtllm", None)
        if trtllm_ops is None or not hasattr(trtllm_ops, "indexer_topk_prefill"):
            raise RuntimeError(
                "GLM DSA prefill top-k requires torch.ops.trtllm.indexer_topk_prefill."
            )

        seq_len_sum = max(seq_len_sum, 1)
        max_logits_bytes = max(1, max_logits_mb) * 1024 * 1024
        max_query_rows = max(1, max_logits_bytes // (seq_len_sum * 4))

        row_starts_i32 = row_starts.to(torch.int32).contiguous()
        row_ends_i32 = row_ends.to(torch.int32).contiguous()
        for start in range(0, num_prefill_tokens, max_query_rows):
            end = min(start + max_query_rows, num_prefill_tokens)
            logits = deep_gemm.fp8_mqa_logits(
                q_fp8[start:end].contiguous(),
                kv_fp8,
                weights[start:end].contiguous(),
                row_starts_i32[start:end].contiguous(),
                row_ends_i32[start:end].contiguous(),
                clean_logits=False,
                max_seqlen_k=int(causal_lens[start:end].max().item()),
            )
            logits.nan_to_num_(
                nan=float("-inf"),
                posinf=float("-inf"),
                neginf=float("-inf"),
            )
            trtllm_ops.indexer_topk_prefill(
                logits.contiguous(),
                row_starts_i32[start:end].contiguous(),
                row_ends_i32[start:end].contiguous(),
                workspace_indices[start:end],
                topk,
            )
        valid_topk = workspace_indices >= 0
        workspace_indices = torch.where(
            valid_topk,
            workspace_indices
            + kv_workspace_bases.to(torch.int32)
            .index_select(
                0,
                token_req,
            )
            .unsqueeze(1),
            workspace_indices,
        )
        topk_lens = torch.minimum(
            causal_lens,
            torch.full_like(causal_lens, topk),
        ).to(torch.int32)
        return GlmDsaPrefillTopK(
            workspace_indices=workspace_indices,
            topk_lens=topk_lens,
            block_tables=block_tables,
            seq_lens=seq_lens.to(device=q.device, dtype=torch.int32),
            max_seq_len=max_seq_len,
            kv_workspace_slots=kv_workspace_slots,
        )

    def _forward_dsa_indexer(
        self,
        *,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        q_lora: torch.Tensor,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
        comm_manager: CommManager,
        key_only: bool = False,
    ) -> GlmDsaIndexerOutput | None:
        if positions.numel() == 0:
            return None
        if not hasattr(ctx.token_to_kv_pool, "set_index_k_buffer"):
            return None

        indexer_hidden_states = comm_manager.pre_attn_comm(hidden_states, ctx)
        if key_only:
            index_k = self.indexer.forward_key_only(indexer_hidden_states, positions)
            ctx.token_to_kv_pool.set_index_k_buffer(
                self.attn_mqa.layer_id,
                out_cache_loc,
                index_k,
            )
            return None

        indexer_output = self.indexer(indexer_hidden_states, q_lora, positions)
        ctx.token_to_kv_pool.set_index_k_buffer(
            self.attn_mqa.layer_id,
            out_cache_loc,
            indexer_output.key,
        )
        return indexer_output

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
        comm_manager: CommManager,
        block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        qkv = self.fused_qkv_a_proj_with_mqa(
            hidden_states,
            block_scale,
            torch.bfloat16,
        )
        # The fused QKV-A weight may be zero-padded on its output dim to a
        # multiple of 128 so the FP8 block-scale GEMM stays numerically valid
        # (see GlmMoeDsaForCausalLM._pad_fused_qkv_a_proj_for_fp8_blockscale).
        # Drop the padding columns before the split / comm. No-op when the
        # projection output already matches the logical width.
        _qkv_width = self.q_lora_rank + self.kv_lora_rank + self.qk_rope_head_dim
        if qkv.shape[-1] != _qkv_width:
            qkv = qkv[..., :_qkv_width]
        qkv = comm_manager.pre_attn_comm(qkv, ctx)
        q_a, latent_cache = qkv.split(
            [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
            dim=-1,
        )
        kv_a = latent_cache[..., : self.kv_lora_rank]
        q_norm = torch.empty_like(q_a)
        if q_a.size(0) > 0:
            self.fused_qk_layernorm(
                input_q_a=q_a,
                input_kv_a=kv_a,
                output_q_a=q_norm,
            )

        decode_metadata = getattr(ctx.attn_backend, "forward_decode_metadata", None)
        decode_window = self._resolve_decode_window(
            ctx,
            decode_metadata,
            total_tokens=hidden_states.shape[0],
        )
        num_decode_tokens = decode_window.num_tokens
        num_prefill_tokens = decode_window.start
        decode_start = decode_window.start
        decode_end = decode_window.end

        carried_prefill_topk = getattr(ctx, "dsa_prefill_topk", None)
        carried_decode_topk = getattr(ctx, "dsa_decode_topk", None)
        carried_decode_topk_covers_window = self._decode_topk_covers_window(
            carried_decode_topk,
            decode_start,
            decode_end,
        )
        should_compute_indexer = not self.skip_indexer_topk or (
            self.is_nextn
            and (
                (num_prefill_tokens > 0 and carried_prefill_topk is None)
                or (num_decode_tokens > 0 and not carried_decode_topk_covers_window)
            )
        )

        indexer_output = None
        full_context_decode_topk = None
        if should_compute_indexer:
            full_context_decode_topk = (
                self._try_compute_decode_full_context_topk_indices(
                    ctx,
                    num_tokens=hidden_states.shape[0],
                    device=hidden_states.device,
                )
            )
            key_only_indexer = (
                full_context_decode_topk is not None
                and _glm_dsa_is_pure_decode_token_mode(ctx.forward_mode)
                and ctx.num_extends == 0
            )
            indexer_output = self._forward_dsa_indexer(
                positions=positions,
                hidden_states=hidden_states,
                q_lora=q_norm,
                ctx=ctx,
                out_cache_loc=out_cache_loc,
                comm_manager=comm_manager,
                key_only=key_only_indexer,
            )
        q = self.q_b_proj(q_norm)[0]

        if not should_compute_indexer:
            prefill_topk = carried_prefill_topk
            decode_topk = carried_decode_topk
        else:
            prefill_topk = (
                self._compute_prefill_topk_indices(
                    indexer_output,
                    ctx,
                    num_prefill_tokens,
                )
                if indexer_output is not None
                else None
            )
            decode_topk = full_context_decode_topk
            if decode_topk is None and indexer_output is not None:
                decode_topk = self._compute_decode_topk_indices(indexer_output, ctx)
            ctx.dsa_prefill_topk = prefill_topk
            ctx.dsa_decode_topk = decode_topk
        attn_output = torch.empty(
            q.size(0),
            self.num_local_heads * self.v_head_dim,
            dtype=q.dtype,
            device=q.device,
        )

        if ctx.num_extends > 0:
            prefill_ctx = replace(
                ctx,
                bs=ctx.num_extends,
                input_num_tokens=num_prefill_tokens,
                forward_mode=ForwardMode.EXTEND,
            )
            if prefill_topk is None:
                raise RuntimeError(
                    "GLM DSA sparse prefill requires computed top-k indices."
                )
            self.forward_dsa_sparse_prefill(
                positions[:num_prefill_tokens],
                q[:num_prefill_tokens],
                latent_cache[:num_prefill_tokens],
                prefill_ctx,
                out_cache_loc[:num_prefill_tokens],
                attn_output[:num_prefill_tokens],
                prefill_topk=prefill_topk,
            )

        if num_decode_tokens > 0:
            decode_ctx = replace(
                ctx,
                bs=decode_window.num_reqs,
                num_extends=0,
                input_num_tokens=num_decode_tokens,
                forward_mode=ForwardMode.DECODE,
            )
            if decode_topk is None:
                raise RuntimeError(
                    "GLM DSA sparse decode requires computed top-k indices."
                )
            topk_indices, topk_lens = self._slice_decode_topk(
                decode_topk,
                decode_start,
                decode_end,
            )
            self.forward_absorb(
                positions[decode_start:decode_end],
                q[decode_start:decode_end],
                latent_cache[decode_start:decode_end],
                decode_ctx,
                out_cache_loc[decode_start:decode_end],
                attn_output[decode_start:decode_end],
                topk_indices=topk_indices,
                topk_lens=topk_lens,
            )

        if ctx.draft_first_step_reduce:
            attn_output = attn_output.index_select(0, ctx.gather_ids)
        output, _ = self.o_proj(attn_output)
        return output

    def forward_dsa_sparse_prefill(
        self,
        positions: torch.Tensor,
        q: torch.Tensor,
        latent_cache: torch.Tensor,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
        output: torch.Tensor,
        *,
        prefill_topk: GlmDsaPrefillTopK,
    ) -> torch.Tensor:
        Q, _ = self.forward_absorb_qkv_proj(
            q,
            latent_cache,
            positions,
            ctx,
            out_cache_loc,
        )
        attn_output = ctx.attn_backend.forward_sparse_prefill(
            q=Q,
            layer=self.attn_mqa,
            token_to_kv_pool=ctx.token_to_kv_pool,
            block_tables=prefill_topk.block_tables,
            seq_lens=prefill_topk.seq_lens,
            workspace_indices=prefill_topk.workspace_indices,
            topk_lens=prefill_topk.topk_lens,
            kv_workspace_slots=prefill_topk.kv_workspace_slots,
            max_seq_len=prefill_topk.max_seq_len,
        )
        attn_output = attn_output.view(-1, self.num_local_heads, self.kv_lora_rank)
        output_view = output.view(-1, self.num_local_heads, self.v_head_dim)
        torch.bmm(
            attn_output.transpose(0, 1),
            self.w_vc,
            out=output_view.transpose(0, 1),
        )
        return output

    def forward_absorb(
        self,
        positions: torch.Tensor,
        q: torch.Tensor,
        latent_cache: torch.Tensor,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
        output: torch.Tensor,
        topk_indices: torch.Tensor | None = None,
        topk_lens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        Q, K = self.forward_absorb_qkv_proj(
            q,
            latent_cache,
            positions,
            ctx,
            out_cache_loc,
        )
        return self.forward_absorb_attn_v_proj(
            Q,
            K,
            ctx,
            out_cache_loc,
            output,
            topk_indices=topk_indices,
            topk_lens=topk_lens,
        )

    def forward_absorb_attn_v_proj(
        self,
        Q,
        K,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
        output: torch.Tensor,
        topk_indices: torch.Tensor | None = None,
        topk_lens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        need_save_kv = False
        if self.attention_backend not in self._MLA_KERNEL_BACKENDS:
            need_save_kv = not self.use_fused_set_kv_buffer

        attn_output = self.attn_mqa(
            Q,
            K,
            K[..., : self.kv_lora_rank],
            ctx,
            out_cache_loc,
            save_kv_cache=need_save_kv,
            topk_indices=topk_indices,
            topk_lens=topk_lens,
        )
        attn_output = attn_output.view(-1, self.num_local_heads, self.kv_lora_rank)
        output_view = output.view(-1, self.num_local_heads, self.v_head_dim)
        torch.bmm(
            attn_output.transpose(0, 1),
            self.w_vc,
            out=output_view.transpose(0, 1),
        )
        return output


class GlmMoeDsaDecoderLayer(DeepseekV3DecoderLayer):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        is_nextn: bool = False,
        prefix: str = "",
        alt_stream: torch.cuda.Stream | None = None,
    ) -> None:
        nn.Module.__init__(self)
        self.mapping = mapping
        self.hidden_size = config.hidden_size
        rope_theta = get_rope_theta(config)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)

        self.self_attn = GlmMoeDsaAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            q_lora_rank=(
                config.q_lora_rank if hasattr(config, "q_lora_rank") else None
            ),
            kv_lora_rank=config.kv_lora_rank,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=(
                None
                if "self_attn" in getattr(config, "disable_quant_module", [])
                else quant_config
            ),
            layer_id=layer_id,
            prefix=add_prefix("self_attn", prefix),
            reduce_attn_results=False,
            alt_stream=alt_stream,
            mapping=self.mapping,
            is_nextn=is_nextn,
        )

        self.layer_id = layer_id
        self.is_moe_layer = self._is_moe_layer(layer_id, is_nextn, config)
        if self.is_moe_layer:
            self.mlp = DeepseekV3MoE(
                config=config,
                mapping=self.mapping,
                quant_config=quant_config,
                layer_index=layer_id,
                prefix=add_prefix("mlp", prefix),
                alt_stream=alt_stream,
            )
        else:
            self.mlp = DeepseekV3MLP(
                hidden_size=config.hidden_size,
                intermediate_size=(
                    config.ffn_hidden_size
                    if hasattr(config, "ffn_hidden_size")
                    else config.intermediate_size
                ),
                hidden_act=config.hidden_act,
                mapping=self.mapping,
                quant_config=(
                    None
                    if "dense_mlp" in getattr(config, "disable_quant_module", [])
                    else quant_config
                ),
                prefix=add_prefix("mlp", prefix),
                is_shared_expert=False,
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.comm_manager = CommManager(
            mapping=self.mapping,
            layer_id=self.layer_id,
            is_moe=self.is_moe_layer,
            prev_is_moe=self._is_moe_layer(layer_id - 1, is_nextn, config),
            input_layernorm=self.input_layernorm,
            post_attn_layernorm=self.post_attention_layernorm,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> torch.Tensor:
        num_global_tokens, max_num_tokens_per_gpu = self.comm_manager.get_num_tokens(
            ctx
        )

        if not ctx.forward_mode.is_idle():
            hidden_states, residual = self.comm_manager.input_reduce_norm(
                hidden_states, residual
            )
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                ctx=ctx,
                out_cache_loc=out_cache_loc,
                comm_manager=self.comm_manager,
            )
            if ctx.draft_first_step_reduce:
                residual = residual.index_select(0, ctx.gather_ids)
            hidden_states, residual = self.comm_manager.post_attn_reduce_norm(
                hidden_states, residual, ctx
            )
            hidden_states = self.forward_mlp(
                hidden_states,
                residual,
                ctx,
                num_global_tokens,
                max_num_tokens_per_gpu,
            )
        else:
            hidden_states = self.forward_mlp(
                hidden_states,
                residual,
                ctx,
                num_global_tokens,
                max_num_tokens_per_gpu,
            )
        return hidden_states, residual

    def forward_mlp(
        self,
        hidden_states,
        residual,
        ctx: ForwardContext,
        num_global_tokens,
        max_num_tokens_per_gpu,
    ):
        hidden_states = self.comm_manager.pre_mlp_comm(hidden_states, ctx)
        if self.is_moe_layer:
            hidden_states = self.mlp(
                hidden_states, num_global_tokens, max_num_tokens_per_gpu
            )
        else:
            hidden_states = self.mlp(hidden_states)
        hidden_states, residual = self.comm_manager.post_mlp_fused(
            hidden_states, residual, ctx
        )
        return hidden_states


class GlmMoeDsaModel(DeepseekV3Model):
    def __init__(
        self,
        config: PretrainedConfig,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.mapping = mapping
        self.padding_id = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )
        self.alt_stream = torch.cuda.Stream()
        self.layers = nn.ModuleList(
            [
                GlmMoeDsaDecoderLayer(
                    config,
                    layer_id,
                    mapping=self.mapping,
                    quant_config=quant_config,
                    prefix=add_prefix(f"layers.{layer_id}", prefix),
                    alt_stream=self.alt_stream,
                )
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layers_to_capture: set = set()


def pad_fused_qkv_a_proj_weight_for_fp8_blockscale(attn) -> None:
    """Pad one attention module's fused QKV-A projection output dim to 128.

    The FP8 block-scale dense GEMM (deep_gemm / default ``mm`` path) returns NaN
    when the output dim ``N`` is not a multiple of the 128 scale block. GLM-5.1's
    fused QKV-A projection has ``N = q_lora_rank + kv_lora_rank + qk_rope_head_dim``
    (e.g. 2624), which is not 128-aligned, so attention output goes NaN. We
    zero-pad the FP8 weight rows up to the next 128 multiple;
    ``weight_scale_inv`` already has ``ceil(N/128)`` row blocks (covering the
    padded rows) and the downstream ``qkv.split(...)`` drops the padding rows, so
    real outputs are unchanged. No-op for bf16 weights or already-aligned ``N``.

    Shared by the main model (per decoder layer) and the NextN draft model (its
    single DSA decoder), both of which carry the same fused QKV-A projection.

    Args:
        attn: A GLM DSA attention module exposing ``fused_qkv_a_proj_with_mqa``.
    """
    fp8_dtypes = (torch.float8_e4m3fn, getattr(torch, "float8_e4m3fnuz", None))
    fp8_dtypes = tuple(d for d in fp8_dtypes if d is not None)
    proj = getattr(attn, "fused_qkv_a_proj_with_mqa", None)
    weight = getattr(proj, "weight", None)
    if weight is None or weight.dtype not in fp8_dtypes:
        return
    n = weight.shape[0]
    if n % 128 == 0:
        return
    n_pad = ((n + 127) // 128) * 128
    pad = weight.new_zeros(n_pad - n, weight.shape[1])
    proj.weight = torch.nn.Parameter(
        torch.cat([weight.data, pad], dim=0), requires_grad=False
    )


class GlmMoeDsaForCausalLM(DeepseekV3ForCausalLM):
    model_cls = GlmMoeDsaModel

    def _record_fused_indexer_projection_shard(
        self,
        *,
        module_name: str,
        shard_id: int,
        loaded_shards: dict[str, set[int]],
        modules_dict: dict[str, nn.Module],
    ) -> None:
        shards = loaded_shards.setdefault(module_name, set())
        shards.add(int(shard_id))
        if shards != {0, 1}:
            return

        module = modules_dict.get(module_name)
        if isinstance(module, GlmDsaIndexer):
            module.set_wk_weights_proj_loaded()

    def _load_fused_indexer_projection_shard(
        self,
        *,
        module_name: str,
        shard_id: int,
        loaded_weight: torch.Tensor,
        params_dict: dict[str, torch.Tensor],
        modules_dict: dict[str, nn.Module],
        loaded_shards: dict[str, set[int]],
    ) -> bool:
        param = params_dict.get(f"{module_name}.wk_weights_proj.weight")
        if param is None:
            return False

        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        weight_loader(param, loaded_weight, shard_id)
        self._record_fused_indexer_projection_shard(
            module_name=module_name,
            shard_id=shard_id,
            loaded_shards=loaded_shards,
            modules_dict=modules_dict,
        )
        return True

    def _flush_fused_indexer_fp8_wk(
        self,
        *,
        module_name: str,
        pending_fp8_wk: dict[str, dict[str, torch.Tensor]],
        params_dict: dict[str, torch.Tensor],
        modules_dict: dict[str, nn.Module],
        loaded_shards: dict[str, set[int]],
    ) -> None:
        entry = pending_fp8_wk.get(module_name)
        if not entry or "weight" not in entry or "scale" not in entry:
            return
        weight_block_size = getattr(self.quant_config, "weight_block_size", None)
        if weight_block_size is None:
            return

        weight_fp8 = entry["weight"]
        scale = entry["scale"]
        weight_bf16 = block_dequant(
            weight_fp8,
            scale,
            list(weight_block_size),
        ).to(torch.bfloat16)
        if self._load_fused_indexer_projection_shard(
            module_name=module_name,
            shard_id=0,
            loaded_weight=weight_bf16,
            params_dict=params_dict,
            modules_dict=modules_dict,
            loaded_shards=loaded_shards,
        ):
            del pending_fp8_wk[module_name]

    def _try_load_fused_indexer_projection(
        self,
        *,
        name: str,
        loaded_weight: torch.Tensor,
        params_dict: dict[str, torch.Tensor],
        modules_dict: dict[str, nn.Module],
        pending_fp8_wk: dict[str, dict[str, torch.Tensor]],
        loaded_shards: dict[str, set[int]],
    ) -> None:
        if ".indexer.wk_weights_proj." in name:
            return

        if ".indexer.weights_proj.weight" in name:
            module_name = name.rsplit(".weights_proj.weight", 1)[0]
            self._load_fused_indexer_projection_shard(
                module_name=module_name,
                shard_id=1,
                loaded_weight=loaded_weight,
                params_dict=params_dict,
                modules_dict=modules_dict,
                loaded_shards=loaded_shards,
            )
            return

        if ".indexer.wk." not in name:
            return

        module_name = name.rsplit(".wk.", 1)[0]
        if name.endswith(".weight") and loaded_weight.dtype in (
            torch.float8_e4m3fn,
            torch.float8_e4m3fnuz,
        ):
            pending_fp8_wk.setdefault(module_name, {})["weight"] = loaded_weight
            self._flush_fused_indexer_fp8_wk(
                module_name=module_name,
                pending_fp8_wk=pending_fp8_wk,
                params_dict=params_dict,
                modules_dict=modules_dict,
                loaded_shards=loaded_shards,
            )
            return

        if name.endswith(".weight"):
            self._load_fused_indexer_projection_shard(
                module_name=module_name,
                shard_id=0,
                loaded_weight=loaded_weight,
                params_dict=params_dict,
                modules_dict=modules_dict,
                loaded_shards=loaded_shards,
            )
            return

        if "weight_scale_inv" in name:
            pending_fp8_wk.setdefault(module_name, {})["scale"] = loaded_weight
            self._flush_fused_indexer_fp8_wk(
                module_name=module_name,
                pending_fp8_wk=pending_fp8_wk,
                params_dict=params_dict,
                modules_dict=modules_dict,
                loaded_shards=loaded_shards,
            )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        params_dict = dict(self.named_parameters())
        modules_dict = dict(self.named_modules())
        pending_fp8_wk: dict[str, dict[str, torch.Tensor]] = {}
        loaded_fused_indexer_shards: dict[str, set[int]] = {}

        def base_weights():
            for name, loaded_weight in weights:
                layer_id = get_layer_id(name)
                if layer_id is not None and layer_id >= self.config.num_hidden_layers:
                    continue
                if "rotary_emb.inv_freq" in name:
                    continue
                if ".indexer." not in name:
                    yield name, loaded_weight
                    continue

                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = self.get_param(params_dict, name)
                if param is None:
                    continue
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                self._try_load_fused_indexer_projection(
                    name=name,
                    loaded_weight=loaded_weight,
                    params_dict=params_dict,
                    modules_dict=modules_dict,
                    pending_fp8_wk=pending_fp8_wk,
                    loaded_shards=loaded_fused_indexer_shards,
                )

        super().load_weights(base_weights())
        self._pad_fused_qkv_a_proj_for_fp8_blockscale()

    def _pad_fused_qkv_a_proj_for_fp8_blockscale(self) -> None:
        """Pad each decoder layer's fused QKV-A projection to a 128-multiple.

        See :func:`pad_fused_qkv_a_proj_weight_for_fp8_blockscale` for why this
        is needed (FP8 block-scale GEMM returns NaN for non-128-aligned ``N``).
        """
        for layer in getattr(self.model, "layers", []):
            attn = getattr(layer, "self_attn", None)
            if attn is not None:
                pad_fused_qkv_a_proj_weight_for_fp8_blockscale(attn)


EntryClass = [GlmMoeDsaForCausalLM]

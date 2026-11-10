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

"""Kimi-K3.1: Kimi-K3 plus a DSA lightning indexer on every MLA layer.

Everything but the MLA layers is Kimi-K3 (``kimi_k3.py``). An MLA layer keeps
K3's gated NoPE MLA projections and output gate, and attends only the
``index_topk`` history rows its own indexer selects (DeepSeek-V3.2 style
sparse attention). The indexer scores, for query token ``t`` and history token
``s <= t``::

    q_I = wq_b(c_q)                  # [index_n_heads, index_head_dim]
    k_I = LayerNorm(wk(x_s))         # [index_head_dim], weight and bias
    w   = weights_proj(x_t)          # [index_n_heads]
    I   = d^-0.5 * H^-0.5 * sum_h w_h * relu(q_I,h . k_I)

with ``c_q`` the normalized MLA q latent and ``x`` the attention input the
MLA a-projection reads. There is no RoPE part and no Hadamard rotation, and
every DSA layer selects its own top-k. The checkpoint ships no modeling code,
so these semantics follow DeepSeek-V3.2/GLM-5 prior art and the tensor shapes.

The selection and the sparse prefill/decode cores are GLM-DSA's
(:class:`GlmDsaSparseAttentionMixin`); the index keys live in the K3 cache
family's index-K planes (``KimiK3Recipe``, ``HybridKDADSATokenToKVPool``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING

import torch
from tokenspeed_kernel.ops.activation.triton import sigmoid_mul
from tokenspeed_kernel.ops.attention.dsa import cute_dsl as dsa_cute_dsl
from tokenspeed_kernel.ops.attention.mla import mla_project_value
from tokenspeed_kernel.ops.gemm import kimi3_mla_gate_projection, mm
from tokenspeed_kernel.selection import NoKernelFoundError, select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature
from torch import nn

from tokenspeed.runtime.configs.kimi_k3_config import KimiLinearConfig
from tokenspeed.runtime.configs.numerics import BITWISE_ENVELOPES
from tokenspeed.runtime.distributed.comm_manager import CommManager
from tokenspeed.runtime.distributed.mapping import DenseLayerMapping, Mapping
from tokenspeed.runtime.execution.breakable_cuda_graph import (
    break_point,
    is_breakable_capture_active,
)
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.hybrid.linear import (
    HybridLinearAttnBackend,
)
from tokenspeed.runtime.layers.layernorm import LayerNorm
from tokenspeed.runtime.layers.linear import ReplicatedLinear
from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig
from tokenspeed.runtime.models.glm5 import (
    GlmDsaIndexerOutput,
    GlmDsaSparseAttentionMixin,
)
from tokenspeed.runtime.models.kimi_k3 import (
    KimiK3ForConditionalGeneration,
    KimiLinearMLAAttention,
)
from tokenspeed.runtime.utils import add_prefix
from tokenspeed.runtime.utils.cuda_stream import StreamFork
from tokenspeed.runtime.utils.env import global_server_args_dict

if TYPE_CHECKING:
    from tokenspeed_kernel.ops.communication._iris.kimi_k3 import K3IrisState

    from tokenspeed.runtime.execution.context import ForwardContext

# q_a + kv_a + RoPE-key width the Iris MoE tail projects ahead (MLA_QKV_A_WIDTH).
_PRECOMPUTED_QKV_A_WIDTH = 2112

# The checkpoint names no epsilon for the index-key LayerNorm; 1e-6 is DeepSeek-V3.2's.
_INDEX_K_NORM_EPS = 1e-6

# GB300 AL8: the front saves 2 us/DSA layer at 8-32 rows and costs 7 at 64.
_DECODE_FRONT_MAX_ROWS = 32


class KimiK31Indexer(nn.Module):
    """K3.1 lightning indexer: projections only, no RoPE, no Hadamard.

    The weights are replicated on every attention rank (the score sums over
    all index heads before the top-k, so a head shard would need a
    collective of full score rows). The checkpoint stores them in BF16:
    ``self_attn.*`` is in its quantization ignore list.
    """

    def __init__(self, config: KimiLinearConfig, *, prefix: str) -> None:
        super().__init__()
        self.index_topk = config.index_topk
        self.index_n_heads = config.index_n_heads
        self.index_head_dim = config.index_head_dim
        # The positive score scale; the top-k scorer multiplies by it.
        self.weights_softmax_scale = (
            self.index_head_dim**-0.5 * self.index_n_heads**-0.5
        )
        self.wq_b = ReplicatedLinear(
            config.q_lora_rank,
            self.index_n_heads * self.index_head_dim,
            bias=False,
            quant_config=None,
            prefix=add_prefix("wq_b", prefix),
        )
        self.wk = ReplicatedLinear(
            config.hidden_size,
            self.index_head_dim,
            bias=False,
            quant_config=None,
            prefix=add_prefix("wk", prefix),
        )
        self.weights_proj = ReplicatedLinear(
            config.hidden_size,
            self.index_n_heads,
            bias=False,
            quant_config=None,
            prefix=add_prefix("weights_proj", prefix),
        )
        self.k_norm = LayerNorm(self.index_head_dim, eps=_INDEX_K_NORM_EPS)
        # [wk; weights_proj] stacked after load: one GEMM over the attention input.
        self.wk_weights_proj_weight: torch.Tensor | None = None

    def fuse_key_weight_projections(self) -> None:
        """Stack ``wk`` and ``weights_proj`` into one weight.

        The weight is ``[index_head_dim + index_n_heads, hidden]``. The two
        parameters become views of its rows, so later in-place weight loads keep
        it current, and a repeated call (a weight update's) keeps it. Bitwise
        numerics keep the two planned GEMMs.
        """
        if (
            global_server_args_dict["numerics"] in BITWISE_ENVELOPES
            or self.wk_weights_proj_weight is not None
        ):
            return
        fused = torch.cat((self.wk.weight.data, self.weights_proj.weight.data))
        self.wk.weight.data = fused[: self.index_head_dim]
        self.weights_proj.weight.data = fused[self.index_head_dim :]
        self.wk_weights_proj_weight = fused

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_lora: torch.Tensor,
        positions: torch.Tensor,
    ) -> GlmDsaIndexerOutput:
        """Project the index query, key and head weights.

        Args:
            hidden_states: ``[T, hidden]`` attention input.
            q_lora: ``[T, q_lora_rank]`` normalized MLA q latent.
            positions: Unused: the indexer has no positional part.

        Returns:
            The index query (``q_lora`` with ``query_proj`` to project it on
            demand, outside bitwise numerics), the pre-norm ``key``
            ``[T, index_head_dim]`` with ``key_norm`` for the index-K write to
            apply, and raw ``weights`` ``[T, index_n_heads]``.
        """
        del positions
        if self.wk_weights_proj_weight is not None:
            return self.from_key_weights(
                q_lora, mm(hidden_states, self.wk_weights_proj_weight)
            )
        query, query_proj = self._query(q_lora)
        return GlmDsaIndexerOutput(
            query=query,
            key=self.wk(hidden_states)[0],
            weights=self.weights_proj(hidden_states)[0],
            key_norm=self.k_norm,
            query_proj=query_proj,
        )

    def from_key_weights(
        self, q_lora: torch.Tensor, key_weights: torch.Tensor
    ) -> GlmDsaIndexerOutput:
        """``forward`` over index keys and head weights already projected.

        Args:
            q_lora: ``[T, q_lora_rank]`` normalized MLA q latent.
            key_weights: ``[T, index_head_dim + index_n_heads]`` attention
                input times ``[wk; weights_proj]^T`` with unit inner stride;
                columns of a wider projection's output are fine.

        Returns:
            As ``forward``, with ``key`` and ``weights`` views of
            ``key_weights``.
        """
        query, query_proj = self._query(q_lora)
        # Strided views; the index-K writes and the top-k scorers take strided rows.
        key, weights = key_weights.split(
            (self.index_head_dim, self.index_n_heads), dim=-1
        )
        return GlmDsaIndexerOutput(
            query=query,
            key=key,
            weights=weights,
            key_norm=self.k_norm,
            query_proj=query_proj,
        )

    def _query(
        self, q_lora: torch.Tensor
    ) -> tuple[torch.Tensor, Callable[[torch.Tensor], torch.Tensor] | None]:
        """The index query, and the projection still to apply to it or None.

        Under bitwise numerics the query is projected now; otherwise it is
        ``q_lora`` with the projection to apply on demand.
        """
        if global_server_args_dict["numerics"] in BITWISE_ENVELOPES:
            # One GEMM over all rows: a row's bits must not depend on its batch.
            return self._project_query(q_lora), None
        return q_lora, self._project_query

    def _project_query(self, q_lora: torch.Tensor) -> torch.Tensor:
        return self.wq_b(q_lora)[0].view(-1, self.index_n_heads, self.index_head_dim)


def _scorers_take_fp8_index_queries(indexer: KimiK31Indexer) -> bool:
    """Whether both paged top-k scorers (prefill and decode) of an
    ``fp8_scaled`` plane are registered for this indexer's FP8 E4M3 queries
    on this platform."""
    signature = format_signature(
        q=dense_tensor_format(torch.float8_e4m3fn),
        weights=dense_tensor_format(torch.bfloat16),
    )
    traits = {
        "index_heads": indexer.index_n_heads,
        "head_dim": indexer.index_head_dim,
        "topk": indexer.index_topk,
        "index_k_format": "fp8_scaled",
    }
    try:
        for mode in ("dsa_prefill_topk", "dsa_decode_topk"):
            select_kernel("attention", mode, signature, traits=traits)
    except NoKernelFoundError:
        return False
    return True


def _fused_decode_topk_solution(indexer: KimiK31Indexer) -> str | None:
    """``"cute_dsl"`` where the fused decode top-k serves this indexer, else None.

    It serves BF16 queries, and the E4M3 ones the index-K write quantizes, over
    a packed ``fp8_scaled`` plane, 1-8 rows per request, and is pinned only
    where ``max_num_seqs`` fits the requests one of its rounds schedules. It scores
    bit-identically to the DeepGEMM leaf and gives exact ties at the threshold
    the lowest positions.
    """
    signatures = (
        format_signature(
            q=dense_tensor_format(torch.bfloat16),
            weights=dense_tensor_format(torch.bfloat16),
        ),
        format_signature(
            q=dense_tensor_format(torch.float8_e4m3fn),
            weights=dense_tensor_format(torch.float32),
        ),
    )
    traits = {
        "q_len": frozenset(range(1, 9)),
        "index_heads": indexer.index_n_heads,
        "head_dim": indexer.index_head_dim,
        "page_size": 64,
        "topk": indexer.index_topk,
        "index_k_format": "fp8_scaled",
        "index_k_layout": "packed",
    }
    try:
        for signature in signatures:
            select_kernel(
                "attention",
                "dsa_decode_topk",
                signature,
                traits=traits,
                solution="cute_dsl",
            )
    except NoKernelFoundError:
        return None
    max_requests = int(global_server_args_dict["max_num_seqs"] or 0)
    if max_requests > dsa_cute_dsl.FUSED_DECODE_TOPK_MAX_REQUESTS:
        return None
    return "cute_dsl"


def _decode_front_kernel(
    indexer: KimiK31Indexer, q_lora_rank: int, kv_lora_rank: int, q_b_rows: int
) -> Callable | None:
    """The fused decode front (``dsa_decode_front``) serving this layer, or None.

    One launch replaces a decode round's q_a / kv_a RMSNorm, its stacked
    [q_b; wq_b] projection and its index-K write with query quantization. It
    hands the fused decode top-k E4M3 queries, so a host selects it only with
    ``index_k_write_quantizes_query``.
    """
    signature = format_signature(
        q_a=dense_tensor_format(torch.bfloat16),
        weight=dense_tensor_format(torch.bfloat16),
    )
    traits = {
        "q_lora_rank": q_lora_rank,
        "kv_lora_rank": kv_lora_rank,
        "q_b_rows": q_b_rows,
        "index_heads": indexer.index_n_heads,
        "head_dim": indexer.index_head_dim,
        "page_size": 64,
        "index_k_format": "fp8_scaled",
        "index_k_layout": "packed",
    }
    try:
        return select_kernel(
            "attention",
            "dsa_decode_front",
            signature,
            traits=traits,
            solution="cute_dsl",
        )
    except NoKernelFoundError:
        return None


class KimiK31SparseMLAAttention(GlmDsaSparseAttentionMixin, KimiLinearMLAAttention):
    """K3's gated NoPE MLA attending its own indexer's top-k (DSA).

    The a-projection, output gate and ``o_proj`` are K3's. The q latent is
    normalized unfused (``fused_qk_layernorm`` then ``q_b_proj``) because the
    indexer reads the normalized latent, which K3's fused query projection
    does not expose. The 64-wide unrotated slice stays in the latent row
    (576 values) and in the absorbed query, and the DSA cores take its dot
    product like any other key channel.

    A decode round of the K3 decode chain runs its norms, the stacked
    [q_b; wq_b] projection and the index-K write with query quantization in
    one launch where the platform registers ``dsa_decode_front``
    (``_takes_decode_front``).
    """

    def __init__(
        self,
        config: KimiLinearConfig,
        mapping: Mapping,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        max_position_embeddings: int,
        quant_config: QuantizationConfig | None,
        layer_id: int,
        prefix: str,
        reduce_attn_results: bool,
        alt_stream: torch.cuda.Stream | None,
        *,
        qkv_parallel: DenseLayerMapping | None,
        output_parallel: DenseLayerMapping | None,
    ) -> None:
        if qkv_parallel is not None or output_parallel is not None:
            raise NotImplementedError(
                "Kimi-K3.1 DSA layers run attention TP only; unset "
                "TOKENSPEED_KIMI_K3_QKV_PROJ_TP_SIZE / TOKENSPEED_KIMI_K3_O_PROJ_TP_SIZE"
            )
        if not config.mla_use_output_gate:
            raise ValueError("Kimi-K3.1 DSA layers require the gated MLA layout")
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
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            layer_id=layer_id,
            prefix=prefix,
            reduce_attn_results=reduce_attn_results,
            alt_stream=alt_stream,
            qkv_parallel=qkv_parallel,
            output_parallel=output_parallel,
        )
        self.index_topk = config.index_topk
        # Every K3.1 DSA layer computes its own top-k (index_topk_freq=1).
        self.is_nextn = False
        self.skip_indexer_topk = False
        # The replicated indexer lets ranks split prefill top-k rows and all-gather.
        self.prefill_topk_tp_split = True
        # K-major w_kc/w_vc take the grouped GEMM unless a bitwise plan is pinned.
        self.absorbed_weights_k_major = self.absorption_kernel_plan is None
        self.indexer = KimiK31Indexer(config, prefix=add_prefix("indexer", prefix))
        # FP8 index queries where the platform registers fp8_scaled scorers.
        self.index_query_fp8 = _scorers_take_fp8_index_queries(self.indexer)
        # The fused decode top-k takes BF16 or unit-scale E4M3 index queries.
        self.decode_topk_solution = (
            None if self.index_query_fp8 else _fused_decode_topk_solution(self.indexer)
        )
        # A decode round quantizes its index query in the index-K write's launch.
        self.index_k_write_quantizes_query = self.decode_topk_solution is not None
        # A decode chain round's norms, q_b / wq_b projection and index-K write in one launch.
        self.decode_front = (
            _decode_front_kernel(
                self.indexer,
                self.q_lora_rank,
                self.kv_lora_rank,
                self.q_b_proj.weight.shape[0],
            )
            if self.index_k_write_quantizes_query
            else None
        )
        self._decode_topk_indices_buffer: torch.Tensor | None = None
        self._decode_topk_lens_buffer: torch.Tensor | None = None
        self._retired_decode_workspaces: list[torch.Tensor] = []
        # Decode rounds select top-k on the alternate stream while the query absorbs.
        self.decode_topk_fork = StreamFork(alt_stream)
        # The attention group's K3 Iris state, once communication is prepared.
        self.topk_iris_state: K3IrisState | None = None
        # [q_b_proj; indexer wq_b] stacked after load: one GEMM over q_norm.
        self.q_b_wq_b_weight: torch.Tensor | None = None
        # [q_a | kv_a | gate | wk | weights_proj], once the decode chain packs it.
        self.decode_projection: torch.Tensor | None = None

    def _gather_prefill_topk_rows(
        self, local: torch.Tensor, local_lens: torch.Tensor, gather_dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather the int16 selection through Iris where the K3 state takes it."""
        state = self.topk_iris_state
        if state is not None and gather_dtype == torch.int16:
            gathered = state.gather_prefill_topk(local, local_lens)
            if gathered is not None:
                return gathered
        return super()._gather_prefill_topk_rows(local, local_lens, gather_dtype)

    def precomputable_qkv_a_weight(self) -> torch.Tensor | None:
        """The BF16 q_a/kv_a rows a caller may project ahead of ``forward``.

        Returns None unless ``forward(precomputed_qkv_a=...)`` would otherwise
        take the split TP projection, whose QKV-a rows ignore the gate.
        """
        weight = self.fused_qkv_a_proj_with_mqa.weight
        if (
            self.qkv_parallel is not None
            or weight.dtype != torch.bfloat16
            or self._fused_qkv_a_pad_rows
            or self._qkv_a_width != _PRECOMPUTED_QKV_A_WIDTH
        ):
            return None
        return weight[: self._qkv_a_width]

    def fuse_query_projections(self) -> None:
        """Stack ``q_b_proj`` and the indexer's ``wq_b`` into one ``[q_b rows +
        index_heads * index_head_dim, q_lora_rank]`` weight, as
        ``KimiK31Indexer.fuse_key_weight_projections`` stacks ``wk``, for
        decode rounds to project both from ``q_norm`` in one GEMM. The two
        parameters become views of its rows, so later in-place weight loads keep
        it current, and a repeated call (a weight update's) keeps it. Bitwise
        numerics and a quantized ``q_b_proj`` keep the separate GEMMs."""
        q_b, wq_b = self.q_b_proj.weight, self.indexer.wq_b.weight
        if (
            global_server_args_dict["numerics"] in BITWISE_ENVELOPES
            or q_b.dtype != torch.bfloat16
            or wq_b.dtype != torch.bfloat16
            or self.q_b_wq_b_weight is not None
        ):
            return
        rows = q_b.shape[0]
        fused = torch.cat((q_b.data, wq_b.data))
        q_b.data = fused[:rows]
        wq_b.data = fused[rows:]
        self.q_b_wq_b_weight = fused

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: ForwardContext,
        comm_manager: CommManager,
        block_scale: torch.Tensor | None = None,
        attnres_partial_args: tuple | None = None,
        *,
        projection_out: torch.Tensor | None,
        precomputed_qkv_a: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``precomputed_qkv_a`` holds ``hidden_states @ precomputable_qkv_a_weight().T``."""
        if hidden_states.shape[0] == 0:
            # No rows and no projection collective (attention TP only).
            return hidden_states
        if precomputed_qkv_a is None:
            q_a, latent_cache, gate = self._project_qkv_gate_a(
                hidden_states, ctx, comm_manager, block_scale, attnres_partial_args
            )
        else:
            if block_scale is not None or attnres_partial_args is not None:
                raise ValueError("precomputed QKV-a rows take the plain BF16 path")
            qkv = comm_manager.pre_attn_comm(precomputed_qkv_a, ctx)
            q_a, latent_cache = qkv.split(
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                dim=-1,
            )
            gate = kimi3_mla_gate_projection(
                hidden_states,
                self.fused_qkv_a_proj_with_mqa.weight,
                self._qkv_a_width,
            )
        attn_output, gate = self._attend(
            positions,
            hidden_states,
            q_a,
            latent_cache,
            gate,
            ctx,
            comm_manager,
            index_key_weights=None,
        )
        return self._gate_project_output(
            attn_output, gate, ctx, projection_out=projection_out
        )

    def o_proj_input(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: ForwardContext,
        comm_manager: CommManager,
        *,
        projection,
    ) -> torch.Tensor:
        """This rank's gated o-projection input, as the K3 decode chain takes it.

        It follows ``KimiLinearMLAAttention.o_proj_input``, attending the
        indexer's top-k like ``forward``. The indexer takes its keys and head
        weights from the packed projection instead of projecting them itself,
        and a decode round runs the fused decode front where
        ``_takes_decode_front`` admits it.

        Args:
            positions: The token positions.
            hidden_states: The attention input [T, hidden].
            ctx: The forward context.
            comm_manager: The layer's communication manager.
            projection: Computes the packed [q_a | kv_a | gate | wk |
                weights_proj] projection (``pack_decode_projection``).
        """
        packed = comm_manager.pre_attn_comm(projection(hidden_states), ctx)
        indexer = self.indexer
        q_a, latent_cache, gate, index_key_weights = packed.split(
            [
                self.q_lora_rank,
                self.kv_lora_rank + self.qk_rope_head_dim,
                self._gate_width,
                indexer.index_head_dim + indexer.index_n_heads,
            ],
            dim=-1,
        )
        attn_output, gate = self._attend(
            positions,
            hidden_states,
            q_a,
            latent_cache,
            gate,
            ctx,
            comm_manager,
            index_key_weights=index_key_weights,
        )
        return attn_output if gate is None else sigmoid_mul(attn_output, gate)

    def decode_projection_parts(self) -> tuple[torch.Tensor, ...]:
        """The weights stacked row-wise into the decode chain's input projection.

        In output order: K3's [q_a | kv_a | gate], then the indexer's ``wk`` and
        ``weights_proj``, which read the same attention input.
        """
        indexer = self.indexer
        return (
            self.fused_qkv_a_proj_with_mqa.weight,
            indexer.wk.weight,
            indexer.weights_proj.weight,
        )

    def pack_decode_projection(self) -> torch.Tensor:
        """The decode chain's input projection weight, packed on the first call.

        It stacks ``decode_projection_parts`` into one ``[rows, hidden]`` tensor.
        The parameters (and the indexer's stacked ``[wk; weights_proj]``)
        become views of its rows, so the values, the steady-state memory and
        every other path's GEMM operands stay as loaded, and in-place weight
        loads keep it current. Call it before graph capture and before
        anything records the weights' addresses.
        """
        if self.decode_projection is not None:
            return self.decode_projection
        parts = self.decode_projection_parts()
        if len({part.dtype for part in parts}) != 1:
            raise ValueError(
                "the decode projection packs one dtype, got "
                f"{[part.dtype for part in parts]}"
            )
        packed = torch.cat([part.data for part in parts])
        offset = 0
        for part in parts:
            rows = part.shape[0]
            part.data = packed[offset : offset + rows]
            offset += rows
        indexer = self.indexer
        if indexer.wk_weights_proj_weight is not None:
            indexer.wk_weights_proj_weight = packed[parts[0].shape[0] :]
        self.decode_projection = packed
        return packed

    def decode_weights(self) -> tuple[torch.Tensor, ...]:
        """The weights one decode step reads, in reading order.

        The packed [q_a | kv_a | gate] projection, q_b with the index query
        projection, the index key and head-weight projections (inside the first
        once the decode chain has packed them), the absorbed key and value
        projections (once derived) and the o-projection.
        """
        indexer = self.indexer
        queries = (self.q_b_wq_b_weight,)
        if self.q_b_wq_b_weight is None:
            queries = (self.q_b_proj.weight, indexer.wq_b.weight)
        keys = (indexer.wk_weights_proj_weight,)
        if indexer.wk_weights_proj_weight is None:
            keys = (indexer.wk.weight, indexer.weights_proj.weight)
        absorbed = tuple(w for w in (self.w_kc, self.w_vc) if w is not None)
        projection = self.fused_qkv_a_proj_with_mqa.weight
        if self.decode_projection is not None:
            projection, keys = self.decode_projection, ()
        return (projection, *queries, *keys, *absorbed, self.o_proj.weight)

    def _attend(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        q_a: torch.Tensor,
        latent_cache: torch.Tensor,
        gate: torch.Tensor | None,
        ctx: ForwardContext,
        comm_manager: CommManager,
        *,
        index_key_weights: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Sparse attention from the raw a-projection rows.

        ``index_key_weights`` holds the indexer's keys and head weights the
        caller projected with the a-projection, or None for the indexer to
        project them.

        Returns:
            The attention output and the output gate still to apply (None
            when the value projection applied it).
        """
        kv_a = latent_cache[..., : self.kv_lora_rank]
        # Decode gates in the value projection; an FP8 o_proj gates in its input quant.
        fuse_value_gate = (
            gate is not None
            and ctx.num_extends == 0
            and self.absorbed_weights_k_major
            and not self.o_proj.supports_fp8_group128_input
        )
        if self._takes_decode_front(ctx, q_a, kv_a, index_key_weights):
            # The decode front normalizes, projects and writes the index keys in the break.
            attn_output = self._sparse_attention(
                positions,
                hidden_states,
                q_a,
                latent_cache,
                ctx,
                comm_manager,
                gate if fuse_value_gate else None,
                None,
                None,
                index_key_weights,
                normalized=False,
            )
            return attn_output, None if fuse_value_gate else gate
        # Normalizes the latent in place and the q latent into q_norm.
        q_norm = torch.empty_like(q_a)
        self.fused_qk_layernorm(input_q_a=q_a, input_kv_a=kv_a, output_q_a=q_norm)
        projected_q = index_query = None
        # Decode rounds only: a prefill projects wq_b for its rank's top-k share of rows.
        if self.q_b_wq_b_weight is not None and ctx.num_extends == 0:
            # Token-shaped, so it may run before the eager break.
            projected_q, index_query = mm(q_norm, self.q_b_wq_b_weight).split(
                (self.q_b_proj.weight.shape[0], self.indexer.wq_b.weight.shape[0]),
                dim=-1,
            )
            index_query = index_query.view(
                -1, self.indexer.index_n_heads, self.indexer.index_head_dim
            )
        attn_output = self._sparse_attention(
            positions,
            hidden_states,
            q_norm,
            latent_cache,
            ctx,
            comm_manager,
            gate if fuse_value_gate else None,
            projected_q,
            index_query,
            index_key_weights,
            normalized=True,
        )
        return attn_output, None if fuse_value_gate else gate

    def _takes_decode_front(
        self,
        ctx: ForwardContext,
        q_a: torch.Tensor,
        kv_a: torch.Tensor,
        index_key_weights: torch.Tensor | None,
    ) -> bool:
        """Whether this round runs the fused decode front (``_decode_front``).

        A decode round of the decode chain (keys and head weights projected
        with the a-projection) qualifies where the front was selected, the
        layer computes its own top-k, the stacked [q_b; wq_b] weight exists,
        the round's index-K write quantizes its query, its KV is not placed
        across ranks, the front addresses the rows, there are at most
        ``_DECODE_FRONT_MAX_ROWS`` of them, and no breakable capture records
        the round (its eager break would replay the capture-time choice over
        padded rows).
        """
        if (
            ctx.num_extends != 0
            or q_a.shape[0] > _DECODE_FRONT_MAX_ROWS
            or self.skip_indexer_topk
            or self.decode_front is None
            or self.q_b_wq_b_weight is None
            or not self.index_k_write_quantizes_query
            or index_key_weights is None
            or is_breakable_capture_active()
        ):
            return False
        backend = ctx.attn_backend
        if isinstance(backend, HybridLinearAttnBackend):
            backend = backend.full_attn_backend
        if backend.cache_placement(self.attn_mqa) is not None:
            return False
        key, weights = index_key_weights.split(
            (self.indexer.index_head_dim, self.indexer.index_n_heads), dim=-1
        )
        return dsa_cute_dsl.decode_front_rows_supported(q_a, kv_a, key, weights)

    def _decode_front(
        self,
        q_a: torch.Tensor,
        latent_cache: torch.Tensor,
        index_key_weights: torch.Tensor,
        ctx: ForwardContext,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]
    ]:
        """A decode round's front in one launch, inside the eager break.

        It normalizes ``q_a`` into ``q_norm`` and the latent in place exactly as
        ``fused_qk_layernorm`` does, projects ``q_norm`` through the stacked
        [q_b; wq_b] weight (FP32 accumulation in its own K order), writes the
        round's index keys exactly as ``set_index_k_buffer_and_quantize_query``
        does and quantizes the index query as it does.

        Returns:
            ``(q_norm, projected_q, index_query, written_index_query)``: the
            BF16 ``q_norm``, ``q_b_proj`` rows and ``[T, index_heads,
            index_head_dim]`` index query (views of one projection, as the
            unfused GEMM's split gives them), and the E4M3 query with its FP32
            head weights for ``forward_dsa_core``.
        """
        indexer = self.indexer
        key, head_weights = index_key_weights.split(
            (indexer.index_head_dim, indexer.index_n_heads), dim=-1
        )
        pool = ctx.token_to_kv_pool
        norms = self.fused_qk_layernorm
        q_b_rows = self.q_b_proj.weight.shape[0]
        q_norm, projected, q_fp8, weights = self.decode_front(
            q_a,
            latent_cache[..., : self.kv_lora_rank],
            key,
            head_weights,
            q_norm_weight=norms.weight_q_a,
            kv_norm_weight=norms.weight_kv_a,
            norm_eps=norms.q_a_norm.variance_epsilon,
            query_weight=self.q_b_wq_b_weight,
            q_b_rows=q_b_rows,
            key_norm_weight=indexer.k_norm.weight,
            key_norm_bias=indexer.k_norm.bias,
            key_norm_eps=indexer.k_norm.variance_epsilon,
            index_k_cache=pool.get_index_k_buffer(self.attn_mqa.layer_id),
            slots=ctx.attn_backend.write_locations(self.attn_mqa, ForwardMode.DECODE),
            page_size=pool.index_k_page_size,
            softmax_scale=indexer.weights_softmax_scale,
        )
        projected_q, index_query = projected.split(
            (q_b_rows, self.indexer.wq_b.weight.shape[0]), dim=-1
        )
        index_query = index_query.view(
            -1, indexer.index_n_heads, indexer.index_head_dim
        )
        return q_norm, projected_q, index_query, (q_fp8, weights)

    def project_latent_values(
        self,
        attn_output: torch.Tensor,
        output: torch.Tensor,
        gate: torch.Tensor | None,
    ) -> None:
        if not self.absorbed_weights_k_major:
            return super().project_latent_values(attn_output, output, gate)
        mla_project_value(attn_output, self.w_vc, gate=gate, out=output)

    @break_point
    def _sparse_attention(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        q_latent: torch.Tensor,
        latent_cache: torch.Tensor,
        ctx: ForwardContext,
        comm_manager: CommManager,
        output_gate: torch.Tensor | None,
        projected_q: torch.Tensor | None,
        index_query: torch.Tensor | None,
        index_key_weights: torch.Tensor | None,
        *,
        normalized: bool,
    ) -> torch.Tensor:
        """The eager break: index-K write, top-k, and the sparse cores.

        The cache writes and the data-dependent top-k read the live ``ctx``,
        so every DSA operation stays here, and ``forward_dsa_core`` narrows a
        padded prefill-graph replay to its live rows: the contract
        ``ModelConfig.dsa_padded_prefill_replay`` declares. Anything moved out
        of the break must be token-shaped and read no backend metadata. The
        DSA leaf is the hybrid wrapper's full-attention child.

        ``q_latent`` is ``q_norm`` when ``normalized``; otherwise it is the raw
        q_a rows of a round ``_takes_decode_front`` admitted, the latent is
        still unnormalized, and ``_decode_front`` produces the normalized rows,
        the projections and the index-K write here.
        """
        backend = ctx.attn_backend
        if isinstance(backend, HybridLinearAttnBackend):
            backend = backend.full_attn_backend
        ctx = replace(ctx, attn_backend=backend)
        written_index_query = None
        if not normalized:
            q_latent, projected_q, index_query, written_index_query = (
                self._decode_front(q_latent, latent_cache, index_key_weights, ctx)
            )
        with self.decode_topk_fork.scope(enable=ctx.num_extends == 0):
            return self.forward_dsa_core(
                positions,
                hidden_states,
                q_latent,
                latent_cache,
                ctx,
                comm_manager,
                output_gate=output_gate,
                projected_q=projected_q,
                index_query=index_query,
                index_key_weights=index_key_weights,
                written_index_query=written_index_query,
            )


class KimiK31ForConditionalGeneration(KimiK3ForConditionalGeneration):
    """Kimi-K3.1 top-level model: the K3 model over a DSA text config."""

    def __init__(self, config, mapping: Mapping, **kwargs) -> None:
        if not config.text_config.dsa_layer_ids:
            raise ValueError(
                "KimiK31ForConditionalGeneration requires text_config.dsa_config"
            )
        super().__init__(config, mapping, **kwargs)

    def load_weights(self, weights) -> None:
        super().load_weights(weights)
        self._fuse_indexer_projections()

    def post_load_weights(self) -> None:
        super().post_load_weights()
        self._fuse_indexer_projections()

    def prepare_communication_runtime(self, max_num_tokens: int) -> bool:
        prepared = super().prepare_communication_runtime(max_num_tokens)
        if self.language_model is not None:
            for layer in self.language_model.model.layers:
                if isinstance(layer.self_attn, KimiK31SparseMLAAttention):
                    layer.self_attn.topk_iris_state = layer.k3_comm.iris_state
        return prepared

    def _fuse_indexer_projections(self) -> None:
        if self.language_model is None:
            return
        for layer in self.language_model.model.layers:
            if isinstance(layer.self_attn, KimiK31SparseMLAAttention):
                layer.self_attn.indexer.fuse_key_weight_projections()
                layer.self_attn.fuse_query_projections()


EntryClass = [KimiK31ForConditionalGeneration]

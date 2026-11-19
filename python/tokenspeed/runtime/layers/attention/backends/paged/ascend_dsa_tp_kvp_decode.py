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

"""Projection fusion and resources for native TP8/KVP8 DSA Decode."""

import math
import weakref
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F
from tokenspeed_kernel.ops.attention.dsa import prepare_dsa_resources
from tokenspeed_kernel.ops.attention.dsa.ascend import (
    validate_dsa_context_parallel,
    validate_dsa_prolog,
)
from tokenspeed_kernel.ops.attention.dsa.paged import (
    select_dsa_projection_kernels,
    select_paged_dsa_attention,
    select_paged_dsa_indexer,
)

from tokenspeed.runtime.layers.attention.dcp.ascend import (
    AscendDCPContext,
    AscendDCPMetadata,
)
from tokenspeed.runtime.layers.attention.dcp.placement import CachePlacement


@dataclass(frozen=True)
class _AscendCachePlacement(CachePlacement):
    # The placement owner prepares native DSA inputs before the first Prolog.
    prepare_decode_metadata: Callable | None = None


def prepare_ascend_dsa_cp_metadata(
    locations, placement, *, initial_tokens, window_tokens
):
    """Prepare the leaf's CP metadata before Prolog through its placement owner."""
    if (
        not isinstance(placement, _AscendCachePlacement)
        or placement.prepare_decode_metadata is None
    ):
        raise ValueError("Native DSA requires Ascend CP cache placement")
    slots = placement.prepare_decode_metadata(
        locations,
        init_tokens=initial_tokens,
        local_tokens=window_tokens,
    )
    return slots


def _nk(weight, widths, kl1=512):
    # Bit-preserving private NK panel stream; no dtype conversion.
    if weight.ndim != 2 or sum(widths) != weight.shape[0] or weight.shape[1] % kl1:
        raise ValueError(
            "DSA NK panels must cover the weight rows and tile its input width"
        )
    panels = []
    col = 0
    for width in widths:
        for start in range(0, weight.shape[1], kl1):
            tile = weight[col : col + width, start : start + kl1]
            panels.append(
                tile.reshape(width // 16, 16, kl1 // 16, 16)
                .permute(2, 0, 1, 3)
                .contiguous()
                .view(-1)
            )
        col += width
    return torch.cat(panels)


def pack_qb_panels(qb):
    """Encode the native N128 tiles without changing coefficient bits."""
    if qb.device.type != "cpu" or tuple(qb.shape) not in ((20480, 1536), (8192, 768)):
        raise ValueError("Expected CPU QB [20480,1536] or [8192,768]")
    packed = _nk(
        qb, [128] * (qb.shape[0] // 128), kl1=256 if qb.shape[1] == 768 else 512
    )
    return packed.reshape(-1, 16, 16).permute(1, 0, 2).contiguous().reshape(16, -1)


def pack_down_qa_panels(down):
    """Encode N96 QA tiles followed by the genuine-NZ KV/Indexer suffix."""
    if down.device.type != "cpu" or tuple(down.shape) not in (
        (2240, 4096),
        (1472, 2048),
    ):
        raise ValueError("Expected CPU down [2240,4096] or [1472,2048]")
    columns, hidden = down.shape
    query_rank = columns - 704
    qa = _nk(down[:query_rank], [96] * (query_rank // 96))
    aux = (
        down[query_rank:]
        .reshape(44, 16, hidden // 16, 16)
        .permute(0, 2, 3, 1)
        .contiguous()
        .view(-1)
    )
    packed = torch.cat((qa, aux))
    return (
        packed.reshape(columns // 16, hidden // 16, 16, 16)
        .permute(1, 2, 0, 3)
        .contiguous()
        .reshape(hidden, columns)
    )


DSA_DECODE_ROWS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)


def supports_fused_dsa_tp_kvp_decode(layer) -> bool:
    """Check the loaded geometry owned by the fused projection implementation."""
    indexer = layer.indexer
    geometry = (
        layer.hidden_size,
        layer.num_heads,
        layer.num_local_heads,
        layer.q_lora_rank,
    )
    if geometry not in ((4096, 64, 8, 1536), (2048, 32, 4, 768)):
        return False
    if (
        layer.kv_lora_rank,
        layer.qk_nope_head_dim,
        layer.qk_rope_head_dim,
        layer.v_head_dim,
        indexer.index_head_dim,
    ) != (512, 128, 64, 128, 128):
        return False
    hidden, _, local_heads, query_rank = geometry
    heads = indexer.index_n_heads
    if (
        (hidden == 2048 and heads != 16)
        or (hidden == 4096 and heads != 32)
        or layer.use_output_gate
    ):
        return False
    weights = (
        (layer.q_a_proj.weight, (query_rank, hidden), torch.bfloat16),
        (layer.kv_a_proj_with_mqa.weight, (576, hidden), torch.bfloat16),
        (layer.q_b_proj.weight, (local_heads * 192, query_rank), torch.bfloat16),
        (indexer.wq_b.weight, (heads * 128, query_rank), torch.bfloat16),
        (indexer.wk.weight, (128, hidden), torch.bfloat16),
        (indexer.weights_proj.weight, (heads, hidden), torch.float32),
    )
    return all(
        value.shape == shape and value.dtype == dtype for value, shape, dtype in weights
    ) and all(
        rotary is None or (not rotary.is_neox_style and rotary.rotary_dim == 64)
        for rotary in (layer.rotary_emb, indexer.rotary_emb)
    )


def pack_dsa_qb(mla_qb, index_qb):
    """Pack actual attention and Indexer channels into the native carrier.

    The 32-head layout splits each 128-wide indexer head across two 64-wide
    carrier lanes. The 64-head attention carrier uses 128-wide indexer lanes.
    """
    if tuple(mla_qb.shape) == (32, 192, 768) and tuple(index_qb.shape) == (
        16,
        128,
        768,
    ):
        index_lanes = index_qb.reshape(32, 64, 768)
    elif tuple(mla_qb.shape) == (64, 192, 1536) and tuple(index_qb.shape) in (
        (32, 128, 1536),
        (64, 128, 1536),
    ):
        index_lanes = (
            F.pad(index_qb, (0, 0, 0, 0, 0, 32))
            if index_qb.shape[0] == 32
            else index_qb
        )
    else:
        raise ValueError(
            "Expected 32 attention and 16 indexer heads, or 64 attention and 32 or 64 indexer heads"
        )
    return pack_qb_panels(
        torch.cat((mla_qb, index_lanes), 1).reshape(-1, mla_qb.shape[2])
    )


def pack_score_fp32(weight):
    """Opaque ND buffer containing FP32 NZ_KN coefficients, bit preserving."""
    if (
        tuple(weight.shape) not in ((16, 2048), (32, 4096), (64, 4096))
        or weight.dtype != torch.float32
    ):
        raise ValueError(
            "FP32 score pack requires [16,2048] or [32/64,4096] float32 weights"
        )
    return (
        weight.reshape(-1, 8, weight.shape[1] // 16, 16)
        .permute(0, 2, 3, 1)
        .contiguous()
        .reshape(weight.shape)
    )


def allocate_dsa_projection_outputs(*, local_heads, index_heads, device):
    """One model-owned set of PyTorch producer buffers for all sequential layers.

    Each bucket is contiguous and keeps its address through graph capture/replay.
    Consumers must finish on the model stream before the next layer overwrites
    these buffers. Native HCCL registration is owned separately by DSAResources.
    """
    # Supported Indexer16/32 producers preserve FP32 score weights.
    return {
        rows: (
            torch.empty(
                (8, rows, local_heads, 576), dtype=torch.bfloat16, device=device
            ),
            torch.empty((rows, index_heads, 128), dtype=torch.bfloat16, device=device),
            torch.empty((rows, index_heads), dtype=torch.float32, device=device),
        )
        for rows in DSA_DECODE_ROWS
    }


class AscendDSATPKVPProjection:
    def __init__(self, layer, *, rank, group, eps_q, eps_kv, outputs=None):
        if not supports_fused_dsa_tp_kvp_decode(layer):
            raise ValueError("Unsupported fused DSA projection geometry or weights")
        self.rank = rank
        self.hidden_size = layer.hidden_size
        self.index_heads = layer.indexer.index_n_heads
        self.layer_id = layer.layer_id
        self.attention_layer = layer.attn_mqa
        self.initial_tokens = layer.index_init_tokens
        self.window_tokens = layer.index_local_tokens
        validate_dsa_prolog(index_heads=self.index_heads)
        self._projection = select_dsa_projection_kernels()
        self.eps_q, self.eps_kv = eps_q, eps_kv
        self.eps_index = layer.indexer.k_norm.variance_epsilon

        def gathered(value):
            value = value.detach().contiguous()
            outputs = [torch.empty_like(value) for _ in range(8)]
            dist.all_gather(outputs, value, group=group)
            return torch.cat(outputs, 0)

        qb = (
            gathered(layer.q_b_proj.weight)
            .cpu()
            .view(layer.num_heads, 192, layer.q_lora_rank)
        )
        self.query_scale = 1.0
        if layer.scale_q_lora:
            self.query_scale = math.sqrt(layer.hidden_size / layer.q_lora_rank)
        iqb = (
            layer.indexer.wq_b.weight.detach()
            .cpu()
            .view(self.index_heads, 128, layer.q_lora_rank)
        )
        self.qb = self._projection.prepare_weight(pack_dsa_qb(qb, iqb).npu())
        down = torch.cat(
            (
                layer.q_a_proj.weight.detach(),
                layer.kv_a_proj_with_mqa.weight.detach(),
                layer.indexer.wk.weight.detach(),
            ),
            0,
        )
        self.down = self._projection.prepare_weight(
            pack_down_qa_panels(down.detach().cpu()).npu()
        )
        self.uk = gathered(layer.w_kc)
        self.gamma_q = layer.q_a_layernorm.weight.detach().contiguous()
        self.gamma_kv = layer.kv_a_layernorm.weight.detach().contiguous()
        self.gamma_index = layer.indexer.k_norm.weight.detach().contiguous()
        self.score_scale = (
            layer.indexer.weights_softmax_scale if self.index_heads == 16 else 1.0
        )
        score_weight = layer.indexer.weights_proj.weight.detach().float()
        if self.index_heads != 16:
            score_weight = score_weight * layer.indexer.weights_softmax_scale
        self.score_weight = pack_score_fp32(score_weight.contiguous())
        self.w_vc = layer.w_vc.detach().contiguous()
        self.w_o = layer.o_proj.weight.detach().contiguous()
        self.rotary_mode = int(layer.rotary_emb is not None) | (
            int(layer.indexer.rotary_emb is not None) << 1
        )
        self.rotary_emb = (
            layer.rotary_emb
            if layer.rotary_emb is not None
            else layer.indexer.rotary_emb
        )
        if self.rotary_emb is not None:
            if self.rotary_emb.is_neox_style or self.rotary_emb.rotary_dim != 64:
                raise ValueError("Fused DSA RoPE requires 64 adjacent-pair channels")
            self.cos_sin = self.rotary_emb.cos_sin_cache.float().contiguous().npu()
        else:
            self.cos_sin = None
        self.window = None
        # Standalone operator callers may own one set; the model runtime always
        # supplies its shared buffers so memory does not grow with layer count.
        self.outputs = (
            allocate_dsa_projection_outputs(
                local_heads=layer.num_local_heads,
                index_heads=self.index_heads,
                device=layer.q_a_proj.weight.device,
            )
            if outputs is None
            else outputs
        )

    def prolog(self, hidden, positions, ctx):
        if self.window is None:
            raise RuntimeError("Initialize DSA transport before forward/capture")
        query, index_query, score = self.outputs[hidden.shape[0]]
        pool = ctx.token_to_kv_pool
        placement = ctx.attn_backend.cache_placement(self.attention_layer)
        locations = ctx.attn_backend.write_locations(
            self.attention_layer, ctx.forward_mode
        )
        slots = prepare_ascend_dsa_cp_metadata(
            locations,
            placement,
            initial_tokens=self.initial_tokens,
            window_tokens=self.window_tokens,
        )
        kv_buffer = pool.get_key_buffer(self.layer_id)
        if kv_buffer.dtype != torch.bfloat16:
            raise ValueError("Native DSA requires BF16 KV cache")
        kv = kv_buffer.view(-1, 576)
        index_k = pool.get_component(self.layer_id, "dsa_index_k").view(-1, 128)
        if self.rotary_mode:
            if (
                positions is None
                or positions.ndim != 1
                or positions.numel() != hidden.shape[0]
                or positions.dtype not in (torch.int32, torch.int64)
            ):
                raise ValueError(
                    "RoPE positions must be integer indices for all gathered DSA rows"
                )
            positions = positions.to(dtype=torch.int64).contiguous()
        else:
            positions = None
        self._projection.prolog(
            hidden.contiguous(),
            self.down,
            self.gamma_q,
            self.gamma_kv,
            self.gamma_index,
            self.score_weight,
            self.qb,
            self.uk,
            self.window,
            kv,
            index_k,
            slots,
            eps_q=self.eps_q,
            eps_kv=self.eps_kv,
            eps_index=self.eps_index,
            query_scale=self.query_scale,
            score_scale=self.score_scale,
            positions=positions,
            cos_sin=self.cos_sin,
            rotary_mode=self.rotary_mode,
            out=(query, index_query, score),
        )
        return query, index_query, score

    def output_projection(self, output):
        return self._projection.output_projection(output, self.w_vc, self.w_o)


class _LayerDSATPKVPDecode(AscendDSATPKVPProjection):
    """Bind shared HCCL views around the unchanged host DSA computation."""

    def __init__(self, layer, *, rank, group, eps_q, eps_kv, owner, outputs):
        super().__init__(
            layer, rank=rank, group=group, eps_q=eps_q, eps_kv=eps_kv, outputs=outputs
        )
        self.owner = owner

    def prolog(self, hidden, positions, ctx):
        self.owner.bind_backend(ctx.attn_backend)
        stream = torch.npu.current_stream()
        hidden.record_stream(stream)
        if positions is not None and positions.dtype in (torch.int32, torch.int64):
            positions = positions.to(dtype=torch.int64).contiguous()
            positions.record_stream(stream)
        return super().prolog(hidden, positions, ctx)


class AscendDSATPKVPRuntime:
    """Prepare native DSA once before profiling; retain it across cache rebinds."""

    def __init__(self, layers, mapping):
        from tokenspeed.runtime.distributed.process_group_manager import (
            process_group_manager as pg,
        )

        self.layers = tuple(layers)
        self.ranks = tuple(mapping.attn.tp_group)
        if len(self.ranks) != 8 or not self.layers:
            raise ValueError(
                "Native DSA requires a complete eight-rank attention group"
            )
        rank = self.ranks.index(dist.get_rank())
        group = pg.get_dedicated_device_group(self.ranks, "ascend_dsa")
        pg.init_process_group(self.ranks, backend="gloo")
        cpu_group = pg.get_process_group("gloo", self.ranks)
        probe = torch.ones(1, device=self.layers[0].q_a_proj.weight.device)
        dist.all_reduce(probe, group=group)
        name = group._get_backend(probe.device).get_hccl_comm_name(dist.get_rank())
        self.group = name
        self.resources = prepare_dsa_resources(
            group=name, rank=rank, heads=self.layers[0].num_heads
        )
        self.window = self.resources.window
        self.outputs = self.resources.outputs
        self.backends = weakref.WeakSet()
        self._bound_leaves = weakref.WeakSet()
        self.runtimes = []
        self.projection_outputs = {}
        self.closed = False
        try:
            torch.npu.synchronize()
            dist.barrier(group=cpu_group)
            first = self.layers[0]
            self.projection_outputs = allocate_dsa_projection_outputs(
                local_heads=first.num_local_heads,
                index_heads=first.indexer.index_n_heads,
                device=first.q_a_proj.weight.device,
            )
            for layer in self.layers:
                if (
                    layer.num_heads,
                    layer.num_local_heads,
                    layer.indexer.index_n_heads,
                ) != (
                    first.num_heads,
                    first.num_local_heads,
                    first.indexer.index_n_heads,
                ):
                    raise ValueError(
                        "Native DSA shared outputs require uniform attention and Indexer heads"
                    )
                runtime = _LayerDSATPKVPDecode(
                    layer,
                    rank=rank,
                    group=pg.get_device_process_group(self.ranks),
                    eps_q=layer.q_a_layernorm.variance_epsilon,
                    eps_kv=layer.kv_a_layernorm.variance_epsilon,
                    owner=self,
                    outputs=self.projection_outputs,
                )
                runtime.window = self.window
                layer.configure_fused_tp_kvp_decode(runtime)
                self.runtimes.append(runtime)
        except BaseException:
            self.close()
            raise

    def bind_backend(self, backend):
        from tokenspeed.runtime.layers.attention.backends.paged.ascend_dsa import (
            AscendDSABackend,
        )

        if self.closed:
            raise RuntimeError("DSA runtime is closed")
        if backend in self.backends:
            return
        pending = [backend]
        found = False
        while pending:
            node = pending.pop()
            if isinstance(node, AscendDSABackend):
                if tuple(node._kvp_ranks) != self.ranks:
                    raise ValueError("DSA model and cache backend groups differ")
                node._decode.configure_fused_decode(
                    group=self.group,
                    ranks=self.ranks,
                    attention_outputs=self.outputs,
                )
                self._bound_leaves.add(node)
                found = True
            pending.extend(node.child_backends())
        if not found:
            raise ValueError("Native DSA requires an Ascend DSA cache backend")
        self.backends.add(backend)

    def close(self):
        """The caller has drained forwards and destroyed every referencing graph."""
        if self.closed:
            return
        for layer in self.layers:
            layer.configure_fused_tp_kvp_decode(None)
        for runtime in self.runtimes:
            runtime.window = None
        for backend in self._bound_leaves:
            backend._decode.clear_fused_decode()
        self.resources.release(self.window)
        self.resources = None
        self.window = None
        self.outputs.clear()
        self.projection_outputs.clear()
        self.runtimes.clear()
        self.backends.clear()
        self._bound_leaves.clear()
        self.closed = True


@dataclass(frozen=True)
class _AscendSparseSelection:
    query: torch.Tensor
    indices: torch.Tensor
    valid_chunks: torch.Tensor
    q_ends: torch.Tensor
    kv_lengths: torch.Tensor
    table: torch.Tensor
    sparse_mode: int
    fused: bool


@dataclass(frozen=True)
class _FusedDecodeBinding:
    """Borrowed producer views; the model runtime owns their native registration."""

    group: str
    attention_outputs: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]


class AscendDSATPKVPDecode:
    def __init__(self, config, spec, *, page_size):
        self._indexer_spec = spec
        self.kernel_page_size = page_size
        self._dcp = AscendDCPMetadata(
            degree=config.kvp_size,
            rank=config.kvp_rank,
            ranks=tuple(config.kvp_group),
            virtual_block_count=None,
        )
        self._decode_indexer = select_paged_dsa_indexer()
        self._decode_attention = select_paged_dsa_attention()
        self._collective_context = None
        self._fused_decode = None
        self.placement = None

    def bind_cache(self, placement):
        self.placement = placement
        self._dcp.bind_virtual_block_count(placement.virtual_block_count)

    def refresh(self, metadata):
        self._dcp.clear_forward_metadata()

    def allocate_decode_buffers(self, rows, columns, device):
        self._dcp.allocate_decode_buffers(rows, columns, device)

    def prepare_prolog_metadata(
        self, locations, metadata, *, init_tokens, local_tokens
    ):
        start = metadata.num_extends
        width = locations.numel() // (metadata.seq_lens.numel() - start)
        self._prepare_decode_metadata(
            locations,
            metadata.seq_lens[start:],
            metadata.page_table[start:],
            width=width,
            initial=init_tokens,
            window=local_tokens,
        )
        return self._dcp.write_slots

    def configure_fused_decode(
        self,
        *,
        group: str,
        ranks: tuple[int, ...],
        attention_outputs: dict[int, tuple],
    ) -> None:
        """Bind CP resources and stable producer outputs outside capture."""
        if self._dcp.degree <= 1 or not group:
            raise ValueError("DSA fusion requires a context-parallel communicator")
        if tuple(ranks) != tuple(self._dcp.ranks):
            raise ValueError("Fused DSA resources must cover the CP group ranks")
        if not attention_outputs:
            raise ValueError("Fused DSA requires prepared SFA producer outputs")
        spec = self._indexer_spec
        for rows, outputs in attention_outputs.items():
            expected = (
                (spec.num_attention_heads, rows, spec.kv_lora_rank),
                (1, rows, spec.num_attention_heads),
                (1, rows, spec.num_attention_heads),
            )
            if len(outputs) != 3 or any(
                value.shape != shape or value.dtype != torch.float32
                for value, shape in zip(outputs, expected)
            ):
                raise ValueError(
                    "Fused DSA requires prepared FP32 SFA producer outputs"
                )
        validate_dsa_context_parallel(
            rank_size=self._dcp.degree, num_heads=spec.num_attention_heads
        )
        self._fused_decode = _FusedDecodeBinding(group, attention_outputs)

    def clear_fused_decode(self) -> None:
        """Drop borrowed views after the owner has drained/destroyed its graphs."""
        self._fused_decode = None

    def _context_collectives(self) -> AscendDCPContext:
        """Use the existing device group for shapes without native fusion."""
        if self._collective_context is None:
            self._collective_context = AscendDCPContext(
                degree=self._dcp.degree,
                rank=self._dcp.rank,
                ranks=self._dcp.ranks,
                auxiliary_namespace=None,
            )
        if not self._collective_context.ready:
            raise RuntimeError("Ascend DSA CP process group was not configured")
        return self._collective_context

    def _select_collective_topk(self, q, local_indices, local_values):
        collective = self._context_collectives()
        degree = self._dcp.degree
        tokens, _, candidates = local_values.shape
        padded_tokens = (tokens + degree - 1) // degree * degree
        if padded_tokens != tokens:
            local_values = torch.cat(
                (
                    local_values,
                    torch.full(
                        (padded_tokens - tokens, 1, candidates),
                        float("-inf"),
                        dtype=local_values.dtype,
                        device=local_values.device,
                    ),
                )
            )
        rows = padded_tokens // degree
        received = torch.empty(
            degree * rows * candidates,
            dtype=local_values.dtype,
            device=local_values.device,
        )
        collective.all_to_all(
            received, local_values.squeeze(1).contiguous().view(-1), auxiliary=False
        )
        packed_query = collective.all_gather(q.contiguous(), auxiliary=False).view(
            degree, *q.shape
        )
        values = received.view(degree, rows, candidates).transpose(0, 1)
        _, positions = values.reshape(rows, degree * candidates).topk(
            self._indexer_spec.index_topk, dim=1
        )
        positions = collective.all_gather(
            positions.to(torch.int32).contiguous(), auxiliary=False
        )[:tokens]
        indices, valid_chunks = self._decode_indexer.select_local_indices(
            local_indices, positions, self._dcp.rank
        )
        return packed_query, indices, valid_chunks

    def _prepare_decode_metadata(
        self, locations, lengths, table, *, width, initial, window
    ):
        """Expand causal query rows and resolve owner-local Index-K write slots.

        The fused prologue prepares this same metadata before writing both
        caches. Storage-view reuse makes subsequent layers share one result;
        eager and graph replay refresh through the same DCP metadata owner.
        """
        placement = self.placement
        page = self.kernel_page_size
        self._dcp.prepare_metadata(
            seq_lens=lengths,
            page_table=table,
            page_size=page,
            block_granularity=(placement.block_granularity),
            init_tokens=initial,
            local_tokens=window,
            query_width=width,
            query_group_size=width,
            slots=locations,
        )
        return self._dcp.metadata(start=0, end=table.shape[0] * width)

    def _select_decode_indices(
        self,
        query,
        index_query,
        index_cache,
        weights,
        local,
        *,
        initial,
        window,
        query_width,
        packed_query,
        fused: bool,
    ):
        """Select owner-local indices; exchange only queries and TopK candidates."""
        kernels = self._decode_indexer
        q_ends = self._dcp.query_ends
        # Verify candidates share page/K tiles but retain their causal ends.
        index_metadata = self._dcp.indexer_metadata(
            q_ends, local, query_group_size=query_width
        )
        local_indices, local_values = kernels.select_candidates(
            index_query,
            index_cache,
            weights,
            index_metadata.q_ends,
            index_metadata.seq_lens,
            index_metadata.page_table,
            self._indexer_spec.index_topk,
            index_metadata.init_counts,
            index_metadata.local_counts,
            sparse_mode=3,
            visible_lengths=index_metadata.visible_lengths,
        )
        if fused:
            if local_values.shape[1:] != (1, 2048):
                raise ValueError("Fused DSA TopK requires a prepared [T,1,2048] shape")
            indices, valid_chunks = kernels.merge_topk(
                local_indices,
                local_values,
                group=self._fused_decode.group,
                rank_size=self._dcp.degree,
            )
        else:
            packed_query, indices, valid_chunks = self._select_collective_topk(
                query, local_indices, local_values
            )
        return _AscendSparseSelection(
            packed_query,
            indices,
            valid_chunks,
            q_ends,
            local.seq_lens,
            local.page_table,
            3,
            fused,
        )

    def _run_collective_attention(self, selection, kv_cache, layer, kernels):
        output, maximum, total = kernels.attention_partial(
            selection.query,
            kv_cache,
            selection.indices,
            selection.valid_chunks,
            selection.q_ends,
            selection.kv_lengths,
            selection.table,
            layer.scaling,
            sparse_mode=selection.sparse_mode,
        )
        lse = (maximum + total.log()).squeeze(0)
        return self._context_collectives().combine(
            output.transpose(0, 1), lse, merge=kernels.merge_partials
        )

    def _run_fused_attention(self, selection, kv_cache, scale, kernels):
        """Write registered SFA producers and return merged local heads [H,T,D]."""
        binding = self._fused_decode
        if binding is None:
            raise RuntimeError(
                "Initialize fused DSA resources before context-parallel execution"
            )
        rows = selection.query.shape[1]
        if rows not in binding.attention_outputs:
            raise ValueError("Fused DSA LSE requires prepared SFA producer outputs")
        output, maximum, total = kernels.attention_partial(
            selection.query,
            kv_cache,
            selection.indices,
            selection.valid_chunks,
            selection.q_ends,
            selection.kv_lengths,
            selection.table,
            scale,
            sparse_mode=selection.sparse_mode,
            out=binding.attention_outputs[rows],
        )
        return kernels.merge_lse_direct(
            output,
            maximum,
            total,
            group=binding.group,
            rank_size=self._dcp.degree,
        )

    def _run_indexed_sparse_attention(
        self,
        selection,
        layer,
        token_to_kv_pool,
        *,
        head_major_output,
    ):
        """Consume owner-local KV and return this rank's attention-head partition."""
        spec = self._indexer_spec
        kv_cache = token_to_kv_pool.get_key_buffer(layer.layer_id).view(
            -1,
            self.kernel_page_size,
            1,
            spec.kv_cache_dim,
        )
        expected_width = spec.kv_lora_rank + spec.qk_rope_head_dim
        if selection.query.shape[-1] != expected_width:
            raise ValueError(
                f"Ascend DSA query width must be {expected_width}, got "
                f"{selection.query.shape[-1]}"
            )
        kernels = self._decode_attention
        if selection.fused:
            result = self._run_fused_attention(
                selection, kv_cache, layer.scaling, kernels
            )
            return (
                result
                if head_major_output
                else result.transpose(0, 1).contiguous().flatten(1)
            )
        result = self._run_collective_attention(selection, kv_cache, layer, kernels)
        return (
            result.transpose(0, 1).contiguous()
            if head_major_output
            else result.flatten(1)
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool,
        bs: int,
        topk_indices: torch.Tensor | None = None,
        topk_lens: torch.Tensor | None = None,
        *,
        fused: bool,
        **kwargs,
    ) -> torch.Tensor:
        tokens_per_request = q.shape[0] // bs
        initial, window = kwargs["index_init_tokens"], kwargs["index_local_tokens"]
        packed_query = kwargs.get("packed_query")
        if fused:
            if packed_query is None:
                raise ValueError("Fused DSA requires the fused Prolog query")
            if self._fused_decode is None:
                raise RuntimeError(
                    "Initialize fused DSA resources before using packed queries"
                )
            if q.shape[0] not in self._fused_decode.attention_outputs:
                raise ValueError("Fused DSA requires prepared SFA producer outputs")
        elif packed_query is not None:
            raise ValueError("Ordinary DSA does not accept a fused Prolog query")

        metadata = kwargs["metadata"]
        start = metadata.num_extends
        index_cache = token_to_kv_pool.get_component(
            layer.layer_id, "dsa_index_k"
        ).view(-1, self.kernel_page_size, 1, self._indexer_spec.index_head_dim)
        local = self._prepare_decode_metadata(
            out_cache_loc,
            metadata.seq_lens[start : start + bs],
            metadata.page_table[start : start + bs],
            width=tokens_per_request,
            initial=initial,
            window=window,
        )
        # Main KV was written by MLA prologue. Fused prologue also writes Index-K;
        # ordinary projections hand Index-K to this one owner-local store.
        if not kwargs.get("index_cache_written", False):
            self._decode_indexer.store_index_keys(
                kwargs["index_key"].contiguous(), index_cache, self._dcp.write_slots
            )
        selection = self._select_decode_indices(
            q,
            kwargs["index_query"],
            index_cache,
            kwargs["index_weights"],
            local,
            initial=initial,
            window=window,
            query_width=tokens_per_request,
            packed_query=packed_query,
            fused=fused,
        )
        step_counter = kwargs.get("step_counter")
        if step_counter is not None:
            step_counter.record_cache()
        return self._run_indexed_sparse_attention(
            selection,
            layer,
            token_to_kv_pool,
            head_major_output=kwargs.get("head_major_output", False),
        )

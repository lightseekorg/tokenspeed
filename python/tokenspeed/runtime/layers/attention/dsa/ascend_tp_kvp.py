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

import torch
import torch.distributed as dist
import torch.nn.functional as F
from tokenspeed_kernel.ops.attention.dsa import prepare_dsa_resources
from tokenspeed_kernel.ops.attention.dsa.ascend import validate_dsa_prolog
from tokenspeed_kernel.ops.attention.dsa.paged import select_dsa_projection_kernels

from tokenspeed.runtime.layers.attention.backends.paged.ascend_dsa import (
    AscendDSABackend,
    prepare_ascend_dsa_cp_metadata,
)


def _nk(weight, widths, kl1=512):
    # Bit-preserving private NK panel stream; no dtype conversion.
    assert sum(widths) == weight.shape[0] and weight.shape[1] % kl1 == 0
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


def supports_fused_dsa_decode(layer) -> bool:
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

    The32-head attention layout splits each128-wide Indexer head across two64-wide carrier lanes,
    retaining all32 attention heads and16 Indexer heads without extra heads.
    The historical attention64 carrier retains its128-wide Indexer lanes.
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
            "Expected attention32/Indexer16 or attention64/Indexer32,64 QB weights"
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
        if not supports_fused_dsa_decode(layer):
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
            int(layer.indexer.use_rotary) << 1
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
                layer.configure_fused_decode(runtime)
                self.runtimes.append(runtime)
        except BaseException:
            self.close()
            raise

    def bind_backend(self, backend):
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
            layer.configure_fused_decode(None)
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

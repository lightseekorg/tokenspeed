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

"""TP8 verify-8 attention branch 0 and post-attention reduce/residual/RMSNorm.

Each layer uses two persistent-kernel roles: rank 0 indexes and publishes
selection; ranks 1-7 own 9/10-head shards and run QKV/query preparation,
sparse MLA and output projections. TOKENSPEED_LONGCAT_LAYER_ENGINE=1 enables
eligible single-request, eight-row decode with replicated attention input;
other forwards retain standard attention.

The engine omits rank 0's MLA KV and workers' index-K writes. Switching an
existing request to standard attention can therefore expose stale cache
planes on those ranks.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from tokenspeed_kernel.ops.longcat.cute_dsl import (
    HEADS_TOTAL,
    NUM_TOKENS,
    TOPK,
    IndexerWeights,
    create_selection_exchange_state,
    engine_head_range,
    make_indexer_engine_workspace,
    make_o_proj_collective_workspace,
    make_worker_engine_workspace,
    run_indexer_engine,
    run_worker_engine,
)

from tokenspeed.runtime.distributed.process_group_manager import (
    process_group_manager as pg_manager,
)
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.execution.forward_step import get_is_capture_mode
from tokenspeed.runtime.models.longcat.selection import DSADecodeSelection, DSASelection

_QKV_NORM_EPS = 1e-5  # the engine's fused q/kv RMSNorm epsilon
_INITIAL_TOKENS = 16
_PAGE_SIZE = 64  # both kernels hard-code 64-row pages
_MAX_SLOTS = 1 << 23  # selection words carry 24-bit signed slot ids
_LOCAL_TOKENS = 1024


def _bf16(t: torch.Tensor) -> torch.Tensor:
    """Contiguous bf16 copy/view of a norm weight (the kernels take bf16)."""
    return t.detach().to(torch.bfloat16).contiguous()


def _check_bf16_shape(name, t, shape, engine: str) -> None:
    if t.dtype != torch.bfloat16:
        raise ValueError(f"{engine} engine needs bf16 {name}, got {t.dtype}")
    if tuple(t.shape) != shape:
        raise ValueError(
            f"{engine} engine expects {name} {shape}, got {tuple(t.shape)}"
        )


def _check_attention_layer(attn, engine: str) -> None:
    """Validate the original TP8 model shards, each with eight local heads."""
    if attn.attn_mqa.fp8_kv_scales is not None:
        raise ValueError(f"{engine} engine requires an unscaled KV cache")
    _check_bf16_shape("q_b_proj", attn.q_b_proj.weight, (8 * 192, 1536), engine)
    _check_bf16_shape("o_proj", attn.o_proj.weight, (8192, 8 * 128), engine)
    _check_bf16_shape("w_kc", attn.w_kc, (8, 128, 512), engine)
    _check_bf16_shape("w_vc", attn.w_vc, (8, 512, 128), engine)
    if tuple(attn.rotary_emb.cos_sin_cache.shape[1:]) != (64,):
        raise ValueError(f"{engine} engine expects a [max_pos, 64] rotary table")
    if (
        attn.q_lora_rank != 1536
        or attn.kv_lora_rank != 512
        or attn.qk_rope_head_dim != 64
    ):
        raise ValueError(f"{engine} engine expects LongCat 2.0 MLA dims")
    if (
        attn.num_heads != HEADS_TOTAL
        or attn.qk_nope_head_dim != 128
        or attn.v_head_dim != 128
    ):
        raise ValueError(f"{engine} engine expects 64 heads of 128/64/128")
    if attn.index_topk != TOPK:
        raise ValueError(f"{engine} engine expects top-k {TOPK}")


def _check_qkv_norms(attn, engine: str) -> None:
    for norm in (attn.q_a_layernorm, attn.kv_a_layernorm):
        if abs(float(norm.variance_epsilon) - _QKV_NORM_EPS) > 1e-12:
            raise ValueError(f"{engine} engine hard-codes q/kv RMSNorm eps 1e-5")


def _selection_exchange(group, device: torch.device, rank: int):
    """Allocate zeroed selection exchange separately for each layer.

    Layers reuse generation tags, so sharing storage could accept another layer's
    selection. Initialize every rank before any peer publication.
    """
    state = create_selection_exchange_state(group, rank=rank, device=device)
    state.symmetric_buffer.zero_()
    torch.cuda.synchronize(device)
    dist.barrier(group=group)
    return state


class LongcatAttention0Engine:
    """Attention-0 engine constructed after each layer's weights load.

    First eligible warm-up prepares weights and TP8 transports before capture.
    forward keeps generations, Lamport tags and receive parity on device.
    """

    _collective = None
    # int32/int64 kernel inputs of the current forward, shared by all layers
    # (identical positions / slots / block table); reset by ``begin_forward``.
    _step_inputs: dict | None = None
    _block_table_width: int | None = None

    def __init__(self, attn, post_attention_layernorm, mapping) -> None:
        self.attn = attn
        self.tp_rank = int(mapping.attn.tp_rank)
        self.device = attn.o_proj.weight.device
        # Torch process group: weight gathers and the symmetric-memory
        # rendezvous of the kernel transports.
        self.group = pg_manager.get_device_process_group(mapping.attn.tp_group)
        self.world_size = int(mapping.attn.tp_size)
        self.post_norm = post_attention_layernorm
        self._check_layer(attn, post_attention_layernorm)
        self.kv_layer_id = int(attn.attn_mqa.layer_id)
        self.selection = DSASelection(owner_layer_id=self.kv_layer_id)
        self.collective = None
        self.exchange = None
        self.indexer_ws = None
        self.worker_ws = None
        self.post_norm_gamma = self.q_gamma = self.kv_gamma = None
        self.post_norm_eps = self.sm_scale = self.k_scale = None
        self.w_qkv_a = self.cos_sin_cache = None
        self.indexer_weights = self.indexer_scale = None
        self.q_b_weight = self.w_kc = self.w_vc_t = self.o_proj_weight = None
        self.local_heads = None
        self._ready = False

    def _prepare(self) -> None:
        """Snapshot/re-shard finalized weights and collectively build TP8 transports.

        Defer until the first eligible forward: loader randomization and quant-method
        post-processing may follow post_load_weights. All ranks must agree on
        eligible before entering the gathers and symmetric-memory rendezvous.
        """
        if get_is_capture_mode():
            raise RuntimeError(
                "attention0 engine initialised under CUDA-graph capture; run an "
                "eager warm-up forward first"
            )
        attn = self.attn
        if attn.attn_mqa.fp8_kv_scales is not None:
            raise RuntimeError(
                "attention0 engine needs an unscaled KV cache, got "
                f"{attn.attn_mqa.fp8_kv_scales}"
            )
        if LongcatAttention0Engine._collective is None:
            LongcatAttention0Engine._collective = make_o_proj_collective_workspace(
                self.group, self.device
            )
        self.collective = LongcatAttention0Engine._collective
        self.exchange = _selection_exchange(self.group, self.device, self.tp_rank)
        self.post_norm_gamma = _bf16(self.post_norm.weight.data)
        self.post_norm_eps = float(self.post_norm.variance_epsilon)
        self.sm_scale = float(attn.scaling)
        self.k_scale = 1.0
        self.q_gamma = _bf16(attn.q_a_layernorm.weight.data)
        self.kv_gamma = _bf16(attn.kv_a_layernorm.weight.data)
        width = attn.q_lora_rank + attn.kv_lora_rank + attn.qk_rope_head_dim
        self.w_qkv_a = attn.fused_qkv_a_proj_with_mqa.weight.data[:width]
        rotary = attn.rotary_emb.cos_sin_cache
        self.cos_sin_cache = rotary if rotary.dtype == torch.float32 else rotary.float()
        # The re-sharding gathers are collective: rank 0 joins them and
        # keeps nothing (it hosts the indexer instead of attention heads).
        self._build_worker_shard(attn)
        if self.tp_rank == 0:
            indexer = attn.indexer
            self.indexer_weights = IndexerWeights(
                w_qkv_a=self.w_qkv_a,
                wq_b=indexer.wq_b.weight.data,
                wk=indexer.wk_weights_proj.weight.data,
                rms_w=_bf16(indexer.k_norm.weight.data),
            )
            self.indexer_scale = float(indexer.weights_softmax_scale)
        self._ready = True

    # ------------------------------------------------------------------
    @staticmethod
    def _check_layer(attn, post_norm) -> None:
        width = attn.q_lora_rank + attn.kv_lora_rank + attn.qk_rope_head_dim
        qkv_a = attn.fused_qkv_a_proj_with_mqa.weight
        if qkv_a.dtype != torch.bfloat16:
            raise ValueError(
                f"attention0 engine needs bf16 fused_qkv_a_proj_with_mqa, got {qkv_a.dtype}"
            )
        if qkv_a.shape[0] < width or qkv_a.shape[1] != 8192:
            raise ValueError(f"attention0 engine expects qkv_a [>= {width}, 8192]")
        _check_attention_layer(attn, "attention0")
        if attn.initial_tokens != _INITIAL_TOKENS or attn.local_tokens != _LOCAL_TOKENS:
            raise ValueError("attention0 engine expects 16 initial + 1024 local tokens")
        _check_qkv_norms(attn, "attention0")
        indexer = attn.indexer
        if indexer is None or type(indexer.k_norm).__name__ != "RMSNorm":
            raise ValueError("attention0 engine needs an RMSNorm index-K norm")
        _check_bf16_shape(
            "indexer.wq_b", indexer.wq_b.weight, (32 * 128, 1536), "attention0"
        )
        _check_bf16_shape(
            "indexer.wk_weights_proj",
            indexer.wk_weights_proj.weight,
            (128 + 32, 8192),
            "attention0",
        )
        if getattr(attn.rotary_emb, "is_neox_style", True):
            raise ValueError("attention0 engine implements interleaved RoPE only")

    def _gather(self, t: torch.Tensor, dim: int) -> torch.Tensor:
        """All-gather a weight shard along ``dim`` in TP rank order (plain NCCL;
        the runtime's comm_ops all_gather is sized for activations)."""
        t = t.contiguous()
        out = torch.empty(
            (self.world_size,) + tuple(t.shape), dtype=t.dtype, device=t.device
        )
        dist.all_gather_into_tensor(out, t, group=self.group)
        return out.movedim(0, dim).reshape(
            t.shape[:dim] + (self.world_size * t.shape[dim],) + t.shape[dim + 1 :]
        )

    def _build_worker_shard(self, attn) -> None:
        """All-gather the TP8 head shards and cut this rank's TP7 shard
        (rank 0 only participates in the gathers)."""
        if self.tp_rank == 0:
            for t in (attn.q_b_proj.weight.data, attn.w_kc, attn.w_vc):
                del_ = self._gather(t, 0)
                del del_
            del_ = self._gather(attn.o_proj.weight.data, 1)
            del del_
            self.local_heads = 0
            return
        offset, local_heads = engine_head_range(self.tp_rank)
        heads = slice(offset, offset + local_heads)
        # q_b_proj: [8*192, 1536] per rank, head-major rows -> [64*192, 1536].
        q_b_all = self._gather(attn.q_b_proj.weight.data, 0)
        self.q_b_weight = q_b_all[
            offset * 192 : (offset + local_heads) * 192
        ].contiguous()
        del q_b_all
        # w_kc: production [8, 128(k), 512(n)] -> engine [H, 512(n), 128(k)].
        w_kc_all = self._gather(attn.w_kc, 0)
        self.w_kc = w_kc_all[heads].transpose(1, 2).contiguous()
        del w_kc_all
        # w_vc: production [8, 512(k), 128(n)] -> engine w_vc_t [H, 128, 512].
        w_vc_all = self._gather(attn.w_vc, 0)
        self.w_vc_t = w_vc_all[heads].transpose(1, 2).contiguous()
        del w_vc_all
        # o_proj: [8192, 8*128] per rank (input sharded) -> [8192, 64*128].
        w_o_all = self._gather(attn.o_proj.weight.data, 1)
        self.o_proj_weight = w_o_all[
            :, offset * 128 : (offset + local_heads) * 128
        ].contiguous()
        del w_o_all
        self.local_heads = local_heads

    # ------------------------------------------------------------------
    @staticmethod
    def eligible(attn, hidden_states: torch.Tensor, ctx, comm_manager) -> bool:
        """Decode step of one request with exactly eight query rows."""
        if ctx.forward_mode is None or not ctx.forward_mode.is_decode():
            return False
        if ctx.num_extends != 0 or ctx.bs != 1:
            return False
        if hidden_states.shape[0] != NUM_TOKENS or ctx.draft_narrowing is not None:
            return False
        if comm_manager.needs_pre_attn_all_gather() or not comm_manager.use_all_reduce(
            comm_manager.is_moe
        ):
            return False
        metadata = getattr(ctx.attn_backend, "forward_decode_metadata", None)
        if metadata is None or getattr(metadata, "page_table", None) is None:
            return False
        window = attn._resolve_decode_window(ctx, metadata, total_tokens=NUM_TOKENS)
        return (
            window.start == 0
            and window.num_reqs == 1
            and window.num_tokens == NUM_TOKENS
            and window.q_len_per_req == NUM_TOKENS
        )

    @classmethod
    def begin_forward(cls) -> None:
        """Invalidate the per-forward input views (call once per model forward)."""
        cls._step_inputs = None

    def _prepare_step_inputs(self, ctx, positions, out_cache_loc) -> dict:
        """Int32/int64 views the kernels expect, converted by the first
        engine layer of the forward and reused by the rest."""
        cached = LongcatAttention0Engine._step_inputs
        if cached is not None:
            return cached
        metadata = ctx.attn_backend.forward_decode_metadata
        # The backend sizes every decode block table to the padded page count
        # of ``max_context_len`` (eager and graph alike); the indexer engine
        # compiles and sizes its logits per width, so pin the first width
        # and refuse to recompile silently.
        block_table = metadata.page_table[0:1]
        if block_table.dtype != torch.int32:
            block_table = block_table.to(torch.int32)
        block_table = block_table.contiguous()
        width = int(block_table.shape[1])
        pinned = LongcatAttention0Engine._block_table_width
        if pinned is None:
            LongcatAttention0Engine._block_table_width = width
        elif pinned != width:
            raise RuntimeError(
                f"attention0 engine block table width changed {pinned} -> {width}"
            )
        inputs = dict(
            positions=positions[:NUM_TOKENS].to(torch.int32),
            out_cache_loc32=out_cache_loc[:NUM_TOKENS].to(torch.int32),
            out_cache_loc64=out_cache_loc[:NUM_TOKENS].to(torch.int64),
            seq_lens=metadata.seq_lens_k[0:1].to(torch.int32),
            block_table=block_table,
        )
        LongcatAttention0Engine._step_inputs = inputs
        return inputs

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        ctx,
        quant_scales: tuple[torch.Tensor | None, torch.Tensor | None] = (None, None),
    ) -> tuple[torch.Tensor, torch.Tensor, DSASelection]:
        """Run the engine step.

        Args:
            positions: ``[8]`` token positions.
            hidden_states: ``[8, 8192]`` bf16 normalized attention input.
            residual: ``[8, 8192]`` bf16 residual stream.
            ctx: Forward context of a decode batch that passed ``eligible``.

        Returns:
            ``(normed, residual_out, selection)``: the post-attention RMSNorm
            output, the updated residual (all-reduced o_proj + residual), and
            the decode selection for attention branch 1.
        """
        out_cache_loc = ctx.attn_backend.write_locations(
            self.attn.attn_mqa, ForwardMode.DECODE
        )
        pool = ctx.token_to_kv_pool
        kv_cache = pool.get_key_buffer(self.kv_layer_id).view(-1, 576)
        if not self._ready:
            if int(pool.arena.kv_page_size) != _PAGE_SIZE:
                raise RuntimeError("attention0 engine needs 64-row KV pages")
            if kv_cache.shape[0] >= _MAX_SLOTS:
                raise RuntimeError(
                    f"attention0 engine transports 24-bit slot ids; pool has "
                    f"{kv_cache.shape[0]} slots"
                )
            self._prepare()
        inputs = self._prepare_step_inputs(ctx, positions, out_cache_loc)
        hidden_states = hidden_states.contiguous()
        residual = residual.contiguous()
        collective = self.collective
        if self.tp_rank == 0:
            if self.indexer_ws is None:
                self.indexer_ws = make_indexer_engine_workspace(
                    self.device, int(inputs["block_table"].shape[1])
                )
            index_k = (
                pool.get_index_k_buffer(self.kv_layer_id).view(torch.uint8).reshape(-1)
            )
            run_indexer_engine(
                hidden_states,
                self.indexer_weights,
                self.cos_sin_cache,
                inputs["positions"],
                inputs["out_cache_loc64"],
                index_k,
                kv_cache,
                inputs["seq_lens"],
                inputs["block_table"],
                self.indexer_ws,
                softmax_scale=self.indexer_scale,
                collective=collective,
                residual=residual,
                post_norm_gamma=self.post_norm_gamma,
                norm_eps=self.post_norm_eps,
                peer_bases=self.exchange.peer_bases,
                quant_global=quant_scales[0],
                quant2_global=quant_scales[1],
            )
            ws = self.indexer_ws
            topk_slots = ws.topk.view(NUM_TOKENS, TOPK)
            topk_lens = ws.lens.view(NUM_TOKENS)
        else:
            if self.worker_ws is None:
                self.worker_ws = make_worker_engine_workspace(
                    self.device, self.local_heads, kv_cache.dtype
                )
            ws = self.worker_ws
            slots, lens, flag = self._selection_views()
            run_worker_engine(
                hidden_states,
                self.w_qkv_a,
                self.q_gamma,
                self.kv_gamma,
                self.q_b_weight,
                self.w_kc,
                self.w_vc_t,
                self.o_proj_weight,
                self.post_norm_gamma,
                self.cos_sin_cache,
                inputs["positions"],
                inputs["out_cache_loc32"],
                inputs["block_table"],
                kv_cache,
                slots,
                lens,
                flag,
                ws,
                sm_scale=self.sm_scale,
                k_scale=self.k_scale,
                rank=self.tp_rank,
                collective=collective,
                norm_eps=self.post_norm_eps,
                selection_slot_words=self.exchange.words,
                residual=residual,
                quant_global=quant_scales[0],
                quant2_global=quant_scales[1],
            )
            topk_slots, topk_lens = ws.topk_out, ws.lens_out
        self.selection.prefill = None
        self.selection.decode = DSADecodeSelection(
            topk_indices=topk_slots, topk_lens=topk_lens
        )
        return ws.norm_out, ws.residual_out, self.selection

    def quantized_outputs(self, quant_scales):
        """Return NVFP4 (values,scales) per quant_scales entry, or None if absent.

        Values are uint8[8,4096]; scales use fp4_quantize's swizzled E4M3[128*512]
        plane. Indexer P5 and worker P7 each pack their own workspace outputs.
        """
        ws = self.indexer_ws if self.tp_rank == 0 else self.worker_ws
        if ws is None:
            return tuple(None for _ in quant_scales)
        pairs = ((ws.quant_out, ws.quant_scale), (ws.quant2_out, ws.quant2_scale))
        return tuple(
            pair if scale is not None else None
            for pair, scale in zip(pairs, quant_scales)
        )

    def _selection_views(self):
        """Slot-0 views of this layer's exchange region; the kernel adds
        ``(generation & 1) * words`` itself (``selection_slot_words``), which
        keeps one captured step valid for every replay."""
        state = self.exchange
        words = state.words
        flat = state.symmetric_buffer.view(-1)
        slots = flat[: NUM_TOKENS * TOPK].view(NUM_TOKENS, TOPK)
        lens = flat[NUM_TOKENS * TOPK : NUM_TOKENS * TOPK + NUM_TOKENS]
        flag = flat[2 * words : 2 * words + 1].view(torch.uint32)
        return slots, lens, flag

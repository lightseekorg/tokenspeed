# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from tokenspeed.runtime.distributed.comm_ops import all_gather_into_tensor
from tokenspeed.runtime.execution.cache_loc_kernel import (
    compute_out_cache_loc_uniform,
    dflash_prepare_decode,
)
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.drafter._dflash_fused_kv import (
    _fused_norm_rope_stacked_scatter,
    _get_kv_buffer_ptrs,
)
from tokenspeed.runtime.execution.drafter.base import BaseDrafter
from tokenspeed.runtime.execution.forward_batch_info import (
    CaptureHiddenMode,
    ForwardMode,
)
from tokenspeed.runtime.layers.logits_processor import LogitsMetadata
from tokenspeed.runtime.utils import get_colorful_logger
from tokenspeed.runtime.utils.nvtx import nvtx_range

if TYPE_CHECKING:
    from tokenspeed.runtime.execution.input_buffer import InputBuffers
    from tokenspeed.runtime.execution.model_runner import ModelRunner
    from tokenspeed.runtime.execution.runtime_states import RuntimeStates
    from tokenspeed.runtime.layers.logits_processor import LogitsProcessorOutput

logger = get_colorful_logger(__name__)


class DFlash(BaseDrafter):
    """DFlash block drafter backed by a native TokenSpeed draft model."""

    def __init__(
        self,
        spec_num_tokens: int,
        spec_num_steps: int,
        page_size: int,
        draft_model_runner: ModelRunner | None = None,
        req_to_page: torch.Tensor | None = None,
        attn_backend=None,
        token_to_kv_pool=None,
        runtime_states: RuntimeStates | None = None,
        input_buffers: InputBuffers | None = None,
        vocab_size: int | None = None,
    ) -> None:
        super().__init__(
            spec_num_tokens=spec_num_tokens,
            spec_num_steps=spec_num_steps,
            draft_model_runner=draft_model_runner,
            runtime_states=runtime_states,
            input_buffers=input_buffers,
            page_size=page_size,
            req_to_page=req_to_page,
            attn_backend=attn_backend,
            token_to_kv_pool=token_to_kv_pool,
            vocab_size=vocab_size,
        )
        if draft_model_runner is None:
            raise ValueError("Native DFLASH requires a draft model runner.")
        server_args = draft_model_runner.server_args
        if not server_args.speculative_draft_model_path:
            raise ValueError("DFLASH requires --speculative-draft-model-path.")

        self.device = torch.device(draft_model_runner.device)
        self.model = draft_model_runner.model

        cfg = self.model.config
        dflash_cfg = getattr(cfg, "dflash_config", {}) or {}
        self.target_layer_ids = [int(x) for x in dflash_cfg.get("target_layer_ids", [])]
        if not self.target_layer_ids:
            raise ValueError(
                "DFLASH draft config must define dflash_config.target_layer_ids."
            )
        if "mask_token_id" not in dflash_cfg:
            raise ValueError(
                "DFLASH draft config must define dflash_config.mask_token_id."
            )
        self.mask_token_id = int(dflash_cfg["mask_token_id"])
        self.block_size = int(getattr(cfg, "block_size", spec_num_tokens))
        if self.block_size != int(spec_num_tokens):
            logger.warning(
                "DFLASH block size mismatch: checkpoint block_size=%s, "
                "runtime speculative_num_draft_tokens=%s.",
                self.block_size,
                spec_num_tokens,
            )
        self.hidden_size = int(getattr(cfg, "hidden_size"))
        self.idle_forward_steps = 1
        self._init_native_buffers()
        self._greedy_gathered_max: torch.Tensor | None = None
        self._greedy_gathered_ids: torch.Tensor | None = None
        self._greedy_gather_cap = 0
        self._init_fused_kv_helper()
        self._init_incremental_proj()

    def _init_native_buffers(self) -> None:
        if self.input_buffers is None:
            raise ValueError("Native DFLASH requires input buffers.")
        if self.req_to_page is None:
            raise ValueError("Native DFLASH requires req_to_page.")
        if self.attn_backend is None or self.token_to_kv_pool is None:
            raise ValueError("Native DFLASH requires draft attention components.")

        max_bs = self.input_buffers.max_bs
        self.draft_seq_lens_buf = torch.zeros(
            (max_bs,), dtype=torch.int32, device=self.device
        )
        self.draft_out_cache_loc_buf = torch.empty(
            (max_bs * self.spec_num_tokens,),
            dtype=torch.int32,
            device=self.device,
        )
        self.draft_input_lengths_buf = torch.full(
            (max_bs,),
            self.spec_num_tokens,
            dtype=torch.int32,
            device=self.device,
        )
        self.draft_extend_seq_lens_cpu = torch.full(
            (max_bs,),
            self.spec_num_tokens,
            dtype=torch.int32,
            pin_memory=True,
        )
        self.block_offsets = torch.arange(
            self.spec_num_tokens, dtype=torch.int64, device=self.device
        )
        self.block_ids_buf = torch.full(
            (max_bs, self.spec_num_tokens),
            self.mask_token_id,
            dtype=torch.int32,
            device=self.device,
        )
        self.block_positions_buf = torch.empty(
            (max_bs, self.spec_num_tokens), dtype=torch.int64, device=self.device
        )
        self.next_tokens_buf = torch.empty(
            (max_bs, self.spec_num_tokens), dtype=torch.int32, device=self.device
        )
        self.current_tokens_buf = torch.empty(
            (max_bs,), dtype=torch.int32, device=self.device
        )
        self.decode_offsets_buf = (
            torch.arange(max_bs, dtype=torch.int64, device=self.device)
            * self.spec_num_tokens
            - 1
        )
        self.gather_indices_buf = torch.empty(
            (max_bs,), dtype=torch.int64, device=self.device
        )

    def bind_target_model(self, target_model) -> None:
        language_model = getattr(target_model, "language_model", target_model)
        self.target_model = target_model
        self.target_language_model = language_model
        self.embed_tokens = target_model.get_input_embeddings()
        self.lm_head = target_model.lm_head
        self.logits_processor = language_model.logits_processor

    def _greedy_gather_capacity(self) -> int:
        """Max element count for the greedy head's tensor-parallel all-gather
        scratch: a full ``max_bs`` decode block.

        The greedy head samples the last ``spec_num_tokens - 1`` block
        positions per request and all-gathers them across the TP group, so the
        worst case is ``tp_size * max_bs * (spec_num_tokens - 1)``.
        """
        tp_size = int(self.logits_processor.tp_size)
        return tp_size * self.input_buffers.max_bs * max(self.spec_num_tokens - 1, 1)

    def _ensure_greedy_gather_buffers(
        self,
        max_dtype: torch.dtype,
        ids_dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Lazily create the greedy all-gather scratch ONCE at its maximum
        capacity, then reuse it in place for every batch size.

        Sizing to the max ``max_bs`` block (rather than growing per batch size)
        is required for CUDA-graph correctness. Graphs are captured for
        increasing batch sizes (``[1, 2, ..., max_bs]``); a buffer grown lazily
        would be freed and reallocated when a larger bs needs more room, leaving
        every smaller-bs graph captured earlier with an
        ``all_gather_into_tensor`` recorded against freed memory. On replay
        those small-bs decode steps read garbage (out-of-vocab) draft token ids,
        which flow into the next verify forward's embedding lookup and trigger a
        CUDA illegal memory access. A fixed max-capacity buffer is allocated
        during warmup (before capture) and shared by every captured graph.

        Returns the (max, id) scratch tensors; callers slice ``[:needed]``.
        """
        cap = self._greedy_gather_capacity()
        if (
            self._greedy_gathered_max is None
            or self._greedy_gathered_ids is None
            or self._greedy_gather_cap < cap
            or self._greedy_gathered_max.dtype != max_dtype
            or self._greedy_gathered_max.device != device
            or self._greedy_gathered_ids.dtype != ids_dtype
        ):
            self._greedy_gathered_max = torch.empty(
                (cap,), dtype=max_dtype, device=device
            )
            self._greedy_gathered_ids = torch.empty(
                (cap,), dtype=ids_dtype, device=device
            )
            self._greedy_gather_cap = cap
        return self._greedy_gathered_max, self._greedy_gathered_ids

    def _greedy_sample_from_vocab_parallel_head(
        self,
        hidden_states: torch.Tensor,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not hasattr(self.lm_head, "weight") or not hasattr(
            self.lm_head, "shard_indices"
        ):
            metadata = LogitsMetadata(forward_mode=ForwardMode.DECODE)
            logits = self.logits_processor._get_logits(
                hidden_states, self.lm_head, metadata
            )
            argmax = torch.argmax(logits, dim=-1)
            if out is not None:
                out.copy_(argmax.view_as(out))
                return out
            return argmax.to(torch.int32)

        shard = self.lm_head.shard_indices
        weight = self.lm_head.weight
        hidden_states = hidden_states.to(weight.dtype)

        num_org = int(shard.num_org_elements)
        num_org_padded = int(shard.num_org_elements_padded)
        num_added = int(shard.num_added_elements)
        org_vocab_start = int(shard.org_vocab_start_index)
        added_vocab_start = int(shard.added_vocab_start_index)

        chunk_len = int(hidden_states.shape[0])
        if num_org > 0:
            base_logits = torch.matmul(hidden_states, weight[:num_org].T)
            local_max, local_arg = torch.max(base_logits, dim=-1)
        else:
            local_max = torch.full(
                (chunk_len,),
                torch.finfo(weight.dtype).min,
                dtype=weight.dtype,
                device=hidden_states.device,
            )
            local_arg = torch.zeros(
                (chunk_len,), dtype=torch.int64, device=hidden_states.device
            )

        if num_added > 0:
            added_start = num_org_padded
            added_end = num_org_padded + num_added
            added_weight = weight[added_start:added_end]
            added_logits = torch.matmul(hidden_states, added_weight.T)
            added_max, added_arg = torch.max(added_logits, dim=-1)
            use_added = added_max > local_max
            local_max = torch.where(use_added, added_max, local_max)
            local_arg = torch.where(
                use_added,
                added_arg.to(local_arg.dtype) + num_org_padded,
                local_arg,
            )

        if num_added == 0:
            global_ids = local_arg + org_vocab_start
        else:
            global_ids = torch.empty(
                (chunk_len,), dtype=torch.int64, device=hidden_states.device
            )
            is_base = local_arg < num_org
            global_ids[is_base] = org_vocab_start + local_arg[is_base]
            global_ids[~is_base] = added_vocab_start + (
                local_arg[~is_base] - num_org_padded
            )

        tp_size = int(self.logits_processor.tp_size)
        if tp_size == 1:
            if out is not None:
                out.copy_(global_ids.view_as(out))
                return out
            return global_ids.to(torch.int32)

        needed = tp_size * chunk_len
        gathered_max, gathered_ids = self._ensure_greedy_gather_buffers(
            local_max.dtype, global_ids.dtype, hidden_states.device
        )
        gathered_max = gathered_max[:needed]
        gathered_ids = gathered_ids[:needed]
        all_gather_into_tensor(
            gathered_max,
            local_max.contiguous(),
            self.logits_processor.tp_group,
        )
        all_gather_into_tensor(
            gathered_ids,
            global_ids.contiguous(),
            self.logits_processor.tp_group,
        )

        gathered_max = gathered_max.view(tp_size, chunk_len)
        gathered_ids = gathered_ids.view(tp_size, chunk_len)
        best_rank = torch.argmax(gathered_max, dim=0).unsqueeze(0)
        result = torch.gather(gathered_ids, 0, best_rank).view(-1)
        if out is not None:
            out.copy_(result.view_as(out))
            return out
        return result.to(torch.int32)

    @nvtx_range("dflash_update_native_cache", color="purple")
    def _update_native_cache_from_target(
        self,
        base_ctx: ForwardContext,
        logits_output: LogitsProcessorOutput,
        accept_lengths: torch.Tensor,
    ) -> None:
        hidden = logits_output.hidden_states
        if hidden is None:
            raise RuntimeError("DFLASH requires target hidden states.")
        if hidden.shape[0] != base_ctx.input_num_tokens:
            raise RuntimeError(
                "DFLASH hidden-state/token mismatch: "
                f"hidden_tokens={hidden.shape[0]}, input_tokens={base_ctx.input_num_tokens}."
            )

        bs = base_ctx.bs
        # The target verify forward emits spec_num_tokens hidden states per
        # decode request (the candidate block); input_lengths_buf only tracks
        # the committed-token count there, so split decode rows by
        # spec_num_tokens. Prefill rows keep their real chunk lengths.
        lengths = self.input_buffers.input_lengths_buf[:bs].to(torch.int64).clone()
        lengths[base_ctx.num_extends :] = self.spec_num_tokens
        req_pool_indices = self.input_buffers.req_pool_indices_buf[:bs]
        positions = self.input_buffers.positions_buf[: base_ctx.input_num_tokens]
        cache_locs = self.input_buffers.out_cache_loc_buf[: base_ctx.input_num_tokens]

        decode_only = base_ctx.num_extends == 0
        if (
            decode_only
            and torch.cuda.is_available()
            and torch.cuda.is_current_stream_capturing()
        ):
            old_lens = self.runtime_states.valid_cache_lengths.index_select(
                0, req_pool_indices
            )
            self.draft_seq_lens_buf[:bs].copy_(
                old_lens.to(torch.int32) + accept_lengths[:bs].to(torch.int32)
            )
            self._write_native_cache(hidden, positions, cache_locs, decode_only=True)
            return

        hidden_chunks = torch.split(hidden, lengths.detach().cpu().tolist(), dim=0)
        pos_chunks = torch.split(positions, lengths.detach().cpu().tolist(), dim=0)
        loc_chunks = torch.split(cache_locs, lengths.detach().cpu().tolist(), dim=0)

        selected_hidden = []
        selected_positions = []
        selected_cache_locs = []
        new_seq_lens = torch.empty((bs,), dtype=torch.int32, device=self.device)

        for row, (chunk, pos_chunk, loc_chunk) in enumerate(
            zip(hidden_chunks, pos_chunks, loc_chunks, strict=True)
        ):
            if row < base_ctx.num_extends:
                take = int(chunk.shape[0])
            else:
                take = int(accept_lengths[row].item())
            if take <= 0:
                pool_idx = req_pool_indices[row]
                new_seq_lens[row] = self.runtime_states.valid_cache_lengths[pool_idx]
                continue

            chunk = chunk[:take].contiguous()
            pos_chunk = pos_chunk[:take].contiguous()
            loc_chunk = loc_chunk[:take].contiguous()
            selected_hidden.append(chunk)
            selected_positions.append(pos_chunk)
            selected_cache_locs.append(loc_chunk)
            new_seq_lens[row] = (pos_chunk[-1] + 1).to(torch.int32)

        self.draft_seq_lens_buf[:bs].copy_(new_seq_lens)
        if not selected_hidden:
            return

        target_hidden = torch.cat(selected_hidden, dim=0)
        target_positions = torch.cat(selected_positions, dim=0)
        target_cache_locs = torch.cat(selected_cache_locs, dim=0)
        self._write_native_cache(
            target_hidden,
            target_positions,
            target_cache_locs,
            decode_only=decode_only,
        )

    def _write_native_cache(
        self,
        target_hidden: torch.Tensor,
        target_positions: torch.Tensor,
        target_cache_locs: torch.Tensor,
        decode_only: bool = False,
    ) -> None:
        target_hidden = target_hidden.to(
            device=self.device,
            dtype=self.draft_model_runner.model.fc.weight.dtype,
        )
        expected_width = int(self.draft_model_runner.model.fc.in_features)
        actual_width = int(target_hidden.shape[-1])
        if actual_width != expected_width:
            raise RuntimeError(
                "DFLASH captured hidden width mismatch: "
                f"expected {expected_width}, got {actual_width}. "
                "Check dflash_config.target_layer_ids against the target model."
            )
        with torch.inference_mode():
            ctx_hidden = self.draft_model_runner.model.project_target_hidden(
                target_hidden
            )
            if decode_only:
                self._write_native_cache_fused(
                    ctx_hidden, target_positions, target_cache_locs
                )
                return

            for layer in self.draft_model_runner.model.layers:
                attn = layer.self_attn
                k, v = attn.kv_proj_only(ctx_hidden)
                k = attn.apply_k_norm(k)
                k = attn.apply_k_rope(target_positions, k)
                k = k.view(-1, attn.num_kv_heads, attn.head_dim)
                v = v.view(-1, attn.num_kv_heads, attn.head_dim)
                self.token_to_kv_pool.set_kv_buffer(
                    attn.attn,
                    target_cache_locs,
                    k,
                    v,
                    attn.attn.k_scale,
                    attn.attn.v_scale,
                )

    def _init_fused_kv_helper(self) -> None:
        """Pre-stack KV weights, k_norm, eps, and cos_sin_cache at construction."""
        self._fused_kv_enabled = False
        self._fused_kv_workspace_capacity = 0
        self._fused_kv_workspace_dtype = None
        self._fused_kv_proj_workspace = None
        # torch.mm(out=...) is always used; workspace pre-allocated at warmup.
        self._fused_kv_k_buffers = []
        self._fused_kv_v_buffers = []
        # Aux stream for overlapping KV cache write with draft block preparation
        self._kv_aux_stream: torch.cuda.Stream | None = None
        self._kv_fork_event: torch.cuda.Event | None = None
        self._kv_join_event: torch.cuda.Event | None = None
        if torch.cuda.is_available():
            self._kv_aux_stream = torch.cuda.Stream(device=self.device)
            self._kv_fork_event = torch.cuda.Event()
            self._kv_join_event = torch.cuda.Event()
        try:
            layers = self.draft_model_runner.model.layers
            if not layers:
                return
            first_attn = layers[0].self_attn
            is_neox = bool(getattr(first_attn.rotary_emb, "is_neox_style", True))
            if not is_neox:
                return

            from tokenspeed.runtime.layers.dense.unquant import UnquantizedLinearMethod

            for layer in layers:
                attn = layer.self_attn
                if not isinstance(
                    getattr(attn.qkv_proj, "quant_method", None),
                    UnquantizedLinearMethod,
                ):
                    return
                if not hasattr(attn.qkv_proj, "weight"):
                    return

            num_kv_heads = int(first_attn.num_kv_heads)
            head_dim = int(first_attn.head_dim)
            kv_size = int(first_attn.kv_size)
            rotary_dim = int(getattr(first_attn.rotary_emb, "rotary_dim", head_dim))
            n_layers = len(layers)

            self._fused_kv_num_kv_heads = num_kv_heads
            self._fused_kv_head_dim = head_dim
            self._fused_kv_kv_size = kv_size
            self._fused_kv_rotary_dim = rotary_dim
            self._fused_kv_n_layers = n_layers
            self._fused_kv_layer_out_dim = 2 * kv_size

            kv_weight_rows = []
            k_norm_rows = []
            eps_values = []
            for layer in layers:
                attn = layer.self_attn
                kv_weight_rows.append(
                    attn.qkv_proj.weight[attn.q_size : attn.q_size + 2 * attn.kv_size]
                )
                k_norm_rows.append(attn.k_norm.weight)
                eps_values.append(float(attn.k_norm.variance_epsilon))

            flat_kv_weight = torch.cat(kv_weight_rows, dim=0)
            self._fused_kv_flat_weight_t = flat_kv_weight.t().contiguous()
            self._fused_kv_k_norm_weight = torch.stack(k_norm_rows, dim=0).contiguous()
            self._fused_kv_eps = torch.tensor(
                eps_values, dtype=torch.float32, device=self.device
            )

            cos_sin_cache = first_attn.rotary_emb.cos_sin_cache
            if cos_sin_cache.device != self.device:
                cos_sin_cache = cos_sin_cache.to(self.device)
            self._fused_kv_cos_sin_cache = cos_sin_cache

            self._fused_kv_k_buffers = [
                self.token_to_kv_pool.get_key_buffer(layer.self_attn.attn.layer_id)
                for layer in layers
            ]
            self._fused_kv_v_buffers = [
                self.token_to_kv_pool.get_value_buffer(layer.self_attn.attn.layer_id)
                for layer in layers
            ]

            self._fused_kv_k_ptrs, self._fused_kv_v_ptrs = _get_kv_buffer_ptrs(
                self._fused_kv_k_buffers, self._fused_kv_v_buffers
            )

            self._fused_kv_inv_k_scales = None
            self._fused_kv_inv_v_scales = None
            if self._fused_kv_k_buffers[0].dtype == torch.float8_e4m3fn:
                has_scale = any(
                    getattr(layer.self_attn.attn, "k_scale", None) is not None
                    or getattr(layer.self_attn.attn, "v_scale", None) is not None
                    for layer in layers
                )
                if has_scale:
                    inv_k_vals = []
                    inv_v_vals = []
                    for layer in layers:
                        attn = layer.self_attn.attn
                        k_s = getattr(attn, "k_scale", None)
                        v_s = getattr(attn, "v_scale", None)
                        inv_k_vals.append(1.0 / float(k_s) if k_s is not None else 1.0)
                        inv_v_vals.append(1.0 / float(v_s) if v_s is not None else 1.0)
                    self._fused_kv_inv_k_scales = torch.tensor(
                        inv_k_vals, dtype=torch.float32, device=self.device
                    )
                    self._fused_kv_inv_v_scales = torch.tensor(
                        inv_v_vals, dtype=torch.float32, device=self.device
                    )

            self._fused_kv_enabled = True

            max_total_ctx = self.input_buffers.max_bs * self.spec_num_tokens
            ws_dtype = self.draft_model_runner.model.fc.weight.dtype
            self._fused_kv_proj_workspace = torch.empty(
                (max_total_ctx, n_layers * self._fused_kv_layer_out_dim),
                dtype=ws_dtype,
                device=self.device,
            )
            self._fused_kv_workspace_capacity = max_total_ctx
            self._fused_kv_workspace_dtype = ws_dtype

            logger.info(
                "DFLASH fused KV materialization enabled. "
                "n_layers=%d, num_kv_heads=%d, head_dim=%d",
                n_layers,
                num_kv_heads,
                head_dim,
            )
        except Exception as e:
            logger.warning(
                "DFLASH fused KV initialization failed, falling back to sequential: %s",
                e,
            )
            self._fused_kv_enabled = False

    def _init_incremental_proj(self) -> None:
        self._incremental_proj_enabled = False
        self._incremental_kv_write_done = False
        if not self._fused_kv_enabled:
            return
        if self._kv_aux_stream is None:
            return
        try:
            fc = self.draft_model_runner.model.fc
            hidden_norm = self.draft_model_runner.model.hidden_norm
            fc_weight = fc.weight.data
            hidden_size = fc_weight.shape[0]
            n_captures = len(self.target_layer_ids)
            in_features = fc_weight.shape[1]
            if in_features != n_captures * hidden_size:
                logger.warning(
                    "Incremental proj disabled: fc.in_features=%d != n_captures(%d) * hidden(%d)",
                    in_features,
                    n_captures,
                    hidden_size,
                )
                return

            ws_dtype = fc_weight.dtype
            max_tokens = self.input_buffers.max_bs * (self.spec_num_tokens + 1)
            self._incr_n_captures = n_captures
            self._incr_hidden_norm = hidden_norm
            self._incr_sub_weights_t = []
            for i in range(n_captures):
                sub_w = fc_weight[:, i * hidden_size : (i + 1) * hidden_size]
                self._incr_sub_weights_t.append(sub_w.t().contiguous())

            self._incr_acc_buf = torch.zeros(
                (max_tokens, hidden_size), dtype=ws_dtype, device=self.device
            )
            self._incr_slot_bufs = [
                torch.empty(
                    (max_tokens, hidden_size), dtype=ws_dtype, device=self.device
                )
                for _ in range(n_captures)
            ]
            self._incr_capture_events = [torch.cuda.Event() for _ in range(n_captures)]
            self._incr_num_tokens = 0
            self._incremental_proj_enabled = True
            logger.info(
                "DFLASH incremental projection enabled. "
                "n_captures=%d, hidden_size=%d, max_tokens=%d",
                n_captures,
                hidden_size,
                max_tokens,
            )
        except Exception as e:
            logger.warning("DFLASH incremental projection init failed: %s", e)
            self._incremental_proj_enabled = False

    def _prepare_incremental_proj(
        self, num_tokens: int, positions: torch.Tensor, cache_locs: torch.Tensor
    ) -> None:
        self._incr_num_tokens = num_tokens
        self._incr_positions = positions
        self._incr_cache_locs = cache_locs
        self._incremental_kv_write_done = False
        self._incr_acc_buf[:num_tokens].zero_()
        self.target_language_model.model._dflash_incr_active = True

    def _on_capture_slot_ready(self, capture_idx: int, num_tokens: int) -> None:
        event = self._incr_capture_events[capture_idx]
        event.record(torch.cuda.current_stream())

        with torch.cuda.stream(self._kv_aux_stream):
            self._kv_aux_stream.wait_event(event)
            hidden = self._incr_slot_bufs[capture_idx][:num_tokens]
            acc = self._incr_acc_buf[:num_tokens]
            torch.addmm(
                acc,
                hidden,
                self._incr_sub_weights_t[capture_idx],
                beta=1.0,
                alpha=1.0,
                out=acc,
            )

            if capture_idx == self._incr_n_captures - 1:
                ctx_hidden = self._incr_hidden_norm(acc)
                self._write_native_cache_fused(
                    ctx_hidden, self._incr_positions, self._incr_cache_locs
                )
                self._incremental_kv_write_done = True
                self._kv_join_event.record(self._kv_aux_stream)

    def _ensure_fused_workspace(self, total_ctx: int, dtype: torch.dtype) -> None:
        """Ensure the projection workspace is large enough.

        The workspace is pre-allocated at init to max_bs * spec_num_tokens,
        so this should always be a no-op.
        """
        if (
            self._fused_kv_workspace_capacity >= total_ctx
            and self._fused_kv_workspace_dtype == dtype
            and self._fused_kv_proj_workspace is not None
        ):
            return
        raise RuntimeError(
            f"DFLASH fused KV workspace too small: need {total_ctx}, "
            f"have {self._fused_kv_workspace_capacity}. "
            "This should not happen — workspace is pre-allocated at init."
        )

    def _write_native_cache_fused(
        self,
        ctx_hidden: torch.Tensor,
        target_positions: torch.Tensor,
        target_cache_locs: torch.Tensor,
    ) -> None:
        """Fused KV materialization for decode-only batches.

        One stacked GEMM for all 6 layers' K|V projection, then one Triton
        kernel for fused RMSNorm + RoPE + direct scatter into KV pool.
        Total: 1 GEMM + 1 Triton launch.
        """
        layers = self.draft_model_runner.model.layers
        if not self._fused_kv_enabled:
            for layer in layers:
                attn = layer.self_attn
                k, v = attn.kv_proj_only(ctx_hidden)
                k = attn.apply_k_norm(k)
                k = attn.apply_k_rope(target_positions, k)
                k = k.view(-1, attn.num_kv_heads, attn.head_dim)
                v = v.view(-1, attn.num_kv_heads, attn.head_dim)
                self.token_to_kv_pool.set_kv_buffer(
                    attn.attn,
                    target_cache_locs,
                    k,
                    v,
                    attn.attn.k_scale,
                    attn.attn.v_scale,
                )
            return

        total_ctx = int(ctx_hidden.shape[0])
        self._ensure_fused_workspace(total_ctx, ctx_hidden.dtype)

        proj_out_2d = self._fused_kv_proj_workspace[:total_ctx]
        torch.mm(ctx_hidden, self._fused_kv_flat_weight_t, out=proj_out_2d)

        proj_out = proj_out_2d.view(
            total_ctx, self._fused_kv_n_layers, self._fused_kv_layer_out_dim
        )

        _fused_norm_rope_stacked_scatter(
            proj_out,
            self._fused_kv_k_norm_weight,
            self._fused_kv_eps,
            self._fused_kv_cos_sin_cache,
            target_positions,
            target_cache_locs,
            self._fused_kv_k_buffers,
            self._fused_kv_v_buffers,
            self._fused_kv_num_kv_heads,
            self._fused_kv_head_dim,
            self._fused_kv_rotary_dim,
            self._fused_kv_inv_k_scales,
            self._fused_kv_inv_v_scales,
        )

    @staticmethod
    def _current_tokens_from_output(
        output_tokens: torch.Tensor,
        accept_lengths: torch.Tensor,
        num_extends: int,
        spec_num_tokens: int,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bs = accept_lengths.shape[0]
        current = (
            out
            if out is not None
            else torch.empty((bs,), dtype=torch.int32, device=output_tokens.device)
        )
        if num_extends > 0:
            current[:num_extends] = output_tokens[:num_extends]
        num_decodes = bs - num_extends
        if num_decodes > 0:
            offsets = (
                torch.arange(
                    num_decodes, dtype=torch.int64, device=output_tokens.device
                )
                * spec_num_tokens
                - 1
                + num_extends
            )
            safe_accept_lengths = (
                accept_lengths[num_extends:].to(torch.int64).clamp(1, spec_num_tokens)
            )
            current[num_extends:] = output_tokens[offsets + safe_accept_lengths]
        return current

    def get_candidates(self, base_ctx: ForwardContext) -> torch.Tensor | None:
        num_extends = base_ctx.num_extends
        num_decodes = base_ctx.bs - num_extends
        if num_decodes == 0:
            return None
        num_decode_tokens = num_decodes * self.spec_num_tokens
        num_prefill_tokens = base_ctx.input_num_tokens - num_decode_tokens
        return self.input_buffers.input_ids_buf[
            num_prefill_tokens : base_ctx.input_num_tokens
        ].reshape(num_decodes, self.spec_num_tokens)

    def draft(self, current_tokens: torch.Tensor) -> torch.Tensor:
        return self._draft_native(current_tokens)

    @nvtx_range("dflash_native_draft", color="purple")
    def _draft_native(
        self,
        current_tokens: torch.Tensor,
        kv_sync_event: torch.cuda.Event = None,
        prepared: bool = False,
    ) -> torch.Tensor:
        bs = current_tokens.shape[0]
        req_pool_indices = self.input_buffers.req_pool_indices_buf[:bs]
        prefix_lens = self.draft_seq_lens_buf[:bs]
        seq_lens_after = prefix_lens + int(self.spec_num_tokens)

        block_ids = self.block_ids_buf[:bs]
        # NOTE: callers (run/_run_overlap) write current_tokens directly into
        # block_ids_buf[:bs, 0] before invoking _draft_native
        block_positions = self.block_positions_buf[:bs]
        cache_locs = self.draft_out_cache_loc_buf[: bs * self.spec_num_tokens]
        if not prepared:
            torch.add(
                prefix_lens.unsqueeze(1),
                self.block_offsets,
                out=block_positions,
            )

            compute_out_cache_loc_uniform(
                out_cache_loc_ptr=cache_locs,
                req_pool_indices=req_pool_indices,
                uniform_input_length=self.spec_num_tokens,
                cache_start=prefix_lens,
                req_to_pages=self.req_to_page,
                page_size=self.page_size,
            )

        is_capturing = (
            torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
        )
        if not is_capturing:
            self.attn_backend.init_forward_metadata(
                bs=bs,
                num_extends=bs,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens_after,
                req_to_page=self.req_to_page,
                forward_mode=ForwardMode.DECODE,
                extend_seq_lens=None,
                extend_seq_lens_cpu=self.draft_extend_seq_lens_cpu[:bs],
                extend_prefix_lens=None,
                extend_prefix_lens_cpu=None,
            )
        else:
            self.attn_backend.fill_block_decode_seq_lens(bs, seq_lens_after)

        ctx = ForwardContext(
            attn_backend=self.attn_backend,
            token_to_kv_pool=self.token_to_kv_pool,
            req_to_page=self.req_to_page,
            bs=bs,
            num_extends=bs,
            input_num_tokens=bs * self.spec_num_tokens,
            forward_mode=ForwardMode.DECODE,
            capture_hidden_mode=CaptureHiddenMode.FULL,
        )

        flat_ids = block_ids.reshape(-1)
        input_embeds = self.embed_tokens(flat_ids, reduce_results=False)

        with torch.inference_mode():
            logits_output = self.draft_model_runner.forward(
                ctx=ctx,
                input_ids=flat_ids,
                positions=block_positions.reshape(-1),
                out_cache_loc=cache_locs,
                captured_hidden_states=None,
                input_embeds=input_embeds,
                kv_sync_event=kv_sync_event,
            )

        draft_hidden = logits_output.hidden_states
        if draft_hidden is None:
            raise RuntimeError(
                "Native DFLASH draft model did not return hidden states."
            )
        draft_hidden = draft_hidden.view(bs, self.spec_num_tokens, self.hidden_size)

        next_tokens = self.next_tokens_buf[:bs]
        next_tokens[:, 0] = block_ids[:, 0]
        self._greedy_sample_from_vocab_parallel_head(
            draft_hidden[:, 1:, :].reshape(-1, self.hidden_size),
            out=next_tokens[:, 1:],
        )
        next_tokens.clamp_(min=0)
        return next_tokens

    @nvtx_range("drafter:dflash", color="purple")
    def run(
        self,
        base_ctx: ForwardContext,
        logits_output: LogitsProcessorOutput,
        output_tokens: torch.Tensor,
        accept_lengths: torch.Tensor,
    ) -> torch.Tensor:
        if not hasattr(self, "target_model"):
            raise RuntimeError("DFLASH drafter is not bound to a target model.")

        from tokenspeed.runtime.execution.cuda_graph_wrapper import (
            get_is_cuda_graph_phase,
        )

        decode_only = base_ctx.num_extends == 0
        capturing = (
            torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
        )
        can_overlap = (
            decode_only
            and self._fused_kv_enabled
            and self._kv_aux_stream is not None
            and (capturing or not get_is_cuda_graph_phase())
        )
        if can_overlap:
            return self._run_overlap(
                base_ctx, logits_output, output_tokens, accept_lengths
            )

        # Default sequential path
        self._update_native_cache_from_target(base_ctx, logits_output, accept_lengths)
        bs = base_ctx.bs
        current_tokens = self.block_ids_buf[:bs, 0]
        if base_ctx.num_extends == 0:
            draft_cache_locs = self.draft_out_cache_loc_buf[: bs * self.spec_num_tokens]
            max_draft_prefix = (
                self.req_to_page.shape[1] * self.page_size - self.spec_num_tokens
            )
            dflash_prepare_decode(
                output_tokens=output_tokens,
                accept_lengths=accept_lengths[:bs],
                req_pool_indices=self.input_buffers.req_pool_indices_buf[:bs],
                valid_cache_lengths=self.runtime_states.valid_cache_lengths,
                req_to_pages=self.req_to_page,
                draft_seq_lens=self.draft_seq_lens_buf[:bs],
                block_ids=self.block_ids_buf[:bs],
                block_positions=self.block_positions_buf[:bs],
                out_cache_loc=draft_cache_locs,
                spec_num_tokens=self.spec_num_tokens,
                page_size=self.page_size,
                max_draft_prefix=max_draft_prefix,
            )
            return self._draft_native(current_tokens, prepared=True)

        self._current_tokens_from_output(
            output_tokens,
            accept_lengths,
            base_ctx.num_extends,
            self.spec_num_tokens,
            out=current_tokens,
        )
        return self.draft(current_tokens)

    def _run_overlap(
        self,
        base_ctx: ForwardContext,
        logits_output: LogitsProcessorOutput,
        output_tokens: torch.Tensor,
        accept_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Overlap _update_native_cache_from_target (aux stream) with draft (main).
        Called from run() only when decode-only + fused KV + aux stream are all
        satisfied.
        """
        hidden = logits_output.hidden_states
        if hidden is None and not self._incremental_kv_write_done:
            raise RuntimeError("DFLASH requires target hidden states.")

        bs = base_ctx.bs
        req_pool_indices = self.input_buffers.req_pool_indices_buf[:bs]
        max_draft_prefix = (
            self.req_to_page.shape[1] * self.page_size - self.spec_num_tokens
        )

        current_tokens = self.block_ids_buf[:bs, 0]
        draft_cache_locs = self.draft_out_cache_loc_buf[: bs * self.spec_num_tokens]
        dflash_prepare_decode(
            output_tokens=output_tokens,
            accept_lengths=accept_lengths[:bs],
            req_pool_indices=req_pool_indices,
            valid_cache_lengths=self.runtime_states.valid_cache_lengths,
            req_to_pages=self.req_to_page,
            draft_seq_lens=self.draft_seq_lens_buf[:bs],
            block_ids=self.block_ids_buf[:bs],
            block_positions=self.block_positions_buf[:bs],
            out_cache_loc=draft_cache_locs,
            spec_num_tokens=self.spec_num_tokens,
            page_size=self.page_size,
            max_draft_prefix=max_draft_prefix,
        )

        if not self._incremental_kv_write_done:
            # Fork: aux stream runs full KV write (project + fused GEMM + scatter)
            positions = self.input_buffers.positions_buf[: base_ctx.input_num_tokens]
            cache_locs = self.input_buffers.out_cache_loc_buf[
                : base_ctx.input_num_tokens
            ]
            main_stream = torch.cuda.current_stream()
            self._kv_fork_event.record(main_stream)

            if not (
                torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
            ):
                hidden.record_stream(self._kv_aux_stream)
                positions.record_stream(self._kv_aux_stream)
                cache_locs.record_stream(self._kv_aux_stream)
                if self._fused_kv_proj_workspace is not None:
                    self._fused_kv_proj_workspace.record_stream(self._kv_aux_stream)

            with torch.cuda.stream(self._kv_aux_stream):
                self._kv_aux_stream.wait_event(self._kv_fork_event)
                self._write_native_cache(
                    hidden, positions, cache_locs, decode_only=True
                )
                self._kv_join_event.record(self._kv_aux_stream)

        # Main stream: draft forward overlaps with aux KV write
        return self._draft_native(
            current_tokens, kv_sync_event=self._kv_join_event, prepared=True
        )

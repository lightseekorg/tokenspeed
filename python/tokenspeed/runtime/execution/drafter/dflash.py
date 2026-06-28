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
import torch.nn.functional as F

from tokenspeed.runtime.distributed.comm_ops import all_gather_into_tensor
from tokenspeed.runtime.execution.cache_loc_kernel import compute_out_cache_loc_uniform
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.drafter._dflash_fused_kv import (
    _fused_norm_rope_stacked,
    _fused_norm_rope_stacked_scatter,
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

    def _init_native_buffers(self) -> None:
        if self.input_buffers is None:
            raise ValueError("Native DFLASH requires input buffers.")
        if self.req_to_page is None:
            raise ValueError("Native DFLASH requires req_to_page.")
        if self.attn_backend is None or self.token_to_kv_pool is None:
            raise ValueError("Native DFLASH requires draft attention components.")

        max_bs = self.input_buffers.max_bs
        self.draft_seq_lens_buf = torch.zeros_like(self.input_buffers.seq_lens_buf)
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
        self.draft_extend_prefix_lens_cpu = torch.zeros(
            (max_bs,), dtype=torch.int32, pin_memory=True,
        )
        self.block_offsets = torch.arange(
            self.spec_num_tokens, dtype=torch.int64, device=self.device
        )
        self.block_ids_buf = torch.empty(
            (max_bs, self.spec_num_tokens), dtype=torch.int32, device=self.device
        )
        self.block_positions_buf = torch.empty(
            (max_bs, self.spec_num_tokens), dtype=torch.int64, device=self.device
        )

    def bind_target_model(self, target_model) -> None:
        language_model = getattr(target_model, "language_model", target_model)
        self.target_model = target_model
        self.target_language_model = language_model
        self.embed_tokens = target_model.get_input_embeddings()
        self.lm_head = target_model.lm_head
        self.logits_processor = language_model.logits_processor

    def _greedy_sample_from_vocab_parallel_head(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if not hasattr(self.lm_head, "weight") or not hasattr(
            self.lm_head, "shard_indices"
        ):
            metadata = LogitsMetadata(forward_mode=ForwardMode.DECODE)
            logits = self.logits_processor._get_logits(
                hidden_states, self.lm_head, metadata
            )
            return torch.argmax(logits, dim=-1).to(torch.int32)

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
            return global_ids.to(torch.int32)

        needed = tp_size * chunk_len
        if (
            self._greedy_gather_cap < needed
            or self._greedy_gathered_max is None
            or self._greedy_gathered_ids is None
            or self._greedy_gathered_max.dtype != local_max.dtype
            or self._greedy_gathered_max.device != hidden_states.device
        ):
            self._greedy_gathered_max = torch.empty(
                (needed,), dtype=local_max.dtype, device=hidden_states.device
            )
            self._greedy_gathered_ids = torch.empty(
                (needed,), dtype=global_ids.dtype, device=hidden_states.device
            )
            self._greedy_gather_cap = needed

        gathered_max = self._greedy_gathered_max[:needed]
        gathered_ids = self._greedy_gathered_ids[:needed]
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
        return torch.gather(gathered_ids, 0, best_rank).view(-1).to(torch.int32)

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
        lengths = self.input_buffers.input_lengths_buf[:bs].to(torch.int64)
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
            self._write_native_cache(
                hidden, positions, cache_locs, decode_only=True
            )
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

            self._fused_kv_enabled = True
            logger.info(
                "DFLASH fused KV materialization enabled. "
                "n_layers=%d, num_kv_heads=%d, head_dim=%d",
                n_layers, num_kv_heads, head_dim,
            )
        except Exception as e:
            logger.warning(
                "DFLASH fused KV initialization failed, falling back to sequential: %s",
                e,
            )
            self._fused_kv_enabled = False

    def _ensure_fused_workspace(self, total_ctx: int, dtype: torch.dtype) -> None:
        """Ensure the projection workspace is large enough."""
        if (
            self._fused_kv_workspace_capacity >= total_ctx
            and self._fused_kv_workspace_dtype == dtype
            and self._fused_kv_proj_workspace is not None
        ):
            return
        new_cap = max(1, total_ctx)
        if self._fused_kv_workspace_capacity > 0:
            new_cap = max(new_cap, self._fused_kv_workspace_capacity * 2)
        self._fused_kv_proj_workspace = torch.empty(
            (new_cap, self._fused_kv_n_layers * self._fused_kv_layer_out_dim),
            dtype=dtype,
            device=self.device,
        )
        self._fused_kv_workspace_capacity = new_cap
        self._fused_kv_workspace_dtype = dtype

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
        )

    @staticmethod
    def _current_tokens_from_output(
        output_tokens: torch.Tensor,
        accept_lengths: torch.Tensor,
        num_extends: int,
        spec_num_tokens: int,
    ) -> torch.Tensor:
        bs = accept_lengths.shape[0]
        current = torch.empty((bs,), dtype=torch.int32, device=output_tokens.device)
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
            current[num_extends:] = output_tokens[
                offsets + accept_lengths[num_extends:].to(torch.int64)
            ]
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
        self, current_tokens: torch.Tensor, kv_sync_event: torch.cuda.Event = None
    ) -> torch.Tensor:
        bs = current_tokens.shape[0]
        req_pool_indices = self.input_buffers.req_pool_indices_buf[:bs]
        prefix_lens = self.draft_seq_lens_buf[:bs].clone()
        seq_lens_after = self.draft_seq_lens_buf[:bs]
        seq_lens_after.copy_(prefix_lens + int(self.spec_num_tokens))

        block_ids = self.block_ids_buf[:bs]
        block_ids.fill_(int(self.mask_token_id))
        block_ids[:, 0].copy_(current_tokens.to(torch.int32))
        block_positions = self.block_positions_buf[:bs]
        block_positions.copy_(
            prefix_lens.to(torch.int64).unsqueeze(1) + self.block_offsets
        )

        cache_locs = self.draft_out_cache_loc_buf[: bs * self.spec_num_tokens]
        compute_out_cache_loc_uniform(
            out_cache_loc_ptr=cache_locs,
            req_pool_indices=req_pool_indices,
            uniform_input_length=self.spec_num_tokens,
            cache_start=prefix_lens,
            req_to_pages=self.req_to_page,
            page_size=self.page_size,
        )

        extend_prefix_lens_cpu = self.draft_extend_prefix_lens_cpu[:bs]
        if not (torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()):
            extend_prefix_lens_cpu.copy_(prefix_lens.to(dtype=torch.int32))

        self.attn_backend.init_forward_metadata(
            bs=bs,
            num_extends=bs,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens_after,
            req_to_page=self.req_to_page,
            forward_mode=ForwardMode.EXTEND,
            extend_seq_lens=self.draft_input_lengths_buf[:bs],
            extend_seq_lens_cpu=self.draft_extend_seq_lens_cpu[:bs],
            extend_prefix_lens=prefix_lens,
            extend_prefix_lens_cpu=extend_prefix_lens_cpu,
        )

        ctx = ForwardContext(
            attn_backend=self.attn_backend,
            token_to_kv_pool=self.token_to_kv_pool,
            req_to_page=self.req_to_page,
            bs=bs,
            num_extends=bs,
            input_num_tokens=bs * self.spec_num_tokens,
            forward_mode=ForwardMode.EXTEND,
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

        next_tokens = torch.empty(
            (bs, self.spec_num_tokens), dtype=torch.int32, device=self.device
        )
        next_tokens[:, 0] = current_tokens.to(torch.int32)
        sampled = self._greedy_sample_from_vocab_parallel_head(
            draft_hidden[:, 1:, :].reshape(-1, self.hidden_size)
        )
        next_tokens[:, 1:] = sampled.view(bs, self.spec_num_tokens - 1)
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

        from tokenspeed.runtime.execution.cuda_graph_wrapper import get_is_cuda_graph_phase

        decode_only = base_ctx.num_extends == 0
        capturing = torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
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
        current_tokens = self._current_tokens_from_output(
            output_tokens,
            accept_lengths,
            base_ctx.num_extends,
            self.spec_num_tokens,
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

        Requires decode-only batch and fused KV path. Falls back to sequential
        run() otherwise.
        """
        if not hasattr(self, "target_model"):
            raise RuntimeError("DFLASH drafter is not bound to a target model.")

        decode_only = base_ctx.num_extends == 0
        can_overlap = (
            decode_only
            and self._fused_kv_enabled
            and self._kv_aux_stream is not None
        )
        if not can_overlap:
            return self.run(
                base_ctx, logits_output, output_tokens, accept_lengths
            )

        hidden = logits_output.hidden_states
        if hidden is None:
            raise RuntimeError("DFLASH requires target hidden states.")

        # draft_seq_lens_buf on main stream (draft block prep depends on it)
        bs = base_ctx.bs
        req_pool_indices = self.input_buffers.req_pool_indices_buf[:bs]
        old_lens = self.runtime_states.valid_cache_lengths.index_select(
            0, req_pool_indices
        )
        new_lens = old_lens.to(torch.int32) + accept_lengths[:bs].to(torch.int32)
        max_draft_prefix = self.req_to_page.shape[1] * self.page_size - self.spec_num_tokens
        new_lens.clamp_(max=max_draft_prefix)
        self.draft_seq_lens_buf[:bs].copy_(new_lens)

        # Fork: aux stream runs full KV write (project + fused GEMM + scatter)
        positions = self.input_buffers.positions_buf[: base_ctx.input_num_tokens]
        cache_locs = self.input_buffers.out_cache_loc_buf[: base_ctx.input_num_tokens]
        main_stream = torch.cuda.current_stream()
        self._kv_fork_event.record(main_stream)

        if not (torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()):
            hidden.record_stream(self._kv_aux_stream)
            positions.record_stream(self._kv_aux_stream)
            cache_locs.record_stream(self._kv_aux_stream)
            if self._fused_kv_proj_workspace is not None:
                self._fused_kv_proj_workspace.record_stream(self._kv_aux_stream)

        with torch.cuda.stream(self._kv_aux_stream):
            self._kv_aux_stream.wait_event(self._kv_fork_event)
            self._write_native_cache(hidden, positions, cache_locs, decode_only=True)
            self._kv_join_event.record(self._kv_aux_stream)

        # Main stream: draft block prep overlaps with aux KV write
        current_tokens = self._current_tokens_from_output(
            output_tokens, accept_lengths, 0, self.spec_num_tokens,
        )
        return self._draft_native(current_tokens, kv_sync_event=self._kv_join_event)

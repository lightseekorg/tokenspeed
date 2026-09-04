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

import dataclasses

import torch
from tokenspeed_kernel.ops.attention import (
    dsa_decode,
    dsa_plan,
    dsa_prefill,
)
from tokenspeed_kernel.ops.attention.triton.dsa_topk import (
    workspace_topk_to_global_slots,
)
from tokenspeed_kernel.platform import current_platform

from tokenspeed.runtime.configs.model_config import AttentionArch
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.paged.base import (
    PagedAttentionBackend,
)
from tokenspeed.runtime.layers.attention.backends.paged.mla import MLAAttnBackend
from tokenspeed.runtime.layers.attention.backends.paged.trtllm_mla import (
    TRTLLMMLABackend,
)
from tokenspeed.runtime.layers.attention.backends.support import CudaGraphSupport
from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
from tokenspeed.runtime.layers.attention.configs.dsa import DSAConfig
from tokenspeed.runtime.layers.attention.kernel_page_sizes import (
    DSA_SPARSE_PAGE_SIZE,
)
from tokenspeed.runtime.layers.attention.kpool import KPoolRuntime
from tokenspeed.runtime.layers.attention.registry import register_backend


def _make_dense_leaf(
    config: AttnConfig, spec: DSAConfig, platform, kernel_page_size: int
) -> PagedAttentionBackend:
    # The dense delegate interprets spec.backend_name itself (the MLA leaf's
    # kernel-solution map only knows its own names) — the 'dsa' name that
    # selected THIS wrapper must not leak through.
    dense_spec = dataclasses.replace(spec, backend_name=None)
    if platform.is_nvidia:
        return TRTLLMMLABackend(config, dense_spec, kernel_page_size=kernel_page_size)
    if platform.is_amd:
        return MLAAttnBackend(config, dense_spec, kernel_page_size=kernel_page_size)
    raise RuntimeError(f"DSA backend does not support platform {platform.vendor!r}.")


class DSABackend(PagedAttentionBackend):
    """DSA leaf for sparse MLA attention.

    Dense MLA metadata and dense attention calls are delegated to a platform
    leaf sharing the same kernel page size; the sparse path maps its top-k
    slots through the same page table.
    """

    default_kernel_page_size = DSA_SPARSE_PAGE_SIZE

    # DSA's sparse indexer reads this backend's chunked_prefill_metadata from
    # inside the captured prefill segment, but the prefill graph rebinds only
    # the live ForwardContext at replay — the backend metadata object stays
    # frozen at capture-time (dummy) values. Keep prefills eager.
    cuda_graph_support = CudaGraphSupport(prefill_graph=False)

    def __init__(self, config: AttnConfig, spec: DSAConfig, *, kernel_page_size: int):
        super().__init__(config, spec, kernel_page_size=kernel_page_size)
        platform = current_platform()
        self._dense_backend = _make_dense_leaf(config, spec, platform, kernel_page_size)
        self.index_topk = spec.index_topk
        self.kv_lora_rank = spec.kv_lora_rank
        self.qk_nope_head_dim = spec.qk_nope_head_dim
        self.qk_rope_head_dim = spec.qk_rope_head_dim
        self.v_head_dim = spec.v_head_dim
        self.kv_cache_dim = spec.kv_cache_dim
        self.scaling = spec.scaling
        self.data_type = config.kv_cache_dtype
        self.q_data_type = config.dtype
        self.num_local_heads = spec.num_attention_heads // spec.attn_tp_size
        self._prefill_page_table: torch.Tensor | None = None
        self.kpool_runtime = (
            KPoolRuntime(spec.index_kpool, spec.index_topk)
            if spec.index_kpool is not None
            else None
        )
        if self.kpool_runtime is not None:
            # GLM-5.3-Flash (the one KPool consumer) handles padded prefill
            # replay explicitly in its model code, so the class-level DSA
            # restriction does not apply to it.
            self.cuda_graph_support = CudaGraphSupport(prefill_graph=True)

    def set_request_slots(self, req_pool_indices: torch.Tensor) -> None:
        # KPool's tail state is indexed by request-pool slot, and its
        # per-forward plan must not outlive the metadata build that
        # produced it: the router publishes the slots after every build,
        # which is exactly the reset point.
        if self.kpool_runtime is not None:
            self.kpool_runtime.reset_forward(req_pool_indices)

    def require_kpool_runtime(self) -> KPoolRuntime:
        """Return the configured KPool runtime for sparse pooled indexing."""
        if self.kpool_runtime is None:
            raise RuntimeError("DSA backend was created without KPool configuration")
        return self.kpool_runtime

    def kpool_prefill_page_table(self, num_requests: int) -> torch.Tensor:
        """The kernel-page history rows KPool prefill maps its top-k through."""
        table = self._prefill_page_table
        if table is None and self.chunked_prefill_metadata is not None:
            table = self.chunked_prefill_metadata.page_table
        if table is None:
            raise RuntimeError("DSA KPool prefill requires a full-history page table")
        if num_requests < 0 or table.shape[0] < num_requests:
            raise RuntimeError(
                "DSA KPool prefill page-table row mismatch: "
                f"table={table.shape[0]}, requests={num_requests}"
            )
        return table[:num_requests]

    def kpool_decode_page_table(
        self, row_start: int, num_requests: int
    ) -> torch.Tensor:
        """The kernel-page history rows KPool decode maps its top-k through."""
        metadata = self.forward_decode_metadata
        table = None if metadata is None else metadata.page_table
        row_end = row_start + num_requests
        if (
            table is None
            or row_start < 0
            or num_requests < 0
            or table.shape[0] < row_end
        ):
            rows = None if table is None else table.shape[0]
            raise RuntimeError(
                "DSA KPool decode page-table row mismatch: "
                f"table={rows}, rows=[{row_start}, {row_end})"
            )
        return table[row_start:row_end]

    # ------------------------------------------------------------------
    # Delegation surface
    # ------------------------------------------------------------------

    @property
    def forward_decode_metadata(self):
        return self._dense_backend.forward_decode_metadata

    @property
    def forward_prefill_metadata(self):
        return self._dense_backend.forward_prefill_metadata

    @property
    def chunked_prefill_metadata(self):
        return self._dense_backend.chunked_prefill_metadata

    @property
    def max_num_pages(self) -> int:
        # The dense leaf pads its table width to the fused-kernel block
        # constraint; the router sizes this leaf's tables the same way.
        return self._dense_backend.max_num_pages

    @max_num_pages.setter
    def max_num_pages(self, value: int) -> None:
        del value  # derived from the dense leaf

    @property
    def decode_seq_lens_buffer(self) -> torch.Tensor:
        return self._dense_backend.decode_seq_lens_buffer

    def child_backends(self):
        return (self._dense_backend,)

    def set_cache_pool(self, cache_pool) -> None:
        super().set_cache_pool(cache_pool)
        self._dense_backend.set_cache_pool(cache_pool)

    def register_step_counter(self, step_counter):
        self.step_counter = step_counter
        self._dense_backend.step_counter = step_counter

    def override_num_extends(self, num_extends: int):
        return self._dense_backend.override_num_extends(num_extends)

    def init_cuda_graph_state(self, max_bs: int) -> None:
        self._dense_backend.init_cuda_graph_state(max_bs)

    # Capture is inherited: the leaf default routes through this wrapper's
    # refresh, whose lazy arm builds the piggybacked _dsa_seq_lens_2d /
    # _dsa_plan once per bs on the dense leaf's cached metadata.

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def refresh_decode_metadata(
        self,
        bs: int,
        actual_bs: int,
        seq_lens: torch.Tensor,
        page_table: torch.Tensor,
        *,
        num_extends: int = 0,
        for_graph_replay: bool = False,
    ) -> None:
        self._dense_backend.refresh_decode_metadata(
            bs,
            actual_bs,
            seq_lens,
            page_table,
            num_extends=num_extends,
            for_graph_replay=for_graph_replay,
        )
        metadata = self.forward_decode_metadata
        if getattr(metadata, "_dsa_seq_lens_2d", None) is None:
            # First refresh at a lazily-built bs (no capture ran): allocate the
            # per-token view once; subsequent refreshes update it in place.
            metadata._dsa_seq_lens_2d = (
                seq_lens[:bs]
                .unsqueeze(1)
                .expand(-1, self.spec_num_tokens)
                .reshape(-1, 1)
                .contiguous()
            )
            metadata._dsa_plan = dsa_plan(
                seq_lens_2d=metadata._dsa_seq_lens_2d,
                page_size=self.kernel_page_size,
            )
            return
        metadata._dsa_seq_lens_2d.copy_(
            seq_lens[:bs].unsqueeze(1).expand(-1, self.spec_num_tokens).reshape(-1, 1)
        )
        dsa_plan(
            seq_lens_2d=metadata._dsa_seq_lens_2d,
            page_size=self.kernel_page_size,
            out=metadata._dsa_plan,
        )

    def advance_draft_forward_metadata(self, seq_lens: torch.Tensor) -> None:
        metadata = self.forward_decode_metadata
        if metadata is None or metadata.seq_lens_k is None:
            raise RuntimeError("DSA draft decode metadata was not initialized")
        metadata.seq_lens_k.copy_(seq_lens[: metadata.seq_lens_k.numel()])

        dsa_plan(
            seq_lens_2d=metadata.seq_lens_k.unsqueeze(1),
            page_size=self.kernel_page_size,
            out=metadata._dsa_plan,
        )

    def init_forward_metadata(
        self,
        bs: int,
        num_extends: int,
        seq_lens: torch.Tensor,
        page_table: torch.Tensor,
        forward_mode: ForwardMode,
        *,
        extend_seq_lens: torch.Tensor,
        extend_seq_lens_cpu: torch.Tensor,
        extend_prefix_lens: torch.Tensor,
        extend_prefix_lens_cpu: torch.Tensor,
        extend_with_prefix: bool,
        **kwargs,
    ):
        if not (forward_mode.is_extend_or_mixed() or forward_mode.is_idle()):
            raise RuntimeError(
                "DSA decode metadata goes through refresh_decode_metadata; "
                f"init_forward_metadata only serves extend/mixed ({forward_mode})"
            )
        self._dense_backend.init_forward_metadata(
            bs,
            num_extends,
            seq_lens,
            page_table,
            forward_mode,
            extend_seq_lens=extend_seq_lens,
            extend_seq_lens_cpu=extend_seq_lens_cpu,
            extend_prefix_lens=extend_prefix_lens,
            extend_prefix_lens_cpu=extend_prefix_lens_cpu,
            extend_with_prefix=extend_with_prefix,
            **kwargs,
        )
        # Target mixed batches carry decode rows needing the per-token plan.
        # A draft's plan is rebuilt by the wrapper's refresh_decode_metadata
        # after this init (the unified draft contract).
        if forward_mode.is_mixed() and not self.is_draft:
            metadata = self.forward_decode_metadata
            # Per-token context lengths: the paged-MQA-logits kernel only supports
            # next_n == 1, so each verify token is its own row (bs * spec_num_tokens
            # rows). The per-token causal bound is applied downstream in the top-k.
            # See deep_gemm_dsa_decode_topk.
            metadata._dsa_seq_lens_2d = (
                seq_lens.unsqueeze(1)
                .expand(-1, self.spec_num_tokens)
                .reshape(-1, 1)
                .contiguous()
            )
            if num_extends < bs:
                # Decode rows only: skip the extend requests' per-token block.
                seq_lens_2d = metadata._dsa_seq_lens_2d[
                    num_extends * self.spec_num_tokens :
                ]
            else:
                # The dsa_plan is unused, alias to full-batch seq_lens_2d to
                # generate dsa_plan as a placeholder
                seq_lens_2d = metadata._dsa_seq_lens_2d
            metadata._dsa_plan = dsa_plan(
                seq_lens_2d=seq_lens_2d, page_size=self.kernel_page_size
            )

        self._prefill_page_table = None
        if num_extends > 0 and forward_mode.is_extend_or_mixed():
            cmeta = self._dense_backend.chunked_prefill_metadata
            if cmeta is not None:
                # The sparse indexer's top-k maps through the same kernel page
                # table the extend rows read (DSA's sparse page size equals
                # the leaf's kernel page size by construction).
                self._prefill_page_table = page_table[:num_extends]
                cmeta.page_table = self._prefill_page_table

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def _validate_logit_cap(self, logits_soft_cap: float) -> None:
        if logits_soft_cap and logits_soft_cap > 0:
            raise NotImplementedError(
                "TokenSpeed DSA fused dense attention does not support "
                f"logits_soft_cap={logits_soft_cap}. Sparse DSA kernels must "
                "preserve the capped-score semantics before enabling this model."
            )

    def _validate_dense_context(self, seq_lens: torch.Tensor, bs: int) -> None:
        if seq_lens is None or bs <= 0:
            return
        active_seq_lens = seq_lens[:bs]
        if active_seq_lens.numel() == 0:
            return
        max_seq_len = int(active_seq_lens.max().item())
        if max_seq_len > self.index_topk:
            raise NotImplementedError(
                "TokenSpeed DSA dense attention is exact only when every "
                f"request has seq_len <= index_topk ({self.index_topk}); got "
                f"max seq_len {max_seq_len}. Sparse DSA top-k indices are "
                "required for longer contexts."
            )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool,
        bs: int,
        save_kv_cache: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        # The model drives DSA prefill through forward_extend_chunked /
        # forward_sparse_prefill directly.
        raise NotImplementedError(
            "DSA prefill runs through forward_extend_chunked / forward_sparse_prefill"
        )

    def forward_extend_chunked(
        self,
        q,
        k,
        v,
        scaling,
        logits_soft_cap,
        *,
        cum_seq_lens_q,
        cum_seq_lens_kv,
        max_q_len,
        max_kv_len,
        seq_lens,
        batch_size,
        causal,
        out: torch.Tensor | None = None,
    ):
        self._validate_logit_cap(logits_soft_cap)
        self._validate_dense_context(seq_lens, batch_size)
        return self._dense_backend.forward_extend_chunked(
            q,
            k,
            v,
            scaling,
            logits_soft_cap,
            cum_seq_lens_q=cum_seq_lens_q,
            cum_seq_lens_kv=cum_seq_lens_kv,
            max_q_len=max_q_len,
            max_kv_len=max_kv_len,
            seq_lens=seq_lens,
            batch_size=batch_size,
            causal=causal,
            out=out,
        )

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool,
        bs: int,
        save_kv_cache: bool = True,
        topk_indices: torch.Tensor | None = None,
        topk_lens: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        self._validate_logit_cap(layer.logit_cap)
        if topk_indices is not None:
            return self.forward_sparse_decode(
                q=q,
                k=k,
                v=v,
                layer=layer,
                out_cache_loc=out_cache_loc,
                token_to_kv_pool=token_to_kv_pool,
                bs=bs,
                save_kv_cache=save_kv_cache,
                topk_indices=topk_indices,
                topk_lens=topk_lens,
            )
        metadata = self.forward_decode_metadata
        if metadata is not None and metadata.seq_lens_k is not None:
            num_extends = int(metadata.num_extends or 0)
            self._validate_dense_context(metadata.seq_lens_k[num_extends:], bs)
        return self._dense_backend.forward_decode(
            q=q,
            k=k,
            v=v,
            layer=layer,
            out_cache_loc=out_cache_loc,
            token_to_kv_pool=token_to_kv_pool,
            bs=bs,
            save_kv_cache=save_kv_cache,
            **kwargs,
        )

    def forward_sparse_prefill(
        self,
        *,
        q: torch.Tensor,
        layer,
        token_to_kv_pool,
        page_table: torch.Tensor,
        seq_lens: torch.Tensor,
        kv_seq_lens: torch.Tensor | None = None,
        workspace_indices: torch.Tensor,
        topk_lens: torch.Tensor,
        kv_workspace_slots: torch.Tensor | None = None,
        max_seq_len: int,
    ) -> torch.Tensor:
        if layer.logit_cap and layer.logit_cap > 0:
            self._validate_logit_cap(layer.logit_cap)
        if getattr(token_to_kv_pool, "quant_method", None) == "per_token_head":
            raise RuntimeError(
                "DSA sparse prefill does not support "
                "kv_cache_quant_method='per_token_head' yet."
            )
        if workspace_indices.shape[0] != q.shape[0]:
            raise RuntimeError(
                "DSA sparse prefill metadata token mismatch: "
                f"indices={workspace_indices.shape[0]}, q_tokens={q.shape[0]}"
            )
        if topk_lens.shape[0] != q.shape[0]:
            raise RuntimeError(
                "DSA sparse prefill top-k length mismatch: "
                f"lens={topk_lens.shape[0]}, q_tokens={q.shape[0]}"
            )
        if kv_seq_lens is not None and (
            kv_seq_lens.dim() != 1 or kv_seq_lens.numel() != q.shape[0]
        ):
            raise RuntimeError(
                "DSA sparse prefill physical length mismatch: "
                f"lens={tuple(kv_seq_lens.shape)}, q_tokens={q.shape[0]}"
            )
        if q.shape[0] == 0:
            return q.new_empty((0, layer.tp_q_head_num * layer.v_head_dim))
        # KPool selection can append up to pool_size - 1 visible tail tokens,
        # so its workspace may be wider than the configured pooled top-k.
        if workspace_indices.dim() != 2 or workspace_indices.shape[1] <= 0:
            raise RuntimeError(
                "DSA sparse prefill top-k shape mismatch: "
                f"indices={tuple(workspace_indices.shape)}"
            )
        if kv_workspace_slots is None:
            raise RuntimeError(
                "DSA sparse prefill requires kv_workspace_slots to "
                "map workspace-local top-k rows back to KV cache slots."
            )
        topk_slots = workspace_topk_to_global_slots(
            workspace_indices=workspace_indices,
            kv_workspace_slots=kv_workspace_slots,
        )
        q_view = q.view(q.shape[0], layer.tp_q_head_num, layer.head_dim)
        if self.data_type == torch.float8_e4m3fn and q_view.dtype != self.data_type:
            q_view = q_view.to(self.data_type)
        kv_cache = token_to_kv_pool.get_key_buffer(layer.layer_id)

        k_scale = (
            layer.k_scale_float
            if getattr(layer, "k_scale_float", None) is not None
            else 1.0
        )
        out = dsa_prefill(
            q=q_view,
            kv_cache=kv_cache,
            sparse_kv_cache=None,
            topk_slots=topk_slots,
            topk_lens=topk_lens.to(device=q.device, dtype=torch.int32).contiguous(),
            kv_seq_lens=(
                kv_seq_lens.to(device=q.device, dtype=torch.int32).contiguous()
                if kv_seq_lens is not None
                else None
            ),
            max_seqlen_k=max_seq_len,
            qk_nope_head_dim=self.qk_nope_head_dim,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            softmax_scale=layer.scaling,
            page_size=self.kernel_page_size,
            logit_cap=layer.logit_cap,
            k_scale=k_scale,
        )
        # GLM's sparse-prefill path writes both the latent KV and index_k before
        # entering this method, but bypasses the backend's forward and its
        # normal PD readiness hook. Publish the layer only after the dependent
        # sparse-attention launch has been enqueued, so layerwise transfer cannot
        # observe either cache field before it is ready.
        if self.step_counter is not None:
            self.step_counter.record_cache()
        return out.reshape(-1, layer.tp_q_head_num * layer.v_head_dim)

    def forward_sparse_decode(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool,
        bs: int,
        save_kv_cache: bool,
        topk_indices: torch.Tensor,
        topk_lens: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.kernel_page_size != DSA_SPARSE_PAGE_SIZE:
            raise RuntimeError(
                f"DSA sparse decode requires kernel_page_size="
                f"{DSA_SPARSE_PAGE_SIZE} for "
                f"sparse KV layout, got {self.kernel_page_size}."
            )
        if getattr(token_to_kv_pool, "quant_method", None) == "per_token_head":
            raise RuntimeError(
                "DSA sparse decode does not support "
                "kv_cache_quant_method='per_token_head' yet."
            )
        allow_fp8_query = (
            self.data_type == torch.float8_e4m3fn and q.dtype == torch.float8_e4m3fn
        )
        if q.dtype != torch.bfloat16 and not allow_fp8_query:
            raise RuntimeError(
                "DSA sparse decode requires BF16 query tensors, or FP8 query "
                f"tensors on FP8 KV sparse paths, got {q.dtype}."
            )
        if save_kv_cache:
            assert k is not None
            token_to_kv_pool.set_mla_kv_buffer(
                layer,
                out_cache_loc,
                k[..., : self.kv_lora_rank],
                k[..., self.kv_lora_rank :],
            )

        if topk_indices.dtype != torch.int32:
            topk_indices = topk_indices.to(torch.int32)
        if topk_indices.shape[-1] != self.index_topk and topk_lens is None:
            raise RuntimeError(
                "DSA sparse decode top-k width mismatch: "
                f"indices={topk_indices.shape[-1]}, expected={self.index_topk}"
            )
        num_tokens = q.shape[0]
        # Spec-verify feeds q_len_per_req query rows per request while plain
        # decode and the draft model's own decode steps feed one; derive the
        # width from the actual batch shape (bs is the decode request count)
        # rather than spec_num_tokens, which the draft backend inherits from the
        # shared config.
        if bs > 0 and num_tokens % bs == 0:
            q_len_per_req = num_tokens // bs
        else:
            q_len_per_req = 1
        num_reqs = num_tokens // q_len_per_req
        metadata = self.forward_decode_metadata
        if metadata is None or metadata.seq_lens_k is None:
            raise RuntimeError("DSA sparse decode requires decode metadata.")
        num_extends = int(metadata.num_extends or 0)
        available_reqs = max(0, int(metadata.seq_lens_k.shape[0]) - num_extends)
        if available_reqs < num_reqs:
            if available_reqs <= 0 or q.shape[0] % available_reqs != 0:
                raise RuntimeError(
                    "DSA sparse decode metadata batch mismatch: "
                    f"seq_lens={available_reqs}, requests={num_reqs}, "
                    f"q_tokens={q.shape[0]}."
                )
            num_reqs = available_reqs
            q_len_per_req = q.shape[0] // available_reqs
        seq_lens = metadata.seq_lens_k[num_extends : num_extends + num_reqs]
        if seq_lens.numel() != num_reqs:
            raise RuntimeError(
                "DSA sparse decode metadata batch mismatch: "
                f"seq_lens={seq_lens.numel()}, requests={num_reqs}."
            )
        num_tokens = q.shape[0]
        expected_tokens = num_reqs * int(q_len_per_req)
        if num_tokens != expected_tokens:
            raise RuntimeError(
                "DSA sparse decode token shape mismatch: "
                f"q_tokens={num_tokens}, requests={num_reqs}, "
                f"q_len_per_req={q_len_per_req}."
            )
        if topk_lens is not None:
            if topk_lens.dim() != 1 or topk_lens.numel() != num_tokens:
                raise RuntimeError(
                    "DSA sparse decode top-k length mismatch: "
                    f"lens={tuple(topk_lens.shape)}, q_tokens={num_tokens}."
                )
            topk_lens = topk_lens.to(device=q.device, dtype=torch.int32).contiguous()

        # Physical KV length per query row: verify row t of a request sees
        # seq_len - (width - 1 - t) tokens (the block's own future is masked
        # by the kernel's per-row length, not by top-k selection).
        seq_lens = seq_lens.to(device=q.device, dtype=torch.int32).contiguous()
        if q_len_per_req == 1:
            kv_seq_lens = seq_lens
        else:
            offsets = torch.arange(
                q_len_per_req, device=q.device, dtype=torch.int32
            ) - (q_len_per_req - 1)
            kv_seq_lens = (
                seq_lens.unsqueeze(1).add(offsets).clamp_min(0).reshape(-1).contiguous()
            )

        q_view = q.view(num_tokens, layer.tp_q_head_num, layer.head_dim)
        if self.data_type == torch.float8_e4m3fn:
            q_view = q_view.to(self.data_type)
        kv_cache = token_to_kv_pool.get_key_buffer(layer.layer_id)

        k_scale = (
            layer.k_scale_float
            if getattr(layer, "k_scale_float", None) is not None
            else 1.0
        )
        max_seqlen_k = int(
            getattr(metadata, "max_seq_len_k", 0) or self.max_context_len
        )
        out = dsa_decode(
            q=q_view,
            kv_cache=kv_cache,
            sparse_kv_cache=None,
            topk_slots=topk_indices.view(num_tokens, -1),
            topk_lens=topk_lens,
            max_seqlen_k=max_seqlen_k,
            qk_nope_head_dim=self.qk_nope_head_dim,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            softmax_scale=layer.scaling,
            page_size=self.kernel_page_size,
            q_len_per_req=q_len_per_req,
            kv_seq_lens=kv_seq_lens,
            logit_cap=layer.logit_cap,
            k_scale=k_scale,
        )
        return out.reshape(-1, layer.tp_q_head_num * layer.v_head_dim)


register_backend("dsa", {AttentionArch.DSA}, DSABackend)

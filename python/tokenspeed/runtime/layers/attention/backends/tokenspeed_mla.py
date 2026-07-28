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

"""
CuteDSL MLA attention backend for TokenSpeed scheduling.

Uses CuTe DSL JIT-compiled kernels for MLA decode and prefill on Blackwell SM100 GPUs:
- tokenspeed_mla_decode for decode/verify
- tokenspeed_mla_prefill for prefill
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import triton
from tokenspeed_kernel.ops.attention.tokenspeed_mla import (
    get_num_sm,
    tokenspeed_mla_decode,
    tokenspeed_mla_prefill,
    warmup_compile_prefill,
)

from tokenspeed.runtime.configs.flat_cache_runtime import flat_cache_debug_enabled
from tokenspeed.runtime.configs.model_config import AttentionArch
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend
from tokenspeed.runtime.layers.attention.backends.trtllm_mla import (
    TRTLLM_BLOCK_CONSTRAINT,
    TRTLLMMLAChunkedPrefillMetadata,
)
from tokenspeed.runtime.layers.attention.chunk import (
    build_chunked_prefill_metadata_arrays,
)
from tokenspeed.runtime.layers.attention.configs.mla import MLAConfig
from tokenspeed.runtime.layers.attention.registry import register_backend
from tokenspeed.runtime.utils.env import global_server_args_dict
from tokenspeed.runtime.utils.pdl import pdl_enabled

if TYPE_CHECKING:
    from tokenspeed.runtime.layers.paged_attention import PagedAttention

logger = logging.getLogger(__name__)

# CuteDSL decode workspace. The kernel's own `get_workspace_size` formula is
#   B * H * q_len * split_kv * (D + 1) * (acc_dtype.width // 8)   (bytes)
# and `B * split_kv <= num_SMs`, so a closed-form upper bound (Float32 acc) is
#   num_SMs * H * q_len * (D + 1) * 4
# Buffer is per-device and does NOT need zero-init.
_cutedsl_workspace_buffer: dict[torch.device, torch.Tensor] = {}

# Initial q_len capacity for the per-device decode workspace. The buffer grows
# on demand before each launch, so larger verify/draft batches are supported.
_CUTEDSL_INITIAL_Q_LEN_CAPACITY = 8


def get_cutedsl_workspace_buffer(
    device: torch.device,
    num_heads_per_tp: int,
    kv_lora_rank: int,
    q_len_capacity: int = _CUTEDSL_INITIAL_Q_LEN_CAPACITY,
) -> torch.Tensor:
    """Get or grow the per-device CuteDSL workspace buffer."""
    num_sms = get_num_sm(device)
    required = num_sms * num_heads_per_tp * q_len_capacity * (kv_lora_rank + 1) * 4

    existing = _cutedsl_workspace_buffer.get(device)
    if existing is None or existing.numel() < required:
        _cutedsl_workspace_buffer[device] = torch.empty(
            required, dtype=torch.int8, device=device
        )
    return _cutedsl_workspace_buffer[device]


@dataclass
class CuteDSLMLAPrefillMetadata:
    max_seq_len: int
    cum_seq_lens: torch.Tensor
    seq_lens: torch.Tensor
    # FlatKV only: absolute latent write locations for the extend tokens,
    # flattened in q/k/v token order (positions [prefix, seq) per request).
    flat_out_cache_loc: torch.Tensor | None = None


@dataclass
class CuteDSLMLADecodeMetadata:
    num_extends: int = 0
    block_kv_indices: torch.Tensor | None = None
    max_seq_len_k: int | None = None
    seq_lens_k: torch.Tensor | None = None
    # FlatKV only: absolute latent write locations, one per batch row
    # (position seq_len - 1); decode consumers slice [num_extends:].
    flat_out_cache_loc: torch.Tensor | None = None


class CuteDSLMLABackend(AttentionBackend):
    """CuteDSL MLA attention backend for Blackwell SM100 GPUs.

    Decode uses CuTe DSL JIT-compiled kernels via tokenspeed_mla_decode().
    Prefill uses CuTe DSL FMHA kernel via tokenspeed_mla_prefill().
    """

    _logged_decode = False
    _logged_prefill = False

    # FlatKV contract capability: this backend consumes only history-family
    # (full-attention) tables; state groups belong to the linear sub-backend.
    flat_cache_consumer_families = frozenset({"history"})

    def __init__(self, config: MLAConfig):
        super().__init__(config)

        # Latched the first time flat cache metadata arrives. Once bound, a
        # forward without flat metadata is a hard error: the FlatKV contract
        # forbids falling back to legacy req_to_page metadata.
        self._flat_bound = False
        # Set by the registry before graph capture; see mark_flat_contract.
        self._flat_contract_bound = False

        self.max_context_len = config.context_len
        self.page_size = config.page_size

        # MLA dimensions
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.kv_cache_dim = config.kv_cache_dim
        self.scaling = config.scaling
        self.data_type = config.kv_cache_dtype
        self.q_data_type = config.dtype

        # Workspace buffers — sized from config's num_heads / kv_lora_rank.
        num_heads_per_tp = config.num_attention_heads // config.attn_tp_size
        self.cutedsl_workspace = get_cutedsl_workspace_buffer(
            config.device, num_heads_per_tp, self.kv_lora_rank
        )

        # Pre-compile prefill kernel variants so JIT doesn't run during serving.
        # The backend may be constructed once per attention layer (60x for
        # Kimi-K2.5), but `warmup_compile_prefill` is idempotent: each config
        # is only JIT'd once and cached in a module-global dict.
        # tokenspeed_mla requires --kv-cache-dtype fp8_e4m3, so tokenspeed's
        # FP8 prefill path (deepseek_v3.py:946 `use_fp8_prefill`) is always
        # on and feeds fp8_e4m3fn q/k/v to the kernel — bf16 is unreachable
        # for this backend.
        d_qk = self.qk_nope_head_dim + self.qk_rope_head_dim
        warmup_compile_prefill(
            q_dtype=torch.float8_e4m3fn,
            d_qk=d_qk,
            d_v=self.v_head_dim,
            enable_pdl=pdl_enabled(),
        )

        # Validate page_size
        if self.page_size not in (32, 64):
            raise ValueError(
                f"tokenspeed_mla backend requires page_size 32 or 64, got {self.page_size}"
            )

        # tokenspeed_mla's CuTe DSL kernel only supports fp8_e4m3 KV cache; check
        # at startup so misconfiguration surfaces here, not in the first forward.
        kv_cache_dtype = global_server_args_dict.get("kv_cache_dtype", "auto")
        if kv_cache_dtype != "fp8_e4m3":
            raise NotImplementedError(
                f"tokenspeed_mla backend requires --kv-cache-dtype fp8_e4m3, "
                f"got {kv_cache_dtype!r}."
            )

        self.num_local_heads = num_heads_per_tp

        # Metadata
        self.forward_decode_metadata: CuteDSLMLADecodeMetadata | None = None
        self.forward_prefill_metadata: CuteDSLMLAPrefillMetadata | None = None
        self.decode_cuda_graph_metadata: dict[int, CuteDSLMLADecodeMetadata] = {}
        self.decode_cuda_graph_kv_indices = None
        # FlatKV contract decode graph: persistent write-location buffer whose
        # address the captured graph records; replay refreshes it in place.
        # Allocated in init_cuda_graph_state only when _flat_contract_bound.
        self.decode_cuda_graph_flat_out_cache_loc: torch.Tensor | None = None
        self.chunked_prefill_metadata: TRTLLMMLAChunkedPrefillMetadata | None = None

    def mark_flat_contract(self) -> None:
        """Mark this MLA backend as a Kimi-K3 FlatKV contract sub-backend.

        Called by the registry when the backend is constructed for the
        Kimi-K3 LCM contract path. Enables flat CUDA-graph
        capture/replay with stable full-attention block-table and write-location
        buffers; DeepSeek's shared backend is never marked and keeps the
        non-flat graph path unchanged.
        """
        self._flat_contract_bound = True

    def _calc_padded_blocks(self, max_seq_len: int) -> int:
        """Calculate block count padded to satisfy the fused-kernel constraint."""
        blocks = triton.cdiv(max_seq_len, self.page_size)
        constraint = TRTLLM_BLOCK_CONSTRAINT // self.page_size
        if blocks % constraint != 0:
            blocks = triton.cdiv(blocks, constraint) * constraint
        return blocks

    def _create_block_kv_indices(
        self,
        batch_size: int,
        max_blocks: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        req_to_page: torch.Tensor,
        block_kv_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build page-table from req_to_page using vectorized tensor indexing."""
        if block_kv_indices is None:
            block_kv_indices = torch.zeros(
                (batch_size, max_blocks), dtype=torch.int32, device=self.device
            )

        copy_len = min(max_blocks, req_to_page.shape[1])

        block_kv_indices[:batch_size, :copy_len] = req_to_page[
            req_pool_indices[:batch_size], :copy_len
        ]

        return block_kv_indices

    # ---- FlatKV full-attention-table metadata ----

    def _resolve_flat_full_table(
        self, flat_cache_metadata, flat_cache_forward_op, bs: int
    ) -> tuple[torch.Tensor, int]:
        """Fresh full-attention table + flat page size for this forward.

        Host-side structural validation only (no GPU sync): operation
        freshness, row coverage, and kernel-page divisibility.
        """
        table = flat_cache_metadata.require_full_attention_table(
            active_forward_op=flat_cache_forward_op
        )
        if table.shape[0] < bs:
            raise RuntimeError(
                f"flat full-attention table has {table.shape[0]} rows but the "
                f"batch has {bs} requests"
            )
        flat_page_size = int(flat_cache_metadata.block_size)
        if flat_page_size <= 0 or flat_page_size % self.page_size:
            raise RuntimeError(
                f"flat page size {flat_page_size} is not a positive multiple "
                f"of the kernel page size {self.page_size}"
            )
        if table.stride(0) != table.shape[1] and table.shape[0] > 1:
            # The chunked-prefill kernel derives the row stride from shape[1].
            table = table.contiguous()
        return table, flat_page_size

    def _maybe_debug_check_flat_table(
        self, table: torch.Tensor, seq_lens: torch.Tensor, flat_page_size: int
    ) -> None:
        """TOKENSPEED_FLAT_DEBUG=1 (GPU sync): no -1/null page inside a live
        table range."""
        if (
            not flat_cache_debug_enabled()
            or table.numel() == 0
            or seq_lens.numel() == 0
        ):
            return
        bs = seq_lens.shape[0]
        live_pages = (
            (seq_lens.to(torch.int64) + flat_page_size - 1) // flat_page_size
        ).clamp_max_(table.shape[1])
        columns = torch.arange(table.shape[1], device=table.device)
        live_mask = columns.unsqueeze(0) < live_pages.unsqueeze(1)
        live_entries = table[:bs][live_mask]
        if live_entries.numel() and not bool((live_entries > 0).all().item()):
            raise RuntimeError(
                "flat full-attention table contains -1 or the null page 0 "
                "inside a live range"
            )

    def _maybe_debug_check_flat_write_pages(self, pages: torch.Tensor) -> None:
        """TOKENSPEED_FLAT_DEBUG=1 (GPU sync): latent writes must never land
        in the null page 0 or a -1 hole."""
        if not flat_cache_debug_enabled() or pages.numel() == 0:
            return
        if not bool((pages > 0).all().item()):
            raise RuntimeError(
                "flat MLA write location resolves to the null page 0 or a "
                "-1 table hole"
            )

    def _flat_expand_block_kv_indices(
        self,
        bs: int,
        max_blocks: int,
        flat_table: torch.Tensor,
        flat_page_size: int,
        block_kv_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Expand flat pages into kernel pages for ``block_kv_indices``.

        A flat page holds ``flat_page_size`` consecutive tokens, i.e.
        ``ratio = flat_page_size // self.page_size`` kernel pages; kernel page
        ``page * ratio + k`` covers tokens ``page * flat_page_size +
        [k * page_size, (k + 1) * page_size)`` — the absolute token location
        formula is preserved. -1 table tails clamp to the null page 0, whose
        kernel expansion stays inside the physical null page.
        """
        if block_kv_indices is None:
            block_kv_indices = torch.zeros(
                (bs, max_blocks), dtype=torch.int32, device=self.device
            )
        ratio = flat_page_size // self.page_size
        flat_cols = min(triton.cdiv(max_blocks, ratio), flat_table.shape[1])
        if flat_cols <= 0:
            return block_kv_indices
        expanded = (
            flat_table[:bs, :flat_cols].clamp_min(0).to(torch.int32).unsqueeze(-1)
            * ratio
            + torch.arange(ratio, dtype=torch.int32, device=flat_table.device)
        ).reshape(bs, flat_cols * ratio)
        copy_len = min(max_blocks, flat_cols * ratio)
        block_kv_indices[:bs, :copy_len] = expanded[:, :copy_len]
        return block_kv_indices

    def _flat_decode_out_cache_loc(
        self,
        bs: int,
        seq_lens: torch.Tensor,
        flat_table: torch.Tensor,
        flat_page_size: int,
        out: torch.Tensor | None = None,
        q_len_per_req: int = 1,
    ) -> torch.Tensor:
        """Absolute latent write locations per batch row.

        Plain decode writes one location (position ``seq-1``); speculative
        target-verify writes ``q_len_per_req`` trailing positions
        (``seq-q_len .. seq-1``), flattened request-major to match the query
        layout. ``out`` (CUDA-graph replay): write the locations in place into
        the persistent buffer the graph recorded — same ``data_ptr`` — instead
        of allocating a fresh tensor. No host sync on either path.
        """
        last = (seq_lens[:bs].to(torch.int64) - 1).clamp_min(0)
        if q_len_per_req == 1:
            pos = last.unsqueeze(1)
        else:
            steps = torch.arange(
                1 - q_len_per_req, 1, device=seq_lens.device, dtype=torch.int64
            )
            pos = (last.unsqueeze(1) + steps).clamp_min(0)  # [bs, q_len]
        page_idx = torch.div(pos, flat_page_size, rounding_mode="floor")
        pages = flat_table[:bs].gather(1, page_idx)
        self._maybe_debug_check_flat_write_pages(pages)
        locs = (
            pages.clamp_min(0).to(torch.int64) * flat_page_size + (pos % flat_page_size)
        ).reshape(-1)
        if out is not None:
            out[: bs * q_len_per_req].copy_(locs)
            return out
        return locs

    def _flat_extend_out_cache_loc(
        self,
        flat_table: torch.Tensor,
        extend_prefix_lens_cpu: torch.Tensor,
        extend_seq_lens_cpu: torch.Tensor,
        flat_page_size: int,
    ) -> torch.Tensor:
        """Absolute latent write locations for extend tokens.

        Positions ``[prefix_len, seq_len)`` per request, flattened in q/k/v
        token order. Bounds come from the CPU length mirrors — no GPU sync.
        """
        device = flat_table.device
        chunks: list[torch.Tensor] = []
        pages_for_check: list[torch.Tensor] = []
        for row, (start, num_new) in enumerate(
            zip(extend_prefix_lens_cpu.tolist(), extend_seq_lens_cpu.tolist())
        ):
            start = int(start)
            num_new = int(num_new)
            if num_new <= 0:
                continue
            max_col = (start + num_new - 1) // flat_page_size
            if max_col >= flat_table.shape[1]:
                raise RuntimeError(
                    "flat extend write locs out of table bounds: "
                    f"table {tuple(flat_table.shape)} req={row} prefix={start} "
                    f"new={num_new} flat_page_size={flat_page_size} needs "
                    f"column {max_col}"
                )
            pos = torch.arange(start, start + num_new, dtype=torch.int64, device=device)
            pages = flat_table[row].gather(0, pos // flat_page_size)
            pages_for_check.append(pages)
            chunks.append(pages.to(torch.int64) * flat_page_size + pos % flat_page_size)
        if not chunks:
            return torch.empty(0, dtype=torch.int64, device=device)
        if flat_cache_debug_enabled():
            self._maybe_debug_check_flat_write_pages(torch.cat(pages_for_check))
        return torch.cat(chunks)

    # ---- Metadata initialization ----

    def init_forward_metadata(
        self,
        bs: int,
        num_extends: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode,
        req_to_page: torch.Tensor,
        seq_lens_cpu: torch.Tensor | None = None,
        **kwargs,
    ):
        flat_cache_metadata = kwargs.pop("flat_cache_metadata", None)
        flat_cache_forward_op = kwargs.pop("flat_cache_forward_op", None)
        flat_table = None
        flat_page_size = None
        if flat_cache_metadata is not None and self.is_draft:
            # The MTP draft runs on its own classic paged pool; the target's
            # flat metadata rides the shared forward kwargs — ignore it.
            flat_cache_metadata = None
            flat_cache_forward_op = None
        if flat_cache_metadata is not None:
            self._flat_bound = True
            flat_table, flat_page_size = self._resolve_flat_full_table(
                flat_cache_metadata, flat_cache_forward_op, bs
            )
            self._maybe_debug_check_flat_table(
                flat_table, seq_lens[:bs], flat_page_size
            )
        elif self._flat_bound and bs > 0 and not forward_mode.is_idle():
            # Missing FlatKV metadata must never select the legacy req_to_page
            # path after the backend is bound to the contract.
            raise RuntimeError(
                "tokenspeed_mla is bound to the FlatKV contract but received "
                "no flat cache metadata; refusing the legacy req_to_page path"
            )

        if forward_mode.is_extend_or_mixed():
            self._init_prefill_metadata(
                seq_lens[:num_extends],
                req_pool_indices=req_pool_indices[:num_extends],
                req_to_page=req_to_page,
                extend_prefix_lens=kwargs.pop("extend_prefix_lens"),
                extend_prefix_lens_cpu=kwargs.pop("extend_prefix_lens_cpu"),
                extend_seq_lens=kwargs.pop("extend_seq_lens"),
                extend_seq_lens_cpu=kwargs.pop("extend_seq_lens_cpu"),
                flat_table=flat_table,
                flat_page_size=flat_page_size,
            )
        # Drafter steps 1..N are pure DECODE on full bs regardless of target
        # mode, so under is_draft we also fill decode_metadata under EXTEND
        # so the multi-step loop has metadata. The wrapper pre-writes
        # draft_seq_lens before calling here so `seq_lens` aliases the
        # drafter's live buffer.
        if (
            forward_mode.is_decode()
            or forward_mode.is_mixed()
            or (forward_mode.is_extend() and self.is_draft)
        ):
            # Target-verify decodes q_len tokens per request; the flat write
            # locations must cover every one of them.
            verify_q_len = (
                self.spec_num_tokens
                if (
                    self.spec_num_tokens > 1
                    and not self.is_draft
                    and forward_mode.is_decode()
                )
                else 1
            )
            self._init_decode_metadata(
                bs,
                num_extends,
                req_pool_indices,
                seq_lens,
                req_to_page,
                flat_table=flat_table,
                flat_page_size=flat_page_size,
                q_len_per_req=verify_q_len,
            )

    @contextmanager
    def override_num_extends(self, num_extends: int):
        assert self.forward_decode_metadata is not None
        prev = self.forward_decode_metadata.num_extends
        self.forward_decode_metadata.num_extends = num_extends
        try:
            yield
        finally:
            self.forward_decode_metadata.num_extends = prev

    def select_out_cache_loc(self, layer, out_cache_loc, forward_mode=None):
        """FlatKV write-location hook for model-owned latent KV writes.

        Legacy (non-flat) forwards return the caller's locations untouched —
        byte-for-byte the base-class behavior, so DeepSeek-style MLA models
        are unaffected. Once flat metadata is bound, the group-derived
        locations stored in the current forward's metadata are authoritative
        and the caller's req_to_page-derived locations are discarded.
        """
        if not self._flat_bound:
            return out_cache_loc
        if forward_mode is None or forward_mode.is_idle():
            return out_cache_loc
        if forward_mode.is_decode():
            metadata = self.forward_decode_metadata
            if metadata is None or metadata.flat_out_cache_loc is None:
                raise RuntimeError(
                    "flat MLA decode write locations are missing; "
                    "init_forward_metadata must run with flat cache metadata"
                )
            locs = metadata.flat_out_cache_loc[metadata.num_extends :]
        else:
            metadata = self.forward_prefill_metadata
            if metadata is None or metadata.flat_out_cache_loc is None:
                raise RuntimeError(
                    "flat MLA prefill write locations are missing; "
                    "init_forward_metadata must run with flat cache metadata"
                )
            locs = metadata.flat_out_cache_loc
        if out_cache_loc is not None and locs.shape[0] != out_cache_loc.shape[0]:
            raise RuntimeError(
                f"flat MLA write locations cover {locs.shape[0]} tokens but "
                f"the caller provided {out_cache_loc.shape[0]}"
            )
        return locs

    def _init_decode_metadata(
        self,
        bs: int,
        num_extends: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        req_to_page: torch.Tensor,
        flat_table: torch.Tensor | None = None,
        flat_page_size: int | None = None,
        q_len_per_req: int = 1,
    ):
        max_blocks = self._calc_padded_blocks(self.max_context_len)
        if flat_table is not None:
            # FlatKV path: kernel pages and write locations both derive from
            # the full-attention table; req_to_page is never consulted.
            block_kv_indices = self._flat_expand_block_kv_indices(
                bs, max_blocks, flat_table, flat_page_size
            )
            flat_out_cache_loc = self._flat_decode_out_cache_loc(
                bs, seq_lens, flat_table, flat_page_size, q_len_per_req=q_len_per_req
            )
        else:
            block_kv_indices = self._create_block_kv_indices(
                bs, max_blocks, req_pool_indices, seq_lens, req_to_page
            )
            flat_out_cache_loc = None

        self.forward_decode_metadata = CuteDSLMLADecodeMetadata(
            block_kv_indices=block_kv_indices,
            max_seq_len_k=self.max_context_len,
            seq_lens_k=seq_lens,
            num_extends=num_extends,
            flat_out_cache_loc=flat_out_cache_loc,
        )

    def _init_prefill_metadata(
        self,
        seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor | None = None,
        req_to_page: torch.Tensor | None = None,
        extend_prefix_lens: torch.Tensor | None = None,
        extend_prefix_lens_cpu: torch.Tensor | None = None,
        extend_seq_lens: torch.Tensor | None = None,
        extend_seq_lens_cpu: torch.Tensor | None = None,
        flat_table: torch.Tensor | None = None,
        flat_page_size: int | None = None,
    ):
        # Worst-case bound to avoid GPU->CPU sync from seq_lens.max().item().
        # TODO: track a loose CPU upper bound (advance by chunked_prefill_size /
        # accept_lengths.max(); correct when accurate values land) for tighter
        # kernel-grid sizing without syncing.
        max_seq_len = self.max_context_len
        cum_seq_lens = torch.zeros(
            len(seq_lens) + 1, dtype=torch.int32, device=seq_lens.device
        )
        torch.cumsum(seq_lens, dim=0, out=cum_seq_lens[1:])

        assert (
            seq_lens.dtype == torch.int32
        ), f"seq_lens must be int32, got {seq_lens.dtype}"
        num_extends = extend_seq_lens.shape[0]
        if flat_table is not None:
            flat_out_cache_loc = self._flat_extend_out_cache_loc(
                flat_table[:num_extends],
                extend_prefix_lens_cpu,
                extend_seq_lens_cpu,
                flat_page_size,
            )
        else:
            flat_out_cache_loc = None
        self.forward_prefill_metadata = CuteDSLMLAPrefillMetadata(
            max_seq_len=max_seq_len,
            cum_seq_lens=cum_seq_lens,
            seq_lens=seq_lens,
            flat_out_cache_loc=flat_out_cache_loc,
        )
        cum_extend_seq_lens = torch.zeros(
            num_extends + 1, device=self.device, dtype=torch.int32
        )
        torch.cumsum(extend_seq_lens, dim=0, out=cum_extend_seq_lens[1:])
        max_extend_seq_len = extend_seq_lens_cpu.max().item()
        if flat_table is not None:
            # FlatKV path: chunked history reads gather from the
            # full-attention table (rows are batch-ordered, prefill rows
            # first), at the flat page granularity — the produced KV slots
            # are the same absolute token locations either way.
            chunk_req_to_page = flat_table[:num_extends]
            chunk_req_pool_indices = torch.arange(
                num_extends, dtype=torch.int64, device=flat_table.device
            )
            chunk_page_size = flat_page_size
        else:
            chunk_req_to_page = req_to_page
            chunk_req_pool_indices = req_pool_indices
            chunk_page_size = self.page_size
        (
            chunked_loop_num,
            chunk_kv_indices_list,
            chunked_seq_len,
            cu_chunked_seq_len,
            max_chunk_len_per_loop,
        ) = build_chunked_prefill_metadata_arrays(
            extend_prefix_lens,
            extend_prefix_lens_cpu,
            chunk_req_to_page,
            chunk_req_pool_indices,
            chunk_page_size,
        )
        self.chunked_prefill_metadata = TRTLLMMLAChunkedPrefillMetadata(
            extend_prefix_lens=extend_prefix_lens,
            extend_prefix_lens_cpu=extend_prefix_lens_cpu,
            extend_seq_lens=extend_seq_lens,
            extend_seq_lens_cpu=extend_seq_lens_cpu,
            req_pool_indices=req_pool_indices,
            cum_extend_seq_lens=cum_extend_seq_lens,
            max_extend_seq_len=max_extend_seq_len,
            chunked_loop_num=chunked_loop_num,
            chunk_kv_indices_list=chunk_kv_indices_list,
            chunked_seq_len=chunked_seq_len,
            cu_chunked_seq_len=cu_chunked_seq_len,
            max_chunk_len_per_loop=max_chunk_len_per_loop,
        )

    # ---- CUDA Graph ----

    def init_cuda_graph_state(self, max_bs: int, seq_lens_buf: torch.Tensor):
        assert (
            seq_lens_buf.dtype == torch.int32
            and seq_lens_buf.dim() == 1
            and seq_lens_buf.shape[0] >= max_bs
        ), (
            f"seq_lens_buf must be int32 with shape[0] >= {max_bs}, "
            f"got {seq_lens_buf.dtype} {tuple(seq_lens_buf.shape)}"
        )
        # Alias controller's seq_lens_buf — backend never mutates it.
        self.cuda_graph_seq_lens_buf = seq_lens_buf
        max_blocks = self._calc_padded_blocks(self.max_context_len)
        self.decode_cuda_graph_kv_indices = torch.zeros(
            (max_bs, max_blocks), dtype=torch.int32, device=self.device
        )
        if self._flat_contract_bound:
            # The FlatKV decode graph uses one absolute latent write slot per
            # row, refreshed in place from the current full-attention table.
            # Allocate the stable-address buffer outside the graph pool, like
            # the KV indices.
            self.decode_cuda_graph_flat_out_cache_loc = torch.zeros(
                max_bs * max(1, self.spec_num_tokens),
                dtype=torch.int64,
                device=self.device,
            )
        else:
            self.decode_cuda_graph_flat_out_cache_loc = None

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode,
        flat_cache_group_ids: tuple[str, ...] = (),
        **kwargs,
    ):
        # Structural gate: the contract sub-backend (or a wrapper passing flat
        # group ids) takes the FlatKV capture path; DeepSeek's unmarked
        # backend, which never sees either signal, keeps the non-flat path.
        flat = bool(flat_cache_group_ids) or self._flat_contract_bound
        if flat and self.is_draft:
            raise NotImplementedError(
                "tokenspeed_mla draft worker does not take the FlatKV path"
            )
        if forward_mode.is_extend_or_mixed():
            raise NotImplementedError(
                f"tokenspeed_mla CUDA graph capture not supported for {forward_mode}"
            )

        max_blocks = self._calc_padded_blocks(self.max_context_len)
        block_kv_indices = self.decode_cuda_graph_kv_indices[:bs, :max_blocks]

        if flat:
            # The captured FlatKV graph reads the kernel page table and latent
            # write locations from persistent buffers; replay refreshes both
            # in place from the current full-attention table. Latch _flat_bound
            # so the recorded forward_decode takes the flat write-location
            # branch (select_out_cache_loc), matching the eager path.
            self._flat_bound = True
            if self.decode_cuda_graph_flat_out_cache_loc is None:
                raise RuntimeError(
                    "tokenspeed_mla FlatKV graph capture: the flat write-loc "
                    "buffer is not allocated; init_cuda_graph_state ran before "
                    "the backend was marked as the FlatKV contract sub-backend"
                )
            # Placeholders resolve to the null page 0 until the first replay.
            block_kv_indices.zero_()
            capture_q_len = (
                self.spec_num_tokens
                if (self.spec_num_tokens > 1 and not self.is_draft)
                else 1
            )
            flat_out_cache_loc = self.decode_cuda_graph_flat_out_cache_loc[
                : bs * capture_q_len
            ]
            flat_out_cache_loc.zero_()
            metadata = CuteDSLMLADecodeMetadata(
                block_kv_indices=block_kv_indices,
                max_seq_len_k=self.max_context_len,
                seq_lens_k=self.cuda_graph_seq_lens_buf[:bs],
                num_extends=0,
                flat_out_cache_loc=flat_out_cache_loc,
            )
        else:
            metadata = CuteDSLMLADecodeMetadata(
                block_kv_indices=block_kv_indices,
                max_seq_len_k=self.max_context_len,
                seq_lens_k=self.cuda_graph_seq_lens_buf[:bs],
                num_extends=0,
            )
        self.decode_cuda_graph_metadata[bs] = metadata
        self.forward_decode_metadata = metadata

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode = None,
        req_to_page: torch.Tensor = None,
        **kwargs,
    ):
        flat_cache_metadata = kwargs.get("flat_cache_metadata")
        if forward_mode is not None and forward_mode.is_extend_or_mixed():
            raise NotImplementedError(
                f"tokenspeed_mla CUDA graph replay not supported for {forward_mode}"
            )

        metadata = self.decode_cuda_graph_metadata[bs]
        # The captured metadata is the source of truth: a flat write-loc buffer
        # exists iff this bs was captured on the FlatKV path.
        flat = metadata.flat_out_cache_loc is not None
        if flat and self.is_draft:
            raise NotImplementedError(
                "tokenspeed_mla draft worker does not take the FlatKV path"
            )

        if flat:
            self._flat_replay_refresh_decode(
                bs,
                seq_lens,
                metadata,
                flat_cache_metadata,
                kwargs.get("flat_cache_forward_op"),
            )
            self.forward_decode_metadata = metadata
            return

        # seq_lens_k aliases seq_lens_buf; only block indices need refresh.
        # When the buffer is aliased to a peer backend (e.g. drafter aliasing
        # the target's kv_indices), the peer's replay has already populated it
        # with identical content.
        if req_to_page is not None and not self._block_table_aliased:
            self._create_block_kv_indices(
                bs,
                metadata.block_kv_indices.shape[1],
                req_pool_indices[:bs],
                seq_lens[:bs],
                req_to_page,
                metadata.block_kv_indices,
            )

        self.forward_decode_metadata = metadata

    def _flat_replay_refresh_decode(
        self,
        bs: int,
        seq_lens: torch.Tensor,
        metadata: "CuteDSLMLADecodeMetadata",
        flat_cache_metadata,
        flat_cache_forward_op,
    ) -> None:
        """Refresh captured flat decode buffers from this replay's
        full-attention table.

        ``bs`` is the padded batch size. The operation-bound metadata carries
        one table row per REAL request; the trailing padded rows have no
        request and resolve to the null page 0 so the captured kernels
        dereference the permanently-zero dummy page.
        The refresh writes into the SAME persistent buffers the graph
        recorded (``metadata.block_kv_indices`` / ``metadata.flat_out_cache_loc``)
        — stable addresses, no reallocation, no host sync.
        """
        if metadata.flat_out_cache_loc is None:
            raise RuntimeError(
                "tokenspeed_mla FlatKV graph replay: captured decode metadata "
                "has no flat write-location buffer"
            )
        # Target verify graphs are captured with q_len write locations per
        # request (request-major flattened); mirror the capture-time width.
        replay_q_len = (
            self.spec_num_tokens
            if (self.spec_num_tokens > 1 and not self.is_draft)
            else 1
        )
        max_blocks = metadata.block_kv_indices.shape[1]
        real_bs = 0
        if flat_cache_metadata is not None:
            # Row coverage is not required here (bs is the padded batch size
            # and the table carries one row per REAL request), so pass 0.
            table, flat_page_size = self._resolve_flat_full_table(
                flat_cache_metadata, flat_cache_forward_op, 0
            )
            real_bs = min(int(table.shape[0]), bs)
            if real_bs > 0:
                self._flat_expand_block_kv_indices(
                    real_bs,
                    max_blocks,
                    table,
                    flat_page_size,
                    metadata.block_kv_indices,
                )
                self._flat_decode_out_cache_loc(
                    real_bs,
                    seq_lens,
                    table,
                    flat_page_size,
                    out=metadata.flat_out_cache_loc[: real_bs * replay_q_len],
                    q_len_per_req=replay_q_len,
                )
        # Padded (and bs==0 idle) rows: null page 0 for both the kernel page
        # table and the write location, so they never touch a live page. Note
        # these rows DO scribble latent slot 0 of page 0 (bytes [0, 576) of
        # each aliased slab); that range stays inside the aliased KDA
        # conv_state component, which is never consumed from page 0 — the
        # recurrent_state bytes read for fresh requests sit above it and
        # remain zero.
        if real_bs < bs:
            metadata.block_kv_indices[real_bs:bs].zero_()
            metadata.flat_out_cache_loc[
                real_bs * replay_q_len : bs * replay_q_len
            ].zero_()

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    # ---- Forward: Decode ----

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: PagedAttention,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool,
        bs: int,
        save_kv_cache: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        # q is whole Q [T, H, head_dim]; k is whole latent [T, 1, head_dim].
        if save_kv_cache:
            assert k is not None
            if self._flat_bound:
                # FlatKV: write locations derive from the full-attention
                # table, never from the caller's req_to_page-derived locs.
                out_cache_loc = self.select_out_cache_loc(
                    layer, out_cache_loc, ForwardMode.DECODE
                )
            token_to_kv_pool.set_mla_kv_buffer(
                layer,
                out_cache_loc,
                k[..., : self.kv_lora_rank],
                k[..., self.kv_lora_rank :],
            )

        metadata = self.forward_decode_metadata
        num_extends = metadata.num_extends
        q_len_per_req = q.shape[0] // bs
        query = q.view(bs, q_len_per_req, layer.tp_q_head_num, layer.head_dim)

        softmax_scale = layer.scaling
        if self.data_type == torch.float8_e4m3fn:
            query = query.to(self.data_type)
            k_scale = (
                layer.k_scale_float
                if getattr(layer, "k_scale_float", None) is not None
                else 1.0
            )
            softmax_scale = k_scale * layer.scaling

        # Prepare KV cache: [num_pages, page_size, kv_cache_dim] (3D for CuteDSL)
        k_cache = token_to_kv_pool.get_key_buffer(layer.layer_id)
        if self.data_type != k_cache.dtype:
            k_cache = k_cache.to(self.data_type)
        kv_cache = k_cache.view(-1, self.page_size, self.kv_cache_dim)

        if not CuteDSLMLABackend._logged_decode:
            logger.info(
                "CuteDSL MLA decode kernel invoked (tokenspeed_mla_decode, query_dtype=%s, kv_dtype=%s)",
                query.dtype,
                kv_cache.dtype,
            )
            CuteDSLMLABackend._logged_decode = True

        self.cutedsl_workspace = get_cutedsl_workspace_buffer(
            query.device, layer.tp_q_head_num, self.kv_lora_rank, query.shape[1]
        )

        raw_out = tokenspeed_mla_decode(
            query=query,
            kv_cache=kv_cache,
            workspace_buffer=self.cutedsl_workspace,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            block_tables=metadata.block_kv_indices[num_extends:],
            seq_lens=metadata.seq_lens_k[num_extends:],
            max_seq_len=metadata.max_seq_len_k,
            softmax_scale=softmax_scale,
            enable_pdl=pdl_enabled(),
        )

        return raw_out.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    # ---- Forward: Extend/Prefill ----

    def forward_extend_chunked(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
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
        if causal:
            step_counter = getattr(self, "step_counter", None)
            if step_counter is not None:
                step_counter.record_cache()

        head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        # The CuteDSL FMHA prefill kernel assumes packed (contiguous) Q/K/V; its
        # TMA descriptors ignore input strides. On the BF16 (NoPE, e.g. Kimi-K3)
        # path V arrives as a non-contiguous slice ``kv[..., qk_nope:]`` of the
        # fused kv_b_proj output (its stride skips the interleaved k_nope block),
        # so without this the kernel reads interleaved garbage and produces an
        # attention output orthogonal to the correct result. Force contiguity on
        # all three; Q/K are already contiguous so ``.contiguous()`` is a no-op.
        q = q.reshape(-1, self.num_local_heads, head_dim).contiguous()
        k = k.reshape(-1, self.num_local_heads, head_dim).contiguous()
        v = v.reshape(-1, self.num_local_heads, self.v_head_dim).contiguous()

        # CuteDSL FMHA MLA: if Q is FP8, ensure K/V match. `.to()` is a no-op
        # when the source dtype already matches.
        if q.dtype == torch.float8_e4m3fn:
            k = k.to(torch.float8_e4m3fn)
            v = v.to(torch.float8_e4m3fn)

        if not CuteDSLMLABackend._logged_prefill:
            logger.info("CuteDSL MLA prefill kernel invoked (tokenspeed_mla_prefill)")
            CuteDSLMLABackend._logged_prefill = True

        result = tokenspeed_mla_prefill(
            query=q,
            key=k,
            value=v,
            seq_lens=seq_lens,
            cum_seq_lens=cum_seq_lens_kv,
            max_seq_len=max_kv_len,
            batch_size=batch_size,
            softmax_scale=scaling,
            is_causal=causal,
            return_lse=True,
            cum_seq_lens_q=cum_seq_lens_q,
            max_seq_len_q=max_q_len,
            enable_pdl=pdl_enabled(),
            out=out,
        )

        if isinstance(result, tuple):
            out, lse = result[0], result[1]
        else:
            out, lse = result, None

        if out.dtype != self.q_data_type:
            out = out.to(self.q_data_type)

        return out, lse


register_backend("tokenspeed_mla", {AttentionArch.MLA}, CuteDSLMLABackend)

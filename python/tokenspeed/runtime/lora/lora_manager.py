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

"""LoRA adapter weight manager (segment-grouped Triton path).

Adapted from sglang/Punica's S-LoRA design.

Memory layout
-------------
For each layer the manager owns:

* ``qkv_A_buffers[layer]``: ``(n_slots, 3 * max_rank, hidden)`` — fused
  q_proj/k_proj/v_proj A matrices, stack-major (q first, then k, then v).
* ``qkv_B_buffers[layer]``: ``(n_slots, q_per_tp + 2 * kv_per_tp, max_rank)``
  — fused output-side, ``[q_per_tp | kv_per_tp | kv_per_tp]`` along dim 1.
* ``o_A_buffers[layer]``:   ``(n_slots, max_rank, in_per_tp)`` — row-parallel
  A, sharded along input dim.
* ``o_B_buffers[layer]``:   ``(n_slots, hidden, max_rank)`` — full B.

No-LoRA requests use ``NO_LORA_SLOT`` (-1), matching vLLM's convention.
Real adapters occupy slots ``0 .. max_loras - 1``.

Tensor parallelism
------------------
* QKV is column-parallel: A is full, B is sharded along output dim
  (``q_per_tp + 2 * kv_per_tp``).  No collective inside the LoRA path.
* O is row-parallel: A is sharded along input dim, B is full.  The host
  module (qwen3 ``o_proj``) runs with ``reduce_results=False`` and has its
  partial sum all-reduced downstream by ``post_attention_layernorm``; the
  LoRA delta rides that same reduction (full ``B @ lora_a`` is added to the
  partial output and the downstream reduce sums it ``tp_size`` times — see
  ``apply_o_lora`` for the resulting numerical caveat).
"""

from __future__ import annotations

import os
from collections import OrderedDict

import torch
from tokenspeed_kernel.ops.lora.triton import (
    lora_expand_fwd,
    lora_expand_grouped_v2_fwd,
    lora_expand_prefill_fwd,
    lora_gate_up_expand_fwd,
    lora_qkv_expand_fwd,
    lora_shrink_fwd,
    lora_shrink_prefill_fwd,
)

from tokenspeed.runtime.lora.adapter_io import (
    LORA_HEAD_LAYER_ID,
    PEFT_HEAD_MODULE,
    PEFT_MODULES,
    read_adapter_scaling,
    resolve_adapter_weight_path,
)
from tokenspeed.runtime.lora.lora_batch import (
    NO_LORA_SLOT,
    LoraBatchInfo,
    build_decode_lora_groups,
)
from tokenspeed.runtime.lora.lora_buffers import LORA_BUFFER_GROUPS, LoraWeightBuffers
from tokenspeed.runtime.lora.lora_cache import LoraCpuCache
from tokenspeed.runtime.lora.moe_lora import MoeLoraBuffers, MoeLoraContext
from tokenspeed.runtime.utils import get_colorful_logger

# Segments longer than this use the prefill (chunked-SGMV) expand kernel,
# which specialises strides and loop counts at compile time.  Shorter
# segments (decode) use the decode-tuned kernels.  Threshold chosen from
# benchmarks: chunked-SGMV wins above ~32 tokens/segment at rank ≥ 64.
_CHUNKED_THRESHOLD = 32

# With max_group_size-based grid, the kernel degenerates to the segmented
# layout when every group has 1 token (n_unique = n), so no threshold is
# needed for correctness.  Keep a minimum of 1 (always use grpv2).
_TRITON_GROUPED_DECODE_MIN_GROUP_SIZE = 1

logger = get_colorful_logger(__name__)


# ── Manager ─────────────────────────────────────────────────────────────────


def _use_triton_grouped_decode(bi: LoraBatchInfo) -> bool:
    """Return whether grouped Triton decode expand should beat basic decode."""
    return (
        bi.single_lora_slot == NO_LORA_SLOT
        and bi.num_groups > 0
        and bi.bs // bi.num_groups >= _TRITON_GROUPED_DECODE_MIN_GROUP_SIZE
    )


class LoraManager:
    """Owns GPU-resident LoRA weights and dispatches the segmented-GEMM path.

    Public surface (used by the model + executor):

    * :meth:`load_adapter` / :meth:`unload_adapter` — adapter lifecycle.
    * :attr:`batch_info` — persistent :class:`LoraBatchInfo` whose tensor
      pointers are stable across forward steps (so they can be baked into
      the captured CUDA graph).
    * :meth:`prepare_loras` — fill the persistent batch_info for one step.
    * :meth:`apply_qkv_lora` / :meth:`apply_o_lora` — Triton-backed deltas.
    """

    def __init__(
        self,
        model_config,
        max_loras: int,
        max_lora_rank: int,
        max_num_tokens: int,
        dtype: torch.dtype,
        device: torch.device,
        tp_rank: int = 0,
        tp_size: int = 1,
        tp_group=None,
        max_loras_cpu: int | None = None,
        lora_buffer_groups: set[str] | frozenset[str] = LORA_BUFFER_GROUPS,
        lora_moe_compressed_shared_outer: bool = False,
    ) -> None:
        self.max_loras = max_loras
        self.max_lora_rank = max_lora_rank
        self.max_num_tokens = max_num_tokens
        self.dtype = dtype
        self.device = device
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.tp_group = tp_group
        unknown_groups = set(lora_buffer_groups) - LORA_BUFFER_GROUPS
        if unknown_groups:
            raise ValueError(f"Unknown LoRA buffer groups: {sorted(unknown_groups)}")
        self.lora_buffer_groups = frozenset(lora_buffer_groups)
        self.enable_attn_lora = "attn" in self.lora_buffer_groups
        self.enable_mlp_lora = "mlp" in self.lora_buffer_groups
        self.enable_moe_lora = "moe" in self.lora_buffer_groups
        self.enable_head_lora = "lm_head" in self.lora_buffer_groups
        self.lora_moe_compressed_shared_outer = lora_moe_compressed_shared_outer
        # Tier-2 CPU cache cap. Defaults to 4× the GPU pool so adapter
        # spill-out to disk is rare in steady state.
        self.max_loras_cpu: int = (
            max_loras_cpu if max_loras_cpu is not None else 4 * max_loras
        )
        if self.max_loras_cpu < max_loras:
            raise ValueError(
                f"max_loras_cpu ({self.max_loras_cpu}) must be ≥ "
                f"max_loras ({max_loras}); GPU-resident adapters live in "
                "the CPU pool too."
            )

        self.n_layers: int = model_config.num_hidden_layers
        hidden: int = model_config.hidden_size
        n_heads: int = model_config.num_attention_heads
        n_kv: int = model_config.num_key_value_heads
        # Use the model's explicit head_dim when available (some architectures like
        # Qwen3.5 decouple head_dim from hidden/n_heads, e.g. hidden=2048, n_heads=16
        # but head_dim=256).
        head_dim: int = getattr(model_config, "head_dim", None) or (hidden // n_heads)
        # attn_output_gate doubles the Q projection size (2× heads in qkv_proj).
        # The o_proj input is n_heads × head_dim (no doubling).
        q_multiplier: int = 2 if getattr(model_config, "attn_output_gate", False) else 1
        q_size_base: int = (n_heads // tp_size) * head_dim

        self.q_size_per_tp: int = q_multiplier * q_size_base
        self.kv_size_per_tp: int = max(1, n_kv // tp_size) * head_dim
        self.o_in_per_tp: int = q_size_base  # o_proj reads un-gated heads
        self.hidden_size: int = hidden

        from tokenspeed.runtime.layers.vocab_parallel_embedding import pad_vocab_size

        vocab_size: int = model_config.vocab_size
        self.vocab_per_tp: int = pad_vocab_size(vocab_size) // tp_size

        # Qwen3MLP is TP-aware: ``gate_up_proj`` is column-parallel (each rank
        # holds ``intermediate_size // tp_size`` output cols) and ``down_proj``
        # is row-parallel (each rank holds ``intermediate_size // tp_size``
        # input cols).  The LoRA deltas ride the partial outputs of those base
        # linears, and the existing downstream all-reduce sums per-rank
        # partials — see ``apply_down_lora``/``apply_gate_up_lora``.
        self.intermediate_size: int = getattr(
            model_config, "intermediate_size", 4 * hidden
        )
        self.intermediate_per_tp: int = self.intermediate_size // self.tp_size
        self.moe_intermediate_size: int = getattr(
            model_config, "moe_intermediate_size", self.intermediate_size
        )
        self.moe_intermediate_per_tp: int = self.moe_intermediate_size // self.tp_size
        self.num_experts: int = int(
            getattr(
                model_config,
                "num_experts",
                getattr(
                    model_config,
                    "num_local_experts",
                    getattr(model_config, "n_routed_experts", 0),
                ),
            )
        )

        # CPU-side flag: True when at least one segment in the current
        # batch_info uses a real adapter. CudaGraphWrapper
        # reads this to pick the with-LoRA vs no-LoRA captured graph.
        self.has_active_lora: bool = False

        # ── Tier 1: GPU pool ─────────────────────────────────────────────
        # Real adapters take slots 0 .. max_loras - 1. Base/no-LoRA requests
        # use NO_LORA_SLOT in batch metadata and do not consume a GPU slot.
        self._n_slots: int = max_loras
        self._slot_to_name: list[str | None] = [None] * self._n_slots
        self._name_to_slot: dict[str, int] = {}
        self._gpu_lru: OrderedDict[str, None] = OrderedDict()  # alias of _lru

        # Mid-decode unload safety: track which adapter names were active in
        # the most recent prepare_loras call, and defer GPU weight eviction for
        # any adapter that is unloaded while still in use.
        self._active_names: set[str] = set()
        self._pending_eviction: set[tuple[str, int]] = (
            set()
        )  # (name, lora_id) to evict when safe

        # ── Tier 2: pinned CPU pool ─────────────────────────────────────
        # ``_cpu_cache[name]`` holds parsed weights in pinned host memory.
        # ``_cpu_lru`` tracks LRU order for CPU eviction back to disk.  An
        # adapter is "CPU-resident" iff its name is in ``_cpu_cache``.
        # GPU-resident adapters are also kept in ``_cpu_cache`` (we pay
        # the host RAM cost once; reload to GPU is cheap and re-evicting
        # GPU then re-promoting only needs an H2D copy, not a disk read).
        self._name_to_id: dict[str, int] = {}
        self._id_to_name: dict[int, str] = {}
        self._next_id: int = 1

        # ── Tier 2/3: CPU pinned pool + disk source of truth ─────────────
        self._cpu_store = LoraCpuCache(
            capacity=self.max_loras_cpu,
            is_gpu_resident=lambda name: name in self._name_to_slot,
        )
        # Compatibility aliases for existing tests/debug tooling.
        self._cpu_cache = self._cpu_store.cache
        self._cpu_lru = self._cpu_store.lru
        self._adapter_paths = self._cpu_store.adapter_paths
        self._pending_loads = self._cpu_store.pending_loads

        # Per-slot rank + scaling for real adapter slots only.
        self._lora_ranks: torch.Tensor = torch.zeros(
            self._n_slots, dtype=torch.int32, device=device
        )
        self._slot_ranks: list[int] = [0] * self._n_slots
        self._slot_scalings: list[float] = [0.0] * self._n_slots
        self._scalings: torch.Tensor = torch.zeros(
            self._n_slots, dtype=torch.float32, device=device
        )

        # ── Persistent batch_info ──────────────────────────────────────────
        # All tensors are sized for the worst case so their pointers are
        # stable across forward steps; per-step updates are in-place.
        # ``num_segments`` may equal ``bs`` (one segment per token in the
        # current path — no sort-by-adapter yet).
        self._batch_info = LoraBatchInfo(
            bs=0,
            num_segments=0,
            max_len=0,
            seg_lens=torch.zeros(max_num_tokens, dtype=torch.int32, device=device),
            seg_indptr=torch.zeros(
                max_num_tokens + 1, dtype=torch.int32, device=device
            ),
            weight_indices=torch.full(
                (max_num_tokens,), NO_LORA_SLOT, dtype=torch.int32, device=device
            ),
            lora_ranks=self._lora_ranks,
            scalings=self._scalings,
            permutation=None,
        )

        # CPU staging buffers (pinned) for the per-step H2D copy.
        self._seg_lens_cpu = torch.zeros(
            max_num_tokens, dtype=torch.int32, pin_memory=True
        )
        self._weight_indices_cpu = torch.full(
            (max_num_tokens,), NO_LORA_SLOT, dtype=torch.int32, pin_memory=True
        )
        # Adapter-group buffers for the decode grouped expand kernel.
        # Computed on CPU in prepare_loras (no GPU sync) and transferred
        # non-blocking.  Using stable GPU addresses so decode CUDA graphs
        # can capture the pointers; num_groups on axis=1 changes per step
        # so the graph grid must be re-evaluated outside the captured region.
        _mg = self._n_slots  # upper bound: one group per loaded adapter
        self._sort_order_cpu = torch.zeros(
            max_num_tokens, dtype=torch.int64, pin_memory=True
        )
        self._group_slots_cpu = torch.zeros(_mg, dtype=torch.int32, pin_memory=True)
        self._group_starts_cpu = torch.zeros(_mg, dtype=torch.int32, pin_memory=True)
        self._group_sizes_cpu = torch.zeros(_mg, dtype=torch.int32, pin_memory=True)
        self._sort_order_buf = torch.zeros(
            max_num_tokens, dtype=torch.int64, device=device
        )
        self._group_slots_buf = torch.zeros(_mg, dtype=torch.int32, device=device)
        self._group_starts_buf = torch.zeros(_mg, dtype=torch.int32, device=device)
        self._group_sizes_buf = torch.zeros(_mg, dtype=torch.int32, device=device)

        # ── GPU weight buffers ─────────────────────────────────────────────
        # Attention:
        #   qkv_A_buffers: (n_slots, 3 * max_rank, hidden) — stacked q/k/v A.
        #   qkv_B_buffers: (n_slots, q_per_tp + 2 * kv_per_tp, max_rank).
        #   o_A_buffers:   (n_slots, max_rank, o_in_per_tp).
        #   o_B_buffers:   (n_slots, hidden, max_rank).
        # MLP (TP-aware, mirrors qwen3 ``Qwen3MLP``):
        #   gate_up_A_buffers: (n_slots, 2 * max_rank, hidden)             — A replicated.
        #   gate_up_B_buffers: (n_slots, 2 * intermediate_per_tp, max_rank) — column-parallel.
        #   down_A_buffers:    (n_slots, max_rank, intermediate_per_tp)    — row-parallel.
        #   down_B_buffers:    (n_slots, hidden, max_rank)                 — B replicated.
        self._weight_buffers = LoraWeightBuffers(
            n_layers=self.n_layers,
            n_slots=self._n_slots,
            max_lora_rank=self.max_lora_rank,
            hidden_size=self.hidden_size,
            q_size_per_tp=self.q_size_per_tp,
            kv_size_per_tp=self.kv_size_per_tp,
            o_in_per_tp=self.o_in_per_tp,
            intermediate_per_tp=self.intermediate_per_tp,
            vocab_per_tp=self.vocab_per_tp,
            dtype=self.dtype,
            device=self.device,
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            buffer_groups=self.lora_buffer_groups,
        )
        self.qkv_A_buffers = self._weight_buffers.qkv_A_buffers
        self.qkv_B_buffers = self._weight_buffers.qkv_B_buffers
        self.o_A_buffers = self._weight_buffers.o_A_buffers
        self.o_B_buffers = self._weight_buffers.o_B_buffers
        self.gate_up_A_buffers = self._weight_buffers.gate_up_A_buffers
        self.gate_up_B_buffers = self._weight_buffers.gate_up_B_buffers
        self.down_A_buffers = self._weight_buffers.down_A_buffers
        self.down_B_buffers = self._weight_buffers.down_B_buffers
        self.lm_head_A_buffer = (
            self._weight_buffers.lm_head_A_buffer if self.enable_head_lora else None
        )
        self.lm_head_B_buffer = (
            self._weight_buffers.lm_head_B_buffer if self.enable_head_lora else None
        )
        self._qkv_output_offset = self._weight_buffers.qkv_output_offset
        self._max_qkv_out_dim = self._weight_buffers.max_qkv_out_dim
        self._o_slice_offsets = self._weight_buffers.o_slice_offsets
        self._gate_up_slice_offsets = self._weight_buffers.gate_up_slice_offsets
        self._down_slice_offsets = self._weight_buffers.down_slice_offsets
        self._moe_lora_buffers = MoeLoraBuffers(
            n_layers=self.n_layers,
            n_slots=self._n_slots,
            max_lora_rank=self.max_lora_rank,
            num_experts=self.num_experts,
            hidden_size=self.hidden_size,
            intermediate_per_tp=self.moe_intermediate_per_tp,
            dtype=self.dtype,
            device=self.device,
            shard_weights=self._weight_buffers.shard_weights,
            enabled=self.enable_moe_lora,
            compressed_shared_outer=self.lora_moe_compressed_shared_outer,
        )
        # Compatibility alias for tests/debug tooling that inspected the old
        # manager-owned storage directly.
        self._moe_lora_weights = self._moe_lora_buffers.weights_by_layer

        logger.info(
            "LoraManager initialized: max_loras=%d max_rank=%d "
            "tp_rank=%d/%d device=%s dtype=%s buffer_groups=%s "
            "moe_compressed_shared_outer=%s",
            max_loras,
            max_lora_rank,
            tp_rank,
            tp_size,
            device,
            dtype,
            ",".join(sorted(self.lora_buffer_groups)),
            self.lora_moe_compressed_shared_outer,
        )

    # ── Public API ──────────────────────────────────────────────────────────

    @property
    def batch_info(self) -> LoraBatchInfo:
        return self._batch_info

    @property
    def moe_lora_context(self) -> MoeLoraContext:
        return self._moe_lora_buffers.build_context(
            batch_info=self._batch_info,
            scalings=self._scalings,
            has_active_lora=self.has_active_lora,
        )

    def load_adapter(self, name: str, path: str) -> int:
        """Register a PEFT adapter from *path* and warm the CPU pool.

        ``path`` is recorded as the adapter's durable disk path; it must
        remain accessible for the lifetime of the manager because the CPU
        pool may evict the adapter back to disk under memory pressure.

        Returns the integer ``lora_id`` to use in subsequent
        ``prepare_loras`` calls.
        """
        if name in self._name_to_id:
            logger.warning("Adapter '%s' is already loaded; re-loading.", name)
            self._evict_by_name(name)
            self._evict_from_cpu(name)

        # Resolve the durable disk path now (used by future re-reads when
        # the CPU pool evicts these weights).
        adapter_path = path
        weight_path = resolve_adapter_weight_path(adapter_path)
        if not os.path.exists(weight_path):
            raise FileNotFoundError(
                f"Adapter weights not found at {weight_path!r} or {path!r}"
            )

        lora_id = self._next_id
        self._next_id += 1
        self._name_to_id[name] = lora_id
        self._id_to_name[lora_id] = name
        self._cpu_store.set_path(name, adapter_path)

        # Warm the CPU pool — bounded by ``max_loras_cpu``, may evict
        # other CPU-resident adapters back to disk.
        self._cpu_store.ensure(name)

        logger.info(
            "Registered adapter '%s' (lora_id=%d) from %s; CPU pool: %d/%d",
            name,
            lora_id,
            path,
            len(self._cpu_cache),
            self.max_loras_cpu,
        )
        return lora_id

    def unload_adapter(self, name: str) -> None:
        if name not in self._name_to_id:
            raise KeyError(f"Adapter '{name}' is not loaded.")

        lora_id = self._name_to_id.pop(name)
        # Keep _id_to_name[lora_id] alive until eviction fires so retracted
        # requests that resume with this lora_id can still be recognised and
        # get the correct weights rather than silently falling back to the base
        # model.  It is cleared when the GPU slot is finally freed below.

        if name in self._active_names:
            # The adapter was used in the most recent forward step; its GPU
            # weights may still be needed if the scheduler is mid-decode or has
            # retracted a request that it will later reschedule.  Defer the
            # weight zeroing until a batch arrives that does not include this
            # adapter — at that point the previous step is confirmed complete.
            logger.warning(
                "Adapter '%s' (lora_id=%d) unloaded while potentially in-flight; "
                "GPU slot eviction deferred until next batch that does not use it.",
                name,
                lora_id,
            )
            # Store (name, lora_id) so the flush can distinguish this entry from
            # a same-name re-registration that might occur before flushing.
            self._pending_eviction.add((name, lora_id))
            # CPU weights kept alive so retracted requests can still reload.
        else:
            # Safe to evict immediately — not active in the current batch.
            del self._id_to_name[lora_id]
            self._evict_by_name(name)
            self._cpu_store.remove(name)

        logger.info("Unloaded adapter '%s' (lora_id=%d)", name, lora_id)

    def get_id(self, name: str) -> int | None:
        return self._name_to_id.get(name)

    def _flush_one_pending(self, name: str, lora_id: int) -> None:
        """Carry out the deferred GPU+CPU eviction for one (name, lora_id) entry.

        Guards against two edge cases:
        - Re-registration: the same name was re-loaded after unload; the new
          slot should NOT be zeroed.  Detected by checking that _id_to_name
          still maps lora_id → name (the old entry, kept alive for retracted
          requests) and that the current name→id mapping no longer exists.
        - Already-evicted slot: LRU pressure may have freed the GPU slot
          before the deferred flush fires; _evict_by_name is idempotent.
        """
        # If the same name was re-registered, _name_to_id[name] exists again
        # with a NEW lora_id.  Skip GPU eviction — the slot now belongs to the
        # new adapter.  The CPU copy for the OLD weights was already removed or
        # never loaded under the new id.
        if self._name_to_id.get(name) is not None:
            # Name was re-registered; clear the stale _id_to_name entry for the
            # old lora_id only if it still points to this name.
            if self._id_to_name.get(lora_id) == name:
                del self._id_to_name[lora_id]
            return

        # Clear the reverse mapping kept alive for retracted-request safety.
        if self._id_to_name.get(lora_id) == name:
            del self._id_to_name[lora_id]

        self._evict_by_name(name)  # idempotent if already LRU-evicted
        self._cpu_store.remove(name)

    def flush_pending_evictions(self) -> None:
        """Evict all deferred adapter weights immediately.

        Safe to call at any time, including while the GPU is running a forward
        pass.  Because _reset_slot no longer issues GPU zero operations, the
        only side effects are CPU-side: removing the slot from _name_to_slot
        and cleaning up CPU weight caches.  The GPU memory retains stale values
        until the slot is reused, but no kernel ever reads from an evicted slot
        (prepare_loras only assigns weight_indices to slots present in
        _name_to_slot).

        Call this when the server has no pending decode requests and you want
        to reclaim GPU slots occupied by deferred-unloaded adapters.
        """
        for name, lora_id in list(self._pending_eviction):
            logger.info("Flushing deferred eviction for adapter '%s'.", name)
            self._flush_one_pending(name, lora_id)
            self._pending_eviction.discard((name, lora_id))

    def prepare_loras(
        self,
        lora_ids: list[int],
        per_request_token_counts: list[int] | int = 1,
    ) -> int:
        """Fill :attr:`batch_info` for the upcoming forward.

        Each request becomes one segment.  Returns the total number of
        tokens written.  All updates are in place on the persistent
        batch_info tensors so the captured CUDA graph keeps replaying
        against the same pointers.
        """
        bs = len(lora_ids)

        # Phase 1: resolve all unique adapters and promote them to GPU before
        # assigning any per-request slots.  A single-pass approach would silently
        # produce wrong outputs: if the batch needs more adapters than max_loras,
        # _find_free_slot evicts an already-assigned adapter (e.g. A), then the
        # later request for A gets NO_LORA_SLOT and runs as the base model.
        unique_names: dict[int, str] = {}
        for lid in lora_ids:
            if lid == 0 or lid in unique_names:
                continue
            name = self._id_to_name.get(lid)
            if name is not None:
                unique_names[lid] = name

        n_unique = len(unique_names)
        if n_unique > self.max_loras:
            raise RuntimeError(
                f"Batch requires {n_unique} unique LoRA adapters but "
                f"max_loras={self.max_loras}. Reduce adapter diversity per "
                "batch (use pack scheduling) or increase max_loras."
            )

        # The previous forward step is now complete (prepare_loras is called
        # synchronously before each forward).  Flush any deferred evictions for
        # adapters that are NOT needed by the current batch.
        current_batch_names = set(unique_names.values())
        for pending_name, pending_lora_id in list(self._pending_eviction):
            if pending_name not in current_batch_names:
                logger.info(
                    "Deferred eviction: removing adapter '%s' GPU weights now.",
                    pending_name,
                )
                self._flush_one_pending(pending_name, pending_lora_id)
                self._pending_eviction.discard((pending_name, pending_lora_id))

        # Track which adapters are active in this batch for mid-decode unload safety.
        self._active_names = current_batch_names

        # Promote all needed adapters before touching per_request_slots so that
        # LRU eviction only targets adapters NOT in this batch.  After each
        # promotion, move the adapter to MRU so subsequent promotions within
        # this loop don't evict an already-promoted or already-resident batch
        # adapter (which would be LRU if it was loaded in a previous step).
        for name in unique_names.values():
            self._ensure_in_gpu(name)
            self._gpu_lru.move_to_end(name)  # protect from intra-phase eviction

        # Phase 2: assign per-request slots from the now-stable _name_to_slot
        # map (no further evictions occur here).
        per_request_slots: list[int] = []
        for lid in lora_ids:
            if lid == 0:
                per_request_slots.append(NO_LORA_SLOT)
                continue
            name = self._id_to_name.get(lid)
            if name is None:
                logger.warning("Unknown lora_id %d; treating as base model.", lid)
                per_request_slots.append(NO_LORA_SLOT)
                continue
            slot = self._name_to_slot[name]  # guaranteed present after phase 1
            per_request_slots.append(slot)
            self._gpu_lru.move_to_end(name)

        # Per-request seg_lens.
        if isinstance(per_request_token_counts, int):
            seg_lens_list = [per_request_token_counts] * bs
        else:
            if len(per_request_token_counts) != bs:
                raise ValueError(
                    "per_request_token_counts length must match lora_ids length"
                )
            seg_lens_list = list(per_request_token_counts)

        total_tokens = sum(seg_lens_list)
        if total_tokens > self.max_num_tokens:
            raise ValueError(
                f"LoRA batch_info overflow: {total_tokens} > {self.max_num_tokens}"
            )
        max_len = max(seg_lens_list) if seg_lens_list else 0

        bi = self._batch_info

        # For decode batches (max_len == 1): compute adapter groups on CPU
        # so the grouped expand kernel can batch same-adapter tokens into a
        # full BLOCK_S=16 GEMM tile, recovering tensor-core efficiency.
        if max_len == 1 and bs > 1:
            sort_order, group_slots, group_starts, group_sizes = (
                build_decode_lora_groups(per_request_slots)
            )
            ng = len(group_slots)
            active_count = len(sort_order)
            self._sort_order_cpu[:active_count] = torch.as_tensor(
                sort_order, dtype=torch.int64
            )
            self._group_slots_cpu[:ng] = torch.as_tensor(group_slots, dtype=torch.int32)
            self._group_starts_cpu[:ng] = torch.as_tensor(
                group_starts, dtype=torch.int32
            )
            self._group_sizes_cpu[:ng] = torch.as_tensor(group_sizes, dtype=torch.int32)
            bi.sort_order = self._sort_order_buf
            bi.group_slots = self._group_slots_buf
            bi.group_starts = self._group_starts_buf
            bi.group_sizes = self._group_sizes_buf
            bi.sort_order[:active_count].copy_(
                self._sort_order_cpu[:active_count], non_blocking=True
            )
            bi.group_slots[:ng].copy_(self._group_slots_cpu[:ng], non_blocking=True)
            bi.group_starts[:ng].copy_(self._group_starts_cpu[:ng], non_blocking=True)
            bi.group_sizes[:ng].copy_(self._group_sizes_cpu[:ng], non_blocking=True)
            bi.num_groups = ng
            bi.max_group_size = max(group_sizes) if group_sizes else 0
        else:
            bi.sort_order = bi.group_slots = bi.group_starts = bi.group_sizes = None
            bi.num_groups = 0
            bi.max_group_size = 0

        first_slot = per_request_slots[0] if per_request_slots else NO_LORA_SLOT
        bi.single_lora_slot = (
            first_slot
            if first_slot != NO_LORA_SLOT
            and all(slot == first_slot for slot in per_request_slots)
            else NO_LORA_SLOT
        )
        bi.single_lora_rank = (
            self._slot_ranks[bi.single_lora_slot]
            if bi.single_lora_slot != NO_LORA_SLOT
            else 0
        )
        bi.multi_lora_start_slot = NO_LORA_SLOT
        bi.multi_lora_count = 0
        bi.multi_lora_segment_len = 0
        bi.multi_lora_rank = 0
        if (
            bs > 1
            and bi.single_lora_slot == NO_LORA_SLOT
            and max_len > _CHUNKED_THRESHOLD
            and len(set(seg_lens_list)) == 1
            and all(slot != NO_LORA_SLOT for slot in per_request_slots)
        ):
            start_slot = per_request_slots[0]
            consecutive_slots = all(
                slot == start_slot + i for i, slot in enumerate(per_request_slots)
            )
            rank = self._slot_ranks[start_slot]
            scaling = self._slot_scalings[start_slot]
            same_rank_and_scaling = all(
                self._slot_ranks[slot] == rank and self._slot_scalings[slot] == scaling
                for slot in per_request_slots
            )
            if consecutive_slots and rank > 0 and same_rank_and_scaling:
                bi.multi_lora_start_slot = start_slot
                bi.multi_lora_count = bs
                bi.multi_lora_segment_len = seg_lens_list[0]
                bi.multi_lora_rank = rank

        # Stage on CPU then a single non-blocking H2D.
        self._seg_lens_cpu[:bs] = torch.as_tensor(seg_lens_list, dtype=torch.int32)
        self._weight_indices_cpu[:bs] = torch.as_tensor(
            per_request_slots, dtype=torch.int32
        )

        self.has_active_lora = any(s != NO_LORA_SLOT for s in per_request_slots)

        bi = self._batch_info
        bi.bs = bs
        bi.num_segments = bs
        bi.max_len = max_len

        # Skip the H2D copies and on-device cumsum when no adapter is active:
        # the no-LoRA CUDA graph omits all LoRA kernels and never reads
        # weight_indices / seg_lens / seg_indptr, so updating them is wasted work.
        if self.has_active_lora:
            bi.seg_lens[:bs].copy_(self._seg_lens_cpu[:bs], non_blocking=True)
            bi.weight_indices[:bs].copy_(
                self._weight_indices_cpu[:bs], non_blocking=True
            )
            bi.seg_indptr[0] = 0
            torch.cumsum(bi.seg_lens[:bs], dim=0, out=bi.seg_indptr[1 : bs + 1])

        return total_tokens

    def apply_qkv_lora(
        self,
        hidden_states: torch.Tensor,
        qkv: torch.Tensor,
        layer_id: int,
    ) -> torch.Tensor:
        """Fused QKV LoRA delta: ``qkv += B @ A @ x * scaling``.

        ``hidden_states``: ``(s, hidden)`` (full input).
        ``qkv``:           ``(s, q_per_tp + 2 * kv_per_tp)`` (output of qkv_proj
        on this rank).  Updated in place via the kernel's fused-add.
        """
        if hidden_states.shape[0] == 0:
            return qkv
        if not self.enable_attn_lora:
            return qkv
        bi = self._batch_info
        if bi.bs == 0 or not self.has_active_lora:
            return qkv

        A_buf = self.qkv_A_buffers[layer_id]
        B_buf = self.qkv_B_buffers[layer_id]
        # lora_a: (s, 3 * max_rank)
        lora_a = (
            lora_shrink_prefill_fwd(hidden_states, A_buf, bi, stack_num=3)
            if bi.max_len > _CHUNKED_THRESHOLD
            else lora_shrink_fwd(hidden_states, A_buf, bi, stack_num=3)
        )
        if bi.max_len > _CHUNKED_THRESHOLD:
            lora_expand_prefill_fwd(
                lora_a,
                B_buf,
                bi,
                self._qkv_output_offset,
                self._max_qkv_out_dim,
                base_output=qkv,
            )
        else:
            lora_qkv_expand_fwd(
                lora_a,
                B_buf,
                bi,
                self._qkv_output_offset,
                self._max_qkv_out_dim,
                base_output=qkv,
            )
        return qkv

    def apply_o_lora(
        self,
        attn_output: torch.Tensor,
        o_output: torch.Tensor,
        layer_id: int,
    ) -> torch.Tensor:
        """Row-parallel O-projection LoRA delta.

        ``attn_output``: ``(s, q_per_tp)`` per-rank attention output (input
        to o_proj).
        ``o_output``: ``(s, hidden)`` partial sum from the host o_proj
        (``reduce_results=False`` on this codebase).  Updated in place.

        Each rank computes ``B @ A_local @ x_local`` — a partial of shape
        ``(s, hidden)``.  A is sharded along its input dim and B is
        replicated, so the sum of partials over ranks equals
        ``B @ A_full @ x_full``.  The host layer's downstream fused
        all-reduce in ``post_attention_layernorm`` sums the base partial
        and the LoRA partial together, producing the correct full output.
        """
        if attn_output.shape[0] == 0:
            return o_output
        if not self.enable_attn_lora:
            return o_output
        bi = self._batch_info
        if bi.bs == 0 or not self.has_active_lora:
            return o_output

        A_buf = self.o_A_buffers[layer_id]
        B_buf = self.o_B_buffers[layer_id]
        # lora_a (partial per rank): (s, max_rank).  No internal all-reduce —
        # the partial flows into B and the result rides the downstream sum.
        lora_a = (
            lora_shrink_prefill_fwd(attn_output, A_buf, bi, stack_num=1)
            if bi.max_len > _CHUNKED_THRESHOLD
            else lora_shrink_fwd(attn_output, A_buf, bi, stack_num=1)
        )
        if bi.max_len > _CHUNKED_THRESHOLD:
            lora_expand_prefill_fwd(
                lora_a,
                B_buf,
                bi,
                self._o_slice_offsets,
                self.hidden_size,
                base_output=o_output,
            )
        elif _use_triton_grouped_decode(bi):
            lora_expand_grouped_v2_fwd(lora_a, B_buf, bi, base_output=o_output)
        else:
            lora_expand_fwd(lora_a, B_buf, bi, base_output=o_output)
        return o_output

    def apply_gate_up_lora(
        self,
        hidden_states: torch.Tensor,
        gate_up: torch.Tensor,
        layer_id: int,
    ) -> torch.Tensor:
        """Fused gate/up LoRA delta: ``gate_up += B @ A @ x * scaling``.

        ``hidden_states``: ``(s, hidden)``.
        ``gate_up``:       ``(s, 2 * intermediate_per_tp)`` — output of the
        column-parallel ``gate_up_proj`` (each rank holds its own output
        shard).  Updated in place via the kernel's fused-add.
        """
        if hidden_states.shape[0] == 0:
            return gate_up
        if not self.enable_mlp_lora:
            return gate_up
        bi = self._batch_info
        if bi.bs == 0 or not self.has_active_lora:
            return gate_up

        A_buf = self.gate_up_A_buffers[layer_id]
        B_buf = self.gate_up_B_buffers[layer_id]
        # lora_a: (s, 2 * max_rank) — gate's lora_a in [:, :r], up's in [:, r:].
        lora_a = (
            lora_shrink_prefill_fwd(hidden_states, A_buf, bi, stack_num=2)
            if bi.max_len > _CHUNKED_THRESHOLD
            else lora_shrink_fwd(hidden_states, A_buf, bi, stack_num=2)
        )
        if bi.max_len > _CHUNKED_THRESHOLD:
            lora_expand_prefill_fwd(
                lora_a,
                B_buf,
                bi,
                self._gate_up_slice_offsets,
                self.intermediate_per_tp,
                base_output=gate_up,
            )
        else:
            lora_gate_up_expand_fwd(
                lora_a,
                B_buf,
                bi,
                self.intermediate_per_tp,
                base_output=gate_up,
            )
        return gate_up

    def apply_down_lora(
        self,
        x: torch.Tensor,
        down_output: torch.Tensor,
        layer_id: int,
    ) -> torch.Tensor:
        """Down-projection LoRA delta (row-parallel under MLP TP).

        ``x``:           ``(s, intermediate_per_tp)`` — input to the
        row-parallel ``down_proj`` (this rank's input shard).
        ``down_output``: ``(s, hidden)`` — partial output of ``down_proj``
        before its all-reduce.  Updated in place.

        Each rank's delta is ``B @ A_local @ x_local``: A is sharded along
        the input dim and B is replicated, so summing per-rank deltas yields
        the full ``B @ A_full @ x_full``.  The base linear runs with
        ``reduce_results=False``; the downstream all-reduce that sums the
        base partial also sums the LoRA partials.
        """
        if x.shape[0] == 0:
            return down_output
        if not self.enable_mlp_lora:
            return down_output
        bi = self._batch_info
        if bi.bs == 0 or not self.has_active_lora:
            return down_output

        A_buf = self.down_A_buffers[layer_id]
        B_buf = self.down_B_buffers[layer_id]
        lora_a = (
            lora_shrink_prefill_fwd(x, A_buf, bi, stack_num=1)
            if bi.max_len > _CHUNKED_THRESHOLD
            else lora_shrink_fwd(x, A_buf, bi, stack_num=1)
        )
        if bi.max_len > _CHUNKED_THRESHOLD:
            lora_expand_prefill_fwd(
                lora_a,
                B_buf,
                bi,
                self._down_slice_offsets,
                self.hidden_size,
                base_output=down_output,
            )
        elif _use_triton_grouped_decode(bi):
            lora_expand_grouped_v2_fwd(lora_a, B_buf, bi, base_output=down_output)
        else:
            lora_expand_fwd(lora_a, B_buf, bi, base_output=down_output)
        return down_output

    def apply_lm_head_lora(
        self,
        hidden_states: torch.Tensor,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """lm_head LoRA delta: ``logits += B @ A @ x * scaling``.

        ``hidden_states``: ``(s, hidden)`` — one token per request (pruned).
        ``logits``:        ``(s, vocab_per_tp)`` — pre-all-gather logits shard.
        Applied before the TP all-gather so each rank contributes its vocab
        shard correctly.

        Note: when ``extend_return_logprob`` is True the caller may pass more
        than ``bi.bs`` tokens.  In that case this method is a no-op because
        the per-token slot mapping is not available here; sampling logits are
        still correct for the last token of each request.
        """
        if hidden_states.shape[0] == 0:
            return logits
        if not self.enable_head_lora:
            return logits
        bi = self._batch_info
        if bi.bs == 0 or not self.has_active_lora:
            return logits
        if hidden_states.shape[0] != bi.bs:
            return logits

        slots = bi.weight_indices[: bi.bs]  # (bs,)
        valid = slots != NO_LORA_SLOT
        if not valid.any():
            return logits

        # Fast path: all requests use the same adapter slot.
        # Use plain matmul to avoid a gather of the B matrix (vocab_per_tp × rank
        # bytes) for every request.  Guarded from CUDA graph capture because the
        # Python branch is frozen at capture time — replaying with a different
        # single_lora_slot would silently use stale weights.
        if (
            bi.single_lora_slot != NO_LORA_SLOT
            and not torch.cuda.is_current_stream_capturing()
        ):
            slot = bi.single_lora_slot
            scaling = self._scalings[slot].item()
            A = self.lm_head_A_buffer[slot]  # (r, hidden)
            B = self.lm_head_B_buffer[slot]  # (vocab_per_tp, r)
            lora_a = hidden_states @ A.T  # (bs, r)
            delta = lora_a @ B.T  # (bs, vocab_per_tp)
            return logits + delta * scaling

        valid_slots = slots.clamp(min=0)
        # A: (bs, r, hidden),  B: (bs, vocab_per_tp, r)
        A = self.lm_head_A_buffer[valid_slots]
        B = self.lm_head_B_buffer[valid_slots]
        # lora_a: (bs, r) = A @ hidden_states[..., None]
        lora_a = torch.bmm(A, hidden_states.unsqueeze(-1)).squeeze(-1)
        # delta: (bs, vocab_per_tp)
        delta = torch.bmm(B, lora_a.unsqueeze(-1)).squeeze(-1)
        # Zero out requests with no adapter; scale the rest.
        scale = self._scalings[valid_slots] * valid.to(self._scalings.dtype)
        return logits + delta * scale.unsqueeze(-1)

    def apply_moe_gate_up_lora(
        self,
        layer_id: int,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        gate_up_output: torch.Tensor,
        *,
        sorted_token_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compatibility wrapper; MoE-specific work lives in MoeLoraContext."""
        if not self.enable_moe_lora:
            return gate_up_output
        return self.moe_lora_context.apply_gate_up_lora(
            layer_id,
            hidden_states,
            topk_ids,
            gate_up_output,
            sorted_token_ids=sorted_token_ids,
        )

    def apply_moe_down_lora(
        self,
        layer_id: int,
        intermediate: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        down_output: torch.Tensor,
        *,
        sorted_token_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compatibility wrapper; MoE-specific work lives in MoeLoraContext."""
        if not self.enable_moe_lora:
            return down_output
        return self.moe_lora_context.apply_down_lora(
            layer_id,
            intermediate,
            topk_ids,
            topk_weights,
            down_output,
            sorted_token_ids=sorted_token_ids,
        )

    def set_adapter_scaling(self, name: str, scaling: float) -> None:
        slot = self._name_to_slot.get(name)
        if slot is not None:
            self._slot_scalings[slot] = scaling
            self._scalings[slot] = scaling

    # ── Slot allocation ─────────────────────────────────────────────────────

    def _ensure_in_gpu(self, name: str) -> int:
        if name in self._name_to_slot:
            return self._name_to_slot[name]
        # Tier-2 → Tier-1 promotion; may need to read from disk if the
        # CPU pool has evicted this adapter since registration.
        self._cpu_store.ensure(name)
        slot = self._find_free_slot()
        self._load_to_slot(name, slot)
        self._name_to_slot[name] = slot
        self._slot_to_name[slot] = name
        self._gpu_lru[name] = None
        return slot

    def prefetch(self, name: str) -> None:
        """Best-effort async warm of the CPU pool for *name*.

        Called from the request-admission path: when a request with a
        non-zero ``lora_id`` arrives the manager kicks off a background
        disk read so the safetensors I/O is overlapped with the previous
        forward step rather than blocking ``prepare_loras`` of the step
        that actually consumes the adapter.

        No-op when the adapter is already CPU-resident or a load is
        already in flight.  Silently ignores unknown adapters (the
        request will fall back to base via NO_LORA_SLOT).
        """
        self._cpu_store.prefetch(name)

    def _evict_from_cpu(self, name: str) -> None:
        """Public helper, takes the lock.  Caller must ensure *name* is
        not currently GPU-resident."""
        self._cpu_store.evict(name)

    def _find_free_slot(self) -> int:
        for slot in range(self._n_slots):
            if self._slot_to_name[slot] is None:
                return slot
        for candidate_name in list(self._gpu_lru.keys()):
            slot = self._name_to_slot[candidate_name]
            logger.debug("Evicting adapter '%s' from GPU slot %d", candidate_name, slot)
            self._evict_by_name(candidate_name)
            return slot
        raise RuntimeError(
            "LoRA GPU pool is full and no evictable adapter was found. "
            f"Increase max_loras (current: {self.max_loras})."
        )

    def _load_to_slot(self, name: str, slot: int) -> None:
        cpu_weights = self._cpu_cache[name]
        rank = self._get_rank_for(name)
        scaling = self._get_scaling_for(name, rank)
        self._reset_slot(slot)
        self._lora_ranks[slot] = rank
        self._slot_ranks[slot] = rank
        self._slot_scalings[slot] = scaling
        self._scalings[slot] = scaling
        self._weight_buffers.load_adapter_to_slot(cpu_weights, slot, rank)
        self._moe_lora_buffers.load_adapter_to_slot(cpu_weights, slot, rank)

        logger.debug("Loaded adapter '%s' into GPU slot %d (rank=%d)", name, slot, rank)

    def _get_rank_for(self, name: str) -> int:
        cpu_weights = self._cpu_cache.get(name, {})
        if not cpu_weights:
            return self.max_lora_rank
        # Check layer 0 first (dense attn/MLP modules).
        if 0 in cpu_weights:
            for mod in PEFT_MODULES:
                if mod in cpu_weights[0]:
                    return cpu_weights[0][mod][0].shape[0]
            for mod, tensors in cpu_weights[0].items():
                if mod.startswith("experts."):
                    lora_A = tensors[0]
                    if lora_A.dim() == 3:
                        return lora_A.shape[1]
                    return lora_A.shape[0]
        # Fall back to lm_head (head-only adapters).
        if LORA_HEAD_LAYER_ID in cpu_weights:
            head = cpu_weights[LORA_HEAD_LAYER_ID]
            if PEFT_HEAD_MODULE in head:
                return head[PEFT_HEAD_MODULE][0].shape[0]
        return self.max_lora_rank

    def _get_scaling_for(self, name: str, rank: int) -> float:
        return read_adapter_scaling(self._adapter_paths.get(name), rank)

    def _evict_by_name(self, name: str) -> None:
        if name in self._name_to_slot:
            slot = self._name_to_slot.pop(name)
            self._slot_to_name[slot] = None
            self._reset_slot(slot)
        self._gpu_lru.pop(name, None)

    def _reset_slot(self, slot: int) -> None:
        # GPU weight tensors are intentionally NOT zeroed here.
        #
        # Correctness argument: prepare_loras assigns weight_indices[i] only to
        # slots present in _name_to_slot.  _evict_by_name removes the slot from
        # _name_to_slot before calling _reset_slot, so no kernel ever reads from
        # an evicted slot's GPU memory regardless of what values are there.
        # _load_to_slot overwrites the stale values when the slot is reused.
        #
        # Skipping the GPU zeros removes potentially hundreds of kernel launches
        # per eviction (one zero_() per buffer per layer) and — more critically —
        # eliminates a CUDA stream race: graph.replay() uses a dedicated stream
        # while tensor.zero_() runs on the default stream; without explicit
        # inter-stream synchronisation, an immediate GPU zero could race with an
        # in-flight graph kernel still reading the old weights.
        #
        # MoE buffers: _moe_lora_buffers.clear_slot does both GPU zeroing AND
        # CPU dict cleanup (weights_by_layer.pop).  Keep the CPU cleanup, skip
        # the GPU zeros.
        self._moe_lora_buffers.clear_slot_cpu_only(slot)
        self._lora_ranks[slot] = 0
        self._slot_ranks[slot] = 0
        self._slot_scalings[slot] = 0.0
        self._scalings[slot] = 0.0

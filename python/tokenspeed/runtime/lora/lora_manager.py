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

Slot 0 is the no-adapter sentinel (rank 0, scaling 0).  The Triton
kernels short-circuit on slot 0, so the captured CUDA graph stays a no-op
when no request uses an adapter.

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

import json
import os
import re
from collections import OrderedDict
from dataclasses import dataclass

import torch

from tokenspeed.runtime.distributed.comm_ops import all_reduce as comm_all_reduce
from tokenspeed.runtime.lora.triton_ops import (
    qkv_lora_b_fwd,
    sgemm_lora_a_fwd,
    sgemm_lora_b_fwd,
)
from tokenspeed.runtime.utils import get_colorful_logger

logger = get_colorful_logger(__name__)

_PEFT_ATTN_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")


# ── Batch info ──────────────────────────────────────────────────────────────


@dataclass
class LoraBatchInfo:
    """Per-step segment metadata read by the Triton kernels.

    All tensors live on the LoRA device.  When the captured CUDA graph
    needs persistent storage (for in-place updates between replays), the
    LoraManager pre-allocates these tensors with maximum sizes; runtime
    fills the prefix and updates :attr:`bs` / :attr:`max_len`.
    """

    bs: int
    num_segments: int
    max_len: int
    seg_lens: torch.Tensor  # (num_segments,) int32
    seg_indptr: torch.Tensor  # (num_segments + 1,) int32
    weight_indices: torch.Tensor  # (num_segments,) int32
    lora_ranks: torch.Tensor  # (n_slots,) int32 (slot 0 ⇒ rank 0)
    scalings: torch.Tensor  # (n_slots,) float32
    permutation: torch.Tensor | None = None  # unused (no sort by adapter yet)


# ── Adapter file IO ─────────────────────────────────────────────────────────


def _load_safetensors(path: str) -> dict[str, torch.Tensor]:
    from safetensors import safe_open

    tensors: dict[str, torch.Tensor] = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors


def _parse_adapter_weights(
    tensors: dict[str, torch.Tensor],
) -> dict[int, dict[str, tuple[torch.Tensor, torch.Tensor]]]:
    """``{layer_id: {module_name: (lora_A, lora_B)}}`` (CPU, fp32 from PEFT)."""
    pattern = re.compile(
        r"base_model\.model\.model\.layers\.(\d+)\.self_attn\."
        r"(q_proj|k_proj|v_proj|o_proj)\.lora_(A|B)\.weight"
    )
    weights: dict[int, dict[str, dict[str, torch.Tensor]]] = {}
    for key, tensor in tensors.items():
        m = pattern.match(key)
        if not m:
            continue
        layer_id, module, ab = int(m.group(1)), m.group(2), m.group(3)
        weights.setdefault(layer_id, {}).setdefault(module, {})[ab] = tensor

    result: dict[int, dict[str, tuple[torch.Tensor, torch.Tensor]]] = {}
    for layer_id, modules in weights.items():
        result[layer_id] = {}
        for module, ab_dict in modules.items():
            result[layer_id][module] = (ab_dict["A"], ab_dict["B"])
    return result


# ── Manager ─────────────────────────────────────────────────────────────────


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
    ) -> None:
        self.max_loras = max_loras
        self.max_lora_rank = max_lora_rank
        self.max_num_tokens = max_num_tokens
        self.dtype = dtype
        self.device = device
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.tp_group = tp_group

        self.n_layers: int = model_config.num_hidden_layers
        hidden: int = model_config.hidden_size
        n_heads: int = model_config.num_attention_heads
        n_kv: int = model_config.num_key_value_heads
        head_dim: int = hidden // n_heads

        self.q_size_per_tp: int = (n_heads // tp_size) * head_dim
        self.kv_size_per_tp: int = max(1, n_kv // tp_size) * head_dim
        self.o_in_per_tp: int = self.q_size_per_tp
        self.hidden_size: int = hidden

        # Slot 0 = no-adapter sentinel.  Real adapters take 1 .. max_loras.
        self._n_slots: int = max_loras + 1
        self._slot_to_name: list[str | None] = [None] * self._n_slots
        self._name_to_slot: dict[str, int] = {}
        self._lru: OrderedDict[str, None] = OrderedDict()

        self._cpu_cache: dict[
            str, dict[int, dict[str, tuple[torch.Tensor, torch.Tensor]]]
        ] = {}
        self._name_to_id: dict[str, int] = {}
        self._id_to_name: dict[int, str] = {}
        self._next_id: int = 1
        self._pinned: set[str] = set()
        self._adapter_paths: dict[str, str] = {}

        # Per-slot rank + scaling.  Rank 0 means "no adapter"; the Triton
        # kernels skip on rank 0, so slot 0's row is permanently zero.
        self._lora_ranks: torch.Tensor = torch.zeros(
            self._n_slots, dtype=torch.int32, device=device
        )
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
            weight_indices=torch.zeros(
                max_num_tokens, dtype=torch.int32, device=device
            ),
            lora_ranks=self._lora_ranks,
            scalings=self._scalings,
            permutation=None,
        )

        # CPU staging buffers (pinned) for the per-step H2D copy.
        self._seg_lens_cpu = torch.zeros(
            max_num_tokens, dtype=torch.int32, pin_memory=True
        )
        self._weight_indices_cpu = torch.zeros(
            max_num_tokens, dtype=torch.int32, pin_memory=True
        )

        # ── GPU weight buffers ─────────────────────────────────────────────
        # qkv_A_buffers: (n_slots, 3 * max_rank, hidden) — stacked q/k/v A.
        # qkv_B_buffers: (n_slots, q_per_tp + 2 * kv_per_tp, max_rank).
        # o_A_buffers:   (n_slots, max_rank, o_in_per_tp).
        # o_B_buffers:   (n_slots, hidden, max_rank).
        self.qkv_A_buffers: list[torch.Tensor] = []
        self.qkv_B_buffers: list[torch.Tensor] = []
        self.o_A_buffers: list[torch.Tensor] = []
        self.o_B_buffers: list[torch.Tensor] = []

        # Cumulative output offsets [0, q, q+kv, q+2*kv] for qkv_lora_b.
        self._qkv_output_offset = torch.tensor(
            [
                0,
                self.q_size_per_tp,
                self.q_size_per_tp + self.kv_size_per_tp,
                self.q_size_per_tp + 2 * self.kv_size_per_tp,
            ],
            dtype=torch.int32,
            device=device,
        )
        self._max_qkv_out_dim = max(self.q_size_per_tp, self.kv_size_per_tp)

        self._alloc_gpu_buffers()

        logger.info(
            "LoraManager initialized: max_loras=%d max_rank=%d "
            "tp_rank=%d/%d device=%s dtype=%s",
            max_loras,
            max_lora_rank,
            tp_rank,
            tp_size,
            device,
            dtype,
        )

    # ── Public API ──────────────────────────────────────────────────────────

    @property
    def batch_info(self) -> LoraBatchInfo:
        return self._batch_info

    def load_adapter(self, name: str, path: str, pinned: bool = False) -> int:
        """Load a PEFT adapter from *path* (CPU side)."""
        if name in self._name_to_id:
            logger.warning("Adapter '%s' is already loaded; re-loading.", name)
            self._evict_by_name(name)

        adapter_path = path
        safetensors = os.path.join(adapter_path, "adapter_model.safetensors")
        if not os.path.exists(safetensors):
            safetensors = path

        raw = _load_safetensors(safetensors)
        weights = _parse_adapter_weights(raw)
        self._cpu_cache[name] = weights

        lora_id = self._next_id
        self._next_id += 1
        self._name_to_id[name] = lora_id
        self._id_to_name[lora_id] = name
        self._adapter_paths[name] = adapter_path
        if pinned:
            self._pinned.add(name)

        logger.info("Loaded adapter '%s' (lora_id=%d) from %s", name, lora_id, path)
        return lora_id

    def unload_adapter(self, name: str) -> None:
        if name not in self._name_to_id:
            raise KeyError(f"Adapter '{name}' is not loaded.")
        self._evict_by_name(name)
        self._cpu_cache.pop(name, None)
        lora_id = self._name_to_id.pop(name)
        del self._id_to_name[lora_id]
        self._pinned.discard(name)
        logger.info("Unloaded adapter '%s'", name)

    def get_id(self, name: str) -> int | None:
        return self._name_to_id.get(name)

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
        # Resolve names → slots; LRU bookkeeping.
        per_request_slots: list[int] = []
        for lid in lora_ids:
            if lid == 0:
                per_request_slots.append(0)
                continue
            name = self._id_to_name.get(lid)
            if name is None:
                logger.warning("Unknown lora_id %d; treating as base model.", lid)
                per_request_slots.append(0)
                continue
            slot = self._ensure_in_gpu(name)
            per_request_slots.append(slot)
            self._lru.move_to_end(name)

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

        # Stage on CPU then a single non-blocking H2D.
        self._seg_lens_cpu[:bs] = torch.as_tensor(seg_lens_list, dtype=torch.int32)
        self._weight_indices_cpu[:bs] = torch.as_tensor(
            per_request_slots, dtype=torch.int32
        )

        bi = self._batch_info
        bi.seg_lens[:bs].copy_(self._seg_lens_cpu[:bs], non_blocking=True)
        bi.weight_indices[:bs].copy_(self._weight_indices_cpu[:bs], non_blocking=True)
        # cumsum on device — same number of segments as bs.
        bi.seg_indptr[0] = 0
        torch.cumsum(bi.seg_lens[:bs], dim=0, out=bi.seg_indptr[1 : bs + 1])
        bi.bs = bs
        bi.num_segments = bs
        bi.max_len = max_len
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
        bi = self._batch_info
        if bi.bs == 0:
            return qkv

        A_buf = self.qkv_A_buffers[layer_id]
        B_buf = self.qkv_B_buffers[layer_id]
        # lora_a: (s, 3 * max_rank)
        lora_a = sgemm_lora_a_fwd(hidden_states, A_buf, bi, stack_num=3)
        qkv_lora_b_fwd(
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

        TP correctness caveat: the delta computed here is the *full*
        ``B @ A @ x`` (after an internal all-reduce on lora_a).  The host
        layer's downstream fused all-reduce in post_attention_layernorm
        sums this delta ``tp_size`` times, overcounting the LoRA
        contribution at TP > 1.  This is a pre-existing TP issue
        independent of the kernel path; fixing it cleanly requires
        coordinating with the host module's reduce policy.
        """
        if attn_output.shape[0] == 0:
            return o_output
        bi = self._batch_info
        if bi.bs == 0:
            return o_output

        A_buf = self.o_A_buffers[layer_id]
        B_buf = self.o_B_buffers[layer_id]
        # lora_a (partial per rank): (s, max_rank)
        lora_a = sgemm_lora_a_fwd(attn_output, A_buf, bi, stack_num=1)
        # All-reduce so each rank has the full ``A @ x``.  Routes through
        # the comm_ops backend (graph-capturable).
        if self.tp_size > 1 and self.tp_group is not None:
            lora_a = comm_all_reduce(lora_a, self.tp_rank, self.tp_group)
        sgemm_lora_b_fwd(lora_a, B_buf, bi, base_output=o_output)
        return o_output

    def set_adapter_scaling(self, name: str, scaling: float) -> None:
        slot = self._name_to_slot.get(name)
        if slot is not None:
            self._scalings[slot] = scaling

    # ── Slot allocation ─────────────────────────────────────────────────────

    def _alloc_gpu_buffers(self) -> None:
        r = self.max_lora_rank
        h = self.hidden_size
        q = self.q_size_per_tp
        kv = self.kv_size_per_tp
        o_in = self.o_in_per_tp
        n = self._n_slots

        for _ in range(self.n_layers):
            # qkv_A: stack q/k/v along dim 1.  All three see the full input.
            self.qkv_A_buffers.append(
                torch.zeros((n, 3 * r, h), dtype=self.dtype, device=self.device)
            )
            # qkv_B: stack q/k/v along dim 1, with their per-rank output sizes.
            self.qkv_B_buffers.append(
                torch.zeros((n, q + 2 * kv, r), dtype=self.dtype, device=self.device)
            )
            self.o_A_buffers.append(
                torch.zeros((n, r, o_in), dtype=self.dtype, device=self.device)
            )
            self.o_B_buffers.append(
                torch.zeros((n, h, r), dtype=self.dtype, device=self.device)
            )

    def _ensure_in_gpu(self, name: str) -> int:
        if name in self._name_to_slot:
            return self._name_to_slot[name]
        slot = self._find_free_slot()
        self._load_to_slot(name, slot)
        self._name_to_slot[name] = slot
        self._slot_to_name[slot] = name
        self._lru[name] = None
        return slot

    def _find_free_slot(self) -> int:
        for slot in range(1, self._n_slots):
            if self._slot_to_name[slot] is None:
                return slot
        for candidate_name in list(self._lru.keys()):
            if candidate_name in self._pinned:
                continue
            slot = self._name_to_slot[candidate_name]
            logger.debug("Evicting adapter '%s' from GPU slot %d", candidate_name, slot)
            del self._name_to_slot[candidate_name]
            self._slot_to_name[slot] = None
            del self._lru[candidate_name]
            return slot
        raise RuntimeError(
            "LoRA GPU pool is full and all adapters are pinned. "
            f"Increase max_loras (current: {self.max_loras}) or unpin an adapter."
        )

    def _load_to_slot(self, name: str, slot: int) -> None:
        cpu_weights = self._cpu_cache[name]
        rank = self._get_rank_for(name)
        scaling = self._get_scaling_for(name, rank)
        self._lora_ranks[slot] = rank
        self._scalings[slot] = scaling

        for layer_id, modules in cpu_weights.items():
            for mod, (lora_A_full, lora_B_full) in modules.items():
                actual_rank = lora_A_full.shape[0]
                lora_A_gpu = lora_A_full.to(device=self.device, dtype=self.dtype)
                lora_B_gpu = lora_B_full.to(device=self.device, dtype=self.dtype)

                lora_A_shard, lora_B_shard = self._shard_weights(
                    mod, lora_A_gpu, lora_B_gpu
                )
                r = min(actual_rank, self.max_lora_rank)

                if mod in ("q_proj", "k_proj", "v_proj"):
                    qkv_idx = ("q_proj", "k_proj", "v_proj").index(mod)
                    rank_off = qkv_idx * self.max_lora_rank
                    out_off, out_size = self._qkv_b_slice(mod)
                    # A — stack along rank dim: rows [qkv_idx*max_rank:+r] hold
                    # the actual (rank, hidden) of this projection.
                    self.qkv_A_buffers[layer_id][
                        slot, rank_off : rank_off + r, :
                    ].copy_(lora_A_shard[:r])
                    # B — stack along output dim with its sharded out size.
                    self.qkv_B_buffers[layer_id][
                        slot, out_off : out_off + out_size, :r
                    ].copy_(lora_B_shard[:, :r])
                else:  # o_proj
                    self.o_A_buffers[layer_id][slot, :r, :].copy_(lora_A_shard[:r])
                    self.o_B_buffers[layer_id][slot, :, :r].copy_(lora_B_shard[:, :r])

        logger.debug("Loaded adapter '%s' into GPU slot %d (rank=%d)", name, slot, rank)

    def _qkv_b_slice(self, module: str) -> tuple[int, int]:
        """``(offset, size)`` of one projection inside the fused QKV B buffer."""
        if module == "q_proj":
            return 0, self.q_size_per_tp
        if module == "k_proj":
            return self.q_size_per_tp, self.kv_size_per_tp
        return self.q_size_per_tp + self.kv_size_per_tp, self.kv_size_per_tp

    def _get_rank_for(self, name: str) -> int:
        cpu_weights = self._cpu_cache.get(name, {})
        if cpu_weights and 0 in cpu_weights and "q_proj" in cpu_weights[0]:
            return cpu_weights[0]["q_proj"][0].shape[0]
        return self.max_lora_rank

    def _get_scaling_for(self, name: str, rank: int) -> float:
        adapter_path = self._adapter_paths.get(name)
        if adapter_path:
            config_file = os.path.join(adapter_path, "adapter_config.json")
            if os.path.exists(config_file):
                try:
                    with open(config_file) as f:
                        cfg = json.load(f)
                    alpha = float(cfg.get("lora_alpha", rank))
                    r = int(cfg.get("r", rank))
                    return alpha / r if r > 0 else 1.0
                except Exception:
                    pass
        return 1.0

    def _shard_weights(
        self,
        module: str,
        lora_A: torch.Tensor,
        lora_B: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.tp_size == 1:
            return lora_A, lora_B
        if module in ("q_proj", "k_proj", "v_proj"):
            out_total = lora_B.shape[0]
            out_per = out_total // self.tp_size
            return (
                lora_A,
                lora_B[self.tp_rank * out_per : (self.tp_rank + 1) * out_per],
            )
        # row-parallel o_proj: shard A along input dim
        in_total = lora_A.shape[1]
        in_per = in_total // self.tp_size
        return (
            lora_A[:, self.tp_rank * in_per : (self.tp_rank + 1) * in_per],
            lora_B,
        )

    def _evict_by_name(self, name: str) -> None:
        if name in self._name_to_slot:
            slot = self._name_to_slot.pop(name)
            self._slot_to_name[slot] = None
            for layer_id in range(self.n_layers):
                self.qkv_A_buffers[layer_id][slot].zero_()
                self.qkv_B_buffers[layer_id][slot].zero_()
                self.o_A_buffers[layer_id][slot].zero_()
                self.o_B_buffers[layer_id][slot].zero_()
            self._lora_ranks[slot] = 0
            self._scalings[slot] = 0.0
        self._lru.pop(name, None)

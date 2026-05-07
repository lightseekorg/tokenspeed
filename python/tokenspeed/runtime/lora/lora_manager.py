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

"""LoRA adapter weight manager.

Handles loading PEFT adapters from disk, maintaining a fixed-size GPU memory
pool (one slot per adapter), LRU eviction when the pool is full, and
providing the per-layer A/B buffers that the model's forward pass reads.

Memory layout
-------------
For each module (q_proj, k_proj, v_proj, o_proj) and each layer:

  A_buffers[module][layer]:  [n_slots, max_rank, in_dim_per_tp]
  B_buffers[module][layer]:  [n_slots, out_dim_per_tp, max_rank]

Slot 0 is permanently zeroed — it represents "no adapter" and ensures that
requests without a LoRA adapter produce a zero delta.

Tensor-parallelism notes
------------------------
* Column-parallel projections (q, k, v): lora_A sees the full input,
  lora_B is sharded along the output dimension.
* Row-parallel projection (o): lora_A is sharded along the input dimension;
  the partial A outputs must be all_reduced before applying lora_B.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from tokenspeed.runtime.utils import get_colorful_logger

if TYPE_CHECKING:
    pass

logger = get_colorful_logger(__name__)

# Module names as they appear in PEFT adapter_model.safetensors keys
_PEFT_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")


def _load_safetensors(path: str) -> dict[str, torch.Tensor]:
    """Load all tensors from a safetensors file to CPU."""
    from safetensors import safe_open

    tensors: dict[str, torch.Tensor] = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors


def _parse_adapter_weights(
    tensors: dict[str, torch.Tensor],
    n_layers: int,
) -> dict[int, dict[str, tuple[torch.Tensor, torch.Tensor]]]:
    """
    Returns {layer_id: {module_name: (lora_A, lora_B)}} with CPU tensors.

    lora_A shape: (rank, in_features)
    lora_B shape: (out_features, rank)
    """
    # Pattern: base_model.model.model.layers.{i}.self_attn.{module}.lora_{A/B}.weight
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


class LoraManager:
    """
    Manages LoRA adapter weights for serving.

    Parameters
    ----------
    model_config:
        HuggingFace-style config object with hidden_size, num_attention_heads,
        num_key_value_heads, num_hidden_layers.
    max_loras:
        Maximum number of adapters resident in GPU memory simultaneously.
        (Non-pinned adapters are evicted LRU when this is exceeded.)
    max_lora_rank:
        Upper bound on rank across all adapters.  GPU buffers are allocated
        for this rank; adapters with smaller rank use a sub-slice.
    dtype:
        Data type for GPU buffers (should match the base model).
    device:
        GPU device.
    tp_rank:
        Tensor-parallel rank of this process.
    tp_size:
        Tensor-parallel world size.
    tp_group:
        torch.distributed ProcessGroup for all_reduce (only needed if
        tp_size > 1).
    """

    def __init__(
        self,
        model_config,
        max_loras: int,
        max_lora_rank: int,
        dtype: torch.dtype,
        device: torch.device,
        tp_rank: int = 0,
        tp_size: int = 1,
        tp_group=None,
    ) -> None:
        self.max_loras = max_loras
        self.max_lora_rank = max_lora_rank
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

        # Per-rank dimensions (column-parallel shards q/k/v; row-parallel shards o input)
        self.q_size_per_tp: int = (n_heads // tp_size) * head_dim
        self.kv_size_per_tp: int = max(1, n_kv // tp_size) * head_dim
        self.o_in_per_tp: int = (n_heads // tp_size) * head_dim  # = q_size_per_tp
        self.hidden_size: int = hidden

        # ── Slot management ───────────────────────────────────────────────
        # Slot 0 = "no adapter" (permanently zeroed).  Real adapters occupy
        # slots 1 .. max_loras.
        self._n_slots: int = max_loras + 1
        self._slot_to_name: list[str | None] = [None] * self._n_slots
        self._name_to_slot: dict[str, int] = {}
        self._lru: OrderedDict[str, None] = OrderedDict()  # name → None; oldest first

        # CPU weight cache: name → parsed layer weights
        self._cpu_cache: dict[
            str, dict[int, dict[str, tuple[torch.Tensor, torch.Tensor]]]
        ] = {}

        # Scaling per slot (float32 on GPU)
        self._scalings: torch.Tensor = torch.zeros(
            self._n_slots, dtype=torch.float32, device=device
        )

        # Integer adapter ID registry (Python-side, separate from slot IDs)
        self._name_to_id: dict[str, int] = {}
        self._id_to_name: dict[int, str] = {}
        self._next_id: int = 1

        # Pinned adapters (never evicted)
        self._pinned: set[str] = set()
        # Adapter name → filesystem path (for scaling lookup)
        self._adapter_paths: dict[str, str] = {}

        # ── GPU buffers ───────────────────────────────────────────────────
        self.A_buffers: dict[str, list[torch.Tensor]] = {}
        self.B_buffers: dict[str, list[torch.Tensor]] = {}
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

    # ── Public API ──────────────────────────────────────────────────────

    def load_adapter(self, name: str, path: str, pinned: bool = False) -> int:
        """Load a PEFT adapter from *path* and return its integer lora_id.

        The adapter weights are loaded to CPU.  GPU slot assignment happens
        lazily in :meth:`prepare_loras`.
        """
        if name in self._name_to_id:
            logger.warning("Adapter '%s' is already loaded; re-loading.", name)
            self._evict_by_name(name)

        adapter_path = path
        # Support adapter subdirectory layout
        import os

        safetensors = os.path.join(adapter_path, "adapter_model.safetensors")
        if not os.path.exists(safetensors):
            # Try the path as-is (maybe a direct .safetensors file)
            safetensors = path

        raw = _load_safetensors(safetensors)
        weights = _parse_adapter_weights(raw, self.n_layers)
        self._cpu_cache[name] = weights

        lora_id = self._next_id
        self._next_id += 1
        self._name_to_id[name] = lora_id
        self._id_to_name[lora_id] = name
        self._adapter_paths[name] = adapter_path  # store for scaling lookup
        if pinned:
            self._pinned.add(name)

        logger.info("Loaded adapter '%s' (lora_id=%d) from %s", name, lora_id, path)
        return lora_id

    def unload_adapter(self, name: str) -> None:
        """Remove an adapter from the manager and free its GPU slot."""
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

    def prepare_loras(self, lora_ids: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """Ensure all adapters in *lora_ids* are in GPU slots.

        Returns
        -------
        weight_indices : torch.Tensor  shape [len(lora_ids)], dtype=int64
            Per-request GPU slot index.  0 = base model (zero delta).
        scalings : torch.Tensor  shape [n_slots], dtype=float32
            Per-slot lora_alpha/r scaling factor.
        """
        weight_indices: list[int] = []
        for lid in lora_ids:
            if lid == 0:
                weight_indices.append(0)
                continue
            name = self._id_to_name.get(lid)
            if name is None:
                logger.warning("Unknown lora_id %d; treating as base model.", lid)
                weight_indices.append(0)
                continue
            slot = self._ensure_in_gpu(name)
            weight_indices.append(slot)
            # Mark recently used
            self._lru.move_to_end(name)

        return (
            torch.tensor(weight_indices, dtype=torch.int64, device=self.device),
            self._scalings,
        )

    # ── Per-layer LoRA application ───────────────────────────────────────

    def apply_qkv_lora(
        self,
        hidden_states: torch.Tensor,
        qkv: torch.Tensor,
        layer_id: int,
        weight_indices: torch.Tensor,
        scalings: torch.Tensor,
    ) -> torch.Tensor:
        """Add LoRA delta to the fused QKV output.

        hidden_states : [tokens, hidden_size]  (full, not sharded)
        qkv           : [tokens, q_size_per_tp + 2*kv_size_per_tp]
        weight_indices: [n_requests] → slot index per request
        scalings      : [n_slots]

        For column-parallel projections (q, k, v):
          - lora_A is FULL (not sharded)
          - lora_B is sharded by tp_rank (stored that way in the buffer)
        """
        tokens = hidden_states.shape[0]
        if tokens == 0:
            return qkv

        # weight_indices is already per-token (expanded by model_executor before
        # the forward pass).  Single-request decode still needs broadcast.
        w_idx = weight_indices
        if w_idx.shape[0] == 1 and tokens > 1:
            w_idx = w_idx.expand(tokens)

        q_delta = self._apply_col_parallel_lora(
            hidden_states, layer_id, "q_proj", w_idx, scalings
        )
        k_delta = self._apply_col_parallel_lora(
            hidden_states, layer_id, "k_proj", w_idx, scalings
        )
        v_delta = self._apply_col_parallel_lora(
            hidden_states, layer_id, "v_proj", w_idx, scalings
        )
        delta = torch.cat([q_delta, k_delta, v_delta], dim=-1)
        return qkv + delta

    def apply_o_lora(
        self,
        attn_output: torch.Tensor,
        o_output: torch.Tensor,
        layer_id: int,
        weight_indices: torch.Tensor,
        scalings: torch.Tensor,
    ) -> torch.Tensor:
        """Add LoRA delta to the o_proj output.

        attn_output : [tokens, q_size_per_tp]  (row-parallel input, sharded)
        o_output    : [tokens, hidden_size]     (before external all_reduce)

        For row-parallel projection (o):
          - lora_A is sharded along in_dim (matching attn_output's shard)
          - lora_B is FULL
          - A partial all_reduce is needed across TP ranks before applying B
        """
        tokens = attn_output.shape[0]
        if tokens == 0:
            return o_output

        w_idx = weight_indices
        if w_idx.shape[0] == 1 and tokens > 1:
            w_idx = w_idx.expand(tokens)

        o_delta = self._apply_row_parallel_lora(
            attn_output, layer_id, "o_proj", w_idx, scalings
        )
        return o_output + o_delta

    # ── Private helpers ──────────────────────────────────────────────────

    def _alloc_gpu_buffers(self) -> None:
        r = self.max_lora_rank
        h = self.hidden_size
        q = self.q_size_per_tp
        kv = self.kv_size_per_tp
        o_in = self.o_in_per_tp

        # Module → (A shape per slot, B shape per slot)
        shape_map = {
            "q_proj": ((r, h), (q, r)),  # column-parallel
            "k_proj": ((r, h), (kv, r)),  # column-parallel
            "v_proj": ((r, h), (kv, r)),  # column-parallel
            "o_proj": ((r, o_in), (h, r)),  # row-parallel; A sharded
        }

        for mod, (a_shape, b_shape) in shape_map.items():
            self.A_buffers[mod] = []
            self.B_buffers[mod] = []
            for _ in range(self.n_layers):
                A = torch.zeros(
                    self._n_slots, *a_shape, dtype=self.dtype, device=self.device
                )
                B = torch.zeros(
                    self._n_slots, *b_shape, dtype=self.dtype, device=self.device
                )
                self.A_buffers[mod].append(A)
                self.B_buffers[mod].append(B)

    def _ensure_in_gpu(self, name: str) -> int:
        """Return the GPU slot for *name*, loading it if necessary."""
        if name in self._name_to_slot:
            return self._name_to_slot[name]

        slot = self._find_free_slot(name)
        self._load_to_slot(name, slot)
        self._name_to_slot[name] = slot
        self._slot_to_name[slot] = name
        self._lru[name] = None  # track in LRU
        return slot

    def _find_free_slot(self, _requesting_name: str) -> int:
        """Find or evict a slot."""
        # Try an empty slot (skip slot 0 which is the "no lora" sentinel)
        for slot in range(1, self._n_slots):
            if self._slot_to_name[slot] is None:
                return slot

        # No empty slot — evict LRU non-pinned adapter
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
        """Copy CPU weights for *name* into GPU slot *slot*."""
        cpu_weights = self._cpu_cache[name]
        rank = self._get_rank_for(name)

        # Compute scaling from adapter_config.json if available
        scaling = self._get_scaling_for(name, rank)
        self._scalings[slot] = scaling

        for layer_id, modules in cpu_weights.items():
            for mod, (lora_A_full, lora_B_full) in modules.items():
                actual_rank = lora_A_full.shape[0]  # (rank, in_dim)
                lora_A_gpu = lora_A_full.to(device=self.device, dtype=self.dtype)
                lora_B_gpu = lora_B_full.to(device=self.device, dtype=self.dtype)

                # Shard for TP
                lora_A_shard, lora_B_shard = self._shard_weights(
                    mod, lora_A_gpu, lora_B_gpu
                )

                # Write into the pre-allocated buffer at this slot
                r = min(actual_rank, self.max_lora_rank)
                self.A_buffers[mod][layer_id][slot, :r].copy_(lora_A_shard[:r])
                self.B_buffers[mod][layer_id][slot, :, :r].copy_(lora_B_shard[:, :r])

        logger.debug("Loaded adapter '%s' into GPU slot %d (rank=%d)", name, slot, rank)

    def _get_rank_for(self, name: str) -> int:
        """Return the rank of the adapter's first layer's q_proj."""
        cpu_weights = self._cpu_cache.get(name, {})
        if cpu_weights and 0 in cpu_weights and "q_proj" in cpu_weights[0]:
            return cpu_weights[0]["q_proj"][0].shape[0]
        return self.max_lora_rank

    def _get_scaling_for(self, name: str, rank: int) -> float:
        """Read lora_alpha/r from adapter_config.json; default to 1.0."""
        import json
        import os

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
        """Shard A/B for tensor parallelism.

        Column-parallel (q, k, v):  A unsharded, B output-sharded
        Row-parallel    (o):        A input-sharded, B unsharded
        """
        if self.tp_size == 1:
            return lora_A, lora_B

        if module in ("q_proj", "k_proj", "v_proj"):
            # column-parallel: shard B along output dimension
            out_total = lora_B.shape[0]
            out_per = out_total // self.tp_size
            lora_B_shard = lora_B[self.tp_rank * out_per : (self.tp_rank + 1) * out_per]
            return lora_A, lora_B_shard
        else:
            # row-parallel (o_proj): shard A along input dimension
            in_total = lora_A.shape[1]
            in_per = in_total // self.tp_size
            lora_A_shard = lora_A[
                :, self.tp_rank * in_per : (self.tp_rank + 1) * in_per
            ]
            return lora_A_shard, lora_B

    def _evict_by_name(self, name: str) -> None:
        if name in self._name_to_slot:
            slot = self._name_to_slot.pop(name)
            self._slot_to_name[slot] = None
            # Zero out the slot
            for mod in _PEFT_MODULES:
                for layer_id in range(self.n_layers):
                    self.A_buffers[mod][layer_id][slot].zero_()
                    self.B_buffers[mod][layer_id][slot].zero_()
            self._scalings[slot] = 0.0
        self._lru.pop(name, None)

    def _apply_col_parallel_lora(
        self,
        x: torch.Tensor,
        layer_id: int,
        module: str,
        w_idx: torch.Tensor,
        scalings: torch.Tensor,
    ) -> torch.Tensor:
        """Compute LoRA delta for a column-parallel projection.

        x      : [tokens, hidden_size]
        A_buf  : [n_slots, max_rank, hidden_size]
        B_buf  : [n_slots, out_per_tp, max_rank]
        returns: [tokens, out_per_tp]
        """
        A_buf = self.A_buffers[module][layer_id]  # [slots, r, h]
        B_buf = self.B_buffers[module][layer_id]  # [slots, out, r]
        scale = scalings[w_idx]  # [tokens]

        # Gather per-token A/B rows
        A_sel = A_buf[w_idx]  # [tokens, r, h]
        B_sel = B_buf[w_idx]  # [tokens, out, r]

        # lora_a: [tokens, r]  = einsum('ti,tri->tr', x, A_sel)
        lora_a = torch.bmm(A_sel, x.unsqueeze(-1)).squeeze(-1)
        # lora_b: [tokens, out] = einsum('tri,ti->tr', B_sel, lora_a)
        delta = torch.bmm(B_sel, lora_a.unsqueeze(-1)).squeeze(-1)
        return delta * scale.unsqueeze(-1).to(delta.dtype)

    def _apply_row_parallel_lora(
        self,
        x_shard: torch.Tensor,
        layer_id: int,
        module: str,
        w_idx: torch.Tensor,
        scalings: torch.Tensor,
    ) -> torch.Tensor:
        """Compute LoRA delta for a row-parallel projection.

        x_shard: [tokens, in_per_tp]   (sharded input)
        A_buf  : [n_slots, max_rank, in_per_tp]
        B_buf  : [n_slots, hidden, max_rank]
        returns: [tokens, hidden]
        """
        A_buf = self.A_buffers[module][layer_id]
        B_buf = self.B_buffers[module][layer_id]
        scale = scalings[w_idx]

        A_sel = A_buf[w_idx]  # [tokens, r, in_per_tp]
        B_sel = B_buf[w_idx]  # [tokens, hidden, r]

        # Partial A output
        lora_a = torch.bmm(A_sel, x_shard.unsqueeze(-1)).squeeze(-1)  # [tokens, r]

        # All-reduce partial lora_a across TP
        if self.tp_size > 1 and self.tp_group is not None:
            dist.all_reduce(lora_a, group=self.tp_group)

        delta = torch.bmm(B_sel, lora_a.unsqueeze(-1)).squeeze(-1)  # [tokens, h]
        return delta * scale.unsqueeze(-1).to(delta.dtype)

    def set_adapter_scaling(self, name: str, scaling: float) -> None:
        """Override the scaling factor for a loaded adapter."""
        slot = self._name_to_slot.get(name)
        if slot is not None:
            self._scalings[slot] = scaling

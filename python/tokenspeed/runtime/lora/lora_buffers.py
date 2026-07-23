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

"""GPU-resident LoRA weight buffer layout and slot loading."""

from __future__ import annotations

import torch

from tokenspeed.runtime.lora.adapter_io import (
    LORA_HEAD_LAYER_ID,
    PEFT_HEAD_MODULE,
    AdapterWeights,
)

LORA_BUFFER_GROUPS = frozenset({"attn", "mlp", "moe", "lm_head"})


class LoraWeightBuffers:
    def __init__(
        self,
        *,
        n_layers: int,
        n_slots: int,
        max_lora_rank: int,
        hidden_size: int,
        q_size_per_tp: int,
        kv_size_per_tp: int,
        o_in_per_tp: int,
        intermediate_per_tp: int,
        vocab_per_tp: int,
        dtype: torch.dtype,
        device: torch.device,
        tp_rank: int,
        tp_size: int,
        buffer_groups: set[str] | frozenset[str] = LORA_BUFFER_GROUPS,
    ) -> None:
        self.n_layers = n_layers
        self.n_slots = n_slots
        self.max_lora_rank = max_lora_rank
        self.hidden_size = hidden_size
        self.q_size_per_tp = q_size_per_tp
        self.kv_size_per_tp = kv_size_per_tp
        self.o_in_per_tp = o_in_per_tp
        self.intermediate_per_tp = intermediate_per_tp
        self.vocab_per_tp = vocab_per_tp
        self.dtype = dtype
        self.device = device
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        unknown_groups = set(buffer_groups) - LORA_BUFFER_GROUPS
        if unknown_groups:
            raise ValueError(f"Unknown LoRA buffer groups: {sorted(unknown_groups)}")
        self.buffer_groups = frozenset(buffer_groups)
        self.enable_attn = "attn" in self.buffer_groups
        self.enable_mlp = "mlp" in self.buffer_groups
        self.enable_head = "lm_head" in self.buffer_groups

        self.qkv_A_buffers: list[torch.Tensor] = []
        self.qkv_B_buffers: list[torch.Tensor] = []
        self.o_A_buffers: list[torch.Tensor] = []
        self.o_B_buffers: list[torch.Tensor] = []
        self.gate_up_A_buffers: list[torch.Tensor] = []
        self.gate_up_B_buffers: list[torch.Tensor] = []
        self.down_A_buffers: list[torch.Tensor] = []
        self.down_B_buffers: list[torch.Tensor] = []
        # lm_head LoRA — single pair of buffers (not per-layer).
        # A: (n_slots, r, hidden)  — replicated across TP ranks.
        # B: (n_slots, vocab_per_tp, r) — column-parallel shard.
        self.lm_head_A_buffer: torch.Tensor
        self.lm_head_B_buffer: torch.Tensor

        self.qkv_output_offset = torch.tensor(
            [
                0,
                q_size_per_tp,
                q_size_per_tp + kv_size_per_tp,
                q_size_per_tp + 2 * kv_size_per_tp,
            ],
            dtype=torch.int32,
            device=device,
        )
        self.max_qkv_out_dim = max(q_size_per_tp, kv_size_per_tp)

        self.o_slice_offsets = torch.tensor(
            [0, hidden_size], dtype=torch.int32, device=device
        )
        self.gate_up_slice_offsets = torch.tensor(
            [0, intermediate_per_tp, 2 * intermediate_per_tp],
            dtype=torch.int32,
            device=device,
        )
        self.down_slice_offsets = torch.tensor(
            [0, hidden_size], dtype=torch.int32, device=device
        )

        self._alloc()

    def _alloc(self) -> None:
        r = self.max_lora_rank
        h = self.hidden_size
        q = self.q_size_per_tp
        kv = self.kv_size_per_tp
        o_in = self.o_in_per_tp
        i = self.intermediate_per_tp
        v = self.vocab_per_tp
        n = self.n_slots

        for _ in range(self.n_layers):
            if self.enable_attn:
                self.qkv_A_buffers.append(
                    torch.zeros((n, 3 * r, h), dtype=self.dtype, device=self.device)
                )
                self.qkv_B_buffers.append(
                    torch.zeros(
                        (n, q + 2 * kv, r), dtype=self.dtype, device=self.device
                    )
                )
                self.o_A_buffers.append(
                    torch.zeros((n, r, o_in), dtype=self.dtype, device=self.device)
                )
                self.o_B_buffers.append(
                    torch.zeros((n, h, r), dtype=self.dtype, device=self.device)
                )
            if self.enable_mlp:
                self.gate_up_A_buffers.append(
                    torch.zeros((n, 2 * r, h), dtype=self.dtype, device=self.device)
                )
                self.gate_up_B_buffers.append(
                    torch.zeros((n, 2 * i, r), dtype=self.dtype, device=self.device)
                )
                self.down_A_buffers.append(
                    torch.zeros((n, r, i), dtype=self.dtype, device=self.device)
                )
                self.down_B_buffers.append(
                    torch.zeros((n, h, r), dtype=self.dtype, device=self.device)
                )
        if self.enable_head:
            self.lm_head_A_buffer = torch.zeros(
                (n, r, h), dtype=self.dtype, device=self.device
            )
            self.lm_head_B_buffer = torch.zeros(
                (n, v, r), dtype=self.dtype, device=self.device
            )

    def load_adapter_to_slot(
        self,
        cpu_weights: AdapterWeights,
        slot: int,
        rank: int,
    ) -> None:
        for layer_id, modules in cpu_weights.items():
            if layer_id == LORA_HEAD_LAYER_ID:
                if PEFT_HEAD_MODULE in modules:
                    self._load_lm_head_to_slot(modules[PEFT_HEAD_MODULE], slot, rank)
                continue
            for mod, (lora_A_full, lora_B_full) in modules.items():
                if mod.startswith("experts."):
                    continue
                self._check_module_enabled(mod)
                lora_A_shard_cpu, lora_B_shard_cpu = self.shard_weights(
                    mod, lora_A_full, lora_B_full
                )
                r = min(lora_A_shard_cpu.shape[0], rank)
                lora_A_shard = lora_A_shard_cpu[:r].to(
                    device=self.device,
                    dtype=self.dtype,
                    non_blocking=True,
                )
                lora_B_shard = lora_B_shard_cpu[:, :r].to(
                    device=self.device,
                    dtype=self.dtype,
                    non_blocking=True,
                )

                if mod in ("q_proj", "k_proj", "v_proj"):
                    qkv_idx = ("q_proj", "k_proj", "v_proj").index(mod)
                    rank_off = qkv_idx * r
                    out_off, out_size = self.qkv_b_slice(mod)
                    self.qkv_A_buffers[layer_id][
                        slot, rank_off : rank_off + r, :
                    ].copy_(lora_A_shard, non_blocking=True)
                    self.qkv_B_buffers[layer_id][
                        slot, out_off : out_off + out_size, :r
                    ].copy_(lora_B_shard, non_blocking=True)
                elif mod == "o_proj":
                    self.o_A_buffers[layer_id][slot, :r, :].copy_(
                        lora_A_shard, non_blocking=True
                    )
                    self.o_B_buffers[layer_id][slot, :, :r].copy_(
                        lora_B_shard, non_blocking=True
                    )
                elif mod in ("gate_proj", "up_proj"):
                    gate_up_idx = 0 if mod == "gate_proj" else 1
                    rank_off = gate_up_idx * r
                    out_off = gate_up_idx * self.intermediate_per_tp
                    self.gate_up_A_buffers[layer_id][
                        slot, rank_off : rank_off + r, :
                    ].copy_(lora_A_shard, non_blocking=True)
                    self.gate_up_B_buffers[layer_id][
                        slot, out_off : out_off + self.intermediate_per_tp, :r
                    ].copy_(lora_B_shard, non_blocking=True)
                else:
                    self.down_A_buffers[layer_id][slot, :r, :].copy_(
                        lora_A_shard, non_blocking=True
                    )
                    self.down_B_buffers[layer_id][slot, :, :r].copy_(
                        lora_B_shard, non_blocking=True
                    )

    def _load_lm_head_to_slot(
        self,
        ab: tuple[torch.Tensor, torch.Tensor],
        slot: int,
        rank: int,
    ) -> None:
        if not self.enable_head:
            raise ValueError(
                "Adapter targets lm_head, but LoRA buffer group 'head' is disabled."
            )
        lora_A_full, lora_B_full = ab
        lora_A_cpu, lora_B_cpu = self.shard_weights(
            PEFT_HEAD_MODULE, lora_A_full, lora_B_full
        )
        r = min(lora_A_cpu.shape[0], rank)
        self.lm_head_A_buffer[slot, :r, :].copy_(
            lora_A_cpu[:r].to(device=self.device, dtype=self.dtype, non_blocking=True),
            non_blocking=True,
        )
        self.lm_head_B_buffer[slot, :, :r].copy_(
            lora_B_cpu[:, :r].to(
                device=self.device, dtype=self.dtype, non_blocking=True
            ),
            non_blocking=True,
        )

    def zero_slot(self, slot: int) -> None:
        if self.enable_attn:
            for layer_id in range(self.n_layers):
                self.qkv_A_buffers[layer_id][slot].zero_()
                self.qkv_B_buffers[layer_id][slot].zero_()
                self.o_A_buffers[layer_id][slot].zero_()
                self.o_B_buffers[layer_id][slot].zero_()
        if self.enable_mlp:
            for layer_id in range(self.n_layers):
                self.gate_up_A_buffers[layer_id][slot].zero_()
                self.gate_up_B_buffers[layer_id][slot].zero_()
                self.down_A_buffers[layer_id][slot].zero_()
                self.down_B_buffers[layer_id][slot].zero_()
        if self.enable_head:
            self.lm_head_A_buffer[slot].zero_()
            self.lm_head_B_buffer[slot].zero_()

    def _check_module_enabled(self, module: str) -> None:
        if module in ("q_proj", "k_proj", "v_proj", "o_proj"):
            if not self.enable_attn:
                raise ValueError(
                    f"Adapter targets {module}, but LoRA buffer group 'attn' "
                    "is disabled."
                )
            return
        if module in ("gate_proj", "up_proj", "down_proj"):
            if not self.enable_mlp:
                raise ValueError(
                    f"Adapter targets {module}, but LoRA buffer group 'mlp' "
                    "is disabled."
                )
            return
        if module == PEFT_HEAD_MODULE:
            if not self.enable_head:
                raise ValueError(
                    "Adapter targets lm_head, but LoRA buffer group 'head' "
                    "is disabled."
                )
            return
        raise ValueError(f"Unsupported dense LoRA module: {module}")

    def qkv_b_slice(self, module: str) -> tuple[int, int]:
        """Return ``(offset, size)`` of a projection inside fused QKV B."""
        if module == "q_proj":
            return 0, self.q_size_per_tp
        if module == "k_proj":
            return self.q_size_per_tp, self.kv_size_per_tp
        return self.q_size_per_tp + self.kv_size_per_tp, self.kv_size_per_tp

    def shard_weights(
        self,
        module: str,
        lora_A: torch.Tensor,
        lora_B: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.tp_size == 1:
            return lora_A, lora_B
        # Column-parallel (attn q/k/v, MLP gate/up, lm_head): shard B along output dim.
        if module in (
            "q_proj",
            "k_proj",
            "v_proj",
            "gate_proj",
            "up_proj",
            PEFT_HEAD_MODULE,
        ):
            out_total = lora_B.shape[0]
            out_per = out_total // self.tp_size
            return (
                lora_A,
                lora_B[self.tp_rank * out_per : (self.tp_rank + 1) * out_per],
            )
        # Row-parallel (attn o_proj, MLP down_proj): shard A along input dim.
        in_total = lora_A.shape[1]
        in_per = in_total // self.tp_size
        return (
            lora_A[:, self.tp_rank * in_per : (self.tp_rank + 1) * in_per],
            lora_B,
        )

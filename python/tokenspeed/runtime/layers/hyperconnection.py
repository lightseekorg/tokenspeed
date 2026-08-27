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

"""Hyper-connection layers used by Qwen4-Exp."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from tokenspeed_kernel.platform import pdl_enabled
from torch import nn


@dataclass(frozen=True)
class HyperConnectionConfig:
    """Shape and normalization parameters for a hyper-connection stream."""

    hc_count: int
    hidden_size: int
    hc_lowrank: int
    rms_norm_eps: float = 1e-6
    params_dtype: torch.dtype = torch.bfloat16
    hc_per_branch_norm: bool = True


def _matching_rows(base: torch.Tensor, derived: torch.Tensor):
    """Row offset of ``derived`` inside ``base``, or ``None`` when unrelated.

    Attention communication either hands the residual back untouched (all-reduce)
    or replaces it with a contiguous row slice of itself (reduce-scatter), so any
    row-wise transform of ``base`` stays valid for ``derived`` in both cases.
    """
    if derived is base:
        return 0
    if derived.dtype is not base.dtype or derived.device != base.device:
        return None
    if derived.shape[1:] != base.shape[1:] or derived.stride() != base.stride():
        return None
    if derived.untyped_storage().data_ptr() != base.untyped_storage().data_ptr():
        return None
    row_stride = base.stride(0)
    delta = derived.storage_offset() - base.storage_offset()
    if row_stride <= 0 or delta < 0 or delta % row_stride:
        return None
    start = delta // row_stride
    if start + derived.shape[0] > base.shape[0]:
        return None
    return start


class GroupedGemmaRMSNorm(nn.Module):
    """Gemma RMSNorm with optional independently-normalized feature groups."""

    def __init__(self, hidden_size: int, eps: float, group_size: int | None = None):
        super().__init__()
        if group_size is not None and hidden_size % group_size:
            raise ValueError("hidden_size must be divisible by group_size")
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps
        self.group_size = group_size
        self.register_buffer("gemma_weight", self.weight.data + 1.0, persistent=False)
        self.weight.weight_loader = self._weight_loader

    def _weight_loader(self, param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
        if param.size() != loaded_weight.size():
            raise ValueError(
                f"Shape mismatch: {param.size()} != {loaded_weight.size()}."
            )
        param.data.copy_(loaded_weight)
        self.gemma_weight = param.data + 1.0

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x_float = x.float()
        if self.group_size is None:
            variance = x_float.square().mean(dim=-1, keepdim=True)
            normalized = x_float * torch.rsqrt(variance + self.variance_epsilon)
        else:
            grouped = x_float.unflatten(-1, (-1, self.group_size))
            variance = grouped.square().mean(dim=-1, keepdim=True)
            normalized = (
                grouped * torch.rsqrt(variance + self.variance_epsilon)
            ).flatten(-2)
        return (normalized * self.gemma_weight.float()).to(input_dtype)

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        from tokenspeed.runtime.layers.attention.linear.layernorm_gated import (
            rmsnorm_fn,
        )

        return rmsnorm_fn(
            x,
            self.gemma_weight,
            None,
            z=None,
            eps=self.variance_epsilon,
            group_size=self.group_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_cuda:
            return self.forward_cuda(x)
        return self.forward_native(x)


@triton.jit
def _hc_mix_epilogue_kernel(
    gate_ptr,
    normalized_ptr,
    out_ptr,
    gate_row_stride,
    normalized_row_stride,
    out_row_stride,
    hidden_size,
    HC_COUNT: tl.constexpr,
    BLOCK: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    """Sigmoid gate, per-branch weighting and branch mean in one pass.

    Each program owns one token and one slice of ``hidden_size``, so the widened
    ``[..., hc_count * hidden_size]`` sigmoid and product tensors the eager path
    materializes stay in registers and only the mixed row reaches memory.
    """

    row = tl.program_id(0)
    offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < hidden_size
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()
    mixed = tl.zeros([BLOCK], dtype=tl.float32)
    for branch in tl.static_range(HC_COUNT):
        column = branch * hidden_size + offsets
        gate = tl.load(
            gate_ptr + row * gate_row_stride + column, mask=mask, other=0.0
        ).to(tl.float32)
        value = tl.load(
            normalized_ptr + row * normalized_row_stride + column, mask=mask, other=0.0
        ).to(tl.float32)
        mixed += tl.sigmoid(gate) * value
    tl.store(out_ptr + row * out_row_stride + offsets, mixed / HC_COUNT, mask=mask)
    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _hc_combine_kernel(
    hyper_ptr,
    block_ptr,
    inject_ptr,
    out_ptr,
    hyper_row_stride,
    block_row_stride,
    inject_row_stride,
    out_row_stride,
    hidden_size,
    HC_COUNT: tl.constexpr,
    BLOCK: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    """Gate one sublayer output and add it into every residual branch.

    The sublayer row is loaded once and reused for all ``hc_count`` branches, so
    the broadcast product the eager path materializes at the full residual width
    never reaches memory.
    """

    row = tl.program_id(0)
    offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < hidden_size
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()
    value = tl.load(
        block_ptr + row * block_row_stride + offsets, mask=mask, other=0.0
    ).to(tl.float32)
    for branch in tl.static_range(HC_COUNT):
        logit = tl.load(inject_ptr + row * inject_row_stride + branch).to(tl.float32)
        column = branch * hidden_size + offsets
        residual = tl.load(
            hyper_ptr + row * hyper_row_stride + column, mask=mask, other=0.0
        ).to(tl.float32)
        tl.store(
            out_ptr + row * out_row_stride + column,
            residual + value * 2.0 * tl.sigmoid(logit),
            mask=mask,
        )
    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


class GatedResidualSimple(nn.Module):
    """Low-rank gated mixing and gated sublayer injection for HC streams."""

    def __init__(
        self,
        config: HyperConnectionConfig,
        *,
        use_mix: bool = True,
        use_combine: bool = True,
    ) -> None:
        super().__init__()
        self.config = config
        self.hc_count = config.hc_count
        self.hidden_size = config.hidden_size
        self.hc_lowrank = config.hc_lowrank
        self.use_mix = use_mix
        self.use_combine = use_combine
        hc_size = self.hc_count * self.hidden_size
        group_size = self.hidden_size if config.hc_per_branch_norm else None
        norm_size = hc_size if config.hc_per_branch_norm else self.hidden_size
        self.hc_norm = GroupedGemmaRMSNorm(
            norm_size, config.rms_norm_eps, group_size=group_size
        )
        # The low-rank mix gate and the inject logits both read the whole
        # normalized stream, so fuse their projections into one GEMM and fetch it
        # once: rows [0, hc_lowrank) hold the mix-down weight and the next
        # hc_count rows the inject weight. Checkpoint shards are routed there by
        # the stacked mapping in load_qwen4_exp_weights. Both projections are
        # scaled by 1 / hc_count, folded into the weight at load time, which is
        # exact for a power-of-two hc_count.
        projection_rows = (config.hc_lowrank if use_mix else 0) + (
            self.hc_count if use_combine else 0
        )
        self.mix_inject_proj = (
            nn.Linear(hc_size, projection_rows, bias=False) if projection_rows else None
        )
        if self.mix_inject_proj is not None:
            self.mix_inject_proj.weight.weight_loader = self._load_projection_shard
        self.input_mix_weight_up = (
            nn.Linear(config.hc_lowrank, hc_size, bias=False) if use_mix else None
        )

    def _load_projection_shard(
        self,
        param: torch.Tensor,
        loaded_weight: torch.Tensor,
        shard_id: str,
    ) -> None:
        """Route a checkpoint projection into the fused weight's row range."""
        mix_rows = self.hc_lowrank if self.use_mix else 0
        inject_rows = self.hc_count if self.use_combine else 0
        offsets = {"mix": (0, mix_rows), "inject": (mix_rows, inject_rows)}
        start, size = offsets[shard_id]
        if size == 0:
            # This stream does not use that half of the projection.
            return
        if loaded_weight.shape[0] != size or param.shape[0] != mix_rows + inject_rows:
            raise ValueError(
                f"hyper-connection {shard_id} shard shape mismatch: param "
                f"{tuple(param.shape)}, loaded {tuple(loaded_weight.shape)}"
            )
        param.data[start : start + size].copy_(
            (loaded_weight / self.hc_count).to(param.device, param.dtype)
        )

    def _split_projection(self, projected: torch.Tensor):
        """Split the fused projection into mix gate and inject logits."""
        if not self.use_combine:
            return projected, None
        if not self.use_mix:
            return None, projected
        return projected[..., : self.hc_lowrank], projected[..., self.hc_lowrank :]

    def _inject_logits(self, normalized: torch.Tensor):
        """Inject logits alone, for residual rows ``mix`` never saw."""
        if not self.use_combine:
            return None
        weight = self.mix_inject_proj.weight
        if self.use_mix:
            weight = weight[self.hc_lowrank :]
        return F.linear(normalized, weight)

    def _normalize(self, hyper_input: torch.Tensor) -> torch.Tensor:
        if self.config.hc_per_branch_norm:
            return self.hc_norm(hyper_input)
        return self.hc_norm(
            hyper_input.unflatten(-1, (self.hc_count, self.hidden_size))
        ).flatten(-2)

    def norm_for(self, value: torch.Tensor, residuals):
        """Residual pair for ``value``, reusing the norm from ``residuals``.

        The norm is row-wise, so a residual that communication left untouched or
        sliced along rows can borrow the tensor :meth:`mix` already produced
        instead of paying for a second full-width pass.
        """
        hyper_input, normalized, inject_logits = residuals
        start = _matching_rows(hyper_input, value)
        if start is None:
            fresh = self._normalize(value)
            return value, fresh, self._inject_logits(fresh)
        if start == 0 and value.shape[0] == hyper_input.shape[0]:
            return value, normalized, inject_logits
        rows = slice(start, start + value.shape[0])
        return value, normalized[rows], inject_logits[rows]

    def _mix_epilogue_torch(
        self, gate: torch.Tensor, normalized: torch.Tensor
    ) -> torch.Tensor:
        weights = torch.sigmoid(gate).unflatten(-1, (self.hc_count, self.hidden_size))
        branches = normalized.unflatten(-1, (self.hc_count, self.hidden_size))
        return (weights * branches).mean(dim=-2).to(self.config.params_dtype)

    def _mix_epilogue_cuda(
        self, gate: torch.Tensor, normalized: torch.Tensor
    ) -> torch.Tensor:
        rows = gate.shape[0]
        mixed = torch.empty(
            (rows, self.hidden_size),
            dtype=self.config.params_dtype,
            device=gate.device,
        )
        # Row strides let the fused projection's split views feed the kernel
        # without a contiguity copy; only the last dim must be dense.
        if gate.stride(-1) != 1:
            gate = gate.contiguous()
        if normalized.stride(-1) != 1:
            normalized = normalized.contiguous()
        block = min(triton.next_power_of_2(self.hidden_size), 1024)
        use_pdl = pdl_enabled()
        _hc_mix_epilogue_kernel[(rows, triton.cdiv(self.hidden_size, block))](
            gate,
            normalized,
            mixed,
            gate.stride(0),
            normalized.stride(0),
            mixed.stride(0),
            self.hidden_size,
            HC_COUNT=self.hc_count,
            BLOCK=block,
            ENABLE_PDL=use_pdl,
            **({"launch_pdl": True} if use_pdl else {}),
        )
        return mixed

    def mix(self, hyper_input: torch.Tensor):
        """Mix ``hc_count`` residual branches into one sublayer input.

        Returns:
            A pair containing the mixed ``[..., hidden_size]`` input and the
            residual tuple required by :meth:`combine`.
        """
        expected = self.hc_count * self.hidden_size
        if hyper_input.shape[-1] != expected:
            raise ValueError(
                f"hyper input width must be {expected}, got {hyper_input.shape[-1]}"
            )
        normalized = self._normalize(hyper_input)
        if hyper_input.shape[0] == 0:
            mixed = hyper_input.new_empty((*hyper_input.shape[:-1], self.hidden_size))
            inject_logits = hyper_input.new_empty(
                (*hyper_input.shape[:-1], self.hc_count)
            )
            return mixed, (hyper_input, normalized, inject_logits)
        gate, inject_logits = self._split_projection(self.mix_inject_proj(normalized))
        gate = self.input_mix_weight_up(F.silu(gate))
        if gate.is_cuda and gate.dim() == 2:
            mixed = self._mix_epilogue_cuda(gate, normalized)
        else:
            mixed = self._mix_epilogue_torch(gate, normalized)
        return mixed, (
            hyper_input,
            normalized,
            inject_logits,
        )

    def _combine_torch(
        self,
        block_output: torch.Tensor,
        hyper_input: torch.Tensor,
        inject_logits: torch.Tensor,
    ) -> torch.Tensor:
        inject = 2 * torch.sigmoid(inject_logits)
        combined = hyper_input.unflatten(
            -1, (self.hc_count, self.hidden_size)
        ) + block_output.unsqueeze(-2) * inject.unsqueeze(-1)
        return combined.flatten(-2).to(self.config.params_dtype)

    def _combine_cuda(
        self,
        block_output: torch.Tensor,
        hyper_input: torch.Tensor,
        inject_logits: torch.Tensor,
    ) -> torch.Tensor:
        rows = block_output.shape[0]
        combined = torch.empty(
            (rows, self.hc_count * self.hidden_size),
            dtype=self.config.params_dtype,
            device=block_output.device,
        )
        # Row strides let sliced residuals and the fused projection's inject view
        # feed the kernel without a contiguity copy; only the last dim must be
        # dense.
        if hyper_input.stride(-1) != 1:
            hyper_input = hyper_input.contiguous()
        if block_output.stride(-1) != 1:
            block_output = block_output.contiguous()
        if inject_logits.stride(-1) != 1:
            inject_logits = inject_logits.contiguous()
        block = min(triton.next_power_of_2(self.hidden_size), 1024)
        use_pdl = pdl_enabled()
        _hc_combine_kernel[(rows, triton.cdiv(self.hidden_size, block))](
            hyper_input,
            block_output,
            inject_logits,
            combined,
            hyper_input.stride(0),
            block_output.stride(0),
            inject_logits.stride(0),
            combined.stride(0),
            self.hidden_size,
            HC_COUNT=self.hc_count,
            BLOCK=block,
            ENABLE_PDL=use_pdl,
            **({"launch_pdl": True} if use_pdl else {}),
        )
        return combined

    def combine(self, block_output: torch.Tensor, residuals) -> torch.Tensor:
        """Inject one sublayer output back into every residual branch."""
        hyper_input, _, inject_logits = residuals
        if block_output.shape[0] == 0:
            return hyper_input.to(self.config.params_dtype)
        if block_output.is_cuda and block_output.dim() == 2:
            return self._combine_cuda(block_output, hyper_input, inject_logits)
        return self._combine_torch(block_output, hyper_input, inject_logits)


__all__ = [
    "GatedResidualSimple",
    "GroupedGemmaRMSNorm",
    "HyperConnectionConfig",
]

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
from tokenspeed_kernel import (
    gated_residual_combine,
    gated_residual_mix,
    grouped_gemma_rmsnorm,
    prepare_gated_residual_weight_cache,
)
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

    def __post_init__(self) -> None:
        if self.hc_count <= 1:
            raise ValueError("hc_count must be greater than one")
        if self.hidden_size <= 0 or self.hc_lowrank <= 0:
            raise ValueError("hidden_size and hc_lowrank must be positive")


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
    """GPU Gemma RMSNorm with optional independently-normalized feature groups."""

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return grouped_gemma_rmsnorm(
            x,
            self.weight,
            self.group_size,
            self.variance_epsilon,
        )


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
        # scaled by 1 / hc_count. Fold that scale only when it is an exact
        # binary exponent change; non-power-of-two counts scale projection
        # results at runtime instead of quantizing the checkpoint weights.
        self._fold_projection_scale = not (self.hc_count & (self.hc_count - 1))
        self._projection_scale = (
            1.0 if self._fold_projection_scale else 1.0 / self.hc_count
        )
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
        if self.input_mix_weight_up is not None:
            self.input_mix_weight_up.weight.weight_loader = self._load_up_weight

    def _load_up_weight(self, param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
        """Load the fixed-shape projection and prepare its derived GPU weight."""
        if param.shape != loaded_weight.shape:
            raise ValueError(
                f"hyper-connection up weight shape mismatch: param "
                f"{tuple(param.shape)}, loaded {tuple(loaded_weight.shape)}"
            )
        param.data.copy_(loaded_weight)
        prepare_gated_residual_weight_cache(param, self.hc_lowrank)

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
        weight = (
            loaded_weight / self.hc_count
            if self._fold_projection_scale
            else loaded_weight
        )
        param.data[start : start + size].copy_(weight.to(param.device, param.dtype))

    def _inject_logits(self, normalized: torch.Tensor):
        """Inject logits alone, for residual rows ``mix`` never saw."""
        if not self.use_combine:
            return None
        weight = self.mix_inject_proj.weight
        if self.use_mix:
            weight = weight[self.hc_lowrank :]
        logits = F.linear(normalized, weight)
        if self._projection_scale != 1.0:
            logits = logits * self._projection_scale
        return logits

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
        mixed, inject_logits = gated_residual_mix(
            normalized,
            self.mix_inject_proj.weight,
            self.input_mix_weight_up.weight,
            self.hc_count,
            self.hidden_size,
            self.hc_lowrank,
            projection_scale=self._projection_scale,
        )
        mixed = mixed.to(self.config.params_dtype)
        return mixed, (
            hyper_input,
            normalized,
            inject_logits,
        )

    def combine(self, block_output: torch.Tensor, residuals) -> torch.Tensor:
        """Inject one sublayer output back into every residual branch."""
        hyper_input, _, inject_logits = residuals
        return gated_residual_combine(
            block_output,
            hyper_input,
            inject_logits,
            self.hc_count,
            self.hidden_size,
        ).to(self.config.params_dtype)


__all__ = [
    "GatedResidualSimple",
    "GroupedGemmaRMSNorm",
    "HyperConnectionConfig",
]

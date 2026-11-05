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

"""Ascend graph-safe sampling primitives provided by flash-npu-kernel."""

from __future__ import annotations

from dataclasses import dataclass

import torch

try:
    import flash_ops  # noqa: F401
except ImportError as exc:
    raise ImportError(
        "Ascend full sampling requires flash_ops built with the npu_sample "
        "operators (fused_topk_topp_renorm and sorted_topk_topp_renorm)"
    ) from exc

for _name in (
    "fused_topk_topp_renorm",
    "npu_sampling_workspace",
    "npu_sample_from_probs",
    "npu_verify_chain_greedy",
    "npu_chain_speculative_sampling_target_only",
    "npu_min_p_renorm",
):
    if not hasattr(torch.ops.custom, _name):
        raise RuntimeError(
            f"Missing custom::{_name}; rebuild flash_ops with sampling operators"
        )


@dataclass(frozen=True)
class SamplingWorkspace:
    """Retain the operator-allocated scratch tensors across graph replay."""

    partial_sums: torch.Tensor
    selected_blocks: torch.Tensor
    residuals: torch.Tensor
    rejected_ids: torch.Tensor
    sample_positions: torch.Tensor

    @classmethod
    def create(cls, max_rows: int, vocab_size: int, device: torch.device | str):
        return cls(
            *torch.ops.custom.npu_sampling_workspace(
                max_rows, vocab_size, torch.device(device)
            )
        )


def fused_topk_topp_renorm(probs, top_ks, top_ps, *, out=None):
    """Filter each row through the native top-k/top-p operator."""
    if out is not None and (
        out.shape != probs.shape
        or out.dtype != torch.float32
        or out.device != probs.device
    ):
        raise ValueError(
            "out must match probs in shape and device and have float32 dtype"
        )
    result = torch.ops.custom.fused_topk_topp_renorm(probs, top_ks, top_ps)
    if out is not None:
        out.copy_(result)
        return out
    return result


def sample_from_probs(probs, coins, workspace, *, out=None):
    """Sample using caller-provided coins, outputs and retained scratch."""
    if out is None:
        out = torch.empty((probs.shape[0],), dtype=torch.int32, device=probs.device)
    return torch.ops.custom.npu_sample_from_probs(
        probs,
        coins,
        workspace.partial_sums,
        workspace.selected_blocks,
        workspace.residuals,
        workspace.rejected_ids,
        workspace.sample_positions,
        out,
    )


def chain_speculative_sampling_target_only(
    predicts,
    accept_index,
    accept_token_num,
    candidates,
    uniform_samples,
    uniform_samples_for_final_sampling,
    target_probs,
    workspace,
    *,
    threshold_single=1.0,
    threshold_acc=1.0,
):
    """Verify a chain through the packaged Torch operator with caller-owned outputs."""
    torch.ops.custom.npu_chain_speculative_sampling_target_only(
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        uniform_samples,
        uniform_samples_for_final_sampling,
        target_probs,
        workspace.partial_sums,
        workspace.selected_blocks,
        workspace.residuals,
        workspace.rejected_ids,
        workspace.sample_positions,
        threshold_single,
        threshold_acc,
    )


def verify_chain_greedy(
    predicts, accept_index, accept_token_num, candidates, target_predict
):
    """Verify the greedy prefix into the caller-owned output tensors."""
    torch.ops.custom.npu_verify_chain_greedy(
        predicts, accept_index, accept_token_num, candidates, target_predict
    )


min_p_renorm_prob = torch.ops.custom.npu_min_p_renorm


__all__ = [
    "SamplingWorkspace",
    "chain_speculative_sampling_target_only",
    "fused_topk_topp_renorm",
    "min_p_renorm_prob",
    "sample_from_probs",
    "verify_chain_greedy",
]

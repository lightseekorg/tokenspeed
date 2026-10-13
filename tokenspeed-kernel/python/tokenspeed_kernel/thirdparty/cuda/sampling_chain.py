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

"""Speculative decoding chain sampling ops: verify_chain_greedy, chain_speculative_sampling_target_only."""

import functools
from pathlib import Path

import torch
from tokenspeed_kernel.platform import pdl_enabled


@functools.cache
def _load_sampling_chain_module():
    import tvm_ffi

    objs_dir = Path(__file__).parent / "objs" / "sampling_chain"
    so_path = objs_dir / "sampling_chain.so"
    if not so_path.exists():
        raise RuntimeError(
            f"tokenspeed_kernel sampling_chain library not found at {so_path}. "
            "Run: pip install -e tokenspeed_kernel/python/"
        )
    return tvm_ffi.load_module(str(so_path))


def verify_chain_greedy(
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    target_predict: torch.Tensor,
    batch_size: int,
    num_draft_tokens: int,
    enable_pdl: bool | None = None,
) -> None:
    enable_pdl = pdl_enabled() if enable_pdl is None else enable_pdl
    _load_sampling_chain_module().verify_chain_greedy(
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        target_predict,
        int(batch_size),
        int(num_draft_tokens),
        enable_pdl,
    )


def chain_speculative_sampling_target_only(
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    uniform_samples: torch.Tensor,
    uniform_samples_for_final_sampling: torch.Tensor,
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor | None = None,
    threshold_single: float = 1.0,
    threshold_acc: float = 1.0,
    deterministic: bool = True,
    enable_pdl: bool | None = None,
    *,
    use_draft_prob: bool,
    reject_draft_prob_threshold: float,
) -> None:
    """Chain speculative verification with one of two accept rules.

    Args:
        predicts: ``[bs * N]`` int32 output tokens, written for the accepted
            prefix and the slot after it.
        accept_index: ``[bs, N]`` int32 output positions, ``-1`` where unused.
        accept_token_num: ``[bs]`` int32 accepted draft counts (bonus token
            excluded).
        candidates: ``[bs, N]`` int32 chains; column 0 is the verified token,
            columns ``1..N-1`` the drafts.
        uniform_samples: ``[bs, N]`` fp32 accept coins.
        uniform_samples_for_final_sampling: ``[bs]`` fp32 residual coins.
        target_probs: ``[bs, N, V]`` fp32 target distributions.
        draft_probs: ``[bs, N, V]`` fp32 draft distributions. Under
            ``use_draft_prob`` row ``i`` is the distribution candidate
            ``i + 1`` was sampled from (required); otherwise it is an optional
            residual subtrahend and ``None`` means all zeros (no GMEM traffic).
        threshold_single: Target-only rule: accept outright at or above this
            target probability.
        threshold_acc: Target-only rule: accept with probability
            ``target_prob / threshold_acc``.
        deterministic: Fixed-order block scan for the residual draw.
        enable_pdl: Programmatic dependent launch; ``None`` takes the platform
            default.
        use_draft_prob: Standard rejection sampling (``coin * q(x) < p(x)``,
            residual ``norm(relu(p - q))``) instead of the target-only rule.
        reject_draft_prob_threshold: ``draft_probs`` entries above this are
            the "no recorded proposal" sentinel: the candidate is rejected and
            the row samples the full target. The binding rejects values below
            1.0 (a real probability would read as the sentinel); the serving
            layer validates the full usable range once at server-args time.
    """
    if use_draft_prob and draft_probs is None:
        raise ValueError(
            "chain_speculative_sampling_target_only: use_draft_prob requires the "
            "recorded draft_probs"
        )
    enable_pdl = pdl_enabled() if enable_pdl is None else enable_pdl
    _load_sampling_chain_module().chain_speculative_sampling_target_only(
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        uniform_samples,
        uniform_samples_for_final_sampling,
        target_probs,
        draft_probs,
        float(threshold_single),
        float(threshold_acc),
        bool(deterministic),
        bool(use_draft_prob),
        float(reject_draft_prob_threshold),
        enable_pdl,
    )

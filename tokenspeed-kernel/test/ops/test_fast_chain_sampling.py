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

"""Statistical correctness for the opt-in fast chain-spec sampling path.

The runtime opt-in (TOKENSPEED_SPEC_FAST_CHAIN_SAMPLING=1, see
`tokenspeed/runtime/sampling/backends/flashinfer.py`) replaces the sequential

    top_k_renorm_prob -> top_p_renorm_prob -> chain_speculative_sampling_target_only

verify chain with

    top_k_top_p_sampling_from_logits(filter_apply_order="joint") -> verify_chain_greedy

These three properties must hold (the third one is the explicit caveat):

1. The fast path's empirical output matches the joint-filter analytic
   reference within statistical noise (joint-mode top-k AND top-p computed
   on the original distribution).
2. The original sequential path's empirical output matches the sequential
   analytic reference within statistical noise (top-k renormalize then top-p
   on renormalized distribution).
3. The two filter modes are *not* identical -- in typical configs they
   differ by a small but non-zero total variation distance (~5%). The fast
   path therefore aligns verify with plain-decode's filter semantics rather
   than preserving bit-distribution-equivalence with the sequential path.
"""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.sampling.cuda import (
    chain_speculative_sampling_target_only,
    verify_chain_greedy,
)
from tokenspeed_kernel.ops.sampling.flashinfer import (
    top_k_renorm_prob,
    top_k_top_p_sampling_from_logits,
    top_p_renorm_prob,
)

# Small enough to run in a few seconds on a single GPU.
_VOCAB = 1024
_TOP_K = 50
_TOP_P = 0.95
_BATCH = 4096
_NUM_CHUNKS = 64  # → 262,144 trials per path
_MODE_NOISE_FLOOR = 0.02
_JOINT_VS_SEQUENTIAL_MAX_TV = 0.10


def _total_variation(p: torch.Tensor, q: torch.Tensor) -> float:
    return 0.5 * (p - q).abs().sum().item()


def _sequential_filter_reference(
    probs: torch.Tensor, top_k: int, top_p: float
) -> torch.Tensor:
    """Sequential top_k_renorm then top_p_renorm (the original spec verify)."""
    top_k_t = torch.tensor([top_k], dtype=torch.int32, device=probs.device)
    top_p_t = torch.tensor([top_p], dtype=torch.float32, device=probs.device)
    ref = top_k_renorm_prob(probs.unsqueeze(0), top_k_t)
    ref = top_p_renorm_prob(ref, top_p_t, is_deterministic=True)
    return ref[0]


def _joint_filter_reference(
    probs: torch.Tensor, top_k: int, top_p: float
) -> torch.Tensor:
    """flashinfer joint mode: top-k AND top-p judged on the *original* distribution."""
    vocab = probs.size(0)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=0)
    in_top_p = cumsum <= top_p
    # Include the first token that crosses top_p so retained mass >= top_p
    # (matches the flashinfer convention).
    last_in = int(in_top_p.sum().item())
    if last_in < vocab:
        in_top_p[last_in] = True
    in_top_k = torch.arange(vocab, device=probs.device) < top_k
    joint_mask = in_top_k & in_top_p
    kept_sorted = torch.where(joint_mask, sorted_probs, torch.zeros_like(sorted_probs))
    kept_sorted = kept_sorted / kept_sorted.sum()
    ref = torch.zeros(vocab, device=probs.device)
    ref[sorted_idx] = kept_sorted
    return ref


def _accumulate(samples: torch.Tensor, counts: torch.Tensor) -> None:
    ones = torch.ones_like(samples, dtype=torch.float32)
    counts.scatter_add_(0, samples.long(), ones)


@pytest.fixture
def setup(device: str):
    """Single fixed target distribution + batched expanded scalars."""
    torch.manual_seed(0)
    logits = torch.randn(1, _VOCAB, device=device) * 2.5
    target_probs_raw = torch.softmax(logits, dim=-1).contiguous()
    ref_seq = _sequential_filter_reference(target_probs_raw[0], _TOP_K, _TOP_P)
    ref_joint = _joint_filter_reference(target_probs_raw[0], _TOP_K, _TOP_P)
    draft_token = int(ref_seq.argmax().item())
    target_probs_raw_bs = target_probs_raw.expand(_BATCH, -1).contiguous()
    top_k_t_bs = torch.full((_BATCH,), _TOP_K, dtype=torch.int32, device=device)
    top_p_t_bs = torch.full((_BATCH,), _TOP_P, dtype=torch.float32, device=device)
    return {
        "device": device,
        "target_probs_raw": target_probs_raw,
        "target_probs_raw_bs": target_probs_raw_bs,
        "top_k_t_bs": top_k_t_bs,
        "top_p_t_bs": top_p_t_bs,
        "ref_seq": ref_seq,
        "ref_joint": ref_joint,
        "draft_token": draft_token,
    }


def _fast_path_step(setup_dict) -> torch.Tensor:
    """One BATCH-sized call of joint-mode sampling, matching the runtime
    fast path. Returns shape [BATCH] sampled tokens.

    The runtime path pre-scales logits by temperature before sampling; for
    this test we work directly from the raw probs and a temperature of 1.0,
    so we pass log(probs) -- top_k_top_p_sampling_from_logits applies its
    own softmax internally.
    """
    logits = setup_dict["target_probs_raw_bs"].log()
    return top_k_top_p_sampling_from_logits(
        logits,
        setup_dict["top_k_t_bs"],
        setup_dict["top_p_t_bs"],
        filter_apply_order="joint",
        deterministic=True,
    )


def _sequential_path_step(setup_dict) -> torch.Tensor:
    """One BATCH-sized call of the original sequential chain-spec verify on a
    2-position linear chain. Each batch row is an independent trial.

    Returns shape [BATCH] -- the token chosen at chain position 0, which is
    the only relevant output for distribution comparison.
    """
    device = setup_dict["device"]
    bs = _BATCH
    num_draft_tokens = 2  # 1 chain candidate + 1 bonus slot

    target_probs_filtered = top_k_renorm_prob(
        setup_dict["target_probs_raw_bs"], setup_dict["top_k_t_bs"]
    )
    target_probs_filtered = top_p_renorm_prob(
        target_probs_filtered, setup_dict["top_p_t_bs"], is_deterministic=True
    )
    target_3d = (
        target_probs_filtered.unsqueeze(1)
        .expand(bs, num_draft_tokens, _VOCAB)
        .contiguous()
    )

    candidates = torch.empty((bs, num_draft_tokens), dtype=torch.int64, device=device)
    candidates[:, 0] = setup_dict["draft_token"]
    candidates[:, 1] = 0

    predicts = torch.full(
        (bs * num_draft_tokens,), -1, dtype=torch.int32, device=device
    )
    accept_index = torch.full(
        (bs, num_draft_tokens), -1, dtype=torch.int32, device=device
    )
    accept_token_num = torch.zeros(bs, dtype=torch.int32, device=device)
    coins = torch.rand((bs, num_draft_tokens), dtype=torch.float32, device=device)
    coins_final = torch.rand((bs,), dtype=torch.float32, device=device)

    chain_speculative_sampling_target_only(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_token_num,
        candidates=candidates,
        uniform_samples=coins,
        uniform_samples_for_final_sampling=coins_final,
        target_probs=target_3d,
        draft_probs=None,
        threshold_single=1.0,
        threshold_acc=1.0,
        deterministic=True,
    )
    return predicts.view(bs, num_draft_tokens)[:, 0]


def _empirical_dist(sampler, setup_dict) -> torch.Tensor:
    counts = torch.zeros(_VOCAB, device=setup_dict["device"])
    for _ in range(_NUM_CHUNKS):
        _accumulate(sampler(setup_dict), counts)
    return counts / counts.sum()


@pytest.mark.gpu
def test_fast_path_matches_joint_filter_reference(setup) -> None:
    """fast path's empirical output ≈ joint-filter analytic reference."""
    torch.manual_seed(1)
    emp = _empirical_dist(_fast_path_step, setup)
    tv = _total_variation(emp, setup["ref_joint"])
    assert (
        tv < _MODE_NOISE_FLOOR
    ), f"fast path deviates from joint-filter reference: TV={tv:.4f}"


@pytest.mark.gpu
def test_sequential_path_matches_sequential_filter_reference(setup) -> None:
    """sequential path's empirical output ≈ sequential-filter analytic reference."""
    torch.manual_seed(2)
    emp = _empirical_dist(_sequential_path_step, setup)
    tv = _total_variation(emp, setup["ref_seq"])
    assert (
        tv < _MODE_NOISE_FLOOR
    ), f"sequential path deviates from sequential-filter reference: TV={tv:.4f}"


@pytest.mark.gpu
def test_joint_vs_sequential_filter_gap_is_bounded_and_nonzero(setup) -> None:
    """The two filter modes are NOT identical (documented semantic shift).

    Assert TV is bounded by JOINT_VS_SEQUENTIAL_MAX_TV in this config and is
    strictly non-zero. Catches both regressions (gap blew up) and accidental
    semantic convergence (gap collapsed to zero, suggesting tests aren't
    exercising both filter modes anymore).
    """
    tv = _total_variation(setup["ref_seq"], setup["ref_joint"])
    assert tv < _JOINT_VS_SEQUENTIAL_MAX_TV, (
        f"joint vs sequential filter gap unexpectedly large: TV={tv:.4f} "
        f"(typical ~0.05 for top_k=50/top_p=0.95)"
    )
    assert tv > 0.0, (
        "joint and sequential filter references are identical -- are the "
        "filter modes truly distinct in this config?"
    )

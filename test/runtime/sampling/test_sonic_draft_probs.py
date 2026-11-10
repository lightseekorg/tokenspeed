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

"""sonic verifies drafts sampled from a recorded q by the draft-probability rule.

Under ``--enable-speculative-sampling`` the drafter samples each draft from its
own distribution q and records it (``draft_probs``). Verify accepts a draft x
with probability ``min(1, p(x) / q(x))`` and otherwise draws from
``max(p - q, 0)``, so the emitted token still follows the target p, and a
draft is accepted with probability ``sum(min(p, q))``. A slot with no recorded
proposal (the sentinel above ``spec_reject_draft_prob_threshold``) rejects its
draft and draws from p.
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys

import pytest
import torch

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(
    est_time=120,
    suite="runtime-1gpu",
    disabled_on_runners=["amd-*"],
    disabled_on_runners_reason="sonic's kernels need NVIDIA's libdevice (fast_logf).",
)

from tokenspeed_kernel.platform import current_platform  # noqa: E402

from tokenspeed.runtime.layers.logits_processor import (  # noqa: E402
    LogitsProcessorOutput,
)
from tokenspeed.runtime.sampling.backends.base import (  # noqa: E402
    SamplingBackendConfig,
)
from tokenspeed.runtime.sampling.backends.sonic import (  # noqa: E402
    SonicSamplingBackend,
)
from tokenspeed.runtime.sampling.sampling_batch_info import (  # noqa: E402
    SamplingBatchInfo,
)
from tokenspeed.runtime.sampling.sampling_params import SamplingParams  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and current_platform().is_nvidia),
    reason="sonic's kernels run on NVIDIA GPUs",
)

VOCAB = 1024
POOL = 8
BS = 4
N = 2
THRESHOLD = 2.0
ITERS = 6000


def _backend(speculative_sampling: bool = True) -> SonicSamplingBackend:
    return SonicSamplingBackend(
        SamplingBackendConfig(
            synthetic_acceptance_length=None,
            enable_speculative_sampling=speculative_sampling,
            sampling_stream="batch",
            logprob_order="torch",
            max_bs=BS,
            max_draft_tokens_per_req=N,
            max_req_pool_size=POOL,
            vocab_size=VOCAB,
            device="cuda",
            spec_reject_draft_prob_threshold=THRESHOLD,
        )
    )


def _prepare(backend: SonicSamplingBackend) -> torch.Tensor:
    params = []
    for i in range(BS):
        sp = SamplingParams(temperature=1.0, top_k=-1, top_p=1.0)
        sp.verify(VOCAB)
        sp.resolve_seed(f"d{i}")
        sp.normalize(None)
        params.append(sp)
    slots = list(range(1, BS + 1))
    backend.prepare_step(
        request_ids=[f"d{i}" for i in range(BS)],
        request_pool_indices=slots,
        sampling_params_list=params,
        num_tokens_per_req=N,
    )
    return torch.tensor(slots, dtype=torch.int64, device="cuda")


def _target() -> tuple[torch.Tensor, torch.Tensor]:
    """bf16 logits whose 300 strongest tokens carry the law, and that law over sonic's top-128 support."""
    row = (torch.randn(VOCAB) * 2.0).to(torch.bfloat16)
    row[300:] = -40.0
    values, ids = row.float().topk(128)
    probs = torch.zeros(VOCAB)
    probs[ids] = torch.softmax(values, -1)
    return row, probs


def _chi2_deviate(counts: torch.Tensor, probs: torch.Tensor, n: int) -> float:
    """Normal deviate of the chi-square statistic over cells expecting >= 20 draws."""
    keep = probs * n >= 20
    observed = counts[keep].double()
    expected = probs[keep].double() * n
    chi2 = (((observed - expected) ** 2) / expected).sum().item()
    dof = int(keep.sum().item()) - 1
    return (chi2 - dof) / math.sqrt(2 * dof)


def _verify_many(backend, slots, row, draft_probs, draws):
    """First emitted token counts and accepted drafts over ``ITERS`` verifies, drafts from ``draws``."""
    logits = row.cuda().expand(BS * N, VOCAB).contiguous()
    offsets = torch.zeros(POOL + 1, dtype=torch.int32, device="cuda")
    info = SamplingBatchInfo(
        req_pool_indices=slots,
        valid_cache_lengths=offsets,
        vocab_size=VOCAB,
        device="cuda",
        draft_probs=draft_probs,
    )
    candidates = torch.zeros(BS, N, dtype=torch.int32, device="cuda")
    counts = torch.zeros(VOCAB, dtype=torch.long, device="cuda")
    accepted = 0
    for step in range(ITERS):
        offsets.fill_(step * N)
        candidates[:, 1] = draws[step]
        predict, accept = backend.verify(
            LogitsProcessorOutput(next_token_logits=logits),
            info,
            candidates,
            tree=None,
        )
        first = predict.view(BS, N)[:, 0]
        counts.scatter_add_(0, first.long(), torch.ones_like(first, dtype=torch.long))
        accepted += int((accept == 2).sum().item())
    return counts.cpu(), accepted


@pytest.mark.parametrize("proposal", ["close", "unrelated"])
def test_drafts_from_q_emit_the_target_and_are_accepted_with_min_p_over_q(proposal):
    torch.manual_seed(31)
    row, probs = _target()
    if proposal == "close":
        # A drafter that roughly knows the target: its logits plus noise.
        q = torch.softmax(row.float() + torch.randn(VOCAB) * 0.7, -1)
    else:
        q = torch.softmax(torch.randn(VOCAB) * 2.0, -1)
    draft_probs = torch.zeros(POOL + 1, N, VOCAB, device="cuda")
    draft_probs[:, 0] = q.cuda()
    draws = torch.multinomial(q, ITERS * BS, replacement=True).view(ITERS, BS)
    backend = _backend()
    counts, accepted = _verify_many(
        backend, _prepare(backend), row, draft_probs, draws.int().cuda()
    )
    total = ITERS * BS
    assert counts[probs == 0].sum().item() == 0
    deviate = _chi2_deviate(counts, probs, total)
    assert abs(deviate) < 5.0, deviate
    expected = torch.minimum(probs, q).sum().item()
    spread = 4 * math.sqrt(expected * (1 - expected) / total) + 0.01
    assert abs(accepted / total - expected) < spread, (accepted / total, expected)


def test_a_slot_without_a_proposal_rejects_and_draws_the_target():
    torch.manual_seed(32)
    row, probs = _target()
    # The sentinel the runtime writes for a slot with no recorded proposal.
    draft_probs = torch.full((POOL + 1, N, VOCAB), THRESHOLD + 1, device="cuda")
    top = int(probs.argmax())
    draws = torch.full((ITERS, BS), top, dtype=torch.int32, device="cuda")
    backend = _backend()
    counts, accepted = _verify_many(backend, _prepare(backend), row, draft_probs, draws)
    assert accepted == 0
    deviate = _chi2_deviate(counts, probs, ITERS * BS)
    assert abs(deviate) < 5.0, deviate


def _kernel_test_utils():
    """tokenspeed-kernel's test helpers, for ``assert_no_triton_compile``."""
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = os.path.join(os.path.dirname(root), "tokenspeed-kernel", "test", "utils.py")
    spec = importlib.util.spec_from_file_location("sonic_kernel_test_utils", path)
    utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utils)
    return utils


def test_draft_prob_verify_compiles_nothing_after_warm_up():
    from tokenspeed_kernel.thirdparty.sonic_sampler.ops import (
        multistep,
        prologue,
        verify,
    )

    utils = _kernel_test_utils()
    backend = _backend()
    slots = _prepare(backend)
    # The same slots 8 bytes past 16-byte alignment, as a mixed round passes its decode rows.
    shifted = torch.cat([slots[:1], slots])[1:]
    logits = torch.randn(BS * N, VOCAB, device="cuda").bfloat16()
    offsets = torch.zeros(POOL + 1, dtype=torch.int64, device="cuda")
    drafts = torch.zeros((BS, N), dtype=torch.int32, device="cuda")
    # The runtime's table: every pool slot, one row per verify position.
    draft_probs = torch.full((POOL + 1, N, VOCAB), 1.0 / VOCAB, device="cuda")
    kernels = (
        prologue.bitpacked_reduction_kernel,
        multistep.cumulative_unpack_kernel,
        verify.chain_speculative_verification_kernel,
    )
    with utils.assert_no_triton_compile(*kernels):
        for request_slots in (slots, shifted):
            for bs in range(1, BS + 1):
                backend._fused_multistep(
                    logits[: bs * N],
                    request_slots[:bs],
                    offsets,
                    drafts[:bs],
                    draft_probs=draft_probs,
                )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

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
        admission_ids=list(range(1, BS + 1)),
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
        prologue.whole_reduction_kernel,
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


def _reference_dspark_markov_probs(
    base_logits: torch.Tensor,
    embed: torch.Tensor,
    proj: torch.Tensor,
    anchors: torch.Tensor,
    temps: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
) -> torch.Tensor:
    """Reference FP32 softmax + top-K/top-P renorm matching DSpark Markov."""
    from tokenspeed.runtime.sampling.backends.dspark_renorm import (
        torch_small_vocab_topk_topp_renorm,
    )

    bias = embed[anchors.long()].float() @ proj.float().t()
    raw = base_logits.float() + bias
    greedy = top_ks == 1
    probs = torch.softmax(raw / temps.float().unsqueeze(1).clamp_min(1e-20), dim=-1)
    argmax_ids = raw.argmax(dim=-1, keepdim=True)
    one_hot = torch.zeros_like(probs).scatter_(1, argmax_ids, 1.0)
    probs = torch.where(greedy.unsqueeze(1), one_hot, probs)
    return torch_small_vocab_topk_topp_renorm(probs.clone(), top_ks, top_ps)


@pytest.mark.parametrize("vocab", [512, 1024, 129280])
def test_sonic_dspark_markov_proposals_and_cuda_graph_zero_jit(vocab: int):
    from tokenspeed_kernel.ops.sampling.triton import dspark_markov

    from tokenspeed.runtime.layers.vocab_parallel_embedding import (
        ParallelLMHead,
        VocabParallelEmbedding,
    )
    from tokenspeed.runtime.models.deepseek_v41_dspark_heads import (
        DSparkVanillaMarkov,
        dspark_sampled_workspace,
        sample_dspark_block_sampled,
    )

    torch.manual_seed(41)
    block, rank = 4, 8
    backend = SonicSamplingBackend(
        SamplingBackendConfig(
            synthetic_acceptance_length=None,
            enable_speculative_sampling=True,
            sampling_stream="batch",
            logprob_order="torch",
            max_bs=BS,
            max_draft_tokens_per_req=block + 1,
            max_req_pool_size=POOL,
            vocab_size=vocab,
            device="cuda",
            spec_reject_draft_prob_threshold=THRESHOLD,
        )
    )
    specs = [
        # Disabled top_k (-1 -> _TOP_K_DISABLED): must NOT truncate at 128.
        SamplingParams(temperature=0.8, top_k=-1, top_p=1.0),
        SamplingParams(temperature=0.75, top_k=-1, top_p=0.9),
        SamplingParams(temperature=0.9, top_k=32, top_p=0.85),
        SamplingParams(temperature=0.0, top_k=-1, top_p=1.0),
    ]
    for i, sp in enumerate(specs):
        sp.verify(vocab)
        sp.resolve_seed(f"sonic-markov-{vocab}-{i}")
        sp.normalize(None)
    slots = torch.tensor([1, 2, 3, 4], dtype=torch.int64, device="cuda")
    backend.prepare_step(
        admission_ids=[10 + i for i in range(BS)],
        request_pool_indices=slots.tolist(),
        sampling_params_list=specs,
        num_tokens_per_req=block + 1,
    )

    pools = backend.speculative_sampling_pools()
    ws = dspark_sampled_workspace(
        max_rows=BS,
        vocab_size=vocab,
        rank=rank,
        embed_dtype=torch.bfloat16,
        pools=pools,
        topk_topp_workspace_bytes=backend.dspark_topk_topp_workspace_size(BS, vocab),
        renorm_probs=backend.renorm_dspark_probabilities,
        device=torch.device("cuda"),
    )
    base_logits = torch.randn(BS, block, vocab, device="cuda") * 0.25
    embedding = VocabParallelEmbedding(vocab, rank, params_dtype=torch.bfloat16).to(
        "cuda"
    )
    projection = ParallelLMHead(vocab, rank, params_dtype=torch.bfloat16).to("cuda")
    with torch.no_grad():
        embedding.weight.normal_(std=0.1)
        projection.weight.normal_(std=0.1)
    markov_head = DSparkVanillaMarkov(embedding, projection)
    # Force flat-enough spectrum on row 1 so top-p=0.9 spans > 128 tokens when vocab >= 1024.
    base_logits[1, :, :400] = 0.0
    base_logits[1, :, 400:] = -20.0
    anchors = torch.tensor([3, 7, 11, 15], dtype=torch.int32, device="cuda")
    base_pos = torch.tensor([10, 20, 30, 40], dtype=torch.int64, device="cuda")
    draft_probs = torch.full((POOL + 1, block + 1, vocab), THRESHOLD + 1, device="cuda")
    next_tokens = torch.zeros(BS, block + 1, dtype=torch.int32, device="cuda")

    sample_dspark_block_sampled(
        base_logits,
        anchors,
        markov_head,
        slots,
        slots,
        base_pos,
        draft_probs,
        ws,
        vocab,
        next_tokens[:, 1:],
    )
    expected_q0 = _reference_dspark_markov_probs(
        base_logits[:, 0],
        embedding.weight,
        projection.weight,
        anchors,
        pools.temperature.index_select(0, slots),
        pools.top_k.index_select(0, slots),
        pools.top_p.index_select(0, slots),
    )
    torch.testing.assert_close(draft_probs[slots, 0], expected_q0, atol=2e-3, rtol=2e-2)
    assert int((draft_probs[slots[1], 0] > 0).sum().item()) > 128
    assert draft_probs[slots[3], 0, expected_q0[3].argmax()].item() == pytest.approx(
        1.0
    )
    assert next_tokens[3, 1].item() == int(expected_q0[3].argmax().item())

    # Capture + replay DSpark Markov + Sonic verify under CUDAGraph with 0 Triton compiles.
    target_logits = torch.randn(BS * (block + 1), vocab, device="cuda").bfloat16()
    verify_offsets = torch.zeros(POOL + 1, dtype=torch.int64, device="cuda")
    verify_info = SamplingBatchInfo(
        req_pool_indices=slots,
        valid_cache_lengths=verify_offsets,
        vocab_size=vocab,
        device="cuda",
        draft_probs=draft_probs,
    )
    eager_tokens = next_tokens.clone()
    eager_draft_probs = draft_probs[slots, :block].clone()
    eager_predict, eager_accept = backend.verify(
        LogitsProcessorOutput(next_token_logits=target_logits),
        verify_info,
        next_tokens,
        tree=None,
    )
    eager_predict = eager_predict.clone()
    eager_accept = eager_accept.clone()

    utils = _kernel_test_utils()
    kernels = (
        dspark_markov._dspark_markov_init_kernel,
        dspark_markov._dspark_markov_stage1_kernel,
        dspark_markov._dspark_markov_stage2_norm_kernel,
        dspark_markov._dspark_scatter_and_mass_kernel,
        dspark_markov._dspark_select_and_embed_kernel,
    )
    with utils.assert_no_triton_compile(*kernels):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            sample_dspark_block_sampled(
                base_logits,
                anchors,
                markov_head,
                slots,
                slots,
                base_pos,
                draft_probs,
                ws,
                vocab,
                next_tokens[:, 1:],
            )
            cap_predict, cap_accept = backend.verify(
                LogitsProcessorOutput(next_token_logits=target_logits),
                verify_info,
                next_tokens,
                tree=None,
            )
        next_tokens.zero_()
        draft_probs[slots, :block].zero_()
        graph.replay()
        torch.testing.assert_close(next_tokens, eager_tokens, rtol=0, atol=0)
        torch.testing.assert_close(
            draft_probs[slots, :block], eager_draft_probs, rtol=0, atol=0
        )
        torch.testing.assert_close(cap_predict, eager_predict, rtol=0, atol=0)
        torch.testing.assert_close(cap_accept, eager_accept, rtol=0, atol=0)
        graph.reset()


def test_sonic_dspark_markov_verify_matches_target_distribution():
    from tokenspeed.runtime.layers.vocab_parallel_embedding import (
        ParallelLMHead,
        VocabParallelEmbedding,
    )
    from tokenspeed.runtime.models.deepseek_v41_dspark_heads import (
        DSparkVanillaMarkov,
        dspark_sampled_workspace,
        sample_dspark_block_sampled,
    )

    torch.manual_seed(73)
    backend = _backend()
    # Restrict target support within Sonic's top-128 support so rejection sampling
    # recovers the exact target distribution and sum(min(p, q)) acceptance rate.
    row, target_probs = _target()
    params = []
    for i in range(BS):
        sp = SamplingParams(temperature=1.0, top_k=96, top_p=0.95)
        sp.verify(VOCAB)
        sp.resolve_seed(f"sonic-dspark-stat-{i}")
        sp.normalize(None)
        params.append(sp)
    slots = torch.tensor(list(range(1, BS + 1)), dtype=torch.int64, device="cuda")
    backend.prepare_step(
        admission_ids=list(range(101, 101 + BS)),
        request_pool_indices=slots.tolist(),
        sampling_params_list=params,
        num_tokens_per_req=N,
    )
    # Apply the same top_k=96 / top_p=0.95 filter to the expected target probs.
    pools = backend.speculative_sampling_pools()
    target_logits = row.cuda().expand(BS * N, VOCAB).contiguous()
    from tokenspeed.runtime.sampling.backends.dspark_renorm import (
        torch_small_vocab_topk_topp_renorm,
    )

    ref_p = torch_small_vocab_topk_topp_renorm(
        torch.softmax(row.float().unsqueeze(0), dim=-1),
        pools.top_k.index_select(0, slots[:1]).cpu(),
        pools.top_p.index_select(0, slots[:1]).cpu(),
    ).squeeze(0)

    block, rank = 1, 8
    ws = dspark_sampled_workspace(
        max_rows=BS,
        vocab_size=VOCAB,
        rank=rank,
        embed_dtype=torch.bfloat16,
        pools=pools,
        topk_topp_workspace_bytes=backend.dspark_topk_topp_workspace_size(BS, VOCAB),
        renorm_probs=backend.renorm_dspark_probabilities,
        device=torch.device("cuda"),
    )
    base_logits = (
        row.float()
        .unsqueeze(0)
        .unsqueeze(0)
        .expand(BS, block, VOCAB)
        .contiguous()
        .cuda()
    )
    embedding = VocabParallelEmbedding(VOCAB, rank, params_dtype=torch.bfloat16).to(
        "cuda"
    )
    projection = ParallelLMHead(VOCAB, rank, params_dtype=torch.bfloat16).to("cuda")
    with torch.no_grad():
        embedding.weight.normal_(std=0.2)
        projection.weight.normal_(std=0.2)
    markov_head = DSparkVanillaMarkov(embedding, projection)
    anchors = torch.zeros(BS, dtype=torch.int32, device="cuda")
    base_pos = torch.zeros(BS, dtype=torch.int64, device="cuda")
    draft_probs = torch.full((POOL + 1, N, VOCAB), THRESHOLD + 1, device="cuda")
    next_tokens = torch.zeros(BS, N, dtype=torch.int32, device="cuda")
    offsets = torch.zeros(POOL + 1, dtype=torch.int32, device="cuda")
    info = SamplingBatchInfo(
        req_pool_indices=slots,
        valid_cache_lengths=offsets,
        vocab_size=VOCAB,
        device="cuda",
        draft_probs=draft_probs,
    )
    counts = torch.zeros(VOCAB, dtype=torch.long, device="cuda")
    accepted = 0
    for step in range(ITERS):
        base_pos.fill_(step)
        sample_dspark_block_sampled(
            base_logits,
            anchors,
            markov_head,
            slots,
            slots,
            base_pos,
            draft_probs,
            ws,
            VOCAB,
            next_tokens[:, 1:],
        )
        offsets.fill_(step * N)
        predict, accept = backend.verify(
            LogitsProcessorOutput(next_token_logits=target_logits),
            info,
            next_tokens,
            tree=None,
        )
        first = predict.view(BS, N)[:, 0]
        counts.scatter_add_(0, first.long(), torch.ones_like(first, dtype=torch.long))
        accepted += int((accept == 2).sum().item())

    expected_q = draft_probs[slots[0], 0].cpu()
    total = ITERS * BS
    assert counts.cpu()[ref_p == 0].sum().item() == 0
    deviate = _chi2_deviate(counts.cpu(), ref_p, total)
    assert abs(deviate) < 5.0, deviate
    expected_acc = torch.minimum(ref_p, expected_q).sum().item()
    spread = 4 * math.sqrt(expected_acc * (1 - expected_acc) / total) + 0.015
    assert abs(accepted / total - expected_acc) < spread, (
        accepted / total,
        expected_acc,
    )


def test_torch_small_vocab_topk_topp_renorm_matches_fused_extension():
    from tokenspeed_kernel.ops.sampling.cuda import fused_topk_topp_renorm

    from tokenspeed.runtime.sampling.backends.dspark_renorm import (
        torch_small_vocab_topk_topp_renorm,
    )
    from tokenspeed.runtime.sampling.sampling_params import _TOP_K_DISABLED

    torch.manual_seed(17)
    vocab = 1024
    probs = torch.softmax(torch.randn(5, vocab, device="cuda") * 2.5, dim=-1)
    top_ks = torch.tensor(
        [1, _TOP_K_DISABLED, _TOP_K_DISABLED, 17, 127], dtype=torch.int32, device="cuda"
    )
    top_ps = torch.tensor(
        [1.0, 1.0, 0.85, 0.9, 1.0], dtype=torch.float32, device="cuda"
    )

    fused_in = probs.clone()
    fused_in[0] = float("nan")
    torch_in = fused_in.clone()

    fused_out = fused_topk_topp_renorm(fused_in, top_ks, top_ps, in_place_skip=True)
    torch_out = torch_small_vocab_topk_topp_renorm(torch_in, top_ks, top_ps)

    assert torch_out[0].isnan().all()
    torch.testing.assert_close(torch_out[1:], fused_out[1:], atol=1e-6, rtol=1e-5)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

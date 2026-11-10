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

"""The sonic sampler on vocab-sharded logits draws what it draws from the gathered rows.

Rank 2 of four simulated ranks hands the sampler its raw LM-head shard; the
other ranks' candidates arrive through the gather, which the test answers by
running their selection over their shards. Tokens, accepted drafts and decode
counts must be bitwise those of the same backend on the whole rows the logits
processor produces: bf16 rows gathered, scaled and softcapped by the
processor's own kernel; fp32 rows (Beam's capped head) gathered as they are.
"""

from __future__ import annotations

import importlib.util
import os
import sys

import pytest
import torch

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(
    est_time=60,
    suite="runtime-1gpu",
    disabled_on_runners=["amd-*"],
    disabled_on_runners_reason="sonic's kernels need NVIDIA's libdevice (fast_logf).",
)

from tokenspeed_kernel.ops.sampling import sonic  # noqa: E402
from tokenspeed_kernel.ops.sampling.sonic import MAX_K, pick  # noqa: E402
from tokenspeed_kernel.platform import current_platform  # noqa: E402

from tokenspeed.runtime.layers import sampled_vocab_shard as shard_module  # noqa: E402
from tokenspeed.runtime.layers.logits_processor import (  # noqa: E402
    LogitsProcessorOutput,
    fused_softcap_generic,
)
from tokenspeed.runtime.layers.sampled_vocab_shard import (  # noqa: E402
    SampledVocabShard,
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

RANKS = 4
RANK = 2
SHARD = 8192
VOCAB = RANKS * SHARD
SCALE = 0.17677669529663687
SOFTCAP = 20.0
N = 4
POOL = 9
STEPS = 4

# One request per sampling setting, each in its own slot.
SETTINGS = (
    dict(temperature=0.0),
    dict(temperature=0.7),
    dict(temperature=0.9, top_k=20),
    dict(top_p=0.8),
    dict(min_p=0.1),
    dict(repetition_penalty=1.3, frequency_penalty=0.4, presence_penalty=0.2),
    dict(temperature=0.8, logit_bias={str(t): 15.0 for t in (97, 9000, 17000, 30000)}),
    # A grammar request: its mask (``_mask``) allows every third token.
    dict(top_p=0.9, json_schema="{}"),
)
BS = len(SETTINGS)
GRAMMAR_REQUEST = 7


def _config(**overrides) -> SamplingBackendConfig:
    fields = dict(
        synthetic_acceptance_length=None,
        enable_speculative_sampling=False,
        sampling_stream="batch",
        logprob_order="torch",
        max_bs=BS,
        max_draft_tokens_per_req=N,
        max_req_pool_size=POOL,
        vocab_size=VOCAB,
        device="cuda",
        tp_group=tuple(range(RANKS)),
        # One process stands in for the group: nothing to broadcast.
        enable_tp_sync=False,
    )
    fields.update(overrides)
    return SamplingBackendConfig(**fields)


def _prepare(backend: SonicSamplingBackend, n: int) -> torch.Tensor:
    params = []
    for i, setting in enumerate(SETTINGS):
        sp = SamplingParams(**{"top_k": -1, "top_p": 1.0, **setting})
        sp.verify(VOCAB)
        sp.resolve_seed(f"r{i}")
        sp.normalize(None)
        params.append(sp)
    slots = list(range(1, BS + 1))
    backend.prepare_step(
        request_ids=[f"r{i}" for i in range(BS)],
        request_pool_indices=slots,
        sampling_params_list=params,
        num_tokens_per_req=n,
    )
    return torch.tensor(slots, dtype=torch.int64, device="cuda")


def _mask(rows: int, n: int) -> torch.Tensor:
    """xgrammar's int32 bitmask: the grammar request allows every third token, the others all."""
    allowed = torch.ones((rows, VOCAB), dtype=torch.bool)
    allowed.view(BS, n, VOCAB)[GRAMMAR_REQUEST] = torch.arange(VOCAB) % 3 == 0
    bits = allowed.view(rows, -1, 32).to(torch.int64) << torch.arange(32)
    bits = bits.sum(-1)
    return torch.where(bits >= 2**31, bits - 2**32, bits).to("cuda", torch.int32)


def _info(slots: torch.Tensor, mask: torch.Tensor) -> SamplingBatchInfo:
    return SamplingBatchInfo(
        req_pool_indices=slots,
        valid_cache_lengths=torch.arange(
            500, 500 + POOL + 1, dtype=torch.int32, device="cuda"
        ),
        vocab_mask=mask,
        vocab_size=VOCAB,
        device="cuda",
    )


class _Group:
    """Four ranks' raw head shards; rank ``RANK`` samples, the gather runs the others' selection."""

    def __init__(self, rows: int, n: int, seed: int, dtype: torch.dtype):
        gen = torch.Generator().manual_seed(seed)
        raw = torch.randn(rows, VOCAB, generator=gen) * 3
        # The greedy request's rows hold one clear maximum (unsharded sonic breaks bf16 ties by block column).
        raw.view(BS, n, VOCAB)[0, :, 12345] = 40.0
        self.capped = dtype == torch.bfloat16
        if self.capped:
            self.raw = raw.to("cuda", torch.bfloat16)
            # The logits processor's whole rows: gathered, then scaled and capped.
            self.whole = self.raw.clone()
            fused_softcap_generic(self.whole, SOFTCAP, SCALE)
        else:
            # Beam's head: capped fp32 rows whose 256 strongest tokens sit 0.1 apart (over a bf16 ulp), the rest far below.
            raw -= 40.0
            head = torch.randperm(VOCAB, generator=gen)[:256]
            raw[:, head] = 20.0 - 0.1 * torch.arange(256, dtype=torch.float32)
            raw.view(BS, n, VOCAB)[0, :, 12345] = 40.0
            self.raw = raw.to("cuda", torch.float32)
            self.whole = self.raw.clone()
        self.drafted: torch.Tensor | None = None

    def shard(self, rank: int) -> SampledVocabShard:
        return SampledVocabShard(
            logits=self.raw[:, rank * SHARD : (rank + 1) * SHARD],
            vocab_start=rank * SHARD,
            logit_scale=SCALE if self.capped else None,
            softcap=SOFTCAP if self.capped else None,
            tp_size=RANKS,
            tp_group=tuple(range(RANKS)),
            gather_full=lambda logits: self.whole.clone(),
        )

    def all_gather_single(self, landing, staged, group):
        """Every rank's candidate pairs, rank-major, as the group would send them."""
        rows = staged.shape[0]
        backend = self.backend
        state = backend.state if self.capped else backend._exact_state
        for peer in range(RANKS):
            out = landing[peer * rows : (peer + 1) * rows]
            if peer == RANK:
                out.copy_(staged)
                continue
            peer_shard = self.shard(peer)
            sonic.shard_candidates(
                peer_shard.logits[:rows].contiguous(),
                state,
                self.slots,
                self.drafted,
                torch.empty_like(backend.scratchpad[:rows]),
                sonic.VocabShard(
                    peer * SHARD,
                    peer_shard.logit_scale,
                    peer_shard.softcap,
                    None,
                    out,
                ),
                pick(SHARD, rows),
                False,
            )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_sample_on_shards_matches_whole_rows(monkeypatch, dtype) -> None:
    whole, sharded = (SonicSamplingBackend(_config()) for _ in range(2))
    for step in range(STEPS):
        group = _Group(BS, 1, step, dtype)
        monkeypatch.setattr(shard_module, "all_gather_single", group.all_gather_single)
        results = []
        for backend, output in (
            (whole, LogitsProcessorOutput(next_token_logits=group.whole.clone())),
            (
                sharded,
                LogitsProcessorOutput(
                    next_token_logits=None, vocab_shard=group.shard(RANK)
                ),
            ),
        ):
            group.backend, group.slots = backend, _prepare(backend, 1)
            sampled, lengths = backend.sample(output, _info(group.slots, _mask(BS, 1)))
            results.append(sampled.clone())
        assert torch.equal(results[1], results[0]), (step, results)
        assert int(results[0][GRAMMAR_REQUEST]) % 3 == 0
    assert torch.equal(sharded.state.decode_counts, whole.state.decode_counts)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_verify_on_shards_matches_whole_rows(monkeypatch, dtype) -> None:
    rows = BS * N
    whole, sharded = (SonicSamplingBackend(_config()) for _ in range(2))
    lengths = []
    for step in range(STEPS):
        group = _Group(rows, N, 50 + step, dtype)
        monkeypatch.setattr(shard_module, "all_gather_single", group.all_gather_single)
        # Drafts: each row's argmax on alternating rows, a weak token on the rest.
        best = group.whole.float().view(BS, N, VOCAB).argmax(dim=2).int()
        lanes = torch.arange(N, device="cuda") % 2 == step % 2
        candidates = torch.where(lanes, best, 5).roll(1, dims=1)
        results = []
        for backend, output in (
            (whole, LogitsProcessorOutput(next_token_logits=group.whole.clone())),
            (
                sharded,
                LogitsProcessorOutput(
                    next_token_logits=None, vocab_shard=group.shard(RANK)
                ),
            ),
        ):
            group.backend, group.slots = backend, _prepare(backend, N)
            group.drafted = backend.v_drafted[:BS]
            predict, accept = backend.verify(
                output, _info(group.slots, _mask(rows, N)), candidates, tree=None
            )
            results.append((predict.clone(), accept.clone()))
        assert torch.equal(results[1][0], results[0][0]), step
        assert torch.equal(results[1][1], results[0][1]), step
        assert int(results[0][0].view(BS, N)[GRAMMAR_REQUEST, 0]) % 3 == 0
        lengths += results[0][1].tolist()
    assert torch.equal(sharded.state.decode_counts, whole.state.decode_counts)
    # The windows were neither all accepted nor all rejected.
    assert min(lengths) < N and max(lengths) > 1


def _kernel_test_utils():
    """tokenspeed-kernel's test helpers, for ``assert_no_triton_compile``."""
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = os.path.join(os.path.dirname(root), "tokenspeed-kernel", "test", "utils.py")
    spec = importlib.util.spec_from_file_location("sonic_kernel_test_utils", path)
    utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utils)
    return utils


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_sharded_launches_compile_nothing_after_warm_up(monkeypatch, dtype) -> None:
    """``warmup_vocab_shard_sampling`` compiles what every batch size, offset slot pointer and draft width the shard path launches needs."""
    from tokenspeed_kernel.thirdparty.sonic_sampler.ops import (
        multistep,
        prologue,
    )
    from tokenspeed_kernel.thirdparty.sonic_sampler.ops import shard as shard_ops
    from tokenspeed_kernel.thirdparty.sonic_sampler.ops import (
        singular,
        verify,
    )

    def all_gather_single(landing, staged, group):
        # Every rank's pairs are this rank's: the gather launches no watched kernel.
        rows = staged.shape[0]
        for peer in range(RANKS):
            landing[peer * rows : (peer + 1) * rows].copy_(staged)

    monkeypatch.setattr(shard_module, "all_gather_single", all_gather_single)
    utils = _kernel_test_utils()
    backend = SonicSamplingBackend(_config())
    shard = _Group(BS * N, N, 7, dtype).shard(RANK)
    # The processor hands over its projection's contiguous rows.
    logits = shard.logits.contiguous()
    backend.warmup_vocab_shard_sampling(
        shard,
        torch.zeros((1, N), dtype=torch.int32, device="cuda"),
        draft_q=None,
        with_vocab_mask=False,
    )
    slots = _prepare(backend, N)
    # The same slots 8 bytes past 16-byte alignment, as a mixed round passes its decode rows.
    shifted = torch.cat([slots[:1], slots])[1:]
    offsets = torch.zeros(POOL + 1, dtype=torch.int64, device="cuda")
    drafts = torch.zeros((BS, N), dtype=torch.int32, device="cuda")
    kernels = (
        prologue.bitpacked_reduction_kernel,
        shard_ops.shard_candidates_kernel,
        singular.cumulative_selection_kernel,
        multistep.cumulative_unpack_kernel,
        verify.chain_speculative_verification_kernel,
    )
    with utils.assert_no_triton_compile(*kernels):
        for request_slots in (slots, shifted):
            for bs in range(1, BS + 1):
                backend._fused_singular(
                    logits[:bs], request_slots[:bs], offsets, shard=shard
                )
                backend._fused_multistep(
                    logits[: bs * N],
                    request_slots[:bs],
                    offsets,
                    drafts[:bs],
                    shard=shard,
                )


def test_shards_need_tp_and_whole_row_readers_off() -> None:
    sharded = SonicSamplingBackend(_config())
    assert sharded.vocab_shard_top_k() == MAX_K
    assert sharded.supports_vocab_shard(torch.bfloat16, SHARD, RANKS)
    assert sharded.supports_vocab_shard(torch.float32, SHARD, RANKS)
    assert not sharded.supports_vocab_shard(torch.float16, SHARD, RANKS)
    assert not sharded.supports_vocab_shard(torch.bfloat16, SHARD, 2)
    for overrides in (
        dict(tp_group=None),
        dict(enable_output_logprobs=True),
        dict(enable_nan_detection=True),
    ):
        backend = SonicSamplingBackend(_config(**overrides))
        assert backend.vocab_shard_top_k() is None
        assert not backend.supports_vocab_shard(torch.bfloat16, SHARD, RANKS)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

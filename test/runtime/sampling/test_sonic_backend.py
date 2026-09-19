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

"""SonicSamplingBackend contract tests.

Pins the seams the runtime relies on:

* registry: ``sonic`` resolves when sonic-sampler is importable;
* flip detection writes slot state once (scalars + indicator bits) and
  steady-state steps never re-scatter;
* greedy rows (``top_k == 1``) match argmax through the fused kernel, for
  ``sample()`` and the chain verifier alike;
* ``verify()`` at ``N == 1`` IS ``sample()`` (the unified decode rule),
  bitwise, greedy and stochastic;
* per-request knobs reach the kernel: top_k support, min_p == 1 collapse,
  logit_bias, grammar bitmask (per draft position under verify), and the
  in-kernel decode counts that drive frequency penalties;
* CUDA-graph replay of ``sample()``/``verify()`` matches eager;
* the in-graph noise draw is a function of (seed, cache length) only;
* an all-greedy step replays the greedy backend's kernels (graph variant).

Logits are bf16 throughout, the lm_head's native dtype every backend is fed.
The fused kernels come from ``tokenspeed_kernel.thirdparty.sonic``; sonic-sampler
itself supplies buffers, indicators and dispatch tables.

Runs both the tuned dispatch path (real vocab) and the fallback tiling
(small vocab, no bucket), since production hits either depending on arch.
"""

from __future__ import annotations

import importlib.util
import math
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest
import torch

from tokenspeed.runtime.sampling.backends.base import SamplingBackendConfig
from tokenspeed.runtime.sampling.sampling_batch_info import SamplingBatchInfo
from tokenspeed.runtime.sampling.sampling_params import SamplingParams

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)
requires_sonic = pytest.mark.skipif(
    importlib.util.find_spec("sonic_sampler") is None,
    reason="requires the optional sonic-sampler package",
)

pytestmark = [requires_cuda, requires_sonic]

if importlib.util.find_spec("sonic_sampler") is not None:
    # Bind sonic to the vendored Triton before any test imports sonic_sampler directly.
    import tokenspeed_kernel.ops.sampling.sonic  # noqa: F401

VOCAB = 1024  # below every tuned bucket -> exercises the fallback tiling
REAL_VOCAB = 151936  # qwen-sized -> exercises the tuned dispatch on sm90/sm100
POOL = 8
MAX_BS = 4
MAX_N = 4


def _backend_cls():
    from tokenspeed.runtime.sampling.backends.sonic import SonicSamplingBackend

    return SonicSamplingBackend


def _make_config(vocab: int = VOCAB, max_n: int = MAX_N) -> SamplingBackendConfig:
    return SamplingBackendConfig(
        max_bs=MAX_BS,
        max_draft_tokens_per_req=max_n,
        max_req_pool_size=POOL,
        vocab_size=vocab,
        device="cuda",
    )


def _make_backend(vocab: int = VOCAB, max_n: int = MAX_N):
    return _backend_cls()(_make_config(vocab, max_n))


def _sp(rid: str, vocab: int = VOCAB, **overrides) -> SamplingParams:
    defaults = dict(temperature=1.0, top_k=-1, top_p=1.0)
    defaults.update(overrides)
    sp = SamplingParams(**defaults)
    sp.verify(vocab)
    sp.resolve_seed(rid)
    sp.normalize(None)
    return sp


def _greedy_sp(rid: str, vocab: int = VOCAB, **overrides) -> SamplingParams:
    return _sp(rid, vocab, temperature=0.0, **overrides)


def _prepare(backend, sps: list[SamplingParams], n: int = 1, slots=None):
    slots = list(range(1, len(sps) + 1)) if slots is None else slots
    backend.prepare_step(
        request_ids=[f"rid_{sp.seed}_{i}" for i, sp in enumerate(sps)],
        request_pool_indices=slots,
        sampling_params_list=sps,
        num_tokens_per_req=n,
    )
    return torch.tensor(slots, dtype=torch.int64, device="cuda")


def _info(req_pool_indices, vocab: int = VOCAB, vocab_mask=None) -> SamplingBatchInfo:
    # Pool-indexed cache lengths: the per-slot Philox offset production always supplies.
    offsets = torch.arange(100, 100 + POOL + 1, dtype=torch.int32, device="cuda")
    return SamplingBatchInfo(
        req_pool_indices=req_pool_indices,
        valid_cache_lengths=offsets,
        vocab_size=vocab,
        vocab_mask=vocab_mask,
        device="cuda",
    )


def _logits_output(logits: torch.Tensor):
    from tokenspeed.runtime.layers.logits_processor import LogitsProcessorOutput

    return LogitsProcessorOutput(next_token_logits=logits)


def _tie_free_logits(
    rows: int, vocab: int = VOCAB, top: int = 1, dtype=torch.bfloat16
) -> torch.Tensor:
    """bf16 logits (the lm_head dtype production feeds the sampler) whose
    ``top`` largest entries per row are strictly ordered and strictly above
    the rest, so argmax / top-k references cannot disagree via bf16 ties."""
    logits = torch.randn(rows, vocab, device="cuda", dtype=torch.float32)
    idx = logits.topk(top, dim=-1).indices
    bump = 8.0 + torch.arange(top, 0, -1, device="cuda", dtype=torch.float32)
    logits.scatter_(1, idx, bump.expand(rows, top))
    return logits.to(dtype)


def _allow_only(tokens: torch.Tensor, vocab: int = VOCAB) -> torch.Tensor:
    """xgrammar-style int32 bitmask ``[rows, ceil(V/32)]`` allowing one token
    per row (SET bit = allowed)."""
    words = (vocab + 31) // 32
    mask = torch.zeros(tokens.numel(), words, dtype=torch.int32, device="cuda")
    rows = torch.arange(tokens.numel(), device="cuda")
    word = tokens // 32
    bit = (tokens % 32).to(torch.int64)
    mask[rows, word] = (torch.ones_like(bit) << bit).to(torch.int32)
    return mask


# --------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------- #
def test_registry_resolves_sonic():
    from tokenspeed.runtime.sampling import registry

    server_args = SimpleNamespace(
        sampling_backend="sonic",
        enable_nan_detection=False,
        enable_output_logprobs=False,
        disable_sampling_tp_sync=True,
    )
    backend = registry.create_sampling_backend(
        server_args,
        max_bs=MAX_BS,
        max_draft_tokens_per_req=MAX_N,
        device="cuda",
        max_req_pool_size=POOL,
        vocab_size=VOCAB,
    )
    assert isinstance(backend, _backend_cls())


# --------------------------------------------------------------------- #
# slot state
# --------------------------------------------------------------------- #
def test_flip_detection_writes_slot_state_once():
    from sonic_sampler.core.flags import Indicator

    backend = _make_backend()
    sp_a = _sp("a", temperature=0.7, top_k=50, top_p=0.9, min_p=0.05, seed=42)
    sp_b = _greedy_sp("b", seed=7)
    _prepare(backend, [sp_a, sp_b], slots=[1, 3])
    torch.cuda.synchronize()

    assert backend._last_rid_per_slot[1] is not None
    assert backend._last_rid_per_slot[3] is not None

    buffers = backend.buffers
    assert buffers.temperature[1].item() == pytest.approx(0.7, abs=1e-2)
    assert buffers.top_k[1].item() == 50
    assert buffers.top_p[1].item() == pytest.approx(0.9, abs=1e-2)
    assert buffers.min_p[1].item() == pytest.approx(0.05, abs=1e-2)
    bits_a = buffers.flags.indicators.target[1]
    assert Indicator.TEMPERATURE in bits_a
    assert Indicator.TOP_K in bits_a
    assert Indicator.TOP_P in bits_a
    assert Indicator.MIN_P in bits_a
    assert Indicator.GREEDY not in bits_a

    # Greedy: top_k == 1, neutral knobs, GREEDY bit and nothing stochastic.
    assert buffers.top_k[3].item() == 1
    assert buffers.temperature[3].item() == pytest.approx(1.0)
    bits_b = buffers.flags.indicators.target[3]
    assert Indicator.GREEDY in bits_b
    assert not (bits_b & Indicator.stochastic())

    # top_k = -1 lands on the bounded top-MAX_K path with the TOP_K bit off.
    sp_c = _sp("c", top_k=-1, seed=9)
    _prepare(backend, [sp_c], slots=[5])
    torch.cuda.synchronize()
    assert buffers.top_k[5].item() == 128
    assert Indicator.TOP_K not in buffers.flags.indicators.target[5]

    # Steady state: same rid on the same slot must not re-scatter.
    buffers.temperature[1].fill_(9.0)
    _prepare(backend, [sp_a, sp_b], slots=[1, 3])
    torch.cuda.synchronize()
    assert buffers.temperature[1].item() == pytest.approx(9.0)


# --------------------------------------------------------------------- #
# greedy
# --------------------------------------------------------------------- #
def test_measured_dispatch_replaces_packaged_bucket():
    """On an arch with measured configs, the tuned (block_n, strategy, warps)
    must be what ``_tuning`` hands the kernels for every batch size."""
    from tokenspeed_kernel.ops.sampling.sonic import _MEASURED_DISPATCH

    backend = _make_backend(REAL_VOCAB)
    major, minor = torch.cuda.get_device_capability()
    if backend.tiling.dispatch is None:
        pytest.skip("no packaged dispatch bucket for this arch")
    key = (major * 10 + minor, backend.tiling.dispatch.size)
    if key not in _MEASURED_DISPATCH:
        pytest.skip("no measured dispatch for this arch/bucket")
    entries = _MEASURED_DISPATCH[key]
    for bs in (1, 4, MAX_BS):
        config, strategy, block_n = backend._tuning(bs, None)
        size, exp_block_n, exp_strategy, first, second = min(
            (e for e in entries if e[0] >= bs), key=lambda e: e[0]
        )
        assert block_n == exp_block_n
        assert strategy.key == exp_strategy
        assert (config.first, config.second) == (first, second)


@pytest.mark.parametrize("vocab", [VOCAB, REAL_VOCAB])
def test_greedy_sample_matches_argmax(vocab):
    torch.manual_seed(0)
    backend = _make_backend(vocab)
    if vocab == REAL_VOCAB and backend.tiling.dispatch is None:
        pytest.skip("no tuned dispatch bucket for this arch; fallback covered by VOCAB")
    req = _prepare(backend, [_greedy_sp(f"g{i}", vocab) for i in range(MAX_BS)])
    logits = _tie_free_logits(MAX_BS, vocab)
    sampled, ones = backend.sample(_logits_output(logits.clone()), _info(req, vocab))
    assert sampled.data_ptr() == backend.out_tok.data_ptr()
    torch.testing.assert_close(sampled.view(-1).cpu(), logits.argmax(-1).int().cpu())
    assert ones.tolist() == [1] * MAX_BS


def test_greedy_verify_chain_accepts_argmax_prefix():
    torch.manual_seed(1)
    n = MAX_N
    backend = _make_backend()
    req = _prepare(backend, [_greedy_sp(f"g{i}") for i in range(MAX_BS)], n=n)
    logits = _tie_free_logits(MAX_BS * n)
    argmax = logits.argmax(-1).int().view(MAX_BS, n)

    # Row i: the first i drafts match the target argmax, the next one does not.
    candidates = torch.randint(0, VOCAB, (MAX_BS, n), dtype=torch.int32, device="cuda")
    for i in range(MAX_BS):
        for j in range(1, n):
            candidates[i, j] = (
                argmax[i, j - 1] if j <= i else (argmax[i, j - 1] + 1) % VOCAB
            )

    predict, accept = backend.verify(
        _logits_output(logits.clone()), _info(req), candidates
    )
    assert predict.data_ptr() == backend.v_drafted.data_ptr()
    predict = predict.view(MAX_BS, n).cpu()
    accept = accept.cpu()
    for i in range(MAX_BS):
        k = min(i, n - 1)
        assert accept[i].item() == k + 1
        # Accepted prefix echoes the drafts; the correction is the argmax there.
        torch.testing.assert_close(predict[i, :k], candidates[i, 1 : k + 1].cpu())
        assert predict[i, k].item() == argmax[i, k].item()


# --------------------------------------------------------------------- #
# verify(N == 1) == sample()
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("greedy", [True, False], ids=["greedy", "stochastic"])
def test_verify_n1_matches_sample(greedy):
    torch.manual_seed(2)
    make = _greedy_sp if greedy else (lambda r: _sp(r, temperature=0.8, top_k=40))
    logits = torch.randn(MAX_BS, VOCAB, device="cuda", dtype=torch.bfloat16)

    backend = _make_backend(max_n=1)
    req = _prepare(backend, [make(f"n1_{i}") for i in range(MAX_BS)])
    sampled, ones = backend.sample(_logits_output(logits.clone()), _info(req))
    sampled = sampled.clone()

    backend2 = _make_backend(max_n=1)
    req2 = _prepare(backend2, [make(f"n1_{i}") for i in range(MAX_BS)])
    candidates = torch.randint(0, VOCAB, (MAX_BS, 1), dtype=torch.int32, device="cuda")
    predict, accept = backend2.verify(
        _logits_output(logits.clone()), _info(req2), candidates
    )
    torch.testing.assert_close(predict.view(-1).cpu(), sampled.view(-1).cpu())
    assert accept.tolist() == [1] * MAX_BS
    assert ones.tolist() == [1] * MAX_BS


# --------------------------------------------------------------------- #
# per-request knobs reach the kernel
# --------------------------------------------------------------------- #
def test_stochastic_sample_stays_in_top_k_support():
    torch.manual_seed(3)
    k = 8
    backend = _make_backend()
    sps = [_sp(f"k{i}", temperature=1.0, top_k=k) for i in range(MAX_BS)]
    logits = _tie_free_logits(MAX_BS, top=k)
    topk = logits.topk(k, dim=-1).indices.cpu()
    for step in range(16):
        req = _prepare(backend, sps)
        info = _info(req)
        info.valid_cache_lengths = info.valid_cache_lengths + step
        sampled, _ = backend.sample(_logits_output(logits.clone()), info)
        for i, tok in enumerate(sampled.view(-1).cpu().tolist()):
            assert (
                tok in topk[i].tolist()
            ), f"step {step} row {i}: {tok} outside top-{k}"


def test_min_p_one_collapses_to_argmax():
    torch.manual_seed(4)
    backend = _make_backend()
    req = _prepare(backend, [_sp(f"m{i}", min_p=1.0) for i in range(MAX_BS)])
    logits = _tie_free_logits(MAX_BS)
    sampled, _ = backend.sample(_logits_output(logits.clone()), _info(req))
    torch.testing.assert_close(sampled.view(-1).cpu(), logits.argmax(-1).int().cpu())


def test_logit_bias_forces_token():
    torch.manual_seed(5)
    backend = _make_backend()
    target = [17, 300, 511, 1000]
    sps = [_sp(f"b{i}", logit_bias={str(t): 100.0}) for i, t in enumerate(target)]
    req = _prepare(backend, sps)
    logits = _tie_free_logits(MAX_BS)
    sampled, _ = backend.sample(_logits_output(logits.clone()), _info(req))
    assert sampled.view(-1).cpu().tolist() == target


def test_grammar_mask_forces_token_sample_and_verify():
    torch.manual_seed(6)
    n = MAX_N
    backend = _make_backend()

    # sample(): one mask row per request.
    allowed = torch.tensor([3, 64, 700, 1023], dtype=torch.int32, device="cuda")
    req = _prepare(backend, [_sp(f"gr{i}", json_schema="{}") for i in range(MAX_BS)])
    logits = _tie_free_logits(MAX_BS)
    sampled, _ = backend.sample(
        _logits_output(logits.clone()), _info(req, vocab_mask=_allow_only(allowed))
    )
    assert sampled.view(-1).cpu().tolist() == allowed.cpu().tolist()

    # verify(): greedy rows with mismatching drafts must correct position 0 under its mask.
    allowed_pos = torch.arange(MAX_BS * n, device="cuda", dtype=torch.int32) * 7 + 1
    req = _prepare(
        backend, [_greedy_sp(f"gv{i}", json_schema="{}") for i in range(MAX_BS)], n=n
    )
    logits = _tie_free_logits(MAX_BS * n)
    candidates = torch.zeros(MAX_BS, n, dtype=torch.int32, device="cuda")
    predict, accept = backend.verify(
        _logits_output(logits.clone()),
        _info(req, vocab_mask=_allow_only(allowed_pos)),
        candidates,
    )
    assert accept.cpu().tolist() == [1] * MAX_BS
    expected = allowed_pos.view(MAX_BS, n)[:, 0].cpu()
    torch.testing.assert_close(predict.view(MAX_BS, n)[:, 0].cpu(), expected)


def test_frequency_penalty_uses_in_kernel_decode_counts():
    """Greedy row with the maximum frequency penalty: the first draw takes the
    argmax and bumps its decode count; the second draw is pushed to the
    runner-up because ``logits[t0] - 2.0 * 1 < logits[t1]``."""
    torch.manual_seed(7)
    backend = _make_backend()
    sps = [_greedy_sp(f"p{i}", frequency_penalty=2.0) for i in range(MAX_BS)]
    logits = torch.full((MAX_BS, VOCAB), -5.0, device="cuda", dtype=torch.bfloat16)
    t0 = torch.tensor([10, 20, 30, 40], device="cuda")
    t1 = t0 + 1
    rows = torch.arange(MAX_BS, device="cuda")
    logits[rows, t0] = 10.0
    logits[rows, t1] = 9.5

    req = _prepare(backend, sps)
    first, _ = backend.sample(_logits_output(logits.clone()), _info(req))
    assert first.view(-1).cpu().tolist() == t0.cpu().tolist()
    counts = backend.buffers.repetition.counts.decode
    assert counts[req, t0].cpu().tolist() == [1] * MAX_BS

    req = _prepare(backend, sps)
    second, _ = backend.sample(_logits_output(logits.clone()), _info(req))
    assert second.view(-1).cpu().tolist() == t1.cpu().tolist()


def test_noise_is_a_function_of_seed_and_cache_length():
    """Per-request determinism: same seed + same cache length -> same token
    regardless of slot or batch mates; a different cache length redraws."""
    torch.manual_seed(10)
    backend = _make_backend()
    logits = torch.randn(1, VOCAB, device="cuda", dtype=torch.bfloat16)
    sp = _sp("det", temperature=1.0, top_k=64, seed=777)
    outs = []
    for slot, mates in ((1, []), (5, [_sp("x", seed=1), _sp("y", seed=2)])):
        sps = mates + [sp]
        slots = list(range(2, 2 + len(mates))) + [slot]
        req = _prepare(backend, sps, slots=slots)
        rows = torch.cat(
            [
                torch.randn(len(mates), VOCAB, device="cuda", dtype=torch.bfloat16),
                logits,
            ]
        )
        # Same cache length in either slot (the test default varies by slot).
        info = _info(req)
        info.valid_cache_lengths = torch.full_like(info.valid_cache_lengths, 100)
        sampled, _ = backend.sample(_logits_output(rows), info)
        outs.append(sampled[-1].item())
    assert outs[0] == outs[1]

    req = _prepare(backend, [sp], slots=[1])
    info = _info(req)
    info.valid_cache_lengths = info.valid_cache_lengths + 1
    moved = []
    for _ in range(8):
        moved.append(backend.sample(_logits_output(logits.clone()), info)[0].item())
        info.valid_cache_lengths = info.valid_cache_lengths + 1
    assert len(set(moved)) > 1


def test_mixed_greedy_and_stochastic_rows_in_one_launch():
    torch.manual_seed(8)
    backend = _make_backend()
    sps = [
        _greedy_sp("mg0"),
        _sp("ms1", top_k=4),
        _greedy_sp("mg2"),
        _sp("ms3", top_k=4),
    ]
    req = _prepare(backend, sps)
    logits = _tie_free_logits(MAX_BS, top=4)
    sampled, _ = backend.sample(_logits_output(logits.clone()), _info(req))
    sampled = sampled.view(-1).cpu().tolist()
    argmax = logits.argmax(-1).cpu().tolist()
    top4 = logits.topk(4, dim=-1).indices.cpu()
    assert sampled[0] == argmax[0] and sampled[2] == argmax[2]
    assert sampled[1] in top4[1].tolist() and sampled[3] in top4[3].tolist()


# --------------------------------------------------------------------- #
# CUDA graph
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("vocab", [VOCAB, REAL_VOCAB])
def test_cuda_graph_replay_matches_eager(vocab):
    torch.manual_seed(9)
    n = MAX_N
    backend = _make_backend(vocab)
    if vocab == REAL_VOCAB and backend.tiling.dispatch is None:
        pytest.skip("no tuned dispatch bucket for this arch; fallback covered by VOCAB")
    sps = [
        _greedy_sp("cg0", vocab),
        _sp("cs1", vocab, top_k=16),
        _greedy_sp("cg2", vocab),
        _sp("cs3", vocab, top_p=0.9),
    ]

    # Capture warm-up: every row gathers slot 0, then clean its counts.
    backend.prepare_capture(bs=MAX_BS, num_tokens_per_req=n)
    logits_buf = torch.zeros(MAX_BS * n, vocab, device="cuda", dtype=torch.bfloat16)
    cand_buf = torch.zeros(MAX_BS, n, dtype=torch.int32, device="cuda")
    req_buf = torch.zeros(MAX_BS, dtype=torch.int64, device="cuda")
    info = _info(req_buf, vocab)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(2):
            backend.sample(_logits_output(logits_buf[:MAX_BS]), info)
            backend.verify(_logits_output(logits_buf), info, cand_buf)
    torch.cuda.current_stream().wait_stream(stream)
    backend.reset_capture_state()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        g_sampled, _ = backend.sample(_logits_output(logits_buf[:MAX_BS]), info)
        g_predict, g_accept = backend.verify(_logits_output(logits_buf), info, cand_buf)

    # Eager reference over the same state, then replay over identical inputs.
    req = _prepare(backend, sps, n=n)
    logits = _tie_free_logits(MAX_BS * n, vocab)
    candidates = torch.randint(0, vocab, (MAX_BS, n), dtype=torch.int32, device="cuda")
    e_sampled, _ = backend.sample(
        _logits_output(logits[:MAX_BS].clone()), _info(req, vocab)
    )
    e_predict, e_accept = backend.verify(
        _logits_output(logits.clone()), _info(req, vocab), candidates
    )
    e_sampled, e_predict, e_accept = (
        e_sampled.clone(),
        e_predict.clone(),
        e_accept.clone(),
    )

    # Same seeds and cache lengths (and no penalties): replay must reproduce eager bitwise.
    logits_buf.copy_(logits)
    cand_buf.copy_(candidates)
    req_buf.copy_(req)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(g_sampled.cpu(), e_sampled.cpu())
    torch.testing.assert_close(g_predict.cpu(), e_predict.cpu())
    torch.testing.assert_close(g_accept.cpu(), e_accept.cpu())


# --------------------------------------------------------------------- #
# review regressions
# --------------------------------------------------------------------- #
def test_registry_survives_missing_sonic():
    """Without sonic-sampler the facade reports ``available = False`` and the
    backend module still imports (the registry imports it for every backend
    name); ``sonic`` is simply not registered."""
    code = textwrap.dedent("""
        import sys
        sys.modules["sonic_sampler"] = None
        from tokenspeed_kernel.ops.sampling import sonic as facade
        assert not facade.available and facade.MAX_K is None
        from tokenspeed.runtime.sampling import registry
        from tokenspeed.runtime.sampling.backends import sonic  # noqa: F401
        assert "sonic" not in registry._BACKEND_REGISTRY
        """)
    subprocess.run([sys.executable, "-c", code], check=True)


def test_fused_greedy_route_picks_a_max_on_ties():
    """The kernel's greedy path may break exact bf16 ties differently from
    ``torch.argmax``; it must still return a maximal token."""
    backend = _make_backend()
    sps = [_greedy_sp(f"tg{i}") for i in range(MAX_BS - 1)] + [_sp("ts", top_k=4)]
    req = _prepare(backend, sps)
    logits = torch.full((MAX_BS, VOCAB), -10.0, device="cuda", dtype=torch.bfloat16)
    tie_cols = torch.tensor([3, 17, 256, VOCAB - 1], device="cuda")
    logits[:, tie_cols] = 5.0
    sampled, _ = backend.sample(_logits_output(logits.clone()), _info(req))
    greedy_rows = sampled.view(-1)[: MAX_BS - 1]
    assert torch.isin(greedy_rows, tie_cols.to(torch.int32)).all(), greedy_rows


def test_verify_outputs_are_persistent_buffers():
    backend = _make_backend()
    req = _prepare(backend, [_sp("pb0", top_k=8), _greedy_sp("pb1")], n=MAX_N)
    logits = _tie_free_logits(2 * MAX_N)
    candidates = torch.randint(0, VOCAB, (2, MAX_N), dtype=torch.int32, device="cuda")
    ptrs = set()
    for _ in range(3):
        predict, accept = backend.verify(
            _logits_output(logits.clone()), _info(req), candidates
        )
        ptrs.add((predict.data_ptr(), accept.data_ptr()))
    assert ptrs == {(backend.v_drafted.data_ptr(), backend.accept_buf.data_ptr())}


def test_logit_bias_admission_writes_only_its_device_row():
    backend = _make_backend(max_n=1)
    _prepare(backend, [_sp("lb", logit_bias={"7": 2.0, "9": -1.0})], slots=[3])
    bias = backend.buffers.bias
    assert bias[3, 7].item() == 2.0 and bias[3, 9].item() == -1.0
    assert bias.count_nonzero().item() == 2


def test_cuda_graph_padded_replay_matches_eager():
    """Production replays a captured batch size with the tail rows padded to
    slot 0 (the reserved sink). The live rows must equal an unpadded eager
    step over the same requests, for ``sample()`` and ``verify()``."""
    torch.manual_seed(11)
    n = MAX_N
    live = 2
    backend = _make_backend()
    sps = [_sp("pp0", top_k=16), _greedy_sp("pp1")]

    backend.prepare_capture(bs=MAX_BS, num_tokens_per_req=n)
    logits_buf = torch.zeros(MAX_BS * n, VOCAB, device="cuda", dtype=torch.bfloat16)
    cand_buf = torch.zeros(MAX_BS, n, dtype=torch.int32, device="cuda")
    req_buf = torch.zeros(MAX_BS, dtype=torch.int64, device="cuda")
    info = _info(req_buf)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(2):
            backend.sample(_logits_output(logits_buf[:MAX_BS]), info)
            backend.verify(_logits_output(logits_buf), info, cand_buf)
    torch.cuda.current_stream().wait_stream(stream)
    backend.reset_capture_state()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        g_sampled, _ = backend.sample(_logits_output(logits_buf[:MAX_BS]), info)
        g_predict, g_accept = backend.verify(_logits_output(logits_buf), info, cand_buf)

    req = _prepare(backend, sps, n=n)
    logits = _tie_free_logits(live * n)
    candidates = torch.randint(0, VOCAB, (live, n), dtype=torch.int32, device="cuda")
    e_sampled, _ = backend.sample(_logits_output(logits[:live].clone()), _info(req))
    e_predict, e_accept = backend.verify(
        _logits_output(logits.clone()), _info(req), candidates
    )
    e_sampled, e_predict, e_accept = (
        e_sampled.clone(),
        e_predict.clone(),
        e_accept.clone(),
    )

    # Padded layout: the live rows, then slot-0 rows whose outputs are discarded.
    logits_buf.zero_()
    logits_buf[: live * n].copy_(logits)
    cand_buf.zero_()
    cand_buf[:live].copy_(candidates)
    req_buf.zero_()
    req_buf[:live].copy_(req)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(g_sampled[:live].cpu(), e_sampled.cpu())
    torch.testing.assert_close(
        g_predict.view(MAX_BS, n)[:live].cpu(), e_predict.view(live, n).cpu()
    )
    torch.testing.assert_close(g_accept[:live].cpu(), e_accept.cpu())


def _chi2_deviate(counts: torch.Tensor, probs: torch.Tensor, n: int) -> float:
    """Normal deviate of the chi-square statistic over cells expecting >= 20 draws."""
    keep = probs * n >= 20
    obs = counts[keep].double()
    exp = probs[keep].double() * n
    chi2 = (((obs - exp) ** 2) / exp).sum().item()
    dof = int(keep.sum().item()) - 1
    return (chi2 - dof) / math.sqrt(2 * dof)


@pytest.mark.parametrize("scale", [2.0, 6.0], ids=["moderate", "peaked"])
def test_stochastic_sample_matches_softmax(scale):
    """Empirical draw vs fp32 softmax over the top-128 bf16 logits (sonic's
    bounded support). The peaked row is where a bf16 Gumbel-max score
    inflates the mode by several sd at this sample size."""
    torch.manual_seed(3)
    iters = 8000
    backend = _make_backend(max_n=1)
    req = _prepare(backend, [_sp(f"chi{i}") for i in range(MAX_BS)])
    row = (torch.randn(VOCAB) * scale).to(torch.bfloat16)
    row[200:] = -40.0
    logits = row.cuda().expand(MAX_BS, VOCAB).contiguous()
    offsets = torch.zeros(POOL + 1, dtype=torch.int32, device="cuda")
    info = SamplingBatchInfo(
        req_pool_indices=req,
        valid_cache_lengths=offsets,
        vocab_size=VOCAB,
        device="cuda",
    )
    counts = torch.zeros(VOCAB, dtype=torch.long, device="cuda")
    for it in range(iters):
        offsets.fill_(it)
        sampled, _ = backend.sample(_logits_output(logits), info)
        counts.scatter_add_(
            0, sampled.long(), torch.ones_like(sampled, dtype=torch.long)
        )
    vals, idx = row.float().topk(128)
    probs = torch.zeros(VOCAB)
    probs[idx] = torch.softmax(vals, -1)
    counts = counts.cpu()
    assert counts.sum().item() == iters * MAX_BS
    assert counts[probs == 0].sum().item() == 0, "a draw left the top-128 support"
    dev = _chi2_deviate(counts, probs, iters * MAX_BS)
    assert abs(dev) < 5.0, dev


def test_rejects_non_bf16_logits():
    backend = _make_backend(max_n=1)
    req = _prepare(backend, [_sp("h0", top_k=8)], slots=[1])
    logits = torch.randn(1, VOCAB, device="cuda", dtype=torch.float16)
    with pytest.raises(ValueError, match="bf16"):
        backend.sample(_logits_output(logits), _info(req))


def test_rejects_vocab_below_sonic_minimum():
    with pytest.raises(ValueError, match="512"):
        _backend_cls()(_make_config(vocab=256, max_n=1))


def test_logprobs_match_log_softmax_and_leave_logits_untouched():
    """Selected-token logprobs over the raw (grammar-masked) distribution, for
    ``sample()``; the caller's logits are not mutated by the mask."""
    torch.manual_seed(13)
    config = _make_config(max_n=1)
    config.enable_output_logprobs = True
    backend = _backend_cls()(config)
    req = _prepare(backend, [_sp("lp0", top_k=8), _greedy_sp("lp1")])
    logits = _tie_free_logits(2)
    allowed = torch.tensor([5, 900], dtype=torch.int32, device="cuda")
    mask = _allow_only(allowed)
    info = _info(req, vocab_mask=mask)
    bits = (mask[:, None, :] >> torch.arange(32, device="cuda")[None, :, None]) & 1
    keep = bits.permute(0, 2, 1).reshape(2, -1)[:, :VOCAB].bool()
    info.apply_vocab_mask = lambda logits, vocab_mask: logits.masked_fill_(
        ~keep, float("-inf")
    )
    out = _logits_output(logits.clone())
    sampled, _ = backend.sample(out, info)
    torch.testing.assert_close(out.next_token_logits.cpu(), logits.cpu())
    reference = torch.log_softmax(logits.float().masked_fill(~keep, float("-inf")), -1)
    expected = reference.gather(1, sampled.long()[:, None]).view(-1)
    torch.testing.assert_close(
        out.next_token_logprobs.float().cpu(), expected.cpu(), atol=1e-2, rtol=1e-2
    )


@pytest.mark.parametrize("draft_rank", [0, 3, 40])
def test_verify_first_token_marginal_matches_the_target(draft_rank):
    """Masked rejection with a point-draw draft d emits d with probability
    p(d) and otherwise draws from p without d, so the first emitted token's
    marginal is the target law regardless of d, and the acceptance rate is
    p(d)."""
    torch.manual_seed(200)
    iters = 6000
    n = 2
    backend = _make_backend(max_n=n)
    row = (torch.randn(VOCAB) * 2.0).to(torch.bfloat16)
    row[300:] = -40.0
    vals, idx = row.float().topk(128)
    probs = torch.zeros(VOCAB)
    probs[idx] = torch.softmax(vals, -1)
    draft = int(idx[draft_rank].item())
    req = _prepare(backend, [_sp(f"sv{i}") for i in range(MAX_BS)], n=n)
    logits = row.cuda().expand(MAX_BS * n, VOCAB).contiguous()
    candidates = torch.zeros(MAX_BS, n, dtype=torch.int32, device="cuda")
    candidates[:, 1] = draft
    offsets = torch.zeros(POOL + 1, dtype=torch.int32, device="cuda")
    info = SamplingBatchInfo(
        req_pool_indices=req,
        valid_cache_lengths=offsets,
        vocab_size=VOCAB,
        device="cuda",
    )
    counts = torch.zeros(VOCAB, dtype=torch.long, device="cuda")
    accepted = 0
    for step in range(iters):
        offsets.fill_(step * n)
        predict, accept = backend.verify(_logits_output(logits), info, candidates)
        first = predict.view(MAX_BS, n)[:, 0]
        counts.scatter_add_(0, first.long(), torch.ones_like(first, dtype=torch.long))
        accepted += int((accept == 2).sum().item())
    counts = counts.cpu()
    total = iters * MAX_BS
    assert counts[probs == 0].sum().item() == 0
    dev = _chi2_deviate(counts, probs, total)
    assert abs(dev) < 5.0, dev
    p_draft = probs[draft].item()
    assert (
        abs(accepted / total - p_draft)
        < 4 * math.sqrt(p_draft * (1 - p_draft) / total) + 0.01
    )


def test_penalized_greedy_pick_is_within_one_bf16_ulp_of_the_maximum():
    """sonic applies penalties in bf16 and ranks packed bf16 values, so a
    penalized greedy pick need not be the fp32 argmax (the fp32 backends'
    answer) but must lie within one bf16 ulp of the exactly penalized maximum."""
    torch.manual_seed(31)
    backend = _make_backend(max_n=1)
    sps = [
        _greedy_sp(f"pg{i}", frequency_penalty=0.87, presence_penalty=0.38)
        for i in range(MAX_BS)
    ]
    req = _prepare(backend, sps)
    counts = backend.buffers.repetition.counts.decode
    violations = 0
    for _ in range(200):
        logits = (torch.randn(MAX_BS, VOCAB, device="cuda") * 3.0).to(torch.bfloat16)
        before = counts[req].clone().float()
        sampled, _ = backend.sample(_logits_output(logits.clone()), _info(req))
        exact = logits.float() - 0.87 * before - 0.38 * (before > 0).float()
        best = exact.max(-1).values
        picked = exact.gather(1, sampled.long()[:, None]).view(-1)
        ulp = torch.finfo(torch.bfloat16).eps * torch.exp2(
            torch.floor(torch.log2(best.abs()))
        )
        violations += int((picked < best - ulp).sum().item())
    assert violations == 0, violations

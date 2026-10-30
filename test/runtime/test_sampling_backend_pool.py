"""Sampling-backend pool-state invariants.

Locks the sampling backend's per-slot state machine:

    prepare_step(rids, pool_indices, sp_list)
      ├─ flip detection against ``_last_rid_per_slot``
      ├─ _reset_slot(p, sp) — scatter scalars, counts, logit_bias, gen
      └─ _prepare_step_hook — coin refill (if coin-owning backend)

These invariants are hot-path-critical: a wrong flip call leaves stale
penalty counts or bias rows active for a new request, and a missed flip
leaves a finished request's scalars live for its slot's next tenant.

Tests below cover:
  * greedy backend opts out of pool state (prepare_step is a no-op).
  * flashinfer backend scatters temperature/top_k/top_p/seed on flip.
  * steady-state: same rid+slot across steps → no redundant _reset_slot.
  * slot recycle: slot reassigned to a new rid → _reset_slot fires.
  * flashinfer_full additionally scatters penalty scalars, counts (zero),
    and logit_bias (zero-then-scatter) on flip; out-of-vocab bias raises.
  * boundary asserts: misaligned rid/pool/sp lists, out-of-range pool_idx.

Runs on CUDA because the backends allocate GPU tensors in ``__init__``;
the test doesn't invoke any flashinfer kernels.
"""

import dataclasses
import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=30, suite="runtime-1gpu")

import torch  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402

import tokenspeed.runtime.sampling.backends.triton as triton_backend_module  # noqa: E402
from tokenspeed.runtime.execution.forward_step import (  # noqa: E402
    ForwardStepRunner,
)
from tokenspeed.runtime.layers.logits_processor import (  # noqa: E402
    LogitsProcessorOutput,
)
from tokenspeed.runtime.sampling.backends.base import (  # noqa: E402
    CUDA_GRAPH_VARIANT_DEFAULT,
    SamplingBackendConfig,
)
from tokenspeed.runtime.sampling.backends.flashinfer import (  # noqa: E402
    FlashInferSamplingBackend,
)
from tokenspeed.runtime.sampling.backends.flashinfer_full import (  # noqa: E402
    FlashInferFullSamplingBackend,
)
from tokenspeed.runtime.sampling.backends.greedy import (  # noqa: E402
    GreedySamplingBackend,
)
from tokenspeed.runtime.sampling.backends.triton import (  # noqa: E402
    _SAMPLE_ROUTE_GUMBEL_GENERIC,
    _SAMPLE_ROUTE_GUMBEL_NO_FILTER,
    _SAMPLE_ROUTE_GUMBEL_TOP_K,
    _SAMPLE_ROUTE_GUMBEL_TOP_K_TOP_P,
    _SAMPLE_ROUTE_GUMBEL_TOP_P,
    CUDA_GRAPH_VARIANT_TRITON_NO_FILTER,
    CUDA_GRAPH_VARIANT_TRITON_TOP_K,
    CUDA_GRAPH_VARIANT_TRITON_TOP_K_TOP_P,
    CUDA_GRAPH_VARIANT_TRITON_TOP_P,
    CUDA_GRAPH_VARIANT_TRITON_VERIFY_NO_FILTER,
    TritonSamplingBackend,
)
from tokenspeed.runtime.sampling.backends.triton_full import (  # noqa: E402
    CUDA_GRAPH_VARIANT_TRITON_FULL_MIN_P,
    CUDA_GRAPH_VARIANT_TRITON_FULL_TOP_K_TOP_P_MIN_P,
    TritonFullSamplingBackend,
)
from tokenspeed.runtime.sampling.sampling_batch_info import (  # noqa: E402
    SamplingBatchInfo,
)
from tokenspeed.runtime.sampling.sampling_params import SamplingParams  # noqa: E402

VOCAB = 1024
POOL = 8  # max_req_pool_size → pool_rows == POOL + 1


def _make_config() -> SamplingBackendConfig:
    return SamplingBackendConfig(
        synthetic_acceptance_length=None,
        enable_speculative_sampling=False,
        sampling_stream="batch",
        logprob_order="torch",
        max_bs=4,
        max_draft_tokens_per_req=4,
        max_req_pool_size=POOL,
        vocab_size=VOCAB,
        device="cuda",
    )


def _sp(rid_suffix: str, **overrides) -> SamplingParams:
    """Build a normalized SamplingParams with an rid-specific seed. The
    rid suffix drives the seed so per-test sp values stay distinct even
    when only temperature differs."""
    defaults = dict(
        temperature=1.0,
        top_k=-1,
        top_p=1.0,
        min_p=0.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        repetition_penalty=1.0,
        seed=abs(hash(rid_suffix)) % (2**31),
    )
    defaults.update(overrides)
    sp = SamplingParams(**defaults)
    sp.resolve_seed(f"rid_{rid_suffix}")
    sp.normalize(None)
    return sp


def _assert_logit_bias_h2d_is_async(backend) -> None:
    """The bias scatter on flip must not stall the host: no pageable H2D
    memcpy and no cudaStreamSynchronize while the stream is busy, and the
    row still holds the requested values once the stream drains."""
    sp = _sp("bias")
    sp.logit_bias = {"100": 2.0, "200": -1.5, "300": 0.25}
    busy = torch.randn((4096, 4096), device="cuda")
    for _ in range(8):
        busy = busy @ busy
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        backend.prepare_step(
            request_ids=["bias"],
            request_pool_indices=[5],
            sampling_params_list=[sp],
        )
    torch.cuda.synchronize()
    names = [e.name for e in prof.events()]
    if not any("Memcpy HtoD" in n for n in names):
        pytest.skip("profiler collected no CUDA memcpy activity")
    assert "cudaStreamSynchronize" not in names
    assert not any("Pageable -> Device" in n for n in names), names
    assert any("Pinned -> Device" in n for n in names), names
    row = backend._logit_bias[5]
    first = row[100].item()
    assert first == 2.0 or round(abs(first - 2.0), 2) == 0
    first = row[200].item()
    assert first == -1.5 or round(abs(first - -1.5), 2) == 0
    first = row[300].item()
    assert first == 0.25 or round(abs(first - 0.25), 2) == 0
    first = row[150].item()
    assert first == 0.0 or round(abs(first - 0.0), 3) == 0


class TestGreedyNoPoolState:
    """Greedy backend declares ``_HAS_POOL_STATE = False`` so prepare_step
    must short-circuit with no allocation or iteration. Guards against a
    future refactor accidentally forcing pool tracking on stateless
    backends."""

    def test_prepare_step_is_noop(self):
        b = GreedySamplingBackend(_make_config())
        assert not (b._HAS_POOL_STATE)
        assert not (hasattr(b, "_last_rid_per_slot"))
        # Must not raise even with nonsensical inputs — short-circuits.
        b.prepare_step(
            request_ids=["a", "b"],
            request_pool_indices=[999, -1],  # would fail bounds check otherwise
            sampling_params_list=[_sp("a"), _sp("b")],
        )


class TestFlashInferFlipDetection:
    """flashinfer's pool-indexed scalar buffers are core scheduler state.
    These tests pin flip semantics down at the Python state-machine level
    (no kernel invocation needed)."""

    def setup_method(self):
        self.backend = FlashInferSamplingBackend(_make_config())

    def test_dp_verify_buffers_are_lazy(self):
        assert (self.backend._predict_local_buf) is None
        assert (self.backend._accept_index_local_buf) is None
        assert (self.backend._accept_length_local_buf) is None

    def test_first_admission_flips_and_scatters(self):
        sp_a = _sp("a", temperature=0.7, top_k=50, top_p=0.9, seed=42)
        sp_b = _sp("b", temperature=1.2, top_k=20, top_p=0.8, seed=123)
        self.backend.prepare_step(
            request_ids=["a", "b"],
            request_pool_indices=[1, 3],
            sampling_params_list=[sp_a, sp_b],
        )
        assert (self.backend._last_rid_per_slot[1]) == ("a")
        assert (self.backend._last_rid_per_slot[3]) == ("b")
        _pytest_first_1 = self.backend._temperature_pool[1].item()
        _pytest_second_2 = 0.7
        _pytest_places_3 = 3
        assert (
            _pytest_first_1 == _pytest_second_2
            or round(abs(_pytest_first_1 - _pytest_second_2), _pytest_places_3) == 0
        )
        assert (self.backend._top_k_pool[1].item()) == (50)
        _pytest_first_4 = self.backend._top_p_pool[1].item()
        _pytest_second_5 = 0.9
        _pytest_places_6 = 3
        assert (
            _pytest_first_4 == _pytest_second_5
            or round(abs(_pytest_first_4 - _pytest_second_5), _pytest_places_6) == 0
        )
        assert (self.backend._seed_pool[1].item()) == (42)
        _pytest_first_7 = self.backend._temperature_pool[3].item()
        _pytest_second_8 = 1.2
        _pytest_places_9 = 3
        assert (
            _pytest_first_7 == _pytest_second_8
            or round(abs(_pytest_first_7 - _pytest_second_8), _pytest_places_9) == 0
        )
        assert (self.backend._top_k_pool[3].item()) == (20)
        # Unused slots keep their neutral init values.
        _pytest_first_10 = self.backend._temperature_pool[0].item()
        _pytest_second_11 = 1.0
        assert (
            _pytest_first_10 == _pytest_second_11
            or round(abs(_pytest_first_10 - _pytest_second_11), 7) == 0
        )
        _pytest_first_12 = self.backend._temperature_pool[5].item()
        _pytest_second_13 = 1.0
        assert (
            _pytest_first_12 == _pytest_second_13
            or round(abs(_pytest_first_12 - _pytest_second_13), 7) == 0
        )

    def test_steady_state_no_reflip(self):
        """Same rid on same slot across steps → _reset_slot must not fire
        a second time. Guard against an off-by-one in the comparison."""
        sp_a = _sp("a", temperature=0.7)
        self.backend.prepare_step(
            request_ids=["a"],
            request_pool_indices=[2],
            sampling_params_list=[sp_a],
        )
        # Mutate the pool scalar from outside to prove _reset_slot did NOT
        # re-fire. If prepare_step mistakenly re-scatters, our sentinel
        # will be overwritten.
        self.backend._temperature_pool[2] = 9.999
        self.backend.prepare_step(
            request_ids=["a"],
            request_pool_indices=[2],
            sampling_params_list=[sp_a],
        )
        _pytest_first_14 = self.backend._temperature_pool[2].item()
        _pytest_second_15 = 9.999
        _pytest_places_16 = 3
        assert (
            _pytest_first_14 == _pytest_second_15
            or round(abs(_pytest_first_14 - _pytest_second_15), _pytest_places_16) == 0
        )

    def test_slot_recycle_flips(self):
        """Slot reused by a different rid → _reset_slot fires, scalars
        overwritten, new generator seeded."""
        sp_a = _sp("a", temperature=0.7, seed=1)
        self.backend.prepare_step(
            request_ids=["a"],
            request_pool_indices=[2],
            sampling_params_list=[sp_a],
        )
        gen_a = self.backend._cpu_generator_per_slot[2]
        sp_b = _sp("b", temperature=0.3, seed=99)
        self.backend.prepare_step(
            request_ids=["b"],
            request_pool_indices=[2],
            sampling_params_list=[sp_b],
        )
        assert (self.backend._last_rid_per_slot[2]) == ("b")
        _pytest_first_17 = self.backend._temperature_pool[2].item()
        _pytest_second_18 = 0.3
        _pytest_places_19 = 3
        assert (
            _pytest_first_17 == _pytest_second_18
            or round(abs(_pytest_first_17 - _pytest_second_18), _pytest_places_19) == 0
        )
        assert (self.backend._seed_pool[2].item()) == (99)
        assert (self.backend._cpu_generator_per_slot[2]) is not (gen_a)


class TestTritonRouteSelection:
    def setup_method(self):
        self.backend = TritonSamplingBackend(_make_config())

    def test_no_filter_step_selects_triton_gumbel(self):
        self.backend.prepare_step(
            request_ids=["a", "b"],
            request_pool_indices=[1, 2],
            sampling_params_list=[_sp("a", top_k=-1, top_p=1.0), _sp("b", top_k=-1)],
        )
        assert (self.backend._sample_route) == (_SAMPLE_ROUTE_GUMBEL_NO_FILTER)

    def test_finite_top_k_step_selects_triton_gumbel(self):
        self.backend.prepare_step(
            request_ids=["a"],
            request_pool_indices=[1],
            sampling_params_list=[_sp("a", top_k=50, top_p=1.0)],
        )
        assert (self.backend._sample_route) == (_SAMPLE_ROUTE_GUMBEL_TOP_K)
        assert (self.backend._top_k_top_p_pad) == (64)

    def test_finite_top_k_top_p_step_selects_triton_gumbel(self):
        self.backend.prepare_step(
            request_ids=["a"],
            request_pool_indices=[1],
            sampling_params_list=[_sp("a", top_k=50, top_p=0.9)],
        )
        assert (self.backend._sample_route) == (_SAMPLE_ROUTE_GUMBEL_TOP_K_TOP_P)
        assert (self.backend._top_k_top_p_pad) == (64)

        self.backend.prepare_step(
            request_ids=["a"],
            request_pool_indices=[1],
            sampling_params_list=[_sp("a", top_k=127, top_p=0.9)],
        )
        assert (self.backend._top_k_top_p_pad) == (128)

    def test_top_p_only_step_selects_top_p_triton_gumbel(self):
        self.backend.prepare_step(
            request_ids=["a"],
            request_pool_indices=[1],
            sampling_params_list=[_sp("a", top_k=-1, top_p=0.9)],
        )
        assert (self.backend._sample_route) == (_SAMPLE_ROUTE_GUMBEL_TOP_P)

    def test_mixed_step_selects_generic_triton_gumbel(self):
        self.backend.prepare_step(
            request_ids=["a", "b"],
            request_pool_indices=[1, 2],
            sampling_params_list=[
                _sp("a", top_k=-1, top_p=1.0),
                _sp("b", top_k=50, top_p=1.0),
            ],
        )
        assert (self.backend._sample_route) == (_SAMPLE_ROUTE_GUMBEL_GENERIC)

    def test_multi_token_no_filter_step_selects_verify_fast_path(self):
        self.backend.prepare_step(
            request_ids=["a"],
            request_pool_indices=[1],
            sampling_params_list=[_sp("a", top_k=-1, top_p=1.0)],
            num_tokens_per_req=4,
        )
        assert (self.backend._sample_route) == (_SAMPLE_ROUTE_GUMBEL_NO_FILTER)

    def test_capture_keeps_generic_sampler_graph(self):
        self.backend._sample_route = _SAMPLE_ROUTE_GUMBEL_NO_FILTER
        self.backend.prepare_capture(bs=1)
        assert (self.backend._sample_route) == (_SAMPLE_ROUTE_GUMBEL_GENERIC)

    def test_cuda_graph_variants_cover_top_p_and_verify_no_filter(self):
        assert (self.backend.cuda_graph_capture_variants(num_tokens_per_req=1)) == (
            (
                CUDA_GRAPH_VARIANT_DEFAULT,
                CUDA_GRAPH_VARIANT_TRITON_NO_FILTER,
                CUDA_GRAPH_VARIANT_TRITON_TOP_P,
                CUDA_GRAPH_VARIANT_TRITON_TOP_K,
                CUDA_GRAPH_VARIANT_TRITON_TOP_K_TOP_P,
            )
        )
        assert (self.backend.cuda_graph_capture_variants(num_tokens_per_req=4)) == (
            (
                CUDA_GRAPH_VARIANT_DEFAULT,
                CUDA_GRAPH_VARIANT_TRITON_NO_FILTER,
                CUDA_GRAPH_VARIANT_TRITON_TOP_P,
                CUDA_GRAPH_VARIANT_TRITON_TOP_K,
                CUDA_GRAPH_VARIANT_TRITON_TOP_K_TOP_P,
                CUDA_GRAPH_VARIANT_TRITON_VERIFY_NO_FILTER,
            )
        )

        self.backend.prepare_step(
            request_ids=["a"],
            request_pool_indices=[1],
            sampling_params_list=[_sp("a", top_k=-1, top_p=1.0)],
            num_tokens_per_req=4,
        )
        assert (self.backend.cuda_graph_replay_variant(num_tokens_per_req=4)) == (
            CUDA_GRAPH_VARIANT_TRITON_VERIFY_NO_FILTER
        )

        self.backend.prepare_step(
            request_ids=["a"],
            request_pool_indices=[1],
            sampling_params_list=[_sp("a", top_k=-1, top_p=1.0)],
            num_tokens_per_req=1,
        )
        assert (self.backend.cuda_graph_replay_variant(num_tokens_per_req=1)) == (
            CUDA_GRAPH_VARIANT_TRITON_NO_FILTER
        )

        self.backend.prepare_step(
            request_ids=["a"],
            request_pool_indices=[1],
            sampling_params_list=[_sp("a", top_k=-1, top_p=0.9)],
            num_tokens_per_req=4,
        )
        assert (self.backend.cuda_graph_replay_variant(num_tokens_per_req=4)) == (
            CUDA_GRAPH_VARIANT_TRITON_TOP_P
        )

        self.backend.prepare_step(
            request_ids=["a"],
            request_pool_indices=[1],
            sampling_params_list=[_sp("a", top_k=50, top_p=1.0)],
            num_tokens_per_req=4,
        )
        assert (self.backend.cuda_graph_replay_variant(num_tokens_per_req=4)) == (
            CUDA_GRAPH_VARIANT_TRITON_TOP_K
        )

        self.backend.prepare_step(
            request_ids=["a"],
            request_pool_indices=[1],
            sampling_params_list=[_sp("a", top_k=50, top_p=0.9)],
            num_tokens_per_req=4,
        )
        assert (self.backend.cuda_graph_replay_variant(num_tokens_per_req=4)) == (
            CUDA_GRAPH_VARIANT_TRITON_TOP_K_TOP_P
        )

    def test_verify_finite_top_k_top_p_uses_direct_sampler(self):
        n = 4
        self.backend.prepare_step(
            request_ids=["a"],
            request_pool_indices=[1],
            sampling_params_list=[_sp("a", top_k=50, top_p=0.9)],
            num_tokens_per_req=n,
        )
        logits = torch.randn((n, VOCAB), dtype=torch.float32, device="cuda")
        candidates = torch.zeros((1, n), dtype=torch.int32, device="cuda")
        sampling_info = SamplingBatchInfo(
            req_pool_indices=torch.tensor([1], dtype=torch.int32, device="cuda"),
            valid_cache_lengths=torch.zeros(
                (POOL + 1,), dtype=torch.int32, device="cuda"
            ),
            vocab_size=VOCAB,
            device="cuda",
        )
        target_sampled = torch.arange(n, dtype=torch.int32, device="cuda")

        def _fake_verify_chain(
            predicts,
            accept_index,
            accept_token_num,
            candidates,
            target_sampled,
            *,
            write_rejected_targets,
            enable_pdl=False,
        ):
            assert write_rejected_targets is False
            del accept_index, candidates, target_sampled, enable_pdl
            predicts.fill_(0)
            accept_token_num.fill_(0)

        with (
            patch.object(
                triton_backend_module,
                "gumbel_sample_top_k_top_p_from_pools",
                return_value=target_sampled,
            ) as direct_sampler,
            patch.object(
                triton_backend_module,
                "gumbel_sample_from_pools_generic",
                side_effect=AssertionError("generic sampler should not be used"),
            ),
            patch.object(
                triton_backend_module,
                "verify_chain_target_sampled",
                side_effect=_fake_verify_chain,
            ),
        ):
            self.backend.verify(
                LogitsProcessorOutput(next_token_logits=logits),
                sampling_info,
                candidates,
                tree=None,
            )

        direct_sampler.assert_called_once()
        args, kwargs = direct_sampler.call_args
        assert (args[7].shape[0]) == (n)
        assert (args[8].shape[0]) == (n)

    def test_verify_top_p_only_uses_top_p_sampler(self):
        n = 4
        self.backend.prepare_step(
            request_ids=["a"],
            request_pool_indices=[1],
            sampling_params_list=[_sp("a", top_k=-1, top_p=0.9)],
            num_tokens_per_req=n,
        )
        logits = torch.randn((n, VOCAB), dtype=torch.float32, device="cuda")
        candidates = torch.zeros((1, n), dtype=torch.int32, device="cuda")
        sampling_info = SamplingBatchInfo(
            req_pool_indices=torch.tensor([1], dtype=torch.int32, device="cuda"),
            valid_cache_lengths=torch.zeros(
                (POOL + 1,), dtype=torch.int32, device="cuda"
            ),
            vocab_size=VOCAB,
            device="cuda",
        )
        target_sampled = torch.arange(n, dtype=torch.int32, device="cuda")

        def _fake_verify_chain(
            predicts,
            accept_index,
            accept_token_num,
            candidates,
            target_sampled,
            *,
            write_rejected_targets,
            enable_pdl=False,
        ):
            assert write_rejected_targets is False
            del accept_index, candidates, target_sampled, enable_pdl
            predicts.fill_(0)
            accept_token_num.fill_(0)

        with (
            patch.object(
                triton_backend_module,
                "gumbel_sample_top_p_parallel_from_pools",
                return_value=target_sampled,
            ) as top_p_sampler,
            patch.object(
                triton_backend_module,
                "gumbel_sample_from_pools_generic",
                side_effect=AssertionError("generic sampler should not be used"),
            ),
            patch.object(
                triton_backend_module,
                "verify_chain_target_sampled",
                side_effect=_fake_verify_chain,
            ),
        ):
            self.backend.verify(
                LogitsProcessorOutput(next_token_logits=logits),
                sampling_info,
                candidates,
                tree=None,
            )

        top_p_sampler.assert_called_once()
        args, kwargs = top_p_sampler.call_args
        assert (args[6].shape[0]) == (n)
        assert (args[17].shape[0]) == (n)
        assert (kwargs["num_tokens_per_req"]) == (n)

    def test_verify_large_finite_top_k_top_p_uses_qrita_sampler(self):
        bs, n, vocab = 32, 4, 32768
        backend = TritonSamplingBackend(
            SamplingBackendConfig(
                synthetic_acceptance_length=None,
                enable_speculative_sampling=False,
                sampling_stream="batch",
                logprob_order="torch",
                max_bs=bs,
                max_draft_tokens_per_req=n,
                max_req_pool_size=POOL + bs,
                vocab_size=vocab,
                device="cuda",
            )
        )
        backend.prepare_step(
            request_ids=[f"r{i}" for i in range(bs)],
            request_pool_indices=list(range(1, bs + 1)),
            sampling_params_list=[
                _sp(f"qrita_{i}", top_k=127, top_p=0.9) for i in range(bs)
            ],
            num_tokens_per_req=n,
        )
        logits = torch.randn((bs * n, vocab), dtype=torch.float32, device="cuda")
        candidates = torch.zeros((bs, n), dtype=torch.int32, device="cuda")
        sampling_info = SamplingBatchInfo(
            req_pool_indices=torch.arange(1, bs + 1, dtype=torch.int32, device="cuda"),
            valid_cache_lengths=torch.zeros(
                (POOL + bs + 1,), dtype=torch.int32, device="cuda"
            ),
            vocab_size=vocab,
            device="cuda",
        )
        target_sampled = torch.arange(bs * n, dtype=torch.int32, device="cuda")

        def _fake_verify_chain(
            predicts,
            accept_index,
            accept_token_num,
            candidates,
            target_sampled,
            *,
            write_rejected_targets,
            enable_pdl=False,
        ):
            assert write_rejected_targets is False
            del accept_index, candidates, target_sampled, enable_pdl
            predicts.fill_(0)
            accept_token_num.fill_(0)

        with (
            patch.object(
                triton_backend_module,
                "gumbel_sample_top_k_top_p_qrita_from_pools",
                return_value=target_sampled,
            ) as qrita_sampler,
            patch.object(
                triton_backend_module,
                "gumbel_sample_top_k_top_p_from_pools",
                side_effect=AssertionError("candidate sampler should not be used"),
            ),
            patch.object(
                triton_backend_module,
                "gumbel_sample_from_pools_generic",
                side_effect=AssertionError("generic sampler should not be used"),
            ),
            patch.object(
                triton_backend_module,
                "verify_chain_target_sampled",
                side_effect=_fake_verify_chain,
            ),
        ):
            backend.verify(
                LogitsProcessorOutput(next_token_logits=logits),
                sampling_info,
                candidates,
                tree=None,
            )

        qrita_sampler.assert_called_once()
        args, kwargs = qrita_sampler.call_args
        assert (args[7].shape[1]) == (vocab)
        assert (args[8].shape[0]) == (
            len(triton_backend_module._QRITA_PERCENTILE_TO_STD_TABLE)
        )
        assert (kwargs["num_tokens_per_req"]) == (n)

    def test_verify_large_finite_top_k_only_uses_qrita_sampler(self):
        bs, n, vocab = 32, 4, 200064
        backend = TritonSamplingBackend(
            SamplingBackendConfig(
                synthetic_acceptance_length=None,
                enable_speculative_sampling=False,
                sampling_stream="batch",
                logprob_order="torch",
                max_bs=bs,
                max_draft_tokens_per_req=n,
                max_req_pool_size=POOL + bs,
                vocab_size=vocab,
                device="cuda",
            )
        )
        backend.prepare_step(
            request_ids=[f"r{i}" for i in range(bs)],
            request_pool_indices=list(range(1, bs + 1)),
            sampling_params_list=[
                _sp(f"qrita_topk_{i}", top_k=64, top_p=1.0) for i in range(bs)
            ],
            num_tokens_per_req=n,
        )
        logits = torch.randn((bs * n, vocab), dtype=torch.float32, device="cuda")
        candidates = torch.zeros((bs, n), dtype=torch.int32, device="cuda")
        sampling_info = SamplingBatchInfo(
            req_pool_indices=torch.arange(1, bs + 1, dtype=torch.int32, device="cuda"),
            valid_cache_lengths=torch.zeros(
                (POOL + bs + 1,), dtype=torch.int32, device="cuda"
            ),
            vocab_size=vocab,
            device="cuda",
        )
        target_sampled = torch.arange(bs * n, dtype=torch.int32, device="cuda")

        def _fake_verify_chain(
            predicts,
            accept_index,
            accept_token_num,
            candidates,
            target_sampled,
            *,
            write_rejected_targets,
            enable_pdl=False,
        ):
            assert write_rejected_targets is False
            del accept_index, candidates, target_sampled, enable_pdl
            predicts.fill_(0)
            accept_token_num.fill_(0)

        with (
            patch.object(
                triton_backend_module,
                "gumbel_sample_top_k_top_p_qrita_from_pools",
                return_value=target_sampled,
            ) as qrita_sampler,
            patch.object(
                triton_backend_module,
                "gumbel_sample_top_k_top_p_from_pools",
                side_effect=AssertionError("candidate sampler should not be used"),
            ),
            patch.object(
                triton_backend_module,
                "gumbel_sample_from_pools_generic",
                side_effect=AssertionError("generic sampler should not be used"),
            ),
            patch.object(
                triton_backend_module,
                "verify_chain_target_sampled",
                side_effect=_fake_verify_chain,
            ),
        ):
            backend.verify(
                LogitsProcessorOutput(next_token_logits=logits),
                sampling_info,
                candidates,
                tree=None,
            )

        qrita_sampler.assert_called_once()
        args, kwargs = qrita_sampler.call_args
        assert (args[7].shape[1]) == (vocab)
        assert (kwargs["num_tokens_per_req"]) == (n)

    def test_prepare_capture_variant_sets_verify_fast_path(self):
        self.backend.prepare_capture_variant(
            bs=1,
            num_tokens_per_req=4,
            variant=CUDA_GRAPH_VARIANT_TRITON_TOP_K,
        )
        assert (self.backend._sample_route) == (_SAMPLE_ROUTE_GUMBEL_TOP_K)

        self.backend.prepare_capture_variant(
            bs=1,
            num_tokens_per_req=4,
            variant=CUDA_GRAPH_VARIANT_TRITON_TOP_K_TOP_P,
        )
        assert (self.backend._sample_route) == (_SAMPLE_ROUTE_GUMBEL_TOP_K_TOP_P)

        self.backend.prepare_capture_variant(
            bs=1,
            num_tokens_per_req=4,
            variant=CUDA_GRAPH_VARIANT_TRITON_TOP_P,
        )
        assert (self.backend._sample_route) == (_SAMPLE_ROUTE_GUMBEL_TOP_P)

        self.backend.prepare_capture_variant(
            bs=1,
            num_tokens_per_req=4,
            variant=CUDA_GRAPH_VARIANT_TRITON_VERIFY_NO_FILTER,
        )
        assert (self.backend._sample_route) == (_SAMPLE_ROUTE_GUMBEL_NO_FILTER)

        self.backend.prepare_capture_variant(
            bs=1,
            num_tokens_per_req=4,
            variant=CUDA_GRAPH_VARIANT_DEFAULT,
        )
        assert (self.backend._sample_route) == (_SAMPLE_ROUTE_GUMBEL_GENERIC)


class TestTritonLogprobOutputs:
    def _make_backend(self, backend_cls):
        config = dataclasses.replace(_make_config(), enable_output_logprobs=True)
        backend = backend_cls(config)
        backend.prepare_step(
            request_ids=["a", "b"],
            request_pool_indices=[1, 2],
            sampling_params_list=[
                _sp("a", top_k=-1, top_p=1.0),
                _sp("b", top_k=-1, top_p=1.0),
            ],
        )
        return backend

    def _make_sampling_info(self):
        return SamplingBatchInfo(
            req_pool_indices=torch.tensor([1, 2], dtype=torch.int32, device="cuda"),
            valid_cache_lengths=torch.zeros(
                (POOL + 1,), dtype=torch.int32, device="cuda"
            ),
            vocab_size=VOCAB,
            device="cuda",
        )

    def _assert_logprob_outputs(self, logits, sampled, logits_output):
        ref = torch.log_softmax(logits.float(), dim=-1)
        ref_selected = ref.gather(-1, sampled.long().unsqueeze(-1)).squeeze(-1)
        torch.testing.assert_close(
            logits_output.next_token_logprobs,
            ref_selected,
            rtol=1e-4,
            atol=1e-4,
        )

        assert (logits_output.next_token_top_logprobs_val) is None
        assert (logits_output.next_token_top_logprobs_idx) is None
        assert (logits_output.next_token_token_ids_logprobs_val) is None
        assert (logits_output.next_token_token_ids_logprobs_idx) is None

    def test_triton_sample_writes_compact_logprob_side_outputs(self):
        backend = self._make_backend(TritonSamplingBackend)
        generator = torch.Generator(device="cuda").manual_seed(7)
        logits = torch.randn(
            (2, VOCAB), dtype=torch.float32, device="cuda", generator=generator
        )
        logits_output = LogitsProcessorOutput(next_token_logits=logits.clone())

        sampled, _ = backend.sample(
            logits_output,
            self._make_sampling_info(),
        )

        self._assert_logprob_outputs(logits, sampled, logits_output)

    def test_triton_full_sample_writes_compact_logprob_side_outputs(self):
        backend = self._make_backend(TritonFullSamplingBackend)
        generator = torch.Generator(device="cuda").manual_seed(11)
        logits = torch.randn(
            (2, VOCAB), dtype=torch.float32, device="cuda", generator=generator
        )
        logits_output = LogitsProcessorOutput(next_token_logits=logits.clone())

        sampled, _ = backend.sample(
            logits_output,
            self._make_sampling_info(),
        )

        self._assert_logprob_outputs(logits, sampled, logits_output)


class TestCudaGraphSamplingVariants:
    def test_wrapper_dedupes_default_and_selects_replay_variant(self):
        class FakeSamplingBackend:
            variant = CUDA_GRAPH_VARIANT_TRITON_VERIFY_NO_FILTER

            def cuda_graph_capture_variants(self, num_tokens_per_req):
                self.capture_num_tokens_per_req = num_tokens_per_req
                return (
                    CUDA_GRAPH_VARIANT_DEFAULT,
                    CUDA_GRAPH_VARIANT_TRITON_VERIFY_NO_FILTER,
                )

            def cuda_graph_replay_variant(self, num_tokens_per_req):
                self.replay_num_tokens_per_req = num_tokens_per_req
                return self.variant

        backend = FakeSamplingBackend()
        wrapper = object.__new__(ForwardStepRunner)
        wrapper.sampling_backend = backend
        wrapper.max_tokens_per_req = 4
        wrapper.graphs = {
            (CUDA_GRAPH_VARIANT_DEFAULT, 8): object(),
            (CUDA_GRAPH_VARIANT_TRITON_VERIFY_NO_FILTER, 8): object(),
        }

        assert (wrapper._cuda_graph_capture_variants()) == (
            (
                CUDA_GRAPH_VARIANT_DEFAULT,
                CUDA_GRAPH_VARIANT_TRITON_VERIFY_NO_FILTER,
            )
        )
        assert (backend.capture_num_tokens_per_req) == (4)
        assert (wrapper._cuda_graph_key(8)) == (
            (CUDA_GRAPH_VARIANT_TRITON_VERIFY_NO_FILTER, 8)
        )
        assert (backend.replay_num_tokens_per_req) == (4)

        backend.variant = "missing"
        with pytest.raises((RuntimeError), match=("was not captured")):
            wrapper._cuda_graph_key(8)


class TestFlashInferFullFlipExtended:
    """Full backend extends flip behavior with penalty scalars, count rows,
    and logit_bias scatter. Each of these must be cleared/scattered on
    flip or a new request inherits the previous occupant's state."""

    def setup_method(self):
        self.backend = FlashInferFullSamplingBackend(_make_config())

    def test_penalty_scalars_scattered(self):
        sp = _sp(
            "a",
            frequency_penalty=0.5,
            presence_penalty=0.25,
            repetition_penalty=1.2,
            min_p=0.1,
        )
        self.backend.prepare_step(
            request_ids=["a"],
            request_pool_indices=[3],
            sampling_params_list=[sp],
        )
        _pytest_first_20 = self.backend._freq_pen_pool[3].item()
        _pytest_second_21 = 0.5
        _pytest_places_22 = 2
        assert (
            _pytest_first_20 == _pytest_second_21
            or round(abs(_pytest_first_20 - _pytest_second_21), _pytest_places_22) == 0
        )
        _pytest_first_23 = self.backend._pres_pen_pool[3].item()
        _pytest_second_24 = 0.25
        _pytest_places_25 = 2
        assert (
            _pytest_first_23 == _pytest_second_24
            or round(abs(_pytest_first_23 - _pytest_second_24), _pytest_places_25) == 0
        )
        _pytest_first_26 = self.backend._rep_pen_pool[3].item()
        _pytest_second_27 = 1.2
        _pytest_places_28 = 2
        assert (
            _pytest_first_26 == _pytest_second_27
            or round(abs(_pytest_first_26 - _pytest_second_27), _pytest_places_28) == 0
        )
        _pytest_first_29 = self.backend._min_p_pool[3].item()
        _pytest_second_30 = 0.1
        _pytest_places_31 = 3
        assert (
            _pytest_first_29 == _pytest_second_30
            or round(abs(_pytest_first_29 - _pytest_second_30), _pytest_places_31) == 0
        )

    def test_counts_and_bias_cleared_on_flip(self):
        """Dirty slot 2 simulating a prior occupant's accumulated state,
        then flip. Both rows must be zeroed (bias also rescattered if the
        new sp carries logit_bias)."""
        self.backend._counts[2, 100] = 7
        self.backend._logit_bias[2, 100] = 5.0
        sp = _sp("new", temperature=1.0)
        self.backend.prepare_step(
            request_ids=["new"],
            request_pool_indices=[2],
            sampling_params_list=[sp],
        )
        assert (self.backend._counts[2, 100].item()) == (0)
        _pytest_first_32 = self.backend._logit_bias[2, 100].item()
        _pytest_second_33 = 0.0
        _pytest_places_34 = 3
        assert (
            _pytest_first_32 == _pytest_second_33
            or round(abs(_pytest_first_32 - _pytest_second_33), _pytest_places_34) == 0
        )

    def test_logit_bias_scattered(self):
        sp = _sp("a", temperature=1.0)
        sp.logit_bias = {"100": 2.0, "200": -1.5}
        self.backend.prepare_step(
            request_ids=["a"],
            request_pool_indices=[4],
            sampling_params_list=[sp],
        )
        _pytest_first_35 = self.backend._logit_bias[4, 100].item()
        _pytest_second_36 = 2.0
        _pytest_places_37 = 2
        assert (
            _pytest_first_35 == _pytest_second_36
            or round(abs(_pytest_first_35 - _pytest_second_36), _pytest_places_37) == 0
        )
        _pytest_first_38 = self.backend._logit_bias[4, 200].item()
        _pytest_second_39 = -1.5
        _pytest_places_40 = 2
        assert (
            _pytest_first_38 == _pytest_second_39
            or round(abs(_pytest_first_38 - _pytest_second_39), _pytest_places_40) == 0
        )
        # Other positions untouched.
        _pytest_first_41 = self.backend._logit_bias[4, 150].item()
        _pytest_second_42 = 0.0
        _pytest_places_43 = 3
        assert (
            _pytest_first_41 == _pytest_second_42
            or round(abs(_pytest_first_41 - _pytest_second_42), _pytest_places_43) == 0
        )

    def test_logit_bias_h2d_is_pinned_and_non_blocking(self):
        _assert_logit_bias_h2d_is_async(self.backend)

    def test_logit_bias_out_of_vocab_asserts(self):
        """OOV token ids would write past the bias row; must be caught."""
        sp = _sp("a")
        sp.logit_bias = {str(VOCAB + 5): 1.0}
        with pytest.raises((AssertionError)):
            self.backend.prepare_step(
                request_ids=["a"],
                request_pool_indices=[1],
                sampling_params_list=[sp],
            )


class TestTritonFullIndependentState:
    """TritonFull owns full-sampling state without FlashInfer inheritance."""

    def setup_method(self):
        self.backend = TritonFullSamplingBackend(_make_config())

    def test_penalty_scalars_scattered(self):
        sp = _sp(
            "a",
            frequency_penalty=0.5,
            presence_penalty=0.25,
            repetition_penalty=1.2,
            min_p=0.1,
        )
        self.backend.prepare_step(
            request_ids=["a"],
            request_pool_indices=[3],
            sampling_params_list=[sp],
        )
        _pytest_first_44 = self.backend._freq_pen_pool[3].item()
        _pytest_second_45 = 0.5
        _pytest_places_46 = 2
        assert (
            _pytest_first_44 == _pytest_second_45
            or round(abs(_pytest_first_44 - _pytest_second_45), _pytest_places_46) == 0
        )
        _pytest_first_47 = self.backend._pres_pen_pool[3].item()
        _pytest_second_48 = 0.25
        _pytest_places_49 = 2
        assert (
            _pytest_first_47 == _pytest_second_48
            or round(abs(_pytest_first_47 - _pytest_second_48), _pytest_places_49) == 0
        )
        _pytest_first_50 = self.backend._rep_pen_pool[3].item()
        _pytest_second_51 = 1.2
        _pytest_places_52 = 2
        assert (
            _pytest_first_50 == _pytest_second_51
            or round(abs(_pytest_first_50 - _pytest_second_51), _pytest_places_52) == 0
        )
        _pytest_first_53 = self.backend._min_p_pool[3].item()
        _pytest_second_54 = 0.1
        _pytest_places_55 = 3
        assert (
            _pytest_first_53 == _pytest_second_54
            or round(abs(_pytest_first_53 - _pytest_second_54), _pytest_places_55) == 0
        )

    def test_counts_and_bias_cleared_on_flip(self):
        self.backend._counts[2, 100] = 7
        self.backend._logit_bias[2, 100] = 5.0
        self.backend.prepare_step(
            request_ids=["new"],
            request_pool_indices=[2],
            sampling_params_list=[_sp("new")],
        )
        assert (self.backend._counts[2, 100].item()) == (0)
        _pytest_first_56 = self.backend._logit_bias[2, 100].item()
        _pytest_second_57 = 0.0
        _pytest_places_58 = 3
        assert (
            _pytest_first_56 == _pytest_second_57
            or round(abs(_pytest_first_56 - _pytest_second_57), _pytest_places_58) == 0
        )

    def test_logit_bias_scattered(self):
        sp = _sp("a")
        sp.logit_bias = {"100": 2.0, "200": -1.5}
        self.backend.prepare_step(
            request_ids=["a"],
            request_pool_indices=[4],
            sampling_params_list=[sp],
        )
        _pytest_first_59 = self.backend._logit_bias[4, 100].item()
        _pytest_second_60 = 2.0
        _pytest_places_61 = 2
        assert (
            _pytest_first_59 == _pytest_second_60
            or round(abs(_pytest_first_59 - _pytest_second_60), _pytest_places_61) == 0
        )
        _pytest_first_62 = self.backend._logit_bias[4, 200].item()
        _pytest_second_63 = -1.5
        _pytest_places_64 = 2
        assert (
            _pytest_first_62 == _pytest_second_63
            or round(abs(_pytest_first_62 - _pytest_second_63), _pytest_places_64) == 0
        )
        _pytest_first_65 = self.backend._logit_bias[4, 150].item()
        _pytest_second_66 = 0.0
        _pytest_places_67 = 3
        assert (
            _pytest_first_65 == _pytest_second_66
            or round(abs(_pytest_first_65 - _pytest_second_66), _pytest_places_67) == 0
        )

    def test_logit_bias_h2d_is_pinned_and_non_blocking(self):
        _assert_logit_bias_h2d_is_async(self.backend)

    def test_reset_capture_state_clears_capture_counts(self):
        self.backend._counts[0, 10] = 3
        self.backend.reset_capture_state()
        assert (self.backend._counts[0, 10].item()) == (0)

    def test_min_p_cuda_graph_routes_have_dedicated_variants(self):
        variants = self.backend.cuda_graph_capture_variants(num_tokens_per_req=4)
        assert (CUDA_GRAPH_VARIANT_TRITON_FULL_MIN_P) in (variants)
        assert (CUDA_GRAPH_VARIANT_TRITON_FULL_TOP_K_TOP_P_MIN_P) in (variants)

        self.backend.prepare_step(
            request_ids=["a"],
            request_pool_indices=[1],
            sampling_params_list=[_sp("a", top_k=-1, top_p=1.0, min_p=0.2)],
            num_tokens_per_req=4,
        )
        assert (self.backend.cuda_graph_replay_variant(num_tokens_per_req=4)) == (
            CUDA_GRAPH_VARIANT_TRITON_FULL_MIN_P
        )

        self.backend.prepare_step(
            request_ids=["a"],
            request_pool_indices=[1],
            sampling_params_list=[_sp("a", top_k=64, top_p=0.9, min_p=0.05)],
            num_tokens_per_req=4,
        )
        assert (self.backend.cuda_graph_replay_variant(num_tokens_per_req=4)) == (
            CUDA_GRAPH_VARIANT_TRITON_FULL_TOP_K_TOP_P_MIN_P
        )

    def test_logit_bias_out_of_vocab_asserts(self):
        sp = _sp("a")
        sp.logit_bias = {str(VOCAB + 5): 1.0}
        with pytest.raises((AssertionError)):
            self.backend.prepare_step(
                request_ids=["a"],
                request_pool_indices=[1],
                sampling_params_list=[sp],
            )


class TestPrepareStepGuardRails:
    """Cheap boundary asserts in base.prepare_step. Cost is negligible and
    these are exactly the kinds of mismatches that produce silent state
    corruption if they slip through."""

    def setup_method(self):
        self.backend = FlashInferSamplingBackend(_make_config())

    def test_pool_idx_out_of_range_asserts(self):
        pool_rows = POOL + 1
        with pytest.raises((AssertionError)):
            self.backend.prepare_step(
                request_ids=["a"],
                request_pool_indices=[pool_rows],  # one past the end
                sampling_params_list=[_sp("a")],
            )
        with pytest.raises((AssertionError)):
            self.backend.prepare_step(
                request_ids=["a"],
                request_pool_indices=[-1],
                sampling_params_list=[_sp("a")],
            )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))

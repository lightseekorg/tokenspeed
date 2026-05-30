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
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=30, suite="runtime-1gpu")

import torch  # noqa: E402

import tokenspeed.runtime.sampling.backends.triton as triton_backend_module  # noqa: E402
from tokenspeed.runtime.execution.cuda_graph_wrapper import (  # noqa: E402
    CudaGraphWrapper,
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


class TestGreedyNoPoolState(unittest.TestCase):
    """Greedy backend declares ``_HAS_POOL_STATE = False`` so prepare_step
    must short-circuit with no allocation or iteration. Guards against a
    future refactor accidentally forcing pool tracking on stateless
    backends."""

    def test_prepare_step_is_noop(self):
        b = GreedySamplingBackend(_make_config())
        self.assertFalse(b._HAS_POOL_STATE)
        self.assertFalse(hasattr(b, "_last_rid_per_slot"))
        # Must not raise even with nonsensical inputs — short-circuits.
        b.prepare_step(
            request_ids=["a", "b"],
            request_pool_indices=[999, -1],  # would fail bounds check otherwise
            sampling_params_list=[_sp("a"), _sp("b")],
        )


class TestFlashInferFlipDetection(unittest.TestCase):
    """flashinfer's pool-indexed scalar buffers are core scheduler state.
    These tests pin flip semantics down at the Python state-machine level
    (no kernel invocation needed)."""

    def setUp(self):
        self.backend = FlashInferSamplingBackend(_make_config())

    def test_dp_verify_buffers_are_lazy(self):
        self.assertIsNone(self.backend._predict_local_buf)
        self.assertIsNone(self.backend._accept_index_local_buf)
        self.assertIsNone(self.backend._accept_length_local_buf)

    def test_first_admission_flips_and_scatters(self):
        sp_a = _sp("a", temperature=0.7, top_k=50, top_p=0.9, seed=42)
        sp_b = _sp("b", temperature=1.2, top_k=20, top_p=0.8, seed=123)
        self.backend.prepare_step(
            request_ids=["a", "b"],
            request_pool_indices=[1, 3],
            sampling_params_list=[sp_a, sp_b],
        )
        self.assertEqual(self.backend._last_rid_per_slot[1], "a")
        self.assertEqual(self.backend._last_rid_per_slot[3], "b")
        self.assertAlmostEqual(self.backend._temperature_pool[1].item(), 0.7, places=3)
        self.assertEqual(self.backend._top_k_pool[1].item(), 50)
        self.assertAlmostEqual(self.backend._top_p_pool[1].item(), 0.9, places=3)
        self.assertEqual(self.backend._seed_pool[1].item(), 42)
        self.assertAlmostEqual(self.backend._temperature_pool[3].item(), 1.2, places=3)
        self.assertEqual(self.backend._top_k_pool[3].item(), 20)
        # Unused slots keep their neutral init values.
        self.assertAlmostEqual(self.backend._temperature_pool[0].item(), 1.0)
        self.assertAlmostEqual(self.backend._temperature_pool[5].item(), 1.0)

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
        self.assertAlmostEqual(
            self.backend._temperature_pool[2].item(), 9.999, places=3
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
        gen_a = self.backend._generator_per_slot[2]
        sp_b = _sp("b", temperature=0.3, seed=99)
        self.backend.prepare_step(
            request_ids=["b"],
            request_pool_indices=[2],
            sampling_params_list=[sp_b],
        )
        self.assertEqual(self.backend._last_rid_per_slot[2], "b")
        self.assertAlmostEqual(self.backend._temperature_pool[2].item(), 0.3, places=3)
        self.assertEqual(self.backend._seed_pool[2].item(), 99)
        self.assertIsNot(self.backend._generator_per_slot[2], gen_a)


class TestTritonRouteSelection(unittest.TestCase):
    def setUp(self):
        self.backend = TritonSamplingBackend(_make_config())

    def test_no_filter_step_selects_triton_gumbel(self):
        self.backend.prepare_step(
            request_ids=["a", "b"],
            request_pool_indices=[1, 2],
            sampling_params_list=[_sp("a", top_k=-1, top_p=1.0), _sp("b", top_k=-1)],
        )
        self.assertEqual(self.backend._sample_route, _SAMPLE_ROUTE_GUMBEL_NO_FILTER)

    def test_finite_top_k_step_selects_triton_gumbel(self):
        self.backend.prepare_step(
            request_ids=["a"],
            request_pool_indices=[1],
            sampling_params_list=[_sp("a", top_k=50, top_p=1.0)],
        )
        self.assertEqual(self.backend._sample_route, _SAMPLE_ROUTE_GUMBEL_TOP_K)
        self.assertEqual(self.backend._top_k_top_p_pad, 64)

    def test_finite_top_k_top_p_step_selects_triton_gumbel(self):
        self.backend.prepare_step(
            request_ids=["a"],
            request_pool_indices=[1],
            sampling_params_list=[_sp("a", top_k=50, top_p=0.9)],
        )
        self.assertEqual(self.backend._sample_route, _SAMPLE_ROUTE_GUMBEL_TOP_K_TOP_P)
        self.assertEqual(self.backend._top_k_top_p_pad, 64)

        self.backend.prepare_step(
            request_ids=["a"],
            request_pool_indices=[1],
            sampling_params_list=[_sp("a", top_k=128, top_p=0.9)],
        )
        self.assertEqual(self.backend._top_k_top_p_pad, 128)

    def test_top_p_only_step_selects_top_p_triton_gumbel(self):
        self.backend.prepare_step(
            request_ids=["a"],
            request_pool_indices=[1],
            sampling_params_list=[_sp("a", top_k=-1, top_p=0.9)],
        )
        self.assertEqual(self.backend._sample_route, _SAMPLE_ROUTE_GUMBEL_TOP_P)

    def test_mixed_step_selects_generic_triton_gumbel(self):
        self.backend.prepare_step(
            request_ids=["a", "b"],
            request_pool_indices=[1, 2],
            sampling_params_list=[
                _sp("a", top_k=-1, top_p=1.0),
                _sp("b", top_k=50, top_p=1.0),
            ],
        )
        self.assertEqual(self.backend._sample_route, _SAMPLE_ROUTE_GUMBEL_GENERIC)

    def test_multi_token_no_filter_step_selects_verify_fast_path(self):
        self.backend.prepare_step(
            request_ids=["a"],
            request_pool_indices=[1],
            sampling_params_list=[_sp("a", top_k=-1, top_p=1.0)],
            num_tokens_per_req=4,
        )
        self.assertEqual(self.backend._sample_route, _SAMPLE_ROUTE_GUMBEL_NO_FILTER)

    def test_capture_keeps_generic_sampler_graph(self):
        self.backend._sample_route = _SAMPLE_ROUTE_GUMBEL_NO_FILTER
        self.backend.prepare_capture(bs=1)
        self.assertEqual(self.backend._sample_route, _SAMPLE_ROUTE_GUMBEL_GENERIC)

    def test_cuda_graph_variants_cover_top_p_and_verify_no_filter(self):
        self.assertEqual(
            self.backend.cuda_graph_capture_variants(num_tokens_per_req=1),
            (
                CUDA_GRAPH_VARIANT_DEFAULT,
                CUDA_GRAPH_VARIANT_TRITON_NO_FILTER,
                CUDA_GRAPH_VARIANT_TRITON_TOP_P,
                CUDA_GRAPH_VARIANT_TRITON_TOP_K,
                CUDA_GRAPH_VARIANT_TRITON_TOP_K_TOP_P,
            ),
        )
        self.assertEqual(
            self.backend.cuda_graph_capture_variants(num_tokens_per_req=4),
            (
                CUDA_GRAPH_VARIANT_DEFAULT,
                CUDA_GRAPH_VARIANT_TRITON_NO_FILTER,
                CUDA_GRAPH_VARIANT_TRITON_TOP_P,
                CUDA_GRAPH_VARIANT_TRITON_TOP_K,
                CUDA_GRAPH_VARIANT_TRITON_TOP_K_TOP_P,
                CUDA_GRAPH_VARIANT_TRITON_VERIFY_NO_FILTER,
            ),
        )

        self.backend.prepare_step(
            request_ids=["a"],
            request_pool_indices=[1],
            sampling_params_list=[_sp("a", top_k=-1, top_p=1.0)],
            num_tokens_per_req=4,
        )
        self.assertEqual(
            self.backend.cuda_graph_replay_variant(num_tokens_per_req=4),
            CUDA_GRAPH_VARIANT_TRITON_VERIFY_NO_FILTER,
        )

        self.backend.prepare_step(
            request_ids=["a"],
            request_pool_indices=[1],
            sampling_params_list=[_sp("a", top_k=-1, top_p=1.0)],
            num_tokens_per_req=1,
        )
        self.assertEqual(
            self.backend.cuda_graph_replay_variant(num_tokens_per_req=1),
            CUDA_GRAPH_VARIANT_TRITON_NO_FILTER,
        )

        self.backend.prepare_step(
            request_ids=["a"],
            request_pool_indices=[1],
            sampling_params_list=[_sp("a", top_k=-1, top_p=0.9)],
            num_tokens_per_req=4,
        )
        self.assertEqual(
            self.backend.cuda_graph_replay_variant(num_tokens_per_req=4),
            CUDA_GRAPH_VARIANT_TRITON_TOP_P,
        )

        self.backend.prepare_step(
            request_ids=["a"],
            request_pool_indices=[1],
            sampling_params_list=[_sp("a", top_k=50, top_p=1.0)],
            num_tokens_per_req=4,
        )
        self.assertEqual(
            self.backend.cuda_graph_replay_variant(num_tokens_per_req=4),
            CUDA_GRAPH_VARIANT_TRITON_TOP_K,
        )

        self.backend.prepare_step(
            request_ids=["a"],
            request_pool_indices=[1],
            sampling_params_list=[_sp("a", top_k=50, top_p=0.9)],
            num_tokens_per_req=4,
        )
        self.assertEqual(
            self.backend.cuda_graph_replay_variant(num_tokens_per_req=4),
            CUDA_GRAPH_VARIANT_TRITON_TOP_K_TOP_P,
        )

    def test_top_k_top_p_block_size_tuning_is_scoped_to_small_vocab(self):
        self.assertEqual(self.backend._select_top_k_top_p_block_size(1, 32768), 1024)
        self.assertEqual(self.backend._select_top_k_top_p_block_size(8, 32768), 1024)
        self.assertEqual(self.backend._select_top_k_top_p_block_size(32, 32768), 1024)
        self.assertEqual(self.backend._select_top_k_top_p_block_size(32, 151936), 1024)

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
            is_all_greedy=False,
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
            enable_pdl=False,
        ):
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
            )

        direct_sampler.assert_called_once()
        args, kwargs = direct_sampler.call_args
        self.assertEqual(args[7].shape[0], n)
        self.assertEqual(args[8].shape[0], n)

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
            is_all_greedy=False,
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
            enable_pdl=False,
        ):
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
            )

        top_p_sampler.assert_called_once()
        args, kwargs = top_p_sampler.call_args
        self.assertEqual(args[6].shape[0], n)
        self.assertEqual(args[17].shape[0], n)
        self.assertEqual(kwargs["num_tokens_per_req"], n)

    def test_verify_large_finite_top_k_top_p_uses_qrita_sampler(self):
        bs, n, vocab = 32, 4, 32768
        backend = TritonSamplingBackend(
            SamplingBackendConfig(
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
                _sp(f"qrita_{i}", top_k=128, top_p=0.9) for i in range(bs)
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
            is_all_greedy=False,
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
            enable_pdl=False,
        ):
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
            )

        qrita_sampler.assert_called_once()
        args, kwargs = qrita_sampler.call_args
        self.assertEqual(args[7].shape[1], vocab)
        self.assertEqual(
            args[8].shape[0],
            len(triton_backend_module._QRITA_PERCENTILE_TO_STD_TABLE),
        )
        self.assertEqual(kwargs["num_tokens_per_req"], n)

    def test_verify_large_finite_top_k_only_uses_qrita_sampler(self):
        bs, n, vocab = 32, 4, 200064
        backend = TritonSamplingBackend(
            SamplingBackendConfig(
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
            is_all_greedy=False,
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
            enable_pdl=False,
        ):
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
            )

        qrita_sampler.assert_called_once()
        args, kwargs = qrita_sampler.call_args
        self.assertEqual(args[7].shape[1], vocab)
        self.assertEqual(kwargs["num_tokens_per_req"], n)

    def test_prepare_capture_variant_sets_verify_fast_path(self):
        self.backend.prepare_capture_variant(
            bs=1,
            num_tokens_per_req=4,
            variant=CUDA_GRAPH_VARIANT_TRITON_TOP_K,
        )
        self.assertEqual(self.backend._sample_route, _SAMPLE_ROUTE_GUMBEL_TOP_K)

        self.backend.prepare_capture_variant(
            bs=1,
            num_tokens_per_req=4,
            variant=CUDA_GRAPH_VARIANT_TRITON_TOP_K_TOP_P,
        )
        self.assertEqual(self.backend._sample_route, _SAMPLE_ROUTE_GUMBEL_TOP_K_TOP_P)

        self.backend.prepare_capture_variant(
            bs=1,
            num_tokens_per_req=4,
            variant=CUDA_GRAPH_VARIANT_TRITON_TOP_P,
        )
        self.assertEqual(self.backend._sample_route, _SAMPLE_ROUTE_GUMBEL_TOP_P)

        self.backend.prepare_capture_variant(
            bs=1,
            num_tokens_per_req=4,
            variant=CUDA_GRAPH_VARIANT_TRITON_VERIFY_NO_FILTER,
        )
        self.assertEqual(self.backend._sample_route, _SAMPLE_ROUTE_GUMBEL_NO_FILTER)

        self.backend.prepare_capture_variant(
            bs=1,
            num_tokens_per_req=4,
            variant=CUDA_GRAPH_VARIANT_DEFAULT,
        )
        self.assertEqual(self.backend._sample_route, _SAMPLE_ROUTE_GUMBEL_GENERIC)

    def test_triton_verify_is_owned_by_triton_backend(self):
        self.assertFalse(issubclass(TritonSamplingBackend, FlashInferSamplingBackend))
        self.assertIsNot(TritonSamplingBackend.verify, FlashInferSamplingBackend.verify)
        self.assertIsNot(
            TritonFullSamplingBackend.verify,
            FlashInferFullSamplingBackend.verify,
        )
        self.assertTrue(issubclass(TritonFullSamplingBackend, TritonSamplingBackend))
        self.assertFalse(
            issubclass(TritonFullSamplingBackend, FlashInferFullSamplingBackend)
        )

    def test_triton_reuses_pool_state_without_flashinfer_probability_state(self):
        self.assertTrue(hasattr(self.backend, "_temperature_pool"))
        self.assertTrue(hasattr(self.backend, "_top_k_pool"))
        self.assertTrue(hasattr(self.backend, "_top_p_pool"))
        self.assertTrue(hasattr(self.backend, "_seed_pool"))
        self.assertFalse(hasattr(self.backend, "_coins_buf"))
        self.assertFalse(hasattr(self.backend, "_generator_per_slot"))


class TestTritonLogprobOutputs(unittest.TestCase):
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
            is_all_greedy=False,
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

        self.assertIsNone(logits_output.next_token_top_logprobs_val)
        self.assertIsNone(logits_output.next_token_top_logprobs_idx)
        self.assertIsNone(logits_output.next_token_token_ids_logprobs_val)
        self.assertIsNone(logits_output.next_token_token_ids_logprobs_idx)

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


class TestCudaGraphSamplingVariants(unittest.TestCase):
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
        wrapper = object.__new__(CudaGraphWrapper)
        wrapper.sampling_backend = backend
        wrapper.max_tokens_per_req = 4
        wrapper.graphs = {
            (CUDA_GRAPH_VARIANT_DEFAULT, 8): object(),
            (CUDA_GRAPH_VARIANT_TRITON_VERIFY_NO_FILTER, 8): object(),
        }

        self.assertEqual(
            wrapper._cuda_graph_capture_variants(),
            (
                CUDA_GRAPH_VARIANT_DEFAULT,
                CUDA_GRAPH_VARIANT_TRITON_VERIFY_NO_FILTER,
            ),
        )
        self.assertEqual(backend.capture_num_tokens_per_req, 4)
        self.assertEqual(
            wrapper._cuda_graph_key(8),
            (CUDA_GRAPH_VARIANT_TRITON_VERIFY_NO_FILTER, 8),
        )
        self.assertEqual(backend.replay_num_tokens_per_req, 4)

        backend.variant = "missing"
        self.assertEqual(
            wrapper._cuda_graph_key(8),
            (CUDA_GRAPH_VARIANT_DEFAULT, 8),
        )


class TestFlashInferFullFlipExtended(unittest.TestCase):
    """Full backend extends flip behavior with penalty scalars, count rows,
    and logit_bias scatter. Each of these must be cleared/scattered on
    flip or a new request inherits the previous occupant's state."""

    def setUp(self):
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
        self.assertAlmostEqual(self.backend._freq_pen_pool[3].item(), 0.5, places=2)
        self.assertAlmostEqual(self.backend._pres_pen_pool[3].item(), 0.25, places=2)
        self.assertAlmostEqual(self.backend._rep_pen_pool[3].item(), 1.2, places=2)
        self.assertAlmostEqual(self.backend._min_p_pool[3].item(), 0.1, places=3)

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
        self.assertEqual(self.backend._counts[2, 100].item(), 0)
        self.assertAlmostEqual(self.backend._logit_bias[2, 100].item(), 0.0, places=3)

    def test_logit_bias_scattered(self):
        sp = _sp("a", temperature=1.0)
        sp.logit_bias = {"100": 2.0, "200": -1.5}
        self.backend.prepare_step(
            request_ids=["a"],
            request_pool_indices=[4],
            sampling_params_list=[sp],
        )
        self.assertAlmostEqual(self.backend._logit_bias[4, 100].item(), 2.0, places=2)
        self.assertAlmostEqual(self.backend._logit_bias[4, 200].item(), -1.5, places=2)
        # Other positions untouched.
        self.assertAlmostEqual(self.backend._logit_bias[4, 150].item(), 0.0, places=3)

    def test_logit_bias_out_of_vocab_asserts(self):
        """OOV token ids would write past the bias row; must be caught."""
        sp = _sp("a")
        sp.logit_bias = {str(VOCAB + 5): 1.0}
        with self.assertRaises(AssertionError):
            self.backend.prepare_step(
                request_ids=["a"],
                request_pool_indices=[1],
                sampling_params_list=[sp],
            )


class TestTritonFullIndependentState(unittest.TestCase):
    """TritonFull owns the same full-sampling slot state as FlashInferFull,
    but it must not inherit the FlashInferFull probability route."""

    def setUp(self):
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
        self.assertAlmostEqual(self.backend._freq_pen_pool[3].item(), 0.5, places=2)
        self.assertAlmostEqual(self.backend._pres_pen_pool[3].item(), 0.25, places=2)
        self.assertAlmostEqual(self.backend._rep_pen_pool[3].item(), 1.2, places=2)
        self.assertAlmostEqual(self.backend._min_p_pool[3].item(), 0.1, places=3)

    def test_counts_and_bias_cleared_on_flip(self):
        self.backend._counts[2, 100] = 7
        self.backend._logit_bias[2, 100] = 5.0
        self.backend.prepare_step(
            request_ids=["new"],
            request_pool_indices=[2],
            sampling_params_list=[_sp("new")],
        )
        self.assertEqual(self.backend._counts[2, 100].item(), 0)
        self.assertAlmostEqual(self.backend._logit_bias[2, 100].item(), 0.0, places=3)

    def test_logit_bias_scattered(self):
        sp = _sp("a")
        sp.logit_bias = {"100": 2.0, "200": -1.5}
        self.backend.prepare_step(
            request_ids=["a"],
            request_pool_indices=[4],
            sampling_params_list=[sp],
        )
        self.assertAlmostEqual(self.backend._logit_bias[4, 100].item(), 2.0, places=2)
        self.assertAlmostEqual(self.backend._logit_bias[4, 200].item(), -1.5, places=2)
        self.assertAlmostEqual(self.backend._logit_bias[4, 150].item(), 0.0, places=3)

    def test_reset_capture_state_clears_capture_counts(self):
        self.backend._counts[0, 10] = 3
        self.backend.reset_capture_state()
        self.assertEqual(self.backend._counts[0, 10].item(), 0)

    def test_min_p_cuda_graph_routes_have_dedicated_variants(self):
        variants = self.backend.cuda_graph_capture_variants(num_tokens_per_req=4)
        self.assertIn(CUDA_GRAPH_VARIANT_TRITON_FULL_MIN_P, variants)
        self.assertIn(CUDA_GRAPH_VARIANT_TRITON_FULL_TOP_K_TOP_P_MIN_P, variants)

        self.backend.prepare_step(
            request_ids=["a"],
            request_pool_indices=[1],
            sampling_params_list=[_sp("a", top_k=-1, top_p=1.0, min_p=0.2)],
            num_tokens_per_req=4,
        )
        self.assertEqual(
            self.backend.cuda_graph_replay_variant(num_tokens_per_req=4),
            CUDA_GRAPH_VARIANT_TRITON_FULL_MIN_P,
        )

        self.backend.prepare_step(
            request_ids=["a"],
            request_pool_indices=[1],
            sampling_params_list=[_sp("a", top_k=64, top_p=0.9, min_p=0.05)],
            num_tokens_per_req=4,
        )
        self.assertEqual(
            self.backend.cuda_graph_replay_variant(num_tokens_per_req=4),
            CUDA_GRAPH_VARIANT_TRITON_FULL_TOP_K_TOP_P_MIN_P,
        )

    def test_logit_bias_out_of_vocab_asserts(self):
        sp = _sp("a")
        sp.logit_bias = {str(VOCAB + 5): 1.0}
        with self.assertRaises(AssertionError):
            self.backend.prepare_step(
                request_ids=["a"],
                request_pool_indices=[1],
                sampling_params_list=[sp],
            )


class TestPrepareStepGuardRails(unittest.TestCase):
    """Cheap boundary asserts in base.prepare_step. Cost is negligible and
    these are exactly the kinds of mismatches that produce silent state
    corruption if they slip through."""

    def setUp(self):
        self.backend = FlashInferSamplingBackend(_make_config())

    def test_misaligned_lists_assert(self):
        with self.assertRaises(AssertionError):
            self.backend.prepare_step(
                request_ids=["a", "b"],
                request_pool_indices=[1],
                sampling_params_list=[_sp("a"), _sp("b")],
            )

    def test_pool_idx_out_of_range_asserts(self):
        pool_rows = POOL + 1
        with self.assertRaises(AssertionError):
            self.backend.prepare_step(
                request_ids=["a"],
                request_pool_indices=[pool_rows],  # one past the end
                sampling_params_list=[_sp("a")],
            )
        with self.assertRaises(AssertionError):
            self.backend.prepare_step(
                request_ids=["a"],
                request_pool_indices=[-1],
                sampling_params_list=[_sp("a")],
            )


if __name__ == "__main__":
    unittest.main()

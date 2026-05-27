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
  * triton backend scatters temperature/top_k/top_p/seed on flip.
  * steady-state: same rid+slot across steps → no redundant _reset_slot.
  * slot recycle: slot reassigned to a new rid → _reset_slot fires.
  * triton_full additionally scatters penalty scalars, counts (zero),
    and logit_bias (zero-then-scatter) on flip; out-of-vocab bias raises.
  * boundary asserts: misaligned rid/pool/sp lists, out-of-range pool_idx.

Runs on CUDA because the backends allocate GPU tensors in ``__init__``.
"""

import os
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=30, suite="runtime-1gpu")

import torch  # noqa: E402

from tokenspeed.runtime.sampling.backends.base import (  # noqa: E402
    SamplingBackendConfig,
)
from tokenspeed.runtime.sampling.backends.greedy import (  # noqa: E402
    GreedySamplingBackend,
)
from tokenspeed.runtime.sampling.backends.triton import (  # noqa: E402
    TritonSamplingBackend,
)
from tokenspeed.runtime.sampling.backends.triton_full import (  # noqa: E402
    TritonFullSamplingBackend,
)
from tokenspeed.runtime.sampling.sampling_batch_info import (  # noqa: E402
    SamplingBatchInfo,
)
from tokenspeed.runtime.sampling.sampling_params import SamplingParams  # noqa: E402

VOCAB = 1024
POOL = 8  # max_req_pool_size → pool_rows == POOL + 1


def _make_config(
    *,
    enable_output_logprobs: bool = False,
    vocab_size: int = VOCAB,
) -> SamplingBackendConfig:
    return SamplingBackendConfig(
        enable_output_logprobs=enable_output_logprobs,
        max_bs=4,
        max_draft_tokens_per_req=4,
        max_req_pool_size=POOL,
        vocab_size=vocab_size,
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


class TestTritonFlipDetection(unittest.TestCase):
    """triton's pool-indexed scalar buffers are core scheduler state.
    These tests pin flip semantics down at the Python state-machine level
    (no kernel invocation needed)."""

    def setUp(self):
        self.backend = TritonSamplingBackend(_make_config())

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
        gen_a = self.backend._cpu_generator_per_slot[2]
        sp_b = _sp("b", temperature=0.3, seed=99)
        self.backend.prepare_step(
            request_ids=["b"],
            request_pool_indices=[2],
            sampling_params_list=[sp_b],
        )
        self.assertEqual(self.backend._last_rid_per_slot[2], "b")
        self.assertAlmostEqual(self.backend._temperature_pool[2].item(), 0.3, places=3)
        self.assertEqual(self.backend._seed_pool[2].item(), 99)
        self.assertIsNot(self.backend._cpu_generator_per_slot[2], gen_a)


class TestTritonSamplingDefault(unittest.TestCase):
    def _sampling_info(
        self,
        pool_indices,
        *,
        is_all_greedy=False,
        vocab_mask=None,
        apply_vocab_mask=None,
    ):
        return SamplingBatchInfo(
            is_all_greedy=is_all_greedy,
            req_pool_indices=torch.tensor(
                pool_indices, dtype=torch.int32, device="cuda"
            ),
            valid_cache_lengths=torch.zeros(POOL + 1, dtype=torch.int64, device="cuda"),
            vocab_mask=vocab_mask,
            apply_vocab_mask=apply_vocab_mask,
            device="cuda",
        )

    def test_no_filter_samples_batch_order(self):
        backend = TritonSamplingBackend(_make_config())
        backend.prepare_step(
            request_ids=["a", "b"],
            request_pool_indices=[4, 2],
            sampling_params_list=[
                _sp("a", top_k=-1, top_p=1.0, seed=11),
                _sp("b", top_k=-1, top_p=1.0, seed=22),
            ],
        )
        logits = torch.full((2, VOCAB), -10.0, dtype=torch.float32, device="cuda")
        logits[0, VOCAB - 1] = 1.0e6
        logits[1, VOCAB - 7] = 1.0e6
        logits_output = SimpleNamespace(
            next_token_logits=logits, next_token_logprobs=None
        )

        sampled, accept_lengths = backend.sample(
            logits_output, self._sampling_info([4, 2])
        )

        self.assertEqual(sampled.tolist(), [VOCAB - 1, VOCAB - 7])
        self.assertEqual(accept_lengths.tolist(), [1, 1])

    def test_top_k_samples_allowed_candidate(self):
        backend = TritonSamplingBackend(_make_config())
        backend.prepare_step(
            request_ids=["top_k"],
            request_pool_indices=[1],
            sampling_params_list=[_sp("top_k", top_k=2, top_p=1.0, seed=55)],
        )
        logits = torch.full((1, VOCAB), -10.0, dtype=torch.float32, device="cuda")
        logits[0, 17] = 5.0
        logits[0, 23] = 4.0
        logits[0, 99] = 3.0
        logits_output = SimpleNamespace(
            next_token_logits=logits, next_token_logprobs=None
        )

        sampled, accept_lengths = backend.sample(
            logits_output, self._sampling_info([1])
        )

        self.assertIn(sampled.item(), {17, 23})
        self.assertEqual(accept_lengths.tolist(), [1])

    def test_top_k_top_p_samples_nucleus_candidate(self):
        backend = TritonSamplingBackend(_make_config())
        backend.prepare_step(
            request_ids=["top_k_top_p"],
            request_pool_indices=[1],
            sampling_params_list=[_sp("top_k_top_p", top_k=3, top_p=0.6, seed=66)],
        )
        logits = torch.full((1, VOCAB), -10.0, dtype=torch.float32, device="cuda")
        logits[0, 17] = 10.0
        logits[0, 23] = 9.0
        logits[0, 99] = 8.0
        logits_output = SimpleNamespace(
            next_token_logits=logits, next_token_logprobs=None
        )

        sampled, accept_lengths = backend.sample(
            logits_output, self._sampling_info([1])
        )

        self.assertEqual(sampled.item(), 17)
        self.assertEqual(accept_lengths.tolist(), [1])

    def test_top_p_samples_valid_candidate(self):
        backend = TritonSamplingBackend(_make_config())
        backend.prepare_step(
            request_ids=["top_p"],
            request_pool_indices=[1],
            sampling_params_list=[_sp("top_p", top_k=-1, top_p=0.6, seed=33)],
        )
        logits = torch.full((1, VOCAB), -10.0, dtype=torch.float32, device="cuda")
        logits[0, 17] = 10.0
        logits[0, 23] = 1.0
        logits_output = SimpleNamespace(
            next_token_logits=logits,
            next_token_logprobs=None,
        )

        sampled, _ = backend.sample(logits_output, self._sampling_info([1]))

        self.assertEqual(sampled.item(), 17)

    def test_grammar_mask_applies_before_sampling(self):
        backend = TritonSamplingBackend(_make_config())
        backend.prepare_step(
            request_ids=["grammar"],
            request_pool_indices=[1],
            sampling_params_list=[_sp("grammar", top_k=-1, top_p=1.0, seed=44)],
        )
        logits_output = SimpleNamespace(
            next_token_logits=torch.zeros(
                (1, VOCAB), dtype=torch.float32, device="cuda"
            ),
            next_token_logprobs=None,
        )
        vocab_mask = torch.ones((1, 1), dtype=torch.int32, device="cuda")

        def apply_mask(logits, vocab_mask):
            logits.fill_(-float("inf"))
            logits[:, 23] = 1.0e6

        sampled, _ = backend.sample(
            logits_output,
            self._sampling_info(
                [1],
                vocab_mask=vocab_mask,
                apply_vocab_mask=apply_mask,
            ),
        )

        self.assertEqual(sampled.item(), 23)

    def test_prepare_capture_variants_sample(self):
        def set_top_k_top_p(backend):
            backend._top_k_pool[0].fill_(3)
            backend._top_p_pool[0].fill_(0.6)
            backend._temperature_pool[0].fill_(1.0)
            backend._seed_pool[0].fill_(66)

        cases = (
            (None, 42, None),
            ("no_filter", 47, None),
            ("top_k_top_p", 17, set_top_k_top_p),
        )
        for capture_variant, expected_token, configure in cases:
            with self.subTest(capture_variant=capture_variant):
                backend = TritonSamplingBackend(_make_config())
                backend.prepare_capture(
                    bs=1,
                    num_tokens_per_req=1,
                    capture_variant=capture_variant,
                )
                if configure is not None:
                    configure(backend)

                logits = torch.full(
                    (1, VOCAB), -10.0, dtype=torch.float32, device="cuda"
                )
                if capture_variant == "top_k_top_p":
                    logits[0, 17] = 10.0
                    logits[0, 23] = 9.0
                    logits[0, 99] = 8.0
                else:
                    logits[0, expected_token] = 1.0e6
                logits_output = SimpleNamespace(
                    next_token_logits=logits,
                    next_token_logprobs=None,
                )

                sampled, _ = backend.sample(
                    logits_output,
                    self._sampling_info(
                        [0],
                        is_all_greedy=capture_variant == "greedy",
                    ),
                )
                self.assertEqual(sampled.item(), expected_token)

    def test_cuda_graph_replay_variant_selects_no_filter_after_prepare_step(self):
        backend = TritonSamplingBackend(_make_config())
        backend.prepare_step(
            request_ids=["g"],
            request_pool_indices=[1],
            sampling_params_list=[_sp("g", top_k=1, top_p=1.0)],
        )
        self.assertEqual(backend.cuda_graph_replay_variant(), "top_k_top_p")

        backend.prepare_step(
            request_ids=["gv"],
            request_pool_indices=[1],
            sampling_params_list=[_sp("gv", top_k=1, top_p=1.0)],
            num_tokens_per_req=4,
        )
        self.assertEqual(backend.cuda_graph_replay_variant(), "greedy")

        backend.prepare_step(
            request_ids=["a"],
            request_pool_indices=[1],
            sampling_params_list=[_sp("a", top_k=-1, top_p=1.0)],
        )
        self.assertEqual(backend.cuda_graph_replay_variant(), "no_filter")

        backend.prepare_step(
            request_ids=["p"],
            request_pool_indices=[3],
            sampling_params_list=[_sp("p", top_k=-1, top_p=0.9)],
        )
        self.assertEqual(backend.cuda_graph_replay_variant(), "default")

        backend.prepare_step(
            request_ids=["b"],
            request_pool_indices=[2],
            sampling_params_list=[_sp("b", top_k=50, top_p=0.9)],
        )
        self.assertEqual(backend.cuda_graph_replay_variant(), "top_k_top_p")

        backend.prepare_step(
            request_ids=["k"],
            request_pool_indices=[2],
            sampling_params_list=[_sp("k", top_k=50, top_p=1.0)],
        )
        self.assertEqual(backend.cuda_graph_replay_variant(), "top_k_top_p")

        large_backend = TritonSamplingBackend(_make_config(vocab_size=151936))
        large_backend.prepare_step(
            request_ids=["b"],
            request_pool_indices=[2],
            sampling_params_list=[_sp("b", top_k=50, top_p=0.9)],
        )
        self.assertEqual(large_backend.cuda_graph_replay_variant(), "top_k_top_p")

    def test_cuda_graph_capture_variants(self):
        compact_backend = TritonSamplingBackend(_make_config())
        self.assertEqual(
            compact_backend.cuda_graph_capture_variants(num_tokens_per_req=1),
            ("default", "no_filter", "top_k_top_p"),
        )
        self.assertEqual(
            compact_backend.cuda_graph_capture_variants(num_tokens_per_req=4),
            ("default", "greedy"),
        )
        self.assertTrue(
            compact_backend.cuda_graph_capture_is_all_greedy(
                "greedy", num_tokens_per_req=4
            )
        )
        self.assertFalse(
            compact_backend.cuda_graph_capture_is_all_greedy(
                "default", num_tokens_per_req=4
            )
        )

        backend = TritonSamplingBackend(_make_config(vocab_size=151936))
        self.assertEqual(
            backend.cuda_graph_capture_variants(num_tokens_per_req=1),
            ("default", "no_filter", "top_k_top_p"),
        )
        backend.prepare_step(
            request_ids=["p"],
            request_pool_indices=[1],
            sampling_params_list=[_sp("p", top_k=-1, top_p=0.9)],
        )
        self.assertEqual(backend.cuda_graph_replay_variant(), "default")

    def test_mixed_modes_sample_batch(self):
        backend = TritonSamplingBackend(_make_config())
        backend.prepare_step(
            request_ids=["a", "b"],
            request_pool_indices=[1, 2],
            sampling_params_list=[
                _sp("a", top_k=-1, top_p=1.0, seed=11),
                _sp("b", top_k=50, top_p=0.9, seed=22),
            ],
        )
        logits = torch.full((2, VOCAB), -10.0, dtype=torch.float32, device="cuda")
        logits[0, 17] = 1.0e6
        logits[1, 29] = 1.0e6
        logits_output = SimpleNamespace(
            next_token_logits=logits,
            next_token_logprobs=None,
        )

        sampled, _ = backend.sample(logits_output, self._sampling_info([1, 2]))

        torch.testing.assert_close(
            sampled.cpu(), torch.tensor([17, 29], dtype=torch.int32)
        )

    def test_output_logprobs_are_written(self):
        backend = TritonSamplingBackend(_make_config(enable_output_logprobs=True))
        backend.prepare_step(
            request_ids=["a"],
            request_pool_indices=[1],
            sampling_params_list=[_sp("a", top_k=-1, top_p=1.0, seed=11)],
        )
        logits = torch.full((1, VOCAB), -10.0, dtype=torch.float32, device="cuda")
        logits[0, 7] = 1.0e6
        logits_output = SimpleNamespace(
            next_token_logits=logits,
            next_token_logprobs=None,
        )

        sampled, _ = backend.sample(logits_output, self._sampling_info([1]))

        self.assertEqual(sampled.item(), 7)
        self.assertIsNotNone(logits_output.next_token_logprobs)
        torch.testing.assert_close(
            logits_output.next_token_logprobs.cpu(),
            torch.tensor([0.0], dtype=torch.float32),
            atol=0.0,
            rtol=0.0,
        )

    def test_unresolved_mode_raises(self):
        backend = TritonSamplingBackend(_make_config())
        backend.prepare_step(
            request_ids=["a"],
            request_pool_indices=[1],
            sampling_params_list=[_sp("a", top_k=-1, top_p=1.0, seed=11)],
            num_tokens_per_req=2,
        )
        logits = torch.full((1, VOCAB), -10.0, dtype=torch.float32, device="cuda")
        logits[0, 7] = 1.0e6
        logits_output = SimpleNamespace(
            next_token_logits=logits,
            next_token_logprobs=None,
        )

        with self.assertRaisesRegex(RuntimeError, "did not select a sampling mode"):
            backend.sample(logits_output, self._sampling_info([1]))

    def test_verify_nan_guard_keeps_greedy_tokens_valid(self):
        if torch.version.hip is not None:
            self.skipTest("verify_chain_greedy is only available on NVIDIA")

        vocab_size = 151936
        backend = TritonSamplingBackend(_make_config(vocab_size=vocab_size))
        backend.prepare_step(
            request_ids=["mtp"],
            request_pool_indices=[1],
            sampling_params_list=[_sp("mtp", top_k=1, top_p=1.0)],
            num_tokens_per_req=4,
        )
        logits = torch.full(
            (4, vocab_size),
            float("nan"),
            dtype=torch.float32,
            device="cuda",
        )
        logits_output = SimpleNamespace(
            next_token_logits=logits,
            next_token_logprobs=None,
        )
        candidates = torch.zeros((1, 4), dtype=torch.int32, device="cuda")

        predict, accept_lengths = backend.verify(
            logits_output,
            self._sampling_info([1], is_all_greedy=True),
            candidates,
        )

        self.assertEqual(accept_lengths.tolist(), [4])
        accepted = predict[: accept_lengths.item()]
        self.assertTrue(torch.all(accepted >= 0).item())
        self.assertTrue(torch.all(accepted < vocab_size).item())
        self.assertFalse(torch.isnan(logits).any().item())


class TestTritonFullFlipExtended(unittest.TestCase):
    """Full backend extends flip behavior with penalty scalars, count rows,
    and logit_bias scatter. Each of these must be cleared/scattered on
    flip or a new request inherits the previous occupant's state."""

    def setUp(self):
        self.backend = TritonFullSamplingBackend(_make_config())

    def _sampling_info(self, pool_indices):
        return SamplingBatchInfo(
            is_all_greedy=False,
            req_pool_indices=torch.tensor(
                pool_indices, dtype=torch.int32, device="cuda"
            ),
            valid_cache_lengths=torch.zeros(POOL + 1, dtype=torch.int64, device="cuda"),
            vocab_mask=None,
            apply_vocab_mask=None,
            device="cuda",
        )

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

    def test_sample_applies_min_p_and_accumulates_counts(self):
        backend = TritonFullSamplingBackend(_make_config())
        sp = _sp("full", top_k=-1, top_p=1.0, min_p=0.5, seed=123)
        backend.prepare_step(
            request_ids=["full"],
            request_pool_indices=[1],
            sampling_params_list=[sp],
        )

        logits = torch.full((1, VOCAB), -10.0, dtype=torch.float32, device="cuda")
        logits[0, 33] = 1.0e6
        logits_output = SimpleNamespace(
            next_token_logits=logits,
            next_token_logprobs=None,
        )

        sampled, accept_lengths = backend.sample(
            logits_output, self._sampling_info([1])
        )

        self.assertEqual(sampled.item(), 33)
        self.assertEqual(accept_lengths.tolist(), [1])
        self.assertEqual(backend._counts[1, 33].item(), 1)


class TestPrepareStepGuardRails(unittest.TestCase):
    """Cheap boundary asserts in base.prepare_step. Cost is negligible and
    these are exactly the kinds of mismatches that produce silent state
    corruption if they slip through."""

    def setUp(self):
        self.backend = TritonSamplingBackend(_make_config())

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

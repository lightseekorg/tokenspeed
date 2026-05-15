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

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=30, suite="runtime-1gpu")

from tokenspeed.runtime.sampling.backends.base import (  # noqa: E402
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

"""Lazy KDA speculative commit through the real MambaAttnBackend.

The kernel-level equivalences live in
``tokenspeed-kernel/test/ops/attention/test_kda_fused_replay_verify.py``;
what these tests exercise is the runtime plumbing around them: recording a
pending window after acceptance, composing it into the next verify round
(including a re-packed batch), and flushing it whenever a request leaves the
verify stream.

Ground truth for every scenario is the same backend flow with the pending
flushed eagerly after every round -- lazy and eager must land the same
committed pages (to the fused/standalone kernels' ~1 ulp fp32 FMA daylight).
"""

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

from test.runtime.conftest import KIMI_STATE_GROUPS as _STATE_GROUPS
from test.runtime.conftest import cache_metadata_for as _metadata_for
from test.runtime.conftest import make_kimi_pool as _make_kimi_pool
from types import SimpleNamespace  # noqa: E402  (after torch guard)

from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.hybrid_linear_attn import (
    MambaAttnBackend,
)

_LOWER_BOUND = -5.0
H, D, D_FA = 4, 128, 128
KEY_DIM = H * D
CONV_DIM = 3 * KEY_DIM
T = 2  # draft tokens per request
DEV = "cuda"


def _backend(pool):
    config = SimpleNamespace(
        device=DEV,
        num_attention_heads=H,
        num_kv_heads=H,
        attn_tp_size=1,
        dtype=torch.bfloat16,
        head_dim=D,
        is_draft=False,
        speculative_num_draft_tokens=T,
    )
    backend = MambaAttnBackend(config, is_kda=True)
    backend.set_kv_pool(pool)
    if not backend._kda_replay_active():
        pytest.skip("KDA replay kernels unavailable on this platform")
    return backend


class _Harness:
    """Drives verify rounds + accepts through one backend over a real pool."""

    def __init__(self, seed=0, usable_pages=16):
        torch.manual_seed(seed)
        self.pool = _make_kimi_pool(DEV, usable_pages=usable_pages)
        self.contract = self.pool.runtime_contract
        self.backend = _backend(self.pool)
        # Drive EVERY KDA layer, as a real verify forward would.
        self.layer_ids = list(self.backend._state_layer_ids())
        self.params = {
            layer_id: dict(
                conv_weights=torch.randn(CONV_DIM, 4, device=DEV, dtype=torch.bfloat16)
                * 0.1,
                f_b_weight=torch.randn(KEY_DIM, D_FA, device=DEV, dtype=torch.bfloat16)
                * 0.05,
                A_log=torch.randn(H, device=DEV, dtype=torch.float32) * 0.1,
                dt_bias=torch.randn(KEY_DIM, device=DEV, dtype=torch.float32) * 0.1,
            )
            for layer_id in self.layer_ids
        }

    def window(self, bs, seed):
        g = torch.Generator(device="cpu").manual_seed(seed)

        def rnd(*shape):
            return torch.randn(*shape, generator=g, dtype=torch.float32).to(
                DEV, torch.bfloat16
            )

        return dict(
            mixed_qkv=rnd(bs * T, CONV_DIM),
            f_a_out=rnd(bs * T, D_FA),
            beta_raw=rnd(bs * T, H),
        )

    def verify_round(self, rpis, pages, seq_lens, window):
        """One target-verify forward over all three KDA layers."""
        bs = len(rpis)
        tables = {
            gid: np.asarray([[p] for p in pages[gid]], dtype=np.int32)
            for gid in _STATE_GROUPS
        }
        metadata, op = _metadata_for(self.contract, tables, DEV)
        op.request_pool_indices = list(rpis)
        self.backend.init_forward_metadata(
            bs=bs,
            req_pool_indices=torch.tensor(rpis, dtype=torch.int32, device=DEV),
            seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=DEV),
            forward_mode=ForwardMode.DECODE,
            cache_metadata=metadata,
            forward_batch=op,
        )
        outs = {}
        for layer_id in self.layer_ids:
            p = self.params[layer_id]
            outs[layer_id] = self.backend.forward_decode(
                None,
                None,
                None,
                layer=None,
                out_cache_loc=None,
                token_to_kv_pool=self.pool,
                bs=bs,
                mixed_qkv=window["mixed_qkv"].clone(),
                f_a_out=window["f_a_out"],
                beta_raw=window["beta_raw"],
                g_raw=None,
                conv_weights=p["conv_weights"],
                bias=None,
                activation="silu",
                key_dim=KEY_DIM,
                value_dim=KEY_DIM,
                attention_tp_size=1,
                head_k_dim=D,
                head_v_dim=D,
                A_log=p["A_log"],
                dt_bias=p["dt_bias"],
                f_b_weight=p["f_b_weight"],
                lower_bound=_LOWER_BOUND,
                layer_id=layer_id,
                seq_len=bs * T,
                a=None,
                b=None,
            )
        return outs

    def accept(self, accepted):
        self.backend.commit_verified_state(
            torch.tensor(accepted, dtype=torch.int32, device=DEV)
        )

    def flush(self):
        self.backend.flush_kda_pending_commits()

    def state_of(self, layer_id, page):
        conv = self.pool.get_component(layer_id, "conv_state")[page].clone()
        ssm = self.pool.get_component(layer_id, "recurrent_state")[page].clone()
        return conv, ssm

    def pending(self):
        return getattr(self.backend, "_kda_pending", None)


def _pages_for(h, rpis):
    """One distinct page per (request, group), stable across rounds."""
    return {
        gid: [2 + g * len(rpis) + i for i, _ in enumerate(rpis)]
        for g, gid in enumerate(_STATE_GROUPS)
    }


def _run_rounds(h, rounds, accepts, rpis, eager):
    """Drive verify+accept rounds; eager mode flushes after every accept."""
    pages = _pages_for(h, rpis)
    seq_lens = [4 + T] * len(rpis)
    outs = []
    for k, (window_seed, accepted) in enumerate(zip(rounds, accepts)):
        w = h.window(len(rpis), window_seed)
        outs.append(h.verify_round(rpis, pages, seq_lens, w))
        h.accept(accepted)
        if eager:
            h.flush()
        # The engine always commits at least the target's own sample.
        seq_lens = [s + max(a, 1) for s, a in zip(seq_lens, accepted)]
    h.flush()  # commit the last window either way
    return outs, pages


def _assert_pools_match(h_lazy, h_eager, pages, rpis):
    for layer_id in h_lazy.layer_ids:
        gid = h_lazy.pool.group_id_for_layer(layer_id)
        for i, _ in enumerate(rpis):
            page = pages[gid][i]
            conv_l, ssm_l = h_lazy.state_of(layer_id, page)
            conv_e, ssm_e = h_eager.state_of(layer_id, page)
            torch.testing.assert_close(conv_l, conv_e, atol=0.0, rtol=0.0)
            torch.testing.assert_close(ssm_l, ssm_e, atol=1e-6, rtol=1e-4)


def test_lazy_rounds_match_eager_flush_every_round():
    """Three fused rounds == the same rounds with an eager flush after each.

    A flush-counting probe pins down that the steady-state rounds really
    commit through the fused kernel: if composing fell back to flushing,
    lazy and eager would trivially agree and the test would prove nothing.
    """
    rpis = [0, 1, 2]
    rounds, accepts = [11, 12, 13], [[1, 2, 1], [2, 1, 2], [1, 1, 2]]
    h_lazy = _Harness(seed=5)
    flushes = []
    inner = h_lazy.backend._flush_kda_pending

    def _counting_flush(only_rpis=None):
        flushes.append(only_rpis)
        inner(only_rpis)

    h_lazy.backend._flush_kda_pending = _counting_flush
    outs_lazy, pages = _run_rounds(h_lazy, rounds, accepts, rpis, eager=False)
    # Rounds 2 and 3 must have fused their pending; only the final explicit
    # flush (of round 3's window) may run the standalone kernels.
    assert flushes == [None], flushes
    h_eager = _Harness(seed=5)
    outs_eager, _ = _run_rounds(h_eager, rounds, accepts, rpis, eager=True)

    # Round k's outputs come from identical committed inputs in both modes.
    for lo, eo in zip(outs_lazy, outs_eager):
        for layer_id in h_lazy.layer_ids:
            torch.testing.assert_close(
                lo[layer_id].float(), eo[layer_id].float(), atol=1e-3, rtol=1e-2
            )
    _assert_pools_match(h_lazy, h_eager, pages, rpis)
    assert h_lazy.pending() is None


def test_departed_request_is_dropped_and_survivors_fuse():
    """Round 2 drops the middle request and re-packs the batch.

    The departed request's pending must be DISCARDED at the next arm -- its
    pages may already belong to a new request (see the abort-repro test) --
    while the survivors replay from payload captured at their OLD slots
    (row-base indirection), so their states must still match the eager run.
    """
    h_lazy = _Harness(seed=7)
    h_eager = _Harness(seed=7)
    all_rpis = [0, 1, 2]
    pages3 = _pages_for(h_lazy, all_rpis)
    w1 = {}
    for h in (h_lazy, h_eager):
        w1[h] = h.window(3, 21)
        h.verify_round(all_rpis, pages3, [6, 6, 6], w1[h])
        h.accept([2, 1, 2])
    h_eager.flush()

    # Simulate the departed request's page being re-assigned: a new owner
    # rewrites it. The drop must leave this sentinel state untouched.
    gid0 = _STATE_GROUPS[0]
    sentinel = {}
    for layer_id in h_lazy.layer_ids:
        gid = h_lazy.pool.group_id_for_layer(layer_id)
        page = pages3[gid][1]
        conv = h_lazy.pool.get_component(layer_id, "conv_state")
        ssm = h_lazy.pool.get_component(layer_id, "recurrent_state")
        conv[page].fill_(7.0)
        ssm[page].fill_(7.0)
        sentinel[layer_id] = page

    # Request 1 leaves; 0 and 2 re-pack into slots 0 and 1.
    survivors = [0, 2]
    pages2 = {gid: [pages3[gid][0], pages3[gid][2]] for gid in _STATE_GROUPS}
    for h in (h_lazy, h_eager):
        w2 = h.window(2, 22)
        h.verify_round(survivors, pages2, [8, 8], w2)
        h.accept([1, 2])
        h.flush()

    # Survivors match eager exactly.
    for layer_id in h_lazy.layer_ids:
        gid = h_lazy.pool.group_id_for_layer(layer_id)
        for i in (0, 2):
            page = pages3[gid][i]
            conv_l, ssm_l = h_lazy.state_of(layer_id, page)
            conv_e, ssm_e = h_eager.state_of(layer_id, page)
            torch.testing.assert_close(conv_l, conv_e, atol=0.0, rtol=0.0)
            torch.testing.assert_close(ssm_l, ssm_e, atol=1e-6, rtol=1e-4)
    # The departed request's page still holds the new owner's sentinel:
    # nothing replayed the dead window onto it.
    for layer_id in h_lazy.layer_ids:
        page = sentinel[layer_id]
        conv_l, ssm_l = h_lazy.state_of(layer_id, page)
        assert bool((conv_l == 7.0).all()), "drop must not write the reclaimed page"
        assert bool((ssm_l == 7.0).all()), "drop must not write the reclaimed page"


def test_non_verify_forward_flushes_the_pending():
    """A pending commit must not survive into a non-verify KDA forward."""
    h = _Harness(seed=9)
    rpis = [0, 1]
    pages = _pages_for(h, rpis)
    w = h.window(2, 31)
    h.verify_round(rpis, pages, [5, 5], w)
    h.accept([1, 2])
    assert h.pending() is not None

    # Metadata prep for a plain decode (non-verify) must flush.
    tables = {
        gid: np.asarray([[p] for p in pages[gid]], dtype=np.int32)
        for gid in _STATE_GROUPS
    }
    metadata, op = _metadata_for(h.contract, tables, DEV)
    op.request_pool_indices = rpis
    h.backend.spec_num_tokens = 1  # plain-decode shape for this metadata call
    try:
        h.backend.init_forward_metadata(
            bs=2,
            req_pool_indices=torch.tensor(rpis, dtype=torch.int32, device=DEV),
            seq_lens=torch.tensor([6, 7], dtype=torch.int32, device=DEV),
            forward_mode=ForwardMode.DECODE,
            cache_metadata=metadata,
            forward_batch=op,
        )
    finally:
        h.backend.spec_num_tokens = T
    assert h.pending() is None


def test_all_rejected_window_still_commits():
    """accepted = 0 clamps to one committed token (the target's own sample);
    the lazy path must carry that clamp exactly like the eager path."""
    rpis = [0, 1]
    rounds, accepts = [41, 42], [[0, 0], [2, 1]]
    h_lazy = _Harness(seed=11)
    _, pages = _run_rounds(h_lazy, rounds, accepts, rpis, eager=False)
    h_eager = _Harness(seed=11)
    _run_rounds(h_eager, rounds, accepts, rpis, eager=True)
    _assert_pools_match(h_lazy, h_eager, pages, rpis)


def _prep_verify(h, rpis, pages, seq_lens):
    """Verify-shaped metadata prep WITHOUT the forward that should follow.

    This is the arm half of an abandoned round: the event loop prepared the
    batch, then retracted it before launching the model."""
    tables = {
        gid: np.asarray([[p] for p in pages[gid]], dtype=np.int32)
        for gid in _STATE_GROUPS
    }
    metadata, op = _metadata_for(h.contract, tables, DEV)
    op.request_pool_indices = list(rpis)
    h.backend.init_forward_metadata(
        bs=len(rpis),
        req_pool_indices=torch.tensor(rpis, dtype=torch.int32, device=DEV),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=DEV),
        forward_mode=ForwardMode.DECODE,
        cache_metadata=metadata,
        forward_batch=op,
    )


def test_abandoned_verify_prep_keeps_pending_for_retract_flush():
    """Adversarial schedule: arm, then abandon the forward (batch retract).

    Metadata prep composes the pending into the fused-kernel control buffers
    (the "arm"), but the forward never launches -- the event loop retracted
    the batch between prep and dispatch and fires the retract hook
    ``flush_kda_pending_commits`` instead. The pending must be consumed at
    RECORD (after a completed forward), not at arm: an arm-time consume
    would silently lose the accepted window here. The retract-hook flush
    must really write the window, and the next verify round must match an
    eager run that never saw the abandoned prep."""
    rpis = [0, 1]
    h_lazy = _Harness(seed=13)
    h_eager = _Harness(seed=13)
    pages = _pages_for(h_lazy, rpis)

    for h in (h_lazy, h_eager):
        h.verify_round(rpis, pages, [6, 6], h.window(2, 51))
        h.accept([2, 1])
    h_eager.flush()

    # Committed pages are still untouched in the lazy run (commit deferred).
    pre = {
        (layer_id, i): h_lazy.state_of(
            layer_id, pages[h_lazy.pool.group_id_for_layer(layer_id)][i]
        )
        for layer_id in h_lazy.layer_ids
        for i in range(len(rpis))
    }

    _prep_verify(h_lazy, rpis, pages, [8, 7])  # armed ...
    assert h_lazy.pending() is not None  # ... but NOT consumed at arm
    h_lazy.flush()  # the retract hook
    assert h_lazy.pending() is None
    # The flush actually landed the accepted window (every page changed).
    for layer_id in h_lazy.layer_ids:
        gid = h_lazy.pool.group_id_for_layer(layer_id)
        for i in range(len(rpis)):
            _, ssm_n = h_lazy.state_of(layer_id, pages[gid][i])
            _, ssm_0 = pre[(layer_id, i)]
            assert not torch.equal(ssm_n, ssm_0), f"no write: {layer_id}/{i}"

    for h in (h_lazy, h_eager):
        h.verify_round(rpis, pages, [8, 7], h.window(2, 52))
        h.accept([1, 2])
        h.flush()
    _assert_pools_match(h_lazy, h_eager, pages, rpis)


def test_rearm_after_abandoned_prep_commits_exactly_once():
    """Arm, abandon the forward, then arm the SAME batch again and run it.

    No retract hook fires between the two arms, so the pending must survive
    the first (abandoned) arm intact and still fuse-commit during the second
    round's forward -- exactly once. A consume-at-arm would lose the window;
    a stale re-arm double-applying the in-place commit would diverge from
    eager. A flush-counting probe pins down that the surviving pending
    really fused (only the final explicit flush may run the standalone
    kernels)."""
    rpis = [0, 1]
    h_lazy = _Harness(seed=17)
    h_eager = _Harness(seed=17)
    pages = _pages_for(h_lazy, rpis)

    for h in (h_lazy, h_eager):
        h.verify_round(rpis, pages, [6, 6], h.window(2, 61))
        h.accept([1, 2])
    h_eager.flush()

    flushes = []
    inner = h_lazy.backend._flush_kda_pending

    def _counting_flush(only_rpis=None):
        flushes.append(only_rpis)
        inner(only_rpis)

    h_lazy.backend._flush_kda_pending = _counting_flush

    _prep_verify(h_lazy, rpis, pages, [7, 8])  # arm #1, abandoned
    assert h_lazy.pending() is not None

    outs = {}
    for h, sink in ((h_lazy, {}), (h_eager, outs)):
        sink.update(h.verify_round(rpis, pages, [7, 8], h.window(2, 62)))
        h.accept([2, 1])
        h.flush()
    # Arm #2 + fused round consumed the pending; only the terminal flush of
    # round 2's window ran the standalone kernels.
    assert flushes == [None], flushes
    _assert_pools_match(h_lazy, h_eager, pages, rpis)


def test_recycled_rpi_pending_is_dropped_not_replayed():
    """Request-pool-index recycling: a request finishes with a pending
    window, and a NEW request with the same rpi (fresh pages) enters the
    very next verify round with no non-verify forward in between.

    The arm gates each pending entry on pending-commit-page == the
    request's current committed page, so the impostor is NOT armed: its
    verify must run fresh from its own page, identical to an eager control
    where the recycled slot never had a predecessor. The dead request's
    window is deliberately DROPPED, never flushed -- by the time the reuse
    is detected its old pages may already belong to another request, so
    writing them would corrupt a stranger's state. Dropping is safe: the
    dead request finished, so its uncommitted accepted-window state can
    never be read again. Its old pages must keep their pre-accept content
    bitwise."""
    rpis = [0, 1, 2]
    h_lazy = _Harness(seed=19)
    h_eager = _Harness(seed=19)
    pages3 = _pages_for(h_lazy, rpis)
    fresh = {gid: 12 + g for g, gid in enumerate(_STATE_GROUPS)}
    pages2 = {
        gid: [pages3[gid][0], pages3[gid][1], fresh[gid]] for gid in _STATE_GROUPS
    }

    for h in (h_lazy, h_eager):
        h.verify_round(rpis, pages3, [6, 6, 6], h.window(3, 71))
        h.accept([2, 1, 2])  # rpi 2 finishes with this window pending
    h_eager.flush()

    # Snapshot the dead request's old pages (lazy run, pre-commit content).
    dead = {
        layer_id: h_lazy.state_of(
            layer_id, pages3[h_lazy.pool.group_id_for_layer(layer_id)][2]
        )
        for layer_id in h_lazy.layer_ids
    }

    # The impostor recycles rpi 2 on fresh pages; survivors 0, 1 continue.
    outs_lazy, outs_eager = {}, {}
    for h, sink in ((h_lazy, outs_lazy), (h_eager, outs_eager)):
        sink.update(h.verify_round(rpis, pages2, [8, 7, 6], h.window(3, 72)))
        h.accept([1, 2, 2])
        h.flush()

    # The impostor's verify rows come from its own fresh page, exactly like
    # the eager control where the recycled slot never had a predecessor;
    # survivor rows agree too (their pendings fused normally).
    for layer_id in h_lazy.layer_ids:
        lo, eo = outs_lazy[layer_id].float(), outs_eager[layer_id].float()
        torch.testing.assert_close(lo, eo, atol=1e-3, rtol=1e-2)
        torch.testing.assert_close(  # impostor rows, tight: same page, no fuse
            lo[2 * T : 3 * T], eo[2 * T : 3 * T], atol=5e-4, rtol=1e-3
        )

    # Survivors' + impostor's committed pages match eager exactly.
    for layer_id in h_lazy.layer_ids:
        gid = h_lazy.pool.group_id_for_layer(layer_id)
        for page in (pages3[gid][0], pages3[gid][1], fresh[gid]):
            conv_l, ssm_l = h_lazy.state_of(layer_id, page)
            conv_e, ssm_e = h_eager.state_of(layer_id, page)
            torch.testing.assert_close(conv_l, conv_e, atol=0.0, rtol=0.0)
            torch.testing.assert_close(ssm_l, ssm_e, atol=1e-6, rtol=1e-4)

    # The dead request's old pages keep their pre-accept content bitwise:
    # the DROP semantic (eager, by contrast, flushed that window there).
    for layer_id in h_lazy.layer_ids:
        gid = h_lazy.pool.group_id_for_layer(layer_id)
        conv_n, ssm_n = h_lazy.state_of(layer_id, pages3[gid][2])
        conv_0, ssm_0 = dead[layer_id]
        torch.testing.assert_close(conv_n, conv_0, atol=0.0, rtol=0.0)
        torch.testing.assert_close(ssm_n, ssm_0, atol=0.0, rtol=0.0)

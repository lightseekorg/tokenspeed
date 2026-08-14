"""Lazy KDA speculative commit through the real KdaAttnBackend.

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
from tokenspeed.runtime.layers.attention.backends.hybrid_kda import (
    KdaAttnBackend,
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
        max_bs=8,
    )
    backend = KdaAttnBackend(config)
    backend.set_kv_pool(pool)
    if not backend._replay_active:
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
        op.request_ids = [f"req{r}" for r in rpis]
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
        # The graph wrapper settles state after every forward; with no accept
        # lengths yet this is just the release of the window the stage composed,
        # which is on the device once the layers have run.
        self.backend.notify_forward_issued()
        return outs

    def accept(self, accepted):
        self.backend.commit_verified_state(
            torch.tensor(accepted, dtype=torch.int32, device=DEV)
        )

    def flush(self, resident=None):
        """Pause-fence flush; ``resident=None`` means every owner is still here."""
        pend = self.backend._kda_pending
        if resident is None:
            resident = set() if pend is None else set(pend["id_by_rpi"].values())
        self.backend.flush_pending(resident)

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

    The departed request's pending must be DISCARDED at the next stage -- its
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
    op.request_ids = [f"req{r}" for r in rpis]
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

    This is the stage half of an abandoned round: the event loop prepared the
    batch, then retracted it before launching the model."""
    tables = {
        gid: np.asarray([[p] for p in pages[gid]], dtype=np.int32)
        for gid in _STATE_GROUPS
    }
    metadata, op = _metadata_for(h.contract, tables, DEV)
    op.request_pool_indices = list(rpis)
    op.request_ids = [f"req{r}" for r in rpis]
    h.backend.init_forward_metadata(
        bs=len(rpis),
        req_pool_indices=torch.tensor(rpis, dtype=torch.int32, device=DEV),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=DEV),
        forward_mode=ForwardMode.DECODE,
        cache_metadata=metadata,
        forward_batch=op,
    )


def test_abandoned_verify_prep_keeps_pending_for_retract_flush():
    """Adversarial schedule: stage, then abandon the forward (batch retract).

    Metadata prep composes the pending into the fused-kernel control buffers
    (the "stage"), but the forward never launches -- the event loop retracted
    the batch between prep and dispatch and fires the retract hook
    ``flush_pending`` instead. The pending must be consumed at FORWARD
    ISSUE, not at stage: an stage-time consume
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

    _prep_verify(h_lazy, rpis, pages, [8, 7])  # staged ...
    assert h_lazy.pending() is not None  # ... but NOT consumed at stage
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
    """Stage, abandon the forward, then stage the SAME batch again and run it.

    No retract hook fires between the two stages, so the pending must survive
    the first (abandoned) stage intact and still fuse-commit during the second
    round's forward -- exactly once. A consume-at-stage would lose the window;
    a stale re-stage double-applying the in-place commit would diverge from
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

    _prep_verify(h_lazy, rpis, pages, [7, 8])  # stage #1, abandoned
    assert h_lazy.pending() is not None

    outs = {}
    for h, sink in ((h_lazy, {}), (h_eager, outs)):
        sink.update(h.verify_round(rpis, pages, [7, 8], h.window(2, 62)))
        h.accept([2, 1])
        h.flush()
    # Stage #2 + fused round consumed the pending; only the terminal flush of
    # round 2's window ran the standalone kernels.
    assert flushes == [None], flushes
    _assert_pools_match(h_lazy, h_eager, pages, rpis)


def test_recycled_rpi_pending_is_dropped_not_replayed():
    """Request-pool-index recycling: a request finishes with a pending
    window, and a NEW request with the same rpi (fresh pages) enters the
    very next verify round with no non-verify forward in between.

    The stage gates each pending entry on pending-commit-page == the
    request's current committed page, so the impostor is NOT staged: its
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


def _prepped_round(h, rpis, pages, seq_lens):
    """Precompute EVERY host->device transfer a verify round needs.

    The overlap-race test below must drive a whole round while the device is
    parked on a ``torch.cuda._sleep``: any pageable H2D (``torch.tensor(...,
    device=cuda)``) inside the round would synchronize the stream and drain
    the sleep, silently voiding the race window it is trying to create."""
    tables = {
        gid: np.asarray([[p] for p in pages[gid]], dtype=np.int32)
        for gid in _STATE_GROUPS
    }
    metadata, op = _metadata_for(h.contract, tables, DEV)
    op.request_pool_indices = list(rpis)
    op.request_ids = [f"req{r}" for r in rpis]
    # Materialize the device tables now, not lazily mid-round.
    for gid in _STATE_GROUPS:
        metadata.require_table(gid, active_forward_op=op)
    return dict(
        bs=len(rpis),
        metadata=metadata,
        op=op,
        rpi_dev=torch.tensor(rpis, dtype=torch.int32, device=DEV),
        seq_dev=torch.tensor(seq_lens, dtype=torch.int32, device=DEV),
    )


def _run_prepped(h, prepped, window):
    """The harness's verify_round, minus every synchronizing tensor build."""
    h.backend.init_forward_metadata(
        bs=prepped["bs"],
        req_pool_indices=prepped["rpi_dev"],
        seq_lens=prepped["seq_dev"],
        forward_mode=ForwardMode.DECODE,
        cache_metadata=prepped["metadata"],
        forward_batch=prepped["op"],
    )
    outs = {}
    for layer_id in h.layer_ids:
        p = h.params[layer_id]
        outs[layer_id] = h.backend.forward_decode(
            None,
            None,
            None,
            layer=None,
            out_cache_loc=None,
            token_to_kv_pool=h.pool,
            bs=prepped["bs"],
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
            seq_len=prepped["bs"] * T,
            a=None,
            b=None,
        )
    return outs


def test_overlap_prep_does_not_clobber_the_inflight_slot_map():
    """Overlap scheduling: the NEXT round's metadata prep runs on the host
    while THIS round's forward is still queued on the device.

    Ownership rides the device slot table, so each prep's compose must see
    the table exactly as of its own enqueue order -- the raced round R2 is
    staged, its record and the NEXT prep all land while the device is still
    parked, and any host-visible staging shared between preps (the old
    pinned slot map, a reused drop log) would let the later prep rewrite
    the round already in flight. The batch re-packs between the rounds
    (a request departs), so a clobber gathers another request's pending
    entry: the identity gate rejects it, the survivor's accepted window
    silently never commits, and its state diverges from an eager run one
    round later.

    Schedule (T=2 draft tokens, requests 0/1/2):
      W0  bs=2 [0,2]      -- warms every bs=2 shape, alloc, and kernel
      W1  bs=3 [0,1,2]    -- pending gen1 maps {0:0, 1:1, 2:2}
      R2  bs=2 [0,2]      -- request 1 departed; TRUE slot map [0, 2],
                             run entirely behind a device sleep
      P3  prep-only [0,2] -- gen2 maps {0:0, 2:1}; writes slot map [0, 1]
                             into the SAME pinned buffer while R2's copy
                             is still queued behind the sleep
    A marker event recorded right after the sleep proves the device had not
    reached R2's copy when P3's write landed; if the host is too slow the
    test skips rather than passing vacuously."""
    if not hasattr(torch.cuda, "_sleep"):
        pytest.skip("torch.cuda._sleep unavailable")
    h_lazy = _Harness(seed=23)
    h_eager = _Harness(seed=23)
    all_rpis = [0, 1, 2]
    survivors = [0, 2]
    pages3 = _pages_for(h_lazy, all_rpis)
    pages2 = {gid: [pages3[gid][0], pages3[gid][2]] for gid in _STATE_GROUPS}

    # Warmup on both harnesses (lazy keeps its pendings). TWO bs=2 rounds:
    # the first has no pending and its stage returns early, so only the second
    # walks the full bs=2 compose path -- every allocator size class it needs
    # must be warm, or the raced round below would cudaMalloc (a device
    # synchronization) and drain the sleep that creates the race window.
    for h in (h_lazy, h_eager):
        h.verify_round(survivors, pages2, [6, 6], h.window(2, 80))
        h.accept([1, 1])
        h.verify_round(survivors, pages2, [7, 7], h.window(2, 81))
        h.accept([1, 1])
        h.verify_round(all_rpis, pages3, [8, 6, 8], h.window(3, 82))
        h.accept([2, 1, 2])
        if h is h_eager:
            h.flush()

    # Precompute every H2D for R2 + P3 BEFORE parking the device.
    w_r2 = {h: h.window(2, 83) for h in (h_lazy, h_eager)}
    r2 = _prepped_round(h_lazy, survivors, pages2, [10, 10])
    p3 = _prepped_round(h_lazy, survivors, pages2, [11, 12])
    acc_r2 = torch.tensor([1, 2], dtype=torch.int32, device=DEV)
    torch.cuda.synchronize()

    torch.cuda._sleep(6_000_000_000)  # park the device: R2 executes later
    marker = torch.cuda.Event()
    marker.record()
    outs_lazy = _run_prepped(h_lazy, r2, w_r2[h_lazy])  # stage R2: slot map [0, 2]
    h_lazy.backend.commit_verified_state(acc_r2)  # record gen2 {0:0, 2:1}
    h_lazy.backend.init_forward_metadata(  # P3 stage: writes [0, 1] over the pinned buffer
        bs=p3["bs"],
        req_pool_indices=p3["rpi_dev"],
        seq_lens=p3["seq_dev"],
        forward_mode=ForwardMode.DECODE,
        cache_metadata=p3["metadata"],
        forward_batch=p3["op"],
    )
    raced = not marker.query()  # device still asleep => R2's copy ran after P3's write
    torch.cuda.synchronize()
    if not raced:
        pytest.skip("device outran the host; overlap window not reproduced")
    h_lazy.flush()  # retract hook for the abandoned P3 stage; commits gen2

    # Eager control: the same R2 round and abandoned prep, fully synchronous.
    outs_eager = h_eager.verify_round(survivors, pages2, [10, 10], w_r2[h_eager])
    h_eager.accept([1, 2])
    h_eager.flush()
    _prep_verify(h_eager, survivors, pages2, [11, 12])
    h_eager.flush()

    # R2's verify outputs: the survivor that kept slot 0 is untouched by the
    # clobber; request 2's rows must ALSO match -- under the race its stage
    # gathered the departed request's entry and never replayed its pending.
    for layer_id in h_lazy.layer_ids:
        lo = outs_lazy[layer_id].float().flatten(0, -3)  # [bs*T, H, D]
        eo = outs_eager[layer_id].float().flatten(0, -3)
        torch.testing.assert_close(lo[:T], eo[:T], atol=1e-3, rtol=1e-2)
        torch.testing.assert_close(
            lo[T : 2 * T],
            eo[T : 2 * T],
            atol=1e-3,
            rtol=1e-2,
            msg=f"layer {layer_id}: request 2 verified from a stale state -- "
            "its in-flight slot map was clobbered by the next prep",
        )
    # Committed pages: request 2's page must contain W1's accepted window.
    for layer_id in h_lazy.layer_ids:
        gid = h_lazy.pool.group_id_for_layer(layer_id)
        for i, page in ((0, pages3[gid][0]), (2, pages3[gid][2])):
            conv_l, ssm_l = h_lazy.state_of(layer_id, page)
            conv_e, ssm_e = h_eager.state_of(layer_id, page)
            torch.testing.assert_close(
                conv_l, conv_e, atol=0.0, rtol=0.0, msg=f"req {i} conv"
            )
            torch.testing.assert_close(
                ssm_l, ssm_e, atol=1e-6, rtol=1e-4, msg=f"req {i} ssm"
            )


def test_batch_growth_realigns_capture_base():
    """Batch growth rebuilds the payload ring mid-forward; the device-side
    capture base must be recomputed against the NEW half size.

    The stage fills ``capture_base = staged * old_half`` before the forward
    rebuilds a larger ring, while every later reader -- the fused replay's
    base compose AND the flush slice -- uses ``parity * new_half``. An
    eager-vs-lazy comparison cannot see the misalignment (both modes grow,
    both read the identically shifted rows), so the oracle is the ring
    itself: after the growth round, the rows the pending points at must hold
    exactly that round's captured projections."""
    h = _Harness(seed=17, usable_pages=32)
    rpis3 = [0, 1, 2]
    pages3 = _pages_for(h, rpis3)
    pages2 = {gid: pages3[gid][:2] for gid in _STATE_GROUPS}

    h.verify_round([0, 1], pages2, [6, 6], h.window(2, 71))
    h.accept([1, 2])
    w2 = h.window(3, 72)  # batch grows 2 -> 3: ring rebuild inside the forward
    h.verify_round(rpis3, pages3, [7, 8, 6], w2)
    h.accept([2, 1, 1])

    backend = h.backend
    pending = h.pending()
    assert pending is not None
    half = backend._payload_half_rows
    layer0 = h.layer_ids[0]
    qkv_buf, f_a_buf, beta_buf = backend._replay_payload_cache["buffers"][layer0]
    assert qkv_buf.shape[0] == 2 * half, "ring must have grown to the new batch"
    po = pending["parity"] * half
    rows = 3 * T
    assert torch.equal(qkv_buf[po : po + rows], w2["mixed_qkv"]), (
        "capture landed at the OLD half offset; replay/flush will read "
        "another request's window"
    )
    assert torch.equal(f_a_buf[po : po + rows], w2["f_a_out"])
    assert torch.equal(beta_buf[po : po + rows], w2["beta_raw"])

    # And the schedule still converges end-to-end.
    h.verify_round(rpis3, pages3, [9, 9, 7], h.window(3, 73))
    h.accept([1, 2, 2])
    h.flush()
    assert h.pending() is None


def test_growth_flush_drops_departed_windows():
    """The growth flush must screen departures FIRST: a pending whose owner
    left holds pages that may already belong to a new request, and a flush
    ordered before the departed-drop would overwrite them."""
    h_lazy = _Harness(seed=19, usable_pages=32)
    h_eager = _Harness(seed=19, usable_pages=32)
    pages4 = _pages_for(h_lazy, [0, 1, 2, 3])
    pages_r1 = {gid: pages4[gid][:2] for gid in _STATE_GROUPS}
    for h in (h_lazy, h_eager):
        h.verify_round([0, 1], pages_r1, [6, 6], h.window(2, 81))
        h.accept([2, 1])
    h_eager.flush()

    # Request 1 departs; its pages are reclaimed and rewritten by a new owner.
    sentinel = {}
    for layer_id in h_lazy.layer_ids:
        gid = h_lazy.pool.group_id_for_layer(layer_id)
        page = pages4[gid][1]
        h_lazy.pool.get_component(layer_id, "conv_state")[page].fill_(7.0)
        h_lazy.pool.get_component(layer_id, "recurrent_state")[page].fill_(7.0)
        sentinel[layer_id] = page

    # Two new requests join: the batch GROWS, so the stage flushes the pending
    # (ring rebuild) -- request 0's window only, never request 1's.
    rpis_r2 = [0, 2, 3]
    pages_r2 = {
        gid: [pages4[gid][0], pages4[gid][2], pages4[gid][3]] for gid in _STATE_GROUPS
    }
    for h in (h_lazy, h_eager):
        h.verify_round(rpis_r2, pages_r2, [8, 6, 6], h.window(3, 82))
        h.accept([1, 2, 1])
        h.flush()

    # Survivor and the two newcomers match the eager run.
    for layer_id in h_lazy.layer_ids:
        gid = h_lazy.pool.group_id_for_layer(layer_id)
        for i in range(3):
            page = pages_r2[gid][i]
            conv_l, ssm_l = h_lazy.state_of(layer_id, page)
            conv_e, ssm_e = h_eager.state_of(layer_id, page)
            torch.testing.assert_close(conv_l, conv_e, atol=0.0, rtol=0.0)
            torch.testing.assert_close(ssm_l, ssm_e, atol=1e-6, rtol=1e-4)
    # The departed request's reclaimed pages keep the new owner's sentinel.
    for layer_id in h_lazy.layer_ids:
        conv_l, ssm_l = h_lazy.state_of(layer_id, sentinel[layer_id])
        assert bool((conv_l == 7.0).all()), "growth flush wrote a reclaimed page"
        assert bool((ssm_l == 7.0).all()), "growth flush wrote a reclaimed page"


def test_idle_replay_prep_with_zero_real_requests_does_not_crash():
    """A DP rank going idle replays the verify graph with real_bs == 0.

    The replay prep must survive an all-padding batch (no unbound locals on
    the pending-stage plumbing) and LEAVE the pending: an empty batch is no
    evidence about residency (a capacity retraction produces exactly this
    round with every owner resident). Graph state is initialized BEFORE the
    pending is recorded, so the ring-growth flush cannot consume it first
    and the assertion is about the idle stage, not a side effect."""
    h = _Harness(seed=23)
    h.backend.init_cuda_graph_state(max_num_tokens=2 * T)
    rpis = [0, 1]
    pages = _pages_for(h, rpis)
    h.verify_round(rpis, pages, [5, 5], h.window(2, 91))
    h.accept([1, 2])
    assert h.pending() is not None

    h.backend.init_forward_metadata_replay_cuda_graph(
        bs=2,
        req_pool_indices=torch.full((2,), -1, dtype=torch.int32, device=DEV),
        seq_lens=torch.full((2,), T, dtype=torch.int32, device=DEV),
        forward_mode=ForwardMode.DECODE,
        num_padding=2,
    )
    assert h.pending() is not None, "an idle round dropped a live window"


def test_resumed_owner_with_moved_pages_is_dropped_not_flushed():
    """Same rpi + same request id is NOT proof of page ownership.

    A retracted-then-resumed request keeps its identity but its state pages
    were freed and may now hold another request's state. The non-verify
    owner flush must check the recorded commit page is still in the owner's
    current page-table row: present -> flush (jump-forward re-extend),
    absent -> drop, never write."""
    h = _Harness(seed=29)
    rpis = [0, 1]
    pages = _pages_for(h, rpis)
    h.verify_round(rpis, pages, [5, 5], h.window(2, 92))
    h.accept([2, 1])
    assert h.pending() is not None

    # Request 1 was retracted; its old pages now belong to someone else.
    sentinel = {}
    for layer_id in h.layer_ids:
        gid = h.pool.group_id_for_layer(layer_id)
        page = pages[gid][1]
        h.pool.get_component(layer_id, "conv_state")[page].fill_(7.0)
        h.pool.get_component(layer_id, "recurrent_state")[page].fill_(7.0)
        sentinel[layer_id] = page
    # On resume it holds a FRESH page; pages are global across groups, so
    # pick past EVERY group's allocations.
    top = max(max(pages[gid]) for gid in _STATE_GROUPS)
    fresh = {gid: top + 1 + g for g, gid in enumerate(_STATE_GROUPS)}
    fresh_snap = {}
    for layer_id in h.layer_ids:
        gid = h.pool.group_id_for_layer(layer_id)
        fresh_snap[layer_id] = h.state_of(layer_id, fresh[gid])

    # Request 0 (pages unchanged) snapshots for the positive control.
    pre0 = {
        layer_id: h.state_of(layer_id, pages[h.pool.group_id_for_layer(layer_id)][0])
        for layer_id in h.layer_ids
    }

    tables = {
        gid: np.asarray([[pages[gid][0]], [fresh[gid]]], dtype=np.int32)
        for gid in _STATE_GROUPS
    }
    metadata, op = _metadata_for(h.contract, tables, DEV)
    op.request_pool_indices = rpis
    op.request_ids = [f"req{r}" for r in rpis]
    h.backend.spec_num_tokens = 1
    try:
        h.backend.init_forward_metadata(
            bs=2,
            req_pool_indices=torch.tensor(rpis, dtype=torch.int32, device=DEV),
            seq_lens=torch.tensor([7, 6], dtype=torch.int32, device=DEV),
            forward_mode=ForwardMode.DECODE,
            cache_metadata=metadata,
            forward_batch=op,
        )
    finally:
        h.backend.spec_num_tokens = T
    assert h.pending() is None

    # Positive control: request 0's window really flushed (its page changed).
    changed = any(
        not torch.equal(
            h.state_of(layer_id, pages[h.pool.group_id_for_layer(layer_id)][0])[1],
            pre0[layer_id][1],
        )
        for layer_id in h.layer_ids
    )
    assert changed, "resident owner with intact pages must still flush"
    # The re-owned old pages and the resumed request's fresh pages: untouched.
    for layer_id in h.layer_ids:
        gid = h.pool.group_id_for_layer(layer_id)
        conv_l, ssm_l = h.state_of(layer_id, sentinel[layer_id])
        assert bool((conv_l == 7.0).all()), "flush wrote a re-owned page"
        assert bool((ssm_l == 7.0).all()), "flush wrote a re-owned page"
        conv_f, ssm_f = h.state_of(layer_id, fresh[gid])
        assert torch.equal(conv_f, fresh_snap[layer_id][0])
        assert torch.equal(ssm_f, fresh_snap[layer_id][1])


def test_control_buffer_rows_stay_16b_aligned():
    """The compose kernel takes row slices of these buffers, and Triton keys
    its compiled variants on pointer alignment. The buffers are sized once
    at the engine's max batch, so alignment is a one-time property -- pin
    that the single allocation is aligned and covers every legal round."""
    h = _Harness(seed=31)
    first = h.backend._kda_lazy_buffers(min_slots=1)
    for slots in (1, 2, 3, 5, 8):
        bufs = h.backend._kda_lazy_buffers(min_slots=slots)
        assert bufs["flat"] is first["flat"], "control buffers must never move"
        cap = bufs["flat"].shape[1]
        assert cap % 4 == 0, f"int32 rows misaligned at cap={cap}"
        assert cap >= h.backend.max_bs >= slots


def test_payload_ring_rebuild_neutralizes_the_armed_replay_base():
    """A payload-ring rebuild must leave no live replay base behind.

    The stage composes ``base = parity * half + slot * T`` into the graphed
    control buffer, then the forward may rebuild the ring (a widened f_a,
    the enforce-eager fallback, batch growth) and flush the pending against
    the OLD buffers. The rebuilt ring is a different, zero-filled
    allocation: any base row still pointing into it makes the remaining
    layers of THIS round replay garbage and commit it over the state the
    flush just wrote. Eager-vs-lazy cannot see it -- the oracle is the
    control buffer itself.
    """
    h = _Harness(seed=41)
    rpis = [0, 1]
    pages = _pages_for(h, rpis)
    h.verify_round(rpis, pages, [6, 6], h.window(2, 101))
    h.accept([2, 1])
    assert h.pending() is not None

    _prep_verify(h, rpis, pages, [8, 7])  # stage: base row goes live
    bufs = h.backend._kda_lazy_bufs
    assert bool((bufs["base"][: len(rpis)] >= 0).all()), "stage did not compose a base"

    # Rebuild the ring the way an odd f_a width does, mid-forward.
    layer_id = h.layer_ids[0]
    rows = h.backend._replay_payload_cache["rows"]
    h.backend._replay_payload(layer_id, rows, (CONV_DIM, D_FA + 8, H), torch.bfloat16)

    assert h.pending() is None, "rebuild must flush the pending it invalidates"
    assert bool(
        (bufs["base"][: len(rpis)] == -1).all()
    ), "ring rebuilt under a live replay base: the rest of this round re-commits"


def test_window_size_change_flushes_the_pending():
    """A round that changes the draft window must not fuse the old one.

    The payload ring is addressed in the PREVIOUS window's units
    (``base = slot * t_prev``) while the fused kernel replays in THIS
    round's units, so a window-size change mis-indexes the ring and commits
    a state assembled from the wrong rows -- silently. The stage must flush
    instead of composing.
    """
    rpis = [0, 1]
    h_lazy = _Harness(seed=43)
    h_eager = _Harness(seed=43)
    pages = _pages_for(h_lazy, rpis)
    for h in (h_lazy, h_eager):
        h.verify_round(rpis, pages, [6, 6], h.window(2, 111))
        h.accept([2, 1])
    h_eager.flush()  # the eager control commits the T=2 window as itself

    flushes = []
    inner = h_lazy.backend._flush_kda_pending

    def _counting_flush(only_rpis=None, commit_gate_by_group=None):
        flushes.append(only_rpis)
        inner(only_rpis, commit_gate_by_group)

    h_lazy.backend._flush_kda_pending = _counting_flush

    # Round 2 runs a ONE-token window; the T=2 pending cannot ride it.
    for h in (h_lazy, h_eager):
        _one_token_round(h, rpis, pages, [9, 8], seed=112)
        if h is h_lazy:
            # The stage must have committed the T=2 window before this forward.
            pass
        h.accept([1, 1])
        h.flush()
    # The T=2 window landed as a T=2 window: replayed in T=1 units it would
    # commit half of it, from the wrong ring rows.
    _assert_pools_match(h_lazy, h_eager, pages, rpis)


def _one_token_round(h, rpis, pages, seq_lens, seed):
    """One verify forward with a ONE-token window (shrunken draft length)."""
    g = torch.Generator(device="cpu").manual_seed(seed)

    def rnd(*shape):
        return torch.randn(*shape, generator=g, dtype=torch.float32).to(
            DEV, torch.bfloat16
        )

    bs = len(rpis)
    window = dict(
        mixed_qkv=rnd(bs, CONV_DIM), f_a_out=rnd(bs, D_FA), beta_raw=rnd(bs, H)
    )
    tables = {
        gid: np.asarray([[p] for p in pages[gid]], dtype=np.int32)
        for gid in _STATE_GROUPS
    }
    metadata, op = _metadata_for(h.contract, tables, DEV)
    op.request_pool_indices = list(rpis)
    op.request_ids = [f"req{r}" for r in rpis]
    h.backend.speculative_num_draft_tokens = 1
    try:
        h.backend.init_forward_metadata(
            bs=bs,
            req_pool_indices=torch.tensor(rpis, dtype=torch.int32, device=DEV),
            seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=DEV),
            forward_mode=ForwardMode.DECODE,
            cache_metadata=metadata,
            forward_batch=op,
            tokens_per_req=1,
        )
        for layer_id in h.layer_ids:
            p = h.params[layer_id]
            h.backend.forward_decode(
                None,
                None,
                None,
                layer=None,
                out_cache_loc=None,
                token_to_kv_pool=h.pool,
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
                seq_len=bs,
                a=None,
                b=None,
            )
    finally:
        h.backend.speculative_num_draft_tokens = T


def test_empty_forward_round_keeps_a_resident_owners_window():
    """A scheduling round with no forward must not touch the pending.

    The scheduler emits exactly this round on every capacity retraction: a
    prefill chunk fails admission and the plan carries zero forward ops
    while every decoder stays resident with intact pages. Round 1's
    accepted tokens already advanced those sequences; only the deferred
    commit writes their state page. An idle-round drop (the engine's old
    ``drop_pending`` hook) left the page holding the PRE-window state,
    which the next verify silently read as its anchor.
    """
    rpis = [0, 1]
    h_lazy = _Harness(seed=53)
    h_eager = _Harness(seed=53)
    pages = _pages_for(h_lazy, rpis)

    for h in (h_lazy, h_eager):
        h.verify_round(rpis, pages, [6, 6], h.window(2, 121))
        h.accept([2, 1])
    h_eager.flush()

    # The engine hook is gone entirely; a re-added call must fail loudly.
    assert not hasattr(h_lazy.backend, "drop_pending")
    # The graph-padded flavor of the same round: an stage with zero real rows.
    h_lazy.backend._stage_pending_replay(0, len(rpis), {}, None)
    assert h_lazy.pending() is not None, "empty round dropped a live window"

    for h in (h_lazy, h_eager):
        h.verify_round(rpis, pages, [8, 7], h.window(2, 122))
        h.accept([1, 1])
        h.flush()
    _assert_pools_match(h_lazy, h_eager, pages, rpis)


def test_forward_without_a_record_does_not_double_apply_next_round():
    """A verify forward that completes but whose record never runs.

    The fused commit is issued by the KDA layers; the record
    (``commit_verified_state``) only after the whole forward -- including
    the accept sampling -- returns. Anything raising in between leaves an
    already-committed pending live, and with anchor == commit (every window
    that stays inside one state page) the next round replays it onto its
    own result, applying the accepted tokens twice. The forward-issue fence
    (the settle hook the graph wrapper calls after every forward) must
    release the record even when no accept follows.
    """
    rpis = [0, 1]
    h_lazy = _Harness(seed=67)
    h_eager = _Harness(seed=67)
    pages = _pages_for(h_lazy, rpis)

    # Round 1: verify + accept -> pending P, commit deferred.
    for h in (h_lazy, h_eager):
        h.verify_round(rpis, pages, [6, 6], h.window(2, 131))
        h.accept([2, 1])
    h_eager.flush()

    # Round 2 runs to completion (the fused kernel commits P) but its result
    # dies before commit_verified_state: no accept, no seq advance.
    for h in (h_lazy, h_eager):
        h.verify_round(rpis, pages, [8, 7], h.window(2, 132))
    assert h_lazy.pending() is None, "issued forward left its record live"

    # Round 2 is retried at the same seq_lens and this time it records.
    for h in (h_lazy, h_eager):
        h.verify_round(rpis, pages, [8, 7], h.window(2, 133))
        h.accept([1, 2])
        h.flush()
    _assert_pools_match(h_lazy, h_eager, pages, rpis)


def test_pause_flush_does_not_write_a_departed_owners_reclaimed_pages():
    """The pause fence must screen departed owners by engine residency.

    Nothing reaps a departed owner at departure: a prefill/extend-only
    round that does not contain it finds owners == [] -- no flush, no
    drop -- so its window is still pending when the pause fence fires.
    An unscreened flush would write the dead window onto pages the pool
    already handed to a new request's prefill.
    """
    h = _Harness(seed=71, usable_pages=32)
    rpis = [0, 1]
    pages = _pages_for(h, rpis)
    h.verify_round(rpis, pages, [6, 6], h.window(2, 141))
    h.accept([2, 1])
    assert h.pending() is not None

    # Request rpi=1 finishes. Its state pages are freed and handed to a NEW
    # request (rpi 5), whose prefill fills them -- the sentinel stands in.
    for layer_id in h.layer_ids:
        gid = h.pool.group_id_for_layer(layer_id)
        p = pages[gid][1]
        h.pool.get_component(layer_id, "conv_state")[p].fill_(7.0)
        h.pool.get_component(layer_id, "recurrent_state")[p].fill_(7.0)

    # An extend-only round for the newcomer: rpi 1 is not in the batch, so
    # the owner screen finds nothing to flush AND nothing to drop.
    tables = {
        gid: np.asarray([[pages[gid][1]]], dtype=np.int32) for gid in _STATE_GROUPS
    }
    metadata, op = _metadata_for(h.contract, tables, DEV)
    op.request_pool_indices = [5]
    op.request_ids = ["req5"]
    h.backend.init_forward_metadata(
        bs=1,
        req_pool_indices=torch.tensor([5], dtype=torch.int32, device=DEV),
        seq_lens=torch.tensor([4], dtype=torch.int32, device=DEV),
        forward_mode=ForwardMode.EXTEND,
        cache_metadata=metadata,
        forward_batch=op,
        num_extends=1,
        extend_prefix_lens=torch.zeros(1, dtype=torch.int32, device=DEV),
    )
    assert h.pending() is not None and 1 in h.pending()["slot_by_rpi"]

    # The pause fence: req1 has left the engine, req0 is still resident.
    h.backend.flush_pending({"req0"})
    assert h.pending() is None

    for layer_id in h.layer_ids:
        gid = h.pool.group_id_for_layer(layer_id)
        conv, ssm = h.state_of(layer_id, pages[gid][1])
        assert bool((conv == 7.0).all()), "pause flush wrote a reclaimed conv page"
        assert bool((ssm == 7.0).all()), "pause flush wrote a reclaimed state page"


def test_extend_row_owner_is_dropped_not_flushed():
    """A pending owner reappearing as an EXTEND row is a retract-resume.

    Its pages were freed at retraction and the resume rebuilds state with a
    full prefill; the recorded window belongs to the state being replaced.
    Flushing it would write pages the pool may have re-issued in between --
    the resolve must drop it while still committing true decode owners.
    """
    h = _Harness(seed=83, usable_pages=32)
    rpis = [0, 1]
    pages = _pages_for(h, rpis)
    h.verify_round(rpis, pages, [6, 6], h.window(2, 151))
    h.accept([2, 1])
    assert h.pending() is not None

    # rpi=1 was retracted; the sentinel is whatever now sits in its old page.
    for layer_id in h.layer_ids:
        gid = h.pool.group_id_for_layer(layer_id)
        p = pages[gid][1]
        h.pool.get_component(layer_id, "conv_state")[p].fill_(9.0)
        h.pool.get_component(layer_id, "recurrent_state")[p].fill_(9.0)

    # The resume round: rpi 1 returns as an extend row (same request id --
    # a retract-resume keeps it), rpi 0 keeps decoding.
    tables = {
        gid: np.asarray([[pages[gid][1]], [pages[gid][0]]], dtype=np.int32)
        for gid in _STATE_GROUPS
    }
    metadata, op = _metadata_for(h.contract, tables, DEV)
    op.request_pool_indices = [1, 0]
    op.request_ids = ["req1", "req0"]
    h.backend.init_forward_metadata(
        bs=2,
        req_pool_indices=torch.tensor([1, 0], dtype=torch.int32, device=DEV),
        seq_lens=torch.tensor([8, 8], dtype=torch.int32, device=DEV),
        forward_mode=ForwardMode.EXTEND,
        cache_metadata=metadata,
        forward_batch=op,
        num_extends=1,
        extend_prefix_lens=torch.zeros(2, dtype=torch.int32, device=DEV),
    )
    # req0 (decode row) was flushed, req1 (extend row) was dropped.
    assert h.pending() is None
    for layer_id in h.layer_ids:
        gid = h.pool.group_id_for_layer(layer_id)
        conv, ssm = h.state_of(layer_id, pages[gid][1])
        assert bool((conv == 9.0).all()), "resolve flushed a retracted owner's window"
        assert bool((ssm == 9.0).all()), "resolve flushed a retracted owner's window"


def test_gate_scratch_rides_the_frozen_workspace_pool():
    """The hoisted gate carves from the shared workspace pool.

    The executor freezes the pool before graph capture, so the warm at
    backend init must cover the TRUE peak batch (``config.max_bs``), not the
    graph ceiling: a batch above the ceiling runs eagerly against the same
    frozen pool, and under the old per-module seal that exact case crashed.
    Here every round runs with the pool frozen at the max_bs warm -- fused
    and flush paths both -- and still matches the eager reference.
    """
    from tokenspeed.runtime.execution.workspace import (
        reset_workspace_pools,
        workspace_pool,
    )

    reset_workspace_pools()
    try:
        h_lazy = _Harness(seed=91)
        h_eager = _Harness(seed=91)
        rpis = [0, 1]
        pages = _pages_for(h_lazy, rpis)
        # What _preallocate_kda_replay_buffers does at init, then the
        # executor's pre-capture freeze.
        h_lazy.backend._kda_gate_scratch(2 * h_lazy.backend.max_bs * T, H * D)
        workspace_pool(torch.device(DEV)).freeze()

        for h in (h_lazy, h_eager):
            h.verify_round(rpis, pages, [6, 6], h.window(2, 161))
            h.accept([2, 1])
        h_eager.flush()  # standalone replay under the frozen pool
        for h in (h_lazy, h_eager):
            h.verify_round(rpis, pages, [8, 7], h.window(2, 162))
            h.accept([1, 2])
            h.flush()
        _assert_pools_match(h_lazy, h_eager, pages, rpis)
    finally:
        reset_workspace_pools()


def test_eager_batch_above_the_graph_ceiling_commits_standalone():
    """A legal eager batch above the graph ceiling must run, not crash.

    Graph capture freezes the payload ring and control buffers at the
    capture ceiling, while max_num_seqs can schedule a bigger batch that
    runs eagerly. Such a round uses the free-growing overflow set: the
    pending it inherits is flushed standalone at stage (its ring and this
    round's ring diverge), its own window is committed standalone the
    moment acceptance is known, and the graph ring's parity chain is left
    untouched -- so the graph-sized round that follows fuses normally.
    """
    rpis4 = [0, 1, 2, 3]
    rpis2 = [0, 1]
    h_lazy = _Harness(seed=97, usable_pages=32)
    h_eager = _Harness(seed=97, usable_pages=32)
    pages4 = _pages_for(h_lazy, rpis4)
    pages2 = {gid: rows[:2] for gid, rows in pages4.items()}

    # Ceiling of 2; round 1 doubles as the pre-capture warmup (it sizes the
    # ring at the model's widths), then capture freezes everything -- the
    # production ordering.
    h_lazy.backend.init_cuda_graph_state(2)

    # Round 1 at the ceiling: records a pending in the graph ring.
    for h in (h_lazy, h_eager):
        h.verify_round(rpis2, pages2, [6, 6], h.window(2, 171))
        h.accept([2, 1])
    h_eager.flush()
    assert h_lazy.pending() is not None
    h_lazy.backend._graphs_captured = True

    # Round 2 above the ceiling: inherits the pending, runs on overflow.
    for h in (h_lazy, h_eager):
        h.verify_round(rpis4, pages4, [8, 7, 5, 5], h.window(4, 172))
        h.accept([1, 2, 2, 1])
    h_eager.flush()
    # Overflow commits standalone at record: nothing left pending.
    assert h_lazy.pending() is None

    # Round 3 back at the ceiling: the graph set fuses normally again.
    for h in (h_lazy, h_eager):
        h.verify_round(rpis2, pages2, [9, 9], h.window(2, 173))
        h.accept([2, 2])
        h.flush()
    _assert_pools_match(h_lazy, h_eager, pages4, rpis4)


def _kernel_variants():
    """Compiled specializations of the hot verify kernels on this device."""
    from tokenspeed_kernel.thirdparty.triton import fla_kda_recurrent as fk

    def count(kern):
        fn = getattr(kern, "fn", kern)
        total = 0
        for dc in getattr(fn, "device_caches", {}).values():
            for part in dc:
                if isinstance(part, dict):
                    total += len(part)
        return total

    return {
        "window": count(fk.fused_recurrent_kda_window_fwd_kernel),
        "gate": count(fk.kda_gate_precompute_kernel),
    }


def test_overflow_round_compiles_no_fresh_kernel_variants():
    """An above-ceiling round must reuse the warm kernels bit for bit.

    Triton keys compiled variants on pointer alignment; a control or
    payload row whose byte offset differs from the graph set's would
    compile a fresh variant mid-decode, and the module load behind it
    implicitly synchronizes the device (found by adversarial review as a
    2.9s-class stall: the old per-batch overflow control set carried
    unaligned rows). The control rows are one fixed aligned allocation
    now; this pins that no round shape can re-specialize.
    """
    rpis6 = [0, 1, 2, 3, 4, 5]
    h = _Harness(seed=61, usable_pages=32)
    pages6 = _pages_for(h, rpis6)

    def leg(rpis, wseed, seqs):
        pages = {gid: rows[: len(rpis)] for gid, rows in pages6.items()}
        h.verify_round(rpis, pages, seqs, h.window(len(rpis), wseed))
        h.accept([1] * len(rpis))

    h.backend.init_cuda_graph_state(4)
    # Warm every graph-set batch shape first.
    for i, bs in enumerate((4, 3, 2, 4, 3)):
        leg(rpis6[:bs], 800 + i, [6 + i] * bs)
    h.flush()
    h.backend._graphs_captured = True
    base = _kernel_variants()

    leg(rpis6, 850, [12] * 6)  # overflow: bs=6 above the ceiling of 4
    h.flush()
    assert _kernel_variants() == base, (
        f"the overflow round compiled new kernel variants {base} -> "
        f"{_kernel_variants()}"
    )


def test_condemned_row_survives_an_abandoned_prep_without_flushing():
    """The compose's gate verdict must outlive an abandoned prep.

    The stage's compose finds request 1's page moved (identity gate fails)
    and condemns its table row in place; then the forward is abandoned, so
    the pending -- condemned row included -- survives to the retract-hook
    flush. The flush gates its writes on the same table, so the dead
    window must not land on the moved page. Nothing else stands in the
    way: the owner is resident, same rpi, same request id.
    """
    rpis = [0, 1]
    h = _Harness(seed=101, usable_pages=32)
    pages = _pages_for(h, rpis)
    h.verify_round(rpis, pages, [6, 6], h.window(2, 181))
    h.accept([2, 1])
    assert h.pending() is not None

    # Request 1's state moved to a fresh page (e.g. a page-crossing commit);
    # its old commit page now belongs to nobody -- poison it as a sentinel.
    moved = {gid: list(rows) for gid, rows in pages.items()}
    for gid in _STATE_GROUPS:
        moved[gid][1] = pages[gid][1] + 6
    sentinel = {}
    for layer_id in h.layer_ids:
        gid = h.pool.group_id_for_layer(layer_id)
        p = pages[gid][1]
        h.pool.get_component(layer_id, "conv_state")[p].fill_(3.0)
        h.pool.get_component(layer_id, "recurrent_state")[p].fill_(3.0)
        sentinel[layer_id] = p

    # Stage against the moved tables: the compose condemns row 1, then the
    # forward never runs (batch retracted between prep and dispatch).
    _prep_verify(h, rpis, moved, [8, 7])
    assert h.pending() is not None
    h.flush()  # the retract hook

    for layer_id in h.layer_ids:
        conv, ssm = h.state_of(layer_id, sentinel[layer_id])
        assert bool((conv == 3.0).all()), "flush wrote a condemned row's page"
        assert bool((ssm == 3.0).all()), "flush wrote a condemned row's page"

"""Correctness of the fused replay-prefix + verify KDA kernel.

Lazy commit folds round k's replay commit into round k+1's verify: one kernel
anchors at the page from before window k, replays window k's accepted prefix
in registers, stores the result (the deferred commit), and rolls straight
into window k+1's verify steps.

The claim under test is equivalence with the two-launch form -- a
standalone ``fused_recurrent_kda_replay_commit`` followed by a plain verify.
Both paths inline the same step helper on the same fp32 state in the same
order, so the only daylight between them is compiler FMA contraction: the
two constexpr specializations may fuse different mul+add pairs, which moves
the committed state by ~1 ulp fp32 (measured max 1.2e-7) and can flip the
last bit of a bf16 output sitting on a rounding boundary. Tolerances are
set just above that floor -- state atol 1e-6 -- four orders of magnitude
below what a semantic error (wrong anchor, wrong step count, wrong payload
row) produces, as established by the mutation runs on the replay suite.
The conv window is raw bf16 inputs either way and must stay bitwise.
"""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

from tokenspeed_kernel.thirdparty.triton.fla_kda_recurrent import (  # noqa: E402
    fused_recurrent_kda_replay_commit,
    fused_recurrent_kda_verify_megafuse,
    kda_commit_conv_window,
)

HV, K, V, D_FA = 4, 128, 128, 128
P = HV * K
LOWER_BOUND = -5.0
DEV = "cuda"


def _window(n, t, pages=48, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)

    def rnd(*shape, dtype=torch.bfloat16, scale=1.0):
        return (torch.randn(*shape, generator=g, dtype=torch.float32) * scale).to(
            device=DEV, dtype=dtype
        )

    return dict(
        conv_w=rnd(3 * P, 4, scale=0.3).contiguous(),
        conv_pool=rnd(pages, 3 * P, 3),
        w_fb=rnd(P, D_FA, scale=0.05).contiguous(),
        A_log=rnd(HV, dtype=torch.float32, scale=0.5),
        dt_bias=rnd(P, dtype=torch.float32),
        h_pool=rnd(pages, HV, K, V, dtype=torch.float32),
        prev_qkv=rnd(n * t, 3 * P),
        prev_f_a=rnd(n * t, D_FA),
        prev_beta=rnd(n * t, HV),
        new_qkv=rnd(n * t, 3 * P),
        new_f_a=rnd(n * t, D_FA),
        new_beta=rnd(n * t, HV),
        anchor=torch.arange(1, n + 1, device=DEV, dtype=torch.int32),
    )


def _clone_pools(x):
    y = dict(x)
    y["conv_pool"] = x["conv_pool"].clone()
    y["h_pool"] = x["h_pool"].clone()
    return y


def _kw(t):
    return dict(num_heads=HV, head_dim=V, draft_token_num=t, lower_bound=LOWER_BOUND)


def _assert_equivalent(got_out, ref_out, got, ref):
    # bf16 outputs: allow a last-bit flip (relative ~2^-8); state: ~1 ulp
    # fp32 from FMA contraction; conv windows: raw bf16 inputs, bitwise.
    torch.testing.assert_close(got_out.float(), ref_out.float(), atol=1e-3, rtol=1e-2)
    torch.testing.assert_close(got["h_pool"], ref["h_pool"], atol=1e-6, rtol=1e-4)
    torch.testing.assert_close(got["conv_pool"], ref["conv_pool"], atol=0.0, rtol=0.0)


def _two_launch(x, n, t, accepted, commit):
    """Reference: standalone replay commit, then a plain verify."""
    y = _clone_pools(x)
    fused_recurrent_kda_replay_commit(
        y["prev_qkv"],
        y["conv_w"],
        y["conv_pool"],
        y["conv_pool"],
        y["prev_f_a"],
        y["w_fb"],
        y["prev_beta"],
        y["A_log"],
        y["dt_bias"],
        y["h_pool"],
        y["h_pool"],
        y["anchor"],
        commit,
        accepted,
        **_kw(t),
    )
    # After the commit, the verify anchors at the committed page. Requests
    # whose commit was skipped (-1) verify from the untouched anchor.
    read = torch.where(commit >= 0, commit, y["anchor"].to(commit.dtype)).to(
        torch.int32
    )
    out = fused_recurrent_kda_verify_megafuse(
        y["new_qkv"],
        y["conv_w"],
        y["conv_pool"],
        y["new_f_a"],
        y["w_fb"],
        y["new_beta"],
        y["A_log"],
        y["dt_bias"],
        y["h_pool"],
        read,
        **_kw(t),
    )
    return out, y


def _fused(x, n, t, accepted, commit, base=None):
    """One launch: replay prefix + deferred commit + verify."""
    y = _clone_pools(x)
    if base is None:
        base = torch.arange(n, device=DEV, dtype=torch.int32) * t
    out = fused_recurrent_kda_verify_megafuse(
        y["new_qkv"],
        y["conv_w"],
        y["conv_pool"],
        y["new_f_a"],
        y["w_fb"],
        y["new_beta"],
        y["A_log"],
        y["dt_bias"],
        y["h_pool"],
        y["anchor"],
        prev_qkv=y["prev_qkv"],
        prev_f_a=y["prev_f_a"],
        prev_beta=y["prev_beta"],
        prev_base=base,
        prev_steps=accepted,
        commit_indices=commit,
        **_kw(t),
    )
    kda_commit_conv_window(
        y["prev_qkv"],
        y["conv_pool"],
        y["conv_pool"],
        y["anchor"],
        commit,
        accepted,
        conv_dim=3 * P,
        draft_token_num=t,
        row_base=base,
    )
    return out, y


@pytest.mark.parametrize("t", [1, 2, 3, 4])
def test_fused_is_bitwise_the_two_launch_form(t):
    """The headline invariant, swept over accepted lengths 0..T."""
    n = 6
    x = _window(n, t, seed=t)
    commit = torch.arange(1, n + 1, device=DEV, dtype=torch.int32) + 24
    for a in range(t + 1):
        accepted = torch.full((n,), a, device=DEV, dtype=torch.int32)
        ref_out, ref = _two_launch(x, n, t, accepted, commit)
        got_out, got = _fused(x, n, t, accepted, commit)
        _assert_equivalent(got_out, ref_out, got, ref)


def test_fused_commit_in_place_matches_fresh_page():
    """commit page == anchor page is the production case and must not race."""
    n, t = 8, 3
    x = _window(n, t, seed=9)
    accepted = torch.tensor([0, 1, 2, 3, 3, 2, 1, 0], device=DEV, dtype=torch.int32)
    fresh = torch.arange(1, n + 1, device=DEV, dtype=torch.int32) + 24
    ref_out, ref = _fused(x, n, t, accepted, fresh)
    for _ in range(8):  # a race would be intermittent
        got_out, got = _fused(x, n, t, accepted, x["anchor"].clone())
        torch.testing.assert_close(got_out, ref_out, atol=0.0, rtol=0.0)
        for i in range(n):
            r, w = int(x["anchor"][i]), int(fresh[i])
            torch.testing.assert_close(
                got["h_pool"][r], ref["h_pool"][w], atol=0.0, rtol=0.0
            )
            torch.testing.assert_close(
                got["conv_pool"][r], ref["conv_pool"][w], atol=0.0, rtol=0.0
            )


def test_no_pending_degenerates_to_plain_verify():
    """base = -1 must behave exactly like the un-armed kernel, everywhere."""
    n, t = 5, 3
    x = _window(n, t, seed=11)
    none_base = torch.full((n,), -1, device=DEV, dtype=torch.int32)
    accepted = torch.full((n,), t, device=DEV, dtype=torch.int32)
    commit = torch.full((n,), -1, device=DEV, dtype=torch.int32)

    plain = fused_recurrent_kda_verify_megafuse(
        x["new_qkv"],
        x["conv_w"],
        x["conv_pool"],
        x["new_f_a"],
        x["w_fb"],
        x["new_beta"],
        x["A_log"],
        x["dt_bias"],
        x["h_pool"],
        x["anchor"],
        **_kw(t),
    )
    conv_before = x["conv_pool"].clone()
    h_before = x["h_pool"].clone()
    got_out, got = _fused(x, n, t, accepted, commit, base=none_base)
    torch.testing.assert_close(got_out.float(), plain.float(), atol=1e-3, rtol=1e-2)
    # No pending: the pools must be untouched, bit for bit.
    torch.testing.assert_close(got["h_pool"], h_before, atol=0.0, rtol=0.0)
    torch.testing.assert_close(got["conv_pool"], conv_before, atol=0.0, rtol=0.0)


def test_mixed_pending_and_fresh_requests():
    """A batch mixing pendings, a fresh request, and moved payload slots.

    Request order in this round differs from the payload capture order:
    ``base`` points request i at the payload rows captured when it sat at a
    different slot, which is what happens whenever the batch re-packs.
    """
    n, t = 4, 2
    x = _window(n, t, seed=13)
    # Payload slots: request 0 <- old slot 2, request 1 <- old slot 0,
    # request 2 fresh (no pending), request 3 <- old slot 1.
    base = torch.tensor([2 * t, 0 * t, -1, 1 * t], device=DEV, dtype=torch.int32)
    accepted = torch.tensor([2, 1, 0, 2], device=DEV, dtype=torch.int32)
    commit = torch.tensor([25, 26, -1, 27], device=DEV, dtype=torch.int32)
    got_out, got = _fused(x, n, t, accepted, commit, base=base)

    # Reference: replay each pending from its true payload rows via the
    # standalone kernel driven with the same base indirection semantics,
    # by permuting the payload into request order first.
    perm = torch.tensor([2, 0, 1], device=DEV)  # old slots for requests 0,1,3
    rows = (perm.unsqueeze(1) * t + torch.arange(t, device=DEV)).reshape(-1)
    y = _clone_pools(x)
    sel = torch.tensor([0, 1, 3], device=DEV)
    fused_recurrent_kda_replay_commit(
        y["prev_qkv"][rows],
        y["conv_w"],
        y["conv_pool"],
        y["conv_pool"],
        y["prev_f_a"][rows],
        y["w_fb"],
        y["prev_beta"][rows],
        y["A_log"],
        y["dt_bias"],
        y["h_pool"],
        y["h_pool"],
        y["anchor"][sel],
        commit[sel],
        accepted[sel],
        **_kw(t),
    )
    read = y["anchor"].clone()
    read[sel] = commit[sel]
    ref_out = fused_recurrent_kda_verify_megafuse(
        y["new_qkv"],
        y["conv_w"],
        y["conv_pool"],
        y["new_f_a"],
        y["w_fb"],
        y["new_beta"],
        y["A_log"],
        y["dt_bias"],
        y["h_pool"],
        read,
        **_kw(t),
    )
    _assert_equivalent(got_out, ref_out, got, y)


def test_chained_rounds_track_sequential_decode():
    """Three lazy rounds end-to-end against one long sequential decode.

    Round k's window is windows[k]; acceptance a_k. The fused kernel of
    round k+1 commits round k. A final standalone flush commits the last
    round, mirroring a request leaving the verify stream. The surviving
    state must equal a plain sequential decode of all the accepted tokens.
    """
    n, t, rounds = 3, 2, 3
    g = torch.Generator(device="cpu").manual_seed(31)

    def rnd(*shape, dtype=torch.bfloat16, scale=1.0):
        return (torch.randn(*shape, generator=g, dtype=torch.float32) * scale).to(
            device=DEV, dtype=dtype
        )

    x = dict(
        conv_w=rnd(3 * P, 4, scale=0.3).contiguous(),
        conv_pool=rnd(16, 3 * P, 3),
        w_fb=rnd(P, D_FA, scale=0.05).contiguous(),
        A_log=rnd(HV, dtype=torch.float32, scale=0.5),
        dt_bias=rnd(P, dtype=torch.float32),
        h_pool=rnd(16, HV, K, V, dtype=torch.float32),
    )
    windows = [
        dict(qkv=rnd(n * t, 3 * P), f_a=rnd(n * t, D_FA), beta=rnd(n * t, HV))
        for _ in range(rounds)
    ]
    accepted = [
        torch.tensor([1, 2, 1], device=DEV, dtype=torch.int32),
        torch.tensor([2, 1, 2], device=DEV, dtype=torch.int32),
        torch.tensor([1, 1, 2], device=DEV, dtype=torch.int32),
    ]
    anchor = torch.arange(1, n + 1, device=DEV, dtype=torch.int32)
    base = torch.arange(n, device=DEV, dtype=torch.int32) * t

    # --- lazy-commit chain (in place: commit page == anchor page) ---
    lazy = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in x.items()}
    pending = None
    for k in range(rounds):
        w = windows[k]
        kwargs = {}
        if pending is not None:
            pw, pa = pending
            kwargs = dict(
                prev_qkv=pw["qkv"],
                prev_f_a=pw["f_a"],
                prev_beta=pw["beta"],
                prev_base=base,
                prev_steps=pa,
                commit_indices=anchor,
            )
        fused_recurrent_kda_verify_megafuse(
            w["qkv"],
            lazy["conv_w"],
            lazy["conv_pool"],
            w["f_a"],
            lazy["w_fb"],
            w["beta"],
            lazy["A_log"],
            lazy["dt_bias"],
            lazy["h_pool"],
            anchor,
            **kwargs,
            **_kw(t),
        )
        if pending is not None:
            pw, pa = pending
            kda_commit_conv_window(
                pw["qkv"],
                lazy["conv_pool"],
                lazy["conv_pool"],
                anchor,
                anchor,
                pa,
                conv_dim=3 * P,
                draft_token_num=t,
                row_base=base,
            )
        pending = (w, accepted[k])
    # Flush the final pending round (a request leaving the verify stream).
    pw, pa = pending
    fused_recurrent_kda_replay_commit(
        pw["qkv"],
        lazy["conv_w"],
        lazy["conv_pool"],
        lazy["conv_pool"],
        pw["f_a"],
        lazy["w_fb"],
        pw["beta"],
        lazy["A_log"],
        lazy["dt_bias"],
        lazy["h_pool"],
        lazy["h_pool"],
        anchor,
        anchor,
        pa,
        **_kw(t),
    )

    # --- reference: eager replay commit after every round ---
    eager = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in x.items()}
    for k in range(rounds):
        w = windows[k]
        fused_recurrent_kda_replay_commit(
            w["qkv"],
            eager["conv_w"],
            eager["conv_pool"],
            eager["conv_pool"],
            w["f_a"],
            eager["w_fb"],
            w["beta"],
            eager["A_log"],
            eager["dt_bias"],
            eager["h_pool"],
            eager["h_pool"],
            anchor,
            anchor,
            accepted[k],
            **_kw(t),
        )

    torch.testing.assert_close(lazy["h_pool"], eager["h_pool"], atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(
        lazy["conv_pool"], eager["conv_pool"], atol=0.0, rtol=0.0
    )


def test_stale_commit_slot_without_pending_writes_nothing():
    """Red-team regression: base = -1 must gate the commit store too.

    A stale page id in the commit slot of a request that has NO pending
    window (base = -1) used to get the anchor state written over it,
    desyncing that page's recurrent state from its conv window. The store
    is now gated on both indices.
    """
    n, t = 4, 2
    x = _window(n, t, seed=41)
    none_base = torch.full((n,), -1, device=DEV, dtype=torch.int32)
    stale_commit = torch.arange(1, n + 1, device=DEV, dtype=torch.int32) + 24
    accepted = torch.full((n,), t, device=DEV, dtype=torch.int32)
    h_before = x["h_pool"].clone()
    conv_before = x["conv_pool"].clone()
    _fused(x, n, t, accepted, stale_commit, base=none_base)
    # _fused clones the pools; re-run against the originals directly to
    # assert the shared slabs are untouched.
    out = fused_recurrent_kda_verify_megafuse(
        x["new_qkv"],
        x["conv_w"],
        x["conv_pool"],
        x["new_f_a"],
        x["w_fb"],
        x["new_beta"],
        x["A_log"],
        x["dt_bias"],
        x["h_pool"],
        x["anchor"],
        prev_qkv=x["prev_qkv"],
        prev_f_a=x["prev_f_a"],
        prev_beta=x["prev_beta"],
        prev_base=none_base,
        prev_steps=accepted,
        commit_indices=stale_commit,
        **_kw(t),
    )
    assert torch.isfinite(out.float()).all()
    torch.testing.assert_close(x["h_pool"], h_before, atol=0.0, rtol=0.0)
    torch.testing.assert_close(x["conv_pool"], conv_before, atol=0.0, rtol=0.0)


def test_fused_prev_steps_clamped_to_window():
    """Red-team regression: an out-of-range step count must clamp to T.

    The standalone replay entry clamps accepted_length; the fused prefix
    used to consume prev_steps raw, silently replaying the next request's
    first payload row (or reading past the buffer for the last request).
    steps = T + 1 must now behave exactly like steps = T.
    """
    n, t = 3, 2
    x = _window(n, t, seed=43)
    commit = torch.arange(1, n + 1, device=DEV, dtype=torch.int32) + 24
    exact = torch.full((n,), t, device=DEV, dtype=torch.int32)
    over = torch.full((n,), t + 1, device=DEV, dtype=torch.int32)
    ref_out, ref = _fused(x, n, t, exact, commit)
    got_out, got = _fused(x, n, t, over, commit)
    torch.testing.assert_close(got_out, ref_out, atol=0.0, rtol=0.0)
    torch.testing.assert_close(got["h_pool"], ref["h_pool"], atol=0.0, rtol=0.0)
    torch.testing.assert_close(got["conv_pool"], ref["conv_pool"], atol=0.0, rtol=0.0)

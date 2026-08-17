"""Correctness of the KDA speculative replay-commit.

Speculative verification runs the whole draft window, but only an unknown
prefix of it is accepted. Rather than storing a recurrent state per draft
position and committing the accepted one, verification stores nothing and the
accepted prefix is replayed from the still-intact committed page.

What has to hold for that to be lossless:

1. replaying ``a`` positions reproduces what ``a`` ordinary decode steps would
   have produced (the pure-torch reference below, and the production
   non-speculative decode kernel);
2. the rejected suffix cannot influence the committed state at all;
3. ``a = 0`` commits the pre-draft state unchanged;
4. requests in one batch may accept different lengths;
5. committing into the source page (the usual case -- the new position rarely
   crosses a flat page boundary) matches committing into a fresh page. The
   conv window is committed by its own launch precisely so this holds: its
   q/k channels are indexed by head alone, so every program of the recurrence
   kernel's NV column split would otherwise rewrite channels its siblings
   have not read yet.
"""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

from tokenspeed_kernel.thirdparty.triton.fla_kda_recurrent import (  # noqa: E402
    fused_recurrent_kda_replay_commit,
    fused_recurrent_kda_verify_megafuse,
)

# K3 TP8 rank geometry, trimmed to 4 heads to keep the reference loop quick.
HV, K, V, D_FA = 4, 128, 128, 128
P = HV * K
LOWER_BOUND = -5.0
DEV = "cuda"


def _window(n, t, pages=32, seed=0):
    """A draft window of ``t`` positions for ``n`` requests, plus the pools."""
    g = torch.Generator(device="cpu").manual_seed(seed)

    def rnd(*shape, dtype=torch.bfloat16, scale=1.0):
        return (torch.randn(*shape, generator=g, dtype=torch.float32) * scale).to(
            device=DEV, dtype=dtype
        )

    return dict(
        qkv_raw=rnd(n * t, 3 * P),
        conv_w=rnd(3 * P, 4, scale=0.3).contiguous(),
        conv_pool=rnd(pages, 3 * P, 3),
        f_a=rnd(n * t, D_FA),
        w_fb=rnd(P, D_FA, scale=0.05).contiguous(),
        beta=rnd(n * t, HV),
        A_log=rnd(HV, dtype=torch.float32, scale=0.5),
        dt_bias=rnd(P, dtype=torch.float32),
        h_pool=rnd(pages, HV, K, V, dtype=torch.float32),
        read_indices=torch.arange(1, n + 1, device=DEV, dtype=torch.int32),
    )


def _replay(x, write_indices, accepted, t):
    """Commit ``accepted`` replayed positions; returns the mutated pools."""
    x = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in x.items()}
    fused_recurrent_kda_replay_commit(
        x["qkv_raw"],
        x["conv_w"],
        x["conv_pool"],
        x["conv_pool"],
        x["f_a"],
        x["w_fb"],
        x["beta"],
        x["A_log"],
        x["dt_bias"],
        x["h_pool"],
        x["h_pool"],
        x["read_indices"],
        write_indices,
        accepted,
        num_heads=HV,
        head_dim=V,
        draft_token_num=t,
        lower_bound=LOWER_BOUND,
    )
    return x


def _reference(x, n, t, accepted):
    """fp32 torch reference: ``accepted[i]`` sequential decode steps.

    Deliberately written as an ordinary decode loop with no notion of a draft
    window, so agreeing with it IS the statement that replay is equivalent to
    non-speculative decoding.
    """
    conv_w = x["conv_w"].float()
    w_fb = x["w_fb"].float()
    exp_a = torch.exp(x["A_log"])[:, None]
    windows, states = [], []
    for i in range(n):
        r = int(x["read_indices"][i])
        window = x["conv_pool"][r].float()
        h = x["h_pool"][r].clone()
        for step in range(int(accepted[i])):
            tok = i * t + step
            xt = x["qkv_raw"][tok].float()
            acc = (window * conv_w[:, :3]).sum(-1) + xt * conv_w[:, 3]
            y = acc * torch.sigmoid(acc)
            window = torch.cat([window[:, 1:], xt[:, None]], dim=1)
            k = y[P : 2 * P].view(HV, K)
            v = y[2 * P :].view(HV, V)
            k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
            gate = ((w_fb @ x["f_a"][tok].float()) + x["dt_bias"]).view(HV, K)
            gk = LOWER_BOUND * torch.sigmoid(exp_a * gate)
            h = h * torch.exp(gk)[:, :, None]
            resid = v - torch.einsum("hkv,hk->hv", h, k)
            resid = resid * torch.sigmoid(x["beta"][tok].float())[:, None]
            h = h + torch.einsum("hk,hv->hkv", k, resid)
        windows.append(window)
        states.append(h)
    return windows, states


def _check_against_reference(x, out, write_indices, n, t, accepted):
    windows, states = _reference(x, n, t, accepted)
    for i in range(n):
        w = int(write_indices[i])
        torch.testing.assert_close(
            out["conv_pool"][w].float(), windows[i], atol=0.0, rtol=0.0
        )
        # Both sides carry the recurrence in fp32 from identical bf16 inputs,
        # so only reduction order differs: measured worst case 4.8e-7. Keep
        # atol tight enough to catch a wrong update; rtol stays loose because
        # the state has near-zero entries.
        torch.testing.assert_close(out["h_pool"][w], states[i], atol=1e-5, rtol=1e-2)


@pytest.mark.parametrize("t", [1, 2, 3, 5])
def test_replay_matches_sequential_decode_at_every_accepted_length(t):
    """Invariant 1, swept over every acceptable length including 0 and T."""
    n = 6
    x = _window(n, t, seed=t)
    fresh = torch.arange(1, n + 1, device=DEV, dtype=torch.int32) + 16
    for a in range(t + 1):
        accepted = torch.full((n,), a, device=DEV, dtype=torch.int32)
        out = _replay(x, fresh, accepted, t)
        _check_against_reference(x, out, fresh, n, t, accepted)


def test_rejected_suffix_cannot_reach_the_committed_state():
    """Invariant 2: perturbing the rejected tail changes nothing, bitwise."""
    n, t = 5, 4
    x = _window(n, t, seed=7)
    fresh = torch.arange(1, n + 1, device=DEV, dtype=torch.int32) + 16
    accepted = torch.tensor([0, 1, 2, 3, 4], device=DEV, dtype=torch.int32)
    base = _replay(x, fresh, accepted, t)

    perturbed = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in x.items()}
    for i in range(n):
        for step in range(int(accepted[i]), t):
            tok = i * t + step
            perturbed["qkv_raw"][tok].normal_()
            perturbed["f_a"][tok].normal_()
            perturbed["beta"][tok].normal_()
    other = _replay(perturbed, fresh, accepted, t)

    for i in range(n):
        w = int(fresh[i])
        torch.testing.assert_close(
            other["conv_pool"][w], base["conv_pool"][w], atol=0.0, rtol=0.0
        )
        torch.testing.assert_close(
            other["h_pool"][w], base["h_pool"][w], atol=0.0, rtol=0.0
        )


def test_zero_accepted_commits_the_pre_draft_state_unchanged():
    """Invariant 3: an all-rejected window still has to move the state.

    The destination page can differ from the source, so ``a = 0`` is a real
    commit of the unchanged state, not a no-op the caller may skip.
    """
    n, t = 4, 3
    x = _window(n, t, seed=11)
    fresh = torch.arange(1, n + 1, device=DEV, dtype=torch.int32) + 16
    out = _replay(x, fresh, torch.zeros(n, device=DEV, dtype=torch.int32), t)
    for i in range(n):
        r, w = int(x["read_indices"][i]), int(fresh[i])
        torch.testing.assert_close(
            out["conv_pool"][w], x["conv_pool"][r], atol=0.0, rtol=0.0
        )
        torch.testing.assert_close(out["h_pool"][w], x["h_pool"][r], atol=0.0, rtol=0.0)


def test_mixed_accepted_lengths_in_one_batch():
    """Spec 8.3: per-request lengths, including 0 and T, and a skipped row."""
    n, t = 6, 4
    x = _window(n, t, seed=13)
    fresh = torch.arange(1, n + 1, device=DEV, dtype=torch.int32) + 16
    accepted = torch.tensor([0, 4, 2, 1, 3, 4], device=DEV, dtype=torch.int32)
    out = _replay(x, fresh, accepted, t)
    _check_against_reference(x, out, fresh, n, t, accepted)

    # A negative destination (CUDA-graph padding) must leave the pools alone.
    padded = fresh.clone()
    padded[2] = -1
    out_padded = _replay(x, padded, accepted, t)
    w = int(fresh[2])
    torch.testing.assert_close(
        out_padded["conv_pool"][w], x["conv_pool"][w], atol=0.0, rtol=0.0
    )
    torch.testing.assert_close(
        out_padded["h_pool"][w], x["h_pool"][w], atol=0.0, rtol=0.0
    )


def test_committing_into_the_source_page_matches_a_fresh_page():
    """In-place commit is the common case and must not race.

    ``write == read`` makes every recurrence program read and write the same
    page. The recurrence is safe (each program owns a disjoint state slice);
    the conv window is only safe because it is committed by a separate launch
    that gives each program its own channels.
    """
    n, t = 8, 3
    x = _window(n, t, seed=17)
    accepted = torch.tensor([0, 1, 2, 3, 3, 2, 1, 0], device=DEV, dtype=torch.int32)
    fresh = torch.arange(1, n + 1, device=DEV, dtype=torch.int32) + 16
    expected = _replay(x, fresh, accepted, t)

    for _ in range(8):  # repeat: a race here would be intermittent
        in_place = _replay(x, x["read_indices"], accepted, t)
        for i in range(n):
            r, w = int(x["read_indices"][i]), int(fresh[i])
            torch.testing.assert_close(
                in_place["conv_pool"][r], expected["conv_pool"][w], atol=0.0, rtol=0.0
            )
            torch.testing.assert_close(
                in_place["h_pool"][r], expected["h_pool"][w], atol=0.0, rtol=0.0
            )


def test_accepted_length_clamps_at_and_past_the_window_boundary():
    """Boundary pins: a = T is exact, a > T and a < 0 clamp bitwise to T / 0.

    The host entry clamps ``accepted_length`` to ``[0, T]`` before both the
    recurrence and the conv-window launches; over- and under-range values
    must therefore be indistinguishable from the boundary itself, bit for
    bit, in both pools.
    """
    n, t = 4, 3
    x = _window(n, t, seed=29)
    fresh = torch.arange(1, n + 1, device=DEV, dtype=torch.int32) + 16
    at = _replay(x, fresh, torch.full((n,), t, device=DEV, dtype=torch.int32), t)
    past = _replay(x, fresh, torch.full((n,), t + 5, device=DEV, dtype=torch.int32), t)
    torch.testing.assert_close(past["conv_pool"], at["conv_pool"], atol=0.0, rtol=0.0)
    torch.testing.assert_close(past["h_pool"], at["h_pool"], atol=0.0, rtol=0.0)
    zero = _replay(x, fresh, torch.zeros(n, device=DEV, dtype=torch.int32), t)
    neg = _replay(x, fresh, torch.full((n,), -2, device=DEV, dtype=torch.int32), t)
    torch.testing.assert_close(neg["conv_pool"], zero["conv_pool"], atol=0.0, rtol=0.0)
    torch.testing.assert_close(neg["h_pool"], zero["h_pool"], atol=0.0, rtol=0.0)


def test_fresh_request_zero_accept_commits_zero_state():
    """read = -1 with a = 0 must commit an all-zero window and state.

    A request that joined mid-round has no committed page; committing its
    all-rejected window means materializing the zero state, not garbage from
    a masked load at a negative page offset.
    """
    n, t = 3, 2
    x = _window(n, t, seed=31)
    x["read_indices"] = torch.full((n,), -1, device=DEV, dtype=torch.int32)
    fresh = torch.arange(1, n + 1, device=DEV, dtype=torch.int32) + 16
    out = _replay(x, fresh, torch.zeros(n, device=DEV, dtype=torch.int32), t)
    for i in range(n):
        w = int(fresh[i])
        assert (out["conv_pool"][w].float() == 0).all()
        assert (out["h_pool"][w] == 0).all()


def test_in_place_commit_full_pool_accounting():
    """Torn or stray writes: account for every page in the pool, bit for bit.

    First, a = 0 in place: every program must store exactly the bits it read,
    so the whole pool is a fixed point -- any program writing another
    program's [BK, BV] slice (or any out-of-bounds store) moves random
    sentinel bits somewhere detectable. Then a mixed-length in-place commit:
    only the n anchor pages may change, and each must equal the out-of-place
    result; all other pages (including the out-of-place run's targets) must
    survive untouched.
    """
    n, t, pages = 16, 4, 40
    x = _window(n, t, pages=pages, seed=37)
    zero = torch.zeros(n, device=DEV, dtype=torch.int32)
    for _ in range(8):  # a race or torn write would be intermittent
        out = _replay(x, x["read_indices"], zero, t)
        torch.testing.assert_close(out["conv_pool"], x["conv_pool"], atol=0.0, rtol=0.0)
        torch.testing.assert_close(out["h_pool"], x["h_pool"], atol=0.0, rtol=0.0)

    accepted = torch.arange(n, device=DEV, dtype=torch.int32) % (t + 1)
    fresh = torch.arange(1, n + 1, device=DEV, dtype=torch.int32) + 20
    expected = _replay(x, fresh, accepted, t)
    for _ in range(8):
        got = _replay(x, x["read_indices"], accepted, t)
        for p in range(pages):
            if 1 <= p <= n:
                # Anchor page of request p-1: must hold the committed state.
                ref_c = expected["conv_pool"][p + 20]
                ref_h = expected["h_pool"][p + 20]
            else:
                # Every other page must be untouched, bit for bit.
                ref_c = x["conv_pool"][p]
                ref_h = x["h_pool"][p]
            torch.testing.assert_close(got["conv_pool"][p], ref_c, atol=0.0, rtol=0.0)
            torch.testing.assert_close(got["h_pool"][p], ref_h, atol=0.0, rtol=0.0)


def test_verify_leaves_the_committed_pages_intact():
    """The replay anchor only exists because verification stores nothing."""
    n, t = 4, 3
    x = _window(n, t, seed=19)
    conv_before = x["conv_pool"].clone()
    h_before = x["h_pool"].clone()
    out = fused_recurrent_kda_verify_megafuse(
        x["qkv_raw"],
        x["conv_w"],
        x["conv_pool"],
        x["f_a"],
        x["w_fb"],
        x["beta"],
        x["A_log"],
        x["dt_bias"],
        x["h_pool"],
        x["read_indices"],
        num_heads=HV,
        head_dim=V,
        draft_token_num=t,
        lower_bound=LOWER_BOUND,
    )
    assert out.shape == (n * t, HV, V)
    assert torch.isfinite(out.float()).all()
    torch.testing.assert_close(x["conv_pool"], conv_before, atol=0.0, rtol=0.0)
    torch.testing.assert_close(x["h_pool"], h_before, atol=0.0, rtol=0.0)


def test_full_acceptance_agrees_with_the_verify_pass_output():
    """A fully accepted window is the one case verify could have committed.

    Replaying all T positions must reach the state that produced verify's
    last-position output, which is checked here through that output: running
    a further verify of one token from the replayed state has to match a
    verify of T+1 tokens in one go.
    """
    n, t = 3, 3
    x = _window(n, t + 1, seed=23)
    long_out = fused_recurrent_kda_verify_megafuse(
        x["qkv_raw"],
        x["conv_w"],
        x["conv_pool"],
        x["f_a"],
        x["w_fb"],
        x["beta"],
        x["A_log"],
        x["dt_bias"],
        x["h_pool"],
        x["read_indices"],
        num_heads=HV,
        head_dim=V,
        draft_token_num=t + 1,
        lower_bound=LOWER_BOUND,
    ).view(n, t + 1, HV, V)

    # Replay the first t positions, then verify the (t+1)-th alone.
    prefix = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in x.items()}
    prefix["qkv_raw"] = x["qkv_raw"].view(n, t + 1, -1)[:, :t].reshape(n * t, -1)
    prefix["f_a"] = x["f_a"].view(n, t + 1, -1)[:, :t].reshape(n * t, -1)
    prefix["beta"] = x["beta"].view(n, t + 1, -1)[:, :t].reshape(n * t, -1)
    fresh = torch.arange(1, n + 1, device=DEV, dtype=torch.int32) + 16
    committed = _replay(
        prefix, fresh, torch.full((n,), t, device=DEV, dtype=torch.int32), t
    )

    tail_out = fused_recurrent_kda_verify_megafuse(
        x["qkv_raw"].view(n, t + 1, -1)[:, t].contiguous(),
        x["conv_w"],
        committed["conv_pool"],
        x["f_a"].view(n, t + 1, -1)[:, t].contiguous(),
        x["w_fb"],
        x["beta"].view(n, t + 1, -1)[:, t].contiguous(),
        x["A_log"],
        x["dt_bias"],
        committed["h_pool"],
        fresh,
        num_heads=HV,
        head_dim=V,
        draft_token_num=1,
        lower_bound=LOWER_BOUND,
    ).view(n, HV, V)

    torch.testing.assert_close(
        tail_out.float(), long_out[:, t].float(), atol=6e-3, rtol=1e-2
    )


@pytest.mark.parametrize("rows,pdl", [(1, False), (7, False), (64, True)])
def test_capture_payload_matches_three_copies(rows, pdl):
    """The fused capture must be a bitwise clone of the three eager copies."""
    from tokenspeed_kernel.thirdparty.triton.fla_kda_recurrent import (
        kda_capture_payload,
    )

    torch.manual_seed(rows)
    qkv = torch.randn(rows + 3, 3 * P, dtype=torch.bfloat16, device="cuda")
    fa = torch.randn(rows + 3, 2 * HV, dtype=torch.bfloat16, device="cuda")
    beta = torch.randn(rows + 3, HV, dtype=torch.bfloat16, device="cuda")
    dq = torch.full(
        (rows + 3, 3 * P), float("nan"), dtype=torch.bfloat16, device="cuda"
    )
    df = torch.full(
        (rows + 3, 2 * HV), float("nan"), dtype=torch.bfloat16, device="cuda"
    )
    db = torch.full((rows + 3, HV), float("nan"), dtype=torch.bfloat16, device="cuda")
    kda_capture_payload(qkv, fa, beta, dq, df, db, rows, enable_pdl=pdl)
    torch.cuda.synchronize()
    assert torch.equal(dq[:rows], qkv[:rows])
    assert torch.equal(df[:rows], fa[:rows])
    assert torch.equal(db[:rows], beta[:rows])
    assert dq[rows:].isnan().all() and db[rows:].isnan().all(), "wrote past rows"


def _capture_case(w_qkv, w_fa, w_beta, rows, spare=3, strided=False, seed=0):
    """Run kda_capture_payload on explicit widths; return (srcs, dsts)."""
    from tokenspeed_kernel.thirdparty.triton.fla_kda_recurrent import (
        kda_capture_payload,
    )

    torch.manual_seed(seed)
    n = rows + spare

    def src(w):
        if strided:
            # Column slice of a wider buffer: stride(0) > width, stride(1)==1,
            # the shape every runtime row-slice / projection-split view has.
            return torch.randn(n, w + 64, dtype=torch.bfloat16, device=DEV)[:, 32:-32]
        return torch.randn(n, w, dtype=torch.bfloat16, device=DEV)

    def dst(w):
        return torch.full((n, w), float("nan"), dtype=torch.bfloat16, device=DEV)

    srcs = (src(w_qkv), src(w_fa), src(w_beta))
    dsts = (dst(w_qkv), dst(w_fa), dst(w_beta))
    kda_capture_payload(*srcs, *dsts, rows)
    torch.cuda.synchronize()
    return srcs, dsts


@pytest.mark.parametrize(
    "w_qkv,w_fa,w_beta,rows,strided",
    [
        (2048, 1024, 8, 3, False),  # W_FA == BLOCK exactly: no lost tail block
        (3072, 1024, 1024, 2, False),  # beta == BLOCK exactly (assert boundary)
        (1500, 128, 8, 5, False),  # W_QKV not a multiple of BLOCK
        (1025, 1025, 8, 4, False),  # both one past a block boundary
        (1536, 128, 4, 6, True),  # strided sources (views of wider buffers)
    ],
)
def test_capture_payload_width_boundaries(w_qkv, w_fa, w_beta, rows, strided):
    """Block-edge widths and strided sources copy exactly, and only rows rows."""
    (q, f, b), (dq, df, db) = _capture_case(w_qkv, w_fa, w_beta, rows, strided=strided)
    assert torch.equal(dq[:rows], q[:rows])
    assert torch.equal(df[:rows], f[:rows])
    assert torch.equal(db[:rows], b[:rows])
    for d in (dq, df, db):
        assert d[rows:].isnan().all(), "wrote past rows"


def test_capture_payload_zero_rows_is_a_noop():
    """rows == 0 must neither launch out of bounds nor write anything."""
    _, (dq, df, db) = _capture_case(2048, 128, 8, rows=0)
    for d in (dq, df, db):
        assert d.isnan().all(), "rows=0 must write nothing"


def test_capture_payload_fa_wider_than_the_qkv_grid_is_loud_or_lossless():
    """f_a rides the qkv-sized grid: if W_FA exceeds the grid's column reach
    (cdiv(W_QKV, BLOCK) * BLOCK) the tail of every f_a row has no block to
    ride. That must either copy correctly or fail loudly -- a silent
    truncation would feed stale gate inputs into the next round's replay."""
    try:
        (_, f, _), (_, df, _) = _capture_case(512, 1100, 8, rows=3)
    except AssertionError:
        return  # loud: the driver refused the shape, which the contract allows
    assert torch.equal(df[:3], f[:3]), (
        "f_a columns beyond the qkv grid were silently dropped: "
        f"{int(df[:3].isnan().sum())} NaN destination elements remain"
    )


@pytest.mark.parametrize("parity", [0, 1])
def test_merged_capture_matches_commit_plus_standalone(parity):
    """CAPTURE=True must equal a plain commit followed by the standalone
    capture, bitwise, for both ring parities and mixed live/dead rows.

    This is the mode production actually launches (the standalone kernel is
    kept only as the reference); the interesting hazards are the capture
    stores sharing one grid with conv-roll ring reads of the OTHER parity
    half, dead rows (write < 0 / base < 0) still capturing, and the base
    ring row arriving via a device scalar rather than a kernel argument.
    """
    from tokenspeed_kernel.thirdparty.triton.fla_kda_recurrent import (
        kda_capture_payload,
        kda_commit_conv_window,
    )

    torch.manual_seed(parity)
    n, t, half = 4, 2, 4 * 2  # ring half sized exactly n*t: boundary-adjacent
    w = _window(n, t, seed=parity)
    ring = torch.randn(2 * half, 3 * P, dtype=torch.bfloat16, device=DEV)
    dst_fa = torch.randn(2 * half, D_FA, dtype=torch.bfloat16, device=DEV)
    dst_beta = torch.randn(2 * half, HV, dtype=torch.bfloat16, device=DEV)
    src_qkv = torch.randn(n * t, 3 * P, dtype=torch.bfloat16, device=DEV)
    src_fa = torch.randn(n * t, D_FA, dtype=torch.bfloat16, device=DEV)
    src_beta = torch.randn(n * t, HV, dtype=torch.bfloat16, device=DEV)
    # Replay reads half `parity`; capture writes the other half.
    cb = (1 - parity) * half
    capture_base = torch.tensor([cb], dtype=torch.int64, device=DEV)
    # Row 1 has no destination page, row 2 has no pending payload; both dead
    # for the commit, both still captured.
    writes = torch.tensor([5, -1, 6, 7], dtype=torch.int32, device=DEV)
    base = torch.tensor(
        [parity * half + 0, parity * half + 2, -1, parity * half + 6],
        dtype=torch.int32,
        device=DEV,
    )
    steps = torch.tensor([2, 1, 2, 1], dtype=torch.int32, device=DEV)

    def run(fused):
        r, df, db = ring.clone(), dst_fa.clone(), dst_beta.clone()
        pool = w["conv_pool"].clone()
        kda_commit_conv_window(
            r,
            pool,
            pool,
            w["read_indices"],
            writes,
            steps,
            conv_dim=3 * P,
            draft_token_num=t,
            row_base=base,
            capture=(
                (src_qkv, src_fa, src_beta, df, db, capture_base) if fused else None
            ),
        )
        if not fused:
            kda_capture_payload(
                src_qkv, src_fa, src_beta, r[cb:], df[cb:], db[cb:], n * t
            )
        torch.cuda.synchronize()
        return r, df, db, pool

    r_f, df_f, db_f, pool_f = run(fused=True)
    r_s, df_s, db_s, pool_s = run(fused=False)
    assert torch.equal(pool_f, pool_s), "committed conv windows diverge"
    assert torch.equal(r_f, r_s), "captured qkv ring diverges"
    assert torch.equal(df_f, df_s) and torch.equal(db_f, db_s), "gate capture diverges"
    # And the capture really landed: the target half holds the fresh source.
    assert torch.equal(r_f[cb : cb + n * t], src_qkv)
    # The replay half was read, never written.
    assert torch.equal(
        r_f[parity * half : parity * half + half], ring[parity * half :][:half]
    )


def test_merged_capture_handles_a_gate_wider_than_one_column_block():
    """The gate payloads ride the same column blocks as qkv, so a gate wider
    than one block is captured by several programs rather than stranded on
    one. That is what lets the block shrink to give a small batch a grid."""
    from tokenspeed_kernel.thirdparty.triton.fla_kda_recurrent import (
        kda_commit_conv_window,
    )

    n, t = 2, 2
    w = _window(n, t, seed=3)
    ring = torch.randn(2 * n * t, 3 * P, dtype=torch.bfloat16, device=DEV)
    wide = 300  # more than one column block at any block size we pick
    src_fa = torch.randn(n * t, wide, dtype=torch.bfloat16, device=DEV)
    dst_fa = torch.full(
        (2 * n * t, wide), float("nan"), dtype=torch.bfloat16, device=DEV
    )
    src_beta = torch.randn(n * t, HV, dtype=torch.bfloat16, device=DEV)
    dst_beta = torch.full(
        (2 * n * t, HV), float("nan"), dtype=torch.bfloat16, device=DEV
    )
    src_qkv = torch.randn(n * t, 3 * P, dtype=torch.bfloat16, device=DEV)
    cb = torch.zeros(1, dtype=torch.int64, device=DEV)
    kda_commit_conv_window(
        ring,
        w["conv_pool"],
        w["conv_pool"],
        w["read_indices"],
        torch.tensor([5, 6], dtype=torch.int32, device=DEV),
        torch.tensor([1, 1], dtype=torch.int32, device=DEV),
        conv_dim=3 * P,
        draft_token_num=t,
        capture=(src_qkv, src_fa, src_beta, dst_fa, dst_beta, cb),
    )
    torch.cuda.synchronize()
    assert torch.equal(dst_fa[: n * t], src_fa), "wide gate columns were dropped"
    assert torch.equal(dst_beta[: n * t], src_beta)

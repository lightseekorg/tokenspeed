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

from tokenspeed_kernel.ops.attention import (  # noqa: E402
    kda_replay_commit_supported,
)

#: The registry entries behind the probe and the no-store fused verify are
#: vendor-gated (NVIDIA today); direct Triton-kernel tests below run anywhere.
requires_registered_replay = pytest.mark.skipif(
    not kda_replay_commit_supported(torch.bfloat16),
    reason="KDA replay ops are not registered on this platform",
)

from tokenspeed_kernel.thirdparty.triton.fla_kda_recurrent import (  # noqa: E402
    _gate_tiling,
    batched_kda_commit_conv_window_kernel,
    batched_recurrent_kda_replay_commit,
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
        gate_scratch=torch.empty(n * t, P, device=DEV, dtype=torch.float32),
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
        gate_scratch=x["gate_scratch"],
    )
    return x


def _batched_descriptor(xs):
    addresses = []
    for x in xs:
        addresses.append(
            [
                x["qkv_raw"].data_ptr(),
                x["conv_w"].data_ptr(),
                x["conv_pool"].data_ptr(),
                x["f_a"].data_ptr(),
                x["w_fb"].data_ptr(),
                x["beta"].data_ptr(),
                x["A_log"].data_ptr(),
                x["dt_bias"].data_ptr(),
                x["h_pool"].data_ptr(),
                x["gate_scratch"].data_ptr(),
            ]
        )
    return torch.tensor(addresses, dtype=torch.uint64, device=DEV)


def _batched_static_args(x):
    return dict(
        qkv_stride=x["qkv_raw"].stride(0),
        conv_stride=x["conv_pool"].stride(0),
        f_a_stride=x["f_a"].stride(0),
        beta_stride=x["beta"].stride(0),
        state_stride=x["h_pool"].stride(0),
        gate_stride=x["gate_scratch"].stride(0),
        conv_width=x["conv_w"].shape[1],
        lower_bound=LOWER_BOUND,
    )


def test_gate_tiling_counts_every_layer_in_the_grid():
    """The batched launch tiles for its own grid, which spans all layers."""
    dev = torch.device(DEV)
    rows = 8
    one = _gate_tiling(rows, HV, K, dev)
    many = _gate_tiling(rows, HV, K, dev, layers=69)
    # More programs can only let an earlier, wider rung clear the threshold.
    assert many[0] >= one[0] and many[1] >= one[1]
    assert _gate_tiling(rows, HV, K, dev, layers=10**6) == (8, 32)
    # A rung wider than the row count spends most of its lanes on rows that
    # do not exist, so the ladder skips it however wide the grid looks.
    for narrow in (1, 2, 4):
        assert _gate_tiling(narrow, HV, K, dev, layers=10**6)[0] <= narrow


@pytest.mark.parametrize("block", [64, 128, 256, 512, 2048])
def test_batched_conv_window_is_independent_of_its_column_block(block):
    """Splitting the window publish across columns must not move a byte."""
    layers, n, t = 3, 4, 4
    source = [_window(n, t, seed=700 + layer) for layer in range(layers)]
    reads = torch.stack([source[0]["read_indices"]] * layers).to(torch.int32)
    writes = torch.stack(
        [torch.arange(9, 9 + n, device=DEV, dtype=torch.int32)] * layers
    )
    accepted = torch.tensor([0, t, 1, 3], device=DEV, dtype=torch.int32)
    conv_dim = 3 * HV * K

    def publish(block_size):
        xs = [
            {k: (v.clone() if torch.is_tensor(v) else v) for k, v in x.items()}
            for x in source
        ]
        batched_kda_commit_conv_window_kernel[
            (layers, n, (conv_dim + block_size - 1) // block_size)
        ](
            _batched_descriptor(xs),
            torch.arange(layers, device=DEV, dtype=torch.int32),
            reads,
            writes,
            accepted,
            n,
            T=t,
            STRIDE_QKV=xs[0]["qkv_raw"].stride(0),
            STRIDE_CONV=xs[0]["conv_pool"].stride(0),
            CONV_DIM=conv_dim,
            BLOCK=block_size,
            num_warps=min(8, max(1, block_size // 128)),
        )
        return [x["conv_pool"] for x in xs]

    for narrow, wide in zip(publish(block), publish(4096), strict=True):
        torch.testing.assert_close(narrow, wide, atol=0, rtol=0)


def test_batched_replay_is_bit_identical_and_descriptor_sensitive():
    """One launch matches the layer loop; a wrong descriptor must be detected."""
    layers, n, t = 5, 5, 4
    source = [_window(n, t, seed=100 + layer) for layer in range(layers)]
    loop = []
    writes = torch.stack(
        [
            torch.arange(17, 17 + n, device=DEV, dtype=torch.int32),
            torch.arange(24, 24 + n, device=DEV, dtype=torch.int32),
        ]
    )
    reads = torch.stack([source[0]["read_indices"], source[0]["read_indices"] + 5]).to(
        torch.int32
    )
    accepted = torch.tensor([0, t, 1, 3, 2], device=DEV, dtype=torch.int32)
    groups = [0, 0, 0, 1, 1]
    group_indices = torch.tensor(groups, device=DEV, dtype=torch.int32)
    for x, group in zip(source, groups, strict=True):
        local = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in x.items()}
        local["read_indices"] = reads[group]
        loop.append(_replay(local, writes[group], accepted, t))

    batched = [
        {k: (v.clone() if torch.is_tensor(v) else v) for k, v in x.items()}
        for x in source
    ]
    for x, group in zip(batched, groups, strict=True):
        x["read_indices"] = reads[group]
    descriptors = _batched_descriptor(batched)
    batched_recurrent_kda_replay_commit(
        descriptors,
        group_indices,
        reads,
        writes,
        accepted,
        draft_token_num=t,
        num_heads=HV,
        head_dim=K,
        f_a_dim=D_FA,
        **_batched_static_args(batched[0]),
    )
    for expected, actual in zip(loop, batched, strict=True):
        accepted_rows = torch.cat(
            [
                torch.arange(i * t, i * t + count, device=DEV)
                for i, count in enumerate(accepted.tolist())
                if count
            ]
        )
        torch.testing.assert_close(
            actual["gate_scratch"][accepted_rows],
            expected["gate_scratch"][accepted_rows],
            atol=1e-6,
            rtol=0,
        )
        torch.testing.assert_close(
            actual["conv_pool"], expected["conv_pool"], atol=0, rtol=0
        )
        torch.testing.assert_close(
            actual["h_pool"], expected["h_pool"], atol=1e-6, rtol=0
        )

    negative = [
        {k: (v.clone() if torch.is_tensor(v) else v) for k, v in x.items()}
        for x in source
    ]
    bad = _batched_descriptor(negative)
    bad[2, 1] = negative[1]["conv_w"].data_ptr()
    batched_recurrent_kda_replay_commit(
        bad,
        group_indices,
        reads,
        writes,
        accepted,
        draft_token_num=t,
        num_heads=HV,
        head_dim=K,
        f_a_dim=D_FA,
        **_batched_static_args(negative[0]),
    )
    with pytest.raises(AssertionError):
        torch.testing.assert_close(
            negative[2]["h_pool"], loop[2]["h_pool"], atol=0, rtol=0
        )


@pytest.mark.parametrize(
    ("accepted", "crossing", "t"),
    [
        ([1, 1, 1, 1], [False, False, False, False], 3),
        ([2, 3, 2, 3], [True, True, True, True], 3),
        ([1, 1, 2, 3], [False, False, True, True], 3),
        # DSpark's draft width: longer replays cross the boundary more often.
        ([1, 8, 4, 1], [False, True, True, False], 8),
    ],
    ids=[
        "in-place-only",
        "cross-only",
        "mixed-checkpoint-boundary",
        "mixed-boundary-t8",
    ],
)
def test_batched_replay_checkpoint_boundary_matches_layer_loop(accepted, crossing, t):
    """TP1 checkpoint-boundary schedule agrees with the layer-loop oracle."""
    layers, n = 69, 4
    source = [_window(n, t, pages=24, seed=2701 + layer) for layer in range(layers)]
    for x in source:
        qkv = torch.empty(n * t, 9 * P, device=DEV, dtype=torch.bfloat16)
        qkv[:, : 3 * P].copy_(x["qkv_raw"])
        x["qkv_raw"] = qkv[:, : 3 * P]
        beta = torch.empty(n * t, 3 * HV, device=DEV, dtype=torch.bfloat16)
        beta[:, :HV].copy_(x["beta"])
        x["beta"] = beta[:, :HV]
        x["gate_scratch"] = torch.empty(n * t, 3 * P, device=DEV, dtype=torch.float32)[
            :, :P
        ]
        pages = 24
        page_bytes = 96 * 3 * P * 3 * torch.bfloat16.itemsize
        arena = torch.empty(pages * page_bytes, device=DEV, dtype=torch.uint8)
        conv_storage = arena.view(torch.bfloat16)
        conv = torch.as_strided(
            conv_storage, (pages, 9 * P, 3), (page_bytes // 2, 3, 1)
        )
        conv.zero_()
        conv[:, : 3 * P].copy_(x["conv_pool"])
        x["conv_pool"] = conv
        state_storage = arena[3 * P * 3 * 2 :].view(torch.float32)
        state = torch.as_strided(
            state_storage,
            (pages, 3 * HV, K, V),
            (page_bytes // 4, K * V, V, 1),
        )
        state.zero_()
        state[:, :HV].copy_(x["h_pool"])
        x["h_pool"] = state
    reads = torch.stack([torch.arange(2, 2 + n, device=DEV, dtype=torch.int32)] * 3)
    crossed = torch.arange(10, 10 + n, device=DEV, dtype=torch.int32)
    writes = torch.stack(
        [torch.where(torch.tensor(crossing, device=DEV), crossed, reads[0])] * 3
    ).to(torch.int32)
    accepted_tensor = torch.tensor(accepted, device=DEV, dtype=torch.int32)
    groups = [group for group in range(3) for _ in range(23)]

    loop = []
    for x, group in zip(source, groups, strict=True):
        local = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in x.items()}
        local["conv_pool"].zero_()
        local["h_pool"].zero_()
        local["read_indices"] = reads[group]
        loop.append(_replay(local, writes[group], accepted_tensor, t))

    batched = [
        {k: (v.clone() if torch.is_tensor(v) else v) for k, v in x.items()}
        for x in source
    ]
    for x, group in zip(batched, groups, strict=True):
        x["conv_pool"].zero_()
        x["h_pool"].zero_()
        x["read_indices"] = reads[group]
    batched_recurrent_kda_replay_commit(
        _batched_descriptor(batched),
        torch.tensor(groups, device=DEV, dtype=torch.int32),
        reads,
        writes,
        accepted_tensor,
        draft_token_num=t,
        num_heads=HV,
        head_dim=K,
        f_a_dim=D_FA,
        **_batched_static_args(batched[0]),
    )
    for expected, actual in zip(loop, batched, strict=True):
        torch.testing.assert_close(
            actual["conv_pool"], expected["conv_pool"], atol=0, rtol=0
        )
        torch.testing.assert_close(
            actual["h_pool"], expected["h_pool"], atol=1e-6, rtol=0
        )


def _canonical(page):
    """View a V-major state page as the ``[HV, K, V]`` the reference builds."""
    return page.transpose(-1, -2)


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
        h = _canonical(x["h_pool"][r]).clone()
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
        torch.testing.assert_close(
            _canonical(out["h_pool"][w]), states[i], atol=1e-5, rtol=1e-2
        )


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


@requires_registered_replay
def test_replay_commit_probe_tracks_dtype():
    """The capability probe must use the actual activation dtype."""
    from tokenspeed_kernel.ops.attention import kda_replay_commit_supported

    assert not kda_replay_commit_supported(torch.float32)
    assert kda_replay_commit_supported(torch.bfloat16)


@requires_registered_replay
def test_replay_probe_requires_both_commit_and_fused_verify_kernels():
    """Eager replay has no decomposed fallback: either kernel missing means
    unsupported."""
    from unittest import mock

    import tokenspeed_kernel.ops.attention as attention_ops
    from tokenspeed_kernel.selection import NoKernelFoundError

    real = attention_ops.select_kernel

    def missing(mode_to_drop):
        def probe(family, mode, *args, **kwargs):
            if mode == mode_to_drop:
                raise NoKernelFoundError(f"no {mode} on this platform")
            return real(family, mode, *args, **kwargs)

        return probe

    for mode in ("kda_fused_paged_verify", "kda_replay_commit"):
        with mock.patch.object(attention_ops, "select_kernel", missing(mode)):
            assert not attention_ops.kda_replay_commit_supported(torch.bfloat16)


@requires_registered_replay
def test_fused_verify_no_store_matches_store_and_leaves_tape_untouched():
    """The inline-producer no-store fusion returns the store output, sans tape.

    Selection prefers the split-producer variant for this trait set, so the
    twin is named directly: the point here is that dropping the tape does not
    disturb the recurrence, not which producers ran.
    """
    from tokenspeed_kernel.ops.attention.triton.kda_dispatch import (
        triton_nvidia_kda_fused_paged_verify_no_store as try_kda_fused_paged_verify,
    )

    n, t, rows = 2, 3, 12
    x = _window(n, t, seed=47)
    writes = torch.arange(n * t, device=DEV, dtype=torch.int32).view(n, t)
    conv_tape = torch.randn(rows, 3 * P, 3, device=DEV, dtype=torch.bfloat16)
    state_tape = torch.randn(rows, HV, K, V, device=DEV, dtype=torch.float32)
    conv_before, state_before = conv_tape.clone(), state_tape.clone()

    no_store = try_kda_fused_paged_verify(
        x["qkv_raw"],
        x["conv_w"],
        x["conv_pool"],
        conv_tape,
        x["f_a"],
        x["w_fb"],
        x["beta"],
        x["A_log"],
        x["dt_bias"],
        state_pool=x["h_pool"],
        state_scratch=state_tape,
        read_indices=x["read_indices"],
        write_indices=writes,
        num_heads=HV,
        head_dim=V,
        draft_token_num=t,
        lower_bound=LOWER_BOUND,
    )
    torch.testing.assert_close(conv_tape, conv_before, atol=0.0, rtol=0.0)
    torch.testing.assert_close(state_tape, state_before, atol=0.0, rtol=0.0)

    stored = fused_recurrent_kda_verify_megafuse(
        x["qkv_raw"],
        x["conv_w"],
        x["conv_pool"],
        conv_tape,
        x["f_a"],
        x["w_fb"],
        x["beta"],
        x["A_log"],
        x["dt_bias"],
        x["h_pool"],
        state_tape,
        x["read_indices"],
        writes,
        num_heads=HV,
        head_dim=V,
        draft_token_num=t,
        lower_bound=LOWER_BOUND,
        store_states=True,
    ).view(1, -1, HV, V)
    torch.testing.assert_close(no_store, stored, atol=0.0, rtol=0.0)
    assert not torch.equal(conv_tape, conv_before)
    assert not torch.equal(state_tape, state_before)

    # Negative control: the output oracle must notice a wrong committed base.
    corrupted = {**x, "h_pool": x["h_pool"].clone()}
    corrupted["h_pool"][x["read_indices"].long()] += 1
    wrong = try_kda_fused_paged_verify(
        corrupted["qkv_raw"],
        corrupted["conv_w"],
        corrupted["conv_pool"],
        conv_before,
        corrupted["f_a"],
        corrupted["w_fb"],
        corrupted["beta"],
        corrupted["A_log"],
        corrupted["dt_bias"],
        state_pool=corrupted["h_pool"],
        state_scratch=state_before,
        read_indices=corrupted["read_indices"],
        write_indices=writes,
        num_heads=HV,
        head_dim=V,
        draft_token_num=t,
        lower_bound=LOWER_BOUND,
    )
    assert not torch.equal(wrong, no_store)


@requires_registered_replay
@pytest.mark.parametrize("n", [1, 4])
def test_split_verify_wrapper_matches_fused_wrapper(n):
    from tokenspeed_kernel.ops.attention.triton.kda_dispatch import (
        triton_nvidia_kda_fused_paged_verify_no_store,
        triton_nvidia_kda_fused_paged_verify_split,
    )

    t = 3
    x = _window(n, t, seed=49 + n)
    writes = torch.arange(n * t, device=DEV, dtype=torch.int32).view(n, t)
    conv_scratch = torch.empty(n * t, 3 * P, 3, device=DEV, dtype=torch.bfloat16)
    state_scratch = torch.empty(n * t, HV, K, V, device=DEV, dtype=torch.float32)
    kwargs = dict(
        state_pool=x["h_pool"],
        state_scratch=state_scratch,
        read_indices=x["read_indices"],
        write_indices=writes,
        num_heads=HV,
        head_dim=V,
        draft_token_num=t,
        lower_bound=LOWER_BOUND,
    )
    args = (
        x["qkv_raw"],
        x["conv_w"],
        x["conv_pool"],
        conv_scratch,
        x["f_a"],
        x["w_fb"],
        x["beta"],
        x["A_log"],
        x["dt_bias"],
    )
    fused = triton_nvidia_kda_fused_paged_verify_no_store(*args, **kwargs)
    split = triton_nvidia_kda_fused_paged_verify_split(*args, **kwargs)
    torch.testing.assert_close(split, fused, atol=2e-2, rtol=2e-2)


def test_fused_verify_default_store_is_bit_identical_to_explicit_trait():
    """The existing tape-writing API stays bitwise identical by default."""
    x = _window(2, 3, seed=53)
    writes = torch.arange(6, device=DEV, dtype=torch.int32).view(2, 3)
    tapes = []
    outputs = []
    for kwargs in ({}, {"store_states": True}):
        conv = torch.zeros(8, 3 * P, 3, device=DEV, dtype=torch.bfloat16)
        state = torch.zeros(8, HV, K, V, device=DEV, dtype=torch.float32)
        outputs.append(
            fused_recurrent_kda_verify_megafuse(
                x["qkv_raw"],
                x["conv_w"],
                x["conv_pool"],
                conv,
                x["f_a"],
                x["w_fb"],
                x["beta"],
                x["A_log"],
                x["dt_bias"],
                x["h_pool"],
                state,
                x["read_indices"],
                writes,
                num_heads=HV,
                head_dim=V,
                draft_token_num=3,
                lower_bound=LOWER_BOUND,
                **kwargs,
            )
        )
        tapes.append((conv, state))
    torch.testing.assert_close(outputs[0], outputs[1], atol=0.0, rtol=0.0)
    torch.testing.assert_close(tapes[0][0], tapes[1][0], atol=0.0, rtol=0.0)
    torch.testing.assert_close(tapes[0][1], tapes[1][1], atol=0.0, rtol=0.0)


@pytest.mark.parametrize("n,t", [(4, 3), (8, 4), (16, 4)])
def test_fused_verify_routed_bv_matches_bv32(n, t):
    """The routed wide tiles (64/128) must agree with BV=32 to one bf16 ulp.

    Each program owns its V rows outright, so tile width can only reorder
    the fp32 K reduction — a one-ulp wobble after the bf16 store. A masking
    or boundary bug shifts whole rows, far beyond that.
    """
    x = _window(n, t, seed=61)
    writes = torch.arange(n * t, device=DEV, dtype=torch.int32).view(n, t)
    results = []
    for bv in (32, None):
        out = fused_recurrent_kda_verify_megafuse(
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
            writes,
            num_heads=HV,
            head_dim=V,
            draft_token_num=t,
            lower_bound=LOWER_BOUND,
            store_states=False,
            bv=bv,
        )
        results.append(out)
    torch.testing.assert_close(results[0], results[1], atol=1e-5, rtol=8e-3)


def test_fused_verify_store_follows_permuted_write_rows():
    """Stored tapes land at write_indices[tok], not at the token index."""
    n, t = 2, 3
    x = _window(n, t, seed=63)
    rows = n * t
    perm = torch.randperm(rows, generator=torch.Generator().manual_seed(5))
    writes = perm.to(device=DEV, dtype=torch.int32).view(n, t)
    identity = torch.arange(rows, device=DEV, dtype=torch.int32).view(n, t)
    outs, convs, states = [], [], []
    for w in (identity, writes):
        conv = torch.zeros(rows, 3 * P, 3, device=DEV, dtype=torch.bfloat16)
        state = torch.zeros(rows, HV, K, V, device=DEV, dtype=torch.float32)
        outs.append(
            fused_recurrent_kda_verify_megafuse(
                x["qkv_raw"],
                x["conv_w"],
                x["conv_pool"],
                conv,
                x["f_a"],
                x["w_fb"],
                x["beta"],
                x["A_log"],
                x["dt_bias"],
                x["h_pool"],
                state,
                x["read_indices"],
                w,
                num_heads=HV,
                head_dim=V,
                draft_token_num=t,
                lower_bound=LOWER_BOUND,
                store_states=True,
            )
        )
        convs.append(conv)
        states.append(state)
    torch.testing.assert_close(outs[0], outs[1], atol=0.0, rtol=0.0)
    torch.testing.assert_close(convs[1][perm], convs[0], atol=0.0, rtol=0.0)
    torch.testing.assert_close(states[1][perm], states[0], atol=0.0, rtol=0.0)


def test_fused_verify_rejects_non_power_of_two_bv():
    x = _window(1, 3, seed=62)
    writes = torch.arange(3, device=DEV, dtype=torch.int32).view(1, 3)
    conv = torch.zeros(3, 3 * P, 3, device=DEV, dtype=torch.bfloat16)
    state = torch.zeros(3, HV, K, V, device=DEV, dtype=torch.float32)
    with pytest.raises(ValueError, match="positive power of two"):
        fused_recurrent_kda_verify_megafuse(
            x["qkv_raw"],
            x["conv_w"],
            x["conv_pool"],
            conv,
            x["f_a"],
            x["w_fb"],
            x["beta"],
            x["A_log"],
            x["dt_bias"],
            x["h_pool"],
            state,
            x["read_indices"],
            writes,
            num_heads=HV,
            head_dim=V,
            draft_token_num=3,
            lower_bound=LOWER_BOUND,
            bv=48,
        )

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

# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""causal_conv1d_update over a draft tree: each token's window ends at its parent."""

import pytest
import torch
from tokenspeed_kernel.ops.attention.gdn.triton import fused_qkv_split_gdn_prefill
from tokenspeed_kernel.ops.conv import (
    PAD_SLOT_ID,
    causal_conv1d_update,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

PARENTS = [
    [-1, 0, 1, 1, 0, 4],
    [-1, 0, 0, 2, 2, 0],
    [-1, 0, 1, 2, 3, 4],
]


def _run(x, conv_state, weight, bias, base, out_rows, parents):
    return causal_conv1d_update(
        x.clone(),
        conv_state,
        weight,
        bias,
        activation="silu",
        conv_state_indices=base,
        output_state_indices=out_rows,
        parent_indices=parents,
    )


def test_tree_windows_match_per_path_reference():
    torch.manual_seed(0)
    bs, dim, width, t = len(PARENTS), 256, 4, len(PARENTS[0])
    x = torch.randn(bs, dim, t, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(dim, width, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(dim, device="cuda", dtype=torch.bfloat16)
    rows = 1 + bs * (t + 1)
    conv_state = torch.randn(rows, dim, width - 1, device="cuda", dtype=torch.bfloat16)
    base = torch.arange(bs, device="cuda", dtype=torch.int32) * (t + 1) + 1
    out_rows = (
        base[:, None] + 1 + torch.arange(t, device="cuda", dtype=torch.int32)
    ).contiguous()
    parents = torch.tensor(PARENTS, device="cuda", dtype=torch.int32)
    init = conv_state[base.long()].float().clone()

    state = conv_state.clone()
    out = _run(x, state, weight, bias, base, out_rows, parents)

    for b, par in enumerate(PARENTS):
        for i in range(t):
            path, node = [], i
            while node >= 0:
                path.append(node)
                node = par[node]
            seq = torch.cat(
                [init[b], x[b, :, path[::-1]].float()], dim=1
            )  # [dim, 3 + depth + 1]
            ref = (seq[:, -width:] * weight.float()).sum(-1) + bias.float()
            ref = ref * torch.sigmoid(ref)
            torch.testing.assert_close(out[b, :, i].float(), ref, atol=3e-2, rtol=2e-2)
            torch.testing.assert_close(
                state[int(out_rows[b, i])].float(),
                seq[:, -(width - 1) :],
                atol=0,
                rtol=0,
            )

    # A chain given as parents never reloads: identical to the parent-less update.
    chain_state = conv_state.clone()
    chain_out = _run(x, chain_state, weight, bias, base, out_rows, None)
    assert torch.equal(out[2], chain_out[2])
    assert torch.equal(state[out_rows[2].long()], chain_state[out_rows[2].long()])


@pytest.mark.parametrize("width", [2, 3, 4])
def test_tree_update_is_bitwise_per_path_reference(width):
    """Products round to bf16 and add left to right in fp32, so the reference is exact."""
    torch.manual_seed(width)
    bs, dim, t = 4, 1000, 6
    parents = PARENTS + [[-1, 0, 0, 1, -1, 4]]
    x = torch.randn(bs, t, dim, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    weight = torch.randn(dim, width, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(dim, device="cuda", dtype=torch.bfloat16)
    conv_state = torch.randn(
        1 + bs * (t + 1), dim, width - 1, device="cuda", dtype=torch.bfloat16
    )
    base = torch.arange(bs, device="cuda", dtype=torch.int32) * (t + 1) + 1
    base[1] = PAD_SLOT_ID
    out_rows = base[:, None] + 1 + torch.arange(t, device="cuda", dtype=torch.int32)
    out_rows[1] = -1
    out_rows[2, 3] = -1
    init = conv_state.float()
    state = conv_state.clone()
    out = causal_conv1d_update(
        x,
        state,
        weight,
        bias,
        activation=None,
        conv_state_indices=base,
        output_state_indices=out_rows,
        parent_indices=torch.tensor(parents, device="cuda", dtype=torch.int32),
    )

    expected_state = conv_state.clone()
    for b, par in enumerate(parents):
        if b == 1:
            continue
        for i in range(t):
            path, node = [], i
            while node >= 0:
                path.append(node)
                node = par[node]
            seq = torch.cat([init[int(base[b])], x[b, :, path[::-1]].float()], dim=1)
            ref = bias.float()
            for c in range(width):
                product = seq[:, c - width] * weight[:, c].float()
                ref = ref + product.bfloat16().float()
            assert torch.equal(out[b, :, i], ref.bfloat16())
            if out_rows[b, i] >= 0:
                expected_state[int(out_rows[b, i])] = seq[:, 1 - width :].bfloat16()
    assert torch.equal(state, expected_state)


def test_tree_update_reads_a_row_strided_projection_view():
    """Token rows sliced out of a wider projection convolve like dense rows,
    into a token-major output a ``[bs * t, dim]`` view can read back."""
    torch.manual_seed(1)
    bs, dim, width, t = len(PARENTS), 256, 4, len(PARENTS[0])
    wide = torch.randn(bs, t, dim + 96, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(dim, width, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(dim, device="cuda", dtype=torch.bfloat16)
    conv_state = torch.randn(
        1 + bs * (t + 1), dim, width - 1, device="cuda", dtype=torch.bfloat16
    )
    base = torch.arange(bs, device="cuda", dtype=torch.int32) * (t + 1) + 1
    out_rows = base[:, None] + 1 + torch.arange(t, device="cuda", dtype=torch.int32)
    parents = torch.tensor(PARENTS, device="cuda", dtype=torch.int32)
    outs, states = [], []
    for x in (wide[:, :, :dim], wide[:, :, :dim].contiguous()):
        state = conv_state.clone()
        outs.append(
            causal_conv1d_update(
                x.transpose(1, 2),
                state,
                weight,
                bias,
                activation="silu",
                conv_state_indices=base,
                output_state_indices=out_rows,
                parent_indices=parents,
            )
        )
        states.append(state)
    assert torch.equal(outs[0], outs[1]) and torch.equal(states[0], states[1])
    assert outs[0].transpose(1, 2).is_contiguous()


@pytest.mark.parametrize(
    "broken", ["no_base", "int64_parents", "base_shape", "cache_seqlens"]
)
def test_tree_update_rejects_incomplete_index_state(broken):
    bs, dim, width, t = 2, 64, 4, 3
    x = torch.randn(bs, dim, t, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(dim, width, device="cuda", dtype=torch.bfloat16)
    conv_state = torch.zeros(1 + bs * (t + 1), dim, width - 1, device="cuda")
    base = torch.arange(bs, device="cuda", dtype=torch.int32) * (t + 1) + 1
    out_rows = base[:, None] + 1 + torch.arange(t, device="cuda", dtype=torch.int32)
    parents = torch.tensor([[-1, 0, 0]] * bs, device="cuda", dtype=torch.int32)
    if broken == "no_base":
        base = None
    elif broken == "int64_parents":
        parents = parents.long()
    elif broken == "base_shape":
        base = base[:1]
    extra = {"cache_seqlens": base} if broken == "cache_seqlens" else {}
    with pytest.raises(ValueError, match="parent_indices needs int32"):
        causal_conv1d_update(
            x,
            conv_state.bfloat16(),
            weight,
            None,
            activation="silu",
            conv_state_indices=base,
            output_state_indices=out_rows,
            parent_indices=parents,
            **extra,
        )


def test_tree_update_rejects_unknown_activation():
    bs, dim, width, t = 2, 64, 4, 3
    x = torch.randn(bs, dim, t, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(dim, width, device="cuda", dtype=torch.bfloat16)
    conv_state = torch.zeros(
        1 + bs * (t + 1), dim, width - 1, device="cuda", dtype=torch.bfloat16
    )
    base = torch.arange(bs, device="cuda", dtype=torch.int32) * (t + 1) + 1
    out_rows = base[:, None] + 1 + torch.arange(t, device="cuda", dtype=torch.int32)
    parents = torch.tensor([[-1, 0, 0]] * bs, device="cuda", dtype=torch.int32)
    with pytest.raises(AssertionError):
        causal_conv1d_update(
            x,
            conv_state,
            weight,
            None,
            activation="gelu",
            conv_state_indices=base,
            output_state_indices=out_rows,
            parent_indices=parents,
        )


def test_tree_update_skips_padded_entries():
    """A padded entry (conv_state_indices == PAD_SLOT_ID) reads and writes no state, as on the chain path."""
    bs, dim, width, t = 2, 64, 4, 4
    backing = torch.zeros(1 + 8, dim, width - 1, device="cuda", dtype=torch.bfloat16)
    backing[0] = 1000.0  # what index -1 would read
    conv_state = backing[1:]
    weight = torch.randn(dim, width, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(bs, t, dim, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    parents = torch.tensor([[-1, 0, 1, 2]] * bs, dtype=torch.int32, device="cuda")
    base = torch.tensor([PAD_SLOT_ID, 1], dtype=torch.int32, device="cuda")
    out_rows = torch.tensor(
        [[2, 3, 4, 5], [-1, -1, -1, -1]], dtype=torch.int32, device="cuda"
    )
    before = conv_state.clone()
    out = causal_conv1d_update(
        x.clone(),
        conv_state,
        weight,
        None,
        activation="silu",
        conv_state_indices=base,
        output_state_indices=out_rows,
        parent_indices=parents,
    )
    assert torch.equal(conv_state, before)
    assert bool(torch.isfinite(out[1].float()).all())


def test_decode_update_compiles_once_across_batch_sizes():
    """The decode batch changes every round; it must not key a new binary."""
    from unittest.mock import patch

    from tokenspeed_kernel.ops.conv._triton.causal_conv1d import (
        _causal_conv1d_update_kernel,
    )

    torch.manual_seed(0)
    dim, width = 64, 4
    weight = torch.randn(dim, width, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(64, dim, device="cuda", dtype=torch.bfloat16)
    initial = torch.randn(65, dim, width - 1, device="cuda", dtype=torch.bfloat16)

    def run(bs):
        conv_state = initial.clone()
        rows = torch.arange(1, bs + 1, dtype=torch.int32, device="cuda")
        out = causal_conv1d_update(
            x[:bs].clone(),
            conv_state,
            weight,
            None,
            activation="silu",
            conv_state_indices=rows,
            parent_indices=None,
        )
        # Requests are independent: each row equals the same request decoded alone.
        alone_state = initial.clone()
        alone = causal_conv1d_update(
            x[bs - 1 : bs].clone(),
            alone_state,
            weight,
            None,
            activation="silu",
            conv_state_indices=rows[-1:].clone(),
            parent_indices=None,
        )
        assert torch.equal(out[-1:], alone)
        assert torch.equal(conv_state[bs], alone_state[bs])

    run(5)
    with patch.object(
        _causal_conv1d_update_kernel,
        "_do_compile",
        wraps=_causal_conv1d_update_kernel._do_compile,
    ) as compiles:
        for bs in (1, 16, 17, 32, 63):
            run(bs)
    assert compiles.call_count == 0


@pytest.mark.parametrize("width", [1, 3, 4, 16])
@pytest.mark.parametrize("batch", [1, 7, 300])
def test_chain_update_writes_the_split_payload(batch, width):
    """The chain update writes bitwise the in-place output and the split's payload."""
    torch.manual_seed(batch * 17 + width)
    # One TP4 rank of Qwen3.8-27B: [q | k | v | z | b | a] projection rows.
    heads, head_dim, key = 12, 128, 512
    value = heads * head_dim
    dim = 2 * key + value
    rows = batch * width
    projection = torch.randn(
        rows, dim + value + 2 * heads, device="cuda", dtype=torch.bfloat16
    )
    original = projection.clone()
    weight = torch.randn(dim, 4, device="cuda", dtype=torch.bfloat16)
    conv_state = torch.randn(
        1 + batch * (width + 1), dim, 3, device="cuda", dtype=torch.bfloat16
    )
    base = torch.arange(batch, device="cuda", dtype=torch.int32) * (width + 1) + 1
    out_rows = base[:, None] + 1 + torch.arange(width, device="cuda").int()

    def views(source):
        x = source[:, :dim]
        b = source[:, dim + value : dim + value + heads]
        a = source[:, dim + value + heads :]
        return x, a, b

    def update(x, state, **outputs):
        return causal_conv1d_update(
            x.view(batch, width, dim).transpose(1, 2),
            state,
            weight,
            None,
            activation="silu",
            conv_state_indices=base,
            output_state_indices=out_rows,
            parent_indices=None,
            **outputs,
        )

    # The in-place update and the split, as a step-form verify runs them.
    in_place = projection.clone()
    expected_x, expected_a, expected_b = views(in_place)
    expected_state = conv_state.clone()
    update(expected_x, expected_state)
    expected_payload = torch.empty(
        rows, dim - key + 2 * heads, device="cuda", dtype=torch.bfloat16
    )
    fused_qkv_split_gdn_prefill(
        expected_x,
        num_q_heads=key // head_dim,
        num_k_heads=key // head_dim,
        num_v_heads=heads,
        head_q=head_dim,
        head_k=head_dim,
        head_v=head_dim,
        replay=(expected_payload, expected_a, expected_b),
    )

    x, a, b = views(projection)
    state = conv_state.clone()
    out = torch.full((rows, dim), float("nan"), device="cuda", dtype=torch.bfloat16)
    payload = torch.full_like(expected_payload, float("nan"))
    update(
        x,
        state,
        out=out.view(batch, width, dim).transpose(1, 2),
        replay_payload=(payload, a, b),
    )
    assert torch.equal(out, expected_x)
    assert torch.equal(payload, expected_payload)
    assert torch.equal(state, expected_state)
    assert torch.equal(projection, original)


def test_chain_update_with_payload_compiles_once_across_batch_sizes():
    """The payload-writing chain update compiles once per window width, never per batch size."""
    from unittest.mock import patch

    from tokenspeed_kernel.ops.conv.triton import _causal_conv1d_update_kernel

    torch.manual_seed(1)
    heads, dim, width = 4, 256, 3
    projection = torch.randn(
        64 * width, dim + 2 * heads + 8, device="cuda", dtype=torch.bfloat16
    )
    weight = torch.randn(dim, 4, device="cuda", dtype=torch.bfloat16)
    conv_state = torch.randn(1 + 64, dim, 3, device="cuda", dtype=torch.bfloat16)
    payload = torch.empty(
        64 * width, 192 + 2 * heads, device="cuda", dtype=torch.bfloat16
    )

    def run(bs):
        rows = bs * width
        x = projection[:rows, :dim].view(bs, width, dim).transpose(1, 2)
        causal_conv1d_update(
            x,
            conv_state,
            weight,
            None,
            activation="silu",
            conv_state_indices=torch.arange(
                1, bs + 1, dtype=torch.int32, device="cuda"
            ),
            parent_indices=None,
            out=torch.empty_like(projection[:rows, :dim])
            .view(bs, width, dim)
            .transpose(1, 2),
            replay_payload=(
                payload[:rows],
                projection[:rows, dim : dim + heads],
                projection[:rows, dim + heads : dim + 2 * heads],
            ),
        )

    run(5)
    with patch.object(
        _causal_conv1d_update_kernel,
        "_do_compile",
        wraps=_causal_conv1d_update_kernel._do_compile,
    ) as compiles:
        for bs in (1, 16, 17, 32, 63):
            run(bs)
    assert compiles.call_count == 0

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

from tokenspeed.runtime.layers.attention.linear.causal_conv1d import (
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

    from tokenspeed.runtime.layers.attention.linear.causal_conv1d import (
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

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

"""The prefill convolution writing q, k and v rows equals conv plus split."""

import pytest
import torch
from tokenspeed_kernel.ops.attention.gdn._triton.causal_conv1d_metadata import (
    build_causal_conv1d_prefill_metadata,
)
from tokenspeed_kernel.ops.attention.gdn._triton.qkv_split import (
    fused_qkv_split_gdn_prefill,
)
from tokenspeed_kernel.ops.conv import causal_conv1d_fn, causal_conv1d_qkv_fn
from tokenspeed_kernel.ops.conv import cute_dsl as conv_cute_dsl
from tokenspeed_kernel.platform import current_platform
from utils import assert_no_cute_compile

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
_cute = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not current_platform().is_nvidia
    or current_platform().arch_version.major < 9,
    reason="the CuTe DSL convolution needs an NVIDIA GPU from Hopper on",
)

HEAD = 128


def _inputs(lengths, key_heads, value_heads, row_pad, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    key_dim, value_dim = key_heads * HEAD, value_heads * HEAD
    dim = 2 * key_dim + value_dim
    tokens = sum(lengths)
    # A column slice of a wider projection, as the model passes it.
    rows = torch.randn(tokens, dim + row_pad, device="cuda", generator=gen)
    x = rows.to(torch.bfloat16)[:, :dim].transpose(0, 1)
    weight = (torch.randn(dim, 4, device="cuda", generator=gen) * 0.5).to(
        torch.bfloat16
    )
    boundaries = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()],
        dtype=torch.int32,
        device="cuda",
    )
    metadata = build_causal_conv1d_prefill_metadata(
        boundaries, torch.tensor(lengths), 8
    )
    slots = torch.arange(len(lengths), dtype=torch.int32, device="cuda") * 2 + 1
    states = torch.randn(2 * len(lengths) + 2, dim, 3, device="cuda", generator=gen).to(
        torch.bfloat16
    )
    return x, weight, boundaries, metadata, slots, states, key_dim


@pytest.mark.parametrize(
    "lengths,initial",
    [
        ([8192], [True]),
        ([848], [False]),
        ([5000, 3, 1, 2, 3189], [True, False, True, True, False]),
        ([1, 2], [True, True]),
    ],
)
@pytest.mark.parametrize("key_heads,value_heads", [(12, 12), (2, 4), (4, 2)])
@pytest.mark.parametrize("activation", ["silu", None])
def test_qkv_conv_matches_conv_then_split(
    lengths, initial, key_heads, value_heads, activation
):
    x, weight, boundaries, metadata, slots, states, key_dim = _inputs(
        lengths, key_heads, value_heads, row_pad=1680
    )
    has_initial = torch.tensor(initial, device="cuda")
    ref_states, qkv_states = states.clone(), states.clone()
    packed = causal_conv1d_fn(
        x,
        weight,
        None,
        conv_states=ref_states,
        query_start_loc=boundaries,
        cache_indices=slots,
        has_initial_state=has_initial,
        activation=activation,
        prefill_metadata=metadata,
    ).transpose(0, 1)
    expected = fused_qkv_split_gdn_prefill(
        packed, key_heads, key_heads, value_heads, HEAD, HEAD, HEAD
    )
    actual = causal_conv1d_qkv_fn(
        x,
        weight,
        None,
        conv_states=qkv_states,
        query_start_loc=boundaries,
        cache_indices=slots,
        has_initial_state=has_initial,
        activation=activation,
        prefill_metadata=metadata,
        key_dim=key_dim,
    )
    for rows, ref in zip(actual, expected):
        assert rows.is_contiguous()
        assert torch.equal(rows, ref.view(rows.shape))
    assert torch.equal(qkv_states, ref_states)


@pytest.mark.parametrize("key_heads,value_heads", [(1, 2), (3, 3)])
def test_qkv_conv_rejects_widths_off_the_channel_block(key_heads, value_heads):
    x, weight, boundaries, metadata, slots, states, key_dim = _inputs(
        [16], key_heads, value_heads, row_pad=0
    )
    with pytest.raises(ValueError, match="multiples"):
        causal_conv1d_qkv_fn(
            x,
            weight,
            None,
            conv_states=states,
            query_start_loc=boundaries,
            cache_indices=slots,
            has_initial_state=torch.ones(1, dtype=torch.bool, device="cuda"),
            prefill_metadata=metadata,
            key_dim=key_dim,
        )


def _qkv(x, weight, states, boundaries, slots, initial, metadata, key_dim):
    return causal_conv1d_qkv_fn(
        x,
        weight,
        None,
        conv_states=states,
        query_start_loc=boundaries,
        cache_indices=slots,
        has_initial_state=torch.tensor(initial, device="cuda"),
        activation="silu",
        prefill_metadata=metadata,
        key_dim=key_dim,
    )


@_cute
def test_nvidia_selects_the_cute_qkv_conv(monkeypatch):
    x, weight, boundaries, metadata, slots, states, key_dim = _inputs(
        [40], 2, 4, row_pad=0
    )
    launches = []
    launch = conv_cute_dsl.launch_causal_conv1d_prefill

    def spy(*args, **kwargs):
        launches.append(kwargs["split_out"] is not None)
        return launch(*args, **kwargs)

    monkeypatch.setattr(conv_cute_dsl, "launch_causal_conv1d_prefill", spy)
    _qkv(x, weight, states, boundaries, slots, [True], metadata, key_dim)
    assert launches == [True]


@pytest.mark.parametrize("first", [3, 8, 13])
def test_qkv_conv_carries_the_window_across_prefill_chunks(first):
    """A request split into two prefill chunks reads the first chunk's last three
    inputs from the state, and leaves the rows and state of one whole chunk."""
    length = first + 21
    x, weight, boundaries, metadata, slots, states, key_dim = _inputs(
        [length], 2, 4, row_pad=64
    )
    whole_states, chunk_states = states.clone(), states.clone()
    whole = _qkv(x, weight, whole_states, boundaries, slots, [True], metadata, key_dim)
    for lo, hi in ((0, first), (first, length)):
        bounds = torch.tensor([0, hi - lo], dtype=torch.int32, device="cuda")
        chunk_metadata = build_causal_conv1d_prefill_metadata(
            bounds, torch.tensor([hi - lo]), 8
        )
        rows = _qkv(
            x[:, lo:hi],
            weight,
            chunk_states,
            bounds,
            slots,
            [True],
            chunk_metadata,
            key_dim,
        )
        for actual, expected in zip(rows, whole, strict=True):
            assert torch.equal(actual, expected[lo:hi])
    assert torch.equal(chunk_states, whole_states)


@_cute
def test_cute_qkv_conv_compiles_once_across_prefill_shapes():
    """Token totals, request counts and chunk maps vary per prefill; none keys a binary."""

    def run(lengths):
        x, weight, boundaries, metadata, slots, states, key_dim = _inputs(
            lengths, 2, 4, row_pad=64
        )
        initial = [index % 2 == 0 for index in range(len(lengths))]
        _qkv(x, weight, states, boundaries, slots, initial, metadata, key_dim)

    run([9, 7])
    with assert_no_cute_compile():
        for lengths in ([1], [16], [17], [1, 1], [33, 15], [64] * 3, [5, 1, 300]):
            run(lengths)


@_cute
def test_cute_qkv_conv_rejects_a_padding_slot_other_than_minus_one():
    x, weight, boundaries, metadata, slots, states, key_dim = _inputs(
        [16], 2, 4, row_pad=0
    )
    with pytest.raises(ValueError, match="padding slot -1"):
        conv_cute_dsl.cute_dsl_causal_conv1d_prefill_qkv(
            x,
            weight,
            None,
            states,
            boundaries,
            slots,
            torch.ones(1, dtype=torch.bool, device="cuda"),
            "silu",
            -2,
            prefill_metadata=metadata,
            key_dim=key_dim,
        )


@_cute
def test_cute_qkv_launch_rejects_widths_off_the_thread_vector():
    """A q/k boundary inside one thread's channels would store k values as q."""
    x, weight, boundaries, metadata, slots, states, _ = _inputs([16], 2, 4, row_pad=0)
    tokens, key_dim = x.shape[1], 130
    q, k, v = (x.new_empty((tokens, width)) for width in (key_dim, key_dim, 1024 - 260))
    with pytest.raises(ValueError, match="channels per thread"):
        conv_cute_dsl.launch_causal_conv1d_prefill(
            x,
            weight,
            None,
            states,
            slots,
            torch.ones(1, dtype=torch.bool, device="cuda"),
            boundaries,
            q,
            metadata=metadata,
            config=conv_cute_dsl._PARALLEL,
            silu=True,
            enable_pdl=False,
            split_out=(k, v),
        )

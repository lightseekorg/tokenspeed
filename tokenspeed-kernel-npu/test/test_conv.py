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

"""Numerics of the Ascend causal conv1d against a per-token reference.

The reference walks one request at a time, in float32, applying the window
arithmetic directly: a pool row holds the last ``width - 1`` tokens oldest
first, and token ``t`` of a chunk convolves ``X[t - (width - 1) .. t]``.
"""

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")

from tokenspeed_kernel_npu.ops.conv import (
    causal_conv1d_prefill,
    causal_conv1d_update,
)

pytestmark = pytest.mark.skipif(
    not torch.npu.is_available(), reason="Ascend conv1d tests require an NPU"
)

# The native op requires channels % 128 == 0.
CHANNELS = 128
# bf16 rounding plus the op's own accumulator order; the reference is fp32.
RTOL = 0.02


def _reference_prefill(x, weight, pool, lengths, cache, activation=None, bias=None):
    """``x`` is ``[channels, tokens]``; returns ``(out, new_pool)``."""
    channels, tokens = x.shape
    width = weight.shape[1]
    window = width - 1
    out = torch.zeros(channels, tokens, dtype=torch.float32, device=x.device)
    new = pool.clone().float()
    start = 0
    for request, length in enumerate(lengths):
        row = int(cache[request])
        history = new[row].clone()
        for token in range(length):
            full = torch.cat([history, x[:, start + token : start + token + 1]], 1)
            out[:, start + token] = (full[:, -width:] * weight.float()).sum(1)
            history = full[:, -window:]
        new[row] = history
        start += length
    if bias is not None:
        out = out + bias.float()[:, None]
    if activation in ("silu", "swish", True):
        out = out * torch.sigmoid(out)
    return out, new


def _token_value(x, history, token, column, parents, window):
    """Value multiplied by ``weight[:, column]`` in the reference."""
    if parents is None:
        position = token - window + column
        return x[:, position] if position >= 0 else history[:, position]
    node, past_root = token, 0
    for _ in range(window - column):
        if node < 0:
            past_root += 1
            continue
        parent = int(parents[node])
        past_root += int(parent < 0)
        node = parent
    # The portable kernel reads window slot (width - 1) - past_root.
    return x[:, node] if node >= 0 else history[:, window - past_root]


def _reference_update(
    x, weight, pool, read_rows, write_rows, parents=None, per_token=False
):
    """``x`` is ``[batch, channels, steps]``; returns ``(out, new_pool)``.

    ``write_rows[request]`` lists the rows that request writes: one row per
    request when ``per_token`` is false (the last token's window), one row per
    token otherwise.
    """
    batch, _, steps = x.shape
    width = weight.shape[1]
    window = width - 1
    out = torch.zeros(batch, CHANNELS, steps, dtype=torch.float32, device=x.device)
    new = pool.clone().float()
    for request in range(batch):
        history = new[read_rows[request]].clone()
        per_request_parents = None if parents is None else parents[request]
        for token in range(steps):
            for column in range(width):
                value = _token_value(
                    x[request], history, token, column, per_request_parents, window
                )
                out[request, :, token] += weight[:, column].float() * value.float()
        if parents is not None:
            for token in range(steps):
                columns = [
                    _token_value(
                        x[request], history, token, column, per_request_parents, window
                    )
                    for column in range(1, width)
                ]
                new[write_rows[request][token]] = torch.stack(columns, -1)
        elif per_token:
            full = torch.cat([history, x[request]], 1)
            for token in range(steps):
                new[write_rows[request][token]] = full[
                    :, token + 1 : token + 1 + window
                ]
        else:
            new[write_rows[request][0]] = torch.cat([history, x[request]], 1)[
                :, -window:
            ]
    return out, new


def _assert_close(got, expected, what):
    error = (got.float() - expected).abs().max().item()
    peak = expected.abs().max().item() + 1e-9
    assert error / peak < RTOL, f"{what}: rel error {error / peak:.4f}"


def _random(seed, **shapes):
    """Small bf16 tensors on the NPU, keyed by name."""
    torch.manual_seed(seed)
    return {
        name: torch.randn(*shape, dtype=torch.bfloat16, device="npu") * 0.1
        for name, shape in shapes.items()
    }


@pytest.mark.parametrize(
    ("width", "lengths", "cache", "zero_rows", "seed"),
    [
        (4, [5, 3, 4], [2, 0, 1], [], 11),
        (4, [1, 2, 3], [0, 1, 2], [0, 1], 13),
        (4, [12], [0], [], 17),
        (3, [4, 6], [1, 0], [], 19),
    ],
)
def test_prefill_matches_reference(width, lengths, cache, zero_rows, seed):
    tokens = sum(lengths)
    tensors = _random(
        seed,
        x=(CHANNELS, tokens),
        weight=(CHANNELS, width),
        pool=(4, CHANNELS, width - 1),
    )
    pool = tensors["pool"]
    for row in zero_rows:
        pool[row] = 0
    pristine = pool.clone()

    # A zeroed row is how the upstream prefill ABI says "no history".
    has_initial_state = torch.tensor(
        [row not in zero_rows for row in cache], device="npu"
    )
    cache_indices = torch.tensor(cache, dtype=torch.int32, device="npu")
    query_start_loc = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(lengths), 0).tolist()),
        dtype=torch.int32,
        device="npu",
    )

    got = causal_conv1d_prefill(
        tensors["x"],
        tensors["weight"],
        None,
        pool,
        query_start_loc,
        cache_indices,
        has_initial_state,
        None,
        -1,
    )
    expected, expected_pool = _reference_prefill(
        tensors["x"], tensors["weight"], pristine, lengths, cache
    )
    _assert_close(got, expected, "prefill output")
    _assert_close(pool, expected_pool, "prefill pool")

    # Rows this batch never named keep their contents.
    for row in range(pristine.shape[0]):
        if row not in cache:
            assert torch.equal(pool[row], pristine[row]), f"row {row} was written"


def test_prefill_applies_bias_and_silu():
    width, lengths, cache, seed = 4, [4, 3], [0, 1], 23
    tokens = sum(lengths)
    tensors = _random(
        seed,
        x=(CHANNELS, tokens),
        weight=(CHANNELS, width),
        pool=(3, CHANNELS, width - 1),
        bias=(CHANNELS,),
    )
    pristine = tensors["pool"].clone()
    query_start_loc = torch.tensor([0, 4, 7], dtype=torch.int32, device="npu")
    got = causal_conv1d_prefill(
        tensors["x"],
        tensors["weight"],
        tensors["bias"],
        tensors["pool"],
        query_start_loc,
        torch.tensor(cache, dtype=torch.int32, device="npu"),
        torch.ones(2, dtype=torch.bool, device="npu"),
        "silu",
        -1,
    )
    expected, _ = _reference_prefill(
        tensors["x"],
        tensors["weight"],
        pristine,
        lengths,
        cache,
        activation="silu",
        bias=tensors["bias"],
    )
    _assert_close(got, expected, "prefill bias + silu")


def test_prefill_leaves_a_padded_request_alone():
    width, lengths, seed = 4, [4, 3], 29
    tokens = sum(lengths)
    tensors = _random(
        seed,
        x=(CHANNELS, tokens),
        weight=(CHANNELS, width),
        pool=(4, CHANNELS, width - 1),
    )
    pristine = tensors["pool"].clone()
    query_start_loc = torch.tensor([0, 4, 7], dtype=torch.int32, device="npu")
    # Request 1 is padding: a negative cache index skipped by pad_slot_id.
    cache_indices = torch.tensor([2, -1], dtype=torch.int32, device="npu")

    got = causal_conv1d_prefill(
        tensors["x"],
        tensors["weight"],
        None,
        tensors["pool"],
        query_start_loc,
        cache_indices,
        torch.ones(2, dtype=torch.bool, device="npu"),
        None,
        -1,
    )
    expected, _ = _reference_prefill(
        tensors["x"], tensors["weight"], pristine, lengths, [2, 0]
    )
    _assert_close(got[:, :4], expected[:, :4], "padded-batch output")
    for row in range(4):
        if row != 2:
            assert torch.equal(
                tensors["pool"][row], pristine[row]
            ), f"padded batch wrote row {row}"


@pytest.mark.parametrize(
    ("width", "steps", "cache", "seed"),
    [
        (4, 1, [1, 0], 31),
        (4, 4, [1, 0], 37),
        (4, 4, [0, 1], 41),
        (3, 1, [1, 0], 43),
    ],
)
def test_update_matches_reference(width, steps, cache, seed):
    tensors = _random(
        seed,
        x=(2, CHANNELS, steps),
        weight=(CHANNELS, width),
        pool=(4, CHANNELS, width - 1),
    )
    pristine = tensors["pool"].clone()
    rows = torch.tensor(cache, dtype=torch.int32, device="npu")

    got = causal_conv1d_update(
        tensors["x"],
        tensors["pool"],
        tensors["weight"],
        None,
        None,
        conv_state_indices=rows,
        output_state_indices=rows.view(-1, 1),
        parent_indices=None,
    )
    write_rows = [[row] for row in cache]
    expected, expected_pool = _reference_update(
        tensors["x"], tensors["weight"], pristine, cache, write_rows
    )
    _assert_close(got, expected, "update output")
    _assert_close(tensors["pool"], expected_pool, "update pool")


def test_update_writes_a_row_other_than_the_one_it_read():
    width, seed = 4, 47
    tensors = _random(
        seed,
        x=(2, CHANNELS, 1),
        weight=(CHANNELS, width),
        pool=(4, CHANNELS, width - 1),
    )
    pristine = tensors["pool"].clone()
    read = torch.tensor([3, 2], dtype=torch.int32, device="npu")
    write = torch.tensor([0, 1], dtype=torch.int32, device="npu")

    got = causal_conv1d_update(
        tensors["x"],
        tensors["pool"],
        tensors["weight"],
        None,
        None,
        conv_state_indices=read,
        output_state_indices=write.view(-1, 1),
        parent_indices=None,
    )
    expected, expected_pool = _reference_update(
        tensors["x"], tensors["weight"], pristine, [3, 2], [[0], [1]]
    )
    _assert_close(got, expected, "out-of-place output")
    _assert_close(tensors["pool"][[0, 1]], expected_pool[[0, 1]], "out-of-place rows")
    # The committed rows the call only read must not move.
    for row in (2, 3):
        assert torch.equal(
            tensors["pool"][row], pristine[row]
        ), f"read row {row} was overwritten"


@pytest.mark.parametrize(
    "parents",
    [
        [[-1, 0, 1, 2], [-1, 0, 0, 2]],
        [[-1, 0, 0, 1], [-1, 0, 1, 1]],
        [[-1, -1, -1, -1], [-1, 0, -1, -1]],
    ],
)
def test_update_draft_tree_matches_reference(parents):
    width, steps, seed = 4, 4, 53
    tensors = _random(
        seed,
        x=(2, CHANNELS, steps),
        weight=(CHANNELS, width),
        pool=(12, CHANNELS, width - 1),
    )
    pristine = tensors["pool"].clone()
    read = torch.tensor([10, 4], dtype=torch.int32, device="npu")
    out_rows = torch.tensor(
        [[0, 1, 2, 3], [5, 6, 7, 8]], dtype=torch.int32, device="npu"
    )

    got = causal_conv1d_update(
        tensors["x"],
        tensors["pool"],
        tensors["weight"],
        None,
        None,
        conv_state_indices=read,
        output_state_indices=out_rows,
        parent_indices=torch.tensor(parents, dtype=torch.int32, device="npu"),
    )
    expected, expected_pool = _reference_update(
        tensors["x"],
        tensors["weight"],
        pristine,
        [10, 4],
        [[0, 1, 2, 3], [5, 6, 7, 8]],
        parents=parents,
    )
    _assert_close(got, expected, "tree output")
    _assert_close(tensors["pool"], expected_pool, "tree pool")


def test_update_applies_bias_and_silu():
    width, seed = 4, 59
    tensors = _random(
        seed,
        x=(2, CHANNELS, 1),
        weight=(CHANNELS, width),
        pool=(2, CHANNELS, width - 1),
        bias=(CHANNELS,),
    )
    pristine = tensors["pool"].clone()
    rows = torch.tensor([0, 1], dtype=torch.int32, device="npu")

    got = causal_conv1d_update(
        tensors["x"],
        tensors["pool"],
        tensors["weight"],
        tensors["bias"],
        "silu",
        conv_state_indices=rows,
        output_state_indices=rows.view(-1, 1),
        parent_indices=None,
    )
    expected, _ = _reference_update(
        tensors["x"], tensors["weight"], pristine, [0, 1], [[0], [1]]
    )
    expected = expected + tensors["bias"].float()[:, None]
    expected = expected * torch.sigmoid(expected)
    _assert_close(got, expected, "update bias + silu")


def test_update_chain_verify_writes_a_row_per_token():
    """The runtime's chain verify passes one scratch row per token."""
    width, steps, seed = 4, 4, 67
    tensors = _random(
        seed,
        x=(2, CHANNELS, steps),
        weight=(CHANNELS, width),
        pool=(12, CHANNELS, width - 1),
    )
    pristine = tensors["pool"].clone()
    # Rows 0 and 5 seed the two requests' committed windows.
    read = torch.tensor([0, 5], dtype=torch.int32, device="npu")
    out_rows = torch.tensor(
        [[1, 2, 3, 4], [6, 7, 8, 9]], dtype=torch.int32, device="npu"
    )

    got = causal_conv1d_update(
        tensors["x"],
        tensors["pool"],
        tensors["weight"],
        None,
        None,
        conv_state_indices=read,
        output_state_indices=out_rows,
        parent_indices=None,
    )
    expected, expected_pool = _reference_update(
        tensors["x"],
        tensors["weight"],
        pristine,
        [0, 5],
        [[1, 2, 3, 4], [6, 7, 8, 9]],
        per_token=True,
    )
    _assert_close(got, expected, "per-token chain output")
    _assert_close(tensors["pool"], expected_pool, "per-token chain pool")


@pytest.mark.parametrize(
    ("name", "kwargs"),
    [
        ("num_accepted_tokens", {"num_accepted_tokens": "rows"}),
        ("cache_seqlens", {"cache_seqlens": "rows"}),
        ("intermediate_conv_window", {"intermediate_conv_window": "window"}),
        ("replay_payload", {"replay_payload": "payload"}),
    ],
)
def test_update_rejects_options_the_native_op_cannot_honor(name, kwargs):
    torch.manual_seed(61)
    x = torch.randn(2, CHANNELS, 1, dtype=torch.bfloat16, device="npu")
    weight = torch.randn(CHANNELS, 4, dtype=torch.bfloat16, device="npu") * 0.1
    pool = torch.zeros(2, CHANNELS, 3, dtype=torch.bfloat16, device="npu")
    rows = torch.tensor([0, 1], dtype=torch.int32, device="npu")

    resolved = {
        "rows": rows,
        "window": torch.zeros(2, CHANNELS, 3, device="npu"),
        "payload": (x.new_empty(2, 260), x.new_empty(2, 1), x.new_empty(2, 1)),
    }
    with pytest.raises(NotImplementedError):
        causal_conv1d_update(
            x,
            pool,
            weight,
            None,
            None,
            conv_state_indices=rows,
            output_state_indices=rows.view(-1, 1),
            parent_indices=None,
            **{key: resolved[value] for key, value in kwargs.items()},
        )

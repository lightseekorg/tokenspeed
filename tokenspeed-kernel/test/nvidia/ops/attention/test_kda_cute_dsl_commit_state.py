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

"""CuTe DSL batched KDA state commit against the Triton batched commit.

Both commits run on copies of the same serving-shaped pools: per-layer FP32
state pages, padded FP32 correction and BF16 key / raw-gate record rows, a
uint64 descriptor table, and record stamps. Whole pools are compared bit for
bit, so a page outside the write set that changes fails as well. Intervals
include in-place commits, zero accepted tokens, padding pages, and records
the stamps reject.
"""

from __future__ import annotations

import pytest
import torch
from utils import assert_no_cute_compile, is_nvidia

if not is_nvidia():
    pytest.skip("requires an NVIDIA GPU", allow_module_level=True)
pytest.importorskip("cutlass")
pytest.importorskip("quack")
from tokenspeed_kernel.platform import ArchVersion, current_platform

if not ArchVersion(10, 0) <= current_platform().arch_version <= ArchVersion(10, 3):
    pytest.skip("requires SM100 or SM103", allow_module_level=True)
from tokenspeed_kernel.ops.attention.kda import (  # noqa: E402
    resolve_kda_batched_replay_commit,
)
from tokenspeed_kernel.ops.attention.kda._cute_dsl.commit_state import (  # noqa: E402
    MAX_WINDOW,
    commit_states,
)
from tokenspeed_kernel.ops.attention.kda.vllm_recoverssm import (  # noqa: E402
    batched_recoverssm_commit,
)

HEADS, DIM = 12, 128
WIDTH = HEADS * DIM
CORRECTION_ROW_PAD, KEY_GATE_ROW_PAD = 16, 32


def _pools(layers, batch, *, window, capacity, slots, seed, extreme):
    """One request's pages, records, and descriptors per layer."""
    gen = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape, scale=1.0):
        return torch.randn(*shape, device="cuda", generator=gen) * scale

    pages = 3 * batch + 2
    rows = slots * capacity
    state = [randn(pages, HEADS, DIM, DIM, scale=0.5) for _ in range(layers)]
    conv = [randn(pages, 3 * WIDTH, 3).bfloat16() for _ in range(layers)]
    qkv = [randn(rows, 3 * WIDTH).bfloat16() for _ in range(layers)]
    correction = [randn(rows, WIDTH + CORRECTION_ROW_PAD) for _ in range(layers)]
    key_gate = [randn(rows, 2 * WIDTH + KEY_GATE_ROW_PAD) for _ in range(layers)]
    for records, gates in zip(key_gate, correction):
        records[:, WIDTH : 2 * WIDTH] *= 3
        if extreme:
            records[::3, WIDTH : 2 * WIDTH] *= 1000  # saturated sigmoids
            records[1::5, :WIDTH] = 0  # zero keys: rsqrt(1e-6)
            records[2::7, :WIDTH] *= 1e-30  # subnormal squares
            gates[::4] *= 1e30
    key_gate = [records.bfloat16() for records in key_gate]
    a_log = [randn(HEADS, scale=0.5) for _ in range(layers)]
    dt_bias = [randn(WIDTH) for _ in range(layers)]
    conv_weights = [randn(3 * WIDTH, 4).bfloat16() for _ in range(layers)]
    descriptors = torch.tensor(
        [
            [
                qkv[i].data_ptr(),
                conv_weights[i].data_ptr(),
                conv[i].data_ptr(),
                correction[i].data_ptr(),
                0,
                key_gate[i].data_ptr(),
                a_log[i].data_ptr(),
                dt_bias[i].data_ptr(),
                state[i].data_ptr(),
                0,
            ]
            for i in range(layers)
        ],
        dtype=torch.uint64,
        device="cuda",
    )
    keep = (qkv, correction, key_gate, a_log, dt_bias, conv_weights)
    return descriptors, state, conv, keep


def _round(layers, batch, *, window, capacity, slots, seed, rejects):
    """Pages, accepted lengths, and record metadata of one commit round."""
    gen = torch.Generator().manual_seed(seed + 1)
    pages = 3 * batch + 2
    order = torch.randperm(pages - 1, generator=gen)[: 2 * batch] + 1
    reads = order[:batch].to(torch.int32).view(1, batch).cuda()
    writes = order[batch:].to(torch.int32).view(1, batch).cuda()
    accepted = torch.randint(0, window + 1, (batch,), generator=gen, dtype=torch.int32)
    accepted[0] = window
    if batch > 1:
        writes[0, 1] = reads[0, 1]  # an in-place commit
        accepted[1] = 0  # copies the source page
    metadata = torch.full((batch, 3), -1, dtype=torch.int64)
    stamps = torch.full((layers, slots, capacity, 2), -1, dtype=torch.int64)
    request_slots = torch.randperm(slots, generator=gen)[:batch]
    for b in range(batch):
        start = int(torch.randint(0, 1000, (), generator=gen))
        generation = int(torch.randint(1, 50, (), generator=gen))
        metadata[b] = torch.tensor([int(request_slots[b]), generation, start])
        for t in range(window):
            stamps[:, request_slots[b], (start + t) % capacity] = torch.tensor(
                [start + t, generation]
            )
    if rejects and batch >= 4:
        reads[0, 2] = 0  # a padding page
        stamps[layers - 1, request_slots[3], int(metadata[3, 2]) % capacity, 1] += 1
        metadata[batch - 1, 0] = slots + 5  # a slot outside the stamps
    return {
        "group_indices": torch.zeros(layers, dtype=torch.int32, device="cuda"),
        "read_indices": reads,
        "write_indices": writes,
        "accepted_length": accepted.cuda(),
        "record_metadata": metadata.cuda(),
        "record_stamps": stamps.cuda(),
    }


def _strides(window, lower_bound=-5.0):
    return {
        "draft_token_num": window,
        "num_heads": HEADS,
        "correction_stride": WIDTH + CORRECTION_ROW_PAD,
        "key_gate_stride": 2 * WIDTH + KEY_GATE_ROW_PAD,
        "state_stride": HEADS * DIM * DIM,
        "lower_bound": lower_bound,
    }


def _triton(descriptors, round_args, window, lower_bound=-5.0):
    batched_recoverssm_commit(
        descriptors,
        **round_args,
        **_strides(window, lower_bound),
        activation_dtype=torch.bfloat16,
        conv_state_dtype=torch.bfloat16,
        head_dim=DIM,
        qkv_stride=3 * WIDTH,
        conv_stride=9 * WIDTH,
        conv_width=4,
        state_dtype=torch.float32,
    )


def _assert_same(got, want):
    for g, w in zip(got, want):
        assert torch.equal(g.view(torch.int32), w.view(torch.int32))


@pytest.mark.parametrize(
    "layers, batch, window, capacity, slots, rejects",
    [
        (4, 1, 8, 8, 4, False),
        (6, 4, 8, 8, 8, True),
        (3, 8, 8, 16, 16, True),
        (2, 16, 8, 8, 16, False),
        (69, 2, 8, 8, 4, False),
        (3, 4, 1, 1, 4, False),
        (3, 4, 2, 4, 4, True),
        (3, 4, MAX_WINDOW, MAX_WINDOW, 4, True),
    ],
)
@pytest.mark.parametrize("extreme", [False, True], ids=["random", "extreme"])
def test_the_state_commit_matches_the_triton_commit(
    layers, batch, window, capacity, slots, rejects, extreme
) -> None:
    descriptors, state, _, keep = _pools(
        layers,
        batch,
        window=window,
        capacity=capacity,
        slots=slots,
        seed=batch * 31 + window,
        extreme=extreme,
    )
    round_args = _round(
        layers,
        batch,
        window=window,
        capacity=capacity,
        slots=slots,
        seed=window,
        rejects=rejects,
    )
    before = [pages.clone() for pages in state]
    _triton(descriptors, round_args, window)
    want = [pages.clone() for pages in state]
    for pages, original in zip(state, before):
        pages.copy_(original)
    commit_states(descriptors, **round_args, **_strides(window))
    torch.cuda.synchronize()
    _assert_same(state, want)
    assert any(not torch.equal(w, b) for w, b in zip(want, before))
    del keep


def test_a_length_past_the_window_writes_nothing() -> None:
    """A request accepting more tokens than the window leaves its target page alone.

    Its stamps cover the extra position too, so only the length bound refuses it.
    """
    layers, batch, window, capacity, slots = 3, 4, 8, 16, 8
    descriptors, state, _, keep = _pools(
        layers,
        batch,
        window=window,
        capacity=capacity,
        slots=slots,
        seed=11,
        extreme=False,
    )
    round_args = _round(
        layers,
        batch,
        window=window,
        capacity=capacity,
        slots=slots,
        seed=12,
        rejects=False,
    )
    slot, generation, start = round_args["record_metadata"][2].tolist()
    round_args["record_stamps"][:, slot, (start + window) % capacity] = torch.tensor(
        [start + window, generation]
    )
    round_args["accepted_length"][2] = window + 1
    target = int(round_args["write_indices"][0, 2])
    before = [pages.clone() for pages in state]
    commit_states(descriptors, **round_args, **_strides(window))
    torch.cuda.synchronize()
    for got, original in zip(state, before):
        assert torch.equal(got[target], original[target])
    assert any(not torch.equal(g, b) for g, b in zip(state, before))
    del keep


def test_the_registered_commit_matches_the_triton_commit() -> None:
    """The CuTe DSL solution commits states and convolution windows as Triton does."""
    layers, batch, window = 5, 6, 8
    kernel = resolve_kda_batched_replay_commit(
        torch.bfloat16,
        num_heads=HEADS,
        head_dim=DIM,
        spec_query_len=window,
        state_dtype=torch.float32,
    )
    assert kernel.name == "cute_dsl_kda_batched_replay_commit"
    descriptors, state, conv, keep = _pools(
        layers, batch, window=window, capacity=window, slots=8, seed=3, extreme=False
    )
    round_args = _round(
        layers, batch, window=window, capacity=window, slots=8, seed=4, rejects=True
    )
    before = [pages.clone() for pages in state + conv]
    _triton(descriptors, round_args, window)
    want = [pages.clone() for pages in state + conv]
    for pages, original in zip(state + conv, before):
        pages.copy_(original)
    kernel(
        descriptors,
        **round_args,
        **_strides(window),
        activation_dtype=torch.bfloat16,
        conv_state_dtype=torch.bfloat16,
        head_dim=DIM,
        qkv_stride=3 * WIDTH,
        conv_stride=9 * WIDTH,
        conv_width=4,
        state_dtype=torch.float32,
    )
    torch.cuda.synchronize()
    _assert_same(state, want[:layers])
    for got, expected in zip(conv, want[layers:]):
        assert torch.equal(got.view(torch.int16), expected.view(torch.int16))
    del keep


def test_an_unbounded_gate_runs_the_triton_commit() -> None:
    layers, batch, window = 3, 4, 8
    kernel = resolve_kda_batched_replay_commit(
        torch.bfloat16,
        num_heads=HEADS,
        head_dim=DIM,
        spec_query_len=window,
        state_dtype=torch.float32,
    )
    descriptors, state, _, keep = _pools(
        layers, batch, window=window, capacity=window, slots=4, seed=5, extreme=False
    )
    round_args = _round(
        layers, batch, window=window, capacity=window, slots=4, seed=6, rejects=False
    )
    before = [pages.clone() for pages in state]
    _triton(descriptors, round_args, window, lower_bound=None)
    want = [pages.clone() for pages in state]
    for pages, original in zip(state, before):
        pages.copy_(original)
    kernel(
        descriptors,
        **round_args,
        **_strides(window, lower_bound=None),
        activation_dtype=torch.bfloat16,
        conv_state_dtype=torch.bfloat16,
        head_dim=DIM,
        qkv_stride=3 * WIDTH,
        conv_stride=9 * WIDTH,
        conv_width=4,
        state_dtype=torch.float32,
    )
    torch.cuda.synchronize()
    _assert_same(state, want)
    del keep


def test_the_state_commit_replays_in_a_cuda_graph_without_tracing() -> None:
    """Captured once, the commit replays a new round over refreshed inputs."""
    layers, window = 4, 8
    descriptors, state, _, keep = _pools(
        layers, 4, window=window, capacity=window, slots=8, seed=7, extreme=False
    )
    first = _round(
        layers, 4, window=window, capacity=window, slots=8, seed=8, rejects=False
    )
    second = _round(
        layers, 4, window=window, capacity=window, slots=8, seed=9, rejects=True
    )
    original = [pages.clone() for pages in state]
    commit_states(descriptors, **first, **_strides(window))  # compiles before capture
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with assert_no_cute_compile(), torch.cuda.graph(graph, stream=stream):
        commit_states(descriptors, **first, **_strides(window))
    for name, tensor in first.items():
        tensor.copy_(second[name])
    for pages, saved in zip(state, original):
        pages.copy_(saved)
    graph.replay()
    torch.cuda.synchronize()
    replayed = [pages.clone() for pages in state]
    for pages, saved in zip(state, original):
        pages.copy_(saved)
    _triton(descriptors, second, window)
    torch.cuda.synchronize()
    _assert_same(replayed, state)
    del keep


@pytest.mark.parametrize(
    "dtype, spec_query_len, state_dtype, name",
    [
        (torch.bfloat16, 8, torch.float32, "cute_dsl_kda_batched_replay_commit"),
        (
            torch.bfloat16,
            MAX_WINDOW,
            torch.float32,
            "cute_dsl_kda_batched_replay_commit",
        ),
        (torch.bfloat16, 16, torch.float32, "triton_nvidia_kda_batched_replay_commit"),
        (torch.bfloat16, 8, torch.bfloat16, "triton_nvidia_kda_batched_replay_commit"),
        (torch.float16, 8, torch.float32, "triton_nvidia_kda_batched_replay_commit"),
        (torch.bfloat16, None, None, "triton_nvidia_kda_batched_replay_commit"),
    ],
)
def test_the_cute_dsl_commit_takes_short_windows_over_fp32_state(
    dtype, spec_query_len, state_dtype, name
) -> None:
    kernel = resolve_kda_batched_replay_commit(
        dtype,
        num_heads=HEADS,
        head_dim=DIM,
        spec_query_len=spec_query_len,
        state_dtype=state_dtype,
    )
    assert kernel.name == name

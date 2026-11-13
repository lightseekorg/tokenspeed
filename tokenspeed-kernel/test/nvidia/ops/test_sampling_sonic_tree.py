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

"""sonic draws every node of a draft tree as ``sample`` draws, keyed by its depth.

``tree_targets`` draws one token per node row. Its noise is keyed by the
slot's seed and offset and the node's depth, not its row, so the nodes of one
depth share their randomness; its penalties count the tokens on the node's
path on top of the slot's decode counts, which the draw leaves unchanged.
``tree_paths`` lists each node's path and ``count_emitted`` adds the tokens a
verify emits to the decode counts.
"""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.sampling import sonic
from tokenspeed_kernel.ops.sampling.sonic import (
    Indicator,
    SlotState,
    configs,
    pick,
    scratchpad_columns,
)
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.thirdparty.sonic_sampler.ops.base import (
    philox_gumbel,
    philox_key,
)

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and current_platform().is_nvidia),
    reason="sonic's kernels run on NVIDIA GPUs",
)

VOCAB = 32768
SLOTS = 9
# Each request's tree: the root, two children, two grandchildren under the first child, one node below the second grandchild.
PARENTS = [-1, 0, 0, 1, 1, 4]
DEPTHS = [0, 1, 1, 2, 2, 3]
NODES = len(PARENTS)
REQUESTS = SLOTS - 1
ROWS = REQUESTS * NODES


@triton.jit
def _noise_kernel(
    seeds_ptr, offsets_ptr, slots_ptr, steps_ptr, out_ptr, vocab, BLOCK: tl.constexpr
):
    """The Gumbel noise sonic's draws subtract, at every token of each row's slot and step."""
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    key = philox_key(
        seeds_ptr, offsets_ptr, tl.load(slots_ptr + row), tl.load(steps_ptr + row)
    )
    tl.store(out_ptr + row * vocab + cols, philox_gumbel(key, cols), mask=cols < vocab)


def _noise(state: SlotState, offsets, slots, steps) -> torch.Tensor:
    out = torch.empty((slots.numel(), VOCAB), device="cuda")
    _noise_kernel[(slots.numel(), triton.cdiv(VOCAB, 4096))](
        state.seeds, offsets, slots, steps, out, VOCAB, BLOCK=4096
    )
    return out


def _state(dtype: torch.dtype, bits: list[Indicator], frequency: float = 0.0):
    """Slot ``s`` reads ``bits[s % len(bits)]`` at temperature 1.0 and top-k 16, with ``frequency``."""
    state = SlotState.allocate(SLOTS, 1, VOCAB, 7, "cuda")
    state.seeds.copy_(torch.arange(SLOTS) * 7919 + 3)
    state.indicators.copy_(
        torch.tensor([int(bits[s % len(bits)]) for s in range(SLOTS)])
    )
    state.top_k.fill_(16)
    scalars = torch.tensor([[1.0, 0.9, 0.1, 1.0, frequency, 0.0]] * SLOTS).T
    return state.with_scalars(scalars.contiguous().to("cuda", dtype))


def _tree(requests: int = REQUESTS) -> tuple[torch.Tensor, torch.Tensor]:
    parents = torch.tensor(PARENTS, dtype=torch.int32, device="cuda")
    depths = torch.tensor(DEPTHS, dtype=torch.int32, device="cuda")
    return parents.repeat(requests, 1), depths.repeat(requests, 1)


def _rows():
    """Each node row's slot, its request's offset table, and its depth."""
    slots = torch.arange(1, SLOTS, dtype=torch.int64, device="cuda")
    offsets = torch.arange(SLOTS, dtype=torch.int64, device="cuda") * 13 + 5
    _, depths = _tree()
    return slots.repeat_interleave(NODES), offsets, depths.view(-1)


def _draw(logits, state, slots, offsets, steps, ancestors, config=None):
    rows = logits.shape[0]
    tokens = torch.empty((rows, 1), dtype=torch.int32, device="cuda")
    scratchpad = torch.empty(
        (rows, scratchpad_columns(VOCAB)), dtype=torch.uint32, device="cuda"
    )
    sonic.tree_targets(
        logits,
        state,
        slots,
        offsets,
        steps,
        ancestors,
        tokens,
        scratchpad,
        config or pick(VOCAB, rows),
        False,
    )
    return tokens.view(-1)


def _no_paths(rows: int = ROWS) -> torch.Tensor:
    return torch.full((rows, NODES - 1), -1, dtype=torch.int32, device="cuda")


def _logits(rows: int, seed: int, dtype: torch.dtype) -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    logits = torch.randn((rows, VOCAB), generator=generator, device="cuda") * 0.5
    head = torch.randint(0, VOCAB, (rows, 64), generator=generator, device="cuda")
    logits.scatter_(1, head, 6.0)
    return logits.to(dtype)


def test_tree_paths_list_each_node_path_below_its_root():
    candidates = torch.randint(
        0, VOCAB, (REQUESTS, NODES), dtype=torch.int32, device="cuda"
    )
    parents, depths = _tree()
    paths = torch.empty((ROWS, NODES - 1), dtype=torch.int32, device="cuda")
    sonic.tree_paths(candidates, parents, depths, paths)
    expected = torch.full((ROWS, NODES - 1), -1, dtype=torch.int32)
    for request in range(REQUESTS):
        for node in range(NODES):
            walk = node
            while walk > 0:
                expected[request * NODES + node, DEPTHS[walk] - 1] = candidates[
                    request, walk
                ]
                walk = PARENTS[walk]
    assert torch.equal(paths.cpu(), expected)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_whole_rows_take_the_gumbel_max_keyed_by_their_depth(dtype):
    slots, offsets, steps = _rows()
    logits = _logits(ROWS, 1, dtype)
    # Two nodes of one depth with one row of logits draw one token.
    logits[2] = logits[1]
    state = _state(dtype, [Indicator.from_params()])
    expected = (logits.float() - _noise(state, offsets, slots, steps)).argmax(dim=1)

    for config in configs(VOCAB):
        tokens = _draw(logits, state, slots, offsets, steps, _no_paths(), config)
        assert torch.equal(tokens, expected.int()), config
    assert int(tokens[1]) == int(tokens[2])


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_candidate_rows_draw_among_their_top_k_keyed_by_their_depth(dtype):
    slots, offsets, steps = _rows()
    generator = torch.Generator(device="cuda").manual_seed(2)
    logits = torch.randn((ROWS, VOCAB), generator=generator, device="cuda") - 4.0
    # Distinct bf16 values, so the top 16 hold apart from the rest and from each other.
    top = torch.randperm(VOCAB, generator=generator, device="cuda")[:16]
    logits[:, top] = 2.0 + 0.0625 * torch.arange(16, device="cuda", dtype=torch.float32)
    logits = logits.to(dtype)
    state = _state(dtype, [Indicator.from_params(top_k=16)])
    scores = logits.float() - _noise(state, offsets, slots, steps)
    expected = top[scores[:, top].argmax(dim=1)]

    tokens = _draw(logits, state, slots, offsets, steps, _no_paths())
    assert torch.equal(tokens, expected.int())


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_depth_zero_nodes_draw_what_sample_draws(dtype):
    rows = REQUESTS
    slots = torch.arange(1, SLOTS, dtype=torch.int64, device="cuda")
    offsets = torch.arange(SLOTS, dtype=torch.int64, device="cuda") * 13 + 5
    logits = _logits(rows, 3, dtype)
    bits = [
        Indicator.from_params(),
        Indicator.from_params(top_k=1),
        Indicator.from_params(top_k=16),
        Indicator.from_params(top_p=0.9),
        Indicator.from_params(min_p=0.1),
    ]
    state = _state(dtype, bits)
    steps = torch.zeros(rows, dtype=torch.int32, device="cuda")

    tokens = _draw(logits, state, slots, offsets, steps, _no_paths(rows))
    assert not state.decode_counts.any()

    sampled = torch.empty((rows, 1), dtype=torch.int32, device="cuda")
    scratchpad = torch.empty(
        (rows, scratchpad_columns(VOCAB)), dtype=torch.uint32, device="cuda"
    )
    sonic.sample(
        logits,
        state,
        slots,
        offsets,
        sampled,
        scratchpad,
        pick(VOCAB, rows),
        False,
    )
    assert torch.equal(tokens, sampled.view(-1))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_penalties_count_the_path_of_each_node(dtype):
    """Greedy rows at frequency penalty 2.0 over 10.0 > 9.5 > 9.0 > 8.5: each token on a row's path, or in its slot's decode counts, costs 2.0."""
    t0, t1, t2, t3 = 10, 20, 30, 40
    logits = torch.full((5, VOCAB), -5.0, device="cuda")
    logits[:, [t0, t1, t2, t3]] = torch.tensor([10.0, 9.5, 9.0, 8.5], device="cuda")
    logits = logits.to(dtype)
    state = _state(dtype, [Indicator.from_params(top_k=1, frequency=2.0)], 2.0)
    state.decode_counts[5, t2] = 1
    slots = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int64, device="cuda")
    offsets = torch.zeros(SLOTS, dtype=torch.int64, device="cuda")
    paths = torch.tensor(
        [[-1, -1], [t0, -1], [t0, t1], [t1, -1], [t0, t1]],
        dtype=torch.int32,
        device="cuda",
    )
    steps = torch.tensor([0, 1, 2, 1, 2], dtype=torch.int32, device="cuda")
    counts = state.decode_counts.clone()

    tokens = _draw(logits, state, slots, offsets, steps, paths)
    assert tokens.tolist() == [t0, t1, t2, t0, t3]
    assert torch.equal(state.decode_counts, counts)


def test_count_emitted_adds_each_request_emitted_tokens_only():
    state = SlotState.allocate(SLOTS, 1, VOCAB, 7, "cuda")
    slots = torch.tensor([2, 5, 7], dtype=torch.int64, device="cuda")
    tokens = torch.tensor(
        [11, 12, 13, 14, 21, 21, 23, 24, 31, 32, 33, 34],
        dtype=torch.int32,
        device="cuda",
    )
    lengths = torch.tensor([1, 3, 4], dtype=torch.int32, device="cuda")

    sonic.count_emitted(state, slots, tokens, lengths)
    expected = torch.zeros_like(state.decode_counts)
    expected[2, 11] = 1
    expected[5, 21] = 2
    expected[5, 23] = 1
    expected[7, [31, 32, 33, 34]] = 1
    assert torch.equal(state.decode_counts, expected)

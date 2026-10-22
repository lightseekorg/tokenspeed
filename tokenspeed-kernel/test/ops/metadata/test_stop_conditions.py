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

"""Stop conditions cut accepted widths where the control plane ends a request."""

from dataclasses import replace

import pytest
import torch
from tokenspeed_kernel.ops.metadata import apply_stop_conditions
from tokenspeed_kernel.ops.metadata.stop_conditions import (
    _apply_stop_conditions_kernel,
    _apply_stop_conditions_torch,
)
from tokenspeed_kernel.platform import CapabilityRequirement
from tokenspeed_kernel.registry import KernelRegistry, Priority, register_kernel
from tokenspeed_kernel.selection import select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature
from utils import assert_no_triton_compile

_STOP_IDS = 16
_STOP = 7


def test_ascend_selects_portable_fallback_and_prefers_device_plugin(
    monkeypatch, h100_platform
):
    # Keep the real built-in registration, but isolate device plugins and the
    # selection cache so the fallback is exercised even on plugin-enabled hosts.
    builtin = KernelRegistry.get().get_by_name("triton_apply_stop_conditions")
    registry = KernelRegistry()
    registry.register(
        builtin, KernelRegistry.get().get_impl("triton_apply_stop_conditions")
    )
    monkeypatch.setattr(KernelRegistry, "_instance", registry)
    ascend = replace(h100_platform, vendor="ascend", device_name="Ascend")
    signature = format_signature(req_pool_indices=dense_tensor_format(torch.int64))

    selected = select_kernel(
        "metadata", "apply_stop_conditions", signature, platform=ascend
    )
    assert selected.name == "triton_apply_stop_conditions"

    # CPU reference stands in for the unavailable device implementation; the
    # real registry must invalidate its cached fallback and prefer the plugin.
    register_kernel(
        "metadata",
        "apply_stop_conditions",
        name="test_ascend_stop_conditions",
        solution="test_ascend",
        capability=CapabilityRequirement(vendors=frozenset({"ascend"})),
        signatures=builtin.format_signatures,
        priority=Priority.PERFORMANT,
    )(_apply_stop_conditions_torch)
    selected = select_kernel(
        "metadata", "apply_stop_conditions", signature, platform=ascend
    )
    assert selected.name == "test_ascend_stop_conditions"
    state = _state(3)
    accepts = torch.tensor([3], dtype=torch.int32)
    selected(
        torch.tensor([1], dtype=torch.int64),
        torch.tensor([5, _STOP, 5], dtype=torch.int32),
        accepts,
        torch.tensor([3], dtype=torch.int32),
        torch.empty(0, dtype=torch.int32),
        *state,
        0,
        0,
        3,
        2,
    )
    assert accepts.tolist() == [2]
    assert state[3].tolist() == [False, True, False]


def _state(pool, device="cpu"):
    stop_token_ids = torch.full((pool, _STOP_IDS), -1, dtype=torch.int32)
    stop_token_ids[:, 0] = _STOP
    return (
        torch.zeros(pool, dtype=torch.int32),  # valid cache lengths
        stop_token_ids,
        torch.full((pool,), 1 << 30, dtype=torch.int32),  # token limits
        torch.zeros(pool, dtype=torch.bool),  # stopped
    )


def _apply(slots, tokens, accepts, inputs, prompts, state, *, extends, outputs, width):
    valid, stop_ids, limits, stopped = state
    apply_stop_conditions(
        slots,
        tokens,
        accepts,
        inputs,
        prompts,
        valid,
        stop_ids,
        limits,
        stopped,
        num_extends=extends,
        num_prefill_outputs=outputs,
        decode_width=width,
        padding_index=valid.shape[0] - 1,
    )


def test_decode_rows_keep_through_the_stop_token_and_within_the_budget():
    pool, width = 8, 4
    valid, stop_ids, limits, stopped = _state(pool)
    valid[1:5] = 10
    limits[3] = 13  # two more tokens fit after the unconsumed one
    stopped[4] = True  # stopped by the previous round
    slots = torch.tensor([1, 2, 3, 4, pool - 1], dtype=torch.int64)
    tokens = torch.tensor(
        [
            [5, _STOP, 5, 5],  # stop token at column 1
            [5, 5, 5, _STOP],  # stop token past the accepted width
            [5, 5, 5, 5],  # the budget ends it after two
            [5, 5, 5, 5],
            [_STOP, 5, 5, 5],  # padding row
        ],
        dtype=torch.int32,
    ).flatten()
    accepts = torch.tensor([4, 3, 4, 4, 4], dtype=torch.int32)
    inputs = torch.full((5,), width, dtype=torch.int32)

    _apply(
        slots,
        tokens,
        accepts,
        inputs,
        torch.empty(0, dtype=torch.int32),
        (valid, stop_ids, limits, stopped),
        extends=0,
        outputs=0,
        width=width,
    )

    assert accepts.tolist() == [2, 3, 2, 0, 4]
    assert stopped.tolist() == [False, True, False, True, True, False, False, False]


def test_only_a_prompt_completing_extend_row_stops():
    pool = 8
    valid, stop_ids, limits, stopped = _state(pool)
    valid[1:4] = torch.tensor([0, 4, 0], dtype=torch.int32)  # prefixes
    limits[3] = 7  # max_new_tokens = 1 on a 6-token prompt
    slots = torch.tensor([1, 2, 3], dtype=torch.int64)
    tokens = torch.tensor([_STOP, _STOP, 5], dtype=torch.int32)
    accepts = torch.ones(3, dtype=torch.int32)
    inputs = torch.tensor([8, 2, 6], dtype=torch.int32)
    prompts = torch.tensor([10, 6, 6], dtype=torch.int32)  # row 0 is mid-prompt

    _apply(
        slots,
        tokens,
        accepts,
        inputs,
        prompts,
        (valid, stop_ids, limits, stopped),
        extends=3,
        outputs=3,
        width=4,
    )

    assert accepts.tolist() == [1, 1, 1]
    assert stopped[1:4].tolist() == [False, True, True]


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the kernel runs on CUDA"
)


def _random_batch(batch, extends, outputs, width, pool, seed):
    gen = torch.Generator().manual_seed(seed)
    slots = torch.randperm(pool - 1, generator=gen)[:batch].to(torch.int64)
    slots[batch - batch // 7 :] = pool - 1
    valid, stop_ids, limits, stopped = _state(pool)
    stop_ids[:, 1] = torch.randint(0, 12, (pool,), generator=gen, dtype=torch.int32)
    valid[:] = torch.randint(0, 50, (pool,), generator=gen, dtype=torch.int32)
    limits[:] = valid + torch.randint(2, 8, (pool,), generator=gen, dtype=torch.int32)
    stopped[:] = torch.rand(pool, generator=gen) < 0.2
    num_tokens = outputs + (batch - extends) * width
    tokens = torch.randint(0, 12, (num_tokens,), generator=gen, dtype=torch.int32)
    accepts = torch.randint(1, width + 1, (batch,), generator=gen, dtype=torch.int32)
    accepts[:extends] = 1
    inputs = torch.full((batch,), width, dtype=torch.int32)
    inputs[:extends] = torch.randint(1, 9, (extends,), generator=gen, dtype=torch.int32)
    prompts = valid[slots[:extends]] + inputs[:extends]
    prompts[::2] += 3  # every other extend row is mid-prompt
    return slots, tokens, accepts, inputs, prompts, (valid, stop_ids, limits, stopped)


@requires_cuda
@pytest.mark.parametrize(
    ("batch", "extends", "outputs", "width"),
    ((9, 0, 0, 4), (12, 5, 3, 3), (6, 6, 6, 1), (1, 0, 0, 1), (200, 70, 70, 5)),
)
def test_kernel_matches_the_tensor_path(batch, extends, outputs, width):
    pool = max(64, batch + 2)
    cpu = _random_batch(batch, extends, outputs, width, pool, seed=batch)
    *cpu_args, cpu_state = cpu
    gpu_args = [t.cuda() for t in cpu_args]
    gpu_state = tuple(t.cuda() for t in cpu_state)

    _apply(*cpu_args, cpu_state, extends=extends, outputs=outputs, width=width)
    _apply(*gpu_args, gpu_state, extends=extends, outputs=outputs, width=width)
    torch.cuda.synchronize()

    assert torch.equal(gpu_args[2].cpu(), cpu_args[2])
    assert torch.equal(gpu_state[3].cpu(), cpu_state[3])


@requires_cuda
def test_kernel_compiles_once_across_batch_shapes():
    """Batch, extend and output counts change every round; none may key a new binary."""

    def run(batch, extends, outputs):
        args = _random_batch(batch, extends, outputs, 4, 512, seed=batch)
        *tensors, state = args
        _apply(
            *[t.cuda() for t in tensors],
            tuple(t.cuda() for t in state),
            extends=extends,
            outputs=outputs,
            width=4,
        )

    run(7, 3, 3)
    with assert_no_triton_compile(_apply_stop_conditions_kernel):
        for batch, extends, outputs in ((1, 0, 0), (1, 1, 1), (16, 16, 9), (17, 1, 0)):
            run(batch, extends, outputs)
        run(300, 33, 32)

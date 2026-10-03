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

from __future__ import annotations

from types import SimpleNamespace

import pytest
import tokenspeed_kernel.benchmark.generators.residual as residual_generator
import torch
from tokenspeed_kernel.benchmark.harness import BenchmarkRequest
from tokenspeed_kernel.registry import KernelRegistry, KernelSpec


@pytest.mark.parametrize(
    ("num_valid_blocks", "block_write", "has_delta", "storage_rows"),
    [(4, True, False, 8), (8, False, True, 8), (1, False, False, 1)],
)
def test_attn_res_generator_passes_runtime_block_storage(
    fresh_registry,
    monkeypatch,
    mi350_platform,
    num_valid_blocks,
    block_write,
    has_delta,
    storage_rows,
) -> None:
    _ = fresh_registry
    from tokenspeed_kernel.ops import residual as residual_ops

    KernelRegistry.get().register(
        KernelSpec(
            name="unit_attn_res",
            family="residual",
            mode="attn_res_fwd",
            solution="unit",
        ),
        lambda **_kwargs: None,
    )
    calls: list[tuple] = []

    def attn_res_fwd(layer_residual, blocks, *_args, delta, **kwargs):
        calls.append(
            (blocks.shape[0], delta is not None, kwargs, layer_residual.clone())
        )
        if delta is not None:
            layer_residual.add_(delta)

    monkeypatch.setattr(
        residual_generator,
        "_random_tensors",
        lambda _seed, *shapes: [torch.randn(s, dtype=torch.bfloat16) for s in shapes],
    )
    monkeypatch.setattr(residual_generator, "load_builtin_kernels", lambda: None)
    monkeypatch.setattr(
        residual_ops,
        "select_attn_res_fwd_kernel",
        lambda *_args, **_kwargs: (SimpleNamespace(name="unit_attn_res"), 0),
    )
    monkeypatch.setattr(residual_ops, "attn_res_fwd", attn_res_fwd)

    prepared = residual_generator.prepare_attn_res_fwd(
        BenchmarkRequest(
            family="residual",
            mode="attn_res_fwd",
            parameters={
                "model_profile": "kimi_k3_tp8",
                "tokens": 4,
                "hidden_size": 16,
                "block_slots": 8,
                "num_valid_blocks": num_valid_blocks,
                "block_write": block_write,
                "has_delta": has_delta,
                "eps": 1e-5,
                "dtype": "bfloat16",
            },
            solution=None,
            registration=None,
            cold_cache=True,
            seed=42,
        ),
        mi350_platform,
    )
    prepared.invocation.reset()
    prepared.invocation.invoke()
    # Restoring the residual keeps a delta from accumulating across calls.
    prepared.invocation.reset()
    prepared.invocation.invoke()

    rows, passes_delta, kwargs, residual = calls[-1]
    assert torch.equal(residual, calls[0][3])
    assert rows == storage_rows
    assert passes_delta is has_delta
    assert kwargs["num_valid_blocks"] == num_valid_blocks
    assert kwargs["block_write_idx"] == (num_valid_blocks if block_write else -1)
    assert prepared.registration.name == "unit_attn_res"

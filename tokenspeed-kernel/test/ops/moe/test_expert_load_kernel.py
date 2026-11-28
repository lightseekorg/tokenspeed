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

"""Tests for fused masked expert-load recording."""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.moe import record_expert_load
from utils import assert_no_triton_compile


def _reference_record_expert_load(
    topk_ids: torch.Tensor,
    live_rows: torch.Tensor,
    physical_load: torch.Tensor,
    recording_enabled: torch.Tensor | None = None,
) -> torch.Tensor:
    out = physical_load.clone()
    if topk_ids.shape[0] == 0 or topk_ids.shape[1] == 0:
        return out
    if recording_enabled is not None and not bool(recording_enabled.item()):
        return out
    valid = (topk_ids >= 0) & live_rows[: topk_ids.shape[0]].unsqueeze(1)
    out.scatter_add_(
        0,
        topk_ids.masked_fill(~valid, 0).reshape(-1).long(),
        valid.reshape(-1).to(out.dtype),
    )
    return out


@pytest.mark.parametrize("id_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("load_dtype", [torch.int32, torch.int64])
def test_record_expert_load_cpu_parity_and_validation(
    id_dtype: torch.dtype,
    load_dtype: torch.dtype,
) -> None:
    ids = torch.tensor(
        [[0, 2, -1, 1], [3, -1, 3, 2], [0, 1, 2, 3], [2, 2, 1, 0]],
        dtype=id_dtype,
    )
    # Non-contiguous row slice and trailing longer mask.
    ids_view = ids[:, ::2]
    live_rows = torch.tensor([True, False, True, False, True, False], dtype=torch.bool)
    load = torch.tensor([10, 0, 5, 1], dtype=load_dtype)
    expected = _reference_record_expert_load(ids_view, live_rows, load)

    record_expert_load(ids_view, live_rows, load)
    assert torch.equal(load, expected)

    paused = torch.tensor([False], dtype=torch.bool)
    record_expert_load(ids_view, live_rows, load, recording_enabled=paused)
    assert torch.equal(load, expected)

    enabled = torch.tensor(True, dtype=torch.bool)
    expected_after = _reference_record_expert_load(
        ids_view, live_rows, load, recording_enabled=enabled
    )
    record_expert_load(ids_view, live_rows, load, recording_enabled=enabled)
    assert torch.equal(load, expected_after)

    empty_ids = torch.empty((0, 4), dtype=id_dtype)
    record_expert_load(empty_ids, live_rows, load)
    assert torch.equal(load, expected_after)

    with pytest.raises(ValueError, match="live_rows"):
        record_expert_load(ids, live_rows[:2], load)
    with pytest.raises(ValueError, match="out of range"):
        record_expert_load(
            torch.tensor([[4, 0]], dtype=id_dtype),
            torch.tensor([True]),
            torch.zeros(4, dtype=load_dtype),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize("id_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("load_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("toggle_mode", ["none", "false", "true"])
def test_record_expert_load_gpu_matches_reference_and_captures_graph(
    id_dtype: torch.dtype,
    load_dtype: torch.dtype,
    toggle_mode: str,
) -> None:
    torch.manual_seed(17)
    rows, top_k, physical_experts = 97, 8, 48
    backing_ids = torch.randint(-2, physical_experts, (rows, top_k + 4), device="cuda")
    topk_ids = backing_ids[:, 1 : 1 + top_k].to(id_dtype)
    live_backing = (torch.arange(rows + 13, device="cuda") % 5) != 0
    live_rows = live_backing[3:]
    load_bank = torch.randint(0, 7, (3, physical_experts), dtype=load_dtype, device="cuda")
    physical_load = load_bank[1]
    toggle = (
        None
        if toggle_mode == "none"
        else torch.tensor(toggle_mode == "true", dtype=torch.bool, device="cuda")
    )

    expected = _reference_record_expert_load(topk_ids, live_rows, physical_load, toggle)
    record_expert_load(topk_ids, live_rows, physical_load, recording_enabled=toggle)
    assert torch.equal(physical_load, expected)

    # Empty forward is a safe no-op.
    record_expert_load(
        topk_ids[:0], live_rows, physical_load, recording_enabled=toggle
    )
    assert torch.equal(physical_load, expected)

    # Verify CUDA Graph capture & replay accumulate deterministically.
    initial = physical_load.clone()
    step_delta = expected - _reference_record_expert_load(
        topk_ids, live_rows, torch.zeros_like(physical_load), toggle
    )
    del step_delta
    one_step = _reference_record_expert_load(
        topk_ids, live_rows, torch.zeros_like(physical_load), toggle
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        record_expert_load(topk_ids, live_rows, physical_load, recording_enabled=toggle)
    physical_load.copy_(initial)
    graph.replay()
    graph.replay()
    assert torch.equal(physical_load, initial + 2 * one_step)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize("id_dtype", [torch.int32, torch.int64])
def test_record_expert_load_does_not_recompile_across_batches(
    id_dtype: torch.dtype,
) -> None:
    from tokenspeed_kernel.ops.moe._triton import expert_load

    num_experts = 64
    # Intentionally slice at unaligned offsets to exercise do_not_specialize_on_alignment.
    load_bank = torch.zeros(3 * num_experts + 5, dtype=torch.int64, device="cuda")
    ids_bank = torch.randint(
        0, num_experts, (1024, 20), dtype=id_dtype, device="cuda"
    )
    live_bank = torch.ones(1050, dtype=torch.bool, device="cuda")
    live_bank[::7] = False

    record_expert_load(ids_bank[:1, :8], live_bank[:1], load_bank[:num_experts])
    with assert_no_triton_compile(expert_load._record_expert_load_kernel):
        for offset, (rows, top_k) in enumerate(
            [(2, 8), (17, 6), (48, 8), (96, 8), (192, 12), (528, 8), (720, 16), (1024, 1)]
        ):
            load_start = (offset % 5) + 1
            layer_load = load_bank[load_start : load_start + num_experts]
            record_expert_load(
                ids_bank[:rows, 1 : 1 + top_k],
                live_bank[offset:],
                layer_load,
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_record_expert_load_handles_offsets_past_2pow31() -> None:
    free_bytes, _ = torch.cuda.mem_get_info()
    # Need >2 GiB int64 load pool + >8 GiB int32 strided ids pool (~10.5 GiB total).
    if free_bytes < 12 * (1 << 30):
        pytest.skip("requires >= 12 GiB free GPU memory for 2^31 offset test")

    # 1) Element & byte offset > 2^31 into physical_load (int64 elements * 8 bytes).
    big_num_experts = (1 << 28) + 8  # > 2^31 bytes from base
    big_load = torch.zeros(big_num_experts, dtype=torch.int64, device="cuda")
    tail_idx = big_num_experts - 3

    # 2) Element stride > 2^31 inside topk_ids (`row * stride_ids_m`).
    big_stride = (1 << 31) + 4
    id_pool = torch.zeros(big_stride + 8, dtype=torch.int32, device="cuda")
    id_pool[:4] = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device="cuda")
    id_pool[big_stride : big_stride + 4] = torch.tensor(
        [tail_idx, tail_idx, -1, 5], dtype=torch.int32, device="cuda"
    )
    strided_ids = torch.as_strided(id_pool, size=(2, 4), stride=(big_stride, 1))
    live_rows = torch.tensor([False, True], dtype=torch.bool, device="cuda")

    record_expert_load(strided_ids, live_rows, big_load)
    assert big_load[0].item() == 0
    assert big_load[5].item() == 1
    assert big_load[tail_idx].item() == 2

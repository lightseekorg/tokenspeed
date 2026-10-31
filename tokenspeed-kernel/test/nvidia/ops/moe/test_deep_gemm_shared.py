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

"""Checks for shared-expert weight layout, ownership and input staging."""

from dataclasses import replace
from typing import NamedTuple

import pytest
import torch
from tokenspeed_kernel.ops.moe import deep_gemm as mega_backend
from tokenspeed_kernel.ops.moe import moe_supports_shared_expert_fusion
from tokenspeed_kernel.ops.moe._deep_gemm.mega_moe_stage import (
    _stage_mxfp4_mega_moe_inputs_kernel,
    mega_moe_shared_sf_row,
    stage_mxfp4_mega_moe_inputs,
)
from tokenspeed_kernel.ops.moe.deep_gemm import (
    DeepGemmSharedExpertWeights,
    _DeepGemmMegaMoEState,
)
from tokenspeed_kernel.selection import kernel_override
from utils import assert_no_triton_compile

_HIDDEN_SIZE = 256
_SHARED_WIDTH = 2304


def _canonical_weights(seed):
    rng = torch.Generator().manual_seed(seed)
    h, i = _HIDDEN_SIZE, _SHARED_WIDTH
    l1 = torch.randint(1, 120, (2 * i, h), generator=rng, dtype=torch.uint8)
    l1[i:].add_(128)  # Distinct gate/up signs catch swapped halves.
    l2 = torch.randint(1, 120, (h, i), generator=rng, dtype=torch.uint8)
    s1 = torch.randint(100, 150, (2 * i, h // 32), generator=rng, dtype=torch.uint8)
    s2 = torch.randint(100, 150, (h, i // 32), generator=rng, dtype=torch.uint8)
    return (l1.view(torch.float8_e4m3fn), s1), (l2.view(torch.float8_e4m3fn), s2)


def _prepare(weights, routed_width, previous):
    return DeepGemmSharedExpertWeights.prepare(
        *weights,
        hidden_size=_HIDDEN_SIZE,
        routed_intermediate_size=routed_width,
        intermediate_size=_SHARED_WIDTH,
        previous=previous,
    )


def _tensor_bytes(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.contiguous().view(torch.uint8)


def _canonical_gate_up_rows(width: int) -> torch.Tensor:
    # The native matrix stores eight gate rows, then their eight up rows.
    gate_rows = torch.arange(width).reshape(-1, 8)
    physical_gate_rows = gate_rows + torch.arange(width // 8)[:, None] * 8
    return torch.cat((physical_gate_rows.flatten(), (physical_gate_rows + 8).flatten()))


def _decode_scales(packed: torch.Tensor) -> torch.Tensor:
    """Undo the 4-by-32 UTCCP permutation and unpack each int32 low byte first."""
    rows = packed.shape[0]
    physical_rows = torch.arange(rows).reshape(-1, 32, 4).transpose(1, 2).reshape(-1)
    words = packed[physical_rows].to(torch.int64)
    shifts = torch.tensor([0, 8, 16, 24])
    return ((words[:, :, None] >> shifts) & 255).to(torch.uint8).reshape(rows, -1)


def _assert_canonical_and_padding(
    prepared: DeepGemmSharedExpertWeights,
    canonical: tuple[
        tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]
    ],
) -> None:
    width = prepared.routed_intermediate_size
    hidden_size = prepared.hidden_size
    live_width = prepared.intermediate_size
    l1, l1_sf = prepared.l1_weights
    l2, l2_sf = prepared.l2_weights
    assert l1.dtype == l2.dtype == torch.float8_e4m3fn
    assert l1_sf.dtype == l2_sf.dtype == torch.int32
    assert l1_sf.stride() == (1, 2 * width)
    assert l2_sf.stride() == (1, hidden_size)

    gate_up_rows = _canonical_gate_up_rows(width)
    decoded_l1 = l1.view(torch.uint8)[gate_up_rows].reshape(2, width, hidden_size)
    decoded_l1_sf = _decode_scales(l1_sf)[gate_up_rows].reshape(
        2, width, hidden_size // 32
    )
    decoded_l2 = l2.view(torch.uint8)
    decoded_l2_sf = _decode_scales(l2_sf)
    assert torch.equal(
        decoded_l1[:, :live_width],
        canonical[0][0].view(torch.uint8).reshape(2, live_width, hidden_size),
    )
    assert torch.equal(
        decoded_l1_sf[:, :live_width],
        canonical[0][1].reshape(2, live_width, hidden_size // 32),
    )
    assert torch.equal(decoded_l2[:, :live_width], canonical[1][0].view(torch.uint8))
    assert torch.equal(decoded_l2_sf[:, : live_width // 32], canonical[1][1])
    assert torch.count_nonzero(decoded_l1[:, live_width:]).item() == 0
    assert torch.count_nonzero(decoded_l2[:, live_width:]).item() == 0
    assert torch.all(decoded_l1_sf[:, live_width:] == 127).item()
    assert torch.all(decoded_l2_sf[:, live_width // 32 :] == 127).item()


@pytest.mark.parametrize("routed_width", [2304, 2560])
def test_shared_weights_preserve_layout_and_reload_storage(routed_width):
    canonical = _canonical_weights(3)
    inputs = (*canonical[0], *canonical[1])
    snapshots = [_tensor_bytes(t).clone() for t in inputs]
    prepared = _prepare(canonical, routed_width, None)
    _assert_canonical_and_padding(prepared, canonical)
    for tensor, snapshot in zip(inputs, snapshots):
        assert torch.equal(_tensor_bytes(tensor), snapshot)
    derived = (*prepared.l1_weights, *prepared.l2_weights)
    pointers = [t.data_ptr() for t in derived]
    assert set(pointers).isdisjoint(t.data_ptr() for t in inputs)

    reloaded = _canonical_weights(29)
    inputs = (*reloaded[0], *reloaded[1])
    snapshots = [_tensor_bytes(t).clone() for t in inputs]
    assert _prepare(reloaded, routed_width, prepared) is prepared
    _assert_canonical_and_padding(prepared, reloaded)
    assert [t.data_ptr() for t in derived] == pointers
    for tensor, snapshot in zip(inputs, snapshots):
        assert torch.equal(_tensor_bytes(tensor), snapshot)

    snapshots = [_tensor_bytes(t).clone() for t in derived]
    other_width = 2560 if routed_width == 2304 else 2304
    with pytest.raises(ValueError, match="reload"):
        _prepare(reloaded, other_width, prepared)
    for tensor, snapshot in zip(derived, snapshots):
        assert torch.equal(_tensor_bytes(tensor), snapshot)


@pytest.mark.parametrize(
    "invalid", ["weight_dtype", "scale_dtype", "scale_layout", "width"]
)
def test_shared_weights_reject_incompatible_native_layout(invalid):
    l1, l2 = _canonical_weights(3)
    if invalid == "weight_dtype":
        l2 = (l2[0].to(torch.bfloat16), l2[1])
    elif invalid == "scale_dtype":
        l2 = (l2[0], l2[1].float())
    elif invalid == "scale_layout":
        l1 = (l1[0], l1[1][:, ::4])  # Group-128 scales cannot stand in for group-32.
    with pytest.raises(ValueError):
        _prepare((l1, l2), 2048 if invalid == "width" else 2560, None)


@pytest.mark.parametrize(
    "family,mode",
    [
        ("gemm", "mm"),
        ("activation", "silu_and_mul"),
        ("quantization", "fp8_with_scale"),
    ],
)
def test_operator_override_prevents_shared_fusion(family, mode):
    plan = {
        "apply_kernel_name": "deep_gemm_mxfp4_mega_moe_apply",
        "features": frozenset(),
        "input_dtype": torch.bfloat16,
        "internal_activation_dtype": "input",
    }
    assert moe_supports_shared_expert_fusion(plan)
    with kernel_override(family, mode, "test_override"):
        assert not moe_supports_shared_expert_fusion(plan)


class _ProcessGroup:
    def size(self):
        return 2


class _SharedInputWorkspace(NamedTuple):
    group: _ProcessGroup
    num_experts: int
    num_max_tokens_per_rank: int
    num_topk: int
    mma_type: str
    shared_l1_acts_sf: torch.Tensor | None


def test_shared_workspace_cache_and_native_input_layout(monkeypatch):
    # Only native allocation and its tile query are substituted; workspace
    # selection and scale-view selection run the production implementation.
    def allocate(
        group, experts, capacity, top_k, hidden, intermediate, *, num_shared_experts
    ):
        scales = torch.empty((hidden // 128, 512), dtype=torch.int32).t()
        return _SharedInputWorkspace(
            group,
            experts,
            128,
            top_k,
            "fp8xfp4",
            scales if num_shared_experts else None,
        )

    def tile(ranks, experts, capacity, count, top_k, mma_type, /):
        assert (ranks, experts, capacity, top_k, mma_type) == (2, 4, 128, 2, "fp8xfp4")
        return {0: 8, 1: 16, 33: 32}[count]

    monkeypatch.setattr(
        mega_backend, "get_symm_buffer_for_mega_moe", allocate, raising=False
    )
    monkeypatch.setattr(mega_backend, "get_block_m_for_mega_moe", tile, raising=False)
    monkeypatch.setattr(mega_backend, "_symm_buffer_cache", {})
    group = _ProcessGroup()
    unused = (torch.empty(0), torch.empty(0))
    state = _DeepGemmMegaMoEState(unused, unused, torch.device("cuda", 0))
    shared = _prepare(_canonical_weights(3), 2560, None)
    fused_state = replace(state, shared=shared)

    def workspace(prepared):
        return mega_backend._get_symm_buffer(
            state=prepared,
            process_group=group,
            num_experts=4,
            top_k=2,
            hidden_size=_HIDDEN_SIZE,
            intermediate_size=2560,
            max_num_tokens=100,
        )

    routed, fused = workspace(state), workspace(fused_state)
    assert routed is not fused
    assert routed.shared_l1_acts_sf is None
    assert workspace(state) is routed
    assert workspace(fused_state) is fused
    for tokens in (0, 1, 33):
        scales, block_m = mega_backend._shared_input_layout(shared, fused, tokens)
        assert scales is fused.shared_l1_acts_sf
        assert scales is not None
        assert block_m == {0: 8, 1: 16, 33: 32}[tokens]
        assert mega_backend._shared_input_layout(None, routed, tokens) == (None, 0)


def _reference_sf_rows(num_tokens: int, block_m: int) -> list[int]:
    """Enumerate native pages in physical column-major 4-by-32 tile order."""
    rows = [0] * num_tokens
    page_rows = ((block_m + 127) // 128) * 128
    for page in range((num_tokens + block_m - 1) // block_m):
        for tile in range(page_rows // 128):
            for column in range(32):
                for row in range(4):
                    logical_in_page = tile * 128 + row * 32 + column
                    token = page * block_m + logical_in_page
                    if logical_in_page < block_m and token < num_tokens:
                        rows[token] = page * page_rows + tile * 128 + column * 4 + row
    return rows


@pytest.mark.parametrize("block_m", (8, 96, 128, 192))
def test_shared_sf_addresses_cover_boundaries_and_capacity_without_aliasing(
    block_m: int,
) -> None:
    capacity = 4 * block_m + 129
    page_rows = ((block_m + 127) // 128) * 128
    capacity_rows = ((capacity + block_m - 1) // block_m) * page_rows
    counts = sorted(
        {0, 1, block_m - 1, block_m, block_m + 1, 127, 128, 129, capacity - 1, capacity}
    )
    scale_columns = _HIDDEN_SIZE // 128
    for num_tokens in counts:
        logical_rows = torch.arange(num_tokens, dtype=torch.int64)
        physical_rows = mega_moe_shared_sf_row(logical_rows, block_m)
        expected = torch.tensor(
            _reference_sf_rows(num_tokens, block_m), dtype=torch.int64
        )
        assert torch.equal(physical_rows, expected)
        assert torch.unique(physical_rows).numel() == num_tokens
        active_rows = ((num_tokens + block_m - 1) // block_m) * page_rows
        assert torch.all((physical_rows >= 0) & (physical_rows < active_rows)).item()
        # The native shared scale view is MN-major: columns are full pages apart.
        addresses = (
            physical_rows[:, None]
            + torch.arange(scale_columns)[None, :] * capacity_rows
        )
        assert torch.unique(addresses).numel() == num_tokens * scale_columns
        assert torch.all(
            (addresses >= 0) & (addresses < capacity_rows * scale_columns)
        ).item()
        for token in {
            0,
            1,
            block_m - 1,
            block_m,
            block_m + 1,
            127,
            128,
            129,
            capacity - 1,
        }:
            if token < num_tokens:
                assert mega_moe_shared_sf_row(token, block_m) == expected[token].item()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="requires NVIDIA GPU",
)
def test_shared_scale_store_handles_offsets_beyond_int32():
    if torch.cuda.mem_get_info()[0] < 10 * 1024**3:
        pytest.skip("large-stride scale storage needs 10 GiB of free GPU memory")
    x = torch.tensor([[1.0, 16.0, 256.0]], dtype=torch.bfloat16, device="cuda")
    x = x.repeat_interleave(128, dim=1)
    ids = torch.zeros((1, 1), dtype=torch.int64, device="cuda")
    weights = torch.ones((1, 1), dtype=torch.float32, device="cuda")
    x_fp8 = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    x_sf = torch.empty((1, 3), dtype=torch.int32, device="cuda")
    # The last column lies beyond 2**31 elements. Only three scale words are
    # touched; the large backing allocation is never filled or scanned.
    shared_sf = torch.empty_strided(
        (128, 3), (1, (1 << 30) + 128), dtype=torch.int32, device="cuda"
    )
    try:
        stage_mxfp4_mega_moe_inputs(
            x,
            weights,
            ids,
            x_fp8,
            x_sf,
            torch.empty_like(ids),
            torch.empty_like(weights),
            shared_l1_acts_sf=shared_sf,
            shared_block_m=16,
        )
        torch.testing.assert_close(shared_sf[0], x_sf[0], atol=0, rtol=0)
    finally:
        del shared_sf
        torch.cuda.empty_cache()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="requires NVIDIA GPU",
)
def test_shared_staging_reuses_warmed_native_tiles():
    import deep_gemm

    max_tokens, hidden, top_k = 4096, 256, 6

    def block_m(tokens):
        return deep_gemm.get_block_m_for_mega_moe(
            4, 384, max_tokens, tokens, top_k, "fp8xfp4"
        )

    x = torch.zeros((max_tokens, hidden), dtype=torch.bfloat16, device="cuda")
    ids = torch.zeros((max_tokens, top_k), dtype=torch.int64, device="cuda")
    weights = torch.ones((max_tokens, top_k), dtype=torch.float32, device="cuda")
    x_fp8 = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    x_sf = torch.empty((max_tokens, hidden // 128), dtype=torch.int32, device="cuda")
    ids_out, weights_out = torch.empty_like(ids), torch.empty_like(weights)
    # Reserve enough 128-row scale pages for every tile size in the sweep.
    shared_sf = torch.empty(
        (hidden // 128, max_tokens * 128), dtype=torch.int32, device="cuda"
    ).t()

    def stage(tokens, tile):
        stage_mxfp4_mega_moe_inputs(
            x[:tokens],
            weights[:tokens],
            ids[:tokens],
            x_fp8[:tokens],
            x_sf[:tokens],
            ids_out[:tokens],
            weights_out[:tokens],
            shared_l1_acts_sf=shared_sf,
            shared_block_m=tile,
        )

    # Warm exactly the buckets reached by the production warmup schedule.
    warmups = {}
    for tokens in mega_backend._warmup_m_values(max_tokens):
        if tokens:
            warmups.setdefault(block_m(tokens), tokens)
    for tile, tokens in warmups.items():
        stage(tokens, tile)

    with assert_no_triton_compile(_stage_mxfp4_mega_moe_inputs_kernel):
        for tokens in range(max_tokens + 1):
            stage(tokens, block_m(tokens))
    torch.cuda.synchronize()

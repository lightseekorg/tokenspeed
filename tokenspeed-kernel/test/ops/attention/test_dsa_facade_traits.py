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

"""The DSA facades read their selection traits off the tensors they are handed.

The index-key plane's dtype names its format (README, "Index-K plane
formats"), so a bf16 plane selects a leaf declaring ``index_k_format="bf16"``
and never an FP8 one; ``slot_order`` on the sparse cores is a trait plus a
keyword that only declaring cores receive. Fake leaves on CPU; no kernel runs.
"""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.attention import dsa as dsa_pkg
from tokenspeed_kernel.platform import Platform
from tokenspeed_kernel.registry import KernelRegistry, KernelSpec, Priority
from tokenspeed_kernel.selection import NoKernelFoundError
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

HEAD_DIM = 128
FP8_ROW_BYTES = HEAD_DIM + HEAD_DIM // 128 * 4


def _topk_signature():
    return frozenset(
        {
            format_signature(
                q=dense_tensor_format(torch.bfloat16),
                weights=dense_tensor_format(torch.float32),
            )
        }
    )


def _register_topk_leaf(mode: str, name: str, *, index_k_format: str, layouts):
    calls: list[dict] = []

    def leaf(**kwargs):
        calls.append(kwargs)
        tokens = kwargs["q"].shape[0]
        return (
            torch.full((tokens, int(kwargs["topk"])), -1, dtype=torch.int32),
            torch.zeros((tokens,), dtype=torch.int32),
        )

    spec = KernelSpec(
        name=name,
        family="attention",
        mode=mode,
        solution=name,
        format_signatures=_topk_signature(),
        traits={
            "head_dim": frozenset({HEAD_DIM}),
            "page_size": frozenset({64}),
            "index_k_format": frozenset({index_k_format}),
            "index_k_layout": frozenset(layouts),
        },
        features=frozenset({"batch_invariant", "forced_initial_local"}),
        priority=Priority.PORTABLE,
    )
    KernelRegistry.get().register(spec, leaf)
    return calls


@pytest.fixture
def topk_leaves(fresh_registry, h100_platform):
    _ = fresh_registry
    real_platform = Platform.get()
    Platform.override(h100_platform)
    fp8 = {
        mode: _register_topk_leaf(
            mode,
            f"fp8_{mode}",
            index_k_format="fp8_scaled",
            layouts=("packed", "page_planar"),
        )
        for mode in ("dsa_decode_topk", "dsa_prefill_topk")
    }
    bf16 = {
        mode: _register_topk_leaf(
            mode, f"bf16_{mode}", index_k_format="bf16", layouts=("packed",)
        )
        for mode in ("dsa_decode_topk", "dsa_prefill_topk")
    }
    yield fp8, bf16
    Platform.override(real_platform)


def _decode_topk(index_k_cache: torch.Tensor):
    return dsa_pkg.dsa_decode_topk(
        torch.zeros((2, 16, HEAD_DIM), dtype=torch.bfloat16),
        torch.zeros((2, 16), dtype=torch.float32),
        torch.tensor([64, 64], dtype=torch.int32),
        torch.zeros((2, 1), dtype=torch.int32),
        page_size=64,
        topk=4,
        softmax_scale=1.0,
        batch_invariant=True,
        index_k_cache=index_k_cache,
    )


def _prefill_topk(index_k_cache: torch.Tensor, **extra):
    return dsa_pkg.dsa_prefill_topk(
        torch.zeros((2, 16, HEAD_DIM), dtype=torch.bfloat16),
        torch.zeros((2, 16), dtype=torch.float32),
        torch.arange(16, dtype=torch.int64),
        torch.tensor([0, 0], dtype=torch.int32),
        torch.tensor([8, 16], dtype=torch.int32),
        topk=4,
        softmax_scale=1.0,
        batch_invariant=True,
        index_k_cache=index_k_cache,
        page_size=64,
        **extra,
    )


def test_bf16_plane_selects_the_bf16_leaf(topk_leaves):
    fp8, bf16 = topk_leaves
    plane = torch.zeros((128, HEAD_DIM), dtype=torch.bfloat16)
    _decode_topk(plane)
    _prefill_topk(plane)
    assert len(bf16["dsa_decode_topk"]) == 1
    assert len(bf16["dsa_prefill_topk"]) == 1
    assert bf16["dsa_decode_topk"][0]["index_k_cache"] is plane
    assert not fp8["dsa_decode_topk"] and not fp8["dsa_prefill_topk"]


def test_uint8_planes_select_the_fp8_leaf_by_layout(topk_leaves):
    fp8, bf16 = topk_leaves
    _decode_topk(torch.zeros((128, FP8_ROW_BYTES), dtype=torch.uint8))
    _prefill_topk(torch.zeros((2, 64, FP8_ROW_BYTES), dtype=torch.uint8))
    assert len(fp8["dsa_decode_topk"]) == 1 and len(fp8["dsa_prefill_topk"]) == 1
    assert not bf16["dsa_decode_topk"] and not bf16["dsa_prefill_topk"]
    # A page-planar plane never reaches a packed-only bf16 leaf either way.
    assert dsa_pkg._index_k_plane_traits(
        torch.zeros((2, 64, FP8_ROW_BYTES), dtype=torch.uint8), HEAD_DIM
    ) == {"index_k_format": "fp8_scaled", "index_k_layout": "page_planar"}


def test_bf16_plane_only_with_the_bf16_leaf_registered_is_the_only_match(
    fresh_registry, h100_platform
):
    _ = fresh_registry
    real_platform = Platform.get()
    Platform.override(h100_platform)
    try:
        _register_topk_leaf(
            "dsa_decode_topk",
            "fp8_only",
            index_k_format="fp8_scaled",
            layouts=("packed", "page_planar"),
        )
        # Honest labelling: a bf16 plane is never scored as FP8 bytes.
        with pytest.raises(NoKernelFoundError):
            _decode_topk(torch.zeros((128, HEAD_DIM), dtype=torch.bfloat16))
    finally:
        Platform.override(real_platform)


def test_unknown_plane_dtypes_and_shapes_are_refused(topk_leaves):
    with pytest.raises(TypeError, match="no registered format"):
        _decode_topk(torch.zeros((128, HEAD_DIM), dtype=torch.float16))
    with pytest.raises(ValueError, match="packed \\[slots, head_dim\\]"):
        _decode_topk(torch.zeros((2, 64, HEAD_DIM), dtype=torch.bfloat16))


def test_workspace_rows_are_fp8_scaled(topk_leaves):
    fp8, bf16 = topk_leaves
    dsa_pkg.dsa_prefill_topk(
        torch.zeros((2, 16, HEAD_DIM), dtype=torch.bfloat16),
        torch.zeros((2, 16), dtype=torch.float32),
        torch.arange(16, dtype=torch.int64),
        torch.tensor([0, 0], dtype=torch.int32),
        torch.tensor([8, 16], dtype=torch.int32),
        topk=4,
        softmax_scale=1.0,
        batch_invariant=True,
        index_k_fp8=torch.zeros((16, HEAD_DIM), dtype=torch.float8_e4m3fn),
        index_k_scale=torch.zeros((16, 1), dtype=torch.float32),
        page_size=64,
    )
    assert len(fp8["dsa_prefill_topk"]) == 1 and not bf16["dsa_prefill_topk"]


# --- sparse cores: slot_order -------------------------------------------------


def _register_core(mode: str, name: str, *, slot_orders, priority):
    calls: list[dict] = []

    def leaf(**kwargs):
        calls.append(kwargs)
        q = kwargs["q"]
        return torch.zeros((q.shape[0], q.shape[1], 512), dtype=q.dtype)

    traits = {
        "q_len": frozenset({1}),
        "kv_lora_rank": frozenset({512}),
        "qk_rope_head_dim": frozenset({64}),
        "has_kv_cache": frozenset({True}),
        "has_sparse_kv_cache": frozenset({False}),
        "logit_cap": frozenset({False}),
        "return_lse": frozenset({False}),
        "topk_layout": frozenset({"global_slots"}),
    }
    if slot_orders is not None:
        traits["slot_order"] = frozenset(slot_orders)
    spec = KernelSpec(
        name=name,
        family="attention",
        mode=mode,
        solution=name,
        format_signatures=frozenset(
            {format_signature(q=dense_tensor_format(torch.bfloat16))}
        ),
        traits=traits,
        priority=priority,
    )
    KernelRegistry.get().register(spec, leaf)
    return calls


@pytest.fixture
def cores(fresh_registry, h100_platform):
    _ = fresh_registry
    real_platform = Platform.get()
    Platform.override(h100_platform)
    silent = {
        mode: _register_core(
            mode, f"silent_{mode}", slot_orders=None, priority=Priority.PERFORMANT
        )
        for mode in ("dsa_decode", "dsa_prefill")
    }
    sorting = {
        mode: _register_core(
            mode,
            f"sorting_{mode}",
            slot_orders=("sorted", "selection"),
            priority=Priority.REFERENCE,
        )
        for mode in ("dsa_decode", "dsa_prefill")
    }
    yield silent, sorting
    Platform.override(real_platform)


def _run_core(mode: str, **extra):
    facade = dsa_pkg.dsa_decode if mode == "dsa_decode" else dsa_pkg.dsa_prefill
    return facade(
        q=torch.zeros((2, 8, 576), dtype=torch.bfloat16),
        kv_cache=torch.zeros((128, 576), dtype=torch.bfloat16),
        sparse_kv_cache=None,
        topk_slots=torch.full((2, 4), -1, dtype=torch.int32),
        topk_lens=torch.zeros((2,), dtype=torch.int32),
        max_seqlen_k=64,
        qk_nope_head_dim=128,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        softmax_scale=1.0,
        page_size=64,
        **extra,
    )


@pytest.mark.parametrize("mode", ["dsa_decode", "dsa_prefill"])
def test_selection_order_is_served_by_a_silent_core_without_the_keyword(cores, mode):
    silent, sorting = cores
    _run_core(mode, slot_order="selection")
    assert len(silent[mode]) == 1 and "slot_order" not in silent[mode][0]
    assert not sorting[mode]


@pytest.mark.parametrize("mode", ["dsa_decode", "dsa_prefill"])
def test_sorted_order_reaches_a_declaring_core_as_the_keyword(cores, mode):
    silent, sorting = cores
    _run_core(mode, slot_order="sorted", solution=f"sorting_{mode}")
    assert sorting[mode][0]["slot_order"] == "sorted"
    _run_core(mode, slot_order="selection", solution=f"sorting_{mode}")
    assert sorting[mode][1]["slot_order"] == "selection"
    assert not silent[mode]


@pytest.mark.parametrize("mode", ["dsa_decode", "dsa_prefill"])
def test_sorted_order_refuses_a_silent_core(cores, mode):
    silent, _ = cores
    with pytest.raises(ValueError, match="does not declare the slot_order"):
        _run_core(mode, slot_order="sorted")
    assert not silent[mode]
    with pytest.raises(ValueError, match="slot_order must be one of"):
        _run_core(mode, slot_order="shuffled")

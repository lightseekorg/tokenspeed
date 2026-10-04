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
and never an FP8 one; index keys handed as rows in workspace-row order
(``index_k_fp8`` + ``index_k_scale``, ``index_k_bf16``) select only leaves
declaring ``INDEX_K_WORKSPACE_ROWS_FEATURE`` for their format; ``slot_order``
on the top-k leaves and the sparse cores is a trait plus a keyword that only
declaring kernels receive. Fake leaves on CPU; no kernel runs.
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


def _register_topk_leaf(
    mode: str,
    name: str,
    *,
    index_k_format: str,
    layouts,
    workspace_rows: bool,
    slot_orders=None,
):
    calls: list[dict] = []

    def leaf(**kwargs):
        calls.append(kwargs)
        tokens = kwargs["q"].shape[0]
        return (
            torch.full((tokens, int(kwargs["topk"])), -1, dtype=torch.int32),
            torch.zeros((tokens,), dtype=torch.int32),
        )

    features = {"batch_invariant", "forced_initial_local"}
    if workspace_rows:
        features.add(dsa_pkg.INDEX_K_WORKSPACE_ROWS_FEATURE)
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
            **({"slot_order": frozenset(slot_orders)} if slot_orders else {}),
        },
        features=frozenset(features),
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
            workspace_rows=False,
        )
        for mode in ("dsa_decode_topk", "dsa_prefill_topk")
    }
    bf16 = {
        mode: _register_topk_leaf(
            mode,
            f"bf16_{mode}",
            index_k_format="bf16",
            layouts=("packed",),
            workspace_rows=False,
        )
        for mode in ("dsa_decode_topk", "dsa_prefill_topk")
    }
    yield fp8, bf16
    Platform.override(real_platform)


def _decode_topk(index_k_cache: torch.Tensor, slot_order: str = "selection", **extra):
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
        slot_order=slot_order,
        **extra,
    )


def _prefill_topk(index_k_cache: torch.Tensor, slot_order: str = "selection", **extra):
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
        slot_order=slot_order,
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
            workspace_rows=False,
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


def test_candidate_lens_cpu_reaches_only_leaves_declaring_the_feature(
    fresh_registry, h100_platform
):
    """An opaque ``*args, **kwargs`` wrapper (the AMD gluon registrations)
    that does not declare ``CANDIDATE_LENS_CPU_FEATURE`` never receives the
    keyword its launcher cannot take; a leaf declaring it does."""
    _ = fresh_registry
    real_platform = Platform.get()
    Platform.override(h100_platform)
    try:
        seen: dict[str, list] = {"silent": [], "declaring": []}

        def register(name: str, *, declares: bool):
            def launcher(**kwargs):
                seen[name].append(kwargs)
                tokens = kwargs["q"].shape[0]
                return (
                    torch.full((tokens, int(kwargs["topk"])), -1, dtype=torch.int32),
                    torch.zeros((tokens,), dtype=torch.int32),
                )

            def wrapper(*args, **kwargs):
                return launcher(*args, **kwargs)

            features = {"batch_invariant", "forced_initial_local"}
            if declares:
                features.add(dsa_pkg.CANDIDATE_LENS_CPU_FEATURE)
            KernelRegistry.get().register(
                KernelSpec(
                    name=name,
                    family="attention",
                    mode="dsa_prefill_topk",
                    solution=name,
                    format_signatures=_topk_signature(),
                    traits={"index_k_format": frozenset({"fp8_scaled"})},
                    features=frozenset(features),
                    priority=Priority.PORTABLE,
                ),
                wrapper,
            )

        register("silent", declares=False)
        register("declaring", declares=True)
        lens = torch.tensor([8, 16], dtype=torch.int64)
        plane = torch.zeros((128, FP8_ROW_BYTES), dtype=torch.uint8)
        _prefill_topk(plane, candidate_lens_cpu=lens, solution="silent")
        _prefill_topk(plane, candidate_lens_cpu=lens, solution="declaring")
        assert "candidate_lens_cpu" not in seen["silent"][0]
        assert seen["declaring"][0]["candidate_lens_cpu"] is lens
    finally:
        Platform.override(real_platform)


# --- top-k leaves: slot_order ------------------------------------------------


@pytest.fixture
def ordering_topk_leaves(fresh_registry, h100_platform):
    """A silent bf16 leaf and one declaring ``slot_order`` per top-k mode."""
    _ = fresh_registry
    real_platform = Platform.get()
    Platform.override(h100_platform)
    silent = {
        mode: _register_topk_leaf(
            mode,
            f"silent_{mode}",
            index_k_format="bf16",
            layouts=("packed",),
            workspace_rows=False,
        )
        for mode in ("dsa_decode_topk", "dsa_prefill_topk")
    }
    ordering = {
        mode: _register_topk_leaf(
            mode,
            f"ordering_{mode}",
            index_k_format="bf16",
            layouts=("packed",),
            workspace_rows=False,
            slot_orders=("sorted", "selection"),
        )
        for mode in ("dsa_decode_topk", "dsa_prefill_topk")
    }
    yield silent, ordering
    Platform.override(real_platform)


_TOPK_CALLS = {"dsa_decode_topk": _decode_topk, "dsa_prefill_topk": _prefill_topk}


@pytest.mark.parametrize("mode", ["dsa_decode_topk", "dsa_prefill_topk"])
def test_topk_selection_order_is_served_by_a_silent_leaf_without_the_keyword(
    ordering_topk_leaves, mode
):
    silent, _ = ordering_topk_leaves
    plane = torch.zeros((128, HEAD_DIM), dtype=torch.bfloat16)
    _TOPK_CALLS[mode](plane, slot_order="selection", solution=f"silent_{mode}")
    assert len(silent[mode]) == 1 and "slot_order" not in silent[mode][0]


@pytest.mark.parametrize("mode", ["dsa_decode_topk", "dsa_prefill_topk"])
def test_topk_sorted_order_reaches_a_declaring_leaf_as_the_keyword(
    ordering_topk_leaves, mode
):
    """Only the top-k leaf knows a slot's position, so the ascending-position
    order of ``"sorted"`` is its to emit: it receives the keyword."""
    _, ordering = ordering_topk_leaves
    plane = torch.zeros((128, HEAD_DIM), dtype=torch.bfloat16)
    _TOPK_CALLS[mode](plane, slot_order="sorted", solution=f"ordering_{mode}")
    assert ordering[mode][0]["slot_order"] == "sorted"
    _TOPK_CALLS[mode](plane, slot_order="selection", solution=f"ordering_{mode}")
    assert ordering[mode][1]["slot_order"] == "selection"


@pytest.mark.parametrize("mode", ["dsa_decode_topk", "dsa_prefill_topk"])
def test_topk_sorted_order_refuses_a_silent_leaf(ordering_topk_leaves, mode):
    """A leaf that cannot order by position is refused rather than letting the
    core sort by slot, which follows the page placement, not the positions."""
    plane = torch.zeros((128, HEAD_DIM), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="does not declare the slot_order"):
        _TOPK_CALLS[mode](plane, slot_order="sorted", solution=f"silent_{mode}")
    with pytest.raises(ValueError, match="slot_order must be one of"):
        _TOPK_CALLS[mode](plane, slot_order="random", solution=f"ordering_{mode}")


# --- workspace rows: index keys handed in workspace-row order -----------------

ROWS = 16
FP8_ROWS = {
    "index_k_fp8": torch.zeros((ROWS, HEAD_DIM), dtype=torch.float8_e4m3fn),
    "index_k_scale": torch.zeros((ROWS, HEAD_DIM // 128), dtype=torch.float32),
}
BF16_ROWS = {"index_k_bf16": torch.zeros((ROWS, HEAD_DIM), dtype=torch.bfloat16)}


def _prefill_topk_rows(**rows):
    # page_size names the page geometry a leaf pins (the stubs declare 64)
    # even though rows carry no plane; a leaf without the trait ignores it.
    return dsa_pkg.dsa_prefill_topk(
        torch.zeros((2, 16, HEAD_DIM), dtype=torch.bfloat16),
        torch.zeros((2, 16), dtype=torch.float32),
        torch.arange(ROWS, dtype=torch.int64),
        torch.tensor([0, 0], dtype=torch.int32),
        torch.tensor([8, 16], dtype=torch.int32),
        topk=4,
        softmax_scale=1.0,
        batch_invariant=True,
        page_size=64,
        **rows,
        slot_order="selection",
    )


@pytest.fixture
def row_leaves(fresh_registry, h100_platform):
    """Per format, one leaf that only reads planes and one that also takes
    rows in workspace-row order (``INDEX_K_WORKSPACE_ROWS_FEATURE``)."""
    _ = fresh_registry
    real_platform = Platform.get()
    Platform.override(h100_platform)
    leaves = {}
    for index_k_format, layouts in (
        ("fp8_scaled", ("packed", "page_planar")),
        ("bf16", ("packed",)),
    ):
        for takes_rows in (False, True):
            name = f"{index_k_format}_{'rows' if takes_rows else 'plane'}"
            leaves[name] = _register_topk_leaf(
                "dsa_prefill_topk",
                name,
                index_k_format=index_k_format,
                layouts=layouts,
                workspace_rows=takes_rows,
            )
    yield leaves
    Platform.override(real_platform)


def test_workspace_rows_select_the_declaring_leaf_of_their_format(row_leaves):
    _prefill_topk_rows(**FP8_ROWS)
    _prefill_topk_rows(**BF16_ROWS)
    assert len(row_leaves["fp8_scaled_rows"]) == 1
    assert len(row_leaves["bf16_rows"]) == 1
    assert not row_leaves["fp8_scaled_plane"] and not row_leaves["bf16_plane"]
    fp8_call = row_leaves["fp8_scaled_rows"][0]
    assert fp8_call["index_k_fp8"] is FP8_ROWS["index_k_fp8"]
    assert fp8_call["index_k_scale"] is FP8_ROWS["index_k_scale"]
    assert "index_k_bf16" not in fp8_call
    bf16_call = row_leaves["bf16_rows"][0]
    assert bf16_call["index_k_bf16"] is BF16_ROWS["index_k_bf16"]
    assert "index_k_fp8" not in bf16_call and "index_k_scale" not in bf16_call
    assert bf16_call["index_k_cache"] is None


def test_a_plane_only_leaf_never_receives_the_row_keywords(row_leaves):
    """The row keywords are routed by the registered feature, like
    ``candidate_lens_cpu``: a leaf without it is not handed them even when it
    is the one selected (here through the plane)."""
    _prefill_topk(torch.zeros((128, FP8_ROW_BYTES), dtype=torch.uint8))
    _prefill_topk(torch.zeros((128, HEAD_DIM), dtype=torch.bfloat16))
    for name in ("fp8_scaled_plane", "fp8_scaled_rows", "bf16_plane", "bf16_rows"):
        for call in row_leaves[name]:
            assert not {"index_k_fp8", "index_k_scale", "index_k_bf16"} & set(call)


def test_a_plane_only_leaf_is_never_selected_for_workspace_rows(
    fresh_registry, h100_platform
):
    """The failure for rows without a declaring leaf is at selection -- the
    plane-only leaf (what ``triton_dsa_prefill_topk_fp8`` is) never runs."""
    _ = fresh_registry
    real_platform = Platform.get()
    Platform.override(h100_platform)
    try:
        plane_only = _register_topk_leaf(
            "dsa_prefill_topk",
            "fp8_plane",
            index_k_format="fp8_scaled",
            layouts=("packed", "page_planar"),
            workspace_rows=False,
        )
        with pytest.raises(NoKernelFoundError):
            _prefill_topk_rows(**FP8_ROWS)
        with pytest.raises(NoKernelFoundError):
            _prefill_topk_rows(**BF16_ROWS)
        assert not plane_only
        # The plane itself still reaches it.
        _prefill_topk(torch.zeros((128, FP8_ROW_BYTES), dtype=torch.uint8))
        assert len(plane_only) == 1
    finally:
        Platform.override(real_platform)


@pytest.mark.parametrize("how", ["override", "solution", "env"])
def test_an_override_cannot_force_a_plane_only_leaf_onto_workspace_rows(
    row_leaves, monkeypatch, how
):
    """An override skips traits, not required features: forcing the plane-only
    leaf (``override=`` by name or the environment override, which name the
    missing feature; ``solution=``, which filters on it) for rows is a
    ``NoKernelFoundError``, never a keyword the leaf's launcher does not take."""
    if how == "env":
        monkeypatch.setenv(
            "TOKENSPEED_KERNEL_OVERRIDE_ATTENTION_DSA_PREFILL_TOPK",
            "fp8_scaled_plane",
        )
        forced = {}
    else:
        forced = {how: "fp8_scaled_plane"}
    message = "with solution" if how == "solution" else "index_k_workspace_rows"
    with pytest.raises(NoKernelFoundError, match=message):
        _prefill_topk_rows(**FP8_ROWS, **forced)
    assert not row_leaves["fp8_scaled_plane"]
    # The same override still serves the plane.
    _prefill_topk(torch.zeros((128, FP8_ROW_BYTES), dtype=torch.uint8), **forced)
    assert len(row_leaves["fp8_scaled_plane"]) == 1
    # Forcing the declaring leaf by override hands it the rows.
    monkeypatch.delenv(
        "TOKENSPEED_KERNEL_OVERRIDE_ATTENTION_DSA_PREFILL_TOPK", raising=False
    )
    _prefill_topk_rows(**FP8_ROWS, override="fp8_scaled_rows")
    assert row_leaves["fp8_scaled_rows"][0]["index_k_fp8"] is FP8_ROWS["index_k_fp8"]


def test_workspace_rows_come_in_one_format_and_without_a_plane(row_leaves):
    with pytest.raises(ValueError, match="two index-K formats"):
        _prefill_topk_rows(**FP8_ROWS, **BF16_ROWS)
    with pytest.raises(ValueError, match="two sources of index keys"):
        _prefill_topk_rows(
            **BF16_ROWS,
            index_k_cache=torch.zeros((128, HEAD_DIM), dtype=torch.bfloat16),
        )
    with pytest.raises(ValueError, match="two sources of index keys"):
        _prefill_topk_rows(
            **FP8_ROWS,
            index_k_cache=torch.zeros((128, FP8_ROW_BYTES), dtype=torch.uint8),
        )
    # Neither a plane nor rows: refused at the facade, not inside a leaf.
    with pytest.raises(ValueError, match="needs its index keys"):
        _prefill_topk_rows()
    with pytest.raises(ValueError, match="provided together"):
        _prefill_topk_rows(index_k_fp8=FP8_ROWS["index_k_fp8"])
    assert not any(row_leaves.values())


@pytest.mark.parametrize(
    "rows, message",
    [
        (
            {"index_k_bf16": torch.zeros((ROWS, HEAD_DIM), dtype=torch.uint8)},
            "index_k_bf16 holds bf16 rows",
        ),
        (
            {"index_k_bf16": torch.zeros((ROWS, HEAD_DIM // 2), dtype=torch.bfloat16)},
            "index_k_bf16 holds bf16 rows",
        ),
        (
            {"index_k_bf16": torch.zeros((ROWS + 1, HEAD_DIM), dtype=torch.bfloat16)},
            "index_k_bf16 holds bf16 rows",
        ),
        (
            {
                **FP8_ROWS,
                "index_k_fp8": torch.zeros((ROWS, HEAD_DIM), dtype=torch.int8),
            },
            "index_k_fp8 holds FP8 rows",
        ),
        (
            {
                **FP8_ROWS,
                "index_k_fp8": torch.zeros((ROWS + 1, HEAD_DIM), dtype=torch.uint8),
            },
            "index_k_fp8 holds FP8 rows",
        ),
        (
            {**FP8_ROWS, "index_k_scale": torch.zeros((ROWS, 1), dtype=torch.bfloat16)},
            "index_k_scale holds fp32 scales",
        ),
        (
            {**FP8_ROWS, "index_k_scale": torch.zeros((ROWS, 2), dtype=torch.float32)},
            "index_k_scale holds fp32 scales",
        ),
    ],
)
def test_workspace_rows_are_one_key_per_workspace_row_in_their_format(
    row_leaves, rows, message
):
    """Both forms are checked alike before selection: dtype, key width, and
    one row per entry of ``kv_workspace_slots``."""
    with pytest.raises(ValueError, match=message):
        _prefill_topk_rows(**rows)
    assert not any(row_leaves.values())
    # The FP8 bytes may come as uint8 (the gathered plane's bytes) as well.
    _prefill_topk_rows(
        **{**FP8_ROWS, "index_k_fp8": torch.zeros((ROWS, HEAD_DIM), dtype=torch.uint8)}
    )
    assert len(row_leaves["fp8_scaled_rows"]) == 1


def _probe(index_k_format: str, solution: str | None = None) -> str:
    return dsa_pkg.select_dsa_prefill_topk_for_rows(
        index_k_format=index_k_format,
        q_dtype=torch.bfloat16,
        weights_dtype=torch.float32,
        index_heads=16,
        head_dim=HEAD_DIM,
        topk=4,
        page_size=64,
        batch_invariant=True,
        solution=solution,
    )


def test_the_rows_selection_can_be_probed_without_running(row_leaves):
    """``select_dsa_prefill_topk_for_rows`` is the selection the rows form
    makes, for a host to run at construction: the declaring leaf of the
    format, a ``NoKernelFoundError`` without one, and no leaf runs."""
    assert _probe("fp8_scaled") == "fp8_scaled_rows"
    assert _probe("bf16") == "bf16_rows"
    assert _probe("bf16", solution="bf16_rows") == "bf16_rows"
    with pytest.raises(NoKernelFoundError):
        _probe("bf16", solution="bf16_plane")
    with pytest.raises(NoKernelFoundError):
        _probe("fp8_scaled", solution="fp8_scaled_plane")
    with pytest.raises(ValueError, match="workspace-row order"):
        _probe("fp16")
    assert not any(row_leaves.values())


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

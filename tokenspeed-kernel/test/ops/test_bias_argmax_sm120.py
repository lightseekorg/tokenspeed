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

"""Exact native-rounded IDs, metadata rejection, stream and cache contracts."""

from __future__ import annotations

from dataclasses import replace
from unittest import mock

import pytest
import torch
from tokenspeed_kernel._triton import triton
from tokenspeed_kernel.ops.sampling import bias_argmax as ops
from tokenspeed_kernel.ops.sampling import create_bias_argmax_workspace, try_bias_argmax
from tokenspeed_kernel.ops.sampling.triton import bias_argmax as kernels

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="requires NVIDIA SM120",
)
DTYPES = [torch.float16, torch.bfloat16, torch.float32]


def _expected(logits, bias, offset):
    return (logits + bias.to(logits.dtype)).max(dim=-1).indices + offset


def _check(logits, bias, workspace, output_dtype, offset, column):
    storage = torch.full((logits.shape[0], 5), -91, dtype=output_dtype, device="cuda")
    out = storage[:, column]
    before_logits, before_bias = logits.clone(), bias.clone()
    expected = _expected(logits, bias, offset).to(output_dtype)
    assert try_bias_argmax(logits, bias, out, workspace, offset)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    for other in range(5):
        if other != column:
            assert torch.all(storage[:, other] == -91)
    torch.testing.assert_close(logits, before_logits, rtol=0, atol=0, equal_nan=True)
    torch.testing.assert_close(bias, before_bias, rtol=0, atol=0, equal_nan=True)
    return storage


@pytest.mark.parametrize("logits_dtype", DTYPES)
@pytest.mark.parametrize("bias_dtype", DTYPES)
@pytest.mark.parametrize("rows,vocab", [(1, 1), (2, 4097), (8, 32771), (16, 262144)])
def test_native_dtype_matrix(logits_dtype, bias_dtype, rows, vocab):
    torch.manual_seed(42)
    logits = torch.randn((rows * 2, vocab * 2), dtype=logits_dtype, device="cuda")[
        ::2, ::2
    ]
    bias = torch.randn((vocab * 2, rows * 2), dtype=bias_dtype, device="cuda")[
        ::2, ::2
    ].T
    workspace = create_bias_argmax_workspace(rows, vocab, "cuda")
    _check(logits, bias, workspace, torch.int32, 123, 1)


@pytest.mark.parametrize("dtype", DTYPES)
def test_nan_ties_infinities_tail_and_signed_zero(dtype):
    logits = torch.full((9, 8197), -float("inf"), dtype=dtype, device="cuda")
    bias = torch.zeros_like(logits)
    logits[1, 4] = logits[1, 7001] = 8
    logits[2, 3] = logits[2, 6123] = float("inf")
    logits[3, 5001] = float("nan")
    logits[4, 2] = logits[4, 5001] = float("nan")
    logits[5, :] = float("nan")
    logits[6, 8196] = 6
    logits[7, :] = 0
    logits[7, 0] = -0.0
    logits[8, 3] = float("inf")
    bias[8, 3] = -float("inf")
    workspace = create_bias_argmax_workspace(9, 8197, "cuda")
    _check(logits, bias, workspace, torch.int64, 2**40, 3)


@pytest.mark.parametrize(
    "dtype,correction",
    [(torch.float16, [0.0003, 0.0004]), (torch.bfloat16, [0.003, 0.0031])],
)
def test_add_rounding_changes_winner(dtype, correction):
    logits = torch.ones((1, 2), dtype=dtype, device="cuda")
    bias = torch.tensor([correction], dtype=torch.float32, device="cuda")
    assert _expected(logits, bias, 0).item() == 0
    assert (logits.float() + bias.to(dtype).float()).argmax().item() == 1
    _check(logits, bias, create_bias_argmax_workspace(1, 2, "cuda"), torch.int32, 0, 1)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_bias_cast_rounding_changes_winner(dtype):
    logits = torch.full((1, 2), -1.0, dtype=dtype, device="cuda")
    # Bias conversion ties at 1; omitting it preserves two distinct small
    # results even if the subsequent addition still rounds to logits.dtype.
    bias = torch.tensor([[1.0002, 1.0003]], dtype=torch.float32, device="cuda")
    assert (logits.float() + bias).to(dtype).argmax().item() == 1
    assert _expected(logits, bias, 0).item() == 0
    _check(logits, bias, create_bias_argmax_workspace(1, 2, "cuda"), torch.int32, 0, 2)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_maximum_representable_offset(dtype):
    logits = torch.zeros((2, 4097), device="cuda")
    logits[:, -1] = 5
    bias = torch.zeros_like(logits)
    offset = torch.iinfo(dtype).max - 4096
    _check(
        logits, bias, create_bias_argmax_workspace(2, 4097, "cuda"), dtype, offset, 1
    )


@pytest.mark.parametrize("field", ["values", "indices"])
@pytest.mark.parametrize("malformation", ["rank", "dtype", "stride", "shape", "device"])
def test_malformed_workspace_without_writes(field, malformation):
    rows, vocab = 2, 8197
    logits = torch.randn((rows, vocab), device="cuda")
    bias = torch.zeros_like(logits)
    out = torch.full((rows,), -37, dtype=torch.int32, device="cuda")
    workspace = create_bias_argmax_workspace(rows, vocab, "cuda")
    tensor = getattr(workspace, field)
    if malformation == "rank":
        changed = tensor.flatten()
    elif malformation == "dtype":
        changed = tensor.to(torch.float16)
    elif malformation == "stride":
        changed = torch.empty(
            (rows, tensor.shape[1] * 2), dtype=tensor.dtype, device="cuda"
        )[:, ::2]
    elif malformation == "shape":
        changed = tensor[:1]
    else:
        changed = tensor.cpu()
    workspace = replace(workspace, **{field: changed})
    for scratch in (workspace.values, workspace.indices):
        scratch.fill_(123)
    before = [tensor.clone() for tensor in (out, workspace.values, workspace.indices)]
    with mock.patch.object(ops, "select_kernel") as selection:
        assert not try_bias_argmax(logits, bias, out, workspace, 0)
        selection.assert_not_called()
    for actual, expected in zip((out, workspace.values, workspace.indices), before):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "case",
    [
        "out_logits",
        "out_bias",
        "values_logits",
        "indices_bias",
        "scratch_overlap",
        "out_values",
        "out_indices",
    ],
)
def test_read_write_aliases_reject_without_writes(case):
    logits = torch.randn((2, 8197), device="cuda")
    bias = torch.zeros_like(logits)
    out = torch.full((2,), -91, dtype=torch.int32, device="cuda")
    workspace = create_bias_argmax_workspace(2, 8197, "cuda")
    if case == "out_logits":
        out = logits.view(torch.int32).flatten()[:2]
    elif case == "out_bias":
        out = bias.view(torch.int32).flatten()[:2]
    elif case == "values_logits":
        workspace = replace(workspace, values=logits.flatten()[:6].view(2, 3))
    elif case == "indices_bias":
        workspace = replace(
            workspace, indices=bias.view(torch.int32).flatten()[:6].view(2, 3)
        )
    elif case == "scratch_overlap":
        workspace = replace(workspace, indices=workspace.values.view(torch.int32))
    elif case == "out_values":
        out = workspace.values.view(torch.int32).flatten()[:2]
    else:
        out = workspace.indices.flatten()[:2]
    tensors = (logits, bias, out, workspace.values, workspace.indices)
    # Bytewise comparison also handles uninitialized scratch NaN payloads.
    before = [tensor.contiguous().view(torch.uint8).clone() for tensor in tensors]
    with mock.patch.object(ops, "select_kernel") as selection:
        assert not try_bias_argmax(logits, bias, out, workspace, 0)
        selection.assert_not_called()
    for actual, expected in zip(tensors, before):
        assert torch.equal(actual.contiguous().view(torch.uint8), expected)


def test_read_read_alias_is_legal():
    logits = torch.randn((2, 4097), device="cuda")
    _check(
        logits, logits, create_bias_argmax_workspace(2, 4097, "cuda"), torch.int64, 0, 0
    )


@pytest.mark.parametrize(
    "case",
    [
        "overflow",
        "negative_offset",
        "bool_offset",
        "zero_stride",
        "wrong_device",
        "none_workspace",
        "none_out",
        "wrong_out_dtype",
        "wrong_bias_shape",
    ],
)
def test_invalid_metadata_without_launch(case, monkeypatch):
    logits = torch.randn((2, 4097), device="cuda")
    bias = torch.zeros_like(logits)
    out = torch.full((2,), -91, dtype=torch.int32, device="cuda")
    workspace = create_bias_argmax_workspace(2, 4097, "cuda")
    offset = 0
    if case == "overflow":
        offset = torch.iinfo(torch.int32).max - 4095
    elif case == "negative_offset":
        offset = -1
    elif case == "bool_offset":
        offset = True
    elif case == "zero_stride":
        bias = bias[:1].expand(2, -1)
    elif case == "wrong_device":
        monkeypatch.setattr(
            torch.cuda, "current_device", lambda: logits.device.index + 1
        )
    elif case == "none_workspace":
        workspace = None
    elif case == "none_out":
        out = None
    elif case == "wrong_out_dtype":
        out = out.float()
    else:
        bias = bias[:, :-1]
    before = None if out is None else out.clone()
    with mock.patch.object(ops, "select_kernel") as selection:
        assert not try_bias_argmax(logits, bias, out, workspace, offset)
        selection.assert_not_called()
    if out is not None:
        torch.testing.assert_close(out, before, rtol=0, atol=0)


def test_cached_launchers_new_pointers_all_column_alignments(monkeypatch):
    monkeypatch.setattr(kernels, "_LAUNCHERS", {})
    workspace = create_bias_argmax_workspace(2, 8197, "cuda")
    alive = []
    for repeat in range(2):
        for column in range(4):
            logits = torch.full((2, 8197), -3.0, device="cuda")
            logits[:, 17 + repeat * 4 + column] = 8
            bias = torch.zeros_like(logits)
            storage = _check(logits, bias, workspace, torch.int32, 0, column)
            assert len(kernels._LAUNCHERS) == (column + 1 if repeat == 0 else 4)
            assert all(logits.data_ptr() != item[0].data_ptr() for item in alive)
            alive.append((logits, bias, storage))
    for case, (logits, bias, storage) in enumerate(alive):
        assert torch.all(storage[:, case % 4] == 17 + case)


def test_cached_launchers_follow_streams_graph_contents_and_buckets(monkeypatch):
    monkeypatch.setattr(kernels, "_LAUNCHERS", {})
    workspace = create_bias_argmax_workspace(16, 32771, "cuda")
    pointers = (workspace.values.data_ptr(), workspace.indices.data_ptr())
    captures = []
    for rows in (1, 16, 2, 8):
        logits = torch.zeros((rows, 32771), device="cuda")
        bias = torch.zeros_like(logits)
        storage = torch.full((rows, 5), -91, dtype=torch.int32, device="cuda")
        out = storage[:, 1]
        assert try_bias_argmax(logits, bias, out, workspace, 123)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            logits[:, 17] = 8
            assert try_bias_argmax(logits, bias, out, workspace, 123)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                assert try_bias_argmax(logits, bias, out, workspace, 123)
        torch.cuda.current_stream().wait_stream(stream)
        assert torch.all(out == 140)
        captures.append((graph, logits, bias, out))
    for step, bucket in enumerate((0, 2, 1, 3, 0, 1, 2, 3)):
        graph, logits, bias, out = captures[bucket]
        logits.normal_()
        bias.normal_()
        if step % 2:
            bias[:, 4099] = float("nan")
        expected = _expected(logits, bias, 123).to(out.dtype)
        graph.replay()
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
        assert pointers == (workspace.values.data_ptr(), workspace.indices.data_ptr())


def test_cache_keeps_dynamic_launch_hooks(monkeypatch):
    monkeypatch.setattr(kernels, "_LAUNCHERS", {})
    logits = torch.randn((2, 8197), device="cuda")
    bias = torch.zeros_like(logits)
    workspace = create_bias_argmax_workspace(2, 8197, "cuda")
    _check(logits, bias, workspace, torch.int32, 0, 1)
    seen = []

    def record(metadata):
        seen.append(metadata.get()["name"])

    monkeypatch.setattr(triton.knobs.runtime, "launch_enter_hook", record)
    _check(logits, bias, workspace, torch.int32, 0, 1)
    assert seen == ["_bias_argmax_partials", "_bias_argmax_finalize"]


def test_jit_pre_run_hooks_bypass_cache(monkeypatch):
    monkeypatch.setattr(kernels, "_LAUNCHERS", {})
    logits = torch.randn((2, 8197), device="cuda")
    bias = torch.zeros_like(logits)
    workspace = create_bias_argmax_workspace(2, 8197, "cuda")
    _check(logits, bias, workspace, torch.int32, 0, 1)
    seen = []

    def before(*args, **kwargs):
        seen.append(len(args))

    monkeypatch.setattr(kernels._bias_argmax_partials, "pre_run_hooks", [before])
    monkeypatch.setattr(kernels._bias_argmax_finalize, "pre_run_hooks", [before])
    _check(logits, bias, workspace, torch.int32, 0, 1)
    assert len(seen) == 2


@pytest.mark.parametrize(
    "group,setting,value",
    [
        ("runtime", "debug", True),
        ("runtime", "add_stages_inspection_hook", object()),
        ("compilation", "instrumentation_mode", "test-only-no-launch"),
        ("compilation", "fpsan_homomorphic_casts", True),
    ],
)
def test_instrumentation_bypasses_cache(group, setting, value, monkeypatch):
    monkeypatch.setattr(getattr(triton.knobs, group), setting, value)
    assert not kernels._can_reuse_launchers()


@pytest.mark.parametrize("field", ["logits", "bias", "out", "values", "indices"])
@pytest.mark.parametrize("transform", ["negative_view", "sparse_layout"])
def test_unresolved_views_and_sparse_layout_reject_without_selection(field, transform):
    logits = torch.randn((2, 4097), device="cuda")
    bias = torch.zeros_like(logits)
    out = torch.full((2,), -91, dtype=torch.int32, device="cuda")
    workspace = create_bias_argmax_workspace(2, 4097, "cuda")
    workspace.values.fill_(17)
    workspace.indices.fill_(23)
    named = {
        "logits": logits,
        "bias": bias,
        "out": out,
        "values": workspace.values,
        "indices": workspace.indices,
    }
    original = named[field]
    named[field] = (
        torch._neg_view(original)
        if transform == "negative_view"
        else original.to_sparse()
    )
    workspace = replace(workspace, values=named["values"], indices=named["indices"])
    before = {name: tensor.to_dense().clone() for name, tensor in named.items()}
    with mock.patch.object(ops, "select_kernel") as selection:
        assert not try_bias_argmax(
            named["logits"], named["bias"], named["out"], workspace, 0
        )
        selection.assert_not_called()
    for name, tensor in named.items():
        torch.testing.assert_close(tensor.to_dense(), before[name], rtol=0, atol=0)


def test_bias_cast_overflow_creates_first_nan():
    logits = torch.tensor([[-float("inf"), 0.0]], dtype=torch.float16, device="cuda")
    bias = torch.tensor([[100000.0, 1.0]], dtype=torch.float32, device="cuda")
    assert _expected(logits, bias, 0).item() == 0
    assert (logits.float() + bias).to(logits.dtype).argmax().item() == 1
    _check(logits, bias, create_bias_argmax_workspace(1, 2, "cuda"), torch.int32, 0, 1)


def test_active_jit_cache_hook_rejects_before_selection_without_writes(monkeypatch):
    logits = torch.randn((2, 4097), device="cuda")
    bias = torch.zeros_like(logits)
    out = torch.full((2,), -91, dtype=torch.int32, device="cuda")
    workspace = create_bias_argmax_workspace(2, 4097, "cuda")
    workspace.values.fill_(17)
    workspace.indices.fill_(23)
    tensors = (logits, bias, out, workspace.values, workspace.indices)
    before = [tensor.clone() for tensor in tensors]
    seen = []

    def suppress(**kwargs):
        seen.append(True)
        return True

    monkeypatch.setattr(triton.knobs.runtime, "jit_cache_hook", suppress)
    with mock.patch.object(ops, "select_kernel") as selection:
        assert not try_bias_argmax(logits, bias, out, workspace, 0)
        selection.assert_not_called()
    assert not seen
    for actual, expected in zip(tensors, before):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_interpreter_mode_rejects_before_selection_without_writes(monkeypatch):
    logits = torch.randn((2, 4097), device="cuda")
    bias = torch.zeros_like(logits)
    out = torch.full((2,), -91, dtype=torch.int32, device="cuda")
    workspace = create_bias_argmax_workspace(2, 4097, "cuda")
    workspace.values.fill_(17)
    workspace.indices.fill_(23)
    tensors = (logits, bias, out, workspace.values, workspace.indices)
    before = [tensor.clone() for tensor in tensors]
    monkeypatch.setattr(triton.knobs.runtime, "interpret", True)
    with mock.patch.object(ops, "select_kernel") as selection:
        assert not try_bias_argmax(logits, bias, out, workspace, 0)
        selection.assert_not_called()
    for actual, expected in zip(tensors, before):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_nested_logits_reject_before_shape_or_selection():
    logits = torch.nested.nested_tensor(
        [torch.randn(4097, device="cuda"), torch.randn(4096, device="cuda")]
    )
    bias = torch.zeros((2, 4097), device="cuda")
    out = torch.full((2,), -91, dtype=torch.int32, device="cuda")
    workspace = create_bias_argmax_workspace(2, 4097, "cuda")
    workspace.values.fill_(17)
    workspace.indices.fill_(23)
    logits_before = logits.to_padded_tensor(0).clone()
    tensors = (bias, out, workspace.values, workspace.indices)
    before = [tensor.clone() for tensor in tensors]
    with mock.patch.object(ops, "select_kernel") as selection:
        assert not try_bias_argmax(logits, bias, out, workspace, 0)
        selection.assert_not_called()
    torch.testing.assert_close(
        logits.to_padded_tensor(0), logits_before, rtol=0, atol=0
    )
    for actual, expected in zip(tensors, before):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

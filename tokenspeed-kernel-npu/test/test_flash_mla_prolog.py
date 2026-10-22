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


"""CPU tests for package discovery, admission, and the exact flash_ops call."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from tokenspeed_kernel_npu.ops import mla_prolog as adapter


@pytest.fixture(autouse=True)
def clear_loader_cache():
    loader = adapter._flash_mla_prolog
    loader.cache_clear()
    yield
    loader.cache_clear()


@pytest.mark.parametrize(
    "kind", ["missing", "dependency_error", "no_op", "old_schema", "supported"]
)
def test_package_discovery(monkeypatch, kind):
    loader = Mock()
    if kind in ("missing", "dependency_error"):
        loader.side_effect = ModuleNotFoundError(
            "missing module",
            name="flash_ops" if kind == "missing" else "broken_dependency",
        )
    monkeypatch.setattr(adapter.importlib, "import_module", loader)
    arguments = (
        []
        if kind == "old_schema"
        else [
            SimpleNamespace(name="enable_rope"),
            SimpleNamespace(name="ckvkr_repo_mode"),
        ]
    )
    op = Mock(_schema=SimpleNamespace(arguments=arguments))
    custom = SimpleNamespace()
    if kind != "no_op":
        custom.npu_mla_prolog_v3 = SimpleNamespace(default=op)
    monkeypatch.setattr(adapter.torch, "ops", SimpleNamespace(custom=custom))
    if kind == "dependency_error":
        with pytest.raises(ModuleNotFoundError, match="missing module"):
            adapter.mla_prolog_available()
        return
    assert adapter.mla_prolog_available() == (kind == "supported")
    assert adapter.mla_prolog_available() == (kind == "supported")
    loader.assert_called_once_with("flash_ops")


@pytest.fixture
def inputs(monkeypatch):
    device = SimpleNamespace(type="npu")

    def tensor(shape, dtype):
        return SimpleNamespace(
            shape=shape,
            ndim=len(shape),
            dtype=dtype,
            device=device,
            is_contiguous=lambda: True,
            new_empty=Mock(side_effect=lambda shape: torch.empty(shape)),
        )

    shapes = [
        (2, 3072),
        (3072, 1536),
        (1536, 6144),
        (32, 128, 512),
        (3072, 576),
        (1536,),
        (512,),
        (2, 64, 1, 576),
    ]
    args = [tensor(shape, torch.bfloat16) for shape in shapes]
    args.append(tensor((2,), torch.int64))
    monkeypatch.setitem(
        sys.modules, "torch_npu", SimpleNamespace(get_npu_format=lambda weight: 29)
    )
    return args


@pytest.mark.parametrize(
    "case",
    [
        "supported",
        "index_int32",
        "cpu",
        "empty",
        "hidden",
        "heads",
        "weight_shape",
        "dtype",
        "weight_stride",
        "cache_stride",
        "cache_page",
        "cache_width",
        "index_dtype",
        "index_shape",
        "index_device",
        "nz_format",
    ],
)
def test_metadata_admission(inputs, monkeypatch, case):
    args = inputs
    if case == "cpu":
        args[0].device = SimpleNamespace(type="cpu")
    elif case == "empty":
        args[0].shape = (0, 3072)
    elif case == "hidden":
        args[0].shape = (2, 1234)
    elif case == "heads":
        args[3].shape = (31, 128, 512)
    elif case == "weight_shape":
        args[1].shape = (1536, 3072)
    elif case == "dtype":
        args[0].dtype = torch.float32
    elif case == "weight_stride":
        args[3].is_contiguous = lambda: False
    elif case == "cache_stride":
        args[7].is_contiguous = lambda: False
    elif case == "cache_page":
        args[7].shape = (2, 17, 1, 576)
    elif case == "cache_width":
        args[7].shape = (2, 64, 1, 512)
    elif case == "index_dtype":
        args[8].dtype = torch.float32
    elif case == "index_int32":
        args[8].dtype = torch.int32
    elif case == "index_shape":
        args[8].shape = (1,)
    elif case == "index_device":
        args[8].device = SimpleNamespace(type="cpu")
    elif case == "nz_format":
        monkeypatch.setattr(
            sys.modules["torch_npu"], "get_npu_format", lambda weight: 2
        )
    assert adapter._supports_mla_prolog(*args) == (case in ("supported", "index_int32"))


def test_backend_int32_indices_are_widened_without_mutating_source(inputs, monkeypatch):
    op = Mock(return_value=("q", "q_aux", None, None, None))
    monkeypatch.setattr(adapter, "_flash_mla_prolog", lambda: op)
    source = inputs[8]
    source.dtype = torch.int32
    converted = torch.tensor([63, 64], dtype=torch.int64)
    source.to = Mock(return_value=converted)
    assert adapter.mla_prolog(
        *inputs, rmsnorm_epsilon_cq=1e-5, rmsnorm_epsilon_ckv=1e-5
    ) == ("q", "q_aux")
    source.to.assert_called_once_with(dtype=torch.int64)
    assert op.call_args.kwargs["cache_index"] is converted
    assert source.dtype == torch.int32


def test_exact_packaged_nope_call(inputs, monkeypatch):
    op = Mock(return_value=("q", "q_aux", None, None, None))
    monkeypatch.setattr(adapter, "_flash_mla_prolog", lambda: op)
    result = adapter.mla_prolog(
        *inputs, rmsnorm_epsilon_cq=1e-5, rmsnorm_epsilon_ckv=2e-5
    )
    assert result == ("q", "q_aux")
    positional, keywords = op.call_args
    assert positional[:7] == tuple(inputs[:7])
    assert positional[9] is inputs[7]
    assert positional[10] is not inputs[7]
    assert positional[7].shape == (2, 64)
    assert keywords["cache_index"] is inputs[8]
    assert keywords["enable_rope"] is False
    assert keywords["ckvkr_repo_mode"] == 1
    assert keywords["cache_mode"] == "PA_BSND"
    assert keywords["rmsnorm_epsilon_ckv"] == 2e-5
    assert keywords["weight_quant_mode"] == keywords["kv_cache_quant_mode"] == 0


def test_query_norm_can_be_returned_for_indexer_reuse(inputs, monkeypatch):
    op = Mock(return_value=("q", "q_aux", None, "q_norm", None))
    monkeypatch.setattr(adapter, "_flash_mla_prolog", lambda: op)
    result = adapter.mla_prolog(
        *inputs,
        rmsnorm_epsilon_cq=1e-5,
        rmsnorm_epsilon_ckv=2e-5,
        return_query_norm=True,
    )
    assert result == ("q", "q_aux", "q_norm")
    assert op.call_args.kwargs["query_norm_flag"] is True


def test_large_lite_metadata_admission(inputs):
    inputs[0].shape = (2, 4096)
    inputs[1].shape = (4096, 1536)
    inputs[2].shape = (1536, 64 * 192)
    inputs[3].shape = (64, 128, 512)
    inputs[4].shape = (4096, 576)
    assert adapter._supports_mla_prolog(*inputs)


def test_ineligible_inputs_do_not_write(inputs, monkeypatch):
    op = Mock()
    monkeypatch.setattr(adapter, "_flash_mla_prolog", lambda: op)
    inputs[7].is_contiguous = lambda: False
    assert (
        adapter.mla_prolog(*inputs, rmsnorm_epsilon_cq=1e-5, rmsnorm_epsilon_ckv=1e-5)
        is None
    )
    op.assert_not_called()


def test_missing_package_keeps_fallback(inputs, monkeypatch):
    monkeypatch.setattr(adapter, "_flash_mla_prolog", lambda: None)
    assert (
        adapter.mla_prolog(*inputs, rmsnorm_epsilon_cq=1e-5, rmsnorm_epsilon_ckv=1e-5)
        is None
    )


def test_operator_errors_are_not_retried(inputs, monkeypatch):
    op = Mock(side_effect=RuntimeError("operator failure"))
    monkeypatch.setattr(adapter, "_flash_mla_prolog", lambda: op)
    with pytest.raises(RuntimeError, match="operator failure"):
        adapter.mla_prolog(*inputs, rmsnorm_epsilon_cq=1e-5, rmsnorm_epsilon_ckv=1e-5)
    op.assert_called_once()


@pytest.mark.parametrize("heads", [4, 8, 16, 32, 64])
def test_local_head_prolog_admission(inputs, heads):
    inputs[2].shape = (1536, heads * 192)
    inputs[3].shape = (heads, 128, 512)
    assert adapter._supports_mla_prolog(*inputs)

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

"""Loader checks run on CPU with --noconftest; warning checks use kernel fixtures."""

import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch


@pytest.fixture
def native_loader(tmp_path, monkeypatch):
    # An independent module keeps synthetic failures out of the serving caches.
    path = (
        Path(__file__).resolve().parents[2]
        / "python/tokenspeed_kernel/thirdparty/cuda/dsv4_attention.py"
    )
    spec = importlib.util.spec_from_file_location("_test_native_topk", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "_objs_dir", lambda: tmp_path)
    return module


@pytest.mark.parametrize(
    ("case", "reason"),
    [
        ("available", None),
        ("missing_file", "missing native library"),
        ("load_error", "native module load failed (OSError)"),
        ("missing_symbol", "missing export dsv4_persistent_topk"),
        ("missing_ffi", "native module load failed (ModuleNotFoundError)"),
    ],
)
def test_native_availability(native_loader, tmp_path, monkeypatch, case, reason):
    artifact = tmp_path / "dsv4_attention/dsv4_attention.so"
    artifact.parent.mkdir()
    if case != "missing_file":
        artifact.touch()
    ffi = ModuleType("tvm_ffi")
    native = ModuleType("native_topk")
    if case != "missing_symbol":
        native.dsv4_persistent_topk = lambda *args: None

    def load_module(path):
        if case == "load_error":
            raise OSError("private deployment path")
        return native

    ffi.load_module = load_module
    monkeypatch.setitem(sys.modules, "tvm_ffi", None if case == "missing_ffi" else ffi)
    result = native_loader.persistent_topk_unavailable_reason()
    assert native_loader.has_persistent_topk() == (reason is None)
    if reason is None:
        assert result is None
    else:
        assert reason in result
        assert "private deployment path" not in result
    # Make a repair visible: both the success and failure result must stay cached.
    artifact.touch()
    native.dsv4_persistent_topk = lambda *args: None
    ffi.load_module = lambda path: native
    monkeypatch.setitem(sys.modules, "tvm_ffi", ffi)
    assert native_loader.persistent_topk_unavailable_reason() == result


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="kernel imports require the GPU test environment",
)
def test_warning_once_without_consuming_suppressed_calls(
    native_loader, h100_platform, monkeypatch, caplog
):
    from tokenspeed_kernel.ops.attention.qsa import cuda
    from tokenspeed_kernel.selection import kernel_override
    from tokenspeed_kernel.thirdparty.cuda import dsv4_attention

    # Restore the original callable/flag afterwards; do not clear shared caches.
    monkeypatch.setattr(
        dsv4_attention,
        "persistent_topk_unavailable_reason",
        native_loader.persistent_topk_unavailable_reason,
    )
    monkeypatch.setattr(cuda, "_topk_availability_reported", False)
    monkeypatch.setattr(cuda, "current_platform", lambda: h100_platform)
    monkeypatch.delenv(
        "TOKENSPEED_KERNEL_OVERRIDE_ATTENTION_QSA_BLOCK_TOPK_SELECT", raising=False
    )
    caplog.set_level(logging.WARNING, logger=cuda.__name__)
    caplog.clear()  # Ignore unrelated backend warnings from the initial import.
    with kernel_override("attention", "qsa_block_topk_select", "triton"):
        cuda.warn_if_cuda_topk_unavailable(
            torch.float32, 512, solution="cuda", override="cuda"
        )
    assert not caplog.records
    with kernel_override("attention", "qsa_block_topk_select", "cuda"):
        for topk in (512, 1024):
            cuda.warn_if_cuda_topk_unavailable(
                torch.float32, topk, solution=None, override=None
            )
    assert len(caplog.records) == 1
    assert "missing native library" in caplog.text
    assert "dsv4_attention CUDA group" in caplog.text

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

"""CPU checks of the Ascend dual-norm output and aliasing contract."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn


class ReferenceNorm(nn.Module):
    def __init__(self, width, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.linspace(0.5, 1.5, width))
        self.variance_epsilon = eps

    def forward(self, x, *, out):
        value = nn.functional.rms_norm(
            x.float(), (x.shape[-1],), self.weight, self.variance_epsilon
        )
        out.copy_(value.to(x.dtype))
        return out


@pytest.fixture
def dual_norm():
    # Load the actual class without bootstrapping accelerator-only modules.
    source = (
        Path(__file__).resolve().parents[3]
        / "python/tokenspeed/runtime/layers/layernorm.py"
    )
    cls = next(
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "FusedRMSNorm"
    )
    namespace = dict(
        torch=torch,
        nn=nn,
        RMSNorm=ReferenceNorm,
        _platform=SimpleNamespace(is_npu=True),
        _is_amd=False,
    )
    exec(
        compile(ast.Module(body=[cls], type_ignores=[]), str(source), "exec"), namespace
    )
    return namespace["FusedRMSNorm"](ReferenceNorm(24, 1e-5), ReferenceNorm(16, 1e-3))


@pytest.mark.parametrize(
    "separate_q,separate_kv",
    [(False, False), (True, False), (False, True), (True, True)],
)
@pytest.mark.parametrize("rows", [0, 3])
@torch.no_grad()
def test_ascend_dual_norm_preserves_slices_and_output_buffers(
    dual_norm, separate_q, separate_kv, rows
):
    torch.manual_seed(7)
    packed = torch.randn(rows, 48)
    original = packed.clone()
    q, kv = packed[:, :24], packed[:, 24:40]
    q_out = torch.empty_like(q) if separate_q else None
    kv_out = torch.empty_like(kv) if separate_kv else None
    expected_q = nn.functional.rms_norm(
        original[:, :24], (24,), dual_norm.weight_q_a, 1e-5
    )
    expected_kv = nn.functional.rms_norm(
        original[:, 24:40], (16,), dual_norm.weight_kv_a, 1e-3
    )

    returned_q, returned_kv = dual_norm(q, kv, q_out, kv_out)

    assert returned_q is q and returned_kv is kv
    torch.testing.assert_close(q if q_out is None else q_out, expected_q)
    torch.testing.assert_close(kv if kv_out is None else kv_out, expected_kv)
    torch.testing.assert_close(packed[:, 40:], original[:, 40:])
    if separate_q:
        torch.testing.assert_close(q, original[:, :24])
    if separate_kv:
        torch.testing.assert_close(kv, original[:, 24:40])

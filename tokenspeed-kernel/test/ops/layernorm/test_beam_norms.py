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

"""Which kernel ``layernorm.qk_rmsnorm_ssmax`` runs, decided on the host.

CPU only, nothing launches: ranked selection runs the fused Triton kernel exactly
where its host guard admits the call (Beam's TP1 and TP8 head pairs on NVIDIA SM100
to SM103), the torch kernel everywhere else (other TP sizes and platforms), and a
by-name call outside the guard raises instead of falling back. The kernels' byte
equality is ``test/nvidia/ops/layernorm/test_beam_norms_cuda.py``.
"""

from __future__ import annotations

import dataclasses
import importlib

import pytest
import torch
from tokenspeed_kernel.ops.layernorm import gluon
from tokenspeed_kernel.ops.layernorm import qk_rmsnorm_ssmax as qk
from tokenspeed_kernel.platform import ArchVersion, Platform
from tokenspeed_kernel.registry import KernelRegistry, load_builtin_kernels

EPS = 1e-6  # Beam rms_norm_eps
BF16 = torch.bfloat16


@pytest.fixture(autouse=True)
def beam_norm_kernels():
    """The operator registered (an earlier module may have reset the registry); no
    cached selection leaks between tests."""
    if KernelRegistry.get().get_by_name(qk.TORCH_QK_RMSNORM_SSMAX) is None:
        load_builtin_kernels()
        importlib.import_module(qk.__name__)
    KernelRegistry.get().clear_cache()
    yield
    KernelRegistry.get().clear_cache()


class _OnCuda:
    """A CPU tensor whose metadata reports a CUDA tensor: selection and the host guards
    read metadata only, so the CUDA case is decided on a host without a GPU."""

    is_cuda = True

    def __init__(self, tensor: torch.Tensor) -> None:
        self._tensor = tensor

    def __getattr__(self, name):
        return getattr(self._tensor, name)


def _on_cuda(arguments: tuple) -> tuple:
    return tuple(_OnCuda(a) if isinstance(a, torch.Tensor) else a for a in arguments)


@pytest.fixture
def platforms(b200_platform, b300_platform, h100_platform, mi350_platform):
    """(platform, in the fused kernels' range NVIDIA SM100 to SM103)."""
    sm107 = dataclasses.replace(b200_platform, arch_version=ArchVersion(10, 7))
    return (
        (b200_platform, True),
        (b300_platform, True),
        (h100_platform, False),
        (sm107, False),
        (mi350_platform, False),
    )


# ---------------------------------------------------------------------------------------
# layernorm.qk_rmsnorm_ssmax
# ---------------------------------------------------------------------------------------


def _qk(tokens=8, heads=(6, 1), head_dim=128, eps=EPS, change=None):
    """Beam's ``qkv.split`` views and ``[head_dim]`` weights, with one ``change``."""
    q_size, kv_size = heads[0] * head_dim, heads[1] * head_dim
    qkv = torch.zeros(tokens, q_size + 2 * kv_size, dtype=BF16)
    q, k, _ = qkv.split([q_size, kv_size, kv_size], dim=-1)
    q_weight = torch.zeros(head_dim, dtype=BF16)
    positions = torch.zeros(tokens, dtype=torch.int64)
    if change == "3d":
        q, k = q.reshape(tokens, heads[0], head_dim), k.reshape(
            tokens, heads[1], head_dim
        )
    elif change == "strided":
        q = torch.zeros(tokens, 2 * q_size, dtype=BF16)[:, ::2]
    elif change == "overlapping_rows":
        q = torch.zeros(tokens * q_size, dtype=BF16).as_strided(
            (tokens, q_size), (64, 1)
        )
    elif change == "strided_weight":
        q_weight = torch.zeros(2 * head_dim, dtype=BF16)[::2]
    elif change == "strided_positions":
        positions = torch.zeros(2 * tokens, dtype=torch.int64)[::2]
    elif change == "int32_positions":
        positions = positions.int()
    return q, k, q_weight, q_weight.clone(), eps, positions


QK_CASES = {
    "beam_tp1": (_qk(heads=(48, 8)), True),
    "beam_tp8": (_qk(heads=(6, 1)), True),
    "beam_tp4": (_qk(heads=(12, 2)), False),
    "beam_tp16": (_qk(heads=(3, 1)), False),
    "int32_positions": (_qk(change="int32_positions"), True),
    "tokens_0": (_qk(tokens=0), False),
    "cross_pair": (_qk(heads=(48, 1)), False),
    "head_dim_64": (_qk(head_dim=64), False),
    "3d": (_qk(change="3d"), False),
    "strided": (_qk(change="strided"), False),
    "overlapping_rows": (_qk(change="overlapping_rows"), False),
    "strided_weight": (_qk(change="strided_weight"), False),
    "strided_positions": (_qk(change="strided_positions"), False),
    "eps_zero": (_qk(eps=0.0), False),
    "eps_int": (_qk(eps=1), False),
}


@pytest.mark.parametrize("case", QK_CASES)
def test_qk_ranked_selection_runs_triton_exactly_where_its_guard_admits(
    case, platforms, monkeypatch
):
    arguments, admitted = QK_CASES[case]
    for on_cuda in (True, False):
        call = _on_cuda(arguments) if on_cuda else arguments
        for platform, in_range in platforms:
            monkeypatch.setattr(Platform, "_instance", platform)
            KernelRegistry.get().clear_cache()  # the cache key omits the vendor
            guard = qk.qk_rmsnorm_ssmax_rejection(*call)
            assert (guard == "") is (admitted and on_cuda and in_range), guard
            for dequant in (False, True):
                selected = qk.select_qk_rmsnorm_ssmax_kernel(
                    *call, dequant=dequant, solution=None, override=None
                )
                amd_fused = (
                    platform.is_cdna4
                    and on_cuda
                    and not dequant
                    and case in ("beam_tp8", "int32_positions")
                )
                assert selected.name == (
                    "gluon_qk_rmsnorm_ssmax_gfx950"
                    if amd_fused
                    else (
                        qk.TRITON_FUSED_QK_RMSNORM_SSMAX
                        if guard == ""
                        else qk.TORCH_QK_RMSNORM_SSMAX
                    )
                ), (platform.device_name, on_cuda, dequant)


@pytest.mark.parametrize(
    "change", ["dtype", "positions", "eps", "rows", "shape", "dequant"]
)
def test_qk_by_name_rejects_before_any_device_launch(
    change, mi350_platform, monkeypatch
):
    """Each case raises for its own reason: CUDA metadata and the gfx950
    platform admit the default arguments, so only ``change`` can reject."""
    monkeypatch.setattr(Platform, "_instance", mi350_platform)
    args = list(_qk())
    extra = {}
    if change == "dtype":
        args[0] = args[0].float()
    elif change == "positions":
        args[5] = args[5].float()
    elif change == "eps":
        args[4] = 0.0
    elif change == "rows":
        args = list(_qk(tokens=0))
    elif change == "shape":
        args[0] = args[0][:, :640]
    else:
        extra["dequant"] = object()
    with pytest.raises(ValueError):
        gluon.gluon_qk_rmsnorm_ssmax_gfx950(*_on_cuda(tuple(args)), **extra)

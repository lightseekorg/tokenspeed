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

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("deep_gemm")

from tokenspeed_kernel import (
    dsv4_mega_moe_apply,
    dsv4_mega_moe_plan,
    dsv4_mega_moe_process_weights,
    dsv4_mega_moe_warmup,
)
from tokenspeed_kernel.ops.moe.deep_gemm import dsv4_mega_moe as ops
from tokenspeed_kernel.thirdparty.deep_gemm.mega_moe_bf16 import (
    _BF16_SWIGLU,
    _HEADER,
    _SWIGLU,
    _make_include_overlay,
    _patch_swiglu,
)


def _quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = torch.exp2(torch.ceil(torch.log2(x.abs().amax() / 448)))
    return (x / scale).to(torch.float8_e4m3fn), scale


@pytest.mark.parametrize("weight", [0.011372136883437634, 0.01872287318110466])
def test_bf16_rounding_changes_fp8_payload_or_scale(weight: float) -> None:
    weighted = F.silu(torch.tensor(1.0)) * weight
    raw, raw_scale = _quantize(weighted)
    rounded, rounded_scale = _quantize(weighted.bfloat16().float())
    assert raw.float() != rounded.float() or raw_scale != rounded_scale
    # Rounding before weighting is not the reference operation either.
    early, early_scale = _quantize(
        F.silu(torch.tensor(1.0)).bfloat16().float() * weight
    )
    assert early.float() != rounded.float() or early_scale != rounded_scale


def test_swiglu_patch_is_scoped_and_rejects_unknown_sources() -> None:
    situ = "activation_values[i][k] = __fmul2_rn(__fmul2_rn(gate, up), weights);"
    source = "// SiTU\n" + situ + "\n// SwiGLU\n" + _SWIGLU + "\n// Amax reduction"
    patched = _patch_swiglu(source)
    assert patched == source.replace(_SWIGLU, _BF16_SWIGLU)
    assert patched.index("__float22bfloat162_rn") < patched.index("// Amax reduction")
    for unsupported in ("", _SWIGLU * 2, patched):
        with pytest.raises(RuntimeError, match="Unsupported DeepGEMM"):
            _patch_swiglu(unsupported)


def test_include_overlay_is_atomic_and_does_not_modify_installation(
    tmp_path: Path,
) -> None:
    include = tmp_path / "installed" / "include"
    header = include / _HEADER
    header.parent.mkdir(parents=True)
    header.write_text(_SWIGLU)
    untouched = header.with_name("other.cuh")
    untouched.write_text("// unchanged")
    (include / "cutlass").mkdir()
    (include / "deep_gemm" / "common").mkdir()
    cache = tmp_path / "cache"
    with ThreadPoolExecutor(max_workers=4) as pool:
        roots = list(
            pool.map(lambda _: _make_include_overlay(include, cache), range(8))
        )
    assert all(root == roots[0] for root in roots)
    assert header.read_text() == _SWIGLU
    assert (roots[0] / "include" / _HEADER).read_text() == _BF16_SWIGLU
    assert (roots[0] / "include" / _HEADER.parent / "other.cuh").resolve() == untouched
    assert (roots[0] / "include" / "cutlass").resolve() == include / "cutlass"
    header.write_text(_SWIGLU + "\n// changed dependency")
    assert _make_include_overlay(include, cache) != roots[0]


@pytest.mark.parametrize("fast_math", [False, True])
@pytest.mark.parametrize("activation_clamp", [None, 1.0])
def test_fused_mega_moe_rounds_weighted_swiglu_before_fp8(
    fast_math: bool, activation_clamp: float | None
) -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires SM100-family GPU and DeepGEMM")

    # Identity MXFP4 projections make both GEMMs exact. Only the weighted
    # SwiGLU -> BF16 -> FP8 boundary can explain the counterexamples below.
    hidden, experts = 512, 4
    group = SimpleNamespace(size=lambda: 1, rank=lambda: 0, barrier=lambda: None)
    plan = dsv4_mega_moe_plan(
        num_experts=experts,
        num_local_experts=experts,
        top_k=1,
        hidden_size=hidden,
        intermediate_size=hidden,
        max_num_tokens=4,
        process_group=group,
        activation_clamp=activation_clamp,
        input_dtype=torch.bfloat16,
        solution="deep_gemm",
    )
    diagonal = torch.arange(hidden, device="cuda")
    packed = torch.zeros(
        (experts, hidden, hidden // 2), dtype=torch.uint8, device="cuda"
    )
    # E2M1 code 2 is 1.0; low nibble is the even K element.
    packed[:, diagonal, diagonal // 2] = (2 << (4 * (diagonal % 2))).to(torch.uint8)
    scales = torch.full(
        (experts, hidden, hidden // 32), 127, dtype=torch.uint8, device="cuda"
    )
    state = dsv4_mega_moe_process_weights(
        plan,
        torch.cat((packed, packed), dim=1),
        torch.cat((scales, scales), dim=1),
        packed,
        scales,
    )
    dsv4_mega_moe_warmup(plan, state)
    inputs = torch.full(
        (1, hidden),
        1.0 if activation_clamp is None else 2.0,
        dtype=torch.bfloat16,
        device="cuda",
    )
    ids = torch.zeros((1, 1), dtype=torch.int64, device="cuda")
    weights = torch.empty((1, 1), dtype=torch.float32, device="cuda")
    buffer = ops._get_symm_buffer(
        state=state.backend_state,
        process_group=group,
        num_experts=experts,
        top_k=1,
        hidden_size=hidden,
        intermediate_size=hidden,
        max_num_tokens=4,
    )

    def apply() -> torch.Tensor:
        return dsv4_mega_moe_apply(
            plan, state, inputs, weights, ids, fast_math=fast_math
        )

    for weight in (0.011372136883437634, 0.01872287318110466):
        weights.fill_(weight)
        weighted = (F.silu(torch.tensor(1.0)) * weight).bfloat16().float()
        payload, scale = _quantize(weighted)
        expected = torch.full_like(inputs, (payload.float() * scale).item())
        torch.testing.assert_close(apply(), expected, rtol=0, atol=0)
        # Expert zero's first ring token. Inspect SF as well as final output:
        # the second witness changes amax's UE8M0 exponent but not dequant output.
        expected_sf = int(scale.view(torch.int32).item()) >> 23
        assert buffer.l2_acts_sf[0, 0].item() == expected_sf * 0x01010101
        torch.testing.assert_close(
            buffer.l2_acts[0].float(),
            torch.full((hidden,), payload.float().item(), device="cuda"),
            rtol=0,
            atol=0,
        )

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = apply()
    weights.fill_(0.011372136883437634)
    graph.replay()
    torch.testing.assert_close(
        captured, torch.full_like(inputs, 0.0078125), rtol=0, atol=0
    )

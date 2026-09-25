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

import importlib
from pathlib import Path

import pytest
import torch
from tokenspeed_kernel.thirdparty.petit_gluon import petit_kernel

mega_moe = importlib.import_module("lib.moe.rocm.mega_moe")


def test_vendored_source_keeps_upstream_import_roots() -> None:
    vendor_root = Path(petit_kernel.__file__).resolve().parents[1]
    for source_root in (vendor_root / "lib", vendor_root / "petit_kernel"):
        for path in source_root.rglob("*.py"):
            assert "tokenspeed_kernel" not in path.read_text()


@mega_moe.g.jit
def _petit_compiler_contract_kernel():
    storage = mega_moe.l.allocate_shared_memory(
        mega_moe.l.uint32,
        [68],
        mega_moe.l.SwizzledSharedLayout(1, 1, 1, [0]),
    )
    shared = mega_moe.l.full((), 0, mega_moe.l.uint64).to(
        mega_moe.l.pointer_type(mega_moe.l.uint32, 3)
    )
    mega_moe.l.store(shared, 0)
    storage._keep_alive()


def test_gluon_imports_use_compatible_triton() -> None:
    assert mega_moe.g.__name__ == "triton.experimental.gluon"
    assert mega_moe.l.__name__ == "triton.experimental.gluon.language"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU compiler")
def test_gluon_petit_compiler_contract() -> None:
    compiled = _petit_compiler_contract_kernel.warmup(grid=(1,), num_warps=1)

    assert compiled.metadata.shared == 68 * 4
    assert "addrspace(3)" in compiled.asm["llir"]


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or not getattr(torch.cuda.get_device_properties(0), "gcnArchName", "").startswith(
        "gfx950"
    ),
    reason="requires a GFX950 compiler",
)
@pytest.mark.parametrize(
    ("num_experts", "topk", "model_dim", "activation_function", "has_bias"),
    (
        (
            128,
            4,
            2880,
            petit_kernel.MegaMoeActivationFunction.swiglu,
            True,
        ),
        (
            384,
            6,
            7168,
            petit_kernel.MegaMoeActivationFunction.silu,
            False,
        ),
    ),
)
def test_supported_mega_moe_profiles_compile(
    num_experts: int,
    topk: int,
    model_dim: int,
    activation_function: petit_kernel.MegaMoeActivationFunction,
    has_bias: bool,
) -> None:
    config = petit_kernel.MegaMoeConfig(
        world_size=8,
        num_experts=num_experts,
        topk=topk,
        model_dim=model_dim,
        activation=petit_kernel.MegaMoeActivation.mxfp4,
        activation_function=activation_function,
        stages=petit_kernel.MegaMoeStages.two_stage,
        inter_dim=3072,
        has_bias=has_bias,
    )
    adapter = mega_moe._MegaMoESolutions()[config._solution_id_for_tokens(1)]
    stage1, stage2, combine = adapter.Kernels(True, False, False, False)
    device = torch.device("cuda", 0)
    uint8 = torch.empty(1, dtype=torch.uint8, device=device)
    int32 = torch.empty(1, dtype=torch.int32, device=device)
    float32 = torch.empty(1, dtype=torch.float32, device=device)
    bfloat16 = torch.empty(1, dtype=torch.bfloat16, device=device)
    bias = bfloat16 if has_bias else None

    compiled = (
        mega_moe.MegaMoEStage1.warmup(
            uint8,
            uint8,
            1,
            bias,
            uint8,
            0,
            uint8,
            int32,
            float32,
            stage1,
            None,
            grid=(stage1.kNumSMs,),
            num_warps=stage1.kNumWarps,
            enable_fp_fusion=False,
        ),
        mega_moe.MegaMoEStage2.warmup(
            uint8,
            uint8,
            bias,
            uint8,
            0,
            stage2,
            None,
            grid=(stage2.kStage2GridBlocks,),
            num_warps=stage2.kNumWarps,
            enable_fp_fusion=False,
        ),
        mega_moe.MegaMoECombine.warmup(
            bfloat16,
            1,
            config.compute_model_dim,
            uint8,
            0,
            combine,
            None,
            grid=(combine.kNumSMs,),
            num_warps=combine.kNumWarps,
            enable_fp_fusion=False,
        ),
    )

    assert [kernel.metadata.shared for kernel in compiled] == [32784, 32784, 0]
    assert all(kernel.metadata.triton_version == "3.8.0" for kernel in compiled)


@pytest.mark.parametrize(
    (
        "num_experts",
        "topk",
        "model_dim",
        "inter_dim",
        "activation_function",
        "has_bias",
        "compute_model_dim",
    ),
    (
        (
            128,
            4,
            2880,
            3072,
            petit_kernel.MegaMoeActivationFunction.swiglu,
            True,
            3072,
        ),
        (
            384,
            6,
            7168,
            3072,
            petit_kernel.MegaMoeActivationFunction.silu,
            False,
            7168,
        ),
    ),
)
def test_supported_mega_moe_profiles(
    num_experts: int,
    topk: int,
    model_dim: int,
    inter_dim: int,
    activation_function: petit_kernel.MegaMoeActivationFunction,
    has_bias: bool,
    compute_model_dim: int,
) -> None:
    config = petit_kernel.MegaMoeConfig(
        world_size=8,
        num_experts=num_experts,
        topk=topk,
        model_dim=model_dim,
        activation=petit_kernel.MegaMoeActivation.mxfp4,
        activation_function=activation_function,
        stages=petit_kernel.MegaMoeStages.two_stage,
        inter_dim=inter_dim,
        has_bias=has_bias,
    )

    assert config.compute_model_dim == compute_model_dim
    assert config.max_tokens_per_rank == 1024


def test_native_mxfp4_repack_preserves_shapes() -> None:
    weight = (
        torch.arange(256 * 128, dtype=torch.int64).to(torch.uint8).reshape(1, 256, 128)
    )
    scales = torch.arange(256 * 8, dtype=torch.int64).to(torch.uint8).reshape(1, 256, 8)

    packed_weight, packed_scales = petit_kernel.repack_moe_kernel_layout(
        weight,
        scales,
        layout=petit_kernel.MoeKernelLayout.native_mxfp4,
        petit_format=True,
    )

    assert packed_weight.shape == weight.shape
    assert packed_scales.shape == scales.shape
    assert packed_weight.is_contiguous()
    assert packed_scales.is_contiguous()
    assert not torch.equal(packed_weight, weight)
    assert not torch.equal(packed_scales, scales)

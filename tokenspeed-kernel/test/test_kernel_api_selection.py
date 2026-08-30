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

"""Golden selection tests for top-level tokenspeed-kernel public APIs.

Each case invokes a real public API (``mm``, ``moe_plan``/``moe_apply``,
attention, sampling) with :class:`SelectedKernel` calls intercepted by a spy,
and asserts the auto-selected kernel name.  Cases run on every host: the
platform each case targets is injected via ``Platform.override`` with the
fixture platforms from ``conftest.py``, so an NVIDIA CI machine also checks
the AMD golden selections and vice versa.  Only kernels whose registration is
import-guarded on missing optional backend packages are skipped.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass

import pytest
import tokenspeed_kernel
import tokenspeed_kernel.numerics.reference.gemm as _gemm_reference
import tokenspeed_kernel.ops.attention as _attention_pkg
import tokenspeed_kernel.ops.attention.cuda as _attention_cuda
import tokenspeed_kernel.ops.attention.cuda.dsv4 as _attention_cuda_dsv4
import tokenspeed_kernel.ops.attention.flash_attn as _attention_flash_attn
import tokenspeed_kernel.ops.attention.flash_mla as _attention_flash_mla
import tokenspeed_kernel.ops.attention.flashinfer as _attention_flashinfer
import tokenspeed_kernel.ops.attention.flashinfer.gated_delta_rule as _attention_flashinfer_gdn
import tokenspeed_kernel.ops.attention.gluon as _attention_gluon
import tokenspeed_kernel.ops.attention.triton as _attention_triton
import tokenspeed_kernel.ops.gemm as _gemm_pkg
import tokenspeed_kernel.ops.gemm.cuda as _gemm_cuda
import tokenspeed_kernel.ops.gemm.deep_gemm as _gemm_deep_gemm
import tokenspeed_kernel.ops.gemm.flashinfer as _gemm_flashinfer
import tokenspeed_kernel.ops.gemm.gluon as _gemm_gluon
import tokenspeed_kernel.ops.gemm.triton as _gemm_triton
import tokenspeed_kernel.ops.gemm.trtllm as _gemm_trtllm
import tokenspeed_kernel.ops.mhc as _mhc_pkg
import tokenspeed_kernel.ops.mhc.deep_gemm as _mhc_deep_gemm
import tokenspeed_kernel.ops.mhc.gluon as _mhc_gluon
import tokenspeed_kernel.ops.mhc.triton as _mhc_triton
import tokenspeed_kernel.ops.moe as _moe_pkg
import tokenspeed_kernel.ops.moe.cuda as _moe_cuda
import tokenspeed_kernel.ops.moe.deep_gemm as _moe_deep_gemm
import tokenspeed_kernel.ops.moe.flashinfer as _moe_flashinfer
import tokenspeed_kernel.ops.moe.gluon as _moe_gluon
import tokenspeed_kernel.ops.moe.gluon.dsv4 as _moe_gluon_dsv4
import tokenspeed_kernel.ops.moe.triton as _moe_triton
import tokenspeed_kernel.ops.quantization as _quantization_pkg
import tokenspeed_kernel.ops.quantization.flashinfer as _quantization_flashinfer
import tokenspeed_kernel.ops.quantization.triton as _quantization_triton
import tokenspeed_kernel.ops.quantization.trtllm as _quantization_trtllm
import tokenspeed_kernel.ops.sampling as _sampling_pkg
import tokenspeed_kernel.ops.sampling.cute_dsl as _sampling_cute_dsl
import tokenspeed_kernel.ops.sampling.gluon as _sampling_gluon
import torch
from tokenspeed_kernel.ops.attention import GdnChunkPrefillResult
from tokenspeed_kernel.ops.attention.triton import dsa as _attention_triton_dsa
from tokenspeed_kernel.ops.attention.triton import (
    dsa_topk as _attention_triton_dsa_topk,
)
from tokenspeed_kernel.ops.attention.triton import dsv4 as _attention_triton_dsv4
from tokenspeed_kernel.ops.attention.triton import (
    gated_delta_rule as _attention_triton_gdn,
)
from tokenspeed_kernel.ops.attention.triton import (
    merge_state as _attention_triton_merge_state,
)
from tokenspeed_kernel.ops.attention.triton import (
    mha_decode as _attention_triton_mha_decode,
)
from tokenspeed_kernel.ops.attention.triton import (
    mha_prefill as _attention_triton_mha_prefill,
)
from tokenspeed_kernel.ops.attention.triton import (
    mla_decode as _attention_triton_mla_decode,
)
from tokenspeed_kernel.ops.attention.triton import (
    mla_prefill as _attention_triton_mla_prefill,
)
from tokenspeed_kernel.ops.attention.triton import rel_mha as _attention_triton_rel_mha
from tokenspeed_kernel.ops.moe.deep_gemm import deepep_fp8 as _moe_deep_gemm_deepep_fp8
from tokenspeed_kernel.ops.moe.flashinfer import (
    cutedsl_deepep_nvfp4 as _moe_cutedsl_deepep_nvfp4,
)
from tokenspeed_kernel.ops.moe.flashinfer import cutlass_fp8 as _moe_cutlass_fp8
from tokenspeed_kernel.ops.moe.flashinfer import cutlass_nvfp4 as _moe_cutlass_nvfp4
from tokenspeed_kernel.ops.moe.flashinfer import cutlass_unquant as _moe_cutlass_unquant
from tokenspeed_kernel.ops.moe.flashinfer import trtllm_fp8 as _moe_trtllm_fp8
from tokenspeed_kernel.ops.moe.flashinfer import trtllm_mxfp4 as _moe_trtllm_mxfp4
from tokenspeed_kernel.ops.moe.flashinfer import trtllm_mxint4 as _moe_trtllm_mxint4
from tokenspeed_kernel.ops.moe.flashinfer import trtllm_nvfp4 as _moe_trtllm_nvfp4
from tokenspeed_kernel.ops.moe.flashinfer import trtllm_unquant as _moe_trtllm_unquant
from tokenspeed_kernel.ops.moe.gluon import mxfp4 as _moe_gluon_mxfp4
from tokenspeed_kernel.ops.moe.triton import bf16 as _moe_triton_bf16
from tokenspeed_kernel.ops.moe.triton import mxfp4 as _moe_triton_mxfp4
from tokenspeed_kernel.platform import ArchVersion, Platform, PlatformInfo
from tokenspeed_kernel.registry import KernelRegistry, error_fn
from tokenspeed_kernel.selection import SelectedKernel, spec_matches_traits

_RELOAD_MODULES = [
    # Attention registration modules.
    _attention_cuda_dsv4,
    _attention_cuda,
    _attention_flash_attn,
    _attention_flash_mla,
    _attention_flashinfer_gdn,
    _attention_flashinfer,
    _attention_gluon,
    _attention_triton_mha_prefill,
    _attention_triton_mha_decode,
    _attention_triton_mla_prefill,
    _attention_triton_mla_decode,
    _attention_triton_rel_mha,
    _attention_triton_merge_state,
    _attention_triton_dsv4,
    _attention_triton_dsa,
    _attention_triton_dsa_topk,
    _attention_triton_gdn,
    _attention_triton,
    _attention_pkg,
    # GEMM registration modules.
    _gemm_reference,
    _gemm_cuda,
    _gemm_deep_gemm,
    _gemm_flashinfer,
    _gemm_gluon,
    _gemm_triton,
    _gemm_trtllm,
    _gemm_pkg,
    # mHC registration modules.
    _mhc_deep_gemm,
    _mhc_gluon,
    _mhc_triton,
    _mhc_pkg,
    # MoE registration modules.
    _moe_cuda,
    _moe_deep_gemm_deepep_fp8,
    _moe_deep_gemm,
    _moe_cutedsl_deepep_nvfp4,
    _moe_cutlass_fp8,
    _moe_cutlass_nvfp4,
    _moe_cutlass_unquant,
    _moe_trtllm_fp8,
    _moe_trtllm_mxfp4,
    _moe_trtllm_mxint4,
    _moe_trtllm_nvfp4,
    _moe_trtllm_unquant,
    _moe_flashinfer,
    _moe_gluon_dsv4,
    _moe_gluon_mxfp4,
    _moe_gluon,
    _moe_triton_bf16,
    _moe_triton_mxfp4,
    _moe_triton,
    _moe_pkg,
    # Quantization registration modules.
    _quantization_flashinfer,
    _quantization_triton,
    _quantization_trtllm,
    _quantization_pkg,
    # Sampling registration modules.
    _sampling_cute_dsl,
    _sampling_gluon,
    _sampling_pkg,
    # Top-level public API re-exports.
    tokenspeed_kernel,
]


@pytest.fixture(autouse=True)
def _kernel_registry(fresh_registry):
    """Reload real registrations into the fresh registry for each case."""
    for mod in _RELOAD_MODULES:
        importlib.reload(mod)


def test_builtin_moe_preprocessor_links_are_callables():
    kernel_registry = KernelRegistry.get()
    errors = []
    for kernel_spec in kernel_registry.list_kernels("moe", "apply"):
        preprocessor = kernel_spec.weight_preprocessor
        if preprocessor is not None and not callable(preprocessor):
            errors.append(f"{kernel_spec.name}: non-callable preprocessor")

    process_weight_kernels = kernel_registry.list_kernels("moe", "process_weights")
    assert process_weight_kernels == []

    assert errors == []


def test_dsv4_padded_heads_platform_policy(
    mi350_platform: PlatformInfo,
    h100_platform: PlatformInfo,
) -> None:
    host_platform = Platform.get()
    try:
        Platform.override(mi350_platform)
        assert tokenspeed_kernel.dsv4_padded_heads(16) == 16
        assert tokenspeed_kernel.dsv4_padded_heads(32) == 32
        Platform.override(h100_platform)
        assert tokenspeed_kernel.dsv4_padded_heads(16) == 64
        assert tokenspeed_kernel.dsv4_padded_heads(65) == 128
    finally:
        Platform.override(host_platform)


def test_moe_process_weights_returns_for_no_preprocessing_plan():
    module = torch.nn.Module()

    result = tokenspeed_kernel.moe_process_weights(
        {"weight_preprocessor": None},
        module,
    )

    assert result is None


def test_moe_process_weights_dispatches_plan_preprocessor_callable():
    calls = []

    def preprocess(plan, w):
        calls.append((plan, w))

    module = torch.nn.Module()
    plan = {"weight_preprocessor": preprocess}

    result = tokenspeed_kernel.moe_process_weights(plan, module)

    assert result is None
    assert calls == [(plan, module)]


@dataclass(frozen=True)
class KernelApiSelectionCase:
    id: str
    family: str
    mode: str
    arch: str
    expected: str
    # Whether a platform would run this case natively.  Evaluated against the
    # host platform to decide if a missing kernel registration is a failure
    # (the host should have the backend) or a skip (optional backend absent).
    matches: Callable[[PlatformInfo], bool]
    invoke: Callable[[], object]


def _is_hopper(platform: PlatformInfo) -> bool:
    return platform.is_hopper


def _is_blackwell_sm100(platform: PlatformInfo) -> bool:
    return platform.is_blackwell and platform.arch_version == ArchVersion(10, 0)


def _is_blackwell_sm103(platform: PlatformInfo) -> bool:
    return platform.is_blackwell and platform.arch_version == ArchVersion(10, 3)


def _is_blackwell_non_sm100(platform: PlatformInfo) -> bool:
    return platform.is_blackwell and platform.arch_version != ArchVersion(10, 0)


def _is_blackwell_plus(platform: PlatformInfo) -> bool:
    return platform.is_blackwell_plus


def _is_hopper_plus(platform: PlatformInfo) -> bool:
    return platform.is_nvidia and platform.arch_version >= ArchVersion(9, 0)


def _is_hopper_plus_with_flashmla(platform: PlatformInfo) -> bool:
    return (
        _is_hopper_plus(platform)
        and _attention_flash_mla.flash_mla_with_kvcache is not error_fn
    )


def _is_nvidia(platform: PlatformInfo) -> bool:
    return platform.is_nvidia


def _is_nvidia_with_dsv4_cuda(platform: PlatformInfo) -> bool:
    return platform.is_nvidia and _attention_cuda_dsv4.has_fused_qnorm_rope_kv_insert()


def _is_nvidia_with_cute_dsl(platform: PlatformInfo) -> bool:
    return platform.is_nvidia and _sampling_cute_dsl.is_available()


def _is_hopper_plus_with_deep_gemm(platform: PlatformInfo) -> bool:
    # The FP8 DeepEP apply kernel only registers when the optional DeepGEMM
    # package exposes the masked grouped GEMM, so gate on that too.
    return (
        platform.is_nvidia
        and platform.arch_version >= ArchVersion(9, 0)
        and _moe_deep_gemm_deepep_fp8.m_grouped_fp8_gemm_nt_masked is not None
    )


def _is_cdna4(platform: PlatformInfo) -> bool:
    return platform.is_cdna4


def _is_cdna5(platform: PlatformInfo) -> bool:
    return platform.is_cdna5


def _is_supported_gpu(platform: PlatformInfo) -> bool:
    return platform.is_nvidia or platform.is_amd


def _fp8_dtype() -> torch.dtype:
    return torch.float8_e4m3fn


def _quantize_mxfp8() -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.empty((4, 128), dtype=torch.bfloat16)
    return tokenspeed_kernel.quantize_mxfp8(x)


def _mm_dense() -> torch.Tensor:
    a = torch.empty((4, 16), dtype=torch.bfloat16)
    b = torch.empty((32, 16), dtype=torch.bfloat16)
    return tokenspeed_kernel.mm(a, b)


def _mm_dense_cdna4_aligned() -> torch.Tensor:
    a = torch.empty((16, 64), dtype=torch.bfloat16)
    b = torch.empty((128, 64), dtype=torch.bfloat16)
    return tokenspeed_kernel.mm(a, b)


def _bmm_dense() -> torch.Tensor:
    a = torch.empty((4, 2, 16), dtype=torch.bfloat16)
    b = torch.empty((4, 32, 16), dtype=torch.bfloat16)
    return tokenspeed_kernel.bmm(a, b)


def _dsv4_linear_fp32() -> torch.Tensor:
    hidden_states = torch.empty((2, 4096), dtype=torch.bfloat16)
    weight = torch.empty((256, 4096), dtype=torch.bfloat16)
    return tokenspeed_kernel.dsv4_linear_fp32(hidden_states, weight)


def _mm_mxfp8() -> torch.Tensor:
    a = torch.empty((4, 128), dtype=_fp8_dtype())
    b = torch.empty((128, 128), dtype=_fp8_dtype())
    a_scales = torch.empty((4, 1), dtype=torch.float32)
    b_scales = torch.empty((1, 1), dtype=torch.float32)
    return tokenspeed_kernel.mm(
        a,
        b,
        A_scales=a_scales,
        B_scales=b_scales,
        out_dtype=torch.bfloat16,
        block_size=[128, 128],
        quant="mxfp8",
    )


def test_gemm_mxfp8_online_activation_signature_uses_quantized_storage() -> None:
    a = torch.empty((4, 128), dtype=torch.bfloat16)
    b = torch.empty((128, 128), dtype=_fp8_dtype())
    b_scales = torch.empty((1, 1), dtype=torch.float32)

    signature = _gemm_pkg._gemm_format_signature(
        a,
        b,
        None,
        b_scales,
        torch.bfloat16,
        "mxfp8",
        [128, 128],
    )

    a_format = signature.format_for("a")
    b_format = signature.format_for("b")
    assert a_format is not None
    assert b_format is not None
    assert a_format.storage_dtype == _fp8_dtype()
    assert b_format.storage_dtype == _fp8_dtype()
    assert a_format.scale is not None
    assert b_format.scale is not None
    assert a_format.scale.block_shape == (128, 128)
    assert b_format.scale.block_shape == (128, 128)


def test_bmm_mxfp8_online_activation_signature_uses_quantized_storage() -> None:
    a = torch.empty((2, 4, 128), dtype=torch.bfloat16)
    b = torch.empty((2, 128, 128), dtype=_fp8_dtype())
    b_scales = torch.empty((2, 1, 1), dtype=torch.float32)

    signature = _gemm_pkg._gemm_format_signature(
        a,
        b,
        None,
        b_scales,
        torch.bfloat16,
        "mxfp8",
        [128, 128],
    )

    a_format = signature.format_for("a")
    b_format = signature.format_for("b")
    assert a_format is not None
    assert b_format is not None
    assert a_format.storage_dtype == _fp8_dtype()
    assert b_format.storage_dtype == _fp8_dtype()
    assert a_format.scale is not None
    assert b_format.scale is not None
    assert a_format.scale.block_shape == (128, 128)
    assert b_format.scale.block_shape == (128, 128)


def test_gemm_mxfp8_online_activation_preserves_repeated_rows() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for online mxfp8 GEMM verification")
    if not (Platform.get().is_nvidia or Platform.get().is_cdna4):
        pytest.skip("online mxfp8 GEMM verification requires NVIDIA or AMD CDNA4")

    torch.manual_seed(0)
    num_tokens = 16
    hidden_size = 2048
    output_size = 128
    block_size = [128, 128]
    a = torch.randn((1, hidden_size), device="cuda", dtype=torch.bfloat16).repeat(
        num_tokens, 1
    )
    b = (
        torch.randn((output_size, hidden_size), device="cuda", dtype=torch.float32)
        * 0.1
    ).to(_fp8_dtype())
    b_scales = (
        torch.rand(
            (
                (output_size + block_size[0] - 1) // block_size[0],
                (hidden_size + block_size[1] - 1) // block_size[1],
            ),
            device="cuda",
            dtype=torch.float32,
        )
        + 0.01
    )

    out = tokenspeed_kernel.mm(
        a,
        b,
        B_scales=b_scales,
        out_dtype=torch.bfloat16,
        quant="mxfp8",
        block_size=block_size,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(out[1:], out[:1].expand_as(out[1:]), rtol=0, atol=0)


def test_gemm_fp8_scaled_signature_uses_fp8_format_with_scale() -> None:
    a = torch.empty((4, 128), dtype=_fp8_dtype())
    b = torch.empty((128, 128), dtype=_fp8_dtype())
    a_scales = torch.empty((1,), dtype=torch.float32)
    b_scales = torch.empty((1,), dtype=torch.float32)

    signature = _gemm_pkg._gemm_format_signature(
        a,
        b,
        a_scales,
        b_scales,
        torch.bfloat16,
        "fp8",
        None,
    )

    for role in ("a", "b"):
        tensor_format = signature.format_for(role)
        assert tensor_format is not None
        assert tensor_format.format == "scaled-fp8"
        assert tensor_format.storage_dtype == _fp8_dtype()
        assert tensor_format.scale is not None
        assert tensor_format.scale.granularity == "tensor"
        assert tensor_format.scale.storage_dtype == torch.float32


def test_bmm_fp8_scaled_signature_uses_fp8_format_with_scale() -> None:
    a = torch.empty((2, 4, 128), dtype=_fp8_dtype())
    b = torch.empty((2, 128, 128), dtype=_fp8_dtype())
    a_scales = torch.empty((1,), dtype=torch.float32)
    b_scales = torch.empty((1,), dtype=torch.float32)

    signature = _gemm_pkg._gemm_format_signature(
        a,
        b,
        a_scales,
        b_scales,
        torch.bfloat16,
        "fp8",
        None,
    )

    for role in ("a", "b"):
        tensor_format = signature.format_for(role)
        assert tensor_format is not None
        assert tensor_format.format == "scaled-fp8"
        assert tensor_format.storage_dtype == _fp8_dtype()
        assert tensor_format.scale is not None
        assert tensor_format.scale.granularity == "tensor"
        assert tensor_format.scale.storage_dtype == torch.float32


def test_gemm_fp8_scaled_signature_uses_channel_granularity() -> None:
    a = torch.empty((4, 128), dtype=_fp8_dtype())
    b = torch.empty((128, 128), dtype=_fp8_dtype())
    a_scales = torch.empty((4,), dtype=torch.float32)
    b_scales = torch.empty((128,), dtype=torch.float32)

    signature = _gemm_pkg._gemm_format_signature(
        a,
        b,
        a_scales,
        b_scales,
        torch.bfloat16,
        "fp8",
        None,
    )

    for role in ("a", "b"):
        tensor_format = signature.format_for(role)
        assert tensor_format is not None
        assert tensor_format.scale is not None
        assert tensor_format.scale.granularity == "channel"


def test_bmm_fp8_scaled_signature_uses_channel_granularity() -> None:
    a = torch.empty((4, 2, 128), dtype=_fp8_dtype())
    b = torch.empty((4, 32, 128), dtype=_fp8_dtype())
    a_scales = torch.empty((4, 2), dtype=torch.float32)
    b_scales = torch.empty((4, 32), dtype=torch.float32)

    signature = _gemm_pkg._gemm_format_signature(
        a,
        b,
        a_scales,
        b_scales,
        torch.bfloat16,
        "fp8",
        None,
    )

    for role in ("a", "b"):
        tensor_format = signature.format_for(role)
        assert tensor_format is not None
        assert tensor_format.scale is not None
        assert tensor_format.scale.granularity == "channel"


def test_gemm_quantized_reference_dispatches_fp8_inputs() -> None:
    fp8_dtype = _fp8_dtype()
    a = torch.zeros((4, 128), dtype=fp8_dtype)
    a_bf16 = torch.zeros((4, 128), dtype=torch.bfloat16)
    b = torch.zeros((128, 128), dtype=fp8_dtype)
    tensor_scales = torch.ones((1,), dtype=torch.float32)
    block_a_scales = torch.ones((4, 1), dtype=torch.float32)
    block_b_scales = torch.ones((1, 1), dtype=torch.float32)

    blockscale = tokenspeed_kernel.mm(
        a,
        b,
        A_scales=block_a_scales,
        B_scales=block_b_scales,
        out_dtype=torch.bfloat16,
        block_size=[128, 128],
        quant="mxfp8",
        override="torch_mm_fp8_blockscale",
    )
    assert blockscale.shape == (4, 128)
    assert blockscale.dtype == torch.bfloat16

    online_blockscale = tokenspeed_kernel.mm(
        a_bf16,
        b,
        B_scales=block_b_scales,
        out_dtype=torch.bfloat16,
        block_size=[128, 128],
        quant="mxfp8",
        override="torch_mm_fp8_blockscale",
    )
    assert online_blockscale.shape == (4, 128)
    assert online_blockscale.dtype == torch.bfloat16

    tensor_scaled = tokenspeed_kernel.mm(
        a,
        b,
        A_scales=tensor_scales,
        B_scales=tensor_scales,
        out_dtype=torch.bfloat16,
        quant="fp8",
        override="torch_mm_fp8_scaled_mnk",
    )
    assert tensor_scaled.shape == (4, 128)
    assert tensor_scaled.dtype == torch.bfloat16


def test_bmm_quantized_reference_dispatches_fp8_inputs() -> None:
    fp8_dtype = _fp8_dtype()
    a = torch.zeros((2, 4, 128), dtype=fp8_dtype)
    a_bf16 = torch.zeros((2, 4, 128), dtype=torch.bfloat16)
    b = torch.zeros((2, 128, 128), dtype=fp8_dtype)
    tensor_scales = torch.ones((1,), dtype=torch.float32)
    channel_a_scales = torch.ones((2, 4), dtype=torch.float32)
    channel_b_scales = torch.ones((2, 128), dtype=torch.float32)
    block_a_scales = torch.ones((2, 4, 1), dtype=torch.float32)
    block_b_scales = torch.ones((2, 1, 1), dtype=torch.float32)

    blockscale = tokenspeed_kernel.bmm(
        a,
        b,
        A_scales=block_a_scales,
        B_scales=block_b_scales,
        out_dtype=torch.bfloat16,
        block_size=[128, 128],
        quant="mxfp8",
        override="torch_bmm_fp8_blockscale",
    )
    assert blockscale.shape == (2, 4, 128)
    assert blockscale.dtype == torch.bfloat16

    online_blockscale = tokenspeed_kernel.bmm(
        a_bf16,
        b,
        B_scales=block_b_scales,
        out_dtype=torch.bfloat16,
        block_size=[128, 128],
        quant="mxfp8",
        override="torch_bmm_fp8_blockscale",
    )
    assert online_blockscale.shape == (2, 4, 128)
    assert online_blockscale.dtype == torch.bfloat16

    tensor_scaled = tokenspeed_kernel.bmm(
        a,
        b,
        A_scales=tensor_scales,
        B_scales=tensor_scales,
        out_dtype=torch.bfloat16,
        quant="fp8",
        override="torch_bmm_fp8_scaled",
    )
    assert tensor_scaled.shape == (2, 4, 128)
    assert tensor_scaled.dtype == torch.bfloat16

    channel_scaled = tokenspeed_kernel.bmm(
        a,
        b,
        A_scales=channel_a_scales,
        B_scales=channel_b_scales,
        out_dtype=torch.bfloat16,
        quant="fp8",
        override="torch_bmm_fp8_scaled",
    )
    assert channel_scaled.shape == (2, 4, 128)
    assert channel_scaled.dtype == torch.bfloat16


def _copy_out_mm_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scales: torch.Tensor | None,
    B_scales: torch.Tensor | None,
    out_dtype: torch.dtype,
    *,
    alpha: torch.Tensor | None = None,
    block_size: list[int] | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    assert A_scales is None
    assert B_scales is None
    assert block_size is None
    output = A @ B.T
    if alpha is not None:
        output = output * alpha.to(dtype=output.dtype)
    output = output.to(out_dtype)
    if out is not None:
        out.copy_(output)
        return out
    return output


def test_mm_non_native_out_kernel_copies_to_out(monkeypatch) -> None:
    torch.manual_seed(1)
    a = torch.randn((4, 8), dtype=torch.float32)
    b = torch.randn((16, 8), dtype=torch.float32)
    out = torch.empty((4, 16), dtype=torch.float32)

    def select_copy_out_kernel(*args, **kwargs) -> SelectedKernel:
        return SelectedKernel("test_mm_copy_out_kernel", _copy_out_mm_kernel)

    monkeypatch.setattr(_gemm_pkg, "select_kernel", select_copy_out_kernel)

    actual = tokenspeed_kernel.mm(a, b, out=out, override="test_mm_copy_out_kernel")
    expected = a @ b.T

    assert actual is out
    torch.testing.assert_close(out, expected)


def _mm_nvfp4() -> torch.Tensor:
    a = torch.empty((4, 64), dtype=torch.uint8)
    b = torch.empty((128, 64), dtype=torch.uint8)
    a_scales = torch.empty((4, 1), dtype=torch.float32)
    b_scales = torch.empty((128, 1), dtype=torch.float32)
    alpha = torch.empty((), dtype=torch.float32)
    return tokenspeed_kernel.mm(
        a,
        b,
        A_scales=a_scales,
        B_scales=b_scales,
        out_dtype=torch.bfloat16,
        alpha=alpha,
        quant="nvfp4",
    )


def test_gemm_nvfp4_signature_uses_fixed_block_shape() -> None:
    a = torch.empty((4, 64), dtype=torch.uint8)
    b = torch.empty((128, 64), dtype=torch.uint8)
    a_scales = torch.empty((4, 1), dtype=torch.float32)
    b_scales = torch.empty((128, 1), dtype=torch.float32)

    signature = _gemm_pkg._gemm_format_signature(
        a,
        b,
        a_scales,
        b_scales,
        torch.bfloat16,
        "nvfp4",
        None,
    )

    for role in ("a", "b"):
        tensor_format = signature.format_for(role)
        assert tensor_format is not None
        assert tensor_format.scale is not None
        assert tensor_format.scale.block_shape == (16,)


def test_bmm_nvfp4_signature_uses_fixed_block_shape() -> None:
    a = torch.empty((2, 4, 64), dtype=torch.uint8)
    b = torch.empty((2, 128, 64), dtype=torch.uint8)
    a_scales = torch.empty((2, 4, 1), dtype=torch.float32)
    b_scales = torch.empty((2, 128, 1), dtype=torch.float32)

    signature = _gemm_pkg._gemm_format_signature(
        a,
        b,
        a_scales,
        b_scales,
        torch.bfloat16,
        "nvfp4",
        None,
    )

    for role in ("a", "b"):
        tensor_format = signature.format_for(role)
        assert tensor_format is not None
        assert tensor_format.scale is not None
        assert tensor_format.scale.block_shape == (16,)


def _mm_mxfp4() -> torch.Tensor:
    a = torch.empty((4, 32), dtype=torch.uint8)
    b = torch.empty((128, 32), dtype=torch.uint8)
    a_scales = torch.empty((4, 2), dtype=torch.uint8)
    b_scales = torch.empty((128, 2), dtype=torch.uint8)
    return tokenspeed_kernel.mm(
        a,
        b,
        A_scales=a_scales,
        B_scales=b_scales,
        out_dtype=torch.bfloat16,
        quant="mxfp4",
    )


def _attention_prefill() -> object:
    q = torch.empty((4, 16, 64), dtype=torch.bfloat16)
    k = torch.empty((4, 8, 64), dtype=torch.bfloat16)
    v = torch.empty((4, 8, 64), dtype=torch.bfloat16)
    cu_seqlens = torch.tensor([0, 4], dtype=torch.int32)
    return tokenspeed_kernel.mha_prefill(
        q,
        k,
        v,
        cu_seqlens,
        cu_seqlens_cpu=[0, 4],
        max_seqlen=4,
    )


def _attention_extend() -> object:
    q = torch.empty((4, 16, 64), dtype=torch.bfloat16)
    cu_seqlens_q = torch.tensor([0, 2, 4], dtype=torch.int32)
    cu_seqlens_kv = torch.tensor([0, 64, 192], dtype=torch.int32)
    k_cache = torch.empty((8, 64, 8, 64), dtype=torch.bfloat16)
    v_cache = torch.empty((8, 64, 8, 64), dtype=torch.bfloat16)
    page_table = torch.empty((2, 4), dtype=torch.int32)
    cache_seqlens = torch.tensor([64, 128], dtype=torch.int32)
    return tokenspeed_kernel.mha_extend_with_kvcache(
        q,
        cu_seqlens_q,
        cu_seqlens_kv,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        max_seqlen_q=2,
        max_seqlen_k=128,
    )


def _attention_decode() -> object:
    q = torch.empty((2, 16, 64), dtype=torch.bfloat16)
    k_cache = torch.empty((8, 64, 8, 64), dtype=torch.bfloat16)
    v_cache = torch.empty((8, 64, 8, 64), dtype=torch.bfloat16)
    page_table = torch.empty((2, 4), dtype=torch.int32)
    cache_seqlens = torch.tensor([64, 128], dtype=torch.int32)
    return tokenspeed_kernel.mha_decode_with_kvcache(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        max_seqlen_k=128,
        max_seqlen_q=1,
    )


def _attention_mla_decode(
    batch_size: int,
    *,
    override: str | None = None,
) -> object:
    q = torch.empty((batch_size, 1, 64, 576), dtype=torch.bfloat16)
    kv_cache = torch.empty((batch_size, 64, 1, 576), dtype=torch.bfloat16)
    page_table = torch.arange(batch_size, dtype=torch.int32).view(batch_size, 1)
    cache_seqlens = torch.full((batch_size,), 64, dtype=torch.int32)
    return tokenspeed_kernel.mla_decode_with_kvcache(
        q=q,
        kv_cache=kv_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        max_seqlen_k=64,
        qk_nope_head_dim=128,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        softmax_scale=1.0,
        override=override,
    )


def _attention_mla_decode_fp8_k3() -> object:
    q = torch.empty((1, 1, 12, 576), dtype=torch.bfloat16)
    kv_cache = torch.empty((2, 64, 1, 576), dtype=torch.float8_e4m3fn)
    page_table = torch.tensor([[0, 1]], dtype=torch.int32)
    cache_seqlens = torch.tensor([128], dtype=torch.int32)
    return tokenspeed_kernel.mla_decode_with_kvcache(
        q=q,
        kv_cache=kv_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        max_seqlen_k=300_000,
        qk_nope_head_dim=128,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        softmax_scale=192**-0.5,
    )


def _attention_mla_decode_fp8q_k3() -> object:
    q = torch.empty((1, 1, 12, 576), dtype=torch.float8_e4m3fn)
    kv_cache = torch.empty((2, 64, 1, 576), dtype=torch.float8_e4m3fn)
    page_table = torch.tensor([[0, 1]], dtype=torch.int32)
    cache_seqlens = torch.tensor([128], dtype=torch.int32)
    return tokenspeed_kernel.mla_decode_with_kvcache(
        q=q,
        kv_cache=kv_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        max_seqlen_k=300_000,
        qk_nope_head_dim=128,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        softmax_scale=192**-0.5,
    )


def _attention_mla_decode_fp8q_unsupported_heads() -> object:
    q = torch.empty((1, 1, 32, 576), dtype=torch.float8_e4m3fn)
    kv_cache = torch.empty((2, 64, 1, 576), dtype=torch.float8_e4m3fn)
    page_table = torch.tensor([[0, 1]], dtype=torch.int32)
    cache_seqlens = torch.tensor([128], dtype=torch.int32)
    return tokenspeed_kernel.mla_decode_with_kvcache(
        q=q,
        kv_cache=kv_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        max_seqlen_k=300_000,
        qk_nope_head_dim=128,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        softmax_scale=192**-0.5,
    )


def _attention_mla_decode_projected_value_gfx1250(heads: int = 12) -> object:
    q = torch.empty((1, 1, heads, 576), dtype=torch.float8_e4m3fn)
    kv_cache = torch.empty((64, 64, 1, 576), dtype=torch.float8_e4m3fn)
    page_table = torch.arange(64, dtype=torch.int32).view(1, 64)
    cache_seqlens = torch.tensor([4096], dtype=torch.int32)
    value_weight = torch.empty((heads, 512, 128), dtype=torch.bfloat16)
    out = torch.empty((1, heads * 128), dtype=torch.bfloat16)
    return tokenspeed_kernel.mla_decode_with_kvcache(
        q=q,
        kv_cache=kv_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        max_seqlen_k=4096,
        qk_nope_head_dim=128,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        softmax_scale=192**-0.5,
        value_weight=value_weight,
        out=out,
    )


def _attention_mla_project_value_gfx1250(
    *,
    heads: int = 12,
    use_gate: bool = False,
) -> object:
    attention = torch.empty((1, heads, 512), dtype=torch.bfloat16)
    weight = torch.empty((heads, 512, 128), dtype=torch.bfloat16)
    out = torch.empty((1, heads * 128), dtype=torch.bfloat16)
    gate = torch.empty_like(out) if use_gate else None
    return tokenspeed_kernel.mla_project_value(
        attention,
        weight,
        gate=gate,
        out=out,
    )


def _attention_mla_normalize_project_query_gfx1250(heads: int = 12) -> object:
    query = torch.empty((1, 1536), dtype=torch.bfloat16)
    kv = torch.empty((1, 512), dtype=torch.bfloat16)
    query_norm_weight = torch.empty((1536,), dtype=torch.bfloat16)
    kv_norm_weight = torch.empty((512,), dtype=torch.bfloat16)
    projection_weight = torch.empty((heads * 192, 1536), dtype=torch.bfloat16)
    return tokenspeed_kernel.mla_normalize_project_query(
        query,
        kv,
        query_norm_weight,
        kv_norm_weight,
        projection_weight,
        eps=1e-6,
        prepare_absorbed_query=True,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
    )


def _attention_rel_prefill() -> object:
    q = torch.empty((4, 16, 64), dtype=torch.bfloat16)
    k = torch.empty((4, 8, 64), dtype=torch.bfloat16)
    v = torch.empty((4, 8, 64), dtype=torch.bfloat16)
    rel_logits = torch.empty((4, 16, 64), dtype=torch.bfloat16)
    cu_seqlens = torch.tensor([0, 4], dtype=torch.int32)
    return tokenspeed_kernel.rel_mha_prefill(
        q,
        k,
        v,
        rel_logits,
        cu_seqlens,
        cu_seqlens_cpu=[0, 4],
        max_seqlen=4,
        softmax_scale=1.0 / 64,
    )


def _attention_rel_extend() -> object:
    q = torch.empty((4, 16, 64), dtype=torch.bfloat16)
    rel_logits = torch.empty((4, 16, 64), dtype=torch.bfloat16)
    cu_seqlens_q = torch.tensor([0, 2, 4], dtype=torch.int32)
    cu_seqlens_kv = torch.tensor([0, 64, 192], dtype=torch.int32)
    k_cache = torch.empty((8, 64, 8, 64), dtype=torch.bfloat16)
    v_cache = torch.empty((8, 64, 8, 64), dtype=torch.bfloat16)
    page_table = torch.empty((2, 4), dtype=torch.int32)
    cache_seqlens = torch.tensor([64, 128], dtype=torch.int32)
    return tokenspeed_kernel.rel_mha_extend_with_kvcache(
        q=q,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_kv,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        max_seqlen_q=2,
        max_seqlen_k=128,
        rel_logits=rel_logits,
        softmax_scale=1.0 / 64,
    )


def _attention_rel_extend_page256_sliding() -> object:
    q = torch.empty((4, 16, 128), dtype=torch.bfloat16)
    rel_logits = torch.empty((4, 16, 512), dtype=torch.bfloat16)
    cu_seqlens_q = torch.tensor([0, 2, 4], dtype=torch.int32)
    cu_seqlens_kv = torch.tensor([0, 256, 768], dtype=torch.int32)
    k_cache = torch.empty((4, 256, 8, 128), dtype=torch.bfloat16)
    v_cache = torch.empty((4, 256, 8, 128), dtype=torch.bfloat16)
    page_table = torch.empty((2, 3), dtype=torch.int32)
    cache_seqlens = torch.tensor([256, 512], dtype=torch.int32)
    return tokenspeed_kernel.rel_mha_extend_with_kvcache(
        q=q,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_kv,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        max_seqlen_q=2,
        max_seqlen_k=512,
        rel_logits=rel_logits,
        window_left=255,
        softmax_scale=1.0 / 128,
    )


def _attention_rel_decode() -> object:
    q = torch.empty((2, 16, 64), dtype=torch.bfloat16)
    rel_logits = torch.empty((2, 16, 64), dtype=torch.bfloat16)
    k_cache = torch.empty((8, 64, 8, 64), dtype=torch.bfloat16)
    v_cache = torch.empty((8, 64, 8, 64), dtype=torch.bfloat16)
    page_table = torch.empty((2, 4), dtype=torch.int32)
    cache_seqlens = torch.tensor([64, 128], dtype=torch.int32)
    cu_seqlens_q = torch.tensor([0, 1, 2], dtype=torch.int32)
    return tokenspeed_kernel.rel_mha_decode_with_kvcache(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        max_seqlen_k=128,
        rel_logits=rel_logits,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=1,
        softmax_scale=1.0 / 64,
    )


def _attention_rel_decode_page128_sliding() -> object:
    q = torch.empty((2, 16, 128), dtype=torch.bfloat16)
    rel_logits = torch.empty((2, 16, 128), dtype=torch.bfloat16)
    k_cache = torch.empty((4, 128, 8, 128), dtype=torch.bfloat16)
    v_cache = torch.empty((4, 128, 8, 128), dtype=torch.bfloat16)
    page_table = torch.empty((2, 2), dtype=torch.int32)
    cache_seqlens = torch.tensor([128, 256], dtype=torch.int32)
    cu_seqlens_q = torch.tensor([0, 1, 2], dtype=torch.int32)
    return tokenspeed_kernel.rel_mha_decode_with_kvcache(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        max_seqlen_k=256,
        rel_logits=rel_logits,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=1,
        window_left=127,
        softmax_scale=1.0 / 128,
    )


def _attention_rel_decode_multiquery(window_left: int) -> object:
    batch = 2
    prediction = 4
    q = torch.empty((batch * prediction, 8, 128), dtype=torch.bfloat16)
    rel_logits = torch.empty((batch * prediction, 8, 512), dtype=torch.bfloat16)
    k_cache = torch.empty((12, 128, 2, 128), dtype=torch.bfloat16)
    v_cache = torch.empty((12, 128, 2, 128), dtype=torch.bfloat16)
    page_table = torch.empty((batch, 6), dtype=torch.int32)
    cache_seqlens = torch.tensor([300, 641], dtype=torch.int32)
    cu_seqlens_q = torch.tensor([0, prediction, 2 * prediction], dtype=torch.int32)
    return tokenspeed_kernel.rel_mha_decode_with_kvcache(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        max_seqlen_k=641,
        rel_logits=rel_logits,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=prediction,
        window_left=window_left,
        softmax_scale=1.0 / 128,
    )


def _attention_rel_decode_multiquery_sliding() -> object:
    return _attention_rel_decode_multiquery(window_left=511)


def _attention_rel_decode_multiquery_full() -> object:
    return _attention_rel_decode_multiquery(window_left=-1)


def _attention_rel_decode_page256_sliding() -> object:
    q = torch.empty((2, 16, 128), dtype=torch.bfloat16)
    rel_logits = torch.empty((2, 16, 512), dtype=torch.bfloat16)
    k_cache = torch.empty((4, 256, 8, 128), dtype=torch.bfloat16)
    v_cache = torch.empty((4, 256, 8, 128), dtype=torch.bfloat16)
    page_table = torch.empty((2, 2), dtype=torch.int32)
    cache_seqlens = torch.tensor([256, 512], dtype=torch.int32)
    cu_seqlens_q = torch.tensor([0, 1, 2], dtype=torch.int32)
    return tokenspeed_kernel.rel_mha_decode_with_kvcache(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        max_seqlen_k=512,
        rel_logits=rel_logits,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=1,
        window_left=255,
        softmax_scale=1.0 / 128,
    )


def _attention_dsa_decode() -> object:
    q = torch.empty((2, 8, 576), dtype=torch.bfloat16)
    sparse_kv_cache = torch.empty((64, 656), dtype=torch.uint8)
    topk_slots = torch.empty((2, 512), dtype=torch.int32)
    topk_lens = torch.empty((2,), dtype=torch.int32)
    return tokenspeed_kernel.dsa_decode(
        q=q,
        kv_cache=None,
        sparse_kv_cache=sparse_kv_cache,
        topk_slots=topk_slots,
        topk_lens=topk_lens,
        max_seqlen_k=64,
        qk_nope_head_dim=192,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        softmax_scale=1.0,
        page_size=64,
    )


def _attention_dsv4_selected(width: int = 640) -> object:
    q = torch.empty((1, 16, 512), dtype=torch.bfloat16)
    kv = torch.empty((width, 512), dtype=torch.bfloat16)
    indices = torch.arange(width, dtype=torch.int32).unsqueeze(0)
    lens = torch.tensor([width], dtype=torch.int32)
    attn_sink = torch.empty((16,), dtype=torch.float32)
    return tokenspeed_kernel.dsv4_prefill(
        q,
        kv,
        indices,
        lens,
        attn_sink,
        512**-0.5,
    )


def _attention_dsv4_selected_short() -> object:
    return _attention_dsv4_selected(128)


def _attention_dsv4_selected_i64() -> object:
    q = torch.empty((1, 16, 512), dtype=torch.bfloat16)
    kv = torch.empty((640, 512), dtype=torch.bfloat16)
    return tokenspeed_kernel.dsv4_prefill(
        q,
        kv,
        torch.arange(640, dtype=torch.int32).unsqueeze(0),
        torch.tensor([640], dtype=torch.int64),
        torch.empty((16,), dtype=torch.float32),
        512**-0.5,
    )


def _attention_dsv4_paged_selected(with_extra: bool = True) -> object:
    q = torch.empty((2, 16, 512), dtype=torch.bfloat16)
    swa_cache = torch.empty((2, 64 * 584), dtype=torch.uint8)
    swa_slots = torch.empty((2, 256), dtype=torch.int32)
    swa_lens = torch.empty((2,), dtype=torch.int32)
    attn_sink = torch.empty((16,), dtype=torch.float32)
    kwargs = {}
    if with_extra:
        kwargs = {
            "extra_kv_cache": torch.empty((2, 64 * 584), dtype=torch.uint8),
            "extra_slots": torch.empty((2, 1, 128), dtype=torch.int32),
            "extra_lens": torch.empty((2,), dtype=torch.int32),
            "extra_page_size": 64,
        }
    return tokenspeed_kernel.dsv4_decode(
        q=q,
        swa_kv_cache=swa_cache,
        swa_slots=swa_slots,
        swa_lens=swa_lens,
        swa_page_size=64,
        attn_sink=attn_sink,
        softmax_scale=512**-0.5,
        **kwargs,
    )


def _attention_dsv4_paged_selected_swa_only() -> object:
    return _attention_dsv4_paged_selected(with_extra=False)


def _attention_dsv4_paged_selected_pro_tp8() -> object:
    tokens = 6
    q = torch.empty((tokens, 16, 512), dtype=torch.bfloat16)
    swa_cache = torch.empty((2, 64 * 584), dtype=torch.uint8)
    swa_slots = torch.empty((tokens, 128), dtype=torch.int32)
    swa_lens = torch.empty((tokens,), dtype=torch.int32)
    extra_cache = torch.empty((16, 64 * 584), dtype=torch.uint8)
    extra_slots = torch.empty((tokens, 1024), dtype=torch.int32)
    extra_lens = torch.empty((tokens,), dtype=torch.int32)
    attn_sink = torch.empty((16,), dtype=torch.float32)
    return tokenspeed_kernel.dsv4_decode(
        q=q,
        swa_kv_cache=swa_cache,
        swa_slots=swa_slots,
        swa_lens=swa_lens,
        swa_page_size=64,
        attn_sink=attn_sink,
        softmax_scale=512**-0.5,
        extra_kv_cache=extra_cache,
        extra_slots=extra_slots,
        extra_lens=extra_lens,
        extra_page_size=64,
    )


def _attention_dsv4_paged_selected_pro_tp8_i64() -> object:
    tokens = 6
    return tokenspeed_kernel.dsv4_decode(
        q=torch.empty((tokens, 16, 512), dtype=torch.bfloat16),
        swa_kv_cache=torch.empty((2, 64 * 584), dtype=torch.uint8),
        swa_slots=torch.empty((tokens, 128), dtype=torch.int32),
        swa_lens=torch.empty((tokens,), dtype=torch.int32),
        swa_page_size=64,
        attn_sink=torch.empty((16,), dtype=torch.float32),
        softmax_scale=512**-0.5,
        extra_kv_cache=torch.empty((16, 64 * 584), dtype=torch.uint8),
        extra_slots=torch.empty((tokens, 1024), dtype=torch.int32),
        extra_lens=torch.empty((tokens,), dtype=torch.int64),
        extra_page_size=64,
    )


def _attention_dsv4_swa_cache_insert() -> object:
    q = torch.empty((1, 2, 512), dtype=torch.bfloat16)
    kv = torch.empty((1, 512), dtype=torch.bfloat16)
    cache = torch.empty((1, 64 * 584), dtype=torch.uint8)
    slot_mapping = torch.zeros((1,), dtype=torch.int64)
    positions = torch.zeros((1,), dtype=torch.int64)
    cos_sin_cache = torch.empty((1, 64), dtype=torch.float32)
    q_out = torch.empty_like(q)
    return tokenspeed_kernel.dsv4_swa_cache_insert(
        q,
        kv,
        cache,
        slot_mapping,
        positions,
        cos_sin_cache,
        1e-6,
        64,
        q_out=q_out,
    )


def _attention_dsa_decode_fp8_dense_rank128_q4(
    dtype: torch.dtype = torch.float8_e4m3fn,
) -> object:
    q = torch.empty((2, 4, 8, 192), dtype=dtype)
    kv_cache = torch.empty((64, 192), dtype=dtype)
    topk_slots = torch.empty((8, 2048), dtype=torch.int32)
    topk_lens = torch.empty((8,), dtype=torch.int32)
    return tokenspeed_kernel.dsa_decode(
        q=q,
        kv_cache=kv_cache,
        sparse_kv_cache=None,
        topk_slots=topk_slots,
        topk_lens=topk_lens,
        max_seqlen_k=64,
        qk_nope_head_dim=128,
        kv_lora_rank=128,
        qk_rope_head_dim=64,
        softmax_scale=1.0,
        page_size=64,
        q_len_per_req=4,
    )


def _attention_dsa_decode_fp8_e5m2_dense_rank128_q4() -> object:
    return _attention_dsa_decode_fp8_dense_rank128_q4(torch.float8_e5m2)


def _attention_dsa_decode_fp8_dense_rank512(
    dtype: torch.dtype = torch.float8_e4m3fn,
) -> object:
    q = torch.empty((2, 4, 8, 576), dtype=dtype)
    kv_cache = torch.empty((64, 576), dtype=dtype)
    topk_slots = torch.empty((8, 2048), dtype=torch.int32)
    topk_lens = torch.empty((8,), dtype=torch.int32)
    return tokenspeed_kernel.dsa_decode(
        q=q,
        kv_cache=kv_cache,
        sparse_kv_cache=None,
        topk_slots=topk_slots,
        topk_lens=topk_lens,
        max_seqlen_k=64,
        qk_nope_head_dim=192,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        softmax_scale=1.0,
        page_size=64,
        q_len_per_req=4,
    )


def _attention_dsa_decode_fp8_e5m2_dense_rank512() -> object:
    return _attention_dsa_decode_fp8_dense_rank512(torch.float8_e5m2)


def _attention_dsa_decode_fp8_sparse_rank512(
    dtype: torch.dtype = torch.float8_e4m3fn,
) -> object:
    q = torch.empty((2, 4, 8, 576), dtype=dtype)
    sparse_kv_cache = torch.empty((64, 656), dtype=torch.uint8)
    topk_slots = torch.empty((8, 2048), dtype=torch.int32)
    topk_lens = torch.empty((8,), dtype=torch.int32)
    return tokenspeed_kernel.dsa_decode(
        q=q,
        kv_cache=None,
        sparse_kv_cache=sparse_kv_cache,
        topk_slots=topk_slots,
        topk_lens=topk_lens,
        max_seqlen_k=64,
        qk_nope_head_dim=192,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        softmax_scale=1.0,
        page_size=64,
        q_len_per_req=4,
    )


def _attention_dsa_decode_fp8_e5m2_sparse_rank512() -> object:
    return _attention_dsa_decode_fp8_sparse_rank512(torch.float8_e5m2)


def _attention_dsa_prefill() -> object:
    q = torch.empty((2, 8, 576), dtype=torch.bfloat16)
    sparse_kv_cache = torch.empty((64, 656), dtype=torch.uint8)
    topk_slots = torch.empty((2, 512), dtype=torch.int32)
    topk_lens = torch.empty((2,), dtype=torch.int32)
    return tokenspeed_kernel.dsa_prefill(
        q=q,
        kv_cache=None,
        sparse_kv_cache=sparse_kv_cache,
        topk_slots=topk_slots,
        topk_lens=topk_lens,
        max_seqlen_k=64,
        qk_nope_head_dim=192,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        softmax_scale=1.0,
        page_size=64,
    )


def _attention_dsa_prefill_fp8_dense(
    dtype: torch.dtype = torch.float8_e4m3fn,
) -> object:
    q = torch.empty((2, 8, 576), dtype=dtype)
    kv_cache = torch.empty((64, 576), dtype=dtype)
    topk_slots = torch.empty((2, 1024), dtype=torch.int32)
    topk_lens = torch.empty((2,), dtype=torch.int32)
    return tokenspeed_kernel.dsa_prefill(
        q=q,
        kv_cache=kv_cache,
        sparse_kv_cache=None,
        topk_slots=topk_slots,
        topk_lens=topk_lens,
        max_seqlen_k=64,
        qk_nope_head_dim=192,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        softmax_scale=1.0,
        page_size=64,
    )


def _attention_dsa_prefill_fp8_e5m2_dense() -> object:
    return _attention_dsa_prefill_fp8_dense(torch.float8_e5m2)


def _attention_dsa_decode_fp8_dense_rank128() -> object:
    q = torch.empty((2, 8, 192), dtype=torch.float8_e4m3fn)
    kv_cache = torch.empty((64, 192), dtype=torch.float8_e4m3fn)
    topk_slots = torch.empty((2, 2048), dtype=torch.int32)
    topk_lens = torch.empty((2,), dtype=torch.int32)
    return tokenspeed_kernel.dsa_decode(
        q=q,
        kv_cache=kv_cache,
        sparse_kv_cache=None,
        topk_slots=topk_slots,
        topk_lens=topk_lens,
        max_seqlen_k=2048,
        qk_nope_head_dim=128,
        kv_lora_rank=128,
        qk_rope_head_dim=64,
        softmax_scale=1.0,
        page_size=64,
    )


def _attention_dsa_prefill_bf16_dense_rank128() -> object:
    q = torch.empty((2, 8, 192), dtype=torch.bfloat16)
    kv_cache = torch.empty((64, 192), dtype=torch.bfloat16)
    topk_slots = torch.empty((2, 1024), dtype=torch.int32)
    topk_lens = torch.empty((2,), dtype=torch.int32)
    return tokenspeed_kernel.dsa_prefill(
        q=q,
        kv_cache=kv_cache,
        sparse_kv_cache=None,
        topk_slots=topk_slots,
        topk_lens=topk_lens,
        max_seqlen_k=1024,
        qk_nope_head_dim=192,
        kv_lora_rank=128,
        qk_rope_head_dim=64,
        softmax_scale=1.0,
        page_size=64,
    )


def _attention_dsa_prefill_fp8_dense_rank128() -> object:
    q = torch.empty((2, 8, 192), dtype=torch.float8_e4m3fn)
    kv_cache = torch.empty((64, 192), dtype=torch.float8_e4m3fn)
    topk_slots = torch.empty((2, 1024), dtype=torch.int32)
    topk_lens = torch.empty((2,), dtype=torch.int32)
    return tokenspeed_kernel.dsa_prefill(
        q=q,
        kv_cache=kv_cache,
        sparse_kv_cache=None,
        topk_slots=topk_slots,
        topk_lens=topk_lens,
        max_seqlen_k=1024,
        qk_nope_head_dim=128,
        kv_lora_rank=128,
        qk_rope_head_dim=64,
        softmax_scale=1.0,
        page_size=64,
    )


def _attention_dsa_prefill_fp8_packed_rank512() -> object:
    q = torch.empty((2, 8, 576), dtype=torch.float8_e4m3fn)
    sparse_kv_cache = torch.empty((64, 656), dtype=torch.uint8)
    topk_slots = torch.empty((2, 1024), dtype=torch.int32)
    topk_lens = torch.empty((2,), dtype=torch.int32)
    return tokenspeed_kernel.dsa_prefill(
        q=q,
        kv_cache=None,
        sparse_kv_cache=sparse_kv_cache,
        topk_slots=topk_slots,
        topk_lens=topk_lens,
        max_seqlen_k=1024,
        qk_nope_head_dim=192,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        softmax_scale=1.0,
        page_size=64,
    )


def _attention_dsa_decode_topk(*, weights_dtype: torch.dtype = torch.float32) -> object:
    q = torch.empty((2, 2, 128), dtype=torch.bfloat16)
    weights = torch.empty((2, 2), dtype=weights_dtype)
    index_k = torch.zeros((128, 132), dtype=torch.uint8)
    seq_lens = torch.tensor([64, 64], dtype=torch.int32)
    block_table = torch.zeros((2, 1), dtype=torch.int32)
    return tokenspeed_kernel.dsa_decode_topk(
        q,
        weights,
        seq_lens,
        block_table,
        page_size=64,
        topk=512,
        softmax_scale=1.0,
        index_k_cache=index_k,
    )


def _attention_dsa_decode_topk_bf16_weights() -> object:
    return _attention_dsa_decode_topk(weights_dtype=torch.bfloat16)


def _attention_dsa_decode_topk_logical() -> object:
    q = torch.empty((2, 2, 128), dtype=torch.bfloat16)
    return tokenspeed_kernel.dsa_decode_topk(
        q,
        torch.empty((2, 2), dtype=torch.float32),
        torch.tensor([64, 64], dtype=torch.int32),
        torch.zeros((2, 1), dtype=torch.int32),
        page_size=64,
        topk=512,
        softmax_scale=1.0,
        index_k_cache=torch.zeros((128, 132), dtype=torch.uint8),
        topk_layout="logical_offsets",
        block_table_base_offsets=torch.tensor([3, 5], dtype=torch.int32),
    )


def _attention_dsa_prefill_topk(
    *,
    page_size: int = 64,
    solution: str | None = None,
    override: str | None = None,
    weights_dtype: torch.dtype = torch.float32,
) -> object:
    q = torch.empty((2, 2, 128), dtype=torch.bfloat16)
    weights = torch.empty((2, 2), dtype=weights_dtype)
    index_k = torch.zeros((128, 132), dtype=torch.uint8)
    kv_workspace_slots = torch.arange(64, dtype=torch.int64)
    row_starts = torch.tensor([0, 8], dtype=torch.int32)
    row_ends = torch.tensor([8, 16], dtype=torch.int32)
    return tokenspeed_kernel.dsa_prefill_topk(
        q,
        weights,
        kv_workspace_slots,
        row_starts,
        row_ends,
        topk=512,
        softmax_scale=1.0,
        index_k_cache=index_k,
        page_size=page_size,
        solution=solution,
        override=override,
    )


def _attention_dsa_prefill_topk_bf16_weights() -> object:
    return _attention_dsa_prefill_topk(weights_dtype=torch.bfloat16)


def _attention_dsa_decode_topk_standard(
    index_heads: int,
    q_dtype: torch.dtype = torch.bfloat16,
) -> object:
    q = torch.empty((2, index_heads, 128), dtype=q_dtype)
    q_scales = (
        torch.ones((2, index_heads), dtype=torch.float32)
        if q_dtype == torch.float8_e4m3fn
        else None
    )
    return tokenspeed_kernel.dsa_decode_topk(
        q,
        torch.empty((2, index_heads), dtype=torch.bfloat16),
        torch.tensor([64, 64], dtype=torch.int32),
        torch.zeros((2, 1), dtype=torch.int32),
        page_size=64,
        topk=512,
        softmax_scale=1.0,
        index_k_cache=torch.zeros((128, 132), dtype=torch.uint8),
        q_scales=q_scales,
    )


def _attention_dsa_prefill_topk_standard(
    index_heads: int,
    q_dtype: torch.dtype = torch.bfloat16,
) -> object:
    q = torch.empty((2, index_heads, 128), dtype=q_dtype)
    q_scales = (
        torch.ones((2, index_heads), dtype=torch.float32)
        if q_dtype == torch.float8_e4m3fn
        else None
    )
    return tokenspeed_kernel.dsa_prefill_topk(
        q,
        torch.empty((2, index_heads), dtype=torch.float32),
        torch.arange(64, dtype=torch.int64),
        torch.tensor([0, 8], dtype=torch.int32),
        torch.tensor([8, 16], dtype=torch.int32),
        topk=512,
        softmax_scale=1.0,
        index_k_cache=torch.zeros((128, 132), dtype=torch.uint8),
        page_size=64,
        q_scales=q_scales,
    )


@pytest.mark.parametrize("index_heads", [32, 64])
@pytest.mark.parametrize("mode", ["decode", "prefill"])
def test_dsa_topk_selection_receives_index_heads(
    monkeypatch: pytest.MonkeyPatch,
    index_heads: int,
    mode: str,
) -> None:
    """The public request exposes head count for exact DSA registrations."""
    captured: dict[str, object] = {}

    class _SelectedKernel:
        name = "test_dsa_topk"

        def __call__(self, **kwargs):
            tokens = kwargs["q"].shape[0]
            topk = int(kwargs["topk"])
            return (
                torch.full((tokens, topk), -1, dtype=torch.int32),
                torch.zeros((tokens,), dtype=torch.int32),
            )

    def select_dsa_topk(*args, **kwargs):
        captured.update(kwargs["traits"])
        return _SelectedKernel()

    monkeypatch.setattr(_attention_pkg, "select_kernel", select_dsa_topk)
    q = torch.empty((1, index_heads, 128), dtype=torch.bfloat16)
    weights = torch.empty((1, index_heads), dtype=torch.float32)
    index_k_cache = torch.empty((64, 132), dtype=torch.uint8)

    if mode == "decode":
        tokenspeed_kernel.dsa_decode_topk(
            q,
            weights,
            torch.tensor([1], dtype=torch.int32),
            torch.zeros((1, 1), dtype=torch.int32),
            page_size=64,
            topk=1,
            softmax_scale=1.0,
            index_k_cache=index_k_cache,
        )
    else:
        tokenspeed_kernel.dsa_prefill_topk(
            q,
            weights,
            torch.tensor([0], dtype=torch.int64),
            torch.tensor([0], dtype=torch.int32),
            torch.tensor([1], dtype=torch.int32),
            topk=1,
            softmax_scale=1.0,
            index_k_cache=index_k_cache,
            page_size=64,
        )

    assert captured["index_heads"] == index_heads


def test_dsa_prefill_topk_forwards_cpu_candidate_lens_to_deep_gemm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The optional host mirror reaches DeepGEMM without affecting selection."""
    captured: dict[str, object] = {}

    class _SelectedKernel:
        name = "deep_gemm_dsa_prefill_topk"

        def __call__(self, **kwargs):
            captured.update(kwargs)
            return (
                torch.full((2, 1), -1, dtype=torch.int32),
                torch.zeros((2,), dtype=torch.int32),
            )

    monkeypatch.setattr(
        _attention_pkg,
        "select_kernel",
        lambda *args, **kwargs: _SelectedKernel(),
    )
    candidate_lens_cpu = torch.tensor([8, 16], dtype=torch.int64)

    tokenspeed_kernel.dsa_prefill_topk(
        torch.empty((2, 2, 128), dtype=torch.bfloat16),
        torch.empty((2, 2), dtype=torch.float32),
        torch.arange(16, dtype=torch.int64),
        torch.tensor([0, 0], dtype=torch.int32),
        torch.tensor([8, 16], dtype=torch.int32),
        topk=1,
        softmax_scale=1.0,
        index_k_cache=torch.zeros((128, 132), dtype=torch.uint8),
        page_size=64,
        candidate_lens_cpu=candidate_lens_cpu,
    )

    assert captured["candidate_lens_cpu"] is candidate_lens_cpu


@pytest.mark.parametrize("mode", ["decode", "prefill"])
@pytest.mark.parametrize(
    ("cache", "expected"),
    [
        (torch.empty((64, 132), dtype=torch.uint8), "packed"),
        (torch.empty((1, 64 * 132), dtype=torch.uint8), "page_planar"),
    ],
)
def test_dsa_topk_selection_receives_cache_layout(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    cache: torch.Tensor,
    expected: str,
) -> None:
    captured: dict[str, object] = {}

    class _SelectedKernel:
        name = "test_dsa_topk"

        def __call__(self, **kwargs):
            tokens = kwargs["q"].shape[0]
            topk = int(kwargs["topk"])
            return (
                torch.full((tokens, topk), -1, dtype=torch.int32),
                torch.zeros((tokens,), dtype=torch.int32),
            )

    def select_dsa_topk(*args, **kwargs):
        captured.update(kwargs["traits"])
        return _SelectedKernel()

    monkeypatch.setattr(_attention_pkg, "select_kernel", select_dsa_topk)
    q = torch.empty((1, 32, 128), dtype=torch.bfloat16)
    weights = torch.empty((1, 32), dtype=torch.float32)

    if mode == "decode":
        tokenspeed_kernel.dsa_decode_topk(
            q,
            weights,
            torch.tensor([1], dtype=torch.int32),
            torch.zeros((1, 1), dtype=torch.int32),
            page_size=64,
            topk=1,
            softmax_scale=1.0,
            index_k_cache=cache,
        )
    else:
        tokenspeed_kernel.dsa_prefill_topk(
            q,
            weights,
            torch.tensor([0], dtype=torch.int64),
            torch.tensor([0], dtype=torch.int32),
            torch.tensor([1], dtype=torch.int32),
            topk=1,
            softmax_scale=1.0,
            index_k_cache=cache,
            page_size=64,
        )

    assert captured["index_k_layout"] == expected


@pytest.mark.parametrize("missing", ["values", "scales"])
def test_dsa_prefill_topk_rejects_incomplete_workspace_rows(missing: str) -> None:
    inputs = {
        "index_k_fp8": torch.empty((1, 128), dtype=torch.float8_e4m3fn),
        "index_k_scale": torch.ones((1, 1), dtype=torch.float32),
    }
    inputs.pop("index_k_fp8" if missing == "values" else "index_k_scale")

    with pytest.raises(ValueError, match="must be provided together"):
        tokenspeed_kernel.dsa_prefill_topk(
            torch.empty((1, 32, 128), dtype=torch.bfloat16),
            torch.empty((1, 32), dtype=torch.float32),
            torch.tensor([0], dtype=torch.int64),
            torch.tensor([0], dtype=torch.int32),
            torch.tensor([1], dtype=torch.int32),
            topk=1,
            softmax_scale=1.0,
            page_size=64,
            **inputs,
        )


def _attention_dsa_plan() -> object:
    seq_lens_2d = torch.tensor([[64], [64]], dtype=torch.int32)
    return tokenspeed_kernel.dsa_plan(seq_lens_2d=seq_lens_2d, page_size=64)


def _attention_merge_state() -> object:
    out_a = torch.empty((4, 16, 64), dtype=torch.bfloat16)
    out_b = torch.empty((4, 16, 64), dtype=torch.bfloat16)
    lse_a = torch.empty((4, 16), dtype=torch.float32)
    lse_b = torch.empty((4, 16), dtype=torch.float32)
    return tokenspeed_kernel.attn_merge_state(out_a, lse_a, out_b, lse_b)


def _mhc_pre() -> object:
    residual = torch.empty((1, 4, 16), dtype=torch.bfloat16)
    fn = torch.empty((24, 64), dtype=torch.float32)
    hc_scale = torch.empty((3,), dtype=torch.float32)
    hc_base = torch.empty((24,), dtype=torch.float32)
    return tokenspeed_kernel.mhc_pre(
        residual,
        fn,
        hc_scale,
        hc_base,
        1e-6,
        1e-6,
        2,
    )


def _mhc_post() -> object:
    hidden_states = torch.empty((1, 16), dtype=torch.bfloat16)
    residual = torch.empty((1, 4, 16), dtype=torch.bfloat16)
    post = torch.empty((1, 4, 1), dtype=torch.float32)
    comb = torch.empty((1, 4, 4), dtype=torch.float32)
    return tokenspeed_kernel.mhc_post(hidden_states, residual, post, comb)


def _attention_gdn_chunk_prefill() -> object:
    q = torch.empty((1, 4, 16, 64), dtype=torch.bfloat16)
    k = torch.empty((1, 4, 16, 64), dtype=torch.bfloat16)
    v = torch.empty((1, 4, 16, 64), dtype=torch.bfloat16)
    g = torch.empty((1, 4, 16), dtype=torch.bfloat16)
    beta = torch.empty((1, 4, 16), dtype=torch.bfloat16)
    initial_state = torch.empty((1, 16, 64, 64), dtype=torch.bfloat16)
    cu_seqlens = torch.tensor([0, 4], dtype=torch.int32)
    return tokenspeed_kernel.gdn_chunk_prefill(
        q,
        k,
        v,
        g,
        beta,
        scale=64**-0.5,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        qk_l2norm=True,
        solution="triton",
    )


def _sampling_argmax() -> object:
    if not torch.cuda.is_available():
        pytest.skip("argmax dispatches through kernel selection only for CUDA tensors")
    logits = torch.empty((4, 4096), dtype=torch.float32, device="cuda")
    return tokenspeed_kernel.argmax(logits)


def _assert_moe_plan(plan: dict, *, apply: str, preprocessor: str | None) -> None:
    assert plan["apply_kernel_name"] == apply
    actual_preprocessor = plan["weight_preprocessor"]
    actual_name = (
        None
        if actual_preprocessor is None
        else getattr(actual_preprocessor, "__name__", repr(actual_preprocessor))
    )
    assert actual_name == preprocessor


@pytest.mark.parametrize(
    "platform_fixture,weight_dtype,deepep_mode,expected_apply",
    [
        (
            "b200_platform",
            "nvfp4",
            "low_latency",
            "flashinfer_cutedsl_deepep_nvfp4_moe_apply",
        ),
        ("b200_platform", "fp8", "auto", "deep_gemm_deepep_fp8_moe_apply"),
    ],
)
def test_deepep_selects_apply_kernel_by_weight_dtype_without_pinned_solution(
    platform_fixture: str,
    weight_dtype: str,
    deepep_mode: str,
    expected_apply: str,
    request: pytest.FixtureRequest,
) -> None:
    """DeepEP plans must resolve through traits, not a hardcoded solution.

    ``moe_plan`` used to force ``solution="flashinfer_cutedsl_deepep"``, which no
    kernel registers, so an unpinned DeepEP plan could never resolve. The
    ``supports_all_to_all_ep`` + ``weight_dtype`` traits are what select the
    kernel owning the dispatch/combine legs for each quantization.
    """
    registry = KernelRegistry.get()
    if registry.get_by_name(expected_apply) is None:
        pytest.skip(f"{expected_apply!r} is unavailable (optional backend missing)")

    platform = request.getfixturevalue(platform_fixture)
    real_platform = Platform.get()
    try:
        Platform.override(platform)
        registry.clear_cache()
        plan = tokenspeed_kernel.moe_plan(
            weight_dtype,
            input_dtype=torch.bfloat16,
            activation="silu",
            a2a_backend="deepep",
            ep_size=2,
            ispp=256,
            fp8_scale_block_shape=(128, 128) if weight_dtype == "fp8" else None,
            internal_activation_dtype="input",
            deepep_group=object(),
            deepep_mode=deepep_mode,
        )
    finally:
        Platform.override(real_platform)
        registry.clear_cache()

    assert plan["apply_kernel_name"] == expected_apply
    assert plan["a2a_backend"] == "deepep"


@pytest.mark.parametrize("deepep_mode", [None, "auto", "normal"])
def test_nvfp4_deepep_rejects_modes_without_normal_legs(
    deepep_mode: str | None,
    b200_platform,
) -> None:
    """The nvfp4 masked GEMMs cannot consume normal dispatch buffers."""
    registry = KernelRegistry.get()
    kernel_name = "flashinfer_cutedsl_deepep_nvfp4_moe_apply"
    if registry.get_by_name(kernel_name) is None:
        pytest.skip(f"{kernel_name!r} is unavailable (optional backend missing)")

    real_platform = Platform.get()
    try:
        Platform.override(b200_platform)
        registry.clear_cache()
        with pytest.raises(ValueError, match="does not support deepep_mode"):
            tokenspeed_kernel.moe_plan(
                "nvfp4",
                input_dtype=torch.bfloat16,
                activation="silu",
                routing_mode="precomputed_topk",
                a2a_backend="deepep",
                ep_size=2,
                ispp=256,
                internal_activation_dtype="input",
                deepep_group=object(),
                deepep_mode=deepep_mode,
            )
    finally:
        Platform.override(real_platform)
        registry.clear_cache()


@pytest.mark.parametrize(
    "kernel_name",
    [
        "flashinfer_cutedsl_deepep_nvfp4_moe_apply",
        "deep_gemm_deepep_fp8_moe_apply",
    ],
)
def test_deepep_apply_kernels_only_register_bf16(kernel_name: str) -> None:
    """DeepEP low-latency dispatch accepts BF16 activations only."""
    spec = KernelRegistry.get().get_by_name(kernel_name)
    if spec is None:
        pytest.skip(f"{kernel_name!r} is unavailable (optional backend missing)")
    assert spec.storage_dtypes_for_role("x") == frozenset({torch.bfloat16})


def test_deepep_plan_carries_mode_and_low_latency_capacity(b200_platform) -> None:
    """Mode and capacity live on the plan, not on the first forward's shapes.

    The DeepEP buffer is allocated once, when a layer first dispatches. Sizing
    the low-latency legs from that batch would make decode depend on whichever
    batch arrived first, so the plan pins both up front.
    """
    registry = KernelRegistry.get()
    kernel_name = "deep_gemm_deepep_fp8_moe_apply"
    if registry.get_by_name(kernel_name) is None:
        pytest.skip(f"{kernel_name!r} is unavailable (optional backend missing)")

    real_platform = Platform.get()
    try:
        Platform.override(b200_platform)
        registry.clear_cache()
        plan = tokenspeed_kernel.moe_plan(
            "fp8",
            input_dtype=torch.bfloat16,
            activation="silu",
            a2a_backend="deepep",
            ep_size=2,
            ispp=256,
            fp8_scale_block_shape=(128, 128),
            deepep_group=object(),
            deepep_mode="auto",
            deepep_low_latency_max_num_tokens_per_gpu=256,
        )
    finally:
        Platform.override(real_platform)
        registry.clear_cache()

    assert plan["deepep_mode"] == "auto"
    assert plan["deepep_low_latency_max_num_tokens_per_gpu"] == 256


def test_moe_plan_defaults_deepep_mode_to_auto() -> None:
    plan = tokenspeed_kernel.moe_plan(
        "unquant",
        input_dtype=torch.bfloat16,
        activation="silu",
        routing_mode="precomputed_topk",
        ep_size=1,
        ispp=128,
        solution="triton",
    )
    assert plan["deepep_mode"] == "auto"
    assert plan["deepep_low_latency_max_num_tokens_per_gpu"] is None


@pytest.mark.parametrize(
    "deepep_mode,a2a_backend,match",
    [
        ("ll", "deepep", "deepep_mode must be"),
        ("normal", "none", "requires an all-to-all backend"),
    ],
)
def test_moe_plan_rejects_invalid_deepep_mode(
    deepep_mode: str, a2a_backend: str, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        tokenspeed_kernel.moe_plan(
            "fp8",
            input_dtype=torch.bfloat16,
            activation="silu",
            a2a_backend=a2a_backend,
            ep_size=2,
            ispp=256,
            fp8_scale_block_shape=(128, 128),
            deepep_mode=deepep_mode,
        )


def test_gluon_mxfp4_swiglu_args_default_missing_values_to_standard_swiglu() -> None:
    if not hasattr(_moe_gluon_mxfp4, "_swiglu_args"):
        pytest.skip("Gluon MXFP4 SwiGLU args are AMD-only")

    w = torch.nn.Module()
    w.swiglu_arg = type("SwigluArg", (), {"alpha": None, "limit": None})()

    assert _moe_gluon_mxfp4._swiglu_args(w) == (1.0, 0.0, 0.0)

    w.swiglu_arg = type("SwigluArg", (), {"alpha": 1.702, "limit": 7.0})()
    w.swiglu_beta = 1.0

    assert _moe_gluon_mxfp4._swiglu_args(w) == (1.702, 7.0, 1.0)


def test_gluon_dsa_prefill_topk_rejects_unsupported_page_size() -> None:
    registry = KernelRegistry.get()
    if registry.get_by_name("gluon_dsa_prefill_topk_fp8_gfx950") is None:
        pytest.skip("Gluon DSA top-k is AMD-only")

    with pytest.raises(tokenspeed_kernel.NoKernelFoundError, match="traits"):
        _attention_dsa_prefill_topk(page_size=32, solution="gluon")


def test_gluon_dsa_prefill_topk_exact_override_checks_page_size() -> None:
    registry = KernelRegistry.get()
    kernel_name = "gluon_dsa_prefill_topk_fp8_gfx950"
    if registry.get_by_name(kernel_name) is None:
        pytest.skip("Gluon DSA top-k is AMD-only")

    with pytest.raises(ValueError, match="page_size=64"):
        _attention_dsa_prefill_topk(page_size=32, override=kernel_name)


def test_gluon_mxfp4_apply_priority_prefers_dynamic_over_precomputed() -> None:
    """The dynamic gluon mxfp4 apply outranks the precomputed one.

    ``moe_plan`` never requests a ``routing_mode`` trait, so both the
    ``kernel_routing`` (dynamic) and ``precomputed_topk`` apply kernels match a
    gluon mxfp4 plan's traits. Selection therefore falls to declared priority.
    The dynamic entry (``SPECIALIZED + 3``) is the one that forwards caller
    top-k into BOTH the decode and package-prefill fast paths, so it must win
    over the precomputed entry (``SPECIALIZED + 2``), which only runs the
    generic ragged path. This test pins that ordering so a future priority
    bump doesn't silently route the AMD mxfp4 MoE onto the slower entry.
    """
    registry = KernelRegistry.get()
    dynamic = registry.get_by_name("gluon_mxfp4_dynamic_moe_apply")
    precomputed = registry.get_by_name("gluon_mxfp4_precomputed_moe_apply")
    if dynamic is None or precomputed is None:
        pytest.skip("gluon mxfp4 apply kernels are AMD-only")

    # Trait profiles differ only by routing_mode; everything else that gates
    # selection is identical, so priority is the tiebreaker.
    assert dynamic.traits.get("routing_mode") == frozenset({"kernel_routing"})
    assert precomputed.traits.get("routing_mode") == frozenset({"precomputed_topk"})
    for trait in (
        "weight_dtype",
        "activation",
        "internal_activation_dtype",
        "supports_bias",
        "ispp_alignment",
    ):
        assert dynamic.traits.get(trait) == precomputed.traits.get(trait)
    assert dynamic.priority > precomputed.priority


def test_gluon_mxfp4_plan_selects_dynamic_apply_on_cdna4(
    mi350_platform: PlatformInfo,
) -> None:
    """A CDNA4 gluon mxfp4 plan resolves to the dynamic apply, not precomputed.

    This is the end-to-end confirmation of the priority test above: with the
    platform overridden to CDNA4 (so both AMD apply kernels satisfy their
    capability gate), ``moe_plan`` picks ``gluon_mxfp4_dynamic_moe_apply``.
    That is the entry whose decode + package-prefill paths honor precomputed
    ``topk_weights`` / ``topk_ids``.
    """
    registry = KernelRegistry.get()
    if registry.get_by_name("gluon_mxfp4_dynamic_moe_apply") is None:
        pytest.skip("gluon mxfp4 apply kernels are AMD-only")

    real_platform = Platform.get()
    try:
        Platform.override(mi350_platform)
        registry.clear_cache()
        plan = tokenspeed_kernel.moe_plan(
            "mxfp4",
            input_dtype=torch.bfloat16,
            activation="swiglu",
            ispp=128,
            internal_activation_dtype="input",
            with_bias=True,
            solution="gluon",
        )
    finally:
        Platform.override(real_platform)
        registry.clear_cache()

    _assert_moe_plan(
        plan,
        apply="gluon_mxfp4_dynamic_moe_apply",
        preprocessor="gluon_mxfp4_gfx950_moe_weights",
    )
    # support_routing is True because the selected (dynamic) kernel advertises
    # kernel_routing; precomputed top-k is still forwarded as an optimization.
    assert plan["support_routing"] is True


def test_triton_mxfp4_supports_input_activation_dtype(
    mi350_platform: PlatformInfo,
) -> None:
    registry = KernelRegistry.get()
    real_platform = Platform.get()
    try:
        Platform.override(mi350_platform)
        registry.clear_cache()
        plan = tokenspeed_kernel.moe_plan(
            "mxfp4",
            input_dtype=torch.bfloat16,
            activation="swiglu",
            routing_mode="precomputed_topk",
            ispp=128,
            internal_activation_dtype="input",
            solution="triton",
        )
        assert plan["apply_kernel_name"] == "triton_mxfp4_precomputed_moe_apply"
    finally:
        Platform.override(real_platform)
        registry.clear_cache()


@pytest.mark.parametrize(
    "ep_size,solution,kernel_name,preprocessor",
    [
        (
            8,
            None,
            "gluon_mxfp4_a16w4_situ_ep_precomputed_moe_apply",
            "validate_linear_mxfp4_moe_weights",
        ),
        (
            8,
            "gluon",
            "gluon_mxfp4_a16w4_situ_ep_precomputed_moe_apply",
            "validate_linear_mxfp4_moe_weights",
        ),
    ],
)
def test_kimi3_mxfp4_situ_selection_on_cdna4(
    mi350_platform: PlatformInfo,
    ep_size: int,
    solution: str | None,
    kernel_name: str,
    preprocessor: str,
) -> None:
    registry = KernelRegistry.get()
    if registry.get_by_name(kernel_name) is None:
        pytest.skip(f"{kernel_name} is unavailable")
    real_platform = Platform.get()
    try:
        Platform.override(mi350_platform)
        registry.clear_cache()
        plan = tokenspeed_kernel.moe_plan(
            "mxfp4",
            input_dtype=torch.bfloat16,
            activation="situ",
            routing_mode="precomputed_topk",
            ep_size=ep_size,
            ispp=3072,
            internal_activation_dtype="input",
            solution=solution,
        )
    finally:
        Platform.override(real_platform)
        registry.clear_cache()

    _assert_moe_plan(plan, apply=kernel_name, preprocessor=preprocessor)
    assert plan["activation"] == "situ"
    assert plan["support_routing"] is False


def _make_fake_gluon_mxfp4_layer(top_k: int) -> torch.nn.Module:
    """Minimal ``w`` exposing only the attributes the apply wrapper reads.

    The downstream ``gluon_mxfp_dynamic_mxfp4_fused_moe`` is spied on, so the
    weight/scale tensors are never dereferenced by a real kernel launch.
    """
    w = torch.nn.Module()
    w.top_k = top_k
    w.w13_weight_triton_tensor = object()
    w.w2_weight_triton_tensor = object()
    w.w13_precision_config = type("PC", (), {"b_mx_scale": object()})()
    w.w2_precision_config = type(
        "PC", (), {"b_mx_scale": object(), "out_dtype": torch.bfloat16}
    )()
    return w


@pytest.mark.parametrize(
    "num_tokens, expect_forwarded",
    [
        (1, True),  # M <= _DIRECT_DECODE_MAX_M -> direct decode
        (2, True),  # M == _DIRECT_DECODE_MAX_M -> direct decode
        (3, True),  # previously-gapped interval (2, 4): now forwarded too
        (4, True),  # M >= _PRECOMPUTED_MFMA_MIN_M -> precomputed MFMA decode
        (16, True),  # large M still forwards for the generic precomputed route
    ],
)
def test_gluon_mxfp4_dynamic_apply_forwards_precomputed_topk_by_batch_size(
    num_tokens: int,
    expect_forwarded: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression test for precomputed-top-k forwarding across batch sizes.

    ``gluon_mxfp4_dynamic_moe_apply`` now forwards the caller's precomputed
    ``topk_weights`` / ``topk_ids`` for every batch size when they are
    supplied, and lets the downstream dispatch pick the tuned kernel. This
    guards against regressing the old M=3 gap, where batches in the open
    interval (_DIRECT_DECODE_MAX_M, _PRECOMPUTED_MFMA_MIN_M) silently dropped
    the precomputed top-k and recomputed routing from ``router_logits``.
    """
    if not hasattr(_moe_gluon_mxfp4, "gluon_mxfp4_dynamic_moe_apply"):
        pytest.skip("gluon mxfp4 dynamic apply is AMD-only")

    captured: dict[str, object] = {}

    def fake_fused_moe(*args, **kwargs):
        captured["precomputed_topk_weights"] = kwargs.get("precomputed_topk_weights")
        captured["precomputed_topk_ids"] = kwargs.get("precomputed_topk_ids")
        return "sentinel"

    monkeypatch.setattr(
        _moe_gluon_mxfp4, "gluon_mxfp_dynamic_mxfp4_fused_moe", fake_fused_moe
    )

    w = _make_fake_gluon_mxfp4_layer(top_k=1)
    x = torch.empty((num_tokens, 16), dtype=torch.bfloat16)
    router_logits = torch.empty((num_tokens, 8), dtype=torch.float32)
    topk_weights = torch.ones((num_tokens, 1), dtype=torch.float32)
    topk_ids = torch.zeros((num_tokens, 1), dtype=torch.int32)

    out = _moe_gluon_mxfp4.gluon_mxfp4_dynamic_moe_apply(
        {},
        x,
        w,
        router_logits,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
    )

    assert out == "sentinel"
    if expect_forwarded:
        assert captured["precomputed_topk_weights"] is topk_weights
        assert captured["precomputed_topk_ids"] is topk_ids
    else:
        # Reserved for batch sizes that intentionally drop precomputed top-k;
        # currently none do, so this branch guards against a future regression.
        assert captured["precomputed_topk_weights"] is None
        assert captured["precomputed_topk_ids"] is None


@pytest.mark.parametrize(
    "num_tokens,expected_decode",
    [
        pytest.param(32, True, id="bpe-16"),
        pytest.param(33, False, id="bpe-16.5"),
    ],
)
def test_gluon_mxfp4_gfx1250_apply_selects_kernel_by_average_bpe(
    num_tokens: int,
    expected_decode: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not hasattr(_moe_gluon_mxfp4, "gluon_mxfp4_gfx1250_precomputed_moe_apply"):
        pytest.skip("gfx1250 Gluon MXFP4 apply is AMD-only")

    captured: dict[str, object] = {}

    def fake_fused_moe(*args, **kwargs):
        captured["decode"] = kwargs.get("decode")
        return "sentinel"

    monkeypatch.setattr(
        _moe_gluon_mxfp4.fused_mxfp_gfx1250,
        "gluon_mxfp_precomputed_mxfp4_fused_moe",
        fake_fused_moe,
    )

    num_experts = 4
    top_k = 2
    w = torch.nn.Module()
    w.w13_weight_triton_tensor = torch.empty((num_experts, 0, 0))
    w.w2_weight_triton_tensor = object()
    w.w13_precision_config = type("PC", (), {"b_mx_scale": object()})()
    w.w2_precision_config = type(
        "PC", (), {"b_mx_scale": object(), "out_dtype": torch.bfloat16}
    )()
    x = torch.empty((num_tokens, 16), dtype=torch.bfloat16)
    router_logits = torch.empty((num_tokens, num_experts), dtype=torch.float32)
    topk_weights = torch.ones((num_tokens, top_k), dtype=torch.float32)
    topk_ids = torch.zeros((num_tokens, top_k), dtype=torch.int32)

    out = _moe_gluon_mxfp4.gluon_mxfp4_gfx1250_precomputed_moe_apply(
        {},
        x,
        w,
        router_logits,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
    )

    assert out == "sentinel"
    assert captured["decode"] is expected_decode


def _moe_apply_unquant_trtllm() -> object:
    plan = tokenspeed_kernel.moe_plan(
        "unquant",
        input_dtype=torch.bfloat16,
        activation="swiglu",
        requires_deferred_finalize=True,
        ep_size=2,
        ispp=128,
        internal_activation_dtype="input",
    )
    _assert_moe_plan(
        plan,
        apply="flashinfer_trtllm_unquant_moe_apply",
        preprocessor="flashinfer_trtllm_unquant_moe_weights",
    )
    x = torch.empty((4, 16), dtype=torch.bfloat16)
    router_logits = torch.empty((4, 8), dtype=torch.float32)
    return tokenspeed_kernel.moe_apply(
        plan,
        x,
        torch.nn.Module(),
        router_logits,
        do_finalize=False,
    )


def _dsv4_select_experts_bias() -> object:
    router_logits = torch.empty((2, 256), dtype=torch.float32)
    correction_bias = torch.empty((256,), dtype=torch.float32)
    return tokenspeed_kernel.dsv4_select_experts(
        router_logits,
        6,
        True,
        correction_bias=correction_bias,
        need_scores=False,
    )


def _dsv4_select_experts_hash() -> object:
    router_logits = torch.empty((2, 384), dtype=torch.bfloat16)
    hash_indices_table = torch.zeros((8, 6), dtype=torch.int32)
    input_ids = torch.zeros((2,), dtype=torch.int64)
    return tokenspeed_kernel.dsv4_select_experts(
        router_logits,
        6,
        True,
        hash_indices_table=hash_indices_table,
        input_ids=input_ids,
    )


def _moe_apply_unquant_cutlass() -> object:
    plan = tokenspeed_kernel.moe_plan(
        "unquant",
        input_dtype=torch.bfloat16,
        activation="swiglu",
        ep_size=2,
        ispp=128,
        internal_activation_dtype="input",
    )
    _assert_moe_plan(
        plan,
        apply="flashinfer_cutlass_unquant_moe_apply",
        preprocessor="flashinfer_cutlass_unquant_moe_weights",
    )
    x = torch.empty((4, 16), dtype=torch.bfloat16)
    router_logits = torch.empty((4, 8), dtype=torch.float32)
    return tokenspeed_kernel.moe_apply(plan, x, torch.nn.Module(), router_logits)


def _moe_apply_fp8_cutlass() -> object:
    plan = tokenspeed_kernel.moe_plan(
        "fp8",
        input_dtype=torch.bfloat16,
        activation="silu",
        ep_size=2,
        ispp=128,
        fp8_scale_block_shape=(128, 128),
        internal_activation_dtype="input",
    )
    _assert_moe_plan(
        plan,
        apply="flashinfer_cutlass_fp8_moe_apply",
        preprocessor="flashinfer_cutlass_fp8_moe_weights",
    )
    x = torch.empty((4, 16), dtype=torch.bfloat16)
    router_logits = torch.empty((4, 8), dtype=torch.float32)
    return tokenspeed_kernel.moe_apply(plan, x, torch.nn.Module(), router_logits)


def _moe_apply_fp8_trtllm() -> object:
    plan = tokenspeed_kernel.moe_plan(
        "fp8",
        input_dtype=torch.bfloat16,
        activation="silu",
        ep_size=2,
        ispp=128,
        fp8_scale_block_shape=(128, 128),
        internal_activation_dtype="input",
    )
    _assert_moe_plan(
        plan,
        apply="flashinfer_trtllm_fp8_moe_apply",
        preprocessor="flashinfer_trtllm_fp8_moe_process_weights",
    )
    x = torch.empty((4, 16), dtype=torch.bfloat16)
    router_logits = torch.empty((4, 8), dtype=torch.float32)
    return tokenspeed_kernel.moe_apply(plan, x, torch.nn.Module(), router_logits)


def _moe_apply_nvfp4_trtllm() -> object:
    plan = tokenspeed_kernel.moe_plan(
        "nvfp4",
        input_dtype=torch.bfloat16,
        activation="swiglu",
        requires_deferred_finalize=True,
        ep_size=2,
        ispp=128,
        internal_activation_dtype="input",
    )
    _assert_moe_plan(
        plan,
        apply="flashinfer_trtllm_nvfp4_moe_apply",
        preprocessor="flashinfer_trtllm_nvfp4_moe_weights",
    )
    x = torch.empty((4, 16), dtype=torch.bfloat16)
    router_logits = torch.empty((4, 8), dtype=torch.float32)
    return tokenspeed_kernel.moe_apply(
        plan,
        x,
        torch.nn.Module(),
        router_logits,
        do_finalize=False,
    )


def _moe_apply_nvfp4_cutlass() -> object:
    plan = tokenspeed_kernel.moe_plan(
        "nvfp4",
        input_dtype=torch.bfloat16,
        activation="swiglu",
        ep_size=2,
        ispp=128,
        internal_activation_dtype="input",
        solution="flashinfer_cutlass",
    )
    _assert_moe_plan(
        plan,
        apply="flashinfer_cutlass_nvfp4_moe_apply",
        preprocessor="flashinfer_cutlass_nvfp4_moe_weights",
    )
    x = torch.empty((4, 16), dtype=torch.bfloat16)
    router_logits = torch.empty((4, 8), dtype=torch.float32)
    return tokenspeed_kernel.moe_apply(plan, x, torch.nn.Module(), router_logits)


def _moe_apply_nvfp4_trtllm_routed() -> object:
    plan = tokenspeed_kernel.moe_plan(
        "nvfp4",
        input_dtype=torch.bfloat16,
        activation="swiglu",
        routing_mode="precomputed_topk",
        ep_size=2,
        ispp=128,
        internal_activation_dtype="input",
        solution="flashinfer_trtllm",
    )
    _assert_moe_plan(
        plan,
        apply="flashinfer_trtllm_nvfp4_routed_moe_apply",
        preprocessor="flashinfer_trtllm_nvfp4_moe_weights",
    )
    assert plan["support_routing"] is False
    x = torch.empty((4, 16), dtype=torch.bfloat16)
    router_logits = torch.empty((4, 8), dtype=torch.float32)
    topk_weights = torch.empty((4, 2), dtype=torch.float32)
    topk_ids = torch.empty((4, 2), dtype=torch.int32)
    return tokenspeed_kernel.moe_apply(
        plan,
        x,
        torch.nn.Module(),
        router_logits,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
    )


def _moe_apply_nvfp4_trtllm_unconstrained_routing() -> object:
    # No routing_mode requested: the kernel-routing registration must keep
    # winning under solution "flashinfer_trtllm" (its callers pass only
    # router_logits), so the routed variant sits at a lower priority.
    plan = tokenspeed_kernel.moe_plan(
        "nvfp4",
        input_dtype=torch.bfloat16,
        activation="swiglu",
        ep_size=2,
        ispp=128,
        internal_activation_dtype="input",
        solution="flashinfer_trtllm",
    )
    _assert_moe_plan(
        plan,
        apply="flashinfer_trtllm_nvfp4_moe_apply",
        preprocessor="flashinfer_trtllm_nvfp4_moe_weights",
    )
    assert plan["support_routing"] is True
    x = torch.empty((4, 16), dtype=torch.bfloat16)
    router_logits = torch.empty((4, 8), dtype=torch.float32)
    return tokenspeed_kernel.moe_apply(plan, x, torch.nn.Module(), router_logits)


def _moe_apply_unquant_trtllm_routed() -> object:
    plan = tokenspeed_kernel.moe_plan(
        "unquant",
        input_dtype=torch.bfloat16,
        activation="swiglu",
        routing_mode="precomputed_topk",
        ep_size=2,
        ispp=128,
        internal_activation_dtype="input",
        solution="flashinfer_trtllm",
    )
    _assert_moe_plan(
        plan,
        apply="flashinfer_trtllm_unquant_routed_moe_apply",
        preprocessor="flashinfer_trtllm_unquant_moe_weights",
    )
    assert plan["support_routing"] is False
    x = torch.empty((4, 16), dtype=torch.bfloat16)
    router_logits = torch.empty((4, 8), dtype=torch.float32)
    topk_weights = torch.empty((4, 2), dtype=torch.float32)
    topk_ids = torch.empty((4, 2), dtype=torch.int32)
    return tokenspeed_kernel.moe_apply(
        plan,
        x,
        torch.nn.Module(),
        router_logits,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
    )


def _moe_apply_nvfp4_deepep_cutedsl() -> object:
    plan = tokenspeed_kernel.moe_plan(
        "nvfp4",
        input_dtype=torch.bfloat16,
        activation="silu",
        a2a_backend="deepep",
        ep_size=2,
        ispp=128,
        internal_activation_dtype="input",
        deepep_group=object(),
        deepep_mode="low_latency",
        solution="flashinfer_cutedsl",
    )
    _assert_moe_plan(
        plan,
        apply="flashinfer_cutedsl_deepep_nvfp4_moe_apply",
        preprocessor="flashinfer_cutedsl_deepep_nvfp4_moe_weights",
    )
    x = torch.empty((4, 16), dtype=torch.bfloat16)
    router_logits = torch.empty((4, 8), dtype=torch.float32)
    return tokenspeed_kernel.moe_apply(plan, x, torch.nn.Module(), router_logits)


def _moe_apply_fp8_deepep_deep_gemm() -> object:
    plan = tokenspeed_kernel.moe_plan(
        "fp8",
        input_dtype=torch.bfloat16,
        activation="silu",
        routing_mode="precomputed_topk",
        a2a_backend="deepep",
        ep_size=2,
        ispp=256,
        fp8_scale_block_shape=(128, 128),
        internal_activation_dtype="input",
        deepep_group=object(),
        solution="deep_gemm",
    )
    _assert_moe_plan(
        plan,
        apply="deep_gemm_deepep_fp8_moe_apply",
        preprocessor="deep_gemm_deepep_fp8_moe_weights",
    )
    assert plan["support_routing"] is False
    x = torch.empty((4, 16), dtype=torch.bfloat16)
    router_logits = torch.empty((4, 8), dtype=torch.float32)
    topk_weights = torch.empty((4, 2), dtype=torch.float32)
    topk_ids = torch.empty((4, 2), dtype=torch.int32)
    return tokenspeed_kernel.moe_apply(
        plan,
        x,
        torch.nn.Module(),
        router_logits,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
    )


def _moe_apply_mxfp4_trtllm() -> object:
    plan = tokenspeed_kernel.moe_plan(
        "mxfp4",
        input_dtype=torch.bfloat16,
        activation="swiglu",
        ep_size=2,
        ispp=128,
        internal_activation_dtype="input",
        with_bias=True,
    )
    _assert_moe_plan(
        plan,
        apply="flashinfer_trtllm_mxfp4_moe_apply",
        preprocessor="flashinfer_trtllm_mxfp4_moe_weights",
    )
    x = torch.empty((4, 16), dtype=torch.bfloat16)
    router_logits = torch.empty((4, 8), dtype=torch.float32)
    return tokenspeed_kernel.moe_apply(plan, x, torch.nn.Module(), router_logits)


def _moe_apply_mxfp4_triton() -> object:
    plan = tokenspeed_kernel.moe_plan(
        "mxfp4",
        input_dtype=torch.bfloat16,
        activation="swiglu",
        routing_mode="precomputed_topk",
        ispp=128,
        internal_activation_dtype="mxfp4",
        with_bias=False,
        solution="triton",
    )
    _assert_moe_plan(
        plan,
        apply="triton_mxfp4_precomputed_moe_apply",
        preprocessor="triton_mxfp4_moe_weights",
    )
    x = torch.empty((4, 16), dtype=torch.bfloat16)
    router_logits = torch.empty((4, 8), dtype=torch.float32)
    topk_weights = torch.empty((4, 2), dtype=torch.float32)
    topk_ids = torch.empty((4, 2), dtype=torch.int64)
    return tokenspeed_kernel.moe_apply(
        plan,
        x,
        torch.nn.Module(),
        router_logits,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
    )


def _moe_apply_unquant_triton() -> object:
    plan = tokenspeed_kernel.moe_plan(
        "unquant",
        input_dtype=torch.bfloat16,
        activation="swiglu",
        routing_mode="precomputed_topk",
        ispp=128,
        internal_activation_dtype="input",
        with_bias=False,
        solution="triton",
    )
    _assert_moe_plan(
        plan,
        apply="triton_bf16_precomputed_moe_apply",
        preprocessor=None,
    )
    x = torch.empty((4, 16), dtype=torch.bfloat16)
    router_logits = torch.empty((4, 8), dtype=torch.float32)
    topk_weights = torch.empty((4, 2), dtype=torch.float32)
    topk_ids = torch.empty((4, 2), dtype=torch.int64)
    return tokenspeed_kernel.moe_apply(
        plan,
        x,
        torch.nn.Module(),
        router_logits,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
    )


def _moe_apply_mxfp4_gluon() -> object:
    plan = tokenspeed_kernel.moe_plan(
        "mxfp4",
        input_dtype=torch.bfloat16,
        activation="swiglu",
        ispp=128,
        internal_activation_dtype="fp8",
        with_bias=True,
    )
    _assert_moe_plan(
        plan,
        apply="gluon_mxfp4_moe_apply",
        preprocessor="gluon_mxfp4_gfx950_moe_weights",
    )
    x = torch.empty((4, 16), dtype=torch.bfloat16)
    router_logits = torch.empty((4, 8), dtype=torch.float32)
    return tokenspeed_kernel.moe_apply(plan, x, torch.nn.Module(), router_logits)


def _moe_apply_mxint4_trtllm() -> object:
    plan = tokenspeed_kernel.moe_plan(
        "mxint4",
        input_dtype=torch.bfloat16,
        activation="swiglu",
        ep_size=2,
        ispp=256,
        internal_activation_dtype="input",
    )
    _assert_moe_plan(
        plan,
        apply="flashinfer_trtllm_mxint4_moe_apply",
        preprocessor="flashinfer_trtllm_mxint4_moe_weights",
    )
    x = torch.empty((4, 16), dtype=torch.bfloat16)
    router_logits = torch.empty((4, 8), dtype=torch.float32)
    return tokenspeed_kernel.moe_apply(plan, x, torch.nn.Module(), router_logits)


def _moe_apply_mxfp4_dynamic_tp() -> object:
    plan = tokenspeed_kernel.moe_plan(
        "mxfp4",
        input_dtype=torch.bfloat16,
        activation="silu",
        ep_size=1,
        ispp=2048,
        internal_activation_dtype="input",
    )
    _assert_moe_plan(
        plan,
        apply="gluon_mxfp4_dynamic_moe_apply",
        preprocessor="gluon_mxfp4_gfx950_moe_weights",
    )
    x = torch.empty((4, 16), dtype=torch.bfloat16)
    router_logits = torch.empty((4, 8), dtype=torch.float32)
    return tokenspeed_kernel.moe_apply(
        plan,
        x,
        torch.nn.Module(),
        router_logits,
    )


def _case(
    matches: Callable[[PlatformInfo], bool],
    arch: str,
    family: str,
    mode: str,
    expected: str,
    invoke: Callable[[], object],
    *,
    id_suffix: str | None = None,
) -> KernelApiSelectionCase:
    case_id = f"{arch}/{family}.{mode}/{expected}"
    if id_suffix is not None:
        case_id = f"{case_id}/{id_suffix}"
    return KernelApiSelectionCase(
        id=case_id,
        arch=arch,
        family=family,
        mode=mode,
        expected=expected,
        matches=matches,
        invoke=invoke,
    )


_CASES = [
    # Attention API x architecture golden cases.
    _case(
        _is_hopper_plus_with_flashmla,
        "hopper-plus",
        "attention",
        "dsv4_decode",
        "flashmla_dsv4_decode",
        _attention_dsv4_paged_selected,
        id_suffix="extra-segment",
    ),
    _case(
        _is_hopper,
        "hopper",
        "attention",
        "mha_prefill",
        "fa3_mha_prefill",
        _attention_prefill,
    ),
    _case(
        _is_hopper,
        "hopper",
        "attention",
        "mha_extend_with_kvcache",
        "fa3_mha_extend_with_kvcache_cached",
        _attention_extend,
    ),
    _case(
        _is_hopper,
        "hopper",
        "attention",
        "mha_decode_with_kvcache",
        "fa3_mha_decode_with_kvcache_cached",
        _attention_decode,
    ),
    _case(
        _is_hopper,
        "hopper",
        "attention",
        "attn_merge_state",
        "cuda_attn_merge_state",
        _attention_merge_state,
    ),
    _case(
        _is_nvidia_with_dsv4_cuda,
        "hopper",
        "attention",
        "dsv4_swa_cache_insert",
        "cuda_dsv4_swa_cache_insert",
        _attention_dsv4_swa_cache_insert,
    ),
    _case(
        _is_blackwell_sm100,
        "blackwell-sm100",
        "attention",
        "mha_prefill",
        "fa4_mha_prefill",
        _attention_prefill,
    ),
    _case(
        _is_blackwell_sm100,
        "blackwell-sm100",
        "attention",
        "mha_extend_with_kvcache",
        "fa4_mha_extend_with_kvcache_cached",
        _attention_extend,
    ),
    _case(
        _is_blackwell_sm100,
        "blackwell-sm100",
        "attention",
        "mha_decode_with_kvcache",
        "fa4_mha_decode_with_kvcache",
        _attention_decode,
    ),
    _case(
        _is_blackwell_sm100,
        "blackwell-sm100",
        "attention",
        "attn_merge_state",
        "cuda_attn_merge_state",
        _attention_merge_state,
    ),
    _case(
        _is_blackwell_sm103,
        "blackwell-sm103",
        "attention",
        "mha_extend_with_kvcache",
        "flashinfer_trtllm_mha_extend_with_kvcache",
        _attention_extend,
    ),
    _case(
        _is_blackwell_sm103,
        "blackwell-sm103",
        "attention",
        "mha_decode_with_kvcache",
        "flashinfer_trtllm_mha_decode_with_kvcache",
        _attention_decode,
    ),
    _case(
        _is_blackwell_sm103,
        "blackwell-sm103",
        "attention",
        "attn_merge_state",
        "cuda_attn_merge_state",
        _attention_merge_state,
    ),
    _case(
        _is_blackwell_sm103,
        "blackwell-sm103",
        "attention",
        "rel_mha_decode_with_kvcache",
        "fa4_rel_mha_decode_with_kvcache",
        _attention_rel_decode_multiquery_sliding,
    ),
    _case(
        _is_cdna4,
        "cdna4",
        "attention",
        "dsv4_decode",
        "gluon_dsv4_decode_split_gfx950",
        _attention_dsv4_paged_selected_pro_tp8,
        id_suffix="pro-tp8",
    ),
    _case(
        _is_cdna4,
        "cdna4",
        "attention",
        "dsv4_decode",
        "triton_dsv4_decode",
        _attention_dsv4_paged_selected_pro_tp8_i64,
        id_suffix="pro-tp8-int64-metadata",
    ),
    _case(
        _is_cdna4,
        "cdna4",
        "attention",
        "dsv4_decode",
        "triton_dsv4_decode",
        _attention_dsv4_paged_selected,
        id_suffix="extra-segment",
    ),
    _case(
        _is_cdna4,
        "cdna4",
        "attention",
        "dsv4_decode",
        "triton_dsv4_decode",
        _attention_dsv4_paged_selected_swa_only,
        id_suffix="swa-only",
    ),
    _case(
        _is_cdna4,
        "cdna4",
        "attention",
        "dsv4_prefill",
        "gluon_dsv4_prefill_gfx950",
        _attention_dsv4_selected,
        id_suffix="width640",
    ),
    _case(
        _is_cdna4,
        "cdna4",
        "attention",
        "dsv4_prefill",
        "triton_dsv4_prefill",
        _attention_dsv4_selected_i64,
        id_suffix="width640-int64-metadata",
    ),
    _case(
        _is_cdna4,
        "cdna4",
        "attention",
        "dsv4_prefill",
        "triton_dsv4_prefill",
        _attention_dsv4_selected_short,
        id_suffix="width128",
    ),
    _case(
        _is_cdna4,
        "cdna4",
        "attention",
        "dsv4_swa_cache_insert",
        "triton_dsv4_swa_cache_insert",
        _attention_dsv4_swa_cache_insert,
    ),
    _case(
        _is_cdna4,
        "cdna4",
        "attention",
        "mha_prefill",
        "gluon_mha_prefill_gfx950",
        _attention_prefill,
    ),
    _case(
        _is_cdna4,
        "cdna4",
        "attention",
        "mha_extend_with_kvcache",
        "gluon_mha_extend_gfx950",
        _attention_extend,
    ),
    _case(
        _is_cdna4,
        "cdna4",
        "attention",
        "mha_decode_with_kvcache",
        "gluon_mha_decode_gfx950",
        _attention_decode,
    ),
    _case(
        _is_cdna4,
        "cdna4",
        "attention",
        "mla_decode_with_kvcache",
        "gluon_mla_decode_bf16xfp8_gfx950_bh16bn128",
        _attention_mla_decode_fp8_k3,
    ),
    _case(
        _is_cdna4,
        "cdna4",
        "attention",
        "mla_decode_with_kvcache",
        "gluon_mla_decode_fp8xfp8_gfx950_bh16bn128",
        _attention_mla_decode_fp8q_k3,
    ),
    _case(
        _is_cdna4,
        "cdna4",
        "attention",
        "mla_decode_with_kvcache",
        "triton_mla_decode_with_kvcache",
        _attention_mla_decode_fp8q_unsupported_heads,
    ),
    _case(
        _is_cdna5,
        "cdna5",
        "attention",
        "mla_decode_projected_value",
        "gluon_mla_decode_projected_value_gfx1250",
        _attention_mla_decode_projected_value_gfx1250,
    ),
    _case(
        _is_cdna5,
        "cdna5",
        "attention",
        "mla_decode_projected_value",
        "gluon_mla_decode_projected_value_gfx1250",
        lambda: _attention_mla_decode_projected_value_gfx1250(16),
        id_suffix="h16",
    ),
    _case(
        _is_cdna5,
        "cdna5",
        "attention",
        "mla_project_value",
        "gluon_mla_project_value_gfx1250",
        _attention_mla_project_value_gfx1250,
    ),
    _case(
        _is_cdna5,
        "cdna5",
        "attention",
        "mla_project_value",
        "gluon_mla_project_value_gfx1250",
        lambda: _attention_mla_project_value_gfx1250(use_gate=True),
        id_suffix="sigmoid-gate",
    ),
    _case(
        _is_cdna5,
        "cdna5",
        "attention",
        "mla_normalize_project_query",
        "gluon_mla_normalize_project_query_gfx1250",
        _attention_mla_normalize_project_query_gfx1250,
    ),
    _case(
        _is_cdna5,
        "cdna5",
        "attention",
        "mla_normalize_project_query",
        "gluon_mla_normalize_project_query_gfx1250",
        lambda: _attention_mla_normalize_project_query_gfx1250(16),
        id_suffix="h16",
    ),
    _case(
        _is_cdna4,
        "cdna4",
        "attention",
        "rel_mha_prefill",
        "gluon_rel_mha_prefill_gfx950",
        _attention_rel_prefill,
    ),
    _case(
        _is_cdna4,
        "cdna4",
        "attention",
        "rel_mha_extend_with_kvcache",
        "gluon_rel_mha_extend_gfx950",
        _attention_rel_extend,
    ),
    _case(
        _is_cdna4,
        "cdna4",
        "attention",
        "rel_mha_decode_with_kvcache",
        "gluon_rel_mha_decode_gfx950",
        _attention_rel_decode,
    ),
    _case(
        _is_cdna4,
        "cdna4",
        "attention",
        "rel_mha_decode_with_kvcache_page128_sliding",
        "gluon_rel_mha_decode_gfx950",
        _attention_rel_decode_page128_sliding,
    ),
    _case(
        _is_cdna4,
        "cdna4",
        "attention",
        "rel_mha_extend_with_kvcache_page256_sliding",
        "gluon_rel_mha_extend_gfx950",
        _attention_rel_extend_page256_sliding,
    ),
    _case(
        _is_cdna4,
        "cdna4",
        "attention",
        "rel_mha_decode_with_kvcache_page256_sliding",
        "gluon_rel_mha_decode_gfx950",
        _attention_rel_decode_page256_sliding,
    ),
    _case(
        _is_cdna4,
        "cdna4",
        "attention",
        "attn_merge_state",
        "triton_attn_merge_state",
        _attention_merge_state,
    ),
    _case(
        _is_cdna4,
        "cdna4",
        "attention",
        "dsa_decode",
        "gluon_dsa_decode_gfx950",
        _attention_dsa_decode,
    ),
    _case(
        _is_cdna4,
        "cdna4",
        "attention",
        "dsa_decode_fp8_dense_rank128",
        "gluon_dsa_decode_gfx950",
        _attention_dsa_decode_fp8_dense_rank128_q4,
    ),
    _case(
        _is_cdna4,
        "cdna4",
        "attention",
        "dsa_decode_fp8_e5m2_dense_rank128",
        "gluon_dsa_decode_gfx950",
        _attention_dsa_decode_fp8_e5m2_dense_rank128_q4,
    ),
    _case(
        _is_cdna4,
        "cdna4",
        "attention",
        "dsa_decode_fp8_dense_rank512",
        "gluon_dsa_decode_gfx950",
        _attention_dsa_decode_fp8_dense_rank512,
    ),
    _case(
        _is_cdna4,
        "cdna4",
        "attention",
        "dsa_decode_fp8_e5m2_dense_rank512",
        "gluon_dsa_decode_gfx950",
        _attention_dsa_decode_fp8_e5m2_dense_rank512,
    ),
    _case(
        _is_cdna4,
        "cdna4",
        "attention",
        "dsa_decode_fp8_sparse_rank512",
        "gluon_dsa_decode_gfx950",
        _attention_dsa_decode_fp8_sparse_rank512,
    ),
    _case(
        _is_cdna4,
        "cdna4",
        "attention",
        "dsa_decode_fp8_e5m2_sparse_rank512",
        "gluon_dsa_decode_gfx950",
        _attention_dsa_decode_fp8_e5m2_sparse_rank512,
    ),
    _case(
        _is_cdna4,
        "cdna4",
        "attention",
        "dsa_prefill",
        "gluon_dsa_prefill_gfx950",
        _attention_dsa_prefill,
    ),
    _case(
        _is_cdna4,
        "cdna4",
        "attention",
        "dsa_prefill_fp8_dense_rank512",
        "gluon_dsa_prefill_fp8_dense_gfx950",
        _attention_dsa_prefill_fp8_dense,
    ),
    _case(
        _is_cdna4,
        "cdna4",
        "attention",
        "dsa_prefill_fp8_e5m2_dense_rank512",
        "gluon_dsa_prefill_fp8_dense_gfx950",
        _attention_dsa_prefill_fp8_e5m2_dense,
    ),
    _case(
        _is_cdna4,
        "cdna4",
        "attention",
        "dsa_decode_topk",
        "gluon_dsa_decode_topk_fp8_gfx950",
        _attention_dsa_decode_topk,
    ),
    _case(
        _is_cdna4,
        "cdna4",
        "attention",
        "dsa_decode_topk",
        "gluon_dsa_decode_topk_fp8_gfx950",
        _attention_dsa_decode_topk_bf16_weights,
        id_suffix="bf16-weights",
    ),
    _case(
        _is_cdna4,
        "cdna4",
        "attention",
        "dsa_decode_topk",
        "triton_dsa_decode_topk_fp8",
        _attention_dsa_decode_topk_logical,
        id_suffix="logical-offsets",
    ),
    _case(
        _is_cdna4,
        "cdna4",
        "attention",
        "dsa_prefill_topk",
        "gluon_dsa_prefill_topk_fp8_gfx950",
        _attention_dsa_prefill_topk,
    ),
    _case(
        _is_cdna4,
        "cdna4",
        "attention",
        "dsa_prefill_topk",
        "gluon_dsa_prefill_topk_fp8_gfx950",
        _attention_dsa_prefill_topk_bf16_weights,
        id_suffix="bf16-weights",
    ),
    *(
        _case(
            _is_cdna4,
            "cdna4",
            "attention",
            operation,
            expected,
            lambda heads=heads, dtype=dtype, invoke=invoke: invoke(heads, dtype),
            id_suffix=f"h{heads}-{str(dtype).removeprefix('torch.')}",
        )
        for operation, expected, invoke in (
            (
                "dsa_decode_topk",
                "gluon_dsa_decode_topk_standard_gfx950",
                _attention_dsa_decode_topk_standard,
            ),
            (
                "dsa_prefill_topk",
                "gluon_dsa_prefill_topk_standard_gfx950",
                _attention_dsa_prefill_topk_standard,
            ),
        )
        for heads in (32, 64)
        for dtype in (torch.bfloat16, torch.float8_e4m3fn)
    ),
    _case(
        _is_cdna4,
        "cdna4",
        "attention",
        "dsa_plan",
        "triton_dsa_plan",
        _attention_dsa_plan,
    ),
    _case(
        _is_cdna5,
        "cdna5",
        "attention",
        "dsa_decode",
        "gluon_dsa_decode_gfx1250",
        _attention_dsa_decode,
    ),
    _case(
        _is_cdna5,
        "cdna5",
        "attention",
        "dsa_prefill",
        "gluon_dsa_prefill_gfx1250",
        _attention_dsa_prefill,
    ),
    _case(
        _is_cdna5,
        "cdna5",
        "attention",
        "dsa_decode",
        "gluon_dsa_decode_gfx1250",
        _attention_dsa_decode_fp8_dense_rank128,
    ),
    _case(
        _is_cdna5,
        "cdna5",
        "attention",
        "dsa_decode_fp8_e5m2_dense_rank128",
        "gluon_dsa_decode_gfx1250",
        _attention_dsa_decode_fp8_e5m2_dense_rank128_q4,
    ),
    _case(
        _is_cdna5,
        "cdna5",
        "attention",
        "dsa_decode_fp8_dense_rank512",
        "gluon_dsa_decode_gfx1250",
        _attention_dsa_decode_fp8_dense_rank512,
    ),
    _case(
        _is_cdna5,
        "cdna5",
        "attention",
        "dsa_decode_fp8_e5m2_dense_rank512",
        "gluon_dsa_decode_gfx1250",
        _attention_dsa_decode_fp8_e5m2_dense_rank512,
    ),
    _case(
        _is_cdna5,
        "cdna5",
        "attention",
        "dsa_decode_fp8_sparse_rank512",
        "gluon_dsa_decode_gfx1250",
        _attention_dsa_decode_fp8_sparse_rank512,
    ),
    _case(
        _is_cdna5,
        "cdna5",
        "attention",
        "dsa_decode_fp8_e5m2_sparse_rank512",
        "gluon_dsa_decode_gfx1250",
        _attention_dsa_decode_fp8_e5m2_sparse_rank512,
    ),
    _case(
        _is_cdna5,
        "cdna5",
        "attention",
        "dsa_prefill",
        "gluon_dsa_prefill_gfx1250",
        _attention_dsa_prefill_bf16_dense_rank128,
    ),
    _case(
        _is_cdna5,
        "cdna5",
        "attention",
        "dsa_prefill",
        "gluon_dsa_prefill_fp8_dense_gfx1250",
        _attention_dsa_prefill_fp8_dense,
    ),
    _case(
        _is_cdna5,
        "cdna5",
        "attention",
        "dsa_prefill_fp8_e5m2_dense_rank512",
        "gluon_dsa_prefill_fp8_dense_gfx1250",
        _attention_dsa_prefill_fp8_e5m2_dense,
    ),
    _case(
        _is_cdna5,
        "cdna5",
        "attention",
        "dsa_prefill_fp8_dense_rank128",
        "triton_dsa_prefill",
        _attention_dsa_prefill_fp8_dense_rank128,
    ),
    _case(
        _is_cdna5,
        "cdna5",
        "attention",
        "dsa_prefill_fp8_packed_rank512",
        "triton_dsa_prefill",
        _attention_dsa_prefill_fp8_packed_rank512,
    ),
    _case(
        _is_cdna5,
        "cdna5",
        "attention",
        "dsa_decode_topk",
        "gluon_dsa_decode_topk_fp8_gfx1250",
        _attention_dsa_decode_topk,
    ),
    _case(
        _is_cdna5,
        "cdna5",
        "attention",
        "dsa_decode_topk",
        "gluon_dsa_decode_topk_fp8_gfx1250",
        _attention_dsa_decode_topk_bf16_weights,
        id_suffix="bf16-weights",
    ),
    _case(
        _is_cdna5,
        "cdna5",
        "attention",
        "dsa_prefill_topk",
        "gluon_dsa_prefill_topk_fp8_gfx1250",
        _attention_dsa_prefill_topk,
    ),
    _case(
        _is_cdna5,
        "cdna5",
        "attention",
        "dsa_prefill_topk",
        "gluon_dsa_prefill_topk_fp8_gfx1250",
        _attention_dsa_prefill_topk_bf16_weights,
        id_suffix="bf16-weights",
    ),
    _case(
        _is_supported_gpu,
        "supported-gpu",
        "attention",
        "gdn_chunk_prefill",
        "triton_gdn_chunk_prefill",
        _attention_gdn_chunk_prefill,
    ),
    # GEMM API x architecture golden cases.
    _case(_is_supported_gpu, "supported-gpu", "gemm", "mm", "torch_mm", _mm_dense),
    _case(
        _is_supported_gpu,
        "supported-gpu",
        "gemm",
        "bmm",
        "torch_bmm",
        _bmm_dense,
    ),
    _case(
        _is_cdna4,
        "cdna4",
        "gemm",
        "mm",
        "torch_mm",
        _mm_dense_cdna4_aligned,
    ),
    _case(
        _is_hopper,
        "hopper",
        "gemm",
        "mm",
        "deep_gemm_mm_fp8_blockscale",
        _mm_mxfp8,
    ),
    _case(
        _is_blackwell_sm100,
        "blackwell-sm100",
        "gemm",
        "mm",
        "flashinfer_mm_fp8_blockscale",
        _mm_mxfp8,
    ),
    _case(
        _is_blackwell_plus,
        "blackwell-plus",
        "gemm",
        "mm",
        "cublaslt_mm_nvfp4",
        _mm_nvfp4,
    ),
    _case(
        _is_cdna4,
        "cdna4",
        "gemm",
        "mm",
        "triton_mm_fp8_blockscale",
        _mm_mxfp8,
    ),
    _case(
        _is_hopper_plus,
        "hopper-plus",
        "gemm",
        "dsv4_linear_fp32",
        "cuda_dsv3_dsv4_linear_fp32",
        _dsv4_linear_fp32,
    ),
    # Quantization API x architecture golden cases.
    _case(
        _is_hopper,
        "hopper",
        "quantization",
        "mxfp8",
        "triton_quantize_mxfp8",
        _quantize_mxfp8,
    ),
    _case(
        _is_blackwell_sm100,
        "blackwell-sm100",
        "quantization",
        "mxfp8",
        "flashinfer_quantize_mxfp8",
        _quantize_mxfp8,
    ),
    _case(
        _is_cdna4,
        "cdna4",
        "gemm",
        "mm",
        "triton_mm_mxfp4",
        _mm_mxfp4,
    ),
    # Sampling API x architecture golden cases.
    _case(
        _is_nvidia_with_cute_dsl,
        "nvidia-cutedsl",
        "sampling",
        "argmax",
        "cute_dsl_argmax",
        _sampling_argmax,
    ),
    _case(
        _is_cdna4,
        "cdna4",
        "sampling",
        "argmax",
        "gluon_argmax_gfx950",
        _sampling_argmax,
    ),
    # MoE API x architecture golden cases.
    _case(
        _is_hopper_plus,
        "hopper-plus",
        "moe",
        "dsv4_select_experts",
        "cuda_dsv4_select_experts",
        _dsv4_select_experts_bias,
        id_suffix="bias",
    ),
    _case(
        _is_hopper_plus,
        "hopper-plus",
        "moe",
        "dsv4_select_experts",
        "cuda_dsv4_select_experts",
        _dsv4_select_experts_hash,
        id_suffix="hash",
    ),
    _case(
        _is_cdna4,
        "cdna4",
        "moe",
        "dsv4_select_experts",
        "gluon_dsv4_select_experts_gfx950",
        _dsv4_select_experts_bias,
    ),
    _case(
        _is_hopper,
        "hopper",
        "moe",
        "apply",
        "flashinfer_cutlass_unquant_moe_apply",
        _moe_apply_unquant_cutlass,
    ),
    _case(
        _is_hopper,
        "hopper",
        "moe",
        "apply",
        "flashinfer_cutlass_fp8_moe_apply",
        _moe_apply_fp8_cutlass,
    ),
    _case(
        _is_blackwell_sm100,
        "blackwell-sm100",
        "moe",
        "apply",
        "flashinfer_trtllm_fp8_moe_apply",
        _moe_apply_fp8_trtllm,
    ),
    _case(
        _is_blackwell_sm100,
        "blackwell-sm100",
        "moe",
        "apply",
        "flashinfer_trtllm_unquant_moe_apply",
        _moe_apply_unquant_trtllm,
    ),
    _case(
        _is_blackwell_sm100,
        "blackwell-sm100",
        "moe",
        "apply",
        "flashinfer_trtllm_nvfp4_moe_apply",
        _moe_apply_nvfp4_trtllm,
    ),
    _case(
        _is_blackwell_sm100,
        "blackwell-sm100",
        "moe",
        "apply",
        "flashinfer_cutlass_nvfp4_moe_apply",
        _moe_apply_nvfp4_cutlass,
    ),
    _case(
        _is_blackwell_sm100,
        "blackwell-sm100",
        "moe",
        "apply",
        "flashinfer_trtllm_nvfp4_routed_moe_apply",
        _moe_apply_nvfp4_trtllm_routed,
    ),
    _case(
        _is_blackwell_sm100,
        "blackwell-sm100",
        "moe",
        "apply",
        "flashinfer_trtllm_nvfp4_moe_apply",
        _moe_apply_nvfp4_trtllm_unconstrained_routing,
    ),
    _case(
        _is_blackwell_sm100,
        "blackwell-sm100",
        "moe",
        "apply",
        "flashinfer_trtllm_unquant_routed_moe_apply",
        _moe_apply_unquant_trtllm_routed,
    ),
    _case(
        _is_blackwell_plus,
        "blackwell-plus",
        "moe",
        "apply",
        "flashinfer_cutedsl_deepep_nvfp4_moe_apply",
        _moe_apply_nvfp4_deepep_cutedsl,
    ),
    _case(
        _is_hopper_plus_with_deep_gemm,
        "hopper-plus",
        "moe",
        "apply",
        "deep_gemm_deepep_fp8_moe_apply",
        _moe_apply_fp8_deepep_deep_gemm,
    ),
    _case(
        _is_blackwell_sm100,
        "blackwell-sm100",
        "moe",
        "apply",
        "flashinfer_trtllm_mxfp4_moe_apply",
        _moe_apply_mxfp4_trtllm,
    ),
    _case(
        _is_blackwell_sm100,
        "blackwell-sm100",
        "moe",
        "apply",
        "flashinfer_trtllm_mxint4_moe_apply",
        _moe_apply_mxint4_trtllm,
    ),
    _case(
        _is_cdna4,
        "cdna4",
        "mhc",
        "pre",
        "triton_mhc_pre",
        _mhc_pre,
    ),
    _case(
        _is_cdna4,
        "cdna4",
        "mhc",
        "post",
        "triton_mhc_post",
        _mhc_post,
    ),
    _case(
        _is_cdna4,
        "cdna4",
        "moe",
        "apply",
        "triton_mxfp4_precomputed_moe_apply",
        _moe_apply_mxfp4_triton,
    ),
    _case(
        _is_supported_gpu,
        "supported-gpu",
        "moe",
        "apply",
        "triton_bf16_precomputed_moe_apply",
        _moe_apply_unquant_triton,
    ),
    _case(
        _is_cdna4,
        "cdna4",
        "moe",
        "apply",
        "gluon_mxfp4_moe_apply",
        _moe_apply_mxfp4_gluon,
    ),
    _case(
        _is_cdna4,
        "cdna4",
        "moe",
        "apply",
        "gluon_mxfp4_dynamic_moe_apply",
        _moe_apply_mxfp4_dynamic_tp,
    ),
]


@pytest.fixture
def selected_kernel_spy(monkeypatch):
    active_case: dict[str, KernelApiSelectionCase | None] = {"case": None}
    calls: list[str] = []

    def fake_call(self: SelectedKernel, *args, **kwargs):
        case = active_case["case"]
        assert case is not None, "selected_kernel_spy used without an active case"
        calls.append(self.name)

        if case.family == "gemm":
            if case.mode == "dsv4_linear_fp32":
                hidden_states, weight = args[:2]
                return torch.empty(
                    (*hidden_states.shape[:-1], weight.shape[0]),
                    dtype=torch.float32,
                    device=hidden_states.device,
                )
            a, b, _a_scales, _b_scales, out_dtype = args[:5]
            if case.mode == "bmm":
                n = b.shape[-2]
                return torch.empty(
                    (a.shape[0], a.shape[1], n), dtype=out_dtype, device=a.device
                )
            n = b.shape[-1] if b.shape[0] == a.shape[-1] else b.shape[0]
            return torch.empty((a.shape[0], n), dtype=out_dtype, device=a.device)

        if case.family == "attention":
            if case.mode == "attn_merge_state":
                return torch.empty_like(kwargs["out_a"]), torch.empty_like(
                    kwargs["lse_a"]
                )
            if case.mode == "dsa_plan":
                return torch.empty((1, 4), dtype=torch.int32)
            if case.mode in {
                "mla_decode_projected_value",
                "mla_normalize_project_query",
                "mla_project_value",
            }:
                return kwargs["out"]
            q = kwargs["q"]
            if case.mode == "gdn_chunk_prefill":
                return GdnChunkPrefillResult(
                    out=torch.empty_like(q),
                    final_state=kwargs.get("initial_state"),
                )
            if kwargs.get("return_lse", False):
                lse = torch.empty(q.shape[:-1], dtype=torch.float32, device=q.device)
                return torch.empty_like(q), lse
            return torch.empty_like(q)

        if case.family == "sampling":
            (logits,) = args[:1]
            out = kwargs.get("out")
            if out is not None:
                return out
            return torch.empty(
                (logits.shape[0],), dtype=torch.int64, device=logits.device
            )

        if case.family == "moe":
            if case.mode == "dsv4_select_experts":
                router_logits, top_k = args[:2]
                shape = (router_logits.shape[0], top_k)
                return (
                    torch.empty(shape, dtype=torch.float32),
                    torch.empty(shape, dtype=torch.int32),
                    torch.empty_like(router_logits, dtype=torch.float32),
                )
            return torch.empty_like(kwargs["x"])

        if case.family == "mhc":
            if case.mode == "pre":
                residual = args[0]
                return (
                    torch.empty(
                        (*residual.shape[:-2], residual.shape[-1]),
                        dtype=residual.dtype,
                    ),
                    torch.empty(
                        (*residual.shape[:-1], 1),
                        dtype=torch.float32,
                    ),
                    torch.empty(
                        (*residual.shape[:-2], residual.shape[-2], residual.shape[-2]),
                        dtype=torch.float32,
                    ),
                )
            return torch.empty_like(args[1])

        return None

    monkeypatch.setattr(SelectedKernel, "__call__", fake_call)
    return active_case, calls


def _find_case(*, arch: str, family: str, mode: str) -> KernelApiSelectionCase:
    for case in _CASES:
        if case.arch == arch and case.family == family and case.mode == mode:
            return case
    raise AssertionError(f"missing golden case for {arch}/{family}.{mode}")


# Fixture platforms (see conftest.py) each case's arch tag runs under.
_ARCH_FIXTURES: dict[str, tuple[str, ...]] = {
    "hopper": ("h100_platform",),
    "hopper-plus": ("h100_platform", "b200_platform", "b300_platform"),
    "blackwell-sm100": ("b200_platform",),
    "blackwell-sm103": ("b300_platform",),
    "blackwell-plus": ("b200_platform", "b300_platform"),
    "cdna4": ("mi350_platform",),
    "cdna5": ("mi450_platform",),
    "supported-gpu": (
        "h100_platform",
        "b200_platform",
        "b300_platform",
        "mi350_platform",
    ),
    "nvidia-cutedsl": ("h100_platform", "b200_platform", "b300_platform"),
}


def test_mxfp8_quantizer_capabilities_match_architecture(
    h100_platform: PlatformInfo,
    b200_platform: PlatformInfo,
) -> None:
    if not Platform.get().is_nvidia:
        pytest.skip("FlashInfer quantization kernels are registered only on NVIDIA")

    registry = KernelRegistry.get()
    h100_names = {
        spec.name
        for spec in registry.get_for_operator(
            "quantization", "mxfp8", platform=h100_platform
        )
    }
    b200_names = {
        spec.name
        for spec in registry.get_for_operator(
            "quantization", "mxfp8", platform=b200_platform
        )
    }

    assert "flashinfer_quantize_mxfp8" not in h100_names
    assert "triton_quantize_mxfp8" in h100_names
    assert "flashinfer_quantize_mxfp8" in b200_names
    assert "triton_quantize_mxfp8" in b200_names


def test_b300_rel_decode_registration_and_selection(
    b300_platform: PlatformInfo,
    selected_kernel_spy,
) -> None:
    if (
        not Platform.get().is_nvidia
        or importlib.util.find_spec("flash_attn.cute") is None
    ):
        pytest.skip("B300 registration simulation requires NVIDIA FA4")

    case = _find_case(
        arch="blackwell-sm103",
        family="attention",
        mode="rel_mha_decode_with_kvcache",
    )
    real_platform = Platform.get()
    active_case, calls = selected_kernel_spy
    active_case["case"] = case

    try:
        Platform.override(b300_platform)
        KernelRegistry.reset()
        importlib.reload(_attention_flash_attn)
        registry = KernelRegistry.get()

        expected_spec = registry.get_by_name(case.expected)
        assert expected_spec is not None
        assert expected_spec.capability.satisfied_by(b300_platform)

        plain_decode = registry.get_by_name("fa4_mha_decode_with_kvcache")
        assert plain_decode is not None
        assert not plain_decode.capability.satisfied_by(b300_platform)

        case.invoke()
        _attention_rel_decode_multiquery_full()

        assert calls == [case.expected, case.expected]
    finally:
        Platform.override(real_platform)
        KernelRegistry.reset()
        importlib.reload(_attention_flash_attn)


def test_attn_merge_state_routes_to_triton_on_cdna4(
    mi350_platform: PlatformInfo,
    selected_kernel_spy,
) -> None:
    case = _find_case(arch="cdna4", family="attention", mode="attn_merge_state")
    registry = KernelRegistry.get()
    expected_spec = registry.get_by_name(case.expected)
    assert expected_spec is not None
    assert expected_spec.capability.satisfied_by(mi350_platform)

    real_platform = Platform.get()
    active_case, calls = selected_kernel_spy
    active_case["case"] = case
    try:
        Platform.override(mi350_platform)
        registry.clear_cache()

        case.invoke()

        assert calls == ["triton_attn_merge_state"]
    finally:
        Platform.override(real_platform)
        registry.clear_cache()


_GLUON_MLA_FIXED_KERNELS = (
    "gluon_mla_decode_bf16xbf16_gfx950_bh16bn64",
    "gluon_mla_decode_bf16xbf16_gfx950_bh64",
    "gluon_mla_decode_bf16xbf16_gfx950_bh16_multiblock",
    "gluon_mla_decode_bf16xbf16_gfx950_bh64_small",
)


@pytest.mark.parametrize(
    "trait,value,matches",
    [
        pytest.param("num_q_heads", 12, True, id="matched"),
        pytest.param("num_q_heads", 16, True, id="h16"),
        pytest.param("num_q_heads", 32, False, id="unsupported-heads"),
        pytest.param("value_head_dim", 64, False, id="unsupported-value"),
        pytest.param("page_size", 128, False, id="unsupported-page"),
        pytest.param("support_logit_cap", True, False, id="unsupported-logit-cap"),
    ],
)
def test_gluon_mla_projected_value_gfx1250_traits_are_narrow(
    trait: str,
    value: object,
    matches: bool,
) -> None:
    spec = KernelRegistry.get().get_by_name("gluon_mla_decode_projected_value_gfx1250")
    if spec is None:
        pytest.skip("gfx1250 Gluon MLA registration is unavailable")
    traits = {
        "batch_size": 1,
        "q_len": 1,
        "num_q_heads": 12,
        "page_size": 64,
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
        "value_head_dim": 128,
        "gate_kind": "sigmoid",
        "support_logit_cap": False,
    }
    traits[trait] = value
    assert spec_matches_traits(spec, traits) is matches


def _require_gluon_mla_fixed_kernel(name: str):
    registry = KernelRegistry.get()
    if registry.get_by_name(_GLUON_MLA_FIXED_KERNELS[0]) is None:
        pytest.skip("gfx950 Gluon MLA registrations are unavailable")
    spec = registry.get_by_name(name)
    assert spec is not None
    return spec


@pytest.mark.parametrize("name", _GLUON_MLA_FIXED_KERNELS)
def test_gluon_mla_fixed_entrypoints_are_registered(name: str) -> None:
    _require_gluon_mla_fixed_kernel(name)


@pytest.mark.parametrize(
    "name,expected_batches",
    [
        pytest.param(
            "gluon_mla_decode_bf16xbf16_gfx950_bh16_multiblock",
            frozenset({1}),
            id="bh16-multiblock",
        ),
        pytest.param(
            "gluon_mla_decode_bf16xbf16_gfx950_bh64_small",
            frozenset({2, 4}),
            id="bh64-small",
        ),
    ],
)
@pytest.mark.parametrize("batch", [1, 2, 3, 4, 64])
def test_gluon_mla_small_batch_registrations_have_disjoint_traits(
    name: str,
    expected_batches: frozenset[int],
    batch: int,
) -> None:
    spec = _require_gluon_mla_fixed_kernel(name)

    traits = {
        "batch_size": batch,
        "batch_size_div_64": batch % 64 == 0,
        "q_len": 1,
        "num_q_heads": 64,
        "page_size": 64,
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
        "support_logit_cap": False,
        "return_lse": False,
    }
    assert spec_matches_traits(spec, traits) is (batch in expected_batches)


@pytest.mark.parametrize(
    "batch,expected",
    [
        pytest.param(
            1,
            "gluon_mla_decode_bf16xbf16_gfx950_bh16_multiblock",
            id="b1-bh16-multiblock",
        ),
        pytest.param(
            2,
            "gluon_mla_decode_bf16xbf16_gfx950_bh64_small",
            id="b2-bh64-small",
        ),
        pytest.param(
            4,
            "gluon_mla_decode_bf16xbf16_gfx950_bh64_small",
            id="b4-bh64-small",
        ),
        pytest.param(
            64,
            "gluon_mla_decode_bf16xbf16_gfx950_bh64",
            id="b64-bh64",
        ),
    ],
)
def test_gluon_mla_fixed_regime_auto_selection(
    batch: int,
    expected: str,
    mi350_platform: PlatformInfo,
    selected_kernel_spy,
) -> None:
    _require_gluon_mla_fixed_kernel(expected)
    case = _case(
        _is_cdna4,
        "cdna4",
        "attention",
        "mla_decode_with_kvcache",
        expected,
        lambda: _attention_mla_decode(batch),
    )
    host_platform = Platform.get()
    active_case, calls = selected_kernel_spy
    active_case["case"] = case
    registry = KernelRegistry.get()
    try:
        Platform.override(mi350_platform)
        registry.clear_cache()
        case.invoke()
    finally:
        Platform.override(host_platform)
        registry.clear_cache()

    assert calls == [expected]


_CASE_PLATFORM_PARAMS = [
    pytest.param(
        case,
        fixture_name,
        id=f"{case.id}@{fixture_name.removesuffix('_platform')}",
    )
    for case in _CASES
    for fixture_name in _ARCH_FIXTURES[case.arch]
]


@pytest.mark.parametrize(("case", "platform_fixture"), _CASE_PLATFORM_PARAMS)
def test_kernel_api_selection(
    case: KernelApiSelectionCase,
    platform_fixture: str,
    selected_kernel_spy,
    request: pytest.FixtureRequest,
):
    platform = request.getfixturevalue(platform_fixture)
    host_platform = Platform.get()

    registry = KernelRegistry.get()
    expected_spec = registry.get_by_name(case.expected)
    if expected_spec is None:
        # Registrations are import-guarded on optional backend packages, so a
        # missing spec is only a failure when this host should run the case
        # natively.
        assert not case.matches(host_platform), (
            f"{case.expected!r} is not registered on "
            f"{host_platform.device_name} ({host_platform.arch_version})"
        )
        pytest.skip(f"{case.expected!r} is not registered (optional backend missing)")
    assert expected_spec.capability.satisfied_by(platform), (
        f"{case.expected!r} is registered but not compatible with "
        f"{platform.device_name} ({platform.arch_version})"
    )

    active_case, calls = selected_kernel_spy
    active_case["case"] = case
    try:
        Platform.override(platform)
        registry.clear_cache()

        case.invoke()
    finally:
        Platform.override(host_platform)
        registry.clear_cache()

    assert calls == [case.expected]

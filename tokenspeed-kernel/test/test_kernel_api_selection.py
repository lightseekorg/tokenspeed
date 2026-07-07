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

"""Golden selection tests for top-level tokenspeed-kernel public APIs."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Callable

import pytest
import tokenspeed_kernel
import tokenspeed_kernel.numerics.reference.gemm as _gemm_reference
import tokenspeed_kernel.ops.attention as _attention_pkg
import tokenspeed_kernel.ops.attention.cuda as _attention_cuda
import tokenspeed_kernel.ops.attention.flash_attn as _attention_flash_attn
import tokenspeed_kernel.ops.attention.flash_mla as _attention_flash_mla
import tokenspeed_kernel.ops.attention.flashinfer as _attention_flashinfer
import tokenspeed_kernel.ops.attention.flashinfer.gated_delta_rule as _attention_flashinfer_gdn
import tokenspeed_kernel.ops.attention.gluon as _attention_gluon
import tokenspeed_kernel.ops.attention.triton as _attention_triton
import tokenspeed_kernel.ops.gemm as _gemm_pkg
import tokenspeed_kernel.ops.gemm.deep_gemm as _gemm_deep_gemm
import tokenspeed_kernel.ops.gemm.flashinfer as _gemm_flashinfer
import tokenspeed_kernel.ops.gemm.triton as _gemm_triton
import tokenspeed_kernel.ops.gemm.trtllm as _gemm_trtllm
import tokenspeed_kernel.ops.moe as _moe_pkg
import tokenspeed_kernel.ops.moe.flashinfer as _moe_flashinfer
import tokenspeed_kernel.ops.moe.gluon as _moe_gluon
import tokenspeed_kernel.ops.moe.triton as _moe_triton
import tokenspeed_kernel.ops.sampling as _sampling_pkg
import tokenspeed_kernel.ops.sampling.cute_dsl as _sampling_cute_dsl
import tokenspeed_kernel.ops.sampling.gluon as _sampling_gluon
import torch
from tokenspeed_kernel.ops.attention.gdn_utils import GdnChunkPrefillResult
from tokenspeed_kernel.ops.attention.triton import dsa as _attention_triton_dsa
from tokenspeed_kernel.ops.attention.triton import (
    dsa_topk as _attention_triton_dsa_topk,
)
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
from tokenspeed_kernel.ops.moe.triton import fp8 as _moe_triton_fp8
from tokenspeed_kernel.ops.moe.triton import mxfp4 as _moe_triton_mxfp4
from tokenspeed_kernel.platform import ArchVersion, Platform, PlatformInfo
from tokenspeed_kernel.registry import KernelRegistry
from tokenspeed_kernel.selection import SelectedKernel

_RELOAD_MODULES = [
    # Attention registration modules.
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
    _attention_triton_merge_state,
    _attention_triton_dsa,
    _attention_triton_dsa_topk,
    _attention_triton_gdn,
    _attention_triton,
    _attention_pkg,
    # GEMM registration modules.
    _gemm_reference,
    _gemm_deep_gemm,
    _gemm_flashinfer,
    _gemm_triton,
    _gemm_trtllm,
    _gemm_pkg,
    # MoE registration modules.
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
    _moe_gluon_mxfp4,
    _moe_gluon,
    _moe_triton_fp8,
    _moe_triton_mxfp4,
    _moe_triton,
    _moe_pkg,
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
    matches: Callable[[PlatformInfo], bool]
    invoke: Callable[[], object]


def _is_hopper(platform: PlatformInfo) -> bool:
    return platform.is_hopper


def _is_blackwell_sm100(platform: PlatformInfo) -> bool:
    return platform.is_blackwell and platform.arch_version == ArchVersion(10, 0)


def _is_blackwell_non_sm100(platform: PlatformInfo) -> bool:
    return platform.is_blackwell and platform.arch_version != ArchVersion(10, 0)


def _is_blackwell_plus(platform: PlatformInfo) -> bool:
    return platform.is_blackwell_plus


def _is_hopper_plus(platform: PlatformInfo) -> bool:
    return platform.is_nvidia and platform.arch_version >= ArchVersion(9, 0)


def _is_nvidia(platform: PlatformInfo) -> bool:
    return platform.is_nvidia


def _is_nvidia_with_cute_dsl(platform: PlatformInfo) -> bool:
    return platform.is_nvidia and _sampling_cute_dsl.is_available()


def _is_cdna4(platform: PlatformInfo) -> bool:
    return platform.is_cdna4


def _is_supported_gpu(platform: PlatformInfo) -> bool:
    return platform.is_nvidia or platform.is_amd


def _fp8_dtype() -> torch.dtype:
    return Platform.get().fp8e4m3fn.dtype


def _mm_dense() -> torch.Tensor:
    a = torch.empty((4, 16), dtype=torch.bfloat16)
    b = torch.empty((32, 16), dtype=torch.bfloat16)
    return tokenspeed_kernel.mm(a, b)


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
        solution="triton",
    )


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
        solution="triton",
    )


def _attention_dsa_decode_topk() -> object:
    q = torch.empty((2, 2, 128), dtype=torch.bfloat16)
    weights = torch.empty((2, 2), dtype=torch.float32)
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


def _attention_dsa_prefill_topk() -> object:
    q = torch.empty((2, 2, 128), dtype=torch.bfloat16)
    weights = torch.empty((2, 2), dtype=torch.float32)
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
        page_size=64,
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


def _moe_apply_nvfp4_deepep_cutedsl() -> object:
    plan = tokenspeed_kernel.moe_plan(
        "nvfp4",
        input_dtype=torch.bfloat16,
        activation="swiglu",
        a2a_backend="deepep",
        ep_size=2,
        ispp=128,
        internal_activation_dtype="input",
        deepep_group=object(),
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
        ispp=128,
        internal_activation_dtype="fp8",
        with_bias=True,
    )
    _assert_moe_plan(
        plan,
        apply="triton_mxfp4_moe_apply",
        preprocessor="triton_mxfp4_moe_weights",
    )
    x = torch.empty((4, 16), dtype=torch.bfloat16)
    router_logits = torch.empty((4, 8), dtype=torch.float32)
    return tokenspeed_kernel.moe_apply(plan, x, torch.nn.Module(), router_logits)


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


def _moe_apply_mxfp4_precomputed_tp() -> object:
    plan = tokenspeed_kernel.moe_plan(
        "mxfp4",
        input_dtype=torch.bfloat16,
        activation="silu",
        ep_size=1,
        ispp=2048,
        internal_activation_dtype="fp8",
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


def _moe_apply_fp8_precomputed_tp() -> object:
    plan = _moe_pkg.moe_plan(
        "fp8",
        input_dtype=torch.bfloat16,
        activation="silu",
        ep_size=1,
        fp8_scale_block_shape=(128, 128),
        solution="triton",
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


def _moe_apply_fp8_precomputed_ep() -> object:
    plan = _moe_pkg.moe_plan(
        "fp8",
        input_dtype=torch.bfloat16,
        activation="silu",
        ep_size=2,
        fp8_scale_block_shape=(128, 128),
        solution="triton",
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


def _moe_apply_mxfp4_precomputed_ep() -> object:
    plan = tokenspeed_kernel.moe_plan(
        "mxfp4",
        input_dtype=torch.bfloat16,
        activation="silu",
        ep_size=4,
        internal_activation_dtype="input",
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


def test_mxfp4_ep_topk_localization_masks_remote_experts() -> None:
    w = torch.nn.Module()
    w.num_experts = 8
    w.num_local_experts = 2
    w.ep_rank = 2
    w.ep_size = 4
    topk_weights = torch.tensor(
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        dtype=torch.float32,
    )
    topk_ids = torch.tensor([[0, 4, 5], [6, 7, 3]], dtype=torch.int64)

    local_weights, local_ids, num_experts = _moe_triton_mxfp4._local_topk_for_ep(
        topk_weights,
        topk_ids,
        w,
    )

    assert num_experts == 2
    torch.testing.assert_close(
        local_weights,
        torch.tensor([[0.0, 0.2, 0.3], [0.0, 0.0, 0.0]], dtype=torch.float32),
    )
    assert torch.equal(
        local_ids,
        torch.tensor([[-1, 0, 1], [-1, -1, -1]], dtype=torch.int64),
    )


def _case(
    matches: Callable[[PlatformInfo], bool],
    arch: str,
    family: str,
    mode: str,
    expected: str,
    invoke: Callable[[], object],
) -> KernelApiSelectionCase:
    return KernelApiSelectionCase(
        id=f"{arch}/{family}.{mode}/{expected}",
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
        _is_blackwell_non_sm100,
        "blackwell-non-sm100",
        "attention",
        "mha_extend_with_kvcache",
        "flashinfer_trtllm_mha_extend_with_kvcache",
        _attention_extend,
    ),
    _case(
        _is_blackwell_non_sm100,
        "blackwell-non-sm100",
        "attention",
        "mha_decode_with_kvcache",
        "flashinfer_trtllm_mha_decode_with_kvcache",
        _attention_decode,
    ),
    _case(
        _is_blackwell_non_sm100,
        "blackwell-non-sm100",
        "attention",
        "attn_merge_state",
        "cuda_attn_merge_state",
        _attention_merge_state,
    ),
    _case(
        _is_cdna4,
        "cdna4",
        "attention",
        "mha_prefill",
        "gluon_mha_prefill_fp16_gfx950",
        _attention_prefill,
    ),
    _case(
        _is_cdna4,
        "cdna4",
        "attention",
        "mha_extend_with_kvcache",
        "gluon_mha_extend_fp16_gfx950",
        _attention_extend,
    ),
    _case(
        _is_cdna4,
        "cdna4",
        "attention",
        "mha_decode_with_kvcache",
        "gluon_mha_decode_fp16_gfx950",
        _attention_decode,
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
        "triton_dsa_decode",
        _attention_dsa_decode,
    ),
    _case(
        _is_cdna4,
        "cdna4",
        "attention",
        "dsa_prefill",
        "triton_dsa_prefill",
        _attention_dsa_prefill,
    ),
    _case(
        _is_cdna4,
        "cdna4",
        "attention",
        "dsa_decode_topk",
        "triton_dsa_decode_topk_fp8",
        _attention_dsa_decode_topk,
    ),
    _case(
        _is_cdna4,
        "cdna4",
        "attention",
        "dsa_prefill_topk",
        "triton_dsa_prefill_topk_fp8",
        _attention_dsa_prefill_topk,
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
        _is_blackwell_plus,
        "blackwell-plus",
        "moe",
        "apply",
        "flashinfer_cutedsl_deepep_nvfp4_moe_apply",
        _moe_apply_nvfp4_deepep_cutedsl,
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
        _is_hopper,
        "hopper",
        "moe",
        "apply",
        "triton_mxfp4_moe_apply",
        _moe_apply_mxfp4_triton,
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
        "triton_mxfp4_precomputed_moe_apply",
        _moe_apply_mxfp4_precomputed_tp,
    ),
    _case(
        _is_cdna4,
        "cdna4",
        "moe",
        "apply",
        "triton_mxfp4_ep_precomputed_moe_apply",
        _moe_apply_mxfp4_precomputed_ep,
    ),
    _case(
        _is_cdna4,
        "cdna4",
        "moe",
        "apply",
        "triton_fp8_ep_precomputed_moe_apply",
        _moe_apply_fp8_precomputed_tp,
    ),
    _case(
        _is_cdna4,
        "cdna4",
        "moe",
        "apply",
        "triton_fp8_ep_precomputed_moe_apply",
        _moe_apply_fp8_precomputed_ep,
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
            a, b, _a_scales, _b_scales, out_dtype = args[:5]
            n = b.shape[-1] if b.shape[0] == a.shape[-1] else b.shape[0]
            return torch.empty((a.shape[0], n), dtype=out_dtype, device=a.device)

        if case.family == "attention":
            if case.mode == "attn_merge_state":
                return torch.empty_like(kwargs["out_a"]), torch.empty_like(
                    kwargs["lse_a"]
                )
            if case.mode == "dsa_plan":
                return torch.empty((1, 4), dtype=torch.int32)
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
            return torch.empty_like(kwargs["x"])

        return None

    monkeypatch.setattr(SelectedKernel, "__call__", fake_call)
    return active_case, calls


def _find_case(*, arch: str, family: str, mode: str) -> KernelApiSelectionCase:
    for case in _CASES:
        if case.arch == arch and case.family == family and case.mode == mode:
            return case
    raise AssertionError(f"missing golden case for {arch}/{family}.{mode}")


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


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.id)
def test_kernel_api_selection(case: KernelApiSelectionCase, selected_kernel_spy):
    platform = Platform.get()
    if not case.matches(platform):
        pytest.skip(
            f"{case.id} only applies to its {case.arch} architecture case; "
            f"current platform is {platform.device_name} ({platform.arch_version})"
        )

    registry = KernelRegistry.get()
    expected_spec = registry.get_by_name(case.expected)
    assert expected_spec is not None, (
        f"{case.expected!r} is not registered on "
        f"{platform.device_name} ({platform.arch_version})"
    )
    assert expected_spec.capability.satisfied_by(platform), (
        f"{case.expected!r} is registered but not compatible with "
        f"{platform.device_name} ({platform.arch_version})"
    )

    active_case, calls = selected_kernel_spy
    active_case["case"] = case
    registry.clear_cache()

    case.invoke()

    assert calls == [case.expected]

import pytest
import tokenspeed_kernel.ops.attention.flashinfer as flashinfer_attention
import torch
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.selection import select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature


def _skip_without_flashinfer_fp8_mha_platform():
    platform = current_platform()
    if not (platform.is_nvidia and platform.is_hopper_plus):
        pytest.skip("FlashInfer FP8 MHA path is registered only on Hopper+ NVIDIA GPUs")


def _fp8_cache_signature():
    return format_signature(
        q=dense_tensor_format(torch.float8_e4m3fn),
        k_cache=dense_tensor_format(torch.float8_e4m3fn),
        v_cache=dense_tensor_format(torch.float8_e4m3fn),
    )


def _bf16_query_fp8_cache_signature():
    return format_signature(
        q=dense_tensor_format(torch.bfloat16),
        k_cache=dense_tensor_format(torch.float8_e4m3fn),
        v_cache=dense_tensor_format(torch.float8_e4m3fn),
    )


def test_flashinfer_mha_decode_selects_bf16_query_fp8_kv_cache_signature():
    _skip_without_flashinfer_fp8_mha_platform()

    kernel = select_kernel(
        "attention",
        "mha_decode_with_kvcache",
        _bf16_query_fp8_cache_signature(),
        traits={
            "head_dim": 128,
            "page_size": 1,
            "sliding_window": False,
            "support_logit_cap": False,
            "support_sinks": False,
            "return_lse": False,
        },
        solution="flashinfer",
    )

    assert kernel.name == "flashinfer_mha_decode_with_kvcache"


def test_flashinfer_trtllm_mha_extend_selects_fp8_kv_cache_signature():
    _skip_without_flashinfer_fp8_mha_platform()

    kernel = select_kernel(
        "attention",
        "mha_extend_with_kvcache",
        _fp8_cache_signature(),
        traits={
            "head_dim": 128,
            "page_size": 64,
            "is_causal": True,
            "sliding_window": False,
            "support_logit_cap": False,
            "support_sinks": False,
            "return_lse": False,
        },
        solution="flashinfer",
    )

    assert kernel.name == "flashinfer_trtllm_mha_extend_with_kvcache"


def test_flashinfer_mha_extend_selects_bf16_query_fp8_kv_cache_signature():
    _skip_without_flashinfer_fp8_mha_platform()

    kernel = select_kernel(
        "attention",
        "mha_extend_with_kvcache",
        _bf16_query_fp8_cache_signature(),
        traits={
            "head_dim": 128,
            "page_size": 1,
            "is_causal": True,
            "sliding_window": False,
            "support_logit_cap": False,
            "support_sinks": False,
            "return_lse": False,
        },
        solution="flashinfer",
    )

    assert kernel.name == "flashinfer_mha_extend_with_kvcache"


def test_flashinfer_decode_tensor_core_policy_matches_fp8_gqa_path():
    assert flashinfer_attention._should_use_decode_tensor_cores(
        torch.float8_e4m3fn,
        num_q_heads=32,
        num_kv_heads=2,
    )
    assert flashinfer_attention._should_use_decode_tensor_cores(
        torch.bfloat16,
        num_q_heads=32,
        num_kv_heads=2,
    )
    assert not flashinfer_attention._should_use_decode_tensor_cores(
        torch.bfloat16,
        num_q_heads=2,
        num_kv_heads=2,
    )


def test_nvfp4_dense_uses_flashinfer_weight_layout_when_available(monkeypatch):
    from tokenspeed.runtime.layers.dense import nvfp4 as dense_nvfp4
    from tokenspeed.runtime.layers.linear import ReplicatedLinear
    from tokenspeed.runtime.layers.quantization.nvfp4 import Nvfp4Config

    layer = ReplicatedLinear(
        input_size=32,
        output_size=96,
        bias=False,
        quant_config=Nvfp4Config(group_size=16),
    )
    layer.weight.data.zero_()
    layer.weight_scale.data.zero_()
    layer.input_scale.data.fill_(2.0)
    layer.weight_scale_2.data.fill_(3.0)

    def fake_prepare(weight, scale):
        assert weight.shape == (96, 16)
        assert scale.shape == (96, 2)
        return (
            torch.empty((128, 24), dtype=torch.uint8),
            torch.empty((128, 4), dtype=torch.float8_e4m3fn),
            96,
            8,
        )

    monkeypatch.setattr(
        dense_nvfp4,
        "prepare_nvfp4_weight_for_fp4_gemm",
        fake_prepare,
    )

    layer.quant_method.process_weights_after_loading(layer)

    assert layer._nvfp4_dense_kernel_name == "flashinfer_mm_nvfp4"
    assert layer.output_size_per_partition == 96
    assert layer.weights_padding_cols == 8
    assert layer.weight.shape == (128, 24)
    assert layer.weight_scale_interleaved.shape == (128, 4)
    assert not hasattr(layer, "weight_scale")
    torch.testing.assert_close(layer.alpha, torch.tensor(6.0))
    torch.testing.assert_close(layer.input_scale_inv, torch.tensor(0.5))


def test_prepare_nvfp4_weight_for_fp4_gemm_matches_sglang_padding():
    from tokenspeed_kernel.ops.quantization import flashinfer as flashinfer_quant
    from tokenspeed_kernel.registry import error_fn

    if flashinfer_quant.prepare_nvfp4_weight_for_fp4_gemm is error_fn:
        pytest.skip(
            "FlashInfer NVFP4 weight preparation is only available on Blackwell"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight = torch.empty((96, 24), dtype=torch.uint8, device=device)
    scale = torch.empty((96, 3), dtype=torch.float8_e4m3fn, device=device)

    weight, scale, output_size, weights_padding_cols = (
        flashinfer_quant.prepare_nvfp4_weight_for_fp4_gemm(weight, scale)
    )

    assert output_size == 96
    assert weights_padding_cols == 8
    assert weight.shape == (96, 32)
    assert scale.shape == (128, 4)


def test_nvfp4_dense_cutlass_scale_swizzle_matches_sglang_layout():
    from tokenspeed.runtime.layers.dense.nvfp4 import swizzle_blockscale_2d

    scales = torch.arange(130 * 5, dtype=torch.int16).reshape(130, 5)
    expected = torch.zeros((256, 8), dtype=scales.dtype)
    expected[:130, :5] = scales
    expected = expected.reshape(1, 2, 4, 32, 2, 4)
    expected = expected.permute(0, 1, 4, 3, 2, 5).contiguous().reshape(256, 8)

    actual = swizzle_blockscale_2d(scales)

    assert actual.shape == (256, 8)
    torch.testing.assert_close(actual, expected)


def test_nvfp4_dense_flashinfer_forward_pads_activation_and_slices_output(
    monkeypatch,
):
    from tokenspeed.runtime.layers.dense import nvfp4 as dense_nvfp4
    from tokenspeed.runtime.layers.linear import ReplicatedLinear
    from tokenspeed.runtime.layers.quantization.nvfp4 import Nvfp4Config

    layer = ReplicatedLinear(
        input_size=32,
        output_size=96,
        bias=False,
        quant_config=Nvfp4Config(group_size=16),
    )
    layer.weight.data.zero_()
    layer.weight_scale.data.zero_()
    layer.input_scale.data.fill_(2.0)
    layer.weight_scale_2.data.fill_(3.0)

    monkeypatch.setattr(
        dense_nvfp4,
        "prepare_nvfp4_weight_for_fp4_gemm",
        lambda weight, scale: (
            torch.empty((128, 24), dtype=torch.uint8),
            torch.empty((128, 4), dtype=torch.float8_e4m3fn),
            96,
            8,
        ),
    )
    layer.quant_method.process_weights_after_loading(layer)

    def fake_fp4_quantize(x, scale, **kwargs):
        assert x.shape == (2, 32)
        assert set(kwargs) == {"enable_pdl"}
        assert isinstance(kwargs["enable_pdl"], bool)
        torch.testing.assert_close(scale, torch.tensor(0.5))
        return (
            torch.empty((2, 16), dtype=torch.uint8),
            torch.empty((128, 2), dtype=torch.float8_e4m3fn),
        )

    captured = {}

    def fake_mm(A, B, **kwargs):
        captured["A_shape"] = A.shape
        captured["B_shape"] = B.shape
        captured["B_scales_shape"] = kwargs["B_scales"].shape
        captured["override"] = kwargs["override"]
        captured["expected_kernel_name"] = kwargs["expected_kernel_name"]
        captured["quant"] = kwargs["quant"]
        captured["out_dtype"] = kwargs["out_dtype"]
        return torch.empty((2, 128), dtype=kwargs["out_dtype"])

    monkeypatch.setattr(dense_nvfp4, "fp4_quantize", fake_fp4_quantize)
    monkeypatch.setattr(dense_nvfp4.tokenspeed_kernel, "mm", fake_mm)

    out, bias = layer(torch.empty((2, 32), dtype=torch.bfloat16))

    assert bias is None
    assert out.shape == (2, 96)
    assert captured == {
        "A_shape": torch.Size([2, 24]),
        "B_shape": torch.Size([24, 128]),
        "B_scales_shape": torch.Size([4, 128]),
        "override": "flashinfer_mm_nvfp4",
        "expected_kernel_name": "flashinfer_mm_nvfp4",
        "quant": "nvfp4",
        "out_dtype": torch.bfloat16,
    }

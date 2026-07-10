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
from math import prod
from types import SimpleNamespace

import pytest
import tokenspeed_kernel.ops.moe.trtllm.mxfp4 as native_mxfp4
import tokenspeed_kernel.platform as platform_module
import tokenspeed_kernel.thirdparty.trtllm_native_moe as native_loader
import torch
from tokenspeed_kernel.ops.moe.flashinfer.trtllm_mxfp4 import (
    flashinfer_trtllm_mxfp4_moe_apply,
    flashinfer_trtllm_mxfp4_moe_weights,
)
from tokenspeed_kernel.registry import KernelRegistry


def _require_native_sm100() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("native TRT-LLM MXFP4 MoE requires CUDA")
    device = torch.device("cuda", torch.cuda.current_device())
    if torch.cuda.get_device_capability(device) != (10, 0):
        pytest.skip("native TRT-LLM MXFP4 MoE requires SM100")
    if not native_loader.has_native_mxfp4_moe():
        pytest.skip("native TRT-LLM MXFP4 MoE shared library is not built")
    return device


def test_native_mxfp4_registration_uses_precomputed_topk_and_shared_preprocessor(
    fresh_registry, monkeypatch
) -> None:
    original_has_native_mxfp4_moe = native_loader.has_native_mxfp4_moe
    original_current_platform = platform_module.current_platform
    monkeypatch.setattr(
        platform_module,
        "current_platform",
        lambda: SimpleNamespace(is_nvidia=True),
    )
    monkeypatch.setattr(native_loader, "has_native_mxfp4_moe", lambda: True)
    importlib.reload(native_mxfp4)

    spec = KernelRegistry.get().get_by_name("trtllm_mxfp4_moe_apply")
    assert spec is not None
    assert spec.solution == "trtllm"
    flashinfer_mxfp4 = importlib.import_module(
        "tokenspeed_kernel.ops.moe.flashinfer.trtllm_mxfp4"
    )
    assert (
        spec.weight_preprocessor is flashinfer_mxfp4.flashinfer_trtllm_mxfp4_moe_weights
    )
    assert spec.traits["routing_mode"] == frozenset({"precomputed_topk"})
    assert spec.traits["supports_deferred_finalize"] == frozenset({False})

    monkeypatch.setattr(
        native_loader,
        "has_native_mxfp4_moe",
        original_has_native_mxfp4_moe,
    )
    monkeypatch.setattr(
        platform_module,
        "current_platform",
        original_current_platform,
    )
    importlib.reload(native_mxfp4)


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (native_loader._TACTIC_ABI.encode("ascii"), True),
        (b"different-abi", False),
        (b"\xff", False),
    ],
)
def test_native_mxfp4_tactic_abi_match(payload: bytes, expected: bool) -> None:
    class FakeLibrary:
        pass

    def tactic_abi() -> bytes:
        return payload

    library = FakeLibrary()
    setattr(library, native_loader._TACTIC_ABI_SYMBOL, tactic_abi)
    assert native_loader._matches_offline_tactic_abi(library) is expected


def test_native_mxfp4_tactic_abi_requires_symbol() -> None:
    assert not native_loader._matches_offline_tactic_abi(object())


def test_native_mxfp4_runner_rejects_unvendored_modes() -> None:
    _require_native_sm100()
    runner_class = native_loader._load_runner_class()

    with pytest.raises(RuntimeError, match="supports only SwiGLU"):
        runner_class(1, True)
    with pytest.raises(RuntimeError, match="requires MXFP8 activations"):
        runner_class(0, False)


def test_native_mxfp4_forwards_plain_topk_and_raw_scale_bytes(monkeypatch) -> None:
    x = torch.randn(2, 8, dtype=torch.bfloat16)
    x_quant = torch.zeros(2, 8, dtype=torch.float8_e4m3fn)
    x_scale = torch.tensor([120, 121], dtype=torch.uint8)
    topk_weights = torch.tensor([[0.75, 0.25], [0.625, 0.375]])
    topk_ids = torch.tensor([[0, -1], [2, 3]], dtype=torch.int32)

    layer = SimpleNamespace(
        w13_weight=torch.zeros(2, 16, 4, dtype=torch.uint8),
        w13_weight_scale=torch.zeros(2, 16, 1, dtype=torch.float8_e4m3fn),
        w13_weight_bias=torch.zeros(2, 16, dtype=torch.float32),
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=torch.full((2,), 7.0),
        w2_weight=torch.zeros(2, 8, 4, dtype=torch.uint8),
        w2_weight_scale=torch.zeros(2, 8, 1, dtype=torch.float8_e4m3fn),
        w2_weight_bias=torch.zeros(2, 8, dtype=torch.float32),
        hidden_size=8,
        hidden_size_padded=8,
        hidden_size_original=8,
        intermediate_size=12,
        intermediate_size_per_partition=8,
        tp_size=2,
        num_experts=4,
        num_local_experts=2,
        top_k=2,
        ep_rank=1,
    )

    quantize_calls = []
    runner_calls = []

    def fake_quantize(value, swizzled, *, alignment):
        quantize_calls.append((value, swizzled, alignment))
        return x_quant, x_scale

    def fake_run_native_mxfp4_moe(**kwargs):
        runner_calls.append(kwargs)
        return kwargs["output"]

    monkeypatch.setattr(native_mxfp4, "mxfp8_quantize", fake_quantize)
    monkeypatch.setattr(
        native_mxfp4,
        "run_native_mxfp4_moe",
        fake_run_native_mxfp4_moe,
    )
    monkeypatch.setattr(
        native_mxfp4,
        "select_native_mxfp4_moe_tactic",
        lambda **kwargs: (16, 50),
    )

    result = native_mxfp4.trtllm_mxfp4_moe_apply(
        {},
        x,
        layer,
        torch.randn(2, 4),
        topk_weights=topk_weights,
        topk_ids=topk_ids,
    )

    assert len(quantize_calls) == 1
    assert quantize_calls[0][0] is x
    assert quantize_calls[0][1:] == (False, 8)
    assert result.shape == (2, 8)
    assert len(runner_calls) == 1
    kwargs = runner_calls[0]
    assert kwargs["hidden_states"] is x_quant
    assert kwargs["hidden_states_scale"].dtype == torch.uint8
    assert kwargs["hidden_states_scale"].ndim == 1
    assert kwargs["gemm1_weights_scale"].dtype == torch.uint8
    assert kwargs["gemm2_weights_scale"].dtype == torch.uint8
    assert kwargs["topk_weights"].dtype == torch.bfloat16
    assert kwargs["topk_ids"] is topk_ids
    assert kwargs["local_expert_offset"] == 2
    assert kwargs["local_num_experts"] == 2
    assert kwargs["intermediate_size"] == 8
    assert kwargs["valid_intermediate_size"] == 6
    assert kwargs["valid_hidden_size"] == 8
    assert kwargs["tactic"] == (16, 50)


def test_native_mxfp4_rejects_redundant_experts() -> None:
    with pytest.raises(ValueError, match="does not support redundant experts"):
        native_mxfp4.trtllm_mxfp4_moe_apply(
            {},
            torch.empty(0, 8, dtype=torch.bfloat16),
            SimpleNamespace(ep_num_redundant_experts=1),
            torch.empty(0, 4),
            topk_weights=torch.empty(0, 2),
            topk_ids=torch.empty(0, 2, dtype=torch.int32),
        )


def test_native_mxfp4_tactic_selection_is_limited_to_vendored_v4_tp8(
    monkeypatch,
) -> None:
    monkeypatch.setattr(native_loader, "_LIBRARY_HANDLE", object())
    monkeypatch.setattr(native_loader, "_OFFLINE_TACTICS_COMPATIBLE", True)
    monkeypatch.setattr(native_loader, "_is_valid_offline_tactic", lambda **_: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (10, 0))
    common = {
        "hidden_size": 7168,
        "intermediate_size": 384,
        "valid_hidden_size": 7168,
        "valid_intermediate_size": 384,
        "local_num_experts": 384,
        "top_k": 6,
        "device": torch.device("cuda:0"),
    }

    assert native_loader.select_native_mxfp4_moe_tactic(num_tokens=32, **common) == (
        16,
        50,
    )
    assert native_loader.select_native_mxfp4_moe_tactic(num_tokens=63, **common) == (
        16,
        50,
    )
    assert native_loader.select_native_mxfp4_moe_tactic(
        num_tokens=32,
        **{
            **common,
            "intermediate_size": 3072,
            "valid_intermediate_size": 3072,
            "local_num_experts": 48,
        },
    ) == (8, 196)
    assert native_loader.select_native_mxfp4_moe_tactic(
        num_tokens=129,
        **{
            **common,
            "intermediate_size": 3072,
            "valid_intermediate_size": 3072,
            "local_num_experts": 48,
        },
    ) == (8, 136)
    assert native_loader.select_native_mxfp4_moe_tactic(
        num_tokens=257,
        **{
            **common,
            "intermediate_size": 3072,
            "valid_intermediate_size": 3072,
            "local_num_experts": 48,
        },
    ) == (16, 54)
    assert native_loader.select_native_mxfp4_moe_tactic(
        num_tokens=32, **{**common, "local_num_experts": 12}
    ) == (-1, -1)
    assert native_loader.select_native_mxfp4_moe_tactic(num_tokens=16384, **common) == (
        -1,
        -1,
    )

    monkeypatch.setattr(native_loader, "_is_valid_offline_tactic", lambda **_: False)
    assert native_loader.select_native_mxfp4_moe_tactic(num_tokens=32, **common) == (
        -1,
        -1,
    )

    monkeypatch.setattr(native_loader, "_OFFLINE_TACTICS_COMPATIBLE", False)
    assert native_loader.select_native_mxfp4_moe_tactic(num_tokens=32, **common) == (
        -1,
        -1,
    )


def test_native_mxfp4_rejects_non_int32_topk_ids() -> None:
    layer = SimpleNamespace(top_k=2)
    x = torch.zeros(1, 8, dtype=torch.bfloat16)

    with pytest.raises(TypeError, match="topk_ids must be int32"):
        native_mxfp4._validate_topk(
            x,
            layer,
            torch.ones(1, 2),
            torch.zeros(1, 2, dtype=torch.int64),
        )


def test_native_mxfp4_rejects_invalid_explicit_tactic(monkeypatch) -> None:
    monkeypatch.setattr(native_loader, "_is_valid_offline_tactic", lambda **_: False)

    with pytest.raises(ValueError, match="invalid native TRT-LLM MoE tactic"):
        native_loader.run_native_mxfp4_moe(
            hidden_states=torch.empty(1, 8),
            hidden_states_scale=torch.empty(1),
            gemm1_weights=torch.empty(1),
            gemm1_weights_scale=torch.empty(1),
            gemm1_bias=None,
            gemm1_alpha=None,
            gemm1_beta=None,
            gemm1_clamp_limit=None,
            gemm2_weights=torch.empty(1),
            gemm2_weights_scale=torch.empty(1),
            gemm2_bias=None,
            num_experts=4,
            top_k=1,
            intermediate_size=8,
            valid_hidden_size=8,
            valid_intermediate_size=8,
            local_expert_offset=0,
            local_num_experts=1,
            topk_weights=torch.ones(1, 1, dtype=torch.bfloat16),
            topk_ids=torch.zeros(1, 1, dtype=torch.int32),
            output=torch.empty(1, 8, dtype=torch.bfloat16),
            tactic=(7, 999),
        )


def test_native_mxfp4_runner_rejects_unsafe_tactic_before_dispatch() -> None:
    device = _require_native_sm100()
    runner = native_loader._runner()
    hidden_states = torch.empty((1, 1), dtype=torch.float8_e4m3fn, device=device)
    byte = torch.empty(1, dtype=torch.uint8, device=device)
    topk_weights = torch.ones((1, 1), dtype=torch.bfloat16, device=device)
    topk_ids = torch.zeros((1, 1), dtype=torch.int32, device=device)
    output = torch.empty((1, 1), dtype=torch.bfloat16, device=device)

    def run(tactic: list[int]):
        return runner.run_moe(
            None,
            None,
            hidden_states,
            byte,
            byte,
            byte,
            None,
            None,
            None,
            None,
            byte,
            byte,
            None,
            None,
            None,
            None,
            4,
            1,
            None,
            None,
            1,
            1,
            1,
            0,
            1,
            None,
            1,
            tactic,
            topk_weights,
            topk_ids,
            output,
        )

    with pytest.raises(RuntimeError, match="exactly \\(tileN, config\\)"):
        run([8])
    with pytest.raises(RuntimeError, match="unsupported tileN: 7"):
        run([7, 0])
    with pytest.raises(RuntimeError, match="config index is out of bounds"):
        run([8, 1_000_000])


@pytest.mark.parametrize(
    ("intermediate_size", "local_num_experts"),
    [(384, 384), (3072, 48)],
)
def test_native_mxfp4_offline_tactics_are_valid_for_actual_runner(
    intermediate_size: int,
    local_num_experts: int,
) -> None:
    device = _require_native_sm100()
    hidden_size = 7168
    top_k = 6

    # Non-power-of-two values exercise the floor-to-bucket rule used by
    # TensorRT-LLM's dynamic-M autotuning cache. Exact-shape validation still
    # protects bucket tactics that do not support every M in their interval.
    for num_tokens in (3, 33, 63, 65, 127, 129, 255, 257, 511, 1025, 4097, 8191):
        tactic = native_loader.select_native_mxfp4_moe_tactic(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            valid_hidden_size=hidden_size,
            valid_intermediate_size=intermediate_size,
            local_num_experts=local_num_experts,
            top_k=top_k,
            device=device,
        )
        valid_tactics = {
            tuple(config)
            for config in native_loader._runner().get_valid_configs(
                top_k,
                hidden_size,
                intermediate_size,
                local_num_experts,
                num_tokens,
                hidden_size,
                intermediate_size,
            )
        }
        assert tactic == (-1, -1) or tactic in valid_tactics, (
            num_tokens,
            intermediate_size,
            local_num_experts,
            tactic,
        )


@pytest.mark.parametrize(
    ("num_local_experts", "tp_size", "logical_intermediate_size", "ep_rank"),
    [(4, 2, 256, 0), (2, 1, 128, 1)],
)
def test_native_mxfp4_matches_flashinfer_for_nonzero_weights(
    num_local_experts: int,
    tp_size: int,
    logical_intermediate_size: int,
    ep_rank: int,
) -> None:
    device = _require_native_sm100()

    num_tokens = 2
    hidden_size = 512
    intermediate_size = 128
    num_experts = 4
    top_k = 2

    def packed_mxfp4(shape: tuple[int, ...]) -> torch.Tensor:
        values = torch.arange(prod(shape), dtype=torch.int64, device=device).reshape(
            shape
        )
        low = values.remainder(3).add(1).to(torch.uint8)
        high = values.div(3, rounding_mode="floor").remainder(3).add(1).to(torch.uint8)
        return low | (high << 4)

    def e8m0_scales(shape: tuple[int, ...]) -> torch.Tensor:
        return (
            torch.arange(prod(shape), dtype=torch.int64, device=device)
            .remainder(2)
            .add(121)
            .reshape(shape)
            .to(torch.uint8)
        )

    weights = SimpleNamespace(
        w13_weight=torch.nn.Parameter(
            packed_mxfp4((num_local_experts, 2 * intermediate_size, hidden_size // 2)),
            requires_grad=False,
        ),
        w13_weight_scale=torch.nn.Parameter(
            e8m0_scales((num_local_experts, 2 * intermediate_size, hidden_size // 32)),
            requires_grad=False,
        ),
        w2_weight=torch.nn.Parameter(
            packed_mxfp4((num_local_experts, hidden_size, intermediate_size // 2)),
            requires_grad=False,
        ),
        w2_weight_scale=torch.nn.Parameter(
            e8m0_scales((num_local_experts, hidden_size, intermediate_size // 32)),
            requires_grad=False,
        ),
        num_experts=num_experts,
        num_local_experts=num_local_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        intermediate_size=logical_intermediate_size,
        tp_size=tp_size,
        ep_rank=ep_rank,
        w13_input_layout="concatenated",
    )
    flashinfer_trtllm_mxfp4_moe_weights({}, weights)
    # This test compares implementations rather than autotuning FlashInfer.
    weights._flashinfer_trtllm_autotuned = True

    x = (
        0.05
        + torch.arange(num_tokens * hidden_size, dtype=torch.float32, device=device)
        .reshape(num_tokens, hidden_size)
        .remainder(17)
        .mul(0.002)
    ).to(torch.bfloat16)
    topk_weights = torch.tensor(
        [[0.75, 0.25], [0.625, 0.375]], dtype=torch.float32, device=device
    )
    topk_ids = torch.tensor([[0, 2], [3, 1]], dtype=torch.int32, device=device)
    router_logits = torch.full(
        (num_tokens, num_experts), -1e20, dtype=torch.float32, device=device
    )
    router_logits.scatter_(1, topk_ids.to(torch.int64), topk_weights.log())

    reference = flashinfer_trtllm_mxfp4_moe_apply({}, x, weights, router_logits)
    actual = native_mxfp4.trtllm_mxfp4_moe_apply(
        {}, x, weights, router_logits, topk_weights, topk_ids
    )

    assert torch.isfinite(actual).all()
    assert reference.abs().max() > 0.25
    torch.testing.assert_close(actual.float(), reference.float(), rtol=2e-2, atol=2e-2)


def test_native_mxfp4_runner_supports_cuda_graph_replay() -> None:
    device = _require_native_sm100()
    flashinfer = pytest.importorskip("flashinfer")
    mxfp8_quantize = getattr(flashinfer, "mxfp8_quantize", None)
    if mxfp8_quantize is None:
        pytest.skip("installed FlashInfer does not expose mxfp8_quantize")

    num_tokens = 2
    hidden_size = 7168
    intermediate_size = 384
    local_num_experts = 1
    num_experts = 4
    top_k = 1
    x = torch.zeros((num_tokens, hidden_size), dtype=torch.bfloat16, device=device)
    x_quant, x_scale = mxfp8_quantize(x, False, alignment=hidden_size)
    configs = native_loader._runner().get_valid_configs(
        top_k,
        hidden_size,
        intermediate_size,
        local_num_experts,
        num_tokens,
        hidden_size,
        intermediate_size,
    )
    assert configs

    kwargs = {
        "hidden_states": x_quant,
        "hidden_states_scale": x_scale.view(torch.uint8).flatten(),
        "gemm1_weights": torch.zeros(
            (local_num_experts, 2 * intermediate_size, hidden_size // 2),
            dtype=torch.uint8,
            device=device,
        ),
        "gemm1_weights_scale": torch.zeros(
            (local_num_experts, 2 * intermediate_size, hidden_size // 32),
            dtype=torch.uint8,
            device=device,
        ),
        "gemm1_bias": None,
        "gemm1_alpha": None,
        "gemm1_beta": None,
        "gemm1_clamp_limit": None,
        "gemm2_weights": torch.zeros(
            (local_num_experts, hidden_size, intermediate_size // 2),
            dtype=torch.uint8,
            device=device,
        ),
        "gemm2_weights_scale": torch.zeros(
            (local_num_experts, hidden_size, intermediate_size // 32),
            dtype=torch.uint8,
            device=device,
        ),
        "gemm2_bias": None,
        "num_experts": num_experts,
        "top_k": top_k,
        "intermediate_size": intermediate_size,
        "valid_hidden_size": hidden_size,
        "valid_intermediate_size": intermediate_size,
        "local_expert_offset": 0,
        "local_num_experts": local_num_experts,
        "topk_weights": torch.ones(
            (num_tokens, top_k), dtype=torch.bfloat16, device=device
        ),
        "topk_ids": torch.zeros((num_tokens, top_k), dtype=torch.int32, device=device),
        "output": torch.empty(
            (num_tokens, hidden_size), dtype=torch.bfloat16, device=device
        ),
        "tactic": tuple(configs[0]),
    }

    side_stream = torch.cuda.Stream(device=device)
    side_stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(side_stream):
        for _ in range(3):
            native_loader.run_native_mxfp4_moe(**kwargs)
    torch.cuda.current_stream(device).wait_stream(side_stream)
    torch.cuda.synchronize(device)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = native_loader.run_native_mxfp4_moe(**kwargs)
    # Poison the buffer after capture so the assertions prove replay actually
    # executes and overwrites it instead of observing capture/warmup output.
    kwargs["output"].fill_(float("nan"))
    torch.cuda.synchronize(device)
    graph.replay()
    torch.cuda.synchronize(device)

    assert result.data_ptr() == kwargs["output"].data_ptr()
    assert torch.isfinite(result).all()
    assert torch.count_nonzero(result) == 0

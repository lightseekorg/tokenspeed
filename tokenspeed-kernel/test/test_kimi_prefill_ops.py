# Copyright (c) 2026 LightSeek Foundation

"""Portable contract tests for the Kimi prefill kernel entry points."""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from tokenspeed_kernel.ops.gemm import kimi3 as kimi3_module
from tokenspeed_kernel.ops.gemm import (
    kimi3_latent_projection,
    kimi3_mla_qkv_gate_projection,
    kimi3_qkvfab_projection,
    kimi3_router_projection,
    kimi3_shared_down_projection,
    kimi3_shared_situ_projection,
)
from tokenspeed_kernel.ops.moe.sigmoid_topk import (
    _moe_sigmoid_bias_topk as moe_sigmoid_bias_topk,
)


def test_sigmoid_bias_topk_torch_is_byte_exact() -> None:
    torch.manual_seed(2)
    logits = torch.randn(5, 32)
    bias = torch.randn(32)
    scores = logits.sigmoid()
    expected_ids = torch.topk(scores + bias, 8, dim=-1, sorted=False).indices
    expected_weights = scores.gather(1, expected_ids)
    expected_weights = expected_weights / expected_weights.sum(dim=-1, keepdim=True)

    actual_weights, actual_ids = moe_sigmoid_bias_topk(
        logits, bias, 8, solution="torch"
    )
    assert torch.equal(actual_ids, expected_ids.to(torch.int32))
    assert torch.equal(actual_weights, expected_weights)


def test_kimi3_qkvfab_projection_falls_back_for_noncanonical_shape() -> None:
    hidden_states = torch.randn(4, 64, dtype=torch.bfloat16)
    weight = torch.randn(288, 64, dtype=torch.bfloat16)
    actual = kimi3_qkvfab_projection(hidden_states, weight)
    expected = torch.nn.functional.linear(hidden_states, weight)
    torch.testing.assert_close(actual, expected)


def test_kimi3_latent_projection_falls_back_for_noncanonical_shape() -> None:
    hidden_states = torch.randn(4, 64, dtype=torch.bfloat16)
    weight = torch.randn(32, 64, dtype=torch.bfloat16)
    actual = kimi3_latent_projection(hidden_states, weight)
    expected = torch.nn.functional.linear(hidden_states, weight)
    torch.testing.assert_close(actual, expected)


def test_kimi3_router_projection_falls_back_for_noncanonical_shape() -> None:
    hidden_states = torch.randn(3, 64, dtype=torch.bfloat16)
    weight = torch.randn(8, 64, dtype=torch.bfloat16)
    actual = kimi3_router_projection(hidden_states, weight)
    expected = torch.nn.functional.linear(hidden_states.float(), weight.float())
    torch.testing.assert_close(actual, expected)


def test_kimi3_router_projection_auto_splits_on_token_count() -> None:
    """auto keeps the CUDA kernel at small M and switches to cublas above it.

    The CUDA kernel's per-thread token loop runs on CUDA cores, so its time
    grows linearly with M while the tensor-core GEMM stays flat; the dispatch
    threshold is where they cross. Solution selection is observed by mocking
    the two terminal paths -- no GPU needed.
    """
    hidden_states = torch.randn(1, kimi3_module.KIMI3_HIDDEN_SIZE).to(torch.bfloat16)
    weight = torch.randn(
        kimi3_module.KIMI3_ROUTER_SIZE, kimi3_module.KIMI3_HIDDEN_SIZE
    ).to(torch.bfloat16)
    platform = SimpleNamespace(is_cdna4=False, is_cdna5=False, is_hopper_plus=True)

    def solution_for(m: int) -> str:
        x = hidden_states.expand(m, -1).contiguous()
        with (
            mock.patch.object(kimi3_module.Platform, "get", return_value=platform),
            mock.patch.object(kimi3_module, "pdl_enabled", return_value=False),
            mock.patch.object(
                kimi3_module, "_mm_out_dtype_supported", return_value=True
            ),
            mock.patch.object(
                type(x), "is_cuda", new_callable=mock.PropertyMock
            ) as is_cuda,
            mock.patch.object(torch, "mm", return_value=torch.empty(0)) as mm,
        ):
            is_cuda.return_value = True
            try:
                kimi3_router_projection(x, weight)
            except Exception:
                pass  # the mocked/CPU terminal paths need not succeed
            return "cublas" if mm.called else "cuda-or-torch"

    assert solution_for(kimi3_module._ROUTER_CUDA_MAX_TOKENS) == "cuda-or-torch"
    assert solution_for(kimi3_module._ROUTER_CUDA_MAX_TOKENS + 1) == "cublas"


def test_kimi3_router_projection_cublas_requires_out_dtype_support() -> None:
    hidden_states = torch.randn(8, kimi3_module.KIMI3_HIDDEN_SIZE).to(torch.bfloat16)
    weight = torch.randn(
        kimi3_module.KIMI3_ROUTER_SIZE, kimi3_module.KIMI3_HIDDEN_SIZE
    ).to(torch.bfloat16)
    with mock.patch.object(kimi3_module, "_mm_out_dtype_supported", return_value=False):
        try:
            kimi3_router_projection(hidden_states, weight, solution="cublas")
            raise AssertionError("expected ValueError")
        except ValueError as exc:
            assert "out_dtype" in str(exc)


def test_kimi3_shared_projection_falls_back_for_noncanonical_shape() -> None:
    hidden_states = torch.randn(3, 64, dtype=torch.bfloat16)
    gate_up_weight = torch.randn(32, 64, dtype=torch.bfloat16)
    down_weight = torch.randn(64, 16, dtype=torch.bfloat16)
    activated = kimi3_shared_situ_projection(
        hidden_states,
        gate_up_weight,
        beta=1.5,
        linear_beta=2.5,
    )
    projected = kimi3_shared_down_projection(activated, down_weight)

    gate_up = torch.nn.functional.linear(hidden_states, gate_up_weight)
    gate, up = gate_up.float().chunk(2, dim=-1)
    gate = 1.5 * torch.tanh(gate / 1.5) * torch.sigmoid(gate)
    up = 2.5 * torch.tanh(up / 2.5)
    expected = torch.nn.functional.linear((gate * up).to(torch.bfloat16), down_weight)
    torch.testing.assert_close(projected, expected)


def test_kimi3_mla_projection_owns_schedule_selection() -> None:
    weight = torch.randn(10, 5)
    with mock.patch.object(
        kimi3_module.Platform,
        "get",
        return_value=SimpleNamespace(is_cdna4=True, is_cdna5=False),
    ):
        decode = kimi3_mla_qkv_gate_projection(torch.ones(1, 5), weight, 6)
        prefill = kimi3_mla_qkv_gate_projection(torch.ones(33, 5), weight, 6)

    assert decode.packed is not None
    assert prefill.packed is None
    expected = torch.nn.functional.linear(torch.ones(33, 5), weight)
    torch.testing.assert_close(prefill.qkv, expected[:, :6])
    torch.testing.assert_close(prefill.gate, expected[:, 6:])


def test_kimi3_mla_projection_preserves_non_cdna_prefill_schedule() -> None:
    hidden_states = torch.ones(33, 5)
    weight = torch.randn(10, 5)
    with mock.patch.object(
        kimi3_module.Platform,
        "get",
        return_value=SimpleNamespace(is_cdna4=False, is_cdna5=False),
    ):
        projection = kimi3_mla_qkv_gate_projection(hidden_states, weight, 6)

    assert projection.packed is None
    expected = torch.nn.functional.linear(hidden_states, weight)
    torch.testing.assert_close(projection.qkv, expected[:, :6])
    torch.testing.assert_close(projection.gate, expected[:, 6:])


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("solution", ["auto", "torch"])
@pytest.mark.parametrize("rows", [2, 33])
def test_kimi3_shared_down_preserves_strided_cpu_output(dtype, solution, rows) -> None:
    hidden_states = torch.randn(rows, 128, dtype=dtype)
    weight = torch.randn(16, 128, dtype=dtype)
    lane = torch.full((rows, 24), -7, dtype=dtype)
    out = lane[:, 8:]
    expected = torch.empty_strided(out.shape, out.stride(), dtype=dtype)
    torch.mm(hidden_states, weight.T, out=expected)
    with (
        mock.patch.object(
            kimi3_module.Platform, "get", return_value=SimpleNamespace(is_cdna5=True)
        ),
        mock.patch.object(
            kimi3_module, "use_gluon_wmma_dense_gfx1250", return_value=True
        ),
    ):
        actual = kimi3_shared_down_projection(
            hidden_states, weight, out=out, solution=solution
        )
    assert actual is out
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.all(lane[:, :8] == -7)


@pytest.mark.parametrize("solution", ["invalid", "triton_gemv", "gluon_largem_gfx1250"])
def test_kimi3_shared_down_rejects_unsupported_strided_selector(solution) -> None:
    hidden_states = torch.ones(2, 128, dtype=torch.bfloat16)
    weight = torch.ones(16, 128, dtype=torch.bfloat16)
    out = torch.zeros(2, 32, dtype=torch.bfloat16)[:, 16:]
    with pytest.raises(ValueError, match="solution|contiguous"):
        kimi3_shared_down_projection(hidden_states, weight, out=out, solution=solution)
    assert torch.count_nonzero(out) == 0


@pytest.mark.parametrize("layout", ["shape", "dtype", "inner_stride", "overlap"])
def test_kimi3_shared_down_validates_strided_output(layout) -> None:
    hidden_states = torch.ones(2, 128, dtype=torch.bfloat16)
    weight = torch.ones(16, 128, dtype=torch.bfloat16)
    lane = torch.zeros(2, 32, dtype=torch.bfloat16)
    if layout == "shape":
        out = lane[:, 17:]
    elif layout == "dtype":
        out = lane.float()[:, 16:]
    elif layout == "inner_stride":
        out = lane[:, ::2]
    else:
        out = lane[0, :16].expand(2, -1)
    with pytest.raises(ValueError, match="out must"):
        kimi3_shared_down_projection(hidden_states, weight, out=out, solution="auto")

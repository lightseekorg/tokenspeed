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

import math
from collections.abc import Callable

import pytest
import tokenspeed_kernel.ops.attention as attention_ops
import tokenspeed_kernel.ops.attention.flashinfer as flashinfer_ops
import torch
from tokenspeed_kernel.ops.kvcache.triton import fused_fp8_set_kv_buffer
from tokenspeed_kernel.registry import KernelRegistry
from tokenspeed_kernel.selection import SelectedKernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature


def _extend_inputs() -> dict:
    return {
        "q": torch.empty((2, 4, 64), dtype=torch.bfloat16),
        "cu_seqlens_q": torch.tensor([0, 2], dtype=torch.int32),
        "cu_seqlens_kv": torch.tensor([0, 8], dtype=torch.int32),
        "k_cache": torch.empty((1, 64, 2, 64), dtype=torch.bfloat16),
        "v_cache": torch.empty((1, 64, 2, 64), dtype=torch.bfloat16),
        "page_table": torch.zeros((1, 1), dtype=torch.int32),
        "cache_seqlens": torch.tensor([8], dtype=torch.int32),
        "max_seqlen_q": 2,
        "max_seqlen_k": 8,
    }


def _decode_inputs() -> dict:
    return {
        "q": torch.empty((1, 4, 64), dtype=torch.bfloat16),
        "k_cache": torch.empty((1, 64, 2, 64), dtype=torch.bfloat16),
        "v_cache": torch.empty((1, 64, 2, 64), dtype=torch.bfloat16),
        "page_table": torch.zeros((1, 1), dtype=torch.int32),
        "cache_seqlens": torch.tensor([8], dtype=torch.int32),
        "max_seqlen_q": 1,
        "max_seqlen_k": 8,
    }


@pytest.mark.parametrize(
    "api,inputs",
    [
        (attention_ops.mha_extend_with_kvcache, _extend_inputs),
        (attention_ops.mha_decode_with_kvcache, _decode_inputs),
    ],
    ids=["extend", "decode"],
)
def test_public_cached_mha_only_forwards_non_none_fp8_options(
    monkeypatch: pytest.MonkeyPatch,
    api: Callable,
    inputs: Callable[[], dict],
) -> None:
    calls = []

    def kernel(**kwargs):
        calls.append(kwargs)
        return kwargs["q"]

    selected = SelectedKernel("capture_cached_mha_kwargs", kernel)
    monkeypatch.setattr(
        attention_ops, "select_kernel", lambda *args, **kwargs: selected
    )

    api(**inputs())
    assert "k_scale" not in calls[-1]
    assert "v_scale" not in calls[-1]
    assert "output_dtype" not in calls[-1]

    mixed_inputs = inputs()
    mixed_inputs["k_cache"] = mixed_inputs["k_cache"].to(torch.float8_e4m3fn)
    mixed_inputs["v_cache"] = mixed_inputs["v_cache"].to(torch.float8_e4m3fn)
    api(
        **mixed_inputs,
        k_scale=0.25,
        v_scale=0.5,
        output_dtype=torch.bfloat16,
    )
    assert calls[-1]["q"].dtype is torch.bfloat16
    assert calls[-1]["k_cache"].dtype is torch.float8_e4m3fn
    assert calls[-1]["v_cache"].dtype is torch.float8_e4m3fn
    assert calls[-1]["k_scale"] == 0.25
    assert calls[-1]["v_scale"] == 0.5
    assert calls[-1]["output_dtype"] is torch.bfloat16


def _require_flashinfer_mha(name: str) -> Callable:
    kernel = getattr(flashinfer_ops, name, None)
    if kernel is None:
        pytest.skip("FlashInfer MHA wrappers are unavailable on this platform")
    return kernel


def test_flashinfer_cached_mha_registrations_split_mixed_extend() -> None:
    mixed_signature = format_signature(
        q=dense_tensor_format(torch.bfloat16),
        k_cache=dense_tensor_format(torch.float8_e4m3fn),
        v_cache=dense_tensor_format(torch.float8_e4m3fn),
    )
    extend_name = "flashinfer_fa2_mixed_mha_extend_with_kvcache"
    extend_spec = KernelRegistry.get().get_by_name(extend_name)
    assert extend_spec is not None
    assert extend_spec.supports_format_signature(mixed_signature)
    assert extend_spec.traits["support_sinks"] == frozenset({False})
    assert extend_spec.traits["support_logit_cap"] == frozenset({False, True})
    assert extend_spec.traits["return_lse"] == frozenset({False, True})

    decode_name = "flashinfer_trtllm_mha_decode_with_kvcache"
    decode_spec = KernelRegistry.get().get_by_name(decode_name)
    assert decode_spec is not None
    assert decode_spec.supports_format_signature(mixed_signature)

    same_dtype_extend = KernelRegistry.get().get_by_name(
        "flashinfer_trtllm_mha_extend_with_kvcache"
    )
    assert same_dtype_extend is not None
    assert not same_dtype_extend.supports_format_signature(mixed_signature)
    for dtype in (torch.bfloat16, torch.float16, torch.float8_e4m3fn):
        same_dtype_signature = format_signature(
            q=dense_tensor_format(dtype),
            k_cache=dense_tensor_format(dtype),
            v_cache=dense_tensor_format(dtype),
        )
        assert same_dtype_extend.supports_format_signature(same_dtype_signature)
        assert decode_spec.supports_format_signature(same_dtype_signature)
        assert not extend_spec.supports_format_signature(same_dtype_signature)


@pytest.mark.parametrize(
    "api,inputs,kernel_name",
    [
        (
            attention_ops.mha_extend_with_kvcache,
            _extend_inputs,
            "flashinfer_fa2_mixed_mha_extend_with_kvcache",
        ),
        (
            attention_ops.mha_decode_with_kvcache,
            _decode_inputs,
            "flashinfer_trtllm_mha_decode_with_kvcache",
        ),
    ],
    ids=["extend", "decode"],
)
def test_public_mixed_fp8_cache_selects_flashinfer(
    monkeypatch: pytest.MonkeyPatch,
    api: Callable,
    inputs: Callable[[], dict],
    kernel_name: str,
) -> None:
    _require_flashinfer_mha(kernel_name)
    calls = []

    def capture_call(self, *args, **kwargs):
        calls.append((self.name, kwargs))
        return kwargs["q"]

    monkeypatch.setattr(SelectedKernel, "__call__", capture_call)
    mixed_inputs = inputs()
    mixed_inputs["k_cache"] = mixed_inputs["k_cache"].to(torch.float8_e4m3fn)
    mixed_inputs["v_cache"] = mixed_inputs["v_cache"].to(torch.float8_e4m3fn)

    output = api(
        **mixed_inputs,
        k_scale=0.25,
        v_scale=0.5,
        output_dtype=torch.bfloat16,
        solution="flashinfer",
    )

    assert [name for name, _ in calls] == [kernel_name]
    assert calls[0][1]["q"].dtype is torch.bfloat16
    assert calls[0][1]["k_cache"].dtype is torch.float8_e4m3fn
    assert calls[0][1]["v_cache"].dtype is torch.float8_e4m3fn
    assert calls[0][1]["output_dtype"] is torch.bfloat16
    assert output.dtype is torch.bfloat16


def test_flashinfer_fa2_extend_uses_native_fp8_cache_and_reuses_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kernel = _require_flashinfer_mha("flashinfer_fa2_mixed_mha_extend_with_kvcache")
    instances = []

    class FakePagedPrefillWrapper:
        def __init__(self, workspace, kv_layout, backend):
            self.init_args = (workspace, kv_layout, backend)
            self.plan_calls = []
            self.run_calls = []
            instances.append(self)

        def plan(self, *args, **kwargs):
            self.plan_calls.append((args, kwargs))

        def run(self, q, paged_kv_cache, **kwargs):
            self.run_calls.append((q, paged_kv_cache, kwargs))
            output = torch.empty_like(q, dtype=torch.bfloat16)
            if kwargs["return_lse"]:
                return output, torch.empty(q.shape[:2], dtype=torch.float32)
            return output

    monkeypatch.setattr(
        flashinfer_ops,
        "BatchPrefillWithPagedKVCacheWrapper",
        FakePagedPrefillWrapper,
    )
    flashinfer_ops._fa2_mixed_extend_states.clear()

    # Serving metadata can be created under inference mode and therefore has
    # no tensor version counter. Its per-forward identity still drives replans.
    with torch.inference_mode():
        inputs = {
            "q": torch.empty((5, 4, 64), dtype=torch.bfloat16),
            "cu_seqlens_q": torch.tensor([0, 2, 5], dtype=torch.int32),
            "cu_seqlens_kv": torch.tensor([0, 8, 78], dtype=torch.int32),
            "k_cache": torch.empty((4, 64, 2, 64), dtype=torch.float8_e4m3fn),
            "v_cache": torch.empty((4, 64, 2, 64), dtype=torch.float8_e4m3fn),
            # Drop inactive -1 tails without dropping active page 0.
            "page_table": torch.tensor([[0, -1], [2, 3]], dtype=torch.int32),
            "cache_seqlens": torch.tensor([8, 70], dtype=torch.int32),
            "max_seqlen_q": 3,
            "max_seqlen_k": 70,
        }
    k_cache = inputs["k_cache"]
    v_cache = inputs["v_cache"]
    k_cache_ptr = k_cache.data_ptr()
    v_cache_ptr = v_cache.data_ptr()

    output = kernel(
        **inputs,
        k_scale=0.25,
        v_scale=0.5,
        output_dtype=torch.bfloat16,
        is_causal=True,
        window_left=16,
        logit_cap=4.0,
        return_lse=True,
    )

    assert len(instances) == 1
    wrapper = instances[0]
    assert wrapper.init_args[1:] == ("NHD", "fa2")
    assert len(wrapper.plan_calls) == 1
    plan_args, plan_kwargs = wrapper.plan_calls[0]
    assert torch.equal(plan_args[0], torch.tensor([0, 2, 5], dtype=torch.int32))
    assert torch.equal(plan_args[1], torch.tensor([0, 1, 3], dtype=torch.int32))
    assert torch.equal(plan_args[2], torch.tensor([0, 2, 3], dtype=torch.int32))
    assert torch.equal(plan_args[3], torch.tensor([8, 6], dtype=torch.int32))
    assert plan_kwargs["num_qo_heads"] == 4
    assert plan_kwargs["num_kv_heads"] == 2
    assert plan_kwargs["head_dim_qk"] == 64
    assert plan_kwargs["head_dim_vo"] == 64
    assert plan_kwargs["page_size"] == 64
    assert plan_kwargs["causal"] is True
    assert plan_kwargs["window_left"] == 16
    assert plan_kwargs["logits_soft_cap"] == pytest.approx(4.0)
    assert plan_kwargs["q_data_type"] is torch.bfloat16
    assert plan_kwargs["kv_data_type"] is torch.float8_e4m3fn
    assert plan_kwargs["o_data_type"] is torch.bfloat16

    assert len(wrapper.run_calls) == 1
    run_q, run_cache, run_kwargs = wrapper.run_calls[0]
    assert run_q is inputs["q"]
    assert run_cache[0] is k_cache
    assert run_cache[1] is v_cache
    assert run_kwargs == {
        "k_scale": 0.25,
        "v_scale": 0.5,
        "return_lse": True,
        "window_left": 16,
    }
    assert isinstance(output, tuple)
    assert output[0].dtype is torch.bfloat16
    assert output[1].dtype is torch.float32
    assert k_cache.data_ptr() == k_cache_ptr
    assert v_cache.data_ptr() == v_cache_ptr
    # A second layer shares metadata but owns different Q/K/V tensors. It must
    # reuse both the wrapper and the plan.
    second_layer = dict(inputs)
    second_layer["q"] = inputs["q"].clone()
    second_layer["k_cache"] = inputs["k_cache"].clone()
    second_layer["v_cache"] = inputs["v_cache"].clone()
    kernel(
        **second_layer,
        k_scale=0.25,
        v_scale=0.5,
        output_dtype=torch.bfloat16,
        is_causal=True,
        window_left=16,
        logit_cap=4.0,
        return_lse=True,
    )
    assert len(instances) == 1
    assert len(wrapper.plan_calls) == 1
    assert len(wrapper.run_calls) == 2

    # A new forward has a new cu-seqlens tensor and refreshes only the plan.
    next_forward = dict(second_layer)
    next_forward["cu_seqlens_q"] = inputs["cu_seqlens_q"].clone()
    kernel(
        **next_forward,
        k_scale=0.25,
        v_scale=0.5,
        output_dtype=torch.bfloat16,
        is_causal=True,
        window_left=16,
        logit_cap=4.0,
        return_lse=True,
    )
    assert len(instances) == 1
    assert len(wrapper.plan_calls) == 2
    assert len(wrapper.run_calls) == 3


def test_flashinfer_fa2_mixed_extend_fails_closed_for_sinks() -> None:
    kernel = _require_flashinfer_mha("flashinfer_fa2_mixed_mha_extend_with_kvcache")
    inputs = _extend_inputs()
    inputs["k_cache"] = inputs["k_cache"].to(torch.float8_e4m3fn)
    inputs["v_cache"] = inputs["v_cache"].to(torch.float8_e4m3fn)

    with pytest.raises(RuntimeError, match="does not support attention sinks"):
        kernel(**inputs, sinks=torch.zeros(4, dtype=torch.float32))


def test_flashinfer_fa2_mixed_extend_rejects_active_page_holes() -> None:
    with pytest.raises(ValueError, match="active page_table entries"):
        flashinfer_ops._compact_paged_kv_metadata(
            torch.tensor([[-1]], dtype=torch.int32),
            torch.tensor([1], dtype=torch.int32),
            page_size=64,
            num_cache_pages=1,
        )


def test_flashinfer_decode_applies_mixed_fp8_cache_scales_and_output_dtype(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kernel = _require_flashinfer_mha("flashinfer_trtllm_mha_decode_with_kvcache")
    captured = {}

    def decode_kernel(**kwargs):
        captured.update(kwargs)
        return torch.empty_like(kwargs["query"], dtype=kwargs["out_dtype"])

    monkeypatch.setattr(
        flashinfer_ops, "trtllm_batch_decode_with_kv_cache", decode_kernel
    )
    monkeypatch.setattr(
        flashinfer_ops, "_workspace_buffer", torch.empty(1, dtype=torch.uint8)
    )
    inputs = _decode_inputs()
    inputs["k_cache"] = inputs["k_cache"].to(torch.float8_e4m3fn)
    inputs["v_cache"] = inputs["v_cache"].to(torch.float8_e4m3fn)

    output = kernel(
        **inputs,
        k_scale=0.25,
        v_scale=0.5,
        output_dtype=torch.bfloat16,
    )

    assert captured["bmm1_scale"] == pytest.approx(0.25 / math.sqrt(64))
    assert captured["bmm2_scale"] == pytest.approx(0.5)
    assert captured["query"].dtype is torch.bfloat16
    assert captured["kv_cache"][0].dtype is torch.float8_e4m3fn
    assert captured["kv_cache"][1].dtype is torch.float8_e4m3fn
    assert captured["out_dtype"] is torch.bfloat16
    assert output.dtype is torch.bfloat16


@pytest.mark.parametrize(
    "kernel_name,inputs_factory,low_level_name",
    [
        (
            "flashinfer_trtllm_mha_extend_with_kvcache",
            _extend_inputs,
            "trtllm_batch_context_with_kv_cache",
        ),
        (
            "flashinfer_trtllm_mha_decode_with_kvcache",
            _decode_inputs,
            "trtllm_batch_decode_with_kv_cache",
        ),
    ],
    ids=["extend", "decode"],
)
@pytest.mark.parametrize(
    "dtype", [torch.bfloat16, torch.float8_e4m3fn], ids=["bf16", "fp8"]
)
def test_flashinfer_cached_mha_preserves_same_dtype_paths(
    monkeypatch: pytest.MonkeyPatch,
    kernel_name: str,
    inputs_factory: Callable[[], dict],
    low_level_name: str,
    dtype: torch.dtype,
) -> None:
    kernel = _require_flashinfer_mha(kernel_name)
    captured = {}

    def cached_kernel(**kwargs):
        captured.update(kwargs)
        return torch.empty_like(kwargs["query"], dtype=kwargs["out_dtype"])

    monkeypatch.setattr(flashinfer_ops, low_level_name, cached_kernel)
    monkeypatch.setattr(
        flashinfer_ops, "_workspace_buffer", torch.empty(1, dtype=torch.uint8)
    )
    inputs = inputs_factory()
    inputs["q"] = inputs["q"].to(dtype)
    inputs["k_cache"] = inputs["k_cache"].to(dtype)
    inputs["v_cache"] = inputs["v_cache"].to(dtype)

    output = kernel(**inputs)

    assert captured["query"].dtype is dtype
    assert captured["kv_cache"][0].dtype is dtype
    assert captured["kv_cache"][1].dtype is dtype
    assert captured["bmm1_scale"] == pytest.approx(1.0 / math.sqrt(64))
    assert captured["bmm2_scale"] == pytest.approx(1.0)
    assert captured["out_dtype"] is dtype
    assert output.dtype is dtype


def _fp8_paged_attention_reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    *,
    seq_len: int,
    query_start: int,
    k_scale: float,
    v_scale: float,
) -> torch.Tensor:
    """Reference attention over the values represented by an FP8 cache."""

    q_float = q.float()
    page_size = k_cache.shape[1]
    num_pages = (seq_len + page_size - 1) // page_size
    active_pages = page_table[:num_pages].to(torch.long)
    k_float = k_cache[active_pages].flatten(0, 1)[:seq_len].float() * k_scale
    v_float = v_cache[active_pages].flatten(0, 1)[:seq_len].float() * v_scale
    group_size = q.shape[1] // k_float.shape[1]
    k_float = k_float.repeat_interleave(group_size, dim=1)
    v_float = v_float.repeat_interleave(group_size, dim=1)

    scores = torch.einsum("qhd,khd->hqk", q_float, k_float) / math.sqrt(q.shape[-1])
    query_positions = query_start + torch.arange(q.shape[0], device=q.device)
    key_positions = torch.arange(seq_len, device=q.device)
    scores.masked_fill_(
        key_positions[None, :] > query_positions[:, None],
        -float("inf"),
    )
    probabilities = scores.softmax(dim=-1)
    return torch.einsum("hqk,khd->qhd", probabilities, v_float)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
    reason="FP8 cached MHA kernels require NVIDIA Blackwell.",
)
@pytest.mark.parametrize("mode", ["extend", "decode"])
def test_flashinfer_fp8_cached_mha_matches_dequantized_reference(mode: str) -> None:
    """Exercise real BF16-query/FP8-cache MHA with non-unit K/V scales."""

    torch.manual_seed(20260715)
    device = torch.device("cuda", torch.cuda.current_device())
    page_size = 128
    head_dim = 128
    num_q_heads = 16
    num_kv_heads = 1
    k_scale = 0.25
    v_scale = 0.5
    if mode == "extend":
        # TP4-local MiniMax-M3 shape with variable prefix lengths. Request 0
        # crosses a page boundary and maps to non-contiguous physical pages.
        query_lens = [3, 2]
        prefix_lens = [130, 5]
        physical_pages = [[3, 1], [4]]
        num_cache_pages = 6
    else:
        query_lens = [1]
        prefix_lens = [8]
        physical_pages = [[2]]
        num_cache_pages = 4
    seq_lens = [
        prefix_len + query_len
        for prefix_len, query_len in zip(prefix_lens, query_lens, strict=True)
    ]
    total_q = sum(query_lens)
    total_kv = sum(seq_lens)

    q = torch.randn(
        total_q,
        num_q_heads,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    k_source = torch.randn(
        total_kv,
        num_kv_heads,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    v_source = torch.randn_like(k_source)
    k_cache = torch.zeros(
        num_cache_pages,
        page_size,
        num_kv_heads,
        head_dim,
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    v_cache = torch.zeros_like(k_cache)
    cache_locs = []
    for page_ids, seq_len in zip(physical_pages, seq_lens, strict=True):
        logical_positions = torch.arange(seq_len, dtype=torch.int32, device=device)
        page_ids_tensor = torch.tensor(page_ids, dtype=torch.int32, device=device)
        cache_locs.append(
            page_ids_tensor[logical_positions // page_size] * page_size
            + logical_positions % page_size
        )
    cache_locs = torch.cat(cache_locs)
    fused_fp8_set_kv_buffer(
        k_source,
        v_source,
        k_cache,
        v_cache,
        cache_locs,
        k_scale=k_scale,
        v_scale=v_scale,
        page_size=page_size,
    )
    k_cache_before = k_cache.clone()
    v_cache_before = v_cache.clone()

    max_pages_per_request = max(len(page_ids) for page_ids in physical_pages)
    page_table = torch.full(
        (len(seq_lens), max_pages_per_request),
        -1,
        dtype=torch.int32,
        device=device,
    )
    for batch_idx, page_ids in enumerate(physical_pages):
        page_table[batch_idx, : len(page_ids)] = torch.tensor(
            page_ids, dtype=torch.int32, device=device
        )
    cache_seqlens = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    if mode == "extend":
        cu_seqlens_q = torch.tensor(
            [0, query_lens[0], total_q], dtype=torch.int32, device=device
        )
        cu_seqlens_kv = torch.tensor(
            [0, seq_lens[0], total_kv], dtype=torch.int32, device=device
        )
        output = attention_ops.mha_extend_with_kvcache(
            q=q,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            max_seqlen_q=max(query_lens),
            max_seqlen_k=max(seq_lens),
            is_causal=True,
            k_scale=k_scale,
            v_scale=v_scale,
            output_dtype=torch.bfloat16,
            solution="flashinfer",
        )
    else:
        output = attention_ops.mha_decode_with_kvcache(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            max_seqlen_k=max(seq_lens),
            max_seqlen_q=1,
            k_scale=k_scale,
            v_scale=v_scale,
            output_dtype=torch.bfloat16,
            solution="flashinfer",
        )

    references = []
    q_start = 0
    for batch_idx, (query_len, prefix_len, seq_len) in enumerate(
        zip(query_lens, prefix_lens, seq_lens, strict=True)
    ):
        q_end = q_start + query_len
        references.append(
            _fp8_paged_attention_reference(
                q[q_start:q_end],
                k_cache,
                v_cache,
                page_table[batch_idx],
                seq_len=seq_len,
                query_start=prefix_len,
                k_scale=k_scale,
                v_scale=v_scale,
            )
        )
        q_start = q_end
    reference = torch.cat(references)
    assert output.dtype is torch.bfloat16
    assert k_cache.dtype is torch.float8_e4m3fn
    assert v_cache.dtype is torch.float8_e4m3fn
    assert torch.equal(k_cache, k_cache_before)
    assert torch.equal(v_cache, v_cache_before)
    torch.testing.assert_close(output.float(), reference, rtol=3e-2, atol=3e-2)

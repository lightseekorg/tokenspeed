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

"""Real V4.1 fused-decode attention checks on supported CUDA targets."""

import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from test.runtime import test_deepseek_v41_cache as cache_test
from test.runtime.test_deepseek_v41_engram import _mapping
from test.runtime.test_deepseek_v41_model import (
    _Backend,
    _config,
    _ctx,
    _initialize,
    _quant,
)

import pytest
import torch
from tokenspeed_kernel.ops.attention import dsv41
from tokenspeed_kernel.ops.attention.dsv41.flash_mla import (
    is_flash_mla_fused_decode_available,
)
from tokenspeed_kernel.platform import current_platform, pdl_enabled
from tokenspeed_kernel.selection import NoKernelFoundError

from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.breakable_cuda_graph import (
    BreakableCapture,
    active_forward,
)
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.execution.output_layout import ForwardOutputLayout
from tokenspeed.runtime.layers.attention.backends.specific.deepseek_v41 import (
    DeepseekV41AttentionBackend,
)
from tokenspeed.runtime.layers.attention.configs.deepseek_v41 import DeepseekV41Config
from tokenspeed.runtime.layers.linear import LinearBase
from tokenspeed.runtime.models import deepseek_v41 as v41_model
from tokenspeed.runtime.models.deepseek_v41 import (
    DeepseekV41Attention,
    _ReferenceFp8LinearMethod,
    prepare_grouped_fp8_projection,
    v41_mxfp8_config,
)


@pytest.fixture
def pdl_for_fp8_gemm():
    previous = pdl_enabled()
    pdl_enabled(True)
    try:
        yield
    finally:
        pdl_enabled(previous)


def _supported_config(groups=8, q_rank=128):
    config = _config()
    config.num_attention_heads = 64
    config.head_dim = 512
    config.qk_rope_head_dim = 64
    config.q_lora_rank = q_rank
    config.o_groups = groups
    config.o_lora_rank = 128
    return config


def _attention(*, seed, device="cuda", groups=8, q_rank=128):
    torch.manual_seed(seed)
    config = _supported_config(groups, q_rank)
    with torch.device(device):
        attn = DeepseekV41Attention(
            config,
            _mapping(0, 1, 1),
            1,  # Not a KV/index owner: keep this test focused on attention math.
            config.num_hidden_layers,
            v41_mxfp8_config(_quant()),
            "layers.1.attn",
            aux_stream=None,
        )
    _initialize(attn)
    return attn


def _random_codes(shape, device):
    return (torch.randn(shape, device=device) * 0.25).to(torch.float8_e4m3fn)


def _checkpoint_scales(rows, cols, device, exponent=123):
    return torch.full(
        (rows // 32, cols // 32), exponent, dtype=torch.uint8, device=device
    )


def _assert_bitwise(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    torch.testing.assert_close(
        actual.clone(memory_format=torch.contiguous_format).view(torch.uint8),
        expected.clone(memory_format=torch.contiguous_format).view(torch.uint8),
        rtol=0,
        atol=0,
    )


def _load_q_projection(attn, *, seed, scale_exponent):
    torch.manual_seed(seed)
    layer = attn.wq_b
    weight = _random_codes(layer.weight.shape, layer.weight.device)
    scales = _checkpoint_scales(*weight.shape, weight.device, exponent=scale_exponent)
    loader = getattr(layer.weight, "weight_loader", None)
    if loader is None:
        layer.weight.data.copy_(weight)
    else:
        loader(layer.weight, weight)
    layer.weight_scale_inv.weight_loader(layer.weight_scale_inv, scales)
    layer.quant_method.process_weights_after_loading(layer)


def _load_wo_a(attn, *, seed, scale_exponent):
    torch.manual_seed(seed)
    layer = attn.wo_a
    weight = _random_codes(layer.weight.shape, layer.weight.device)
    scales = _checkpoint_scales(*weight.shape, weight.device, exponent=scale_exponent)
    expanded = scales.view(torch.float8_e8m0fnu).view(torch.uint8)
    expanded = expanded.repeat_interleave(32, dim=0)
    attn.wo_a_scale_inv.copy_(expanded)
    prepared = prepare_grouped_fp8_projection(weight, expanded, attn.n_local_groups)
    assert prepared is not None
    if layer.weight.dtype == weight.dtype:
        layer.weight.data.copy_(weight)
    else:
        layer.weight.data = weight
    assert layer.weight.dtype == torch.float8_e4m3fn
    if attn.wo_a_scales is None:
        attn.wo_a_scales = prepared
    else:
        attn.wo_a_scales.copy_(prepared)
    attn.wo_a_fp8 = True


def _load_decode_projections(attn, *, seed):
    _load_q_projection(attn, seed=seed, scale_exponent=123)
    _load_wo_a(attn, seed=seed + 1, scale_exponent=123)
    for module in attn.modules():
        if (
            isinstance(module, LinearBase)
            and isinstance(module.quant_method, _ReferenceFp8LinearMethod)
            and module is not attn.wq_b
        ):
            module.quant_method.process_weights_after_loading(module)
            module.quant_method.prepare_weight_scales(module)
    attn.prepare_output_projections()
    assert attn.wo_b_fp8 and attn.wo_b_scales is not None


def _real_backend(*, max_bs=1):
    recipe = cache_test._recipe("cuda")
    config = recipe.attn_config
    spec = replace(config.component(DeepseekV41Config), num_attention_heads=64)
    config = replace(config, components=(spec,))
    assert spec.num_attention_heads == 64 and spec.head_dim == 512
    assert spec.kv_owners[1] < 0 and spec.index_sources[1] < 0
    default_packing = (
        cache_test._config("cuda")
        .component(DeepseekV41Config)
        .row_layout()
        .group_packing
    )
    assert spec.row_layout().group_packing == default_packing
    recipe.attn_config = config
    pool = cache_test._pool(recipe, "cuda")
    backend = DeepseekV41AttentionBackend(config, spec)
    backend.set_cache_pool(pool)
    backend.init_cuda_graph_state(
        max_bs, max_tokens_per_req=1, overlap_schedule_depth=1
    )
    tables = cache_test._tables("cuda")
    return backend, tables


def _decode(attn, backend, tables, hidden):
    backend.refresh_decode_metadata(
        1,
        1,
        torch.tensor([0], device="cuda", dtype=torch.int64),
        torch.tensor([1], device="cuda", dtype=torch.int64),
        forward_mode=ForwardMode.DECODE,
        block_tables=tables,
        num_extends=0,
        for_graph_replay=False,
    )
    positions = backend.query_metadata(ForwardMode.DECODE).positions
    return attn(
        positions,
        hidden,
        _ctx(backend, 1, ForwardMode.DECODE),
        reduce_output=True,
    )


def _require_fused_cuda():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if not is_flash_mla_fused_decode_available():
        pytest.skip("requires the native V4.1 fused-decode kernel")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.inference_mode()
def test_kvp_keeps_canonical_weights_without_unused_fused_copies(pdl_for_fp8_gemm):
    _require_fused_cuda()
    torch.manual_seed(4400)
    config = _supported_config()
    mapping = Mapping(
        rank=0,
        world_size=2,
        attn_tp_size=2,
        attn_qcp_size=2,
        attn_kvp_size=2,
        dense_tp_size=1,
        moe_tp_size=1,
        moe_ep_size=2,
    )
    with torch.device("cuda"):
        attn = DeepseekV41Attention(
            config,
            mapping,
            1,
            config.num_hidden_layers,
            v41_mxfp8_config(_quant()),
            "layers.1.attn",
            aux_stream=None,
        )
    _initialize(attn)
    _load_decode_projections(attn, seed=4401)
    assert mapping.attn.head_tp_size == 1 and mapping.attn.kvp_size == 2
    assert attn.wo_a_fp8 and attn.wq_b.weight.dtype == torch.float8_e4m3fn
    attn.prepare_fused_decode()
    assert all(
        t is None
        for t in (
            attn.fused_q_weight,
            attn.fused_q_scales,
            attn.fused_wo_a_weight,
            attn.fused_wo_a_scales,
        )
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("seed", [4100, 4110, 4120])
@torch.inference_mode()
def test_fused_decode_matches_canonical_and_keeps_prefill_weights(
    pdl_for_fp8_gemm, monkeypatch, seed
):
    _require_fused_cuda()
    attn = _attention(seed=seed)
    if attn.wq_b.weight.dtype != torch.float8_e4m3fn:
        pytest.skip("fused decode requires canonical FP8 Q weights")
    _load_decode_projections(attn, seed=seed + 1)
    canonical = (
        attn.wq_b.weight.clone(),
        attn.wq_b.weight_scale_inv.clone(),
        attn.wo_a.weight.clone(),
        attn.wo_a_scales.clone(),
    )
    hidden = torch.randn(1, 128, device="cuda", dtype=torch.bfloat16)

    # Observe the actual wo_b operands while running the real projection. Each
    # path must match its own FP64 dot product, independently of quantization
    # differences earlier in attention. No GEMM is replaced by an oracle.
    projection_inputs = []
    fused_attention_outputs = []
    project = v41_model.fp8_projection
    attend = dsv41.selected_attention_fp8

    def observe_attention(*args, **kwargs):
        result = attend(*args, **kwargs)
        fused_attention_outputs.append(result)
        return result

    def observe_projection(x, scales, weight, weight_scales, out):
        result = project(x, scales, weight, weight_scales, out)
        projection_inputs.append((x, scales))
        return result

    monkeypatch.setattr(v41_model, "fp8_projection", observe_projection)
    monkeypatch.setattr(dsv41, "selected_attention_fp8", observe_attention)

    def projection_reference(operands):
        codes, scales = operands
        exponents = scales.clone(memory_format=torch.contiguous_format).view(
            torch.uint8
        )
        x = (
            codes.double().unflatten(-1, (-1, 32))
            * torch.exp2(exponents.double() - 127)[..., None]
        ).flatten(-2)
        weight = (
            attn.wo_b.weight.double().unflatten(-1, (-1, 32))
            * torch.exp2(attn.wo_b.weight_scale_inv.double() - 127)[..., None]
        ).flatten(-2)
        return x @ weight.T

    plain_backend, plain_tables = _real_backend()
    plain_decode = _decode(attn, plain_backend, plain_tables, hidden)
    plain_reference = projection_reference(projection_inputs[-1])
    torch.testing.assert_close(
        plain_decode.float(),
        plain_reference.to(torch.bfloat16).float(),
        rtol=torch.finfo(torch.bfloat16).eps,
        atol=2e-6,
    )

    plain_prefill_backend, plain_prefill_tables = _real_backend()
    plain_meta = cache_test._extend(
        plain_prefill_backend, plain_prefill_tables, [1], [0], [0], [1]
    )
    plain_prefill = attn(
        plain_meta.positions,
        hidden,
        _ctx(plain_prefill_backend, 1, ForwardMode.EXTEND),
        reduce_output=True,
    )

    attn.prepare_fused_decode()
    assert attn.n_local_heads == 64 and attn.head_dim == 512
    assert attn.n_local_groups == 8
    assert attn.rotary_emb.dim == 64
    assert all(
        tensor is not None
        for tensor in (
            attn.fused_q_weight,
            attn.fused_q_scales,
            attn.fused_wo_a_weight,
            attn.fused_wo_a_scales,
        )
    )
    fused_backend, fused_tables = _real_backend()
    fused_decode = _decode(attn, fused_backend, fused_tables, hidden)
    assert (
        fused_attention_outputs
    ), "The numeric comparison must execute fused attention"
    fused_reference = projection_reference(projection_inputs[-1])
    torch.testing.assert_close(
        fused_decode.float(),
        fused_reference.to(torch.bfloat16).float(),
        rtol=torch.finfo(torch.bfloat16).eps,
        atol=2e-6,
    )
    # Dense projection can cancel individual components. Use the same
    # relative-L2 envelope as the quantized projection test rather than an
    # elementwise relative error near zero. The exact same-input checks above
    # independently guard the scale/group layout and final projection wiring.
    delta = (fused_decode.float() - plain_decode.float()).norm(dim=-1)
    assert torch.all(delta / plain_decode.float().norm(dim=-1) < 0.055)
    rounding = (
        2
        * torch.finfo(torch.bfloat16).eps
        * (fused_reference.abs() + plain_reference.abs())
        + 4e-6
    )
    assert torch.all(
        (
            (fused_decode.double() - plain_decode.double())
            - (fused_reference - plain_reference)
        ).abs()
        <= rounding
    )

    fused_prefill_backend, fused_prefill_tables = _real_backend()
    fused_meta = cache_test._extend(
        fused_prefill_backend, fused_prefill_tables, [1], [0], [0], [1]
    )
    fused_prefill = attn(
        fused_meta.positions,
        hidden,
        _ctx(fused_prefill_backend, 1, ForwardMode.EXTEND),
        reduce_output=True,
    )
    # Prefill now runs the fused FP8 attention, so it is no longer bitwise equal
    # to the plain path. Use the same relative-L2 envelope as the decode
    # comparison above.
    prefill_delta = (fused_prefill.float() - plain_prefill.float()).norm(dim=-1)
    assert torch.all(prefill_delta / plain_prefill.float().norm(dim=-1) < 0.055)
    for actual, expected in zip(
        (
            attn.wq_b.weight,
            attn.wq_b.weight_scale_inv,
            attn.wo_a.weight,
            attn.wo_a_scales,
        ),
        canonical,
    ):
        _assert_bitwise(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.inference_mode()
def test_fused_decode_graph_sees_in_place_weight_refresh(pdl_for_fp8_gemm):
    _require_fused_cuda()
    attn = _attention(seed=4200)
    if attn.wq_b.weight.dtype != torch.float8_e4m3fn:
        pytest.skip("fused decode requires canonical FP8 Q weights")
    _load_decode_projections(attn, seed=4201)
    attn.prepare_fused_decode()
    prepared = (
        attn.fused_q_weight,
        attn.fused_q_scales,
        attn.fused_wo_a_weight,
        attn.fused_wo_a_scales,
    )
    addresses = tuple(tensor.data_ptr() for tensor in prepared)
    hidden = torch.randn(1, 128, device="cuda", dtype=torch.bfloat16)
    backend, tables = _real_backend()
    backend.init_forward_metadata_capture_cuda_graph(
        1,
        torch.tensor([0], device="cuda", dtype=torch.int64),
        torch.tensor([1], device="cuda", dtype=torch.int32),
        ForwardMode.DECODE,
        block_tables=tables,
    )
    positions = backend.query_metadata(ForwardMode.DECODE).positions

    def call():
        return attn(
            positions,
            hidden,
            _ctx(backend, 1, ForwardMode.DECODE),
            reduce_output=True,
        )

    call()  # Warm up kernels and RoPE tables.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = call()
    graph.replay()
    old = captured.clone()

    _load_q_projection(attn, seed=4202, scale_exponent=124)
    _load_wo_a(attn, seed=4203, scale_exponent=121)
    attn.prepare_fused_decode()
    assert (
        tuple(
            tensor.data_ptr()
            for tensor in (
                attn.fused_q_weight,
                attn.fused_q_scales,
                attn.fused_wo_a_weight,
                attn.fused_wo_a_scales,
            )
        )
        == addresses
    )
    fresh = _attention(seed=4200)
    _load_decode_projections(fresh, seed=4201)
    _load_q_projection(fresh, seed=4202, scale_exponent=124)
    _load_wo_a(fresh, seed=4203, scale_exponent=121)
    fresh.prepare_fused_decode()
    for updated, independently_prepared in zip(
        (
            attn.fused_q_weight,
            attn.fused_q_scales,
            attn.fused_wo_a_weight,
            attn.fused_wo_a_scales,
        ),
        (
            fresh.fused_q_weight,
            fresh.fused_q_scales,
            fresh.fused_wo_a_weight,
            fresh.fused_wo_a_scales,
        ),
    ):
        _assert_bitwise(updated, independently_prepared)
    backend.refresh_decode_metadata(
        1,
        1,
        torch.tensor([0], device="cuda", dtype=torch.int64),
        torch.tensor([1], device="cuda", dtype=torch.int64),
        forward_mode=ForwardMode.DECODE,
        block_tables=tables,
        num_extends=0,
        for_graph_replay=True,
    )
    graph.replay()
    new = captured.clone()

    eager_backend, eager_tables = _real_backend()
    eager = _decode(fresh, eager_backend, eager_tables, hidden)
    torch.testing.assert_close(new, eager, rtol=0, atol=0)
    assert not torch.allclose(old, new, rtol=0.01, atol=0.002)
    graph.reset()


def test_cpu_or_unsupported_geometry_keeps_the_plain_attention_path():
    config = _config()
    attn = DeepseekV41Attention(
        config,
        _mapping(0, 1, 1),
        1,
        config.num_hidden_layers,
        None,
        "layers.1.attn",
        aux_stream=None,
    )
    _initialize(attn)
    positions = torch.tensor([0])
    backend = _Backend(positions, torch.zeros_like(positions))
    hidden = torch.randn(1, config.hidden_size, dtype=torch.bfloat16)
    before = attn(positions, hidden, _ctx(backend, 1, ForwardMode.DECODE), True)
    attn.prepare_fused_decode()
    after = attn(positions, hidden, _ctx(backend, 1, ForwardMode.DECODE), True)
    assert attn.fused_q_weight is None
    torch.testing.assert_close(after, before, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.inference_mode()
@pytest.mark.parametrize("groups,q_rank", [(4, 128), (8, 160)])
def test_unsupported_fused_geometry_falls_back_to_plain_decode(
    pdl_for_fp8_gemm, groups, q_rank
):
    _require_fused_cuda()
    attn = _attention(seed=4300, groups=groups, q_rank=q_rank)
    if attn.wq_b.weight.dtype != torch.float8_e4m3fn:
        pytest.skip("fused decode requires canonical FP8 Q weights")
    _load_decode_projections(attn, seed=4301)
    canonical = (
        attn.wq_b.weight.clone(),
        attn.wq_b.weight_scale_inv.clone(),
        attn.wo_a.weight.clone(),
        attn.wo_a_scales.clone(),
    )
    attn.prepare_fused_decode()
    assert attn.fused_q_weight is None
    for actual, expected in zip(
        (
            attn.wq_b.weight,
            attn.wq_b.weight_scale_inv,
            attn.wo_a.weight,
            attn.wo_a_scales,
        ),
        canonical,
    ):
        _assert_bitwise(actual, expected)

    hidden = torch.randn(1, 128, device="cuda", dtype=torch.bfloat16)
    plain_backend, plain_tables = _real_backend()
    try:
        plain = _decode(attn, plain_backend, plain_tables, hidden)
    except NoKernelFoundError:
        if q_rank == 160:
            pytest.skip("canonical Q GEMM has no kernel for K=160 on this target")
        raise
    fallback_backend, fallback_tables = _real_backend()
    fallback = _decode(attn, fallback_backend, fallback_tables, hidden)
    torch.testing.assert_close(fallback, plain, rtol=0, atol=0)


def test_tp_attention_is_not_prepared_for_fused_decode():
    config = _supported_config()
    with torch.device("meta"):
        attn = DeepseekV41Attention(
            config,
            _mapping(0, 2, 2),
            1,
            config.num_hidden_layers,
            v41_mxfp8_config(_quant()),
            "layers.1.attn",
            aux_stream=None,
        )
    attn.wo_a_fp8 = True
    attn.prepare_fused_decode()
    assert attn.n_local_heads == 32
    assert attn.fused_q_weight is None


def test_zeroed_fused_output_has_the_kernels_layout():
    rows, groups = 5, 8
    codes, scales = dsv41.allocate_fp8_output(
        torch.device("cpu"), groups, rows, zeroed=True
    )
    assert codes.dtype == torch.float8_e4m3fn
    assert codes.shape == (rows, groups, 64 * 512 // groups)
    assert codes.is_contiguous()
    assert scales.dtype == torch.int32
    assert scales.shape == (rows, groups, 64 * 512 // groups // 128)
    # Rows are the fastest dimension, over storage padded to four rows.
    assert scales.stride(0) == 1
    assert scales.stride(1) == (rows + 3) // 4 * 4 * scales.shape[2]
    assert not codes.view(torch.uint8).any()
    assert not scales.any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fused_prefill_support_follows_the_query_group():
    backend, _ = _real_backend()
    assert backend.supports_fused_prefill
    backend.spec = replace(backend.spec, attention_query_group=(0, 1))
    assert backend.supports_fused_attention
    assert not backend.supports_fused_prefill


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("guard", ["query_group", "query_shard"])
@pytest.mark.parametrize("mode", [ForwardMode.EXTEND, ForwardMode.MIXED])
def test_fused_prefill_rejects_layouts_it_cannot_serve(guard, mode):
    backend, tables = _real_backend()
    meta = cache_test._extend(backend, tables, [3], [0], [0], [3])
    if guard == "query_group":
        backend.spec = replace(backend.spec, attention_query_group=(0, 1))
        key_rows, message = None, "operator query context parallelism"
    else:
        # The guard reads only whether key_rows is present.
        key_rows, message = object(), "model query shard"
    q = torch.zeros(3, 64, 512, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match=message):
        backend.forward_v41(
            q,
            torch.zeros(3, 512, device="cuda", dtype=torch.bfloat16),
            layer_id=1,
            positions=meta.positions,
            request_indices=meta.request_indices,
            forward_mode=mode,
            index_q=None,
            index_weights=None,
            attn_sink=torch.zeros(64, device="cuda", dtype=torch.float32),
            softmax_scale=512**-0.5,
            index_process_group=None,
            swa_rope_cache=torch.zeros(1, 64, device="cuda", dtype=torch.float32),
            swa_norm_weight=None,
            swa_norm_eps=0.0,
            fused_decode_groups=8,
            key_rows=key_rows,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("canonical", [False, True])
def test_fused_mixed_swa_matches_separate_extend_and_decode(canonical):
    _require_fused_cuda()
    backend, tables = _real_backend(max_bs=2)
    angle = torch.arange(512, device="cuda")[:, None] * torch.linspace(
        0.001, 0.03, 32, device="cuda"
    )
    rope = torch.cat((angle.cos(), angle.sin()), dim=-1)
    cache_test._check_mixed_swa_matches_separate(
        backend, tables, layer=1, rope=rope, canonical=canonical
    )


def _mixed_context(backend, tables, prefill_tokens):
    lengths = torch.tensor([prefill_tokens], dtype=torch.int32)
    backend.init_forward_metadata(
        2,
        1,
        torch.tensor([0, 1], device="cuda"),
        torch.tensor([prefill_tokens, 1], dtype=torch.int32, device="cuda"),
        ForwardMode.MIXED,
        block_tables=tables,
        extend_seq_lens=lengths.cuda(),
        extend_seq_lens_cpu=lengths,
        extend_prefix_lens=torch.tensor([0], dtype=torch.int32, device="cuda"),
        extend_prefix_lens_cpu=torch.tensor([0], dtype=torch.int32),
        extend_replay_lens_cpu=torch.tensor([0], dtype=torch.int32),
        extend_prompt_lens_cpu=lengths,
        extend_with_prefix=False,
        query_shard=None,
    )
    meta = backend.query_metadata(ForwardMode.MIXED)
    return meta, ForwardContext(
        attn_backend=backend,
        token_to_kv_pool=None,
        bs=2,
        num_extends=1,
        input_num_tokens=prefill_tokens + 1,
        forward_mode=ForwardMode.MIXED,
        output_layout=ForwardOutputLayout(1, 1, 1, 1),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.inference_mode()
def test_fused_mixed_model_matches_separate_forwards(pdl_for_fp8_gemm):
    _require_fused_cuda()
    attn = _attention(seed=4170)
    if attn.wq_b.weight.dtype != torch.float8_e4m3fn:
        pytest.skip("fused attention requires canonical FP8 Q weights")
    _load_decode_projections(attn, seed=4171)
    attn.prepare_fused_decode()
    backend, tables = _real_backend(max_bs=2)
    hidden = torch.randn(4, 128, device="cuda", dtype=torch.bfloat16)
    before = backend.cache_pool.arena.buffer.clone()
    meta, ctx = _mixed_context(backend, tables, 3)
    mixed = attn(meta.positions, hidden, ctx, reduce_output=True)

    backend.cache_pool.arena.buffer.copy_(before)
    meta = cache_test._extend(backend, tables, [3], [0], [0], [3])
    extend = attn(
        meta.positions,
        hidden[:3],
        _ctx(backend, 3, ForwardMode.EXTEND),
        reduce_output=True,
    )
    backend.refresh_decode_metadata(
        1,
        1,
        torch.tensor([1], device="cuda"),
        torch.tensor([1], dtype=torch.int32, device="cuda"),
        forward_mode=ForwardMode.DECODE,
        block_tables={gid: table[1:2] for gid, table in tables.items()},
        for_graph_replay=False,
    )
    meta = backend.query_metadata(ForwardMode.DECODE)
    decode = attn(
        meta.positions,
        hidden[3:],
        _ctx(backend, 1, ForwardMode.DECODE),
        reduce_output=True,
    )
    separate = torch.cat((extend, decode))
    # Different GEMM row counts can change BF16 accumulation rounding.
    error = (mixed.float() - separate.float()).norm(dim=-1)
    assert torch.all(error < 0.01 * separate.float().norm(dim=-1))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("mode", [ForwardMode.EXTEND, ForwardMode.MIXED])
def test_fused_attention_clears_skipped_rows_in_a_reused_destination(mode):
    _require_fused_cuda()
    backend, tables = _real_backend(max_bs=2)
    if mode.is_mixed():
        meta, _ = _mixed_context(backend, tables, 2)
    else:
        meta = cache_test._extend(backend, tables, [2], [0], [0], [2])
    # The prefill window skips the inserted row, including in a mixed batch.
    hole = meta.positions.new_tensor([-1])
    positions = torch.cat((meta.positions[:1], hole, meta.positions[1:]))
    requests = torch.cat((meta.request_indices[:1], hole, meta.request_indices[1:]))
    rows = positions.numel()
    q = torch.randn(rows, 64, 512, dtype=torch.bfloat16, device="cuda")
    swa = torch.randn(rows, 512, dtype=torch.bfloat16, device="cuda")
    angle = torch.randn(2, 32, device="cuda")
    rope = torch.cat((angle.cos(), angle.sin()), dim=-1)
    sink = torch.zeros(64, device="cuda")
    parent = dsv41.allocate_fp8_output(q.device, 8, rows + 2, zeroed=False)
    parent[0].view(torch.uint8).fill_(0x38)
    parent[1].fill_(123)
    out = parent[0][1:-1], parent[1][1:-1]

    def attend(destination):
        return backend.forward_v41(
            q,
            swa,
            layer_id=1,
            positions=positions,
            request_indices=requests,
            forward_mode=mode,
            index_q=None,
            index_weights=None,
            attn_sink=sink,
            softmax_scale=512**-0.5,
            index_process_group=None,
            swa_rope_cache=rope,
            swa_norm_weight=None,
            swa_norm_eps=0.0,
            fused_decode_groups=8,
            key_rows=None,
            fp8_out=destination,
        )

    before = backend.cache_pool.arena.buffer.clone()
    expected = attend(None)
    expected_cache = backend.cache_pool.arena.buffer.clone()
    backend.cache_pool.arena.buffer.copy_(before)
    assert attend(out) is out
    torch.testing.assert_close(out[0].view(torch.uint8), expected[0].view(torch.uint8))
    torch.testing.assert_close(out[1], expected[1])
    assert not torch.count_nonzero(out[0][1].view(torch.uint8))
    assert not torch.count_nonzero(out[1][1])
    assert torch.equal(backend.cache_pool.arena.buffer, expected_cache)
    for row in (0, -1):
        assert torch.all(parent[0][row].view(torch.uint8) == 0x38)
        assert torch.all(parent[1][row] == 123)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.inference_mode()
@pytest.mark.parametrize("fp8_wo_b", [True, False])
def test_prefill_graph_projects_fused_outputs_across_live_batches(
    pdl_for_fp8_gemm, monkeypatch, fp8_wo_b
):
    _require_fused_cuda()
    attn = _attention(seed=4172)
    _load_decode_projections(attn, seed=4173)
    attn.prepare_fused_decode()
    attn.wo_b_fp8 = fp8_wo_b
    backend, tables = _real_backend(max_bs=2)
    reference, ref_tables = _real_backend(max_bs=2)
    bucket = 19  # Packed scales have physical room for 20 rows.
    hidden = torch.randn(bucket, 128, device="cuda", dtype=torch.bfloat16)
    meta = cache_test._extend(backend, tables, [bucket], [0], [0], [bucket])
    ctx = _ctx(backend, bucket, ForwardMode.EXTEND)

    # Observe the real wo_a operands on the capture stream. They must be
    # refreshed by the eager break before every subsequent replay.
    captured_operands = []
    eager_operands = []
    if fp8_wo_b:
        project = v41_model.grouped_fp8_projection_quantize_fp8
    else:
        project = v41_model.grouped_fp8_projection

    def observe(codes, scales, *args):
        if torch.cuda.is_current_stream_capturing():
            captured_operands.append((codes, scales))
        else:
            eager_operands[:] = [codes, scales]
        return project(codes, scales, *args)

    monkeypatch.setattr(v41_model, project.__name__, observe)
    with active_forward(ctx):
        attn(meta.positions, hidden, ctx, reduce_output=True)
        graph = BreakableCapture()
        with graph:
            result = attn(meta.positions, hidden, ctx, reduce_output=True)
    ((codes, scales),) = captured_operands
    assert codes.is_contiguous() and scales.stride() == (1, 32 * 20, 20)
    physical_scales = scales.as_strided((20, 8, 32), scales.stride())
    reference.cache_pool.arena.buffer.copy_(backend.cache_pool.arena.buffer)

    for mode, live in (
        (ForwardMode.EXTEND, 17),
        (ForwardMode.MIXED, 5),
        (ForwardMode.IDLE, 0),
        (ForwardMode.EXTEND, 1),
        (ForwardMode.MIXED, bucket),
    ):
        hidden.normal_()
        hidden[live:].fill_(float("nan"))
        if mode.is_idle():
            empty = meta.positions[:0]
            backend.refresh_decode_metadata(
                0,
                0,
                empty,
                empty,
                forward_mode=mode,
                block_tables={gid: table[:0] for gid, table in tables.items()},
                for_graph_replay=False,
            )
            ctx = _ctx(backend, 0, mode)
            ctx.bs = ctx.num_extends = 0
            ctx.output_layout = ForwardOutputLayout(0, 0, 0, 1)
            ctx.global_num_tokens = [0, 5]
            ctx.all_decode_or_idle = False
            with active_forward(ctx):
                graph.replay(valid_rows=0)
            assert not result.count_nonzero()
            assert not codes.view(torch.uint8).count_nonzero()
            assert not physical_scales.count_nonzero()
            assert torch.equal(
                backend.cache_pool.arena.buffer, reference.cache_pool.arena.buffer
            )
            continue
        if mode.is_mixed():
            _, ctx = _mixed_context(backend, tables, live - 1)
            ref_meta, ref_ctx = _mixed_context(reference, ref_tables, live - 1)
        else:
            cache_test._extend(backend, tables, [live], [0], [0], [live])
            ref_meta = cache_test._extend(
                reference, ref_tables, [live], [0], [0], [live]
            )
            ctx = _ctx(backend, live, mode)
            ref_ctx = _ctx(reference, live, mode)
        expected = attn(ref_meta.positions, hidden[:live], ref_ctx, reduce_output=True)
        expected_codes, expected_scales = eager_operands
        with active_forward(ctx):
            graph.replay(valid_rows=live)
        torch.testing.assert_close(
            codes[:live].view(torch.uint8),
            expected_codes.view(torch.uint8),
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(scales[:live], expected_scales, atol=0, rtol=0)
        # The graph projects the padded bucket. A different M can change the
        # BF16 rounding in the final GEMM even with byte-identical input rows.
        error = (result[:live].float() - expected.float()).norm(dim=-1)
        assert torch.all(error <= 0.01 * expected.float().norm(dim=-1))
        assert not torch.count_nonzero(result[live:])
        assert not torch.count_nonzero(codes[live:].view(torch.uint8))
        assert not torch.count_nonzero(physical_scales[live:])
        torch.testing.assert_close(
            backend.cache_pool.arena.buffer,
            reference.cache_pool.arena.buffer,
            atol=0,
            rtol=0,
        )


_QUERY_WARMUP_SCRIPT = """
import sys
from pathlib import Path

import torch
from test.runtime.test_deepseek_v41_fused_decode import (
    _initialize, _load_decode_projections, _mapping, _quant, _supported_config,
)
from tokenspeed.runtime.models.deepseek_v41 import (
    DeepseekV41Attention, _warmup_query_projections, v41_mxfp8_config,
)
from tokenspeed_kernel.ops.gemm import fp8_projection
from tokenspeed_kernel.platform import pdl_enabled

pdl_enabled(True)
cache, groups = Path(sys.argv[1]), int(sys.argv[2])
config = _supported_config(groups=groups)
attentions = []
with torch.device("cuda"), torch.inference_mode():
    # Window-only layers precede the first indexer in real checkpoints.
    for layer_id in (1, 2):
        attn = DeepseekV41Attention(
            config, _mapping(0, 1, 1), layer_id, config.num_hidden_layers,
            v41_mxfp8_config(_quant()), f"layers.{layer_id}.attn", aux_stream=None,
        )
        _initialize(attn)
        _load_decode_projections(attn, seed=layer_id)
        attn.prepare_fused_decode()
        assert (attn.fused_q_weight is not None) == (groups == 8)
        attentions.append(attn)
    _warmup_query_projections(attentions, 300)
    compiled = set(cache.rglob("*"))
    assert compiled, "the warmup compiled nothing into the JIT cache"
    for attn in attentions:
        layers = [attn.wq_a_wkv, attn.wq_b]
        if attn.indexer is not None:
            layers.append(attn.indexer.wq_b)
        for tokens in (1, 7, 16, 17, 100, 255, 256, 300):
            for layer in layers:
                width = layer.weight.shape[1]
                codes = torch.zeros(tokens, width, dtype=torch.float8_e4m3fn)
                scales = torch.zeros(width // 128, (tokens + 3) // 4 * 4,
                                     dtype=torch.int32).mT[:tokens]
                layer.quant_method.apply(layer, codes, None, scales, torch.bfloat16)
            if attn.fused_q_weight is not None:
                width = attn.fused_q_weight.shape[1]
                codes = torch.zeros(tokens, width, dtype=torch.float8_e4m3fn)
                scales = torch.zeros(width // 128, (tokens + 3) // 4 * 4,
                                     dtype=torch.int32).mT[:tokens]
                fp8_projection(codes, scales, attn.fused_q_weight,
                               attn.fused_q_projection_scales, None)
    torch.cuda.synchronize()
    new = set(cache.rglob("*")) - compiled
    assert not new, f"attention-front projections compiled after warmup: {new}"
"""


@pytest.mark.skipif(
    not (torch.cuda.is_available() and current_platform().is_blackwell),
    reason="DeepGEMM group-32 projections require Blackwell",
)
@pytest.mark.parametrize("groups", [8, 16])
def test_warmup_covers_every_deep_gemm_front_projection(tmp_path, groups):
    pytest.importorskip("deep_gemm")
    cache = tmp_path / "jit"
    result = subprocess.run(
        [sys.executable, "-c", _QUERY_WARMUP_SCRIPT, str(cache), str(groups)],
        env={**os.environ, "DG_JIT_CACHE_DIR": str(cache)},
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert result.returncode == 0, result.stdout + result.stderr

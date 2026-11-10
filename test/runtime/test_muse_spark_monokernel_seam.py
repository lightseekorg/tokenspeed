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

"""The Muse Spark decode monokernel seam: which forwards take the kernels,
and which models the kernels serve. CPU only."""

from __future__ import annotations

import os
import re
import sys
from types import SimpleNamespace

import pytest
import torch
from tokenspeed_kernel.platform import current_platform

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from tokenspeed.runtime.configs.muse_spark_config import (  # noqa: E402
    MuseSparkConfig,
)
from tokenspeed.runtime.distributed.mapping import Mapping  # noqa: E402
from tokenspeed.runtime.execution.context import ForwardContext  # noqa: E402
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode  # noqa: E402
from tokenspeed.runtime.execution.output_layout import (  # noqa: E402
    ForwardOutputLayout,
)
from tokenspeed.runtime.layers.moe.weights.loaders import (  # noqa: E402
    load_w2,
    load_w13,
)
from tokenspeed.runtime.layers.paged_attention import PagedAttention  # noqa: E402
from tokenspeed.runtime.models.muse_spark_monokernel import (  # noqa: E402
    MAX_REQUESTS,
    MAX_ROWS,
    NOT_TP8,
    OTHER_CONSTANTS,
    OTHER_KV,
    OTHER_SHAPES,
    SCALED_KV,
    CompiledLaunches,
    MuseSparkMonokernelState,
    enable_decode_monokernels,
    hands_whole_rows,
    kv_reasons,
    model_reasons,
    projection_hand_off,
    reserve_monokernel_heap,
    row_lengths,
    takes_monokernels,
)
from tokenspeed.runtime.utils.env import global_server_args_dict  # noqa: E402


def _ctx(mode: ForwardMode, bs: int, rows: int) -> ForwardContext:
    return ForwardContext(
        attn_backend=None,
        token_to_kv_pool=None,
        bs=bs,
        num_extends=bs if mode.is_extend() else 0,
        input_num_tokens=rows,
        forward_mode=mode,
        output_layout=ForwardOutputLayout(0, 0, bs, 1),
    )


def test_decode_steps_of_up_to_max_requests_take_the_kernels() -> None:
    """A launch computes the rows of up to MAX_REQUESTS requests: a decode
    token each, or a speculative verify's tokens each, MAX_ROWS rows at
    most. Prefill, more requests and wider verifies keep the regular path."""
    assert takes_monokernels(1, _ctx(ForwardMode.DECODE, 1, 1))
    assert takes_monokernels(4, _ctx(ForwardMode.DECODE, 1, 4))
    assert takes_monokernels(8, _ctx(ForwardMode.DECODE, 1, 8))
    # Two requests' rows only while MAX_REQUESTS allows them.
    two = MAX_REQUESTS >= 2
    assert takes_monokernels(2, _ctx(ForwardMode.DECODE, 2, 2)) == two
    assert takes_monokernels(MAX_ROWS, _ctx(ForwardMode.DECODE, 2, MAX_ROWS)) == two
    assert not takes_monokernels(MAX_ROWS + 2, _ctx(ForwardMode.DECODE, 2, 18))
    assert not takes_monokernels(3, _ctx(ForwardMode.DECODE, 3, 3))
    assert not takes_monokernels(1, _ctx(ForwardMode.EXTEND, 1, 1))
    assert not takes_monokernels(1, _ctx(ForwardMode.IDLE, 1, 1))


def test_each_request_split_of_the_same_rows_compiles_ahead_once() -> None:
    # 2 rows are two decode requests (chain 1) or one 2-token verify (chain 2),
    # 8 rows one 8-token verify or two 4-token ones: different binaries, each
    # compiled on every rank before its first launch, and only then.
    compiled = CompiledLaunches()
    for rows, chain in ((2, 1), (2, 2), (8, 8), (8, 4)):
        assert compiled.first(5, rows, chain, False, False)
    for rows, chain in ((2, 1), (2, 2), (8, 8), (8, 4)):
        assert not compiled.first(5, rows, chain, False, False)
    assert compiled.first(5, 2, 1, True, False)
    assert compiled.first(6, 2, 1, False, False)


def test_a_launch_that_stores_a_tap_compiles_ahead_once() -> None:
    # Storing a draft's tap is a compile-time argument of the kernel (TAP):
    # the same layer and rows with and without a tap are two binaries, so the
    # tapped one compiles on every rank before its first launch too.
    compiled = CompiledLaunches()
    assert compiled.first(60, 8, 8, False, False)
    assert compiled.first(60, 8, 8, False, True)
    assert not compiled.first(60, 8, 8, False, True)


def test_launches_between_fused_layers_hand_the_projection_on() -> None:
    """Past one row a launch whose next layer is fused writes that layer's
    q | k | v projection into the copy the next launch reads, and a launch
    after a fused layer reads its own there: layers 1 and 60 border the
    regular layers 0 and 61. One row keeps every projection in its launch."""
    fused = set(range(1, 61))
    proj = torch.arange(2 * 8 * 4, dtype=torch.int32).view(2, 8, 4)
    first_in, first_out = projection_hand_off(1, 8, fused, proj)
    assert first_in is None and first_out.data_ptr() == proj[0].data_ptr()
    mid_in, mid_out = projection_hand_off(2, 8, fused, proj)
    assert mid_in.data_ptr() == proj[0].data_ptr()
    assert mid_out.data_ptr() == proj[1].data_ptr()
    last_in, last_out = projection_hand_off(60, 8, fused, proj)
    assert last_in.data_ptr() == proj[0].data_ptr() and last_out is None
    # Every launch reads the copy the launch before it wrote.
    for i in range(2, 61):
        assert projection_hand_off(i, 8, fused, proj)[0].data_ptr() == (
            projection_hand_off(i - 1, 8, fused, proj)[1].data_ptr()
        )
    assert projection_hand_off(30, 1, fused, proj) == (None, None)


def test_a_launch_hands_whole_rows_where_the_model_reads_them() -> None:
    """Between two launches the residual stays owned columns; a launch
    gathers whole rows before a regular layer, and before a layer that a
    draft taps unless the launches store the taps themselves."""
    served = set(range(1, 61))
    assert not hands_whole_rows(5, served, frozenset(), False)
    assert hands_whole_rows(60, served, frozenset(), False)
    # A tapped layer whose taps the model captures (the launches store none,
    # for example because another tapped layer follows layer 0).
    assert hands_whole_rows(5, served, frozenset({1, 6}), False)
    assert not hands_whole_rows(5, served, frozenset({1, 6}), True)


def test_a_verify_step_gives_each_row_its_token_length() -> None:
    """Verify metadata holds each request's length after the step; row t of
    a request's rows is its token at position seq - chain + t."""
    offsets = torch.arange(1 - MAX_ROWS, 1, dtype=torch.int32)
    seq_lens = torch.tensor([100, 7], dtype=torch.int32)
    assert row_lengths(seq_lens, offsets, 1, 8).tolist() == list(range(93, 101))
    # Two requests: each request's rows end at its own length.
    assert row_lengths(seq_lens, offsets, 2, 3).tolist() == [98, 99, 100, 5, 6, 7]
    # One decode row a request reads the metadata as is.
    assert row_lengths(seq_lens, offsets, 2, 1).tolist() == [100, 7]


def _text_config(**overrides) -> dict:
    """Muse Spark Open's decoder (four of its 62 layers)."""
    layer_types = ["sliding_attention", "full_attention"] * 2
    config = {
        "vocab_size": 202048,
        "hidden_size": 8192,
        "head_dim": 64,
        "num_attention_heads": 128,
        "num_key_value_heads": 16,
        "num_hidden_layers": len(layer_types),
        "max_position_embeddings": 1048576,
        "rms_norm_eps": 1e-5,
        "post_norm_eps": 1e-8,
        "final_logit_softcapping": 20.0,
        "qk_scale_factor": 5.4730064863875,
        "output_multiplier": 0.17677669529663687,
        "residual_gate_temperature": 0.3,
        "sliding_window": 2048,
        "layer_types": layer_types,
        "layer_rope_theta": [
            0.0 if t == "full_attention" else 500000.0 for t in layer_types
        ],
        "mlp_layer_types": ["sparse"] * len(layer_types),
        "rope_parameters": {"rope_theta": 500000.0, "rope_type": "default"},
        "moe_hidden_size": 4096,
        "moe_intermediate_size": 4096,
        "num_local_experts": 256,
        "num_experts_per_tok": 8,
        "num_shared_experts": 0,
        "routed_scaling_factor": 1.0,
    }
    config.update(overrides)
    return config


def _config(**overrides) -> MuseSparkConfig:
    return MuseSparkConfig(
        text_config=_text_config(**overrides),
        vision_config={
            "hidden_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "intermediate_size": 32,
            "layer_types": ["full_attention"],
            "layer_norm_eps": 1e-5,
            "merge_size": 2,
            "patch_size": 2,
            "patch_temporal": 2,
            "pos_emb_height": 2,
            "pos_emb_width": 2,
            "max_position_embeddings": 4,
            "rope_parameters": {"rope_theta": 10000, "rope_type": "axial"},
            "hidden_act": "gelu",
        },
        audio_config={},
        image_token_id=200,
        video_token_id=201,
        audio_token_id=202,
        out_hidden_size=128,
        projector_hidden_size=64,
        projector_hidden_act="gelu",
    )


def _mapping(moe_tp: int, moe_ep: int) -> Mapping:
    return Mapping(
        rank=0,
        world_size=8,
        attn_tp_size=8,
        attn_dp_size=1,
        dense_tp_size=8,
        dense_dp_size=1,
        moe_tp_size=moe_tp,
        moe_ep_size=moe_ep,
        moe_dp_size=1,
        vision_tp_size=1,
        vision_dp_size=8,
        linear_attn_tp_size=8,
        pp_size=1,
        pp_layer_partition=None,
        nprocs_per_node=8,
        nnodes=1,
        base_gpu_id=0,
        gpu_id_step=1,
    )


def test_muse_spark_open_at_tp8_ep1_has_monokernels() -> None:
    assert model_reasons(_config(), _mapping(8, 1)) == []


@pytest.mark.parametrize("moe_tp, moe_ep", [(1, 8), (4, 2), (2, 4)])
def test_other_moe_splits_keep_the_regular_path(moe_tp, moe_ep) -> None:
    """Only MoE TP8 / EP1 is served: on a TP1 / EP8 server a rank holds whole
    experts, not the slice of every expert that the kernel reads."""
    assert model_reasons(_config(), _mapping(moe_tp, moe_ep)) == [NOT_TP8]


class _UnpackedLayer(torch.nn.Module):
    """What enable_decode_monokernels touches of a monokernel decoder layer:
    its launch."""

    def __init__(self) -> None:
        super().__init__()
        self.fused = None


def test_a_model_the_kernels_cannot_pack_fails_startup() -> None:
    """--decode-monokernel never serves a layer without its kernel: on a
    split the kernels do not serve, loading fails and names every layer
    with the reason (docs/design/monokernel.md)."""
    model = torch.nn.Module()
    model.config = _config()
    model.mapping = _mapping(4, 2)
    model.layers = torch.nn.ModuleList(_UnpackedLayer() for _ in range(4))
    with pytest.raises(ValueError, match=re.escape(f"{NOT_TP8}: layers 0, 1, 2, 3")):
        enable_decode_monokernels(model)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not current_platform().is_cdna4,
    reason="the stuck-poll record is a gfx950 kernel's",
)
@pytest.mark.parametrize(
    "taps, stored",
    [([], 0), ([2, 31, 58], 3), ([1, 31], 0)],
    ids=["no-draft", "launches-store-taps", "a-tap-after-layer-0"],
)
def test_the_state_allocates_what_it_reserved_on_the_iris_heap(
    monkeypatch, taps, stored
) -> None:
    """The model reserves its state's Iris heap buffers before the heap
    exists, and the heap is created with only those bytes on top of the
    communication buffers: what the state allocates once it exists is
    exactly what was reserved, with and without the draft's taps."""
    from tokenspeed_kernel.ops.communication import _iris
    from tokenspeed_kernel.ops.model.muse_spark.monokernel import FusedDecodeLayer

    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList(_UnpackedLayer() for _ in range(62))
    # Sliding and full layers' plans differ in size; the largest serves all.
    for index in range(1, 61):
        model.layers[index].fused = FusedDecodeLayer(
            decode=None, weights=None, scratch_pairs=4096 + index, late_weights=None
        )
    model.layers_to_capture = taps
    monkeypatch.setattr(_iris, "_reserved", {})
    reserve_monokernel_heap(model)
    reserved = _iris._reserved["muse_spark_monokernel"]

    # A heap that records each allocation at the Torch allocator's alignment.
    allocated: list[int] = []

    class Heap:
        def zeros(self, shape, dtype):
            allocated.append(_iris.allocation_bytes(tuple(shape), dtype))
            return torch.zeros(shape, dtype=dtype, device="cuda")

    class Group:
        def __init__(self, ctx, group=None, rank=0, device=None):
            self.bases = torch.zeros(8, dtype=torch.int64)

        def finish_prepare(self) -> None:
            pass

    monkeypatch.setattr(_iris, "existing_context", Heap)
    monkeypatch.setattr(_iris, "IrisGroup", Group)
    monkeypatch.setitem(global_server_args_dict, "monokernel_stuck_record_dir", None)
    fused = {i: layer.fused for i, layer in enumerate(model.layers) if layer.fused}
    state = MuseSparkMonokernelState(fused, None, None, 0, taps)
    assert state.scratch.numel() == 4096 + 60
    assert len(state.tap_slots) == stored
    assert sum(allocated) == reserved


def test_an_fp8_kv_cache_takes_the_kernels(monkeypatch) -> None:
    """The kernels read and write the regular path's unit-scale FP8 e4m3."""
    monkeypatch.setitem(global_server_args_dict, "kv_cache_dtype", "fp8_e4m3")
    assert model_reasons(_config(), _mapping(8, 1)) == []


def test_an_mxfp8_kv_cache_keeps_the_regular_path(monkeypatch) -> None:
    monkeypatch.setitem(global_server_args_dict, "kv_cache_dtype", "mxfp8")
    assert model_reasons(_config(), _mapping(8, 1)) == [OTHER_KV]


def _attention() -> PagedAttention:
    """One Muse Spark rank's attention: 16 q heads, 2 KV heads of 64."""
    return PagedAttention(16, 64, 0.125, 2, 1, rotary_emb=None, qk_norm=None)


@pytest.mark.parametrize("kv_cache_dtype", ["fp8_e4m3", "auto"])
def test_a_layer_with_fp8_kv_scales_keeps_the_regular_path_on_an_fp8_cache(
    monkeypatch, kv_cache_dtype
) -> None:
    """A checkpoint's k_scale applies to an FP8 cache, and the kernels apply
    none; a BF16 cache ignores the scales."""
    monkeypatch.setitem(global_server_args_dict, "kv_cache_dtype", kv_cache_dtype)
    attention = _attention()
    assert kv_reasons(attention) == []
    attention.load_kv_cache_scale("model.layers.1.self_attn.k_scale", torch.tensor(2.0))
    want = [SCALED_KV] if kv_cache_dtype == "fp8_e4m3" else []
    assert kv_reasons(attention) == want


@pytest.mark.parametrize(
    "override, reason",
    [
        ({"output_multiplier": 0.125}, OTHER_CONSTANTS),
        ({"qk_scale_factor": 4.0}, OTHER_CONSTANTS),
        ({"num_local_experts": 128}, OTHER_SHAPES),
        ({"moe_hidden_size": 2048}, OTHER_SHAPES),
    ],
)
def test_other_baked_in_values_keep_the_regular_path(override, reason) -> None:
    """The kernel bakes Muse Spark Open's dimensions and constants in: a
    checkpoint with other values has no monokernel."""
    assert model_reasons(_config(**override), _mapping(8, 1)) == [reason]

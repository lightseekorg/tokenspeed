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

"""Muse Spark experts under attention DP8 / EP8 on eight gfx950 GPUs.

Each rank routes its own rows: MXFP4 layers through Gluon Petit, BF16 layers
through the gathered expert exchange. Experts and projections hold one exact
nonzero per row, so an independent FP32 reference reproduces the kernels'
rounding points. The remaining differences are FP32 summation order before a
BF16 rounding; BF16 experts also round each rank's partial sum to BF16. The
reference reuses the layer's row-wise router and pre-expert norm, which other
tests cover, so both sides route identically.

It needs eight gfx950 GPUs and skips elsewhere, including the 1-GPU suite.
"""

from __future__ import annotations

import os
import socket
import sys

import pytest
import torch
import torch.multiprocessing as mp
from tokenspeed_kernel.platform import current_platform

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="runtime-1gpu")

from tokenspeed.runtime.configs.muse_spark_config import (
    FULL_ATTENTION,
    SLIDING_ATTENTION,
    MuseSparkConfig,
)
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.layers.quantization.mxfp4 import Mxfp4Config

WORLD_SIZE = 8
HIDDEN = 4096
EXPERTS = 256
TOP_K = 8
LOCAL_EXPERTS = EXPERTS // WORLD_SIZE
# E2M1 codes 2, 3, 4 are 1.0, 1.5, 2.0; with E8M0 scale 123 they are scaled by 1/16.
_LEVELS = (1.0, 1.5, 2.0)
_MXFP4_SCALE = 123


def _text_config() -> dict:
    layer_types = [SLIDING_ATTENTION, FULL_ATTENTION]
    return {
        "vocab_size": 256,
        "hidden_size": HIDDEN,
        "head_dim": 64,
        "num_attention_heads": HIDDEN // 64,
        "num_key_value_heads": 8,
        "num_hidden_layers": 2,
        "max_position_embeddings": 4096,
        "rms_norm_eps": 1e-5,
        "post_norm_eps": 1e-8,
        "final_logit_softcapping": 20.0,
        "qk_scale_factor": 5.4730064863875,
        "output_multiplier": 0.125,
        "residual_gate_temperature": 0.3,
        "sliding_window": 128,
        "layer_types": layer_types,
        "layer_rope_theta": [500000.0, 0.0],
        "mlp_layer_types": ["sparse", "sparse"],
        "rope_parameters": {"rope_theta": 500000.0, "rope_type": "default"},
        "moe_hidden_size": HIDDEN,
        "moe_intermediate_size": HIDDEN,
        "num_local_experts": EXPERTS,
        "num_experts_per_tok": TOP_K,
        "num_shared_experts": 0,
        "routed_scaling_factor": 1.0,
    }


def _config() -> MuseSparkConfig:
    return MuseSparkConfig(
        text_config=_text_config(),
        vision_config={},
        audio_config={},
        image_token_id=200,
        video_token_id=201,
        audio_token_id=202,
        out_hidden_size=128,
        projector_hidden_size=64,
        projector_hidden_act="gelu",
    )


def _attention_dp_mapping(rank: int) -> Mapping:
    """``--tp 1 --data-parallel-size 8 --ep-size 8`` on one node."""
    return Mapping(
        rank=rank,
        world_size=WORLD_SIZE,
        attn_tp_size=1,
        attn_dp_size=WORLD_SIZE,
        dense_tp_size=1,
        moe_tp_size=1,
        moe_ep_size=WORLD_SIZE,
        vision_tp_size=1,
        vision_dp_size=1,
        pp_size=1,
        nprocs_per_node=WORLD_SIZE,
        nnodes=1,
    )


def _quark_mxfp4_config() -> Mxfp4Config:
    return Mxfp4Config.from_config(
        {
            "quant_method": "quark",
            "global_quant_config": {
                "weight": {
                    "dtype": "fp4",
                    "qscheme": "per_group",
                    "group_size": 32,
                    "scale_format": "e8m0",
                    "is_dynamic": False,
                },
                "input_tensors": {
                    "dtype": "fp4",
                    "qscheme": "per_group",
                    "group_size": 32,
                    "scale_format": "e8m0",
                    "is_dynamic": True,
                },
            },
            "export": {"pack_method": "reorder", "weight_format": "real_quantized"},
            "exclude": [],
        }
    )


def _expert_columns(expert: int, device: torch.device):
    """The one input column each gate, up, and down row of ``expert`` reads."""
    channel = torch.arange(HIDDEN, device=device)
    return (
        (channel * 7) % HIDDEN,
        (channel * 7 + 3) % HIDDEN,
        (channel * 11 + expert) % HIDDEN,
    )


def _expert_levels(expert: int) -> tuple[float, float]:
    """Gate/down and up multipliers of ``expert``'s single nonzero per row."""
    return _LEVELS[expert % 3] / 16, _LEVELS[(expert + 1) % 3] / 16


def _load_mxfp4_experts(experts: torch.nn.Module, ep_rank: int) -> None:
    """Fill serialized MXFP4 experts with one nonzero E2M1 value per row."""
    device = experts.w13_weight.device
    experts.w13_weight.data.zero_()
    experts.w2_weight.data.zero_()
    rows = torch.arange(HIDDEN, device=device)
    for local in range(LOCAL_EXPERTS):
        expert = ep_rank * LOCAL_EXPERTS + local
        gate_cols, up_cols, down_cols = _expert_columns(expert, device)
        for half, cols, code in (
            (0, gate_cols, 2 + expert % 3),
            (1, up_cols, 2 + (expert + 1) % 3),
        ):
            experts.w13_weight.data[local, half * HIDDEN + rows, cols // 2] = (
                code << (cols % 2) * 4
            ).to(torch.uint8)
        experts.w2_weight.data[local, rows, down_cols // 2] = (
            (2 + expert % 3) << (down_cols % 2) * 4
        ).to(torch.uint8)
    experts.w13_weight_scale.data.fill_(_MXFP4_SCALE)
    experts.w2_weight_scale.data.fill_(_MXFP4_SCALE)


def _load_bf16_experts(experts: torch.nn.Module, ep_rank: int) -> None:
    """Fill BF16 experts with the same single-nonzero pattern as the MXFP4 ones."""
    device = experts.w13_weight.device
    experts.w13_weight.data.zero_()
    experts.w2_weight.data.zero_()
    rows = torch.arange(HIDDEN, device=device)
    for local in range(LOCAL_EXPERTS):
        expert = ep_rank * LOCAL_EXPERTS + local
        gate_cols, up_cols, down_cols = _expert_columns(expert, device)
        gate_level, up_level = _expert_levels(expert)
        experts.w13_weight.data[local, rows, gate_cols] = gate_level
        experts.w13_weight.data[local, HIDDEN + rows, up_cols] = up_level
        experts.w2_weight.data[local, rows, down_cols] = gate_level


def _load_dense_weights(moe: torch.nn.Module, seed: int) -> None:
    """Load router, norms, and exact permutation projections, the same on every rank."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    device = moe.gate.weight.device

    def randn(*shape: int, scale: float) -> torch.Tensor:
        return (
            torch.randn(*shape, generator=generator, dtype=torch.float32) * scale
        ).to(device)

    moe.gate.weight.data.copy_(randn(HIDDEN, EXPERTS, scale=0.05))
    moe.gate.e_score_correction_bias.data.copy_(randn(EXPERTS, scale=0.1))
    moe.gate.process_weights_after_loading(moe.gate)
    moe.pre_expert_norm.weight.data.copy_(randn(HIDDEN, scale=0.1))
    moe.post_expert_norm.weight.data.copy_(randn(HIDDEN, scale=0.1))
    # Permutations scaled by powers of two keep both projections exact.
    for projection, offset in ((moe.pre_expert_proj, 5), (moe.post_expert_proj, 9)):
        weight = torch.zeros(HIDDEN, HIDDEN, dtype=torch.float32)
        rows = torch.arange(HIDDEN)
        weight[rows, (rows * 3 + offset) % HIDDEN] = 2.0 ** (rows % 3 - 1).float()
        projection.weight.data.copy_(weight.to(device))


def _mxfp4_round(value: torch.Tensor, input_quantization: bool) -> torch.Tensor:
    """Petit's MXFP4 activation rounding: E8M0 group scales, nearest-even E2M1.

    Petit rounds an input group's scale up (max / 6) and an intermediate's to
    the nearest power of two, so neither matches the shared Quark reference.
    """
    shape = value.shape
    blocks = value.float().reshape(-1, 32)
    maximum = blocks.abs().amax(dim=1)
    if input_quantization:
        bits = (maximum * (1.0 / 6.0)).contiguous().view(torch.int32)
        exponent = (bits >> 23) + ((bits & 0x7FFFFF) != 0).int()
    else:
        bits = maximum.contiguous().view(torch.int32)
        exponent = (((bits + 0x00400000) & 0x7F800000) >> 23).clamp(min=2) - 2
    scale = (exponent << 23).view(torch.float32)[:, None]
    scale = torch.where(scale == 0, 1.0, scale)
    normalized = blocks.abs() / scale
    levels = torch.tensor([0.0, 1.0, 2.0, 4.0, 0.5, 1.5, 3.0, 6.0], device=value.device)
    closest = (normalized[..., None] - levels).abs().argmin(dim=-1)
    return (levels[closest] * blocks.sign() * scale).reshape(shape)


def _reference_routes(
    compressed: torch.Tensor, topk_ids: torch.Tensor, quantized: bool
) -> torch.Tensor:
    """Every selected expert's BF16 output ``[rows, top_k, HIDDEN]`` in FP32."""
    inputs = _mxfp4_round(compressed, True) if quantized else compressed.float()
    routes = torch.zeros(
        (*topk_ids.shape, HIDDEN), dtype=torch.float32, device=compressed.device
    )
    for expert in torch.unique(topk_ids).tolist():
        row, slot = torch.where(topk_ids == expert)
        gate_cols, up_cols, down_cols = _expert_columns(expert, compressed.device)
        gate_level, up_level = _expert_levels(expert)
        gate = inputs[row[:, None], gate_cols] * gate_level
        up = inputs[row[:, None], up_cols] * up_level
        activated = gate * torch.sigmoid(gate) * up
        activated = (
            _mxfp4_round(activated, False)
            if quantized
            else activated.to(torch.bfloat16).float()
        )
        routes[row, slot] = (
            (activated[:, down_cols] * gate_level).to(torch.bfloat16).float()
        )
    return routes


def _reference_output(
    moe: torch.nn.Module,
    hidden_states: torch.Tensor,
    router_input: torch.Tensor,
    quantized: bool,
) -> torch.Tensor:
    """FP32 Muse expert block for ``hidden_states`` past the row-wise prefix."""
    if hidden_states.shape[0] == 0:
        return hidden_states.new_empty((0, HIDDEN))
    compressed = moe.pre_expert_norm(moe.pre_expert_proj(hidden_states), torch.bfloat16)
    route_weights, topk_ids = moe._select_experts(moe.gate(router_input, False), None)
    routes = _reference_routes(compressed, topk_ids.long(), quantized)
    post = moe.post_expert_norm
    # Muse rounds the centered norm weight to BF16.
    routes = routes * (post.weight.float() + post.weight_offset).bfloat16().float()
    routes = routes * torch.rsqrt(routes.square().mean(-1, keepdim=True) + post.eps)
    weighted = routes * route_weights[..., None]
    if quantized:
        # Petit returns every route; this rank sums them and rounds once.
        combined = weighted.sum(1).to(torch.bfloat16)
    else:
        # Each owner rank rounds its experts' partial sum to BF16 before the
        # FP32 reduce-scatter adds the partials.
        owner = topk_ids.long() // LOCAL_EXPERTS
        combined = sum(
            (weighted * (owner == rank)[..., None]).sum(1).to(torch.bfloat16).float()
            for rank in range(WORLD_SIZE)
        ).to(torch.bfloat16)
    return (combined.float() @ moe.post_expert_proj.weight.float().T).to(torch.bfloat16)


def _attention_dp_worker(rank: int, port: int) -> None:
    from tokenspeed.runtime.distributed.process_group_manager import (
        process_group_manager,
    )
    from tokenspeed.runtime.layers.moe import utils as moe_utils
    from tokenspeed.runtime.models.muse_spark import MuseSparkMoE
    from tokenspeed.runtime.utils.env import global_server_args_dict

    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    mapping = _attention_dp_mapping(rank)
    process_group_manager.init_distributed(
        mapping,
        distributed_init_method=f"tcp://127.0.0.1:{port}",
        backend="nccl",
        timeout=300,
        device_id=device,
    )
    process_group_manager.init_process_group(mapping.moe.ep_group)
    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setitem(global_server_args_dict, "mapping", mapping)
            # Server limits read while preparing the communication backends.
            for key, value in (
                ("chunked_prefill_size", 1024),
                ("max_prefill_tokens", 1024),
                ("max_model_len", 4096),
                ("max_num_seqs", 64),
            ):
                patch.setitem(global_server_args_dict, key, value)
            patch.setattr(
                moe_utils, "ALL2ALL_BACKEND", moe_utils.All2AllBackend.GLUON_PETIT
            )
            patch.setattr(moe_utils, "MOE_BACKEND", moe_utils.MoeBackend.GLUON_PETIT)
            torch.set_default_dtype(torch.bfloat16)
            config = _config()
            with torch.device(device):
                layers = {
                    quantized: MuseSparkMoE(
                        config=config,
                        mapping=mapping,
                        quant_config=_quark_mxfp4_config() if quantized else None,
                        layer_id=int(quantized),
                        prefix=f"model.layers.{int(quantized)}.mlp",
                        alt_stream=torch.cuda.Stream(),
                    )
                    for quantized in (True, False)
                }
            petit, bf16 = layers[True], layers[False]
            assert petit.exchanges_rank_local_rows and bf16.exchanges_rank_local_rows
            assert petit.uses_petit and not bf16.uses_petit
            _load_mxfp4_experts(petit.experts, mapping.moe.ep_rank)
            _load_bf16_experts(bf16.experts, mapping.moe.ep_rank)
            for seed, moe in enumerate(layers.values()):
                _load_dense_weights(moe, seed)
            # Repacking creates Petit's symmetric workspace on every rank.
            petit.experts.process_weights_after_loading(petit.experts)
            bf16.experts.process_weights_after_loading(bf16.experts)
            _check_eager_rows(layers, rank, device)
            _check_graph_replay(layers, rank, device)
            torch.cuda.synchronize()
    finally:
        torch.distributed.destroy_process_group()


def _inputs(rows: int, rank: int, step: int, device: torch.device):
    generator = torch.Generator(device="cpu").manual_seed(1000 * step + rank)
    hidden = torch.randn(rows, HIDDEN, generator=generator, dtype=torch.float32).to(
        device
    )
    return hidden.to(torch.bfloat16), hidden


def _assert_matches(
    output: torch.Tensor, expected: torch.Tensor, quantized: bool
) -> None:
    # A BF16-rounded rank partial may differ by one BF16 ulp of its own
    # magnitude (normalized routes, about one); the post-expert permutation
    # scales it by up to two. Partials can cancel, so bound that absolutely.
    torch.testing.assert_close(
        output,
        expected,
        atol=1e-5 if quantized else 2 * 2.0**-8,
        rtol=8e-3,
        msg=lambda message: f"{'Petit' if quantized else 'BF16'} experts: {message}",
    )


def _check_eager_rows(layers: dict, rank: int, device: torch.device) -> None:
    # Ranks without rows, uneven tile boundaries, a full-capacity rank, and a
    # round in which no rank holds rows (every rank skips the exchange).
    for step, counts in enumerate(
        (
            [0, 1, 3, 17, 64, 5, 0, 130],
            [1] * WORLD_SIZE,
            [1024, 0, 0, 0, 0, 0, 0, 1],
            [0] * WORLD_SIZE,
        )
    ):
        hidden, router_input = _inputs(counts[rank], rank, step, device)
        for quantized, moe in layers.items():
            output = moe.forward_rank_local(hidden, router_input, counts)
            assert output.shape == (counts[rank], HIDDEN)
            expected = _reference_output(moe, hidden, router_input, quantized)
            _assert_matches(output, expected, quantized)


def _check_graph_replay(layers: dict, rank: int, device: torch.device) -> None:
    counts = [4, 0, 9, 1, 4, 16, 2, 7]
    hidden, router_input = _inputs(counts[rank], rank, 0, device)
    # Both layers share one graph, so Petit and the gathered collectives
    # replay in order with refreshed inputs.
    eager = [
        moe.forward_rank_local(hidden, router_input, counts) for moe in layers.values()
    ]
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = [
            moe.forward_rank_local(hidden, router_input, counts)
            for moe in layers.values()
        ]
    for step in (0, 1, 2):
        new_hidden, new_router_input = _inputs(counts[rank], rank, 10 + step, device)
        if step:
            hidden.copy_(new_hidden)
            router_input.copy_(new_router_input)
        graph.replay()
        for (quantized, moe), output, reference in zip(layers.items(), captured, eager):
            expected = _reference_output(moe, hidden, router_input, quantized)
            _assert_matches(output, expected, quantized)
            if step == 0:
                torch.testing.assert_close(output, reference, atol=0, rtol=0)


def test_attention_dp_experts_match_reference() -> None:
    if (
        not torch.cuda.is_available()
        or not current_platform().is_cdna4
        or torch.cuda.device_count() < WORLD_SIZE
    ):
        pytest.skip(f"requires {WORLD_SIZE} gfx950 GPUs")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    mp.spawn(_attention_dp_worker, args=(port,), nprocs=WORLD_SIZE, join=True)

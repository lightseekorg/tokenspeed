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

"""CPU behavior tests for original-precision expert loading and execution."""

import pytest
import torch

from tokenspeed.runtime.configs.inkling2 import Inkling2ModelConfig
from tokenspeed.runtime.distributed import Mapping
from tokenspeed.runtime.layers.moe import inkling2 as inkling2_moe
from tokenspeed.runtime.layers.moe.inkling2 import Inkling2MoE
from tokenspeed.runtime.layers.quantization.inkling2 import Inkling2QuantConfig
from tokenspeed.runtime.moe.expert_location import (
    ExpertLocationMetadata,
    set_global_expert_location_metadata,
)
from tokenspeed.runtime.utils.env import global_server_args_dict


def _fold(rows: torch.Tensor) -> torch.Tensor:
    """Sum BF16 ``[T, S, H]`` rows over ``S`` in FP32 order from +0.0, rounding once."""
    folded = torch.zeros(rows[:, 0].shape)
    for slot in rows.unbind(1):
        folded += slot.float()
    return folded.to(torch.bfloat16)


def _config(use_gate_bias=True, hidden=64, intermediate=32):
    return Inkling2ModelConfig(
        model_max_length=128,
        use_value_norm_no_gamma=True,
        qk_l2_clip_threshold=10.0,
        rel_l2_clip_threshold=8.0,
        n_layers_per_attn_res_block=1,
        attn_res_normalize_values=True,
        mlp_gate_softcap=4.0,
        mlp_up_softcap=25.0,
        vocab_size=64,
        hidden_size=hidden,
        intermediate_size=intermediate,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=64,
        d_rel=16,
        rel_extent=128,
        local_layer_ids=[],
        sliding_window_size=128,
        swa_num_attention_heads=1,
        swa_num_key_value_heads=1,
        swa_head_dim=64,
        rms_norm_eps=1e-6,
        q_bias=False,
        o_bias=False,
        sconv_kernel_size=4,
        n_routed_experts=4,
        n_shared_experts=0,
        num_experts_per_tok=2,
        route_scale=8.0,
        use_gate_bias=use_gate_bias,
        log_scaling_n_floor=128,
        log_scaling_alpha=0.1,
        unpadded_vocab_size=64,
        final_logit_softcapping=30.0,
    )


def _precision():
    prefix = "model.llm.layers.0.mlp"

    def dense(dtype, names):
        return dict(
            compute_dtype=dtype,
            output_dtype=dtype,
            tensors=names,
            weight_transform="cast",
            weight_block_shape=[],
            required_activation_quantization=None,
        )

    return Inkling2QuantConfig.from_config(
        {
            "quant_method": "inkling2",
            "model": {
                "parameter_groups": [
                    dense("bfloat16", [prefix + ".gate.weight"]),
                    dense("float32", [prefix + ".gate.bias"]),
                    dict(
                        compute_dtype="nvfp4",
                        output_dtype="uint8",
                        tensors=[
                            prefix + ".experts.w13_weight",
                            prefix + ".experts.w2_weight",
                        ],
                        weight_transform="nvfp4_nearest",
                        weight_block_shape=[1, 16],
                        weight_scale_format="amax_denominator",
                        required_activation_quantization=dict(
                            format="nvfp4_e2m1",
                            block_size=16,
                            scale_dtype="float8_e4m3fn",
                            rounding="nearest",
                            amax_type="per_token",
                            scale_selection_rule="fouroversix_mae",
                        ),
                    ),
                ]
            },
        }
    )


def _layer(rank, world_size, ep_size, use_gate_bias=True, hidden=64, intermediate=32):
    return Inkling2MoE(
        _config(use_gate_bias, hidden, intermediate),
        Mapping(
            rank=rank,
            world_size=world_size,
            moe_tp_size=1,
            moe_ep_size=ep_size,
            moe_dp_size=1,
        ),
        layer_id=0,
        quant_config=_precision(),
        prefix="model.llm.layers.0.mlp",
    )


def test_cpu_forward_retains_quantized_parameters_and_rounding():
    layer = _layer(0, 1, 1)
    with torch.no_grad():
        layer.gate.weight.zero_()
        layer.gate.bias.zero_()
        layer.experts.w13_weight.fill_(0x11)
        layer.experts.w2_weight.fill_(0x11)
        layer.experts.w13_scale.fill_(0.125)
        layer.experts.w2_scale.fill_(0.125)
        layer.experts.w13_amax.fill_(2688.0)
        layer.experts.w2_amax.fill_(2688.0)
        layer.experts.w13_outer_scale_denominator.fill_(2688.0)
        layer.experts.w2_outer_scale_denominator.fill_(2688.0)
    packed = layer.experts.w13_weight.clone()
    rows = layer(
        torch.ones((3, 64), dtype=torch.bfloat16), None, reduce=True, fold_into=None
    )
    # The layer folds the token's two routes itself.
    assert rows.shape == (3, 1, 64)
    result = rows[:, 0]
    gate = torch.tensor(4.0, dtype=torch.float32)
    activated = (
        (
            ((4.0 * torch.tanh(gate / 4.0)) * torch.sigmoid(gate))
            * (25.0 * torch.tanh(gate / 25.0))
        )
        * 4.0
    ).to(torch.bfloat16)
    expected = ((activated.float() * 2.0).to(torch.bfloat16).float() * 2.0).to(
        torch.bfloat16
    )
    assert torch.equal(result, expected.expand(3, 64))
    assert torch.equal(layer.experts.w13_weight, packed)
    assert layer(
        torch.empty((0, 64), dtype=torch.bfloat16), None, reduce=True, fold_into=None
    ).shape == (0, 1, 64)


def test_absent_gate_bias_routes_like_a_zero_bias():
    generator = torch.Generator().manual_seed(7)
    absent = _layer(0, 1, 1, use_gate_bias=False)
    zero = _layer(0, 1, 1, use_gate_bias=True)
    with torch.no_grad():
        absent.gate.weight.copy_(
            torch.randn(absent.gate.weight.shape, generator=generator).bfloat16()
        )
        # Distinct random experts, so a different route changes the output.
        for weight in (absent.experts.w13_weight, absent.experts.w2_weight):
            weight.copy_(
                torch.randint(0, 256, weight.shape, generator=generator).to(
                    weight.dtype
                )
            )
        absent.experts.w13_scale.fill_(0.125)
        absent.experts.w2_scale.fill_(0.125)
        absent.experts.w13_amax.fill_(2688.0)
        absent.experts.w2_amax.fill_(2688.0)
        absent.experts.w13_outer_scale_denominator.fill_(2688.0)
        absent.experts.w2_outer_scale_denominator.fill_(2688.0)
        zero_parameters = dict(zero.named_parameters())
        for name, value in absent.named_parameters():
            zero_parameters[name].copy_(value)
        zero.gate.bias.zero_()
    x = torch.randn((9, 64), generator=generator).bfloat16()
    expected = zero(x, None, reduce=True, fold_into=None)
    assert torch.equal(absent(x, None, reduce=True, fold_into=None), expected)
    with torch.no_grad():
        zero.gate.bias.copy_(torch.tensor([0.0, 100.0, 200.0, 300.0]))
    assert not torch.equal(zero(x, None, reduce=True, fold_into=None), expected)


def test_ep_loader_copies_exact_expert_shard_and_replicates_denominator():
    layer = _layer(1, 2, 2)
    source = (
        torch.arange(4, dtype=torch.uint8)[:, None, None].expand(4, 64, 32).contiguous()
    )
    layer.experts.w13_weight.weight_loader(layer.experts.w13_weight, source)
    assert torch.equal(layer.experts.w13_weight, source[2:])
    denominator = torch.tensor([2688.0], dtype=torch.float32)
    layer.experts.w13_outer_scale_denominator.weight_loader(
        layer.experts.w13_outer_scale_denominator, denominator
    )
    assert torch.equal(layer.experts.w13_outer_scale_denominator, denominator)
    with pytest.raises(ValueError, match="dtype mismatch"):
        layer.experts.w13_weight.weight_loader(
            layer.experts.w13_weight, source.to(torch.bfloat16)
        )


def _checkpoint():
    generator = torch.Generator().manual_seed(0)
    tensors = {
        "w13_weight": torch.randint(0, 256, (4, 64, 32), generator=generator),
        "w2_weight": torch.randint(0, 256, (4, 64, 16), generator=generator),
    }
    tensors = {name: value.to(torch.uint8) for name, value in tensors.items()}
    for leaf, rows, width in (("w13", 64, 64), ("w2", 64, 32)):
        tensors[f"{leaf}_scale"] = torch.full(
            (4, rows, width // 16), 0.125, dtype=torch.float8_e4m3fn
        )
        tensors[f"{leaf}_amax"] = torch.tensor([672.0, 1344.0, 2688.0, 5376.0])
        tensors[f"{leaf}_outer_scale_denominator"] = torch.tensor([2688.0])
    return tensors


def _load(layer, checkpoint, router):
    for name, value in checkpoint.items():
        param = getattr(layer.experts, name)
        param.weight_loader(param, value)
    layer.gate.weight.weight_loader(layer.gate.weight, router)
    layer.gate.bias.weight_loader(layer.gate.bias, torch.zeros(4))


def test_expert_placement_moves_routes_to_replicas_without_changing_output(
    monkeypatch,
):
    # Four experts on six slots: a permutation plus replicas of experts 2, 0.
    placement = ExpertLocationMetadata.from_physical_to_logical_map(
        torch.tensor([[2, 0, 3, 1, 2, 0]]), 4, ep_size=1, ep_rank=0, ep_rank_nodes=(0,)
    )
    placement.enable_load_recording(Mapping(rank=0, world_size=1))
    placement.reserve_load_rows(16)
    monkeypatch.setitem(global_server_args_dict, "ep_dispatch_algorithm", "static")
    checkpoint = _checkpoint()
    router = torch.randn((4, 64), generator=torch.Generator().manual_seed(1)).to(
        torch.bfloat16
    )
    trivial = _layer(0, 1, 1)
    _load(trivial, checkpoint, router)
    set_global_expert_location_metadata(placement)
    try:
        placed = _layer(0, 1, 1)
    finally:
        set_global_expert_location_metadata(None)
    _load(placed, checkpoint, router)
    assert torch.equal(
        placed.experts.w13_weight, checkpoint["w13_weight"][[2, 0, 3, 1, 2, 0]]
    )
    x = torch.randn((9, 64), generator=torch.Generator().manual_seed(2)).to(
        torch.bfloat16
    )
    expected = trivial(x, None, reduce=True, fold_into=None)
    assert torch.equal(placed(x, None, reduce=True, fold_into=None), expected)
    load = placement.physical_load[0]
    assert load.sum().item() == 9 * 2 and load[4:].sum().item() > 0
    # Inkling2 has no zero experts and routes every token on every rank.
    monkeypatch.setitem(
        global_server_args_dict, "ep_dispatch_algorithm", "static_with_zero_expert"
    )
    set_global_expert_location_metadata(placement)
    try:
        with pytest.raises(ValueError, match="no zero experts"):
            _layer(0, 1, 1)
    finally:
        set_global_expert_location_metadata(None)


def test_two_rank_placement_rows_fold_to_the_single_rank_output(monkeypatch):
    # Rank 0 holds experts 2, 0, 3 and rank 1 holds 1, 2, 0; others' rows read zero.
    physical_to_logical = torch.tensor([[2, 0, 3, 1, 2, 0]])
    monkeypatch.setitem(global_server_args_dict, "ep_dispatch_algorithm", "static")
    rows = []

    def all_reduce(tensor, **_):
        # The test runs both ranks in one process; their sum is taken below.
        rows.append(tensor)
        return tensor

    monkeypatch.setattr(inkling2_moe, "all_reduce", all_reduce)
    checkpoint = _checkpoint()
    router = torch.randn((4, 64), generator=torch.Generator().manual_seed(1)).to(
        torch.bfloat16
    )
    x = torch.randn((9, 64), generator=torch.Generator().manual_seed(2)).to(
        torch.bfloat16
    )
    trivial = _layer(0, 1, 1)
    _load(trivial, checkpoint, router)
    for rank in (0, 1):
        set_global_expert_location_metadata(
            ExpertLocationMetadata.from_physical_to_logical_map(
                physical_to_logical, 4, ep_size=2, ep_rank=rank, ep_rank_nodes=(0, 0)
            )
        )
        try:
            placed = _layer(rank, 2, 2)
        finally:
            set_global_expert_location_metadata(None)
        _load(placed, checkpoint, router)
        assert placed.expert_start == 3 * rank
        placed(x, None, reduce=True, fold_into=None)
        received = len(rows)
        # Left to the caller, the reduction's input comes back unreduced.
        assert torch.equal(placed(x, None, reduce=False, fold_into=None), rows[-1])
        assert len(rows) == received
    # The all-reduce adds the ranks' folded BF16 rows in FP32 and rounds once.
    # With two routes per token, a token's routes are on one rank (the same
    # slot-order sum) or one on each (rows exact on their own): the same bits.
    assert torch.equal(
        _fold(torch.stack(rows, 1)[:, :, 0]),
        trivial(x, None, reduce=True, fold_into=None)[:, 0],
    )
    # Without EP there is no all-reduce to leave to the caller.
    with pytest.raises(ValueError, match="all-reduced batch"):
        trivial(x, None, reduce=False, fold_into=None)


def test_router_loader_refuses_storage_precision_change():
    layer = _layer(0, 1, 1)
    with pytest.raises(ValueError, match="expected torch.bfloat16"):
        layer.gate.weight.weight_loader(
            layer.gate.weight, torch.ones((4, 64), dtype=torch.float32)
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an admitted GPU")
def test_gpu_complete_moe_keeps_router_fp32_and_expert_bf16_boundaries():
    from tokenspeed_kernel.ops.moe.block_scaled import block_scaled_router_mm

    router_weight = torch.zeros((4, 64), dtype=torch.bfloat16, device="cuda")
    router_weight[0, 0] = 1.0
    router_weight[0, 1] = 2.0**-9
    router_input = torch.ones((3, 64), dtype=torch.bfloat16, device="cuda")
    logits = block_scaled_router_mm(router_input, router_weight, solution=None)
    expected_logits = torch.zeros((3, 4), dtype=torch.float32)
    expected_logits[:, 0] = 1.0 + 2.0**-9
    assert logits.dtype == torch.float32
    assert torch.equal(logits.cpu(), expected_logits)
    assert not torch.equal(logits.cpu(), logits.cpu().bfloat16().float())

    cpu = _layer(0, 1, 1)
    with torch.no_grad():
        cpu.gate.weight.zero_()
        cpu.gate.bias.zero_()
        cpu.experts.w13_weight.fill_(0x11)
        cpu.experts.w2_weight.fill_(0x11)
        cpu.experts.w13_scale.fill_(0.125)
        cpu.experts.w2_scale.fill_(0.125)
        cpu.experts.w13_amax.fill_(2688.0)
        cpu.experts.w2_amax.fill_(2688.0)
        cpu.experts.w13_outer_scale_denominator.fill_(2688.0)
        cpu.experts.w2_outer_scale_denominator.fill_(2688.0)
    x = torch.ones((3, 64), dtype=torch.bfloat16)
    expected = cpu(x, None, reduce=True, fold_into=None)
    # Built on the GPU, as the loader builds it, so its plan picks a GPU kernel.
    with torch.device("cuda"):
        gpu = _layer(0, 1, 1)
    gpu.load_state_dict(cpu.state_dict())
    result = gpu(x.cuda(), None, reduce=True, fold_into=None)
    assert torch.equal(result.cpu(), expected)


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or not (10, 0) <= torch.cuda.get_device_capability() <= (10, 3),
    reason="the batched-GEMM expert layout targets the sm_100 family",
)
def test_gpu_expert_loads_land_in_the_planned_layout_on_every_load():
    """Experts built on the GPU at whole-K-tile widths take a kernel that
    reads its own weight layout: every load through the weight loaders, the
    first and a reload alike, routes like the Triton kernel on the
    checkpoint layout."""
    from tokenspeed_kernel.ops.moe.block_scaled import block_scaled_moe_routes_plan

    hidden, intermediate = 512, 256
    generator = torch.Generator(device="cuda").manual_seed(3)

    def codes(*shape):
        return torch.randint(0, 256, shape, generator=generator, device="cuda").to(
            torch.uint8
        )

    def scales(*shape):
        values = torch.rand(shape, generator=generator, device="cuda") * 0.5 + 0.25
        return values.to(torch.float8_e4m3fn)

    checkpoint = {
        "gate.weight": torch.randn(
            4, hidden, generator=generator, device="cuda"
        ).bfloat16(),
        "gate.bias": torch.zeros(4, device="cuda"),
        "experts.w13_weight": codes(4, 2 * intermediate, hidden // 2),
        "experts.w13_scale": scales(4, 2 * intermediate, hidden // 16),
        "experts.w13_amax": torch.rand(4, generator=generator, device="cuda") + 0.5,
        "experts.w13_outer_scale_denominator": torch.tensor([2688.0], device="cuda"),
        "experts.w2_weight": codes(4, hidden, intermediate // 2),
        "experts.w2_scale": scales(4, hidden, intermediate // 16),
        "experts.w2_amax": torch.rand(4, generator=generator, device="cuda") + 0.5,
        "experts.w2_outer_scale_denominator": torch.tensor([2688.0], device="cuda"),
    }
    with torch.device("cuda"):
        planned = _layer(0, 1, 1, hidden=hidden, intermediate=intermediate)
        reference = _layer(0, 1, 1, hidden=hidden, intermediate=intermediate)
    reference.experts.plan = block_scaled_moe_routes_plan(
        formats=reference.experts.formats,
        hidden=hidden,
        intermediate=intermediate,
        device=torch.device("cuda"),
        solution="triton",
    )

    def load(layer):
        for name, param in layer.named_parameters():
            param.weight_loader(param, checkpoint[name])

    load(planned)
    load(reference)
    x = torch.randn(9, hidden, generator=generator, device="cuda").bfloat16()
    expected = reference(x, None, reduce=True, fold_into=None)
    assert expected.abs().amax() > 0
    assert torch.equal(planned(x, None, reduce=True, fold_into=None), expected)
    load(planned)
    assert torch.equal(planned(x, None, reduce=True, fold_into=None), expected)

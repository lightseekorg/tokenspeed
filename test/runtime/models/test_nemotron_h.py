from __future__ import annotations

from types import SimpleNamespace

import torch

from tokenspeed.runtime.configs.nemotron_h_config import NemotronHConfig
from tokenspeed.runtime.layers.moe.core.types import MoELayerSpec
from tokenspeed.runtime.layers.moe.topk import BypassedTopKOutput, TopKConfig
from tokenspeed.runtime.layers.paged_attention import PagedAttention
from tokenspeed.runtime.models.nemotron_h import NemotronHForCausalLM


def _fake_model(*, ep_rank: int = 0, ep_size: int = 1):
    model = object.__new__(NemotronHForCausalLM)
    model.config = NemotronHConfig(
        layers_block_type=["moe"],
        n_routed_experts=4,
    )
    model.mapping = SimpleNamespace(
        moe=SimpleNamespace(ep_rank=ep_rank, ep_size=ep_size)
    )
    return model


def _recording_param(calls):
    param = torch.nn.Parameter(torch.empty(1))

    def weight_loader(param, loaded_weight, *, shard_id, local_expert_id):
        calls.append(
            {
                "param": param,
                "loaded_weight": loaded_weight,
                "shard_id": shard_id,
                "local_expert_id": local_expert_id,
            }
        )

    param.weight_loader = weight_loader
    return param


def test_nemotron_h_expert_up_proj_loads_as_non_gated_w13():
    model = _fake_model(ep_rank=1, ep_size=2)
    calls = []
    param = _recording_param(calls)
    params_dict = {"model.layers.0.mixer.experts.w13_weight": param}
    loaded_weight = torch.tensor([1.0])

    handled = model._try_load_expert_weight(
        "model.layers.0.mixer.experts.2.up_proj.weight",
        loaded_weight,
        params_dict,
    )

    assert handled
    assert len(calls) == 1
    assert calls[0]["param"] is param
    assert calls[0]["loaded_weight"] is loaded_weight
    assert calls[0]["shard_id"] == "w13"
    assert calls[0]["local_expert_id"] == 0


def test_nemotron_h_expert_down_proj_loads_as_w2():
    model = _fake_model()
    calls = []
    param = _recording_param(calls)
    params_dict = {"model.layers.0.mixer.experts.w2_weight": param}
    loaded_weight = torch.tensor([1.0])

    handled = model._try_load_expert_weight(
        "model.layers.0.mixer.experts.3.down_proj.weight",
        loaded_weight,
        params_dict,
    )

    assert handled
    assert calls[0]["shard_id"] == "w2"
    assert calls[0]["local_expert_id"] == 3


def test_nemotron_h_expert_gate_proj_is_skipped():
    model = _fake_model()
    calls = []
    params_dict = {"model.layers.0.mixer.experts.w13_weight": _recording_param(calls)}

    handled = model._try_load_expert_weight(
        "model.layers.0.mixer.experts.0.gate_proj.weight",
        torch.tensor([1.0]),
        params_dict,
    )

    assert handled
    assert calls == []


def test_nemotron_h_loads_modelopt_kv_scales_onto_attention_module():
    model = _fake_model()
    attn = PagedAttention(
        num_heads=1,
        head_dim=8,
        scaling=8**-0.5,
        num_kv_heads=1,
        layer_id=0,
    )
    modules_dict = {"model.layers.0.mixer.attn": attn}

    assert model._try_load_kv_scale(
        "model.layers.0.mixer.attn.k_scale",
        torch.tensor(0.5),
        modules_dict,
    )
    assert model._try_load_kv_scale(
        "model.layers.0.mixer.attn.v_scale",
        torch.tensor(0.25),
        modules_dict,
    )

    assert attn.k_scale == 0.5
    assert attn.v_scale == 0.25
    assert attn.k_scale_float == 0.5
    assert attn.v_scale_float == 0.25


def test_nemotron_h_nvfp4_moe_routes_bias_in_hidden_state_dtype(monkeypatch):
    from tokenspeed.runtime.layers.moe.backends.nvfp4 import flashinfer_trtllm
    from tokenspeed.runtime.layers.moe.backends.nvfp4.flashinfer_trtllm import (
        Nvfp4FlashinferTrtllmBackend,
    )

    captured = {}

    def fake_fp4_quantize(x, *args, **kwargs):
        del args, kwargs
        return (
            torch.empty((x.shape[0], x.shape[1] // 2), dtype=torch.uint8),
            torch.empty((x.shape[0], x.shape[1] // 16), dtype=torch.uint8),
        )

    def fake_moe_fused(**kwargs):
        captured["routing_logits_dtype"] = kwargs["routing_logits"].dtype
        captured["routing_bias_dtype"] = kwargs["routing_bias"].dtype
        captured["per_token_scale"] = kwargs["per_token_scale"]
        captured["routed_scaling_factor"] = kwargs["routed_scaling_factor"]
        captured["output_shape"] = kwargs["output"].shape
        captured["output_dtype"] = kwargs["output"].dtype
        return (kwargs["output"],)

    monkeypatch.setattr(
        flashinfer_trtllm, "trtllm_fp4_block_scale_moe_raw", fake_moe_fused
    )
    monkeypatch.setattr(
        "tokenspeed_kernel.ops.quantization.flashinfer.fp4_quantize",
        fake_fp4_quantize,
    )

    correction_bias = torch.ones(4, dtype=torch.float32)
    backend = Nvfp4FlashinferTrtllmBackend(
        "test",
        MoELayerSpec(
            top_k=2,
            num_experts=4,
            num_local_experts=4,
            hidden_size=32,
            intermediate_size=16,
            activation="relu2",
            tp_rank=0,
            tp_size=1,
            ep_rank=0,
            ep_size=1,
        ),
        SimpleNamespace(group_size=16),
        {"correction_bias": correction_bias},
    )
    layer = SimpleNamespace(
        hidden_size_unpadded=32,
        hidden_size_padded=32,
        intermediate_size_per_partition=16,
        w13_input_scale_quant=torch.tensor(1.0),
        gemm1_weights_fp4_shuffled=torch.nn.Parameter(
            torch.empty((4, 16, 16), dtype=torch.uint8),
            requires_grad=False,
        ),
        gemm1_scales_fp4_shuffled=torch.nn.Parameter(
            torch.empty((4, 16, 2), dtype=torch.uint8),
            requires_grad=False,
        ),
        gemm2_weights_fp4_shuffled=torch.nn.Parameter(
            torch.empty((4, 32, 8), dtype=torch.uint8),
            requires_grad=False,
        ),
        gemm2_scales_fp4_shuffled=torch.nn.Parameter(
            torch.empty((4, 32, 1), dtype=torch.uint8),
            requires_grad=False,
        ),
        g1_scale_c=torch.nn.Parameter(torch.empty((4, 16)), requires_grad=False),
        g1_alphas=torch.nn.Parameter(torch.empty((4, 16)), requires_grad=False),
        g2_alphas=torch.nn.Parameter(torch.empty((4, 32)), requires_grad=False),
    )
    hidden_states = torch.empty((3, 32), dtype=torch.bfloat16)
    topk_output = BypassedTopKOutput(
        hidden_states=hidden_states,
        router_logits=torch.empty((3, 4), dtype=torch.float32),
        topk_config=TopKConfig(top_k=2, correction_bias=correction_bias),
    )

    backend.forward(layer, hidden_states, topk_output, 3, 3)

    assert captured["routing_logits_dtype"] == torch.float32
    assert captured["routing_bias_dtype"] == torch.bfloat16
    assert captured["per_token_scale"] is None
    assert captured["routed_scaling_factor"] is None
    assert captured["output_shape"] == (3, 32)
    assert captured["output_dtype"] == torch.bfloat16


def test_nemotron_h_nvfp4_relu2_g1_scale_c_is_contiguous(monkeypatch):
    from tokenspeed.runtime.layers.moe.backends.nvfp4 import flashinfer_trtllm
    from tokenspeed.runtime.layers.moe.backends.nvfp4.flashinfer_trtllm import (
        Nvfp4FlashinferTrtllmBackend,
    )

    def row_permute(_cache, tensor, _epilogue_tile_m, **_kwargs):
        return torch.arange(tensor.shape[0], dtype=torch.long)

    monkeypatch.setattr(flashinfer_trtllm, "_get_w13_permute_indices", row_permute)
    monkeypatch.setattr(
        flashinfer_trtllm, "get_w2_permute_indices_with_cache", row_permute
    )
    monkeypatch.setattr(
        flashinfer_trtllm,
        "nvfp4_block_scale_interleave",
        lambda tensor: tensor.contiguous(),
    )

    backend = Nvfp4FlashinferTrtllmBackend(
        "test",
        MoELayerSpec(
            top_k=2,
            num_experts=2,
            num_local_experts=2,
            hidden_size=256,
            intermediate_size=16,
            activation="relu2",
            tp_rank=0,
            tp_size=1,
            ep_rank=0,
            ep_size=1,
        ),
        SimpleNamespace(group_size=16),
    )
    layer = SimpleNamespace(
        w13_weight=torch.nn.Parameter(
            torch.empty((2, 16, 128), dtype=torch.uint8),
            requires_grad=False,
        ),
        w13_weight_scale=torch.nn.Parameter(
            torch.empty((2, 16, 16), dtype=torch.uint8),
            requires_grad=False,
        ),
        w2_weight=torch.nn.Parameter(
            torch.empty((2, 256, 8), dtype=torch.uint8),
            requires_grad=False,
        ),
        w2_weight_scale=torch.nn.Parameter(
            torch.empty((2, 256, 1), dtype=torch.uint8),
            requires_grad=False,
        ),
        w13_weight_scale_2=torch.nn.Parameter(torch.ones(2, dtype=torch.float32)),
        w2_weight_scale_2=torch.nn.Parameter(torch.full((2,), 3.0)),
        w13_input_scale=torch.nn.Parameter(torch.full((2,), 2.0)),
        w2_input_scale=torch.nn.Parameter(torch.full((2,), 4.0)),
    )

    backend.process_weights_after_loading(layer)

    assert layer.g1_scale_c.is_contiguous()
    assert layer.g1_scale_c.stride() == (1,)
    assert torch.equal(layer.g1_scale_c, torch.full((2,), 0.25))

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

"""CPU capture/schema checks; the graph test requires an explicitly free GPU.

DEEPSEEK_V41_REFERENCE_DIR enables JSON-only local checkpoint coverage.
"""

import json
import os
from pathlib import Path
from test.runtime.test_deepseek_v41_engram import _mapping
from test.runtime.test_deepseek_v41_model import (
    _checkpoint,
    _config,
    _ctx,
    _loader_model,
    _quant,
)
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from tokenspeed.runtime.configs.model_config import ModelConfig
from tokenspeed.runtime.execution.drafter import get_drafter_impl
from tokenspeed.runtime.execution.drafter.deepseek_v4_dspark import DeepseekV4DSpark
from tokenspeed.runtime.execution.forward_batch_info import (
    CaptureHiddenMode,
    ForwardMode,
)
from tokenspeed.runtime.layers.attention import registry as attention_registry
from tokenspeed.runtime.layers.linear import LinearBase
from tokenspeed.runtime.layers.moe.expert import MoELayer
from tokenspeed.runtime.models.deepseek_v41 import (
    DeepseekV41ForCausalLM,
    DeepseekV41Model,
)
from tokenspeed.runtime.models.deepseek_v41_dspark import (
    DeepseekV41ForCausalLMDSpark,
    _quantized_kv,
    _WindowAttention,
)
from tokenspeed.runtime.utils.env import global_server_args_dict
from tokenspeed.runtime.utils.hf_transformers_utils import get_config


@pytest.mark.parametrize(
    "capture_mode",
    [CaptureHiddenMode.FULL, CaptureHiddenMode.LAST, CaptureHiddenMode.NULL, None],
)
def test_target_capture_is_mean_layer_input_after_engram(capture_mode):
    embeddings = torch.tensor([[1.0, 2.0], [11.0, 12.0]])
    streams = torch.tensor([0.0, 2.0, 6.0, 12.0]).view(1, 4, 1)
    events = []

    class Engram(nn.Module):
        def __init__(self, layer_id):
            super().__init__()
            self.layer_id = layer_id
            self.layer_hash_index = layer_id - 37

        def forward(self, hidden, hashes, mask):
            events.append(("engram", self.layer_id))
            assert hashes.tolist() == [self.layer_hash_index] * 2
            assert mask.tolist() == [True, True]
            return hidden + 100 * (self.layer_hash_index + 1)

    class Layer(nn.Module):
        def __init__(self, layer_id):
            super().__init__()
            self.layer_id = layer_id
            self.engram = Engram(layer_id) if layer_id >= 37 else None

        def forward(self, hidden, pre_mix, positions, input_ids, ctx):
            events.append(("layer", self.layer_id))
            # Distinct HC streams and a non-mean final mix catch weighted/output taps.
            return embeddings[:, None, :] + streams + 10 * self.layer_id, pre_mix

    target = DeepseekV41ForCausalLM.__new__(DeepseekV41ForCausalLM)
    nn.Module.__init__(target)
    target.model = DeepseekV41Model.__new__(DeepseekV41Model)
    nn.Module.__init__(target.model)
    target.model.config = SimpleNamespace(
        num_hidden_layers=40, hidden_size=2, hc_mult=4, engram_layer_ids=[37, 38, 39]
    )
    target.model.layers = nn.ModuleList(Layer(i) for i in range(40))
    target.model.engram_hash = Mock(return_value=torch.tensor([[0, 1, 2], [0, 1, 2]]))
    target.model.norm = SimpleNamespace(weight=torch.ones(2), variance_epsilon=1e-6)
    target.set_dspark_layers_to_capture([37, 38, 39])
    assert target.model.dspark_capture_layers == (37, 38, 39)
    assert target.capture_aux_hidden_states
    ctx = _ctx(None, 2, ForwardMode.EXTEND)
    ctx.capture_hidden_mode = capture_mode
    _, captures = target.model(
        torch.tensor([1, 2]),
        torch.tensor([0, 1]),
        ctx,
        input_embeds=embeddings,
        pp_inbound=None,
        engram_previous_tokens=torch.full((2, 3), -1, dtype=torch.int64),
        engram_token_mask=torch.ones(2, dtype=torch.bool),
    )
    assert events == [("layer", i) for i in range(37)] + [
        ("engram", 37),
        ("layer", 37),
        ("engram", 38),
        ("layer", 38),
        ("engram", 39),
        ("layer", 39),
    ]
    if capture_mode in (None, CaptureHiddenMode.NULL):
        assert captures is None
    else:
        # Layer 36/37/38 outputs + Engram 100/200/300 + unweighted stream mean 5.
        expected = torch.cat(
            [embeddings + 465, embeddings + 575, embeddings + 685], dim=-1
        )
        assert len(captures) == 3  # No extra final HC/pre-norm capture.
        assert all(capture.shape == (2, 2) for capture in captures)
        torch.testing.assert_close(
            torch.cat(captures, dim=-1), expected, rtol=0, atol=0
        )


@pytest.mark.parametrize(
    "layer_ids", [[], [38, 37, 39], [37, 37, 39], [-1, 37, 39], [37, 38, 40]]
)
def test_target_capture_rejects_invalid_taps(layer_ids):
    target = DeepseekV41ForCausalLM.__new__(DeepseekV41ForCausalLM)
    nn.Module.__init__(target)
    target.model = SimpleNamespace(config=SimpleNamespace(num_hidden_layers=40))
    target.set_dspark_layers_to_capture([37, 38, 39])
    with pytest.raises(ValueError, match="ordered target layer IDs"):
        target.set_dspark_layers_to_capture(layer_ids)
    assert target.model.dspark_capture_layers == (37, 38, 39)


def _draft_config():
    config = _config()
    config.dspark_num_stages = config.num_nextn_predict_layers = 3
    config.dspark_block_size = 5
    config.dspark_target_layer_ids = [37, 38, 39]
    config.dspark_n_routed_experts = 4
    config.dspark_num_experts_per_tok = 2
    config.dspark_markov_rank = 32
    config.dspark_noise_token_id = 7
    config.sliding_window = 128
    return config


def _draft_checkpoint(model):
    config = model.config
    raw = _checkpoint(config)
    weights = {
        name.replace("layers.", "mtp.", 1): tensor
        for name, tensor in raw.items()
        if name.startswith("layers.")
    }
    h = config.hidden_size
    weights["mtp.0.main_proj.weight"] = torch.full(
        (h, h * 3), 0.125, dtype=torch.float8_e4m3fn
    )
    weights["mtp.0.main_proj.scale"] = torch.full(
        (h // 32, h * 3 // 32), 121, dtype=torch.uint8
    ).view(torch.float8_e8m0fnu)
    weights["mtp.0.main_norm.weight"] = torch.ones(h, dtype=torch.bfloat16)
    weights["mtp.2.norm.weight"] = torch.ones(h, dtype=torch.bfloat16)
    for name in ("embed", "head"):
        weights[f"mtp.2.markov_head.{name}.weight"] = torch.full(
            (config.vocab_size, config.dspark_markov_rank), 0.125, dtype=torch.bfloat16
        )
    weights["mtp.2.confidence_head.proj.weight"] = torch.ones(
        (1, h + config.dspark_markov_rank), dtype=torch.bfloat16
    )
    return weights


@pytest.mark.parametrize("rank", [0, 1, 2, 3])
def test_draft_checkpoint_strict_shards(monkeypatch, rank):
    config = _draft_config()
    _loader_model(monkeypatch, config, rank, "cpu")
    model = DeepseekV41ForCausalLMDSpark(
        SimpleNamespace(text_config=config), _mapping(rank, 4, 4), _quant(), "", False
    )
    weights = _draft_checkpoint(model.model)
    model.load_weights(reversed(list(weights.items())))
    assert model.checkpoint_load_report["loaded"] > 0
    assert config.n_shared_experts == model.model.config.n_shared_experts == 1
    assert all(layer.ffn.shared_experts is not None for layer in model.model.layers)
    assert all(
        f"mtp.{stage}.ffn.shared_experts.{shard}.{suffix}" in weights
        for stage in range(3)
        for shard in ("w1", "w2", "w3")
        for suffix in ("weight", "scale")
    )
    assert not any("hc_head" in name for name, _ in model.named_parameters())
    assert get_drafter_impl("DSPARK", model) is DeepseekV4DSpark
    location = model.get_model_config_for_expert_location(model.config)
    assert (location.num_layers, location.num_logical_experts) == (3, 4)
    with pytest.raises(ValueError, match="Missing"):
        model.load_weights(
            (n, w) for n, w in weights.items() if n != "mtp.0.main_norm.weight"
        )
    with pytest.raises(ValueError, match="Missing"):
        model.load_weights(
            (n, w)
            for n, w in weights.items()
            if n != "mtp.1.ffn.shared_experts.w2.scale"
        )
    with pytest.raises(ValueError, match="Duplicate"):
        model.load_weights(
            [
                *weights.items(),
                ("mtp.0.main_norm.weight", weights["mtp.0.main_norm.weight"]),
            ]
        )
    with pytest.raises(ValueError, match="Unexpected"):
        model.load_weights([("mtp.3.norm.weight", torch.ones(config.hidden_size))])


def test_window_attention_matches_dense_reference():
    torch.manual_seed(41)
    batch, block, heads, dim, window = 2, 5, 2, 64, 8
    q = torch.randn(batch * block, heads, dim, dtype=torch.bfloat16)
    current = torch.randn(batch * block, dim, dtype=torch.bfloat16)
    cache = torch.randn(4, window, dim, dtype=torch.bfloat16)
    slots, starts = torch.tensor([2, 1]), torch.tensor([3, 12])
    positions = (starts[:, None] + 1 + torch.arange(block)).flatten()
    sink = torch.tensor([0.1, -0.3])
    backend = _WindowAttention(positions, cache, slots, starts, block)
    actual = backend.forward_v41(
        q,
        current,
        layer_id=0,
        positions=positions,
        request_indices=backend.meta.request_indices,
        forward_mode=ForwardMode.DECODE,
        index_q=None,
        index_weights=None,
        attn_sink=sink,
        softmax_scale=dim**-0.5,
        index_process_group=None,
    )
    decoded = _quantized_kv(current).reshape(batch, block, dim)
    expected = torch.empty_like(q).reshape(batch, block, heads, dim)
    for b in range(batch):
        kv = torch.cat(
            (cache[slots[b], : min(window, int(starts[b]) + 1)], decoded[b])
        ).float()
        for j in range(block):
            for h in range(heads):
                scores = kv @ q[b * block + j, h].float() * dim**-0.5
                probs = torch.cat((scores, sink[h : h + 1])).softmax(0)[:-1]
                expected[b, j, h] = probs @ kv
    torch.testing.assert_close(
        actual, expected.reshape_as(actual), rtol=0.01, atol=0.01
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_draft_forward_graph_and_context_seeding(monkeypatch):
    config = _draft_config()
    config.hidden_size = 256
    config.moe_intermediate_size = 256
    config.head_dim = 512
    config.qk_rope_head_dim = 64
    config.q_lora_rank = 128
    config.num_attention_heads = config.o_groups = 2
    monkeypatch.setitem(global_server_args_dict, "ep_num_redundant_experts", 0)
    with torch.device("cuda:0"):
        adapter = DeepseekV41ForCausalLMDSpark(
            SimpleNamespace(text_config=config), _mapping(0, 1, 1), _quant(), "", False
        )
    adapter.load_weights(_draft_checkpoint(adapter.model).items())
    with torch.no_grad():
        adapter.model.embed_tokens.weight.fill_(0.1)
        adapter.lm_head.weight.fill_(0.1)
    adapter.set_embed_and_head(
        adapter.model.embed_tokens.weight, adapter.lm_head.weight
    )
    model = adapter.model
    for module in model.modules():
        if isinstance(module, LinearBase):
            module.quant_method.process_weights_after_loading(module)
        elif isinstance(module, MoELayer):
            module.process_weights_after_loading(module)
    windows = torch.zeros(3, 3, 128, 512, dtype=torch.bfloat16, device="cuda:0")
    hidden = torch.randn(
        2, 4, 3 * config.hidden_size, dtype=torch.bfloat16, device="cuda:0"
    )
    slots = torch.tensor([1, 2], device="cuda:0")
    positions = torch.arange(4, device="cuda:0").expand(2, -1)
    model.write_context_windows_batched(
        hidden,
        positions,
        slots,
        torch.ones_like(positions, dtype=torch.bool),
        windows,
        0,
    )
    assert windows[0].count_nonzero() == 0
    assert windows[1:, :, :4].count_nonzero() > 0
    main = hidden[:, -1].contiguous()
    bonus = torch.tensor([3, 4], device="cuda:0")
    starts = torch.tensor([4, 4], device="cuda:0")
    ctx = _ctx(None, 10, ForwardMode.DECODE)
    ctx.bs, ctx.num_extends = 2, 0

    def forward():
        return model.forward_backbone(main, bonus, starts, windows, slots, ctx)

    with torch.inference_mode():
        for _ in range(3):
            forward()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = forward()
        for position in (5, 127, 128, 129):
            starts.fill_(position)
            before = windows.clone()
            expected = forward()
            windows.copy_(before)
            graph.replay()
            torch.testing.assert_close(captured, expected, rtol=0, atol=0)
            assert torch.isfinite(captured).all()
        graph.reset()


@pytest.mark.parametrize("checkpoint_source", ["temporary", "reference"])
def test_checkpoint_model_config_and_no_draft_paged_attention(
    monkeypatch, tmp_path, checkpoint_source
):
    if checkpoint_source == "reference":
        root = os.environ.get("DEEPSEEK_V41_REFERENCE_DIR")
        if root is None:
            pytest.skip("set DEEPSEEK_V41_REFERENCE_DIR for JSON-only integration")
        assert Path(root).is_dir()
    else:
        root = str(tmp_path)
        text = vars(_draft_config()).copy()
        # Stage count must come from the index, not a pre-populated config field.
        del text["dspark_num_stages"]
        text.update(model_type="deepseek_v41_text", max_position_embeddings=512)
        (tmp_path / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "deepseek_v41",
                    "architectures": ["DeepseekV41ForCausalLM"],
                    "dtype": "bfloat16",
                    "text_config": text,
                }
            ),
            encoding="utf-8",
        )
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "weight_map": {
                        f"mtp.{stage}.ffn.shared_experts.w1.weight": "not-loaded.safetensors"
                        for stage in range(3)
                    }
                }
            ),
            encoding="utf-8",
        )
    args = SimpleNamespace(
        mapping=_mapping(0, 4, 4),
        speculative_algorithm="DSPARK",
        speculative_num_steps=None,
        speculative_num_draft_tokens=None,
        _speculative_widths_explicit=False,
        attention_backend=None,
        drafter_attention_backend=None,
        prefix_granularity=64,
        load_format="auto",
        disaggregation_mode="null",
    )
    configs = [
        ModelConfig(
            model_path=root,
            trust_remote_code=False,
            revision=None,
            context_length=512,
            model_override_args="{}",
            dtype="bfloat16",
            quantization=None,
            override_config_file=None,
            is_draft_worker=is_draft,
            server_args=args,
        )
        for is_draft in (False, True)
    ]
    target, draft = configs
    assert target.hf_config.architectures == ["DeepseekV41ForCausalLM"]
    assert draft.hf_config.architectures == ["DeepseekV41ForCausalLMDSpark"]
    assert draft.hf_text_config.dspark_target_layer_ids == [37, 38, 39]
    assert draft.num_attention_layers == draft.hf_text_config.dspark_num_stages == 3
    assert draft.hf_text_config.num_nextn_predict_layers == 3
    assert draft.spec_block_size == args.speculative_num_steps == 5
    assert args.speculative_num_draft_tokens == 6
    assert draft.dspark_prefix_replay_tokens == 128
    assert (
        target.hf_text_config.n_shared_experts
        == draft.hf_text_config.n_shared_experts
        == 1
    )
    side = attention_registry._resolve_attn_side(draft, None)
    assert side.is_dspark and not side.is_deepseek_v4

    # Stop at the allocation boundary, after the real registry chooses both sides.
    config_builder = Mock(return_value=SimpleNamespace(component=lambda cls: None))
    monkeypatch.setattr(attention_registry, "_create_attn_config", config_builder)
    monkeypatch.setattr(
        attention_registry, "_resolve_cache_family", Mock(return_value="deepseek_v41")
    )
    monkeypatch.setattr(
        attention_registry, "_resolve_full_attn_backend_name", Mock(return_value=None)
    )
    monkeypatch.setattr(
        attention_registry,
        "profile_available_cache_memory_bytes",
        Mock(side_effect=RuntimeError("CPU test allocation boundary")),
    )
    args.gpu_memory_utilization = 0.9
    with pytest.raises(RuntimeError, match="CPU test allocation boundary"):
        attention_registry.create_attn_components(
            server_args=args,
            model_config=target,
            gpu_id=0,
            rank=0,
            gpu_memory=0,
            enable_memory_saver=False,
            draft_model_config=draft,
            decode_input_tokens=6,
            overlap_schedule_depth=0,
        )
    config_builder.assert_called_once_with(args, target)


def test_reference_draft_architecture():
    root = os.environ.get("DEEPSEEK_V41_REFERENCE_DIR")
    if root is None:
        pytest.skip("set DEEPSEEK_V41_REFERENCE_DIR for checkpoint config coverage")
    config = get_config(
        root,
        trust_remote_code=False,
        revision=None,
        model_override_args=None,
        is_draft_worker=True,
        speculative_algorithm="DSPARK",
        local_files_only=True,
    )
    assert config.architectures == ["DeepseekV41ForCausalLMDSpark"]
    assert config.text_config.dspark_block_size == 5
    assert config.text_config.dspark_target_layer_ids == [37, 38, 39]
    index = json.loads((Path(root) / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    assert {int(n.split(".")[1]) for n in index if n.startswith("mtp.")} == {0, 1, 2}
    assert all(f"mtp.{i}.ffn.shared_experts.w1.weight" in index for i in range(3))

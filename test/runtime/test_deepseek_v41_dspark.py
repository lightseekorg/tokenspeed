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
    _Backend,
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
from tokenspeed_kernel.ops.attention import dsv41
from torch import nn

from tokenspeed.runtime.configs.model_config import ModelConfig
from tokenspeed.runtime.execution.drafter import get_drafter_impl
from tokenspeed.runtime.execution.drafter.deepseek_v41_dspark import (
    DeepseekV41DSpark,
)
from tokenspeed.runtime.execution.forward_batch_info import (
    CaptureHiddenMode,
    ForwardMode,
)
from tokenspeed.runtime.layers.attention import registry as attention_registry
from tokenspeed.runtime.layers.attention.backends.specific.deepseek_v41 import (
    V41DecoderView,
    V41PrefillSpan,
)
from tokenspeed.runtime.layers.linear import LinearBase
from tokenspeed.runtime.layers.moe.expert import MoELayer
from tokenspeed.runtime.models import deepseek_v41 as v41
from tokenspeed.runtime.models.deepseek_v41 import (
    DeepseekV41ForCausalLM,
    DeepseekV41Model,
)
from tokenspeed.runtime.models.deepseek_v41_dspark import (
    DeepseekV41ForCausalLMDSpark,
    _quantized_kv,
    _WindowAttention,
    _WindowSelection,
    _write_window_rows,
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

        def forward(self, hidden, pre_mix, positions, image_mask, ctx):
            events.append(("layer", self.layer_id))
            rows = v41._row_plan(self.layer_id, 20, ctx)
            assert rows.keep_rows is None and rows.source is rows.query
            # Distinct HC streams and a non-mean final mix catch weighted/output taps.
            return embeddings[:, None, :] + streams + 10 * self.layer_id, pre_mix

    target = DeepseekV41ForCausalLM.__new__(DeepseekV41ForCausalLM)
    nn.Module.__init__(target)
    target.model = DeepseekV41Model.__new__(DeepseekV41Model)
    nn.Module.__init__(target.model)
    target.model.config = SimpleNamespace(
        num_hidden_layers=40, hidden_size=2, hc_mult=4, engram_layer_ids=[37, 38, 39]
    )
    target.model.ced_decoder_start = 20
    target.model.decoder_uses_engram = True
    target.model.layers = nn.ModuleList(Layer(i) for i in range(40))
    target.model.engram_hash = Mock(return_value=torch.tensor([[0, 1, 2], [0, 1, 2]]))
    target.model.norm = SimpleNamespace(weight=torch.ones(2), variance_epsilon=1e-6)
    target.set_dspark_layers_to_capture([37, 38, 39])
    assert target.model.dspark_capture_layers == (37, 38, 39)
    assert target.capture_aux_hidden_states
    positions = torch.tensor([0, 1])
    backend = _Backend(positions, torch.zeros(2, dtype=torch.int64))
    ctx = _ctx(backend, 2, ForwardMode.EXTEND)
    ctx.capture_hidden_mode = capture_mode
    _, captures = target.model(
        torch.tensor([1, 2]),
        positions,
        ctx,
        input_embeds=embeddings,
        pp_inbound=None,
        engram_previous_tokens=torch.full((2, 3), -1, dtype=torch.int64),
        engram_token_mask=torch.ones(2, dtype=torch.bool),
        image_mask=None,
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
        SimpleNamespace(text_config=config), _mapping(rank, 4, 4), _quant()
    )
    weights = _draft_checkpoint(model.model)
    model.load_weights(reversed(list(weights.items())))
    assert model.checkpoint_load_report["loaded"] > 0
    # The context write's merged projection holds every stage's wkv shard,
    # loaded from the same checkpoint tensors as the stage's own wq_a_wkv.
    merged = model.model.context_wkv
    d, q_lora = config.head_dim, config.q_lora_rank
    for stage, layer in enumerate(model.model.layers):
        rows = slice(stage * d, (stage + 1) * d)
        torch.testing.assert_close(
            merged.weight[rows], layer.attn.wq_a_wkv.weight[q_lora:], rtol=0, atol=0
        )
        torch.testing.assert_close(
            merged.weight_scale_inv[rows],
            layer.attn.wq_a_wkv.weight_scale_inv[q_lora:],
            rtol=0,
            atol=0,
        )
    assert config.n_shared_experts == model.model.config.n_shared_experts == 1
    assert all(layer.ffn.shared_experts is not None for layer in model.model.layers)
    assert all(
        f"mtp.{stage}.ffn.shared_experts.{shard}.{suffix}" in weights
        for stage in range(3)
        for shard in ("w1", "w2", "w3")
        for suffix in ("weight", "scale")
    )
    assert not any("hc_head" in name for name, _ in model.named_parameters())
    assert get_drafter_impl("DSPARK", model) is DeepseekV41DSpark
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


def _paged_slots(positions, rows_per_page):
    """Map absolute positions onto pages 1.. of a fake window field; -1 stays."""
    slots = rows_per_page + positions
    return torch.where(positions < 0, torch.full_like(slots, -1), slots)


def _history_slots(starts, window, rows_per_page):
    wanted = starts[:, None] - (window - 1) + torch.arange(window, device=starts.device)
    wanted = wanted.masked_fill(wanted < 0, -1)
    return _paged_slots(wanted, rows_per_page).to(torch.int32)


def _window_backend(positions, cache, history, block, rows_per_page):
    """One stage's borrowed window over the block geometry ``dspark_block`` derives."""
    n = history.shape[0]
    anchors = torch.zeros(n, dtype=torch.int64, device=history.device)
    _, _, _, request_indices, page, row, indices = dsv41.dspark_block(
        anchors, anchors, history, 0, rows_per_page, 1, block
    )
    return _WindowAttention(
        positions,
        cache,
        history,
        block,
        _WindowSelection(page, row, indices),
        request_indices,
    )


def test_window_attention_matches_dense_reference():
    torch.manual_seed(41)
    batch, block, heads, dim, window, rows = 2, 5, 2, 64, 8, 4
    q = torch.randn(batch * block, heads, dim, dtype=torch.bfloat16)
    current = torch.randn(batch * block, dim, dtype=torch.bfloat16)
    cache = torch.randn(6, rows, dim, dtype=torch.bfloat16)
    starts = torch.tensor([3, 12])
    history = _history_slots(starts, window, rows)
    positions = (starts[:, None] + 1 + torch.arange(block)).flatten()
    sink = torch.tensor([0.1, -0.3])
    backend = _window_backend(positions, cache, history, block, rows)
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
        swa_rope_cache=None,
    )
    decoded = _quantized_kv(current).reshape(batch, block, dim)
    expected = torch.empty_like(q).reshape(batch, block, heads, dim)
    for b in range(batch):
        live = history[b][history[b] >= 0].long()
        assert live.numel() == min(window, int(starts[b]) + 1)
        kv = torch.cat((cache[live // rows, live % rows], decoded[b])).float()
        for j in range(block):
            for h in range(heads):
                scores = kv @ q[b * block + j, h].float() * dim**-0.5
                probs = torch.cat((scores, sink[h : h + 1])).softmax(0)[:-1]
                expected[b, j, h] = probs @ kv
    torch.testing.assert_close(
        actual, expected.reshape_as(actual), rtol=0.01, atol=0.01
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_window_attention_kernel_matches_reference():
    """The workspace kernel path reproduces the fp32 reference on real shapes.

    Rows cover an empty, a partial and a full 128-row history plus one graph
    padding row (all -1 slots); q arrives 64-head padded with a -inf padded
    sink exactly as ``DeepseekV41Attention.forward`` hands it over, and the
    window field is strided like the LCM arena so the gather cannot rely on a
    flat view.
    """
    from tokenspeed_kernel.ops.attention.dsv41.flash_mla import (
        is_flash_mla_v41_available,
    )
    from tokenspeed_kernel.platform import current_platform
    from tokenspeed_kernel.registry import KernelRegistry
    from tokenspeed_kernel.selection import select_kernel
    from tokenspeed_kernel.signature import dense_tensor_format, format_signature

    from tokenspeed.runtime.models.deepseek_v41 import DeepseekV41RotaryEmbedding

    torch.manual_seed(7)
    device = torch.device("cuda:0")
    block, real_heads, padded_heads, dim, window, rows = 5, 2, 64, 512, 128, 64
    config = _config()
    config.head_dim, config.qk_rope_head_dim = dim, 64
    config.max_position_embeddings = 4096
    rotary = DeepseekV41RotaryEmbedding(config, 0)
    rotary._prepare_cache(device)
    starts = torch.tensor([0, 40, 300, -1], device=device)
    batch = starts.numel()
    history = _history_slots(starts.clamp_min(0), window, rows).to(device)
    history[3] = -1
    positions = starts.clamp_min(0)[:, None] + 1 + torch.arange(block, device=device)
    positions = positions.flatten()
    # 97-row strides: reshape(-1, dim) on this field would copy, not view.
    field = torch.randn(9, 97, dim, dtype=torch.bfloat16, device=device)[:, :rows]
    field_before = field.clone()
    q = torch.zeros(
        batch * block, padded_heads, dim, dtype=torch.bfloat16, device=device
    )
    q[:, :real_heads] = torch.randn(batch * block, real_heads, dim, device=device)
    swa = torch.randn(batch * block, dim, dtype=torch.bfloat16, device=device)
    sink = torch.full((padded_heads,), -float("inf"), device=device)
    sink[:real_heads] = torch.tensor([0.1, -0.3], device=device)
    backend = _window_backend(positions, field, history, block, rows)
    kwargs = dict(
        layer_id=0,
        positions=positions,
        request_indices=backend.meta.request_indices,
        forward_mode=ForwardMode.DECODE,
        index_q=None,
        index_weights=None,
        attn_sink=sink,
        softmax_scale=dim**-0.5,
        index_process_group=None,
        swa_rope_cache=rotary.cos_sin_cache,
    )
    actual = backend.forward_v41(q, swa, **kwargs)
    kv = backend._workspace(swa, positions, rotary.cos_sin_cache)
    expected = backend._reference_forward(q, kv, sink, dim**-0.5)
    # Both branches feed the kernel the very same quantized block rows.
    torch.testing.assert_close(
        backend._workspace(swa, positions, rotary.cos_sin_cache), kv, rtol=0, atol=0
    )
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(
        actual[:, :real_heads], expected[:, :real_heads], rtol=0.02, atol=0.004
    )
    assert torch.equal(field, field_before)
    # The same launch on the same bytes must be deterministic: graph replay
    # parity in test_draft_forward_graph_and_context_seeding relies on it.
    torch.testing.assert_close(
        backend.forward_v41(q, swa, **kwargs), actual, rtol=0, atol=0
    )
    selected = select_kernel(
        "attention",
        "dsv41_selected_attention",
        format_signature(x=dense_tensor_format(torch.bfloat16)),
        features=None,
        platform=None,
        traits={"flashmla_eligible": is_flash_mla_v41_available()},
        solution=None,
        override=None,
    )
    # Expect what the registry can serve here: FlashMLA inside its declared
    # arch window (sm90-sm103, not sm107) when importable, a specialized Gluon
    # kernel on the AMD parts that register one, the portable Triton kernel
    # everywhere else.
    registry, platform = KernelRegistry.get(), current_platform()

    def _registered_here(name: str) -> bool:
        spec = registry.get_by_name(name)
        return spec is not None and spec.capability.satisfied_by(platform)

    if is_flash_mla_v41_available() and _registered_here(
        "flashmla_dsv41_selected_attention"
    ):
        expected = "flashmla_dsv41_selected_attention"
    else:
        specialized = [
            name
            for name in (
                "gluon_dsv41_selected_attention_gfx950",
                "gluon_dsv41_selected_attention_gfx1250",
            )
            if _registered_here(name)
        ]
        expected = specialized[0] if specialized else "triton_dsv41_selected_attention"
    assert selected.name == expected


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
            SimpleNamespace(text_config=config), _mapping(0, 1, 1), _quant()
        )
    adapter.load_weights(_draft_checkpoint(adapter.model).items())
    # The target lends its BF16 embedding and head; the draft's own are placeholders.
    adapter.set_embed_and_head(
        *(
            nn.Parameter(torch.full_like(p, 0.1, dtype=torch.bfloat16))
            for p in (adapter.model.embed_tokens.weight, adapter.lm_head.weight)
        )
    )
    model = adapter.model
    for module in model.modules():
        if isinstance(module, LinearBase):
            module.quant_method.process_weights_after_loading(module)
        elif isinstance(module, MoELayer):
            module.process_weights_after_loading(module)
    # Three stage fields over eight 64-row pages; page 0 is the null page.
    windows = torch.zeros(3, 8, 64, 512, dtype=torch.bfloat16, device="cuda:0")
    pool = SimpleNamespace(dspark_kv=lambda stage: windows[stage])
    hidden = torch.randn(
        8, 3 * config.hidden_size, dtype=torch.bfloat16, device="cuda:0"
    )
    positions = torch.cat((torch.arange(4), torch.arange(130, 134))).to("cuda:0")
    slots = _paged_slots(positions, 64)
    model.write_context_kv(hidden, positions, slots, pool)
    assert windows[:, 0].count_nonzero() == 0
    written = windows[:, slots // 64, slots % 64]
    assert written.count_nonzero() > 0
    assert windows.count_nonzero() == written.count_nonzero()
    # The fused row kernel reproduces the eager reference chain (kv_norm,
    # RoPE, FP8 round trip, masked scatter) bit for bit, and the merged
    # projection agrees with each stage's own wkv columns.
    main_x = model._main_input(hidden)
    kv, _ = model.context_wkv(main_x, block_scale=None, output_dtype=None)
    reference = torch.zeros_like(windows)
    for stage, layer in enumerate(model.layers):
        stage_kv = kv[:, stage * 512 : (stage + 1) * 512]
        qkv, _ = layer.attn.wq_a_wkv(main_x, block_scale=None, output_dtype=None)
        torch.testing.assert_close(
            stage_kv, qkv[:, config.q_lora_rank :], rtol=0.02, atol=0.02
        )
        _write_window_rows(
            reference[stage], slots, model._main_kv(layer.attn, stage_kv, positions)
        )
    torch.testing.assert_close(windows, reference, rtol=0, atol=0)
    # A -1 slot is a padding or nonresident row: the null page stays zero.
    model.write_context_kv(
        hidden[:1], positions[:1], torch.tensor([-1], device="cuda:0"), pool
    )
    assert windows[:, 0].count_nonzero() == 0

    _assert_drafter_run_writes_rows_and_drafts(adapter, windows, pool)

    bonus = torch.tensor([3, 4], device="cuda:0")
    starts = torch.tensor([4, 4], device="cuda:0")
    history = torch.empty(2, 128, dtype=torch.int32, device="cuda:0")
    history.copy_(_history_slots(starts, 128, 64))
    ctx = _ctx(None, 10, ForwardMode.DECODE)
    ctx.bs, ctx.num_extends = 2, 0

    def forward():
        return model.forward_backbone(bonus, starts, history, pool, ctx)

    with torch.inference_mode():
        for _ in range(3):
            forward()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = forward()
        for position in (5, 127, 128, 129, 133):
            starts.fill_(position)
            history.copy_(_history_slots(starts, 128, 64))
            expected = forward()
            graph.replay()
            torch.testing.assert_close(captured, expected, rtol=0, atol=0)
            assert torch.isfinite(captured).all()
        graph.reset()


@torch.inference_mode()
def _assert_drafter_run_writes_rows_and_drafts(adapter, windows, pool):
    """Drive run() through a fake V4.1 backend: extend rows seed, decode rows draft."""
    from tokenspeed.runtime.execution.input_buffer import InputBuffers

    width, block = 6, 5
    ib = InputBuffers(3, 1024, 8, device="cuda:0")
    drafter = DeepseekV41DSpark(
        spec_num_tokens=width,
        spec_num_steps=block,
        draft_model_runner=SimpleNamespace(
            model=adapter, mapping=adapter.mapping, device="cuda:0"
        ),
        attn_backend=None,
        token_to_kv_pool=None,
        runtime_states=None,
        input_buffers=ib,
        vocab_size=adapter.model.config.vocab_size,
    )
    target = SimpleNamespace(
        set_dspark_layers_to_capture=Mock(),
        logits_processor=SimpleNamespace(tp_group=adapter.mapping.attn.tp_group),
    )
    drafter.wire_target(target)
    # One extend request of 5 tokens at positions 198..202 whose CED decoder
    # kept its last 3 rows (200..202), and one decode row whose verify window
    # covers positions 300..305 with 4 accepted tokens. The taps -- and the
    # rows the drafter writes -- follow the decoder view, not the raw extend.
    extend_positions = torch.arange(200, 203, device="cuda:0")
    decode_positions = torch.arange(300, 306, device="cuda:0")
    metas = {
        ForwardMode.EXTEND: SimpleNamespace(
            positions=torch.arange(198, 203, device="cuda:0"),
            swa_write_slots=_paged_slots(torch.arange(198, 203, device="cuda:0"), 64),
            request_indices=torch.zeros(5, dtype=torch.int64, device="cuda:0"),
        ),
        ForwardMode.DECODE: SimpleNamespace(
            positions=decode_positions,
            swa_write_slots=_paged_slots(decode_positions, 64),
            request_indices=torch.ones(6, dtype=torch.int64, device="cuda:0"),
        ),
    }
    kept = SimpleNamespace(
        positions=extend_positions,
        swa_write_slots=_paged_slots(extend_positions, 64),
        request_indices=torch.zeros(3, dtype=torch.int64, device="cuda:0"),
    )
    view_positions = torch.cat((extend_positions, decode_positions))
    view = V41DecoderView(
        SimpleNamespace(
            positions=view_positions,
            swa_write_slots=_paged_slots(view_positions, 64),
            request_indices=torch.cat(
                (kept.request_indices, metas[ForwardMode.DECODE].request_indices)
            ),
        ),
        kept,
        (V41PrefillSpan(0, 0, 200, 3, 200),),
        torch.tensor([2, 3, 4, 5, 6, 7, 8, 9, 10], device="cuda:0"),
        torch.tensor([2, 3, 4, 5, 6, 7, 8], device="cuda:0"),
    )
    seen = {"groups": set()}

    def window_slots(group_id, start_pos, request_indices):
        seen["groups"].add(group_id)
        seen["start_pos"] = start_pos.clone()
        seen["requests"] = request_indices.clone()
        return _history_slots(start_pos, 128, 64), None

    backend = SimpleNamespace(
        query_metadata=lambda mode: metas[mode],
        decoder_view=lambda: view,
        window_slots=window_slots,
    )
    ib.extend_seq_lens_cpu[:1] = 5
    ctx = _ctx(None, 3 + width, ForwardMode.MIXED)
    ctx.bs, ctx.num_extends = 2, 1
    ctx.attn_backend, ctx.token_to_kv_pool = backend, pool
    hidden = torch.randn(
        3 + width, drafter.hidden_width, dtype=torch.bfloat16, device="cuda:0"
    )
    output_tokens = torch.arange(1, 2 + width, device="cuda:0", dtype=torch.int32)
    accept_lengths = torch.tensor([1, 4], device="cuda:0", dtype=torch.int32)
    windows.zero_()
    next_tokens = drafter.run(
        base_ctx=ctx,
        logits_output=SimpleNamespace(hidden_states=hidden),
        output_tokens=output_tokens,
        accept_lengths=accept_lengths,
    )
    assert next_tokens.shape == (2, width)
    assert torch.isfinite(next_tokens.float()).all()
    # The extend row proposes nothing beyond its bonus; the decode row does.
    assert (next_tokens[0] == output_tokens[0]).all()
    assert next_tokens[1, 0] == output_tokens[1 + 3]
    assert seen["start_pos"].tolist() == [303]
    assert seen["requests"].tolist() == [1]
    assert seen["groups"] == {"v41.swa"}
    # Exactly the kept extend rows and the verify rows were written; the two
    # extend rows the decoder dropped (198, 199) were never captured.
    written = torch.cat((extend_positions, decode_positions))
    slots = _paged_slots(written, 64)
    assert windows[:, 0].count_nonzero() == 0
    assert (windows[:, slots // 64, slots % 64] != 0).any(dim=-1).all()
    assert (
        windows.count_nonzero() == windows[:, slots // 64, slots % 64].count_nonzero()
    )
    dropped = _paged_slots(torch.arange(198, 200, device="cuda:0"), 64)
    assert windows[:, dropped // 64, dropped % 64].count_nonzero() == 0


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
    assert target.is_multimodal and not draft.is_multimodal
    assert draft.hf_config.architectures == ["DeepseekV41ForCausalLMDSpark"]
    assert draft.hf_text_config.dspark_target_layer_ids == [37, 38, 39]
    assert draft.num_attention_layers == draft.hf_text_config.dspark_num_stages == 3
    assert draft.hf_text_config.num_nextn_predict_layers == 3
    assert draft.spec_block_size == args.speculative_num_steps == 5
    assert args.speculative_num_draft_tokens == 6
    assert draft.dspark_prefix_replay_tokens == 0
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

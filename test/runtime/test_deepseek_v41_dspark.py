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
import sys
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
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.drafter import get_drafter_impl
from tokenspeed.runtime.execution.drafter.deepseek_v41_dspark import (
    DeepseekV41DSpark,
)
from tokenspeed.runtime.execution.forward_batch_info import (
    CaptureHiddenMode,
    ForwardMode,
)
from tokenspeed.runtime.execution.output_layout import ForwardOutputLayout
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
    DeepseekV41DSparkModel,
    DeepseekV41ForCausalLMDSpark,
    _quantized_kv,
    _WindowAttention,
    _WindowSelection,
    _write_window_rows,
)
from tokenspeed.runtime.utils.env import global_server_args_dict
from tokenspeed.runtime.utils.hf_transformers_utils import get_config
from tokenspeed.runtime.utils.server_args import ServerArgs

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=30, suite="runtime-1gpu")


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

        def forward(self, hidden, hashes, mask, *, query_shard=None):
            assert query_shard is None
            events.append(("engram", self.layer_id))
            assert hashes.tolist() == [[self.layer_hash_index] * 3] * 2
            assert mask.tolist() == [True, True]
            return hidden + 100 * (self.layer_hash_index + 1)

    class Layer(nn.Module):
        def __init__(self, layer_id):
            super().__init__()
            self.layer_id = layer_id
            self.engram = Engram(layer_id) if layer_id >= 37 else None

        def forward(
            self,
            hidden,
            pre_mix,
            positions,
            image_mask,
            ctx,
            *,
            pending_post,
            allow_ffn_reduce_fusion,
            capture_input,
        ):
            events.append(("layer", self.layer_id))
            assert allow_ffn_reduce_fusion == (
                self.layer_id not in (19, 20) and self.layer_id < 36
            )
            rows = v41._row_plan(self.layer_id, 20, ctx)
            assert rows.keep_rows is None and rows.source is rows.query
            # Distinct HC streams and a non-mean final mix catch weighted/output taps.
            return (
                embeddings[:, None, :] + streams + 10 * self.layer_id,
                pre_mix,
                SimpleNamespace(finish=lambda residual: residual),
                hidden.mean(dim=1) if capture_input else None,
            )

    target = DeepseekV41ForCausalLM.__new__(DeepseekV41ForCausalLM)
    nn.Module.__init__(target)
    target.model = DeepseekV41Model.__new__(DeepseekV41Model)
    nn.Module.__init__(target.model)
    target.model.config = SimpleNamespace(
        num_hidden_layers=40,
        hidden_size=2,
        hc_mult=4,
        engram_layer_ids=[37, 38, 39],
        engram_max_ngram_size=4,
        engram_n_heads=1,
    )
    target.model.ced_decoder_start = 20
    target.model.decoder_uses_engram = True
    target.model.layers = nn.ModuleList(Layer(i) for i in range(40))
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
        engram_hash_ids=torch.arange(3).view(1, 3, 1).expand(2, 3, 3),
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_window_workspace_preserves_deferred_norm():
    device = "cuda"
    torch.manual_seed(418)
    config = _config()
    config.head_dim, config.qk_rope_head_dim = 512, 64
    config.max_position_embeddings = 512
    positions = torch.tensor([4, 5, 6, 7], device=device)
    history = torch.tensor([[64, 65, -1, -1]], device=device)
    cache = torch.randn(3, 64, 512, dtype=torch.bfloat16, device=device)
    backend = _window_backend(positions, cache, history, 4, 64)
    norm = v41.RMSNorm(512, eps=config.rms_norm_eps).to(
        device=device, dtype=torch.bfloat16
    )
    with torch.no_grad():
        norm.weight.uniform_(0.5, 1.5)
    rotary = v41.DeepseekV41RotaryEmbedding(config, 0)
    rotary._prepare_cache(torch.device(device))
    rope_cache = rotary.cos_sin_cache
    swa = torch.randn(4, 512, dtype=torch.bfloat16, device=device)
    original = swa.clone()
    expected = backend._workspace(
        v41._norm(swa, norm),
        positions,
        rope_cache,
        None,
        norm.variance_epsilon,
    )
    actual = backend._workspace(
        swa,
        positions,
        rope_cache,
        norm.weight,
        norm.variance_epsilon,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(swa, original, rtol=0, atol=0)


def test_window_workspace_refuses_deferred_norm_off_the_fused_path():
    # The model defers the SWA norm only to the fused 512-wide CUDA rows; any
    # other path must not drop it.
    history = torch.tensor([[-1, -1]])
    backend = _WindowAttention(
        torch.tensor([0, 1]),
        torch.zeros(1, 64, 64, dtype=torch.bfloat16),
        history,
        2,
        _WindowSelection(
            torch.zeros_like(history),
            torch.zeros_like(history),
            torch.full((2, 4), -1, dtype=torch.int32),
        ),
        torch.zeros(2, dtype=torch.int64),
    )
    with pytest.raises(ValueError, match="requires 512-wide CUDA rows"):
        backend._workspace(
            torch.randn(2, 64, dtype=torch.bfloat16),
            torch.tensor([0, 1]),
            None,
            torch.ones(64, dtype=torch.bfloat16),
            1e-6,
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
        swa_norm_weight=None,
        swa_norm_eps=0.0,
        fused_decode_groups=None,
        key_rows=None,
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
    sink = torch.full(
        (padded_heads,), -float("inf"), dtype=torch.float32, device=device
    )
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
        swa_norm_weight=None,
        swa_norm_eps=config.rms_norm_eps,
        fused_decode_groups=None,
        key_rows=None,
    )
    actual = backend.forward_v41(q, swa, **kwargs)
    kv = backend._workspace(
        swa, positions, rotary.cos_sin_cache, None, config.rms_norm_eps
    )
    current = dsv41.dspark_rows(
        swa,
        None,
        None,
        positions,
        rotary.cos_sin_cache,
        None,
        None,
        torch.empty_like(swa),
    )
    separate = torch.cat(
        (
            field[backend.selection.page, backend.selection.row],
            current.reshape(batch, block, dim),
        ),
        dim=1,
    )
    torch.testing.assert_close(kv, separate, rtol=0, atol=0)
    expected = backend._reference_forward(q, kv, sink, dim**-0.5)
    # Both branches feed the kernel the very same quantized block rows.
    torch.testing.assert_close(
        backend._workspace(
            swa, positions, rotary.cos_sin_cache, None, config.rms_norm_eps
        ),
        kv,
        rtol=0,
        atol=0,
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
    _assert_drafts_follow_their_rows(adapter, windows, pool)

    bonus = torch.tensor([3, 4], device="cuda:0")
    starts = torch.tensor([4, 4], device="cuda:0")
    history = torch.empty(2, 128, dtype=torch.int32, device="cuda:0")
    history.copy_(_history_slots(starts, 128, 64))
    ctx = _ctx(None, 10, ForwardMode.DECODE)
    ctx.bs, ctx.num_extends = 2, 0
    ctx.output_layout = ForwardOutputLayout(0, 0, 2, 5)

    confidence_features = torch.empty(
        2, model.block_size, config.hidden_size, dtype=torch.bfloat16, device="cuda:0"
    )

    def forward():
        return model.forward_backbone(
            bonus,
            starts,
            history,
            pool,
            ctx,
            confidence_hidden_out=confidence_features,
        )

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
            expected_features = confidence_features.clone()
            ordinary = model.forward_backbone(bonus, starts, history, pool, ctx)
            torch.testing.assert_close(ordinary, expected, rtol=0, atol=0)
            values = expected_features.float()
            normalized = (
                values
                * torch.rsqrt(
                    values.square().mean(-1, keepdim=True) + model.norm.variance_epsilon
                )
                * model.norm.weight.float()
            ).to(expected.dtype)
            torch.testing.assert_close(expected, normalized)
            graph.replay()
            torch.testing.assert_close(
                confidence_features, expected_features, rtol=0, atol=0
            )
            torch.testing.assert_close(captured, expected, rtol=0, atol=0)
            assert torch.isfinite(captured).all()
        graph.reset()
    with monkeypatch.context() as adaptive:
        adaptive.setitem(global_server_args_dict, "enable_dsv41_adaptive_dspark", True)
        _assert_adaptive_inputs_and_state(adapter)
        _assert_drafts_follow_their_rows(adapter, windows, pool)


@torch.inference_mode()
def _assert_drafts_follow_their_rows(adapter, windows, pool):
    """Permuting a decode batch permutes its drafts: no row reads another's state.

    Three decode requests with different bonus tokens, accept lengths and
    windows run as ``[A, B, C]`` and as ``[C, A, B]``; each request's block
    must come out the same in both orders. The block sampler once read the
    drafter's strided bonus column as contiguous, which gave every row after
    the first another row's anchor -- a single-request batch never sees that.
    """
    from tokenspeed.runtime.execution.input_buffer import InputBuffers
    from tokenspeed.runtime.execution.runtime_states import RuntimeStates

    width, block, n = 6, 5, 3
    ib = InputBuffers(n, 1024, 8, device="cuda:0")
    drafter = DeepseekV41DSpark(
        spec_num_tokens=width,
        spec_num_steps=block,
        draft_model_runner=SimpleNamespace(
            model=adapter,
            mapping=adapter.mapping,
            device="cuda:0",
            server_args=ServerArgs(model="unused", disaggregation_mode="null"),
        ),
        attn_backend=None,
        token_to_kv_pool=None,
        runtime_states=RuntimeStates(
            8,
            adapter.model.config.vocab_size,
            adapter.model.config.vocab_size,
            width,
            device="cuda:0",
        ),
        input_buffers=ib,
        vocab_size=adapter.model.config.vocab_size,
    )
    drafter.wire_target(
        SimpleNamespace(
            set_dspark_layers_to_capture=Mock(),
            logits_processor=SimpleNamespace(tp_group=adapter.mapping.attn.tp_group),
        )
    )
    torch.manual_seed(3)
    # The synthetic checkpoint's Markov weights are constants, which makes the
    # bigram bias the same for every anchor; random weights make the drafts
    # depend on which anchor each row reads.
    for weight in (
        adapter.model.markov_embedding.weight,
        adapter.model.markov_projection.weight,
    ):
        weight.copy_(torch.randn_like(weight, dtype=torch.float32).to(weight.dtype))
    # Positions stay inside the eight fake pages (slot = 64 + position).
    starts = [300, 40, 380]
    accepts = [4, 1, 6]
    hidden = torch.randn(
        n, width, drafter.hidden_width, dtype=torch.bfloat16, device="cuda:0"
    )
    tokens = torch.randint(
        0, adapter.model.config.vocab_size, (n, width), device="cuda:0"
    ).to(torch.int32)

    def run(order, *, packed=False):
        lengths = [accepts[i] if packed else width for i in order]
        positions = torch.cat(
            [
                torch.arange(starts[i], starts[i] + length, device="cuda:0")
                for i, length in zip(order, lengths, strict=True)
            ]
        )
        meta = SimpleNamespace(
            positions=positions,
            swa_write_slots=_paged_slots(positions, 64),
            request_indices=torch.arange(n, device="cuda:0").repeat_interleave(
                torch.tensor(lengths, device="cuda:0")
            ),
        )
        backend = SimpleNamespace(
            query_metadata=lambda mode: meta,
            decoder_view=lambda: V41DecoderView(meta, None, (), None, None),
            window_slots=lambda group, start_pos, requests: (
                _history_slots(start_pos, 128, 64),
                None,
            ),
        )
        ctx = _ctx(None, sum(lengths), ForwardMode.DECODE)
        if packed:
            ctx.decode_query_indptr = torch.tensor(
                [0, *torch.tensor(lengths).cumsum(0).tolist()], device="cuda:0"
            )
        ctx.bs, ctx.num_extends = n, 0
        ctx.output_layout = ForwardOutputLayout(0, 0, n, width)
        ctx.attn_backend, ctx.token_to_kv_pool = backend, pool
        windows.zero_()
        slots = torch.tensor([i + 1 for i in order], device="cuda:0")
        ib.state_write_req_pool_indices_buf[:n].copy_(slots)
        result = drafter.run(
            base_ctx=ctx,
            logits_output=SimpleNamespace(
                hidden_states=torch.cat(
                    [
                        hidden[i, :length]
                        for i, length in zip(order, lengths, strict=True)
                    ]
                )
            ),
            output_tokens=tokens[order].reshape(-1),
            accept_lengths=torch.tensor(
                [accepts[i] for i in order], device="cuda:0", dtype=torch.int32
            ),
        ).clone()
        if drafter.confidence_scores is not None:
            expected = adapter.model.confidence_logits(
                drafter.confidence_features[:n], result[:, :-1]
            )
            torch.testing.assert_close(
                drafter.confidence_scores.index_select(0, slots),
                expected,
                rtol=0,
                atol=0,
            )
        return result

    forward = run([0, 1, 2])
    forward_scores = (
        None
        if drafter.confidence_scores is None
        else drafter.confidence_scores[1:4].clone()
    )
    # Dropping unverified suffix rows must preserve the next draft exactly:
    # the anchor, bonus and accepted context are unchanged.
    torch.testing.assert_close(run([0, 1, 2], packed=True), forward, rtol=0, atol=0)
    torch.testing.assert_close(
        run([2, 0, 1], packed=True), forward[[2, 0, 1]], rtol=0, atol=0
    )
    rotated = run([2, 0, 1])
    torch.testing.assert_close(rotated, forward[[2, 0, 1]], rtol=0, atol=0)
    if forward_scores is not None:
        torch.testing.assert_close(
            drafter.confidence_scores[1:4], forward_scores, rtol=0, atol=0
        )
    if forward_scores is not None:
        from test.runtime.test_deepseek_v41_cache import _tables, _verify_backend

        # Exercise packed target metadata using the real draft candidates.
        assert torch.isfinite(forward_scores).all()
        drafter.prepare_verify_layout((1,) * n, num_tokens=n)
        drafter.set_decode_graph_query_extent(n)
        backend = _verify_backend("cuda:0", n, width)
        slots = torch.tensor([1, 2, 3], device="cuda:0")
        ib.req_pool_indices_buf[:n].copy_(slots)
        ib.seq_lens_buf[:n].copy_(
            torch.tensor([position + width for position in starts], device="cuda:0")
        )
        ib.input_ids_buf[: n * width].copy_(forward.flatten())
        backend.refresh_decode_metadata(
            n,
            n,
            slots,
            ib.seq_lens_buf[:n],
            forward_mode=ForwardMode.DECODE,
            # This check only prepares metadata; it never writes target KV.
            block_tables={
                key: torch.cat((table, torch.zeros_like(table[:1])))
                for key, table in _tables("cuda:0").items()
            },
            for_graph_replay=False,
        )
        ib.positions_buf[: n * width].copy_(
            backend.query_metadata(ForwardMode.DECODE).positions
        )
        ctx = _ctx(backend, n * width, ForwardMode.DECODE)
        ctx.bs = n
        ctx.output_layout = ForwardOutputLayout(0, 0, n, width)
        with drafter.target_forward_context(ctx):
            assert ctx.input_num_tokens == n
            assert ctx.decode_query_indptr.tolist() == [0, 1, 2, 3]
            torch.testing.assert_close(ib.input_ids_buf[:n], forward[:, 0])
        torch.testing.assert_close(ib.input_ids_buf[: n * width], forward.flatten())
        drafter.set_decode_graph_query_extent(None)
    # The requests differ, so identical drafts would mean the rows collapsed.
    assert not torch.equal(forward[0], forward[1])
    assert not torch.equal(forward[1], forward[2])
    # A target head whose local vocabulary diverges from the Markov
    # projection's must fail instead of drafting with wrong shard offsets.
    drafter.lm_head = SimpleNamespace(weight=adapter.lm_head.weight[:-1])
    with pytest.raises(ValueError, match="local vocabulary"):
        run([0, 1, 2])


def _assert_drafter_run_writes_rows_and_drafts(adapter, windows, pool):
    """Drive run() through a fake V4.1 backend: extend rows seed, decode rows draft."""
    from tokenspeed.runtime.execution.input_buffer import InputBuffers

    width, block = 6, 5
    ib = InputBuffers(3, 1024, 8, device="cuda:0")
    drafter = DeepseekV41DSpark(
        spec_num_tokens=width,
        spec_num_steps=block,
        draft_model_runner=SimpleNamespace(
            model=adapter,
            mapping=adapter.mapping,
            device="cuda:0",
            server_args=ServerArgs(model="unused", disaggregation_mode="null"),
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
    ctx.output_layout = ForwardOutputLayout(1, 1, 1, width)
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

    class ConfigArgs(SimpleNamespace):
        validate_tp_batch_invariant_weights = (
            ServerArgs.validate_tp_batch_invariant_weights
        )

    args = ConfigArgs(
        tp_batch_invariant="none",
        mapping=_mapping(0, 4, 4),
        speculative_algorithm="DSPARK",
        speculative_num_steps=None,
        speculative_num_draft_tokens=None,
        _speculative_widths_explicit=False,
        attention_backend=None,
        drafter_attention_backend=None,
        load_format="auto",
        disaggregation_mode="null",
        decode_monokernel=False,
        validate_tp_batch_invariant_weights=Mock(),
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
    assert side.is_dspark

    # Stop at the allocation boundary, after the real registry chooses both sides.
    config_builder = Mock(
        return_value=SimpleNamespace(component=lambda cls: None, kvp_size=1)
    )
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
            graph_reserve_bytes=0,
            post_profile_bytes=0,
            probe_batch_rows=None,
            profiled_cache_bytes=None,
            reuse_target_backend=None,
            reuse_draft_backend=None,
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


class TestDSparkBaseLogits:
    def test_dspark_base_logits_use_public_fp32_head_math(self):
        model = object.__new__(DeepseekV41DSparkModel)
        hidden = torch.tensor([[[1.0, -2.0]], [[0.5, 4.0]]], dtype=torch.bfloat16)
        head = torch.tensor([[0.5, 0.25], [-1.0, 2.0]], dtype=torch.bfloat16)

        logits = model.local_base_logits(hidden, head)

        assert logits.dtype == torch.float32
        assert logits.shape == (2, 1, 2)
        assert torch.equal(logits, hidden.float() @ head.float().T)
        with pytest.raises(ValueError, match="hidden dtype"):
            model.local_base_logits(hidden, head.float())

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
    def test_dspark_base_logits_read_the_shared_head_in_place_under_cuda_graph(self):
        model = object.__new__(DeepseekV41DSparkModel)
        hidden = torch.tensor([[1.0, -2.0]], dtype=torch.bfloat16, device="cuda")
        head = torch.tensor(
            [[0.5, 0.25], [-1.0, 2.0]], dtype=torch.bfloat16, device="cuda"
        )

        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            for _ in range(3):
                model.local_base_logits(hidden, head)
        torch.cuda.current_stream().wait_stream(warmup_stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            logits = model.local_base_logits(hidden, head)
        graph.replay()
        torch.cuda.synchronize()
        assert logits.dtype == torch.float32
        assert torch.equal(logits, hidden.float() @ head.float().T)

        # No FP32 copy sits between the target's head and the draft: an
        # in-place weight update is visible to the next replay as is.
        head.add_(1)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(logits, hidden.float() @ head.float().T)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
    @pytest.mark.parametrize("batch", [1, 16])
    def test_dspark_base_logits_accumulate_fp32_on_the_tp4_vocab_shard(self, batch):
        # V4.1 Flash: 5120 hidden, a 129280 / 4 vocabulary shard, block 5.
        model = object.__new__(DeepseekV41DSparkModel)
        generator = torch.Generator(device="cuda").manual_seed(batch)

        def draw_hidden():
            return torch.randn(
                batch, 5, 5120, device="cuda", generator=generator
            ).bfloat16()

        hidden = draw_hidden()
        head = (
            torch.randn(32320, 5120, device="cuda", generator=generator) * 0.02
        ).bfloat16()

        def assert_fp32_head_math(logits):
            assert logits.dtype == torch.float32
            assert logits.shape == (batch, 5, 32320)
            # Exact BF16 products summed in FP32 differ from FP64 only by
            # summation order; one BF16 rounding of the logits (2**-9
            # relative, logits near 1) would exceed this.
            torch.testing.assert_close(
                logits.double(),
                hidden.double() @ head.double().T,
                atol=1e-4,
                rtol=1e-4,
            )

        assert_fp32_head_math(model.local_base_logits(hidden, head))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            logits = model.local_base_logits(hidden, head)
        hidden.copy_(draw_hidden())
        graph.replay()
        torch.cuda.synchronize()
        assert_fp32_head_math(logits)


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("mode", ["prefill", "decode", "null"])
def test_tp1_ep2_draft_checkpoint_and_idle_schedule(monkeypatch, rank, mode):
    from test.runtime.test_deepseek_v41_model import _mock_loader_hardware

    from tokenspeed.runtime.configs.deepseek_v41_config import DeepseekV41Config
    from tokenspeed.runtime.distributed.mapping import Mapping
    from tokenspeed.runtime.execution.input_buffer import InputBuffers
    from tokenspeed.runtime.execution.model_runner import ModelRunner

    mapping = Mapping(
        rank=rank,
        world_size=2,
        attn_tp_size=1,
        attn_dp_size=2,
        dense_tp_size=1,
        dense_dp_size=2,
        moe_tp_size=1,
        moe_ep_size=2,
        moe_dp_size=1,
        vision_tp_size=1,
        vision_dp_size=1,
        linear_attn_tp_size=1,
        nprocs_per_node=2,
        nnodes=1,
    )
    _mock_loader_hardware(monkeypatch, mapping)
    monkeypatch.delitem(global_server_args_dict, "disaggregation_mode", raising=False)
    adapter = DeepseekV41ForCausalLMDSpark(
        DeepseekV41Config(text_config=vars(_draft_config())), mapping, _quant()
    )
    adapter.load_weights(_draft_checkpoint(adapter.model).items())
    # The runner's GPU initialization is outside this CPU checkpoint test.
    runner = Mock(spec=ModelRunner)
    runner.model, runner.mapping, runner.device = adapter, mapping, "cpu"
    runner.server_args = ServerArgs(model="unused", disaggregation_mode=mode)
    kwargs = dict(
        spec_num_tokens=6,
        spec_num_steps=5,
        draft_model_runner=runner,
        attn_backend=None,
        token_to_kv_pool=None,
        runtime_states=None,
        input_buffers=InputBuffers(3, 1024, 8, device="cpu"),
        vocab_size=adapter.model.config.vocab_size,
    )
    drafter = DeepseekV41DSpark(**kwargs)
    # Idle and active ranks must perform the same number of block forwards.
    assert drafter.idle_forward_global_num_tokens(
        [0, 18], [0, 3], global_num_decodes=[0, 3]
    ) == [[0, 15]]
    assert drafter.idle_forward_global_num_tokens(
        [128, 18], [1, 3], global_num_decodes=[0, 3]
    ) == [[0, 15]]
    assert drafter.idle_forward_global_num_tokens(
        [140, 18], [3, 3], global_num_decodes=[2, 3]
    ) == [[10, 15]]
    assert (
        drafter.idle_forward_global_num_tokens(
            [128, 128], [1, 1], global_num_decodes=[0, 0]
        )
        == []
    )
    assert (
        adapter.lm_head.weight.shape[0]
        == adapter.model.markov_projection.weight.shape[0]
    )
    assert adapter.checkpoint_load_report["loaded"] > 0


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_v41_confidence_uses_predecessor_embeddings_and_fp32(monkeypatch, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA/ROCm")
    config = _draft_config()
    _loader_model(monkeypatch, config, 0, "cpu")
    adapter = DeepseekV41ForCausalLMDSpark(
        SimpleNamespace(text_config=config), _mapping(0, 4, 4), _quant()
    )
    model = adapter.model
    model.markov_embedding.to(device)
    model.confidence_projection.to(device)
    with torch.no_grad():
        model.markov_embedding.weight.zero_()
        model.markov_embedding.weight[:, 0].copy_(
            torch.arange(config.vocab_size, device=device)
        )
        model.confidence_projection.weight.zero_()
        model.confidence_projection.weight[0, 0] = 0.25
        model.confidence_projection.weight[0, config.hidden_size] = -0.5
    hidden = torch.zeros(
        2,
        config.dspark_block_size,
        config.hidden_size,
        dtype=torch.bfloat16,
        device=device,
    )
    hidden[..., 0] = 2
    previous = torch.arange(10, device=device).view(2, 5)
    expected = 0.5 - 0.5 * previous.float()
    score = model.confidence_logits(hidden, previous)
    assert score.dtype == torch.float32
    torch.testing.assert_close(score, expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match="aligned block"):
        model.confidence_logits(hidden, previous[:, :-1])
    if device == "cuda":
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                model.confidence_logits(hidden, previous)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = model.confidence_logits(hidden, previous)
        previous.add_(1)
        graph.replay()
        torch.testing.assert_close(
            captured, 0.5 - 0.5 * previous.float(), rtol=0, atol=0
        )
        graph.reset()


@torch.inference_mode()
def _assert_adaptive_inputs_and_state(adapter):
    from test.runtime.test_deepseek_v41_cache import _tables, _verify_backend
    from test.runtime.test_deepseek_v41_engram import _config, _Tokenizer

    from tokenspeed.runtime.execution.input_buffer import InputBuffers
    from tokenspeed.runtime.execution.runtime_states import RuntimeStates
    from tokenspeed.runtime.models.deepseek_v41_engram import EngramHashState

    device, width = "cuda:0", 6
    ib = InputBuffers(2, 1024, 4, device=device)
    hash_state = EngramHashState(_config(), _Tokenizer(), device)
    ib.init_ngram_buffers(3, hash_state.input_parameters, len(_Tokenizer()))
    original_hashes = torch.arange(
        ib.ngram_hash_ids_buf[:12].numel(),
        device=device,
        dtype=ib.ngram_hash_ids_buf.dtype,
    ).reshape_as(ib.ngram_hash_ids_buf[:12])
    original_mask = torch.arange(12, device=device) % 2 == 0
    state = RuntimeStates(4, 128, 128, width, device=device)
    backend = _verify_backend(device, 2, width)
    tables = _tables(device)
    drafter = DeepseekV41DSpark(
        spec_num_tokens=width,
        spec_num_steps=5,
        draft_model_runner=SimpleNamespace(
            model=adapter, mapping=adapter.mapping, device=device
        ),
        attn_backend=backend,
        token_to_kv_pool=backend.cache_pool,
        runtime_states=state,
        input_buffers=ib,
        vocab_size=128,
    )
    drafter.wire_target(adapter)
    original = torch.arange(12, device=device, dtype=ib.input_ids_buf.dtype)

    def prepare(ids, slots):
        drafter.prepare_request_state(ids, slots, 0)
        ib.req_pool_indices_buf[:2].copy_(torch.tensor(slots, device=device))
        ib.input_ids_buf[:12].copy_(original)
        ib.ngram_hash_ids_buf[:12].copy_(original_hashes)
        ib.ngram_token_mask_buf[:12].copy_(original_mask)
        ib.seq_lens_buf[:2].copy_(torch.tensor([18, 30], device=device))
        backend.refresh_decode_metadata(
            2,
            2,
            ib.req_pool_indices_buf[:2],
            ib.seq_lens_buf[:2],
            forward_mode=ForwardMode.DECODE,
            block_tables=tables,
            for_graph_replay=False,
        )
        ib.positions_buf[:12].copy_(
            backend.query_metadata(ForwardMode.DECODE).positions
        )
        ctx = _ctx(backend, 12, ForwardMode.DECODE)
        ctx.bs = 2
        ctx.output_layout = ForwardOutputLayout(0, 0, 2, width)
        return ctx

    ctx = prepare(["a", "b"], [1, 3])
    drafter.confidence_scores[1].copy_(torch.tensor([1, -1, 1, 1, 1], device=device))
    drafter.confidence_scores[3].copy_(torch.tensor([1, 1, -1, 1, 1], device=device))
    saved_positions = ib.positions_buf[:12].clone()
    drafter.prepare_verify_layout((2, 3), num_tokens=5)
    drafter.set_decode_graph_query_extent(5)
    ctx.global_num_tokens = [12] * 8
    with drafter.target_forward_context(ctx):
        assert ctx.global_num_tokens == [5] * 8
        assert ctx.input_num_tokens == 5
        assert ctx.decode_query_indptr.tolist() == [0, 2, 5]
        assert ib.input_ids_buf[:5].tolist() == [0, 1, 6, 7, 8]
        assert ib.positions_buf[:5].tolist() == [12, 13, 24, 25, 26]
        torch.testing.assert_close(drafter.verify_candidates().flatten(), original)
        rows = torch.tensor([0, 1, 6, 7, 8], device=device)
        torch.testing.assert_close(ib.ngram_hash_ids_buf[:5], original_hashes[rows])
        torch.testing.assert_close(ib.ngram_token_mask_buf[:5], original_mask[rows])
    assert ctx.global_num_tokens == [12] * 8
    torch.testing.assert_close(ib.input_ids_buf[:12], original)
    torch.testing.assert_close(ib.ngram_hash_ids_buf[:12], original_hashes)
    torch.testing.assert_close(ib.ngram_token_mask_buf[:12], original_mask)
    torch.testing.assert_close(ib.positions_buf[:12], saved_positions)

    # Reusing a layout still reads this round's inputs, not the saved
    # candidates from the preceding round.
    ctx = prepare(["a", "b"], [1, 3])
    changed = original.flip(0)
    ib.input_ids_buf[:12].copy_(changed)
    with drafter.target_forward_context(ctx):
        assert ctx.decode_query_indptr.tolist() == [0, 2, 5]
        torch.testing.assert_close(
            ib.input_ids_buf[:5], changed[torch.tensor([0, 1, 6, 7, 8], device=device)]
        )
        torch.testing.assert_close(drafter.verify_candidates().flatten(), changed)
    torch.testing.assert_close(ib.input_ids_buf[:12], changed)

    # The allocator publishes the layout for the reordered request rows.
    ctx = prepare(["b", "a"], [3, 1])
    drafter.prepare_verify_layout((3, 2), num_tokens=5)
    with pytest.raises(RuntimeError, match="target failed"):
        with drafter.target_forward_context(ctx):
            assert ctx.decode_query_indptr.tolist() == [0, 3, 5]
            raise RuntimeError("target failed")
    torch.testing.assert_close(ib.input_ids_buf[:12], original)
    torch.testing.assert_close(ib.ngram_hash_ids_buf[:12], original_hashes)
    torch.testing.assert_close(ib.ngram_token_mask_buf[:12], original_mask)

    image = torch.empty(drafter.slot_state_bytes(), dtype=torch.uint8, device=device)
    stream = torch.cuda.current_stream()
    drafter.export_slot_state(1, image, stream, request_id="a")
    drafter.import_slot_state(2, image, stream, request_id="a")
    drafter.prepare_request_state(["a"], [2], 0)
    torch.testing.assert_close(
        drafter.confidence_scores[2], drafter.confidence_scores[1]
    )
    drafter.export_slot_state(1, image, stream, request_id="unprepared")
    drafter.import_slot_state(0, image, stream, request_id="unprepared")
    assert torch.isposinf(drafter.confidence_scores[0]).all()

    ctx = prepare(["new", "b"], [1, 3])
    assert torch.isposinf(drafter.confidence_scores[1]).all()
    drafter.set_decode_graph_query_extent(None)
    with drafter.target_forward_context(ctx):
        assert ctx.decode_query_indptr is None
    drafter.on_target_weights_updated()
    ctx = prepare(["new", "b"], [1, 3])
    with drafter.target_forward_context(ctx):
        assert ctx.decode_query_indptr is None
        assert ctx.input_num_tokens == 12

    # Full-capacity execution must not reuse the previous packed layout.
    ctx = prepare(["a", "b"], [1, 3])
    drafter.set_decode_graph_query_extent(12)
    with drafter.target_forward_context(ctx):
        assert ctx.decode_query_indptr is None
        assert drafter._verify_candidates is None
        torch.testing.assert_close(ib.input_ids_buf[:12], original)
        torch.testing.assert_close(
            backend.query_metadata(ForwardMode.DECODE).positions, saved_positions
        )
    drafter.set_decode_graph_query_extent(None)

    # One captured six-row forward must serve different live CSR layouts,
    # preserve fresh candidates, and restore full scheduler-width inputs.
    ctx = prepare(["a", "b"], [1, 3])
    drafter.prepare_verify_layout((2, 3), num_tokens=6)
    drafter.set_decode_graph_query_extent(6)

    def packed_step():
        with drafter.target_forward_context(ctx):
            return (
                ib.input_ids_buf[:6].clone(),
                ib.ngram_hash_ids_buf[:6].clone(),
                backend.query_metadata(ForwardMode.DECODE).positions.clone(),
                drafter.verify_candidates().clone(),
            )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            packed_step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = packed_step()
    for counts in ((3, 2), (1, 5), (2, 3)):
        changed = original + sum(counts)
        ib.input_ids_buf[:12].copy_(changed)
        drafter.prepare_verify_layout(counts, num_tokens=6)
        graph.replay()
        live = sum(counts)
        indices = torch.tensor(
            [r * width + c for r, n in enumerate(counts) for c in range(n)],
            device=device,
        )
        torch.testing.assert_close(captured[0][:live], changed[indices])
        torch.testing.assert_close(captured[1][:live], original_hashes[indices])
        torch.testing.assert_close(captured[2][:live], saved_positions[indices])
        torch.testing.assert_close(captured[3].flatten(), changed)
        torch.testing.assert_close(ib.input_ids_buf[:12], changed)
        torch.testing.assert_close(ib.ngram_hash_ids_buf[:12], original_hashes)
    graph.reset()
    drafter.set_decode_graph_query_extent(None)


@pytest.mark.parametrize(
    "field,value",
    [
        (None, None),
        ("speculative_algorithm", "EAGLE3"),
        ("sampling_backend", "flashinfer"),
        ("disable_prefill_graph", False),
        ("kv_parallel_size", 2),
        ("decode_context_parallel_size", 2),
        ("query_context_parallel_size", 2),
        ("pipeline_parallel_size", 2),
        ("dp_sampling", True),
        ("grammar_backend", "xgrammar"),
        ("enable_output_logprobs", True),
        ("enable_output_cumulative_probs", True),
        ("enable_speculative_sampling", True),
        ("speculative_eagle_topk", 2),
        ("synthetic_acceptance_length", 2.5),
        ("synthetic_acceptance_length", 4),
        ("enforce_eager", True),
        ("disable_sampling_tp_sync", True),
    ],
)
def test_adaptive_v41_launch_contract(field, value):
    args = ServerArgs.__new__(ServerArgs)
    args.enable_dsv41_adaptive_dspark = True
    args.speculative_algorithm = "DSPARK"
    args.sampling_backend = "greedy"
    args.enforce_eager = False
    args.disable_prefill_graph = True
    args.grammar_backend = "none"
    args.mapping = _mapping(0, 1, 1)
    if field is not None:
        setattr(args, field, value)
        with pytest.raises(ValueError):
            args.validate_dsv41_adaptive_dspark()
    else:
        args.validate_dsv41_adaptive_dspark()
    # With no opt-in, existing launch configurations incur no new restrictions.
    args.enable_dsv41_adaptive_dspark = False
    args.validate_dsv41_adaptive_dspark()


@pytest.mark.parametrize(
    "bs,capacity", [(32, 128), (32, 144), (32, 192), (16, 64), (4, 16)]
)
def test_verify_graph_token_capacity(bs, capacity):
    drafter = object.__new__(DeepseekV41DSpark)
    drafter.device = torch.device("cpu")
    drafter.spec_num_tokens = 6
    drafter.adaptive_min_batch_size = 1
    drafter.confidence_scores = torch.empty(1)
    drafter.verify_graph_token_step = 16
    drafter._verify_rows = torch.empty(bs * 6, dtype=torch.int64)
    drafter._verify_indptr = torch.empty(bs + 1, dtype=torch.int64)
    drafter._verify_layout_counts = None
    drafter.prepare_decode_graph_token_capture_layout(bs=bs, query_tokens=capacity)
    assert drafter._verify_indptr[bs].item() == capacity
    assert all(1 <= n <= 6 for n in drafter._verify_layout_counts)
    assert capacity in drafter.decode_graph_query_capacities(bs)


@pytest.mark.parametrize("actual,padded", [(0, 32), (29, 32), (16, 16)])
def test_small_or_padded_batches_keep_full_verification(actual, padded):
    from tokenspeed.runtime.execution.drafter.verify_budget import VerifyBudgetPolicy

    drafter = object.__new__(DeepseekV41DSpark)
    drafter.spec_num_tokens = 6
    drafter._verify_step_id = 0
    drafter._active_verify_timing = None
    drafter.verify_budget_policy = VerifyBudgetPolicy(True, 32, 6, 0.02)
    drafter.draft_model = SimpleNamespace(mapping=_mapping(0, 1, 1))
    ctx = ForwardContext(
        attn_backend=None,
        token_to_kv_pool=None,
        bs=actual,
        num_extends=0,
        input_num_tokens=actual * 6,
        forward_mode=ForwardMode.DECODE,
        output_layout=ForwardOutputLayout(0, 0, actual, 6),
    )
    assert (
        drafter.prepare_decode_graph_token_layout(
            ctx, actual_bs=actual, padded_bs=padded
        )
        == padded * 6
    )
    drafter.verify_budget_policy = None
    assert (
        drafter.prepare_decode_graph_token_layout(
            ctx, actual_bs=actual, padded_bs=padded
        )
        == padded * 6
    )


def test_fine_verify_graph_capture_plan_and_keys():
    from tokenspeed.runtime.execution.forward_step import ForwardStepRunner

    drafter = object.__new__(DeepseekV41DSpark)
    drafter.device = torch.device("cpu")
    drafter.spec_num_tokens = 6
    drafter.adaptive_min_batch_size = 1
    drafter.verify_budget_policy = None
    drafter.confidence_scores = torch.empty(1)
    drafter.verify_graph_token_step = 8
    runner = object.__new__(ForwardStepRunner)
    runner.disable = False
    runner.capture_bs = [1, 8, 16, 32]
    runner.sampling_backend = None
    runner.max_tokens_per_req = 6
    runner.drafter = drafter
    plan = runner.capture_plan
    for bs in runner.capture_bs:
        capacities = [
            runner._split_query_variant(v, bs)[1]
            for v, ladder in plan.items()
            if bs in ladder
        ]
        assert sorted(capacities) == list(drafter.decode_graph_query_capacities(bs))
    runner.graphs = {(v, bs): object() for v, ladder in plan.items() for bs in ladder}
    runner._decode_query_tokens = 152
    key = runner._cuda_graph_key(32)
    assert runner._split_query_variant(key[0], key[1]) == ("default", 152)
    runner._decode_query_tokens = 192
    assert runner._cuda_graph_key(32) == ("default", 32)


@pytest.fixture(autouse=True)
def adaptive_test_small_batches(monkeypatch):
    # Small model fixtures exercise the adaptive path; serving defaults to 32.
    monkeypatch.setitem(global_server_args_dict, "dsv41_dspark_min_batch_size", 1)


@pytest.mark.parametrize("minimum", [0, -1, True, 1.5])
def test_invalid_adaptive_minimum_batch_size(minimum):
    args = ServerArgs(model="unused", disaggregation_mode="null")
    args.dsv41_dspark_min_batch_size = minimum
    with pytest.raises(ValueError, match="minimum batch size"):
        args.validate_dsv41_adaptive_dspark()


def test_budget_switch_validation_is_repeatable():
    args = ServerArgs.__new__(ServerArgs)
    args.enable_dsv41_adaptive_dspark = True
    args.speculative_algorithm = "DSPARK"
    args.sampling_backend = "greedy"
    args.disable_prefill_graph = True
    args.grammar_backend = "none"
    args.mapping = _mapping(0, 1, 1)
    args.validate_dsv41_adaptive_dspark()
    args.validate_dsv41_adaptive_dspark()
    assert args.enable_dsv41_adaptive_dspark is True
    assert args.dsv41_dspark_min_batch_size == 32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@torch.inference_mode()
def test_budget_allocation_graphs_are_prepared_and_reused(tmp_path, monkeypatch):
    from tokenspeed_kernel.ops.attention.dsv41 import verify_layout

    from tokenspeed.runtime.execution.drafter.verify_budget import VerifyBudgetPolicy

    drafter = object.__new__(DeepseekV41DSpark)
    from tokenspeed.runtime.execution.drafter.verify_calibration import (
        VerifyCalibration,
    )
    from tokenspeed.runtime.execution.drafter.verify_feedback import VerifyFeedback
    from tokenspeed.runtime.execution.model_runner import ModelRunner

    tp = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    mapping = _mapping(rank, tp, tp)
    if tp > 1:
        from tokenspeed.runtime.distributed.process_group_manager import (
            process_group_manager,
        )

        process_group_manager.init_distributed(
            mapping,
            distributed_init_method="env://",
            backend="nccl",
            timeout=120,
            device_id=torch.device("cuda", int(os.environ["LOCAL_RANK"])),
        )
        process_group_manager.init_process_group(mapping.attn.tp_group, backend="nccl")
    drafter.draft_model = Mock(spec=ModelRunner)
    drafter.draft_model.mapping = mapping
    drafter.tp_group = mapping.attn.tp_group
    drafter.verify_feedback = VerifyFeedback(draft_width=5)
    drafter._pending_verify_feedback = []
    drafter._verify_step_id = 0
    drafter._verify_timing_epoch = 0
    drafter._active_verify_timing = None
    drafter._verify_step_samples = []
    drafter._aligned_verify_samples = ()
    drafter._verify_startup_ready = False
    drafter._verify_collecting = True
    drafter._verify_cache_initialized = True
    drafter._verify_cache = None
    drafter._cached_verify_profiles = ()
    drafter._verify_collectors = {}
    drafter._verify_calibrations = ()
    drafter._verify_policy_ms = None
    drafter.device = "cuda"
    drafter.spec_num_tokens = 6
    drafter.block_size = 5
    drafter.adaptive_min_batch_size = 4
    drafter.verify_budget_policy = VerifyBudgetPolicy(True, 4, 6, 0.02)
    drafter.confidence_scores = torch.empty(1, device="cuda")
    drafter.verify_graph_token_step = 4
    drafter._verify_rows = torch.empty(24, dtype=torch.int64, device="cuda")
    drafter._verify_indptr = torch.empty(5, dtype=torch.int64, device="cuda")
    drafter._verify_layout_counts = None
    drafter._verify_budget_graphs = {}
    drafter._verify_budget_buffers = {}
    stream = torch.cuda.Stream()
    pool = torch.cuda.graph_pool_handle()
    drafter.capture_round_graphs(2, stream=stream, pool=pool)
    assert not drafter._verify_budget_graphs
    drafter.capture_round_graphs(4, stream=stream, pool=pool)
    graphs = dict(drafter._verify_budget_graphs)
    drafter.capture_round_graphs(4, stream=stream, pool=pool)
    assert drafter._verify_budget_graphs == graphs
    scores = torch.randn(4, 5, device="cuda")
    for capacity in (20, 4, 12, 8, 16, 12):
        scores.neg_()
        drafter.prepare_verify_budget_layout(
            scores, batch_size=4, query_capacity=capacity
        )
        rows = torch.empty(capacity, dtype=torch.int64, device="cuda")
        indptr = torch.empty(5, dtype=torch.int64, device="cuda")
        counts = torch.empty(4, dtype=torch.int64, device="cuda")
        valid = torch.empty((), dtype=torch.bool, device="cuda")
        verify_layout(
            scores,
            query_capacity=capacity,
            rows_out=rows,
            indptr_out=indptr,
            counts_out=counts,
            valid_out=valid,
        )
        torch.testing.assert_close(
            drafter._verify_rows[:capacity], rows, atol=0, rtol=0
        )
        torch.testing.assert_close(drafter._verify_indptr, indptr, atol=0, rtol=0)
    assert drafter._prepare_calibrated_budget(actual_bs=4, padded_bs=4) == 24
    profile = VerifyCalibration(4, 100, 200, 6, 10.0, ((12, 1.0),), 3)
    drafter.install_verify_calibration((profile,), policy_ms=0.0)
    assert drafter._prepare_calibrated_budget(actual_bs=4, padded_bs=4) == 24
    ids, slots = ["a", "b", "c", "d"], [0, 1, 2, 3]
    drafter.verify_feedback.prepare(ids, slots, num_extends=0)
    drafter.verify_feedback.publish(
        drafter.verify_feedback.snapshot(),
        [[0.0 if rank == 0 else -1000.0] * 5] * 4,
        context_lengths=[120 if rank == 0 else 10000] * 4,
    )
    for _ in range(2):
        drafter.verify_feedback.prepare(ids, slots, num_extends=0)
    assert drafter._prepare_calibrated_budget(actual_bs=4, padded_bs=4) == 12
    assert drafter._verify_indptr.tolist() == [0, 3, 6, 9, 12]
    assert drafter._verify_budget_buffers[4].valid.item()
    assert drafter._prepare_calibrated_budget(actual_bs=4, padded_bs=8) == 48
    profile = VerifyCalibration(4, 100, 125, 6, 10.0, ((12, 1.0),), 3)
    drafter.install_verify_calibration((profile,), policy_ms=0.0)
    assert drafter._prepare_calibrated_budget(actual_bs=4, padded_bs=4) == 24
    from tokenspeed.runtime.execution.context import ForwardContext
    from tokenspeed.runtime.execution.input_buffer import InputBuffers
    from tokenspeed.runtime.execution.output_layout import ForwardOutputLayout

    drafter.input_buffers = InputBuffers(4, 24, 0, device="cuda")
    drafter.input_buffers.req_pool_indices_buf[:4].copy_(torch.arange(4, device="cuda"))
    drafter.input_buffers.seq_lens_buf[:4].fill_(120)
    drafter.confidence_scores = torch.zeros((4, 5), device="cuda")
    ctx = ForwardContext(
        attn_backend=None,
        token_to_kv_pool=None,
        bs=4,
        num_extends=0,
        input_num_tokens=24,
        forward_mode=ForwardMode.DECODE,
        output_layout=ForwardOutputLayout(0, 0, 4, 6),
    )
    assert (
        drafter.prepare_decode_graph_token_layout(ctx, actual_bs=4, padded_bs=4) == 24
    )
    drafter.confidence_scores.add_(1)
    drafter.finish_round(ctx, graph_bs=4)
    torch.cuda.synchronize()
    drafter._poll_verify_feedback()
    assert len(drafter._verify_step_samples) == 1
    sample = drafter._verify_step_samples[0]
    assert (sample.step, sample.batch_size, sample.query_capacity) == (1, 4, 24)
    assert (sample.min_context, sample.max_context) == (120, 120)
    assert sample.elapsed_ms > 0
    drafter._synchronize_verify_samples(drafter._verify_budget_buffers[4])
    assert len(drafter._aligned_verify_samples) == 1
    aligned = drafter._aligned_verify_samples[0]
    assert (aligned.step, aligned.batch_size, aligned.query_capacity) == (1, 4, 24)
    assert aligned.elapsed_ms >= sample.elapsed_ms
    # Padded rounds do not contribute measurements for an exact-BS profile.
    assert (
        drafter.prepare_decode_graph_token_layout(ctx, actual_bs=4, padded_bs=8) == 48
    )
    assert drafter._active_verify_timing is None
    drafter.finish_round(ctx, graph_bs=8)
    assert not drafter._pending_verify_feedback
    # Aligned real-step metadata drives the same collector on every rank.
    # Synthetic durations test the installation/probing protocol, not speed.
    from tokenspeed.runtime.execution.drafter.verify_calibration import VerifyStepSample

    drafter._verify_calibrations = ()
    drafter._verify_startup_ready = False
    drafter._verify_collecting = True
    for tier, capacity in enumerate((24, 20, 16, 12, 8, 4)):
        drafter._aligned_verify_samples = tuple(
            VerifyStepSample(
                100 + tier * 10 + index,
                4,
                capacity,
                120,
                120,
                1000.0 if index < 2 else float(capacity),
            )
            for index in range(10)
        )
        drafter._update_verify_calibration()
        if tier == 0:
            assert drafter._prepare_calibrated_budget(actual_bs=4, padded_bs=4) == 20
            assert drafter._verify_collecting
    assert len(drafter._verify_calibrations) == 1
    automatic = drafter._verify_calibrations[0]
    assert automatic.native_step_ms == 24.0
    assert automatic.sample_count == 8
    assert automatic.adaptive_step_ms == tuple(
        (q, float(q)) for q in (4, 8, 12, 16, 20)
    )
    assert not drafter._verify_collecting
    # Rebuilding graphs invalidates already queued timing and score feedback.
    drafter.prepare_decode_graph_token_layout(ctx, actual_bs=4, padded_bs=4)
    drafter.finish_round(ctx, graph_bs=4)
    old_snapshot = drafter.verify_feedback.snapshot()
    drafter.release_round_graphs()
    torch.cuda.synchronize()
    drafter._poll_verify_feedback()
    assert not drafter._verify_step_samples
    assert drafter.verify_feedback.snapshot() == ()
    drafter.verify_feedback.prepare(ids, slots, num_extends=0)
    drafter.verify_feedback.publish(
        old_snapshot, [[0.0] * 5] * 4, context_lengths=[120] * 4
    )
    for _ in range(2):
        drafter.verify_feedback.prepare(ids, slots, num_extends=0)
    assert drafter.verify_feedback.scores() is None
    assert drafter.verify_feedback.context_bounds() is None
    assert not drafter._verify_budget_graphs
    assert not drafter._verify_budget_buffers
    with pytest.raises(ValueError, match="not prepared"):
        drafter.prepare_verify_budget_layout(scores, batch_size=4, query_capacity=12)

    # Re-capture restores completed regions without recalibration.
    drafter.capture_round_graphs(4, stream=stream, pool=pool)
    assert drafter._verify_calibrations == (automatic,)

    from transformers import PretrainedConfig

    from tokenspeed.runtime.utils.server_args import prepare_server_args

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    runner = Mock(spec=ModelRunner)
    runner.server_args = prepare_server_args(["--model", str(tmp_path / "checkpoint")])
    runner.model_config = Mock(spec=ModelConfig)
    runner.model_config.hf_config = PretrainedConfig(hidden_size=128)
    drafter.draft_model_runner = runner
    drafter._verify_cache_initialized = False
    drafter._initialize_verify_cache()
    if rank == 0:
        assert drafter._verify_cache is not None
    drafter._save_verify_calibration()
    drafter._cached_verify_profiles = ()
    drafter._verify_calibrations = ()
    drafter._verify_cache_initialized = False
    drafter._restore_verify_calibration(4)
    assert drafter._verify_calibrations == (automatic,)
    assert not drafter._verify_collecting
    drafter.on_target_weights_updated()
    assert not drafter._verify_calibrations
    assert not drafter._cached_verify_profiles
    assert drafter._verify_cache is None
    assert drafter._verify_collecting

    if tp > 1:
        torch.distributed.destroy_process_group()

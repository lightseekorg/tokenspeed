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

"""Looped Inkling2: the cache-layer view, loading, and the loop forward.

The forward tests run the real decoder layers and residual state on CPU, with
the GPU-only attention, MoE and convolution kernels replaced by fixed
functions of their inputs, and compare against a torch transcription of the
looped reference: each later pass restarts from the core's entry and gates
messages from the previous pass's residual blocks into the core's inputs.
"""

from __future__ import annotations

from pathlib import Path
from test.runtime.models.test_inkling2_config import _checkpoint
from test.runtime.models.test_inkling2_model import _mapping
from test.runtime.test_inkling2_quantization import _group

import pytest
import torch
from tokenspeed_kernel.ops.layernorm.precision import rmsnorm_precision

from tokenspeed.runtime.configs.inkling2 import (
    Inkling2LoopMMConfig,
    Inkling2MMConfig,
    Inkling2ModelConfig,
    Inkling2MTPConfig,
)
from tokenspeed.runtime.configs.model_config import (
    ModelConfig,
    _configure_inkling2_loop,
    _derive_num_attention_layers,
)
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.forward_batch_info import (
    CaptureHiddenMode,
    ForwardMode,
)
from tokenspeed.runtime.execution.output_layout import ForwardOutputLayout
from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig
from tokenspeed.runtime.layers.attention.kv_cache.factory import (
    create_cache_arena,
    create_cache_pool,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.inkling import InklingRecipe
from tokenspeed.runtime.layers.attention_residual import (
    AttentionResidual,
    AttentionResidualState,
)
from tokenspeed.runtime.layers.moe.inkling2 import Inkling2MoE
from tokenspeed.runtime.layers.quantization.inkling2 import Inkling2QuantConfig
from tokenspeed.runtime.models.inkling2 import Inkling2ForConditionalGeneration
from tokenspeed.runtime.models.inkling2_attention import (
    Inkling2Attention,
    Inkling2ShortConvolution,
)
from tokenspeed.runtime.models.inkling2_loop import (
    Inkling2LoopMessages,
    Inkling2MessageController,
)
from tokenspeed.runtime.utils.env import global_server_args_dict
from tokenspeed.runtime.utils.hf_transformers_utils import get_config
from tokenspeed.runtime.utils.server_args import ServerArgs

EPS = 1e-6


def _looped_checkpoint() -> dict:
    """The released looped geometry: 24 decoder layers, a 12-layer core run
    three times, and draft taps that count layer visits."""
    values = _checkpoint()
    values.update(
        model_type="inkling2_loop",
        architectures=["Inkling2LoopForConditionalGeneration"],
    )
    values["text_config"].update(
        hidden_size=2304,
        num_hidden_layers=24,
        num_attention_heads=24,
        swa_num_attention_heads=24,
        local_layer_ids=[i for i in range(24) if i % 6 != 5],
        n_routed_experts=128,
        intermediate_size=1152,
        log_scaling_n_floor=8192,
        loop_config={
            "core_start": 6,
            "core_end": 18,
            "num_passes": 3,
            "gate_rank": 64,
            "is_varloop": False,
        },
    )
    values["mtp_config"].update(
        hidden_size=2304,
        num_hidden_layers=3,
        num_attention_heads=24,
        local_layer_ids=[0, 2],
        target_layer_ids=[5, 23, 41, 48],
    )
    return values


# ---- the cache-layer view ----


def test_looped_visits_and_cache_layers() -> None:
    config = Inkling2LoopMMConfig(**_looped_checkpoint())
    text = config.text_config
    assert text.num_hidden_layers == 24
    assert text.loop_core_layers == range(6, 18)
    assert text.num_loop_passes == 3
    assert text.num_layer_visits == text.num_cache_layers == 48
    visits = text.layer_visits
    assert visits[:6] == [(layer, 0) for layer in range(6)]
    assert visits[6:42] == [(layer, p) for p in range(3) for layer in range(6, 18)]
    assert visits[42:] == [(layer, 0) for layer in range(18, 24)]
    assert [text.cache_layer_id(layer, p) for layer, p in visits] == [
        *range(18),
        *range(24, 48),
        *range(18, 24),
    ]
    assert text.cache_layer_id(6, 1) == 24
    assert text.cache_layer_id(17, 2) == 47
    with pytest.raises(ValueError, match="no pass 1"):
        text.cache_layer_id(5, 1)
    with pytest.raises(ValueError, match="no pass 3"):
        text.cache_layer_id(6, 3)
    labels = text.cache_layer_types
    assert len(labels) == 48
    full = [i for i, label in enumerate(labels) if label == "full_attention"]
    # Full layers 5, 11, 17 and 23, then the later passes of 11 and 17.
    assert full == [5, 11, 17, 23, 29, 35, 41, 47]
    assert text.local_cache_layer_ids == [i for i in range(48) if i not in full]
    assert all(labels.count(f"sliding_attention_{k}") == 8 for k in range(5))
    assert _derive_num_attention_layers(config, 24) == 48


def test_unlooped_cache_view_is_one_cache_layer_per_decoder_layer() -> None:
    config = Inkling2MMConfig(**_checkpoint())
    text = config.text_config
    assert text.loop_config is None
    assert text.loop_core_layers == range(0)
    assert text.num_layer_visits == text.num_cache_layers == 48
    assert text.layer_visits == [(layer, 0) for layer in range(48)]
    assert text.local_cache_layer_ids == text.local_layer_ids
    labels = text.cache_layer_types
    assert labels[:6] == [
        *(f"sliding_attention_{k}" for k in range(5)),
        "full_attention",
    ]
    assert labels[6:12] == [
        *(f"sliding_attention_{k}" for k in range(5)),
        "full_attention",
    ]
    assert _derive_num_attention_layers(config, 48) == 48


@pytest.mark.parametrize(
    "changes,message",
    [
        ({"core_start": 5}, "align with attention residual blocks"),
        ({"core_end": 30}, "within the decoder layers"),
        ({"core_start": 18, "core_end": 6}, "within the decoder layers"),
        ({"num_passes": 1}, "at least two passes"),
        ({"gate_rank": 0}, "positive gate rank"),
        ({"gate_rank": 64.0}, "integer fields"),
        ({"extra": 1}, "integer fields"),
        ({"is_varloop": 0}, "boolean is_varloop"),
    ],
)
def test_invalid_loop_config(changes: dict, message: str) -> None:
    values = _looped_checkpoint()["text_config"]
    values["loop_config"].update(changes)
    with pytest.raises(ValueError, match=message):
        Inkling2ModelConfig(**values)


def test_loop_config_requires_is_varloop() -> None:
    values = _looped_checkpoint()["text_config"]
    del values["loop_config"]["is_varloop"]
    with pytest.raises(ValueError, match="boolean is_varloop"):
        Inkling2ModelConfig(**values)


@pytest.mark.parametrize("is_varloop", [False, True])
def test_varloop_checkpoint_serves_its_own_passes(is_varloop: bool) -> None:
    values = _looped_checkpoint()
    values["text_config"]["loop_config"]["is_varloop"] = is_varloop
    text = Inkling2LoopMMConfig(**values).text_config
    assert text.num_loop_passes == 3
    assert text.num_cache_layers == 48


def _varloop_config(num_passes: int | None, share_cache: bool) -> Inkling2LoopMMConfig:
    values = _looped_checkpoint()
    values["text_config"]["loop_config"]["is_varloop"] = True
    config = Inkling2LoopMMConfig(**values)
    config.text_config.configure_loop(num_passes=num_passes, share_cache=share_cache)
    return config


@pytest.mark.parametrize("num_passes", [1, 2, 4, 5])
def test_pass_count_override_derives_visits_and_caches(num_passes: int) -> None:
    """Visits run the prelude, the core num_passes times and the coda; every
    visit keeps its own KV cache and convolution state."""
    config = _varloop_config(num_passes, False)
    text = config.text_config
    assert text.num_loop_passes == num_passes
    num_visits = 12 + 12 * num_passes
    assert text.num_layer_visits == text.num_cache_layers == num_visits
    assert text.num_conv_layers == num_visits
    assert text.layer_visits == (
        [(layer, 0) for layer in range(6)]
        + [(layer, p) for p in range(num_passes) for layer in range(6, 18)]
        + [(layer, 0) for layer in range(18, 24)]
    )
    ids = [text.cache_layer_id(layer, p) for layer, p in text.layer_visits]
    assert sorted(ids) == list(range(num_visits))
    assert ids == [text.conv_layer_id(layer, p) for layer, p in text.layer_visits]
    labels = text.cache_layer_types
    assert len(labels) == num_visits
    # Full layers 5, 11, 17 and 23, and 11 and 17 again each later pass.
    assert labels.count("full_attention") == 4 + 2 * (num_passes - 1)
    assert _derive_num_attention_layers(config, 24) == num_visits


@pytest.mark.parametrize("num_passes", [None, 1, 4])
def test_shared_cache_keeps_one_kv_layer_per_decoder_layer(
    num_passes: int | None,
) -> None:
    """Every pass of a core layer addresses that layer's one KV cache; the
    convolution state stays one per visit."""
    config = _varloop_config(num_passes, True)
    text = config.text_config
    passes = 3 if num_passes is None else num_passes
    assert text.num_loop_passes == passes
    assert text.num_cache_layers == 24
    assert text.num_conv_layers == text.num_layer_visits == 12 + 12 * passes
    visits = text.layer_visits
    assert [text.cache_layer_id(layer, p) for layer, p in visits] == [
        layer for layer, _ in visits
    ]
    conv_ids = [text.conv_layer_id(layer, p) for layer, p in visits]
    assert sorted(conv_ids) == list(range(text.num_layer_visits))
    assert text.local_cache_layer_ids == text.local_layer_ids
    labels = text.cache_layer_types
    assert [i for i, label in enumerate(labels) if label == "full_attention"] == [
        5,
        11,
        17,
        23,
    ]
    assert _derive_num_attention_layers(config, 24) == 24


def test_loop_overrides_need_a_varloop_checkpoint() -> None:
    fixed = Inkling2LoopMMConfig(**_looped_checkpoint()).text_config
    unlooped = Inkling2MMConfig(**_checkpoint()).text_config
    for text in (fixed, unlooped):
        for overrides in (
            {"num_passes": 2, "share_cache": False},
            {"num_passes": None, "share_cache": True},
        ):
            with pytest.raises(ValueError, match="is_varloop=true"):
                text.configure_loop(**overrides)
    varloop = _varloop_config(None, False).text_config
    with pytest.raises(ValueError, match="at least 1 pass"):
        varloop.configure_loop(num_passes=0, share_cache=False)


def _serve_args(path: Path, **fields) -> ServerArgs:
    args = ServerArgs(model=str(path), device="cpu", max_model_len=2048)
    for name, value in fields.items():
        setattr(args, name, value)
    return args


def _target_config(path: Path, args: ServerArgs) -> ModelConfig:
    return ModelConfig(
        str(path),
        trust_remote_code=False,
        model_override_args="{}",
        dtype="bfloat16",
        server_args=args,
        context_length=2048,
    )


@pytest.mark.parametrize(
    "num_passes,share_cache,cache_layers",
    [(None, False, 48), (1, False, 24), (4, False, 60), (None, True, 24)],
)
def test_server_args_configure_the_varloop_target(
    tmp_path: Path, num_passes: int | None, share_cache: bool, cache_layers: int
) -> None:
    _varloop_config(None, False).save_pretrained(tmp_path)
    args = _serve_args(
        tmp_path,
        inkling2_loop_num_passes=num_passes,
        inkling2_loop_share_cache=share_cache,
    )
    metadata = _target_config(tmp_path, args)
    text = metadata.hf_text_config
    # The cache recipes and the model read the same text config.
    assert metadata.hf_config.get_text_config() is text
    assert text.num_loop_passes == (3 if num_passes is None else num_passes)
    assert text.loop_share_cache == share_cache
    assert metadata.num_attention_layers == cache_layers


@pytest.mark.parametrize("share_cache", [False, True])
def test_every_visit_checkpoints_its_convolutions_on_its_cache_pages(
    tmp_path: Path, share_cache: bool
) -> None:
    """Each layer visit keeps its own ShortConv checkpoints, on the pages of
    the visit's KV cache layer: every pass of a core layer shares those pages
    under a shared cache."""
    _varloop_config(None, False).save_pretrained(tmp_path)
    args = _serve_args(
        tmp_path,
        inkling2_loop_share_cache=share_cache,
        max_num_seqs=2,
        max_total_tokens=2048,
        attention_backend="triton",
        kv_cache_dtype="bfloat16",
        enable_prefix_caching=False,
        disable_kvstore=True,
    )
    args.mapping.rank = 0
    metadata = _target_config(tmp_path, args)
    text = metadata.hf_text_config
    attn_config = MHAConfig.generate(args, metadata, is_draft=False)
    recipe = InklingRecipe(
        server_args=args,
        model_config=metadata,
        attn_config=attn_config,
        draft_model_config=None,
        draft_attn_config=None,
        cache_budget_bytes=8 << 30,
        probe_batch_rows=None,
        decode_input_tokens=1,
        overlap_schedule_depth=0,
    )
    setup = recipe.setup()
    arena = create_cache_arena(setup.spec, device="cpu", enable_memory_saver=False)
    layers = text.num_cache_layers
    pool = create_cache_pool(
        setup.spec.layer_view(first_layer=0, num_layers=layers),
        attn_config,
        arena,
        num_layers=layers,
        rank=0,
        field_layer_offset=0,
    )
    plan = arena.plan
    checkpoints = set()
    for layer, pass_idx in text.layer_visits:
        conv = text.conv_layer_id(layer, pass_idx)
        cache = text.cache_layer_id(layer, pass_idx)
        prefix = (
            f"layer.{cache}.conv{conv}"
            if share_cache and pass_idx > 0
            else f"layer.{cache}"
        )
        k, v = pool.kvconv_checkpoint_buffers(conv)
        buffers = {
            "kvconv_k": k,
            "kvconv_v": v,
            "attnconv": pool.hiddenconv_checkpoint(conv, "attnconv")[1],
            "mlpconv": pool.hiddenconv_checkpoint(conv, "mlpconv")[1],
        }
        for name, buffer in buffers.items():
            field = plan.field(f"{prefix}.{name}")
            assert buffer.data_ptr() == arena.field(field.field_id).data_ptr()
            # Groups page a plane apart: a checkpoint is its group and offset.
            checkpoints.add((field.group_id, buffer.data_ptr()))
        kv_planes = {plan.field(f"layer.{cache}.{side}").plane_id for side in "kv"}
        assert {
            plan.field(f"{prefix}.kvconv_k").plane_id,
            plan.field(f"{prefix}.kvconv_v").plane_id,
        } == kv_planes
    assert len(checkpoints) == 4 * text.num_conv_layers == 4 * 48


def test_server_args_reject_overrides_of_a_fixed_loop(tmp_path: Path) -> None:
    Inkling2LoopMMConfig(**_looped_checkpoint()).save_pretrained(tmp_path)
    for fields in (
        {"inkling2_loop_num_passes": 2},
        {"inkling2_loop_share_cache": True},
    ):
        with pytest.raises(ValueError, match="is_varloop=true"):
            _target_config(tmp_path, _serve_args(tmp_path, **fields))


def test_speculation_serves_only_the_checkpoint_pass_count(tmp_path: Path) -> None:
    """A DSpark draft taps fixed visits: another pass count fails, the
    checkpoint's own count (and a shared cache) does not, nor does a
    model-free DUMMY draft at another count."""
    config = _varloop_config(None, False)
    args = _serve_args(
        tmp_path, speculative_algorithm="DSPARK", inkling2_loop_num_passes=2
    )
    with pytest.raises(ValueError, match="fixed layer visits"):
        _configure_inkling2_loop(args, config.text_config)
    args.inkling2_loop_num_passes = 3
    args.inkling2_loop_share_cache = True
    _configure_inkling2_loop(args, config.text_config)
    assert config.text_config.num_loop_passes == 3
    assert config.text_config.num_cache_layers == 24
    dummy = _varloop_config(None, False)
    args = _serve_args(
        tmp_path, speculative_algorithm="DUMMY", inkling2_loop_num_passes=2
    )
    _configure_inkling2_loop(args, dummy.text_config)
    assert dummy.text_config.num_loop_passes == 2


def test_loop_config_and_model_type_go_together() -> None:
    with pytest.raises(ValueError, match="model_type inkling2_loop"):
        Inkling2MMConfig(**_looped_checkpoint())
    with pytest.raises(ValueError, match="model_type inkling2_loop"):
        Inkling2LoopMMConfig(**_checkpoint())


@pytest.mark.parametrize(
    "looped, architecture",
    [
        # The first inkling2_loop export kept the non-looped name.
        (True, "Inkling2ForConditionalGeneration"),
        (False, "Inkling2LoopForConditionalGeneration"),
    ],
)
def test_architecture_and_model_type_go_together(
    looped: bool, architecture: str
) -> None:
    values = _looped_checkpoint() if looped else _checkpoint()
    values["architectures"] = [architecture]
    config_cls = Inkling2LoopMMConfig if looped else Inkling2MMConfig
    with pytest.raises(ValueError, match="checkpoints use architectures"):
        config_cls(**values)


def test_looped_architecture_resolves_to_the_inkling2_model() -> None:
    from tokenspeed.runtime.models.inkling2 import Inkling2LoopForConditionalGeneration
    from tokenspeed.runtime.models.registry import ModelRegistry

    model_cls, architecture = ModelRegistry.resolve_model_cls(
        ["Inkling2LoopForConditionalGeneration"]
    )
    assert model_cls is Inkling2LoopForConditionalGeneration
    assert architecture == "Inkling2LoopForConditionalGeneration"


def test_looped_checkpoint_loads_by_model_type(tmp_path: Path) -> None:
    Inkling2LoopMMConfig(**_looped_checkpoint()).save_pretrained(tmp_path)
    config = get_config(str(tmp_path), trust_remote_code=False, revision=None)
    assert type(config) is Inkling2LoopMMConfig
    assert config.model_type == "inkling2_loop"
    assert type(config.text_config) is Inkling2ModelConfig
    assert (
        config.text_config.loop_config
        == _looped_checkpoint()["text_config"]["loop_config"]
    )
    assert config.text_config.num_layer_visits == 48


def test_loop_messages_require_normalized_residual_values() -> None:
    values = _looped_checkpoint()["text_config"]
    values["attn_res_normalize_values"] = False
    with pytest.raises(ValueError, match="attn_res_normalize_values=True"):
        Inkling2ModelConfig(**values)


def test_draft_taps_count_layer_visits() -> None:
    draft = Inkling2MTPConfig.from_target(Inkling2LoopMMConfig(**_looped_checkpoint()))
    assert draft.target_layer_ids == [5, 23, 41, 48]
    assert draft.dspark_target_layer_ids == [5, 23, 41, 48]
    assert draft.dflash_config == {
        "num_target_layers": 48,
        "target_layer_ids": [5, 23, 41],
    }
    assert draft.target_num_hidden_layers == 24


@pytest.mark.parametrize("taps", [[5, 23, 24], [5, 23, 41, 47], [5, 23, 49]])
def test_draft_taps_end_with_the_visit_sentinel(taps: list[int]) -> None:
    values = _looped_checkpoint()
    values["mtp_config"]["target_layer_ids"] = taps
    with pytest.raises(ValueError, match="final-norm sentinel 48"):
        Inkling2MTPConfig.from_target(Inkling2LoopMMConfig(**values))


# ---- a tiny looped model ----

# Eight layers in blocks of two; the core, layers 2-5 (two blocks), runs three
# times, so a pass's message bank holds a stored block and a partial.
TINY_LOOP = {
    "core_start": 2,
    "core_end": 6,
    "num_passes": 3,
    "gate_rank": 4,
    "is_varloop": False,
}
# The same core in a variable-loop checkpoint, which serves other pass counts
# and a shared cache (configure_loop).
TINY_VARLOOP = {**TINY_LOOP, "is_varloop": True}
# configure_loop arguments: the checkpoint's passes, one pass, more passes than
# the checkpoint's, and a shared cache at the default and another count.
LOOP_OVERRIDES = [
    None,
    {"num_passes": 1, "share_cache": False},
    {"num_passes": 4, "share_cache": False},
    {"num_passes": None, "share_cache": True},
    {"num_passes": 2, "share_cache": True},
]
TINY_LOCAL = [0, 1, 2, 4, 6]
WIDTH = 32


def _tiny_config(loop: dict | None) -> Inkling2MMConfig:
    values = _checkpoint()
    values["text_config"].update(
        hidden_size=WIDTH,
        num_hidden_layers=8,
        vocab_size=64,
        unpadded_vocab_size=60,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        d_rel=2,
        rel_extent=8,
        local_layer_ids=list(TINY_LOCAL),
        swa_head_dim=8,
        swa_num_attention_heads=4,
        swa_num_key_value_heads=2,
        sliding_window_size=8,
        n_layers_per_attn_res_block=2,
        n_routed_experts=4,
        num_experts_per_tok=2,
        intermediate_size=32,
        loop_config=loop,
    )
    values["vision_config"] = None
    values["mtp_config"] = None
    values["eos_token_id"] = 2
    config_cls = Inkling2MMConfig if loop is None else Inkling2LoopMMConfig
    values.update(
        model_type=config_cls.model_type, architectures=[config_cls.architecture]
    )
    return config_cls(**values)


def _manifest(config: Inkling2MMConfig) -> Inkling2QuantConfig:
    text = config.text_config
    bf16 = [
        "model.llm.embed.weight",
        "model.llm.embed_norm.weight",
        "model.llm.norm.weight",
    ]
    fp32 = ["model.llm.attn_res.output_query"]
    if text.loop_core_layers:
        fp32 += [
            "model.llm.middle_cycle_cross_pass_queries",
            "model.llm.middle_cycle_cross_pass_kv_queries",
        ]
    experts = {"nvfp4": [], "mxfp8": []}
    for layer in range(text.num_hidden_layers):
        prefix = f"model.llm.layers.{layer}"
        local = layer in text.local_layer_ids
        fp32 += [f"model.llm.attn_res.mlp_queries.{layer}", f"{prefix}.mlp.gate.bias"]
        if layer:
            fp32.append(f"model.llm.attn_res.attn_queries.{layer - 1}")
        bf16 += [
            f"{prefix}.attn.{projection}.weight"
            for projection in (
                "wq_du",
                "wk_dv",
                "wv_dv",
                "wr_du",
                "gate_attn",
                "attn_sink_proj",
                "wo_ud",
            )
        ]
        bf16 += [
            f"{prefix}.attn.{leaf}"
            for leaf in ("q_l2_clip.scale", "rel_l2_clip.scale", "rel_logits_proj.proj")
        ]
        bf16 += [
            f"{prefix}.{leaf}.weight"
            for leaf in ("mlp.gate", "attn_sconv", "mlp_sconv", "mlp_norm")
        ]
        norms = (
            ("attn_norm",)
            if local
            else ("q_input_norm", "k_input_norm", "v_input_norm")
        )
        bf16 += [f"{prefix}.{norm}.weight" for norm in norms]
        for stream in ("k_sconv", "v_sconv"):
            leaves = ("weight",) if local else ("static_w", "weight_proj")
            bf16 += [f"{prefix}.attn.{stream}.{leaf}" for leaf in leaves]
        if layer in text.loop_core_layers:
            bf16 += [
                f"{prefix}.controllers.{name}.{leaf}_weight"
                for name in ("attn", "k", "v", "mlp")
                for leaf in ("down", "gain", "forget")
            ]
        experts["nvfp4" if layer == 0 else "mxfp8"] += [
            f"{prefix}.mlp.experts.{leaf}_weight" for leaf in ("w13", "w2")
        ]
    groups = [_group(compute, names) for compute, names in experts.items()]
    groups += [_group("bfloat16", bf16), _group("float32", fp32)]
    output = _group("bfloat16", ["model.llm.unembed.weight"])
    output["fwd_output_dtype"] = "float32"
    groups.append(output)
    return Inkling2QuantConfig({"model": {"parameter_groups": groups}})


def _unloaded(
    loop: dict | None, overrides: dict | None = None
) -> Inkling2ForConditionalGeneration:
    config = _tiny_config(loop)
    if overrides is not None:
        config.text_config.configure_loop(**overrides)
    return Inkling2ForConditionalGeneration(
        config=config,
        mapping=_mapping(),
        quant_config=_manifest(config),
        is_multimodal_active=False,
        mm_attention_backend=None,
    )


def _checkpoint_weights(
    model: Inkling2ForConditionalGeneration, seed: int
) -> list[tuple[str, torch.Tensor]]:
    """Random checkpoint tensors for every target parameter, under their
    checkpoint names. The same seed gives the same physical layers whether
    or not the model loops."""
    projections = ("wq_du", "wk_dv", "wv_dv", "wr_du", "gate_attn", "attn_sink_proj")
    weights = []
    for name, param in model.named_parameters():
        name = name.replace("model.", "model.llm.", 1)
        name = name.replace("model.llm.embed_tokens.weight", "model.llm.embed.weight")
        name = name.replace("lm_head.weight", "model.llm.unembed.weight")
        # Each tensor draws from its own name's stream.
        generator = torch.Generator().manual_seed(seed + sum(map(ord, name)) * 131)
        if name.endswith(".attn.qkvr.weight"):
            for projection, size in zip(projections, (32, 16, 16, 8, 32, 4)):
                value = torch.randn((size, WIDTH), generator=generator) * 0.3
                weights.append(
                    (name.replace(".qkvr.", f".{projection}."), value.bfloat16())
                )
            continue
        for leaf in ("w13", "w2"):
            for suffix in ("scale", "amax", "outer_scale_denominator"):
                name = name.replace(
                    f".experts.{leaf}_{suffix}", f".experts.{leaf}_weight_{suffix}"
                )
        if param.dtype in (torch.float32, torch.bfloat16):
            value = torch.randn(param.shape, generator=generator)
            if name.endswith("norm.weight"):
                value = value.abs() * 0.5 + 0.75
            elif ".controllers." in name:
                # Trained-like scale: unit-variance projections.
                value = value * param.shape[1] ** -0.5
            value = value.to(param.dtype)
        else:
            value = torch.ones_like(param)
        weights.append((name, value))
    return weights


def _loaded(
    loop: dict | None, seed: int, overrides: dict | None = None
) -> Inkling2ForConditionalGeneration:
    model = _unloaded(loop, overrides)
    model.load_weights(_checkpoint_weights(model, seed))
    return model


@pytest.mark.parametrize(
    "loop,overrides",
    [(TINY_LOOP, None), *((TINY_VARLOOP, o) for o in LOOP_OVERRIDES[1:])],
)
def test_looped_model_builds_per_pass_caches(
    loop: dict, overrides: dict | None
) -> None:
    """One paged attention a KV cache layer (one for all passes under a
    shared cache), and every pass its own convolution state."""
    model = _unloaded(loop, overrides)
    text = model.text_config
    for layer in model.model.layers:
        passes = range(text.layer_passes(layer.layer_id))
        ids = tuple(text.cache_layer_id(layer.layer_id, p) for p in passes)
        conv_ids = tuple(text.conv_layer_id(layer.layer_id, p) for p in passes)
        attention = [layer.attn.attn, *(layer.attn.loop_attn or ())]
        assert tuple(a.layer_id for a in attention) == tuple(dict.fromkeys(ids))
        assert tuple(layer.attn.pass_attention(p).layer_id for p in passes) == ids
        for conv in (
            layer.attn.k_sconv,
            layer.attn.v_sconv,
            layer.attn_sconv,
            layer.mlp_sconv,
        ):
            assert conv.conv_layer_ids == conv_ids
        is_local = layer.layer_id in TINY_LOCAL
        assert all(a.sliding_window_size == (7 if is_local else -1) for a in attention)
        assert (layer.controllers is None) == (layer.layer_id not in range(2, 6))
    plain = _unloaded(None)
    assert all(layer.attn.loop_attn is None for layer in plain.model.layers)
    assert all(layer.controllers is None for layer in plain.model.layers)
    assert plain.model.middle_cycle_cross_pass_queries is None


def test_looped_load_requires_every_loop_tensor() -> None:
    model = _unloaded(TINY_LOOP)
    weights = _checkpoint_weights(model, 3)
    loaded = model.load_weights(weights)
    assert loaded == {name for name, _ in model.named_parameters()}
    controller = model.model.layers[3].controllers["v"]
    expected = dict(weights)
    assert torch.equal(
        controller.down_weight,
        expected["model.llm.layers.3.controllers.v.down_weight"],
    )
    assert torch.equal(
        controller.gate_weight,
        torch.cat((controller.gain_weight, controller.forget_weight)),
    )
    assert model.model.middle_cycle_cross_pass_queries.dtype == torch.float32
    for missing in (
        "model.llm.layers.5.controllers.mlp.gain_weight",
        "model.llm.middle_cycle_cross_pass_kv_queries",
    ):
        with pytest.raises(ValueError, match="Missing checkpoint tensors"):
            _unloaded(TINY_LOOP).load_weights(
                [(n, w) for n, w in weights if n != missing]
            )


@pytest.mark.parametrize(
    "mode,interval,loop,rejects",
    [
        ("prefill", 1, TINY_LOOP, True),
        ("prefill", 0, TINY_LOOP, False),
        ("decode", 1, TINY_LOOP, False),
        ("prefill", 1, None, False),
    ],
)
def test_layerwise_cache_transfer_is_rejected_at_construction(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    interval: int,
    loop: dict | None,
    rejects: bool,
) -> None:
    """A looped checkpoint on a PD prefill server with layerwise transfer
    fails when the model is built, not on its first forward."""
    monkeypatch.setitem(global_server_args_dict, "disaggregation_mode", mode)
    monkeypatch.setitem(
        global_server_args_dict, "disaggregation_layerwise_interval", interval
    )
    if rejects:
        with pytest.raises(
            NotImplementedError, match="--disaggregation-layerwise-interval 0"
        ):
            _unloaded(loop)
    else:
        assert _unloaded(loop).text_config.loop_config == loop


def test_unlooped_model_rejects_loop_tensors() -> None:
    model = _unloaded(None)
    extra = ("model.llm.layers.2.controllers.attn.down_weight", torch.zeros(4, 64))
    with pytest.raises(ValueError, match="No checkpoint precision|Unrecognized"):
        model.load_weights([*_checkpoint_weights(model, 3), extra])


# ---- the torch reference of the looped forward ----


def _rms(x: torch.Tensor) -> torch.Tensor:
    x = x.double()
    return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + EPS)


def _block_attn_res(sources: list[torch.Tensor], query: torch.Tensor) -> torch.Tensor:
    """The reference depth attention with RMS-normalized keys and values."""
    values = torch.stack([x.double() for x in sources])
    inv_rms = torch.rsqrt(values.square().mean(dim=-1) + EPS)
    scale = values.shape[-1] ** -0.5
    logits = torch.einsum("ntd,d,nt->tn", values, query.double() * scale, inv_rms)
    weights = torch.softmax(logits, dim=-1)
    return torch.einsum("tn,ntd->td", weights, values * inv_rms.unsqueeze(-1)).float()


def _ref_norm(x: torch.Tensor, weight: torch.Tensor | None) -> torch.Tensor:
    x = x.float()
    out = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + EPS)
    if weight is not None:
        out = out * weight.float()
    return out.bfloat16()


def _ref_controller(
    controller: Inkling2MessageController,
    x: torch.Tensor,
    message: torch.Tensor,
    normalized: torch.Tensor | None = None,
) -> torch.Tensor:
    if normalized is None:
        normalized = _ref_norm(torch.stack((x, message), dim=1), None).flatten(1)
    hidden = (normalized.float() @ controller.down_weight.float().T).bfloat16()
    gain = hidden.float() @ controller.gain_weight.float().T
    forget = hidden.float() @ controller.forget_weight.float().T
    return (1.0 - torch.tanh(forget)) * x + torch.tanh(gain) * message


def _round_sum(*terms: torch.Tensor) -> torch.Tensor:
    total = terms[0].float()
    for term in terms[1:]:
        total = total + term.float()
    return total.bfloat16()


class _RefState:
    """The reference residual state: completed blocks and the BF16 partial."""

    def __init__(self, partial: torch.Tensor, blocks: list[torch.Tensor]) -> None:
        self.partial = partial
        self.blocks = blocks


def _attention_kernel(sources: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """A fixed BF16 function of the Q, K and V sources standing in for the
    GPU attention."""
    q, k, v = sources if len(sources) == 3 else sources * 3
    return torch.tanh(q.float() * 0.7 + k.float() * 0.2 - v.float() * 0.4).bfloat16()


def _moe_kernel(x: torch.Tensor) -> torch.Tensor:
    return torch.sin(x.float() * 1.3).bfloat16()


def _reference_forward(
    model: Inkling2ForConditionalGeneration, hidden: torch.Tensor
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """The looped reference forward (its _run_loop) with the same kernels;
    returns the final normalized hidden and the partial entering each visit."""
    text, llm = model.text_config, model.model
    attn_res, block = llm.attn_res, text.n_layers_per_attn_res_block
    core = text.loop_core_layers
    queries = llm.middle_cycle_cross_pass_queries
    kv_queries = llm.middle_cycle_cross_pass_kv_queries
    state = _RefState(hidden, [])
    taps: list[torch.Tensor] = []
    entry: _RefState | None = None
    bank: list[torch.Tensor] | None = None

    def message(layer: int, name: str) -> torch.Tensor:
        i = layer - core.start
        rows = queries if name in ("attn", "mlp") else kv_queries
        return _block_attn_res(bank, rows[2 * i + int(name in ("mlp", "v"))])

    for layer_id, pass_idx in text.layer_visits:
        if core and layer_id == core.start:
            if pass_idx == 0:
                entry = _RefState(state.partial, list(state.blocks))
            else:
                bank = [*state.blocks[len(entry.blocks) + 1 :], state.partial]
                state = _RefState(entry.partial, list(entry.blocks))
        elif core and layer_id == core.stop:
            bank = None
        taps.append(state.partial)
        layer = llm.layers[layer_id]
        local = layer_id in TINY_LOCAL
        if not state.blocks:
            streams = (state.partial.float(),) * (1 if local else 3)
        else:
            query = attn_res.attn_query(layer_id).view(-1, WIDTH)
            streams = tuple(
                _block_attn_res([*state.blocks, state.partial], q) for q in query
            )
        norms = layer.input_norms()
        if bank is not None:
            if local:
                streams, norms = streams * 3, norms * 3
            streams = tuple(
                _ref_controller(layer.controllers[name], x, message(layer_id, name))
                for name, x in zip(("attn", "k", "v"), streams)
            )
        attention = _attention_kernel(
            tuple(_ref_norm(x, n.weight) for n, x in zip(norms, streams))
        )
        boundary = layer_id % block == 0
        if boundary:
            sources = [*state.blocks, state.partial, attention]
        else:
            sources = [*state.blocks, (state.partial.float() + attention.float())]
        mlp_input = _block_attn_res(sources, attn_res.mlp_queries[layer_id])
        if bank is not None:
            mlp_input = _ref_controller(
                layer.controllers["mlp"], mlp_input, message(layer_id, "mlp")
            )
        mlp = _moe_kernel(_ref_norm(mlp_input, layer.mlp_norm.weight))
        if boundary:
            state.blocks.append(state.partial)
            state.partial = _round_sum(attention, mlp)
        else:
            state.partial = _round_sum(state.partial, attention, mlp)
    output = _block_attn_res([*state.blocks, state.partial], attn_res.output_query)
    return _ref_norm(output, llm.norm.weight), taps


class _CpuMailbox:
    """The decode chain's mailbox on one CPU rank: the rows a producer
    publishes into a parity, until its consumer reads them (multicast device
    memory on GPUs)."""

    def __init__(self) -> None:
        self.slots: dict[int, torch.Tensor] = {}

    def publish(self, parity: int, rows: torch.Tensor) -> None:
        assert parity not in self.slots, f"parity {parity} published twice"
        self.slots[parity] = rows

    def take(self, parity: int) -> torch.Tensor:
        return self.slots.pop(parity)


class _CpuKernels:
    """Stand-ins for the GPU-only attention, MoE and convolution kernels that
    record which cache layer each call addresses. Published outputs go into a
    ``_CpuMailbox``; a convolution reading the mailbox records its cache layer
    in ``mailbox``."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.attention: list[tuple[int, int]] = []
        self.conv: list[tuple[str, int]] = []
        self.mailbox: list[tuple[str, int]] = []
        kernels = self

        def attention(layer, streams, ctx, tau, *, pass_idx, publish_into, **kwargs):
            kernels.attention.append((layer.layer_id, layer.cache_layer_ids[pass_idx]))
            output = _attention_kernel(streams)
            if publish_into is None:
                return output
            publish_into.mailbox.publish(publish_into.parity, output)
            return None

        def conv(module, x, ctx, *, pass_idx, all_reduce_group, token_range):
            kernels.conv.append((module.stream.name, module.conv_layer_ids[pass_idx]))
            return x

        def conv_from_mailbox(module, source, tokens, ctx, pass_idx):
            kernels.mailbox.append(
                (module.stream.name, module.conv_layer_ids[pass_idx])
            )
            return source.mailbox.take(source.parity)[:tokens]

        def moe(module, x, ctx, *, reduce, fold_into):
            output = _moe_kernel(x)
            if fold_into is None:
                return output
            fold_into.mailbox.publish(fold_into.parity, output)
            return None

        monkeypatch.setattr(Inkling2Attention, "forward", attention)
        monkeypatch.setattr(Inkling2ShortConvolution, "forward", conv)
        monkeypatch.setattr(Inkling2ShortConvolution, "from_mailbox", conv_from_mailbox)
        monkeypatch.setattr(Inkling2MoE, "forward", moe)


class _Backend:
    """The attention backend fields the decoder reads on CPU."""

    enable_layerwise_cache_ready = False


class _DecodeConvMetadata:
    num_extends = 0


class _VerifyBackend(_Backend):
    """A target verify of three tokens: its convolutions publish the accepted
    window after acceptance, the batch the decode chain's mailbox serves
    (``sconv_decodes_without_publish``)."""

    conv_decode_metadata = _DecodeConvMetadata()
    conv_spec_num_tokens = 3
    is_draft = False


def _context(rows: int, mode: ForwardMode, backend: _Backend) -> ForwardContext:
    return ForwardContext(
        attn_backend=backend,
        token_to_kv_pool=None,
        bs=rows if mode.is_decode() else 1,
        num_extends=0 if mode.is_decode() else 1,
        input_num_tokens=rows,
        forward_mode=mode,
        output_layout=ForwardOutputLayout(0, 0, rows, 1),
        capture_hidden_mode=CaptureHiddenMode.FULL,
    )


def _run(
    model: Inkling2ForConditionalGeneration,
    hidden: torch.Tensor,
    mode: ForwardMode,
    backend: _Backend,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    model.model.layers_to_capture = list(range(model.text_config.num_layer_visits))
    # CPU rows: block-set partials run inline instead of on a CUDA stream.
    model.model.attnres_stream = None
    positions = torch.arange(hidden.shape[0])
    with torch.no_grad():
        output, taps = model.model(
            positions, positions, _context(len(hidden), mode, backend), hidden
        )
    return output, taps[:-1]


# Up to 256 tokens the residual state sweeps block sets; more mix each query.
@pytest.mark.parametrize(
    "tokens,mode", [(3, ForwardMode.DECODE), (300, ForwardMode.EXTEND)]
)
@pytest.mark.parametrize(
    "loop,overrides",
    [(TINY_LOOP, None), *((TINY_VARLOOP, o) for o in LOOP_OVERRIDES[1:])],
)
def test_loop_forward_matches_the_reference(
    monkeypatch: pytest.MonkeyPatch,
    tokens: int,
    mode: ForwardMode,
    loop: dict,
    overrides: dict | None,
) -> None:
    """At the checkpoint's pass count and at overridden ones (1 runs the core
    once with no messages), with and without a shared KV cache."""
    kernels = _CpuKernels(monkeypatch)
    model = _loaded(loop, seed=11, overrides=overrides)
    hidden = (
        torch.randn((tokens, WIDTH), generator=torch.Generator().manual_seed(5))
    ).bfloat16()
    output, taps = _run(model, hidden, mode, _Backend())
    with torch.no_grad():
        expected, expected_taps = _reference_forward(model, hidden)
    text = model.text_config
    passes = text.num_loop_passes
    assert len(taps) == len(expected_taps) == 8 + 4 * (passes - 1)
    for visit, (got, want) in enumerate(zip(taps, expected_taps)):
        torch.testing.assert_close(
            got, want, atol=2e-2, rtol=2e-2, msg=f"visit {visit}"
        )
    torch.testing.assert_close(output, expected, atol=3e-2, rtol=3e-2)
    # Each visit addresses its KV cache layer (its decoder layer's under a
    # shared cache) and its own convolution state, in visit order.
    visits = text.layer_visits
    kv_ids = [text.cache_layer_id(layer, p) for layer, p in visits]
    conv_ids = [text.conv_layer_id(layer, p) for layer, p in visits]
    assert sorted(conv_ids) == list(range(len(visits)))
    if text.loop_share_cache:
        assert kv_ids == [layer for layer, _ in visits]
    else:
        assert kv_ids == conv_ids
    assert [cache for _, cache in kernels.attention] == kv_ids
    assert [conv for stream, conv in kernels.conv if stream == "ATTN"] == conv_ids
    assert [conv for stream, conv in kernels.conv if stream == "MLP"] == conv_ids


@pytest.mark.parametrize("tokens", [3, 300])
def test_first_pass_matches_the_unlooped_layers_bit_for_bit(
    monkeypatch: pytest.MonkeyPatch, tokens: int
) -> None:
    """The prelude and the first pass of the core run the ordinary fused
    path: their visits equal the same layers of the unlooped model."""
    _CpuKernels(monkeypatch)
    looped = _loaded(TINY_LOOP, seed=13)
    plain = _loaded(None, seed=13)
    hidden = torch.randn(
        (tokens, WIDTH), generator=torch.Generator().manual_seed(7)
    ).bfloat16()
    mode = ForwardMode.DECODE if tokens <= 256 else ForwardMode.EXTEND
    _, looped_taps = _run(looped, hidden, mode, _Backend())
    _, plain_taps = _run(plain, hidden, mode, _Backend())
    first_pass = TINY_LOOP["core_end"]
    for visit in range(first_pass):
        assert torch.equal(looped_taps[visit], plain_taps[visit]), visit
    # The second pass restarts from the core's entry but reads messages.
    assert not torch.equal(
        looped_taps[first_pass + 1], plain_taps[TINY_LOOP["core_start"] + 1]
    )


def test_decode_chain_mailbox_matches_the_all_reduce_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the decode chain's mailbox, every visit publishes its attention
    output projection and MoE route fold and convolves them out of the
    mailbox: each later pass's convolutions read their own pass's state, the
    MoE input still takes its message, and the outputs equal the all-reduce
    path's bit for bit."""
    kernels = _CpuKernels(monkeypatch)
    model = _loaded(TINY_LOOP, seed=17)
    hidden = torch.randn(
        (3, WIDTH), generator=torch.Generator().manual_seed(9)
    ).bfloat16()
    expected, expected_taps = _run(model, hidden, ForwardMode.DECODE, _VerifyBackend())
    assert not kernels.mailbox
    kernels.conv.clear()
    # The mailbox replaces the MoE's EP all-reduce, which needs an EP group.
    monkeypatch.setattr(Inkling2MoE, "all_reduces", lambda module, tokens: True)
    mailbox = _CpuMailbox()
    model.model.decode_mailbox = mailbox
    output, taps = _run(model, hidden, ForwardMode.DECODE, _VerifyBackend())
    text = model.text_config
    visit_ids = [text.conv_layer_id(layer, p) for layer, p in text.layer_visits]
    assert kernels.conv == []
    assert kernels.mailbox == [
        (stream, cache) for cache in visit_ids for stream in ("ATTN", "MLP")
    ]
    assert not mailbox.slots
    assert len(taps) == len(expected_taps) == len(visit_ids)
    for visit, (got, want) in enumerate(zip(taps, expected_taps)):
        assert torch.equal(got, want), visit
    assert torch.equal(output, expected)


# ---- the residual state across passes ----


def _state_case(core: range, num_layers: int, tokens: int):
    """Random sublayer outputs for every visit of a looped stack in blocks of
    two, and the residual owner with random pseudo-queries."""
    generator = torch.Generator().manual_seed(tokens + core.start * 17)
    local = [i for i in range(num_layers) if i % 2 == 0]
    attn_res = AttentionResidual(num_layers, WIDTH, 2, local, True)
    with torch.no_grad():
        for param in attn_res.parameters():
            param.copy_(torch.randn(param.shape, generator=generator))
    attn_res.derive_score_weights()
    passes = 3
    visits = (
        [(layer, 0) for layer in range(core.start)]
        + [(layer, p) for p in range(passes) for layer in core]
        + [(layer, 0) for layer in range(core.stop, num_layers)]
    )
    outputs = [
        tuple(
            (torch.randn((tokens, WIDTH), generator=generator) * scale).bfloat16()
            for scale in (0.5, 0.8)
        )
        for _ in visits
    ]
    embedding = torch.randn((tokens, WIDTH), generator=generator).bfloat16()
    return attn_res, local, visits, outputs, embedding


@pytest.mark.parametrize(
    "core,num_layers",
    [(range(2, 6), 8), (range(0, 4), 6), (range(2, 4), 4), (range(4, 8), 8)],
)
@pytest.mark.parametrize("tokens", [3, 300])
def test_state_rewinds_to_the_loop_entry_and_banks_each_pass(
    core: range, num_layers: int, tokens: int
) -> None:
    """Every mixture of every visit and every message bank against the
    reference state handling: snapshot at the core's entry, bank the blocks a
    pass wrote, restart the next pass from the snapshot."""
    attn_res, local, visits, outputs, embedding = _state_case(core, num_layers, tokens)
    state = AttentionResidualState(attn_res, embedding, None, None)
    ref = _RefState(embedding, [])
    entry = ref_entry = None
    for (layer, pass_idx), (attention, mlp) in zip(visits, outputs, strict=True):
        if layer == core.start:
            if pass_idx == 0:
                entry = state.loop_entry(layer)
                ref_entry = _RefState(ref.partial, list(ref.blocks))
            else:
                stored, partial = state.loop_bank(entry)
                ref_bank = [*ref.blocks[len(ref_entry.blocks) + 1 :], ref.partial]
                assert len(stored) == len(ref_bank) - 1
                for got, want in zip([*stored, partial], ref_bank, strict=True):
                    assert torch.equal(got, want)
                state.rewind_loop(entry)
                ref = _RefState(ref_entry.partial, list(ref_entry.blocks))
        mixtures = state.attention_mixtures(layer)
        if not ref.blocks:
            expected = (ref.partial.float(),) * (1 if layer in local else 3)
        else:
            query = attn_res.attn_query(layer).view(-1, WIDTH)
            expected = tuple(
                _block_attn_res([*ref.blocks, ref.partial], q) for q in query
            )
        assert len(mixtures) == len(expected)
        for got, want in zip(mixtures, expected):
            assert got.dtype == torch.float32
            torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)
        boundary = layer % 2 == 0
        sources = (
            [*ref.blocks, ref.partial, attention]
            if boundary
            else [*ref.blocks, ref.partial.float() + attention.float()]
        )
        torch.testing.assert_close(
            state.mlp_mixture(layer, attention),
            _block_attn_res(sources, attn_res.mlp_queries[layer]),
            atol=1e-5,
            rtol=1e-5,
        )
        state.finalize_layer(layer, attention, mlp)
        if boundary:
            ref.blocks.append(ref.partial)
            ref.partial = _round_sum(attention, mlp)
        else:
            ref.partial = _round_sum(ref.partial, attention, mlp)
        assert torch.equal(state.partial, ref.partial)


def test_loop_entry_sits_at_a_block_boundary() -> None:
    attn_res, *_, embedding = _state_case(range(2, 6), 8, 3)
    state = AttentionResidualState(attn_res, embedding, None, None)
    with pytest.raises(ValueError, match="block boundary"):
        state.loop_entry(1)


# ---- messages and controllers ----


def _controller(generator: torch.Generator, width: int, rank: int):
    """A controller whose BF16 low-rank projection is exact in FP32: each
    ``down`` row holds four power-of-two entries, so every GEMM summation
    order rounds the same BF16 ``h``."""
    controller = Inkling2MessageController(width, rank)
    down = torch.zeros((rank, 2 * width))
    for row in range(rank):
        columns = torch.randperm(2 * width, generator=generator)[:4]
        signs = torch.randint(0, 2, (4,), generator=generator) * 2 - 1
        down[row, columns] = signs * 2.0 ** -torch.randint(
            0, 3, (4,), generator=generator
        )
    with torch.no_grad():
        controller.down_weight.copy_(down.bfloat16())
        for param in (controller.gain_weight, controller.forget_weight):
            param.copy_(
                (torch.randn(param.shape, generator=generator) * rank**-0.5).bfloat16()
            )
    controller.derive_gate_weight()
    return controller


def _device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires an admitted GPU")
    return torch.device(name)


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
@pytest.mark.parametrize(
    "tokens,width,stored", [(5, 2304, 1), (300, 2304, 1), (4, 64, 0)]
)
def test_message_read_matches_the_reference_depth_attention(
    device_name: str, tokens: int, width: int, stored: int
) -> None:
    """Softmax over the bank's sources of RMS-normalized keys, mixing
    RMS-normalized values, with each input's own pseudo-query."""
    device = _device(device_name)
    generator = torch.Generator().manual_seed(tokens + width)
    bank = [
        (torch.randn((tokens, width), generator=generator) * (i + 2)).bfloat16()
        for i in range(stored + 1)
    ]
    queries = torch.randn((3, 4, width), generator=generator)
    messages = Inkling2LoopMessages(
        (
            torch.stack(bank[:-1]).to(device)
            if stored
            else bank[0].new_empty((0, tokens, width)).to(device)
        ),
        bank[-1].to(device),
        (queries * width**-0.5).to(device),
        EPS,
    )
    for layer in range(3):
        got = messages.read(layer, 0, 4)
        for index in range(4):
            torch.testing.assert_close(
                got[index].cpu(),
                _block_attn_res(bank, queries[layer, index]),
                atol=2e-6,
                rtol=2e-6,
            )
        tail = messages.read(layer, 3, 1)[0]
        torch.testing.assert_close(tail.cpu(), got[3].cpu(), atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
@pytest.mark.parametrize(
    "tokens,width,rank", [(5, 2304, 64), (300, 2304, 64), (3, 64, 8)]
)
def test_controller_matches_the_reference_math(
    device_name: str, tokens: int, width: int, rank: int
) -> None:
    """``(1 - tanh(forget)) * x + tanh(gain) * message`` from the BF16
    low-rank projection of the normalized pair."""
    device = _device(device_name)
    generator = torch.Generator().manual_seed(tokens * 3 + rank)
    controller = _controller(generator, width, rank)
    x = torch.randn((tokens, width), generator=generator) * 4
    message = torch.randn((tokens, width), generator=generator)
    # The pair's BF16 RMSNorm is the norm kernel's contract, which may round
    # a GPU row one ulp from the CPU statement; the reference takes the
    # kernel's rows, so the comparison isolates the projections and the gate.
    pair = rmsnorm_precision(
        torch.stack((x, message), dim=1).to(device),
        None,
        EPS,
        torch.bfloat16,
        mean_by_reciprocal=False,
    ).cpu()
    reference_pair = _ref_norm(torch.stack((x, message), dim=1), None)
    torch.testing.assert_close(pair, reference_pair, atol=2**-6, rtol=2**-7)
    expected = _ref_controller(controller, x, message, normalized=pair.flatten(1))
    got = controller.to(device)(x.to(device), message.to(device)).cpu()
    assert got.dtype == torch.float32
    # Only the K=rank FP32 gate sums may associate differently.
    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_gated_inputs_match_the_controllers_and_norms(device: str) -> None:
    """One fused controller op gives each input's controller and RMSNorm,
    up to one BF16 step from its projections' summation order."""
    from tokenspeed.runtime.layers.attention_residual import PrecisionRMSNorm
    from tokenspeed.runtime.models.inkling2_loop import (
        LOOP_INPUTS,
        Inkling2LoopMessages,
        Inkling2MessageController,
    )

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires an admitted GPU")
    hidden, rank, tokens = 768, 64, 5
    generator = torch.Generator().manual_seed(3)
    controllers = torch.nn.ModuleDict()
    for name in LOOP_INPUTS:
        controller = Inkling2MessageController(hidden, rank)
        with torch.no_grad():
            controller.down_weight.copy_(
                torch.randn((rank, 2 * hidden), generator=generator) * 0.03
            )
            controller.gain_weight.copy_(
                torch.randn((hidden, rank), generator=generator) * 0.05
            )
            controller.forget_weight.copy_(
                torch.randn((hidden, rank), generator=generator) * 0.05
            )
        controller.derive_gate_weight()
        controllers[name] = controller
    norms = tuple(
        PrecisionRMSNorm(hidden, 1e-6, torch.bfloat16, True, False) for _ in range(3)
    )
    for norm in norms:
        with torch.no_grad():
            norm.weight.copy_(1 + 0.1 * torch.randn((hidden,), generator=generator))
    controllers.to(device)
    norms = tuple(norm.to(device) for norm in norms)
    messages = Inkling2LoopMessages(
        torch.randn((3, tokens, hidden), generator=generator).to(device),
        torch.randn((tokens, hidden), generator=generator).bfloat16().to(device),
        (torch.randn((2, 4, hidden), generator=generator) * hidden**-0.5).to(device),
        1e-6,
    )
    destinations = tuple(
        torch.randn((tokens, hidden), generator=generator).to(device) for _ in range(3)
    )
    for first, count in ((0, 3), (3, 1)):
        inputs = destinations[:count]
        expected = tuple(
            norm(x)
            for norm, x in zip(
                norms[:count], messages.gate(1, first, inputs, controllers)
            )
        )
        actual = messages.gate_normalized(1, first, inputs, controllers, norms[:count])
        for a, e in zip(actual, expected, strict=True):
            torch.testing.assert_close(a.float(), e.float(), rtol=2**-7, atol=2**-7)

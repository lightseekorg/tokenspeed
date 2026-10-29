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

"""EAGLE-4 llama draft: the low-rank candidate head and the static trees
ranking it (on a GPU); the draft config check, the strict checkpoint loader,
and the Nemotron-style draft layer (NoPE, the ``eh_proj`` input merge and the
latent squared-ReLU MoE)."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import pytest
import torch
from transformers import LlamaConfig

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from ci_system.ci_register import register_cuda_ci

from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.drafter.eagle_harness.drafter import Eagle4
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.execution.input_buffer import InputBuffers
from tokenspeed.runtime.execution.runtime_states import RuntimeStates
from tokenspeed.runtime.execution.tree_spec import TreeSpec, TreeSpecConfig
from tokenspeed.runtime.layers.attention.backends.paged.router import (
    CacheGroupRouter,
)
from tokenspeed.runtime.layers.logits_processor import LogitsMetadata
from tokenspeed.runtime.layers.moe import utils as moe_utils
from tokenspeed.runtime.layers.paged_attention import PagedAttention
from tokenspeed.runtime.models.llama_eagle4 import (
    CandidateHeads,
    LlamaForCausalLMEagle4,
    validate_draft_config,
)
from tokenspeed.runtime.utils.env import global_server_args_dict

register_cuda_ci(
    est_time=5,
    suite="runtime-1gpu",
    disabled_on_runners=["amd-*"],
    disabled_on_runners_reason="no AMD MoE kernel for the latent MoE's relu2 experts",
)

_HIDDEN = 32


def _build_model(
    monkeypatch: pytest.MonkeyPatch, config: LlamaConfig, device: str = "cuda"
) -> LlamaForCausalLMEagle4:
    mapping = Mapping(rank=0, world_size=1)
    monkeypatch.setitem(global_server_args_dict, "ep_num_redundant_experts", 0)
    monkeypatch.setitem(global_server_args_dict, "max_model_len", 256)
    monkeypatch.setitem(global_server_args_dict, "mapping", mapping)
    monkeypatch.setitem(global_server_args_dict, "comm_fusion_max_num_tokens", 256)
    # The engine's default --moe-backend (an earlier test may have left another one set).
    monkeypatch.setattr(moe_utils, "MOE_BACKEND", moe_utils.MoeBackend.AUTO)

    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device(device):
            return LlamaForCausalLMEagle4(config, mapping)
    finally:
        torch.set_default_dtype(old_dtype)


@dataclass(frozen=True)
class _DraftModelConfig:
    requires_request_token_history: bool


class _DraftRunner:
    """The draft ModelRunner's parts a drafter reads, around a model built in
    place (a ModelRunner loads its model from a checkpoint)."""

    def __init__(self, model: LlamaForCausalLMEagle4) -> None:
        self.device = torch.device("cuda")
        self.model = model
        self.mapping = Mapping(rank=0, world_size=1)
        self.model_config = _DraftModelConfig(requires_request_token_history=False)


def _static_tree_drafter(model: LlamaForCausalLMEagle4, topk: int) -> Eagle4:
    """An EAGLE-4 drafter over ``model`` bound to a static tree of ``topk``
    children per node, on an unbound draft router."""
    max_bs, steps = 4, 3
    nodes = (steps - 1) * topk
    drafter = Eagle4(
        spec_num_tokens=nodes,
        spec_num_steps=steps,
        draft_model_runner=_DraftRunner(model),
        attn_backend=CacheGroupRouter(
            None,
            is_draft=True,
            spec_num_tokens=1,
            device="cuda",
            consumed_group_ids=None,
        ),
        runtime_states=RuntimeStates(
            req_pool_size=max_bs,
            vocab_size=model.config.vocab_size,
            output_length=nodes,
            device="cuda",
        ),
        input_buffers=InputBuffers(
            max_bs=max_bs,
            max_num_tokens=max_bs * nodes,
            state_write_padding_pool_index=max_bs,
        ),
        vocab_size=model.config.vocab_size,
    )
    drafter.bind_tree(
        TreeSpec(
            TreeSpecConfig(topk=topk, num_steps=steps, num_nodes=nodes),
            max_bs=max_bs,
            device=drafter.device,
        )
    )
    return drafter


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_eagle4_candidate_head_scores_the_low_rank_top_candidates_exactly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Candidates are the top rank-r scores; their logits are the exact LM head's;
    the draft top-k is the best of their softmax."""
    num_candidates, vocab = 128, 256
    config = LlamaConfig(
        vocab_size=vocab,
        hidden_size=_HIDDEN,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=256,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        draft_vocab_size=vocab,
        efficient_lm_head_config={
            "type": "low_rank",
            "rank": 8,
            "num_candidates": num_candidates,
        },
    )
    model = _build_model(monkeypatch, config)
    torch.manual_seed(0)
    for weight in (
        model.lm_head.weight,
        model.efficient_lm_head.weight,
        model.efficient_lm_head_down_proj.weight,
    ):
        weight.data.normal_()
    heads = CandidateHeads(
        lm_head=model.lm_head,
        scores_head=model.efficient_lm_head,
        down_proj=model.efficient_lm_head_down_proj,
        num_candidates=num_candidates,
    )
    processor = model.logits_processor
    processor.draft_top_k = 4
    hidden = torch.randn(3, _HIDDEN, dtype=torch.bfloat16, device="cuda")

    output = processor(
        None,
        hidden,
        heads,
        LogitsMetadata(forward_mode=ForwardMode.DECODE, query_shard=None),
    )
    logits, ids = output.next_token_logits, output.candidate_ids

    scores = heads.down_proj(hidden)[0] @ heads.scores_head.weight.T
    for row in range(hidden.shape[0]):
        want = set(torch.topk(scores[row], num_candidates).indices.tolist())
        assert set(ids[row].tolist()) == want
    exact = (heads.lm_head.weight[ids].float() @ hidden.float().unsqueeze(-1)).squeeze(
        -1
    )
    torch.testing.assert_close(logits, exact.bfloat16().float(), rtol=0, atol=0.13)
    best = output.next_token_ids
    assert torch.equal(best, ids.gather(-1, logits.argmax(-1, keepdim=True)).view(-1))
    top = torch.topk(torch.softmax(logits, -1), 4, dim=-1)
    probs, tokens = output.draft_topk
    torch.testing.assert_close(probs, top.values, rtol=0, atol=0)
    assert torch.equal(tokens, ids.gather(-1, top.indices))
    # A static EAGLE-4 tree ranks the same candidates by log-probability.
    scores, tree_tokens = _static_tree_drafter(model, 4)._score_candidates(output)
    assert torch.equal(tree_tokens.long(), tokens)
    torch.testing.assert_close(scores.exp(), probs, rtol=1e-5, atol=1e-6)


# Toy sizes; the MoE's are the smallest the BF16 squared-ReLU expert kernels plan for.
_MOE_HIDDEN, _LATENT, _EXPERT_INTER, _EXPERTS, _TOP_K, _SHARED_INTER = (
    256,
    128,
    128,
    16,
    4,
    256,
)
_MOE_VOCAB, _RANK = 256, 8


def _qwen38_draft_config(**overrides) -> LlamaConfig:
    """The Qwen3.8 EAGLE-4 draft's config keys: dense SwiGLU, RoPE, concat input."""
    values = dict(
        _torchspec_version="0.1.0",
        architectures=["LlamaForCausalLMEagle4"],
        attention_bias=False,
        draft_vocab_size=_MOE_VOCAB,
        eagle_config={
            "eagle_aux_hidden_state_layer_ids": [3, 31, 59],
            "use_aux_hidden_state": True,
        },
        efficient_lm_head_config={
            "num_candidates": 128,
            "rank": _RANK,
            "type": "low_rank",
        },
        fc_norm=True,
        head_dim=64,
        hidden_act="silu",
        hidden_size=_MOE_HIDDEN,
        intermediate_size=512,
        layer_types=["sliding_attention"],
        max_position_embeddings=262144,
        mlp_bias=False,
        norm_output=True,
        num_attention_heads=4,
        num_aux_hidden_states=3,
        num_hidden_layers=1,
        num_key_value_heads=4,
        rms_norm_eps=1e-6,
        rope_parameters={"rope_theta": 10000000.0, "rope_type": "default"},
        sliding_window=64,
        target_hidden_size=_MOE_HIDDEN,
        tie_word_embeddings=False,
        use_sliding_window=True,
        vocab_size=_MOE_VOCAB,
        torch_dtype="bfloat16",
    )
    values.update(overrides)
    return LlamaConfig(**values)


def _nemotron_draft_values() -> dict:
    """The Nemotron 3.5 Super EAGLE-4 draft's config: NoPE, eh_proj input merge,
    latent squared-ReLU MoE, training-only keys included."""
    return dict(
        _torchspec_version="0.1.0",
        architectures=["LlamaForCausalLMEagle4"],
        attention_bias=False,
        attention_dropout=0.0,
        attn_norm=True,
        draft_vocab_size=_MOE_VOCAB,
        eagle_config={"eagle_aux_hidden_state_layer_ids": [7, 47, 78]},
        efficient_lm_head_config={
            "num_candidates": 128,
            "rank": _RANK,
            "type": "low_rank",
        },
        enable_moe=True,
        expert_bias_update_rate=0.0001,
        expert_parallel_size=1,
        fc_norm=True,
        head_dim=64,
        hidden_act="relu2",
        hidden_size=_MOE_HIDDEN,
        initializer_range=0.02,
        intermediate_size=512,
        layer_types=["sliding_attention"],
        load_balance_loss_alpha=0.0,
        max_position_embeddings=1048576,
        mlp_bias=False,
        moe_architecture="nemotron_latent_relu2",
        moe_expert_parallel_size=1,
        moe_intermediate_size=_EXPERT_INTER,
        moe_latent_size=_LATENT,
        moe_router_dtype="float32",
        moe_shared_expert_intermediate_size=_SHARED_INTER,
        n_group=1,
        n_routed_experts=_EXPERTS,
        n_shared_experts=1,
        norm_output=True,
        norm_topk_prob=True,
        num_attention_heads=4,
        num_experts_per_tok=_TOP_K,
        num_hidden_layers=1,
        num_key_value_heads=4,
        pretraining_tp=1,
        rms_norm_eps=1e-5,
        rope_parameters={"rope_theta": 10000.0, "rope_type": "default"},
        routed_scaling_factor=2.5,
        scoring_func="sigmoid",
        seq_aux_loss_alpha=0.0,
        sliding_window=64,
        target_hidden_size=_MOE_HIDDEN,
        tie_word_embeddings=False,
        topk_group=1,
        topk_method="noaux_tc",
        use_cache=True,
        use_rope=False,
        vocab_size=_MOE_VOCAB,
        torch_dtype="bfloat16",
    )


def _nemotron_draft_config(**overrides) -> LlamaConfig:
    values = _nemotron_draft_values()
    values.update(overrides)
    return LlamaConfig(**values)


def test_eagle4_accepts_the_released_draft_configs() -> None:
    validate_draft_config(_qwen38_draft_config())
    validate_draft_config(_nemotron_draft_config())


@pytest.mark.parametrize(
    "overrides,match",
    [
        (dict(enable_two_heads=True), "does not implement"),
        (dict(enable_post_mlp_norm=True), "enable_post_mlp_norm"),
        (dict(share_draft_embedding_from_target=False), "share_draft_embedding"),
        (dict(moe_architecture="sparse_swiglu"), "moe_architecture"),
        (dict(hidden_act="silu"), "hidden_act"),
        (dict(enable_fc_norm=False), "enable_fc_norm"),
        (dict(target_hidden_size=2 * _MOE_HIDDEN), "target_hidden_size"),
        (dict(draft_vocab_size=_MOE_VOCAB // 2), "draft_vocab_size"),
        (dict(layer_types=None), "layer_types"),
        (dict(eagle_config={"eagle_aux_hidden_state_layer_ids": [1, 2]}), "three"),
        (
            dict(efficient_lm_head_config={"type": "svd", "rank": 8}),
            "efficient_lm_head_config",
        ),
        (dict(n_shared_experts=2), "n_shared_experts"),
        (dict(norm_topk_prob=False), "norm_topk_prob"),
        (dict(routed_scaling_factor=None), "routed_scaling_factor"),
    ],
)
def test_eagle4_refuses_draft_structures_it_does_not_build(
    overrides: dict, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        validate_draft_config(_nemotron_draft_config(**overrides))


def test_eagle4_refuses_a_vocab_its_heads_would_pad(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A vocab the LM heads pad to a multiple of 64 leaves zero rows a shard's
    top-k could pick: the candidate head refuses it."""
    config = _qwen38_draft_config(vocab_size=1000, draft_vocab_size=1000)
    with pytest.raises(ValueError, match="unpadded vocab shards"):
        _build_model(monkeypatch, config, "meta")


def test_eagle4_draft_attends_without_rope_only_when_the_config_says_so(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Nemotron draft ships rope parameters but sets use_rope false."""
    for config, rotary in (
        (_nemotron_draft_config(), False),
        (_qwen38_draft_config(), True),
    ):
        model = _build_model(monkeypatch, config, "meta")
        (attention,) = (m for m in model.modules() if isinstance(m, PagedAttention))
        assert (attention.rotary_emb is not None) == rotary


def _nemotron_draft_checkpoint(
    layer_prefix: str, device: str
) -> dict[str, torch.Tensor]:
    """The Nemotron EAGLE-4 draft's tensors at toy size: random BF16, the
    router bias in FP32."""
    g = None if device == "meta" else torch.Generator(device=device).manual_seed(1)

    def rand(*shape: int, dtype=torch.bfloat16) -> torch.Tensor:
        if g is None:
            return torch.empty(*shape, dtype=dtype, device=device)
        scale = shape[-1] ** -0.5
        return (torch.randn(*shape, generator=g, device=device) * scale).to(dtype)

    h, v = _MOE_HIDDEN, _MOE_VOCAB
    weights = {
        "embed_tokens.weight": rand(v, h),
        "lm_head.weight": rand(v, h),
        "efficient_lm_head.weight": rand(v, _RANK),
        "efficient_lm_head_down_proj.weight": rand(_RANK, h),
        "fc.weight": rand(h, 3 * h),
        "norm.weight": rand(h),
        **{f"fc_norm.{i}.weight": rand(h) for i in range(3)},
    }
    layer = {
        "hidden_norm.weight": rand(h),
        "input_layernorm.weight": rand(h),
        "eh_proj.weight": rand(h, 2 * h),
        "pre_attention_layernorm.weight": rand(h),
        "self_attn.q_proj.weight": rand(h, h),
        "self_attn.k_proj.weight": rand(h, h),
        "self_attn.v_proj.weight": rand(h, h),
        "self_attn.o_proj.weight": rand(h, h),
        "post_attention_layernorm.weight": rand(h),
        "mlp.gate.weight": rand(_EXPERTS, h),
        # As trained: a large common offset whose small per-expert spread a BF16 bias would round away.
        "mlp.gate.e_score_correction_bias": 25.0
        + rand(_EXPERTS, dtype=torch.float32) * 0.25,
        "mlp.fc1_latent_proj.weight": rand(_LATENT, h),
        "mlp.fc2_latent_proj.weight": rand(h, _LATENT),
        "mlp.shared_experts.up_proj.weight": rand(_SHARED_INTER, h),
        "mlp.shared_experts.down_proj.weight": rand(h, _SHARED_INTER),
    }
    for e in range(_EXPERTS):
        layer[f"mlp.experts.{e}.up_proj.weight"] = rand(_EXPERT_INTER, _LATENT)
        layer[f"mlp.experts.{e}.down_proj.weight"] = rand(_LATENT, _EXPERT_INTER)
    weights.update({layer_prefix + name: w for name, w in layer.items()})
    return weights


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("layer_prefix", ["layers.0.", "midlayer.", "model.layers.0."])
def test_eagle4_latent_moe_checkpoint_fills_every_parameter_from_its_tensor(
    monkeypatch: pytest.MonkeyPatch, layer_prefix: str
) -> None:
    model = _build_model(monkeypatch, _nemotron_draft_config())
    weights = _nemotron_draft_checkpoint(layer_prefix, "cuda")

    model.load_weights(weights.items())

    params = dict(model.named_parameters())
    layer = {
        name.removeprefix(layer_prefix): w
        for name, w in weights.items()
        if name.startswith(layer_prefix)
    }
    expected = {
        "model.midlayer.self_attn.qkv_proj.weight": torch.cat(
            [layer[f"self_attn.{p}_proj.weight"] for p in "qkv"]
        ),
        "model.midlayer.mlp.experts.w13_weight": torch.stack(
            [layer[f"mlp.experts.{e}.up_proj.weight"] for e in range(_EXPERTS)]
        ),
        "model.midlayer.mlp.experts.w2_weight": torch.stack(
            [layer[f"mlp.experts.{e}.down_proj.weight"] for e in range(_EXPERTS)]
        ),
    }
    for name, w in layer.items():
        if not name.startswith(("self_attn.q_", "self_attn.k_", "self_attn.v_")):
            if ".experts." not in name:
                expected[f"model.midlayer.{name}"] = w
    for name, w in weights.items():
        if not name.startswith(layer_prefix):
            top = name if name.startswith(("lm_head", "efficient")) else f"model.{name}"
            expected[top] = w
    assert set(expected) == set(params)
    for name, w in expected.items():
        assert params[name].dtype == w.dtype, name
        assert torch.equal(params[name], w), name


@pytest.mark.parametrize(
    "drop,error",
    [
        ("layers.0.mlp.experts.3.down_proj.weight", "expert tensors"),
        ("layers.0.self_attn.k_proj.weight", r"qkv_proj.weight \(k\)"),
        ("layers.0.mlp.gate.e_score_correction_bias", "lacks"),
        ("layers.0.pre_attention_layernorm.weight", "lacks"),
        ("efficient_lm_head.weight", "lacks"),
    ],
)
def test_eagle4_refuses_an_incomplete_checkpoint(
    monkeypatch: pytest.MonkeyPatch, drop: str, error: str
) -> None:
    model = _build_model(monkeypatch, _nemotron_draft_config(), "meta")
    weights = _nemotron_draft_checkpoint("layers.0.", "meta")
    del weights[drop]

    with pytest.raises(ValueError, match=error):
        model.load_weights(weights.items())


def test_eagle4_refuses_an_expert_tensor_given_twice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One expert's tensor under two prefixes cannot stand in for another's."""
    model = _build_model(monkeypatch, _nemotron_draft_config(), "meta")
    weights = _nemotron_draft_checkpoint("layers.0.", "meta")
    del weights["layers.0.mlp.experts.3.down_proj.weight"]
    weights["midlayer.mlp.experts.2.down_proj.weight"] = weights[
        "layers.0.mlp.experts.2.down_proj.weight"
    ]

    with pytest.raises(ValueError, match="repeats"):
        model.load_weights(weights.items())


def test_eagle4_refuses_a_tensor_no_parameter_takes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _build_model(monkeypatch, _nemotron_draft_config(), "meta")
    weights = _nemotron_draft_checkpoint("layers.0.", "meta")
    weights["layers.0.mlp.post_mlp_layernorm.weight"] = weights["norm.weight"]

    with pytest.raises(KeyError, match="post_mlp_layernorm"):
        model.load_weights(weights.items())


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x = x.float()
    return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight.float()).to(
        torch.bfloat16
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_eagle4_eh_proj_turns_the_normed_inputs_into_the_residual(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With attn_norm, eh_proj maps ``[norm(embeds) || norm(hidden)]`` to the
    residual and its norm feeds q/k/v; without it the concat feeds them."""
    model = _build_model(monkeypatch, _nemotron_draft_config())
    model.load_weights(_nemotron_draft_checkpoint("layers.0.", "cuda").items())
    layer = model.model.midlayer
    g = torch.Generator(device="cuda").manual_seed(4)
    merged = torch.randn(5, 2 * _MOE_HIDDEN, generator=g, device="cuda").bfloat16()
    hidden = torch.randn(5, _MOE_HIDDEN, generator=g, device="cuda").bfloat16()

    attn_input, residual = layer._fuse_input(merged, hidden)

    want = merged.float() @ layer.eh_proj.weight.float().T
    torch.testing.assert_close(residual.float(), want, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(
        attn_input.float(),
        _rms_norm(residual, layer.pre_attention_layernorm.weight, 1e-5).float(),
        atol=2e-2,
        rtol=2e-2,
    )

    qwen = _build_model(
        monkeypatch, _qwen38_draft_config(), "meta"
    ).model.midlayer._fuse_input(merged, hidden)
    assert qwen[0] is merged and qwen[1] is hidden


def _latent_moe_reference(
    weights: dict[str, torch.Tensor], x: torch.Tensor
) -> torch.Tensor:
    """The draft MoE as trained: an FP32 sigmoid router, top-k over the
    bias-corrected scores, the unbiased scores renormalized and scaled by
    routed_scaling_factor, squared-ReLU experts on the latent, fc2 back, plus
    the shared expert on the full-width input."""
    w = {name.removeprefix("layers.0.mlp."): t for name, t in weights.items()}
    scores = (x.float() @ w["gate.weight"].float().T).sigmoid()
    chosen = (scores + w["gate.e_score_correction_bias"]).topk(_TOP_K, dim=-1).indices
    gates = scores.gather(-1, chosen)
    gates = gates / gates.sum(-1, keepdim=True) * 2.5
    latent = x @ w["fc1_latent_proj.weight"].T
    routed = torch.zeros(x.shape[0], _LATENT, device=x.device)
    for row in range(x.shape[0]):
        for slot in range(_TOP_K):
            e = chosen[row, slot].item()
            up = latent[row] @ w[f"experts.{e}.up_proj.weight"].T
            act = torch.relu(up).float().square().bfloat16()
            routed[row] += gates[row, slot] * (
                act @ w[f"experts.{e}.down_proj.weight"].T
            )
    routed = routed.bfloat16() @ w["fc2_latent_proj.weight"].T
    up = torch.relu(x @ w["shared_experts.up_proj.weight"].T).float().square()
    return routed + up.bfloat16() @ w["shared_experts.down_proj.weight"].T


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="the BF16 squared-ReLU experts run on SM100-class GPUs",
)
@pytest.mark.parametrize("tokens", [1, 4, 13])
def test_eagle4_latent_moe_matches_the_trained_block(
    monkeypatch: pytest.MonkeyPatch, tokens: int
) -> None:
    model = _build_model(monkeypatch, _nemotron_draft_config())
    weights = _nemotron_draft_checkpoint("layers.0.", "cuda")
    model.load_weights(weights.items())
    moe = model.model.midlayer.mlp
    # The block's one post-load hook the model loader runs that is not a no-op.
    moe.experts.process_weights_after_loading(moe.experts)
    x = torch.randn(tokens, _MOE_HIDDEN, device="cuda").bfloat16()
    ctx = ForwardContext(
        attn_backend=None,
        token_to_kv_pool=None,
        bs=tokens,
        num_extends=0,
        input_num_tokens=tokens,
        forward_mode=ForwardMode.DECODE,
        output_layout=None,
    )

    out = moe(x, None, ctx)

    ref = _latent_moe_reference(
        {k: v for k, v in weights.items() if ".mlp." in k}, x
    ).float()
    assert (out.float() - ref).norm() / ref.norm() < 1e-2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

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

"""Native draft arithmetic and checkpoint loading without a serving cache."""

import json
from pathlib import Path
from test.runtime.models.test_inkling2_config import _checkpoint

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from torch import nn

from tokenspeed.runtime.configs.device_config import DeviceConfig
from tokenspeed.runtime.configs.inkling2 import Inkling2MMConfig, Inkling2MTPConfig
from tokenspeed.runtime.configs.load_config import LoadConfig
from tokenspeed.runtime.configs.model_config import ModelConfig
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.engine.scheduler_utils import (
    resolve_dspark_prefix_replay_tokens,
)
from tokenspeed.runtime.layers.attention_residual import PrecisionRMSNorm
from tokenspeed.runtime.layers.vocab_parallel_embedding import ParallelLMHead
from tokenspeed.runtime.model_loader.loader import DefaultModelLoader
from tokenspeed.runtime.models.base.input_embedding import DecoderInputEmbedding
from tokenspeed.runtime.models.inkling2_mtp import (
    Inkling2ForConditionalGenerationMTP,
    Inkling2MTPHead,
    _add_norm,
    padded_shard_ids_to_vocab,
    vocab_ids_to_padded_shards,
)
from tokenspeed.runtime.utils.server_args import ServerArgs


def _config():
    values = _checkpoint()
    values["eos_token_id"] = 2
    values["text_config"].update(
        hidden_size=32,
        num_hidden_layers=6,
        vocab_size=64,
        unpadded_vocab_size=60,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        local_layer_ids=[0, 1, 3, 4],
        swa_num_attention_heads=4,
        swa_num_key_value_heads=2,
        swa_head_dim=8,
        n_layers_per_attn_res_block=2,
    )
    values["mtp_config"].update(
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=64,
        max_position_embeddings=64,
        block_size=4,
        markov_rank=4,
        sliding_window_size=8,
        target_layer_ids=[0, 2, 6],
    )
    return Inkling2MTPConfig.from_target(Inkling2MMConfig(**values))


def _mapping(tp=1, rank=0):
    return Mapping(
        rank=rank, world_size=tp, attn_tp_size=tp, dense_tp_size=tp, lm_head_tp_size=tp
    )


@pytest.mark.parametrize("prefix_caching", [False, True])
def test_default_loader_loads_native_draft_from_target_directory(
    tmp_path, prefix_caching
):
    from test.runtime.models.test_inkling2_model import _model

    target = _model().config
    target.mtp_config = _config().to_dict()
    target.mtp_config["target_layer_ids"] = [0, 1, 2]
    target.save_pretrained(tmp_path)
    expected = Inkling2ForConditionalGenerationMTP(
        Inkling2MTPConfig.from_target(target), _mapping(), None
    )
    weights = dict(_native_weights(expected))
    expected.load_weights(weights.items())
    save_file(weights, tmp_path / "mtp.safetensors")
    # The target index must not redirect the independent draft loader.
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"target.weight": "absent-target.safetensors"}})
    )
    (tmp_path / "hf_quant_config.json").write_text(
        json.dumps({"quant_method": "inkling2", "model": {"parameter_groups": []}})
    )
    args = ServerArgs(
        model=str(tmp_path),
        speculative_draft_model_path=str(tmp_path),
        speculative_algorithm="DSPARK",
        speculative_num_steps=3,
        speculative_num_draft_tokens=4,
        max_model_len=40,
        device="cpu",
    )
    args.mapping.rank = 0
    common = dict(
        trust_remote_code=False,
        model_override_args="{}",
        dtype="bfloat16",
        server_args=args,
        context_length=40,
    )
    metadata = ModelConfig(str(tmp_path), is_draft_worker=True, **common)
    target_metadata = ModelConfig(str(tmp_path), is_draft_worker=False, **common)
    assert metadata.quantization is None
    assert target_metadata.quantization == "inkling2"
    # Substitute only the unavailable GPU device; loader, registry and arithmetic stay real.
    device = DeviceConfig("cuda")
    device.device = torch.device("cpu")
    device.device_type = "cpu"
    loaded = DefaultModelLoader(
        LoadConfig(weight_loader_prefetch_checkpoints=False)
    ).load_model(model_config=metadata, device_config=device)
    assert isinstance(loaded, Inkling2ForConditionalGenerationMTP)
    assert metadata.quantization is None
    assert metadata.num_attention_layers == 5
    assert (
        resolve_dspark_prefix_replay_tokens(
            speculative_algorithm="DSPARK",
            enable_prefix_caching=prefix_caching,
            enable_kvstore=False,
            disaggregation_mode="null",
            draft_model_path_use_base=True,
            draft_model_config=metadata,
        )
        == 0
    )
    for name, value in expected.named_parameters():
        assert torch.equal(value, dict(loaded.named_parameters())[name])
    assert torch.equal(loaded.fuse_alpha, expected.fuse_alpha)


def _native_weights(model):
    generator = torch.Generator().manual_seed(42)
    weights = []
    for name, parameter in model.named_parameters():
        value = (torch.randn(parameter.shape, generator=generator) * 0.04).bfloat16()
        if "norm.weight" in name:
            value += 1
        if ".qkv_proj." in name:
            d = model.config.head_dim
            q = model.config.num_attention_heads * d
            kv = model.config.num_key_value_heads * d
            for piece, chunk in zip(("q", "k", "v"), value.split([q, kv, kv])):
                weights.append(
                    (name.replace("qkv_proj", piece + "_proj"), chunk.clone())
                )
        elif ".kv_proj_ctx." in name:
            for piece, chunk in zip(("k", "v"), value.chunk(2)):
                weights.append(
                    (name.replace("kv_proj_ctx", piece + "_proj_ctx"), chunk.clone())
                )
        elif ".gate_up_proj." in name:
            for piece, chunk in zip(("gate", "up"), value.chunk(2)):
                weights.append(
                    (name.replace("gate_up_proj", piece + "_proj"), chunk.clone())
                )
        else:
            weights.append((name, value))
    return weights


def _loaded():
    model = Inkling2ForConditionalGenerationMTP(_config(), _mapping(), None)
    weights = _native_weights(model)
    assert len(weights) == 84
    model.load_weights(weights)
    return model, weights


def test_shared_loader_reads_native_file_outside_target_index(tmp_path: Path):
    source, weights = _loaded()
    save_file(dict(weights), tmp_path / "mtp.safetensors")
    # This target shard must not be opened by the draft loader.
    (tmp_path / "model-00001-of-00001.safetensors").write_bytes(b"not a draft")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.weight": "model-00001-of-00001.safetensors"}})
    )
    draft = Inkling2ForConditionalGenerationMTP(_config(), _mapping(), None)
    loader = DefaultModelLoader(LoadConfig(weight_loader_prefetch_checkpoints=False))
    loaded = draft.load_weights(
        loader.get_checkpoint_weights(str(tmp_path), None, draft, _mapping())
    )
    assert len(loaded) == 64
    for name, parameter in draft.named_parameters():
        assert torch.equal(parameter, dict(source.named_parameters())[name])


def test_unloaded_draft_refuses_target_wiring():
    from test.runtime.models.test_inkling2_model import _model

    target = _model()
    target.config.mtp_config = _config().to_dict()
    target.config.mtp_config["target_layer_ids"] = [0, 1, 2]
    draft = Inkling2ForConditionalGenerationMTP(
        Inkling2MTPConfig.from_target(target.config), _mapping(), None
    )
    with pytest.raises(RuntimeError, match="complete checkpoint loading"):
        draft.configure_target(target, target.text_config)
    draft.load_weights(_native_weights(draft))
    draft.configure_target(target, target.text_config)
    assert draft.lm_head is target.lm_head


def test_missing_native_file_does_not_fall_back_to_target(tmp_path: Path):
    save_file({"target.weight": torch.ones(1)}, tmp_path / "model.safetensors")
    draft = Inkling2ForConditionalGenerationMTP(_config(), _mapping(), None)
    loader = DefaultModelLoader(LoadConfig(weight_loader_prefetch_checkpoints=False))
    with pytest.raises(FileNotFoundError, match="mtp.safetensors"):
        list(loader.get_checkpoint_weights(str(tmp_path), None, draft, _mapping()))


def _bind(model):
    torch.manual_seed(51)
    lookup = nn.Embedding(64, 32, dtype=torch.bfloat16)
    norm = PrecisionRMSNorm(32, 1e-6, torch.bfloat16, True, True)
    norm.weight.data.fill_(3)
    head = ParallelLMHead(
        64,
        32,
        bias=False,
        params_dtype=torch.bfloat16,
        quant_config=None,
        prefix="model.llm.unembed",
        tp_rank=0,
        tp_size=1,
        tp_group=(0,),
    )
    head.weight.data.normal_(std=0.1)
    model.attach_shared_modules(
        embed_tokens=DecoderInputEmbedding(lookup, norm), lm_head=head
    )
    return lookup, norm, head


def test_native_loader_and_context_mixtures_preserve_tap_order_and_bf16_store():
    model, weights = _loaded()
    assert len(dict(model.named_parameters())) == 64
    assert all(parameter.dtype == torch.bfloat16 for parameter in model.parameters())
    expected_alpha = model.fuse_weight.float().softmax(-1).bfloat16()
    assert torch.equal(model.fuse_alpha, expected_alpha)
    targets = torch.arange(2 * 3 * 32).reshape(2, 3 * 32).bfloat16() / 100
    for stage in range(6):
        mixed = torch.zeros(2, 32)
        for tap in range(3):
            mixed += (
                targets[:, tap * 32 : (tap + 1) * 32].float()
                * expected_alpha[stage, tap].float()
            )
        norm = (
            model.markov_head.dflare_norm
            if stage == 5
            else model.layers[stage].dflare_norm
        )
        expected = (
            mixed
            * torch.rsqrt(mixed.square().mean(-1, keepdim=True) + 1e-6)
            * norm.weight.float()
        ).bfloat16()
        assert torch.equal(model.mix_context(targets, stage), expected)
    assert torch.equal(model.compute_f_rnn(targets), model.mix_context(targets, 5))
    k, v = model.project_context_kv(targets, 0)
    native = dict(weights)
    context = model.mix_context(targets, 0)
    assert torch.equal(
        k, F.linear(context, native["layers.0.self_attn.k_proj_ctx.weight"])
    )
    assert torch.equal(
        v, F.linear(context, native["layers.0.self_attn.v_proj_ctx.weight"])
    )
    assert model.mix_context(targets[:0], 0).shape == (0, 32)


def test_normalized_anchor_and_learned_masks_ignore_placeholder_ids():
    model, _ = _loaded()
    lookup, norm, head = _bind(model)
    ids = torch.tensor([2, 0, 0, 0, 3, 0, 0, 0])
    actual = model.forward_embed(ids).reshape(2, 4, 32)
    assert torch.equal(actual[:, 0], norm(lookup(torch.tensor([2, 3]))))
    assert torch.equal(actual[:, 1:], model.mask_embed.expand(2, 3, 32))
    changed = ids.clone()
    changed[[1, 2, 3, 5, 6, 7]] = 63
    assert torch.equal(model.forward_embed(changed), actual.reshape(8, 32))
    assert model.lm_head is head and model.embed_tokens.weight is lookup.weight
    assert model.forward_embed(ids[:0]).shape == (0, 32)
    rows = torch.arange(2 * 4 * 32).reshape(2, 4, 32).bfloat16()
    assert torch.equal(model.proposal_rows(rows), rows[:, 1:])


def test_query_block_keeps_bf16_projections_for_the_shared_prologue():
    model, weights = _loaded()
    hidden = torch.linspace(-2, 2, 4 * 32).reshape(4, 32).bfloat16()
    attention = model.layers[0].self_attn
    q, k, v = attention.project_query_block(hidden)
    native = dict(weights)
    for name, actual in (("q", q), ("k", k), ("v", v)):
        projection = F.linear(hidden, native[f"layers.0.self_attn.{name}_proj.weight"])
        assert actual.dtype == torch.bfloat16 and torch.equal(actual, projection)


def test_packed_checkpoint_roundtrip_matches_native_projection_loading():
    source, _ = _loaded()
    destination = Inkling2ForConditionalGenerationMTP(_config(), _mapping(), None)
    destination.load_weights(
        (name, parameter.clone()) for name, parameter in source.named_parameters()
    )
    inputs = torch.randn(3, 32).bfloat16()
    for original, restored in zip(
        source.layers[0].self_attn.project_query_block(inputs),
        destination.layers[0].self_attn.project_query_block(inputs),
    ):
        assert torch.equal(original, restored)


@pytest.mark.parametrize(
    "failure", ["raw_embedding", "head_dtype", "head_bias", "head_quantization"]
)
def test_shared_resources_reject_incompatible_precision_or_embedding_boundary(failure):
    model, _ = _loaded()
    lookup, norm, head = _bind(model)
    adapter = DecoderInputEmbedding(lookup, norm)
    if failure == "raw_embedding":
        adapter = DecoderInputEmbedding(lookup, None)
    elif failure == "head_dtype":
        head.to(torch.float32)
    elif failure == "head_bias":
        head.bias = nn.Parameter(torch.zeros(64, dtype=torch.bfloat16))
    else:
        head.quant_config = model.config
    with pytest.raises(ValueError):
        model.attach_shared_modules(embed_tokens=adapter, lm_head=head)


def test_head_fold_uses_fp64_and_draft_base_stays_uncapped_bf16():
    model, _ = _loaded()
    _, _, head = _bind(model)
    recurrent = model.markov_head
    assert torch.equal(
        recurrent.fold_weight,
        (head.weight.double() @ recurrent.hidden_out.weight.double()).bfloat16(),
    )
    assert torch.equal(
        recurrent.fold_bias,
        (head.weight.double() @ recurrent.hidden_out.bias.double()).bfloat16(),
    )
    head.weight.data.zero_()
    head.weight.data[:, :3] = torch.tensor([128, 0.5, -128], dtype=torch.bfloat16)
    model.refresh_shared_modules()
    hidden = torch.ones(2, 32, dtype=torch.bfloat16)
    base = model.compute_base_logits(hidden)
    assert base.dtype == torch.bfloat16
    assert torch.equal(base, F.linear(hidden, head.weight))
    assert torch.isfinite(base[:, 60:]).all()
    # The drafter passes each block's proposal rows, a strided view.
    blocks = torch.randn(2, 4, 32).bfloat16()
    rows = model.proposal_rows(blocks)
    base = model.compute_base_logits(rows)
    assert base.shape == (2, 3, head.weight.shape[0])
    for block in range(2):
        assert torch.equal(base[block], F.linear(rows[block], head.weight))


def test_fp64_fold_retains_small_terms_between_cancelling_products():
    model, _ = _loaded()
    head = model.markov_head
    with torch.no_grad():
        head.hidden_out.weight.zero_()
        head.hidden_out.weight[:3, 0] = 1
        head.hidden_out.bias.zero_()
        head.hidden_out.bias[:3] = 1
    weights = torch.zeros(1, 32, dtype=torch.bfloat16)
    weights[0, :3] = torch.tensor([2**24, 1, -(2**24)])
    head.refresh_fold(weights, vocab_start=0, vocab_count=1)
    assert head.fold_weight[0, 0] == 1 and head.fold_bias[0] == 1
    assert (weights.float() @ head.hidden_out.weight.float())[0, 0] == 0


def test_dense_mlp_keeps_projection_and_swiglu_bf16_boundaries():
    model, weights = _loaded()
    native = dict(weights)
    hidden = torch.linspace(-1, 1, 3 * 32).reshape(3, 32).bfloat16()
    gate = F.linear(hidden, native["layers.0.mlp.gate_proj.weight"])
    up = F.linear(hidden, native["layers.0.mlp.up_proj.weight"])
    activation = (F.silu(gate.float()) * up.float()).bfloat16()
    expected = F.linear(activation, native["layers.0.mlp.down_proj.weight"])
    actual = model.layers[0].mlp(hidden)
    assert actual.dtype == torch.bfloat16 and torch.equal(actual, expected)


def _head_oracle(head, base, tokens, hidden, features):
    state = hidden.new_zeros(hidden.shape[0], head.markov_rank)
    outputs = []
    for step in range(tokens.shape[1]):
        previous = F.embedding(tokens[:, step], head.markov_w1.weight)
        inputs = torch.cat([state, previous, hidden[:, step], features], dim=-1)
        gate_raw, candidate, out = F.linear(
            inputs, head.rnn_proj.weight, head.rnn_proj.bias
        ).chunk(3, dim=-1)
        gate = gate_raw.sigmoid()
        state = gate * state + (1 - gate) * candidate.tanh()
        correction = torch.addmm(head.fold_bias, out.tanh(), head.fold_weight.T)
        summed = base[:, step] + correction
        logits = (head.softcap * (summed.float() / head.softcap).tanh()).bfloat16()
        logits[:, head.valid_vocab_size :] = -float("inf")
        outputs.append(logits)
    return torch.stack(outputs, dim=1)


def test_bf16_recurrence_feeds_back_each_sample_and_padded_shard_ids_agree():
    model, _ = _loaded()
    _bind(model)
    torch.manual_seed(74)
    hidden = torch.randn(2, 3, 32).bfloat16()
    features = torch.randn(2, 32).bfloat16()
    base = torch.randn(2, 3, 64).bfloat16()
    tokens = torch.tensor([[1, 2, 3], [4, 5, 6]])
    corrections = []

    def sample(logits, _step):
        # The step rows are reused; padding past the shard reads -inf.
        assert torch.isneginf(logits[:, 64:]).all()
        corrections.append(logits[:, :64].clone())
        return logits.argmax(-1)

    sampled = model.markov_head.sample_block(
        base,
        first_prev_tokens=tokens[:, 0],
        hidden_states=hidden,
        f_rnn=features,
        logits_out=torch.empty(2, 96, dtype=torch.bfloat16),
        sampler=sample,
        shard_padding=None,
    )
    corrected = torch.stack(corrections, dim=1)
    previous = torch.cat((tokens[:, :1], sampled[:, :-1]), dim=1)
    assert torch.equal(
        corrected, _head_oracle(model.markov_head, base, previous, hidden, features)
    )
    assert torch.equal(sampled, corrected.argmax(-1))
    # Indices over 16-row shards padded to 40 propose the same vocabulary ids.
    padded_corrections = []

    def sample_padded(logits, _step):
        padded_corrections.append(logits[:, :64].clone())
        return vocab_ids_to_padded_shards(logits.argmax(-1), 40, 16)

    padded_sampled = model.markov_head.sample_block(
        base,
        first_prev_tokens=tokens[:, 0],
        hidden_states=hidden,
        f_rnn=features,
        logits_out=torch.empty(2, 96, dtype=torch.bfloat16),
        sampler=sample_padded,
        shard_padding=(40, 16),
    )
    assert torch.equal(padded_sampled, sampled)
    assert torch.equal(torch.stack(padded_corrections, dim=1), corrected)
    assert torch.equal(
        padded_shard_ids_to_vocab(vocab_ids_to_padded_shards(previous, 40, 16), 40, 16),
        previous,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_gpu_recurrence_matches_teacher_forced_cpu_and_padded_shards():
    config = _config()
    # Widths the GPU path serves: projection K a multiple of 256.
    config.hidden_size, config.markov_rank = 256, 16
    torch.manual_seed(5)
    head = Inkling2MTPHead(config)
    for parameter in head.parameters():
        parameter.data.copy_(torch.randn_like(parameter.float()) * 0.1)
    head.refresh_fold(
        torch.randn(64, 256).bfloat16() * 0.1, vocab_start=0, vocab_count=64
    )
    hidden = torch.randn(3, 5, 256).bfloat16()
    features = torch.randn(3, 256).bfloat16()
    base = torch.randn(3, 5, 64).bfloat16()
    first = torch.tensor([1, 7, 63])
    chosen = []

    def run(device, shard_padding=None):
        head.to(device)
        logits = []

        def sample(step_logits, step):
            # Teacher forcing: both devices see the CPU run's tokens.
            logits.append(step_logits[:, :64].float().cpu())
            if device == "cpu":
                chosen.append(step_logits[:, :60].argmax(-1))
            token = chosen[step].to(device)
            if shard_padding is None:
                return token
            return vocab_ids_to_padded_shards(token, *shard_padding)

        tokens = head.sample_block(
            base.to(device),
            first_prev_tokens=first.to(device),
            hidden_states=hidden.to(device),
            f_rnn=features.to(device),
            logits_out=torch.empty(3, 96, dtype=torch.bfloat16, device=device),
            sampler=sample,
            shard_padding=shard_padding,
        )
        return torch.stack(logits, dim=1), tokens.cpu()

    expected, tokens = run("cpu")
    assert head.step_invariant_projection(hidden, features) is None
    actual, gpu_tokens = run("cuda")
    # The kernel reads the same embedding rows from padded-shard indices.
    padded, padded_tokens = run("cuda", shard_padding=(40, 16))
    assert torch.equal(padded, actual) and torch.equal(padded_tokens, gpu_tokens)
    assert torch.equal(gpu_tokens, tokens)
    # The GPU rounds each FP32 projection sum once, in another order: BF16 agreement.
    torch.testing.assert_close(actual, expected, rtol=0.05, atol=0.05)


def test_fold_and_final_cap_mask_use_native_vocabulary_shard_offsets():
    model, _ = _loaded()
    head = model.markov_head
    weight = torch.ones(8, 32, dtype=torch.bfloat16)
    head.refresh_fold(weight, vocab_start=56, vocab_count=8)
    base = torch.full((2, 8), 1000, dtype=torch.bfloat16)
    logits = head.step_logits(
        base,
        torch.zeros(2, 4, dtype=torch.bfloat16),
        torch.empty(2, 8, dtype=torch.bfloat16),
    )
    assert logits.dtype == torch.bfloat16
    assert torch.equal(logits[:, :4], torch.full((2, 4), 30, dtype=torch.bfloat16))
    assert torch.isneginf(logits[:, 4:]).all()


def test_softcap_runs_only_after_the_bf16_recurrent_correction():
    model, _ = _loaded()
    _bind(model)
    head = model.markov_head
    head.fold_weight.zero_()
    head.fold_bias.fill_(-1000)
    base = torch.full((2, 64), 1000, dtype=torch.bfloat16)
    actual = head.step_logits(
        base,
        torch.zeros(2, 4, dtype=torch.bfloat16),
        torch.empty(2, 64, dtype=torch.bfloat16),
    )
    assert torch.equal(actual[:, :60], torch.zeros(2, 60, dtype=torch.bfloat16))
    assert torch.isneginf(actual[:, 60:]).all()


@pytest.mark.parametrize(
    "field", ["q_bias", "o_bias", "is_causal", "sample_from_anchor"]
)
def test_model_rejects_other_attention_and_anchor_semantics(field):
    config = _config()
    setattr(config, field, True)
    with pytest.raises(ValueError, match="unbiased, non-causal"):
        Inkling2ForConditionalGenerationMTP(config, _mapping(), None)


def test_residual_statistics_precede_the_bf16_residual_store():
    norm = PrecisionRMSNorm(32, 1e-6, torch.bfloat16, True, True)
    hidden = torch.linspace(0.003, 0.2, 32).bfloat16().reshape(1, 32)
    residual = torch.ones_like(hidden)
    output, stored = _add_norm(hidden, residual, norm)
    total = hidden.float() + residual.float()
    expected = (
        total * torch.rsqrt(total.square().mean(-1, keepdim=True) + 1e-6)
    ).bfloat16()
    assert torch.equal(output, expected)
    assert torch.equal(stored, total.bfloat16())
    assert not torch.equal(output, norm(stored))


@pytest.mark.parametrize(
    "failure",
    [
        "missing",
        "unknown",
        "dtype",
        "shape",
        "duplicate",
        "incomplete",
        "packed_first",
        "unpacked_first",
    ],
)
def test_strict_native_loader_rejects_bad_checkpoints(failure):
    model, weights = _loaded()
    packed = (
        "layers.0.self_attn.qkv_proj.weight",
        model.layers[0].self_attn.qkv_proj.weight.clone(),
    )
    if failure == "missing":
        weights = weights[1:]
    elif failure == "unknown":
        weights.append(("unexpected.weight", torch.ones(1, dtype=torch.bfloat16)))
    elif failure == "dtype":
        weights[0] = (weights[0][0], weights[0][1].float())
    elif failure == "shape":
        weights[0] = (weights[0][0], weights[0][1].flatten())
    elif failure == "duplicate":
        weights.append(weights[0])
    elif failure == "incomplete":
        weights = [
            (name, value)
            for name, value in weights
            if name != "layers.0.self_attn.k_proj.weight"
        ]
    elif failure == "packed_first":
        weights.insert(0, packed)
    else:
        weights.append(packed)
    with pytest.raises(ValueError):
        model.load_weights(weights)
    assert model.fuse_alpha is None


@pytest.mark.parametrize("tp", [1, 2])
def test_native_loader_shards_complete_qkv_context_and_dense_weights(tp):
    source, weights = _loaded()
    for rank in range(tp):
        model = Inkling2ForConditionalGenerationMTP(_config(), _mapping(tp, rank), None)
        assert len(model.load_weights(reversed(weights))) == 64
        q = dict(weights)["layers.0.self_attn.q_proj.weight"].chunk(tp)[rank]
        k = dict(weights)["layers.0.self_attn.k_proj.weight"].chunk(tp)[rank]
        v = dict(weights)["layers.0.self_attn.v_proj.weight"].chunk(tp)[rank]
        assert torch.equal(
            model.layers[0].self_attn.qkv_proj.weight, torch.cat((q, k, v))
        )
        assert torch.equal(
            model.layers[0].mlp.down_proj.weight,
            source.layers[0].mlp.down_proj.weight.chunk(tp, dim=1)[rank],
        )

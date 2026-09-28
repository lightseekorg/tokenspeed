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

"""Native-MTP contracts: CPU doubles and synthetic GPU runtime integration.

GPU checks use the real NextN, Eagle, packed cache, MLA and logits processor,
not a full target model or end-to-end speculative generation.
"""

import dataclasses
from contextlib import contextmanager
from copy import copy
from dataclasses import dataclass, replace
from enum import IntEnum, auto
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from runtime.test_dots3_note_model import (
    RUNTIME,
    _Backend,
    _config_class,
    _config_values,
    _definitions,
    _Fp8Config,
    _Linear,
    _method,
    _Pool,
    _RMSNorm,
    _rotate,
    _small_config,
    adapter,
)
from torch import nn


class _LoadableLinear(_Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight.weight_loader = self.weight_loader
        if self.weight_scale_inv is not None:
            self.weight_scale_inv.weight_loader = self.weight_loader


class _MergedLinear(_LoadableLinear):
    def weight_loader(self, param, weight, shard_id):
        destination = param.chunk(2, dim=0)[shard_id]
        assert destination.shape == weight.shape
        destination.data.copy_(weight)


class _MLP(nn.Module):
    def __init__(
        self,
        hidden,
        intermediate,
        activation,
        *,
        mapping,
        quant_config,
        prefix,
        batch_invariant,
    ):
        super().__init__()
        assert activation == "silu"
        self.batch_invariant = batch_invariant
        self.gate_up_proj = _MergedLinear(
            hidden,
            2 * intermediate,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = _LoadableLinear(
            intermediate,
            hidden,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        self.inputs = []

    def forward(self, x):
        self.inputs.append(x.clone())
        gate, up = self.gate_up_proj(x)[0].chunk(2, dim=-1)
        return self.down_proj(torch.nn.functional.silu(gate) * up)[0]


class _Embedding(nn.Embedding):
    def __init__(self, vocab, hidden, *, tp_rank, tp_size, tp_group):
        super().__init__(vocab, hidden, dtype=torch.bfloat16)
        assert tp_size == 1
        self.calls = []

    def forward(self, ids):
        self.calls.append(ids.clone())
        return super().forward(ids)


class _Comm:
    def __init__(
        self,
        *,
        mapping,
        layer_id,
        is_moe,
        prev_is_moe,
        input_layernorm,
        post_attn_layernorm,
        dense_batch_invariant,
        query_sharded,
    ):
        assert not query_sharded
        self.dense_batch_invariant = dense_batch_invariant
        assert layer_id == 0 and not is_moe and not prev_is_moe
        self.input_layernorm = input_layernorm
        self.post_attn_layernorm = post_attn_layernorm
        self.residuals = []
        self.sizing = []

    def get_num_tokens(self, ctx):
        self.sizing.append(ctx.collective_num_tokens)
        return ctx.collective_num_tokens, ctx.collective_num_tokens

    def input_reduce_norm(self, x, residual):
        assert residual is None
        return self.input_layernorm(x), x

    def pre_attn_comm(self, x, ctx):
        return x

    def post_attn_reduce_norm(self, x, residual, ctx):
        self.residuals.append(residual.clone())
        residual = x + residual
        return self.post_attn_layernorm(residual), residual

    def pre_mlp_comm(self, x, ctx):
        return x

    def post_mlp_fused(self, x, residual, ctx):
        return x, residual

    def final_norm(self, x, residual, ctx, norm):
        return norm(x + residual), None


@dataclass
class _Context:
    attn_backend: object
    token_to_kv_pool: object
    bs: int
    num_extends: int
    input_num_tokens: int
    forward_mode: object
    draft_narrowing: object = None
    gather_ids: torch.Tensor | None = None
    global_bs: list[int] | None = None
    global_num_tokens: list[int] | None = None
    collective_num_tokens: int | None = None
    collective_global_num_tokens: list[int] | None = None


@pytest.fixture
def nextn(adapter, monkeypatch):
    namespace = adapter.Dots3NoteDecoderLayer.__init__.__globals__
    monkeypatch.setitem(namespace, "CommManager", _Comm)
    monkeypatch.setitem(namespace, "DeepseekV3MLP", _MLP)
    monkeypatch.setitem(namespace, "ReplicatedLinear", _LoadableLinear)
    namespace = dict(
        namespace,
        copy=copy,
        VocabParallelEmbedding=_Embedding,
        contextmanager=contextmanager,
    )
    _definitions(
        RUNTIME / "execution/context.py", ["report_collective_sizing"], namespace
    )
    _definitions(
        RUNTIME / "models/dots3_note_nextn.py",
        ["Dots3NoteModelNextN", "Dots3NoteForCausalLMNextN"],
        namespace,
    )
    namespace["LogitsMetadata"] = SimpleNamespace(from_forward_context=lambda ctx: ctx)
    return SimpleNamespace(**namespace)


def _model(nextn, *, quantized):
    config = _small_config()
    config.vocab_size = 128
    config.intermediate_size = 256
    mapping = SimpleNamespace(
        attn=SimpleNamespace(
            tp_rank=0, tp_size=1, tp_group=None, qcp_size=1, dcp_size=1
        ),
        moe=SimpleNamespace(ep_rank=0, ep_size=1),
        pp_size=1,
    )
    model = nextn.Dots3NoteForCausalLMNextN.__new__(nextn.Dots3NoteForCausalLMNextN)
    nn.Module.__init__(model)
    model.config = config
    model.mapping = mapping
    model.quant_config = _Fp8Config() if quantized else None
    model.model = nextn.Dots3NoteModelNextN(
        config, mapping, quant_config=model.quant_config, prefix="model"
    )
    model.lm_head = nn.Linear(64, 128, bias=False, dtype=torch.bfloat16)
    model.logits_processor = Mock(
        side_effect=lambda ids, hidden, head, meta: SimpleNamespace(
            hidden_states=hidden
        )
    )
    return model


def test_dense_batch_invariant_reaches_mlp_and_communication(nextn, monkeypatch):
    monkeypatch.setitem(
        nextn.global_server_args_dict, "tp_batch_invariant", "attn+dense"
    )
    layer = _model(nextn, quantized=False).model.layers[0]
    assert layer.mlp.batch_invariant and layer.comm_manager.dense_batch_invariant


def test_logits_processor_uses_shared_head_topology(nextn, monkeypatch):
    model = _model(nextn, quantized=False)
    model.mapping.attn.has_dp = True
    model.mapping.lm_head = SimpleNamespace(
        tp_rank=1, tp_size=2, tp_group=(0, 1), has_tp=True
    )
    processor = Mock()
    monkeypatch.setitem(
        model.resolve_logits_processor.__globals__, "LogitsProcessor", processor
    )
    model.resolve_logits_processor(model.config)
    assert processor.call_args.kwargs == dict(
        skip_all_gather=True,
        do_argmax=True,
        tp_rank=1,
        tp_size=2,
        tp_group=(0, 1),
        dp_lm_head_tp=True,
    )


def test_logits_metadata_keeps_narrowed_dp_row_counts(nextn, monkeypatch):
    # Execute the production metadata/counting code without native/GPU imports.
    namespace = dict(
        __name__=__name__, dataclasses=dataclasses, IntEnum=IntEnum, auto=auto
    )
    _definitions(
        RUNTIME / "execution/forward_batch_info.py", ["CaptureHiddenMode"], namespace
    )
    _definitions(RUNTIME / "layers/logits_processor.py", ["LogitsMetadata"], namespace)
    _definitions(
        RUNTIME / "distributed/comm_manager.py", ["dp_group_row_counts"], namespace
    )
    row_counts = _method(
        RUNTIME / "layers/logits_processor.py",
        "LogitsProcessor",
        "_lm_head_tp_row_counts",
        namespace,
    )
    model = _model(nextn, quantized=False)
    monkeypatch.setitem(
        model.forward.__wrapped__.__globals__,
        "LogitsMetadata",
        namespace["LogitsMetadata"],
    )
    ctx = SimpleNamespace(
        bs=2,
        input_num_tokens=8,
        forward_mode=nextn.ForwardMode.DECODE,
        capture_hidden_mode=namespace["CaptureHiddenMode"].LAST,
        gather_ids=torch.tensor([1, 7]),
        logits_rows_selected=False,
        input_logprob_rows=None,
        query_shard=None,
        all_decode_or_idle=True,
        global_bs=[3, 0, 2, 1],
        global_num_tokens=[12, 0, 8, 4],
        collective_num_tokens=None,
        collective_global_num_tokens=None,
        draft_narrowing=None,
    )
    captured = torch.arange(8 * 64).view(8, 64).bfloat16()

    def decoder(ids, positions, ctx, *, input_embeds, captured_hidden_states):
        hidden = captured_hidden_states
        if ctx.draft_narrowing is not None:
            hidden = hidden.index_select(0, ctx.gather_ids)
        assert ctx.collective_num_tokens == hidden.shape[0]
        return hidden, None

    monkeypatch.setattr(model.model, "forward", decoder)
    head_tp = SimpleNamespace(tp_rank=1, tp_size=2, tp_group=(0, 2))
    model.logits_processor = Mock(
        side_effect=lambda ids, hidden, head, metadata: row_counts(
            head_tp, hidden, metadata
        )
    )
    for narrowing, expected_counts in ((True, [3, 2]), (False, [12, 8])):
        ctx.draft_narrowing = object() if narrowing else None
        counts = model(ctx, torch.arange(8), torch.arange(8), captured)
        assert counts == expected_counts
        metadata = model.logits_processor.call_args.args[3]
        assert isinstance(metadata, namespace["LogitsMetadata"])
        assert metadata.collective_global_num_tokens == (
            ctx.global_bs if narrowing else ctx.global_num_tokens
        )
        assert metadata.global_num_tokens == ctx.global_num_tokens == [12, 0, 8, 4]
        assert ctx.collective_num_tokens is None
        assert ctx.collective_global_num_tokens is None


def _checkpoint(model, *, quantized):
    weights = {}
    for name, module in model.model.named_modules():
        if isinstance(module, _Linear):
            parts = (
                ("gate_proj", "up_proj") if name.endswith("gate_up_proj") else (None,)
            )
            for i, half in enumerate(parts):
                source = f"model.layers.46.{name.removeprefix('layers.0.')}"
                shape = list(module.weight.shape)
                if half is not None:
                    source = source.replace("gate_up_proj", half)
                    shape[0] //= 2
                weights[f"{source}.weight"] = torch.full(
                    shape,
                    i + 1,
                    dtype=torch.float8_e4m3fn if quantized else torch.bfloat16,
                )
                if quantized:
                    weights[f"{source}.weight_scale_inv"] = torch.full(
                        tuple((n + 127) // 128 for n in shape), 0.125 * (i + 1)
                    )
        elif isinstance(module, _RMSNorm):
            source = (
                "shared_head.norm" if name == "norm" else name.removeprefix("layers.0.")
            )
            weights[f"model.layers.46.{source}.weight"] = torch.full_like(
                module.weight, 0.75
            )
    weights["model.mtp.embed_tokens.weight"] = torch.full_like(
        model.model.embed_tokens.weight, 0.5
    )
    return weights


@torch.no_grad()
def test_native_checkpoint_remaps_scales_and_head_only_binding(nextn):
    model = _model(nextn, quantized=True)
    weights = _checkpoint(model, quantized=True)
    assert len(weights) == 29
    assert all(model.checkpoint_weight_name_filter(name) for name in weights)
    assert not any(
        model.checkpoint_weight_name_filter(name)
        for name in (
            "lm_head.weight",
            "model.embed_tokens.weight",
            "model.layers.45.eh_proj.weight",
            "model.layers.47.eh_proj.weight",
            "model.layers.460.eh_proj.weight",
            "model.mtp.other.weight",
        )
    )
    model.load_weights(
        reversed(
            list(weights.items())
            + [
                ("model.layers.45.mlp.experts.0.gate_proj.weight", torch.tensor(99)),
                ("lm_head.weight", torch.tensor(99)),
                ("model.embed_tokens.weight", torch.tensor(99)),
            ]
        )
    )
    layer = model.model.layers[0]
    assert not layer.is_moe_layer and layer.self_attn.indexer is None
    assert layer.self_attn.layer_id == 0 and layer.self_attn.window_left == 512
    assert model.config.num_hidden_layers == 46
    assert model.config.layer_types[0] == "full_attention"
    assert torch.all(layer.self_attn.kv_a_proj_with_mqa.weight == 0.125)
    assert torch.all(layer.self_attn.g_proj.weight == 0.125)
    assert torch.all(layer.self_attn.w_kc == 0.125)
    assert torch.all(layer.mlp.gate_up_proj.weight.float()[:256] == 1)
    assert torch.all(layer.mlp.gate_up_proj.weight.float()[256:] == 2)
    assert torch.all(layer.mlp.gate_up_proj.weight_scale_inv[:2] == 0.125)
    assert torch.all(layer.mlp.gate_up_proj.weight_scale_inv[2:] == 0.25)
    assert torch.all(model.model.eh_proj.weight.float() == 1)
    assert torch.all(model.model.eh_proj.weight_scale_inv == 0.125)
    assert torch.all(model.model.norm.weight == 0.75)
    embed = model.model.embed_tokens.weight
    norm = model.model.norm.weight
    target_embed = nn.Parameter(torch.zeros_like(embed))
    head = nn.Parameter(torch.zeros_like(model.lm_head.weight))
    model.set_embed_and_head(target_embed, head)
    assert model.lm_head.weight is head
    assert model.model.embed_tokens.weight is embed and torch.all(embed == 0.5)
    assert model.model.norm.weight is norm and torch.all(norm == 0.75)
    with pytest.raises(ValueError, match="shapes"):
        model.set_embed_and_head(target_embed, nn.Parameter(torch.zeros(1)))
    assert model.get_hot_token_id() is None


@pytest.mark.parametrize(
    "missing",
    [
        "model.mtp.embed_tokens.weight",
        "model.layers.46.shared_head.norm.weight",
        "model.layers.46.mlp.up_proj.weight_scale_inv",
        "model.layers.46.eh_proj.weight_scale_inv",
        "model.layers.46.self_attn.g_proj.weight_scale_inv",
        "model.layers.46.self_attn.q_b_proj.weight",
    ],
)
def test_missing_checkpoint_tensors_fail(nextn, missing):
    model = _model(nextn, quantized=True)
    weights = _checkpoint(model, quantized=True)
    del weights[missing]
    with pytest.raises(ValueError, match="Missing|Incomplete"):
        model.load_weights(weights.items())


@pytest.mark.parametrize(
    "source",
    [
        "model.layers.46.self_attn.indexer.wk.weight",
        "model.layers.46.mlp.experts.0.gate_proj.weight",
        "model.layers.46.shared_head.head.weight",
        "model.mtp.other.weight",
    ],
)
def test_unsupported_mtp_tensors_fail(nextn, source):
    model = _model(nextn, quantized=True)
    with pytest.raises(KeyError, match="Unsupported"):
        model.load_weights([(source, torch.ones(1))])


@pytest.mark.parametrize(
    "bad_scale", [torch.ones(2, 2), torch.ones(1, 1, dtype=torch.bfloat16)]
)
def test_eh_proj_scale_grid_is_checked(nextn, bad_scale):
    model = _model(nextn, quantized=True)
    weights = _checkpoint(model, quantized=True)
    weights["model.layers.46.eh_proj.weight_scale_inv"] = bad_scale
    with pytest.raises((ValueError, AssertionError)):
        model.load_weights(weights.items())


class _DraftBackend(_Backend):
    def __init__(self, *args, frontier, **kwargs):
        super().__init__(*args, **kwargs)
        # MLA draft metadata is bs-wide, even when its write window is bs*N.
        self.leaf.forward_decode_metadata.q_len_per_req = 1
        self.frontier = frontier
        self.publish_count = 0
        self.queries = []

    def publish_accepted_prefix(self):
        assert sum(len(loc) for _, loc in self.pool.writes) == len(
            self.prefill_locs
        ) + len(self.decode_locs)
        self.publish_count += 1
        self.leaf.forward_decode_metadata.seq_lens_k.copy_(self.frontier)

    @contextmanager
    def override_num_extends(self, count):
        previous = self.num_extends
        self.num_extends = count
        try:
            yield
        finally:
            self.num_extends = previous

    def forward(self, *args, **kwargs):
        assert self.publish_count == 1
        assert kwargs["record_kv_cache"] == (not self.mode.is_decode_or_idle())
        self.queries.append(args[0].clone())
        return super().forward(*args, **kwargs)


@pytest.mark.parametrize("mode_name", ["DECODE", "MIXED", "EXTEND"])
@torch.no_grad()
def test_accepted_rows_full_kv_writes_residual_and_postnorm(nextn, mode_name):
    torch.manual_seed(42)
    model = _model(nextn, quantized=False)
    for name, param in model.model.named_parameters():
        param.copy_(torch.randn_like(param) * 0.1)
    layer = model.model.layers[0]
    attn = layer.self_attn
    attn.prepare_weights()
    mode = getattr(nextn.ForwardMode, mode_name)
    prefixes, lengths = ([2], [3]) if mode_name != "DECODE" else ([], [])
    # Accept lengths include the guaranteed target token: 1 means zero proposals
    # accepted; 2 is partial; 4 accepts every proposal of a verify-width-4 round.
    decode_lengths = [6, 6, 6, 0] if mode_name != "EXTEND" else []
    accepted = [1, 2, 4, 1] if decode_lengths else []
    frontier = torch.tensor(
        ([5] if prefixes else []) + ([3, 4, 6, 0] if accepted else [])
    )
    pool = _Pool(32, 96, ([5] if prefixes else []) + decode_lengths)
    backend = _DraftBackend(
        attn,
        pool,
        mode,
        prefixes,
        lengths,
        decode_lengths,
        q_len_per_req=4,
        frontier=frontier,
    )
    bs = len(frontier)
    for req in range(bs - (1 if accepted else 0)):
        rows = torch.randn(2, 96).bfloat16() * 0.1
        pool.set_mla_kv_buffer(
            attn.attn_mqa, pool.slots(req, 0, 2), rows[:, :32], rows[:, 32:]
        )
    pool.writes.clear()
    pool.set_mla_kv_buffer = Mock(side_effect=AssertionError("legacy writer"))
    num_prefill = sum(lengths)
    rows = num_prefill + 4 * len(decode_lengths)
    gather = torch.tensor(
        ([num_prefill - 1] if prefixes else [])
        + [num_prefill + 4 * i + n - 1 for i, n in enumerate(accepted)]
    )
    ctx = _Context(backend, pool, bs, len(prefixes), rows, mode, backend, gather)
    # Padding beyond logical input rows must never reach the cache.
    input_ids = torch.arange(rows + 2) % 128
    positions = torch.tensor(
        ([2, 3, 4] if prefixes else []) + ([2, 3, 4, 5] * len(decode_lengths)) + [0, 0]
    )
    captured = torch.randn(rows + 2, 64).bfloat16()
    result = model(ctx, input_ids, positions, captured_hidden_states=captured)
    fused = model.model.eh_proj.calls[0]
    expected_fused = torch.cat(
        (
            model.model.enorm(model.model.embed_tokens(input_ids)),
            model.model.hnorm(captured),
        ),
        -1,
    )
    torch.testing.assert_close(fused, expected_fused, rtol=0, atol=0)
    hidden = torch.nn.functional.linear(fused, model.model.eh_proj.weight)
    normalized = layer.input_layernorm(hidden)
    _, q, latent, rope = attn.project(normalized[:rows])
    q[..., attn.qk_nope_head_dim :] = _rotate(
        q[..., attn.qk_nope_head_dim :], positions[:rows], attn.rotary_emb.base
    )
    rope = _rotate(rope[:, None], positions[:rows], attn.rotary_emb.base).squeeze(1)
    torch.testing.assert_close(
        backend.queries[0], attn.absorb_query(q[gather]), rtol=0, atol=0
    )
    torch.testing.assert_close(
        layer.comm_manager.residuals[0], hidden[gather], rtol=0, atol=0
    )
    assert backend.publish_count == 1 and backend.num_extends == len(prefixes)
    assert layer.comm_manager.sizing == [bs]
    assert ctx.collective_num_tokens is None
    assert layer.mlp.inputs[0].shape == (bs, 64)
    for req, n in enumerate(frontier.tolist()):
        selected = backend.selected[0][req]
        torch.testing.assert_close(
            selected[selected >= 0], pool.slots(req, max(0, n - 513), n)
        )
    for start, end, loc in (
        (0, num_prefill, backend.prefill_locs),
        (num_prefill, rows, backend.decode_locs),
    ):
        if end > start:
            # Ignore the null-page padding rows, which may overwrite each other.
            live = loc != 0
            torch.testing.assert_close(
                pool.rows(loc[live]),
                torch.cat((latent[start:end][live], rope[start:end][live]), -1),
            )
    values = attn.expand_values(
        backend._attention(backend.queries[0], backend.selected[0])
    )
    gate = torch.sigmoid(attn.g_proj(normalized[gather])[0])
    attention_out = attn.o_proj((values * gate[..., None]).flatten(1))[0]
    residual = hidden[gather] + attention_out
    expected = model.model.norm(
        layer.mlp(layer.post_attention_layernorm(residual)) + residual
    )
    torch.testing.assert_close(result.hidden_states, expected, rtol=0, atol=0)
    assert result.hidden_states.shape == (bs, 64)

    # Later Eagle steps use ordinary one-row decode, at frontier (not frontier+1).
    backend = _Backend(
        attn,
        pool,
        nextn.ForwardMode.DECODE,
        [],
        [],
        [n + 1 if n else 0 for n in frontier.tolist()],
        q_len_per_req=1,
    )
    next_ctx = _Context(backend, pool, bs, 0, bs, nextn.ForwardMode.DECODE)
    pool.writes.clear()
    next_output = model(next_ctx, input_ids[:bs], frontier, result.hidden_states)
    assert next_output.hidden_states.shape == (bs, 64)
    assert len(pool.writes) == 1 and len(pool.writes[0][1]) == bs
    torch.testing.assert_close(
        model.model.eh_proj.calls[-1][:, 64:],
        model.model.hnorm(result.hidden_states),
        rtol=0,
        atol=0,
    )


@torch.no_grad()
def test_forward_preserves_explicit_embeddings_positions_and_recurrent_hidden(nextn):
    model = _model(nextn, quantized=False)
    for param in model.model.parameters():
        param.copy_(torch.randn_like(param) * 0.1)
    positions = torch.tensor([9, 20])
    embeds = torch.randn(2, 64).bfloat16()
    captured = torch.randn(2, 64).bfloat16()
    seen = []

    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.comm_manager = SimpleNamespace(
                final_norm=lambda x, residual, ctx, norm: (norm(x + residual), None)
            )

        def forward(self, position, hidden, ctx, residual):
            assert position is positions and residual is None
            seen.append(hidden.clone())
            return hidden * 2, hidden * 3

    model.model.layers[0] = Decoder()
    ctx = _Context(None, None, 2, 0, 2, nextn.ForwardMode.DECODE)
    for _ in range(2):
        result = model(ctx, torch.tensor([1, 2]), positions, captured, embeds)
        expected_input = torch.cat(
            (model.model.enorm(embeds), model.model.hnorm(captured)), -1
        )
        torch.testing.assert_close(
            model.model.eh_proj.calls[-1], expected_input, rtol=0, atol=0
        )
        torch.testing.assert_close(
            result.hidden_states,
            model.model.norm(seen[-1] * 2 + seen[-1] * 3),
            rtol=0,
            atol=0,
        )
        captured = result.hidden_states
        positions.add_(1)
    assert not model.model.embed_tokens.calls
    with pytest.raises(ValueError, match="captured_hidden_states"):
        model(ctx, torch.tensor([1, 2]), positions, input_embeds=embeds)


@torch.no_grad()
def test_idle_needs_no_capture_and_does_not_publish_or_write(nextn):
    model = _model(nextn, quantized=False)
    publisher = Mock()
    ctx = _Context(
        None,
        None,
        0,
        0,
        0,
        nextn.ForwardMode.IDLE,
        publisher,
        torch.empty(0, dtype=torch.long),
    )
    result = model(
        ctx, torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)
    )
    assert result.hidden_states.shape == (0, 64)
    publisher.publish_accepted_prefix.assert_not_called()
    assert not model.model.embed_tokens.calls and not model.model.eh_proj.calls


def test_public_geometry_and_source_config_unchanged(nextn):
    config = _config_class()(**_config_values())
    mapping = _model(nextn, quantized=False).mapping
    with torch.device("meta"):
        model = nextn.Dots3NoteModelNextN(
            config, mapping, quant_config=_Fp8Config(), prefix="model"
        )
    attn = model.layers[0].self_attn
    assert (
        attn.num_heads,
        attn.kv_lora_rank,
        attn.qk_nope_head_dim,
        attn.qk_rope_head_dim,
        attn.v_head_dim,
    ) == (64, 1024, 192, 64, 128)
    assert model.eh_proj.weight.shape == (5120, 10240)
    assert model.eh_proj.weight_scale_inv.shape == (40, 80)
    assert model.layers[0].mlp.gate_up_proj.weight.shape == (2 * 13824, 5120)
    assert attn.rotary_emb.base == 50000 and attn.window_left == 512
    assert config.num_hidden_layers == 46 and config.layer_types[0] == "full_attention"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("mode_name", ["EXTEND", "MIXED", "DECODE"])
@torch.no_grad()
def test_gpu_native_eagle_narrowing(monkeypatch, mode_name):
    from runtime.cache.test_dots3_note_compatibility import build_mtp_pools
    from runtime.cache.test_dots3_note_recipe import inputs as cache_inputs

    from tokenspeed.runtime.distributed.mapping import Mapping
    from tokenspeed.runtime.execution.context import ForwardContext
    from tokenspeed.runtime.execution.drafter.eagle import (
        AcceptedPrefixPublisher,
        Eagle,
        EagleDraftInput,
    )
    from tokenspeed.runtime.execution.factory import configure_draft_target
    from tokenspeed.runtime.execution.forward_batch_info import (
        CaptureHiddenMode,
        ForwardMode,
    )
    from tokenspeed.runtime.execution.input_buffer import InputBuffers
    from tokenspeed.runtime.execution.output_layout import ForwardOutputLayout
    from tokenspeed.runtime.layers.paged_attention import bind_cache_groups
    from tokenspeed.runtime.models.dots3_note_nextn import Dots3NoteForCausalLMNextN
    from tokenspeed.runtime.utils.env import global_server_args_dict

    torch.manual_seed(123)
    monkeypatch.setitem(global_server_args_dict, "chunked_prefill_size", 32)
    monkeypatch.setitem(global_server_args_dict, "mla_chunk_multiplier", 1)
    mapping = Mapping(rank=0, world_size=1)
    config = _config_class()(
        **(_config_values() | {"vocab_size": 128, "max_position_embeddings": 1024})
    )
    # Only vocabulary/context capacity shrink; all projections and the dense MLP
    # retain the checkpoint geometry. No target weights are allocated.
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        with torch.device("cuda"):
            model = Dots3NoteForCausalLMNextN(
                config,
                mapping,
                quant_config=None,
            )
    finally:
        torch.set_default_dtype(previous_dtype)
    for parameter in model.parameters():
        if parameter.ndim == 1:
            parameter.fill_(1)
        else:
            parameter.normal_(std=0.25 / parameter.shape[1] ** 0.5)
    model.model.norm.weight.fill_(0.75)
    model.post_load_weights()
    embed = model.model.embed_tokens.weight
    norm = model.model.norm.weight
    target_embed = nn.Parameter(torch.zeros_like(embed))
    head = nn.Parameter(torch.randn_like(model.lm_head.weight) * 0.01)
    configure_draft_target(
        SimpleNamespace(speculative_algorithm="MTP"),
        SimpleNamespace(
            model=SimpleNamespace(
                get_embed_and_head=lambda: (target_embed, head),
                logits_processor=model.logits_processor,
            ),
            mapping=mapping,
        ),
        SimpleNamespace(
            model=model,
            model_config=SimpleNamespace(requires_request_token_history=False),
        ),
    )
    assert model.lm_head.weight is head
    assert model.model.embed_tokens.weight is embed
    assert model.model.norm.weight is norm and torch.all(norm == 0.75)
    assert config.num_hidden_layers == 46 and len(model.model.layers) == 1
    layer = model.model.layers[0]
    attn = layer.self_attn
    assert attn.num_local_heads == 64 and attn.kv_lora_rank == 1024
    assert not layer.is_moe_layer and attn.indexer is None

    inputs = cache_inputs.__wrapped__(monkeypatch)
    inputs["server_args"].mapping = mapping
    inputs["server_args"].attn_tp_size = 1
    inputs["server_args"].max_num_seqs = 4
    inputs["model_config"].hf_config = config
    inputs["model_config"].hf_text_config = config
    inputs["model_config"].context_len = 640
    inputs["decode_input_tokens"] = 4
    _, pool, router = build_mtp_pools(inputs, device="cuda")
    bind_cache_groups(model, pool)
    leaf = router.leaf_for(attn.attn_mqa)
    cache = pool.get_key_buffer(0)
    assert attn.attn_mqa.group_id == "draft.swa"
    assert cache.shape[1:] == (32, 1, 1088)
    assert cache.stride(0) * cache.element_size() == 71424
    cache.normal_(std=0.1)
    cache[0].zero_()
    mode = getattr(ForwardMode, mode_name)
    prefixes = (
        [31, 511] if mode_name == "EXTEND" else [511] if mode_name == "MIXED" else []
    )
    extends = [2, 3] if mode_name == "EXTEND" else [3] if mode_name == "MIXED" else []
    valid = [] if mode_name == "EXTEND" else [31, 63, 511]
    accepted = [1] * len(extends) + ([1, 2, 4] if valid else [])
    if mode_name == "DECODE":
        valid += [0]
        accepted += [1]
    bs = len(accepted)
    actual_bs = bs - int(mode_name == "DECODE")
    num_extends = len(extends)
    num_tokens = sum(extends) + len(valid) * 4
    seq_lens = [p + n for p, n in zip(prefixes, extends)] + [v + 4 for v in valid]
    if mode_name == "DECODE":
        seq_lens[-1] = 1
    buffers = InputBuffers(bs, num_tokens + 2, state_write_padding_pool_index=0)
    buffers.seq_lens_buf.copy_(torch.tensor(seq_lens, device="cuda", dtype=torch.int32))
    buffers.req_pool_indices_buf.copy_(torch.arange(bs, device="cuda"))
    positions = [p + i for p, n in zip(prefixes, extends) for i in range(n)]
    positions += [v + i for v in valid for i in range(4)]
    buffers.positions_buf[:num_tokens].copy_(torch.tensor(positions, device="cuda"))
    buffers.input_lengths_buf[:num_extends].copy_(torch.tensor(extends, device="cuda"))
    buffers.shifted_prefill_ids_buf[:num_tokens].copy_(
        torch.arange(num_tokens, device="cuda") + 10
    )
    # A real next prompt token at an intermediate chunk end must not be replaced;
    # the last chunk's -1 sentinel must be replaced by the target's sampled token.
    if extends:
        buffers.shifted_prefill_ids_buf[sum(extends) - 1] = -1
    tables = {
        "draft.swa": torch.arange(
            1, bs * leaf.max_num_pages + 1, device="cuda", dtype=torch.int32
        ).view(bs, -1)
    }
    states = SimpleNamespace(
        draft_probs=None,
        valid_cache_lengths=torch.tensor(
            [0] * num_extends + valid, device="cuda", dtype=torch.int32
        ),
    )
    captured = torch.randn(
        num_tokens, config.hidden_size, device="cuda", dtype=torch.bfloat16
    )
    draft_input = EagleDraftInput(
        input_num_tokens=num_tokens,
        num_extends=num_extends,
        forward_mode=mode,
        base_model_output=torch.arange(
            num_extends + 4 * len(valid), device="cuda", dtype=torch.int32
        )
        + 50,
        accept_lengths=torch.tensor(accepted, device="cuda", dtype=torch.int32),
        base_out_hidden_states=captured,
    )
    contexts = []
    outputs = []

    def run_model(*, ctx, input_ids, positions, captured_hidden_states, spec_step_idx):
        if spec_step_idx:
            assert ctx.draft_narrowing is None and ctx.input_num_tokens == bs
            torch.testing.assert_close(
                positions, (frontier + spec_step_idx - 1).to(positions.dtype)
            )
            torch.testing.assert_close(
                captured_hidden_states, outputs[-1].hidden_states, rtol=0, atol=0
            )
            torch.testing.assert_close(input_ids, outputs[-1].next_token_ids)
        contexts.append(ctx)
        output = model(ctx, input_ids, positions, captured_hidden_states)
        outputs.append(output)
        return output

    eagle = Eagle(
        4,
        3,
        SimpleNamespace(
            model=model,
            mapping=mapping,
            device="cuda",
            forward=run_model,
            model_config=SimpleNamespace(requires_request_token_history=False),
        ),
        attn_backend=router,
        token_to_kv_pool=pool,
        runtime_states=states,
        input_buffers=buffers,
        vocab_size=config.vocab_size,
    )

    def refresh(actual, *, graph):
        if extends:
            prefix_cpu = torch.tensor(prefixes, dtype=torch.int32)
            extend_cpu = torch.tensor(extends, dtype=torch.int32)
            router.init_forward_metadata(
                bs,
                num_extends,
                buffers.req_pool_indices_buf,
                buffers.seq_lens_buf,
                mode,
                block_tables=tables,
                block_tables_cpu={gid: table.cpu() for gid, table in tables.items()},
                query_shard=None,
                extend_seq_lens=extend_cpu.cuda(),
                extend_seq_lens_cpu=extend_cpu,
                extend_prefix_lens=prefix_cpu.cuda(),
                extend_prefix_lens_cpu=prefix_cpu,
                extend_replay_lens_cpu=torch.zeros_like(extend_cpu),
                extend_prompt_lens_cpu=prefix_cpu + extend_cpu,
                extend_with_prefix=True,
            )
        # ForwardStepRunner also refreshes draft decode metadata after prefill
        # init: step 0 writes ragged KV, then attends as one decode per request.
        eagle.draft_seq_lens_buf.copy_(buffers.seq_lens_buf)
        router.refresh_decode_metadata(
            bs,
            actual,
            buffers.req_pool_indices_buf,
            eagle.draft_seq_lens_buf,
            forward_mode=ForwardMode.DECODE,
            block_tables=tables,
            num_extends=num_extends,
            for_graph_replay=graph,
        )

    refresh(actual_bs, graph=False)
    ids, gather = eagle._get_first_step_input(draft_input, bs, num_tokens)
    if len(extends) == 2:
        assert ids[extends[0] - 1].item() == 10 + extends[0] - 1
    if extends:
        assert ids[sum(extends) - 1] == draft_input.base_model_output[num_extends - 1]
    frontier = eagle._accepted_frontier(bs, draft_input)
    publisher = AcceptedPrefixPublisher(router, frontier)
    fused = model.model.eh_proj(
        torch.cat(
            (
                model.model.enorm(model.model.embed_tokens(ids)),
                model.model.hnorm(captured),
            ),
            -1,
        )
    )[0]
    normalized = layer.input_layernorm(fused)
    _, q, latent, rope = attn.project(normalized)
    rotated_q, rope = attn.rotary_emb(
        buffers.positions_buf[:num_tokens],
        q[..., attn.qk_nope_head_dim :].clone(),
        rope[:, None].clone(),
    )
    q[..., attn.qk_nope_head_dim :] = rotated_q
    rope = rope.squeeze(1)
    locations = torch.cat(
        [
            router.write_locations(attn.attn_mqa, part)
            for part, count in (
                (ForwardMode.EXTEND, sum(extends)),
                (ForwardMode.DECODE, len(valid)),
            )
            if count
        ]
    )
    writes = Mock(wraps=attn.attn_mqa.latent_prologue)
    monkeypatch.setattr(attn.attn_mqa, "latent_prologue", writes)
    monkeypatch.setattr(
        pool, "set_mla_kv_buffer", Mock(side_effect=AssertionError("legacy writer"))
    )
    advance = router.advance_draft_forward_metadata

    def publish(lengths):
        assert (
            sum(call.kwargs["slots"].numel() for call in writes.call_args_list)
            == num_tokens
        )
        live = locations // 32 != 0
        torch.testing.assert_close(
            cache[locations[live] // 32, locations[live] % 32, 0],
            torch.cat((latent, rope), -1)[live],
            rtol=0,
            atol=0,
        )
        advance(lengths)

    publish_spy = Mock(side_effect=publish)
    monkeypatch.setattr(router, "advance_draft_forward_metadata", publish_spy)
    result, _ = eagle._run_first_step(bs, draft_input, publisher)
    assert publish_spy.call_count == 1
    assert leaf.forward_decode_metadata.num_extends == num_extends
    assert contexts[-1].collective_num_tokens is None
    assert result.hidden_states.shape == (bs, 5120)
    assert result.next_token_logits.shape == (bs, 128)
    assert result.next_token_ids.shape == (bs,)
    torch.testing.assert_close(leaf.forward_decode_metadata.seq_lens_k, frontier)

    # Independent dense FP32 attention over the accepted SWA prefix. Future
    # verify rows have already been written but must not affect these queries.
    queries = attn.absorb_query(q[gather]).float()
    values = []
    for req, length in enumerate(frontier[:actual_bs].tolist()):
        logical = torch.arange(max(0, length - 513), length, device="cuda")
        rows = cache[
            tables["draft.swa"][req, logical // 32].long(), logical % 32, 0
        ].float()
        probabilities = (queries[req] @ rows.T * attn.scaling).softmax(-1)
        values.append(probabilities @ rows[:, :1024])
    values = attn.expand_values(torch.stack(values).bfloat16())
    gate = attn.g_proj(normalized[gather[:actual_bs]])[0].sigmoid()
    attention = attn.o_proj((values * gate[..., None]).flatten(1))[0]
    mlp_input, residual = layer.post_attention_layernorm(
        attention, fused[gather[:actual_bs]].clone()
    )
    expected, _ = model.model.norm(layer.mlp(mlp_input), residual)
    torch.testing.assert_close(
        result.hidden_states[:actual_bs], expected, rtol=0.02, atol=0.02
    )
    torch.testing.assert_close(
        result.next_token_logits,
        torch.nn.functional.linear(result.hidden_states, model.lm_head.weight),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        result.next_token_ids,
        result.next_token_logits.argmax(-1).to(result.next_token_ids.dtype),
    )

    monkeypatch.setattr(attn.attn_mqa, "latent_prologue", writes._mock_wraps)
    monkeypatch.setattr(router, "advance_draft_forward_metadata", advance)
    # Extra collective rows are not requests and must neither be cached nor
    # survive the query/gate/residual narrowing.
    refresh(actual_bs, graph=False)
    padded = model(
        contexts[-1],
        torch.cat((ids, ids[:2])),
        torch.cat((buffers.positions_buf[:num_tokens], buffers.positions_buf[:2])),
        torch.cat((captured, captured[:2])),
    )
    torch.testing.assert_close(
        padded.hidden_states[:actual_bs],
        result.hidden_states[:actual_bs],
        rtol=0,
        atol=0,
    )

    next_tokens = torch.empty((bs, 4), device="cuda", dtype=torch.int32)
    with router.override_num_extends(0):
        eagle._run_multi_step_decode(
            bs,
            result.next_token_ids,
            result.hidden_states,
            next_tokens,
            draft_input,
            (None, None),
            frontier,
        )
    assert outputs[-1].hidden_states.shape == (bs, 5120)
    torch.testing.assert_close(next_tokens[:, 2], outputs[-2].next_token_ids.int())
    torch.testing.assert_close(next_tokens[:, 3], outputs[-1].next_token_ids.int())
    torch.testing.assert_close(leaf.forward_decode_metadata.seq_lens_k, frontier + 2)

    if mode_name == "DECODE":
        # Capture the actual Eagle first step, then change acceptance, live batch
        # size and page tables without changing captured tensor addresses.
        refresh(actual_bs, graph=False)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                eagle._run_first_step(bs, draft_input, publisher)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            recorded, _ = eagle._run_first_step(bs, draft_input, publisher)
        for live, accept in ((2, [4, 1, 1, 1]), (0, [1, 1, 1, 1]), (3, [2, 4, 1, 1])):
            draft_input.accept_lengths.copy_(
                torch.tensor(accept, device="cuda", dtype=torch.int32)
            )
            frontier.copy_(eagle._accepted_frontier(bs, draft_input))
            tables["draft.swa"].copy_(tables["draft.swa"].roll(1, dims=1))
            refresh(live, graph=True)
            before_live_cache = cache[1:].clone() if live == 0 else None
            graph.replay()
            if live == 0:
                torch.testing.assert_close(cache[1:], before_live_cache, rtol=0, atol=0)
            replay_hidden = recorded.hidden_states.clone()
            replay_logits = recorded.next_token_logits.clone()
            refresh(live, graph=False)
            eager, _ = eagle._run_first_step(bs, draft_input, publisher)
            torch.testing.assert_close(
                replay_hidden[:live], eager.hidden_states[:live], rtol=0, atol=0
            )
            torch.testing.assert_close(
                replay_logits[:live], eager.next_token_logits[:live], rtol=0, atol=0
            )

    idle = ForwardContext(
        router,
        pool,
        0,
        0,
        0,
        ForwardMode.IDLE,
        output_layout=ForwardOutputLayout(0, 0, 0, 1),
        capture_hidden_mode=CaptureHiddenMode.LAST,
    )
    before = cache.clone()
    empty = torch.empty(0, device="cuda", dtype=torch.int64)
    output = model(idle, empty, empty)
    assert output.hidden_states.shape == (0, 5120)
    assert output.next_token_logits.shape == (0, 128)
    torch.testing.assert_close(cache, before, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.no_grad()
def test_gpu_real_fp8_loader_requires_all_29_native_tensors():
    from tokenspeed.runtime.distributed.mapping import Mapping
    from tokenspeed.runtime.layers.quantization.fp8 import Fp8Config
    from tokenspeed.runtime.models.dots3_note_nextn import Dots3NoteForCausalLMNextN

    config = _config_class()(
        **(
            _config_values()
            | {
                "vocab_size": 128,
                "hidden_size": 128,
                "intermediate_size": 256,
                "swa_q_lora_rank": 128,
                "max_position_embeddings": 640,
            }
        )
    )
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        with torch.device("cuda"):
            model = Dots3NoteForCausalLMNextN(
                config,
                Mapping(rank=0, world_size=1),
                quant_config=Fp8Config(
                    is_checkpoint_fp8_serialized=True,
                    activation_scheme="dynamic",
                    weight_block_size=[128, 128],
                    scale_fmt=None,
                ),
            )
    finally:
        torch.set_default_dtype(previous_dtype)
    # Explicit checkpoint names, independent of the loader's remapping table.
    shapes = {
        "eh_proj": (128, 256),
        "self_attn.q_a_proj": (128, 128),
        "self_attn.q_b_proj": (64 * 256, 128),
        "self_attn.kv_a_proj_with_mqa": (1088, 128),
        "self_attn.kv_b_proj": (64 * 320, 1024),
        "self_attn.g_proj": (64, 128),
        "self_attn.o_proj": (128, 64 * 128),
        "mlp.gate_proj": (256, 128),
        "mlp.up_proj": (256, 128),
        "mlp.down_proj": (128, 256),
    }
    weights = {}
    for name, shape in shapes.items():
        source = f"model.layers.46.{name}"
        weights[f"{source}.weight"] = torch.full(
            shape,
            2 if name == "mlp.up_proj" else 1,
            device="cuda",
            dtype=torch.float8_e4m3fn,
        )
        scale_shape = tuple((n + 127) // 128 for n in shape)
        weights[f"{source}.weight_scale_inv"] = (
            torch.arange(
                1,
                1 + scale_shape[0] * scale_shape[1],
                device="cuda",
                dtype=torch.float32,
            ).view(scale_shape)
            / 128
        )
    for name, width in {
        "enorm": 128,
        "hnorm": 128,
        "shared_head.norm": 128,
        "input_layernorm": 128,
        "post_attention_layernorm": 128,
        "self_attn.q_a_layernorm": 128,
        "self_attn.kv_a_layernorm": 1024,
        "self_attn.k_rope_only_layernorm": 64,
    }.items():
        weights[f"model.layers.46.{name}.weight"] = torch.full(
            (width,), 0.75, device="cuda", dtype=torch.bfloat16
        )
    weights["model.mtp.embed_tokens.weight"] = torch.full(
        (128, 128), 0.5, device="cuda", dtype=torch.bfloat16
    )
    assert len(weights) == 29
    head = model.lm_head.weight.clone()
    for stream in (list(weights.items()), list(reversed(weights.items()))):
        model.load_weights(stream)
        attn = model.model.layers[0].self_attn
        scale = weights["model.layers.46.self_attn.kv_b_proj.weight_scale_inv"]
        dense = (
            scale.repeat_interleave(128, 0)
            .repeat_interleave(128, 1)
            .bfloat16()
            .view(64, 320, 1024)
        )
        torch.testing.assert_close(attn.w_kc, dense[:, :192], rtol=0, atol=0)
        torch.testing.assert_close(
            attn.w_vc, dense[:, 192:].transpose(1, 2), rtol=0, atol=0
        )
        assert attn.g_proj.weight.dtype == torch.bfloat16
        assert attn.kv_a_proj_with_mqa.weight.dtype == torch.bfloat16
        assert model.model.eh_proj.weight.dtype == torch.float8_e4m3fn
        assert torch.all(
            model.model.layers[0].mlp.gate_up_proj.weight.float()[256:] == 2
        )
        torch.testing.assert_close(
            model.lm_head.weight, head, rtol=0, atol=0, equal_nan=True
        )
    for missing in weights:
        with pytest.raises(ValueError, match="Missing|Incomplete"):
            model.load_weights(
                (name, weight) for name, weight in weights.items() if name != missing
            )

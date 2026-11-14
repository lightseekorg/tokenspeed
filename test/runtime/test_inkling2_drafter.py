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

"""Native block resources, accepted features and recurrent proposals on CPU."""

from dataclasses import dataclass
from pathlib import Path
from test.runtime.models.test_inkling2_model import _model
from test.runtime.models.test_inkling2_mtp import _config as _draft_config
from test.runtime.models.test_inkling2_mtp import _head_oracle, _native_weights

import pytest
import torch
import torch.nn.functional as F

from tokenspeed.runtime.configs.inkling2 import Inkling2MTPConfig
from tokenspeed.runtime.configs.model_config import ModelConfig
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.drafter import get_drafter_impl
from tokenspeed.runtime.execution.drafter.inkling2 import Inkling2DSpark
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.execution.input_buffer import InputBuffers
from tokenspeed.runtime.execution.model_runner import ModelRunner
from tokenspeed.runtime.execution.output_layout import ForwardOutputLayout
from tokenspeed.runtime.execution.runtime_states import RuntimeStates
from tokenspeed.runtime.layers.attention.backends.specific.inkling import (
    InklingAttnBackend,
)
from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig
from tokenspeed.runtime.layers.attention.kv_cache.factory import (
    create_cache_arena,
    create_cache_pool,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.inkling import InklingRecipe
from tokenspeed.runtime.layers.attention.registry import create_paged_router
from tokenspeed.runtime.layers.logits_processor import LogitsProcessorOutput
from tokenspeed.runtime.layers.paged_attention import (
    bind_cache_groups,
    check_block_drafter_storage,
)
from tokenspeed.runtime.models.inkling2_mtp import (
    Inkling2ForConditionalGenerationMTP,
    padded_shard_ids_to_vocab,
    vocab_ids_to_padded_shards,
)
from tokenspeed.runtime.utils.server_args import ServerArgs


@dataclass
class _LoadedRunner:
    """Replace checkpoint runner construction, not its forward or model arithmetic."""

    model: Inkling2ForConditionalGenerationMTP
    model_config: ModelConfig
    server_args: ServerArgs
    device: str
    is_generation: bool
    forward_accepts_spec_step_idx: bool

    forward = ModelRunner.forward

    @property
    def mapping(self):
        return self.server_args.mapping


def build_native_drafter(
    path: Path, max_bs: int, device: str, draft_overrides: dict, context_length: int
):
    with torch.device(device):
        target = _model()
        target.config.mtp_config = _draft_config().to_dict()
        target.config.mtp_config.update(target_layer_ids=[0, 1, 2], **draft_overrides)
        model = Inkling2ForConditionalGenerationMTP(
            Inkling2MTPConfig.from_target(target.config), target.mapping, None
        )
    torch.manual_seed(14)
    with torch.no_grad():
        target.model.embed_tokens.weight.normal_(std=0.15)
        target.model.embed_norm.weight.fill_(1.5)
        target.lm_head.weight.normal_(std=0.15)
    model.load_weights(_native_weights(model))
    model.configure_target(target, target.text_config)
    target_path, draft_path = path / "target", path / "draft"
    target.config.save_pretrained(target_path)
    model.config.save_pretrained(draft_path)
    args = ServerArgs(
        model=str(target_path),
        speculative_draft_model_path=str(draft_path),
        speculative_algorithm="DSPARK",
        speculative_num_steps=model.block_size - 1,
        speculative_num_draft_tokens=model.block_size,
        max_num_seqs=max_bs,
        max_model_len=context_length,
        max_total_tokens=256,
        chunked_prefill_size=32,
        attention_backend="triton",
        drafter_attention_backend="triton",
        device=device,
        kv_cache_dtype="bfloat16",
        enable_prefix_caching=False,
        disable_kvstore=True,
    )
    args.mapping.rank = 0
    common = dict(
        trust_remote_code=False,
        model_override_args="{}",
        dtype="bfloat16",
        server_args=args,
        context_length=context_length,
    )
    target_config = ModelConfig(str(target_path), **common)
    draft_config = ModelConfig(str(draft_path), is_draft_worker=True, **common)
    attention = MHAConfig.generate(args, draft_config, is_draft=True)
    recipe = InklingRecipe(
        server_args=args,
        model_config=target_config,
        attn_config=MHAConfig.generate(args, target_config, is_draft=False),
        draft_model_config=draft_config,
        draft_attn_config=attention,
        cache_budget_bytes=32 << 20,
        probe_batch_rows=4,
        decode_input_tokens=model.block_size,
        overlap_schedule_depth=0,
    )
    setup = recipe.setup()
    arena = create_cache_arena(setup.spec, device=device, enable_memory_saver=False)
    target_attention = recipe.attn_config
    target_pool = create_cache_pool(
        setup.spec.layer_view(first_layer=0, num_layers=2),
        target_attention,
        arena,
        num_layers=2,
        rank=0,
        field_layer_offset=0,
    )
    target_backend = InklingAttnBackend.from_config(
        create_paged_router(
            target_attention, target_config.attention_arch, backend_name="triton"
        ),
        target.text_config,
        target_attention,
        num_layers=2,
        is_draft=False,
        enable_layerwise_cache_ready=False,
    )
    target_backend.set_cache_pool(target_pool)
    target_backend.init_cuda_graph_state(max_bs, max_extend_tokens=32)
    draft_spec = setup.spec.layer_view(first_layer=2, num_layers=5, family="mha")
    pool = create_cache_pool(
        draft_spec, attention, arena, num_layers=5, rank=0, field_layer_offset=2
    )
    bind_cache_groups(target, target_pool)
    bind_cache_groups(model, pool)
    check_block_drafter_storage(model, target_pool)
    backend = create_paged_router(
        attention, draft_config.attention_arch, backend_name="triton"
    )
    backend.set_cache_pool(pool)
    backend.init_cuda_graph_state(max_bs, max_extend_tokens=32)
    buffers = InputBuffers(
        max_bs, max_bs * model.block_size * 2, max_bs + 1, device=device
    )
    states = RuntimeStates(
        req_pool_size=max_bs + 1,
        input_vocab_size=60,
        vocab_size=60,
        output_length=model.block_size,
        device=device,
    )
    runner = _LoadedRunner(model, draft_config, args, device, True, False)
    drafter = Inkling2DSpark(
        spec_num_tokens=model.block_size,
        spec_num_steps=model.block_size - 1,
        draft_model_runner=runner,
        attn_backend=backend,
        token_to_kv_pool=pool,
        runtime_states=states,
        input_buffers=buffers,
        vocab_size=60,
    )
    drafter.wire_target(target)
    return drafter, target, draft_spec, attention, target_backend


def _context(bs: int, extends: int, tokens: int) -> ForwardContext:
    return ForwardContext(
        attn_backend=None,
        token_to_kv_pool=None,
        bs=bs,
        num_extends=extends,
        input_num_tokens=tokens,
        forward_mode=(
            ForwardMode.DECODE
            if not extends
            else ForwardMode.EXTEND if extends == bs else ForwardMode.MIXED
        ),
        output_layout=ForwardOutputLayout(extends, extends, bs - extends, 4),
    )


@pytest.mark.parametrize(
    "block_size,context_length,allowed",
    [(4, 52, True), (4, 53, False), (8, 40, True), (8, 41, False), (8, 65, False)],
)
def test_native_context_limit_includes_speculative_padding(
    tmp_path, monkeypatch, block_size, context_length, allowed
):
    monkeypatch.setenv("TOKENSPEED_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN", "1")
    if allowed:
        build_native_drafter(
            tmp_path, 2, "cpu", {"block_size": block_size}, context_length
        )
    else:
        with pytest.raises(ValueError, match="speculative padding"):
            build_native_drafter(
                tmp_path, 2, "cpu", {"block_size": block_size}, context_length
            )


def test_native_setup_borrows_normalized_embedding_and_head_before_drafting(tmp_path):
    drafter, target, _, _, _ = build_native_drafter(tmp_path, 3, "cpu", {}, 40)
    assert get_drafter_impl("DSPARK", drafter.model) is Inkling2DSpark
    assert drafter.model.lm_head is target.lm_head
    assert drafter.model.markov_head.fold_weight is not None
    ids = torch.tensor([1, 7, 8, 9, 2, 4, 5, 6])
    actual = drafter._first_layer_embeds(ids).view(2, 4, 32)
    raw = F.embedding(torch.tensor([1, 2]), target.model.embed_tokens.weight)
    expected = raw.float() * torch.rsqrt(
        raw.float().square().mean(-1, keepdim=True) + 1e-6
    )
    expected = (expected * target.model.embed_norm.weight.float()).bfloat16()
    assert torch.equal(actual[:, 0], expected)
    assert torch.equal(actual[:, 1:], drafter.model.mask_embed.expand(2, 3, 32))
    assert not drafter._fused_kv_enabled


# A count of 0 takes the batch's first row, as the packed-row clamp does.
@pytest.mark.parametrize("accepted", [0, 1, 2, 3, 4])
def test_recurrent_feature_uses_last_accepted_row_and_proposals_reset(
    tmp_path, accepted
):
    drafter, _, _, _, _ = build_native_drafter(tmp_path, 3, "cpu", {}, 40)
    torch.manual_seed(17)
    features = torch.randn(8, 96).bfloat16()
    ctx = _context(2, 0, 8)
    drafter._prepare_proposal_features(ctx, features, torch.tensor([accepted, 4]))
    expected_features = drafter.model.compute_f_rnn(features[[max(accepted - 1, 0), 7]])
    assert torch.equal(drafter._proposal_features_buf[:2], expected_features)
    hidden = torch.randn(2, 4, 32).bfloat16()
    blocks = torch.tensor([[1, 0, 0, 0], [2, 0, 0, 0]], dtype=torch.int32)
    output = torch.empty_like(blocks)
    actual = drafter._sample_block(hidden, blocks, output).clone()
    rows = hidden[:, 1:]
    base = drafter.model.compute_base_logits(rows)
    previous = torch.cat((blocks[:, :1], actual[:, 1:-1]), dim=1)
    oracle = _head_oracle(
        drafter.model.markov_head, base, previous, rows, expected_features
    )
    assert torch.equal(actual[:, 1:].long(), oracle.argmax(-1))
    assert torch.equal(actual[:, 0], blocks[:, 0])
    assert 0 <= actual.min() and actual.max() < 60
    assert torch.equal(drafter._sample_block(hidden, blocks, output), actual)
    # Reorder requests and overwrite the batch scratch, rather than a slot table.
    drafter._prepare_proposal_features(
        ctx, features.flip(0), torch.tensor([4, accepted])
    )
    changed = drafter.model.compute_f_rnn(features.flip(0)[[3, 4 + accepted - 1]])
    assert torch.equal(drafter._proposal_features_buf[:2], changed)


def test_mixed_prefill_and_decode_features_follow_packed_target_rows(tmp_path):
    drafter, _, _, _, _ = build_native_drafter(tmp_path, 3, "cpu", {}, 40)
    features = torch.arange(7 * 96).view(7, 96).bfloat16() / 128
    drafter.input_buffers.input_lengths_buf[:2] = torch.tensor([3, 1])
    ctx = _context(2, 1, 7)
    drafter._prepare_proposal_features(ctx, features, torch.tensor([1, 2]))
    expected = drafter.model.compute_f_rnn(features[[2, 4]])
    assert torch.equal(drafter._proposal_features_buf[:2], expected)


def test_native_forward_uses_common_empty_output_and_rejects_unknown_inputs(tmp_path):
    drafter, _, _, _, _ = build_native_drafter(tmp_path, 3, "cpu", {}, 40)
    ids = torch.empty(0, dtype=torch.long)
    ctx = _context(0, 0, 0)
    output = drafter.draft_model_runner.forward(
        ctx,
        ids,
        ids,
        captured_hidden_states=None,
        input_embeds=None,
        multimodal_context=None,
        spec_step_idx=None,
        pp_inbound=None,
    )
    assert output.hidden_states.shape == (0, 32)
    assert output.hidden_states.dtype == torch.bfloat16
    assert output.next_token_logits is None
    with pytest.raises(TypeError, match="Unsupported native draft arguments"):
        drafter.model(ctx, ids, ids, unsupported=True)


def _cpu_rope(
    positions, k, head_size, table, *, is_neox, k_rope_out, solution, override
):
    # Substitute only the unavailable GPU rotation. Norm/projections remain real.
    assert is_neox and k_rope_out is None and solution is None and override is None
    rows = k.view(k.shape[0], -1, head_size).float()
    cosine, sine = table[positions].float().chunk(2, -1)
    left, right = rows.chunk(2, -1)
    cosine, sine = cosine[:, None], sine[:, None]
    return (
        torch.cat((left * cosine - right * sine, right * cosine + left * sine), -1)
        .to(k.dtype)
        .reshape_as(k)
    )


def _cpu_context_kernels(monkeypatch):
    from tokenspeed.runtime.layers.attention.kv_cache import mha
    from tokenspeed.runtime.models import inkling2_mtp

    def store(k, v, k_dst, v_dst, loc):
        k_dst.index_copy_(0, loc.long(), k)
        v_dst.index_copy_(0, loc.long(), v)

    monkeypatch.setattr(inkling2_mtp, "apply_k_rope", _cpu_rope)
    monkeypatch.setattr(mha, "store_kv_cache", store)


def test_context_write_preserves_target_fields_and_rebinds_without_shared_fc(
    tmp_path, monkeypatch
):
    _cpu_context_kernels(monkeypatch)
    drafter, _, spec, attention, _ = build_native_drafter(tmp_path, 3, "cpu", {}, 40)
    pool = drafter.token_to_kv_pool
    pool.clear_kv_buffers()
    positions, slots = torch.tensor([0, 7, 8]), torch.tensor([127, 128, 131])
    features = torch.randn(3, 96).bfloat16()
    target_before = {
        field.field_id: pool.arena.field(field.field_id).clone()
        for field in spec.memory_plan.fields
        if int(field.field_id.split(".")[1]) < 2
    }
    expected = []
    for i, layer in enumerate(drafter.model.layers):
        attn = layer.self_attn
        k, v = drafter.model.project_context_kv(features, i)
        raw = k.reshape(-1, attn.head_dim).float()
        normalized = (
            (
                raw
                * torch.rsqrt(raw.square().mean(-1, keepdim=True) + 1e-6)
                * attn.k_norm_ctx.weight.float()
            )
            .bfloat16()
            .view_as(k)
        )
        rotated = _cpu_rope(
            positions,
            normalized,
            attn.head_dim,
            attn.rotary_emb.cos_sin_cache,
            is_neox=True,
            k_rope_out=None,
            solution=None,
            override=None,
        )
        expected.append((rotated.view(3, 2, 8), v.view(3, 2, 8)))
    drafter._write_native_cache(features, positions, slots, decode_only=True)
    for i, (k, v) in enumerate(expected):
        assert torch.equal(pool.get_key_buffer(i)[slots], k)
        assert torch.equal(pool.get_value_buffer(i)[slots], v)
    assert all(
        torch.equal(pool.arena.field(name), value)
        for name, value in target_before.items()
    )
    original = pool.get_key_buffer(0).clone()
    arena = create_cache_arena(spec, device="cpu", enable_memory_saver=False)
    replacement = create_cache_pool(
        spec, attention, arena, num_layers=5, rank=0, field_layer_offset=2
    )
    drafter.attn_backend.set_cache_pool(replacement)
    drafter.set_cache_pool(replacement)
    drafter._write_native_cache(features.flip(0), positions, slots, decode_only=True)
    assert torch.equal(pool.get_key_buffer(0), original)
    assert not torch.equal(
        replacement.get_value_buffer(0)[slots], pool.get_value_buffer(0)[slots]
    )
    assert not drafter._fused_kv_enabled


def test_shared_target_update_selects_features_and_publishes_only_accepted_prefix(
    tmp_path, monkeypatch
):
    _cpu_context_kernels(monkeypatch)
    drafter, _, _, _, target_backend = build_native_drafter(tmp_path, 3, "cpu", {}, 40)
    router = target_backend.inner
    slots = torch.tensor([1, 2], dtype=torch.int64)
    lengths = torch.tensor([9, 15], dtype=torch.int32)
    # The runner refreshes both routers before the target forward.
    for backend in (router, drafter.attn_backend):
        backend.refresh_decode_metadata(
            2,
            2,
            slots,
            lengths,
            forward_mode=ForwardMode.DECODE,
            block_tables={
                name: torch.tensor([[1], [2]], dtype=torch.int32)
                for name in backend.group_ids
            },
            num_extends=0,
        )
    positions = torch.tensor([5, 6, 7, 8, 11, 12, 13, 14])
    drafter.input_buffers.positions_buf[:8].copy_(positions)
    drafter.input_buffers.req_pool_indices_buf[:2].copy_(slots)
    drafter.runtime_states.valid_cache_lengths[slots] = torch.tensor(
        [5, 11], dtype=torch.int32
    )
    ctx = _context(2, 0, 8)
    ctx.attn_backend = router
    hidden = torch.randn(8, 96).bfloat16()
    drafter._update_native_cache_from_target(
        ctx,
        LogitsProcessorOutput(next_token_logits=None, hidden_states=hidden),
        torch.tensor([2, 4], dtype=torch.int32),
    )
    assert drafter.draft_seq_lens_buf[:2].tolist() == [7, 15]
    assert torch.equal(
        drafter._proposal_features_buf[:2], drafter.model.compute_f_rnn(hidden[[1, 7]])
    )
    assert router.decode_window_locations().numel() == 8
    # The side-stream write names rows through the target's window and the
    # fallback write through the draft's: both must be the same rows.
    assert torch.equal(
        drafter.attn_backend.decode_window_locations(),
        router.decode_window_locations(),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an admitted GPU")
def test_verify_context_write_rides_the_side_stream_from_the_last_tap(tmp_path):
    drafter, _, _, _, target_backend = build_native_drafter(tmp_path, 3, "cuda", {}, 40)
    router = target_backend.inner
    tables = {
        name: torch.tensor([[1], [2]], dtype=torch.int32, device="cuda")
        for name in router.group_ids
    }
    router.refresh_decode_metadata(
        2,
        2,
        torch.tensor([1, 2], dtype=torch.int64, device="cuda"),
        torch.tensor([9, 15], dtype=torch.int32, device="cuda"),
        forward_mode=ForwardMode.DECODE,
        block_tables=tables,
        num_extends=0,
    )
    positions = drafter.input_buffers.positions_buf[:8]
    positions.copy_(torch.tensor([5, 6, 7, 8, 11, 12, 13, 14]))
    slots = router.decode_window_locations()[:8]
    torch.manual_seed(5)
    taps = [
        torch.randn(8, 32, device="cuda").bfloat16()
        for _ in range(drafter.model.num_context_features)
    ]
    pool = drafter.token_to_kv_pool
    pool.clear_kv_buffers()
    drafter._write_native_cache(torch.cat(taps, -1), positions, slots, True)
    expected = [
        (pool.get_key_buffer(i).clone(), pool.get_value_buffer(i).clone())
        for i in range(len(drafter.model.layers))
    ]
    pool.clear_kv_buffers()

    # A prefill round keeps the inline write.
    extend = _context(2, 2, 8)
    extend.attn_backend = router
    drafter.prepare_target_forward(extend)
    assert extend.target_capture_sink is None
    # So does a decode round wider than the side stream's tap buffer.
    wide = _context(2, 0, drafter._context_taps_buf.shape[0] + 1)
    wide.attn_backend = router
    drafter.prepare_target_forward(wide)
    assert wide.target_capture_sink is None
    ctx = _context(2, 0, 8)
    ctx.attn_backend = router
    drafter.prepare_target_forward(ctx)
    assert ctx.target_capture_sink is drafter
    for index, tap in enumerate(taps):
        ctx.target_capture_sink.on_target_capture(index, tap)
    # The round's own write is the one already in flight.
    drafter._write_native_cache(torch.cat(taps[::-1], -1), positions, slots, True)
    torch.cuda.synchronize()
    for i, (key, value) in enumerate(expected):
        assert torch.equal(pool.get_key_buffer(i), key)
        assert torch.equal(pool.get_value_buffer(i), value)


def test_padded_shard_argmax_ids_map_to_vocabulary_ids():
    # Four shards of 6 vocabulary rows padded to 8 columns.
    rank = torch.tensor([0, 0, 1, 2, 3, 3])
    column = torch.tensor([0, 5, 0, 3, 2, 5])
    ids = padded_shard_ids_to_vocab(rank * 8 + column, 8, 6)
    assert ids.dtype == torch.int64
    assert torch.equal(ids, rank * 6 + column)
    assert torch.equal(vocab_ids_to_padded_shards(ids, 8, 6), rank * 8 + column)

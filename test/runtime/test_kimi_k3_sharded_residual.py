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

"""CPU tests for the Kimi-K3 residual stream that stays row-sharded across layers."""

import functools
import os
import sys
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from tokenspeed.runtime.models import kimi_k3  # noqa: E402

_ROWS = 16
_HIDDEN = 4


def _peer_rows(rank: int) -> torch.Tensor:
    return torch.full((_ROWS // 8, _HIDDEN), float(rank + 1))


def _fake_all_gather(calls):
    """Gather as if every peer q held rows filled with q + 1."""

    def all_gather(tensor, group, dim=0, backend=None):
        del backend
        assert dim == 0
        calls.append((tensor, group))
        return torch.cat([_peer_rows(q) for q in range(8)])

    return all_gather


def _spec(block_residual, write_idx=-1):
    weight = torch.ones(_HIDDEN)
    return kimi_k3._AttnResMixSpec(
        block_residual=block_residual,
        res_weight=weight,
        rms_weight=weight,
        eps=1e-6,
        out_norm_weight=weight,
        out_norm_eps=1e-6,
        num_valid_blocks=2,
        block_write_idx=write_idx,
        next_weight=None,
        gather_next_projection=False,
    )


def test_stream_records_owned_rows_and_gathers_them_back():
    rank = 3
    blocks = torch.zeros((4, _ROWS, _HIDDEN))
    stream = kimi_k3._ShardedResidualStream("group", rank, 8, _ROWS)
    assert (stream.local_rows, stream.first_row) == (2, 6)
    own = _peer_rows(rank)
    activation = torch.randn(_ROWS, _HIDDEN)

    qkv_a = torch.randn(_ROWS, 3)
    stream.next_mix = _spec(blocks, write_idx=2)
    assert stream.finish((own, activation, qkv_a), blocks) is own
    assert stream.sharded and stream.activation is activation
    assert stream.projection is qkv_a
    assert stream.sharded_blocks == [2]
    blocks[2, 6:8] = own

    calls = []
    with mock.patch.object(kimi_k3, "all_gather", _fake_all_gather(calls)):
        full = stream.unshard(own, blocks)
    expected = torch.cat([_peer_rows(q) for q in range(8)])
    torch.testing.assert_close(full, expected)
    torch.testing.assert_close(blocks[2], expected)
    assert torch.count_nonzero(blocks[(0, 1, 3),]) == 0
    assert [group for _, group in calls] == ["group", "group"]
    torch.testing.assert_close(calls[1][0], own)
    assert not stream.sharded and stream.activation is None
    assert stream.projection is None
    assert stream.sharded_blocks == []

    # A full-row output leaves nothing to gather but the owned snapshots.
    stream.next_mix = _spec(blocks, write_idx=1)
    stream.finish((own, activation, None), blocks)
    calls.clear()
    full_output = torch.randn(_ROWS, _HIDDEN)
    with mock.patch.object(kimi_k3, "all_gather", _fake_all_gather(calls)):
        assert stream.finish(full_output, blocks) is full_output
    assert len(calls) == 1 and not stream.sharded and stream.activation is None
    torch.testing.assert_close(blocks[1], expected)

    # Unsharded residuals are returned as they are.
    with mock.patch.object(kimi_k3, "all_gather", side_effect=AssertionError):
        assert stream.unshard(full_output, blocks) is full_output


def _moe_layer(comm, moe, *, writes_block):
    weight = torch.ones(_HIDDEN)
    norm = SimpleNamespace(weight=weight, variance_epsilon=1e-6)
    return SimpleNamespace(
        is_block_write_layer=writes_block,
        block_write_idx=2,
        prev_valid_blocks=2,
        self_attention_res_proj=SimpleNamespace(weight=weight),
        self_attention_res_norm=norm,
        input_layernorm=norm,
        self_attn=mock.Mock(return_value=torch.zeros(_ROWS, _HIDDEN)),
        mapping=SimpleNamespace(attn=SimpleNamespace(dp_size=1)),
        mlp_res_proj=SimpleNamespace(weight=weight),
        mlp_res_norm=norm,
        post_attention_layernorm=norm,
        is_moe_layer=True,
        block_sparse_moe=moe,
        k3_comm=comm,
        comm_manager=SimpleNamespace(get_num_tokens=lambda _: (_ROWS, _ROWS)),
        _prepare_next_fallback_attnres_partial=mock.Mock(),
    )


@pytest.mark.parametrize("gathering", (False, True))
@pytest.mark.parametrize("writes_block", (False, True))
def test_premixed_layer_skips_its_input_mix_and_passes_the_next_one(
    writes_block, gathering
):
    blocks = torch.zeros((4, _ROWS, _HIDDEN))
    activation = torch.randn(_ROWS, _HIDDEN)
    residual = _peer_rows(5)
    shard, mixed = _peer_rows(5) * 2, torch.randn(_ROWS, _HIDDEN)
    next_residual, next_activation = _peer_rows(5) * 3, torch.randn(_ROWS, _HIDDEN)
    moe = mock.Mock(spec=kimi_k3.KimiLinearMoE)
    next_qkv = torch.randn(_ROWS, 3)
    moe.return_value = (next_residual, next_activation, next_qkv)
    moe.native_latent_moe = None
    pending = object()
    comm = SimpleNamespace(
        acquire_projection_output=mock.Mock(return_value=torch.zeros(_ROWS, _HIDDEN)),
        take_published=mock.Mock(return_value=None),
        mix_for_moe=mock.Mock(
            return_value=(shard, mixed, pending) if gathering else (shard, mixed)
        ),
        reduce_for_attnres=mock.Mock(side_effect=AssertionError("reduced twice")),
    )
    published = object()
    comm.take_published.return_value = published
    layer = _moe_layer(comm, moe, writes_block=writes_block)
    stream = kimi_k3._ShardedResidualStream("group", 5, 8, _ROWS)
    stream.sharded, stream.activation = True, activation
    precomputed = torch.randn(_ROWS, 3)
    stream.projection = precomputed
    stream.next_mix = _spec(blocks, write_idx=3)

    with mock.patch.object(
        kimi_k3, "_apply_attn_res", side_effect=AssertionError("mixed twice")
    ):
        prefix, history = kimi_k3.KimiLinearDecoderLayer._forward_fused_attnres_graph(
            layer, torch.empty(0), residual, SimpleNamespace(), blocks, stream
        )

    assert layer.self_attn.call_args.kwargs["hidden_states"] is activation
    assert layer.self_attn.call_args.kwargs["precomputed_qkv_a"] is precomputed
    assert comm.acquire_projection_output.call_args.args[0] is activation
    assert comm.mix_for_moe.call_args.args[1] is (None if writes_block else residual)
    assert comm.mix_for_moe.call_args.kwargs["residual_is_sharded"] is True
    # The attention partial's publication record reaches its scatter.
    comm.take_published.assert_called_once_with(comm.mix_for_moe.call_args.args[0])
    assert comm.mix_for_moe.call_args.kwargs["published"] is published
    # Too few rows for the MoE's row-sharded inputs: the mix gathers in place.
    assert comm.mix_for_moe.call_args.kwargs["defer_gather"] is False
    assert moe.call_args.kwargs["attention_gather"] is (pending if gathering else None)
    assert moe.call_args.args[:2] == (mixed, shard)
    assert moe.call_args.kwargs["prefix_is_sharded"] is True
    assert moe.call_args.kwargs["next_mix"] is stream.next_mix
    assert prefix is next_residual and history is blocks
    assert stream.sharded and stream.activation is next_activation
    assert stream.projection is next_qkv
    assert stream.sharded_blocks == [3]


def test_declined_sharded_mix_gathers_before_the_ordinary_reduce():
    blocks = torch.zeros((4, _ROWS, _HIDDEN))
    residual = _peer_rows(2)
    reduced, delta = torch.randn(_ROWS, _HIDDEN), torch.randn(_ROWS, _HIDDEN)
    output = torch.randn(_ROWS, _HIDDEN)
    moe = mock.Mock(spec=kimi_k3.KimiLinearMoE, return_value=output)
    moe.native_latent_moe = None
    comm = SimpleNamespace(
        acquire_projection_output=mock.Mock(return_value=torch.zeros(_ROWS, _HIDDEN)),
        take_published=mock.Mock(return_value=None),
        mix_for_moe=mock.Mock(return_value=None),
        reduce_for_attnres=mock.Mock(return_value=(reduced, delta)),
    )
    layer = _moe_layer(comm, moe, writes_block=False)
    stream = kimi_k3._ShardedResidualStream("group", 2, 8, _ROWS)
    stream.sharded, stream.activation = True, torch.randn(_ROWS, _HIDDEN)
    stream.sharded_blocks = [1]
    stream.next_mix = _spec(blocks)
    calls = []
    mlp_input = torch.randn(_ROWS, _HIDDEN)

    with (
        mock.patch.object(kimi_k3, "all_gather", _fake_all_gather(calls)),
        mock.patch.object(kimi_k3, "_apply_attn_res", return_value=mlp_input) as mix,
    ):
        prefix, _ = kimi_k3.KimiLinearDecoderLayer._forward_fused_attnres_graph(
            layer, torch.empty(0), residual, SimpleNamespace(), blocks, stream
        )

    everyone = torch.cat([_peer_rows(q) for q in range(8)])
    torch.testing.assert_close(comm.reduce_for_attnres.call_args.args[1], everyone)
    torch.testing.assert_close(blocks[1], everyone)
    assert mix.call_count == 1 and mix.call_args.args[0] is reduced
    assert moe.call_args.args[:2] == (mlp_input, reduced)
    assert moe.call_args.kwargs["prefix_is_sharded"] is False
    assert prefix is output
    assert not stream.sharded and stream.activation is None
    assert stream.sharded_blocks == []


class _FakeLayer:
    def __init__(self, index, rank):
        self.index = index
        self.rank = rank
        self.seen = []

    def pre_attention_mix(self, block_residual, rows):
        assert rows == _ROWS
        return SimpleNamespace(name=("pre", self.index), block_write_idx=-1)

    def __call__(self, positions, prefix_sum, ctx, block_residual, residual_stream):
        self.seen.append((prefix_sum, residual_stream.next_mix))
        own = _peer_rows(self.rank) * (self.index + 1)
        return (
            residual_stream.finish(
                (own, torch.full((_ROWS, _HIDDEN), float(self.index)), None),
                block_residual,
            ),
            block_residual,
        )


def test_backbone_captures_gathered_rows_and_returns_the_premixed_output():
    rank = 6
    layers = [_FakeLayer(index, rank) for index in range(3)]
    stream = kimi_k3._ShardedResidualStream("group", rank, 8, _ROWS)
    model = SimpleNamespace(
        embed_tokens=lambda input_ids: torch.zeros((_ROWS, _HIDDEN)),
        config=SimpleNamespace(num_hidden_layers=3, attn_res_block_size=2),
        pp_start_layer=0,
        pp_end_layer=3,
        mapping=SimpleNamespace(is_last_pp_rank=True),
        layers=layers,
        layers_to_capture=[],
        dflash_aux_stream="prefix",
        eagle3_layers_to_capture=(2,),
        decode_final=None,
        decode_chain=None,
        _output_attn_res_mix=lambda block_residual: SimpleNamespace(
            name=("output",), block_write_idx=-1
        ),
    )
    calls = []
    with (
        mock.patch.object(
            kimi_k3.KimiLinearModel, "_sharded_residual_stream", return_value=stream
        ),
        mock.patch.object(kimi_k3, "all_gather", _fake_all_gather(calls)),
        mock.patch.object(
            kimi_k3, "_apply_attn_res", side_effect=AssertionError("mixed twice")
        ),
    ):
        hidden, aux = kimi_k3.KimiLinearModel.forward(
            model,
            torch.zeros(_ROWS, dtype=torch.long),
            positions=None,
            ctx=SimpleNamespace(dspark_context_producer=None),
        )

    assert [mix.name for _, mix in (layer.seen[0] for layer in layers)] == [
        ("pre", 1),
        ("pre", 2),
        ("output",),
    ]
    # Layer 1 receives layer 0's owned rows; the capture after it gathers them.
    torch.testing.assert_close(layers[1].seen[0][0], _peer_rows(rank))
    assert len(aux) == 1 and len(calls) == 1
    torch.testing.assert_close(calls[0][0], _peer_rows(rank) * 2)
    torch.testing.assert_close(aux[0], torch.cat([_peer_rows(q) for q in range(8)]))
    torch.testing.assert_close(hidden, torch.full((_ROWS, _HIDDEN), 2.0))
    assert hidden.data_ptr() != stream.activation.data_ptr()


def test_stream_is_limited_to_tp8_prefill_rows():
    group = ("tp",)
    attn = SimpleNamespace(dp_size=1, tp_size=8, tp_rank=4, tp_group=group)
    moe = SimpleNamespace(tp_size=8, ep_size=1, tp_ep_group=group)
    model = SimpleNamespace(
        mapping=SimpleNamespace(pp_size=1, attn=attn, moe=moe), norm=object()
    )

    def stream(rows, capturing=False, **changes):
        hidden = torch.empty((rows, _HIDDEN), device="meta")
        mapping = SimpleNamespace(
            pp_size=changes.get("pp_size", 1),
            attn=SimpleNamespace(**{**vars(attn), **changes.get("attn", {})}),
            moe=SimpleNamespace(**{**vars(moe), **changes.get("moe", {})}),
        )
        with (
            mock.patch.object(
                kimi_k3, "current_platform", return_value=SimpleNamespace(is_cdna4=True)
            ),
            mock.patch.object(kimi_k3, "get_is_capture_mode", return_value=False),
        ):
            return kimi_k3.KimiLinearModel._sharded_residual_stream(
                SimpleNamespace(mapping=mapping, norm=model.norm), hidden, capturing
            )

    accepted = stream(8192)
    assert (accepted.group, accepted.first_row, accepted.local_rows) == (
        group,
        4096,
        1024,
    )
    assert stream(1024).first_row == 512
    for rejected in (
        stream(1016),
        stream(8200),
        stream(2052),
        stream(2048, capturing=True),
        stream(2048, pp_size=2),
        stream(2048, attn={"dp_size": 2}),
        stream(2048, attn={"tp_size": 4}),
        stream(2048, moe={"ep_size": 8}),
        stream(2048, moe={"tp_ep_group": ("other",)}),
    ):
        assert rejected is None


def test_mla_offers_its_bf16_qkv_a_rows_for_precomputation():
    from tokenspeed.runtime.models.kimi_k31 import KimiK31SparseMLAAttention

    weight = torch.empty((2112 + 1536, 7168), dtype=torch.bfloat16, device="meta")

    def offered(**changes):
        attention = SimpleNamespace(
            qkv_parallel=None,
            fused_qkv_a_proj_with_mqa=SimpleNamespace(weight=weight),
            _fused_qkv_a_pad_rows=0,
            _qkv_a_width=2112,
        )
        vars(attention).update(changes)
        return KimiK31SparseMLAAttention.precomputable_qkv_a_weight(attention)

    rows = offered()
    assert rows.shape == (2112, 7168) and rows.data_ptr() == weight.data_ptr()
    assert offered(qkv_parallel=object()) is None
    assert offered(_fused_qkv_a_pad_rows=64) is None
    assert offered(_qkv_a_width=2048) is None
    assert (
        offered(
            fused_qkv_a_proj_with_mqa=SimpleNamespace(
                weight=weight.to(torch.float8_e4m3fn)
            )
        )
        is None
    )


def test_mla_forward_takes_precomputed_qkv_a_and_projects_only_the_gate():
    from tokenspeed.runtime.models import kimi_k31

    tokens, hidden_size = 4, 16
    hidden = torch.randn(tokens, hidden_size)
    weight = torch.randn(14 + 6, hidden_size)
    precomputed = torch.randn(tokens, 14)
    gate = torch.randn(tokens, 6)
    attn_output, output = torch.randn(tokens, 6), torch.randn(tokens, hidden_size)
    attention = SimpleNamespace(
        q_lora_rank=8,
        kv_lora_rank=4,
        qk_rope_head_dim=2,
        _qkv_a_width=14,
        fused_qkv_a_proj_with_mqa=SimpleNamespace(weight=weight),
        fused_qk_layernorm=mock.Mock(),
        absorbed_weights_k_major=True,
        o_proj=SimpleNamespace(supports_fp8_group128_input=False),
        q_b_wq_b_weight=None,
        _takes_decode_front=mock.Mock(return_value=False),
        _sparse_attention=mock.Mock(return_value=attn_output),
        _gate_project_output=mock.Mock(return_value=output),
        _project_qkv_gate_a=mock.Mock(side_effect=AssertionError("projected twice")),
    )
    attention._attend = functools.partial(
        kimi_k31.KimiK31SparseMLAAttention._attend, attention
    )
    comm = SimpleNamespace(pre_attn_comm=lambda value, ctx: value)
    ctx = SimpleNamespace(num_extends=1)
    with mock.patch.object(
        kimi_k31, "kimi3_mla_gate_projection", return_value=gate
    ) as project_gate:
        result = kimi_k31.KimiK31SparseMLAAttention.forward(
            attention,
            torch.empty(0),
            hidden,
            ctx,
            comm,
            projection_out=None,
            precomputed_qkv_a=precomputed,
        )
    assert result is output
    assert project_gate.call_args.args == (hidden, weight, 14)
    norm = attention.fused_qk_layernorm.call_args.kwargs
    torch.testing.assert_close(norm["input_q_a"], precomputed[:, :8])
    torch.testing.assert_close(norm["input_kv_a"], precomputed[:, 8:12])
    assert norm["input_kv_a"].data_ptr() == precomputed[:, 8:].data_ptr()
    latent = attention._sparse_attention.call_args.args[3]
    assert latent.data_ptr() == precomputed[:, 8:].data_ptr() and latent.shape == (4, 6)
    assert attention._gate_project_output.call_args.args[1] is gate
    with pytest.raises(ValueError, match="plain BF16"):
        kimi_k31.KimiK31SparseMLAAttention.forward(
            attention,
            torch.empty(0),
            hidden,
            ctx,
            comm,
            block_scale=torch.ones(1),
            projection_out=None,
            precomputed_qkv_a=precomputed,
        )


def test_kda_layer_takes_the_precomputed_projection():
    blocks = torch.zeros((4, _ROWS, _HIDDEN))
    activation = torch.randn(_ROWS, _HIDDEN)
    shard, mixed = _peer_rows(1) * 2, torch.randn(_ROWS, _HIDDEN)
    moe = mock.Mock(
        spec=kimi_k3.KimiLinearMoE, return_value=torch.randn(_ROWS, _HIDDEN)
    )
    moe.native_latent_moe = None
    comm = SimpleNamespace(
        acquire_projection_output=mock.Mock(return_value=torch.zeros(_ROWS, _HIDDEN)),
        take_published=mock.Mock(return_value=None),
        mix_for_moe=mock.Mock(return_value=(shard, mixed)),
        reduce_for_attnres=mock.Mock(side_effect=AssertionError("reduced twice")),
    )
    layer = _moe_layer(comm, moe, writes_block=False)
    layer.self_attn = mock.Mock(spec=kimi_k3.KimiLinearKDA)
    layer.self_attn.o_proj = None
    layer.self_attn.return_value = torch.zeros(_ROWS, _HIDDEN)
    stream = kimi_k3._ShardedResidualStream("group", 1, 8, _ROWS)
    projection = torch.randn(_ROWS, 5)
    stream.sharded, stream.activation, stream.projection = True, activation, projection
    stream.next_mix = _spec(blocks)
    with mock.patch.object(
        kimi_k3, "_apply_attn_res", side_effect=AssertionError("mixed twice")
    ):
        kimi_k3.KimiLinearDecoderLayer._forward_fused_attnres_graph(
            layer, torch.empty(0), _peer_rows(1), SimpleNamespace(), blocks, stream
        )
    kwargs = layer.self_attn.call_args.kwargs
    assert kwargs["precomputed_projection"] is projection
    assert "precomputed_qkv_a" not in kwargs


def test_kda_projection_slices_a_precomputed_projection():
    proj_local, head_dim, heads = 8, 4, 2
    used = 4 * proj_local + head_dim + heads
    projection = torch.randn(3, used + 2)
    attention = SimpleNamespace(
        local_num_heads=heads,
        head_dim=head_dim,
        qkvgb_proj=SimpleNamespace(
            used_rows=used,
            fp8_channel_quant=False,
            fp8_block_quant=False,
            side_effect=AssertionError("projected twice"),
        ),
        qkv_parallel=None,
    )
    attention.local_num_heads = proj_local // head_dim
    attention._split_qkvfab = functools.partial(
        kimi_k3.KimiLinearKDA._split_qkvfab, attention
    )
    mixed_qkv, gate, f_a, beta = kimi_k3.KimiLinearKDA._project_qkvfab(
        attention, torch.randn(3, 6), None, None, projection
    )
    assert mixed_qkv.data_ptr() == projection.data_ptr()
    torch.testing.assert_close(mixed_qkv, projection[:, : 3 * proj_local])
    torch.testing.assert_close(gate, projection[:, 3 * proj_local : 4 * proj_local])
    torch.testing.assert_close(f_a, projection[:, 4 * proj_local : 4 * proj_local + 4])
    torch.testing.assert_close(beta, projection[:, 4 * proj_local + 4 : used])
    with pytest.raises(ValueError, match="AttnRes partials"):
        kimi_k3.KimiLinearKDA._project_qkvfab(
            attention, torch.randn(3, 6), None, (None,) * 6, projection
        )


def test_kda_offers_only_its_bf16_merged_projection():
    weight = torch.empty((6288, 7168), dtype=torch.bfloat16, device="meta")

    def offered(qkv_parallel=None, **merged_changes):
        merged = kimi_k3.KimiKDAMergedProj.__new__(kimi_k3.KimiKDAMergedProj)
        torch.nn.Module.__init__(merged)
        merged.fp8_block_quant = False
        merged.fp8_channel_quant = False
        merged.quant_method = None
        merged.weight = torch.nn.Parameter(weight, requires_grad=False)
        for name, value in merged_changes.items():
            setattr(merged, name, value)
        attention = SimpleNamespace(qkv_parallel=qkv_parallel, qkvgb_proj=merged)
        return kimi_k3.KimiLinearKDA.precomputable_projection_weight(attention)

    assert offered() is not None and offered().shape == (6288, 7168)
    assert offered(qkv_parallel=object()) is None
    assert offered(fp8_block_quant=True) is None
    assert offered(fp8_channel_quant=True) is None
    assert offered(quant_method=object()) is None

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

from contextlib import nullcontext
from types import SimpleNamespace
from unittest import mock

import pytest

from tokenspeed.runtime.layers.moe.utils import All2AllBackend
from tokenspeed.runtime.models.gpt_oss import GptOssDecoderLayer
from tokenspeed.runtime.utils.server_args import ServerArgs


def _mapping() -> SimpleNamespace:
    return SimpleNamespace(
        nnodes=1,
        world_size=8,
        moe=SimpleNamespace(ep_size=8, tp_size=1),
        attn=SimpleNamespace(tp_size=1, cp_size=1, dp_size=8),
        dense=SimpleNamespace(tp_size=1),
    )


def _validation_args(
    *,
    moe_backend: str,
    draft_moe_backend: str | None,
    all2all_backend: str,
    speculative_algorithm: str | None,
    max_num_seqs: int,
    dtype: str,
    chunked_prefill_size: int,
) -> SimpleNamespace:
    return SimpleNamespace(
        device="cuda",
        moe_backend=moe_backend,
        draft_moe_backend=draft_moe_backend,
        all2all_backend=all2all_backend,
        mapping=_mapping(),
        enable_eplb=False,
        ep_num_redundant_experts=0,
        init_expert_location=None,
        speculative_algorithm=speculative_algorithm,
        speculative_num_draft_tokens=0,
        max_num_seqs=max_num_seqs,
        dtype=dtype,
        chunked_prefill_size=chunked_prefill_size,
        max_prefill_tokens=1024,
    )


def test_petit_moe_runs_on_dp_idle_ranks() -> None:
    with mock.patch(
        "tokenspeed.runtime.models.gpt_oss.get_all2all_backend",
        return_value=All2AllBackend.PETIT,
    ):
        spec = GptOssDecoderLayer.mlp_spec(SimpleNamespace())

    assert spec.runs_on_empty_input is True


@pytest.mark.parametrize("backend", [All2AllBackend.PETIT, All2AllBackend.NONE])
@pytest.mark.parametrize("num_experts", [256, 384])
def test_deepseek_petit_shared_expert_stream_and_placement(
    backend: All2AllBackend, num_experts: int
) -> None:
    from tokenspeed.runtime.models import deepseek_v3 as model

    config = SimpleNamespace(
        n_shared_experts=1,
        routed_scaling_factor=2.5,
        n_routed_experts=num_experts,
        hidden_act="silu",
        hidden_size=7168,
        moe_intermediate_size=2048,
        num_experts_per_tok=8,
        norm_topk_prob=True,
        n_group=8,
        topk_group=4,
    )
    mapping = _mapping()
    mapping.moe.tp_rank = 0
    mapping.moe.ep_rank = 0
    aux_stream = object()
    with (
        mock.patch.object(model, "get_all2all_backend", return_value=backend),
        mock.patch.object(model, "StreamFork") as stream_fork,
        mock.patch.object(model, "MoEGate"),
        mock.patch.object(model, "DeepseekV3MLP") as shared,
        mock.patch.object(model, "MoELayer"),
        mock.patch.object(model, "TopK"),
        mock.patch.dict(model.global_server_args_dict, ep_num_redundant_experts=0),
    ):
        model.DeepseekV3MoE(
            config=config,
            mapping=mapping,
            quant_config=None,
            layer_index=0,
            prefix="model.layers.0.mlp",
            alt_stream=aux_stream,
        )
    stream_fork.assert_called_once_with(
        None if backend is All2AllBackend.PETIT else aux_stream
    )
    assert shared.call_args.kwargs["is_shared_expert"] is (
        backend is not All2AllBackend.PETIT
    )


def test_petit_requires_both_backend_flags() -> None:
    args = _validation_args(
        moe_backend="petit",
        draft_moe_backend=None,
        all2all_backend="none",
        speculative_algorithm=None,
        max_num_seqs=160,
        dtype="bfloat16",
        chunked_prefill_size=1024,
    )

    with pytest.raises(ValueError, match="requires --all2all-backend petit"):
        ServerArgs.validate(args)


def test_petit_rejects_non_petit_draft_backend() -> None:
    args = _validation_args(
        moe_backend="petit",
        draft_moe_backend="triton",
        all2all_backend="petit",
        speculative_algorithm="MTP",
        max_num_seqs=160,
        dtype="bfloat16",
        chunked_prefill_size=1024,
    )

    with pytest.raises(ValueError, match="incompatible draft=triton"):
        ServerArgs.validate(args)


def test_petit_rejects_draft_only_selection() -> None:
    args = _validation_args(
        moe_backend="triton",
        draft_moe_backend="petit",
        all2all_backend="none",
        speculative_algorithm="MTP",
        max_num_seqs=160,
        dtype="bfloat16",
        chunked_prefill_size=1024,
    )

    with pytest.raises(
        ValueError,
        match="requires --all2all-backend petit for the active draft",
    ):
        ServerArgs.validate(args)


def test_petit_draft_inherits_target_backend() -> None:
    args = _validation_args(
        moe_backend="petit",
        draft_moe_backend=None,
        all2all_backend="petit",
        speculative_algorithm="MTP",
        max_num_seqs=160,
        dtype="bfloat16",
        chunked_prefill_size=1024,
    )
    platform = SimpleNamespace(is_cdna4=False)

    with (
        mock.patch(
            "tokenspeed.runtime.utils.server_args.current_platform",
            return_value=platform,
        ),
        pytest.raises(ValueError, match="requires AMD CDNA4"),
    ):
        ServerArgs.validate(args)


def test_petit_rejects_decode_capacity_above_workspace_limit() -> None:
    args = _validation_args(
        moe_backend="petit",
        draft_moe_backend=None,
        all2all_backend="petit",
        speculative_algorithm=None,
        max_num_seqs=8200,
        dtype="bfloat16",
        chunked_prefill_size=1024,
    )
    platform = SimpleNamespace(is_cdna4=True)

    with (
        mock.patch(
            "tokenspeed.runtime.utils.server_args.current_platform",
            return_value=platform,
        ),
        pytest.raises(ValueError, match="1024 decode tokens per rank"),
    ):
        ServerArgs.validate(args)


@pytest.mark.parametrize("dtype", ["auto", "half", "float16", "float", "float32"])
def test_petit_rejects_non_bfloat16_dtype(dtype: str) -> None:
    args = _validation_args(
        moe_backend="petit",
        draft_moe_backend=None,
        all2all_backend="petit",
        speculative_algorithm=None,
        max_num_seqs=160,
        dtype=dtype,
        chunked_prefill_size=1024,
    )

    with pytest.raises(ValueError, match="requires --dtype bfloat16"):
        ServerArgs.validate(args)


@pytest.mark.parametrize("chunked_prefill_size", [-1, 0])
def test_petit_rejects_disabled_chunked_prefill(
    chunked_prefill_size: int,
) -> None:
    args = _validation_args(
        moe_backend="petit",
        draft_moe_backend=None,
        all2all_backend="petit",
        speculative_algorithm=None,
        max_num_seqs=160,
        dtype="bfloat16",
        chunked_prefill_size=chunked_prefill_size,
    )
    platform = SimpleNamespace(is_cdna4=True)

    with (
        mock.patch(
            "tokenspeed.runtime.utils.server_args.current_platform",
            return_value=platform,
        ),
        pytest.raises(ValueError, match="positive value no greater than 1024"),
    ):
        ServerArgs.validate(args)


def test_v4_petit_keeps_hash_input_ids_local() -> None:
    import torch

    from tokenspeed.runtime.models.deepseek_v4 import DeepseekV4DecoderLayer

    input_ids = torch.tensor([11, 22], dtype=torch.int64)
    with mock.patch(
        "tokenspeed.runtime.models.deepseek_v4.get_all2all_backend",
        return_value=All2AllBackend.PETIT,
    ):
        result = DeepseekV4DecoderLayer._pre_mlp_input_ids_comm(
            SimpleNamespace(), input_ids, None
        )
    assert result is input_ids


@pytest.mark.parametrize("rows", [0, 3])
def test_v4_petit_participates_on_empty_ranks_and_scales_once(rows: int) -> None:
    import torch

    from tokenspeed.runtime.models.deepseek_v4 import DeepseekV4MoE

    states = torch.ones((rows, 8))
    experts = mock.Mock(return_value=torch.full_like(states, 2.0))
    output_format = SimpleNamespace(is_bypassed=lambda: False)
    topk = SimpleNamespace(format=output_format)
    fork = SimpleNamespace(branch=lambda: nullcontext())
    moe = SimpleNamespace(
        use_petit=True,
        stream_fork=SimpleNamespace(scope=lambda **kwargs: nullcontext(fork)),
        _select_experts=mock.Mock(return_value=(None, None, None)),
        _make_topk_output=mock.Mock(return_value=topk),
        experts=experts,
        routed_scaling_factor=2.5,
        _forward_shared_experts=mock.Mock(return_value=torch.full_like(states, 7.0)),
    )
    result = DeepseekV4MoE.forward_normal(
        moe, states, torch.zeros(rows, dtype=torch.int64), 5, 3
    )
    experts.assert_called_once_with(
        hidden_states=states,
        topk_output=topk,
        num_global_tokens=5,
        max_num_tokens_per_gpu=3,
    )
    torch.testing.assert_close(result, torch.full_like(states, 12.0))


def test_v4_empty_routing_has_valid_shapes() -> None:
    import torch

    from tokenspeed.runtime.models.deepseek_v4 import DeepseekV4MoE

    gate = mock.Mock(side_effect=AssertionError("empty routing must not launch gate"))
    moe = SimpleNamespace(
        config=SimpleNamespace(num_experts_per_tok=6, n_routed_experts=384), gate=gate
    )
    weights, ids, scores = DeepseekV4MoE._select_experts(
        moe, torch.empty((0, 7168)), torch.empty(0, dtype=torch.int64)
    )
    assert weights.shape == ids.shape == (0, 6)
    assert weights.dtype == torch.float32
    assert ids.dtype == torch.int32
    assert scores.shape == (0, 384)


@pytest.mark.parametrize("backend_name", ["PETIT", "AUTO"])
def test_v4_shared_expert_stream_placement_and_mxfp4_loader(backend_name: str) -> None:
    from tokenspeed.runtime.layers.moe.utils import MoeBackend
    from tokenspeed.runtime.layers.quantization import Fp8Config, Mxfp4Config
    from tokenspeed.runtime.models import deepseek_v4 as model

    config = SimpleNamespace(
        n_shared_experts=1,
        n_routed_experts=384,
        num_experts_per_tok=6,
        hidden_size=7168,
        moe_intermediate_size=3072,
        hidden_act="silu",
        norm_topk_prob=True,
        swiglu_limit=10.0,
        expert_dtype="fp4",
    )
    mapping = SimpleNamespace(
        attn=SimpleNamespace(tp_size=1),
        moe=SimpleNamespace(ep_size=8, tp_size=1, tp_ep_size=8, tp_rank=0, ep_rank=0),
    )
    quant = Fp8Config(is_checkpoint_fp8_serialized=True, weight_block_size=[128, 128])
    backend = MoeBackend[backend_name]
    aux_stream = object()
    with (
        mock.patch.object(model, "get_moe_backend", return_value=backend),
        mock.patch.object(model, "StreamFork") as stream_fork,
        mock.patch.dict(model.global_server_args_dict, {"ep_num_redundant_experts": 0}),
        mock.patch.object(model, "DeepseekV4MoEGate"),
        mock.patch.object(model, "DeepseekV4MLP") as shared,
        mock.patch.object(model, "MoELayer") as experts,
        mock.patch.object(model, "TopK"),
    ):
        model.DeepseekV4MoE(config, mapping, quant, 0, "model.layers.0.ffn", aux_stream)
    stream_fork.assert_called_once_with(
        None if backend is MoeBackend.PETIT else aux_stream
    )
    assert shared.call_args.kwargs["is_shared_expert"] is (
        backend is not MoeBackend.PETIT
    )
    assert shared.call_args.kwargs["swiglu_limit"] == 10.0
    args = experts.call_args.kwargs
    assert isinstance(args["quant_config"], Mxfp4Config)
    assert args["quant_config"].is_checkpoint_mxfp4_serialized
    assert args["swiglu_limit"] == 10.0
    assert args["routing_mode"] == "precomputed_topk"
    assert args["routing_config"]["routed_scaling_factor"] == 1.0

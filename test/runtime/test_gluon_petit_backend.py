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

import argparse
from contextlib import nullcontext
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from tokenspeed.runtime.layers.moe import expert as expert_module
from tokenspeed.runtime.layers.moe.expert import MoELayer
from tokenspeed.runtime.layers.moe.utils import All2AllBackend, MoeBackend
from tokenspeed.runtime.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from tokenspeed.runtime.layers.quantization.mxfp4 import Mxfp4Config
from tokenspeed.runtime.models.base.decoder_layer import CompiledMoEDecoderLayer
from tokenspeed.runtime.models.base.module_spec import ModuleKind
from tokenspeed.runtime.models.base.placement import ParallelGroup, Replicate
from tokenspeed.runtime.utils.server_args import ServerArgs


def _compressed_mxfp4_config(quant_format: str) -> CompressedTensorsConfig:
    return CompressedTensorsConfig.from_config(
        {
            "format": quant_format,
            "config_groups": {
                "group_0": {
                    "targets": ["Linear"],
                    "weights": {
                        "num_bits": 4,
                        "type": "float",
                        "strategy": "group",
                        "group_size": 32,
                        "symmetric": True,
                        "dynamic": False,
                    },
                    "input_activations": None,
                }
            },
        }
    )


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
        disable_prefill_graph=False,
        disable_pdl=False,
        moe_backend=moe_backend,
        draft_moe_backend=draft_moe_backend,
        all2all_backend=all2all_backend,
        mapping=_mapping(),
        enable_eplb=False,
        ep_num_redundant_experts=0,
        init_expert_location=None,
        speculative_algorithm=speculative_algorithm,
        speculative_num_draft_tokens=1,
        max_num_seqs=max_num_seqs,
        dtype=dtype,
        chunked_prefill_size=chunked_prefill_size,
        max_prefill_tokens=1024,
    )


def test_gluon_petit_backend_enums() -> None:
    assert All2AllBackend("gluon_petit").is_gluon_petit()
    assert MoeBackend("gluon_petit").is_gluon_petit()
    assert not MoeBackend("gluon_petit").is_mega_moe()


def test_gluon_petit_cli_backend_names() -> None:
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args(
        [
            "openai/gpt-oss-120b",
            "--moe-backend",
            "gluon_petit",
            "--draft-moe-backend",
            "gluon_petit",
            "--all2all-backend",
            "gluon_petit",
        ]
    )
    assert MoeBackend(args.moe_backend) is MoeBackend.GLUON_PETIT
    assert MoeBackend(args.draft_moe_backend) is MoeBackend.GLUON_PETIT
    assert All2AllBackend(args.all2all_backend) is All2AllBackend.GLUON_PETIT


def test_gluon_petit_compiled_moe_owns_ep_communication() -> None:
    layer = CompiledMoEDecoderLayer.__new__(CompiledMoEDecoderLayer)

    with mock.patch(
        "tokenspeed.runtime.models.base.decoder_layer.get_all2all_backend",
        return_value=All2AllBackend.GLUON_PETIT,
    ):
        spec = layer.mlp_spec()

    assert spec.kind == ModuleKind.MOE
    assert spec.input_placement == Replicate(ParallelGroup.ATTN_TP)
    assert spec.output_placement is None


def test_gluon_petit_requires_matching_backend_pair() -> None:
    args = _validation_args(
        moe_backend="gluon_petit",
        draft_moe_backend=None,
        all2all_backend="none",
        speculative_algorithm=None,
        max_num_seqs=160,
        dtype="bfloat16",
        chunked_prefill_size=1024,
    )

    with pytest.raises(ValueError, match="--all2all-backend gluon_petit"):
        ServerArgs.validate_petit_moe_options(args)


def test_gluon_petit_rejects_mixed_draft_backend() -> None:
    args = _validation_args(
        moe_backend="gluon_petit",
        draft_moe_backend="triton",
        all2all_backend="gluon_petit",
        speculative_algorithm="MTP",
        max_num_seqs=160,
        dtype="bfloat16",
        chunked_prefill_size=1024,
    )

    with pytest.raises(ValueError, match="incompatible draft=triton"):
        ServerArgs.validate_petit_moe_options(args)


def test_gluon_petit_rejects_non_bfloat16_dtype() -> None:
    args = _validation_args(
        moe_backend="gluon_petit",
        draft_moe_backend=None,
        all2all_backend="gluon_petit",
        speculative_algorithm=None,
        max_num_seqs=160,
        dtype="float16",
        chunked_prefill_size=1024,
    )

    with pytest.raises(ValueError, match="requires --dtype bfloat16"):
        ServerArgs.validate_petit_moe_options(args)


def test_gluon_petit_rejects_decode_capacity_above_workspace_limit() -> None:
    args = _validation_args(
        moe_backend="gluon_petit",
        draft_moe_backend=None,
        all2all_backend="gluon_petit",
        speculative_algorithm=None,
        max_num_seqs=8200,
        dtype="bfloat16",
        chunked_prefill_size=1024,
    )

    with pytest.raises(ValueError, match="1024 decode tokens per rank"):
        ServerArgs.validate_petit_moe_options(args)


def test_dsv4_gluon_petit_zero_token_routing_shapes() -> None:
    from tokenspeed.runtime.models.deepseek_v4 import DeepseekV4MoE

    layer = DeepseekV4MoE.__new__(DeepseekV4MoE)
    layer.config = SimpleNamespace(num_experts_per_tok=6, n_routed_experts=384)
    hidden_states = torch.empty((0, 7168), dtype=torch.bfloat16)

    layer.gate = SimpleNamespace(tid2eid=None, e_score_correction_bias=None)
    scores, correction, hashes, input_ids = layer._routing_inputs(
        hidden_states, input_ids=None
    )

    assert scores.shape == (0, 384)
    assert scores.dtype == torch.float32
    assert correction is None
    assert hashes is None
    assert input_ids is None


@pytest.fixture
def petit_args() -> SimpleNamespace:
    return _validation_args(
        moe_backend="gluon_petit",
        draft_moe_backend=None,
        all2all_backend="gluon_petit",
        speculative_algorithm=None,
        max_num_seqs=8192,
        dtype="bfloat16",
        chunked_prefill_size=1024,
    )


def test_validate_calls_petit_validation() -> None:
    # Construct real ServerArgs without model/device initialization, then check
    # that the public validation entry point reaches the backend-pair check.
    with mock.patch.object(ServerArgs, "__post_init__", return_value=None):
        args = ServerArgs(model="test")
    args.moe_backend = "gluon_petit"
    args.all2all_backend = "none"
    with pytest.raises(ValueError, match="--all2all-backend gluon_petit"):
        args.validate()


@pytest.mark.parametrize(
    "overrides,error",
    [
        ({}, None),
        ({"draft_moe_backend": "triton"}, None),  # Inactive draft is ignored.
        ({"speculative_algorithm": "MTP"}, None),  # Draft inherits target.
        ({"speculative_algorithm": "MTP", "draft_moe_backend": "gluon_petit"}, None),
        ({"moe_backend": "triton"}, "incompatible target=triton"),
        (
            {"moe_backend": "triton", "all2all_backend": "none", "dtype": "float16"},
            None,
        ),
        ({"max_num_seqs": 8200}, "1024 decode tokens per rank"),
        (
            {
                "speculative_algorithm": "MTP",
                "speculative_num_draft_tokens": 2,
                "max_num_seqs": 4096,
            },
            None,
        ),
        (
            {
                "speculative_algorithm": "MTP",
                "speculative_num_draft_tokens": 2,
                "max_num_seqs": 4104,
            },
            "1024 decode tokens per rank",
        ),
        ({"chunked_prefill_size": 0}, "1024 prefill tokens per rank"),
        ({"chunked_prefill_size": 1025}, "1024 prefill tokens per rank"),
        ({"max_prefill_tokens": 1025}, "1024 prefill tokens per rank"),
    ],
)
def test_petit_shared_options(petit_args, overrides, error) -> None:
    vars(petit_args).update(overrides)
    with pytest.raises(ValueError, match=error) if error else nullcontext():
        ServerArgs.validate_petit_moe_options(petit_args)


@pytest.mark.parametrize("attn_tp,attn_cp,dense_tp", [(2, 1, 1), (1, 2, 1), (1, 1, 2)])
def test_petit_shared_parallelism(petit_args, attn_tp, attn_cp, dense_tp) -> None:
    petit_args.mapping.attn.tp_size = attn_tp
    petit_args.mapping.attn.cp_size = attn_cp
    petit_args.mapping.dense.tp_size = dense_tp
    with pytest.raises(ValueError, match="attention TP1, CP1, and dense TP1"):
        ServerArgs.validate_petit_moe_options(petit_args)


@pytest.mark.parametrize(
    "mapping_overrides,moe_overrides,options,layer_overrides,is_cdna4,error",
    [
        ({}, {}, {}, {}, True, None),
        (
            {},
            {},
            {},
            {
                "top_k": 16,
                "num_experts": 896,
                "hidden_size": 3584,
                "intermediate_size": 3072,
                "activation": "situ",
                "activation_situ_beta": 4.0,
                "activation_situ_linear_beta": 25.0,
                "routing_mode": "precomputed_topk",
                "quant_config": _compressed_mxfp4_config("mxfp4-pack-quantized"),
            },
            True,
            None,
        ),
        (
            {},
            {},
            {},
            {"quant_config": _compressed_mxfp4_config("pack-quantized")},
            True,
            "serialized MXFP4 expert weights",
        ),
        (
            {},
            {},
            {},
            {
                "quant_config": Mxfp4Config(
                    ignored_layers=[], is_checkpoint_mxfp4_serialized=False
                )
            },
            True,
            "serialized MXFP4 expert weights",
        ),
        ({}, {}, {"init_expert_location": "trivial"}, {}, True, None),
        ({}, {}, {}, {}, False, "requires AMD CDNA4"),
        (
            {},
            {},
            {"moe_mxfp4_fp8_activation": True},
            {},
            True,
            "requested fp8 activations are unsupported",
        ),
        ({"nnodes": 2}, {}, {}, {}, True, "supports one node only"),
        ({"world_size": 4}, {}, {}, {}, True, "world_size=ep_size=8"),
        ({}, {"ep_size": 4}, {}, {}, True, "world_size=ep_size=8"),
        ({}, {"tp_size": 2}, {}, {}, True, "MoE tensor parallel size 1"),
        ({}, {}, {}, {"ep_size": 4}, True, "world_size=ep_size=8"),
        ({}, {}, {}, {"ep_size": 1, "tp_size": 2}, True, "MoE tensor parallel size 1"),
        ({}, {}, {"enable_eplb": True}, {}, True, "trivial expert placement"),
        ({}, {}, {"ep_num_redundant_experts": 8}, {}, True, "trivial expert placement"),
        (
            {},
            {},
            {"init_expert_location": "custom"},
            {},
            True,
            "trivial expert placement",
        ),
        ({}, {}, {}, {"quant_config": None}, True, "serialized MXFP4 expert weights"),
        ({}, {}, {}, {"activation_alpha": 1.5}, True, "nonstandard SiLU alpha"),
    ],
)
def test_petit_layer_constraints(
    petit_args,
    mapping_overrides,
    moe_overrides,
    options,
    layer_overrides,
    is_cdna4,
    error,
) -> None:
    vars(petit_args.mapping).update(mapping_overrides)
    vars(petit_args.mapping.moe).update(moe_overrides)
    vars(petit_args).update(options)
    # These restrictions belong to the layer, so server validation accepts them.
    with mock.patch(
        "tokenspeed.runtime.utils.server_args.current_platform",
        return_value=SimpleNamespace(is_cdna4=is_cdna4),
    ):
        ServerArgs.validate_petit_moe_options(petit_args)
    layer_args = dict(
        top_k=4,
        num_experts=128,
        hidden_size=2880,
        intermediate_size=3072,
        quant_config=Mxfp4Config(
            ignored_layers=[], is_checkpoint_mxfp4_serialized=True
        ),
        layer_index=0,
        tp_rank=0,
        tp_size=1,
        ep_rank=0,
        ep_size=8,
    )
    layer_args.update(layer_overrides)
    with (
        mock.patch.dict(
            expert_module.global_server_args_dict,
            mapping=petit_args.mapping,
            enable_eplb=petit_args.enable_eplb,
            ep_num_redundant_experts=petit_args.ep_num_redundant_experts,
            init_expert_location=petit_args.init_expert_location,
            moe_mxfp4_fp8_activation=options.get("moe_mxfp4_fp8_activation", False),
        ),
        mock.patch.object(
            expert_module,
            "current_platform",
            return_value=SimpleNamespace(is_cdna4=is_cdna4),
        ),
        mock.patch.object(
            expert_module,
            "get_all2all_backend",
            return_value=All2AllBackend.GLUON_PETIT,
        ),
        mock.patch.object(
            expert_module, "get_moe_backend", return_value=MoeBackend.GLUON_PETIT
        ),
        mock.patch.object(
            expert_module.tokenspeed_kernel,
            "moe_plan",
            return_value={"solution": "gluon"},
        ) as plan,
        mock.patch.object(expert_module, "create_layer_weights") as weights,
    ):
        with pytest.raises(ValueError, match=error) if error else nullcontext():
            MoELayer(**layer_args)
        if error:
            plan.assert_not_called()
            weights.assert_not_called()
        else:
            plan.assert_called_once()
            assert plan.call_args.kwargs["ep_size"] == 8
            assert plan.call_args.kwargs["solution"] == "gluon"
            assert plan.call_args.kwargs["a2a_backend"] == "gluon_petit"
            assert plan.call_args.kwargs["activation"] == layer_args.get(
                "activation", "silu"
            )
            weights.assert_called_once()

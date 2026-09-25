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

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from tokenspeed.runtime.layers.moe.utils import All2AllBackend, MoeBackend
from tokenspeed.runtime.models.base.decoder_layer import CompiledMoEDecoderLayer
from tokenspeed.runtime.models.base.module_spec import ModuleKind
from tokenspeed.runtime.models.base.placement import ParallelGroup, Replicate
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


def test_petit_gluon_backend_enums() -> None:
    assert All2AllBackend("petit_gluon").is_petit_gluon()
    assert MoeBackend("petit_gluon").is_petit_gluon()
    assert not MoeBackend("petit_gluon").is_mega_moe()


def test_petit_gluon_compiled_moe_owns_ep_communication() -> None:
    layer = CompiledMoEDecoderLayer.__new__(CompiledMoEDecoderLayer)

    with mock.patch(
        "tokenspeed.runtime.models.base.decoder_layer.get_all2all_backend",
        return_value=All2AllBackend.PETIT_GLUON,
    ):
        spec = layer.mlp_spec()

    assert spec.kind == ModuleKind.MOE
    assert spec.input_placement == Replicate(ParallelGroup.ATTN_TP)
    assert spec.output_placement is None


def test_petit_gluon_requires_matching_backend_pair() -> None:
    args = _validation_args(
        moe_backend="petit_gluon",
        draft_moe_backend=None,
        all2all_backend="none",
        speculative_algorithm=None,
        max_num_seqs=160,
        dtype="bfloat16",
        chunked_prefill_size=1024,
    )

    with pytest.raises(ValueError, match="--all2all-backend petit_gluon"):
        ServerArgs.validate(args)


def test_petit_gluon_rejects_mixed_draft_backend() -> None:
    args = _validation_args(
        moe_backend="petit_gluon",
        draft_moe_backend="triton",
        all2all_backend="petit_gluon",
        speculative_algorithm="MTP",
        max_num_seqs=160,
        dtype="bfloat16",
        chunked_prefill_size=1024,
    )

    with pytest.raises(ValueError, match="incompatible draft=triton"):
        ServerArgs.validate(args)


def test_petit_gluon_rejects_non_bfloat16_dtype() -> None:
    args = _validation_args(
        moe_backend="petit_gluon",
        draft_moe_backend=None,
        all2all_backend="petit_gluon",
        speculative_algorithm=None,
        max_num_seqs=160,
        dtype="float16",
        chunked_prefill_size=1024,
    )

    with pytest.raises(ValueError, match="requires --dtype bfloat16"):
        ServerArgs.validate(args)


def test_petit_gluon_rejects_decode_capacity_above_workspace_limit() -> None:
    args = _validation_args(
        moe_backend="petit_gluon",
        draft_moe_backend=None,
        all2all_backend="petit_gluon",
        speculative_algorithm=None,
        max_num_seqs=8200,
        dtype="bfloat16",
        chunked_prefill_size=1024,
    )

    with (
        mock.patch(
            "tokenspeed.runtime.utils.server_args.current_platform",
            return_value=SimpleNamespace(is_cdna4=True),
        ),
        pytest.raises(ValueError, match="1024 decode tokens per rank"),
    ):
        ServerArgs.validate(args)


def test_dsv4_petit_gluon_zero_token_routing_shapes() -> None:
    from tokenspeed.runtime.models.deepseek_v4 import DeepseekV4MoE

    layer = DeepseekV4MoE.__new__(DeepseekV4MoE)
    layer.config = SimpleNamespace(num_experts_per_tok=6, n_routed_experts=384)
    hidden_states = torch.empty((0, 7168), dtype=torch.bfloat16)

    weights, ids, scores = layer._select_experts(hidden_states, input_ids=None)

    assert weights.shape == (0, 6)
    assert weights.dtype == torch.float32
    assert ids.shape == (0, 6)
    assert ids.dtype == torch.int32
    assert scores.shape == (0, 384)

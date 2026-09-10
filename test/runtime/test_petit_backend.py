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


@pytest.mark.parametrize("dtype", ["half", "float16", "float", "float32"])
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

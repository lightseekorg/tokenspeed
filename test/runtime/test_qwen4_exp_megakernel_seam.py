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

"""The Qwen4-Exp decode megakernel seam: which forwards take the kernels, and
how startup reports the layers that keep the regular path. CPU only."""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from tokenspeed.runtime.execution.context import ForwardContext  # noqa: E402
from tokenspeed.runtime.execution.drafter.eagle import (  # noqa: E402
    AcceptedPrefixPublisher,
)
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode  # noqa: E402
from tokenspeed.runtime.execution.output_layout import (  # noqa: E402
    ForwardOutputLayout,
)
from tokenspeed.runtime.models.qwen4_exp_megakernel import (  # noqa: E402
    ATTENTION_TP,
    EXPERT_PARALLEL,
    NOT_MXFP4,
    coverage,
    forward_chain,
    unserved_report,
)
from tokenspeed.runtime.utils.server_args import (  # noqa: E402
    DECODE_MEGAKERNEL_ARCHITECTURES,
    decode_megakernel_refusal,
)


class _DraftBackend:
    """The draft model's attention backend: QSA only, no recurrent state."""

    @property
    def linear_attn_backend(self):
        raise RuntimeError("the draft backend has no linear attention")

    def advance_draft_forward_metadata(self, seq_lens: torch.Tensor) -> None:
        del seq_lens


def _decode_ctx(bs: int, rows: int, narrowing) -> ForwardContext:
    return ForwardContext(
        attn_backend=narrowing.attn_backend if narrowing else _DraftBackend(),
        token_to_kv_pool=None,
        bs=bs,
        num_extends=0,
        input_num_tokens=rows,
        forward_mode=ForwardMode.DECODE,
        output_layout=ForwardOutputLayout(0, 0, bs, 1),
        draft_narrowing=narrowing,
    )


def test_draft_steps_take_the_kernels() -> None:
    """Every draft step of up to 4 rows takes the kernels: the first step of
    one request (its verified rows, a chain) and the one-row steps after it.
    The draft backend has no recurrent state, and the rule never asks for it.
    A first step of several requests keeps the regular path: a launch
    narrows to the live row of one request only."""
    backend = _DraftBackend()
    first = AcceptedPrefixPublisher(backend, torch.tensor([7]))
    assert forward_chain(4, _decode_ctx(1, 4, first)) == 4
    assert forward_chain(1, _decode_ctx(1, 1, None)) == 1
    assert forward_chain(4, _decode_ctx(4, 4, None)) == 1
    two = AcceptedPrefixPublisher(backend, torch.tensor([7, 9]))
    assert forward_chain(4, _decode_ctx(2, 4, two)) is None
    assert forward_chain(8, _decode_ctx(2, 8, two)) is None
    assert forward_chain(5, _decode_ctx(5, 5, None)) is None


def test_the_report_groups_layers_by_reason() -> None:
    unserved = {3: [ATTENTION_TP], 0: [ATTENTION_TP, EXPERT_PARALLEL], 7: [NOT_MXFP4]}
    assert unserved_report(unserved) == (
        f"{ATTENTION_TP}: layers 0, 3; {EXPERT_PARALLEL}: layer 0; {NOT_MXFP4}: layer 7"
    )


def test_a_target_without_any_megakernel_layer_fails_startup() -> None:
    """The switch never serves silently without kernels: a target that has no
    megakernel layer fails with every reason."""
    unserved = {0: [ATTENTION_TP], 1: [ATTENTION_TP]}
    with pytest.raises(
        ValueError, match=f"no layer of Target.*{ATTENTION_TP}: layers 0, 1"
    ):
        coverage("Target", 2, unserved, required=True)


def test_a_draft_without_any_megakernel_layer_keeps_the_regular_path() -> None:
    warning = coverage("Draft", 1, {0: [NOT_MXFP4]}, required=False)
    assert warning == (
        "Decode megakernel: 0 of 1 Draft layers; the rest keep the regular path "
        f"({NOT_MXFP4}: layer 0)"
    )


@pytest.mark.parametrize("required", [True, False])
def test_partial_coverage_warns_with_the_layers_on_the_regular_path(required) -> None:
    warning = coverage("Target", 4, {2: [NOT_MXFP4]}, required=required)
    assert warning == (
        "Decode megakernel: 3 of 4 Target layers; the rest keep the regular path "
        f"({NOT_MXFP4}: layer 2)"
    )
    assert coverage("Target", 4, {}, required=required) is None


def test_the_switch_refuses_a_model_without_megakernels() -> None:
    for architecture in DECODE_MEGAKERNEL_ARCHITECTURES:
        assert decode_megakernel_refusal([architecture]) is None
    refusal = decode_megakernel_refusal(["Qwen3ForCausalLM"])
    assert refusal is not None and "Qwen3ForCausalLM" in refusal

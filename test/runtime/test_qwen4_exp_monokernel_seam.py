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

"""The Qwen4-Exp decode monokernel seam: which forwards take the kernels, and
how startup reports the layers that keep the regular path. CPU only."""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from tokenspeed.runtime.configs.qwen4_exp_config import Qwen4ExpTextConfig  # noqa: E402
from tokenspeed.runtime.execution.context import ForwardContext  # noqa: E402
from tokenspeed.runtime.execution.drafter.eagle import (  # noqa: E402
    AcceptedPrefixPublisher,
)
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode  # noqa: E402
from tokenspeed.runtime.execution.output_layout import (  # noqa: E402
    ForwardOutputLayout,
)
from tokenspeed.runtime.layers.attention.backends.specific.qwen4_exp import (  # noqa: E402
    Qwen4ExpBackend,
)
from tokenspeed.runtime.layers.attention.configs.base import AttnConfig  # noqa: E402
from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig  # noqa: E402
from tokenspeed.runtime.models.qwen4_exp import (  # noqa: E402
    Qwen4ExpAttentionDecoderLayer,
    Qwen4ExpLinearDecoderLayer,
    Qwen4ExpModel,
    Qwen4ExpMonokernelAttentionDecoderLayer,
    Qwen4ExpMonokernelLinearDecoderLayer,
)
from tokenspeed.runtime.models.qwen4_exp_monokernel import (  # noqa: E402
    ATTENTION_TP,
    EXPERT_PARALLEL,
    NOT_MXFP4,
    DraftVerifyLengths,
    QSAMonokernelLayer,
    forward_chain,
    unserved_error,
    unserved_report,
)
from tokenspeed.runtime.models.qwen4_exp_nextn import (  # noqa: E402
    Qwen4ExpDraftAttentionDecoderLayer,
    Qwen4ExpDraftModel,
    Qwen4ExpMonokernelDraftAttentionDecoderLayer,
)
from tokenspeed.runtime.utils.env import global_server_args_dict  # noqa: E402
from tokenspeed.runtime.utils.server_args import (  # noqa: E402
    DECODE_MONOKERNEL_ARCHITECTURES,
    decode_monokernel_refusal,
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


def test_a_target_verify_without_gated_deltanet_keeps_the_regular_path() -> None:
    """A Qwen4-Exp view without Gated DeltaNet layers has no ReplaySSM
    payload for the kernels to fill: its verify keeps the regular path
    instead of failing on the missing recurrent backend."""
    qsa = MHAConfig(
        backend_name="qsa",
        num_attention_heads=1,
        num_kv_heads=1,
        head_dim=2,
        attn_tp_size=1,
        cache_layer_types=("full_attention",),
        sliding_window_tokens=None,
    )
    config = AttnConfig(
        device="cuda",
        dtype=torch.bfloat16,
        kv_cache_dtype=torch.bfloat16,
        kv_cache_quant_method="none",
        kv_cache_mxfp8=False,
        kernel_page_size=64,
        context_len=1024,
        max_bs=2,
        pd_disaggregation_enabled=False,
        speculative_num_steps=3,
        speculative_num_draft_tokens=4,
        is_draft=False,
        draft_block_decode=False,
        components=(qsa,),
    )
    # The full-attention child only, with no hybrid wrapper.
    backend = Qwen4ExpBackend(config, _DraftBackend(), None, None)
    assert not backend.has_linear_attn

    def verify_ctx(rows: int) -> ForwardContext:
        return ForwardContext(
            attn_backend=backend,
            token_to_kv_pool=None,
            bs=1,
            num_extends=0,
            input_num_tokens=rows,
            forward_mode=ForwardMode.DECODE,
            output_layout=ForwardOutputLayout(0, 0, 1, 1),
            draft_narrowing=None,
        )

    assert forward_chain(4, verify_ctx(4)) is None
    # One row per request still takes the kernels.
    assert forward_chain(1, verify_ctx(1)) == 1


def test_every_fused_draft_layer_reads_the_lengths_from_before_the_publish() -> None:
    """The draft's first step publishes the accepted lengths into the buffer
    that holds the verified rows' lengths. A second fused layer of the same
    forward must still see the lengths from before the publish."""
    seq_lens = torch.tensor([12, 15])

    class _PublishingBackend(_DraftBackend):
        def advance_draft_forward_metadata(self, lengths: torch.Tensor) -> None:
            seq_lens.copy_(lengths)

    narrowing = AcceptedPrefixPublisher(_PublishingBackend(), torch.tensor([10, 11]))
    lengths = DraftVerifyLengths()
    first = lengths.take(seq_lens, narrowing)
    assert seq_lens.tolist() == [10, 11]
    second = lengths.take(seq_lens, narrowing)
    assert first.tolist() == second.tolist() == [12, 15]


def test_the_report_groups_layers_by_reason() -> None:
    unserved = {3: [ATTENTION_TP], 0: [ATTENTION_TP, EXPERT_PARALLEL], 7: [NOT_MXFP4]}
    assert unserved_report(unserved) == (
        f"{ATTENTION_TP}: layers 0, 3; {EXPERT_PARALLEL}: layer 0; {NOT_MXFP4}: layer 7"
    )


def test_the_startup_error_names_every_layer_without_a_monokernel() -> None:
    """The switch never serves a layer without its kernel: startup fails and
    names each such layer with its reasons."""
    unserved = {0: [ATTENTION_TP], 1: [ATTENTION_TP, NOT_MXFP4]}
    assert unserved_error("Target", unserved) == (
        f"--decode-monokernel: 2 Target layers have no monokernel "
        f"({ATTENTION_TP}: layers 0, 1; {NOT_MXFP4}: layer 1). "
        "Serve without --decode-monokernel."
    )


class _RegularLayer(torch.nn.Module):
    """A regular decoder layer that records the forwards it runs."""

    def __init__(self) -> None:
        super().__init__()
        self.layer_id = 0
        self.ple = None
        self.forwards = 0

    def forward(self, positions, hidden_states, residual, ctx, input_ids):
        self.forwards += 1
        return hidden_states, residual


class _MonokernelLayer(QSAMonokernelLayer, _RegularLayer):
    pass


def test_a_monokernel_layer_runs_other_forwards_with_its_regular_code() -> None:
    """A forward that the kernels do not serve (here 5 rows, past the
    kernels' plans) runs the regular layer's code."""
    layer = _MonokernelLayer()
    rows = torch.zeros(5, 2)
    out = layer(
        positions=None,
        hidden_states=rows,
        residual=None,
        ctx=_decode_ctx(5, 5, None),
        input_ids=None,
    )
    assert out[0] is rows and layer.forwards == 1


def test_a_monokernel_layer_refuses_a_decode_step_before_packing() -> None:
    """A decode step that the kernels serve needs the packed layer: without
    it the layer fails instead of running its regular code."""
    layer = _MonokernelLayer()
    with pytest.raises(RuntimeError, match="before post_quant_warmup"):
        layer(
            positions=None,
            hidden_states=torch.zeros(1, 2),
            residual=None,
            ctx=_decode_ctx(1, 1, None),
            input_ids=None,
        )
    assert layer.forwards == 0


def test_the_switch_refuses_a_model_without_monokernels() -> None:
    for architecture in DECODE_MONOKERNEL_ARCHITECTURES:
        assert decode_monokernel_refusal([architecture]) is None
    refusal = decode_monokernel_refusal(["Qwen3ForCausalLM"])
    assert refusal is not None and "Qwen3ForCausalLM" in refusal


@pytest.mark.parametrize("switch", [False, True])
def test_the_switch_builds_the_monokernel_layer_classes(monkeypatch, switch) -> None:
    """With --decode-monokernel, the target builds each layer from its
    monokernel class; without it, from the regular class."""
    monkeypatch.setitem(global_server_args_dict, "decode_monokernel", switch)
    model = Qwen4ExpModel.__new__(Qwen4ExpModel)
    torch.nn.Module.__init__(model)
    built = {
        layer_type: model.decoder_layer_cls(layer_type)
        for layer_type in ("attention", "linear_attention")
    }
    if switch:
        assert built == {
            "attention": Qwen4ExpMonokernelAttentionDecoderLayer,
            "linear_attention": Qwen4ExpMonokernelLinearDecoderLayer,
        }
    else:
        assert built == {
            "attention": Qwen4ExpAttentionDecoderLayer,
            "linear_attention": Qwen4ExpLinearDecoderLayer,
        }


@pytest.mark.parametrize("shares_index", [False, True])
def test_a_draft_that_shares_the_index_builds_the_regular_layer(
    monkeypatch, shares_index
) -> None:
    """Draft steps that reuse the first step's top-k keep the regular draft
    layer even with the switch: the monokernel selects its own top-k."""
    monkeypatch.setitem(global_server_args_dict, "decode_monokernel", True)
    model = Qwen4ExpDraftModel.__new__(Qwen4ExpDraftModel)
    torch.nn.Module.__init__(model)
    model.config = Qwen4ExpTextConfig(index_share_for_mtp_iteration=shares_index)
    expected = (
        Qwen4ExpDraftAttentionDecoderLayer
        if shares_index
        else Qwen4ExpMonokernelDraftAttentionDecoderLayer
    )
    assert model.decoder_layer_cls("attention") is expected

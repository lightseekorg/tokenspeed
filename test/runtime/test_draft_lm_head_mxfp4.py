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

"""``--draft-lm-head-mxfp4``: an MXFP4 copy of the LM head for an MTP draft."""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.platform import current_platform

if not current_platform().is_amd:
    pytest.skip("Dynamic MXFP4 activations are AMD-only", allow_module_level=True)

from tokenspeed_kernel.ops.gemm.triton import triton_mm_mxfp4  # noqa: E402
from tokenspeed_kernel.ops.quantization.triton import mxfp4_quantize  # noqa: E402
from transformers import PretrainedConfig  # noqa: E402

from tokenspeed.runtime.execution.factory import (  # noqa: E402
    _use_mxfp4_draft_lm_head,
)
from tokenspeed.runtime.layers.dense.mxfp4 import Mxfp4LMHead  # noqa: E402
from tokenspeed.runtime.layers.logits_processor import (  # noqa: E402
    LogitsProcessor,
)
from tokenspeed.runtime.layers.vocab_parallel_embedding import (  # noqa: E402
    ParallelLMHead,
)

_VOCAB = 4096
_HIDDEN = 2560


class _Model(torch.nn.Module):
    """A model that owns only its LM head."""

    def __init__(self, lm_head: torch.nn.Module) -> None:
        super().__init__()
        self.lm_head = lm_head


def _bf16_head() -> ParallelLMHead:
    with torch.device("cuda"):
        head = ParallelLMHead(_VOCAB, _HIDDEN, params_dtype=torch.bfloat16)
    with torch.no_grad():
        head.weight.copy_(torch.randn_like(head.weight) * 0.02)
    return head


def _logits(head: torch.nn.Module, hidden: torch.Tensor) -> torch.Tensor:
    processor = LogitsProcessor(
        config=PretrainedConfig(vocab_size=_VOCAB),
        tp_rank=0,
        tp_size=1,
        tp_group=(0,),
        dp_lm_head_tp=False,
    )
    return processor._get_logits(hidden, head, None, require_full_vocab=True)


@pytest.mark.parametrize("rows", [1, 4])
def test_logits_are_the_mxfp4_product(rows: int) -> None:
    torch.manual_seed(0)
    source = _bf16_head()
    hidden = torch.randn(rows, _HIDDEN, device="cuda", dtype=torch.bfloat16)

    logits = _logits(Mxfp4LMHead(source), hidden)

    weight_fp4, weight_scale = mxfp4_quantize(source.weight)
    hidden_fp4, hidden_scale = mxfp4_quantize(hidden)
    expected = triton_mm_mxfp4(
        hidden_fp4, weight_fp4, hidden_scale, weight_scale, torch.bfloat16
    )
    # The same MXFP4 operands; only the FP32 summation order may differ.
    torch.testing.assert_close(logits.float(), expected.float(), atol=1e-3, rtol=2**-8)


def test_draft_gets_its_own_copy_and_the_target_keeps_bf16() -> None:
    torch.manual_seed(0)
    target = _Model(_bf16_head())
    # The draft's own head module, aliasing the target's weight (as
    # set_embed_and_head binds it).
    draft_head = _bf16_head()
    draft_head.weight = target.lm_head.weight
    draft = _Model(draft_head)
    weight_before = target.lm_head.weight.detach().clone()

    _use_mxfp4_draft_lm_head(target, draft)

    assert isinstance(draft.lm_head, Mxfp4LMHead)
    assert draft.lm_head.weight.dtype == torch.uint8
    assert target.lm_head.weight.dtype == torch.bfloat16
    torch.testing.assert_close(target.lm_head.weight, weight_before, rtol=0, atol=0)
    hidden = torch.randn(1, _HIDDEN, device="cuda", dtype=torch.bfloat16)
    torch.testing.assert_close(
        _logits(target.lm_head, hidden),
        hidden @ weight_before.T,
        rtol=0,
        atol=0,
    )


def test_a_draft_that_adopts_the_target_head_is_refused() -> None:
    target = _Model(_bf16_head())
    draft = _Model(target.lm_head)
    with pytest.raises(ValueError, match="own LM head module"):
        _use_mxfp4_draft_lm_head(target, draft)

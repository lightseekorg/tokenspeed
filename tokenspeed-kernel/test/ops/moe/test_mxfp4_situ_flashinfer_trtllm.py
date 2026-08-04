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

"""flashinfer TRTLLM-Gen SiTU MoE vs the portable K3 reference.

Exercises the full in-repo chain -- the weight preprocessor (concatenated
[gate|up] loader layout -> shuffled TRTLLM [up|gate]) and the registered
apply -- against ``a16w4_mxfp4_moe_reference`` on the same MXFP4 weights.
"""

from __future__ import annotations

from importlib.util import find_spec

import pytest
import torch
from kimi3_reference import a16w4_mxfp4_moe_reference
from utils import make_mxfp4_moe_weights

NUM_EXPERTS = 8
TOP_K = 2
HIDDEN = 256  # multiple of 256: no hidden padding path
ISPP = 128  # multiple of 128: no intermediate padding path
SITU_BETA = 4.0  # K3 activation_situ_beta
SITU_LINEAR_BETA = 25.0  # K3 activation_situ_linear_beta


def _situ_runtime_reason() -> str | None:
    if not torch.cuda.is_available():
        return "requires CUDA"
    if not (10, 0) <= torch.cuda.get_device_capability() <= (10, 3):
        return "flashinfer TRTLLM-Gen SiTU targets the sm_100 family"
    if find_spec("flashinfer") is None:
        return "requires flashinfer"
    from tokenspeed_kernel.ops.moe.mxfp4.flashinfer_trtllm import (
        situ_moe_unavailable_reason,
    )

    return situ_moe_unavailable_reason()


_reason = _situ_runtime_reason()
requires_flashinfer_situ = pytest.mark.skipif(_reason is not None, reason=str(_reason))


class _MoEWeights(torch.nn.Module):
    """Minimal module carrying what the preprocessor and apply consume."""

    def __init__(self, raw: dict[str, torch.Tensor]) -> None:
        super().__init__()
        self.w13_weight = torch.nn.Parameter(raw["w13_weight"], requires_grad=False)
        self.w13_weight_scale = torch.nn.Parameter(
            raw["w13_scale"], requires_grad=False
        )
        self.w2_weight = torch.nn.Parameter(raw["w2_weight"], requires_grad=False)
        self.w2_weight_scale = torch.nn.Parameter(raw["w2_scale"], requires_grad=False)
        self.w13_input_layout = "concatenated"
        self.num_experts = NUM_EXPERTS
        self.num_local_experts = NUM_EXPERTS
        self.top_k = TOP_K
        self.hidden_size = HIDDEN
        self.activation_situ_beta = SITU_BETA
        self.activation_situ_linear_beta = SITU_LINEAR_BETA


@requires_flashinfer_situ
def test_flashinfer_situ_routed_moe_matches_portable_reference() -> None:
    from tokenspeed_kernel.ops.moe.mxfp4.flashinfer_trtllm import (
        flashinfer_trtllm_mxfp4_situ_moe_weights,
        flashinfer_trtllm_mxfp4_situ_routed_moe_apply,
    )

    generator = torch.Generator().manual_seed(20260729)
    num_tokens = 16

    raw = make_mxfp4_moe_weights(NUM_EXPERTS, HIDDEN, ISPP, generator, device="cpu")
    hidden_states = (
        torch.randn(num_tokens, HIDDEN, generator=generator) * 0.2
    ).bfloat16()
    topk_ids = torch.stack(
        [
            torch.randperm(NUM_EXPERTS, generator=generator)[:TOP_K]
            for _ in range(num_tokens)
        ]
    ).to(dtype=torch.int32)
    topk_weights = torch.rand(num_tokens, TOP_K, generator=generator).softmax(dim=-1)

    expected = a16w4_mxfp4_moe_reference(
        hidden_states,
        raw["w13_weight"],
        raw["w13_scale"],
        raw["w2_weight"],
        raw["w2_scale"],
        topk_ids,
        topk_weights.float(),
        situ_beta=SITU_BETA,
        situ_linear_beta=SITU_LINEAR_BETA,
    )

    w = _MoEWeights({k: v.clone() for k, v in raw.items()}).cuda()
    flashinfer_trtllm_mxfp4_situ_moe_weights({}, w)
    actual = flashinfer_trtllm_mxfp4_situ_routed_moe_apply(
        {},
        hidden_states.cuda(),
        w,
        router_logits=None,
        topk_weights=topk_weights.cuda(),
        topk_ids=topk_ids.cuda(),
    )

    # The reference runs bf16 activations while the kernel quantizes them to
    # MXFP8, so this is an activation-quantization envelope, not an ulp bound.
    torch.testing.assert_close(
        actual.cpu().float(), expected.float(), atol=8e-2, rtol=8e-2
    )

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
    from tokenspeed_kernel.ops.moe.flashinfer.trtllm_mxfp4 import (
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
@pytest.mark.parametrize("num_tokens", [1, 16])
def test_k3_prepared_front_matches_flashinfer_quant_and_pack(num_tokens: int) -> None:
    """The fused preparation preserves FlashInfer's byte-level input ABI."""
    from flashinfer import mxfp8_quantize
    from tokenspeed_kernel.thirdparty.cuda.moe import moe_pack_topk_quant_mxfp8

    hidden = 3584
    topk = 16
    generator = torch.Generator().manual_seed(20260818 + num_tokens)
    x = torch.randn(
        num_tokens, hidden, generator=generator, dtype=torch.bfloat16
    ).cuda()
    ids = torch.stack(
        [torch.randperm(896, generator=generator)[:topk] for _ in range(num_tokens)]
    ).to(device="cuda", dtype=torch.int32)
    weights = (
        torch.rand(num_tokens, topk, generator=generator)
        .softmax(dim=-1)
        .to(device="cuda", dtype=torch.bfloat16)
    )

    packed, actual_q, actual_s = moe_pack_topk_quant_mxfp8(x, ids, weights)
    expected_q, expected_s = mxfp8_quantize(
        x, False, alignment=hidden, backend="cute-dsl"
    )
    expected_packed = (ids << 16) | weights.view(torch.uint16).to(torch.int32)

    assert torch.equal(packed, expected_packed)
    assert torch.equal(
        actual_q.view(torch.uint8).flatten(), expected_q.view(torch.uint8).flatten()
    )
    assert torch.equal(actual_s, expected_s.view(torch.uint8))


@requires_flashinfer_situ
@pytest.mark.parametrize(
    ("num_tokens", "enable_pdl"),
    [(1, False), (2, True), (4, True), (8, True), (12, True), (16, True)],
)
def test_k3_route_pack_quant_accepts_strided_front_views(
    num_tokens: int,
    enable_pdl: bool,
) -> None:
    from tokenspeed_kernel.ops.moe import kimi3_route_pack_quant_mxfp8

    generator = torch.Generator().manual_seed(20260819 + num_tokens)
    front = torch.randn(
        num_tokens, 6016, generator=generator, dtype=torch.float32
    ).cuda()
    router_logits = front[:, :896]
    routed_input = front[:, 896 : 896 + 3584]
    bias = torch.randn(896, generator=generator, dtype=torch.float32).cuda()
    scaling = 2.5

    weights, ids, packed, actual_q, actual_s = kimi3_route_pack_quant_mxfp8(
        router_logits,
        bias,
        routed_input,
        routed_scaling_factor=scaling,
        renormalize=True,
        enable_pdl=enable_pdl,
    )
    scores = router_logits.sigmoid()
    expected_ids = torch.topk(scores + bias, 16, dim=-1).indices.to(torch.int32)
    expected_weights = scores.gather(1, expected_ids.long())
    expected_weights = (
        expected_weights / expected_weights.sum(dim=-1, keepdim=True) * scaling
    ).to(torch.bfloat16)
    expected_packed = (ids << 16) | weights.view(torch.uint16).to(torch.int32)

    assert torch.equal(ids, expected_ids)
    assert torch.equal(weights, expected_weights)
    assert torch.equal(packed, expected_packed)

    groups = routed_input.reshape(num_tokens, -1, 32)
    raw_scale = groups.abs().amax(dim=-1).clamp_min(1.0e-10) / 448.0
    scale_bits = raw_scale.view(torch.int32)
    expected_s = (((scale_bits >> 23) & 0xFF) + ((scale_bits & 0x7FFFFF) != 0)).to(
        torch.uint8
    )
    multiplier = torch.ldexp(
        torch.ones_like(raw_scale), 127 - expected_s.to(torch.int32)
    )
    expected_q = (
        (groups * multiplier[..., None]).clamp(max=448.0).to(torch.float8_e4m3fn)
    )
    assert torch.equal(
        actual_q.view(torch.uint8).flatten(), expected_q.view(torch.uint8).flatten()
    )
    assert torch.equal(actual_s, expected_s.flatten())


@requires_flashinfer_situ
def test_flashinfer_situ_routed_moe_matches_portable_reference() -> None:
    from tokenspeed_kernel.ops.moe.flashinfer.trtllm_mxfp4 import (
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

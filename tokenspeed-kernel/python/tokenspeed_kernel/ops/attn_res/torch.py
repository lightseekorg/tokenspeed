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
#
# Portable torch fallback for the Attention-Residual mix. Also selected when the
# shape falls outside the Blackwell kernel's supported range.
import torch
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures


@register_kernel(
    "attn_res",
    "fwd",
    name="torch_attn_res_fwd",
    solution="torch",
    signatures=format_signatures(
        ("layer_residual", "block_residual"), "dense", {torch.bfloat16}
    ),
    priority=Priority.PORTABLE,
    tags={"portability"},
)
def torch_attn_res_fwd(
    *, layer_residual, block_residual, res_weight, rms_weight, eps, out_norm_weight=None
) -> torch.Tensor:
    # Candidates [N, T, H] = blocks then layer; RMSNorm + softmax score + mix in
    # fp32 (this sits on the global residual backbone), bf16 out.
    values = torch.cat((block_residual, layer_residual.unsqueeze(0)), dim=0).float()
    rs = (values.square().mean(-1, keepdim=True) + eps).rsqrt()
    score_weight = rms_weight.float() * res_weight.float()  # [H]
    logits = (values * rs * score_weight).sum(-1)  # [N, T]
    probs = logits.softmax(0)  # over candidates
    out = (probs.unsqueeze(-1) * values).sum(0)  # [T, H]
    out = out.to(layer_residual.dtype)
    if out_norm_weight is not None:
        # Fused following RMSNorm: stats over the bf16-rounded mix (matches the
        # separate rmsnorm-kernel path).
        of = out.float()
        rs = (of.square().mean(-1, keepdim=True) + eps).rsqrt()
        out = (of * rs * out_norm_weight.float()).to(layer_residual.dtype)
    return out

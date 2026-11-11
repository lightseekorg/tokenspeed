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

"""Shared preparation of MLA absorbed key/value projection weights."""

import torch
from tokenspeed_kernel.ops.attention.mla import (
    mla_project_value_prefers_contiguous_weight,
)

from tokenspeed.runtime.model_loader.weight_utils import bind_or_copy


def _prepare_mla_kv_b_proj_weights(
    w: torch.Tensor, self_attn
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split ``kv_b_proj`` into the absorbed ``w_kc``/``w_vc`` pair.

    When ``self_attn`` already holds a pair of the same geometry (a live
    weight update re-running ``post_load_weights``), the new values are
    copied into it so captured CUDA graphs keep valid addresses.

    An attention with ``absorbed_weights_k_major`` set runs both absorbed
    products as plain GEMMs and keeps their reduction dim contiguous
    regardless of the fused value-projection kernel's preference.
    """
    w_kc, w_vc = w.unflatten(
        0, (-1, self_attn.qk_nope_head_dim + self_attn.v_head_dim)
    ).split([self_attn.qk_nope_head_dim, self_attn.v_head_dim], dim=1)
    if not getattr(
        self_attn, "absorbed_weights_k_major", False
    ) and mla_project_value_prefers_contiguous_weight(
        dtype=w.dtype,
        heads=w_vc.shape[0],
        latent_dim=w_vc.shape[2],
        value_dim=w_vc.shape[1],
    ):
        w_kc, w_vc = w_kc.contiguous(), w_vc.transpose(1, 2).contiguous()
    else:
        w_kc = w_kc.transpose(1, 2).contiguous().transpose(1, 2)
        w_vc = w_vc.contiguous().transpose(1, 2)
    return (
        bind_or_copy(self_attn.w_kc, w_kc),
        bind_or_copy(self_attn.w_vc, w_vc),
    )

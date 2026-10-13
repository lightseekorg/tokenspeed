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

"""Per-token-head FP8 latent storage: one scale per latent row."""

from __future__ import annotations

import torch

_FP8_E4M3_MAX = 448.0


def store_latent_per_token_head(
    latent_plane: torch.Tensor,
    scale_plane: torch.Tensor,
    rope_plane: torch.Tensor,
    slots: torch.Tensor,
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
    *,
    sanitize: bool,
) -> None:
    """Scale each latent row into FP8 and store it at ``slots``.

    Args:
        latent_plane: FP8 latent bytes ``[slots, 1, kv_lora_rank]``.
        scale_plane: fp32 per-row scale ``[slots, 1, 1]``.
        rope_plane: RoPE rows divided by the scale ``[slots, 1, rope_dim]``.
        slots: Destination slot of each row.
        k_nope: Latent rows ``[rows, 1, kv_lora_rank]``.
        k_rope: Rotated RoPE rows ``[rows, 1, rope_dim]``.
        sanitize: Replace NaN/Inf with finite values first.
    """
    if sanitize:
        k_nope = torch.nan_to_num(k_nope)
        k_rope = torch.nan_to_num(k_rope)
    latent = k_nope.float()
    scale = latent.abs().amax(dim=-1, keepdim=True).clamp(1e-26) / _FP8_E4M3_MAX
    latent_plane[slots] = (
        (latent / scale).to(torch.float8_e4m3fn).view(latent_plane.dtype)
    )
    scale_plane[slots] = scale
    rope_plane[slots] = (k_rope.float() / scale).to(rope_plane.dtype)

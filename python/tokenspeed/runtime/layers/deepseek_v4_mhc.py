# Copyright (c) 2026 LightSeek Foundation
#
# Portions copyright the vLLM project contributors under Apache-2.0.

from __future__ import annotations

import tokenspeed_kernel
import torch


def mhc_fused_hc(
    x_prev: torch.Tensor,
    residual_prev: torch.Tensor,
    post_prev: torch.Tensor,
    comb_prev: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply the previous post-map followed by the current pre-map."""
    residual_cur = mhc_post(x_prev, residual_prev, post_prev, comb_prev)
    layer_input, post_cur, comb_cur = mhc_pre(
        residual_cur,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_eps,
        sinkhorn_iters,
    )
    return residual_cur, layer_input, post_cur, comb_cur


def mhc_pre(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Delegate the DSv4 pre-map to the tokenspeed-kernel boundary."""
    return tokenspeed_kernel.mhc_pre(
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_eps,
        sinkhorn_iters,
    )


def mhc_post(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    """Delegate the DSv4 post-map to the tokenspeed-kernel boundary."""
    return tokenspeed_kernel.mhc_post(hidden_states, residual, post, comb)

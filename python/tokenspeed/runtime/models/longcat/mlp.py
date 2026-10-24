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

"""Dense NVFP4 operands owned by a LongCat layer."""

from __future__ import annotations

import torch

from tokenspeed.runtime.execution.forward_step import get_is_capture_mode
from tokenspeed.runtime.models.longcat.attention0 import _bf16

_HIDDEN = 8192
_INTERMEDIATE = 1536
_ROUTER_LOGITS = 896
_TOPK = 12


def _prepare_nvfp4_weights(mlp, engine: str) -> tuple[torch.Tensor, ...]:
    """Validate and expose the NVFP4 operands without copying their weights."""
    gate = mlp.gate_up_proj
    down = mlp.down_proj
    for name, t, shape, dtype in (
        (
            "gate_up.weight_swiglu_interleaved",
            gate.weight_swiglu_interleaved,
            (2 * _INTERMEDIATE, _HIDDEN // 2),
            torch.uint8,
        ),
        (
            "gate_up.weight_scale_swiglu_interleaved",
            gate.weight_scale_swiglu_interleaved,
            (2 * _INTERMEDIATE, _HIDDEN // 16),
            torch.float8_e4m3fn,
        ),
        ("down.weight", down.weight, (_HIDDEN, _INTERMEDIATE // 2), torch.uint8),
        (
            "down.weight_scale_interleaved",
            down.weight_scale_interleaved,
            (_HIDDEN, _INTERMEDIATE // 16),
            torch.float8_e4m3fn,
        ),
    ):
        if tuple(t.shape) != shape or t.dtype != dtype:
            raise RuntimeError(
                f"{engine} engine expects {name} {shape} {dtype}, got {tuple(t.shape)} {t.dtype}"
            )
    return (
        gate.weight_swiglu_interleaved.data,
        gate.weight_scale_swiglu_interleaved.data.reshape(-1),
        gate.alpha.data.reshape(1).float().contiguous(),
        gate.input_scale_inv.data,
        down.input_scale_inv.data.reshape(1).float().contiguous(),
        down.weight.data,
        down.weight_scale_interleaved.data.reshape(-1),
        down.alpha.data.reshape(1).float().contiguous(),
    )


class LongcatMlpWeights:
    """Prepare a dense MLP and its following norm; MLP0 also owns router operands."""

    def __init__(self, mlp, norm, *, moe=None):
        self.mlp = mlp
        self.norm = norm
        self.moe = moe
        self.device = mlp.down_proj.weight.device
        if not getattr(mlp, "_use_nvfp4_gemm_swiglu_nvfp4_quant", False):
            raise ValueError("LongCat requires NVFP4 dense MLP weights")
        if moe is not None:
            self._check_router(moe)
        self.gate_weight = self.gate_scale = self.gate_alpha = None
        self.gate_input_scale_inv = self.gate_quant_scale = None
        self.down_weight = self.down_scale = self.down_alpha = None
        self.gamma = self.router_weight = self.router_bias = None
        self._ready = False

    @staticmethod
    def _check_router(moe):
        router = moe.router
        if tuple(router.classifier.weight.shape) != (_ROUTER_LOGITS, _HIDDEN):
            raise ValueError("LongCat expects a [896, 8192] router")
        if router.e_score_correction_bias.shape != (_ROUTER_LOGITS,):
            raise ValueError("LongCat expects an [896] router correction bias")
        if moe.topk.topk_config.top_k != _TOPK or moe.topk.topk_config.renormalize:
            raise ValueError("LongCat routing requires top-12 without renormalization")
        if float(moe.routed_scaling_factor) != 9.0:
            raise ValueError("LongCat routing requires scaling factor 9")
        if moe.zero_expert_num != 128 or moe.zero_expert_type not in (
            "identity",
            "copy",
        ):
            raise ValueError("LongCat requires 128 identity zero experts")

    def _prepare(self):
        if get_is_capture_mode():
            raise RuntimeError(
                "LongCat MLP weights require an eager warm-up before capture"
            )
        (
            self.gate_weight,
            self.gate_scale,
            self.gate_alpha,
            self.gate_input_scale_inv,
            self.gate_quant_scale,
            self.down_weight,
            self.down_scale,
            self.down_alpha,
        ) = _prepare_nvfp4_weights(self.mlp, "LongCat")
        self.gamma = _bf16(self.norm.weight.data)
        if self.moe is not None:
            router = self.moe.router
            self.router_weight = _bf16(router.classifier.weight.data)
            self.router_bias = router.e_score_correction_bias.data.float().contiguous()
        self._ready = True

    @property
    def input_scale_inv(self):
        if not self._ready:
            self._prepare()
        return self.gate_input_scale_inv

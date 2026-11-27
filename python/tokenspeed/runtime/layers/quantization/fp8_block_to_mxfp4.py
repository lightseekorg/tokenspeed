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

"""Load-time requantization of block-FP8 routed experts to MXFP4.

``--moe-requant-experts mxfp4`` serves a block-FP8 checkpoint's routed experts
as MXFP4 (E2M1 values, one E8M0 scale per 32 values along K). The MoE layer
loads the checkpoint's FP8 weights and FP32 128x128 block scales into staging
parameters exactly as the FP8 path does, then, before the MoE backend's weight
processing, replaces them with MXFP4 codes and scales:

1. dequantize: ``x = float32(w_fp8) * scale_inv[n // 128, k // 128]`` (one
   IEEE FP32 multiply per element);
2. quantize each run of 32 consecutive K values with the Triton MXFP4 quantizer
   (``tokenspeed_kernel.ops.quantization.quantize_mxfp4``, linear scales):
   ``amax`` rounded to a power of two at mantissa bit 21, scale exponent
   ``floor(log2(amax_r)) - 2`` (E8M0 byte ``+127``), values ``x * 2**-e``
   rounded to nearest-even E2M1 and saturated at 6.

Every step is element-wise or per 32-value group along K. Tensor parallelism
splits W13 along N (gate/up rows) and W2 along K at multiples of 128 (the FP8
block, hence of 32), and expert parallelism splits whole experts, so every
layout quantizes the same groups with the same scale: the MXFP4 bytes a rank
holds are the bytes of the whole-checkpoint requantization at that rank's
slice. ``TOKENSPEED_MOE_REQUANT_DIGEST_DIR`` writes one SHA-256 per
(layer, global expert, projection, 1/8 intermediate block) per rank so two
layouts' bytes can be compared without gathering them.
"""

from __future__ import annotations

import hashlib
import logging
import os

import torch
from torch import nn

logger = logging.getLogger(__name__)

MXFP4_GROUP = 32
# Digest granularity: an eighth of the intermediate size, the TP8 slice.
DIGEST_BLOCKS = 8
_EXPERT_CHUNK = 8
# Whether this process (rank) has logged its requantization; one line per rank.
_requantization_logged = False


def dequantize_fp8_block(
    weight: torch.Tensor, scale_inv: torch.Tensor, block_shape: tuple[int, int]
) -> torch.Tensor:
    """FP32 values of ``weight`` ``[..., N, K]`` with block scales
    ``[..., ceil(N / bn), ceil(K / bk)]``."""
    block_n, block_k = block_shape
    n, k = weight.shape[-2:]
    scale = scale_inv.to(torch.float32)
    scale = scale.repeat_interleave(block_n, dim=-2)[..., :n, :]
    scale = scale.repeat_interleave(block_k, dim=-1)[..., :k]
    return weight.to(torch.float32) * scale


def requantize_fp8_block_to_mxfp4(
    weight: torch.Tensor, scale_inv: torch.Tensor, block_shape: tuple[int, int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """MXFP4 ``(codes [..., N, K/2], scales [..., N, K/32])`` of a block-FP8
    tensor, both uint8; deterministic and independent of how ``weight`` was
    sliced at multiples of the FP8 block."""
    from tokenspeed_kernel.ops.quantization import quantize_mxfp4

    k = weight.shape[-1]
    if k % MXFP4_GROUP or block_shape[1] % MXFP4_GROUP:
        raise ValueError(
            f"MXFP4 requantization needs K ({k}) and the FP8 K block "
            f"({block_shape[1]}) divisible by {MXFP4_GROUP}"
        )
    values = dequantize_fp8_block(weight, scale_inv, block_shape)
    codes, scales = quantize_mxfp4(
        values.reshape(-1, k), scale_size=MXFP4_GROUP, scale_layout="linear"
    )
    return (
        codes.reshape(*weight.shape[:-1], k // 2),
        scales.reshape(*weight.shape[:-1], k // MXFP4_GROUP),
    )


def _requantize_param(
    weight: torch.Tensor, scale_inv: torch.Tensor, block_shape: tuple[int, int]
) -> tuple[torch.Tensor, torch.Tensor]:
    experts, n, k = weight.shape
    codes = torch.empty((experts, n, k // 2), dtype=torch.uint8, device=weight.device)
    scales = torch.empty(
        (experts, n, k // MXFP4_GROUP), dtype=torch.uint8, device=weight.device
    )
    for start in range(0, experts, _EXPERT_CHUNK):
        rows = slice(start, start + _EXPERT_CHUNK)
        codes[rows], scales[rows] = requantize_fp8_block_to_mxfp4(
            weight[rows], scale_inv[rows], block_shape
        )
    return codes, scales


def requantize_moe_layer(layer: nn.Module, block_shape: tuple[int, int]) -> None:
    """Replace ``layer``'s FP8 staging experts with MXFP4 parameters.

    Reads ``w13_weight``/``w2_weight`` (E4M3) and ``w13_weight_scale_inv``/
    ``w2_weight_scale_inv`` (FP32 blocks); leaves ``w13_weight``/``w2_weight``
    (packed E2M1, uint8) and ``w13_weight_scale``/``w2_weight_scale`` (E8M0,
    uint8) in the layout ``create_mxfp4_weight_pair`` gives, one layer at a
    time so the FP8 staging is freed before the next layer converts.
    """
    experts = layer.w13_weight.shape[0]
    for name in ("w13", "w2"):
        weight = getattr(layer, f"{name}_weight")
        scale_inv = getattr(layer, f"{name}_weight_scale_inv")
        codes, scales = _requantize_param(weight.data, scale_inv.data, block_shape)
        layer.register_parameter(
            f"{name}_weight", nn.Parameter(codes, requires_grad=False)
        )
        layer.register_parameter(
            f"{name}_weight_scale", nn.Parameter(scales, requires_grad=False)
        )
        layer.register_parameter(f"{name}_weight_scale_inv", None)
        del weight, scale_inv
    _write_digest(layer)
    _log_requantization(layer, experts, block_shape)


def _log_requantization(
    layer: nn.Module, experts: int, block_shape: tuple[int, int]
) -> None:
    """State once per rank, at its first converted layer, that the block-FP8
    routed experts are requantized to MXFP4 at load."""
    global _requantization_logged
    if _requantization_logged:
        return
    _requantization_logged = True
    logger.info(
        "MoE expert requantization (--moe-requant-experts mxfp4): block-FP8 "
        "routed experts (%dx%d scale blocks) are requantized to MXFP4 at load, "
        "%d experts per MoE layer on this rank; first layer %s, logged once "
        "per rank",
        block_shape[0],
        block_shape[1],
        experts,
        getattr(layer, "prefix", ""),
    )


def block_digests(
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    *,
    expert_start: int,
    intermediate_size: int,
    tp_rank: int,
    tp_size: int,
) -> list[tuple[int, str, int, str]]:
    """``(global expert, projection, block, sha256)`` of every 1/8-intermediate
    block a rank holds; equal across TP/EP layouts iff the bytes are equal."""
    block = intermediate_size // DIGEST_BLOCKS
    if intermediate_size % DIGEST_BLOCKS or (DIGEST_BLOCKS % tp_size):
        raise ValueError("digest blocks need intermediate % 8 == 0 and TP | 8")
    ispp = intermediate_size // tp_size
    first = tp_rank * (ispp // block)
    rows = []

    def sha(*tensors: torch.Tensor) -> str:
        h = hashlib.sha256()
        for t in tensors:
            h.update(t.contiguous().cpu().numpy().tobytes())
        return h.hexdigest()

    for local in range(w13.shape[0]):
        expert = expert_start + local
        for b in range(ispp // block):
            r = slice(b * block, (b + 1) * block)
            up = slice(ispp + b * block, ispp + (b + 1) * block)
            c = slice(b * block // 2, (b + 1) * block // 2)
            s = slice(b * block // MXFP4_GROUP, (b + 1) * block // MXFP4_GROUP)
            rows.append(
                (expert, "gate", first + b, sha(w13[local, r], w13_scale[local, r]))
            )
            rows.append(
                (expert, "up", first + b, sha(w13[local, up], w13_scale[local, up]))
            )
            rows.append(
                (
                    expert,
                    "down",
                    first + b,
                    sha(w2[local, :, c], w2_scale[local, :, s]),
                )
            )
    return rows


def _write_digest(layer: nn.Module) -> None:
    directory = os.environ.get("TOKENSPEED_MOE_REQUANT_DIGEST_DIR")
    if not directory:
        return
    os.makedirs(directory, exist_ok=True)
    rows = block_digests(
        layer.w13_weight.data,
        layer.w13_weight_scale.data,
        layer.w2_weight.data,
        layer.w2_weight_scale.data,
        expert_start=layer.ep_rank * layer.num_local_experts,
        intermediate_size=layer.intermediate_size,
        tp_rank=layer.tp_rank,
        tp_size=layer.tp_size,
    )
    path = os.path.join(
        directory, f"ep{layer.ep_rank}-tp{layer.tp_rank}-pid{os.getpid()}.tsv"
    )
    with open(path, "a", encoding="utf-8") as out:
        for expert, projection, block, digest in rows:
            out.write(f"{layer.prefix}\t{expert}\t{projection}\t{block}\t{digest}\n")


__all__ = [
    "block_digests",
    "dequantize_fp8_block",
    "requantize_fp8_block_to_mxfp4",
    "requantize_moe_layer",
]

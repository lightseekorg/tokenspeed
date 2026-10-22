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

"""Read preconverted W4A4 expert shards; never quantize or write weights."""

import json
import math
from pathlib import Path

import torch


def read_projection(config_path, *, layer_index, ep_rank, ep_size, projection, shape):
    """Validate a manifest-selected CPU ND shard against the local MXFP4 geometry.

    The manifest binds layer, EP rank and projection to a preconverted file.
    Shape is (local experts, output features, input features). The original
    MXFP4 checkpoint still supplies the model's non-expert weights. Offline
    preparation must use that same checkpoint and the declared EP layout.
    """
    manifest_path = Path(config_path)
    config = json.loads(manifest_path.read_text())
    if config.get("format") != "tokenspeed-v41-w4a4" or config.get("version") != 1:
        raise ValueError(
            "Expected a version-1 preconverted W4A4 manifest; online conversion is unsupported"
        )
    if config["ep_size"] != ep_size or not 0 <= ep_rank < ep_size:
        raise ValueError("W4A4 manifest EP layout does not match the model")
    if projection not in ("w13", "w2"):
        raise ValueError("W4A4 projection must be w13 or w2")
    try:
        filename = config["layers"][str(layer_index)][str(ep_rank)][projection]
    except KeyError as error:
        raise ValueError(
            f"Missing preconverted W4A4 shard: layer={layer_index}, EP rank={ep_rank}, projection={projection}"
        ) from error
    path = manifest_path.parent / filename
    data = torch.load(path, map_location="cpu", weights_only=True)
    if data["version"] != 1 or data["key"] != path.stem:
        raise ValueError("Preconverted W4A4 file identity mismatch")
    experts, output_size, input_size = shape
    if min(shape) <= 0 or input_size % 8:
        raise ValueError("Invalid W4A4 expert dimensions")
    weight, scales = data["weight"], data["scales"]
    if (
        data["input_size"] != input_size
        or data["output_size"] != output_size
        or weight.dtype != torch.int32
        or tuple(weight.shape) != (experts, output_size, input_size // 8)
        or scales.dtype != torch.float32
        or tuple(scales.shape) != (experts, output_size)
        or not bool(torch.isfinite(scales).all())
        or not bool((scales > 0).all())
    ):
        raise ValueError(
            f"Invalid preconverted W4A4 weight/scale layout or values: "
            f"layer={layer_index}, EP rank={ep_rank}, projection={projection}, "
            f"expected unpacked shape={shape}, packed={tuple(weight.shape)} "
            f"({weight.dtype}), scales={tuple(scales.shape)} ({scales.dtype})"
        )
    left, right = data["left"], data["right"]
    for matrix in (left, right):
        if (
            matrix.dtype != torch.bfloat16
            or matrix.ndim != 2
            or matrix.shape[0] != matrix.shape[1]
            or not bool(torch.isfinite(matrix).all())
        ):
            raise ValueError("FlatQuant matrices must be finite square BF16 tensors")
    if left.shape[0] * right.shape[0] != input_size:
        raise ValueError("FlatQuant matrix dimensions do not match the projection")
    ratio = data["clip_ratio"]
    if (
        not isinstance(ratio, (float, int))
        or not math.isfinite(ratio)
        or not 0 < ratio <= 1
    ):
        raise ValueError("FlatQuant activation clip ratio must be in (0, 1]")
    return data


def load_projection(
    config_path, *, layer_index, ep_rank, ep_size, projection, shape, device
):
    """Load checked INT4 codes, FP32 scales and activation matrices for NPU GMM.

    Only device transfer, lossless scale bit packing and ND-to-NZ layout
    preparation happen here. Missing artifacts fail without a conversion fallback.
    """
    data = read_projection(
        config_path,
        layer_index=layer_index,
        ep_rank=ep_rank,
        ep_size=ep_size,
        projection=projection,
        shape=shape,
    )
    import torch_npu
    from tokenspeed_kernel_npu.ops.dsv41_moe import (
        AscendGroupedLinear,
        FlatQuantParameters,
    )

    packed = torch_npu.npu_format_cast(
        data["weight"].contiguous().to(device), 29, customize_dtype=3
    )
    scale_words = (
        data["scales"]
        .contiguous()
        .view(torch.int32)
        .to(torch.int64)
        .view(torch.uint64)
        .unsqueeze(1)
        .to(device)
    )
    transform = FlatQuantParameters(
        data["left"].contiguous().to(device),
        data["right"].contiguous().to(device),
        data["clip_ratio"],
    )
    return AscendGroupedLinear(
        "w4a4", packed, scale_words, transform, data["input_size"], data["output_size"]
    )

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

from __future__ import annotations

import time
from typing import Any

import torch

from tokenspeed.runtime.multimodal.inputs import Modality, MultimodalDataItem
from tokenspeed.runtime.utils import get_colorful_logger
from tokenspeed.runtime.utils.env import envs

logger = get_colorful_logger(__name__)

EncoderWarmupItems = dict[Modality, list[MultimodalDataItem]]


def install_encoder_cudagraph_wrappers(
    model: Any, mm_attention_backend: str | None
) -> dict[str, Any]:
    """Install model-provided encoder graph wrappers and return active wrappers."""
    builder = getattr(model, "make_encoder_cudagraph_wrappers", None)
    if not (
        builder is not None
        and getattr(model, "is_multimodal_active", True)
        and envs.TOKENSPEED_MM_ENABLE_ENCODER_CUDA_GRAPH.get()
        and mm_attention_backend != "flashinfer_cudnn"
    ):
        return {}

    active_wrappers = {}
    for encoder_attr, wrapper in builder(model.mapping).items():
        if not hasattr(model, encoder_attr):
            logger.warning(
                "Skipping encoder CUDA graph wrapper for missing attribute %s",
                encoder_attr,
            )
            continue
        setattr(model, encoder_attr, wrapper)
        active_wrappers[encoder_attr] = wrapper
    return active_wrappers


def prewarm_multimodal_encoders(
    model: Any,
    *,
    skip_server_warmup: bool,
    device: str | torch.device,
) -> None:
    """Run model-provided synthetic items through each multimodal encoder seam."""
    if (
        not envs.TOKENSPEED_MM_ENABLE_VISION_PREWARM.get()
        or skip_server_warmup
        or not getattr(model, "is_multimodal_active", False)
    ):
        return

    make_items = getattr(model, "make_encoder_warmup_items", None)
    if make_items is None:
        return

    patches_per_side = envs.TOKENSPEED_MM_VISION_PREWARM_PATCHES_PER_SIDE.get()
    if patches_per_side <= 0:
        return

    for modality, items in make_items(patches_per_side).items():
        if not items:
            continue
        encoder_attr = f"{modality.name.lower()}_encoder"
        encoder = getattr(model, encoder_attr, None)
        if not callable(encoder):
            raise RuntimeError(
                f"Multimodal warmup items require callable model.{encoder_attr}"
            )

        start = time.perf_counter()
        try:
            with torch.inference_mode():
                output = encoder(items)
                del output
                warmup_device = torch.device(device)
                if warmup_device.type == "cuda":
                    torch.cuda.synchronize(warmup_device)
        except Exception:
            logger.exception(
                "Multimodal encoder prewarm failed: modality=%s",
                modality.name.lower(),
            )
            raise
        logger.info(
            "Multimodal encoder prewarm complete: modality=%s "
            "patches_per_side=%d elapsed=%.3f ms",
            modality.name.lower(),
            patches_per_side,
            (time.perf_counter() - start) * 1000.0,
        )

"""Small Triton-Ascend helpers used by layernorm kernels."""

from __future__ import annotations

from typing import Any

import torch
import triton.language.extra.cann.extension as _cann_extension  # type: ignore
from tokenspeed_kernel._triton import tl, triton

_NUM_AICORE = -1
_NUM_VECTORCORE = -1


def _resolve_triton_ascend_op(op_name: str):
    extension_op = getattr(_cann_extension, op_name, None)
    if extension_op is not None:
        return extension_op

    raise RuntimeError(
        f"Failed to resolve Triton-Ascend op {op_name!r} from "
        "triton.language.extra.cann.extension."
    )


insert_slice = _resolve_triton_ascend_op("insert_slice")
extract_slice = _resolve_triton_ascend_op("extract_slice")
get_element = _resolve_triton_ascend_op("get_element")


def init_device_properties_triton() -> None:
    """Initialize cached Ascend device properties used for Triton tiling."""
    global _NUM_AICORE, _NUM_VECTORCORE
    if _NUM_AICORE != -1 and _NUM_VECTORCORE != -1:
        return

    device_properties: dict[str, Any] = (
        triton.runtime.driver.active.utils.get_device_properties(
            torch.npu.current_device()
        )
    )
    _NUM_AICORE = int(device_properties.get("num_aicore", -1))
    _NUM_VECTORCORE = int(device_properties.get("num_vectorcore", -1))
    if _NUM_AICORE <= 0 or _NUM_VECTORCORE <= 0:
        raise RuntimeError(
            f"Failed to detect Ascend core counts from {device_properties!r}"
        )


def get_vectorcore_num() -> int:
    """Return the current NPU's vector-core count."""
    init_device_properties_triton()
    return _NUM_VECTORCORE


__all__ = [
    "extract_slice",
    "get_element",
    "get_vectorcore_num",
    "init_device_properties_triton",
    "insert_slice",
    "tl",
    "triton",
]

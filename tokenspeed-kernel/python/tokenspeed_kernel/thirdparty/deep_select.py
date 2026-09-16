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

"""Optional DeepSelect dependency, provided by tokenspeed-deepselect."""

from __future__ import annotations

from functools import lru_cache
from types import ModuleType

import torch


@lru_cache(maxsize=1)
def deep_select_api() -> ModuleType:
    """Load the optional native top-k API, with an actionable failure message."""
    try:
        import deep_select
    except (ImportError, OSError) as exc:
        raise ImportError(
            "DeepSelect selection requires tokenspeed-deepselect>=1.0.0.post20260910 "
            "including its native CUDA extension."
        ) from exc
    if not callable(getattr(deep_select, "topk", None)) or not callable(
        getattr(deep_select, "get_stride_requirement", None)
    ):
        raise ImportError("Installed DeepSelect does not expose its v1.0.0 top-k API")
    return deep_select


@lru_cache(maxsize=1)
def is_deep_select_available() -> bool:
    """Return whether the optional official Python API and native extension load."""
    try:
        deep_select_api()
    except ImportError:
        return False
    return True


def deep_select_alignment() -> tuple[int, int]:
    """Return native input and output row-stride alignments, in bytes."""
    return deep_select_api().get_stride_requirement()


def deep_select_topk(
    scores: torch.Tensor,
    lengths: torch.Tensor,
    topk: int,
    output: torch.Tensor,
) -> torch.Tensor:
    """Write unsorted FP32 top-k indices to caller-owned aligned output.

    Args:
        scores: CUDA FP32 [queries, positions], with native aligned row stride.
        lengths: CUDA contiguous int32 [queries], valid exclusive row ends.
        topk: Number of indices to select, from 1 through 4096.
        output: CUDA int32 [queries, topk], with native aligned row stride.

    Returns:
        The supplied output, with -1 after rows shorter than topk. Masked
        negative-infinity values require filtering by the operation adapter.
        No values are returned and no BF16 conversion is performed.
    """
    _, indices = deep_select_api().topk(
        scores,
        topk,
        sorted=False,
        begin=None,
        end=lengths,
        indices_type=torch.int32,
        sorted_index=False,
        hint=None,
        output_idx=output,
        output_idx_offset=None,
        idx_oob_fill_value=-1,
        value_oob_fill_value=-float("inf"),
        return_value=False,
        abort_when_nan_found=True,
    )
    return indices

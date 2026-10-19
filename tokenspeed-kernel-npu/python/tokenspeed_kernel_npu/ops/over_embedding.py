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

"""Ascend device-hash and host-table LongCat OE implementation."""

from __future__ import annotations

import torch


def _load_flash_oe_ops() -> None:
    try:
        import flash_ops  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "flash_ops with the standard LongCat OE operator is required"
        ) from exc
    required = (
        "npu_append_packed_oe_lookup",
        "npu_register_host_oe_tables",
    )
    missing = [name for name in required if not hasattr(torch.ops.flash, name)]
    if missing:
        raise RuntimeError(
            "flash_ops is missing required OE operators: " + ", ".join(missing)
        )


def register_host_tables(
    *, oe_tables: tuple[torch.Tensor, ...], device: torch.device | str | int
) -> None:
    """Register mmap-backed OE tables once for direct AI Core access."""
    _load_flash_oe_ops()
    if isinstance(device, int):
        device_index = device
    else:
        npu_device = torch.device(device)
        if npu_device.type != "npu":
            raise ValueError("Ascend Host OE registration requires an NPU device")
        device_index = npu_device.index
        if device_index is None:
            device_index = torch.npu.current_device()
    torch.ops.flash.npu_register_host_oe_tables(list(oe_tables), device_index)


def append_packed_lookup(
    *,
    input_ids: torch.Tensor,
    input_start_offsets: torch.Tensor,
    req_pool_indices: torch.Tensor,
    active_request_mask: torch.Tensor,
    history_token_ids: torch.Tensor,
    committed_lengths: torch.Tensor,
    oe_tables: tuple[torch.Tensor, ...],
    out: torch.Tensor,
    spec,
) -> None:
    """Append, hash, and lookup Host or Device tables in one Flash operator."""
    _load_flash_oe_ops()
    torch.ops.flash.npu_append_packed_oe_lookup(
        input_ids,
        input_start_offsets,
        req_pool_indices,
        active_request_mask,
        history_token_ids,
        committed_lengths,
        list(oe_tables),
        out,
        [fragment.ngram_order for fragment in spec.fragments],
        [fragment.modulus for fragment in spec.fragments],
        list(spec.ignored_token_ids),
        spec.vocab_size,
        -1 if spec.eos_token_id is None else spec.eos_token_id,
        spec.segment_ignored_tokens,
    )


__all__ = ["append_packed_lookup", "register_host_tables"]

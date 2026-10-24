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

"""Compile cache and PyTorch driver for LongCat packed OE lookup."""

from __future__ import annotations

import threading
from typing import Any

import torch


def _cutedsl_available() -> bool:
    try:
        import cutlass  # noqa: F401
        import cutlass.cute  # noqa: F401
        import quack.compile_utils  # noqa: F401
    except ImportError:
        return False
    return True


class LongCatOEAppendPackedLookup:
    """Compile variable-length lookup for two 512-column fragments on first use."""

    def __init__(self) -> None:
        self._compiled: dict[tuple[object, ...], Any] = {}
        self._compile_lock = threading.Lock()
        self._available: bool | None = None

    def is_available(self) -> bool:
        """Return whether the CuTe DSL toolchain can be imported."""
        if self._available is None:
            self._available = _cutedsl_available()
        return self._available

    @staticmethod
    def _stream(device: torch.device):
        from cuda.bindings.driver import CUstream

        return CUstream(torch.cuda.current_stream(device).cuda_stream)

    @staticmethod
    def _device_index(device: torch.device) -> int:
        if device.index is not None:
            return device.index
        return torch.cuda.current_device()

    def _compile(
        self,
        *,
        device: torch.device,
        vocab_size: int,
        fragment_configs: tuple[tuple[int, int, int], ...],
        ignored_token_ids: tuple[int, ...],
        eos_token_id: int | None,
        enable_pdl: bool,
    ) -> Any:
        import cutlass
        import cutlass.cute as cute
        from quack.compile_utils import make_fake_tensor
        from tokenspeed_kernel.thirdparty.cute_dsl.longcat.embedding._kernel import (
            CuteLongCatOEAppendPackedLookup,
        )

        token_count = cute.sym_int(divisibility=1)
        offset_count = cute.sym_int(divisibility=1)
        request_count = cute.sym_int(divisibility=1)
        slot_count = cute.sym_int(divisibility=1)
        context_capacity = cute.sym_int(divisibility=1)
        input_ids = make_fake_tensor(cutlass.Int32, (token_count,), divisibility=1)
        input_start_offsets = make_fake_tensor(
            cutlass.Int32, (offset_count,), divisibility=1
        )
        req_pool_indices = make_fake_tensor(
            cutlass.Int64, (request_count,), divisibility=1
        )
        active_request_mask = make_fake_tensor(
            cutlass.Boolean, (request_count,), divisibility=8
        )
        history_token_ids = make_fake_tensor(
            cutlass.Int32,
            (slot_count, context_capacity),
            divisibility=1,
        )
        committed_lengths = make_fake_tensor(
            cutlass.Int32, (slot_count,), divisibility=1
        )
        oe_tables = tuple(
            make_fake_tensor(
                cutlass.BFloat16,
                (modulus, feature_width),
                divisibility=8,
            )
            for _, modulus, feature_width in fragment_configs
        )
        out = make_fake_tensor(
            cutlass.BFloat16,
            (
                token_count,
                sum(feature_width for _, _, feature_width in fragment_configs),
            ),
            divisibility=8,
        )

        kernel = CuteLongCatOEAppendPackedLookup(
            vocab_size=vocab_size,
            fragment_configs=fragment_configs,
            ignored_token_ids=ignored_token_ids,
            eos_token_id=eos_token_id,
            use_pdl=enable_pdl,
        )
        with torch.cuda.device(device):
            return cute.compile(
                kernel,
                input_ids,
                input_start_offsets,
                req_pool_indices,
                active_request_mask,
                history_token_ids,
                committed_lengths,
                *oe_tables,
                out,
                self._stream(device),
                options="--enable-tvm-ffi --ptxas-options -maxrregcount=64",
            )

    def __call__(
        self,
        input_ids: torch.Tensor,
        input_start_offsets: torch.Tensor,
        req_pool_indices: torch.Tensor,
        active_request_mask: torch.Tensor,
        history_token_ids: torch.Tensor,
        committed_lengths: torch.Tensor,
        oe_tables: tuple[torch.Tensor, ...],
        out: torch.Tensor,
        *,
        vocab_size: int,
        fragment_configs: tuple[tuple[int, int, int], ...],
        ignored_token_ids: tuple[int, ...] = (),
        eos_token_id: int | None = None,
        enable_pdl: bool = False,
    ) -> None:
        """Launch packed lookup for two 512-column fragments."""
        device = input_ids.device
        key = (
            self._device_index(device),
            vocab_size,
            fragment_configs,
            ignored_token_ids,
            eos_token_id,
            enable_pdl,
        )
        compiled = self._compiled.get(key)
        if compiled is None:
            with self._compile_lock:
                compiled = self._compiled.get(key)
                if compiled is None:
                    compiled = self._compile(
                        device=device,
                        vocab_size=vocab_size,
                        fragment_configs=fragment_configs,
                        ignored_token_ids=ignored_token_ids,
                        eos_token_id=eos_token_id,
                        enable_pdl=enable_pdl,
                    )
                    self._compiled[key] = compiled

        compiled(
            input_ids,
            input_start_offsets,
            req_pool_indices,
            active_request_mask,
            history_token_ids,
            committed_lengths,
            *oe_tables,
            out,
            self._stream(device),
        )


longcat_oe_append_packed_lookup = LongCatOEAppendPackedLookup()

__all__ = [
    "LongCatOEAppendPackedLookup",
    "longcat_oe_append_packed_lookup",
]

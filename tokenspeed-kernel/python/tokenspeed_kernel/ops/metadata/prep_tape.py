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

"""Declarative one-launch CUDA-graph replay metadata prep.

Attention backends refill the captured graph's persistent input buffers
before every replay. Recorded eagerly that is a chain of tiny fills, copies
and gathers -- tens of kernel launches for a few kilobytes. A
:class:`PrepTape` records that chain ONCE against the persistent buffers
(pointers never change across replays) and then executes it each step with
one small H2D register upload plus one interpreter-kernel launch.

Usage::

    tape = PrepTape(device)
    tape.fill(buf_a, n=Reg.BS, value=-1)
    tape.copy(dst_view, src_view, n=Reg.REAL_BS)
    tape.gather(dst, table, idx, n=Reg.REAL_BS, oob_value=-1)
    tape.filltail(buf_b, live=Reg.REAL_BS, total=Reg.BS, value=0)
    tape.finalize()
    ...
    tape.run({Reg.BS: bs, Reg.REAL_BS: real_bs})   # every replay

All tensors must be int32, contiguous in their last dimension, and outlive
the tape (their addresses are baked into the descriptors). Only int32
metadata is supported -- that is what graph input buffers are made of.
"""

from __future__ import annotations

import enum

import torch

_REG_FLAG = 1 << 62
_FIELDS = 8

_OP_FILL = 0
_OP_COPY = 1
_OP_GATHER = 2
_OP_COPY2D = 3
_OP_FILLTAIL = 4
_OP_STATE_PAGES = 5


class Reg(enum.IntEnum):
    """Well-known register slots; backends may use any int < 16."""

    BS = 0
    REAL_BS = 1
    TOKENS = 2
    USER0 = 8
    USER1 = 9
    USER2 = 10
    USER3 = 11
    # Pointer slots: per-step tensor base addresses (data_ptr) go here.
    PTR0 = 16
    PTR1 = 17
    PTR2 = 18
    PTR3 = 19
    PTR4 = 20


def _ref(v: "int | Reg") -> int:
    """Encode an immediate or a register reference into a descriptor field.

    Register refs carry the ``2^62`` flag bit with the slot in the low bits;
    real pointers and metadata immediates never reach that bit.
    """
    return (_REG_FLAG | int(v)) if isinstance(v, Reg) else int(v)


def _check(t: torch.Tensor, name: str) -> torch.Tensor:
    if t.dtype != torch.int32:
        raise TypeError(f"{name} must be int32, got {t.dtype}")
    if t.stride(-1) != 1:
        raise ValueError(f"{name} must be unit-stride in the last dim")
    return t


class PrepTape:
    """Records metadata prep ops and replays them in one kernel launch."""

    def __init__(self, device) -> None:
        self._device = torch.device(device)
        self._rows: list[list[int]] = []
        self._keepalive: list[torch.Tensor] = []
        self._descs: torch.Tensor | None = None
        self._regs_dev = torch.zeros(32, dtype=torch.int64, device=self._device)
        self._regs_pinned = torch.zeros(32, dtype=torch.int64, pin_memory=True)

    # -- recording ---------------------------------------------------------

    def _row(self, op, dst, src, idx, n, m, stride, scalar) -> None:
        if self._descs is not None:
            raise RuntimeError("tape already finalized")
        for t in (dst, src, idx):
            if isinstance(t, torch.Tensor):
                self._keepalive.append(t)
        self._rows.append(
            [
                op,
                dst.data_ptr() if isinstance(dst, torch.Tensor) else 0,
                src.data_ptr() if isinstance(src, torch.Tensor) else 0,
                idx.data_ptr() if isinstance(idx, torch.Tensor) else 0,
                _ref(n),
                _ref(m),
                int(stride),
                _ref(scalar),
            ]
        )

    def fill(self, dst: torch.Tensor, n: "int | Reg", value: "int | Reg") -> None:
        """``dst[0:n] = value``."""
        self._row(_OP_FILL, _check(dst, "dst"), None, None, n, 0, 0, value)

    def copy(
        self,
        dst: torch.Tensor,
        src: torch.Tensor,
        n: "int | Reg",
        src_offset: "int | Reg" = 0,
    ) -> None:
        """``dst[0:n] = src[src_offset : src_offset + n]``."""
        self._row(
            _OP_COPY,
            _check(dst, "dst"),
            _check(src, "src"),
            None,
            n,
            0,
            _ref(src_offset),
            0,
        )

    def gather(
        self,
        dst: torch.Tensor,
        src: torch.Tensor,
        idx: torch.Tensor,
        n: "int | Reg",
        oob_value: int = -1,
    ) -> None:
        """``dst[i] = src[idx[i]]`` for ``i < n``; ``idx[i] < 0`` writes
        ``oob_value`` (the padded-row contract)."""
        self._row(
            _OP_GATHER,
            _check(dst, "dst"),
            _check(src, "src"),
            _check(idx, "idx"),
            n,
            0,
            0,
            oob_value,
        )

    def copy2d(
        self,
        dst: torch.Tensor,
        src: torch.Tensor,
        rows: "int | Reg",
        src_cols: "int | Reg",
        pad_value: int = -1,
    ) -> None:
        """Row-wise copy with column padding: ``dst[r, :src_cols] = src[r]``,
        ``dst[r, src_cols:] = pad_value``. ``dst`` is 2-D; ``src`` rows are
        packed at ``src_cols`` stride."""
        d = _check(dst, "dst")
        if d.ndim != 2:
            raise ValueError("copy2d dst must be 2-D")
        self._row(
            _OP_COPY2D,
            d,
            _check(src, "src"),
            None,
            rows,
            src_cols,
            d.stride(0),
            pad_value,
        )

    def filltail(
        self,
        dst: torch.Tensor,
        live: "int | Reg",
        total: "int | Reg",
        value: "int | Reg",
    ) -> None:
        """``dst[live:total] = value`` (padding rows)."""
        self._row(_OP_FILLTAIL, _check(dst, "dst"), None, None, live, total, 0, value)

    def state_pages(
        self,
        state_in: torch.Tensor,
        state_out: torch.Tensor,
        rows_ptr: Reg,
        seq_lens_ptr: Reg,
        bs: "int | Reg",
        max_slots: "int | Reg",
        page_size: int,
    ) -> None:
        """Decode dual-index state paging for one group (before = after-1):
        ``state_in[i] = rows[i, clamp((after-2)//P)]`` (0 when no history),
        ``state_out[i] = rows[i, clamp((after-1)//P)]``. ``rows`` and
        ``seq_lens`` are per-step tensors passed via pointer registers."""
        self._rows.append(
            [
                _OP_STATE_PAGES,
                _check(state_in, "state_in").data_ptr(),
                _ref(rows_ptr),
                _check(state_out, "state_out").data_ptr(),
                _ref(bs),
                _ref(max_slots),
                int(page_size),
                _ref(seq_lens_ptr),
            ]
        )
        self._keepalive += [state_in, state_out]

    # -- execution ---------------------------------------------------------

    def finalize(self) -> None:
        """Upload the descriptor table; the tape becomes immutable."""
        self._descs = torch.tensor(
            self._rows, dtype=torch.int64, device=self._device
        ).view(-1, _FIELDS)

    def __len__(self) -> int:
        return len(self._rows)

    def run(self, regs: dict) -> None:
        """Execute the tape with this step's register values.

        One pinned write + one async H2D + one kernel launch, regardless of
        how many ops the tape holds.
        """
        from tokenspeed_kernel.ops.metadata.triton_tape import run_tape

        if self._descs is None:
            raise RuntimeError("finalize() the tape before run()")
        for k, v in regs.items():
            self._regs_pinned[int(k)] = (
                v.data_ptr() if isinstance(v, torch.Tensor) else v
            )
        self._regs_dev.copy_(self._regs_pinned, non_blocking=True)
        run_tape(self._descs, self._regs_dev)

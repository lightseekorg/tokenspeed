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

"""Lamport mailboxes: a decode all-reduce carried by its producer.

A producer stores this rank's BF16 partial of each row straight into slot
``rank`` of every rank's mailbox (multicast or peer stores), and the consumer
on each rank polls the slots of its rows, sums them in FP32 in rank order,
rounds once and re-arms them. The data marks its own arrival: an armed slot
holds BF16 ``-0.0``, which producers never store (they publish ``+0.0``), so
neither side waits on a barrier or on the other kernel's completion, and the
sum is the same bits on every rank.

A mailbox has two parities. Successive uses on a rank alternate them, so a
producer never writes into slots its rank's previous consumer is still
reading: by the time any rank publishes into parity ``p`` again, every rank
has consumed (and re-armed) the use before.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist

# Rows of one mailbox slot; a producer publishes at most this many rows.
MAILBOX_ROWS = 64
# Rows the single-pass projection producer serves.
PROJECTION_MAX_ROWS = 8


def mailbox_supported(device: torch.device) -> bool:
    """Whether this platform builds mailboxes: NVIDIA Blackwell with the
    CuTe DSL and symmetric-memory multicast."""
    if not torch.cuda.is_available() or torch.version.hip is not None:
        return False
    if torch.cuda.get_device_capability(device)[0] != 10:
        return False
    try:
        import cutlass  # noqa: F401
        from torch.distributed import _symmetric_memory  # noqa: F401
    except ImportError:
        return False
    return True


class Mailbox:
    """Two parities of ``[tp, MAILBOX_ROWS, width]`` BF16 Lamport slots in
    symmetric memory over ``group``, armed on creation.

    Creation rendezvouses the group (every rank must build it, eagerly and
    outside graph capture) and raises when the group has no multicast memory.
    """

    def __init__(
        self, group: dist.ProcessGroup, *, width: int, device: torch.device
    ) -> None:
        from tokenspeed_kernel.ops.moe._cute_dsl.multicast_view import (
            bf16_tensor_on_pointer,
        )
        from tokenspeed_kernel.thirdparty.cute_dsl.k3_decode.common.host import (
            OprojMailbox,
        )

        self.tp = dist.get_world_size(group)
        self.rank = dist.get_rank(group)
        self.width = width
        self._box = OprojMailbox(
            group, device, self.tp, self.rank, shape=(2, self.tp, MAILBOX_ROWS, width)
        )
        shape = (self.tp, MAILBOX_ROWS, width)
        self._multicast = [
            bf16_tensor_on_pointer(
                self.multicast_address(parity),
                shape,
                (MAILBOX_ROWS * width, width, 1),
                self._box.buffer.device.index,
            )
            for parity in range(2)
        ]

    def slots(self, parity: int) -> torch.Tensor:
        """This rank's ``[tp, MAILBOX_ROWS, width]`` slots of ``parity``."""
        return self._box.buffer[parity]

    def peer_slots(self, peer: int, parity: int) -> torch.Tensor:
        """Rank ``peer``'s ``[tp, MAILBOX_ROWS, width]`` slots of ``parity``, for
        producers that store with ordinary tensor copies."""
        shape = (2, self.tp, MAILBOX_ROWS, self.width)
        return self._box.handle.get_buffer(peer, shape, torch.bfloat16)[parity]

    def multicast_address(self, parity: int) -> int:
        """The multicast address of ``parity``'s slots (stores reach every rank)."""
        return self._box.slot(parity)[1]

    def multicast_slots(self, parity: int) -> torch.Tensor:
        """``[tp, MAILBOX_ROWS, width]`` view of ``parity``'s slots at the
        multicast address, for producers' kernels (stores reach every rank)."""
        return self._multicast[parity]


@dataclass(frozen=True)
class MailboxTarget:
    """One use of a mailbox: the parity its producer publishes into and its
    consumer reads (successive uses on a rank alternate parities)."""

    mailbox: Mailbox
    parity: int


def _projection_staged_blocks(k: int) -> int:
    # Of a warp's k / 128 K blocks, the rest stay in registers; on B300 at
    # K 1536, 4 staged blocks beat 8 and 0 by 0.3 us per call.
    return min(4, k // 128 - 1)


_PROJECTIONS: dict[tuple[int, int, int, int, int], object] = {}


def publish_projection(
    x: torch.Tensor, weight: torch.Tensor, mailbox: Mailbox, parity: int
) -> None:
    """Publish this rank's ``x @ weight.T`` into slot ``rank`` of every rank's
    ``parity`` slots: FP32 MMA chains over K, rounded once to BF16 (``-0.0``
    stored as ``+0.0``). Under PDL its weights load before the wait.

    Args:
        x: Contiguous BF16 ``[T, K]`` rows, ``T <= PROJECTION_MAX_ROWS``.
        weight: Contiguous BF16 ``[mailbox.width, K]``, ``K`` a multiple of 128.
        mailbox: The destination mailbox.
        parity: The mailbox parity this use takes (0 or 1).
    """
    tokens, k = x.shape
    n = mailbox.width
    if (
        x.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or weight.shape != (n, k)
        or not x.is_contiguous()
        or not weight.is_contiguous()
        or not 0 < tokens <= PROJECTION_MAX_ROWS
        or k % 128
    ):
        raise ValueError(
            "publish_projection takes contiguous BF16 [T <= 8, K] rows and "
            f"[{n}, K] weights, K a multiple of 128"
        )
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_stream
    from tokenspeed_kernel.thirdparty.cute_dsl.k3_decode.common.host import (
        fake,
        stream,
    )
    from tokenspeed_kernel.thirdparty.cute_dsl.k3_decode.oproj.producer import (
        OprojProducer,
    )
    from tokenspeed_kernel.thirdparty.cute_dsl.latent_moe_tail.primitives import (
        to_cute,
    )

    key = (tokens, k, n, mailbox.tp, x.device.index)
    compiled = _PROJECTIONS.get(key)
    if compiled is None:
        producer = OprojProducer(k, staged_blocks=_projection_staged_blocks(k), n=n)
        rows = MAILBOX_ROWS
        compiled = _PROJECTIONS[key] = cute.compile(
            producer,
            fake((tokens, k), (k, 1)),
            fake((n, k), (k, 1)),
            fake((mailbox.tp, rows, n), (rows * n, n, 1)),
            cutlass.Int64(0),
            cutlass.Int32(0),
            make_fake_stream(),
        )
    compiled(
        to_cute(x),
        to_cute(weight),
        to_cute(mailbox.slots(parity)),
        cutlass.Int64(mailbox.multicast_address(parity)),
        cutlass.Int32(mailbox.rank),
        stream(x.device),
    )

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

"""CuTe DSL MLA value projection for NVIDIA Blackwell.

Registers the vendored Kimi-K3.1 decode value projection (``MlaValue``) for
``mla_project_value``: one launch projects the absorbed latent through ``w_vc``
and, when gated, multiplies by the sigmoid gate, instead of a batched GEMM and
``sigmoid_mul``. It takes the latent-contiguous ``w_vc`` the absorbed decode
weights use outside bitwise numerics.
"""

from __future__ import annotations

import logging

import torch
import torch.distributed as dist
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

_LATENT, _VALUE = 512, 128

logger = logging.getLogger(__name__)


def _addressable(
    attention: torch.Tensor,
    weight: torch.Tensor,
    gate: torch.Tensor | None,
    out: torch.Tensor,
    *,
    heads: int,
) -> bool:
    """Whether the kernel's vector loads and stores can address these tensors.

    ``attention`` must hold ``heads`` heads.
    """
    rows = [(attention, 8, 16)] + [(t, 2, 4) for t in (gate, out) if t is not None]
    # Row offsets stay int32: decode batches span far fewer than 2**31 elements.
    return (
        weight.stride() == (_LATENT * _VALUE, 1, _LATENT)
        and weight.data_ptr() % 16 == 0
        and all(
            t.stride(-1) == 1
            and all(s % elements == 0 for s in t.stride()[:-1])
            and t.data_ptr() % align == 0
            and t.shape[0] * t.stride(0) < 2**31
            for t, elements, align in rows
        )
        and attention.shape[1] == heads
    )


@register_kernel(
    "attention",
    "mla_project_value",
    name="cute_dsl_mla_project_value",
    solution="cute_dsl",
    capability=CapabilityRequirement(
        min_arch_version=ArchVersion(10, 0),
        max_arch_version=ArchVersion(10, 3),
        vendors=frozenset({"nvidia"}),
    ),
    signatures=frozenset(
        {
            format_signature(
                attention=dense_tensor_format(torch.bfloat16),
                weight=dense_tensor_format(torch.bfloat16),
                out=dense_tensor_format(torch.bfloat16),
            )
        }
    ),
    priority=Priority.SPECIALIZED,
    traits={
        "value_head_dim": frozenset({_VALUE}),
        "kv_lora_rank": frozenset({_LATENT}),
        "gate_kind": frozenset({"none", "sigmoid"}),
        "weight_k_major": frozenset({True}),
        "inputs_contiguous": frozenset({True, False}),
    },
)
def cute_dsl_mla_project_value(
    attention: torch.Tensor,
    weight: torch.Tensor,
    gate: torch.Tensor | None,
    out: torch.Tensor,
) -> torch.Tensor:
    """Project each head's values and apply the optional sigmoid gate.

    Computes ``out = bf16(attention @ weight)`` per head, times
    ``sigmoid(gate)`` when a gate is given.

    Args:
        attention: ``[batch, heads, 512]`` BF16 absorbed values with unit inner
            stride and 16-byte aligned rows.
        weight: ``[heads, 512, 128]`` BF16 ``w_vc`` with the latent axis
            contiguous.
        gate: ``[batch, heads * 128]`` BF16 raw gate with unit inner stride and
            4-byte aligned rows, or None.
        out: ``[batch, heads * 128]`` BF16 output with unit inner stride.

    Returns:
        ``out``.

    Raises:
        ValueError: If a stride, an alignment or a row span past int32
            offsets is one the kernel cannot address.
    """
    from tokenspeed_kernel.thirdparty.cute_dsl.k3_decode.common.host import stream
    from tokenspeed_kernel.thirdparty.cute_dsl.k3_decode.mla.value import mla_value

    if not _addressable(attention, weight, gate, out, heads=weight.shape[0]):
        raise ValueError("CuTe DSL MLA value projection cannot address these tensors")
    mla_value(attention, weight, gate, out, stream(attention.device))
    return out


class DcpValueTail:
    """A KVP group's fused decode value tail.

    Every rank's BF16 attention partials of ``heads`` local heads per rank go
    to their head owners through Lamport mailboxes, which merge them as
    ``combine_dcp_attention_partials`` does and project them as
    ``cute_dsl_mla_project_value`` does, in one launch.

    Built by :func:`create_dcp_value_tail`; holds ``2 * ranks * rows * heads``
    partial rows of 1 KiB in symmetric memory.
    """

    def __init__(self, mailbox) -> None:
        self.mailbox = mailbox

    def serves(
        self,
        partials: torch.Tensor,
        lse: torch.Tensor,
        weight: torch.Tensor,
        gate: torch.Tensor | None,
        out: torch.Tensor,
    ) -> bool:
        """Whether :meth:`__call__` serves these tensors.

        Only sizes and dtypes decide, which every rank of a call shares.
        """
        box = self.mailbox
        rows = partials.shape[0]
        return (
            partials.dtype == torch.bfloat16
            and partials.dim() == 3
            and partials.shape[1:] == (box.size * box.heads, _LATENT)
            and 0 < rows <= box.cap
            and lse.dtype == torch.float32
            and lse.shape == partials.shape[:2]
            and lse.stride(-1) == 1
            and weight.shape == (box.heads, _LATENT, _VALUE)
            and weight.dtype == torch.bfloat16
            and out.shape == (rows, box.heads * _VALUE)
            and out.dtype == torch.bfloat16
            and (gate is None or (gate.shape == out.shape and gate.dtype == out.dtype))
        )

    def __call__(
        self,
        partials: torch.Tensor,
        lse: torch.Tensor,
        weight: torch.Tensor,
        gate: torch.Tensor | None,
        out: torch.Tensor,
    ) -> torch.Tensor:
        """Merge every rank's partials of this rank's heads and project them.

        ``partials`` and their natural-log ``lse`` merge and project into
        ``out`` (``[rows, heads * 128]``), times ``sigmoid(gate)`` when given:
        the bits of the records all-to-all, ``combine_dcp_attention_partials``
        and ``cute_dsl_mla_project_value``. Every rank of the group makes the
        same calls; see :meth:`serves`.

        Raises:
            ValueError: If a stride, an alignment, or a row span past int32
                offsets is one the kernel cannot address.
        """
        from tokenspeed_kernel.thirdparty.cute_dsl.k3_decode.common.host import stream

        box = self.mailbox
        if not _addressable(partials, weight, gate, out, heads=box.size * box.heads):
            raise ValueError("the DCP value tail cannot address these tensors")
        from tokenspeed_kernel.thirdparty.cute_dsl.k3_decode.mla.dcp import dcp_value

        dcp_value(partials, lse, self.mailbox, weight, gate, out, stream(out.device))
        return out


def create_dcp_value_tail(
    group: dist.ProcessGroup,
    device: torch.device,
    *,
    heads: int,
    rows: int,
    timeout_ns: int,
    eligible: bool,
) -> DcpValueTail | None:
    """Collectively build a KVP group's :class:`DcpValueTail`.

    Every rank of the group calls this at the same program point, before graph
    capture and before the KV cache is sized; every refusal is voted on, so all
    ranks get a tail or None.

    Args:
        group: CPU (gloo) process group over the 2, 4 or 8 ranks; the CUDA
            symmetric-memory backend exchanges handles through its store.
        device: This rank's CUDA device, with its index.
        heads: Local query heads per rank, equal on every rank.
        rows: Most decode rows of one call, equal on every rank.
        timeout_ns: Bound of every in-kernel wait; a peer that does not reach
            the same call within it fails the kernel with a trap.
        eligible: The caller's own verdict for this rank, voted on with the rest.

    Returns:
        The tail, or None on every rank.
    """
    from tokenspeed_kernel.ops.communication.fabric import multicast_reachable

    degree = group.size()
    local_ok = (
        eligible
        and current_platform().is_nvidia
        and device.type == "cuda"
        and device.index is not None
        and torch.cuda.get_device_capability(device) in ((10, 0), (10, 3))
        and degree in (2, 4, 8)
        and 0 < heads
        and 0 < rows
        # The merge reads the local mailbox through int32 element offsets.
        and 2 * degree * rows * heads * _LATENT < 2**31
        and timeout_ns > 0
    )
    if local_ok:
        # A rendezvous over hosts the fabric cannot map would hang.
        local_ok = multicast_reachable(group)
    if local_ok:
        # Imported before the vote: a rank failing after it would strand its peers.
        try:
            from tokenspeed_kernel.thirdparty.cute_dsl.k3_decode.mla.dcp import (
                DcpValueMailbox,
            )
        except Exception as exc:  # noqa: BLE001 - local failure is voted on
            logger.warning(f"dcp_value: kernel module import failed: {exc!r}")
            local_ok = False
    votes = torch.tensor([int(local_ok), heads, -heads, rows, -rows], dtype=torch.int64)
    dist.all_reduce(votes, op=dist.ReduceOp.MIN, group=group)
    ok, low_heads, high_heads, low_rows, high_rows = votes.tolist()
    if ok and (low_heads != -high_heads or low_rows != -high_rows):
        raise ValueError(
            f"dcp_value: heads or rows differ across ranks "
            f"{dist.get_process_group_ranks(group)}"
        )
    if not ok:
        return None
    return DcpValueTail(
        DcpValueMailbox(group, device, heads=heads, cap=rows, timeout_ns=timeout_ns)
    )

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

from dataclasses import dataclass
from typing import Protocol

import torch
import torch.distributed as dist
from tokenspeed_kernel.selection import select_kernel
from tokenspeed_kernel.signature import FormatSignature


@dataclass(frozen=True)
class ExpertPartition:
    """Contiguous immutable weight ownership, independent of execution EP."""

    num_experts: int
    rank: int
    size: int

    def __post_init__(self) -> None:
        if self.size < 2 or not 0 <= self.rank < self.size:
            raise ValueError("Peer weights require a valid group of at least two ranks")
        if self.num_experts <= 0 or self.num_experts % self.size:
            raise ValueError(
                "Experts must divide evenly across the weight-sharing group"
            )

    @property
    def count(self) -> int:
        return self.num_experts // self.size

    @property
    def first(self) -> int:
        return self.rank * self.count


@dataclass(frozen=True)
class ExpertWeightShard:
    """A processed expert-major tensor and its explicit retention policy.

    ``replicate`` gathers small per-expert constants once at startup. Otherwise,
    the owned shard stays immutable and remote experts use two staging slots.
    """

    name: str
    tensor: torch.Tensor
    replicate: bool


@dataclass(frozen=True)
class PreparedExpertWeights:
    """Expert-major outputs of a kernel weight preprocessor.

    The layout identifies byte-compatible processed weights. Shards name the
    module parameters to rebind; non-expert scalar state remains on the module.
    """

    layout: str
    shards: tuple[ExpertWeightShard, ...]

    def bind(self, module: torch.nn.Module, views: dict[str, torch.Tensor]) -> None:
        if set(views) != {shard.name for shard in self.shards}:
            raise ValueError("Prepared weight parameters and replacement views differ")
        for name, value in views.items():
            module.get_parameter(name).data = value


class PeerWeightStore(Protocol):
    """Weight views remain stable after finalize; calls use the current stream."""

    def publish_layer(
        self, layer: int, shards: tuple[ExpertWeightShard, ...], *, layout: str
    ) -> dict[str, torch.Tensor]: ...

    def finalize(self) -> None: ...

    def weights(self, layer: int) -> dict[str, torch.Tensor]: ...

    def prefetch(self, layer: int) -> None: ...

    def acquire(self, layer: int) -> None: ...

    def release(self, layer: int) -> None: ...

    def close(self) -> None: ...

    @property
    def physical_bytes(self) -> int: ...

    @property
    def staging_bytes(self) -> int: ...


def create_peer_weight_store(
    *,
    partition: ExpertPartition,
    group: dist.ProcessGroup,
    device: torch.device,
    model_identity: str,
    pool_page_bytes: int,
    copy_implementation: str,
    solution: str,
) -> PeerWeightStore:
    """Create immutable peer storage before loading processed expert weights.

    Args:
        partition: This worker's storage ownership, not its execution EP rank.
        group: CPU process group in storage-rank order, used only for setup/close.
        device: Local device for source and staging allocations.
        model_identity: Shared checkpoint/quantization identity for validation.
        pool_page_bytes: Physical staging-page size, fixed for the store lifetime.
        copy_implementation: Explicit transfer implementation; currently ``simple``.
        solution: Registered storage implementation, currently ``cuda``.

    Returns:
        A store accepting layers in model execution order. ``publish_layer``
        returns owned loading views; ``finalize`` prepares full expert views.
        Callers must stop reading a layer before ``release`` and must finish all
        local readers before collective ``close``. Views expire at close.
    """
    from tokenspeed_kernel.ops.communication import peer_weights_cuda  # noqa: F401

    if group.size() != partition.size or group.rank() != partition.rank:
        raise ValueError("Weight ownership and process-group ranks disagree")
    kernel = select_kernel(
        "communication",
        "peer_weight_store",
        FormatSignature(()),
        traits={},
        solution=solution,
        override=None,
    )
    return kernel(
        partition=partition,
        group=group,
        device=device,
        model_identity=model_identity,
        pool_page_bytes=pool_page_bytes,
        copy_implementation=copy_implementation,
    )

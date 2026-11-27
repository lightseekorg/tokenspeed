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

"""Live rows contributed by each expert-parallel source after attention scatter."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from tokenspeed.runtime.distributed.comm_manager import scattered_moe_source_rows
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.utils.env import global_server_args_dict
from tokenspeed.runtime.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter


class MoESourceRows:
    """One fixed-address device vector shared by the model's MoE layers."""

    def __init__(self, mapping: Mapping, *, device, max_rows_per_rank: int):
        self.mapping = mapping
        self.max_rows_per_rank = max_rows_per_rank
        self.counts: tuple[int, ...] | None = None
        device_type = torch.device(device).type
        saver = TorchMemorySaverAdapter.create(
            enable=device_type == "cuda"
            and global_server_args_dict["enable_memory_saver"],
            device=device_type,
        )
        with saver.named_region("runtime_states"):
            self.source_token_counts = torch.zeros(
                len(mapping.moe.tp_ep_group), dtype=torch.int32, device=device
            )

    def upload_scattered(self, padded_tokens: int, global_num_tokens: Sequence[int]):
        """Intersect each captured TP shard with its DP group's live prefix."""
        segments = scattered_moe_source_rows(
            self.mapping,
            [padded_tokens] * self.mapping.world_size,
            global_num_tokens,
            self.mapping.moe.tp_ep_group,
        )
        self.publish(tuple(live for _, live in segments))

    def publish(self, counts: tuple[int, ...]) -> None:
        if len(counts) != self.source_token_counts.numel() or any(
            count < 0 or count > self.max_rows_per_rank for count in counts
        ):
            raise ValueError(
                f"expected one source count in [0,{self.max_rows_per_rank}] per EP rank"
            )
        # A later step can prepare its counts while the previous DMA is in flight.
        staging = torch.tensor(
            counts,
            dtype=torch.int32,
            device="cpu",
            pin_memory=self.source_token_counts.device.type != "cpu",
        )
        self.source_token_counts.copy_(staging, non_blocking=True)
        self.counts = counts

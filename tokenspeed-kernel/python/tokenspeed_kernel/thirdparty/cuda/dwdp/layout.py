# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Composite tensor geometry adapted from TensorRT-LLM's DWDP VMM layout."""

from __future__ import annotations

from dataclasses import dataclass


def align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


@dataclass(frozen=True)
class WeightLayout:
    total_bytes: int
    local_begin: int
    local_end: int
    page_begin: int
    page_end: int
    pre_bytes: int
    post_bytes: int
    granularity: int
    pool_page_bytes: int

    @classmethod
    def create(
        cls,
        *,
        expert_bytes: int,
        experts: int,
        first: int,
        count: int,
        granularity: int,
        pool_page_bytes: int,
    ) -> WeightLayout:
        if (
            expert_bytes <= 0
            or experts <= 0
            or not 0 <= first < first + count <= experts
        ):
            raise ValueError("Invalid expert byte geometry or ownership")
        if granularity <= 0 or pool_page_bytes <= 0 or pool_page_bytes % granularity:
            raise ValueError(
                "Staging pages must be a positive multiple of VMM granularity"
            )
        begin, end = first * expert_bytes, (first + count) * expert_bytes
        page_begin = begin // granularity * granularity
        page_end = align_up(end, granularity)
        total = experts * expert_bytes
        return cls(
            total,
            begin,
            end,
            page_begin,
            page_end,
            align_up(page_begin, pool_page_bytes),
            align_up(align_up(total, granularity) - page_end, pool_page_bytes),
            granularity,
            pool_page_bytes,
        )

    @property
    def source_bytes(self) -> int:
        return self.page_end - self.page_begin

    @property
    def source_offset(self) -> int:
        return self.local_begin - self.page_begin

    @property
    def view_offset(self) -> int:
        return self.pre_bytes - self.page_begin

    @property
    def virtual_bytes(self) -> int:
        return self.pre_bytes + self.source_bytes + self.post_bytes

    @property
    def staging_pages(self) -> int:
        return (self.pre_bytes + self.post_bytes) // self.pool_page_bytes

    def remote_intervals(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (a, b)
            for a, b in ((0, self.page_begin), (self.page_end, self.total_bytes))
            if a < b
        )

    def edge_intervals(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (a, b)
            for a, b in (
                (self.page_begin, self.local_begin),
                (self.local_end, min(self.page_end, self.total_bytes)),
            )
            if a < b
        )

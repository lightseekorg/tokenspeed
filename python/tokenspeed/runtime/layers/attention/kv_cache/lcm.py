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

"""Physical storage for one LCM cache plan."""

from __future__ import annotations

import torch
from tokenspeed_kernel.ops.kvcache.triton import zero_byte_segments

from tokenspeed.runtime.configs.lcm_memory_plan import LcmMemoryPlan


class LcmCachePool:
    """Own the shared backing and typed field views of one LCM plan."""

    def __init__(self, plan: LcmMemoryPlan, device: str):
        self.plan = plan
        self.backing = torch.zeros(plan.arena_bytes, dtype=torch.uint8, device=device)
        self._fields: dict[str, torch.Tensor] = {}

    def field(self, field_id: str, dtype: torch.dtype) -> torch.Tensor:
        view = self._fields.get(field_id)
        if view is not None:
            if view.dtype != dtype:
                raise ValueError(
                    f"LCM field {field_id!r} is already bound as {view.dtype}"
                )
            return view
        try:
            field = self.plan.field(field_id)
        except KeyError as exc:
            raise ValueError(f"LCM field {field_id!r} is not planned") from exc
        if torch.empty((), dtype=dtype).element_size() != field.element_size:
            raise ValueError(f"field {field_id!r}: dtype itemsize does not match plan")
        group = self.plan.group(field.group_id)
        typed = self.backing.view(dtype)
        element_strides = []
        stride = 1
        for extent in reversed(field.shape):
            element_strides.append(stride)
            stride *= extent
        view = typed.as_strided(
            (group.page_count, *field.shape),
            (
                field.page_stride_bytes // field.element_size,
                *reversed(element_strides),
            ),
            self._field_page_byte_offset(field_id, 0) // field.element_size,
        )
        self._fields[field_id] = view
        return view

    def zero_pages(self, page_ids_by_group: dict[str, list[int]]) -> None:
        segments = [
            segment
            for group_id, page_ids in page_ids_by_group.items()
            for segment in self._page_byte_segments(group_id, page_ids)
        ]
        if segments:
            zero_byte_segments(self.backing, segments)

    def pd_contract(self, group_specs):
        from tokenspeed.runtime.pd.cache_protocol import build_lcm_pd_cache_contract

        missing = [
            field.field_id
            for field in self.plan.fields
            if field.field_id not in self._fields
        ]
        if missing:
            raise RuntimeError(f"LCM PD fields have no runtime dtype: {missing}")
        field_dtypes = {
            field_id: str(view.dtype).removeprefix("torch.")
            for field_id, view in self._fields.items()
        }
        return build_lcm_pd_cache_contract(
            plan=self.plan,
            backing=self.backing,
            group_specs=group_specs,
            field_dtypes=field_dtypes,
        )

    def _field_page_byte_offset(self, field_id: str, page_id: int) -> int:
        field = self.plan.field(field_id)
        group = self.plan.group(field.group_id)
        if page_id < 0 or page_id >= group.page_count:
            raise IndexError(
                f"page_id {page_id} outside [0, {group.page_count}) for "
                f"group {group.group_id!r}"
            )
        plane = self.plan.plane(field.plane_id)
        return (
            plane.arena_offset_bytes
            + plane.bytes_per_lcm_block
            - field.page_stride_bytes
            + page_id * field.page_stride_bytes
            + field.field_offset_bytes
        )

    def _page_byte_segments(
        self, group_id: str, page_ids: list[int]
    ) -> list[tuple[int, int]]:
        self.plan.group(group_id)
        fields = [field for field in self.plan.fields if field.group_id == group_id]
        return [
            (
                self._field_page_byte_offset(field.field_id, page_id),
                field.payload_bytes,
            )
            for page_id in page_ids
            for field in fields
        ]

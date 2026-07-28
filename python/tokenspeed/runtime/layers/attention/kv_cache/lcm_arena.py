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

"""Runtime backing allocation and tensor views for an LCM memory plan."""

from __future__ import annotations

from tokenspeed.runtime.configs.lcm_memory_plan import LcmMemoryPlan


class LcmArena:
    """One fixed backing allocation interpreted through planned field views."""

    def __init__(self, plan: LcmMemoryPlan, backing):
        if int(backing.nbytes) != plan.arena_bytes:
            raise ValueError(
                f"backing has {int(backing.nbytes)} bytes; plan requires "
                f"{plan.arena_bytes}"
            )
        self.plan = plan
        self.backing = backing

    @classmethod
    def allocate(cls, plan: LcmMemoryPlan, device: str):
        """Allocate the arena once; all cache tensors are views of it."""
        import torch

        backing = torch.zeros(plan.arena_bytes, dtype=torch.uint8, device=device)
        return cls(plan, backing)

    def field_page_byte_offset(self, field_id: str, page_id: int) -> int:
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

    def page_byte_segments(self, group_id: str, page_ids) -> list[tuple[int, int]]:
        """Return backing byte ranges occupied by the group's selected pages."""
        self.plan.group(group_id)
        fields = [field for field in self.plan.fields if field.group_id == group_id]
        return [
            (
                self.field_page_byte_offset(field.field_id, page_id),
                field.payload_bytes,
            )
            for page_id in page_ids
            for field in fields
        ]

    def field_view(self, field_id: str, dtype):
        """Return a stable typed tensor view for one planned field."""
        import torch

        field = self.plan.field(field_id)
        if torch.empty((), dtype=dtype).element_size() != field.element_size:
            raise ValueError(f"field {field_id!r}: dtype itemsize does not match plan")
        group = self.plan.group(field.group_id)
        base_bytes = self.field_page_byte_offset(field_id, 0)
        typed = self.backing.view(dtype)
        element_strides = []
        stride = 1
        for extent in reversed(field.shape):
            element_strides.append(stride)
            stride *= extent
        return typed.as_strided(
            (group.page_count, *field.shape),
            (
                field.page_stride_bytes // field.element_size,
                *reversed(element_strides),
            ),
            base_bytes // field.element_size,
        )

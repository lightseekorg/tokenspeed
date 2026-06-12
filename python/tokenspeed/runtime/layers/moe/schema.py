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

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ExpertCheckpointSchema:
    gate_proj_name: str = "gate_proj"
    up_proj_name: str = "up_proj"
    down_proj_name: str = "down_proj"
    gate_up_fused_name: str | None = None
    extra_names: dict[str, str] = field(default_factory=dict)

    def get_semantic_name(self, semantic: str) -> str:
        if semantic == "gate_proj":
            return self.gate_proj_name
        if semantic == "up_proj":
            return self.up_proj_name
        if semantic == "down_proj":
            return self.down_proj_name
        if semantic == "gate_up_fused":
            if self.gate_up_fused_name is None:
                raise KeyError("gate_up_fused_name is not defined")
            return self.gate_up_fused_name
        if semantic in self.extra_names:
            return self.extra_names[semantic]
        raise KeyError(f"Unknown MoE checkpoint semantic: {semantic}")

    def make_expert_weight_name(self, expert_id: int, semantic: str) -> str:
        return f"experts.{expert_id}.{self.get_semantic_name(semantic)}."

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
from enum import Enum, auto

from tokenspeed.runtime.models.base.placement import Placement


class ModuleKind(Enum):
    NORM = auto()
    ATTENTION = auto()
    DENSE_MLP = auto()
    MOE = auto()
    GENERIC = auto()


class CallConvention(Enum):
    NORM_WITH_OPTIONAL_RESIDUAL = auto()
    ATTENTION = auto()
    MOE = auto()
    HIDDEN_STATES_ONLY = auto()


class FusionCapability(Enum):
    NONE = auto()
    REDUCE_NORM = auto()


@dataclass(frozen=True, slots=True)
class ModuleSpec:
    input_placement: Placement | None = None
    output_placement: Placement | None = None
    kind: ModuleKind = ModuleKind.GENERIC
    call: CallConvention = CallConvention.HIDDEN_STATES_ONLY
    fusion: FusionCapability = FusionCapability.NONE
    captures_aux: bool = False
    skip_on_idle: bool = False

    @classmethod
    def from_kind(
        cls,
        *,
        input_placement: Placement | None = None,
        output_placement: Placement | None = None,
        kind: ModuleKind = ModuleKind.GENERIC,
        supports_fused_reduce_norm: bool = False,
        captures_aux: bool = False,
        skip_on_idle: bool = False,
    ) -> ModuleSpec:
        if kind == ModuleKind.NORM:
            call = CallConvention.NORM_WITH_OPTIONAL_RESIDUAL
        elif kind == ModuleKind.ATTENTION:
            call = CallConvention.ATTENTION
        elif kind == ModuleKind.MOE:
            call = CallConvention.MOE
        else:
            call = CallConvention.HIDDEN_STATES_ONLY

        fusion = (
            FusionCapability.REDUCE_NORM
            if supports_fused_reduce_norm
            else FusionCapability.NONE
        )

        return cls(
            input_placement=input_placement,
            output_placement=output_placement,
            kind=kind,
            call=call,
            fusion=fusion,
            captures_aux=captures_aux,
            skip_on_idle=skip_on_idle,
        )

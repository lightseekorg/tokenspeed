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

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt

from tokenspeed.runtime.pd.utils import (
    DisaggregationMode,
    PageTransferMetadata,
)
from tokenspeed.runtime.utils.server_args import ServerArgs


@dataclass
class KVArgs:
    engine_rank: int
    kv_data_ptrs: list[int]
    kv_data_lens: list[int]
    kv_item_lens: list[int]
    offsets: List[Tuple[int]]
    aux_data_ptrs: list[int]
    aux_data_lens: list[int]
    aux_item_lens: list[int]
    ib_device: str
    gpu_id: int
    target_layer_num: int
    draft_layer_num: int
    state_data_ptrs: List[int] = field(default_factory=list)
    state_data_lens: List[int] = field(default_factory=list)
    state_item_lens: List[int] = field(default_factory=list)
    state_type: str = "none"
    mamba_offsets: List[int] | None = None


class KVPoll:
    Failed = 0
    Bootstrapping = 1
    Bootstrapped = 2
    WaitingForInput = 3
    Transferring = 4
    Success = 5

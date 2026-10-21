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

from dataclasses import dataclass

from tokenspeed.runtime.pd.cache_protocol import (
    CacheTransferContract,
)
from tokenspeed.runtime.utils import (
    get_colorful_logger,
)

logger = get_colorful_logger(__name__)


@dataclass
class PrefillParallelInfo:
    tp_size: int
    dp_size: int
    cache_fields_by_stage: tuple[tuple[str, ...], ...]
    cache_layout: CacheTransferContract | None = None
    # Prefill chunk-pipeline stage count; each stage sends only its own
    # layers' KV, so Decode plans per stage and unions the routes.
    pp_size: int = 1

    @property
    def prefill_tp_size_per_dp_rank(self):
        # Intra-stage TP width: the registered world spans pp stages.
        return self.tp_size // (self.dp_size * self.pp_size)

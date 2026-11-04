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

"""gfx950 Gluon fused decoder layer registration.

The leaf's modules load on first use, so importing the op family needs no
device (its plan checker runs on CPU).
"""

from __future__ import annotations

from tokenspeed_kernel.ops.monokernel import DECODER_LAYER_SIGNATURE
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel


@register_kernel(
    "decoder_layer",
    "prepare",
    name="gluon_decoder_layer_muse_spark_gfx950",
    solution="gluon",
    capability=CapabilityRequirement(
        min_arch_version=ArchVersion(9, 5),
        max_arch_version=ArchVersion(9, 5),
        vendors=frozenset({"amd"}),
    ),
    signatures={DECODER_LAYER_SIGNATURE},
    traits={
        "recipe": frozenset({"muse_spark"}),
        "verify_width": frozenset({8}),
        "tp_size": frozenset({8}),
        "ep_size": frozenset({1}),
        "kv_cache_dtype": frozenset({"fp8_e4m3"}),
    },
    priority=Priority.SPECIALIZED,
)
class GluonDecoderLayerMuseSparkGfx950:
    """The Muse Spark fused decoder layer on gfx950 (EP1, TP 8, DFlash
    verify width 8, FP8 KV cache); see ``_gluon/muse_spark/prepared.py``."""

    @staticmethod
    def prepare_state(**kwargs):
        from tokenspeed_kernel.ops.monokernel._gluon.muse_spark.prepared import (
            MuseSparkDecoderLayerState,
        )

        return MuseSparkDecoderLayerState(**kwargs)

    @staticmethod
    def heap_bytes(*, world_size: int, max_requests: int) -> int:
        from tokenspeed_kernel.ops.monokernel._gluon.muse_spark.prepared import (
            MuseSparkDecoderLayerState,
        )

        return MuseSparkDecoderLayerState.heap_bytes(
            world_size=world_size, max_requests=max_requests
        )

    def __init__(self, *, plan_keys, weights, state) -> None:
        from tokenspeed_kernel.ops.monokernel._gluon.muse_spark.prepared import (
            DEADLINE_TICKS,
            DEFAULT_PLAN_FIELDS,
            MuseSparkFusedDecoderLayer,
        )

        # Serving: the default plan fields, no stamps, the serving deadline.
        self.layer = MuseSparkFusedDecoderLayer(
            plan_keys=plan_keys,
            weights=weights,
            state=state,
            plan_fields=DEFAULT_PLAN_FIELDS,
            stamps=False,
            deadline_ticks=DEADLINE_TICKS,
        )

    def decline_reasons(self, requests: int) -> list[str]:
        return self.layer.decline_reasons(requests)

    def forward(self, prepared_query, residual, *, positions, cache, next_prologue):
        return self.layer.forward(
            prepared_query,
            residual,
            positions=positions,
            cache=cache,
            next_prologue=next_prologue,
        )

    def forward_last(self, prepared_query, residual, *, positions, cache, next_norm):
        return self.layer.forward_last(
            prepared_query,
            residual,
            positions=positions,
            cache=cache,
            next_norm=next_norm,
        )

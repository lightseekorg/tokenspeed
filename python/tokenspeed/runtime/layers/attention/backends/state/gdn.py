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

"""Gated DeltaNet layers (the Qwen3.5 family) on the shared recurrent-state backend."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tokenspeed_kernel.ops.attention.gdn import (
    gdn_chunk_prefill_capturable,
    gdn_chunk_prefill_stores_state_checkpoints,
    gdn_decode_supports,
)
from tokenspeed_kernel.ops.attention.gdn.triton import pack_gdn_projection

from tokenspeed.runtime.layers.attention.backends.state.prefill_capacity import (
    CapacityPrefillBackend,
)
from tokenspeed.runtime.layers.attention.configs.linear_attn import LinearAttnConfig

if TYPE_CHECKING:
    import torch

    from tokenspeed.runtime.layers.attention.configs.base import (
        AttnConfig,
        SoftmaxAttnConfig,
    )
    from tokenspeed.runtime.layers.attention.kv_cache.base import CachePool


class GdnAttnBackend(CapacityPrefillBackend):
    """GDN layers, captured inline by the prefill graph where the scan allows it.

    The shared base already runs GDN; this class only admits capacity-shaped
    prefills, which needs a chunk-prefill kernel that reads its sequence bounds
    on device (see ``gdn_chunk_prefill_capturable``).
    """

    # The scan reads its bounds on device, so packing would only add work to eager forwards.
    _capacity_layout_when_uncaptured = False
    _verify_scan_writes_replay = True
    _verify_scan_runs_chain_form = True

    def __init__(self, config: AttnConfig, spec: SoftmaxAttnConfig) -> None:
        super().__init__(config, spec)
        linear_attn = config.component(LinearAttnConfig)
        self._scan_geometry: dict[str, int | bool] = {
            "head_dim": linear_attn.head_k_dim,
            "value_head_dim": linear_attn.head_v_dim,
            "num_q_heads": linear_attn.num_k_heads // linear_attn.tp_size,
            "num_v_heads": linear_attn.num_v_heads // linear_attn.tp_size,
            "qk_l2norm": True,
        }
        self._prefill_capturable = gdn_chunk_prefill_capturable(
            config.dtype, **self._scan_geometry
        )

        self._head_k_dim = linear_attn.head_k_dim
        self._head_v_dim = linear_attn.head_v_dim
        # Resolved when a pool binds: the pool's prefix granularity spaces the
        # in-forward checkpoints, and its state dtype picks the verify kernel.
        self._stores_in_forward_state_checkpoints = False
        self._verify_reads_committed_recurrent_state = False
        self._verify_packed_qkv_views = False
        self._decode_packed_qkv_views = False

    def _publish_cache_pool(self, cache_pool: CachePool) -> None:
        super()._publish_cache_pool(cache_pool)
        # In-forward checkpoints sit at prefix-page multiples of a chunk start,
        # so the scan must store at that granularity.
        contract = cache_pool.arena.runtime_contract
        self._stores_in_forward_state_checkpoints = (
            gdn_chunk_prefill_stores_state_checkpoints(
                self.dtype,
                **self._scan_geometry,
                interval_granularity=contract.prefix_granularity,
            )
        )
        # Where the verify kernel takes a separate destination pool, verify
        # reads the committed state in place instead of seeding the scratch
        # with a copy of it. Where a decode or verify kernel takes strided
        # q/k/v (and a/b), it reads the projection views as handed over
        # instead of a packed copy.
        ssm_dtype = self._state_components(self._state_layer_ids()[0])[1].dtype

        def supports(op: str, trait: str) -> bool:
            return gdn_decode_supports(
                op,
                self.dtype,
                head_dim=self._head_k_dim,
                value_head_dim=self._head_v_dim,
                state_dtype=ssm_dtype,
                trait=trait,
                solution=None,
            )

        self._verify_reads_committed_recurrent_state = not self.replay_ssm and supports(
            "gdn_decode_mtp", "separate_output_state"
        )
        self._verify_packed_qkv_views = supports("gdn_decode_mtp", "strided_qkv")
        self._decode_packed_qkv_views = supports("gdn_decode_step", "strided_qkv")

    def _admits_capacity_prefill(self) -> bool:
        return self._prefill_capturable

    def _pack_projection(
        self,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor | None,
        b: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        # Projection views to the dense, aligned rows FlashInfer's kernels need;
        # skipped where the selected kernels read the views by stride.
        return pack_gdn_projection(mixed_qkv, a, b)

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

import torch
from tokenspeed_kernel.ops.attention.gdn import (
    gdn_chunk_prefill_capturable,
    gdn_chunk_prefill_stores_state_checkpoints,
    gdn_decode_mtp,
    gdn_decode_mtp_warm_batch_sizes,
    gdn_decode_supports,
    gdn_tree_verify_needs_node_states,
)
from tokenspeed_kernel.ops.attention.gdn.triton import pack_gdn_projection

from tokenspeed.runtime.layers.attention.backends.state.mamba import (
    gdn_decode_solution,
)
from tokenspeed.runtime.layers.attention.backends.state.prefill_capacity import (
    CapacityPrefillBackend,
)
from tokenspeed.runtime.layers.attention.configs.linear_attn import LinearAttnConfig

if TYPE_CHECKING:
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

    def __init__(self, config: AttnConfig, spec: SoftmaxAttnConfig) -> None:
        super().__init__(config, spec)
        linear_attn = config.component(LinearAttnConfig)
        scan_geometry = {
            "head_dim": linear_attn.head_k_dim,
            "value_head_dim": linear_attn.head_v_dim,
            "num_q_heads": linear_attn.num_k_heads // linear_attn.tp_size,
            "num_v_heads": linear_attn.num_v_heads // linear_attn.tp_size,
            "qk_l2norm": True,
        }
        self._prefill_capturable = gdn_chunk_prefill_capturable(
            config.dtype, **scan_geometry
        )
        # In-forward checkpoints sit at prefix-page multiples of a chunk start,
        # so the scan must store at that granularity.
        self._stores_in_forward_state_checkpoints = (
            gdn_chunk_prefill_stores_state_checkpoints(
                config.dtype,
                **scan_geometry,
                interval_granularity=config.prefix_granularity,
            )
        )

        self._head_k_dim = linear_attn.head_k_dim
        self._head_v_dim = linear_attn.head_v_dim
        # Resolved when a pool binds (its state dtype picks the verify kernel).
        self._verify_reads_committed_recurrent_state = False
        self._verify_packed_qkv_views = False
        self._decode_packed_qkv_views = False

    def _publish_cache_pool(self, cache_pool: CachePool) -> None:
        super()._publish_cache_pool(cache_pool)
        # Where the verify kernel takes a separate destination pool, verify
        # reads the committed state in place instead of seeding the scratch
        # with a copy of it. Where a decode or verify kernel takes strided
        # q/k/v (and a/b), it reads the projection views as handed over
        # instead of a packed copy.
        ssm_dtype = self._state_components(self._state_layer_ids()[0])[1].dtype
        decode_solution = gdn_decode_solution(ssm_dtype, disable_state_update=False)
        verify_solution = gdn_decode_solution(
            ssm_dtype, disable_state_update=self.replay_ssm
        )
        decode_traits = {"head_dim": self._head_k_dim}
        verify_traits = {
            **decode_traits,
            "value_head_dim": self._head_v_dim,
            "verify_tokens": self.speculative_num_draft_tokens,
            "tree": self.draft_tree,
            "output_only": self.replay_ssm
            and not (
                self.draft_tree
                and gdn_tree_verify_needs_node_states(self.speculative_num_draft_tokens)
            ),
        }

        self._verify_reads_committed_recurrent_state = (
            not self.replay_ssm
            and gdn_decode_supports(
                "gdn_decode_mtp",
                self.dtype,
                state_dtype=ssm_dtype,
                trait="separate_output_state",
                traits=verify_traits,
                solution=verify_solution,
            )
        )
        self._verify_packed_qkv_views = gdn_decode_supports(
            "gdn_decode_mtp",
            self.dtype,
            state_dtype=ssm_dtype,
            trait="strided_qkv",
            traits=verify_traits,
            solution=verify_solution,
        )
        self._decode_packed_qkv_views = gdn_decode_supports(
            "gdn_decode_step",
            self.dtype,
            state_dtype=ssm_dtype,
            trait="strided_qkv",
            traits=decode_traits,
            solution=decode_solution,
        )

    def _warm_verify(self, max_bs: int, width: int) -> None:
        """Warm the selected read-only GDN verify using all-padding launches."""
        if not self.replay_ssm or self.draft_tree:
            return
        num_k_heads, num_v_heads, head_k_dim, head_v_dim = self._gdn_replay.geometry
        _, ssm = self._state_components(self._state_layer_ids()[0])
        sizes = gdn_decode_mtp_warm_batch_sizes(max_bs, num_v_heads, width, ssm.dtype)
        if not sizes:
            return
        rows = sizes[-1] * width
        # Dense rows, as the split writes q/k/v (the gates' layout is not specialized).
        query = torch.zeros(
            (rows, num_k_heads, head_k_dim), dtype=self.dtype, device=self.device
        )
        value = torch.zeros(
            (rows, num_v_heads, head_v_dim), dtype=self.dtype, device=self.device
        )
        gates = torch.zeros((rows, num_v_heads), dtype=self.dtype, device=self.device)
        parameters = torch.zeros(num_v_heads, dtype=torch.float32, device=self.device)
        padding = torch.full(
            (sizes[-1],), self.pad_slot_id, dtype=torch.int32, device=self.device
        )
        for bs in sizes:
            tokens = bs * width
            q = query[:tokens].view(bs, width, num_k_heads, head_k_dim)
            gdn_decode_mtp(
                q,
                q,
                value[:tokens].view(bs, width, num_v_heads, head_v_dim),
                A_log=parameters,
                a=gates[:tokens].view(bs, width, num_v_heads),
                dt_bias=parameters,
                b=gates[:tokens].view(bs, width, num_v_heads),
                initial_state=ssm,
                initial_state_indices=padding[:bs],
                scale=None,
                disable_state_update=True,
                use_qk_l2norm=True,
                tree_ancestors=None,
                solution=gdn_decode_solution(ssm.dtype, disable_state_update=True),
            )

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

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
    GdnPendingReplay,
    gdn_chunk_prefill_capturable,
    gdn_chunk_prefill_stores_state_checkpoints,
    gdn_decode_supports,
    gdn_pending_replay_supported,
    gdn_tree_verify_needs_node_states,
    gdn_verify_fuses_output_norm,
)
from tokenspeed_kernel.ops.attention.gdn.triton import pack_gdn_projection
from tokenspeed_kernel.ops.conv.replay import BankedConvReplayPayload

from tokenspeed.runtime.execution.request_slots import RequestSlotLayout
from tokenspeed.runtime.layers.attention.backends.state.prefill_capacity import (
    CapacityPrefillBackend,
)
from tokenspeed.runtime.layers.attention.configs.linear_attn import LinearAttnConfig

if TYPE_CHECKING:
    from tokenspeed.runtime.layers.attention.backends.state.gdn_pending import (
        GdnPendingState,
    )
    from tokenspeed.runtime.layers.attention.backends.state.mamba import (
        MambaForwardMetadata,
    )
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

    Args:
        config: The model's attention configuration.
        spec: The softmax component of ``config``.
        writes_state_checkpoints_in_forward: Whether the scheduler may give
            extend rows in-forward State checkpoints
            (``--enable-state-checkpoint-in-forward``). Captured scans then
            store them through fixed checkpoint slots, so such forwards run
            inline too.
    """

    # The scan reads its bounds on device, so packing would only add work to eager forwards.
    _capacity_layout_when_uncaptured = False
    _verify_scan_writes_replay = True
    _verify_conv_writes_rows = True

    def __init__(
        self,
        config: AttnConfig,
        spec: SoftmaxAttnConfig,
        *,
        writes_state_checkpoints_in_forward: bool,
        defer_recurrent_state: bool,
    ) -> None:
        super().__init__(config, spec)
        linear_attn = config.component(LinearAttnConfig)
        self._scan_geometry: dict[str, int | bool] = {
            "head_dim": linear_attn.head_k_dim,
            "value_head_dim": linear_attn.head_v_dim,
            "num_q_heads": linear_attn.num_k_heads // linear_attn.tp_size,
            "num_v_heads": linear_attn.num_v_heads // linear_attn.tp_size,
            "qk_l2norm": True,
        }
        # A captured scan of a server writing in-forward checkpoints stores
        # them, so the kernel that does must read its bounds on device too.
        self._prefill_capturable = gdn_chunk_prefill_capturable(
            config.dtype,
            **self._scan_geometry,
            state_checkpoints=writes_state_checkpoints_in_forward,
        )
        self._writes_state_checkpoints_in_forward = writes_state_checkpoints_in_forward

        self._defer_recurrent_state = defer_recurrent_state
        self._pending_state: GdnPendingState | None = None
        self._pending_slot_count = RequestSlotLayout(
            config.max_resident_requests
        ).capacity
        self._head_k_dim = linear_attn.head_k_dim
        self._head_v_dim = linear_attn.head_v_dim
        # Resolved when a pool binds: the pool's prefix granularity spaces the
        # in-forward checkpoints (and so the interval a captured scan stores
        # them at), and its state dtype picks the verify kernel.
        self._stores_in_forward_state_checkpoints = False
        self._state_checkpoint_interval: int | None = None
        self._verify_reads_committed_recurrent_state = False
        self._verify_packed_qkv_views = False
        self._decode_packed_qkv_views = False

    def _publish_cache_pool(self, cache_pool: CachePool) -> None:
        if self._pending_state is not None:
            self._pending_state.materialize(None)
            self._pending_state = None
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
        # Every checkpoint offset is a multiple of that granularity, so a
        # captured scan storing a state at each prefix page covers whichever
        # checkpoints a forward carries.
        self._state_checkpoint_interval = (
            contract.prefix_granularity
            if self._writes_state_checkpoints_in_forward
            and self._stores_in_forward_state_checkpoints
            else None
        )
        # Where the verify kernel takes a separate destination pool, verify
        # reads the committed state in place instead of seeding the scratch
        # with a copy of it. Where a decode or verify kernel takes strided
        # q/k/v (and a/b), it reads the projection views as handed over
        # instead of a packed copy.
        ssm_state = self._state_components(self._state_layer_ids()[0])[1]
        ssm_dtype = ssm_state.dtype
        decode_traits = {
            "head_dim": self._head_k_dim,
            "value_head_dim": self._head_v_dim,
            "strided_state": not ssm_state.is_contiguous(),
        }
        verify_traits = {
            **decode_traits,
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
                solution=None,
            )
        )
        self._verify_packed_qkv_views = gdn_decode_supports(
            "gdn_decode_mtp",
            self.dtype,
            state_dtype=ssm_dtype,
            trait="strided_qkv",
            traits=verify_traits,
            solution=None,
        )
        self._decode_packed_qkv_views = gdn_decode_supports(
            "gdn_decode_step",
            self.dtype,
            state_dtype=ssm_dtype,
            trait="strided_qkv",
            traits=decode_traits,
            solution=None,
        )
        # A verify kernel applying the gated output norm saves its launch.
        self._verify_fuses_output_norm = gdn_verify_fuses_output_norm(
            self.dtype, state_dtype=ssm_dtype, traits=verify_traits, solution=None
        )

    def _admits_capacity_prefill(self) -> bool:
        return self._prefill_capturable

    def _capacity_checkpoint_interval(self) -> int | None:
        return self._state_checkpoint_interval

    def _pack_projection(
        self,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor | None,
        b: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        # Single-token decode: projection views to the dense, aligned rows
        # FlashInfer's kernel needs; skipped where the selected kernel reads the
        # views by stride. A chain verify's conv update writes dense rows instead.
        return pack_gdn_projection(mixed_qkv, a, b)

    def _ensure_verify_scratch(self, bs: int, draft_token_num: int) -> None:
        super()._ensure_verify_scratch(bs, draft_token_num)
        replay = self._gdn_replay
        if (
            not self._defer_recurrent_state
            or replay is None
            or self.draft_tree
            or self.is_draft
            or draft_token_num != 4
            or not gdn_pending_replay_supported(
                self.dtype, geometry=replay.geometry, state_dtype=replay.state_dtype
            )
        ):
            return
        if self._pending_state is not None:
            if self._pending_state.capacity < bs:
                raise RuntimeError(
                    "deferred GDN workspace must be sized before capture"
                )
            return
        from tokenspeed.runtime.layers.attention.backends.state.gdn_pending import (
            GdnPendingState,
        )

        tables = self._verify_copy_tables_get()
        self._pending_state = GdnPendingState(
            replay.payload,
            replay.parameters,
            slot_count=self._pending_slot_count,
            geometry=replay.geometry,
            state_addresses=tables["ssm_comp"],
            state_row_strides=tables["ssm_element_stride"],
            rows_aligned=tables["ssm_rows_aligned"],
            group_indices=tables["group_sel"],
        )
        replay.payload = self._pending_state.payload[0]
        self._pending_state.warmup()

    def preallocate_verify_workspace(self, max_bs: int, draft_token_num: int) -> int:
        size = super().preallocate_verify_workspace(max_bs, draft_token_num)
        if self._pending_state is not None:
            size += self._pending_state.nbytes - self._gdn_replay.payload.nbytes
        return size

    def _deferred_verify_metadata(
        self, *, allow_defer: bool
    ) -> MambaForwardMetadata | None:
        metadata = self.forward_metadata
        if (
            allow_defer
            and metadata is not None
            and metadata.mamba_output_indices is not None
        ):
            return metadata
        return None

    def prepare_pending_recurrent_state(
        self, slots: torch.Tensor, frontier: torch.Tensor, *, allow_defer: bool
    ) -> None:
        pending = self._pending_state
        if pending is None:
            return
        metadata = self._deferred_verify_metadata(allow_defer=allow_defer)
        reads = None
        if metadata is not None:
            reads = torch.stack(
                [
                    metadata.state_in_blocks_by_group[group][: slots.numel()]
                    for group in self._state_groups()
                ]
            )
        pending.prepare(slots, frontier, enabled=metadata is not None, reads=reads)

    def prepare_recurrent_state_acceptance(
        self, stopped: torch.Tensor, frontier: torch.Tensor, *, allow_defer: bool
    ) -> None:
        pending = self._pending_state
        if pending is not None:
            # Target preparation is captured, while acceptance can run outside
            # the graph. Refresh its host choice after an eager prefill/mixed
            # forward, which may have disabled deferral since capture.
            pending.enabled = (
                self._deferred_verify_metadata(allow_defer=allow_defer) is not None
            )
            if pending.enabled:
                pending.prepare_acceptance(stopped, frontier)

    def materialize_recurrent_state(self, slots: torch.Tensor | None) -> None:
        if self._pending_state is not None:
            self._pending_state.materialize(slots)

    def discard_recurrent_state(self, slots: torch.Tensor) -> None:
        if self._pending_state is not None:
            self._pending_state.discard(slots)

    def _replay_inputs(self, layer_id, seq_len, a, b, *, A_log, dt_bias):
        payload, av, bv = super()._replay_inputs(
            layer_id, seq_len, a, b, A_log=A_log, dt_bias=dt_bias
        )
        pending = self._pending_state
        if pending is not None and pending.enabled:
            return BankedConvReplayPayload(
                payload, av, bv, pending.bank, pending.payload.stride(0)
            )
        return payload, av, bv

    def _pending_replay_descriptor(
        self, layer_id: int, batch: int
    ) -> GdnPendingReplay | None:
        pending = self._pending_state
        if pending is not None and pending.enabled:
            return pending.descriptor(self._gdn_replay.layer_ids.index(layer_id), batch)
        return None

    def _replay_commit(self, payload, parameters, **tables) -> None:
        pending = self._pending_state
        if pending is not None and pending.enabled:
            pending.accept(
                tables["accepted_length"],
                read_indices=tables["read_indices"],
                write_indices=tables["write_indices"],
            )
        else:
            super()._replay_commit(payload, parameters, **tables)

    def slot_state_exporters(self):
        owners = super().slot_state_exporters()
        return owners if self._pending_state is None else (*owners, self._pending_state)

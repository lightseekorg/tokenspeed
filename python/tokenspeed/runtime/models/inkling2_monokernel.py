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

"""The decode monokernel seam of an Inkling2 target (``docs/design/monokernel.md``).

With ``--decode-monokernel``, the target's layers (the mid local ones
``6 k + 1 .. 6 k + 4``, the first of each attention-residual block, ``6 k``,
layer 0 included, and the full attention ones, ``6 k + 5``, with NVFP4 or
MXFP8 experts) run a verify step of 1, 2, 4 or 8
requests (``BATCHES``: the decode graphs' batch sizes up to 8; a graph batch
past the live requests pads them) as one persistent launch a rank: ``8 bs``
rows, each request's verify block of 8 after the previous one's. Every
other layer, and every other forward, runs the regular code. A launch takes the regular
layer's boundary (the residual state's pending terms and the set partials)
and returns it, so launched and regular layers follow each other in one
forward.

After loading, :meth:`Inkling2Monokernel.pack` packs the layers;
``prepare_communication_runtime`` allocates the launches' mailbox on the
attention TP group's Iris heap (:meth:`Inkling2Monokernel.prepare`).
"""

from __future__ import annotations

import torch
from tokenspeed_kernel.ops.communication.iris import (
    create_iris_heap,
    iris_heap_bytes,
    iris_symmetric_heap,
    reserve_iris_heap,
)
from tokenspeed_kernel.ops.model.inkling2.monokernel import (
    LocalLayer,
    inkling2_full_layer_splits,
    inkling2_local_layer_decode,
    prepare_inkling2_full_layer,
    prepare_inkling2_local_layer,
)

from tokenspeed.runtime.distributed.process_group_manager import (
    process_group_manager as pg_manager,
)
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.layers.attention_residual import AttentionResidualState
from tokenspeed.runtime.models.monokernel import MonokernelState

# One request's verify block: 8 rows (DSPARK's 7 draft tokens and the token
# they extend), chained.
CHAIN = 8
# The verify batch sizes the launches serve, and their rows. Larger batches
# keep the regular path.
BATCHES = (1, 2, 4, 8)
ROWS = tuple(CHAIN * bs for bs in BATCHES)
ATTENTION_PAGE = 128  # KV page (the regular decode's KV block) of bit identity
# Query rows of a block set's partials: the 6 layers' MLP queries and the
# attention queries of the layers after them (one local, three full).
SET_QUERIES = 14


def launch_requests(ctx: ForwardContext, spec_tokens: int) -> int | None:
    """The requests of a forward that the launches serve (a decode verify
    batch of ``BATCHES``, ``spec_tokens`` = ``CHAIN`` tokens a request), or
    None. It depends only on the forward, so every rank decides the same."""
    if ctx.forward_mode is None or not ctx.forward_mode.is_decode():
        return None
    if spec_tokens != CHAIN or ctx.bs not in BATCHES:
        return None
    return ctx.bs if ctx.input_num_tokens == CHAIN * ctx.bs else None


def verify_positions(
    seq_lens: torch.Tensor, offsets: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    """The rows' positions, int64 into ``out[: CHAIN * bs]``: row ``j`` of
    request ``i`` at ``seq_lens[i] - CHAIN + j`` (``offsets`` is
    ``arange(-CHAIN, 0)``). ``seq_lens`` are the regular attention's decode
    lengths, at least the verify floor ``CHAIN``: a padded request of a graph
    batch replays at length 1, which the attention leaf raises to the floor
    (``docs/design/unified_path.md``), so its rows sit at ``0 .. CHAIN - 1``
    as in the regular attention."""
    bs = seq_lens.shape[0]
    rows = out[: CHAIN * bs]
    torch.add(seq_lens.view(bs, 1), offsets.view(1, CHAIN), out=rows.view(bs, CHAIN))
    return rows


class Inkling2Monokernel:
    """The packed layers of one Inkling2 target and the state their launches
    share.

    Args:
        model: The ``Inkling2TextModel``.
        mapping: The parallel mapping; attention TP must be the MoE EP group.
    """

    def __init__(self, model, mapping) -> None:
        config = model.config
        if mapping.attn.tp_group != mapping.moe.ep_group:
            raise ValueError("the Inkling2 monokernel needs attention TP == MoE EP")
        self.model = model
        self.mapping = mapping
        self.rank = mapping.attn.tp_rank
        block = config.n_layers_per_attn_res_block
        local = set(config.local_layer_ids)
        # Layers with one expert format for both projections: the mid local
        # ones, the boundary ones (the first of an attention-residual block,
        # layer 0 too: its block starts from the embedding) and the full
        # attention ones (the last of a block).
        self.block = block
        self.local = local
        self.eligible = [
            i
            for i, layer in enumerate(model.layers)
            if layer.mlp.experts.formats in (("nvfp4", "nvfp4"), ("mxfp8", "mxfp8"))
        ]
        self.layers: dict[int, LocalLayer] = {}
        self.state: MonokernelState | None = None
        self.bases: torch.Tensor | None = None
        # This step's values: its requests and row positions when it may
        # launch, and the launches so far (their count alternates the
        # peer-region parity).
        self.requests: int | None = None
        self.positions: torch.Tensor | None = None
        self.row_offsets: torch.Tensor | None = None
        self.row_positions: torch.Tensor | None = None
        self.launches = 0
        # Each layer's (P, A, M) outputs: the residual state may hold a
        # layer's terms past the next launch.
        self.outputs: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        # A boundary layer's stored-block count, and two buffers of the set
        # partials its sweep writes: a boundary launch reads the previous
        # set's while it writes its own, so consecutive sets alternate.
        self.block_counts: dict[int, torch.Tensor] = {}
        self.set_buffers: tuple[tuple[torch.Tensor, ...], ...] = ()

    def pack(self) -> None:
        """Pack every eligible layer (after loading; again after a weight
        update). Raises when no monokernel serves a layer."""
        for i in self.eligible:
            layer = self.model.layers[i]
            attn, ex = layer.attn, layer.mlp.experts
            experts = ex.formats[0]
            if experts == "nvfp4":
                w13 = (
                    ex.w13_weight,
                    ex.w13_scale,
                    ex.w13_amax,
                    ex.w13_outer_scale_denominator,
                )
                w2 = (
                    ex.w2_weight,
                    ex.w2_scale,
                    ex.w2_amax,
                    ex.w2_outer_scale_denominator,
                )
            else:
                w13 = (ex.w13_weight, ex.w13_scale)
                w2 = (ex.w2_weight, ex.w2_scale)
            tensors = {
                "qkvr": attn.qkvr.weight,
                "wo": attn.wo_ud.weight,
                "router": layer.mlp.gate.weight,
                "router_bias": layer.mlp.gate.bias,
                "head": (
                    attn.q_l2_clip.scale,
                    attn.rel_l2_clip.scale,
                    attn.rel_logits_proj.proj,
                    attn.k_sconv.taps().contiguous(),
                    attn.v_sconv.taps().contiguous(),
                ),
                "attn_taps": layer.attn_sconv.taps().contiguous(),
                "mlp_taps": layer.mlp_sconv.taps().contiguous(),
                "w13": w13,
                "w2": w2,
            }
            if i in self.local:
                packed = prepare_inkling2_local_layer(
                    rows=ROWS,
                    experts=experts,
                    boundary=i % self.block == 0,
                    first=i == 0,
                    tensors=tensors,
                )
            else:
                packed = prepare_inkling2_full_layer(
                    rows=ROWS, experts=experts, tensors=tensors
                )
            if packed is None:
                raise RuntimeError(
                    f"--decode-monokernel: no Inkling2 monokernel for layer {i} here"
                )
            self.layers[i] = packed
            layer.monokernel = self

    def reserve(self) -> None:
        """Reserve the mailbox on the Iris heap (before the heap exists)."""
        pairs = max(packed.scratch_pairs for packed in self.layers.values())
        reserve_iris_heap("inkling2-monokernel", iris_heap_bytes((pairs,), torch.int64))

    def prepare(self, device: torch.device) -> None:
        """Create the heap, allocate the mailbox and the outputs (eager, every
        rank of the TP group, before graph capture)."""
        create_iris_heap()
        group = pg_manager.get_process_group("nccl", self.mapping.attn.tp_group)
        context, iris_group = iris_symmetric_heap(group, self.rank, device)
        pairs = max(packed.scratch_pairs for packed in self.layers.values())
        self.state = MonokernelState(context.zeros((pairs,), dtype=torch.int64))
        iris_group.finish_prepare()
        self.bases = iris_group.bases
        # Row j of a verify block sits at seq_len - CHAIN + j.
        self.row_offsets = torch.arange(-CHAIN, 0, dtype=torch.int64, device=device)
        rows = max(ROWS)
        self.row_positions = torch.zeros(rows, dtype=torch.int64, device=device)
        hidden = self.model.config.hidden_size
        # A launch of r rows writes the first r rows.
        self.outputs = {
            i: tuple(
                torch.zeros(rows, hidden, dtype=torch.bfloat16, device=device)
                for _ in range(3)
            )
            for i in self.layers
        }
        self.block_counts = {
            i: torch.tensor([i // self.block], dtype=torch.int32, device=device)
            for i in self.layers
            if i % self.block == 0
        }
        queries = SET_QUERIES
        self.set_buffers = tuple(
            (
                torch.zeros(queries * rows, dtype=torch.float32, device=device),
                torch.zeros(queries * rows, dtype=torch.float32, device=device),
                torch.zeros(
                    queries * rows * hidden, dtype=torch.float32, device=device
                ),
            )
            for _ in range(2)
        )

    def start_step(self, ctx: ForwardContext) -> None:
        """Before the layers: this step's row positions, for a forward that
        may launch. They come from the verify metadata that the regular
        layers read (:func:`verify_positions`), into a buffer of the seam:
        the scheduler may refill the model's input positions for the next
        step while this step's launches still run."""
        self.launches = 0
        self.requests = None
        self.positions = None
        if self.state is None:
            return
        self.state.start_step()
        backend = ctx.attn_backend
        requests = launch_requests(ctx, backend.conv_spec_num_tokens)
        if requests is not None:
            # The rows' lengths, from the first packed layer's attention leaf
            # (every leaf of a forward holds the same sequence lengths).
            attn = self.model.layers[next(iter(self.layers))].attn.attn
            leaf = backend.inner._leaf_for(attn)
            seq_lens = leaf.forward_decode_metadata.seq_lens[:requests]
            self.positions = verify_positions(
                seq_lens, self.row_offsets, self.row_positions
            )
            self.requests = requests

    def finish_step(self) -> None:
        """After the layers: a new tag for the next step's launches."""
        if self.state is not None:
            self.state.finish_step()
        self.requests = None
        self.positions = None

    def serves(
        self,
        layer_id: int,
        state: AttentionResidualState,
        ctx: ForwardContext,
        mailbox,
        share,
    ) -> bool:
        """Whether this forward launches layer ``layer_id``. It depends only
        on the forward, so every rank decides the same."""
        return (
            self.state is not None
            and layer_id in self.layers
            and self.requests is not None
            and launch_requests(ctx, ctx.attn_backend.conv_spec_num_tokens)
            == self.requests
            and state.sweep_sets
            and state.share is None
            and share is None
            and mailbox is None
            and self.positions is not None
            # Layer 0 starts the first block: no set partials yet.
            and (
                state.set_partials is None
                if layer_id == 0
                else state.set_partials is not None
                and state.set_blocks == state.num_blocks
            )
        )

    def run(
        self,
        layer,
        state: AttentionResidualState,
        ctx: ForwardContext,
        log_scaling_tau: torch.Tensor | None,
    ) -> None:
        """Launch ``layer``'s step and leave the residual state as its regular
        forward does (``log_scaling_tau``: the forward's FP32 attention logit
        scales, which a full layer applies)."""
        L = layer.layer_id
        full = L not in self.local
        attn = layer.attn
        res = state.attn_res
        boundary = L % self.block == 0
        if state.partials_ready is not None:
            torch.cuda.current_stream().wait_event(state.partials_ready)
            state.partials_ready = None
        attn_row = res.attn_rows[L][0]
        mlp_row = res.mlp_rows[L]

        def query(row: int, gain: torch.Tensor) -> tuple:
            first, _ = res.set_rows[state.num_blocks - 1]
            m, s, acc = state.set_partials
            i = row - first
            return (res.score_weights[row], gain, acc[i], m[i], s[i])

        backend = ctx.attn_backend
        leaf = backend.inner._leaf_for(attn.attn)
        # The attention stages equal the regular split-KV decode bit for bit
        # only at its 128-token KV block, which is the cache page. The page is
        # fixed per server: this fails at graph capture, before serving.
        if leaf.kernel_page_size != ATTENTION_PAGE:
            raise RuntimeError(
                f"the Inkling2 monokernel needs a {ATTENTION_PAGE}-token KV page, "
                f"not {leaf.kernel_page_size}"
            )
        pool = ctx.token_to_kv_pool
        heads, head_dim = attn.attn.tp_k_head_num, attn.attn.qk_head_dim
        # The layer's cache rows, one a token slot: [slots, heads, head].
        k_cache = pool.get_key_buffer(L).view(-1, heads, head_dim)
        v_cache = pool.get_value_buffer(L).view(-1, heads, head_dim)
        # The ring commit after acceptance covers the streams this forward
        # registers, as the regular convolutions do: a local layer's K and V
        # as one stream, a full layer's each its own.
        kv = (attn.k_sconv, attn.v_sconv) if full else (attn.k_sconv,)
        for conv in (*kv, layer.attn_sconv, layer.mlp_sconv):
            _register_stream(ctx, L, conv, full)
        rows = CHAIN * self.requests
        outputs = tuple(t[:rows] for t in self.outputs[L])
        sweep = ()
        full_splits, tau = 0, None
        if full:
            # Three queries (the Q, K and V sources) with the stacked gains
            # the regular mix takes.
            gains = layer.input_norm_gains
            queries = (
                tuple(query(attn_row + j, gains[j]) for j in range(3)),
                query(mlp_row, layer.mlp_norm.weight),
            )
            full_splits = inkling2_full_layer_splits(leaf.max_context_len)
            if log_scaling_tau is None:
                raise RuntimeError("a full layer's launch needs the logit scales")
            tau = log_scaling_tau[:rows]
        elif not boundary:
            queries = (
                query(attn_row, layer.attn_norm.weight),
                query(mlp_row, layer.mlp_norm.weight),
            )
        else:
            # The layer stores its block, sweeps the new set's partials over
            # the stored blocks and its own, and mixes its MLP input over
            # them (the set's first query).
            b = state.num_blocks
            if state.blocks.shape[1] != rows:
                raise RuntimeError(
                    "the residual state's blocks are not the launch's rows"
                )
            set_first, set_last = res.set_rows[b]
            if set_last - set_first != SET_QUERIES:
                raise RuntimeError(f"block set {b} has {set_last - set_first} queries")
            m_buf, s_buf, acc_buf = self.set_buffers[b % 2]
            hidden = acc_buf.shape[0] // (SET_QUERIES * max(ROWS))
            new_set = (
                m_buf[: SET_QUERIES * rows].view(SET_QUERIES, rows),
                s_buf[: SET_QUERIES * rows].view(SET_QUERIES, rows),
                acc_buf[: SET_QUERIES * rows * hidden].view(SET_QUERIES, rows, hidden),
            )
            sweep = (
                state.blocks,
                self.block_counts[L],
                res.score_weights[set_first:set_last],
                *new_set,
            )
            new_m, new_s, new_acc = new_set
            mlp_query = (
                res.score_weights[mlp_row],
                layer.mlp_norm.weight,
                new_acc[0],
                new_m[0],
                new_s[0],
            )
            # Layer 0's attention input is the embedding's RMSNorm, not a
            # mix: its query reads only the gain (the MLP query's other
            # fields keep the operand's shapes; no set partials exist yet).
            attn_query = (
                (res.score_weights[attn_row], layer.attn_norm.weight) + mlp_query[2:]
                if L == 0
                else query(attn_row, layer.attn_norm.weight)
            )
            queries = (attn_query, mlp_query)
            outputs = (state.blocks[b],) + outputs[1:]
        inkling2_local_layer_decode(
            self.layers[L],
            rank=self.rank,
            rows=rows,
            chain=CHAIN,
            scratch=self.state.scratch,
            step=self.state.step,
            layer_index=L,
            # Launches alternate the peer copies. Across steps (an odd count
            # of launches a step) the regular layers between the last launch
            # and the next step's first, whose collectives every rank
            # finishes, keep the ranks within one launch.
            parity=self.launches % 2,
            bases=self.bases,
            terms=tuple(state.terms),
            outputs=outputs,
            queries=queries,
            sweep=sweep,
            full_splits=full_splits,
            tau=tau,
            k_cache=k_cache,
            v_cache=v_cache,
            locations=backend.padded_write_locations(attn.attn, ctx.forward_mode, rows),
            positions=self.positions,
            page_table=leaf.forward_decode_metadata.page_table,
            page_size=leaf.kernel_page_size,
            ring=backend.conv_pool.layer_state_wd(L),
            ring_slots=backend.conv_decode_metadata.cache_indices,
        )
        self.launches += 1
        self.state.launched = True
        p_term, a_term, m_term = outputs
        # As state.attention_input leaves the layer: its candidate is the one
        # pending term; a boundary layer stored it as its block and started
        # the new set's partials.
        state.terms = (p_term,)
        state.block_stored = boundary
        if boundary:
            state.set_partials, state.set_blocks = new_set, state.num_blocks + 1
            state.partials_ready = None
        state.finalize_layer(L, a_term, m_term)


def _register_stream(ctx: ForwardContext, layer_id: int, conv, full: bool) -> None:
    from tokenspeed.runtime.models.inkling import _sconv_stream

    # A local layer's adjacent K and V streams register as one.
    dim = (
        2 * conv.dim
        if not full and conv.channel_offset == 0 and conv._group() == "kvconv"
        else conv.dim
    )
    _sconv_stream(ctx, layer_id, conv.channel_offset, dim, conv._group())

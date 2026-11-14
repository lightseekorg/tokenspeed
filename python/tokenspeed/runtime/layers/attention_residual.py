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

"""Depth-wise residual aggregation and normalization with explicit precision."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from tokenspeed_kernel.ops.communication.triton import MulticastRows
from tokenspeed_kernel.ops.layernorm.precision import rmsnorm_precision
from tokenspeed_kernel.ops.residual.attention import (
    attention_residual_combine,
    attention_residual_mix,
    attention_residual_partials,
    residual_sum,
)
from torch import nn


class PrecisionRMSNorm(nn.Module):
    """Apply learned RMSNorm with an explicit output dtype.

    ``mean_by_reciprocal`` takes the mean square as a multiply by FP32
    ``1 / hidden_size`` instead of a correctly rounded divide, as the norm's
    reference kernel does.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float,
        output_dtype: torch.dtype,
        elementwise_affine: bool,
        mean_by_reciprocal: bool,
    ) -> None:
        super().__init__()
        if hidden_size < 1:
            raise ValueError("RMSNorm hidden size must be positive")
        self.eps = eps
        self.output_dtype = output_dtype
        self.mean_by_reciprocal = mean_by_reciprocal
        self.weight = (
            nn.Parameter(torch.ones(hidden_size, dtype=torch.bfloat16))
            if elementwise_affine
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rmsnorm_precision(
            x,
            self.weight,
            self.eps,
            self.output_dtype,
            mean_by_reciprocal=self.mean_by_reciprocal,
        )


class AttentionResidual(nn.Module):
    """Own depth pseudo-queries without retaining per-request attention state.

    The kernels read the queries as FP32 score weights, ``query * hidden**-0.5``,
    stored once per sublayer in ``score_weights``. Rows are grouped by block
    set: set ``k`` holds, in use order, the MLP query of each layer in block
    ``k`` and the attention query of the layer after it (the output query
    after the last layer). Every mix of one set reads the same completed
    blocks.
    """

    def __init__(
        self,
        n_layers: int,
        hidden_dim: int,
        n_layers_per_attnres_block: int,
        local_layer_ids: list[int],
        normalize_values: bool,
    ) -> None:
        super().__init__()
        if (
            n_layers < 1
            or hidden_dim < 1
            or not 0 < n_layers_per_attnres_block <= n_layers
        ):
            raise ValueError(
                "attention residual layer, hidden and block dimensions are invalid"
            )
        if len(set(local_layer_ids)) != len(local_layer_ids) or any(
            layer < 0 or layer >= n_layers for layer in local_layer_ids
        ):
            raise ValueError(
                "local layer IDs must be unique and within the layer count"
            )
        self.n_layers = n_layers
        self.n_layers_per_attnres_block = n_layers_per_attnres_block
        self.local_layer_ids = frozenset(local_layer_ids)
        self.normalize_values = normalize_values
        self.attn_queries = nn.ParameterList(
            [
                nn.Parameter(
                    torch.zeros(
                        (
                            hidden_dim
                            if layer in self.local_layer_ids
                            else (3, hidden_dim)
                        ),
                        dtype=torch.float32,
                    )
                )
                for layer in range(1, n_layers)
            ]
        )
        self.mlp_queries = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(hidden_dim, dtype=torch.float32))
                for _ in range(n_layers)
            ]
        )
        self.output_query = nn.Parameter(torch.zeros(hidden_dim, dtype=torch.float32))
        # (first row, count) of each layer's attention queries, then the output query.
        self.attn_rows: list[tuple[int, int]] = [(0, 0)] * (n_layers + 1)
        self.mlp_rows: list[int] = [0] * n_layers
        rows = 0
        for layer in range(n_layers):
            self.mlp_rows[layer] = rows
            rows += 1
            following = layer + 1
            count = (
                1 if following == n_layers or following in self.local_layer_ids else 3
            )
            self.attn_rows[following] = (rows, count)
            rows += count
        # [first, last) rows of each block set.
        self.set_rows: list[tuple[int, int]] = []
        for first_layer in range(0, n_layers, n_layers_per_attnres_block):
            last_layer = min(first_layer + n_layers_per_attnres_block, n_layers)
            row, count = self.attn_rows[last_layer]
            self.set_rows.append((self.mlp_rows[first_layer], row + count))
        self.register_buffer(
            "score_weights",
            torch.zeros((rows, hidden_dim), dtype=torch.float32),
            persistent=False,
        )

    @property
    def num_blocks(self) -> int:
        """Completed blocks a forward can hold: the embedding and one per block."""
        return -(-self.n_layers // self.n_layers_per_attnres_block)

    def is_block_boundary(self, layer_idx: int) -> bool:
        return layer_idx % self.n_layers_per_attnres_block == 0

    def attn_query(self, layer_idx: int) -> torch.Tensor:
        if layer_idx <= 0:
            raise ValueError("the first attention sublayer has no depth pseudo-query")
        return self.attn_queries[layer_idx - 1]

    def derive_score_weights(self) -> None:
        """Write ``score_weights`` from the loaded queries; safe to repeat."""
        scale = self.score_weights.shape[1] ** -0.5
        queries = [
            *(self.attn_query(layer) for layer in range(1, self.n_layers)),
            self.output_query,
            *self.mlp_queries,
        ]
        rows = [
            *(self.attn_rows[layer] for layer in range(1, self.n_layers + 1)),
            *((row, 1) for row in self.mlp_rows),
        ]
        with torch.no_grad():
            for query, (row, count) in zip(queries, rows):
                self.score_weights[row : row + count].copy_(
                    query.float().view(count, -1) * scale
                )


def norm_gains(norms: tuple[PrecisionRMSNorm, ...]) -> torch.Tensor | None:
    """The [norms, hidden] gains when the norms can run inside the mix kernel."""
    if any(
        norm.weight is None
        or norm.output_dtype != torch.bfloat16
        or norm.eps != norms[0].eps
        for norm in norms
    ):
        return None
    if len(norms) == 1:
        return norms[0].weight.view(1, -1)
    return torch.stack([norm.weight for norm in norms])


@dataclass(frozen=True)
class ResidualShare:
    """This rank's tokens of a sequence-parallel forward and where its mixes go.

    Every source the state holds (stored blocks, the running block sum) is
    real only on rows ``[first, first + count)``. Each mix covers those rows
    and stores them through the multicast buffer of its use, so after its
    barrier every rank holds the whole mix: attention inputs (up to three
    queries) and the output in ``attention``, MLP inputs in ``mlp``, and the
    full block sums the drafter captures in ``gather``. A buffer is written
    again only after the next use of another one, whose barrier every rank
    reaches after reading it.
    """

    first: int
    count: int
    attention: MulticastRows
    mlp: MulticastRows
    gather: MulticastRows


class AttentionResidualState:
    """Retain completed depth blocks only for the current forward call.

    Completed blocks live in one ``[blocks, tokens, hidden]`` buffer. The
    running sum of the current block stays as pending terms until a mix needs
    it: the next attention-side mix rounds the sum once, as ``residual_sum``
    would, stores it, and mixes the stored value. Each mix finishes with the
    RMSNorm that consumes it.
    """

    eps = 1e-6
    # Up to this many tokens one sweep per block set serves all of the set's queries.
    set_sweep_max_tokens = 256

    def __init__(
        self,
        attn_res: AttentionResidual,
        embedding: torch.Tensor,
        aux_stream: torch.cuda.Stream | None,
        share: ResidualShare | None,
    ) -> None:
        self.attn_res = attn_res
        self.blocks = embedding.new_empty((attn_res.num_blocks, *embedding.shape))
        self.num_blocks = 0
        # Whether blocks[num_blocks] already holds the current partial.
        self.block_stored = False
        self.terms: tuple[torch.Tensor, ...] = (embedding,)
        self.sweep_sets = embedding.shape[0] <= self.set_sweep_max_tokens
        if share is not None and self.sweep_sets:
            raise ValueError("a sequence-parallel forward is larger than a sweep")
        self.share = share
        # Partials of the block set over the first set_blocks blocks.
        self.set_blocks = 0
        self.set_partials: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        # A set's partials start on aux_stream once its last block is stored,
        # alongside the attention before their first mix; None runs them
        # inline at that mix. partials_ready orders that mix after them.
        self.aux_stream = aux_stream
        self.partials_ready: torch.cuda.Event | None = None

    def _set_partials(
        self, num_blocks: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        first, last = self.attn_res.set_rows[num_blocks - 1]
        return attention_residual_partials(
            self.blocks[:num_blocks],
            self.attn_res.score_weights[first:last],
            normalize_values=self.attn_res.normalize_values,
            eps=self.eps,
        )

    def _start_set(self, num_blocks: int) -> None:
        """Start the partials of the set over ``num_blocks`` stored blocks on
        the aux stream; the set's first mix waits for them."""
        if not self.sweep_sets or self.aux_stream is None:
            return
        main = torch.cuda.current_stream()
        self.aux_stream.wait_stream(main)
        with torch.cuda.stream(self.aux_stream):
            partials = self._set_partials(num_blocks)
            ready = torch.cuda.Event()
            ready.record()
        for x in partials:
            x.record_stream(main)
        self.set_partials, self.set_blocks = partials, num_blocks
        self.partials_ready = ready

    @property
    def partial(self) -> torch.Tensor:
        """The running sum of the current block, rounded once."""
        if len(self.terms) > 1:
            self.terms = (residual_sum(self.terms),)
        return self.terms[0]

    def _mix(
        self,
        rows: tuple[int, int],
        num_blocks: int,
        terms: tuple[torch.Tensor, ...],
        candidate_out: torch.Tensor | None,
        norms: tuple[PrecisionRMSNorm, ...],
        norm_gains: torch.Tensor | None,
        shared: MulticastRows | None,
    ) -> tuple[torch.Tensor, ...]:
        """Mix query rows ``rows``; in a sequence-parallel forward ``shared``
        is the multicast buffer this use stores its share of the mix into."""
        row, count = rows
        weights = self.attn_res.score_weights[row : row + count]
        mix_args = dict(
            candidate_out=candidate_out,
            normalize_values=self.attn_res.normalize_values,
            eps=self.eps,
            norm_weights=norm_gains,
            norm_eps=norms[0].eps,
            output_dtype=torch.float32 if norm_gains is None else torch.bfloat16,
        )
        if self.sweep_sets:
            first, last = self.attn_res.set_rows[num_blocks - 1]
            if self.set_blocks != num_blocks:
                self.set_partials = self._set_partials(num_blocks)
                self.set_blocks = num_blocks
            elif self.partials_ready is not None:
                torch.cuda.current_stream().wait_event(self.partials_ready)
                self.partials_ready = None
            partials = tuple(
                x[row - first : row - first + count] for x in self.set_partials
            )
            mixed = attention_residual_combine(terms, weights, partials, **mix_args)
        elif self.share is None:
            # Larger forwards mix each query in one pass: no shared sweep to keep.
            mixed = attention_residual_mix(
                self.blocks[:num_blocks],
                terms,
                weights,
                tokens=None,
                out=None,
                **mix_args,
            )
        else:
            if norm_gains is None:
                raise ValueError("a sequence-parallel mix normalizes in its kernel")
            shape = (count, *terms[0].shape)
            attention_residual_mix(
                self.blocks[:num_blocks],
                terms,
                weights,
                tokens=(self.share.first, self.share.count),
                out=shared.multicast(shape),
                **mix_args,
            )
            shared.barrier()
            mixed = shared.local(shape)
        if norm_gains is None:
            return tuple(norm(x) for norm, x in zip(norms, mixed.unbind(0)))
        return tuple(mixed.unbind(0))

    def attention_input(
        self,
        layer_idx: int,
        norms: tuple[PrecisionRMSNorm, ...],
        norm_gains: torch.Tensor | None,
    ) -> tuple[torch.Tensor, ...]:
        """Normalized attention inputs, one per norm (1 local, 3 full-layer).

        Args:
            layer_idx: The decoder layer.
            norms: The input RMSNorms of the layer.
            norm_gains: ``norm_gains(norms)``. Full layers pass the gains
                they stack once after loading; local layers compute it every
                forward, which only views the one norm's weight.
        """
        boundary = self.attn_res.is_block_boundary(layer_idx)
        if self.num_blocks == 0:
            partial = self.partial
            if boundary:
                self.blocks[0].copy_(partial)
                self.block_stored = True
                self._start_set(1)
            return tuple(norm(partial) for norm in norms)
        if boundary:
            candidate_out = self.blocks[self.num_blocks]
        elif len(self.terms) > 1:
            candidate_out = torch.empty_like(self.terms[0])
        else:
            candidate_out = None
        mixed = self._mix(
            self.attn_res.attn_rows[layer_idx],
            self.num_blocks,
            self.terms,
            candidate_out,
            norms,
            norm_gains,
            None if self.share is None else self.share.attention,
        )
        if candidate_out is not None:
            self.terms = (candidate_out,)
        self.block_stored = boundary
        if boundary:
            # The block is stored; the layer's MLP mix reads one more.
            self._start_set(self.num_blocks + 1)
        return mixed

    def mlp_input(
        self, layer_idx: int, attn_output: torch.Tensor, norm: PrecisionRMSNorm
    ) -> torch.Tensor:
        """The normalized MLP input: the partial plus ``attn_output``, unrounded."""
        boundary = self.attn_res.is_block_boundary(layer_idx)
        if boundary and not self.block_stored:
            self.blocks[self.num_blocks].copy_(self.partial)
            self.block_stored = True
        terms = (attn_output,) if boundary else (self.partial, attn_output)
        return self._mix(
            (self.attn_res.mlp_rows[layer_idx], 1),
            self.num_blocks + int(boundary),
            terms,
            None,
            (norm,),
            norm_gains((norm,)),
            None if self.share is None else self.share.mlp,
        )[0]

    def finalize_layer(
        self, layer_idx: int, attn_output: torch.Tensor, mlp_output: torch.Tensor
    ) -> None:
        if self.attn_res.is_block_boundary(layer_idx):
            if not self.block_stored:
                self.blocks[self.num_blocks].copy_(self.partial)
            self.num_blocks += 1
            self.block_stored = False
            self.terms = (attn_output, mlp_output)
        else:
            self.terms = (self.partial, attn_output, mlp_output)

    def compute_output(self, norm: PrecisionRMSNorm) -> torch.Tensor:
        """The normalized output mixture of all blocks and the final partial."""
        output = self._mix(
            self.attn_res.attn_rows[self.attn_res.n_layers],
            self.num_blocks,
            self.terms,
            torch.empty_like(self.terms[0]) if len(self.terms) > 1 else None,
            (norm,),
            norm_gains((norm,)),
            None if self.share is None else self.share.attention,
        )[0]
        # The next forward's mixes reuse the shared buffer.
        return output if self.share is None else output.clone()

    def full_partial(self) -> torch.Tensor:
        """``partial`` on every row: in a sequence-parallel forward each rank
        contributes its share through the gather buffer."""
        partial = self.partial
        if self.share is None:
            return partial
        rows = slice(self.share.first, self.share.first + self.share.count)
        self.share.gather.multicast(partial.shape)[rows] = partial[rows]
        self.share.gather.barrier()
        return self.share.gather.local(partial.shape).clone()

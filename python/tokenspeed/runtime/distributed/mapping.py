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

import math
from functools import cached_property

Group = tuple[int, ...]


def _resolve_parallelism_sizes(world_size: int, *sizes: int | None) -> tuple[int, ...]:
    """Resolve the parallelism sizes given world_size.

    `sizes` is ordered innermost (fastest-varying) to outermost.
    """
    assert all(x is None or x > 0 for x in sizes)

    resolved = [x for x in sizes]
    num_to_resolve = sum(x is None for x in sizes)
    if num_to_resolve > 0:
        provided_size = math.prod(x for x in sizes if x is not None)
        assert provided_size <= world_size
        assert world_size % provided_size == 0
        resolved_size = world_size // provided_size

        for index, size in enumerate(resolved):
            if size is None:
                resolved[index] = resolved_size
                resolved_size = 1

    assert math.prod(resolved) == world_size
    return tuple(resolved)


def _resolve_dcp_size(tp_size: int, dcp_size: int) -> int:
    """Validate DCP within resolved attention TP; DCP adds no world-size dimension."""
    if isinstance(dcp_size, bool) or not isinstance(dcp_size, int) or dcp_size < 1:
        raise ValueError("dcp_size must be a positive integer")
    if tp_size % dcp_size:
        raise ValueError("attention TP size must be divisible by DCP size")
    return dcp_size


def _resolve_qcp_size(tp_size: int, qcp_size: int) -> int:
    """Validate query context parallelism within resolved attention TP.

    QCP shards an extend forward's query rows over the attention TP group;
    it adds no world-size dimension. The shard spans the whole group (1 is
    off): a partial shard would leave the ranks holding the same rows strided
    by the shard width, which no group here (the head group, the DCP group)
    is built with, and ``validate_qcp`` pins the server to the same rule.
    """
    if isinstance(qcp_size, bool) or not isinstance(qcp_size, int) or qcp_size < 1:
        raise ValueError("qcp_size must be a positive integer")
    if tp_size % qcp_size:
        raise ValueError("attention TP size must be divisible by QCP size")
    if qcp_size not in (1, tp_size):
        raise ValueError(
            f"a query shard spans the whole attention TP group: qcp_size={qcp_size} "
            f"must be 1 or the attention TP size {tp_size}"
        )
    return qcp_size


def _make_parallelism_rank(rank: int, size: int, stride: int = 1) -> int:
    """Return the rank of given size and stride."""
    return (rank // stride) % size


def _make_parallelism_group(rank: int, size: int, stride: int = 1) -> Group:
    """Return the group of ranks of given size and stride."""
    base = rank - (rank // stride % size) * stride
    return tuple(base + j * stride for j in range(size))


class MappingBase:

    def __init__(self, rank: int | None = None, world_size: int = 1):
        assert rank is None or rank >= 0
        self._rank = rank
        assert world_size > 0
        self._world_size = world_size

    @property
    def rank(self) -> int:
        assert self._rank is not None, "rank is not initialized"
        return self._rank

    @rank.setter
    def rank(self, rank: int):
        assert self._rank is None, "rank is already initialized"
        assert rank >= 0
        self._rank = rank
        self._on_rank_initialized(rank)

    def _on_rank_initialized(self, rank: int):
        return None

    @property
    def world_size(self) -> int:
        return self._world_size

    @cached_property
    def world_group(self) -> Group:
        return _make_parallelism_group(self.rank, self.world_size, stride=1)


class DenseLayerMapping(MappingBase):

    def __init__(
        self,
        rank: int | None = None,
        world_size: int = 1,
        tp_size: int | None = None,
        dp_size: int | None = None,
    ):
        super().__init__(rank, world_size)
        self.tp_size, self.dp_size = _resolve_parallelism_sizes(
            self.world_size, tp_size, dp_size
        )

    @cached_property
    def has_tp(self) -> bool:
        return self.tp_size > 1

    @cached_property
    def tp_rank(self) -> int:
        return _make_parallelism_rank(self.rank, self.tp_size, stride=1)

    @cached_property
    def tp_group(self) -> Group:
        return _make_parallelism_group(self.rank, self.tp_size, stride=1)

    @cached_property
    def has_dp(self) -> bool:
        return self.dp_size > 1

    @cached_property
    def dp_rank(self) -> int:
        return _make_parallelism_rank(self.rank, self.dp_size, stride=self.tp_size)

    @cached_property
    def dp_group(self) -> Group:
        return _make_parallelism_group(self.rank, self.dp_size, stride=self.tp_size)


def _attention_row_group_size(tp_size: int, qcp_size: int) -> int:
    """Ranks of the attention TP group that hold the same attention rows.

    Attention TP replicates a forward's rows over its group; query context
    parallelism (QCP) shards an extend's rows over that group instead, so
    only ``tp_size // qcp_size`` ranks hold the same rows -- one, since a
    query shard spans the whole group (``_resolve_qcp_size``), which is what
    lets the groups below stay contiguous.
    """
    return tp_size // qcp_size


def _resolve_head_tp_size(
    tp_size: int, qcp_size: int, world_size: int, head_tp_size: int | None
) -> int:
    """Resolve the attention head-TP width.

    ``None`` keeps the head projections on the ranks that hold the same
    attention rows (today's layout: the attention TP group; head-replicated
    under query context parallelism, whose shards hold different rows). A
    wider head group shards ``q_b_proj`` / ``kv_b_proj`` / ``o_proj`` by
    heads over ranks that hold different rows -- attention-DP ranks, or the
    query shards of a QCP group -- and the attention forward exchanges heads
    for tokens around core attention. It therefore requires that no two
    ranks of the group hold the same rows (attention TP 1, or a query shard
    spanning the attention TP group), is the QCP group itself under query
    sharding, and must tile the stage world.
    """
    row_group_size = _attention_row_group_size(tp_size, qcp_size)
    if head_tp_size is None:
        return row_group_size
    if (
        isinstance(head_tp_size, bool)
        or not isinstance(head_tp_size, int)
        or head_tp_size < 1
    ):
        raise ValueError("attention head TP size must be a positive integer")
    if head_tp_size == row_group_size:
        return head_tp_size
    if row_group_size != 1:
        raise ValueError(
            "attention head TP shards heads over ranks that hold different rows "
            "(attention-DP ranks, or the query shards of a QCP group) and needs "
            "attention TP 1 or a query shard over the attention TP group, "
            f"got attn_tp_size={tp_size} with qcp_size={qcp_size}"
        )
    if qcp_size != 1 and head_tp_size != qcp_size:
        raise ValueError(
            "under query context parallelism the attention head TP group is the "
            f"query-shard group: head_tp_size={head_tp_size} must equal "
            f"qcp_size={qcp_size}"
        )
    if world_size % head_tp_size:
        raise ValueError(
            f"attention head TP size {head_tp_size} must divide the stage "
            f"world size {world_size}"
        )
    return head_tp_size


class AttentionLayerMapping(MappingBase):

    def __init__(
        self,
        rank: int | None = None,
        world_size: int = 1,
        tp_size: int | None = None,
        dp_size: int | None = None,
        dcp_size: int = 1,
        head_tp_size: int | None = None,
        qcp_size: int = 1,
    ):
        super().__init__(rank, world_size)
        self.tp_size, self.dp_size = _resolve_parallelism_sizes(
            self.world_size, tp_size, dp_size
        )
        self.dcp_size = _resolve_dcp_size(self.tp_size, dcp_size)
        self.qcp_size = _resolve_qcp_size(self.tp_size, qcp_size)
        # Width of the group the head projections (q_b/kv_b/o_proj) shard
        # over. Equal to the ranks holding the same rows (tp_size, or 1 under
        # query sharding) unless head TP widens it over DP ranks or over the
        # query shards.
        self.head_tp_size = _resolve_head_tp_size(
            self.tp_size, self.qcp_size, self.world_size, head_tp_size
        )

    @property
    def has_head_tp(self) -> bool:
        """Heads are sharded over a group wider than the ranks that hold the
        same attention rows (the attention TP group, or one query shard of it
        under QCP), so the attention forward exchanges heads for tokens around
        core attention."""
        return self.head_tp_size != _attention_row_group_size(
            self.tp_size, self.qcp_size
        )

    @property
    def head_tp_serves_decode_only(self) -> bool:
        """Head TP over attention-DP ranks serves decode rows only: an expanded
        prefill needs every head's K/V for the cached prefix, which the
        head-sharded ``kv_b_proj`` cannot produce. Over the query shards of a
        QCP group the sparse prefill is absorbed and the extend rows take the
        exchange, so that layout serves the prefill role."""
        return self.has_head_tp and not self.has_qcp

    @cached_property
    def head_tp_rank(self) -> int:
        return _make_parallelism_rank(self.rank, self.head_tp_size, stride=1)

    @cached_property
    def head_tp_group(self) -> Group:
        """Contiguous ranks sharing one set of head projections; equals
        ``tp_group`` without head TP (``(rank,)`` under query sharding, whose
        default is head-replicated) and ``qcp_group`` under head TP over the
        query shards."""
        return _make_parallelism_group(self.rank, self.head_tp_size, stride=1)

    @property
    def has_dcp(self) -> bool:
        return self.dcp_size > 1

    @property
    def has_qcp(self) -> bool:
        return self.qcp_size > 1

    @cached_property
    def qcp_rank(self) -> int:
        """Rank within the consecutive query-context-parallel subgroup of
        attention TP; it is this rank's query shard."""
        return _make_parallelism_rank(self.rank, self.qcp_size, stride=1)

    @cached_property
    def qcp_group(self) -> Group:
        return _make_parallelism_group(self.rank, self.qcp_size, stride=1)

    @cached_property
    def dcp_rank(self) -> int:
        """Rank within the consecutive DCP subgroup of attention TP."""
        return _make_parallelism_rank(self.rank, self.dcp_size, stride=1)

    @cached_property
    def dcp_replica_rank(self) -> int:
        return _make_parallelism_rank(
            self.rank, self.tp_size // self.dcp_size, stride=self.dcp_size
        )

    @cached_property
    def dcp_group(self) -> Group:
        return _make_parallelism_group(self.rank, self.dcp_size, stride=1)

    @cached_property
    def has_tp(self) -> bool:
        return self.tp_size > 1

    @cached_property
    def tp_rank(self) -> int:
        return _make_parallelism_rank(self.rank, self.tp_size, stride=1)

    @cached_property
    def tp_group(self) -> Group:
        return _make_parallelism_group(self.rank, self.tp_size, stride=1)

    @cached_property
    def has_dp(self) -> bool:
        return self.dp_size > 1

    @cached_property
    def dp_rank(self) -> int:
        return _make_parallelism_rank(self.rank, self.dp_size, stride=self.tp_size)

    @cached_property
    def dp_group(self) -> Group:
        return _make_parallelism_group(self.rank, self.dp_size, stride=self.tp_size)

    def scatter_index(self, rank: int) -> int:
        """Index of ``rank`` in a dp-major/tp-minor scattered token count
        table."""
        tp_rank = _make_parallelism_rank(rank, self.tp_size, stride=1)
        dp_rank = _make_parallelism_rank(rank, self.dp_size, stride=self.tp_size)
        return dp_rank * self.tp_size + tp_rank


class MoeLayerMapping(MappingBase):
    def __init__(
        self,
        rank: int | None = None,
        world_size: int = 1,
        tp_size: int | None = None,
        ep_size: int | None = None,
        dp_size: int | None = None,
    ):
        super().__init__(rank, world_size)
        self.tp_size, self.ep_size, self.dp_size = _resolve_parallelism_sizes(
            self.world_size, tp_size, ep_size, dp_size
        )

    @cached_property
    def has_tp(self) -> bool:
        return self.tp_size > 1

    @cached_property
    def tp_rank(self) -> int:
        return _make_parallelism_rank(self.rank, self.tp_size, stride=1)

    @cached_property
    def tp_group(self) -> Group:
        return _make_parallelism_group(self.rank, self.tp_size, stride=1)

    @cached_property
    def has_ep(self) -> bool:
        return self.ep_size > 1

    @cached_property
    def ep_rank(self) -> int:
        return _make_parallelism_rank(self.rank, self.ep_size, stride=self.tp_size)

    @cached_property
    def ep_group(self) -> Group:
        return _make_parallelism_group(self.rank, self.ep_size, stride=self.tp_size)

    @cached_property
    def has_tp_ep(self) -> bool:
        return self.tp_ep_size > 1

    @cached_property
    def tp_ep_size(self) -> int:
        return self.tp_size * self.ep_size

    @cached_property
    def tp_ep_rank(self) -> int:
        return _make_parallelism_rank(self.rank, self.tp_ep_size, stride=1)

    @cached_property
    def tp_ep_group(self) -> Group:
        return _make_parallelism_group(self.rank, self.tp_ep_size, stride=1)

    @cached_property
    def has_dp(self) -> bool:
        return self.dp_size > 1

    @cached_property
    def dp_rank(self) -> int:
        return _make_parallelism_rank(
            self.rank, self.dp_size, stride=self.tp_size * self.ep_size
        )

    @cached_property
    def dp_group(self) -> Group:
        return _make_parallelism_group(
            self.rank, self.dp_size, stride=self.tp_size * self.ep_size
        )


class VisionTowerMapping(MappingBase):
    """Parallel mapping for colocated multimodal encoders.

    ``tp_size`` controls weight tensor parallelism inside the encoder, while
    ``dp_size`` controls item data parallelism. The mapping's world is one
    attention TP group, so item-DP composes independently with outer request
    data parallelism.
    """

    def __init__(
        self,
        rank: int | None = None,
        world_size: int = 1,
        tp_size: int | None = None,
        dp_size: int | None = None,
    ):
        super().__init__(rank, world_size)
        self.tp_size, self.dp_size = _resolve_parallelism_sizes(
            self.world_size, tp_size, dp_size
        )

    @cached_property
    def has_tp(self) -> bool:
        return self.tp_size > 1

    @cached_property
    def tp_rank(self) -> int:
        return _make_parallelism_rank(self.rank, self.tp_size, stride=1)

    @cached_property
    def tp_group(self) -> Group:
        return _make_parallelism_group(self.rank, self.tp_size, stride=1)

    @cached_property
    def has_dp(self) -> bool:
        return self.dp_size > 1

    @cached_property
    def dp_rank(self) -> int:
        return _make_parallelism_rank(self.rank, self.dp_size, stride=self.tp_size)

    @cached_property
    def dp_group(self) -> Group:
        return _make_parallelism_group(self.rank, self.dp_size, stride=self.tp_size)


class LinearAttnLayerMapping(MappingBase):
    """Parallel mapping for linear-attention layers (KDA, GDN).

    Linear-attention layers default to the attention TP width, which
    preserves the historical behavior on every existing deployment. Unlike
    MLA — whose per-token latent KV cannot shard by heads and therefore
    needs attention-DP — linear attention is TP-friendly: weights and the
    per-head recurrent state both shard by head. A wider ``tp_size`` (up to
    the stage world) head-shards them across ranks that are data-parallel
    for the full-attention layers (the MLA-DP + linear-attn-TP hybrid).
    The TP group is contiguous (stride 1) inside a pipeline stage.
    """

    def __init__(
        self,
        rank: int | None = None,
        world_size: int = 1,
        tp_size: int | None = None,
    ):
        super().__init__(rank, world_size)
        # dp is the implicit complement: the number of KDA-TP replicas.
        self.tp_size, self.dp_size = _resolve_parallelism_sizes(
            self.world_size, tp_size, None
        )

    @cached_property
    def has_tp(self) -> bool:
        return self.tp_size > 1

    @cached_property
    def tp_rank(self) -> int:
        return _make_parallelism_rank(self.rank, self.tp_size, stride=1)

    @cached_property
    def tp_group(self) -> Group:
        return _make_parallelism_group(self.rank, self.tp_size, stride=1)

    @cached_property
    def dp_rank(self) -> int:
        return _make_parallelism_rank(self.rank, self.dp_size, stride=self.tp_size)

    @cached_property
    def dp_group(self) -> Group:
        return _make_parallelism_group(self.rank, self.dp_size, stride=self.tp_size)


class LmHeadMapping(MappingBase):
    """Parallel mapping for the vocabulary-sharded LM head.

    Without attention DP the head follows the attention TP group, as it
    always has. Under attention DP the head is replicated by default
    (``tp_size`` 1); a wider ``tp_size`` vocab-shards it over a contiguous
    group of attention-DP ranks, which then gather their tokens before the
    logits GEMM and transpose the vocab shards back to their own rows.
    """

    def __init__(
        self,
        rank: int | None = None,
        world_size: int = 1,
        tp_size: int | None = None,
    ):
        super().__init__(rank, world_size)
        self.tp_size, self.dp_size = _resolve_parallelism_sizes(
            self.world_size, tp_size, None
        )

    @cached_property
    def has_tp(self) -> bool:
        return self.tp_size > 1

    @cached_property
    def tp_rank(self) -> int:
        return _make_parallelism_rank(self.rank, self.tp_size, stride=1)

    @cached_property
    def tp_group(self) -> Group:
        return _make_parallelism_group(self.rank, self.tp_size, stride=1)

    @cached_property
    def dp_rank(self) -> int:
        return _make_parallelism_rank(self.rank, self.dp_size, stride=self.tp_size)

    @cached_property
    def dp_group(self) -> Group:
        return _make_parallelism_group(self.rank, self.dp_size, stride=self.tp_size)


def _resolve_lm_head_tp_size(
    attn: AttentionLayerMapping, lm_head_tp_size: int | None
) -> int:
    """Resolve the LM-head TP width against the attention layout.

    The default is today's layout: the attention TP width without attention
    DP, replicated (1) with it. An explicit width under attention DP needs
    attention TP 1 (the logits path gathers whole rows from each rank) and
    must tile the stage world; without DP it must equal the attention TP.
    """
    if lm_head_tp_size is None:
        return 1 if attn.has_dp else attn.tp_size
    if (
        isinstance(lm_head_tp_size, bool)
        or not isinstance(lm_head_tp_size, int)
        or lm_head_tp_size < 1
    ):
        raise ValueError("LM head TP size must be a positive integer")
    if not attn.has_dp:
        if lm_head_tp_size != attn.tp_size:
            raise ValueError(
                "without attention DP the LM head follows the attention TP "
                f"group; got lm_head_tp_size={lm_head_tp_size} with "
                f"attn_tp_size={attn.tp_size}"
            )
        return lm_head_tp_size
    if lm_head_tp_size > 1 and attn.tp_size != 1:
        raise ValueError(
            "LM head TP under attention DP needs attention TP 1, got "
            f"attn_tp_size={attn.tp_size}"
        )
    if attn.world_size % lm_head_tp_size:
        raise ValueError(
            f"LM head TP size {lm_head_tp_size} must divide the stage world "
            f"size {attn.world_size}"
        )
    return lm_head_tp_size


class Mapping(MappingBase):

    def __init__(
        self,
        rank: int | None = None,
        world_size: int = 1,
        *,
        attn_tp_size: int | None = None,
        attn_dp_size: int | None = None,
        attn_dcp_size: int = 1,
        attn_head_tp_size: int | None = None,
        lm_head_tp_size: int | None = None,
        attn_qcp_size: int = 1,
        dense_tp_size: int | None = None,
        dense_dp_size: int | None = None,
        moe_tp_size: int | None = None,
        moe_ep_size: int | None = None,
        moe_dp_size: int | None = None,
        vision_tp_size: int | None = None,
        vision_dp_size: int | None = None,
        linear_attn_tp_size: int | None = None,
        pp_size: int = 1,
        pp_layer_partition: tuple[int, ...] | None = None,
        nprocs_per_node: int | None = None,
        nnodes: int | None = None,
        base_gpu_id: int = 0,
        gpu_id_step: int = 1,
    ):
        super().__init__(rank, world_size)
        # Pipeline parallelism is the outermost dimension: the world splits
        # into pp_size contiguous stages and every per-layer-type mapping
        # resolves inside one stage. Sub-mappings keep the GLOBAL rank — the
        # stride arithmetic stays correct because a stage base is a multiple
        # of every intra-stage stride.
        assert pp_size > 0
        assert (
            world_size % pp_size == 0
        ), f"world_size {world_size} must be divisible by pp_size {pp_size}"
        self.pp_size = pp_size
        # Optional explicit per-stage layer counts (front to back); validated
        # against the model's layer count where the split is resolved
        # (pp_stage_windows).
        self.pp_layer_partition = (
            tuple(int(count) for count in pp_layer_partition)
            if pp_layer_partition
            else None
        )
        stage_world_size = world_size // pp_size
        self.attn = AttentionLayerMapping(
            rank=rank,
            world_size=stage_world_size,
            tp_size=attn_tp_size,
            dp_size=attn_dp_size,
            dcp_size=attn_dcp_size,
            head_tp_size=attn_head_tp_size,
            qcp_size=attn_qcp_size,
        )
        self.lm_head = LmHeadMapping(
            rank=rank,
            world_size=stage_world_size,
            tp_size=_resolve_lm_head_tp_size(self.attn, lm_head_tp_size),
        )
        self.dense = DenseLayerMapping(
            rank=rank,
            world_size=stage_world_size,
            tp_size=dense_tp_size,
            dp_size=dense_dp_size,
        )
        self.moe = MoeLayerMapping(
            rank=rank,
            world_size=stage_world_size,
            tp_size=moe_tp_size,
            ep_size=moe_ep_size,
            dp_size=moe_dp_size,
        )
        # The vision mapping is local to each attention TP group. With both
        # sizes omitted it resolves to weight TP, preserving the legacy mode.
        self.vision = VisionTowerMapping(
            rank=rank,
            world_size=self.attn.tp_size,
            tp_size=vision_tp_size,
            dp_size=vision_dp_size,
        )
        # Linear-attention layers follow the attention TP width unless
        # overridden — the default is behavior-identical to reading
        # mapping.attn.tp_size.
        self.linear_attn = LinearAttnLayerMapping(
            rank=rank,
            world_size=stage_world_size,
            tp_size=(
                linear_attn_tp_size
                if linear_attn_tp_size is not None
                else self.attn.tp_size
            ),
        )
        self.nprocs_per_node, self.nnodes = _resolve_parallelism_sizes(
            self.world_size, nprocs_per_node, nnodes
        )
        assert base_gpu_id >= 0
        assert gpu_id_step > 0
        self.base_gpu_id = base_gpu_id
        self.gpu_id_step = gpu_id_step

    def _on_rank_initialized(self, rank: int):
        self.attn.rank = rank
        self.lm_head.rank = rank
        self.dense.rank = rank
        self.moe.rank = rank
        self.vision.rank = rank
        self.linear_attn.rank = rank

    @cached_property
    def has_pp(self) -> bool:
        return self.pp_size > 1

    @cached_property
    def stage_world_size(self) -> int:
        return self.world_size // self.pp_size

    @cached_property
    def pp_rank(self) -> int:
        return _make_parallelism_rank(
            self.rank, self.pp_size, stride=self.stage_world_size
        )

    @cached_property
    def pp_group(self) -> Group:
        return _make_parallelism_group(
            self.rank, self.pp_size, stride=self.stage_world_size
        )

    @cached_property
    def is_first_pp_rank(self) -> bool:
        return self.pp_rank == 0

    @cached_property
    def is_last_pp_rank(self) -> bool:
        return self.pp_rank == self.pp_size - 1

    @cached_property
    def pp_prev_rank(self) -> int:
        """Global rank of the same intra-stage position one stage upstream."""
        assert not self.is_first_pp_rank
        return self.rank - self.stage_world_size

    @cached_property
    def pp_next_rank(self) -> int:
        """Global rank of the same intra-stage position one stage downstream."""
        assert not self.is_last_pp_rank
        return self.rank + self.stage_world_size

    @cached_property
    def has_attn_tp(self) -> bool:
        return self.attn.has_tp

    @cached_property
    def has_attn_dp(self) -> bool:
        return self.attn.has_dp

    @cached_property
    def node_rank(self) -> int:
        return self.rank // self.nprocs_per_node

    @cached_property
    def local_rank(self) -> int:
        return self.rank % self.nprocs_per_node

    @cached_property
    def gpu_id(self) -> int:
        return self.base_gpu_id + self.local_rank * self.gpu_id_step

    def __repr__(self) -> str:
        rank_str = str(self._rank) if self._rank is not None else "?"
        lines = [
            f"Mapping(rank={rank_str}, world_size={self.world_size})",
            f"  Cluster : {self.nnodes} node(s) x {self.nprocs_per_node} proc(s)",
            f"  Pipeline: pp={self.pp_size}",
            f"  Attention: tp={self.attn.tp_size}  dcp={self.attn.dcp_size}  "
            f"qcp={self.attn.qcp_size}  dp={self.attn.dp_size}  "
            f"head_tp={self.attn.head_tp_size}",
            f"    Vision: tp={self.vision.tp_size}  item_dp={self.vision.dp_size}",
            f"  LM head : tp={self.lm_head.tp_size}",
            f"  Dense   : tp={self.dense.tp_size}  dp={self.dense.dp_size}",
            f"  MoE     : tp={self.moe.tp_size}  ep={self.moe.ep_size}  dp={self.moe.dp_size}",
        ]
        return "\n".join(lines)

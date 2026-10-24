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

"""CuTe DSL entry points for the LongCat fused decoder."""

from tokenspeed_kernel.ops.longcat._cute_dsl.selection import (
    NUM_TOKENS,
    TOPK,
    create_selection_exchange_state,
)
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.attention._o_proj_collective import (
    make_o_proj_collective_workspace,
)
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.attention.indexer_engine import (
    IndexerWeights,
    make_indexer_engine_workspace,
    run_indexer_engine,
)
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.attention.worker_engine import (
    make_worker_engine_workspace,
    run_worker_engine,
)
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.engine._engine_kernel import (
    LongcatEngine,
)
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.engine._plan import (
    plan,
)
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.engine.attention_worker import (
    WorkerShape,
)
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.engine.layer_program import (
    layer_program,
    make_layer_buffers,
)
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.engine.mlp_program import (
    MlpShape,
    mlp_names,
)
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.engine.moe_program import (
    MoeShape,
    make_moe_buffers,
)

HEADS_TOTAL = 64


def engine_head_range(rank: int) -> tuple[int, int]:
    """TP7 head shard of one worker rank.

    Args:
        rank: Worker rank in ``1..7``.

    Returns:
        ``(head_offset, local_heads)`` into the 64 global heads: ranks 1-6
        take 9 heads each, rank 7 the last 10.
    """
    if not 1 <= rank <= 7:
        raise ValueError(f"worker rank must be in 1..7, got {rank}")
    return (rank - 1) * 9, 10 if rank == 7 else 9


__all__ = [
    "HEADS_TOTAL",
    "IndexerWeights",
    "LongcatEngine",
    "MlpShape",
    "MoeShape",
    "NUM_TOKENS",
    "TOPK",
    "WorkerShape",
    "create_selection_exchange_state",
    "engine_head_range",
    "layer_program",
    "make_indexer_engine_workspace",
    "make_layer_buffers",
    "make_moe_buffers",
    "make_o_proj_collective_workspace",
    "make_worker_engine_workspace",
    "mlp_names",
    "plan",
    "run_indexer_engine",
    "run_worker_engine",
]

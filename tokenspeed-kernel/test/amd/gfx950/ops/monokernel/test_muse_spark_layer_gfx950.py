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

"""One Muse Spark monokernel launch against the regular layer's math, on 8
gfx950 ranks.

Each rank prepares the ``muse_spark`` leaf over random weights in the
production layouts (EP1 expert slices from ``make_experts``) and runs one
launch per plan key. The reference composes the regular path's steps on the
same inputs: the regular decode kernel over the FP8 cache (with the forward's
split schedule on full layers, ~141k context), the gated o-norm, the
o_proj all-reduce, the attention boundary, the FP32 router with sigmoid
top-8, the pooler with its norm and MXFP4 quantization, the direct-decode
MoE kernels per rank and their route sum, the routed norm, the post-expert
projection, the FFN boundary, and the next layer's QKV projection and
``gqa_prologue``. Routing is checked untied on the reference logits, so both
sides pick the same experts.

The monokernel's sums run in a different order (deterministic, not bitwise), and the
pooled rows feed an MXFP4 quantization, where an order difference can move a
value across a rounding boundary. So outputs are compared by relative RMS
error (one BF16 rounding per reordered step, with headroom for those rare
MXFP4 steps), and bitwise only where both sides compute alike: a full
layer's gated attention rows, across ranks, across repeated launches, and
between replay and eager.
"""

import socket
import traceback

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tokenspeed_kernel.platform import current_platform

WORLD = 8
if not (
    torch.cuda.is_available()
    and torch.cuda.device_count() >= WORLD
    and current_platform().is_cdna4
):
    pytest.skip("needs 8 gfx950 GPUs", allow_module_level=True)

from muse_layer_builder import (  # noqa: E402
    EPS,
    HEAD_DIM,
    Q_HEADS,
    Q_WIDTH,
    TRAITS,
    build_layer_inputs,
)
from muse_layer_reference import (  # noqa: E402
    SEEDS,
    RoutingTie,
    assert_matches,
    attention_reference,
    failure_records,
    phase_errors,
    regular_reference,
    run_fused,
)
from tokenspeed_kernel.ops.layernorm.fused_norm import (  # noqa: E402
    weightless_rmsnorm_gated_sigmoid,
)
from tokenspeed_kernel.ops.monokernel import (  # noqa: E402
    PlanKey,
    prepare_monokernel_layer,
    prepare_monokernel_state,
)
from tokenspeed_kernel.ops.monokernel._gluon.muse_spark.iris import (  # noqa: E402
    BOX_WORDS,
)

# Cached tokens per request before the verify rows. Full layers read the
# whole context: one request at ~141k (the served context), and two, whose
# split schedule then gives workgroups two items.
CONTEXT = {"sliding": (1500, 2600), "full": (141_000, 141_000)}
PLAN_KEYS = [
    PlanKey(requests, kind, next_kind)
    for kind in ("sliding", "full")
    for requests in (1, 2)
    for next_kind in ("sliding", "full", "regular")
]


def _check_attention_box(state, inputs):
    """The launch's attention output area (``box``: the gated-normed
    attention rows the o_proj senders read) holds the right rows; for a full
    layer, the split-schedule attention followed by the gated norm must equal
    the regular decode kernel and gated-norm kernel bit for bit (the area
    stores -0.0 as +0.0: its empty word is the -0.0 pair)."""
    q, gate = inputs["prepared_query"]
    rows = q.shape[0]
    expected = weightless_rmsnorm_gated_sigmoid(
        attention_reference(inputs), gate, Q_HEADS, HEAD_DIM, EPS, False
    )
    # This launch's copy of the area (epoch & 1), as BF16 rows.
    epoch = int(state.epochs[0])
    box = state.iris.box.view(2, BOX_WORDS)[epoch & 1]
    actual = box.view(torch.bfloat16).view(-1, Q_WIDTH)[:rows]

    def positive_zero(values):
        return torch.where(values == 0, torch.zeros_like(values), values)

    assert torch.equal(
        positive_zero(actual).view(torch.int16),
        positive_zero(expected).view(torch.int16),
    )


def _worker(rank, port, record_dir, errors):
    try:
        torch.cuda.set_device(rank)
        dist.init_process_group(
            "nccl", init_method=f"tcp://localhost:{port}", rank=rank, world_size=WORLD
        )
        state = prepare_monokernel_state(
            recipe="muse_spark",
            group=dist.group.WORLD,
            rank_in_group=rank,
            max_requests=2,
            failure_record_path=f"{record_dir}/rank{rank}.bin",
            device=torch.device("cuda", rank),
        )
        for key in PLAN_KEYS:
            for seed in SEEDS:
                weights, inputs = build_layer_inputs(
                    key, rank, seed, torch.device("cuda", rank),
                    CONTEXT[key.layer_kind][: key.requests],
                )  # fmt: skip
                try:
                    expected = regular_reference(key, weights, inputs)
                    break
                except RoutingTie:
                    continue
            else:
                raise AssertionError(f"{key}: every seed has a routing tie")
            fused = prepare_monokernel_layer(
                recipe="muse_spark", plan_keys=[key], traits=TRAITS,
                weights=weights, state=state,
            )  # fmt: skip
            assert fused is not None
            assert fused.decline_reasons(key.requests) == [], key
            expected["slots"] = inputs["slots"]
            actual = run_fused(fused, key, inputs, inputs["residual"].clone())
            if key.layer_kind == "full":
                _check_attention_box(state, inputs)
            try:
                assert_matches(key, actual, expected)
            except AssertionError as error:
                rows = inputs["residual"].shape[0]
                raise AssertionError(
                    f"{error}; phases {phase_errors(state, expected, rows)}"
                ) from error

            # Deterministic: the same launch again, at the next epoch and
            # the other copy of every two-copy area, gives the same bits.
            again = run_fused(fused, key, inputs, inputs["residual"].clone())
            for name, value in actual.items():
                assert torch.equal(
                    value.view(torch.uint8), again[name].view(torch.uint8)
                ), (key, name)

            # Every rank holds the same residual.
            gathered = [torch.empty_like(actual["residual"]) for _ in range(WORLD)]
            dist.all_gather(gathered, actual["residual"])
            assert all(torch.equal(r, actual["residual"]) for r in gathered), key

            # Graph replay equals eager.
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = run_fused(fused, key, inputs, inputs["residual"].clone())
            graph.replay()
            torch.cuda.synchronize()
            for name, value in actual.items():
                assert torch.equal(
                    value.view(torch.uint8), captured[name].view(torch.uint8)
                ), (key, name)
    except Exception:
        errors[rank] = traceback.format_exc()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def test_monokernel_matches_the_regular_layer(tmp_path):
    with socket.socket() as sock:
        sock.bind(("", 0))
        port = sock.getsockname()[1]
    errors = mp.Manager().dict()
    try:
        mp.spawn(_worker, args=(port, str(tmp_path), errors), nprocs=WORLD, join=True)
    except Exception as error:
        # A kernel's fail_stop traps; its record survives in the host files.
        raise AssertionError(failure_records(tmp_path, WORLD)) from error
    assert not errors, "\n".join(f"rank {r}: {e}" for r, e in errors.items())

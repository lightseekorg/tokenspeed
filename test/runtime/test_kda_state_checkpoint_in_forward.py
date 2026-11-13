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

"""KDA layers write the in-forward State checkpoint the real scheduler plans.

With ``state_checkpoint_in_forward`` a 400-token prompt under a 256-token
budget keeps its chunks [0, 256) and [256, 400); the second spans the page
boundary 384 before the final token, and each of the three KDA State groups
also stores its state there into a third block. That block must hold exactly
what a forward ending at 384 leaves, the spanning forward's own output and
final state must not change, and a later prompt that keeps the first 390
tokens resumes from it at 384.
"""

from __future__ import annotations

import os
import sys

_TEST_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _TEST_DIR)
sys.path.insert(0, os.path.dirname(_TEST_DIR))
import pytest
import torch
from ci_system.ci_register import register_cuda_ci

ts = pytest.importorskip("tokenspeed_scheduler")

from test.runtime.conftest import (  # noqa: E402
    KIMI_STATE_GROUPS,
    block_tables_for,
    requires_cuda,
)
from test.runtime.scheduler_driver import make_extend_result_event  # noqa: E402
from test.runtime.test_kimi_k3_kda import (  # noqa: E402
    _LOWER_BOUND,
    _assert_close,
    _backend_config,
    _stub_contract,
    _StubContractPool,
)

from tokenspeed.runtime.engine.scheduler_utils import pool_to_cache_groups  # noqa: E402
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode  # noqa: E402
from tokenspeed.runtime.layers.attention.backends.state.kda import (  # noqa: E402
    KdaAttnBackend,
    resolve_kda_backend,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (  # noqa: E402
    STATE_CHECKPOINT_SLOT,
    STATE_INPUT_SLOT,
    STATE_LIVE_SLOT,
    validate_state_checkpoint_in_forward,
)

register_cuda_ci(est_time=60, suite="runtime-1gpu")

PAGE = 128
BUDGET = 256
PROMPT = 400
BOUNDARY = 384  # the last page boundary before the final token
HEADS = 4
DIM = 128
WIDTH = 4
LAYERS = (0, 1, 2)
COMPONENTS = ("conv_state", "recurrent_state")


def _prefill_solution() -> str:
    """The prefill backend a server resolves from ``--kda-backend auto``."""
    solution = resolve_kda_backend("auto")
    if solution == "fla":
        pytest.importorskip("fla")
    return solution


def _layer_params(seed: int) -> dict[int, dict[str, torch.Tensor]]:
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape, dtype=torch.bfloat16):
        return torch.randn(*shape, generator=generator, device="cuda", dtype=dtype)

    return {
        layer: dict(
            conv_weights=randn(3 * HEADS * DIM, WIDTH) * 0.1,
            bias=randn(3 * HEADS * DIM) * 0.1,
            A_log=randn(HEADS, dtype=torch.float32) * 0.1,
            dt_bias=randn(HEADS * DIM, dtype=torch.float32) * 0.1,
        )
        for layer in LAYERS
    }


def _token_streams(seed: int, tokens: int) -> dict[int, dict[str, torch.Tensor]]:
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape):
        return torch.randn(
            *shape, generator=generator, device="cuda", dtype=torch.bfloat16
        )

    return {
        layer: dict(
            mixed=randn(tokens, 3 * HEADS * DIM),
            g_raw=randn(tokens, HEADS * DIM) * 0.5,
            beta_raw=randn(tokens, HEADS),
        )
        for layer in LAYERS
    }


class _Server:
    """A real scheduler and a KDA backend over a three-group contract pool."""

    def __init__(self, params, *, in_forward: bool):
        self.params = params
        self.contract = _stub_contract(prefix_granularity=PAGE, usable_pages=32)
        self.pool = _StubContractPool(
            self.contract,
            "cuda",
            conv_dim=3 * HEADS * DIM,
            width=WIDTH,
            num_heads=HEADS,
            head_dim=DIM,
        )
        self.backend = KdaAttnBackend(
            *_backend_config("cuda"),
            enable_prefill_graph=False,
            kda_backend=_prefill_solution(),
        )
        self.backend.set_kv_pool(self.pool)
        self.backend.init_cuda_graph_state(max_bs=4)
        config = ts.SchedulerConfig()
        config.prefix_granularity = PAGE
        config.num_device_pages = 33
        config.num_host_pages = 0
        config.num_snapshot_pages = 1
        config.max_retracted_requests = 0
        config.max_scheduled_tokens = BUDGET
        config.max_batch_size = 1
        config.decode_input_tokens = 1
        config.overlap_schedule_depth = 0
        config.enable_mixed_prefill_decode = False
        config.disable_l2_cache = True
        config.cache_groups = pool_to_cache_groups(self.pool)
        config.state_checkpoint_in_forward = in_forward
        self.scheduler = ts.Scheduler(config)

    def submit(self, request_id: str, tokens: list[int]):
        request = ts.RequestSpec()
        request.request_id = request_id
        request.tokens = tokens
        request.max_new_tokens = 1
        self.scheduler.submit_requests([request])

    def finish(self, request_id: str):
        event = ts.ForwardEvent.Finish()
        event.request_id = request_id
        self.scheduler.advance(ts.ExecutionEvent().add_event(event))
        self.scheduler.next_execution_plan()

    def prefill(self, streams, *, bucket=None):
        """Plan, run and land one prefill forward of one request.

        ``bucket`` pads each layer's inputs to that many rows, as a prefill
        graph's eager break does. Returns ``(begin, end, checkpoint offsets,
        tables, per-layer outputs)``, each output ``[rows, heads * head_dim]``.
        """
        plan = self.scheduler.next_execution_plan()
        (batch,) = [batch for batch in plan.forward if batch.request_ids]
        assert batch.num_extends() == 1
        begin = int(batch.extend_prefix_lens[0])
        end = begin + int(batch.input_lengths[0])
        tables = {
            group: [list(row) for row in rows]
            for group, rows in batch.block_tables.items()
        }
        lengths = torch.tensor([end - begin], dtype=torch.int32)
        prefixes = torch.tensor([begin], dtype=torch.int32)
        offsets = list(batch.extend_state_checkpoint_lens)
        self.backend.init_forward_metadata(
            bs=1,
            num_extends=1,
            req_pool_indices=torch.tensor(
                list(batch.request_pool_indices), dtype=torch.int32, device="cuda"
            ),
            seq_lens=torch.tensor([end], dtype=torch.int32, device="cuda"),
            forward_mode=ForwardMode.EXTEND,
            block_tables=block_tables_for(self.contract, tables, "cuda"),
            extend_seq_lens=lengths.to("cuda"),
            extend_seq_lens_cpu=lengths,
            extend_prefix_lens=prefixes.to("cuda"),
            extend_prefix_lens_cpu=prefixes,
            extend_replay_lens_cpu=torch.zeros_like(prefixes),
            extend_prompt_lens_cpu=torch.as_tensor(
                batch.prefill_lengths, dtype=torch.int32
            ),
            extend_with_prefix=begin > 0,
            extend_state_checkpoint_lens_cpu=torch.tensor(offsets, dtype=torch.int32),
            query_shard=None,
        )
        outputs = {
            layer: self.forward(layer, streams[layer], begin, end, bs=1, bucket=bucket)
            for layer in LAYERS
        }
        torch.cuda.synchronize()
        (request_id,) = batch.request_ids
        done = end == int(batch.prefill_lengths[0])
        self.scheduler.advance(
            ts.ExecutionEvent().add_event(
                make_extend_result_event(request_id, [7] if done else [], None)
            )
        )
        return begin, end, offsets, tables, outputs

    def forward(self, layer, stream, begin, end, *, bs, bucket=None):
        rows = end - begin if bucket is None else bucket

        def padded(name):
            # Nonzero padding: the backend must scrub it, not rely on zeros.
            return torch.nn.functional.pad(
                stream[name][begin:end], (0, 0, 0, rows - (end - begin)), value=3.0
            )

        return self.backend.forward_extend(
            None,
            None,
            None,
            layer=None,
            out_cache_loc=None,
            token_to_kv_pool=self.pool,
            bs=bs,
            forward_mode=ForwardMode.EXTEND,
            mixed_qkv=padded("mixed"),
            save_kv_cache=True,
            g_raw=padded("g_raw"),
            beta_raw=padded("beta_raw"),
            seq_len=rows,
            conv_weights=self.params[layer]["conv_weights"],
            bias=self.params[layer]["bias"],
            activation="silu",
            key_dim=HEADS * DIM,
            value_dim=HEADS * DIM,
            attention_tp_size=1,
            head_k_dim=DIM,
            head_v_dim=DIM,
            A_log=self.params[layer]["A_log"],
            dt_bias=self.params[layer]["dt_bias"],
            layer_id=layer,
            a=None,
            b=None,
            lower_bound=_LOWER_BOUND,
        ).reshape(rows, -1)

    def states(self, tables, slot):
        """Every layer's ``(conv, recurrent)`` state in slot ``slot`` of its row."""
        return {
            (layer, component): self.pool.get_component(layer, component)[
                tables[group][0][slot]
            ].clone()
            for layer, group in zip(LAYERS, KIMI_STATE_GROUPS, strict=True)
            for component in COMPONENTS
        }


def _prompt(server, request_id, tokens, streams, *, bucket=None):
    """Prefill a whole prompt; returns each forward's ``prefill`` result."""
    server.submit(request_id, tokens)
    forwards = [server.prefill(streams, bucket=bucket)]
    while forwards[-1][1] < len(tokens):
        forwards.append(server.prefill(streams, bucket=bucket))
    return forwards


@requires_cuda
def test_the_spanning_forward_stores_what_a_forward_ending_there_leaves():
    params = _layer_params(11)
    streams = _token_streams(12, PROMPT)
    tokens = list(range(PROMPT))

    server = _Server(params, in_forward=True)
    first, second = _prompt(server, "x", tokens, streams)
    assert first[:3] == (0, BUDGET, [0])
    assert second[:3] == (BUDGET, PROMPT, [BOUNDARY - BUDGET]), "no extra chunk"
    tables = second[3]
    assert all(len(tables[group][0]) == 3 for group in KIMI_STATE_GROUPS)

    # A forward that ends at the boundary with the same chunking: its live
    # slot is the reference, bit for bit.
    reference = _Server(params, in_forward=False)
    *_, ending = _prompt(reference, "r", tokens[:BOUNDARY], streams)
    expected = reference.states(ending[3], STATE_LIVE_SLOT)
    for key, state in server.states(tables, STATE_CHECKPOINT_SLOT).items():
        assert torch.equal(state, expected[key]), key

    # Storing the checkpoint leaves the spanning forward's own output and
    # final state as they are without it.
    plain = _Server(params, in_forward=False)
    *_, unchanged = _prompt(plain, "p", tokens, streams)
    assert unchanged[:3] == (BUDGET, PROMPT, [0])
    for layer in LAYERS:
        assert torch.equal(second[4][layer], unchanged[4][layer])
    live = plain.states(unchanged[3], STATE_LIVE_SLOT)
    for key, state in server.states(tables, STATE_LIVE_SLOT).items():
        assert torch.equal(state, live[key]), key

    # A prompt that keeps the first 390 tokens resumes at the boundary from
    # the checkpoint blocks, and its outputs there agree with the cold run's.
    server.finish("x")
    follow = _token_streams(13, PROMPT)
    for layer in LAYERS:
        for name, values in follow[layer].items():
            values[:390] = streams[layer][name][:390]
    server.submit("y", tokens[:390] + list(range(1000, 1000 + PROMPT - 390)))
    begin, end, offsets, hit_tables, resumed = server.prefill(follow)
    assert (begin, end, offsets) == (BOUNDARY, PROMPT, [0])
    for group in KIMI_STATE_GROUPS:
        assert (
            hit_tables[group][0][STATE_INPUT_SLOT]
            == tables[group][0][STATE_CHECKPOINT_SLOT]
        )
    for layer in LAYERS:
        _assert_close(
            resumed[layer][: 390 - BOUNDARY],
            second[4][layer][BOUNDARY - BUDGET : 390 - BUDGET],
            f"layer {layer} resumed output",
        )


@requires_cuda
def test_a_forward_stores_the_checkpoints_of_several_rows():
    # Two fresh rows checkpoint 256 and 128 tokens in. Each checkpoint block
    # holds what a one-row forward of that row ending there leaves.
    params = _layer_params(21)
    lengths, offsets = [300, 200], [256, 128]
    streams = _token_streams(22, sum(lengths))
    server = _Server(params, in_forward=True)
    tables = {group: [[0, 1, 2], [0, 3, 4]] for group in KIMI_STATE_GROUPS}
    tables["full_attention"] = [[1, 2, 3], [4, 5, 0]]
    lengths_cpu = torch.tensor(lengths, dtype=torch.int32)
    server.backend.init_forward_metadata(
        bs=2,
        num_extends=2,
        req_pool_indices=torch.arange(2, dtype=torch.int32, device="cuda"),
        seq_lens=lengths_cpu.to("cuda"),
        forward_mode=ForwardMode.EXTEND,
        block_tables=block_tables_for(server.contract, tables, "cuda"),
        extend_seq_lens=lengths_cpu.to("cuda"),
        extend_seq_lens_cpu=lengths_cpu,
        extend_prefix_lens=torch.zeros(2, dtype=torch.int32, device="cuda"),
        extend_prefix_lens_cpu=torch.zeros(2, dtype=torch.int32),
        extend_replay_lens_cpu=torch.zeros(2, dtype=torch.int32),
        extend_prompt_lens_cpu=lengths_cpu,
        extend_with_prefix=False,
        extend_state_checkpoint_lens_cpu=torch.tensor(offsets, dtype=torch.int32),
        query_shard=None,
    )
    for layer in LAYERS:
        server.forward(layer, streams[layer], 0, sum(lengths), bs=2)
    torch.cuda.synchronize()

    starts = [0, lengths[0]]
    for row, (start, offset) in enumerate(zip(starts, offsets)):
        alone = _Server(params, in_forward=False)
        row_streams = {
            layer: {
                name: values[start : start + offset]
                for name, values in streams[layer].items()
            }
            for layer in LAYERS
        }
        *_, ending = _prompt(alone, "r", list(range(offset)), row_streams)
        expected = alone.states(ending[3], STATE_LIVE_SLOT)
        actual = server.states(
            {group: [rows[row]] for group, rows in tables.items()},
            STATE_CHECKPOINT_SLOT,
        )
        for (layer, component), state in actual.items():
            if component == "conv_state":
                assert torch.equal(state, expected[layer, component])
            else:
                _assert_close(
                    state,
                    expected[layer, component],
                    f"row {row} layer {layer} checkpoint",
                )


@requires_cuda
def test_a_graph_bucket_pads_the_spanning_forward():
    # A prefill graph runs the spanning forward's KDA layers at their eager
    # breaks over the whole bucket: the real rows and every state match the
    # unpadded run, and the padded rows are zero.
    params = _layer_params(41)
    streams = _token_streams(42, PROMPT)
    tokens = list(range(PROMPT))
    padded = _Server(params, in_forward=True)
    _, spanning = _prompt(padded, "x", tokens, streams, bucket=BUDGET)
    exact = _Server(params, in_forward=True)
    _, unpadded = _prompt(exact, "x", tokens, streams)
    assert spanning[:3] == unpadded[:3] == (BUDGET, PROMPT, [BOUNDARY - BUDGET])

    real = PROMPT - BUDGET
    for layer in LAYERS:
        assert spanning[4][layer].shape[0] == BUDGET
        assert torch.equal(spanning[4][layer][:real], unpadded[4][layer])
        assert not spanning[4][layer][real:].any()
    for slot in (STATE_CHECKPOINT_SLOT, STATE_LIVE_SLOT):
        expected = exact.states(unpadded[3], slot)
        for key, state in padded.states(spanning[3], slot).items():
            assert torch.equal(state, expected[key]), (slot, key)


@requires_cuda
def test_startup_accepts_the_option_for_kda_state_groups():
    server = _Server(_layer_params(31), in_forward=True)
    assert server.backend.stores_in_forward_state_checkpoints
    validate_state_checkpoint_in_forward(
        attn_backend=server.backend, kv_pool=server.pool
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

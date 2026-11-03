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

"""A GDN layer writes the in-forward State checkpoint the real scheduler plans.

With ``state_checkpoint_in_forward`` a 400-token prompt under a 256-token
budget keeps its chunks [0, 256) and [256, 400); the second spans the page
boundary 384 before the final token and also stores the state there into a
third block. That block must hold exactly what a forward ending at 384 leaves
-- the conv window and the recurrent state the same chunking writes into slot
1 -- and a later prompt that keeps the first 390 tokens resumes from it at 384.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from test.runtime.test_gdn_state_paging import (
    _ContractPool,
    _extend_kwargs,
    _mamba_config_pair,
)

from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="runtime-1gpu")

PAGE = 64
BUDGET = 256
PROMPT = 400
BOUNDARY = 384  # the last page boundary before the final token


def _scheduler(ts, make_config, *, in_forward: bool):
    groups = [
        ts.CacheGroupConfig(
            group_id="history",
            block_granularity=PAGE,
            total_pages=65,
            attn_kind=ts.AttnKind.Full,
        ),
        ts.CacheGroupConfig(
            group_id="linear_attention",
            block_granularity=0,
            total_pages=65,
            attn_kind=ts.AttnKind.State,
        ),
    ]
    return ts.Scheduler(
        make_config(
            num_device_pages=65,
            max_scheduled_tokens=BUDGET,
            max_batch_size=4,
            prefix_granularity=PAGE,
            num_host_pages=0,
            disable_l2_cache=True,
            enable_l3_storage=False,
            role="fused",
            num_snapshot_pages=1,
            host_group_packing=(),
            max_retracted_requests=0,
            host_write_policy="write_through",
            storage_config=None,
            cache_groups=groups,
            state_checkpoint_in_forward=in_forward,
        )
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
class TestGdnStateCheckpointInForward:
    H = 16
    D = 128
    WIDTH = 4

    def setup_method(self):
        try:
            from test.runtime.scheduler_driver import (
                advance_scheduler,
                make_extend_result_event,
            )

            import tokenspeed_scheduler as ts
            from tokenspeed_kernel.ops.attention.gdn import (
                gdn_chunk_prefill_stores_state_checkpoints,
            )

            from tokenspeed.runtime.engine.scheduler_utils import make_config
            from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
            from tokenspeed.runtime.layers.attention.backends.state.gdn import (
                GdnAttnBackend,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            pytest.skip(f"needs tokenspeed_scheduler + tokenspeed_kernel: {exc}")
        if not gdn_chunk_prefill_stores_state_checkpoints(
            torch.bfloat16,
            head_dim=self.D,
            value_head_dim=self.D,
            num_q_heads=self.H,
            num_v_heads=self.H,
            qk_l2norm=True,
            interval_granularity=PAGE,
        ):
            pytest.skip("no GDN prefill kernel stores state checkpoints here")
        self.ts = ts
        self.ForwardMode = ForwardMode
        self.GdnAttnBackend = GdnAttnBackend
        self.advance_scheduler = advance_scheduler
        self.make_config = make_config
        self.make_extend_result_event = make_extend_result_event
        torch.manual_seed(0)
        h, d = self.H, self.D
        channels = 3 * h * d
        self.raw = torch.randn(PROMPT, channels, device="cuda", dtype=torch.bfloat16)
        self.a = torch.randn(PROMPT, h, device="cuda", dtype=torch.float32)
        self.b = torch.randn_like(self.a)
        self.common = dict(
            conv_weights=torch.randn(
                channels, self.WIDTH, device="cuda", dtype=torch.bfloat16
            )
            * 0.1,
            bias=torch.randn(channels, device="cuda", dtype=torch.bfloat16) * 0.1,
            activation="silu",
            key_dim=h * d,
            value_dim=h * d,
            attention_tp_size=1,
            head_k_dim=d,
            head_v_dim=d,
            A_log=torch.randn(h, device="cuda", dtype=torch.float32) * 0.1,
            dt_bias=torch.randn(h, device="cuda", dtype=torch.float32) * 0.1,
            layer_id=0,
        )

    def _backend(self, state_dtype):
        conv = torch.zeros(
            65, 3 * self.H * self.D, self.WIDTH - 1, device="cuda", dtype=torch.bfloat16
        )
        recurrent = torch.zeros(
            65, self.H, self.D, self.D, device="cuda", dtype=state_dtype
        )
        backend = self.GdnAttnBackend(
            *_mamba_config_pair(
                torch, heads=self.H, head_dim=self.D, max_bs=4, device="cuda"
            )
        )
        backend.set_kv_pool(_ContractPool({0: ("linear_attention", conv, recurrent)}))
        backend.init_cuda_graph_state(max_bs=4)
        return backend, conv, recurrent

    def _plan(self, scheduler, backend):
        """Plan one prefill forward and build its metadata; returns (op, row)."""
        plan = scheduler.next_execution_plan()
        (op,) = [op for op in plan.forward if op.request_ids]
        assert op.num_extends() == 1
        before = int(op.extend_prefix_lens[0])
        length = int(op.input_lengths[0])
        rows = torch.as_tensor(
            op.block_tables_arrays()["linear_attention"], device="cuda"
        )
        self._init_metadata(
            backend,
            op.request_pool_indices,
            before,
            length,
            rows,
            list(op.extend_state_checkpoint_lens),
        )
        return op, [int(v) for v in rows[0].tolist()]

    def _init_metadata(self, backend, pool_indices, before, length, rows, ckpt):
        backend.init_forward_metadata(
            bs=1,
            num_extends=1,
            req_pool_indices=torch.tensor(
                list(pool_indices), dtype=torch.int32, device="cuda"
            ),
            seq_lens=torch.tensor([before + length], dtype=torch.int32, device="cuda"),
            forward_mode=self.ForwardMode.EXTEND,
            block_tables={"linear_attention": rows},
            extend_state_checkpoint_lens_cpu=torch.tensor(ckpt, dtype=torch.int32),
            **_extend_kwargs(
                torch,
                torch.tensor([length], dtype=torch.int32),
                torch.tensor([before], dtype=torch.int32),
                "cuda",
            ),
        )

    def _forward(self, backend, raw, a, b):
        return backend.forward_extend(
            None,
            None,
            None,
            layer=None,
            out_cache_loc=None,
            token_to_kv_pool=backend.kv_pool,
            bs=1,
            forward_mode=self.ForwardMode.EXTEND,
            mixed_qkv=raw,
            a=a,
            b=b,
            seq_len=raw.shape[0],
            **self.common,
        )

    def _land(self, scheduler, op, inputs):
        (rid,) = op.request_ids
        done = int(op.extend_prefix_lens[0]) + int(op.input_lengths[0]) == len(
            inputs[0]
        )
        self.advance_scheduler(
            scheduler, [self.make_extend_result_event(rid, [7] if done else [])]
        )

    def _prefill(self, scheduler, backend, inputs):
        """Plan, run and land one prefill forward; returns (op, row, output)."""
        op, row = self._plan(scheduler, backend)
        before = int(op.extend_prefix_lens[0])
        end = before + int(op.input_lengths[0])
        output = self._forward(backend, *(t[before:end] for t in inputs))
        self._land(scheduler, op, inputs)
        return op, row, output

    def _reference(self, state_dtype, inputs):
        """The state a forward ending at the boundary leaves, with the same chunking.

        Returns the live slot's ``(conv, recurrent)`` state.
        """
        reference = _scheduler(self.ts, self.make_config, in_forward=False)
        backend, conv, recurrent = self._backend(state_dtype)
        self._submit(reference, "r", BOUNDARY)
        self._prefill(reference, backend, inputs)
        last, row, _ = self._prefill(reference, backend, inputs)
        assert list(last.extend_state_checkpoint_lens) == [0]
        torch.cuda.synchronize()
        return conv[row[1]].clone(), recurrent[row[1]].clone()

    def _submit(self, scheduler, rid, length):
        spec = self.ts.RequestSpec()
        spec.request_id = rid
        spec.tokens = list(range(length))
        spec.max_new_tokens = 1
        scheduler.submit_requests([spec])

    def _finish(self, scheduler, rid):
        event = self.ts.ForwardEvent.Finish()
        event.request_id = rid
        self.advance_scheduler(scheduler, [event])
        scheduler.next_execution_plan()

    def test_the_spanning_forward_stores_what_a_forward_ending_there_leaves(
        self, state_dtype
    ):
        inputs = (self.raw, self.a, self.b)
        scheduler = _scheduler(self.ts, self.make_config, in_forward=True)
        backend, conv, recurrent = self._backend(state_dtype)
        self._submit(scheduler, "x", PROMPT)
        first, _, _ = self._prefill(scheduler, backend, inputs)
        second, row, cold_output = self._prefill(scheduler, backend, inputs)
        assert list(first.extend_state_checkpoint_lens) == [0]
        assert (int(second.extend_prefix_lens[0]), int(second.input_lengths[0])) == (
            BUDGET,
            PROMPT - BUDGET,
        ), "no extra chunk"
        assert list(second.extend_state_checkpoint_lens) == [BOUNDARY - BUDGET]
        assert len(row) == 3
        checkpoint_block = row[2]

        # A forward that ends at the boundary with the same chunking: its
        # live slot is the reference, bit for bit.
        ref_conv, ref_recurrent = self._reference(state_dtype, inputs)
        assert torch.equal(conv[checkpoint_block], ref_conv)
        assert torch.equal(recurrent[checkpoint_block], ref_recurrent)

        # A prompt that keeps the first 390 tokens resumes at the boundary
        # from the checkpoint block, and its outputs there agree with the
        # cold run's to the stored state's rounding.
        self._finish(scheduler, "x")
        follow = tuple(t.clone() for t in inputs)
        for tensor in follow:
            tensor[390:] = tensor[390:].flip(0)
        self._submit(scheduler, "y", PROMPT)
        hit, hit_row, resumed = self._prefill(scheduler, backend, follow)
        assert list(hit.extend_prefix_lens) == [BOUNDARY]
        assert hit_row[0] == checkpoint_block
        torch.cuda.synchronize()
        difference = (
            resumed[:, : 390 - BOUNDARY].float()
            - cold_output[:, BOUNDARY - BUDGET : 390 - BUDGET].float()
        ).abs()
        assert difference.max().item() < (
            1e-2 if state_dtype == torch.float32 else 1e-1
        )

    def _retain_capture_shape(self, backend, tokens):
        """Retain the capture metadata for one ``tokens``-token request.

        This is what a prefill graph captures at startup, from a dummy batch on
        spare blocks.
        """
        if not backend.admits_prefill_graph(tokens, 1, self.ForwardMode.EXTEND):
            pytest.skip("the selected GDN scan is not graph-capturable here")
        spare = torch.tensor([[61, 62]], dtype=torch.int32, device="cuda")
        self._init_metadata(backend, [3], 0, tokens, spare, [0])
        assert backend.prepare_prefill_metadata(
            tokens, 1, self.ForwardMode.EXTEND, capture=True
        )

    def test_a_captured_shape_keeps_the_checkpoint_forwards_live_metadata(
        self, state_dtype
    ):
        # The second chunk (144 tokens, one request) has a startup-captured
        # GDN shape. Its eager forward still prepares that shape's metadata,
        # which carries no checkpoint: the forward keeps its own instead and
        # stores the checkpoint.
        inputs = (self.raw, self.a, self.b)
        scheduler = _scheduler(self.ts, self.make_config, in_forward=True)
        backend, conv, recurrent = self._backend(state_dtype)
        self._retain_capture_shape(backend, PROMPT - BUDGET)
        self._submit(scheduler, "x", PROMPT)
        self._prefill(scheduler, backend, inputs)
        op, row = self._plan(scheduler, backend)
        assert list(op.extend_state_checkpoint_lens) == [BOUNDARY - BUDGET]
        live = backend.forward_metadata
        assert not backend.prepare_prefill_metadata(
            PROMPT - BUDGET, 1, self.ForwardMode.EXTEND, capture=False
        )
        assert backend.forward_metadata is live
        self._forward(backend, *(t[BUDGET:] for t in inputs))
        ref_conv, ref_recurrent = self._reference(state_dtype, inputs)
        assert torch.equal(conv[row[2]], ref_conv)
        assert torch.equal(recurrent[row[2]], ref_recurrent)

    def test_a_checkpoint_forward_replays_with_gdn_at_its_eager_break(
        self, state_dtype
    ):
        # The ordinary prefill graph keeps GDN at an eager break that reads
        # the live metadata: a checkpoint forward replayed into a padded
        # 256-token capture stores what a forward ending at the boundary does.
        from tokenspeed.runtime.execution.breakable_cuda_graph import (
            BreakableCapture,
            break_point,
        )

        inputs = (self.raw, self.a, self.b)
        scheduler = _scheduler(self.ts, self.make_config, in_forward=True)
        backend, conv, recurrent = self._backend(state_dtype)
        bucket = 256
        static = [
            torch.zeros(bucket, *t.shape[1:], dtype=t.dtype, device="cuda")
            for t in inputs
        ]

        @break_point
        def gdn(raw, a, b):
            return self._forward(backend, raw, a, b)

        def layer():
            # Captured work on either side of the break.
            return gdn(static[0] * 1, static[1] * 1, static[2] * 1) * 1

        spare = torch.tensor([[61, 62]], dtype=torch.int32, device="cuda")
        self._init_metadata(backend, [3], 0, bucket, spare, [0])
        capture = BreakableCapture()
        with capture:
            layer()
        assert capture.num_segments == 3
        # The capture ran the break on the dummy batch's spare blocks.
        conv.zero_()
        recurrent.zero_()

        self._submit(scheduler, "x", PROMPT)
        self._prefill(scheduler, backend, inputs)
        op, row = self._plan(scheduler, backend)
        assert list(op.extend_state_checkpoint_lens) == [BOUNDARY - BUDGET]
        tokens = PROMPT - BUDGET
        for buffer, tensor in zip(static, inputs):
            buffer[:tokens].copy_(tensor[BUDGET:])
            buffer[tokens:].zero_()
        capture.replay(valid_rows=tokens)
        ref_conv, ref_recurrent = self._reference(state_dtype, inputs)
        assert torch.equal(conv[row[2]], ref_conv)
        assert torch.equal(recurrent[row[2]], ref_recurrent)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_a_checkpoint_row_compiles_no_new_state_input_kernel():
    # An in-forward checkpoint widens a forward's State rows from two entries
    # to three. The block vectors the state kernels take stay unit stride
    # whatever the width, so a one-row batch -- whose table column would carry
    # the table's stride -- compiles no new specialization.
    import importlib.util

    from tokenspeed_kernel.ops.attention.gdn._triton.prefill_state_inputs import (
        _prepare_prefill_state_inputs_kernel,
        prepare_prefill_state_inputs,
    )

    from tokenspeed.runtime.layers.attention.backends.state.checkpoint import (
        compute_state_block_indices,
    )

    kernel_test_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "..",
        "tokenspeed-kernel",
        "test",
    )
    helper_spec = importlib.util.spec_from_file_location(
        "checkpoint_kernel_test_utils", os.path.join(kernel_test_dir, "utils.py")
    )
    helper = importlib.util.module_from_spec(helper_spec)
    helper_spec.loader.exec_module(helper)

    conv = torch.zeros(8, 64, 3, device="cuda", dtype=torch.bfloat16)
    recurrent = torch.zeros(8, 2, 16, 16, device="cuda", dtype=torch.float32)

    def prepare(row: list[int]) -> None:
        state_in, state_out = compute_state_block_indices(
            torch.tensor([row], dtype=torch.int32, device="cuda"),
            torch.tensor([4], dtype=torch.int32, device="cuda"),
            validate=False,
            group_id="linear_attention",
        )
        prepare_prefill_state_inputs(conv, recurrent, state_in, state_out)

    prepare([2, 3])
    with helper.assert_no_triton_compile(_prepare_prefill_state_inputs_kernel):
        prepare([2, 3, 5])
        prepare([4, 6])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("prefix_granularity", [32, 64])
def test_startup_refuses_a_state_consumer_that_cannot_store_the_checkpoint(
    prefix_granularity,
):
    # The scheduler publishes every State group's third block, so startup
    # refuses the option unless each State consumer writes it: GDN only with
    # a checkpointing scan whose chunk divides the prefix granularity, never
    # the families without one.
    from tokenspeed_kernel.ops.attention.gdn import (
        gdn_chunk_prefill_stores_state_checkpoints,
    )

    from tokenspeed.runtime.layers.attention.backends.state.gdn import GdnAttnBackend
    from tokenspeed.runtime.layers.attention.backends.state.mamba import (
        MambaAttnBackend,
    )
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
        validate_state_checkpoint_in_forward,
    )

    if not gdn_chunk_prefill_stores_state_checkpoints(
        torch.bfloat16,
        head_dim=128,
        value_head_dim=128,
        num_q_heads=16,
        num_v_heads=16,
        qk_l2norm=True,
        interval_granularity=64,
    ):
        pytest.skip("no GDN prefill kernel stores state checkpoints here")
    config, spec = _mamba_config_pair(
        torch, heads=16, head_dim=128, max_bs=4, device="cuda"
    )
    conv = torch.zeros(9, 3 * 16 * 128, 3, dtype=torch.bfloat16, device="cuda")
    ssm = torch.zeros(9, 16, 128, 128, dtype=torch.float32, device="cuda")
    pool = _ContractPool({0: ("linear_attention", conv, ssm)})
    # The checkpoints sit at multiples of the bound pool's prefix granularity.
    pool.arena.runtime_contract.prefix_granularity = prefix_granularity

    gdn = GdnAttnBackend(config, spec)
    gdn.set_cache_pool(pool)
    if prefix_granularity % 64 == 0:
        validate_state_checkpoint_in_forward(attn_backend=gdn, kv_pool=pool)
    else:
        with pytest.raises(ValueError, match="cannot store"):
            validate_state_checkpoint_in_forward(attn_backend=gdn, kv_pool=pool)
    with pytest.raises(ValueError, match="cannot store"):
        validate_state_checkpoint_in_forward(
            attn_backend=MambaAttnBackend(config, spec), kv_pool=pool
        )

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

"""A GDN layer resumes from a token-level state hit the real scheduler planned.

The prefix-cache e2e sequence at the layer level: a cold prompt X, then X[:-1]
with one token to generate (its prefill, the decode the scheduler plans before
the prefill lands, the finish), then X again. The scheduler owns every state
row; the backend runs the Triton GDN kernels on them. The repeat of X resumes
at |X| - 1 from the block X[:-1]'s prefill wrote, and its output for the last
prompt token, the state it writes and its first decode agree with the cold run.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import patch

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

PAGE = 16
PROMPT = 70  # |X|: the resume at |X| - 1 is off the page and the kernel chunk
DECODES = 3


def _scheduler(ts, make_config):
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
            max_scheduled_tokens=256,
            max_batch_size=4,
            prefix_granularity=PAGE,
            num_host_pages=0,
            disable_l2_cache=True,
            enable_l3_storage=False,
            role="fused",
            num_snapshot_pages=1,
            max_retracted_requests=0,
            host_write_policy="write_through",
            storage_config=None,
            cache_groups=groups,
        )
    )


@pytest.mark.skipif(not (torch.cuda.is_available()), reason=("requires CUDA"))
class TestGdnTokenLevelHitResumeTest:
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
            from tokenspeed_kernel.ops.attention.gdn.triton import (
                triton_gdn_chunk_prefill,
            )

            import tokenspeed.runtime.layers.attention.backends.state.mamba as mamba
            from tokenspeed.runtime.engine.scheduler_utils import make_config
            from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
        except (ImportError, ModuleNotFoundError) as exc:
            pytest.skip(f"needs tokenspeed_scheduler + tokenspeed_kernel: {exc}")
        self.ts = ts
        self.mamba = mamba
        self.ForwardMode = ForwardMode
        self.advance_scheduler = advance_scheduler
        self.make_config = make_config
        self.make_extend_result_event = make_extend_result_event
        self.triton_prefill = triton_gdn_chunk_prefill
        torch.manual_seed(0)
        h, d = self.H, self.D
        channels = 3 * h * d
        total = PROMPT + DECODES + 1
        self.raw = torch.randn(total, channels, device="cuda", dtype=torch.bfloat16)
        self.a = torch.randn(total, h, device="cuda", dtype=torch.float32)
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

    def _backend(self):
        from tokenspeed.runtime.layers.attention.backends.state.mamba import (
            MambaAttnBackend,
        )

        conv = torch.zeros(
            65, 3 * self.H * self.D, self.WIDTH - 1, device="cuda", dtype=torch.bfloat16
        )
        # bf16 recurrent state: the portable Triton decode serves every device.
        recurrent = torch.zeros(
            65, self.H, self.D, self.D, device="cuda", dtype=torch.bfloat16
        )
        backend = MambaAttnBackend(
            *_mamba_config_pair(
                torch, heads=self.H, head_dim=self.D, max_bs=4, device="cuda"
            )
        )
        backend.set_kv_pool(_ContractPool({0: ("linear_attention", conv, recurrent)}))
        backend.init_cuda_graph_state(max_bs=4)
        return backend, conv, recurrent

    def _run(self, backend, op):
        """Execute one scheduler forward op on the backend; returns its output."""
        (rid,) = op.request_ids
        rows = torch.as_tensor(
            op.block_tables_arrays()["linear_attention"], device="cuda"
        )
        pool_index = torch.tensor(
            list(op.request_pool_indices), dtype=torch.int32, device="cuda"
        )
        if op.num_extends():
            before = int(op.extend_prefix_lens[0])
            length = int(op.input_lengths[0])
            backend.init_forward_metadata(
                bs=1,
                num_extends=1,
                req_pool_indices=pool_index,
                seq_lens=torch.tensor(
                    [before + length], dtype=torch.int32, device="cuda"
                ),
                forward_mode=self.ForwardMode.EXTEND,
                block_tables={"linear_attention": rows},
                **_extend_kwargs(
                    torch,
                    torch.tensor([length], dtype=torch.int32),
                    torch.tensor([before], dtype=torch.int32),
                    "cuda",
                ),
            )
            return backend.forward_extend(
                None,
                None,
                None,
                layer=None,
                out_cache_loc=None,
                token_to_kv_pool=backend.kv_pool,
                bs=1,
                forward_mode=self.ForwardMode.EXTEND,
                mixed_qkv=self.raw[before : before + length],
                a=self.a[before : before + length],
                b=self.b[before : before + length],
                seq_len=length,
                **self.common,
            )
        position = self._computed[rid]
        backend.refresh_decode_metadata(
            1,
            1,
            pool_index,
            torch.tensor([position + 1], dtype=torch.int32, device="cuda"),
            forward_mode=self.ForwardMode.DECODE,
            block_tables={"linear_attention": rows},
        )
        return backend.forward_decode(
            None,
            None,
            None,
            layer=None,
            out_cache_loc=None,
            token_to_kv_pool=backend.kv_pool,
            bs=1,
            # The decode's conv update consumes its input in place; the model
            # projects a fresh one per forward, this harness replays a stream.
            mixed_qkv=self.raw[position : position + 1].clone(),
            a=self.a[position : position + 1],
            b=self.b[position : position + 1],
            **self.common,
        )

    def _submit(self, scheduler, rid, tokens, max_new_tokens):
        spec = self.ts.RequestSpec()
        spec.request_id = rid
        spec.tokens = list(tokens)
        spec.max_new_tokens = max_new_tokens
        scheduler.submit_requests([spec])
        self._computed[rid] = 0

    def _forward(self, scheduler, backend):
        plan = scheduler.next_execution_plan()
        (op,) = [op for op in plan.forward if op.request_ids]
        output = self._run(backend, op)
        (rid,) = op.request_ids
        if op.num_extends():
            self._computed[rid] = int(op.extend_prefix_lens[0]) + int(
                op.input_lengths[0]
            )
        else:
            self._computed[rid] += 1
        return op, output

    def _land(self, scheduler, rid, token):
        self.advance_scheduler(scheduler, [self.make_extend_result_event(rid, [token])])

    def _finish(self, scheduler, rid):
        event = self.ts.ForwardEvent.Finish()
        event.request_id = rid
        self.advance_scheduler(scheduler, [event])

    def test_the_repeat_resumes_from_the_published_prefill_state(self):
        mamba = self.mamba
        from tokenspeed_kernel.selection import kernel_override

        with (
            patch.object(mamba, "gdn_chunk_prefill", self.triton_prefill),
            kernel_override("attention", "gdn_decode_step", "triton"),
        ):
            self._computed = {}
            scheduler = _scheduler(self.ts, self.make_config)
            backend, conv, recurrent = self._backend()
            X = list(range(PROMPT))

            # Cold X: the prefill and the decodes the scheduler plans.
            self._submit(scheduler, "a", X, DECODES)
            cold_prefill, cold_output = self._forward(scheduler, backend)
            cold_state_block = int(
                cold_prefill.block_tables_arrays()["linear_attention"][0][1]
            )
            cold_state = (
                conv[cold_state_block].clone(),
                recurrent[cold_state_block].clone(),
            )
            cold_decodes = []
            for step in range(DECODES):
                op, output = self._forward(scheduler, backend)
                assert (op.num_extends()) == (0)
                cold_decodes.append(output)
                self._land(scheduler, "a", 1000 + step)
            self._finish(scheduler, "a")

            # X[:-1] with one token to generate, as the engine runs it: the
            # prefill (the junction cut ends its first chunk at the cold run's
            # KV pages, so there are two), the decode the scheduler plans
            # before the last chunk lands, then the landing and the finish.
            self._submit(scheduler, "c", X[:-1], 1)
            chunks = []
            while True:
                short_prefill, short_output = self._forward(scheduler, backend)
                assert (short_prefill.num_extends()) == (1)
                begin = int(short_prefill.extend_prefix_lens[0])
                end = begin + int(short_prefill.input_lengths[0])
                torch.testing.assert_close(
                    short_output, cold_output[:, begin:end], atol=1e-1, rtol=1e-2
                )
                chunks.append((begin, end))
                if end == PROMPT - 1:
                    break
                # An intermediate chunk lands no token.
                self.advance_scheduler(
                    scheduler, [self.make_extend_result_event("c", [])]
                )
            assert (chunks[0]) == ((0, 4 * PAGE))
            published = int(
                short_prefill.block_tables_arrays()["linear_attention"][0][1]
            )
            published_state = (conv[published].clone(), recurrent[published].clone())
            early_decode, _ = self._forward(scheduler, backend)
            assert (early_decode.num_extends()) == (0)
            early_row = [
                int(v)
                for v in early_decode.block_tables_arrays()["linear_attention"][0]
            ]
            assert (early_row[0]) == (published)
            assert (early_row[1]) != (published)
            self._land(scheduler, "c", 2000)
            self._finish(scheduler, "c")
            torch.cuda.synchronize()
            # The decode read the published block and wrote its own.
            assert torch.equal(conv[published], published_state[0])
            assert torch.equal(recurrent[published], published_state[1])
            assert (recurrent[early_row[1]].abs().max().item()) > (0.0)

            # X again: resumes at |X| - 1 from the block X[:-1]'s prefill wrote.
            self._submit(scheduler, "b", X, DECODES)
            hit, resumed = self._forward(scheduler, backend)
            assert (list(hit.extend_prefix_lens)) == ([PROMPT - 1])
            assert (list(hit.input_lengths)) == ([1])
            row = [int(v) for v in hit.block_tables_arrays()["linear_attention"][0]]
            assert (row[0]) == (published)
            assert (row[1]) != (published)
            torch.cuda.synchronize()
            # The last prompt token's output and the state after it agree with
            # the cold run to the chunked kernel's rounding.
            difference = (resumed.float() - cold_output[:, PROMPT - 1 :].float()).abs()
            assert (difference.mean().item()) < (1e-3)
            assert (difference.max().item()) < (1e-1)
            assert torch.equal(conv[row[1]], cold_state[0])
            state_difference = (recurrent[row[1]].float() - cold_state[1].float()).abs()
            assert (state_difference.mean().item()) < (1e-3)
            assert (state_difference.max().item()) < (5e-2)
            self._land(scheduler, "b", 1000)

            # Its first decode reads the prompt-end state from slot 0: the
            # block the resumed chunk wrote, or -- the same prefix is already
            # published under that hash by the cold run -- that canonical
            # block, which the publication at this admission dedups to.
            decode, output = self._forward(scheduler, backend)
            assert (decode.num_extends()) == (0)
            decode_row = [
                int(v) for v in decode.block_tables_arrays()["linear_attention"][0]
            ]
            assert (decode_row[0]) in ((row[1], cold_state_block))
            assert (decode_row[1]) not in ((row[1], cold_state_block, published))
            difference = (output.float() - cold_decodes[0].float()).abs()
            assert (difference.mean().item()) < (1e-3)
            assert (difference.max().item()) < (1e-1)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))

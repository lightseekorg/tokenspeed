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

"""CPU-only tests for P-side layerwise KV streaming's device half: the
attention backends publish each producer step on the step counter the
host's transfer workers poll. Which blocks each chunk sends, and when, the
host plans (``ts-pd``)."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

# CPU-only tests scheduled in runtime-1gpu because they import the full runtime.
sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=10, suite="runtime-1gpu")


def test_dsa_sparse_prefill_publishes_one_cache_step_after_cache_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    from tokenspeed.runtime.layers.attention.backends.paged import dsa as dsa_backend

    events = []
    backend = object.__new__(dsa_backend.DSABackend)
    backend.index_topk = 2
    backend.kvp_group = (0,)
    backend.v_head_dim = 1
    backend.data_type = torch.bfloat16
    backend.qk_nope_head_dim = 1
    backend.kv_lora_rank = 1
    backend.qk_rope_head_dim = 0
    backend.kernel_page_size = 64
    backend.kernel_solution = None
    backend.prefill_kernel_plan = None
    backend.decode_kernel_plan = None
    backend.slot_order = "selection"
    backend.num_local_heads = 1
    backend.num_attention_heads = 1
    backend.query_shard_metadata = None
    backend.step_counter = SimpleNamespace(record_cache=lambda: events.append("ready"))

    def fake_dsa_prefill(**_kwargs):
        events.append("attention")
        return torch.ones((1, 1, 1), dtype=torch.bfloat16)

    monkeypatch.setattr(dsa_backend, "dsa_prefill", fake_dsa_prefill)
    output = backend.forward_sparse_prefill(
        q=torch.zeros((1, 1), dtype=torch.bfloat16),
        layer=SimpleNamespace(
            logit_cap=0,
            tp_q_head_num=1,
            v_head_dim=1,
            head_dim=1,
            layer_id=0,
            scaling=1.0,
        ),
        token_to_kv_pool=SimpleNamespace(
            quant_method=None,
            get_key_buffer=lambda _layer_id: torch.zeros(1),
        ),
        kv_seq_lens=torch.ones(1, dtype=torch.int32),
        topk_slots=torch.zeros((1, 2), dtype=torch.int64),
        topk_lens=torch.ones(1, dtype=torch.int32),
        max_seq_len=1,
    )

    assert output.shape == (1, 1)
    assert events == ["attention", "ready"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

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


from __future__ import annotations

import pytest
import torch
from flashinfer.autotuner import AutoTuner
from tokenspeed_kernel.ops.quantization import quantize_fp8
from tokenspeed_kernel.ops.tuning import autotune, get_autotune_max_num_tokens
from tokenspeed_kernel.selection import kernel_override
from tokenspeed_kernel.thirdparty.trtllm_blockwise.tuning import (
    CuteBlockFp8Runner,
    ExistingBlockFp8Runner,
    _configuration,
)
from transformers import PretrainedConfig

from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.layers.quantization.fp8 import Fp8Config
from tokenspeed.runtime.models.deepseek_v3 import DeepseekV3AttentionMLA
from tokenspeed.runtime.utils.env import global_server_args_dict

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="requires Blackwell",
)


def test_history_projection_tunes_beyond_new_tokens_and_reloads(monkeypatch, tmp_path):
    maximum = get_autotune_max_num_tokens()
    history_rows = maximum + 64
    monkeypatch.setitem(
        global_server_args_dict, "chunked_prefill_size", history_rows // 4
    )
    monkeypatch.setitem(global_server_args_dict, "mla_chunk_multiplier", 4)
    torch.manual_seed(19)
    with torch.device("cuda"):
        attention = DeepseekV3AttentionMLA(
            config=PretrainedConfig(rms_norm_eps=1e-6),
            mapping=Mapping(rank=0, world_size=1),
            hidden_size=128,
            num_heads=2,
            qk_nope_head_dim=64,
            qk_rope_head_dim=64,
            v_head_dim=64,
            q_lora_rank=128,
            kv_lora_rank=512,
            quant_config=Fp8Config(),
            layer_id=0,
            skip_rope=True,
            reduce_attn_results=False,
        )
    layer = attention.kv_b_proj
    layer.weight.data.normal_(0, 0.02)
    layer.quant_method.process_weights_after_loading(layer)
    x = torch.randn(history_rows, 512, device="cuda", dtype=torch.bfloat16)
    tuner = AutoTuner.get()
    tuner.clear_cache()

    def lookup(rows):
        current = x[:rows]
        values, scales = quantize_fp8(
            current, granularity="token_group", group_size=128, solution="triton"
        )
        output = torch.empty(rows, 256, device="cuda", dtype=torch.bfloat16)
        inputs = [values, layer.weight, scales, layer.weight_scale_inv, output]
        return tuner.search_cache(
            "tokenspeed::block_fp8_gemm",
            [ExistingBlockFp8Runner(), CuteBlockFp8Runner()],
            tuple(tuple(t.shape) for t in inputs),
            _configuration(maximum),
            inputs=inputs,
        )

    with autotune(tune_mode=True, tuning_buckets=None, round_up=None):
        layer(x[:maximum])
    assert lookup(maximum)[0]
    assert not lookup(history_rows)[0]
    with autotune(tune_mode=True, tuning_buckets=None, round_up=None):
        attention._project_expanded_kv(x[:maximum])
    assert get_autotune_max_num_tokens() == maximum

    def verify(rows):
        current = x[:rows]
        hit, _, tactic, _ = lookup(rows)
        assert hit and tactic != -1, f"No tuned history projection at {rows} rows"
        with kernel_override("gemm", "mm", "flashinfer_mm_fp8_blockscale"):
            expected = attention._project_expanded_kv(current)
        actual = attention._project_expanded_kv(current)
        torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)
        return actual

    before = verify(history_rows).clone()
    verify(history_rows - 1)
    verify(maximum)
    saved = tmp_path / "history-tactics.json"
    tuner.save_configs(str(saved))
    tuner.clear_cache()
    assert tuner.load_configs(str(saved))
    torch.testing.assert_close(verify(history_rows), before, atol=0, rtol=0)
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        actual = attention._project_expanded_kv(x)
    x.normal_()
    graph.replay()
    torch.testing.assert_close(actual, verify(history_rows), atol=0, rtol=0)

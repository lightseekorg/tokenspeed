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

import argparse
import json
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open

from tokenspeed.runtime.metrics.dsv4_numerical_attribution import (
    DSV4NumericalAttributionRecorder,
    begin_numerical_attribution_forward,
    checkpoint_tensor_order,
    configure_dsv4_numerical_attribution,
    numerical_attribution_checkpoint_active,
)


def _row(step: int, *, checkpoint: bool = False) -> dict:
    return {
        "comparison_key": f"key-{step}",
        "case_id": "text-1",
        "population": "text-control",
        "step": step,
        "profile": "tokenspeed-full",
        "call_input_length": 1,
        "capture_checkpoint": checkpoint,
        "forced_id": None,
    }


def test_series_retains_native_dtype_and_writes_exclusively(tmp_path) -> None:
    recorder = DSV4NumericalAttributionRecorder(
        enabled=True,
        global_rank=0,
        mode="series",
        profile="tokenspeed-full",
        spool_dir=tmp_path,
        expected_vocab_size=4,
        expected_observations=2,
        num_layers=1,
    )
    for step in range(2):
        logits = torch.tensor([[step, 2, 1, 0]], dtype=torch.bfloat16)
        recorder.begin_forward(_row(step))
        recorder.record_sampling(logits, logits.argmax(dim=-1), [_row(step)])
    metadata = json.loads((tmp_path / "rank-0.json").read_text(encoding="utf-8"))
    assert [row["raw_argmax"] for row in metadata["observations"]] == [1, 1]
    assert all(
        row["logit_dtype"] == "torch.bfloat16" for row in metadata["observations"]
    )
    with safe_open(tmp_path / "rank-0.safetensors", framework="pt") as tensors:
        assert set(tensors.keys()) == {"key-0", "key-1"}
        assert tensors.get_tensor("key-0").dtype == torch.bfloat16
    with pytest.raises(RuntimeError, match="finalized twice"):
        recorder.finalize_series()


def test_checkpoint_inventory_is_closed_and_ordered() -> None:
    names = checkpoint_tensor_order(2)
    assert names[:4] == (
        "embedding_pre_hc",
        "layer.0.attn_pre_norm",
        "layer.0.attn_output",
        "layer.0.attn_post_hc",
    )
    assert names[-3:] == ("head_input", "final_norm", "logits")
    assert len(names) == 20


def test_checkpoint_writes_exact_closed_boundary_set(tmp_path) -> None:
    recorder = DSV4NumericalAttributionRecorder(
        enabled=True,
        global_rank=2,
        mode="checkpoint",
        profile="tokenspeed-full",
        output_root=tmp_path,
        expected_vocab_size=4,
        num_layers=1,
    )
    row = _row(0, checkpoint=True)
    recorder.begin_forward(row)
    tensors = {
        "embedding_pre_hc": torch.zeros((1, 4096), dtype=torch.bfloat16),
        "layer.0.attn_pre_norm": torch.zeros(4096, dtype=torch.bfloat16),
        "layer.0.attn_output": torch.zeros(4096, dtype=torch.bfloat16),
        "layer.0.attn_post_hc": torch.zeros((4, 4096), dtype=torch.bfloat16),
        "layer.0.ffn_pre_norm": torch.zeros(4096, dtype=torch.bfloat16),
        "layer.0.router_topk_ids": torch.zeros(6, dtype=torch.int64),
        "layer.0.router_weights": torch.zeros(6, dtype=torch.float32),
        "layer.0.ffn_output": torch.zeros(4096, dtype=torch.bfloat16),
        "layer.0.block_post_hc": torch.zeros((4, 4096), dtype=torch.bfloat16),
        "head_input": torch.zeros(4096, dtype=torch.bfloat16),
        "final_norm": torch.zeros(4096, dtype=torch.bfloat16),
        "logits": torch.zeros(4, dtype=torch.bfloat16),
    }
    for name in checkpoint_tensor_order(1):
        recorder.record_boundary(
            name,
            tensors[name],
            preserve_all_rows=name == "embedding_pre_hc",
        )
    recorder.record_sampling(tensors["logits"].unsqueeze(0), torch.tensor([0]), [row])
    path = (
        tmp_path
        / "checkpoints/tokenspeed-full/text-control/text-1-step-00-rank-2.safetensors"
    )
    with safe_open(path, framework="pt") as checkpoint:
        assert set(checkpoint.keys()) == set(checkpoint_tensor_order(1))


def test_flag_off_recorder_rejects_an_observer_call() -> None:
    recorder = DSV4NumericalAttributionRecorder(enabled=False, global_rank=0)
    with pytest.raises(RuntimeError, match="disabled"):
        recorder.begin_forward(_row(0))


def test_checkpoint_boundary_work_is_inactive_during_series(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("DSV4_NUMERICAL_ATTRIBUTION_MODE", "series")
    monkeypatch.setenv("DSV4_NUMERICAL_ATTRIBUTION_PROFILE", "tokenspeed-full")
    monkeypatch.setenv("DSV4_NUMERICAL_ATTRIBUTION_SPOOL", str(tmp_path))
    configure_dsv4_numerical_attribution(
        enabled=True, global_rank=0, expected_vocab_size=4
    )
    begin_numerical_attribution_forward(_row(0, checkpoint=True))
    assert numerical_attribution_checkpoint_active() is False
    configure_dsv4_numerical_attribution(
        enabled=False, global_rank=0, expected_vocab_size=4
    )


def test_server_flag_defaults_off_and_parses_explicit_enablement() -> None:
    from tokenspeed.runtime.utils.server_args import ServerArgs

    assert ServerArgs(model="model").enable_dsv4_numerical_attribution is False
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    parsed = parser.parse_args(
        ["--model", "model", "--enable-dsv4-numerical-attribution"]
    )
    assert parsed.enable_dsv4_numerical_attribution is True


def test_model_executor_build_sampling_info_uses_active_request_path(
    tmp_path, monkeypatch
) -> None:
    from tokenspeed.runtime.execution.model_executor import ModelExecutor
    from tokenspeed.runtime.sampling.sampling_params import SamplingParams

    monkeypatch.setenv("DSV4_NUMERICAL_ATTRIBUTION_MODE", "series")
    monkeypatch.setenv("DSV4_NUMERICAL_ATTRIBUTION_PROFILE", "tokenspeed-incremental")
    monkeypatch.setenv("DSV4_NUMERICAL_ATTRIBUTION_SPOOL", str(tmp_path))
    configure_dsv4_numerical_attribution(
        enabled=True, global_rank=0, expected_vocab_size=4
    )
    anchor = list(range(32))
    observations = [
        {
            "comparison_key": f"key-{step}",
            "case_id": "text-1",
            "population": "text-control",
            "step": step,
            "scored_anchor_id": step,
        }
        for step in range(32)
    ]
    params = SamplingParams(
        temperature=0,
        custom_params={
            "dsv4_numerical_attribution": {
                "profile": "tokenspeed-incremental",
                "case_id": "text-1",
                "population": "text-control",
                "prompt_length": 3,
                "anchor": anchor,
                "anchor_sha256": __import__("hashlib")
                .sha256(
                    json.dumps(anchor, sort_keys=True, separators=(",", ":")).encode()
                )
                .hexdigest(),
                "observations": observations,
                "requested_step": None,
                "checkpoint_step": None,
            },
        },
    )
    executor = object.__new__(ModelExecutor)
    executor.config = SimpleNamespace(
        enable_dsv4_numerical_attribution=True,
        enforce_eager=True,
        spec_algo=None,
        sampling_backend="greedy",
    )
    executor.device = "cpu"
    executor._dsv4_attribution_steps = {}
    executor._dsv4_attribution_completed_requests = set()
    executor.input_buffers = SimpleNamespace(
        req_pool_indices_buf=torch.tensor([1], dtype=torch.int64)
    )
    executor.runtime_states = SimpleNamespace(
        valid_cache_lengths=torch.tensor([0, 0], dtype=torch.int32),
        vocab_size=4,
    )
    forward_op = SimpleNamespace(
        num_extends=lambda: 1,
        request_ids=["request-1"],
        extend_prefix_lens=[0],
        input_lengths=[3],
    )
    info = executor._build_sampling_info(1, [params], forward_op)
    assert info.dsv4_forced_token_ids == [0]
    assert info.dsv4_numerical_attribution_rows[0]["request_id"] == "request-1"
    decode_op = SimpleNamespace(
        num_extends=lambda: 0,
        request_ids=["request-1"],
        extend_prefix_lens=[],
        input_lengths=[1],
    )
    decode_info = executor._build_sampling_info(1, [params], decode_op)
    assert decode_info.dsv4_forced_token_ids == [1]
    assert decode_info.dsv4_numerical_attribution_rows[0]["start_position"] == 3
    for _ in range(30):
        executor._build_sampling_info(1, [params], decode_op)
    drained_info = executor._build_sampling_info(1, [params], decode_op)
    assert drained_info.dsv4_forced_token_ids is None
    assert drained_info.dsv4_numerical_attribution_rows is None
    configure_dsv4_numerical_attribution(
        enabled=False, global_rank=0, expected_vocab_size=4
    )


@pytest.mark.parametrize("decode_verify", [False, True])
def test_greedy_records_raw_argmax_before_forcing(
    tmp_path, monkeypatch, decode_verify
) -> None:
    from tokenspeed.runtime.layers.logits_processor import LogitsProcessorOutput
    from tokenspeed.runtime.sampling.backends.base import SamplingBackendConfig
    from tokenspeed.runtime.sampling.backends.greedy import GreedySamplingBackend
    from tokenspeed.runtime.sampling.sampling_batch_info import SamplingBatchInfo

    monkeypatch.setenv("DSV4_NUMERICAL_ATTRIBUTION_MODE", "series")
    monkeypatch.setenv("DSV4_NUMERICAL_ATTRIBUTION_PROFILE", "tokenspeed-incremental")
    monkeypatch.setenv("DSV4_NUMERICAL_ATTRIBUTION_SPOOL", str(tmp_path))
    recorder = configure_dsv4_numerical_attribution(
        enabled=True, global_rank=0, expected_vocab_size=4
    )
    recorder.expected_observations = 1
    row = {
        **_row(0),
        "profile": "tokenspeed-incremental",
        "forced_id": 3,
    }
    recorder.begin_forward(row)
    backend = GreedySamplingBackend(
        SamplingBackendConfig(
            max_bs=1,
            max_draft_tokens_per_req=1,
            device=torch.device("cpu"),
            enable_tp_sync=False,
        )
    )
    logits = torch.tensor([[0, 5, 2, 1]], dtype=torch.bfloat16)
    output = LogitsProcessorOutput(next_token_logits=logits)
    sampling_info = SamplingBatchInfo(
        dsv4_numerical_attribution_rows=[row],
        dsv4_forced_token_ids=[3],
    )
    if decode_verify:
        tokens, lengths = backend.verify(
            output, sampling_info, torch.zeros((1, 1), dtype=torch.int32)
        )
        assert lengths.tolist() == [1]
    else:
        tokens, _ = backend.sample(output, sampling_info)
    assert tokens.tolist() == [3]
    metadata = json.loads((tmp_path / "rank-0.json").read_text(encoding="utf-8"))
    assert metadata["observations"][0]["raw_argmax"] == 1
    configure_dsv4_numerical_attribution(
        enabled=False, global_rank=0, expected_vocab_size=4
    )

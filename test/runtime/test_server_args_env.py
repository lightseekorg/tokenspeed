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

import pytest

from tokenspeed.runtime.utils.env import (
    global_server_args_dict,
    global_server_args_dict_update,
)
from tokenspeed.runtime.utils.server_args import prepare_server_args


def test_head_tp_exchange_max_rows_reaches_model_initialization():
    original = global_server_args_dict.copy()
    try:
        args = prepare_server_args(
            [
                "--model",
                "x",
                "--head-tp-exchange-max-rows",
                "128",
            ]
        )
        global_server_args_dict_update(args)
        assert global_server_args_dict["head_tp_exchange_max_rows"] == 128
    finally:
        global_server_args_dict.clear()
        global_server_args_dict.update(original)


@pytest.mark.parametrize(
    "roles", [("prefill", "decode", "null"), ("decode", "prefill", "null")]
)
def test_resolved_worker_role_reaches_model_initialization(roles):
    original = global_server_args_dict.copy()
    try:
        for role in roles:
            args = prepare_server_args(["--model", "x", "--disaggregation-mode", role])
            global_server_args_dict_update(args)
            assert global_server_args_dict["disaggregation_mode"] == role
    finally:
        global_server_args_dict.clear()
        global_server_args_dict.update(original)


def test_prefill_kv_limit_reaches_model_initialization_and_can_be_disabled():
    original = global_server_args_dict.copy()
    try:
        for limit in (32768, None, 65536, None):
            argv = ["--model", "x", "--max-model-len", "16384"]
            if limit is not None:
                argv += ["--chunk-prefill-kv-size", str(limit)]
            global_server_args_dict_update(prepare_server_args(argv))
            assert global_server_args_dict["chunk_prefill_kv_size"] == limit
    finally:
        global_server_args_dict.clear()
        global_server_args_dict.update(original)


@pytest.mark.parametrize("rate", ["0", "1", "100"])
def test_pd_ttft_sampling_percentage(rate):
    args = prepare_server_args(
        ["--model", "x", "--ttft-trace-sampling-percentage", rate]
    )
    assert args.ttft_trace_sampling_percentage == int(rate)


@pytest.mark.parametrize("rate", ["-1", "101"])
def test_invalid_pd_ttft_sampling_percentage(rate):
    with pytest.raises(ValueError, match="ttft-trace-sampling-percentage"):
        prepare_server_args(["--model", "x", "--ttft-trace-sampling-percentage", rate])


def test_request_time_stats_compatibility_flag():
    assert prepare_server_args(
        ["--model", "x", "--enable-request-time-stats-logging"]
    ).enable_log_request_stats


def test_adaptive_dspark_rejects_synthetic_acceptance():
    argv = [
        "--model",
        "x",
        "--speculative-algorithm",
        "DSPARK",
        "--sampling-backend",
        "greedy",
        "--grammar-backend",
        "none",
        "--disable-prefill-graph",
        "--enable-dsv41-adaptive-dspark",
    ]
    for length in ("4", "4.5"):
        with pytest.raises(ValueError, match="real acceptance"):
            prepare_server_args(argv + ["--synthetic-acceptance-length", length])


def test_adaptive_dspark_cli_reaches_runtime_and_resets():
    original = global_server_args_dict.copy()
    try:
        args = prepare_server_args(
            [
                "--model",
                "x",
                "--speculative-algorithm",
                "DSPARK",
                "--sampling-backend",
                "greedy",
                "--grammar-backend",
                "none",
                "--disable-prefill-graph",
                "--enable-dsv41-adaptive-dspark",
                "--dsv41-dspark-min-batch-size",
                "64",
            ]
        )
        global_server_args_dict_update(args)
        assert global_server_args_dict["enable_dsv41_adaptive_dspark"] is True
        assert global_server_args_dict["dsv41_dspark_min_batch_size"] == 64
        args = prepare_server_args(["--model", "x"])
        global_server_args_dict_update(args)
        assert global_server_args_dict["enable_dsv41_adaptive_dspark"] is False
        assert global_server_args_dict["dsv41_dspark_min_batch_size"] == 32
    finally:
        global_server_args_dict.clear()
        global_server_args_dict.update(original)


def test_sampled_adaptive_dspark_requires_recorded_proposals():
    argv = [
        "--model",
        "x",
        "--speculative-algorithm",
        "DSPARK",
        "--sampling-backend",
        "flashinfer",
        "--grammar-backend",
        "none",
        "--disable-prefill-graph",
        "--enable-dsv41-adaptive-dspark",
    ]
    with pytest.raises(ValueError, match="enable-speculative-sampling"):
        prepare_server_args(argv)
    with pytest.raises(ValueError, match="sampling-stream per-request"):
        prepare_server_args(argv + ["--enable-speculative-sampling"])
    args = prepare_server_args(
        argv + ["--enable-speculative-sampling", "--sampling-stream", "per-request"]
    )
    assert args.enable_speculative_sampling
    assert args.speculative_rejection_sample_method == "standard"


def test_sampled_adaptive_dspark_rejects_attention_tp():
    with pytest.raises(ValueError, match="attention TP1"):
        prepare_server_args(
            [
                "--model",
                "x",
                "--speculative-algorithm",
                "DSPARK",
                "--sampling-backend",
                "flashinfer",
                "--grammar-backend",
                "none",
                "--disable-prefill-graph",
                "--enable-dsv41-adaptive-dspark",
                "--enable-speculative-sampling",
                "--sampling-stream",
                "per-request",
                "--tensor-parallel-size",
                "2",
            ]
        )


def _dp_local_graph_args():
    return [
        "--model",
        "x",
        "--speculative-algorithm",
        "DSPARK",
        "--sampling-backend",
        "greedy",
        "--moe-backend",
        "mega_moe",
        "--enable-dp-local-cuda-graph",
    ]


@pytest.mark.parametrize("backend", ["mega_moe", "deep_gemm_mega_moe"])
@pytest.mark.parametrize("draft_backend", [None, "mega_moe", "deep_gemm_mega_moe"])
@pytest.mark.parametrize("role", ["null", "decode"])
def test_dp_local_cuda_graph_accepts_fixed_greedy(backend, draft_backend, role):
    argv = _dp_local_graph_args() + [
        "--moe-backend",
        backend,
        "--disaggregation-mode",
        role,
        "--data-parallel-size",
        "2",
        "--enable-expert-parallel",
    ]
    if draft_backend is not None:
        argv += ["--draft-moe-backend", draft_backend]
    args = prepare_server_args(argv)
    assert args.enable_dp_local_cuda_graph
    assert not args.enable_dsv41_adaptive_dspark
    assert args.mapping.attn.dp_size == 2


def test_dp_local_cuda_graph_accepts_sampled_adaptive():
    args = prepare_server_args(
        _dp_local_graph_args()
        + [
            "--sampling-backend",
            "flashinfer",
            "--enable-dsv41-adaptive-dspark",
            "--enable-speculative-sampling",
            "--sampling-stream",
            "per-request",
            "--disable-prefill-graph",
            "--grammar-backend",
            "none",
        ]
    )
    assert args.enable_dp_local_cuda_graph
    assert args.enable_dsv41_adaptive_dspark
    assert args.enable_speculative_sampling


def test_dp_local_cuda_graph_sampled_path_keeps_adaptive_validation():
    with pytest.raises(ValueError, match="enable-speculative-sampling"):
        prepare_server_args(
            _dp_local_graph_args()
            + [
                "--sampling-backend",
                "flashinfer",
                "--enable-dsv41-adaptive-dspark",
                "--disable-prefill-graph",
                "--grammar-backend",
                "none",
            ]
        )


@pytest.mark.parametrize(
    "options, message",
    [
        (["--speculative-algorithm", "MTP"], "require DSPARK"),
        (["--moe-backend", "auto"], "MegaMoE for both target and draft"),
        (["--draft-moe-backend", "deep_gemm"], "MegaMoE for both target and draft"),
        (["--enforce-eager"], "require CUDA graphs"),
        (["--disaggregation-mode", "prefill"], "decode-capable worker"),
        (["--disaggregation-mode", "encode"], "decode-capable worker"),
        (["--sampling-backend", "flashinfer"], "greedy sampling or adaptive"),
        (["--tensor-parallel-size", "2"], "attention/dense/MoE/LM-head TP1"),
        (
            ["--world-size", "2", "--data-parallel-size", "2"],
            "attention/dense/MoE/LM-head TP1",
        ),
        (
            [
                "--world-size",
                "2",
                "--data-parallel-size",
                "2",
                "--enable-expert-parallel",
                "--dense-tp-size",
                "2",
            ],
            "attention/dense/MoE/LM-head TP1",
        ),
        (
            [
                "--world-size",
                "2",
                "--data-parallel-size",
                "2",
                "--enable-expert-parallel",
                "--lm-head-tp-size",
                "2",
            ],
            "attention/dense/MoE/LM-head TP1",
        ),
        (
            [
                "--world-size",
                "2",
                "--data-parallel-size",
                "2",
                "--enable-expert-parallel",
                "--attn-head-tp-size",
                "2",
                "--disaggregation-mode",
                "decode",
                "--disable-prefill-graph",
            ],
            "attention/dense/MoE/LM-head TP1",
        ),
        (["--dp-sampling"], "no DP sampling"),
    ],
)
def test_dp_local_cuda_graph_rejects_unsupported_configuration(options, message):
    with pytest.raises(ValueError, match=message):
        prepare_server_args(_dp_local_graph_args() + options)


@pytest.mark.parametrize(
    "parallel_sizes",
    [
        (2, 1, 1, 1),
        (1, 2, 1, 1),
        (1, 1, 2, 1),
        (1, 1, 1, 2),
    ],
)
def test_dp_local_cuda_graph_rejects_parallel_groups(parallel_sizes):
    args = prepare_server_args(_dp_local_graph_args())
    # Exercise this admission rule independently of earlier topology checks.
    (
        args.pipeline_parallel_size,
        args.kv_parallel_size,
        args.decode_context_parallel_size,
        args.query_context_parallel_size,
    ) = parallel_sizes
    with pytest.raises(ValueError, match="PP1/KVP1/DCP1/QCP1"):
        args.validate_dp_local_cuda_graph()


def test_dp_local_cuda_graph_cli_reaches_runtime_and_resets():
    original = global_server_args_dict.copy()
    try:
        global_server_args_dict_update(prepare_server_args(_dp_local_graph_args()))
        assert global_server_args_dict["enable_dp_local_cuda_graph"] is True
        global_server_args_dict_update(prepare_server_args(["--model", "x"]))
        assert global_server_args_dict["enable_dp_local_cuda_graph"] is False
    finally:
        global_server_args_dict.clear()
        global_server_args_dict.update(original)

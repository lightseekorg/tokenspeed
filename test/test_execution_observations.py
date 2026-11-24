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

"""Real native-engine execution evidence, requiring a GPU lease and local weights.

Set TOKENSPEED_EXECUTION_TEST_ARGS to a JSON list of base engine arguments
(including a local --model and an explicitly chosen, supported numerical envelope).
Inline prefill requires TOKENSPEED_EXECUTION_INLINE_TEST_ARGS naming a model
and backend that support inline capture. Do not put graph or speculation
choices in these lists. DFLASH also requires TOKENSPEED_EXECUTION_DRAFT_MODEL
(TOKENSPEED_EXECUTION_INLINE_DRAFT_MODEL for the inline fixture).
Run this file with the newly built host AND client bindings on PYTHONPATH.
No model is downloaded by these tests.
"""

import json
import os
import struct
import uuid
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(
    torch.version.cuda is None or not torch.cuda.is_available(), reason="requires CUDA"
)


def base_args(variable="TOKENSPEED_EXECUTION_TEST_ARGS"):
    value = os.environ.get(variable)
    if value is None:
        pytest.fail(f"{variable} must supply qualified local model arguments")
    args = json.loads(value)
    model = args[args.index("--model") + 1]
    assert Path(model).is_dir(), "GPU tests require already-local model weights"
    return args


def serving_args(algorithm, route, *, base_variable):
    assert route in (
        "eager",
        "decode_graph",
        "prefill_breakable",
        "prefill_inline",
        "over_ladder",
    )
    args = base_args(base_variable)
    args += [
        "--enable-async-logging",
        "--enable-startup-requests",
        "--enable-output-logprobs",
        "--max-num-seqs",
        "8",
        "--max-model-len",
        "1024",
        "--chunked-prefill-size",
        "128",
        "--cudagraph-capture-sizes",
        "1",
        "2",
        "4",
        "--max-cudagraph-capture-size",
        "4",
        "--prefill-graph-capture-token-sizes",
        "64",
        "128",
        "--prefill-graph-max-tokens",
        "128",
        "--prefill-graph-capture-batch-sizes",
        "4" if route == "prefill_inline" else "1",
    ]
    if route == "eager":
        args += ["--enforce-eager"]
    elif route == "decode_graph":
        args += ["--disable-prefill-graph"]
    if algorithm != "none":
        args += [
            "--speculative-algorithm",
            algorithm,
            "--speculative-num-draft-tokens",
            "8",
        ]
    if algorithm == "DFLASH":
        variable = (
            "TOKENSPEED_EXECUTION_INLINE_DRAFT_MODEL"
            if route == "prefill_inline"
            else "TOKENSPEED_EXECUTION_DRAFT_MODEL"
        )
        draft = os.environ.get(variable)
        if draft is None:
            pytest.fail(f"DFLASH requires local {variable}")
        assert Path(draft).is_dir()
        args += ["--speculative-draft-model-path", draft]
    return args


def launch(args, *, observed, num_engines=1):
    from test.execution_observation_assets import resolve_qualification

    from tokenspeed_engine import _host

    from tokenspeed.runtime.entrypoints.wire_driver import WireDriver

    assert (
        _host.execution_observation_protocol() == "tokenspeed-execution-observations-v1"
    )
    # Resolve the actual named policy and both model gates; no policy-name alias.
    # Keep this report in the captured qualification log alongside native traces.
    print(
        f"[execution-qualification] {json.dumps(resolve_qualification(args))}",
        flush=True,
    )
    return WireDriver(
        args,
        num_engines=num_engines,
        routing="round_robin",
        startup_timeout_s=900,
        env={
            "TOKENSPEED_EXECUTION_EVENT_LOG": "1" if observed else "0",
            "TOKENSPEED_CACHE_EVENT_LOG": "1" if observed else "0",
            "TOKENSPEED_CACHE_EVENT_INSTANCE": "execution-test-" + uuid.uuid4().hex,
        },
    )


def generate(driver, count):
    from tokenspeed.runtime.entrypoints.wire_driver import generate_request

    requests = [
        generate_request(
            [1 + (i + j) % 31 for j in range(17)],
            {"temperature": 0, "ignore_eos": True, "max_new_tokens": 24, "seed": 73},
            return_logprob=True,
        )
        for i in range(count)
    ]
    outputs = driver.generate(requests)
    for output in outputs:
        assert output.finish_reason == "length", output.finish_details
        assert len(output.output_ids) == len(output.output_token_logprobs) == 24
    return outputs


def bits(values):
    return b"".join(struct.pack("<f", value) for value in values)


def closed_trace(outputs, capfd, tmp_path):
    from tokenspeed.runtime.entrypoints.execution_trace import ExecutionTrace

    captured = capfd.readouterr()
    log = tmp_path / "execution.log"
    log.write_text(captured.out + captured.err)
    snapshots = [output.execution_observations for output in outputs]
    assert all(snapshot is not None for snapshot in snapshots)
    (tmp_path / "measured.json").write_text(json.dumps(snapshots, indent=2))
    return ExecutionTrace.read([log])


@pytest.mark.parametrize("algorithm", ["none", "DUMMY", "DFLASH"])
@pytest.mark.parametrize(
    "route",
    ["eager", "decode_graph", "prefill_breakable", "prefill_inline", "over_ladder"],
)
def test_actual_routes_widths_closed_trace_and_noninterference(
    algorithm, route, capfd, tmp_path
):
    args = serving_args(
        algorithm,
        route,
        base_variable=(
            "TOKENSPEED_EXECUTION_INLINE_TEST_ARGS"
            if route == "prefill_inline"
            else "TOKENSPEED_EXECUTION_TEST_ARGS"
        ),
    )
    count = (
        5 if route == "over_ladder" else 3
    )  # includes a padded bucket and an eager overflow
    with launch(args, observed=False) as driver:
        baseline = generate(driver, count)
        assert all(output.execution_observations is None for output in baseline)
    capfd.readouterr()
    with launch(args, observed=True) as driver:
        observed = generate(driver, count)
    trace = closed_trace(observed, capfd, tmp_path)
    width = 1 if algorithm == "none" else 8
    for expected, actual in zip(baseline, observed, strict=True):
        assert actual.output_ids == expected.output_ids
        assert bits(actual.output_token_logprobs) == bits(
            expected.output_token_logprobs
        )
        snapshot = actual.execution_observations
        assert snapshot is not None
        prefill_path = (
            "eager"
            if route in ("eager", "decode_graph")
            else "prefill_inline" if route == "prefill_inline" else "prefill_breakable"
        )
        decode_path = "eager" if route in ("eager", "over_ladder") else "decode_graph"
        trace.require(
            snapshot,
            physical_ranks=[0],
            cells=[("prefill", prefill_path, 0), ("decode", decode_path, width)],
            restored=False,
        )
        if route == "over_ladder":
            trace.require_eager_overflow(
                snapshot,
                physical_ranks=[0],
                verify_width=width,
                capture_max_requests=int(
                    args[args.index("--max-cudagraph-capture-size") + 1]
                ),
            )
        rows = list(
            trace.rounds[
                (
                    snapshot["instance_id"],
                    "fused",
                    0,
                    0,
                    snapshot["admission_id"],
                    snapshot["request_id"],
                )
            ].values()
        )
        assert all(r["verify_width"] == width for r in rows if r["phase"] == "decode")
        decode = [r for r in rows if r["phase"] == "decode"]
        assert all(
            0 <= r["kept_prefix"] <= r["accepted_before_stop"] <= width for r in decode
        )
        assert all(
            not r["force_single"] or r["accepted_before_stop"] == 1 for r in decode
        )
        assert sum(b["count"] for b in snapshot["pre_stop_bins"]) == sum(
            b["count"] for b in snapshot["bins"] if b["phase"] == "decode"
        )
        if decode_path == "decode_graph":
            assert any(
                r["request_capacity"] > count
                for r in rows
                if r["path"] == "decode_graph"
            )


def test_restore_keeps_the_admission_and_observed_continuation(capfd, tmp_path):
    args = serving_args(
        "DUMMY", "prefill_breakable", base_variable="TOKENSPEED_EXECUTION_TEST_ARGS"
    ) + [
        "--debug-force-retraction-interval",
        "3",
    ]
    with launch(args, observed=True) as driver:
        outputs = generate(driver, 3)
    trace = closed_trace(outputs, capfd, tmp_path)
    for output in outputs:
        trace.require(
            output.execution_observations,
            physical_ranks=[0],
            cells=[("decode", "decode_graph", 8)],
            restored=True,
        )
        trace.require_copied_restore(output.execution_observations, physical_ranks=[0])


def test_observations_survive_memory_saver_release_resume(capfd, tmp_path):
    from tokenspeed.runtime.engine.io_struct import (
        ReleaseMemoryOccupationReqInput,
        ResumeMemoryOccupationReqInput,
    )

    args = serving_args(
        "none", "decode_graph", base_variable="TOKENSPEED_EXECUTION_MEMORY_TEST_ARGS"
    )
    assert "--enable-memory-saver" in args
    with launch(args, observed=True) as driver:
        [baseline] = generate(driver, 1)
        outputs = [baseline]
        for _ in range(2):
            assert all(
                reply.success
                for reply in driver.control(
                    ReleaseMemoryOccupationReqInput(tags=["weights"])
                )
            )
            assert all(
                reply.success
                for reply in driver.control(
                    ResumeMemoryOccupationReqInput(tags=["weights"])
                )
            )
            [resumed] = generate(driver, 1)
            outputs.append(resumed)
            assert resumed.output_ids == baseline.output_ids
            assert bits(resumed.output_token_logprobs) == bits(
                baseline.output_token_logprobs
            )
    trace = closed_trace(outputs, capfd, tmp_path)
    assert (
        len({output.execution_observations["admission_id"] for output in outputs}) == 3
    )
    for output in outputs:
        trace.require(
            output.execution_observations,
            physical_ranks=[0],
            cells=[("prefill", "eager", 0), ("decode", "decode_graph", 1)],
            restored=False,
        )


@pytest.mark.parametrize("algorithm", ["none", "DUMMY", "DFLASH"])
def test_mixed_decode_rows_retain_the_shared_target_route(algorithm, capfd, tmp_path):
    from tokenspeed.runtime.entrypoints.wire_driver import generate_request

    args = serving_args(
        algorithm, "prefill_breakable", base_variable="TOKENSPEED_EXECUTION_TEST_ARGS"
    ) + ["--enable-mixed-batch"]
    with launch(args, observed=True) as driver:
        requests = [
            generate_request(
                [i % 31 + 1 for i in range(length)],
                {"temperature": 0, "ignore_eos": True, "max_new_tokens": 48, "seed": 7},
            )
            for length in (17, 513, 17)
        ]
        outputs = driver.generate(requests)
    trace = closed_trace(outputs, capfd, tmp_path)
    measured = []
    for output in outputs:
        assert output.finish_reason == "length"
        trace.check_snapshot(output.execution_observations)
        snapshot = output.execution_observations
        key = (
            snapshot["instance_id"],
            "fused",
            0,
            0,
            snapshot["admission_id"],
            snapshot["request_id"],
        )
        measured += list(trace.rounds[key].values())
    mixed = [
        row for row in measured if row["phase"] == "decode" and row["num_extends"] > 0
    ]
    assert mixed, "the workload must actually reach mixed execution"
    assert all(
        row["path"] in ("eager", "prefill_breakable", "prefill_inline") for row in mixed
    )
    assert all(
        row["verify_width"] == (1 if algorithm == "none" else 8) for row in mixed
    )


def test_idle_dp_peer_gets_no_serving_request_credit(capfd, tmp_path):
    from tokenspeed.runtime.entrypoints.wire_driver import generate_request

    args = base_args("TOKENSPEED_EXECUTION_DP_TEST_ARGS") + ["--enable-async-logging"]
    assert args[args.index("--data-parallel-size") + 1] == "2"
    with launch(args, observed=True, num_engines=2) as driver:
        outputs = []
        for dp in (0, 1):
            request = generate_request(
                [1, 2, 3] * 8,
                {"temperature": 0, "ignore_eos": True, "max_new_tokens": 32},
                data_parallel_rank=dp,
            )
            [output] = driver.generate([request])
            assert output.finish_reason == "length"
            assert output.execution_observations["dp_rank"] == dp
            outputs.append(output)
    trace = closed_trace(outputs, capfd, tmp_path)
    for output in outputs:
        snapshot = output.execution_observations
        trace.check_snapshot(snapshot)
        same_rid = [key for key in trace.started if key[-1] == snapshot["request_id"]]
        assert same_rid
        assert all(key[2] == snapshot["dp_rank"] for key in same_rid)

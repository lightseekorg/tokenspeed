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

"""Native Mooncake cross-instance MODEL serving correctness gate (L3).

One engine (the writer) serves a chunk-aligned prefix carrying a random
codeword and persists it to a native Mooncake store; the test waits for
successful PUT evidence and a drained writeback that holds across several
storage-metrics exports, then stops the writer. A FRESH engine (the reader)
starts against the same store, tenant, model and weights; its FIRST request
(prefix + a question asking for the codeword) must reuse exactly the whole
prefix with positive GET deltas -- nothing warmed its Device/Host caches, so
that reuse is the L3 restore. A third engine with every cache tier disabled
(the oracle) recomputes the same prompt; the reader's greedy tokens and
logprobs must match the oracle's within the case's envelope, and both
answers must contain the codeword, so the agreement depends on the prefix.
A disjoint negative-control prompt on the reader must recompute fully. The
store's data segments live on an independent provider process that
outlives the writer; ``flush_cache`` is never called (it would delete the
remote namespace).

Status: pending GPU validation. Scope: one attention-TP rank, fused role,
TCP store, one writer, reader and oracle per case, greedy decoding. See
docs/operations/native-l3-validation.md.

The gate runs only when L3_SERVING_E2E_CONFIG names an operator config (see
test/runtime/l3_serving_e2e_config.example.json)::

    L3_SERVING_E2E_CONFIG=/path/to/gate.json \
        python3 -m pytest test/runtime/test_l3_serving_e2e.py -v -s

Unset, the test skips. Set (even blank), every configuration or
prerequisite problem FAILS the test.
"""

from __future__ import annotations

import argparse
import os
import signal
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid

import pytest

# The repository root on sys.path so ``test.*`` resolves, in this process
# and in the provider subprocess (which re-runs this file as a script).
sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)

from test.l3_serving_test_utils import (  # noqa: E402
    PROVIDER_READY_TOKEN,
    GateConfigError,
    aligned_prefix_ids,
    aligned_prefix_length,
    await_provider_ready,
    check_token_budgets,
    compare_outputs,
    evaluate_codeword_answers,
    evaluate_completion_finish,
    evaluate_negative_control,
    evaluate_reader_restore,
    evaluate_writer_evidence,
    load_gate_config,
    max_prefetch_tokens,
    metrics_url,
    missing_gauges,
    negative_control_text,
    oracle_engine_argv,
    provider_command,
    provider_lifetime_seconds,
    readability_violations,
    requested_config_path,
    run_bounded,
    run_tenant_id,
    serving_engine_argv,
    snapshot_from_exposition,
    start_provider_process,
    stop_on_exit,
    stop_provider_process,
    suffix_question_text,
    thresholds_from_case,
)
from test.l3_test_utils import mooncake_master, store_library  # noqa: E402

# The provider's Host pool is a transfer buffer only; its geometry executes
# no storage operation, so a handful of small blocks is enough.
_PROVIDER_LCM_BLOCK_BYTES = 1 << 20
_PROVIDER_LCM_BLOCKS = 4
# The provider closes its executor within 5 s of SIGTERM.
_PROVIDER_CLOSE_TIMEOUT_S = 5.0
_PROVIDER_STOP_GRACE_S = 30.0
_PROVIDER_POLL_S = 0.2

# One HTTP scrape must answer inside this budget.
_SCRAPE_TIMEOUT_S = 10.0
# The metrics endpoint and the first storage-metrics export follow engine
# registration; this bounds the wait for both.
_METRICS_READY_BUDGET_S = 120.0
# The engine exports storage metrics at most once a second. Evidence must
# hold unchanged for longer than a few export periods before it counts, so
# a scrape taken just before a pending export cannot settle a phase.
_SETTLE_WINDOW_S = 3.0


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _scrape(port: int):
    with urllib.request.urlopen(
        metrics_url(port), timeout=_SCRAPE_TIMEOUT_S
    ) as response:
        return snapshot_from_exposition(response.read().decode())


def _wait_metrics(port: int, budget_s: float, interval_s: float):
    """The first scrape carrying every mandatory L3 gauge, i.e. after the
    engine's first storage-metrics export."""
    deadline = time.monotonic() + budget_s
    problem = "no scrape yet"
    while time.monotonic() < deadline:
        try:
            snapshot = _scrape(port)
        except (urllib.error.URLError, OSError, ValueError) as error:
            problem = f"scrape failed: {error}"
        else:
            missing = missing_gauges(snapshot)
            if not missing:
                return snapshot
            problem = f"gauges not exported yet: {missing}"
        time.sleep(interval_s)
    raise AssertionError(
        f"the engine's L3 metrics on port {port} were not ready within "
        f"{budget_s}s: {problem}"
    )


def _await_settled(port: int, accept, interval_s: float, budget_s: float, label: str):
    """A scrape that ``accept`` finds no violation in and that stayed
    unchanged across ``_SETTLE_WINDOW_S``; fails with the last violations
    after ``budget_s``."""
    deadline = time.monotonic() + budget_s
    stable = None
    stable_since = 0.0
    violations = ["no scrape yet"]
    while time.monotonic() < deadline:
        snapshot = _scrape(port)
        now = time.monotonic()
        violations = accept(snapshot)
        if violations or snapshot != stable:
            stable = None if violations else snapshot
            stable_since = now
        elif now - stable_since >= _SETTLE_WINDOW_S:
            return snapshot
        time.sleep(min(interval_s, _SETTLE_WINDOW_S / 2))
    raise AssertionError(
        f"{label}: the L3 evidence did not settle within {budget_s}s:\n- "
        + "\n- ".join(violations)
    )


def _generate(
    engine,
    input_ids: list[int],
    max_new_tokens: int,
    return_logprob: bool,
    timeout_s: float,
    label: str,
):
    """One greedy wire request, bounded: a hung engine fails the phase, whose
    ``stop_on_exit`` then stops the engine."""
    from tokenspeed.runtime.entrypoints.wire_driver import generate_request

    request = generate_request(
        input_ids,
        {"max_new_tokens": max_new_tokens, "temperature": 0},
        return_logprob=return_logprob,
    )
    [completion] = run_bounded(lambda: engine.generate([request]), timeout_s, label)
    problems = evaluate_completion_finish(completion.finish_reason, label)
    assert not problems, "\n- " + "\n- ".join(problems)
    assert completion.prompt_tokens == len(input_ids), (
        f"{label}: the engine saw {completion.prompt_tokens} prompt tokens, "
        f"expected {len(input_ids)}: prompt truncation invalidates the gate"
    )
    return completion


def _start_engine(argv: list[str], config):
    """A WireDriver engine; a startup failure stops its own process group
    before raising, so nothing is left to clean up here."""
    from tokenspeed.runtime.entrypoints.wire_driver import WireDriver

    return WireDriver(
        argv,
        num_engines=1,
        routing="round_robin",
        startup_timeout_s=config.timeouts.engine_startup_seconds,
    )


def _shutdown_engine(engine) -> str:
    # SIGTERM to the launcher, then SIGKILL to the engine's own session
    # (WireDriver starts it with start_new_session): its ranks and nothing
    # else.
    engine.shutdown()
    return ""


def _stop_provider_checked(provider) -> str:
    stopped = stop_provider_process(provider, _PROVIDER_STOP_GRACE_S)
    if stopped.returncode != 0:
        raise RuntimeError(
            f"the L3 segment provider exited with {stopped.returncode} (a "
            "clean stop exits 0); log:\n" + stopped.log
        )
    return ""


def _writer_phase(config, case, tenant_id: str, master_address: str, x_ids):
    """Serve the prefix on a fresh engine and wait until it is persisted;
    the engine is stopped before this returns, whatever happens."""
    port = _free_port()
    engine = _start_engine(
        serving_engine_argv(
            case,
            config.mooncake,
            config.async_overrides,
            tenant_id,
            master_address,
            port,
        ),
        config,
    )
    with stop_on_exit(lambda: _shutdown_engine(engine), f"[{case.name}] writer"):
        interval = config.timeouts.metrics_poll_interval_seconds
        _wait_metrics(port, _METRICS_READY_BUDGET_S, interval)
        completion = _generate(
            engine,
            x_ids,
            case.writer_max_new_tokens,
            False,
            config.timeouts.request_seconds,
            f"[{case.name}] writer",
        )
        assert completion.cached_tokens == 0, (
            f"[{case.name}] a fresh writer on a fresh master must recompute its "
            f"prompt, got {completion.cached_tokens} cached tokens"
        )
        thresholds = thresholds_from_case(case)
        _await_settled(
            port,
            lambda snapshot: evaluate_writer_evidence(snapshot, thresholds),
            interval,
            config.timeouts.put_evidence_seconds,
            f"[{case.name}] writer persistence",
        )


def _reader_phase(config, case, tenant_id, master_address, x_ids, y_ids, z_ids):
    """The fresh reader: its FIRST request is the probe (prefix + suffix),
    the second the disjoint negative control. Returns both completions."""
    port = _free_port()
    engine = _start_engine(
        serving_engine_argv(
            case,
            config.mooncake,
            config.async_overrides,
            tenant_id,
            master_address,
            port,
        ),
        config,
    )
    with stop_on_exit(lambda: _shutdown_engine(engine), f"[{case.name}] reader"):
        interval = config.timeouts.metrics_poll_interval_seconds
        budget = config.timeouts.put_evidence_seconds
        before = _wait_metrics(port, _METRICS_READY_BUDGET_S, interval)
        # The first request this engine ever serves consumes the persisted
        # prefix; nothing may warm its Device/Host caches before it.
        probe = _generate(
            engine,
            x_ids + y_ids,
            case.suffix_max_new_tokens,
            True,
            config.timeouts.request_seconds,
            f"[{case.name}] reader probe",
        )
        after_probe = _await_settled(
            port,
            lambda snapshot: readability_violations(snapshot, "reader"),
            interval,
            budget,
            f"[{case.name}] probe",
        )
        violations = evaluate_reader_restore(
            before,
            after_probe,
            thresholds_from_case(case),
            len(x_ids),
            probe.cached_tokens,
        )
        assert not violations, (
            f"[{case.name}] the fresh reader's FIRST request did not restore "
            "the persisted prefix:\n- " + "\n- ".join(violations)
        )
        control = _generate(
            engine,
            z_ids,
            case.suffix_max_new_tokens,
            True,
            config.timeouts.request_seconds,
            f"[{case.name}] negative control",
        )
        after_control = _await_settled(
            port,
            lambda snapshot: readability_violations(snapshot, "reader"),
            interval,
            budget,
            f"[{case.name}] control",
        )
        violations = evaluate_negative_control(
            after_probe, after_control, control.cached_tokens
        )
        assert not violations, (
            f"[{case.name}] the negative control reused cache it must not "
            "have:\n- " + "\n- ".join(violations)
        )
        return probe, control


def _oracle_phase(config, case, probe_ids: list[int]):
    """The recompute oracle: every cache tier off, same chunking, so its
    last prefill chunk is the reader's extend."""
    engine = _start_engine(oracle_engine_argv(case), config)
    with stop_on_exit(lambda: _shutdown_engine(engine), f"[{case.name}] oracle"):
        reference = _generate(
            engine,
            probe_ids,
            case.suffix_max_new_tokens,
            True,
            config.timeouts.request_seconds,
            f"[{case.name}] oracle",
        )
        assert reference.cached_tokens == 0, (
            f"[{case.name}] the oracle must recompute every token, got "
            f"{reference.cached_tokens} cached tokens"
        )
        return reference


def _run_case(config, case, tenant_id: str) -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(case.model_path, trust_remote_code=True)

    def encode(text: str) -> list[int]:
        return tokenizer(text, add_special_tokens=False)["input_ids"]

    def decode(ids: list[int]) -> str:
        return tokenizer.decode(ids, skip_special_tokens=True)

    # Random per case per run: no earlier run or training corpus can supply
    # the fact the suffix asks for.
    codeword = uuid.uuid4().hex[:12]
    x_ids = aligned_prefix_ids(
        encode,
        decode,
        codeword,
        aligned_prefix_length(case.min_prefix_tokens, case.chunked_prefill_tokens),
    )
    y_ids = encode(suffix_question_text())
    z_ids = encode(negative_control_text())
    check_token_budgets(
        case, max_prefetch_tokens(config), len(x_ids), len(y_ids), len(z_ids)
    )
    with mooncake_master() as master_address:
        provider = start_provider_process(
            provider_command(
                sys.executable,
                os.path.abspath(__file__),
                config.mooncake,
                tenant_id,
                master_address,
                config.provider.segment_bytes,
                provider_lifetime_seconds(config.timeouts, len(config.cases)),
            )
        )
        with stop_on_exit(
            lambda: _stop_provider_checked(provider), f"[{case.name}] L3 provider"
        ):
            await_provider_ready(
                provider,
                PROVIDER_READY_TOKEN,
                config.timeouts.engine_startup_seconds,
                _PROVIDER_POLL_S,
            )
            _writer_phase(config, case, tenant_id, master_address, x_ids)
            probe, control = _reader_phase(
                config, case, tenant_id, master_address, x_ids, y_ids, z_ids
            )
    reference = _oracle_phase(config, case, x_ids + y_ids)
    violations = compare_outputs(
        reference.output_ids,
        reference.output_token_logprobs,
        probe.output_ids,
        probe.output_token_logprobs,
        case.max_token_mismatches,
        case.logprob_atol,
        case.min_output_tokens,
    )
    violations += evaluate_codeword_answers(
        codeword,
        decode(reference.output_ids),
        decode(probe.output_ids),
        decode(control.output_ids),
    )
    assert not violations, (
        f"[{case.name}] the restored output does not match the recompute "
        "oracle:\n- " + "\n- ".join(violations)
    )
    print(
        f"[{case.name}] native L3 serving gate passed: a {len(x_ids)}-token "
        f"prefix was persisted, restored whole and answered like the oracle "
        f"over {len(probe.output_ids)} output tokens",
        flush=True,
    )


def test_native_mooncake_cross_instance_serving_gate():
    """The gate: skipped unless L3_SERVING_E2E_CONFIG is set; once set,
    every configuration or prerequisite problem fails."""
    try:
        config_path = requested_config_path(os.environ)
    except GateConfigError as error:
        pytest.fail(str(error))
    if config_path is None:
        pytest.skip(
            "set L3_SERVING_E2E_CONFIG to a gate config JSON to run the native "
            "Mooncake cross-instance serving gate"
        )
    try:
        config = load_gate_config(config_path)
    except GateConfigError as error:
        pytest.fail(str(error))
    if os.environ.get("TOKENSPEED_MOONCAKE_STORE_LIBRARY"):
        pytest.fail(
            "unset TOKENSPEED_MOONCAKE_STORE_LIBRARY for this gate: the master "
            "and provider use the installed wheel, and all engines must use "
            "that same native library"
        )
    try:
        import torch
        import transformers  # noqa: F401 - the tokenizer source
    except ImportError as error:
        pytest.fail(f"the gate is requested but a prerequisite is missing: {error}")
    if not torch.cuda.is_available():
        pytest.fail("the gate is requested but no CUDA device is available")
    if store_library() is None:
        pytest.fail(
            "the gate is requested but the tokenspeed-mooncake store library "
            "(libmooncake_store.so) is missing"
        )
    for case in config.cases:
        _run_case(config, case, run_tenant_id(config.tenant_prefix, uuid.uuid4().hex))


def _provider_main(argv: list[str]) -> None:
    """The independent segment provider process's entry point.

    Connects a store client that mounts ``--segment-bytes`` into the store
    and then idles; cached objects live on this segment, independent of any
    inference worker's lifetime. SIGTERM/SIGINT (or the backstop deadline)
    ends it; it exits 0 only when its executor closed cleanly.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--l3-provider", action="store_true", required=True)
    parser.add_argument("--master", required=True)
    parser.add_argument("--tenant-id", required=True)
    parser.add_argument("--segment-bytes", type=int, required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--device-name", required=True)
    parser.add_argument("--metadata-server", required=True)
    parser.add_argument("--local-hostname", required=True)
    parser.add_argument("--max-seconds", type=float, required=True)
    args = parser.parse_args(argv)

    stop = threading.Event()

    def handle_signal(signum, frame):
        stop.set()

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    from test.l3_test_utils import start_l3_executor

    import torch

    from tokenspeed.runtime.cache.l3.config import MooncakeStoreSettings

    library = store_library()
    if library is None:
        raise SystemExit(
            "the tokenspeed-mooncake store library (libmooncake_store.so) is missing"
        )
    settings = MooncakeStoreSettings.resolve(
        {
            "master_server_address": args.master,
            "local_hostname": args.local_hostname,
            "metadata_server": args.metadata_server,
            "protocol": args.protocol,
            "device_name": args.device_name,
            "global_segment_size": args.segment_bytes,
            "tenant_id": args.tenant_id,
        },
        library=library,
        tp_size=1,
        pp_size=1,
    )
    host_buffer = torch.zeros(
        _PROVIDER_LCM_BLOCKS * _PROVIDER_LCM_BLOCK_BYTES, dtype=torch.uint8
    )
    executor = start_l3_executor(
        host_buffer,
        groups=((1, _PROVIDER_LCM_BLOCK_BYTES, 1),),
        lcm_block_bytes=_PROVIDER_LCM_BLOCK_BYTES,
        owner_rank=0,
        tp_rank=0,
        deployment_id=f"{args.tenant_id}-provider",
        mooncake=settings,
    )
    try:
        print(f"\n{PROVIDER_READY_TOKEN}", flush=True)
        deadline = time.monotonic() + args.max_seconds
        while not stop.is_set() and time.monotonic() < deadline:
            stop.wait(0.5)
    finally:
        executor.close(_PROVIDER_CLOSE_TIMEOUT_S)


if __name__ == "__main__":
    if "--l3-provider" in sys.argv:
        _provider_main(sys.argv[1:])
    else:
        raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))

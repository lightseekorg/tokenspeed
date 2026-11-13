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

"""Direct real/simulated decode forwards; run under the host's GPU lease.

Server flags follow ``--``. The same token JSON and flags (except --simulate-tokens)
are used for both runs. This script opens no serving endpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path


def _read_tokens(
    path, *, batches, contexts, vocab_size, max_batch_size, max_context, width
):
    """Validate CPU inputs before loading weights or joining GPU collectives."""
    if max(batches) > max_batch_size or max(contexts) + width > max_context:
        raise ValueError("Benchmark matrix exceeds configured batch/context capacity")
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, list) or len(raw) < max(batches):
        raise ValueError(
            "Token JSON must contain at least max(batch sizes) token lists"
        )
    selected = raw[: max(batches)]
    for prompt in selected:
        if (
            not isinstance(prompt, list)
            or len(prompt) < max(contexts) + 1
            or any(type(t) is not int or not 0 <= t < vocab_size for t in prompt)
        ):
            raise ValueError("Token input is too short or contains invalid token IDs")
    return selected


def _plan(
    batch, prompts, next_tokens, *, request_token_history, ngram_width, max_new_tokens
):
    from tokenspeed.runtime.execution.types import (
        NGramInputs,
        PlannedForward,
        RequestHistorySeeds,
        StopCriteria,
    )
    from tokenspeed.runtime.sampling.sampling_params import SamplingParams

    extending = batch.num_extends() > 0
    positions = (
        tuple(batch.extend_prefix_lens)
        if extending
        else tuple(len(prompts[rid]) for rid in batch.request_ids)
    )
    ngram = None
    if ngram_width:
        # Keep prefill and decode tail ordering/padding in this shared planner.
        rows = []
        for rid, position in zip(batch.request_ids, positions, strict=True):
            history = prompts[rid]
            current = history[position] if extending else next_tokens[rid]
            tail = tuple(reversed(history[max(0, position - ngram_width) : position]))
            rows.append((current, *tail, *([-1] * (ngram_width - len(tail)))))
        ngram = NGramInputs(tokens=tuple(rows), positions=positions)
    return PlannedForward(
        forward_op=batch,
        sampling_params_list=[
            SamplingParams(
                temperature=0, ignore_eos=True, max_new_tokens=max_new_tokens, seed=1234
            )
        ]
        * len(batch.request_ids),
        dp_metadata=None,
        grammar_inputs=None,
        multimodal_context=None,
        ngram_inputs=ngram,
        # Match the serving planner: Engram n-gram windows do not imply that
        # the model owns a separate full request-token-history buffer.
        request_history_seeds=(
            RequestHistorySeeds(
                slots=tuple(batch.request_pool_indices),
                prefix_lengths=positions,
                tokens=tuple(
                    tuple(prompts[rid][:n])
                    for rid, n in zip(batch.request_ids, positions, strict=True)
                ),
            )
            if extending and request_token_history
            else None
        ),
        input_logprob_plan=None,
        sampled_top_k=0,
        stop_criteria=tuple(
            StopCriteria((), len(prompts[rid]) + max_new_tokens)
            for rid in batch.request_ids
        ),
    )


def _prefill(
    config,
    prompts,
    handle,
    *,
    request_token_history,
    ngram_width,
    max_new_tokens,
    request_prefix,
):
    from tokenspeed_scheduler import (
        ExecutionEvent,
        ForwardEvent,
        RequestSpec,
        Scheduler,
    )

    from tokenspeed.runtime.engine.scheduler_utils import cache_ops_from_plan
    from tokenspeed.runtime.execution.simulation import SimulatedDecodeFixture

    scheduler = Scheduler(config)
    by_id = {f"{request_prefix}-{i}": prompt for i, prompt in enumerate(prompts)}
    specs = []
    for rid, prompt in by_id.items():
        spec = RequestSpec()
        spec.request_id, spec.tokens, spec.max_new_tokens = rid, prompt, max_new_tokens
        specs.append(spec)
    scheduler.submit_requests(specs)
    completed, next_tokens, chunks = {}, {}, []
    while len(completed) < len(prompts):
        plan = scheduler.next_execution_plan()
        if len(plan.forward) != 1 or not plan.forward[0].request_ids:
            raise RuntimeError("Cache cannot admit this real-prefill case")
        batch = plan.forward[0]
        if batch.num_extends() != len(batch.request_ids):
            raise RuntimeError("Cache pressure interrupted the real-prefill case")
        started = time.perf_counter()
        result = handle.execute(
            plan,
            _plan(
                batch,
                by_id,
                next_tokens,
                request_token_history=request_token_history,
                ngram_width=ngram_width,
                max_new_tokens=max_new_tokens,
            ),
            cache_ops=cache_ops_from_plan(plan).device,
        )
        handle.settle()
        result.sync()
        values = result.output_tokens.flatten().tolist()
        if result.output_nan_flags is not None and result.output_nan_flags.any():
            raise RuntimeError("Nonfinite output during real prefill")
        chunks.append(
            {
                "seconds": time.perf_counter() - started,
                "tokens": sum(batch.input_lengths),
                "prefix_lengths": batch.extend_prefix_lens,
            }
        )
        event = ExecutionEvent()
        for i, (rid, prefix, count) in enumerate(
            zip(
                batch.request_ids,
                batch.extend_prefix_lens,
                batch.input_lengths,
                strict=True,
            )
        ):
            feedback = ForwardEvent.ExtendResult()
            feedback.request_id = rid
            if prefix + count == len(by_id[rid]):
                feedback.tokens = [values[i]]
                completed[rid] = feedback
                next_tokens[rid] = values[i]
            else:
                feedback.tokens = []
                event.add_event(feedback)
        scheduler.advance(event)
    event = ExecutionEvent()
    for feedback in completed.values():
        event.add_event(feedback)
    scheduler.advance(event)
    plan = scheduler.next_execution_plan()
    if (
        len(plan.forward) != 1
        or plan.forward[0].num_extends()
        or len(plan.forward[0].request_ids) != len(prompts)
    ):
        raise RuntimeError("Real prefill did not yield the full decode batch")
    ids = plan.forward[0].request_ids
    return (
        SimulatedDecodeFixture(
            scheduler,
            len(chunks),
            plan,
            tuple(len(by_id[rid]) for rid in ids),
            tuple(next_tokens[rid] for rid in ids),
            tuple(tuple(by_id[rid]) for rid in ids),
        ),
        chunks,
    )


def _worker(rank, args, server_args, ports):
    output = Path(args.output)
    with (output / f"rank-{rank}.log").open("w", buffering=1) as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        try:
            _run_rank(rank, args, server_args, ports)
        except BaseException:
            # The supervisor stops sibling ranks on a fatal GPU error. Persist
            # this rank's traceback before spawn's parent can be interrupted.
            traceback.print_exc()
            raise


def _run_rank(rank, args, server_args, ports):
    started = time.perf_counter()
    import torch

    from tokenspeed.runtime.engine.rank_setup import (
        init_distributed,
        load_model_config,
        scheduler_config_from_args,
    )
    from tokenspeed.runtime.engine.scheduler_utils import (
        cache_ops_from_plan,
        engram_context_len,
    )
    from tokenspeed.runtime.execution.device import build_device_side
    from tokenspeed.runtime.execution.simulation import allocate_simulated_decode
    from tokenspeed.runtime.utils.jit_compile_check import install_jit_compile_check
    from tokenspeed.runtime.utils.startup_timing import startup_phase

    install_jit_compile_check()

    output = Path(args.output)
    receipt = output / f"rank-{rank}.json"
    report = {
        "status": "running",
        "rank": rank,
        "simulate": server_args.simulate,
        "nan_detection_enabled": server_args.enable_nan_detection,
        "timing": {},
        "cases": [],
    }
    try:
        mapping = server_args.mapping
        mapping.rank = rank
        if (
            mapping.attn.dp_size != 1
            or mapping.pp_size != 1
            or mapping.attn.kvp_size != 1
        ):
            raise ValueError(
                "Direct benchmark currently requires attention DP=1, PP=1, KVP=1"
            )
        if (
            not server_args.disable_overlap_schedule
            or server_args.enable_prefix_caching
            or server_args.enable_kvstore
        ):
            raise ValueError(
                "Pass --disable-overlap-schedule --disable-prefix-caching --disable-kvstore"
            )
        if (
            not server_args.retraction_snapshot_pool_disabled
            or server_args.kv_offload_config is not None
        ):
            raise ValueError(
                "Pass --retraction-snapshot-ratio 0 and disable KV offloading"
            )
        if server_args.disaggregation_mode != "null":
            raise ValueError("Direct benchmark requires aggregate execution")
        gpu = rank % mapping.nprocs_per_node + server_args.base_gpu_id
        with startup_phase("direct_benchmark.init", rank=rank):
            model_config = load_model_config(
                server_args, server_args.model, draft_of=None
            )
            draft = (
                load_model_config(
                    server_args,
                    server_args.speculative_draft_model_path,
                    draft_of=model_config,
                )
                if server_args.speculative_draft_model_path is not None
                else None
            )
            width = (
                server_args.speculative_num_draft_tokens
                if server_args.speculative_algorithm
                else 1
            )
            raw = _read_tokens(
                args.tokens_json,
                batches=args.batches,
                contexts=args.contexts,
                vocab_size=model_config.input_vocab_size,
                max_batch_size=server_args.max_num_seqs,
                max_context=model_config.context_len,
                width=width,
            )
            report["timing"]["config_seconds"] = time.perf_counter() - started
            phase = time.perf_counter()
            free = init_distributed(
                server_args,
                ports,
                gpu_id=gpu,
                global_rank=rank,
                hidden_size=model_config.hidden_size,
            )
            report["timing"]["distributed_seconds"] = time.perf_counter() - phase
            phase = time.perf_counter()
            device = build_device_side(
                server_args=server_args,
                model_config=model_config,
                draft_model_config=draft,
                gpu_id=gpu,
                global_rank=rank,
                attn_tp_rank=mapping.attn.tp_rank,
                min_per_gpu_mem=free,
                overlap_schedule_depth=0,
                decode_input_tokens=width,
                max_batch_size=server_args.max_num_seqs,
                kv_transfer_builder=lambda *_args, **_kwargs: None,
            )
            report["timing"]["device_build_seconds"] = time.perf_counter() - phase
        report["timing"]["initialization_seconds"] = time.perf_counter() - started
        config = scheduler_config_from_args(
            server_args,
            device.specs,
            storage_config=None,
            max_scheduled_tokens=server_args.chunked_prefill_size,
            max_batch_size=server_args.max_num_seqs,
            decode_input_tokens=width,
            overlap_schedule_depth=0,
            enable_kv_cache_events=False,
            prefix_replay_tokens=0,
        )
        handle = device.handle
        handle.bind_forward_thread()
        report.update(
            decode_width=width,
            world_size=mapping.world_size,
            model=server_args.model,
            tokens_json=str(Path(args.tokens_json).resolve()),
            cache_capacity={
                "device_pages": config.num_device_pages,
                "host_pages": config.num_host_pages,
                "snapshot_pages": config.num_snapshot_pages,
                "prefix_granularity": config.prefix_granularity,
            },
        )
        receipt.write_text(json.dumps(report, indent=2) + "\n")
        for length in args.contexts:
            for batch_size in args.batches:
                phase = time.perf_counter()
                prompts = [p[:length] for p in raw[:batch_size]]
                ngram_width = engram_context_len(model_config.hf_text_config)
                if server_args.simulate:
                    fixture = allocate_simulated_decode(
                        config,
                        prompts,
                        [p[length] for p in raw[:batch_size]],
                        max_new_tokens=1024,
                    )
                    chunks = []
                else:
                    fixture, chunks = _prefill(
                        config,
                        prompts,
                        handle,
                        request_token_history=model_config.requires_request_token_history,
                        ngram_width=ngram_width,
                        max_new_tokens=1024,
                        request_prefix=f"real-L{length}-B{batch_size}",
                    )
                plan = fixture.decode_plan
                handle.execute(plan, None, cache_ops=cache_ops_from_plan(plan).device)
                if server_args.simulate:
                    handle.prepare_simulated_decode(
                        fixture,
                        [
                            [
                                (token + j) % model_config.input_vocab_size
                                for j in range(width)
                            ]
                            for token in fixture.next_token_ids
                        ],
                    )
                prep = time.perf_counter() - phase
                batch = plan.forward[0]
                planned = _plan(
                    batch,
                    dict(zip(batch.request_ids, fixture.prompt_tokens, strict=True)),
                    dict(zip(batch.request_ids, fixture.next_token_ids, strict=True)),
                    request_token_history=model_config.requires_request_token_history,
                    ngram_width=ngram_width,
                    max_new_tokens=1024,
                )
                timing = handle.benchmark_decode(
                    planned,
                    iterations=args.iterations,
                    warmup=args.warmup,
                    cache_flush_bytes=args.evict_mib * 1024**2,
                )
                report["cases"].append(
                    {
                        "batch_size": batch_size,
                        "context_length": length,
                        "preparation_seconds": prep,
                        "process_elapsed_seconds": time.perf_counter()
                        - args.process_started,
                        "prefill_chunks": chunks,
                        **timing,
                    }
                )
                receipt.write_text(json.dumps(report, indent=2) + "\n")
                del fixture
        report["status"] = "complete"
    except BaseException as exc:
        report.update(status="failed", error=repr(exc))
        raise
    finally:
        report["timing"]["rank_total_seconds"] = time.perf_counter() - started
        receipt.write_text(json.dumps(report, indent=2) + "\n")


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokens-json", required=True)
    parser.add_argument(
        "--batches", required=True, type=lambda s: [int(x) for x in s.split(",")]
    )
    parser.add_argument(
        "--contexts", required=True, type=lambda s: [int(x) for x in s.split(",")]
    )
    parser.add_argument("--iterations", required=True, type=int)
    parser.add_argument("--warmup", required=True, type=int)
    parser.add_argument("--evict-mib", required=True, type=int)
    args, remaining = parser.parse_known_args()
    args.process_started = started
    if remaining and remaining[0] == "--":
        remaining = remaining[1:]
    if (
        min(args.batches + args.contexts) <= 0
        or args.iterations < 1
        or args.warmup < 1
        or args.evict_mib < 0
    ):
        parser.error(
            "batch/context/iterations/warmup must be positive; eviction bytes nonnegative"
        )
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    args.output = str(output)
    os.environ["TOKENSPEED_STARTUP_TIMING"] = "1"
    state = {"status": "running", "command": sys.argv, "started_unix": time.time()}
    (output / "run.json").write_text(json.dumps(state, indent=2) + "\n")
    try:
        import torch.multiprocessing as mp

        from tokenspeed.runtime.utils.server_args import PortArgs, prepare_server_args

        server_args = prepare_server_args(remaining)
        if server_args.mapping.nnodes != 1:
            raise ValueError("Direct benchmark currently supports one host")
        ports = PortArgs.init_new(server_args)
        mp.spawn(
            _worker,
            args=(args, server_args, ports),
            nprocs=server_args.mapping.world_size,
            join=True,
        )
        ranks = [
            json.loads((output / f"rank-{i}.json").read_text())
            for i in range(server_args.mapping.world_size)
        ]
        state.update(status="complete", ranks=ranks)
    except BaseException as exc:
        state.update(status="failed", error=repr(exc))
        raise
    finally:
        state["wall_seconds"] = time.perf_counter() - started
        (output / "run.json").write_text(json.dumps(state, indent=2) + "\n")


if __name__ == "__main__":
    main()

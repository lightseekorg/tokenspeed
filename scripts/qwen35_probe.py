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

"""Experiment-only observers around the real Engine's ModelRunner.

No serving code imports this module. Spawned experiment workers install these
observers too. Full logits leave NPU only in correctness mode, after forward.
"""

from __future__ import annotations

import functools
import json
import os
import time
from pathlib import Path

import torch


def rank_directory(root: Path) -> Path:
    """Keep TP workers' evidence separate, preserving the TP=1 file layout."""
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
        root = root / f"rank-{torch.distributed.get_rank()}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def install() -> None:
    """Observe real forwards and audit parameter loading without changing math."""
    from tokenspeed.runtime.execution.model_runner import ModelRunner
    from tokenspeed.runtime.models.qwen3_5 import Qwen3_5ForConditionalGeneration

    if getattr(ModelRunner.forward, "_qwen35_probe", False):
        return
    original_forward = ModelRunner.forward
    original_load = Qwen3_5ForConditionalGeneration.load_weights
    current_request = None
    step = 0
    layer_values = {}
    collectives = []
    collect_communication = False
    profiled_phases = set()

    from tokenspeed.runtime.distributed.comm_backend.hccl import HcclBackend

    def observe_collective(name, original):
        @functools.wraps(original)
        def observed(backend, tensor, *args, **kwargs):
            group_index = 1 if name == "all_gather_single" else 0
            group = kwargs["group"] if "group" in kwargs else args[group_index]
            if collect_communication and len(group) > 1:
                payload = args[0] if name == "all_gather_single" else tensor
                tensors = (payload,) if isinstance(payload, torch.Tensor) else payload
                collectives.append(
                    dict(
                        name=name,
                        shapes=[list(t.shape) for t in tensors],
                        dtypes=[str(t.dtype) for t in tensors],
                        payload_bytes=sum(
                            t.numel() * t.element_size() for t in tensors
                        ),
                    )
                )
            return original(backend, tensor, *args, **kwargs)

        return observed

    for name in ("all_reduce", "all_gather_single", "reduce_scatter"):
        setattr(HcclBackend, name, observe_collective(name, getattr(HcclBackend, name)))

    @functools.wraps(original_load)
    def audited_load(self, weights):
        checkpoint_names = []

        def observed():
            for name, tensor in weights:
                checkpoint_names.append(name)
                yield name, tensor

        loaded = original_load(self, observed())
        parameters = dict(self.named_parameters(remove_duplicate=False))
        loaded_objects = {id(parameters[name]) for name in loaded if name in parameters}
        missing = [
            name for name, p in parameters.items() if id(p) not in loaded_objects
        ]
        devices = sorted({str(p.device) for p in parameters.values()})
        audit = dict(
            rank=(
                torch.distributed.get_rank()
                if torch.distributed.is_initialized()
                else 0
            ),
            pid=os.getpid(),
            visible_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
            logical_device=torch.npu.current_device(),
            checkpoint_tensors=len(checkpoint_names),
            loaded_names=len(loaded),
            parameter_names=len(parameters),
            missing=missing,
            unexpected=sorted(set(loaded) - parameters.keys()),
            devices=devices,
            skipped_vision=sum("visual" in name for name in checkpoint_names),
            skipped_mtp=sum("mtp" in name for name in checkpoint_names),
            unique_parameter_count=sum(p.numel() for p in self.parameters()),
            allocated_after_load=torch.npu.memory_allocated(),
            parameter_shapes={name: list(p.shape) for name, p in parameters.items()},
        )
        root = rank_directory(Path(os.environ["QWEN35_PROBE_DIR"]))
        (root / "weight-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
        assert not missing, f"Unloaded parameters: {missing}"
        assert not audit["unexpected"], audit
        assert devices == [f"npu:{torch.npu.current_device()}"], devices
        if os.environ.get("QWEN35_LAYER_TRACE"):

            def trace_layer(index):
                def hook(module, inputs, output):
                    hidden, residual = output
                    layer_values[str(index)] = (
                        (hidden + residual).detach().float().cpu()
                    )

                return hook

            for index, layer in enumerate(self.model.layers):
                layer.register_forward_hook(trace_layer(index))
        return loaded

    @functools.wraps(original_forward)
    def observed_forward(self, ctx, input_ids, positions, *args, **kwargs):
        nonlocal current_request, step, collect_communication
        root = Path(os.environ["QWEN35_PROBE_DIR"])
        control_path = root / "active.json"
        if not control_path.exists():
            return original_forward(self, ctx, input_ids, positions, *args, **kwargs)
        control = json.loads(control_path.read_text())
        if current_request != control["id"]:
            # The overlap scheduler can finish a previous request's queued
            # decode after its final frame reached the caller. A new request
            # starts at its own prefill, never at that trailing forward.
            if not ctx.forward_mode.is_extend():
                return original_forward(
                    self, ctx, input_ids, positions, *args, **kwargs
                )
            current_request, step = control["id"], 0
            profiled_phases.clear()
            torch.npu.reset_peak_memory_stats()
        request_dir = rank_directory(root) / current_request
        request_dir.mkdir(exist_ok=True)
        if step == 0:
            pool = ctx.token_to_kv_pool
            (request_dir.parent / "cache-plan.json").write_text(
                json.dumps(
                    dict(
                        pool_type=type(pool).__name__,
                        token_capacity=pool.arena.size,
                        arena_bytes=pool.arena.allocated_device_bytes,
                        states={
                            str(layer): [
                                dict(shape=list(t.shape), dtype=str(t.dtype))
                                for t in pool.get_state_buffers(layer)
                            ]
                            for layer in pool.state_group_by_layer
                        },
                    ),
                    indent=2,
                )
                + "\n"
            )
        assert input_ids.device.type == positions.device.type == "npu"
        if control["mode"] == "benchmark":
            output = original_forward(self, ctx, input_ids, positions, *args, **kwargs)
            # No per-forward synchronization, events or CPU logits in timed runs.
            record = dict(
                step=step,
                mode=ctx.forward_mode.name,
                tokens=input_ids.numel(),
                batch_size=ctx.bs,
                allocated=torch.npu.memory_allocated(),
                reserved=torch.npu.memory_reserved(),
                peak_allocated=torch.npu.max_memory_allocated(),
                peak_reserved=torch.npu.max_memory_reserved(),
            )
            with (request_dir / "steps.jsonl").open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            step += 1
            return output
        torch.npu.synchronize()
        start_event = torch.npu.Event(enable_timing=True)
        end_event = torch.npu.Event(enable_timing=True)
        profiler = None
        phase = "decode" if ctx.forward_mode.is_decode() else "prefill"
        if control["mode"] == "profile" and (
            step == 0 or (phase == "decode" and phase not in profiled_phases)
        ):
            import torch_npu

            profiled_phases.add(phase)
            profiler = torch_npu.profiler.profile(
                activities=[
                    torch_npu.profiler.ProfilerActivity.CPU,
                    torch_npu.profiler.ProfilerActivity.NPU,
                ],
                schedule=torch_npu.profiler.schedule(
                    wait=0, warmup=0, active=1, repeat=1
                ),
                record_shapes=True,
                with_stack=False,
                experimental_config=torch_npu.profiler._ExperimentalConfig(
                    profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                    export_type=torch_npu.profiler.ExportType.Text,
                ),
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                    str(request_dir / f"profile-{step}")
                ),
            )
            profiler.start()
        started = time.perf_counter()
        collectives.clear()
        collect_communication = True
        if os.environ.get("QWEN35_LAYER_TRACE"):
            layer_values.clear()
        start_event.record()
        output = original_forward(self, ctx, input_ids, positions, *args, **kwargs)
        collect_communication = False
        end_event.record()
        torch.npu.synchronize()
        wall_ms = (time.perf_counter() - started) * 1000
        if profiler is not None:
            profiler.step()
            profiler.stop()
        assert output.next_token_logits.device.type == "npu"
        record = dict(
            step=step,
            mode=ctx.forward_mode.name,
            tokens=input_ids.numel(),
            batch_size=ctx.bs,
            collectives=collectives.copy(),
            wall_ms=wall_ms,
            event_span_ms=start_event.elapsed_time(end_event),
            allocated=torch.npu.memory_allocated(),
            reserved=torch.npu.memory_reserved(),
            peak_allocated=torch.npu.max_memory_allocated(),
            peak_reserved=torch.npu.max_memory_reserved(),
        )
        with (request_dir / "steps.jsonl").open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        if control["mode"] == "capture":
            logits = output.next_token_logits.detach().float().cpu()
            assert torch.isfinite(logits).all()
            torch.save(
                dict(
                    input_ids=input_ids.cpu(), positions=positions.cpu(), logits=logits
                ),
                request_dir / f"{step:04d}.pt",
            )
            if os.environ.get("QWEN35_LAYER_TRACE"):
                torch.save(layer_values, request_dir / f"layers-{step:04d}.pt")
        step += 1
        return output

    observed_forward._qwen35_probe = True
    ModelRunner.forward = observed_forward
    Qwen3_5ForConditionalGeneration.load_weights = audited_load


def engine_kwargs(model: str) -> dict:
    """The same batch-one, FP16, eager configuration as serve_qwen35.sh."""
    return dict(
        model=model,
        device="npu",
        dtype="float16",
        kv_cache_dtype="float16",
        attention_backend="mha",
        sampling_backend="greedy",
        disable_prefill_graph=True,
        disable_pdl=True,
        enforce_eager=True,
        enable_prefix_caching=False,
        disable_kvstore=True,
        language_model_only=True,
        enable_output_logprobs=True,
        seed=2026,
        mamba_ssm_dtype="float32",
        world_size=1,
        max_model_len=4096,
        max_num_seqs=1,
        max_total_tokens=8192,
        chunked_prefill_size=4096,
        prefix_granularity=128,
        retraction_snapshot_ratio=0,
        disable_autotune=True,
        log_level="info",
    )

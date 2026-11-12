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

"""Validate final-MLP observation on a real checkpoint and full eager forward.

Uses the ordinary TokenSpeed loader, cache recipe, paged-attention router and
ModelRunner. No alternate model implementation or forward replacement is used.
Run in an installed TokenSpeed GPU environment; see the observation design doc.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import inspect
import json
import math
import subprocess
from pathlib import Path

import torch
from safetensors.torch import save_file

from tokenspeed.runtime.engine.rank_setup import init_distributed, load_model_config
from tokenspeed.runtime.execution.breakable_cuda_graph import active_forward
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.execution.model_runner import ModelRunner
from tokenspeed.runtime.execution.output_layout import ForwardOutputLayout
from tokenspeed.runtime.inspection.final_mlp import (
    FinalMLPObservation,
    FinalMLPObserver,
)
from tokenspeed.runtime.layers.attention.backends.paged.router import CacheGroupRouter
from tokenspeed.runtime.layers.attention.registry import create_attn_components
from tokenspeed.runtime.layers.logits_processor import prepare_logits_communication
from tokenspeed.runtime.layers.paged_attention import bind_cache_groups
from tokenspeed.runtime.utils.hf_transformers_utils import get_tokenizer
from tokenspeed.runtime.utils.server_args import PortArgs, ServerArgs


class ColdPrefill:
    """One request, zero cached prefix, production eager attention and weights."""

    def __init__(self, args: argparse.Namespace) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("This integration example requires a GPU")
        torch.cuda.set_device(args.gpu_id)
        self.device = torch.device("cuda", args.gpu_id)
        self.max_tokens = args.max_tokens
        server = ServerArgs(
            model=args.model,
            revision=args.revision,
            trust_remote_code=False,
            dtype="bfloat16",
            kv_cache_dtype="bfloat16",
            quantization=None,
            load_format="safetensors",
            device="cuda",
            world_size=1,
            attn_tp_size=1,
            dense_tp_size=1,
            data_parallel_size=1,
            pipeline_parallel_size=1,
            base_gpu_id=args.gpu_id,
            max_num_seqs=1,
            max_model_len=args.max_tokens,
            max_prefill_tokens=args.max_tokens,
            chunked_prefill_size=args.max_tokens,
            max_total_tokens=math.ceil(args.max_tokens / 64) * 64 + 64,
            attention_backend=args.attention_backend,
            dense_gemm_backend="auto",
            numerics="auto",
            layer_boundary_norm=args.norm_mode,
            enforce_eager=True,
            disable_prefill_graph=True,
            prefill_graph_max_tokens=0,
            decode_monokernel=False,
            speculative_algorithm=None,
            enable_allreduce_fusion=False,
            disable_autotune=True,
            enable_memory_saver=False,
            enable_prefix_caching=False,
            disable_kvstore=True,
        )
        server.mapping.rank = 0
        self.server = server
        self.model_config = load_model_config(server, server.model, draft_of=None)
        init_distributed(
            server,
            PortArgs.init_new(server),
            gpu_id=args.gpu_id,
            global_rank=0,
            hidden_size=self.model_config.hidden_size,
        )
        self.runner = ModelRunner(
            self.model_config,
            server,
            args.gpu_id,
            0,
            checkpoint_load_group=None,
        )
        self.model = self.runner.model
        self.runner.prepare_communication_runtime(args.max_tokens)
        prepare_logits_communication(self.model)
        # Explicit budget: use the production cache builder, but do not size
        # this small analysis run from most of the GPU's free memory.
        attention = create_attn_components(
            server,
            self.model_config,
            args.gpu_id,
            0,
            torch.cuda.get_device_properties(self.device).total_memory,
            enable_memory_saver=False,
            draft_model_config=None,
            decode_input_tokens=1,
            overlap_schedule_depth=0,
            graph_reserve_bytes=0,
            post_profile_bytes=0,
            probe_batch_rows=None,
            profiled_cache_bytes=args.cache_mib * 1024**2,
            reuse_target_backend=None,
            reuse_draft_backend=None,
        )
        self.backend = attention.attn_backend
        self.pool = attention.token_to_kv_pool
        if not isinstance(self.backend, CacheGroupRouter):
            raise ValueError("This example requires ordinary paged Qwen3 attention")
        if any(
            kind != "full_history" for kind in self.backend.geometry.attn_kinds.values()
        ):
            raise ValueError(
                "This example supports full-history Qwen3 cache groups only"
            )
        self.backend.configure_runtime(
            request_slot_capacity=2,
            cache_group_specs=tuple(self.pool.arena.cache_group_specs),
            cache_group_page_counts=self.pool.arena.cache_group_page_counts,
        )
        # Despite its name, this prepares the persistent metadata buffers used
        # by eager forwards too. It does not capture or replay a graph.
        self.backend.init_cuda_graph_state(1, max_extend_tokens=args.max_tokens)
        bind_cache_groups(self.model, self.pool)
        self.tokenizer = get_tokenizer(
            server.tokenizer,
            tokenizer_mode=server.tokenizer_mode,
            trust_remote_code=False,
            revision=args.revision,
        )

    def encode(self, prompt: str) -> torch.Tensor:
        ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        if not 1 <= len(ids) <= self.max_tokens:
            raise ValueError(
                f"Prompt has {len(ids)} tokens; supported range is [1, {self.max_tokens}]"
            )
        return torch.tensor(ids, device=self.device, dtype=torch.int32)

    @torch.no_grad()
    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """Reset native cache and run every transformer layer from prefix zero."""
        rows = ids.numel()
        # The cold-prefix metadata already prevents reading old history. Clear
        # storage as an additional control; this is not part of a timed forward.
        for layer_id in range(self.model_config.num_attention_layers):
            self.pool.get_key_buffer(layer_id).zero_()
            self.pool.get_value_buffer(layer_id).zero_()
        tables_cpu = {}
        for gid in self.backend.group_ids:
            page_tokens = self.backend.geometry.granularity_of(gid)
            pages = math.ceil(rows / page_tokens)
            if pages >= self.pool.arena.cache_group_page_counts[gid]:
                raise ValueError(f"Cache budget cannot hold {rows} rows in {gid}")
            # Page zero is the reserved null page; this one request owns all
            # other pages used by this example. There is no scheduler sharing.
            tables_cpu[gid] = torch.arange(1, pages + 1, dtype=torch.int32)[None]
        tables = {gid: table.to(self.device) for gid, table in tables_cpu.items()}
        length_cpu = torch.tensor([rows], dtype=torch.int32)
        prefix_cpu = torch.zeros(1, dtype=torch.int32)
        length = length_cpu.to(self.device)
        self.backend.init_forward_metadata(
            1,
            1,
            torch.zeros(1, device=self.device, dtype=torch.int64),
            length,
            ForwardMode.EXTEND,
            block_tables=tables,
            extend_seq_lens=length,
            extend_seq_lens_cpu=length_cpu,
            extend_prefix_lens=prefix_cpu.to(self.device),
            extend_prefix_lens_cpu=prefix_cpu,
            extend_replay_lens_cpu=prefix_cpu,
            extend_prompt_lens_cpu=length_cpu,
            extend_with_prefix=False,
            query_shard=None,
            block_tables_cpu=tables_cpu,
        )
        ctx = ForwardContext(
            attn_backend=self.backend,
            token_to_kv_pool=self.pool,
            bs=1,
            num_extends=1,
            input_num_tokens=rows,
            forward_mode=ForwardMode.EXTEND,
            output_layout=ForwardOutputLayout(1, 1, 0, 1),
            gather_ids=torch.tensor([rows - 1], device=self.device, dtype=torch.int64),
        )
        with active_forward(ctx):
            output = self.runner.forward(
                ctx,
                ids,
                torch.arange(rows, device=self.device, dtype=torch.int64),
            )
        if output.next_token_logits is None:
            raise RuntimeError("Full selected logits were not returned")
        return output.next_token_logits.detach().clone()


def compare(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    """Record complete-array equality and errors, not only the winning token."""
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    difference = actual.float() - expected.float()
    return {
        "exact_equal": bool(torch.equal(actual, expected)),
        "finite": finite,
        "different_values": int(torch.count_nonzero(actual != expected)),
        "max_abs_error": float(difference.abs().max()) if finite else None,
        "rms_error": float(difference.square().mean().sqrt()) if finite else None,
        "same_argmax": bool(torch.equal(actual.argmax(-1), expected.argmax(-1))),
    }


def trace_tensors(trace: FinalMLPObservation) -> dict[str, torch.Tensor]:
    return {
        item.name: value
        for item in dataclasses.fields(trace)
        if isinstance(value := getattr(trace, item.name), torch.Tensor)
    }


@torch.no_grad()
def evaluate(
    runtime: ColdPrefill, prompt: str, channels: tuple[int, ...], output: Path
) -> dict:
    ids = runtime.encode(prompt)
    rows = ids.numel()
    for _ in range(2):
        runtime.forward(ids)
    baseline = runtime.forward(ids)
    observer = FinalMLPObserver(
        runtime.model,
        num_rows=rows,
        num_logits_rows=1,
        execution="eager",
        speculative=False,
        monokernel=False,
    )
    with observer:
        with observer.record():
            observed = runtime.forward(ids)
        trace = observer.snapshot()
        replay = observer.replay(trace, ablate_channels=channels)
        retained = {name: value.clone() for name, value in trace_tensors(trace).items()}
        # A different input at the same physical shape must overwrite the
        # private workspace while the first exported snapshot remains intact.
        other_ids = ids.clone()
        other_ids[0] = (other_ids[0] + 1) % runtime.model.config.vocab_size
        with observer.record():
            runtime.forward(other_ids)
        second = observer.snapshot()
        independent = all(
            torch.equal(value, retained[name])
            for name, value in trace_tensors(trace).items()
        )
        workspace_changed = not torch.equal(second.gate_up, trace.gate_up)

    def ablate(module, inputs, activation):
        changed = activation.clone()
        changed[:, list(channels)] = 0
        return changed

    # Independent check: rerun the full, actual model with an activation
    # intervention at the same native seam, rather than trusting suffix code.
    handle = runtime.model.model.layers[-1].mlp.act_fn.register_forward_hook(ablate)
    try:
        full_intervention = runtime.forward(ids)
    finally:
        handle.remove()
    restored = runtime.forward(ids)
    checks = {
        "observer_on_vs_off": compare(observed, baseline),
        "recorded_logits_vs_off": compare(trace.selected_logits, baseline),
        "saved_output_suffix_vs_off": compare(replay.suffix_logits, baseline),
        "recomputed_down_vs_recorded": compare(
            replay.recomputed_mlp_output, trace.mlp_output
        ),
        "recomputed_suffix_vs_off": compare(replay.recomputed_logits, baseline),
        "suffix_vs_full_model_ablation": compare(
            replay.intervened_logits, full_intervention
        ),
        "restored_vs_off": compare(restored, baseline),
    }
    tensors = {
        **trace_tensors(trace),
        "input_ids": ids,
        "positions": torch.arange(rows, device=ids.device, dtype=torch.int64),
        "baseline_logits": baseline,
        "observed_logits": observed,
        "saved_output_suffix_logits": replay.suffix_logits,
        "recomputed_suffix_logits": replay.recomputed_logits,
        "intervened_suffix_logits": replay.intervened_logits,
        "full_model_intervened_logits": full_intervention,
        "restored_logits": restored,
    }
    save_file(
        {name: value.detach().cpu().contiguous() for name, value in tensors.items()},
        str(output),
    )
    return {
        "prompt": prompt,
        "token_ids": ids.tolist(),
        "physical_rows": rows,
        "selected_rows": trace.logit_rows.tolist(),
        "layer_id": trace.layer_id,
        "ablated_channels": list(channels),
        "checks": checks,
        "snapshot_unchanged_after_reuse": independent,
        "second_record_changed": workspace_changed,
        "intervention_effect": compare(full_intervention, baseline),
        "baseline_next_token": int(baseline.argmax(-1)[0]),
        "intervened_next_token": int(full_intervention.argmax(-1)[0]),
        "artifact": output.name,
        "passed": independent
        and workspace_changed
        and all(item["exact_equal"] and item["finite"] for item in checks.values()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        help="Qwen3 checkpoint path or Hub ID, e.g. Qwen/Qwen3-0.6B",
    )
    parser.add_argument(
        "--revision",
        required=True,
        help="Pinned checkpoint commit used for both weights and tokenizer",
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        required=True,
        help="JSON array of raw prompt strings; no chat template is added",
    )
    parser.add_argument(
        "--channels", required=True, help="Comma-separated final-MLP channels to ablate"
    )
    parser.add_argument("--norm-mode", choices=("fused", "unfused"), required=True)
    parser.add_argument("--attention-backend", choices=("mha", "triton"), required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New directory for report and tensor artifacts",
    )
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Reject longer prompts; never truncate",
    )
    parser.add_argument(
        "--cache-mib",
        type=int,
        default=128,
        help="Explicit production-cache allocation budget",
    )
    parser.add_argument(
        "--export-suffix-weights",
        action="store_true",
        help="Save runtime gate/up, down, final norm and padded head once",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompts = json.loads(args.prompts.read_text(encoding="utf-8"))
    if (
        not isinstance(prompts, list)
        or not prompts
        or any(not isinstance(p, str) for p in prompts)
    ):
        raise ValueError("--prompts must name a nonempty JSON array of strings")
    channels = tuple(int(item) for item in args.channels.split(","))
    if not channels or len(set(channels)) != len(channels) or min(channels) < 0:
        raise ValueError("--channels requires unique nonnegative integers")
    if args.max_tokens < 1 or args.cache_mib < 1:
        raise ValueError("Token bound and cache budget must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        runtime = ColdPrefill(args)
        if max(channels) >= runtime.model.config.intermediate_size:
            raise ValueError(
                "A requested channel exceeds this model's intermediate width"
            )
        properties = torch.cuda.get_device_properties(runtime.device)
        source_root = Path(__file__).resolve().parents[1]
        source_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source_root, text=True
        ).strip()
        report = {
            "model": args.model,
            "revision": args.revision,
            "source_commit": source_commit,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "observer_sha256": hashlib.sha256(
                Path(inspect.getfile(FinalMLPObserver)).read_bytes()
            ).hexdigest(),
            "device": str(properties),
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "cuda": torch.version.cuda,
            "dtype": "bfloat16",
            "norm_mode": args.norm_mode,
            "attention_backend": runtime.server.attention_backend,
            "numerics": runtime.server.numerics,
            "autotune": False,
            "cache_prefix_tokens": 0,
            "rms_norm_eps": runtime.model.config.rms_norm_eps,
            "vocab_size": runtime.model.config.vocab_size,
            "head_logit_scale": runtime.model.logits_processor.logit_scale,
            "head_softcap": runtime.model.logits_processor.final_logit_softcapping,
            "scope": "Full eager cold prefill; no sampling, generation, serving lifecycle, or graph validation",
            "results": [],
        }
        refusals = {}
        for label, options in (
            (
                "captured",
                {"execution": "captured", "speculative": False, "monokernel": False},
            ),
            (
                "speculative",
                {"execution": "eager", "speculative": True, "monokernel": False},
            ),
            (
                "monokernel",
                {"execution": "eager", "speculative": False, "monokernel": True},
            ),
        ):
            try:
                FinalMLPObserver(
                    runtime.model, num_rows=1, num_logits_rows=1, **options
                )
            except ValueError as error:
                refusals[label] = str(error)
            else:
                raise AssertionError(f"Unsupported {label} configuration was accepted")
        report["unsupported_configuration_refusals"] = refusals
        if args.export_suffix_weights:
            mlp = runtime.model.model.layers[-1].mlp
            save_file(
                {
                    name: value.detach().cpu().contiguous()
                    for name, value in {
                        "gate_up_weight": mlp.gate_up_proj.weight,
                        "down_weight": mlp.down_proj.weight,
                        "final_norm_weight": runtime.model.model.norm.weight,
                        "lm_head_weight": runtime.model.lm_head.weight,
                    }.items()
                },
                str(args.output_dir / "suffix-weights.safetensors"),
            )
        for index, prompt in enumerate(prompts):
            result = evaluate(
                runtime,
                prompt,
                channels,
                args.output_dir / f"prompt-{index:03d}.safetensors",
            )
            report["results"].append(result)
            report["passed"] = all(item["passed"] for item in report["results"])
            (args.output_dir / "report.json").write_text(
                json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
            print(
                json.dumps(
                    {
                        "prompt": index,
                        "passed": result["passed"],
                        "checks": result["checks"],
                    }
                ),
                flush=True,
            )
        if not report["passed"]:
            raise SystemExit(
                "Numerical validation failed; inspect report.json and saved tensors"
            )
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()

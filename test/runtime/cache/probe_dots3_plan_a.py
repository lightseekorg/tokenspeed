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

"""Production cache diagnostic, not model or scheduler lifecycle acceptance.

Uses normal imports and shares the GPU regression checks with
test_dots3_note_pool.py. No synthetic arena views, initializer shims, or
reference-filled persistent caches. Failed checks produce exit status 1.
"""

import argparse
import importlib
import importlib.metadata
import json
from pathlib import Path
from types import SimpleNamespace


def production_inputs(config):
    """Real checkpoint geometry, bounded single-GPU diagnostic serving limits."""
    import torch

    from tokenspeed.runtime.layers.attention.configs.dots3_note import (
        Dots3NoteAttnConfig,
    )

    hf = SimpleNamespace(**config.get("text_config", config))
    model = SimpleNamespace(
        hf_config=hf,
        hf_text_config=hf,
        num_attention_layers=hf.num_hidden_layers,
        context_len=4098,
        dtype=torch.bfloat16,
    )
    args = SimpleNamespace(
        device="cuda",
        speculative_algorithm=None,
        speculative_draft_model_path=None,
        pipeline_parallel_size=1,
        attention_backend="dots3_note",
        kv_cache_dtype="bfloat16",
        kv_cache_quant_method="none",
        attn_tp_size=8,
        data_parallel_size=1,
        mapping=SimpleNamespace(
            attn=SimpleNamespace(
                tp_size=8,
                dp_size=1,
                dcp_size=1,
                dcp_rank=0,
                dcp_group=(0,),
            )
        ),
        prefix_granularity=64,
        spec_context_pad=0,
        max_num_seqs=4,
        max_total_tokens=8192,
        chunked_prefill_size=512,
        mla_chunk_multiplier=4,
        disaggregation_mode="null",
        enable_prefix_caching=True,
    )
    return dict(
        server_args=args,
        model_config=model,
        attn_config=Dots3NoteAttnConfig.generate(args, model, is_draft=False),
        draft_model_config=None,
        draft_attn_config=None,
        cache_budget_bytes=512 << 20,
        decode_input_tokens=1,
        overlap_schedule_depth=0,
    )


def probe_prefill(*, cached, sliding):
    """Expanded-tensor kernel mask control, NOT model projection/pool prefill."""
    import torch
    from tokenspeed_kernel.ops.attention.dots3_note import swa_prefill
    from tokenspeed_kernel.ops.attention.mla import mla_prefill

    n = 545
    nq = 33 if cached else n
    q = torch.zeros((nq, 16, 256), device="cuda", dtype=torch.bfloat16)
    k = torch.zeros((n, 16, 256), device="cuda", dtype=torch.bfloat16)
    v = torch.zeros((n, 16, 128), device="cuda", dtype=torch.bfloat16)
    v[:32] = 1
    cu_q = torch.tensor([0, nq], device="cuda", dtype=torch.int32)
    cu_k = torch.tensor([0, n], device="cuda", dtype=torch.int32)
    prefill = swa_prefill if sliding else mla_prefill
    mask = {"window_left": 512} if sliding else {"is_causal": True}
    actual = prefill(
        q,
        k,
        v,
        cu_q,
        cu_k,
        nq,
        n,
        1 / 16,
        **mask,
        solution="triton",
    )
    positions = torch.arange(n - nq, n, device="cuda")
    begin = (positions - 512).clamp_min(0) if sliding else torch.zeros_like(positions)
    ones = ((positions + 1).clamp_max(32) - begin).clamp_min(0)
    expected = (ones / (positions + 1 - begin)).view(nq, 1, 1).expand_as(actual)
    torch.testing.assert_close(actual.float(), expected, atol=1e-3, rtol=5e-3)
    return {
        "scope": "expanded Q/K/V kernel control; no model projection or cached-prefix gather",
        "qk_dim": 256,
        "v_dim": 128,
        "q_tokens": nq,
        "kv_tokens": n,
        "window_left": 512 if sliding else -1,
        "max_abs_error": (actual.float() - expected).abs().max().item(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--kernel-import-mode", choices=("normal",), required=True)
    parser.add_argument(
        "--report", required=True, help="JSON path, or - for stdout only"
    )
    args = parser.parse_args()
    report = {
        "kernel_import_mode": args.kernel_import_mode,
        "skipped_package_initializers": [],
        "checks": [],
        "end_to_end_ready": False,
        "scope": "Production recipe/arena/pool/router and registry kernels. Synthetic block tables and projected inputs; no model loading, scheduler lifecycle, distributed TP execution, performance, or GSM8K validation.",
    }

    def save():
        if args.report != "-":
            Path(args.report).write_text(json.dumps(report, indent=2) + "\n")

    def check(name, fn):
        try:
            details = fn()
            import torch

            if torch.cuda.is_initialized():
                torch.cuda.synchronize()
            entry = {"name": name, "status": "passed", "details": details}
        except Exception as exc:
            entry = {
                "name": name,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
        report["checks"].append(entry)
        print(json.dumps(entry), flush=True)
        save()
        if entry["status"] == "failed" and any(
            text in entry["error"].lower()
            for text in (
                "illegal memory access",
                "device-side assert",
                "unspecified launch failure",
            )
        ):
            raise SystemExit(
                "Fatal CUDA failure; stopping rather than reusing a poisoned context"
            )
        return entry["status"] == "passed"

    if not check(
        "normal_kernel_import",
        lambda: importlib.import_module("tokenspeed_kernel").__name__,
    ):
        return 1
    import torch

    torch.manual_seed(1234)
    torch.set_float32_matmul_precision("highest")
    if not check(
        "cuda",
        lambda: {
            "device": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "triton": importlib.metadata.version("tokenspeed-triton"),
        },
    ):
        return 1
    # Import the same assertions pytest runs; importing a test module does not
    # invoke its fixture or replace any production module.
    checks = importlib.import_module("test.runtime.cache.test_dots3_note_pool")
    runtimes = []

    def setup():
        runtime = checks.build_runtime(
            production_inputs(json.loads(args.model_config.read_text()))
        )
        runtimes.append(runtime)
        return {
            "parent_bytes": runtime.arena.plan.lcm_block_bytes,
            "arena_bytes": runtime.arena.plan.arena_bytes,
            "pool": type(runtime.pool).__name__,
            "input_source": "supplied configuration",
            "tp_scope": "TP8 local head geometry only; one GPU, no collectives",
        }

    if not check("production_recipe_arena_pool", setup):
        return 1
    runtime = runtimes[0]
    check("all_layer_binding_zero_copy", lambda: checks.check_binding(runtime))
    for count in (6, 513):
        check(
            f"pool_latent_scatter_gather/{count}",
            lambda n=count: checks.check_latent_scatter(runtime, n),
        )
    check("pool_index_planar_scatter", lambda: checks.check_index_scatter(runtime))
    router_checks = [
        (
            "pool_router_extend_boundary_writes",
            lambda: checks.check_extend_writes(runtime),
        ),
        *[
            (
                f"pool_router_swa_decode/{gid}",
                lambda g=gid: checks.check_swa_decode(runtime, g),
            )
            for gid in ("swa.0", "swa.1", "swa.2")
        ],
        ("pool_scatter_topk_dsa_graph", lambda: checks.check_index_to_dsa(runtime)),
        (
            "pool_router_graph_refresh_writes_padding",
            lambda: checks.check_decode_graph(runtime),
        ),
    ]
    report["blocked_checks"] = []
    if check(
        "normal_router_all_layer_binding", lambda: checks.check_router_binding(runtime)
    ):
        for name, fn in router_checks:
            check(name, fn)
    else:
        report["blocked_checks"] = [name for name, _ in router_checks]
    for cached in (False, True):
        for sliding in (False, True):
            check(
                f"expanded_prefill_control/{cached=}/{sliding=}",
                lambda c=cached, s=sliding: probe_prefill(cached=c, sliding=s),
            )
    save()
    print(
        json.dumps(
            {
                "end_to_end_ready": False,
                "passed": sum(c["status"] == "passed" for c in report["checks"]),
                "failed": sum(c["status"] == "failed" for c in report["checks"]),
                "blocked_checks": report["blocked_checks"],
            }
        ),
        flush=True,
    )
    return int(any(c["status"] != "passed" for c in report["checks"]))


if __name__ == "__main__":
    raise SystemExit(main())

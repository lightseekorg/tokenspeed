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

"""Check the FlashMLA sink path and the head padding helper on Hopper.

``gpu_kernel_check.py`` covers the Triton DSA kernel, which runs on any NVIDIA
GPU. It cannot cover ``dsa/cuda.py``: that module only defines its kernels when
``current_platform().is_hopper_plus`` is true, because FlashMLA is a Hopper
kernel. This script runs on an H20/H100/H800 and checks the two things that
were previously untested:

1. FlashMLA accepts ``attn_sink`` through the wrappers in ``dsa/cuda.py``.
2. The head padding helper produces a sink tensor whose padded entries are
   ``-inf``, so padded heads contribute nothing to the denominator.

Both are checked against a direct softmax reference rather than against
themselves.
"""

from __future__ import annotations

import argparse
import importlib
import sys
import types
from pathlib import Path

_FAILURES: list[str] = []


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _install_shadow_package() -> None:
    """Load only the op modules under test, skipping the heavy package init."""
    root = _repo_root()
    kernel_src = root / "tokenspeed-kernel" / "python" / "tokenspeed_kernel"
    package = types.ModuleType("tokenspeed_kernel")
    package.__path__ = [str(kernel_src)]
    package.__package__ = "tokenspeed_kernel"
    sys.modules["tokenspeed_kernel"] = package
    for suffix in ("ops", "ops.attention", "ops.attention.dsa", "ops.attention.dsv4"):
        name = f"tokenspeed_kernel.{suffix}"
        sub = types.ModuleType(name)
        sub.__path__ = [str(kernel_src.joinpath(*suffix.split(".")))]
        sub.__package__ = name
        sys.modules[name] = sub
    sys.path.insert(0, str(root / "tokenspeed-kernel" / "python"))
    sys.path.insert(0, str(root / "python"))


def _check(condition: bool, label: str, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}" + (f"  ({detail})" if detail else ""))
    if not condition:
        _FAILURES.append(label)


def check_platform() -> bool:
    """Confirm this is really a Hopper-class device before drawing conclusions."""
    import torch
    from tokenspeed_kernel.platform import current_platform

    platform = current_platform()
    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    print(f"device: {name} (sm_{major}{minor})")
    _check(
        bool(platform.is_hopper_plus),
        "device reports hopper_plus (FlashMLA path is defined)",
        f"is_hopper_plus={platform.is_hopper_plus}",
    )
    return bool(platform.is_hopper_plus)


def check_flashmla_import(dsa_cuda) -> bool:
    """The module must expose the FlashMLA symbols on this device."""
    names = ("flash_mla_sparse_fwd", "flash_mla_with_kvcache", "get_mla_metadata")
    present = [n for n in names if hasattr(dsa_cuda, n)]
    _check(
        len(present) == len(names),
        "FlashMLA symbols are importable through dsa/cuda.py",
        f"found={present}",
    )
    return len(present) == len(names)


def check_sink_padding(dsa_cuda) -> None:
    """Padded heads must get a -inf sink so they contribute nothing."""
    import torch

    helper = getattr(dsa_cuda, "_pad_sink_for_heads", None)
    if helper is None:
        _check(False, "_pad_sink_for_heads is exposed for checking")
        return

    actual, padded = 8, 64
    sink = torch.randn(actual, device="cuda", dtype=torch.float32)
    out = helper(sink, padded, actual)
    _check(
        out.shape[0] == padded,
        "padded sink has the kernel head count",
        f"shape={tuple(out.shape)}",
    )
    _check(
        torch.equal(out[:actual], sink),
        "real heads keep their sink values unchanged",
    )
    tail = out[actual:]
    _check(
        torch.isneginf(tail).all().item(),
        "padded heads get -inf sinks",
        f"non_neg_inf={int((~torch.isneginf(tail)).sum().item())}",
    )


def check_flashmla_sink_math(dsa_cuda, *, tokens: int, heads: int) -> None:
    """Run FlashMLA sparse prefill with a sink and check it against a reference.

    The kernel is reached through ``dsa/cuda.py`` exactly as the runtime calls
    it, in the absorbed-MLA layout the runtime feeds it: each query and cache
    row is ``kv_lora_rank + rope`` = 576 wide and the value is the first 512
    lanes. The output is compared with a direct FP32 softmax that adds
    ``exp(sink)`` to the denominator, over random top-k slots with ragged
    lengths, at a head count that forces the -inf padding lanes.
    """
    import torch

    device = "cuda"
    torch.manual_seed(0)
    kv_lora_rank, rope_dim, qk_nope = 512, 64, 192
    d_qk = kv_lora_rank + rope_dim
    page_size, topk, pool = 64, 2048, 8192
    softmax_scale = (qk_nope + rope_dim) ** -0.5

    q = torch.randn(tokens, heads, d_qk, device=device, dtype=torch.bfloat16)
    kv = torch.randn(pool, d_qk, device=device, dtype=torch.bfloat16)
    slots = torch.stack(
        [torch.randperm(pool, device=device)[:topk] for _ in range(tokens)]
    ).to(torch.int32)
    lens = torch.randint(
        topk // 4, topk + 1, (tokens,), device=device, dtype=torch.int32
    )
    lens[0] = topk

    def run(attn_sink, return_lse=False):
        return dsa_cuda.flashmla_dsa_prefill(
            q=q,
            kv_cache=kv,
            sparse_kv_cache=None,
            topk_slots=slots,
            topk_lens=lens,
            max_seqlen_k=topk,
            qk_nope_head_dim=qk_nope,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=rope_dim,
            softmax_scale=softmax_scale,
            page_size=page_size,
            q_len_per_req=1,
            kv_seq_lens=None,
            logit_cap=0.0,
            k_scale=1.0,
            return_lse=return_lse,
            out=None,
            enable_pdl=False,
            attn_sink=attn_sink,
        )

    def reference(attn_sink):
        outs, lses = [], []
        kvf = kv.float()
        for t in range(tokens):
            sel = kvf[slots[t, : lens[t]].long()]
            s = (q[t].float() @ sel.T) * softmax_scale
            m = s.max(dim=-1, keepdim=True).values
            if attn_sink is not None:
                m = torch.maximum(m, attn_sink.view(-1, 1))
            e = torch.exp(s - m)
            denom = e.sum(-1, keepdim=True)
            lses.append((torch.log(denom) + m).squeeze(-1))
            if attn_sink is not None:
                denom = denom + torch.exp(attn_sink.view(-1, 1) - m)
            outs.append((e / denom) @ sel[:, :kv_lora_rank])
        return torch.stack(outs), torch.stack(lses)

    def bf16_tol(ref):
        return 2.0**-7 * ref.abs().amax().item() + 2e-3

    base, base_lse = run(None, return_lse=True)
    ref0, ref0_lse = reference(None)
    err = (base.float() - ref0).abs().max().item()
    _check(
        err <= bf16_tol(ref0),
        "FlashMLA without sink matches the reference",
        f"max|diff|={err:.3e} tol={bf16_tol(ref0):.3e}",
    )
    lerr = (base_lse.float() - ref0_lse).abs().max().item()
    _check(lerr <= 1e-2, "FlashMLA LSE matches the reference", f"max|diff|={lerr:.3e}")

    sink = torch.randn(heads, device=device, dtype=torch.float32) * 2 + 3
    got = run(sink)
    ref1, _ = reference(sink)
    err = (got.float() - ref1).abs().max().item()
    _check(
        err <= bf16_tol(ref1),
        "FlashMLA with learnable sink matches the reference",
        f"max|diff|={err:.3e} tol={bf16_tol(ref1):.3e}",
    )
    delta = (got.float() - base.float()).abs().max().item()
    _check(
        delta > 10 * bf16_tol(ref1),
        "the sink actually changes the output",
        f"max|diff vs no-sink|={delta:.3e}",
    )
    cos = torch.nn.functional.cosine_similarity(
        got.float().flatten(), ref1.flatten(), dim=0
    ).item()
    _check(cos > 0.9999, "sink output cosine similarity", f"cos={cos:.6f}")

    big = torch.full((heads,), 200.0, device=device, dtype=torch.float32)
    dom = run(big)
    _check(
        bool(torch.isfinite(dom).all()) and dom.float().abs().max().item() <= 1e-2,
        "a dominant sink absorbs the attention mass and stays finite",
        f"max|out|={dom.float().abs().max().item():.3e}",
    )

    neg = torch.full((heads,), float("-inf"), device=device, dtype=torch.float32)
    err = (run(neg).float() - base.float()).abs().max().item()
    _check(
        err == 0.0,
        "a -inf sink reproduces prefill without a sink",
        f"max|diff|={err:.3e}",
    )


def bench_flashmla(dsa_cuda) -> None:
    """Time FlashMLA sparse prefill with and without a sink at HY V4 shapes."""
    import torch

    device = "cuda"
    kv_lora_rank, rope_dim, qk_nope = 512, 64, 192
    d_qk = kv_lora_rank + rope_dim
    topk, pool = 2048, 65536
    for heads in (8, 16, 64):
        for tokens in (512, 4096):
            q = torch.randn(tokens, heads, d_qk, device=device, dtype=torch.bfloat16)
            kv = torch.randn(pool, d_qk, device=device, dtype=torch.bfloat16)
            slots = torch.randint(
                0, pool, (tokens, topk), device=device, dtype=torch.int32
            )
            lens = torch.full((tokens,), topk, device=device, dtype=torch.int32)
            sink = torch.randn(heads, device=device, dtype=torch.float32)

            def call(s):
                return dsa_cuda.flashmla_dsa_prefill(
                    q=q,
                    kv_cache=kv,
                    sparse_kv_cache=None,
                    topk_slots=slots,
                    topk_lens=lens,
                    max_seqlen_k=topk,
                    qk_nope_head_dim=qk_nope,
                    kv_lora_rank=kv_lora_rank,
                    qk_rope_head_dim=rope_dim,
                    softmax_scale=0.0625,
                    page_size=64,
                    q_len_per_req=1,
                    kv_seq_lens=None,
                    logit_cap=0.0,
                    k_scale=1.0,
                    return_lse=False,
                    out=None,
                    enable_pdl=False,
                    attn_sink=s,
                )

            res = {}
            for label, s in (("no_sink", None), ("sink", sink)):
                for _ in range(3):
                    call(s)
                torch.cuda.synchronize()
                start, end = torch.cuda.Event(True), torch.cuda.Event(True)
                start.record()
                for _ in range(20):
                    call(s)
                end.record()
                torch.cuda.synchronize()
                res[label] = start.elapsed_time(end) / 20 * 1000
            padded = max(64, (heads + 63) // 64 * 64)
            flops = 2 * tokens * padded * topk * (d_qk + kv_lora_rank)
            print(
                f"[BENCH] flashmla prefill heads={heads:>2} (kernel {padded}) tokens={tokens:>4} "
                f"topk={topk}: no_sink {res['no_sink']:8.1f} us | sink {res['sink']:8.1f} us "
                f"| overhead {100 * (res['sink'] / res['no_sink'] - 1):+5.1f}% "
                f"| {flops / res['sink'] / 1e6:6.1f} TFLOPS(padded)"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=16, help="query tokens")
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--bench", action="store_true")
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device available")
    print(f"torch {torch.__version__}")

    _install_shadow_package()

    print("\n== platform ==")
    try:
        hopper = check_platform()
    except Exception as error:
        _check(False, "platform probe", str(error)[:200])
        print("\nVERDICT: FAIL")
        return 1
    if not hopper:
        print("\nVERDICT: SKIP (not a Hopper-class device; FlashMLA path undefined)")
        return 2

    print("\n== FlashMLA sink plumbing ==")
    try:
        dsa_cuda = importlib.import_module("tokenspeed_kernel.ops.attention.dsa.cuda")
    except Exception as error:
        _check(False, "dsa/cuda.py import", str(error)[:300])
        print("\nVERDICT: FAIL")
        return 1

    check_sink_padding(dsa_cuda)
    if check_flashmla_import(dsa_cuda):
        print("\n== FlashMLA sparse prefill with a sink ==")
        for heads in (args.heads, 64):
            print(f"-- heads={heads}")
            check_flashmla_sink_math(dsa_cuda, tokens=args.tokens, heads=heads)
        if args.bench:
            print("\n== FlashMLA bench ==")
            bench_flashmla(dsa_cuda)

    print()
    if _FAILURES:
        print(f"VERDICT: FAIL ({len(_FAILURES)} checks): {_FAILURES}")
        return 1
    print("VERDICT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())

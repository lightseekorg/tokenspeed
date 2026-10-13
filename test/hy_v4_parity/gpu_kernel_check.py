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

"""Run the HY V4 iHC and sink kernels on a real GPU and check them numerically.

The point of this script is narrow and deliberate: the iHC Triton kernels and the
learnable-sink path were written but never executed on hardware. This executes
them and compares against the same CPU reference formulas the unit tests use, so
a kernel that mis-compiles, mis-indexes, or silently drops the sink is caught
before any model-level work.

It loads ``tokenspeed_kernel.ops.residual`` and ``...dsa`` through a minimal
shadow package rather than importing ``tokenspeed_kernel`` itself, because the
package ``__init__`` pulls in the full runtime (flashinfer, cutlass, cu13
libraries). The shadow package exposes only what these two op modules import.

Usage::

    python test/hy_v4_parity/gpu_kernel_check.py

Exits non-zero on any numerical failure, on a kernel that fails to compile, or
on a device too old to run the Triton path.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import sys
import types
from pathlib import Path

_FAILURES: list[str] = []


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _install_shadow_package() -> None:
    """Expose ``tokenspeed_kernel`` submodules without its runtime ``__init__``.

    Only the support modules and the two op modules under test are made
    importable; anything else would drag the full runtime in (flashinfer,
    cutlass, cu13 libraries). Parent packages are registered as empty modules
    with a ``__path__`` so their ``__init__.py`` never executes, while the real
    module files still load from this working copy.
    """
    root = _repo_root()
    kernel_src = root / "tokenspeed-kernel" / "python" / "tokenspeed_kernel"
    if not kernel_src.is_dir():
        raise SystemExit(f"kernel sources not found at {kernel_src}")

    package = types.ModuleType("tokenspeed_kernel")
    package.__path__ = [str(kernel_src)]
    package.__package__ = "tokenspeed_kernel"
    sys.modules["tokenspeed_kernel"] = package

    # Empty stand-ins for the intermediate packages whose __init__ pulls the
    # heavy runtime in; a __path__ keeps their submodules importable.
    for suffix in ("ops", "ops.attention", "ops.attention.dsa"):
        name = f"tokenspeed_kernel.{suffix}"
        sub = types.ModuleType(name)
        sub.__path__ = [str(kernel_src.joinpath(*suffix.split(".")))]
        sub.__package__ = name
        sys.modules[name] = sub

    sys.path.insert(0, str(root / "tokenspeed-kernel" / "python"))
    sys.path.insert(0, str(root / "python"))

    _stub_unrelated_residual_backends()


# Residual backends that are unrelated to HY V4 and pull in third-party wheels.
# ``deep_gemm`` provides the Hopper mHC kernels (GLM-5.3, not iHC) and is built
# against CUDA 13; ``cuda``/``cute_fused`` are Blackwell-only; ``gluon`` is AMD.
# They are registered by side-effect import in ``ops/residual/__init__.py``, so
# without stubbing them the iHC ops cannot be imported on a CUDA 12 node at all.
_UNRELATED_RESIDUAL_BACKENDS = ("deep_gemm", "cuda", "cute_fused", "gluon")


def _stub_unrelated_residual_backends() -> None:
    """Replace unrelated residual backends with empty modules, loudly.

    Only the ``torch`` and ``triton`` backends implement iHC, so stubbing the
    others does not change which kernel the iHC dispatchers select. Each stub is
    reported so a reader can see exactly what was not loaded.
    """
    for backend in _UNRELATED_RESIDUAL_BACKENDS:
        name = f"tokenspeed_kernel.ops.residual.{backend}"
        sys.modules[name] = types.ModuleType(name)
        print(f"[shadow] stubbed unrelated residual backend: {backend}")


def _load(module_name: str):
    """Import a module from the shadow package, reporting import failures."""
    importlib.invalidate_caches()
    return importlib.import_module(module_name)


def _check(condition: bool, label: str, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}" + (f"  ({detail})" if detail else ""))
    if not condition:
        _FAILURES.append(label)


def _bf16_tolerance(reference, ulps: float = 1.0) -> float:
    """Absolute tolerance for a BF16 output compared against an FP32 reference.

    A BF16 result can be off by up to half a unit in the last place of its own
    magnitude, and that unit grows with the value (8 mantissa bits, so one ulp is
    ``2**-7`` relative). A fixed absolute bound therefore fails spuriously once
    outputs get large -- e.g. the iHC head sums ``hc_mult`` channels, so its peak
    grows with ``hc_mult``. This scales the bound with the reference's peak.
    """
    peak = float(reference.abs().max())
    return max(ulps * peak * 2.0**-7, 1e-3)


def _reference_ihc_pre(x, fn, scale, base, hc, magnitude, hc_eps, rms_eps):
    """The reference ``HYV4HCPreLayer.forward``, transcribed line for line."""
    import torch

    shape = x.size()
    flat = x.flatten(1).float()
    rsqrt = torch.rsqrt(flat.square().mean(-1, keepdim=True) + rms_eps)
    mixes = torch.nn.functional.linear(flat, fn) * rsqrt
    pre = torch.sigmoid(mixes[..., :hc] * scale[0] + base[:hc]) + hc_eps
    post = (
        magnitude * torch.sigmoid(mixes[..., hc : 2 * hc] * scale[1] + base[hc:])
        + hc_eps
    )
    return torch.sum(pre.unsqueeze(-1) * x.reshape(shape), dim=1).to(x.dtype), post


def _reference_ihc_post(block_output, residual, post_gates):
    return (post_gates.unsqueeze(-1) * block_output.unsqueeze(1) + residual).to(
        block_output.dtype
    )


def _reference_ihc_head(x, fn, scale, base, rms_eps, hc_eps):
    import torch

    shape = x.size()
    flat = x.flatten(1).float()
    rsqrt = torch.rsqrt(flat.square().mean(-1, keepdim=True) + rms_eps)
    mixes = torch.nn.functional.linear(flat, fn) * rsqrt
    pre = torch.sigmoid(mixes * scale + base) + hc_eps
    return torch.sum(pre.unsqueeze(-1) * x.reshape(shape), dim=1)


def check_ihc(residual_ops, *, num_tokens: int, hc: int, hidden: int, dtype) -> None:
    """Compare the fused Triton iHC boundaries against the CPU reference."""
    import torch

    device = "cuda"
    torch.manual_seed(0)
    x = torch.randn(num_tokens, hc, hidden, device=device, dtype=torch.float32)
    x = x.to(dtype)
    fn = torch.randn(2 * hc, hc * hidden, device=device, dtype=torch.float32) * 0.02
    scale = torch.rand(2, device=device, dtype=torch.float32) + 0.5
    base = torch.randn(2 * hc, device=device, dtype=torch.float32) * 0.1
    # Deliberately large epsilons. With the released values (hc_eps~1e-6,
    # rms_eps~1e-6) a kernel that drops an epsilon differs by less than bf16
    # rounding and would pass; these values make every term observable while
    # leaving the formula unchanged.
    magnitude, hc_eps, rms_eps = 1.7, 0.05, 0.5

    got_out, got_post = residual_ops.ihc_pre(
        x,
        fn,
        scale,
        base,
        magnitude,
        rms_eps,
        hc_eps,
        norm_weight=None,
        norm_eps=None,
    )
    ref_out, ref_post = _reference_ihc_pre(
        x.cpu(), fn.cpu(), scale.cpu(), base.cpu(), hc, magnitude, hc_eps, rms_eps
    )
    ref_out = ref_out.to(device)

    out_err = (got_out.float() - ref_out.float()).abs().max().item()
    post_err = (got_post.float() - ref_post.to(device).float()).abs().max().item()
    _check(
        out_err <= _bf16_tolerance(ref_out),
        "ihc_pre reduced channels match the reference",
        f"max|diff|={out_err:.3e} tol={_bf16_tolerance(ref_out):.3e}",
    )
    # Gates are FP32 end to end, so they are held to FP32 tolerance; the loose
    # BF16 bound used for activations would hide a missing epsilon.
    _check(
        post_err <= 1e-4,
        "ihc_pre post gates match the reference (fp32 tolerance)",
        f"max|diff|={post_err:.3e}",
    )

    # The fused RMSNorm must match normalizing the (bf16-rounded) reduction.
    weight = torch.rand(hidden, device=device, dtype=torch.float32) + 0.5
    norm_eps = 1e-6
    got_norm, _ = residual_ops.ihc_pre(
        x,
        fn,
        scale,
        base,
        magnitude,
        rms_eps,
        hc_eps,
        norm_weight=weight,
        norm_eps=norm_eps,
    )
    rounded = ref_out.float()
    ref_norm = (
        rounded
        * torch.rsqrt(rounded.pow(2).mean(-1, keepdim=True) + norm_eps)
        * weight.float()
    )
    norm_err = (got_norm.float() - ref_norm.to(device)).abs().max().item()
    _check(
        norm_err <= _bf16_tolerance(ref_norm, ulps=2.0),
        "ihc_pre fused RMSNorm matches the reference",
        f"max|diff|={norm_err:.3e} tol={_bf16_tolerance(ref_norm, ulps=2.0):.3e}",
    )

    # Post boundary: scatter gate * block_output back onto the residual channels.
    block = torch.randn(num_tokens, hidden, device=device, dtype=dtype)
    post_gates = got_post
    got = residual_ops.ihc_post(block, x, post_gates)
    _check(
        tuple(got.shape) == tuple(x.shape),
        "ihc_post returns the full residual channel block",
        f"shape={tuple(got.shape)}",
    )
    ref = _reference_ihc_post(block.cpu(), x.cpu(), post_gates.cpu().float())
    post_apply_err = (got.float() - ref.to(device)).abs().max().item()
    _check(
        post_apply_err <= _bf16_tolerance(ref),
        "ihc_post gate/residual matches the reference",
        f"max|diff|={post_apply_err:.3e} tol={_bf16_tolerance(ref):.3e}",
    )

    # Head: one gate set per channel, no post gates. Same observable epsilons.
    head_fn = torch.randn(hc, hc * hidden, device=device, dtype=torch.float32) * 0.02
    head_scale = torch.rand(1, device=device, dtype=torch.float32) + 0.5
    head_base = torch.randn(hc, device=device, dtype=torch.float32) * 0.1
    got_head = residual_ops.ihc_head(x, head_fn, head_scale, head_base, rms_eps, hc_eps)
    ref_head = _reference_ihc_head(
        x.cpu(), head_fn.cpu(), head_scale.cpu(), head_base.cpu(), rms_eps, hc_eps
    )
    head_err = (got_head.float() - ref_head.to(device)).abs().max().item()
    _check(
        head_err <= _bf16_tolerance(ref_head),
        "ihc_head matches the reference",
        f"max|diff|={head_err:.3e} tol={_bf16_tolerance(ref_head):.3e}",
    )


def check_ihc_rejects_oversized(residual_ops, *, dtype) -> None:
    """The fused kernels cap hc_mult at 8 and must refuse larger values loudly.

    A kernel that silently mis-computes for an unsupported channel count would be
    far worse than one that raises, so the refusal is part of the contract.
    """
    import torch

    hc, hidden = 16, 128
    x = torch.randn(4, hc, hidden, device="cuda").to(dtype)
    fn = torch.randn(2 * hc, hc * hidden, device="cuda") * 0.02
    scale = torch.rand(2, device="cuda") + 0.5
    base = torch.randn(2 * hc, device="cuda") * 0.1
    try:
        residual_ops.ihc_pre(
            x, fn, scale, base, 1.7, 0.5, 0.05, norm_weight=None, norm_eps=None
        )
    except Exception as error:
        _check(
            "8" in str(error),
            "ihc_pre refuses hc_mult > 8 with a clear error",
            str(error)[:80],
        )
        return
    _check(False, "ihc_pre refuses hc_mult > 8 with a clear error", "no error raised")


def check_ihc_shapes(residual_ops, *, dtype) -> None:
    """Odd token counts must not change the result (masking correctness)."""
    import torch

    for num_tokens in (1, 7, 33, 129):
        torch.manual_seed(num_tokens)
        hc, hidden = 4, 256
        x = torch.randn(num_tokens, hc, hidden, device="cuda", dtype=torch.float32).to(
            dtype
        )
        fn = torch.randn(2 * hc, hc * hidden, device="cuda") * 0.02
        scale = torch.rand(2, device="cuda") + 0.5
        base = torch.randn(2 * hc, device="cuda") * 0.1
        got, got_post = residual_ops.ihc_pre(
            x, fn, scale, base, 1.7, 0.5, 0.05, norm_weight=None, norm_eps=None
        )
        ref, ref_post = _reference_ihc_pre(
            x.cpu(), fn.cpu(), scale.cpu(), base.cpu(), hc, 1.7, 0.05, 0.5
        )
        err = (got.float() - ref.to("cuda").float()).abs().max().item()
        post_err = (got_post.float() - ref_post.to("cuda").float()).abs().max().item()
        _check(
            err <= _bf16_tolerance(ref) and post_err <= 1e-4,
            f"ihc_pre matches at num_tokens={num_tokens}",
            f"out max|diff|={err:.3e} post max|diff|={post_err:.3e}",
        )


def check_sink(dsa_triton, *, num_tokens: int, heads: int, dtype) -> None:
    """Execute the modified DSA dense-kv kernel with and without a sink.

    This is the load-bearing check for the learnable-sink change: it runs the
    real kernel that carries the ``HAS_SINK`` branch and verifies against a
    hand-written softmax that

    * a sink changes the output by exactly the extra denominator term, and
    * a ``-inf`` sink reproduces plain attention, which is what the padding path
      relies on for head counts that exceed the real head count.
    """
    import torch

    device = "cuda"
    torch.manual_seed(0)
    kv_lora_rank, qk_rope_head_dim = 128, 64
    kv_dim = kv_lora_rank + qk_rope_head_dim
    topk = 8
    softmax_scale = (kv_lora_rank + qk_rope_head_dim) ** -0.5

    q = torch.randn(num_tokens, heads, kv_dim, device=device, dtype=dtype)
    # The kernel gathers through topk_indices, so the KV pool must be at least
    # as long as the number of selected slots or the gather reads out of bounds.
    num_slots = max(num_tokens, topk)
    kv = torch.randn(num_slots, kv_dim, device=device, dtype=dtype)
    topk_indices = torch.arange(topk, device=device, dtype=torch.int32).repeat(
        num_tokens, 1
    )
    topk_lens = torch.full((num_tokens,), topk, device=device, dtype=torch.int32)

    def run(sink):
        out, lse = dsa_triton._run_dense_kv(
            q,
            kv,
            topk_indices,
            topk_lens,
            softmax_scale=softmax_scale,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            return_lse=True,
            sink=sink,
        )
        return out, lse

    baseline, lse_base = run(None)
    _check(
        torch.isfinite(baseline).all().item(),
        "dsa dense-kv kernel runs without a sink",
        f"max={baseline.abs().max().item():.3e}",
    )

    # Reference: softmax over the *selected* top-k slots only, matching the
    # kernel's gather through topk_indices. All tokens select the same slots
    # here, so one gathered set covers every token.
    selected = topk_indices[0].long()
    qn = q[..., :kv_lora_rank].float()
    qr = q[..., kv_lora_rank:].float()
    kn_all = kv[:, :kv_lora_rank].float()
    kr_all = kv[:, kv_lora_rank:].float()
    kn = kn_all[selected]
    kr = kr_all[selected]
    scores = torch.einsum("thd,kd->thk", qn, kn) + torch.einsum("thd,kd->thk", qr, kr)
    scores = scores * softmax_scale
    ref = torch.einsum("thk,kd->thd", torch.softmax(scores, dim=-1), kn)
    ref_err = (baseline.float() - ref).abs().max().item()
    _check(
        ref_err <= 3e-2,
        "dsa dense-kv output matches a direct softmax over the selected slots",
        f"max|diff|={ref_err:.3e}",
    )

    # With a sink, the kernel must match adding exp(sink - max) to the denominator.
    sink = torch.randn(heads, device=device, dtype=torch.float32)
    with_sink, lse_sink = run(sink)
    m = scores.max(dim=-1, keepdim=True).values
    denom = torch.exp(scores - m).sum(-1) + torch.exp(sink[None, :, None] - m).squeeze(
        -1
    )
    ref_sink = torch.einsum(
        "thk,kd->thd", torch.exp(scores - m) / denom.unsqueeze(-1), kn
    )
    sink_err = (with_sink.float() - ref_sink).abs().max().item()
    _check(
        sink_err <= 3e-2,
        "dsa dense-kv applies the learnable sink as an extra denominator term",
        f"max|diff|={sink_err:.3e}",
    )
    _check(
        not torch.allclose(with_sink.float(), baseline.float(), atol=1e-3),
        "the sink actually changes the output (the branch is live)",
        f"max|diff|={(with_sink.float() - baseline.float()).abs().max().item():.3e}",
    )

    # LSE must include the sink term, or downstream merging would be wrong.
    ref_lse = m.squeeze(-1) + torch.log(denom)
    _check(
        tuple(lse_sink.shape) == tuple(ref_lse.shape),
        "dsa LSE shape matches the reference",
        f"{tuple(lse_sink.shape)} vs {tuple(ref_lse.shape)}",
    )
    lse_err = (lse_sink.float() - ref_lse).abs().max().item()
    _check(
        lse_err <= 1e-2,
        "dsa dense-kv LSE folds in the sink",
        f"max|diff|={lse_err:.3e}",
    )

    # A -inf sink must reproduce the no-sink path exactly (padding relies on it).
    neg_inf = torch.full((heads,), float("-inf"), device=device, dtype=torch.float32)
    padded, _ = run(neg_inf)
    pad_err = (padded.float() - baseline.float()).abs().max().item()
    _check(
        pad_err == 0.0 or pad_err <= 1e-5,
        "a -inf sink reproduces attention without a sink",
        f"max|diff|={pad_err:.3e}",
    )

    # A sink far above every score: exp(sink - max) overflows unless the kernel
    # folds the sink into the running max first. The output must stay finite
    # and tend to zero (the sink absorbs nearly all the probability mass).
    huge = torch.full((heads,), 200.0, device=device, dtype=torch.float32)
    dominated, lse_dom = run(huge)
    _check(
        torch.isfinite(dominated).all().item() and torch.isfinite(lse_dom).all().item(),
        "a dominant sink stays finite (sink is folded into the max)",
        f"finite_out={torch.isfinite(dominated).all().item()} "
        f"finite_lse={torch.isfinite(lse_dom).all().item()}",
    )
    _check(
        dominated.float().abs().max().item() <= 1e-3,
        "a dominant sink absorbs the attention mass",
        f"max|out|={dominated.float().abs().max().item():.3e}",
    )
    _check(
        (lse_dom.float() - 200.0).abs().max().item() <= 1e-3,
        "a dominant sink sets the LSE to the sink value",
        f"max|lse-200|={(lse_dom.float() - 200.0).abs().max().item():.3e}",
    )


def check_sink_trait_filtering(dsa_triton) -> None:
    """The kernels that cannot honour a sink must be marked as such.

    ``sinks`` is declared on the registration in ``dsa/cuda.py`` (FlashMLA)
    and ``dsa/flashinfer.py``; here we confirm this module's kernels carry the
    ``HAS_SINK`` constexpr so the sink cannot be silently dropped at launch.
    """
    source = Path(dsa_triton.__file__).read_text()
    _check("HAS_SINK" in source, "dsa triton kernels take a HAS_SINK constexpr")
    _check(
        source.count("HAS_SINK: tl.constexpr") >= 2,
        "both packed and dense DSA kernels handle the sink",
        f"count={source.count('HAS_SINK: tl.constexpr')}",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=("bfloat16",),
        help="Activation dtype. HY V4 runs BF16 and the iHC kernels register "
        "BF16 signatures only, so this is the only supported value.",
    )
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument(
        "--hc",
        type=int,
        default=4,
        help="iHC channel count. HY V4 uses 4; the fused kernels support up to 8 "
        "and reject larger values by design.",
    )
    parser.add_argument("--hidden", type=int, default=1024)
    args = parser.parse_args()
    if not 1 <= args.hc <= 8:
        parser.error(
            f"--hc must be in 1..8 (the fused iHC kernels cap at 8), got {args.hc}"
        )

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device available")
    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    print(f"device: {name} (sm_{major}{minor}), torch {torch.__version__}")
    dtype = torch.bfloat16

    _install_shadow_package()

    print("\n== iHC kernels ==")
    try:
        residual_ops = _load("tokenspeed_kernel.ops.residual")
    except Exception as error:
        _check(False, "residual ops import", str(error)[:200])
        print("\nVERDICT: FAIL")
        return 1
    check_ihc(
        residual_ops,
        num_tokens=args.tokens,
        hc=args.hc,
        hidden=args.hidden,
        dtype=dtype,
    )
    check_ihc_shapes(residual_ops, dtype=dtype)
    check_ihc_rejects_oversized(residual_ops, dtype=dtype)

    print("\n== learnable sink ==")
    try:
        dsa_triton = _load("tokenspeed_kernel.ops.attention.dsa.triton")
    except Exception as error:
        _check(False, "dsa triton kernel import", str(error)[:300])
        print("\nVERDICT: FAIL")
        return 1
    check_sink(dsa_triton, num_tokens=args.tokens, heads=args.hc, dtype=dtype)
    check_sink_trait_filtering(dsa_triton)

    print()
    if _FAILURES:
        print(f"VERDICT: FAIL ({len(_FAILURES)} checks): {_FAILURES}")
        return 1
    print("VERDICT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())

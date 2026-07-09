"""End-to-end correctness test for the standalone bf16 Gluon MoE.

Reference shape: DeepSeek-V3 TP=8 MoE (E=256, D=7168, I=256, topk=8).
Run:

    PYTHONPATH=<triton repo>/python:<gluon-kernels>/kernels/cdna4/moe \
    HIP_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/triton-bf16-moe \
    python -m gluon_bf16_moe.tests.test_bf16_moe

Also runnable directly as a script (adds the parent dir to sys.path).
"""

from __future__ import annotations

import torch

try:
    from gluon_bf16_moe.moe import gluon_bf16_moe
    from gluon_bf16_moe.tests.torch_reference import routing_softmax_topk, torch_moe_ref
except ModuleNotFoundError:  # allow direct `python test_bf16_moe.py`
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from gluon_bf16_moe.moe import gluon_bf16_moe
    from gluon_bf16_moe.tests.torch_reference import routing_softmax_topk, torch_moe_ref


# DeepSeek-V3 TP=8 MoE expert GEMM shape.
E = 256
D = 7168
I_R = 256
TOPK = 8
REL_TOL = 2e-2


def _build(num_tokens, device, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    hidden = torch.randn(
        num_tokens, D, dtype=torch.bfloat16, device=device, generator=g
    )
    # Scale weights down so the fp32 reference stays in a comparable range.
    w1 = (
        torch.randn(E, 2 * I_R, D, dtype=torch.bfloat16, device=device, generator=g)
        * 0.05
    )
    w2 = torch.randn(E, D, I_R, dtype=torch.bfloat16, device=device, generator=g) * 0.05
    logits = torch.randn(num_tokens, E, dtype=torch.float32, device=device, generator=g)
    topk_ids, topk_weights = routing_softmax_topk(logits, TOPK)
    return hidden, w1, w2, topk_ids, topk_weights


def run_case(num_tokens, device):
    hidden, w1, w2, topk_ids, topk_weights = _build(num_tokens, device)
    out = gluon_bf16_moe(hidden, w1, w2, topk_ids, topk_weights)
    torch.cuda.synchronize()
    ref = torch_moe_ref(hidden, w1, w2, topk_ids, topk_weights)

    diff = (out.float() - ref).abs()
    peak = ref.abs().max().item()
    max_abs = diff.max().item()
    ok = max_abs <= REL_TOL * peak + 1e-2
    print(
        f"  M={num_tokens:>5}  max_abs_diff={max_abs:.4f}  peak={peak:.3f}  "
        f"{'PASS' if ok else 'FAIL'}"
    )
    return ok


def run_splitk_consistency(num_tokens, device):
    """The decode split-K stage-1 path must match the single-launch path."""
    hidden, w1, w2, topk_ids, topk_weights = _build(num_tokens, device)
    a = gluon_bf16_moe(hidden, w1, w2, topk_ids, topk_weights, split_k=1)
    b = gluon_bf16_moe(hidden, w1, w2, topk_ids, topk_weights, split_k=8)
    torch.cuda.synchronize()
    diff = (a.float() - b.float()).abs()
    peak = a.float().abs().max().item()
    max_abs = diff.max().item()
    ok = max_abs <= 1e-2 * peak + 1e-2
    print(
        f"  M={num_tokens:>5}  split_k=1 vs 8  max_abs_diff={max_abs:.4f}  "
        f"{'PASS' if ok else 'FAIL'}"
    )
    return ok


def main():
    assert torch.cuda.is_available(), "needs a gfx950 GPU"
    device = "cuda"
    print(f"bf16 Gluon MoE  DSv3 TP=8: E={E} D={D} I={I_R} topk={TOPK}\n")
    print("[1] end-to-end vs fp32 torch reference (auto split-K):")
    results = [run_case(m, device) for m in (1, 8, 64, 256)]
    print("\n[2] decode split-K stage-1 consistency (split_k=1 vs split_k=8):")
    results += [run_splitk_consistency(m, device) for m in (1, 4, 16)]
    passed = sum(results)
    total = len(results)
    print(f"\nTotal: {passed} passed, {total - passed} failed out of {total}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())

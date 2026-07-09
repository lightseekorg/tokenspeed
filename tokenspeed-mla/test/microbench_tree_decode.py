# Copyright (c) 2026 LightSeek Foundation

"""Manual GB200 correctness/perf harness for FP8 MLA tree-mask decode.

This is intentionally not a default pytest: it compiles and runs the Blackwell
CuTe MLA kernel. It validates the optional custom_mask path against an
independent absorbed-MLA PyTorch reference and includes non-tile-aligned K plus
batched offsets.

Example:
    python tokenspeed-mla/test/microbench_tree_decode.py --iters 50
"""

from __future__ import annotations

import argparse
import math

import torch
from tokenspeed_mla import tokenspeed_mla_decode


DEV = "cuda"
FP8 = torch.float8_e4m3fn
KV_LORA = 512
ROPE = 64
D_QK = KV_LORA + ROPE
H = 128
PAGE = 64


def build_inputs(batch: int, q_len: int, seq_k: int, seed: int):
    generator = torch.Generator(device=DEV).manual_seed(seed)
    query = (
        torch.randn(batch, q_len, H, D_QK, device=DEV, generator=generator) * 0.3
    ).to(FP8)
    pages_per_request = (seq_k + PAGE - 1) // PAGE
    page_count = pages_per_request * batch
    kv_cache = (
        torch.randn(page_count, PAGE, D_QK, device=DEV, generator=generator) * 0.3
    ).to(FP8)
    block_tables = torch.arange(page_count, device=DEV, dtype=torch.int32).view(
        batch, pages_per_request
    )
    seq_lens = torch.full((batch,), seq_k, device=DEV, dtype=torch.int32)
    workspace = torch.empty(1 << 29, dtype=torch.int8, device=DEV)
    return query, kv_cache, block_tables, seq_lens, workspace


def chain_ancestor(batch: int, q_len: int) -> torch.Tensor:
    ancestor = torch.tril(torch.ones(q_len, q_len, dtype=torch.bool))
    return ancestor.unsqueeze(0).expand(batch, q_len, q_len).contiguous()


def random_tree_ancestor(batch: int, q_len: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    ancestor = torch.zeros(batch, q_len, q_len, dtype=torch.bool)
    for b in range(batch):
        parent = [-1]
        for i in range(1, q_len):
            parent.append(int(torch.randint(0, i, (1,), generator=generator)))
        for qi in range(q_len):
            j = qi
            while j != -1:
                ancestor[b, qi, j] = True
                j = parent[j]
    return ancestor


def build_custom_mask(ancestor: torch.Tensor, seq_lens: torch.Tensor):
    batch, q_len, _ = ancestor.shape
    offsets = torch.empty(batch, dtype=torch.int32, device=DEV)
    parts = []
    offset = 0
    for b in range(batch):
        K = int(seq_lens[b])
        hist = K - q_len
        mask = torch.zeros(q_len, K, dtype=torch.bool, device=DEV)
        mask[:, :hist] = True
        mask[:, hist:K] = ancestor[b].to(DEV)
        offsets[b] = offset
        parts.append(mask.reshape(-1))
        offset += q_len * K
    return torch.cat(parts).contiguous(), offsets


def call_kernel(
    query,
    kv_cache,
    block_tables,
    seq_lens,
    workspace,
    seq_k,
    *,
    causal,
    custom_mask=None,
    cmask_off=None,
):
    return tokenspeed_mla_decode(
        query=query,
        kv_cache=kv_cache,
        workspace_buffer=workspace,
        kv_lora_rank=KV_LORA,
        qk_rope_head_dim=ROPE,
        block_tables=block_tables,
        seq_lens=seq_lens,
        max_seq_len=seq_k,
        softmax_scale=1.0 / math.sqrt(D_QK),
        causal_mask=causal,
        custom_mask=custom_mask,
        cmask_off=cmask_off,
    )


def absorbed_ref(query, kv_cache, block_tables, seq_lens, ancestor):
    batch, q_len = query.shape[:2]
    scale = 1.0 / math.sqrt(D_QK)
    out = torch.zeros(batch, q_len, H, KV_LORA, device=DEV, dtype=torch.float32)
    qf = query.float()
    for b in range(batch):
        K = int(seq_lens[b])
        hist = K - q_len
        kv = kv_cache[block_tables[b]].reshape(-1, D_QK).float()[:K]
        for qi in range(q_len):
            valid = torch.zeros(K, dtype=torch.bool, device=DEV)
            valid[:hist] = True
            valid[hist:K] = ancestor[b, qi].to(DEV)
            scores = (qf[b, qi] @ kv.t()) * scale
            probs = torch.softmax(
                scores.masked_fill(~valid.view(1, -1), float("-inf")), dim=-1
            )
            out[b, qi] = probs @ kv[:, :KV_LORA]
    return out


def measure_ms(fn, warmup: int, iters: int):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def parse_cases(raw: str):
    cases = []
    for case in raw.split(";"):
        if not case.strip():
            continue
        batch, q_len, seq_k = [int(part.strip()) for part in case.split(",")]
        if batch <= 0 or q_len <= 0 or seq_k < q_len:
            raise ValueError(f"invalid case {case!r}")
        cases.append((batch, q_len, seq_k))
    return cases


def run_case(batch: int, q_len: int, seq_k: int, args) -> bool:
    query, kv_cache, block_tables, seq_lens, workspace = build_inputs(
        batch, q_len, seq_k, seed=batch * 1000 + q_len * 100 + seq_k
    )
    passed = True
    for name, ancestor_cpu in (
        ("chain", chain_ancestor(batch, q_len)),
        ("random", random_tree_ancestor(batch, q_len, seed=seq_k)),
    ):
        custom_mask, cmask_off = build_custom_mask(ancestor_cpu, seq_lens)
        actual = call_kernel(
            query,
            kv_cache,
            block_tables,
            seq_lens,
            workspace,
            seq_k,
            causal=False,
            custom_mask=custom_mask,
            cmask_off=cmask_off,
        ).float()
        expected = absorbed_ref(query, kv_cache, block_tables, seq_lens, ancestor_cpu)
        max_abs = (actual - expected).abs().max().item()
        ok = max_abs < args.tolerance
        print(
            f"[B={batch} q={q_len} K={seq_k}] {name} tree vs ref "
            f"max_abs={max_abs:.3e} {'OK' if ok else 'FAIL'}"
        )
        passed = passed and ok

    if args.iters > 0:
        random_mask, random_off = build_custom_mask(
            random_tree_ancestor(batch, q_len, seed=seq_k), seq_lens
        )
        causal_ms = measure_ms(
            lambda: call_kernel(
                query, kv_cache, block_tables, seq_lens, workspace, seq_k, causal=True
            ),
            args.warmup,
            args.iters,
        )
        tree_ms = measure_ms(
            lambda: call_kernel(
                query,
                kv_cache,
                block_tables,
                seq_lens,
                workspace,
                seq_k,
                causal=False,
                custom_mask=random_mask,
                cmask_off=random_off,
            ),
            args.warmup,
            args.iters,
        )
        print(f"            timing causal={causal_ms:.4f} ms tree={tree_ms:.4f} ms")

    return passed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--cases",
        default="1,4,256;1,8,250;1,8,300;1,16,250;2,8,250",
        help="Semicolon-separated B,q_len,seq_k cases.",
    )
    parser.add_argument("--iters", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--tolerance", type=float, default=0.2)
    args = parser.parse_args()

    torch.cuda.set_device(args.device)
    all_passed = True
    for batch, q_len, seq_k in parse_cases(args.cases):
        all_passed = run_case(batch, q_len, seq_k, args) and all_passed
    raise SystemExit(0 if all_passed else 1)


if __name__ == "__main__":
    main()

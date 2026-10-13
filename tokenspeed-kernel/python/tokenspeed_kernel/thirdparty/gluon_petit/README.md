# Gluon Petit MegaMoE

This directory contains the vendored runtime closure of the Gluon MegaMoE
implementation derived from
[causalflow-ai/petit-kernel](https://github.com/causalflow-ai/petit-kernel),
whose package metadata identifies version 0.0.5. The upstream BSD 3-Clause
license is reproduced in `LICENSE.txt`.

TokenSpeed's package boundary adds the vendor root to Python's module search
path so the upstream `lib` and `petit_kernel` imports resolve unchanged. The
vendored runtime retains only the supported MegaMoE execution path.
The package uses stock Triton 3.8.0 for the Gluon compiler contract used by the
upstream kernels.

The retained runtime supports the registered GFX950 MegaMoE configurations,
including GPT-OSS 120B (EP8, 128 experts, top-4, 2880x3072, biased OpenAI
SwiGLU) and DeepSeek V4 (EP8, 384 experts, top-6, 7168x3072, bias-free SiLU).
The HIP VMM binding is compiled on first use through
`torch.utils.cpp_extension`, so a ROCm development environment is required.

The TokenSpeed adapter lives in `tokenspeed_kernel.ops.moe.gluon.petit` and
registers this runtime as the `gluon` solution with `gluon_petit_*` function
and kernel registration names. The serving options are `--moe-backend gluon_petit`
and `--all2all-backend gluon_petit`; the runtime maps the MoE backend to `gluon`
when selecting a kernel plan.

## Integration status

The runtime retains the upstream LDS memory contract. The currently pinned `tokenspeed-triton` compiler rejects two constructs used by
that contract: numeric LDS pointer address space `3` and the 8196-word shared
allocation used by MegaMoE. This port therefore uses stock Triton 3.8.0 until
the separately reviewed compiler compatibility fix is available in
`tokenspeed-triton`; the tensor conversion does not change either memory requirement.

## Runtime contract

Select the backend with both `--moe-backend gluon_petit` and
`--all2all-backend gluon_petit`. It requires one 8-GPU GFX950 node, EP8/TP1,
BF16 model activations, serialized MXFP4 expert weights, trivial expert
placement, and no more than 1024 tokens per rank. The registered profiles are
GPT-OSS 120B and the DeepSeek V4 expert geometry with unclamped SiLU.
Checkpoints declaring a SiLU activation clamp are rejected because this kernel
cannot preserve that activation. Explicit FP8 activation requests are also
rejected; this backend quantizes activations to MXFP4.

## Retained API

The vendored runtime exposes MegaMoE only. Unused non-MegaMoE launchers and
their FP8 paths are removed; `Moe2StageConfig` and `fmoe_matmul_2stage_*` are
no longer exported. The registered Gluon Petit backend is unchanged.

## Tensor execution

Dispatch, scheduling, both matrix stages, and combine use distributed Gluon
values with explicit masks for memory side effects. Wave-local polling does
not require other waves to participate in a reduction. The thread compiler
and its scalar stage ABI are removed. Matrix tiles use `amd.cdna4.mfma_scaled` with explicit operand and scale
layouts. The per-wave register contract and K accumulation order are preserved.

Partial scale loads suppress inactive lanes at the instruction: an out-of-range
buffer address alone still writes zeros to LDS and can overwrite valid scales.
The barrier utilities document publication scope and payload acquire semantics.

## Validation

Run the GFX950 helper tests and the EP8 sparse-weight numerical reference:

```bash
pytest tokenspeed-kernel/test/amd/ops/moe/test_gluon_petit_tensor.py \
  tokenspeed-kernel/test/amd/ops/moe/test_gluon_petit_mfma.py
torchrun --standalone --nproc-per-node=8 -m pytest --import-mode=importlib \
  tokenspeed-kernel/test/amd/ops/moe/test_gluon_petit_distributed.py
```

The MFMA test compares all supported tile shapes and wave counts bit for bit
against the original native instruction sequence.
The distributed test covers balanced and uneven input counts, empty ranks,
partial scale tiles, capacity, skewed destinations, and refreshed CUDA graphs.
Set `PETIT_OUTPUT_DIR` to save outputs for exact cross-version comparisons.
`check_gluon_petit_ep8.py` in the same directory additionally compares dense
nonzero weights and records compiled kernel resources; its `--action write`
and `--action check` runs must use the same arguments and reference directory.
Select each checkout through `PYTHONPATH=<checkout>/tokenspeed-kernel/python`.
For cross-checkout pytest comparisons, also set
`PETIT_EXPECTED_RUNTIME_ROOT=<checkout>` and use
`--confcutdir=tokenspeed-kernel/test/amd/ops/moe` to prevent the repository's
parent `conftest.py` from overriding that import path. Include the test directory
in `PYTHONPATH` for its shared utilities.

Use the full-path benchmark for both `gpt_oss_120b` and `dsv4`:

```bash
torchrun --standalone --nproc-per-node=8 \
  benchmarks/amd/gfx950/ops/bench_gluon_petit_megamoe.py \
  --profile gpt_oss_120b --tokens 8 16 32 64 128 256 512 1024 \
  --mode graph --warmup 20 --repeat 100 --graph-iters 16 --seed 42
```

Compare repeated paired runs with the same environment and report `total_ms`.
When evaluating the tensor conversion, keep the original baseline results and
also compare against a baseline with only the partial scale-load predicate fix;
that separates the correctness repair from execution-path differences.

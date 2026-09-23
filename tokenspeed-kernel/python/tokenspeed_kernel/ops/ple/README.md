# PLE kernels

N-gram hashing uses one lane per token/head pair. Only head zero writes
shared indices and raw context tails; each head retains the XOR prefix for
its n-gram length and performs one int64 remainder. EOS and raw-tail semantics
are unchanged. Launch sizing includes request-only carried rows for empty batches.

Host gather masks both host table reads and per-row scale reads by shard
ownership. Out-of-shard outputs are still explicitly zeroed.

Convolution and final-state helpers are inlined into one launch. Program i
handles token i and request i independently, covering zero-length requests
without a separate state launch. This removes launch overhead, not all duplicate
input reads. Verify scratch supplies both carried and per-token checkpoints,
so the runtime disables the unused final-state allocation/computation there.
The public facade requires an explicit `write_final`; a disabled final result
has shape `[0, channels, state_len]`.

PDL retains a wait before producer-owned reads. Static conv weights may load
before that wait only with `weights_independent=True` and no contiguity copy.
Gate norm triggers after gated-value stores, before normalized-value reduction;
the fused conv triggers after its incoming wait. Neither trigger publishes
results, and downstream consumers must still synchronize before reading them.
Overlap is opportunistic, not a correctness assumption.

Validation: `test/runtime/test_qwen4_exp.py` covers shard boundaries, FP8/scales,
int64 IDs and EOS/tails, conv references, empty/ragged requests and verify scratch.
Verify scratch tests cover eager/graph replay and PDL on/off. Run on supported
CUDA hardware before deployment; local syntax checks do not validate Triton JIT
or performance. Compare the previous revision using identical weights, shapes,
PDL settings and warmed graph captures; measure whole PLE latency and model TPS,
not just summed kernel durations.

## Exact remainder and opt-in CuTe prototype

`cute_dsl.ple_gate_norm_cute` is an independent NVIDIA prototype, not the
model's default. A 128-thread CTA owns one token/branch, retains intermediate
BF16/FP16 rounding, and shares the key/query cross-warp reduction exchange.
Weights load before the PDL wait; producer-owned activations load after it.
The compile cache includes device, dtype, shape, strides and PDL mode.

`prepare_ngram_reciprocals` accepts host-known positive int64 moduli. Pass its
uint64 result explicitly as `ple_ngram_ids(mod_reciprocals=...)`; `None` retains
the division baseline. Preparation must happen outside capture, once for the
same immutable moduli. The model enables exact reciprocal remainder by default,
preparing a non-persistent buffer from config-generated moduli at construction
and refreshing it if checkpoint loading replaces those moduli.
It follows module device moves without per-forward preparation or a new
checkpoint field. The explicit None path remains available for A/B tests.

For d > 1, let m = floor(2^64 / d). For unsigned n < 2^63,
q = mulhi(n, m) underestimates floor(n / d) by at most one. Thus r = n - q*d
is corrected by subtracting d once if r >= d. d = 1 returns zero. No FP64
approximation or 32-bit hash truncation is used. Negative hashes are outside
the existing PLE contract.

`test/nvidia/ops/test_ple_prototypes.py` tests the independent eager reference,
Triton parity, PDL on/off, graph replay, int64 boundaries, EOS and verify tails.
BF16 CuTe-vs-Triton uses atol/rtol 0.008, both implementations vs eager use
0.01 (eager and the Triton baseline differ at a double-rounding boundary).
FP32 uses 2e-5; IDs and context windows require exact equality.

Run normally with pytest, or execute the test file directly for a minimal
dependency environment. Direct execution constructs package namespaces only
to avoid importing unrelated GEMM/attention backends; it imports the real PLE
facade, Triton kernels, platform module and CuTe implementation. This does not
test whole-package registration or model integration.

Performance tests warm kernels, capture 32 independent invocations, replay
20 times per timed sample and report the median of five CUDA-event samples.
They exclude JIT and Python dispatch, measure both PDL settings and compare
against the current token-by-head Triton baseline, not the older serial-head
implementation. Performance moduli are near one million; adversarial large
moduli belong to correctness tests. These are kernel microbenchmarks, not
end-to-end serving throughput.

### L20D / SM103 prototype results

Environment: PyTorch 2.14.0+cu130, CuTe DSL 4.5.2, tokenspeed-triton
3.8.10.post20260721. Gate inputs: BF16, hidden size 2560, four branches,
projection-style split strides. N-gram: size three, eight heads per n-gram.
All 39 prototype tests passed. Times below are microseconds per invocation.

| PDL | Tokens | Gate Triton | Gate CuTe | N-gram division | N-gram reciprocal |
| --- | ---: | ---: | ---: | ---: | ---: |
| off | 1 | 2.377 | 2.708 | 1.672 | 1.478 |
| off | 4 | 2.436 | 2.718 | 1.799 | 1.671 |
| off | 16 | 2.439 | 2.758 | 1.991 | 1.635 |
| on | 1 | 2.119 | 2.211 | 1.476 | 1.285 |
| on | 4 | 2.120 | 2.247 | 1.606 | 1.415 |
| on | 16 | 2.173 | 2.248 | 1.801 | 1.415 |

The reciprocal path wins this microbenchmark; the standalone CuTe gate does
not. The initial single-warp CuTe prototype took about 9–11 us; four warps and
joint key/query reduction reduced that substantially but did not beat Triton.
Clocks were not locked, and isolated repeated kernels do not establish a
whole-model throughput gain. Only reciprocal remainder is enabled in the model;
the slower CuTe gate prototype remains opt-in.

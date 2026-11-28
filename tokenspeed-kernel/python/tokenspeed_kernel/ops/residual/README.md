# Depth attention residuals

The depth mix weighs completed depth sources and the current candidate independently for each token and FP32 pseudo-query.
Scores use weightless RMS-normalized keys and pseudo-queries scaled by `hidden_size**-0.5`.
Values can remain raw or use the same normalization.

`attention.residual_sum` sums sources in FP32 and stores the original dtype once.
`attention.residual_weighted_sum` mixes `[tokens, S, width]` sources with fixed weights, adding the FP32 products in source order.
These operations do not mutate their inputs.

The mix runs in two steps, on the split kernels Kimi-K3 uses for its `attn_res_fwd` decode path (`activation/triton.py`).
`attention.attention_residual_partials` sweeps the completed sources once per (token, query) and keeps an online-softmax maximum, sum and weighted FP32 accumulator.
`attention.attention_residual_combine` folds the current candidate into those statistics and finishes the mixture.
Its candidate is the FP32 sum of up to three rows, mixed unrounded; with `candidate_out` the sum rounds once to that tensor, as `residual_sum` would, and the rounded value is mixed.
It can apply the following RMSNorm before the single output store; otherwise the mixture is FP32.
The FP32 association differs from a two-pass softmax, within the FP32 tolerances of the tests.
The K3 calls keep their own contract: raw values, one query, and a BF16-rounded mix.
Under PDL, the combine of `attnres_combine_queries` triggers its dependents before its own wait.

The runtime `AttentionResidualState` retains sources within one forward only.
Completed blocks live in one `[blocks, tokens, hidden]` buffer: the embedding at boundary layer zero, then each block's partial sum.
The partial sum of the current block stays as pending terms; the next attention-side mix rounds and stores it.
Every mix of a block set reads the same blocks, so small forwards sweep each set once for all of its queries.
Larger forwards mix each query in one launch, `attention.attention_residual_mix`: one program per (token, query) runs the sweep and the candidate fold and keeps the statistics in registers instead of FP32 scratch, over exact power-of-two column tiles (4096 + 512 for 4608).
It computes the same equations with another FP32 summation order.
With a token range and a destination it mixes only those rows: a sequence-parallel rank (`ResidualShare` in the runtime state) mixes its share of the tokens into a multicast buffer, and after a barrier every rank holds the whole mix.
This state does not add a per-request cache group.
A looped stack runs a core of its layers several times per forward: the state records where the core starts (`loop_entry`), copies out the blocks each pass wrote (`loop_bank`), and restarts the next pass from the recorded entry (`rewind_loop`), reusing the entry's stored partial and its block-set statistics.
The `attention_mixtures` and `mlp_mixture` methods return the FP32 mixtures before their norms, for inputs that are changed before they are normalized.

`attention.message_gate` gates a message into a row per channel, `(1 - tanh(forget)) * x + tanh(gain) * message`, in FP32 with every operation rounded separately (no FMA contraction), from `[tokens, 2 * width]` gain-then-forget logits.
A looped Inkling2 core uses it for the messages its later passes read from the previous pass.

Token counts are runtime scalars of the kernels.
Source count, hidden width, query count, and normalization policy are model properties.
CPU tests cover equations, boundary transitions, and rounding against an FP64 host reference.
Shared GPU tests compare the kernels with the CPU equations.

# Shifted mHC boundary (`mhc_shifted_post_pre_boundary`, `mega_mhc`)

`try_mhc_shifted_post_pre_boundary` implements the Blackwell SM100-family 16-CTA DSMEM cluster shifted mHC boundary for 4-stream BF16 residuals whose prepacked TF32->BF16-hi+lo mixer weights come from `prepare_mhc_boundary_weight(weight)` (updated in place after checkpoint reloads).
- Weights stream directly through shared memory/TMEM via 16-CTA DSMEM cluster reductions and Sinkhorn normalization with zero global split barriers, making every replay deterministic under CUDA graphs. Each cluster communicates only within its own cluster DSMEM and owns disjoint rows with no cross-cluster waits (`docs/design/monokernel.md`), keeping co-resident 1-wave decode launches (`max_decode_wave_clusters(device) = max(1, 2 * multi_processor_count // 16)` clusters, e.g. `14 * 24 = 336` tokens on 112 SMs; `mhc_boundary_max_decode_tokens` returns `0` on `sm103` where unfused StreamFork overlap is faster, while keeping prefill fallback active) and multi-wave launches deadlock-free.
- `warmup_mhc_boundary` compiles all bucketed token tiles `{1, 2, 3, 4, 6, 8, 12, 16, 24}` eagerly before graph capture; uncompiled shapes decline cleanly (`None`) during capture rather than compiling mid-graph.
- Under `deep_gemm >= 2.8.3` (`is_mega_mhc_capture_safe()`), `try_mhc_shifted_post_pre_norm` and `try_mhc_shifted_post_pre_norm_quantize_fp8` (`mega_mhc`) support CUDA graph capture once the eventual capture stream has warmed eagerly. DeepGEMM builds through `2.8.2.post20261115` assert `CaptureStatus::None` in `mega_mhc.hpp` and stay excluded under breakable capture.

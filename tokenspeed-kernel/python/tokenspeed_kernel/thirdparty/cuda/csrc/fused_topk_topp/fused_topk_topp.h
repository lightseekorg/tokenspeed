#ifndef TOKENSPEED_FUSED_TOPK_TOPP_H_
#define TOKENSPEED_FUSED_TOPK_TOPP_H_

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace fused_topk_topp {

using SizeType32 = int32_t;

constexpr int K_TOPK_MAX = 128;

// One workspace covers both pipelines + intermediate top-K results.
size_t getWorkspaceSize(SizeType32 batchSize, SizeType32 vocabSize);

// Renorm-only fused kernel.
//
// Inputs (all on device):
//   probs[bs, V]   — already softmax'd probabilities.
//   topKs[bs]      — int32, per-row K. K_TOPK_MAX is hard upper bound for the
//                    top-k path; K >= V (e.g. (1<<30)) routes the row through
//                    the radix top-p path.
//   topPs[bs]      — float, per-row P in (0, 1].
//
// Output:
//   outProbs[bs, V] — same shape as probs; non-selected positions are 0; kept
//                     positions are renormalized so the row sums to 1.
//
// CUDA-graph safe: every kernel launch has fixed grid/block; per-row mode is
// resolved by the kernels themselves via topKs[row].
void invokeFusedTopKTopP(float const* probs, SizeType32 const* topKs, float const* topPs,
                        float* outProbs, void* workspace, SizeType32 batchSize,
                        SizeType32 vocabSize, cudaStream_t mainStream,
                        cudaStream_t memsetStream);

}  // namespace fused_topk_topp

#endif  // TOKENSPEED_FUSED_TOPK_TOPP_H_

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

"""Checked source transforms for producer-owned TRT-LLM routing padding.

The private JIT module recompiles the routing producers with their headers.
Installed FlashInfer sources and its other modules are never modified.

The native source patterns come from FlashInfer's NVIDIA TRT-LLM routing code
(https://github.com/flashinfer-ai/flashinfer), licensed under Apache-2.0. See
the FlashInfer entry in the distributed THIRDPARTYNOTICES for source paths,
upstream copyrights and license terms. The MIT header above covers TokenSpeed's
original adapter and padding helpers, not the upstream code being transformed.
"""

import re

_HELPERS = r"""
// Tile metadata and live permutation writes have disjoint ownership. Only
// the owner of the last tile of an expert fills its unused rows. This runs
// before the routing kernel's existing PDL completion trigger.
template <typename Params>
__device__ __forceinline__ void initializeRouteTilePadding(
    Params const& params, int32_t validEnd, int32_t tileEnd) {
  if (params.mRouteMapCapacity == 0 || params.mPtrPermutedIdxToTokenIdx == nullptr) return;
  int32_t row = validEnd;
  // Vector stores shorten the per-expert owner's loop without borrowing lanes
  // from other experts or touching the next expert's first live row.
  for (; row < tileEnd && (row & 3) != 0; ++row) {
    params.mPtrPermutedIdxToTokenIdx[row] = -1;
  }
  for (; row + 4 <= tileEnd; row += 4) {
    *reinterpret_cast<int4*>(params.mPtrPermutedIdxToTokenIdx + row) = make_int4(-1, -1, -1, -1);
  }
  for (; row < tileEnd; ++row) {
    params.mPtrPermutedIdxToTokenIdx[row] = -1;
  }
}

// All threads participate after the expert-count scan. Preserve the full
// invalid-row contract, including unused allocation capacity and the guard,
// without overwriting live rows or launching a separate memset. The caller
// supplies the actual allocation length; no extra guard space is assumed.
template <typename Params>
__device__ __forceinline__ void initializeRouteMapSlack(
    Params const& params, int32_t numTiles) {
  if (params.mRouteMapCapacity == 0 || params.mPtrPermutedIdxToTokenIdx == nullptr) return;
  int32_t const paddedEnd = numTiles * params.mTileTokensDim;
  int32_t const thread = blockIdx.x * blockDim.x + threadIdx.x;
  int32_t const stride = gridDim.x * blockDim.x;
  for (int32_t row = paddedEnd + thread; row < params.mRouteMapCapacity; row += stride) {
    params.mPtrPermutedIdxToTokenIdx[row] = -1;
  }
}
"""


def _replace(source: str, old: str, new: str, count: int) -> str:
    if source.count(old) != count:
        raise RuntimeError(
            f"Unsupported FlashInfer routing source: expected {count} occurrences of {old!r}"
        )
    return source.replace(old, new)


def _patch_producer(source: str, count: int) -> str:
    # Every CTA metadata writer owns [validEnd, tileEnd); these ranges do not
    # overlap each other, live assignments, or the trailing allocation slack.
    tile = re.compile(
        r"(?m)^(?P<indent>[ \t]*)params\.mPtrCtaIdxXyToMnLimit\[[^;]+"
        r"= min\(mnLimit1, mnLimit2\);"
    )
    if len(tile.findall(source)) != count:
        raise RuntimeError("Unsupported FlashInfer routing tile writers")
    source = tile.sub(
        lambda m: m[0]
        + "\n"
        + m["indent"]
        + "initializeRouteTilePadding(params, min(mnLimit1, mnLimit2), mnLimit1);",
        source,
    )
    # Match only the padded-count publication block, not a surrounding branch.
    # numNonExitingCtas is the scan aggregate available to all participating
    # threads. Clearing slack from its single publisher would serialize stores.
    publication = re.compile(
        r"(?m)^(?P<indent>[ \t]*)if \([^\n]+\) \{\n"
        r"(?:[ \t]+int32_t permutedIdxSize;\n"
        r"[ \t]+if \(params\.mIsPow2\) \{\n"
        r"[ \t]+permutedIdxSize = mulLog2<int32_t>\(numNonExitingCtas, params\.mPaddingLog2\);\n"
        r"[ \t]+\} else \{\n"
        r"[ \t]+permutedIdxSize = mulTileN<int32_t>\(numNonExitingCtas, params\.mTileTokensDim\);\n"
        r"[ \t]+\}\n\n?)?"
        r"[ \t]+params\.mPtrPermutedIdxSize\[0\] = permutedIdxSize;\n"
        r"[ \t]+params\.mPtrNumNonExitingCtas\[0\] = numNonExitingCtas;\n"
        r"(?P=indent)\}"
    )
    if len(publication.findall(source)) != count:
        raise RuntimeError("Unsupported FlashInfer routing count publishers")
    return publication.sub(
        lambda m: m["indent"]
        + "initializeRouteMapSlack(params, numNonExitingCtas);\n"
        + m[0],
        source,
    )


def patch_routing_sources(sources: dict[str, str]) -> dict[str, str]:
    """Return patched native sources with an explicit allocation-size contract.

    Keys are native filenames. Missing or changed patch sites fail closed;
    callers must compile all returned sources/headers into one private module.
    """
    required = {
        "trtllm_fused_moe_kernel_launcher.cu",
        "trtllm_fused_moe_runner.cu",
        "trtllm_fused_moe_routing_custom.cu",
        "trtllm_fused_moe_routing_llama4.cu",
        "runner.h",
        "RoutingKernel.h",
        "RoutingKernel.cuh",
    }
    if missing := required - sources.keys():
        raise RuntimeError(
            f"Unsupported FlashInfer routing sources: missing {sorted(missing)}"
        )
    producers = {
        "RoutingKernel.cuh": 3,
        "trtllm_fused_moe_routing_custom.cu": 2,
        "trtllm_fused_moe_routing_llama4.cu": 1,
    }
    # A new native producer must not silently retain uninitialized padding.
    for name, source in sources.items():
        stores = re.findall(r"mPtrPermutedIdxToTokenIdx\[[^;\n]+\]\s*=", source)
        if len(stores) != producers.get(name, 0):
            raise RuntimeError(f"Unsupported FlashInfer routing map writers in {name}")
    patched = dict(sources)
    name = "trtllm_fused_moe_kernel_launcher.cu"
    source = sources[name]
    allocation = re.compile(
        r"permuted_idx_to_token_idx\s*=\s*alloc_tensor\("
        r"\{max_num_padded_tokens(?:\s*\+\s*1)?\},\s*"
        r"dl_int32,\s*hidden_states\.device\(\)\);"
    )
    if len(allocation.findall(source)) != 1:
        raise RuntimeError("Unsupported FlashInfer route-map allocation")
    old = (
        "    prepare_routing();\n\n    // Execute routing\n"
        "    tensorrt_llm::kernels::trtllmgen_moe::Routing::Runner routing_runner(tile_tokens_dim);"
    )
    patched[name] = _replace(
        source,
        old,
        old
        + "\n    routing_runner.setRouteMapCapacity("
        + "static_cast<int32_t>(permuted_idx_to_token_idx.numel()));",
        2,
    )

    name = "runner.h"
    source = _replace(
        sources[name],
        "  explicit Runner(int32_t tileTokensDim);",
        "  explicit Runner(int32_t tileTokensDim);\n\n"
        "  void setRouteMapCapacity(int32_t capacity) { mRouteMapCapacity = capacity; }",
        1,
    )
    patched[name] = _replace(
        source,
        "  friend class MoE::Runner;\n  int32_t mTileTokensDim{8};",
        "  friend class MoE::Runner;\n  int32_t mTileTokensDim{8};\n"
        "  int32_t mRouteMapCapacity{0};",
        1,
    )

    name = "trtllm_fused_moe_runner.cu"
    assignment = "routingData.mPtrPermutedIdxToTokenIdx = permutedIdxToTokenIdx;"
    patched[name] = _replace(
        sources[name],
        assignment,
        assignment + "\n    routingData.mRouteMapCapacity = mRouteMapCapacity;",
        5,
    )

    name = "RoutingKernel.h"
    source = _replace(
        sources[name],
        "  // Note: this array (mPtrPermutedIdxToTokenIdx) is uninitialized\n"
        "  // Any out-of-bounds values are undefined.",
        "  // With mRouteMapCapacity > 0, routing initializes all unused entries\n"
        "  // within that allocation to -1 on every invocation, including replay.",
        1,
    )
    source = _replace(
        source,
        "  int32_t* mPtrPermutedIdxToTokenIdx{nullptr};",
        "  int32_t* mPtrPermutedIdxToTokenIdx{nullptr};\n"
        "  // Zero opts out; otherwise every unused entry in this allocation is -1.\n"
        "  int32_t mRouteMapCapacity{0};",
        1,
    )
    source = _replace(
        source,
        "  int32_t* mPtrPermutedIdxToTokenIdx = nullptr;",
        "  int32_t* mPtrPermutedIdxToTokenIdx = nullptr;\n"
        "  int32_t mRouteMapCapacity = 0;",
        1,
    )
    patched[name] = _replace(
        source,
        "    mPtrPermutedIdxToTokenIdx = data.mPtrPermutedIdxToTokenIdx;",
        "    mPtrPermutedIdxToTokenIdx = data.mPtrPermutedIdxToTokenIdx;\n"
        "    mRouteMapCapacity = data.mRouteMapCapacity;",
        1,
    )

    for name, count in producers.items():
        source = _patch_producer(sources[name], count)
        if name == "RoutingKernel.cuh":
            source = _replace(
                source, "namespace routing {", "namespace routing {\n" + _HELPERS, 1
            )
        elif name == "trtllm_fused_moe_routing_llama4.cu":
            # Unlike the common producers, the warp path reserves rows for
            # remote experts too, but never scatters valid indices into them.
            source = _replace(
                source,
                "initializeRouteTilePadding(params, min(mnLimit1, mnLimit2), mnLimit1);",
                "int32_t const routeLocalExpert = expertIdx - params.mLocalExpertsStartIdx;\n"
                "      bool const routeIsLocal = routeLocalExpert >= 0 &&\n"
                "          routeLocalExpert < (params.mNumLocalExperts << params.mLocalExpertsStrideLog2) &&\n"
                "          (routeLocalExpert & ((1 << params.mLocalExpertsStrideLog2) - 1)) == 0;\n"
                "      initializeRouteTilePadding(params, routeIsLocal ? min(mnLimit1, mnLimit2)\n"
                "          : mnLimit1 - params.mTileTokensDim, mnLimit1);",
                1,
            )
        patched[name] = source
    return patched

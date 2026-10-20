// Copyright (c) 2026 LightSeek Foundation
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// EAGLE-Harness tree policy on the device (EAGLE-4): EAGLE-4's round
// decisions (tree width, chain depth and verify count), taken by single-CTA
// kernels that read and write device memory only, so a round's depths can run
// as captured graphs without host synchronization.
//
// device_policy.py precomputes every cost-data value in Python double precision
// into tables indexed by batch size, cost data, context interval, cost variant
// and depth; the kernels only compare, and evaluate the few operations on
// device inputs (work deltas, ranked scores) with one IEEE rounding per Python
// operation, in Python's order (explicit __d*_rn intrinsics: no FMA contraction).
//
// The call order per round is device_policy.py's. Slots, per depth d: width
// (lanes depth d expands), cumu_start (pool size before depth d's children),
// work_size (node count of the work deltas queued after depth d, 0 for none),
// go (depth d drafts). go is mirrored to mapped pinned memory as
// (seq << 1) | go, seq counting begin_round launches, so the host can poll it
// without a CUDA synchronization. device_policy.py refuses cost data
// EAGLE-4's decisions reject, so the kernels check none.
//
// A chain round whose batch size the width table covers (batch_table's
// kTableWidthCol, W > 0) takes no decision: begin_round writes every depth's
// slots and go flags (depths 1..W-2 draft) and the round's counts (verify W),
// reads no cost data, and the decisions launched in its depths change no
// state or slot.
//
// Latency: the kernels run between the drafter's forwards, with L2 cold, and
// the drafter waits on most of them. Each gathers its inputs into shared
// memory in waves of independent loads, then one thread decides from shared
// memory. begin_round copies the round's cost row (fixed by its batch size
// and context interval) to round_f64 and stages the batch size's
// largest past verify width in the state, so the decisions read their inputs
// in one wave. fp64 division costs several hundred cycles
// on some GPUs: the verify's divisions run in parallel, off its serial steps.

#include <cub/block/block_scan.cuh>

#include <climits>
#include <cstdint>
#include <functional>

#include "tvm_ffi_utils.h"

namespace {

// Scalars of the int32 state tensor: the policy's persistent state, then the
// current round's.
enum State : int {
  kTiny = 0,        // the last compared tree decision ran on one request (kept across rounds)
  kTier,            // the batch tier: 0 small, 1 medium, 2 large, 3 huge
  kSampledNext,     // the stop depth the next sampled round plans
  kSampledPlan,     // this sampled round's planned stop depth
  kSampledSeen,     // first depth whose decision said stop (-1: none)
  kBenchBatchSize,  // exec mode 1's benchmark batch size (loaded before a round needs it)
  kRankedCount,     // cumulative ranked scores the next round sizes on (0: none)
  kBatchSize,       // the round's live requests (0: no round runs)
  kSampled,
  kCostId,
  kInterval,
  kVariant,         // tree: tier at round start, or kTinyVariant; chain: tier row
  kHistMax,         // tree: the batch size's largest past verify width
  kPastVerifyNum,
  kLastDepth,       // depth whose decision stopped the round
  kStopWidth,       // that decision's width
  kCumuCount,       // pool size after the round
  kVerifyCount,
  kFlat,
  kVerifyNum,
  kTableWidth,      // chain: the width table's verify width of the round (0: the rule decides)
  kNumState,
};

// Rows of the [4, depth + 1] slot tensor.
enum SlotRow : int { kWidthRow = 0, kCumuStartRow, kWorkSizeRow, kGoRow, kNumSlotRows };

constexpr int kTiers = 4;
constexpr int kTinyVariant = kTiers;  // cost variant (and verify floor) of a tiny round
constexpr int kVariants = kTiers + 1;
// tier_table: the verify floor per cost variant, then the width per tier.
constexpr int kTierTableSize = kVariants + kTiers;
constexpr int kMaxWidths = 32;  // measured target widths per interval
// A tree decision compares the work delta at its tier's width, min(K, 8 or 4).
constexpr int kMaxNarrow = 8;
// Verify history lengths (EAGLE-4 keeps 16 past and 5 recent widths).
constexpr int kMaxHistory = 32;
// Tree depths whose cost row begin_round copies in one wave.
constexpr int kMaxDepth = 32;

// Columns of batch_table ([max_bs, cols], int64): begin_round's inputs per
// batch size; kTableWidthCol: a chain's width table width (0: the rule decides).
enum BatchCol : int {
  kCostIdCol = 0,
  kChainTierRowCol,
  kIntervalsCol,
  kCacheStepCol,
  kTableWidthCol,
  kBatchCols
};
// Columns of interval_i32 ([costs, intervals, cols], trees).
enum IntervalI32Col : int { kNumWidthsCol = 0, kWidthsCol };
// Columns of interval_f64: draft_cost[-1], model cost ratio, then the stop
// thresholds per cost variant and depth, then the verify cost deltas.
enum IntervalF64Col : int { kDraftCostCol = 0, kModelCostRatioCol, kStopThresholdCol };
// Entries of chain_f64 ([costs, intervals, 2 tier rows, depth, entries]); the
// cliff entries are 0 where no verify cost cliff makes the test look ahead.
enum ChainF64 : int { kGainDenom = 0, kStepCost, kChainThreshold, kCliffGainDenom, kCliffCost,
                      kChainF64 };
// The round's cost row, copied by begin_round: a tree's round_f64 holds the
// draft cost, model cost ratio and the stop threshold per depth (of the
// round's cost variant); a chain's its chain_f64 entries per depth (of the
// round's tier row).
enum RoundF64 : int { kRoundDraftCost = 0, kRoundModelCostRatio, kRoundThresholds };

constexpr int kWarp = 32;
constexpr unsigned kFullMask = 0xffffffffu;

// torch.cumsum's CUB scan: 128 threads of 15 items per tile; two tiles keep
// the decoupled look-back's sums in a fixed order.
constexpr int kRankThreads = 128;
constexpr int kRankItems = 15;
constexpr int kRankTile = kRankThreads * kRankItems;
constexpr int kRankTiles = 2;
// The shapes whose torch.mean(dim=0) reduction order the ranking replicates.
constexpr int kRankMaxRows = 32;
constexpr int kRankMaxCols = 64;

using BlockScan = cub::BlockScan<float, kRankThreads, cub::BLOCK_SCAN_WARP_SCANS>;

constexpr int ceil_div(int a, int b) { return (a + b - 1) / b; }

// One lane or thread per entry: the one-warp kernels load a state scalar,
// tier table entry and history entry per lane, verify_width_choose a width per
// lane; select_verify's last two warps load the scores a verify compares.
static_assert(kNumState <= kWarp && kTierTableSize <= kWarp && kMaxHistory <= kWarp);
static_assert(kMaxNarrow < kWarp && kChainF64 <= kWarp && kMaxWidths <= kWarp);
static_assert(kRankMaxCols <= kRankThreads - 2 * kWarp);
// The ranked scores fit two scan tiles.
static_assert(kRankMaxRows * kRankMaxCols <= kRankTile * kRankTiles);

// Python's // on integers (floor), b != 0; 32-bit division when both fit.
__device__ __forceinline__ int64_t floor_div(int64_t a, int64_t b) {
  if (0 <= a && a <= UINT32_MAX && 0 < b && b <= UINT32_MAX) {
    return static_cast<uint32_t>(a) / static_cast<uint32_t>(b);
  }
  int64_t q = a / b;
  if (a % b != 0 && ((a < 0) != (b < 0))) --q;
  return q;
}

// Up to kPerThread values per thread in registers: a kernel loads all its
// inputs of a wave (load) before it stores any to shared memory (store), so
// the loads overlap. Thread tid holds entries tid + k * threads, n <=
// kPerThread * threads.
template <int kPerThread, typename T>
struct Fetch {
  T v[kPerThread];

  template <typename Src>
  __device__ __forceinline__ void load(int n, int tid, int threads, Src src) {
#pragma unroll
    for (int k = 0; k < kPerThread; ++k) {
      const int i = k * threads + tid;
      if (i < n) v[k] = src(i);
    }
  }

  __device__ __forceinline__ void store(T* dst, int n, int tid, int threads) const {
#pragma unroll
    for (int k = 0; k < kPerThread; ++k) {
      const int i = k * threads + tid;
      if (i < n) dst[i] = v[k];
    }
  }
};

// A width decision: the next depth's width and whether the cost criterion stops.
struct Decision {
  bool decided;  // false: the decision compares work deltas
  int width;
  bool cost_stop;
};

// A round kernel's state and slots in shared memory: the slots of its depth
// d (cur) and of depth d + 1 (next, written back at the end).
struct RoundShared {
  int32_t s[kNumState];
  int32_t cur[kNumSlotRows];
  int32_t next[kNumSlotRows];
  int32_t tier_table[kTierTableSize];
  int64_t seq;
};

// A tree round's past verify width at the current tiny and tier.
__device__ __forceinline__ int tree_past_verify_num(const int32_t* s,
                                                    const int32_t* tier_table) {
  return max(s[kHistMax], tier_table[s[kTiny] ? kTinyVariant : s[kTier]]);
}

// Depth d's step of a round, on the shared copies (one thread).
struct Round {
  int32_t* s;
  int32_t* cur;
  int32_t* next;
  int64_t* go_host;  // [tree_depth + 1], mapped pinned memory
  int64_t seq;
  int depth;
  int tree_depth;
  int node_top_k;
  bool chain;
  const int32_t* tier_table;  // trees
  int verify_num;
  int min_verify_count;       // trees

  // Whether depth d + 1 drafts, also to the host mirror: a relaxed
  // system-scope store, unfenced, as the host reads nothing else on it.
  __device__ void publish_go(bool draft) {
    next[kGoRow] = draft;
    const int64_t value = (seq << 1) | (draft ? 1 : 0);
    *reinterpret_cast<volatile int64_t*>(go_host + depth + 1) = value;
  }

  // Whether depth d's slot asks for work deltas, which the next decision compares.
  __device__ bool deltas_depth(int d) const {
    if (chain) return d % 2 == 1 && d < tree_depth - 1;
    if (s[kTiny] && d < 2) return false;
    if (d == tree_depth - 1) return false;
    return d % 2 == 0;
  }

  // Depth d's decision without work deltas ("narrow": the
  // tree's width at the round's tier); not decided when it compares them.
  __device__ Decision early_decision(int d, int narrow) const {
    if (chain) {
      if (d == tree_depth - 1) return {true, -1, true};
      if (d == 0 || d % 2 == 1) return {true, 1, false};
      return {false, 0, false};
    }
    if (s[kTiny] && d < 2) return {true, node_top_k, false};
    if (d == tree_depth - 1) return {true, -1, true};
    if (d % 2 == 0) return {true, narrow, false};
    return {false, 0, false};
  }

  // The round stops after depth d (its children are the last in the pool);
  // then complete_round (the start of the verify tree write): the ranked node count
  // (verify_count) and whether the ranking is flat; a chain verifies every
  // ranked node.
  __device__ void finish(int decided_width) {
    s[kLastDepth] = depth;
    s[kStopWidth] = decided_width;
    s[kCumuCount] = cur[kCumuStartRow] + node_top_k * cur[kWidthRow];
    publish_go(false);
    if (chain) {
      s[kVerifyCount] = min(s[kCumuCount], verify_num);
      s[kVerifyNum] = s[kVerifyCount];
    } else {
      // past * 2 in int64: a cut round records a measured width up to 2**31 - 1.
      const int64_t past = tree_past_verify_num(s, tier_table);
      s[kVerifyCount] = static_cast<int>(
          min(static_cast<int64_t>(min(verify_num, s[kCumuCount])),
              max(past * 2, static_cast<int64_t>(min_verify_count))));
      s[kFlat] = s[kTiny];
    }
  }

  // Take depth d's decision: sampled rounds stop only at their planned depth;
  // a continuing round writes depth d + 1's slots.
  __device__ void commit(int decided_width, bool cost_stop) {
    const int d = depth;
    bool stop = cost_stop;
    if (s[kSampled]) {
      if (cost_stop && s[kSampledSeen] < 0) s[kSampledSeen] = d;
      stop = d >= s[kSampledPlan];
    }
    if (stop) {
      finish(decided_width);
      return;
    }
    const int start = cur[kCumuStartRow] + node_top_k * cur[kWidthRow];
    next[kWidthRow] = decided_width;
    next[kCumuStartRow] = start;
    next[kWorkSizeRow] = deltas_depth(d + 1) ? min(s[kPastVerifyNum], start) : 0;
    publish_go(true);
  }
};

// The loads of a decision kernel's one wave (one warp) are unconditional,
// at indices clamped into range, so they issue back to back without
// branches; the stores to shared memory follow them.
struct RoundLoad {
  int32_t state_value;
  int32_t slot_value;
  int64_t seq_value;

  __device__ __forceinline__ void load(const int32_t* state, const int32_t* slots,
                                       const int64_t* seq, int depth_slots, int d) {
    const int lane = threadIdx.x;
    state_value = state[min(lane, kNumState - 1)];
    slot_value = slots[min(lane, kNumSlotRows - 1) * depth_slots + d];
    seq_value = *seq;
  }

  __device__ __forceinline__ void store(RoundShared& sh) const {
    const int lane = threadIdx.x;
    if (lane < kNumState) sh.s[lane] = state_value;
    if (lane < kNumSlotRows) {
      sh.cur[lane] = slot_value;
      sh.next[lane] = 0;
    }
    if (lane == 0) sh.seq = seq_value;
  }
};

// The end of a decision kernel (one warp): the changed state scalars and
// depth d + 1's slots back to global memory.
__device__ __forceinline__ void store_round(const RoundShared& sh, int32_t* state,
                                            int32_t state_value, int32_t* slots,
                                            int depth_slots, int d) {
  const int lane = threadIdx.x;
  if (lane < kNumState && sh.s[lane] != state_value) state[lane] = sh.s[lane];
  if (lane < kNumSlotRows) slots[lane * depth_slots + d + 1] = sh.next[lane];
}

// Exec mode 1's batch cut of bs.
__device__ bool batch_cut(const int32_t* s, int exec_mode, int bs) {
  if (exec_mode != 1 || bs == 1) return false;
  const int64_t bench = s[kBenchBatchSize];
  return bench > 4 * bs && bench > bs + 4;
}

// ---------------------------------------------------------------------------
// Round start (finish_round): past_verify_num, the sampled plan, the cost
// context's interval, depth 0's decision and depth 1's slots.
// ---------------------------------------------------------------------------
struct BeginInputs {
  int bs;
  bool sampled;
  int64_t frontier_sum;      // of the first bs cached lengths
  const int64_t* batch_row;  // batch_table's row of bs
  int hist_max;              // trees: the largest past verify width of bs
  int chain_past_verify_num;
};

// A width table round of width W: depths 1..W-2 draft (the last drafted depth
// W-2), and it verifies its whole pool of W nodes. Its stopping depth and width
// are those the confidence rule records for a stop there.
__device__ void begin_table_round(Round& r, int width) {
  int32_t* s = r.s;
  const int last = width - 2;
  s[kTableWidth] = width;
  s[kLastDepth] = last;
  s[kStopWidth] = last == r.tree_depth - 1 ? -1 : 1;
  s[kCumuCount] = width;
  s[kVerifyCount] = min(width, r.verify_num);
  s[kVerifyNum] = s[kVerifyCount];
}

// A width table round's slot of depth d (last: its last drafted depth): width
// one from pool size d + 1, no work deltas, go up to the last drafted depth.
// Depth 1 always runs (it begins the round), so a round stopping at depth 0
// keeps a width-one depth 1 whose child no verify ranks.
__device__ __forceinline__ int table_round_slot(int row, int d, int last) {
  const bool runs = d <= max(last, 1);
  if (row == kWidthRow) return runs ? 1 : 0;
  if (row == kCumuStartRow) return runs ? d + 1 : 0;
  if (row == kWorkSizeRow) return 0;
  return d == 0 || d <= last;
}

__device__ void begin_round(Round& r, const BeginInputs& in) {
  int32_t* s = r.s;
  for (int i = kBatchSize; i < kNumState; ++i) s[i] = 0;
  const int bs = in.bs;
  if (bs == 0) {
    r.publish_go(false);
    return;
  }
  s[kBatchSize] = bs;
  s[kSampled] = in.sampled;
  if (r.chain) {
    s[kPastVerifyNum] = in.chain_past_verify_num;
    const int table_width = static_cast<int>(in.batch_row[kTableWidthCol]);
    if (table_width > 0) {
      begin_table_round(r, table_width);
      return;
    }
  } else {
    s[kHistMax] = in.hist_max;
    s[kPastVerifyNum] = tree_past_verify_num(s, r.tier_table);
  }
  if (in.sampled) {
    // A sampled round stops at the depth the last sampled round planned.
    s[kSampledPlan] = s[kSampledNext];
    s[kSampledSeen] = -1;
  }
  // The cost context's interval of the mean cached length.
  const int64_t* row = in.batch_row;
  const int64_t step = row[kCacheStepCol];
  s[kCostId] = static_cast<int>(row[kCostIdCol]);
  const int64_t average_cache_len = floor_div(in.frontier_sum, bs);
  // (average_cache_len - step) // step, exactly.
  const int64_t interval =
      min(max(floor_div(average_cache_len, step) - 1, static_cast<int64_t>(0)),
          row[kIntervalsCol] - 1);
  s[kInterval] = static_cast<int>(interval);
  // Set at the round's start, not at its first cost read: tiny and tier change
  // only after that read.
  s[kVariant] = r.chain ? static_cast<int>(row[kChainTierRowCol])
                        : (s[kTiny] ? kTinyVariant : s[kTier]);
  // Depth 0: the root (one node) and its K children from the first step.
  r.cur[kWidthRow] = 1;
  r.cur[kCumuStartRow] = 1;
  r.cur[kWorkSizeRow] = r.deltas_depth(0) ? 1 : 0;
  r.cur[kGoRow] = 1;
  // Depth 0's decision never compares work deltas: always decided early.
  const Decision decision = r.early_decision(0, r.chain ? 1 : r.tier_table[kVariants + s[kTier]]);
  r.commit(decision.width, decision.cost_stop);
}

// The cost table begin_round copies the round's row from: trees'
// interval_f64, chains' chain_f64.
struct CostTable {
  const double* f64;
  int num_intervals;
  int cols;  // trees
};

__global__ void __launch_bounds__(kWarp)
    begin_round_kernel(int32_t* state, int32_t* slots, int64_t* go_host, int64_t* seq,
                       int tree_depth, int node_top_k, bool chain, const int32_t* hist,
                       int hist_len, const int64_t* batch_table, const int32_t* tier_table,
                       int tier_table_size, CostTable costs, double* round_f64,
                       const int32_t* live_rows,
                       const int64_t* frontier, int64_t frontier_stride, const bool* sampled,
                       int chain_past_verify_num, int verify_num, int min_verify_count) {
  __shared__ RoundShared sh;
  __shared__ int64_t batch_row[kBatchCols];
  __shared__ int32_t round_bs;
  __shared__ bool round_sampled;
  const int lane = threadIdx.x;
  const int depth_slots = tree_depth + 1;
  // Wave 1.
  const int32_t state_value = lane < kNumState ? state[lane] : 0;
  const int32_t tier_value = lane < tier_table_size ? tier_table[lane] : 0;
  const int64_t seq_value = lane == 0 ? *seq : 0;
  const int32_t bs_value = lane == 0 ? *live_rows : 0;
  const bool sampled_value = lane == 0 ? *sampled : false;
  if (lane < kNumState) sh.s[lane] = state_value;
  if (lane < kNumSlotRows) {
    sh.cur[lane] = 0;
    sh.next[lane] = 0;
  }
  if (lane < kTierTableSize) sh.tier_table[lane] = tier_value;
  if (lane == 0) {
    sh.seq = seq_value;
    round_bs = bs_value;
    round_sampled = sampled_value;
  }
  __syncwarp();
  // Wave 2: the batch size's table row, verify history and cached lengths.
  const int bs = round_bs;
  const bool load = bs > 0;
  const int64_t* table_row = batch_table + static_cast<int64_t>(max(bs - 1, 0)) * kBatchCols;
  const int32_t* history = hist + static_cast<int64_t>(max(bs - 1, 0)) * hist_len;
  const int64_t batch_value = load && lane < kBatchCols ? __ldcg(table_row + lane) : 0;
  const int32_t hist_value = load && lane < hist_len ? __ldcg(history + lane) : INT_MIN;
  // Integer sums are exact in any order.
  int64_t partial = 0;
  if (load) {
    constexpr int kPerLane = 8;
    for (int base = 0; base < bs; base += kPerLane * kWarp) {
      int64_t v[kPerLane];
#pragma unroll
      for (int k = 0; k < kPerLane; ++k) {
        const int i = base + k * kWarp + lane;
        v[k] = i < bs ? __ldcg(frontier + i * frontier_stride) : 0;
      }
#pragma unroll
      for (int k = 0; k < kPerLane; ++k) partial += v[k];
    }
  }
  for (int offset = kWarp / 2; offset > 0; offset /= 2) {
    partial += __shfl_xor_sync(kFullMask, partial, offset);
  }
  const int hist_max = __reduce_max_sync(kFullMask, hist_value);
  if (lane < kBatchCols) batch_row[lane] = batch_value;
  __syncwarp();
  if (lane == 0) {
    Round r{sh.s,       sh.cur,     sh.next, go_host,       sh.seq + 1,      0,
            tree_depth, node_top_k, chain,   sh.tier_table, verify_num, min_verify_count};
    BeginInputs in{bs, round_sampled, partial, batch_row, hist_max, chain_past_verify_num};
    begin_round(r, in);
    *seq = r.seq;
    sh.seq = r.seq;
  }
  __syncwarp();
  const int table_width = sh.s[kTableWidth];
  const int table_last = table_width - 2;
  // A table round's go flags, up to the depth after its last drafted one
  // (at least depth 2, which the drafter asks after depth 1).
  if (table_width > 0) {
    for (int d = lane + 1; d <= max(table_last + 1, 2); d += kWarp) {
      const int64_t value = (sh.seq << 1) | (d <= table_last ? 1 : 0);
      *reinterpret_cast<volatile int64_t*>(go_host + d) = value;
    }
  }
  // Wave 3: the round's cost row, for its decisions (a table round takes none).
  const int64_t row = static_cast<int64_t>(sh.s[kCostId]) * costs.num_intervals + sh.s[kInterval];
  const int variant = sh.s[kVariant];
  if (sh.s[kBatchSize] > 0 && table_width == 0) {
    if (chain) {
      const double* terms = costs.f64 + (row * 2 + variant) * tree_depth * kChainF64;
      Fetch<ceil_div(kMaxDepth * kChainF64, kWarp), double> term_values;
      term_values.load(tree_depth * kChainF64, lane, kWarp,
                       [=](int i) { return __ldcg(terms + i); });
      term_values.store(round_f64, tree_depth * kChainF64, lane, kWarp);
    } else {
      const double* f64_row = costs.f64 + row * costs.cols;
      const double* thresholds = f64_row + kStopThresholdCol + variant * tree_depth;
      Fetch<ceil_div(kRoundThresholds + kMaxDepth, kWarp), double> values;
      values.load(kRoundThresholds + tree_depth, lane, kWarp, [=](int i) {
        return i < kRoundThresholds ? __ldcg(f64_row + kDraftCostCol + i)
                                    : __ldcg(thresholds + i - kRoundThresholds);
      });
      values.store(round_f64, kRoundThresholds + tree_depth, lane, kWarp);
    }
  }
  if (lane < kNumState && sh.s[lane] != state_value) state[lane] = sh.s[lane];
  // Depths 0 and 1 from the round, the others cleared; a table round's every depth.
  for (int row = 0; row < kNumSlotRows; ++row) {
    for (int d = lane; d < depth_slots; d += kWarp) {
      slots[row * depth_slots + d] = table_width > 0 ? table_round_slot(row, d, table_last)
                                     : d == 0        ? sh.cur[row]
                                     : d == 1        ? sh.next[row]
                                                     : 0;
    }
  }
}

// ---------------------------------------------------------------------------
// Depth d's tree width decision, before depth d drafts.
// ---------------------------------------------------------------------------
// row: draft_cost[-1], the model cost ratio and the stop threshold at the
// decision's depth, from round_f64.
__device__ void decide_tree(Round& r, const float* work_deltas, const double* row, int exec_mode,
                            double async_layer_ratio) {
  const int d = r.depth;
  int32_t* s = r.s;
  if (r.cur[kGoRow] != 1) {
    r.publish_go(false);
    return;
  }
  const int narrow = r.tier_table[kVariants + s[kTier]];
  Decision decision = r.early_decision(d, narrow);
  if (!decision.decided) {
    const int bs = s[kBatchSize];
    if (batch_cut(s, exec_mode, bs)) {
      decision = {true, 1, true};
    } else {
      s[kTiny] = bs == 1;
      // (work_deltas[width] - work_deltas[0]) / draft_cost[-1] * model_cost_ratio
      // * async_layer_ratio
      double gain = __dsub_rn(static_cast<double>(work_deltas[narrow]),
                              static_cast<double>(work_deltas[0]));
      gain = __ddiv_rn(gain, row[kRoundDraftCost]);
      gain = __dmul_rn(gain, row[kRoundModelCostRatio]);
      gain = __dmul_rn(gain, async_layer_ratio);
      decision = {true, narrow, gain < row[kRoundThresholds]};
    }
  }
  r.commit(decision.width, decision.cost_stop);
}

// Depth d's chain depth decision, before depth d drafts.
__device__ void decide_chain(Round& r, const float* work_deltas, const double* terms) {
  if (r.cur[kGoRow] != 1) {
    r.publish_go(false);
    return;
  }
  Decision decision = r.early_decision(r.depth, 1);
  if (!decision.decided) {
    // raw_gain * 2.0 / (1.0 + depth * depth_loss), against the step's cost
    const double raw_gain = __dsub_rn(static_cast<double>(work_deltas[1]),
                                      static_cast<double>(work_deltas[0]));
    const double doubled = __dmul_rn(raw_gain, 2.0);
    const double curr_gain = __ddiv_rn(doubled, terms[kGainDenom]);
    bool stop = __ddiv_rn(curr_gain, terms[kStepCost]) < terms[kChainThreshold];
    if (stop && terms[kCliffCost] != 0.0) {
      // A verify cost cliff: look two depths ahead.
      const double gain_two_steps =
          __dadd_rn(curr_gain, __ddiv_rn(doubled, terms[kCliffGainDenom]));
      stop = __ddiv_rn(gain_two_steps, terms[kCliffCost]) < terms[kChainThreshold];
    }
    decision = {true, 1, stop};
  }
  r.commit(decision.width, decision.cost_stop);
}

// Depth d's decision of a tree or a chain round, one warp.
template <bool kChain>
__global__ void __launch_bounds__(kWarp)
    decide_kernel(int32_t* state, int32_t* slots, int64_t* go_host, const int64_t* seq,
                  int tree_depth, int node_top_k, const float* work_deltas,
                  const double* round_f64, const int32_t* tier_table, int depth, int exec_mode,
                  double async_layer_ratio, int verify_num, int min_verify_count) {
  constexpr int kDeltas = kChain ? 2 : kMaxNarrow + 1;
  constexpr int kRow = kChain ? kChainF64 : kRoundThresholds + 1;
  __shared__ RoundShared sh;
  __shared__ float deltas[kDeltas];
  __shared__ double row[kRow];
  const int lane = threadIdx.x;
  const int d = depth;
  const int depth_slots = tree_depth + 1;
  const int top_k = kChain ? 1 : node_top_k;
  // One wave: the round, the work deltas it compares and its cost row at depth d.
  RoundLoad round;
  round.load(state, slots, seq, depth_slots, d);
  const int32_t tier_value = kChain ? 0 : tier_table[min(lane, kTierTableSize - 1)];
  const float delta_value = work_deltas[min(lane, min(top_k, kDeltas - 1))];
  const double row_value =
      round_f64[kChain ? d * kChainF64 + min(lane, kChainF64 - 1)
                       : (lane < kRoundThresholds ? lane : kRoundThresholds + d)];
  round.store(sh);
  if (!kChain && lane < kTierTableSize) sh.tier_table[lane] = tier_value;
  if (lane < kDeltas) deltas[lane] = delta_value;
  if (lane < kRow) row[lane] = row_value;
  __syncwarp();
  // No live round (none begun, none requested, or a warm-up): nothing to decide.
  if (sh.s[kBatchSize] <= 0) return;
  if (kChain && sh.s[kTableWidth] > 0) {
    // begin_round wrote the width table round's slots and go flags; past its
    // last drafted depth a decision only clears the next go flag, as a stopped
    // round's does, and leaves the slots.
    if (lane == 0 && sh.cur[kGoRow] != 1) {
      *reinterpret_cast<volatile int64_t*>(go_host + d + 1) = sh.seq << 1;
    }
    return;
  }
  if (lane == 0) {
    Round r{sh.s,   sh.cur, sh.next, go_host, sh.seq, d, tree_depth, top_k, kChain,
            kChain ? nullptr : sh.tier_table, verify_num, kChain ? 0 : min_verify_count};
    if constexpr (kChain) {
      decide_chain(r, deltas, row);
    } else {
      decide_tree(r, deltas, row, exec_mode, async_layer_ratio);
    }
  }
  __syncwarp();
  store_round(sh, state, round.state_value, slots, depth_slots, d);
}

// ---------------------------------------------------------------------------
// Cumulative ranked scores of top_scores[:rows, :cols], bit for bit as
// torch.cumsum(top_scores.mean(dim=0), dim=0) (or of top_scores.view(-1)
// when flat) on a contiguous tensor.
// ---------------------------------------------------------------------------
__device__ __forceinline__ int last_pow2(int n) {
  int p = 1;
  while (p * 2 <= n) p *= 2;
  return p;
}

// torch's mean over dim 0 of a contiguous [rows, cols] float tensor
// (Reduce.cuh with output vectorization, rows <= 32, cols <= 64): each thread
// sums a column into 4 accumulators by row index mod 4 and combines them in
// order; one column is instead split over last_pow2(rows) lanes, each summing
// rows l and l + lanes, then reduced by halving offsets. The sum is scaled by
// float(cols) / (rows * cols).
__device__ float column_mean(const float* top, int stride, int rows, int cols, int c) {
  const float factor = __fdiv_rn(static_cast<float>(cols), static_cast<float>(rows * cols));
  float total;
  if (cols == 1) {
    float lanes[kRankMaxRows];
    const int width = last_pow2(rows);
    for (int l = 0; l < width; ++l) {
      const float first = __fadd_rn(0.0f, top[l * stride]);
      const float second = l + width < rows ? __fadd_rn(0.0f, top[(l + width) * stride]) : 0.0f;
      lanes[l] = __fadd_rn(__fadd_rn(__fadd_rn(first, second), 0.0f), 0.0f);
    }
    for (int offset = width / 2; offset > 0; offset /= 2) {
      for (int l = 0; l < offset; ++l) lanes[l] = __fadd_rn(lanes[l], lanes[l + offset]);
    }
    total = lanes[0];
  } else {
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int k = 0; k < rows; ++k) acc[k % 4] = __fadd_rn(acc[k % 4], top[k * stride + c]);
    total = __fadd_rn(__fadd_rn(__fadd_rn(acc[0], acc[1]), acc[2]), acc[3]);
  }
  return __fmul_rn(total, factor);
}

// The block prefix of the second tile: the first tile's aggregate, as CUB's
// decoupled look-back hands it over.
struct TilePrefix {
  float prefix;
  __device__ float operator()(float) const { return prefix; }
};

// Shared memory of the ranking: the [rows, cols] scores (row-major, zero to
// the end of the second scan tile), turned in place into the cumulative
// ranked scores.
struct RankShared {
  float values[kRankTile * kRankTiles];
  float means[kRankMaxCols];
  typename BlockScan::TempStorage scan;
};

// top[:rows, :cols] (rows of `stride` floats, a supported shape) loaded into
// registers, then stored to rank.values with zeros past it; all
// kRankThreads threads.
struct ScoreFetch {
  // Thread tid holds column tid % kRankMaxCols of rows tid / kRankMaxCols + k * kRowStep.
  static constexpr int kRowStep = kRankThreads / kRankMaxCols;
  static constexpr int kPerThread = kRankMaxRows / kRowStep;
  float v[kPerThread];

  __device__ __forceinline__ void load(const float* top, int64_t stride, int rows, int cols) {
    const int col = threadIdx.x % kRankMaxCols;
    const int row = threadIdx.x / kRankMaxCols;
    const float* p = top + row * stride + col;
#pragma unroll
    for (int k = 0; k < kPerThread; ++k) {
      if (col < cols && row + k * kRowStep < rows) v[k] = __ldcg(p);
      p += kRowStep * stride;
    }
  }

  __device__ __forceinline__ void store(RankShared& rank, int rows, int cols) const {
    const int tid = threadIdx.x;
    const int n = rows * cols;
    const int col = tid % kRankMaxCols;
    const int row = tid / kRankMaxCols;
#pragma unroll
    for (int k = 0; k < kPerThread; ++k) {
      if (col < cols && row + k * kRowStep < rows) {
        rank.values[(row + k * kRowStep) * cols + col] = v[k];
      }
    }
    for (int i = n + (tid - n % kRankThreads + kRankThreads) % kRankThreads;
         i < kRankTile * kRankTiles; i += kRankThreads) {
      rank.values[i] = 0.0f;
    }
  }
};

// The cumulative ranked scores, in place in rank.values[0, ranked count); all
// kRankThreads threads, after ScoreFetch::store and a barrier.
__device__ void rank_scores(RankShared& rank, int rows, int cols, bool flat) {
  const int tid = threadIdx.x;
  if (!flat) {
    if (tid < cols) rank.means[tid] = column_mean(rank.values, cols, rows, cols, tid);
    __syncthreads();
    for (int i = tid; i < rows * cols; i += kRankThreads) {
      rank.values[i] = i < cols ? rank.means[i] : 0.0f;
    }
    __syncthreads();
  }
  const int n = flat ? rows * cols : cols;
  float tile0_aggregate = 0.0f;
  for (int tile = 0; tile < kRankTiles && tile * kRankTile < n; ++tile) {
    const int base = tile * kRankTile + tid * kRankItems;
    float items[kRankItems];
    for (int i = 0; i < kRankItems; ++i) items[i] = rank.values[base + i];
    if (tile == 0) {
      // The (input, output, op, block_aggregate) overload: an output, not an initial value.
      BlockScan(rank.scan).InclusiveScan(items, items, std::plus<float>(), tile0_aggregate);
    } else {
      TilePrefix prefix{tile0_aggregate};
      BlockScan(rank.scan).InclusiveScan(items, items, std::plus<float>(), prefix);
    }
    __syncthreads();
    for (int i = 0; i < kRankItems; ++i) rank.values[base + i] = items[i];
  }
  __syncthreads();
}

// ---------------------------------------------------------------------------
// Verify width, the verify history, the sampled plan, and this round's ranked
// scores published for the next round.
// ---------------------------------------------------------------------------
__device__ int find_majority(const int32_t* nums, int n) {
  for (int i = 0; i < n; ++i) {
    int count = 0;
    for (int j = 0; j < n; ++j) count += nums[j] == nums[i];
    if (count > n / 2) return nums[i];
  }
  return -1;
}

__device__ void shift_in(int32_t* nums, int n, int value) {
  for (int i = 0; i + 1 < n; ++i) nums[i] = nums[i + 1];
  nums[n - 1] = value;
}

// A round's verify inputs: its (batch size, interval) row of the verify
// tables and its (cost data, interval) row of the interval tables.
struct VerifyRow {
  int32_t tier;  // _cost_tier
  double thresholds[2];  // threshold * 0.95, * 1.05
  int32_t num_widths;
  int32_t index[kMaxWidths];  // measured target widths
  // The verify width's cost deltas by position ([max_widths, max_widths]); 0
  // where cost_delta > 0.04 fails.
  double cost_delta[kMaxWidths * kMaxWidths];
};

// The verify width's state between its serial steps and its score comparisons.
struct VerifyPlan {
  bool compare;  // the width needs the score comparisons
  int width;     // otherwise
  int n;         // measured widths below the ranked count
  int widths[kMaxWidths];
  int last_verify_num;
  const float* ranked;  // the ranked scores it sizes on (shared memory)
  // Per position pair (i, j) and stability threshold: whether width j's
  // gain per cost over width i falls below it.
  bool below[kMaxWidths * kMaxWidths * 2];
};

// verify_width_unranked: the largest measured width below the ranked count.
__device__ int verify_width_unranked(int32_t* recent, int recent_len, const VerifyRow& v,
                                     int verify_count) {
  int best = -1;
  for (int i = 0; i < v.num_widths; ++i) {
    const int w = v.index[i];
    if (w < verify_count) best = max(best, w);
  }
  const int width = best >= 0 ? best + 1 : 1;
  shift_in(recent, recent_len, width);
  return width;
}

// The verify width on cumulative ranked scores (fp32, widened exactly), up to the
// score comparisons: sets plan.width, or plan.compare and their inputs.
// Width 1 is a measured width (and exec mode 1 measures a second one).
__device__ void verify_width_plan(const int32_t* s, int32_t* recent, int recent_len,
                                  const VerifyRow& v, int exec_mode, int verify_count,
                                  const float* ranked, int ranked_len, VerifyPlan& plan) {
  plan.last_verify_num = find_majority(recent, recent_len);
  shift_in(recent, recent_len, verify_count);
  const int num_widths = v.num_widths;
  const int32_t* index = v.index;
  if (batch_cut(s, exec_mode, s[kBatchSize])) {
    // The second-smallest measured verify width.
    plan.width = index[1] + 1;
    return;
  }
  const int limit = min(ranked_len, verify_count);
  int n = 0;
  int largest = 0;
  for (int i = 0; i < num_widths; ++i) {
    if (index[i] < limit) {
      plan.widths[n++] = index[i];
      largest = max(largest, index[i]);
    }
  }
  if (largest == 0) {
    recent[recent_len - 1] = 1;
    plan.width = 1;
    return;
  }
  plan.compare = true;
  plan.n = n;
  plan.ranked = ranked;
}

// The score comparisons, both stability thresholds of every position pair at
// once (all threads): the same operations as the serial loop, off its chain.
__device__ void verify_width_compare(const VerifyRow& v, int max_widths, VerifyPlan& plan) {
  const int n = plan.n;
  for (int p = threadIdx.x; p < n * n * 2; p += kRankThreads) {
    const int stable = p % 2;
    const int i = p / 2 / n;
    const int j = p / 2 % n;
    bool below = false;
    if (i < j) {
      const double delta = v.cost_delta[i * max_widths + j];
      below = delta > 0.0 &&
              __ddiv_rn(__dsub_rn(static_cast<double>(plan.ranked[plan.widths[j]]),
                                  static_cast<double>(plan.ranked[plan.widths[i]])),
                        delta) < v.thresholds[stable];
    }
    plan.below[p] = below;
  }
}

// The verify width after the comparisons (warp 0): the stability mask and the
// chosen width. Each width's mask entry changes only in its own column of
// the pair loop, so lane j replays column j in order.
__device__ int verify_width_choose(int32_t* recent, int recent_len, int verify_count,
                                   const VerifyPlan& plan) {
  const int lane = threadIdx.x;
  const int n = plan.n;
  int mask = INT_MIN;
  int smallest = INT_MAX;  // over widths[1:]
  if (lane < n) {
    mask = plan.widths[lane];
    const int reset = plan.widths[1];
    for (int i = 0; i < lane; ++i) {
      const int stable = plan.last_verify_num <= mask + 1 ? 0 : 1;
      if (plan.below[(i * n + lane) * 2 + stable]) mask = reset;
    }
    if (lane >= 1) smallest = plan.widths[lane];
  }
  int chosen = __reduce_max_sync(kFullMask, mask) + 1;
  smallest = __reduce_min_sync(kFullMask, smallest);
  if (lane == 0) recent[recent_len - 1] = chosen;
  if (chosen == 1 || verify_count < chosen) chosen = smallest + 1;
  return chosen;
}

struct VerifyTables {
  const int32_t* interval_i32;  // widths from kNumWidthsCol
  const double* interval_f64;   // cost deltas from cost_delta_col
  const int32_t* verify_i32;    // [max_bs, intervals]: tier
  const double* verify_f64;     // [max_bs, intervals, 2]: threshold * 0.95, * 1.05
  int num_intervals;
  int interval_i32_cols;
  int interval_f64_cols;
  int cost_delta_col;           // of the [max_widths, max_widths] cost deltas
  int max_widths;
};

// This round's ranked scores and verify width: the scores are published for
// the next round before the verify is sized, which reads the previous round's
// (its own in the first).
__global__ void __launch_bounds__(kRankThreads)
    select_verify_kernel(int32_t* state, int32_t* recent, int recent_len, int32_t* hist,
                         int hist_len, float* published, int64_t published_capacity,
                         const float* top, int64_t top_stride, VerifyTables tables,
                         int tree_depth, int exec_mode) {
  __shared__ int32_t s[kNumState];
  __shared__ int32_t recent_sh[kMaxHistory];
  __shared__ int32_t hist_row[kMaxHistory];
  __shared__ float previous[kRankMaxCols];  // the published scores a verify reads
  __shared__ RankShared rank;
  __shared__ VerifyRow v;
  __shared__ VerifyPlan plan;
  const int tid = threadIdx.x;
  // Wave 1. The verify compares scores below verify_count <= kRankMaxCols.
  const int32_t state_value = tid < kNumState ? state[tid] : 0;
  const int recent_tid = tid - kWarp;
  const int32_t recent_value = 0 <= recent_tid && recent_tid < recent_len ? recent[recent_tid] : 0;
  const int previous_tid = tid - 2 * kWarp;
  const float previous_value =
      0 <= previous_tid && previous_tid < min(static_cast<int64_t>(kRankMaxCols), published_capacity)
          ? published[previous_tid]
          : 0.0f;
  if (tid < kNumState) s[tid] = state_value;
  if (0 <= recent_tid && recent_tid < recent_len) recent_sh[recent_tid] = recent_value;
  if (0 <= previous_tid && previous_tid < kRankMaxCols) previous[previous_tid] = previous_value;
  __syncthreads();
  const int bs = s[kBatchSize];
  if (bs <= 0) return;
  const int rows = bs;
  const int verify_count = s[kVerifyCount];
  const bool flat = s[kFlat] != 0;
  const int own_count = flat ? rows * verify_count : verify_count;
  // Wave 2: the scores, the verify history and the verify inputs, all loaded
  // before any is stored.
  ScoreFetch scores;
  scores.load(top, top_stride, rows, verify_count);
  int32_t* history = hist + static_cast<int64_t>(bs - 1) * hist_len;
  const int64_t verify_row = static_cast<int64_t>(bs - 1) * tables.num_intervals + s[kInterval];
  const int64_t interval_row =
      static_cast<int64_t>(s[kCostId]) * tables.num_intervals + s[kInterval];
  const int32_t* widths_row = tables.interval_i32 + interval_row * tables.interval_i32_cols;
  const double* deltas_row =
      tables.interval_f64 + interval_row * tables.interval_f64_cols + tables.cost_delta_col;
  const int max_widths = tables.max_widths;
  const int32_t hist_value = tid < hist_len ? __ldcg(history + tid) : 0;
  const int32_t index_value = tid < max_widths ? __ldcg(widths_row + kWidthsCol + tid) : 0;
  const int32_t i32_value = tid == 0   ? __ldcg(tables.verify_i32 + verify_row)
                            : tid == 1 ? __ldcg(widths_row + kNumWidthsCol)
                                       : 0;
  const double f64_value = tid < 2 ? __ldcg(tables.verify_f64 + verify_row * 2 + tid) : 0.0;
  Fetch<ceil_div(kMaxWidths * kMaxWidths, kRankThreads), double> deltas;
  deltas.load(max_widths * max_widths, tid, kRankThreads,
              [=](int i) { return __ldcg(deltas_row + i); });
  scores.store(rank, rows, verify_count);
  if (tid < hist_len) hist_row[tid] = hist_value;
  if (tid < max_widths) v.index[tid] = index_value;
  if (tid == 0) v.tier = i32_value;
  if (tid == 1) v.num_widths = i32_value;
  if (tid < 2) v.thresholds[tid] = f64_value;
  deltas.store(v.cost_delta, max_widths * max_widths, tid, kRankThreads);
  __syncthreads();
  rank_scores(rank, rows, verify_count, flat);
  for (int i = tid; i < own_count; i += kRankThreads) published[i] = rank.values[i];
  if (tid == 0) {
    // The verify cost row sets the tier.
    s[kTier] = v.tier;
    const int previous_count = s[kRankedCount];
    plan.compare = false;
    plan.width = 0;
    if (previous_count > 0) {
      verify_width_plan(s, recent_sh, recent_len, v, exec_mode, verify_count, previous,
                        previous_count, plan);
    } else if (s[kSampled]) {
      // A sampled round's own scores depend on its draws.
      plan.width = verify_width_unranked(recent_sh, recent_len, v, verify_count);
    } else {
      // The first round decides on its own scores.
      verify_width_plan(s, recent_sh, recent_len, v, exec_mode, verify_count, rank.values,
                        own_count, plan);
    }
  }
  __syncthreads();
  int width = plan.width;
  if (plan.compare) {
    verify_width_compare(v, max_widths, plan);
    __syncthreads();
    if (tid < kWarp) width = verify_width_choose(recent_sh, recent_len, verify_count, plan);
  }
  if (tid == 0) {
    s[kVerifyNum] = width;
    // The history shifts in the write-back below.
    if (s[kSampled]) {
      // The next sampled round's planned stop depth.
      s[kSampledNext] = s[kSampledSeen] >= 0 ? s[kSampledSeen]
                                             : min(s[kSampledPlan] + 2, tree_depth - 1);
    }
    s[kRankedCount] = own_count;
  }
  __syncthreads();
  if (tid < kNumState && s[tid] != state_value) state[tid] = s[tid];
  if (0 <= recent_tid && recent_tid < recent_len && recent_sh[recent_tid] != recent_value) {
    recent[recent_tid] = recent_sh[recent_tid];
  }
  // The history shifted by one, the width last.
  if (tid < hist_len) {
    history[tid] = tid + 1 < hist_len ? hist_row[tid + 1] : s[kVerifyNum];
  }
}

// ---------------------------------------------------------------------------
// Host entry points
// ---------------------------------------------------------------------------
// The checks below cover what the kernels index past their fixed capacities
// or read through a caller's view; device_policy.py builds every table,
// state and slot tensor contiguous in the kernels' dtypes and layouts.

// tree_depth from the [4, tree_depth + 1] slots, within kMaxDepth.
int round_depth(const TensorView& slots) {
  TVM_FFI_ICHECK(slots.size(1) >= 2 && slots.size(1) - 1 <= kMaxDepth)
      << "eagle_harness_policy: tree depth past " << kMaxDepth;
  return static_cast<int>(slots.size(1)) - 1;
}

// A decision's depth and its work deltas: contiguous [n].
void check_decision(int64_t depth, int tree_depth, const TensorView& work_deltas, int64_t n) {
  TVM_FFI_ICHECK(depth >= 1 && depth < tree_depth) << "eagle_harness_policy: decision depth";
  TVM_FFI_ICHECK(work_deltas.size(0) == n && work_deltas.stride(0) == 1)
      << "eagle_harness_policy: work deltas must be contiguous [" << n << "]";
}

}  // namespace

void eagle_harness_policy_begin_round(TensorView state, TensorView seq, TensorView hist,
                                      TensorView slots, TensorView go_host,
                                      TensorView batch_table, TensorView tier_table,
                                      TensorView cost_f64, TensorView round_f64,
                                      TensorView live_rows, TensorView frontier,
                                      TensorView sampled, bool chain, int64_t node_top_k,
                                      int64_t chain_past_verify_num, int64_t verify_num,
                                      int64_t min_verify_count) {
  const int tree_depth = round_depth(slots);
  TVM_FFI_ICHECK_LE(hist.size(1), kMaxHistory) << "eagle_harness_policy: verify history";
  CostTable costs;
  costs.f64 = static_cast<const double*>(cost_f64.data_ptr());
  costs.num_intervals = static_cast<int>(cost_f64.size(1));
  costs.cols = chain ? 0 : static_cast<int>(cost_f64.size(2));
  begin_round_kernel<<<1, kWarp, 0, get_stream(state.device())>>>(
      static_cast<int32_t*>(state.data_ptr()), static_cast<int32_t*>(slots.data_ptr()),
      static_cast<int64_t*>(go_host.data_ptr()), static_cast<int64_t*>(seq.data_ptr()),
      tree_depth, static_cast<int>(node_top_k), chain,
      static_cast<const int32_t*>(hist.data_ptr()), static_cast<int>(hist.size(1)),
      static_cast<const int64_t*>(batch_table.data_ptr()),
      static_cast<const int32_t*>(tier_table.data_ptr()), static_cast<int>(tier_table.size(0)),
      costs, static_cast<double*>(round_f64.data_ptr()),
      static_cast<const int32_t*>(live_rows.data_ptr()),
      static_cast<const int64_t*>(frontier.data_ptr()), frontier.stride(0),
      static_cast<const bool*>(sampled.data_ptr()), static_cast<int>(chain_past_verify_num),
      static_cast<int>(verify_num), static_cast<int>(min_verify_count));
}

void eagle_harness_policy_decide_tree(TensorView state, TensorView seq, TensorView slots,
                                      TensorView go_host, TensorView work_deltas,
                                      TensorView round_f64, TensorView tier_table, int64_t depth,
                                      int64_t node_top_k, int64_t exec_mode,
                                      double async_layer_ratio, int64_t verify_num,
                                      int64_t min_verify_count) {
  const int tree_depth = round_depth(slots);
  check_decision(depth, tree_depth, work_deltas, node_top_k + 1);
  decide_kernel<false><<<1, kWarp, 0, get_stream(state.device())>>>(
      static_cast<int32_t*>(state.data_ptr()), static_cast<int32_t*>(slots.data_ptr()),
      static_cast<int64_t*>(go_host.data_ptr()), static_cast<const int64_t*>(seq.data_ptr()),
      tree_depth, static_cast<int>(node_top_k), static_cast<const float*>(work_deltas.data_ptr()),
      static_cast<const double*>(round_f64.data_ptr()),
      static_cast<const int32_t*>(tier_table.data_ptr()), static_cast<int>(depth),
      static_cast<int>(exec_mode), async_layer_ratio, static_cast<int>(verify_num),
      static_cast<int>(min_verify_count));
}

void eagle_harness_policy_decide_chain(TensorView state, TensorView seq, TensorView slots,
                                       TensorView go_host, TensorView work_deltas,
                                       TensorView round_f64, int64_t depth, int64_t verify_num) {
  const int tree_depth = round_depth(slots);
  check_decision(depth, tree_depth, work_deltas, 2);
  decide_kernel<true><<<1, kWarp, 0, get_stream(state.device())>>>(
      static_cast<int32_t*>(state.data_ptr()), static_cast<int32_t*>(slots.data_ptr()),
      static_cast<int64_t*>(go_host.data_ptr()), static_cast<const int64_t*>(seq.data_ptr()),
      tree_depth, 1, static_cast<const float*>(work_deltas.data_ptr()),
      static_cast<const double*>(round_f64.data_ptr()), nullptr, static_cast<int>(depth), 0, 0.0,
      static_cast<int>(verify_num), 0);
}

void eagle_harness_policy_select_verify(TensorView state, TensorView recent, TensorView hist,
                                        TensorView published, TensorView top_scores,
                                        TensorView interval_i32, TensorView interval_f64,
                                        TensorView verify_i32, TensorView verify_f64,
                                        int64_t tree_depth, int64_t exec_mode) {
  TVM_FFI_ICHECK(hist.size(1) <= kMaxHistory && recent.size(0) <= kMaxHistory)
      << "eagle_harness_policy: verify history";
  // Live rows up to the batch, verify counts up to the columns.
  TVM_FFI_ICHECK(hist.size(0) <= kRankMaxRows && top_scores.size(1) <= kRankMaxCols &&
                 top_scores.stride(1) == 1)
      << "eagle_harness_policy: ranked scores past the replicated torch shapes";
  VerifyTables tables;
  tables.max_widths = static_cast<int>(interval_i32.size(2)) - kWidthsCol;
  tables.cost_delta_col = kStopThresholdCol + kVariants * static_cast<int>(tree_depth);
  TVM_FFI_ICHECK_LE(tables.max_widths, kMaxWidths) << "eagle_harness_policy: measured widths";
  tables.interval_i32 = static_cast<const int32_t*>(interval_i32.data_ptr());
  tables.interval_f64 = static_cast<const double*>(interval_f64.data_ptr());
  tables.verify_i32 = static_cast<const int32_t*>(verify_i32.data_ptr());
  tables.verify_f64 = static_cast<const double*>(verify_f64.data_ptr());
  tables.num_intervals = static_cast<int>(interval_i32.size(1));
  tables.interval_i32_cols = static_cast<int>(interval_i32.size(2));
  tables.interval_f64_cols = static_cast<int>(interval_f64.size(2));
  select_verify_kernel<<<1, kRankThreads, 0, get_stream(state.device())>>>(
      static_cast<int32_t*>(state.data_ptr()), static_cast<int32_t*>(recent.data_ptr()),
      static_cast<int>(recent.size(0)), static_cast<int32_t*>(hist.data_ptr()),
      static_cast<int>(hist.size(1)), static_cast<float*>(published.data_ptr()),
      published.size(0), static_cast<const float*>(top_scores.data_ptr()),
      top_scores.stride(0), tables, static_cast<int>(tree_depth), static_cast<int>(exec_mode));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(eagle_harness_policy_begin_round, eagle_harness_policy_begin_round);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(eagle_harness_policy_decide_tree, eagle_harness_policy_decide_tree);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(eagle_harness_policy_decide_chain,
                              eagle_harness_policy_decide_chain);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(eagle_harness_policy_select_verify,
                              eagle_harness_policy_select_verify);

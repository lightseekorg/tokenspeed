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

// EAGLE-4 round decisions on the device, in device_policy.py's arithmetic, without host sync.

#include <cub/block/block_scan.cuh>

#include <climits>
#include <cstdint>
#include <functional>

#include "tvm_ffi_utils.h"

namespace {

// Scalars of the int32 state tensor: the policy's persistent state, then the current round's.
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

// Columns of batch_table ([max_bs, cols], int64): begin_round's inputs per batch size.
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
// Columns of interval_f64: draft_cost[-1], cost ratio, thresholds[variant][depth], cost deltas.
enum IntervalF64Col : int { kDraftCostCol = 0, kModelCostRatioCol, kStopThresholdCol };
// chain_f64 [costs, intervals, 2 tier rows, depth, entries]; cliff entries are 0 without a cliff.
enum ChainF64 : int { kGainDenom = 0, kStepCost, kChainThreshold, kCliffGainDenom, kCliffCost,
                      kChainF64 };
// round_f64 (begin_round's copy): a tree's draft cost, cost ratio, thresholds; a chain's chain_f64.
enum RoundF64 : int { kRoundDraftCost = 0, kRoundModelCostRatio, kRoundThresholds };

constexpr int kWarp = 32;
constexpr unsigned kFullMask = 0xffffffffu;

// torch.cumsum's CUB scan tiles (128 threads x 15 items); two tiles keep the look-back's sum order.
constexpr int kRankThreads = 128;
constexpr int kRankItems = 15;
constexpr int kRankTile = kRankThreads * kRankItems;
constexpr int kRankTiles = 2;
// The shapes whose torch.mean(dim=0) reduction order the ranking replicates.
constexpr int kRankMaxRows = 32;
constexpr int kRankMaxCols = 64;

using BlockScan = cub::BlockScan<float, kRankThreads, cub::BLOCK_SCAN_WARP_SCANS>;

constexpr int ceil_div(int a, int b) { return (a + b - 1) / b; }

// One lane (one-warp kernels) or thread (select_verify's last two warps) per entry.
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

// Register staging (entries tid + k * threads): a wave loads all before storing any, to overlap.
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

// A round kernel's state and slots in shared memory: depth d's (cur) and depth d + 1's (next).
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

  // Depth d + 1's go flag, mirrored to the host unfenced: the host reads nothing else on it.
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

  // Depth d's decision when it needs no work deltas (narrow: the tier's width); else not decided.
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

  // Stops the round after depth d: verify count, flat flag (a chain verifies all ranked), sampled plan.
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
    if (s[kSampled]) {
      // The next sampled round's plan: the first depth whose rule said stop, else two deeper.
      s[kSampledNext] = s[kSampledSeen] >= 0 ? s[kSampledSeen]
                                             : min(s[kSampledPlan] + 2, tree_depth - 1);
    }
  }

  // Takes depth d's decision; a sampled round stops only at its planned depth.
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

// A decision kernel's one load wave: unconditional loads at clamped indices, issued back to back.
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

// Writes back a decision kernel's changed state scalars and depth d + 1's slots (one warp).
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

// Round start (finish_round): past_verify_num, sampled plan, cost interval, depth 0's decision.
struct BeginInputs {
  int bs;
  bool sampled;
  int64_t frontier_sum;      // of the first bs cached lengths
  const int64_t* batch_row;  // batch_table's row of bs
  int hist_max;              // trees: the largest past verify width of bs
  int chain_past_verify_num;
};

// Width-W table round: depths 1..W-2 draft, all W nodes verify, stop fields as the rule records.
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

// Table round slot of depth d: depth 1 always runs, so a stop at depth 0 keeps an unranked depth 1.
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
  // Fixed at round start: tiny and tier change only after the round's first cost read.
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

// begin_round's cost row source: interval_f64 for trees, chain_f64 for chains.
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
  // Go flags through last drafted depth + 1, at least depth 2 (the drafter reads it after depth 1).
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

// Tree width decision before depth d drafts; row: draft cost, cost ratio, depth d's stop threshold.
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
      // (work_deltas[narrow] - work_deltas[0]) / draft_cost * ratios, one rounding per op, no FMA.
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
    // Table round: begin_round wrote the slots; past its last depth, only clear the next go flag.
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

// Cumulative ranked scores bit for bit as torch.cumsum(top_scores.mean(dim=0)) (flat: view(-1)).
__device__ __forceinline__ int last_pow2(int n) {
  int p = 1;
  while (p * 2 <= n) p *= 2;
  return p;
}

// Replays Reduce.cuh's mean(dim=0) summation order for rows <= 32, cols <= 64 (cols == 1 differs).
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

// Second tile's block prefix: the first tile's aggregate, as CUB's decoupled look-back passes it.
struct TilePrefix {
  float prefix;
  __device__ float operator()(float) const { return prefix; }
};

// Ranking scratch: [rows, cols] scores row-major, zero-padded to two tiles, scanned in place.
struct RankShared {
  float values[kRankTile * kRankTiles];
  float means[kRankMaxCols];
  typename BlockScan::TempStorage scan;
};

// top[:rows, :cols] to registers, then to rank.values zero-padded; all kRankThreads threads.
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

// Cumulative ranked scores in place in rank.values; all threads, after ScoreFetch::store + barrier.
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

// Verify width and history, and this round's ranked scores for the next round.
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

// A round's verify inputs: its (batch size, interval) and (cost data, interval) table rows.
struct VerifyRow {
  int32_t tier;  // _cost_tier
  double thresholds[2];  // threshold * 0.95, * 1.05
  int32_t num_widths;
  int32_t index[kMaxWidths];  // measured target widths
  // Cost deltas by position pair [max_widths, max_widths]; 0 where cost_delta > 0.04 fails.
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
  // Per pair (i, j) and stability threshold: width j's gain per cost over width i is below it.
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

// Verify width on fp32 ranked scores up to the comparisons: sets plan.width or plan.compare.
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

// All pairs' comparisons at both thresholds in parallel: the serial loop's ops, off its chain.
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

// Mask and chosen width (warp 0): width j's mask changes only in column j, so lane j replays it.
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

// Publishes this round's ranked scores, then sizes the verify on the previous round's (first: own).
__global__ void __launch_bounds__(kRankThreads)
    select_verify_kernel(int32_t* state, int32_t* recent, int recent_len, int32_t* hist,
                         int hist_len, float* published, int64_t published_capacity,
                         const float* top, int64_t top_stride, VerifyTables tables,
                         int exec_mode) {
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
  // Wave 2: scores, verify history and verify inputs, all loaded before any is stored.
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

// Host entry points check fixed capacities and views; device_policy.py builds contiguous tables.

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
      top_scores.stride(0), tables, static_cast<int>(exec_mode));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(eagle_harness_policy_begin_round, eagle_harness_policy_begin_round);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(eagle_harness_policy_decide_tree, eagle_harness_policy_decide_tree);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(eagle_harness_policy_decide_chain,
                              eagle_harness_policy_decide_chain);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(eagle_harness_policy_select_verify,
                              eagle_harness_policy_select_verify);

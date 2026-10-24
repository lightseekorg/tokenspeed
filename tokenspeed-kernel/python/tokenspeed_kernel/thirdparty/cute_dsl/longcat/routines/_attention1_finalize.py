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

"""Attention1 query RoPE and KV-cache scatter, hosted by the caller's engine.

Absorbed-query MMA writes NOPE512; this routine rotates/writes RoPE64 and
scatters normalized latent rows. Default tasks are 8*local_heads query tails
plus eight KV rows; query_per_token uses eight query tasks instead.
Query/cache dtype is FP8 E4M3 (576-byte rows) or BF16 (1152-byte rows).
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Uint32
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.routines._collective_primitives import (
    fence_proxy_async_global,
)
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.routines._primitives import (
    load_acquire_gpu_u32,
    red_add_release_gpu_u32,
)

_TOKENS = 8
_HEADS = 8
_KV_LORA = 512
_ROPE_DIM = 64
_ROPE_HALF = _ROPE_DIM // 2
_Q_HEAD_DIM = 192
_QKV_KV_OFFSET = 1536
_QKV_ROPE_OFFSET = 2048
_QUERY_TASKS = 64
_KV_TASKS = 8
_ACTIVE_TASKS = _QUERY_TASKS + _KV_TASKS
_KV_NORM_TASKS = 8
_QKV_OUTPUT_TILES = 44
_Q_B_TILES_PER_HEAD = 4
_Q_B_ROPE_TILE = 3
_ABSORBED_TILES_PER_HEAD = 8
_THREADS = 192
_SYNC_BARRIER_ID = 15


class LongcatAttention1Finalize:
    """Assemble query tails and scatter latent-cache rows.

    Args:
        local_heads: Positive head count; default task count is 8*local_heads+8.
        kv_dtype: Query/cache type, Float8E4M3FN (576 B/row) or BFloat16 (1152 B).
        kv_norm_in_epilogue: Normalize raw KV latent columns using producer row
            sums; otherwise consume pre-normalized columns.
        eps: KV RMSNorm epsilon for epilogue normalization.
        qkv_output_tiles: Producer QKV-A arrivals per generation.
        wait_absorbed: Acquire absorbed-query completion before publishing the
            query task. If False, wait only for Q-B RoPE; the consumer must also
            acquire the separate absorbed-query counter.
        query_per_token: Cover all heads per token in eight query tasks;
            requires wait_absorbed=False.
        rope_ready_per_head: One complete-RoPE ready word per head instead of
            four Q-B tile words; publish only after all 64 RoPE channels store.
        rope_interleaved: Pair (2i,2i+1); otherwise pair (c,c+32). Both use
            cos_sin rows [cos[0:32],sin[0:32]].

    Raises:
        ValueError: Invalid head count, dtype or QKV arrival count.
    """

    threads_per_cta = _THREADS

    def __init__(
        self,
        *,
        local_heads: int = 8,
        kv_dtype=cutlass.Float8E4M3FN,
        kv_norm_in_epilogue: bool = False,
        eps: float = 1e-5,
        qkv_output_tiles: int = _QKV_OUTPUT_TILES,
        rope_interleaved: bool = False,
        wait_absorbed: bool = True,
        query_per_token: bool = False,
        rope_ready_per_head: bool = False,
    ) -> None:
        if not isinstance(local_heads, int) or local_heads < 1:
            raise ValueError("local_heads must be a positive integer")
        if kv_dtype is not cutlass.Float8E4M3FN and kv_dtype is not cutlass.BFloat16:
            raise ValueError(
                "kv_dtype must be cutlass.Float8E4M3FN or cutlass.BFloat16"
            )
        self.local_heads = local_heads
        self.kv_dtype = kv_dtype
        self.kv_is_bf16 = kv_dtype is cutlass.BFloat16
        self.kv_norm_in_epilogue = bool(kv_norm_in_epilogue)
        self.eps = float(eps)
        if not isinstance(qkv_output_tiles, int) or qkv_output_tiles < 1:
            raise ValueError("qkv_output_tiles must be a positive integer")
        self.qkv_output_tiles = qkv_output_tiles
        self.rope_interleaved = bool(rope_interleaved)
        self.wait_absorbed = bool(wait_absorbed)
        if query_per_token and wait_absorbed:
            raise ValueError("query_per_token needs wait_absorbed=False")
        self.query_per_token = bool(query_per_token)
        self.rope_ready_per_head = bool(rope_ready_per_head)
        self.query_task_count = _TOKENS if query_per_token else local_heads * _TOKENS
        self.active_task_count = self.query_task_count + _KV_TASKS

    @cute.jit
    def finalize_device_routine(
        self,
        q_b_output: cute.Tensor,
        qkv_output: cute.Tensor,
        positions: cute.Tensor,
        cos_sin_cache: cute.Tensor,
        cache_loc: cute.Tensor,
        query_fp8: cute.Tensor,
        kv_cache: cute.Tensor,
        generation: cute.Tensor,
        qkv_completed: cute.Tensor,
        kv_norm_completed: cute.Tensor,
        q_b_tile_ready: cute.Tensor,
        absorbed_query_tile_ready: cute.Tensor,
        attention1_prepare_completed: cute.Tensor,
        kv_row_sumsq: cute.Tensor = None,
        kv_gamma: cute.Tensor = None,
        *,
        logical_worker_id=None,
        routine_cta_rank=None,
        barrier_threads: cutlass.Constexpr = 192,
        wait_ready: cutlass.Constexpr = True,
        tail_cluster_sync: cutlass.Constexpr = True,
    ) -> None:
        """Acquire producer readiness and publish query/KV task completion.

        Args:
            wait_ready: Acquire producer tickets. False requires prior producer
                ordering by the host and is valid only for query tasks.
            tail_cluster_sync: Join the routine's final cluster barrier; False when
                the caller supplies it.
            query_fp8: [tokens,local_heads,576] output, FP8 or BF16 as kv_dtype selects.
            kv_norm_completed: Ignored when KV norm runs in the epilogue.
            kv_row_sumsq: FP32[2,8]; slot 1 (8+token) holds the sum of squares of
                BF16 latent columns [0,512). Required for epilogue KV normalization.
            kv_gamma: BF16[512] weights, required for epilogue KV normalization.
            logical_worker_id: Optional task-group override; task=id*2+routine_cta_rank.
            routine_cta_rank: Optional 0/1 override of block_idx_in_cluster().
            barrier_threads: Participants in routine named barrier 15, e.g. 192 for
                hosted W0-5. Other CTA warps must not be included in this count.
        """
        tidx, _, _ = cute.arch.thread_idx()
        _, bidy, bidz = cute.arch.block_idx()
        _, grid_dim_y, _ = cute.arch.grid_dim()
        if cutlass.const_expr(logical_worker_id is None):
            physical_cluster_id = cutlass.Int32(bidy + grid_dim_y * bidz)
            cta_rank = cute.arch.block_idx_in_cluster()
        else:
            physical_cluster_id = logical_worker_id
            cta_rank = routine_cta_rank
        task = physical_cluster_id * 2 + cta_rank

        if task < self.active_task_count:
            if cutlass.const_expr(wait_ready):
                if tidx == 0:
                    generation_value = load_acquire_gpu_u32(generation.iterator)
                    if task < self.query_task_count:
                        for head_i in cutlass.range_constexpr(
                            self.local_heads if self.query_per_token else 1
                        ):
                            head = task % self.local_heads
                            if cutlass.const_expr(self.query_per_token):
                                head = cutlass.Int32(head_i)
                            rope_tile = head * _Q_B_TILES_PER_HEAD + _Q_B_ROPE_TILE
                            if cutlass.const_expr(self.rope_ready_per_head):
                                rope_tile = head
                            completed = load_acquire_gpu_u32(
                                q_b_tile_ready.iterator + rope_tile
                            )
                            while (completed - generation_value).to(Int32) < Int32(0):
                                completed = load_acquire_gpu_u32(
                                    q_b_tile_ready.iterator + rope_tile
                                )
                        for head_tile in cutlass.range_constexpr(
                            _ABSORBED_TILES_PER_HEAD if self.wait_absorbed else 0
                        ):
                            absorbed_tile = head * _ABSORBED_TILES_PER_HEAD + head_tile
                            completed = load_acquire_gpu_u32(
                                absorbed_query_tile_ready.iterator + absorbed_tile
                            )
                            while (completed - generation_value).to(Int32) < Int32(0):
                                completed = load_acquire_gpu_u32(
                                    absorbed_query_tile_ready.iterator + absorbed_tile
                                )
                    else:
                        if cutlass.const_expr(not self.kv_norm_in_epilogue):
                            kv_target = generation_value * Uint32(_KV_NORM_TASKS)
                            completed = load_acquire_gpu_u32(kv_norm_completed.iterator)
                            while (completed - kv_target).to(Int32) < Int32(0):
                                completed = load_acquire_gpu_u32(
                                    kv_norm_completed.iterator
                                )
                        # KV writes also need raw k_pe outside the norm range.
                        # Full QKV completion covers it and, with epilogue norm,
                        # the producer's row sum-of-squares reductions.
                        qkv_target = generation_value * Uint32(self.qkv_output_tiles)
                        completed = load_acquire_gpu_u32(qkv_completed.iterator)
                        while (completed - qkv_target).to(Int32) < Int32(0):
                            completed = load_acquire_gpu_u32(qkv_completed.iterator)
            cute.arch.barrier(
                barrier_id=_SYNC_BARRIER_ID,
                number_of_threads=barrier_threads,
            )

            if task < self.query_task_count:
                # One (head, rope channel) item per thread per round: a
                # per-(token, head) task owns one item per thread, a
                # per-token task walks all ``local_heads * 64`` of its token.
                # Issue every load before the first store so potentially
                # aliasing stores do not serialize the position/cos/sin loads.
                items = cutlass.const_expr(
                    self.local_heads * _ROPE_DIM if self.query_per_token else _ROPE_DIM
                )
                rounds = cutlass.const_expr(-(-items // barrier_threads))
                token = task // self.local_heads
                if cutlass.const_expr(self.query_per_token):
                    token = task
                position = positions[token]
                r_q1 = cute.make_rmem_tensor((rounds,), Float32)
                r_q2 = cute.make_rmem_tensor((rounds,), Float32)
                r_cos = cute.make_rmem_tensor((rounds,), Float32)
                r_sin = cute.make_rmem_tensor((rounds,), Float32)
                for r in cutlass.range_constexpr(rounds):
                    item = tidx + r * barrier_threads
                    if item < items:
                        head = task % self.local_heads
                        if cutlass.const_expr(self.query_per_token):
                            head = item // _ROPE_DIM
                        rope_channel = item % _ROPE_DIM
                        if cutlass.const_expr(self.rope_interleaved):
                            angle = rope_channel // 2
                            low_channel = angle * 2
                            r_cos[r] = Float32(cos_sin_cache[position, angle])
                            r_sin[r] = Float32(
                                cos_sin_cache[position, _ROPE_HALF + angle]
                            )
                            r_q1[r] = Float32(
                                q_b_output[
                                    token, head, _Q_HEAD_DIM - _ROPE_DIM + low_channel
                                ]
                            )
                            r_q2[r] = Float32(
                                q_b_output[
                                    token,
                                    head,
                                    _Q_HEAD_DIM - _ROPE_DIM + low_channel + 1,
                                ]
                            )
                        else:
                            pair = rope_channel
                            if rope_channel >= _ROPE_HALF:
                                pair = rope_channel - _ROPE_HALF
                            r_cos[r] = Float32(cos_sin_cache[position, pair])
                            r_sin[r] = Float32(
                                cos_sin_cache[position, _ROPE_HALF + pair]
                            )
                            r_q1[r] = Float32(
                                q_b_output[token, head, _Q_HEAD_DIM - _ROPE_DIM + pair]
                            )
                            r_q2[r] = Float32(
                                q_b_output[token, head, _Q_HEAD_DIM - _ROPE_HALF + pair]
                            )
                for r in cutlass.range_constexpr(rounds):
                    item = tidx + r * barrier_threads
                    if item < items:
                        head = task % self.local_heads
                        if cutlass.const_expr(self.query_per_token):
                            head = item // _ROPE_DIM
                        rope_channel = item % _ROPE_DIM
                        q1, q2 = r_q1[r], r_q2[r]
                        cos_value, sin_value = r_cos[r], r_sin[r]
                        value = q1 * cos_value - q2 * sin_value
                        if cutlass.const_expr(self.rope_interleaved):
                            if rope_channel % 2 != 0:
                                value = q2 * cos_value + q1 * sin_value
                        else:
                            if rope_channel >= _ROPE_HALF:
                                value = q2 * cos_value + q1 * sin_value
                        if cutlass.const_expr(self.kv_is_bf16):
                            query_fp8[token, head, _KV_LORA + rope_channel] = BFloat16(
                                value
                            )
                        else:
                            query_fp8[token, head, _KV_LORA + rope_channel] = (
                                cutlass.Float8E4M3FN(BFloat16(value))
                            )
            else:
                token = task - self.query_task_count
                cache_row = cache_loc[token]
                position = positions[token]
                if cutlass.const_expr(self.kv_norm_in_epilogue):
                    # Every thread recomputes the same row scale; the source
                    # element is published by the acquire load above.
                    row_sumsq = Float32(kv_row_sumsq[_TOKENS + token])
                    inv_rms = cute.math.rsqrt(
                        row_sumsq / Float32(_KV_LORA) + Float32(self.eps),
                        fastmath=True,
                    )
                # As above: every round's loads before the first store.
                r_v1 = cute.make_rmem_tensor((3,), Float32)
                r_v2 = cute.make_rmem_tensor((3,), Float32)
                r_cos = cute.make_rmem_tensor((3,), Float32)
                r_sin = cute.make_rmem_tensor((3,), Float32)
                for output_round in cutlass.range_constexpr(3):
                    channel = tidx + output_round * _THREADS
                    if channel < _KV_LORA + _ROPE_DIM:
                        if channel < _KV_LORA:
                            r_v1[output_round] = Float32(
                                qkv_output[token, _QKV_KV_OFFSET + channel]
                            )
                            if cutlass.const_expr(self.kv_norm_in_epilogue):
                                r_v2[output_round] = Float32(kv_gamma[channel])
                        else:
                            rope_channel = channel - _KV_LORA
                            if cutlass.const_expr(self.rope_interleaved):
                                angle = rope_channel // 2
                                low_channel = angle * 2
                                r_cos[output_round] = Float32(
                                    cos_sin_cache[position, angle]
                                )
                                r_sin[output_round] = Float32(
                                    cos_sin_cache[position, _ROPE_HALF + angle]
                                )
                                r_v1[output_round] = Float32(
                                    qkv_output[token, _QKV_ROPE_OFFSET + low_channel]
                                )
                                r_v2[output_round] = Float32(
                                    qkv_output[
                                        token, _QKV_ROPE_OFFSET + low_channel + 1
                                    ]
                                )
                            else:
                                pair = rope_channel
                                if rope_channel >= _ROPE_HALF:
                                    pair = rope_channel - _ROPE_HALF
                                r_cos[output_round] = Float32(
                                    cos_sin_cache[position, pair]
                                )
                                r_sin[output_round] = Float32(
                                    cos_sin_cache[position, _ROPE_HALF + pair]
                                )
                                r_v1[output_round] = Float32(
                                    qkv_output[token, _QKV_ROPE_OFFSET + pair]
                                )
                                r_v2[output_round] = Float32(
                                    qkv_output[
                                        token,
                                        _QKV_ROPE_OFFSET + _ROPE_HALF + pair,
                                    ]
                                )
                for output_round in cutlass.range_constexpr(3):
                    channel = tidx + output_round * _THREADS
                    if channel < _KV_LORA + _ROPE_DIM:
                        value = Float32(0.0)
                        if channel < _KV_LORA:
                            value = r_v1[output_round]
                            if cutlass.const_expr(self.kv_norm_in_epilogue):
                                value = Float32(
                                    BFloat16(value * inv_rms * r_v2[output_round])
                                )
                        else:
                            rope_channel = channel - _KV_LORA
                            k1, k2 = r_v1[output_round], r_v2[output_round]
                            cos_value = r_cos[output_round]
                            sin_value = r_sin[output_round]
                            value = k1 * cos_value - k2 * sin_value
                            if cutlass.const_expr(self.rope_interleaved):
                                if rope_channel % 2 != 0:
                                    value = k2 * cos_value + k1 * sin_value
                            else:
                                if rope_channel >= _ROPE_HALF:
                                    value = k2 * cos_value + k1 * sin_value
                        if cutlass.const_expr(self.kv_is_bf16):
                            kv_cache[cache_row, channel] = BFloat16(value)
                        else:
                            kv_cache[cache_row, channel] = cutlass.Float8E4M3FN(
                                BFloat16(value)
                            )

            # Writer-side proxy fence: the query / KV rows stored above are
            # read by cp.async.bulk (async proxy) in the engine's P5, so each
            # writing thread orders its generic stores before the publish.
            fence_proxy_async_global()
            cute.arch.barrier(
                barrier_id=_SYNC_BARRIER_ID,
                number_of_threads=barrier_threads,
            )
            if tidx == 0:
                # Non-returning: nothing here needs the old count.
                red_add_release_gpu_u32(
                    attention1_prepare_completed.iterator, Uint32(1)
                )

        # The parent may repurpose every cluster after this family.
        if cutlass.const_expr(tail_cluster_sync):
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()


__all__ = ["LongcatAttention1Finalize"]

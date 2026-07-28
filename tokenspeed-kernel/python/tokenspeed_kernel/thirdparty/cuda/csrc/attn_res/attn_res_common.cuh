/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Packed Blackwell forward Attention-Residual kernel from the NVIDIA DevTech
 * Kimi-K3 / KDA optimization work. Vendored unmodified -- do not edit here.
 */
#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>

using bf16_t = __nv_bfloat16;

static constexpr int ATTN_RES_BLOCK = 256;
static constexpr int ATTN_RES_WARPS = ATTN_RES_BLOCK / 32;

__inline__ __device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

__inline__ __device__ float block_reduce_sum(float val, float* ws) {
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;
    val = warp_reduce_sum(val);
    if (lane == 0) ws[wid] = val;
    __syncthreads();
    val = (threadIdx.x < ATTN_RES_WARPS) ? ws[threadIdx.x] : 0.f;
    if (wid == 0) val = warp_reduce_sum(val);
    return val;
}

__device__ __forceinline__
const bf16_t* v_addr(const bf16_t* block_res, const bf16_t* layer_res,
                     int n, int N, int t, int b, int T, int B, int H) {
    if (n < N - 1)
        return block_res + (((long long)n * T + t) * B + b) * H;
    return layer_res + ((long long)t * B + b) * H;
}

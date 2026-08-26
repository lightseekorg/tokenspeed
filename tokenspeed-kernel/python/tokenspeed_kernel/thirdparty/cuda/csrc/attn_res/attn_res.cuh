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

#include <cuda_bf16.h>
#include <cuda_runtime.h>

using bf16_t = __nv_bfloat16;

int attn_res_fwd_grid_size(int dev);

void run_attn_res_fwd_online_v2(
    const bf16_t* block_residual,
    bf16_t* layer_residual,
    const bf16_t* delta,
    const bf16_t* res_weight,
    const bf16_t* rms_weight,
    const bf16_t* out_norm_weight,
    bf16_t* output,
    int N,
    int T,
    int block_stride_m,
    int block_stride_r,
    float rms_eps,
    int num_sm,
    cudaStream_t stream,
    bool enable_pdl);

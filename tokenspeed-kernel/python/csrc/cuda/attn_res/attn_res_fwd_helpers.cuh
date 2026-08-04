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
#include <cute/tensor.hpp>

namespace sm100 {

CUTE_DEVICE
void tcgen05_after_thread_sync() {
    asm volatile("tcgen05.fence::after_thread_sync;");
}

CUTE_DEVICE
void umma_arrive_noelect(uint64_t& bar_ptr) {
    uint64_t bar_addr = cute::cast_smem_ptr_to_uint(&bar_ptr);
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
        :
        : "l"(bar_addr));
}

CUTE_DEVICE
float2 float2_sub(const float2& a, const float2& b) {
    float2 c;
    asm volatile(
        "sub.f32x2 %0, %1, %2;\n"
        : "=l"(reinterpret_cast<uint64_t&>(c))
        : "l"(reinterpret_cast<uint64_t const&>(a)),
          "l"(reinterpret_cast<uint64_t const&>(b)));
    return c;
}

CUTE_DEVICE
float2 float2_mul(const float2& a, const float2& b) {
    float2 c;
    asm volatile(
        "mul.f32x2 %0, %1, %2;\n"
        : "=l"(reinterpret_cast<uint64_t&>(c))
        : "l"(reinterpret_cast<uint64_t const&>(a)),
          "l"(reinterpret_cast<uint64_t const&>(b)));
    return c;
}

CUTE_DEVICE
float2 float2_fma(const float2& a, const float2& b, const float2& c) {
    float2 d;
    asm volatile(
        "fma.rn.f32x2 %0, %1, %2, %3;\n"
        : "=l"(reinterpret_cast<uint64_t&>(d))
        : "l"(reinterpret_cast<uint64_t const&>(a)),
          "l"(reinterpret_cast<uint64_t const&>(b)),
          "l"(reinterpret_cast<uint64_t const&>(c)));
    return d;
}

CUTE_DEVICE
float2 float2_add(const float2& a, const float2& b) {
    float2 c;
    asm volatile(
        "add.rn.f32x2 %0, %1, %2;\n"
        : "=l"(reinterpret_cast<uint64_t&>(c))
        : "l"(reinterpret_cast<uint64_t const&>(a)),
          "l"(reinterpret_cast<uint64_t const&>(b)));
    return c;
}

template <int N, typename T>
CUTE_DEVICE void tmem_ld_32dp32bNx(uint32_t const& src_addr, T* dst_ptr_) {
    uint32_t* dst_ptr = reinterpret_cast<uint32_t*>(dst_ptr_);
    if constexpr (N == 8) {
        asm volatile(
            "tcgen05.ld.sync.aligned.32x32b.x8.b32"
            "{%0, %1, %2, %3, %4, %5, %6, %7},"
            "[%8];\n"
            : "=r"(dst_ptr[0]), "=r"(dst_ptr[1]), "=r"(dst_ptr[2]),
              "=r"(dst_ptr[3]), "=r"(dst_ptr[4]), "=r"(dst_ptr[5]),
              "=r"(dst_ptr[6]), "=r"(dst_ptr[7])
            : "r"(src_addr));
    } else {
        static_assert(N == 4, "attn_res TMEM helpers support x4 and x8");
        asm volatile(
            "tcgen05.ld.sync.aligned.32x32b.x4.b32"
            "{%0, %1, %2, %3}, [%4];\n"
            : "=r"(dst_ptr[0]), "=r"(dst_ptr[1]),
              "=r"(dst_ptr[2]), "=r"(dst_ptr[3])
            : "r"(src_addr));
    }
}

template <int N, typename T>
CUTE_DEVICE void tmem_st_32dp32bNx(uint32_t const& dst_addr, T* src_ptr_) {
    uint32_t* src_ptr = reinterpret_cast<uint32_t*>(src_ptr_);
    if constexpr (N == 8) {
        asm volatile(
            "tcgen05.st.sync.aligned.32x32b.x8.b32"
            "[%8], {%0, %1, %2, %3, %4, %5, %6, %7};\n"
            :
            : "r"(src_ptr[0]), "r"(src_ptr[1]), "r"(src_ptr[2]),
              "r"(src_ptr[3]), "r"(src_ptr[4]), "r"(src_ptr[5]),
              "r"(src_ptr[6]), "r"(src_ptr[7]), "r"(dst_addr));
    } else {
        static_assert(N == 4, "attn_res TMEM helpers support x4 and x8");
        asm volatile(
            "tcgen05.st.sync.aligned.32x32b.x4.b32"
            "[%4], {%0, %1, %2, %3};\n"
            :
            : "r"(src_ptr[0]), "r"(src_ptr[1]),
              "r"(src_ptr[2]), "r"(src_ptr[3]), "r"(dst_addr));
    }
}

} // namespace sm100

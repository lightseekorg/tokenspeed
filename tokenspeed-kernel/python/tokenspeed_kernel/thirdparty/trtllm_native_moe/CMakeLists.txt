# SPDX-FileCopyrightText: Copyright (c) 2026 LightSeek Foundation
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.31.10)
project(tokenspeed_trtllm_native_moe LANGUAGES CXX CUDA)

# Temporary source-vendored build of the TRT-LLM routed-MoE runner.  TODO:
# remove this target and import the runner from tokenspeed-trtllm-kernel once
# that package exports MxE4m3MxE2m1BlockScaleMoERunner.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CUDA_ARCHITECTURES "100f-real")
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
set(TRTLLM_ABI_NAMESPACE _v1)

find_package(Torch REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Torch's CMake package injects the detected device's ordinary sm_100 flag and
# clears CMAKE_CUDA_ARCHITECTURES. These kernels require the family-specific
# Blackwell target, so replace that injected gencode and restore 100f.
string(REGEX REPLACE "-gencode[ =]+arch=compute_[^, ]+,code=sm_[^ ]+" ""
                     CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
set(CMAKE_CUDA_ARCHITECTURES "100f-real")

if(NOT CUTLASS_INCLUDE_DIR OR
   NOT EXISTS "${CUTLASS_INCLUDE_DIR}/cutlass/cutlass.h")
  message(FATAL_ERROR "CUTLASS_INCLUDE_DIR must point at a CUTLASS include directory")
endif()

set(TRTLLM_CPP "${CMAKE_CURRENT_SOURCE_DIR}/csrc/cpp")
set(BLOCK_SCALE_MOE
    "${TRTLLM_CPP}/tensorrt_llm/kernels/trtllmGenKernels/blockScaleMoe")
set(BATCHED_GEMM
    "${TRTLLM_CPP}/tensorrt_llm/kernels/trtllmGenKernels/batchedGemm")
set(BATCHED_GEMM_EXPORT "${BATCHED_GEMM}/trtllmGen_bmm_export")
set(BATCHED_GEMM_CUBINS "${BATCHED_GEMM_EXPORT}/cubins")

# This temporary bridge exposes only the MXFP8-by-MXFP4 SwiGLU runner used by
# DeepSeek V4. Keep its embedded cubins and generated metadata equally narrow;
# otherwise the wheel silently grows by tens of MiB with unreachable TRT-LLM
# GEMM variants.
file(GLOB NATIVE_MOE_CUBIN_ARCHIVES CONFIGURE_DEPENDS
     "${BATCHED_GEMM_CUBINS}/*.cubin.tar.zst")
list(LENGTH NATIVE_MOE_CUBIN_ARCHIVES NATIVE_MOE_CUBIN_COUNT)
if(NOT NATIVE_MOE_CUBIN_COUNT EQUAL 85)
  message(FATAL_ERROR
          "native MoE expects exactly 85 cubin archives, found "
          "${NATIVE_MOE_CUBIN_COUNT}")
endif()
foreach(ARCHIVE IN LISTS NATIVE_MOE_CUBIN_ARCHIVES)
  get_filename_component(ARCHIVE_NAME "${ARCHIVE}" NAME)
  if(NOT ARCHIVE_NAME MATCHES
     "^Bmm_(MxE4m3_MxE2m1MxE4m3_Fp32|Bfloat16_MxE2m1MxE4m3_Fp32)_.*\\.cubin\\.tar\\.zst$")
    message(FATAL_ERROR "unsupported native MoE cubin family: ${ARCHIVE_NAME}")
  endif()
endforeach()

file(READ "${BATCHED_GEMM_EXPORT}/KernelMetaInfo.h" NATIVE_MOE_METADATA)
string(REGEX MATCHALL
       "extern unsigned char const Bmm_[A-Za-z0-9_]+_cubin"
       NATIVE_MOE_METADATA_DECLARATIONS "${NATIVE_MOE_METADATA}")
string(REGEX MATCHALL
       "extern unsigned int const Bmm_[A-Za-z0-9_]+_cubin_len"
       NATIVE_MOE_METADATA_LENGTHS "${NATIVE_MOE_METADATA}")
string(REGEX MATCHALL
       "\\{Bmm_[A-Za-z0-9_]+_cubin,"
       NATIVE_MOE_METADATA_CONFIGS "${NATIVE_MOE_METADATA}")
foreach(METADATA_INVENTORY IN ITEMS NATIVE_MOE_METADATA_DECLARATIONS
                                    NATIVE_MOE_METADATA_LENGTHS
                                    NATIVE_MOE_METADATA_CONFIGS)
  list(LENGTH ${METADATA_INVENTORY} NATIVE_MOE_METADATA_COUNT)
  if(NOT NATIVE_MOE_METADATA_COUNT EQUAL NATIVE_MOE_CUBIN_COUNT)
    message(FATAL_ERROR
            "native MoE ${METADATA_INVENTORY} count "
            "${NATIVE_MOE_METADATA_COUNT} does not match archive count "
            "${NATIVE_MOE_CUBIN_COUNT}")
  endif()
endforeach()
foreach(ARCHIVE IN LISTS NATIVE_MOE_CUBIN_ARCHIVES)
  get_filename_component(ARCHIVE_NAME "${ARCHIVE}" NAME)
  string(REGEX REPLACE "\\.cubin\\.tar\\.zst$" "" ARCHIVE_STEM
                       "${ARCHIVE_NAME}")
  foreach(METADATA_TOKEN IN ITEMS
      "extern unsigned char const ${ARCHIVE_STEM}_cubin[];"
      "extern unsigned int const ${ARCHIVE_STEM}_cubin_len;"
      "{${ARCHIVE_STEM}_cubin,")
    string(FIND "${NATIVE_MOE_METADATA}" "${METADATA_TOKEN}"
           METADATA_POSITION)
    if(METADATA_POSITION EQUAL -1)
      message(FATAL_ERROR
              "native MoE cubin has no matching metadata token "
              "${METADATA_TOKEN}: ${ARCHIVE_NAME}")
    endif()
  endforeach()
endforeach()

file(GLOB ROUTING_SOURCES CONFIGURE_DEPENDS
     "${BLOCK_SCALE_MOE}/routing/*.cu")

add_library(
  tokenspeed_trtllm_native_moe SHARED
  "${CMAKE_CURRENT_SOURCE_DIR}/tactic_abi.cpp"
  "${TRTLLM_CPP}/tensorrt_llm/thop/mxFp4BlockScaleMoe.cpp"
  "${TRTLLM_CPP}/tensorrt_llm/common/envUtils.cpp"
  "${TRTLLM_CPP}/tensorrt_llm/common/logger.cpp"
  "${TRTLLM_CPP}/tensorrt_llm/common/stringUtils.cpp"
  "${TRTLLM_CPP}/tensorrt_llm/common/tllmException.cpp"
  "${BATCHED_GEMM}/KernelRunner.cpp"
  "${BLOCK_SCALE_MOE}/DevKernel.cu"
  "${BLOCK_SCALE_MOE}/runner.cu"
  ${ROUTING_SOURCES})

target_include_directories(
  tokenspeed_trtllm_native_moe
  PRIVATE "${TRTLLM_CPP}" "${TRTLLM_CPP}/include"
          "${CUTLASS_INCLUDE_DIR}")
target_compile_definitions(
  tokenspeed_trtllm_native_moe
  PRIVATE TRTLLM_ABI_NAMESPACE=${TRTLLM_ABI_NAMESPACE} TLLM_ENABLE_CUDA
          TLLM_GEN_EXPORT_INTERFACE ENABLE_BF16 ENABLE_FP8 ENABLE_FP4)
target_compile_options(
  tokenspeed_trtllm_native_moe
  PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
          $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
          $<$<COMPILE_LANGUAGE:CUDA>:--split-compile=0>
          $<$<COMPILE_LANGUAGE:CUDA>:--fatbin-options=-compress-all>)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
  target_compile_options(tokenspeed_trtllm_native_moe PRIVATE
                         $<$<COMPILE_LANGUAGE:CXX>:-mcmodel=medium>)
  target_link_options(tokenspeed_trtllm_native_moe PRIVATE -mcmodel=medium)
endif()
set_property(TARGET tokenspeed_trtllm_native_moe
             PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_property(TARGET tokenspeed_trtllm_native_moe
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# Torch's package exports its C++ ABI setting through TORCH_CXX_FLAGS.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
target_link_libraries(
  tokenspeed_trtllm_native_moe PRIVATE ${TORCH_LIBRARIES} CUDA::cudart
                                       CUDA::cuda_driver ${CMAKE_DL_LIBS})

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(tllm_cubin_archive)
tllm_add_cubin_archive_sources(
  tokenspeed_trtllm_native_moe
  "${BATCHED_GEMM_CUBINS}"
  NAMESPACE batchedGemm tensorrt_llm ${TRTLLM_ABI_NAMESPACE} kernels)

set_target_properties(
  tokenspeed_trtllm_native_moe
  PROPERTIES OUTPUT_NAME "tokenspeed_trtllm_native_moe"
             BUILD_WITH_INSTALL_RPATH TRUE
             INSTALL_RPATH "$ORIGIN"
             LIBRARY_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")

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

// Exercise the real radix kernel with delayed global histogram atomics.
// The delay changes scheduling only; each update still uses CUDA atomicAdd.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "air_top_p.cuh"
#include "nv_util.h"

__device__ int* delayed_histogram;

template <typename T>
__device__ T delayed_atomic_add(T* address, T value) {
    if constexpr (std::is_same_v<T, int>) {
        const auto offset = reinterpret_cast<uintptr_t>(address)
                          - reinterpret_cast<uintptr_t>(delayed_histogram);
        if (blockIdx.x == 0 && threadIdx.x >= 32 && __isGlobal(address)
            && offset < 2048 * sizeof(int)) {
            auto start = clock64();
            while (clock64() - start < 20000000ULL) {
                __nanosleep(1000);
            }
        }
    }
    return atomicAdd(address, value);
}

// Instrument only this header, with no production test hook or source rewriting.
#define atomicAdd delayed_atomic_add
#include "air_topk_stable.cuh"
#undef atomicAdd
#include "fused_topk_topp.cu"

using TopKCounter = nv::air_topk_stable::Counter<float, int>;
using TopPCounter = air_top_p::Counter<float>;

void check_cuda(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

template <typename T>
struct DeviceBuffer {
    T* data = nullptr;
    explicit DeviceBuffer(size_t count) { check_cuda(cudaMalloc(&data, sizeof(T) * count)); }
    ~DeviceBuffer() { cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

__global__ void save_counter(const TopKCounter* source, TopKCounter* dest) {
    if (threadIdx.x == 0) *dest = *source;
}

int run(bool boundary, bool graph_mode) {
    constexpr int batch = 96, vocab = 131072, k = fused_topk_topp::K_TOPK_MAX;
    const size_t topk_size = fused_topk_topp::airTopKWorkspaceBytes(batch, vocab);
    DeviceBuffer<unsigned char> workspace(fused_topk_topp::getWorkspaceSize(batch, vocab));
    DeviceBuffer<float> probs(size_t(batch) * vocab), ps(batch);
    DeviceBuffer<int> ks(batch);
    DeviceBuffer<TopKCounter> pass0(1);
    auto parts = nv::calc_aligned_pointers(workspace.data, {
        topk_size, sizeof(float) * batch * k, sizeof(int) * batch * k,
        air_top_p::getWorkspaceBytes<float>(batch, vocab)});
    TopKCounter* counters;
    int *histograms, *idx1, *idx2;
    float *buf1, *buf2;
    fused_topk_topp::airTopKResolveWorkspace<float, int, 11>(
        parts[0], vocab, batch, counters, histograms, buf1, idx1, buf2, idx2);
    TopPCounter* topp_counters;
    air_top_p::HisT<float>* topp_histograms;
    int* topp_counts;
    float *topp_buf1, *topp_buf2;
    air_top_p::resolveWorkspace<float>(batch, vocab, parts[3], topp_counters,
        topp_histograms, topp_counts, topp_buf1, topp_buf2);
    auto values = static_cast<float*>(parts[1]);
    auto indices = static_cast<int*>(parts[2]);
    int* delayed_row = histograms + (batch - 1) * 2048;
    check_cuda(cudaMemcpyToSymbol(delayed_histogram, &delayed_row, sizeof(delayed_row)));

    std::vector<int> host_ks(batch, 1 << 30);
    host_ks.back() = k;
    std::vector<float> host_ps(batch, 1.0f);
    std::vector<float> row(vocab, boundary ? (1.0f - 129.0f / 256) / (vocab - 129) : 1.0f / vocab);
    if (boundary) {
        // CTA 0 contributes one of 129 equal maxima; CTA 1 contributes 128.
        row[0] = 1.0f / 256;
        std::fill(row.begin() + 2048, row.begin() + 2176, 1.0f / 256);
    }
    check_cuda(cudaMemset(probs.data, 0, size_t(batch) * vocab * sizeof(float)));
    check_cuda(cudaMemcpy(probs.data + size_t(batch - 1) * vocab, row.data(),
        vocab * sizeof(float), cudaMemcpyHostToDevice));
    check_cuda(cudaMemcpy(ks.data, host_ks.data(), batch * sizeof(int), cudaMemcpyHostToDevice));
    check_cuda(cudaMemcpy(ps.data, host_ps.data(), batch * sizeof(float), cudaMemcpyHostToDevice));

    cudaStream_t stream;
    check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    auto launch = [&] {
        fused_topk_topp::unifiedInitKernel<float, int, 2048, 2048><<<batch, 256, 0, stream>>>(
            counters, histograms, topp_counters, topp_histograms, topp_counts,
            batch, vocab, probs.data, ps.data, ks.data, k, false);
        auto radix = nv::air_topk_stable::radix_kernel<float, int, 11, 512, false, true>;
        radix<<<dim3(2, batch), 512, 0, stream>>>(probs.data, nullptr, probs.data, nullptr,
            nullptr, nullptr, values, indices, counters, histograms, vocab, k, false, 0);
        save_counter<<<1, 1, 0, stream>>>(counters + batch - 1, pass0.data);
        radix<<<dim3(2, batch), 512, 0, stream>>>(probs.data, nullptr, probs.data, nullptr,
            buf1, idx1, values, indices, counters, histograms, vocab, k, false, 1);
        nv::air_topk_stable::radix_kernel<float, int, 11, 512, true, true>
            <<<dim3(1, batch), 512, 0, stream>>>(probs.data, nullptr, buf1, idx1, buf2, idx2,
                values, indices, counters, histograms, vocab, k, false, 2);
        check_cuda(cudaGetLastError());
    };
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    if (graph_mode) {
        check_cuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        launch();
        check_cuda(cudaStreamEndCapture(stream, &graph));
        check_cuda(cudaGraphInstantiate(&executable, graph, 0));
    }
    for (int iteration = 0; iteration < 4; ++iteration) {
        if (graph_mode) check_cuda(cudaGraphLaunch(executable, stream));
        else launch();
        check_cuda(cudaStreamSynchronize(stream));
        TopKCounter first, last;
        std::vector<TopPCounter> topp(batch);
        std::vector<int> selected(k);
        std::vector<float> selected_values(k);
        check_cuda(cudaMemcpy(&first, pass0.data, sizeof(first), cudaMemcpyDeviceToHost));
        check_cuda(cudaMemcpy(&last, counters + batch - 1, sizeof(last), cudaMemcpyDeviceToHost));
        check_cuda(cudaMemcpy(topp.data(), topp_counters, sizeof(TopPCounter) * batch, cudaMemcpyDeviceToHost));
        check_cuda(cudaMemcpy(selected.data(), indices + (batch - 1) * k, k * sizeof(int), cudaMemcpyDeviceToHost));
        check_cuda(cudaMemcpy(selected_values.data(), values + (batch - 1) * k, k * sizeof(float), cudaMemcpyDeviceToHost));
        bool pointers_intact = true;
        for (int b = 0; b < batch; ++b) pointers_intact &= topp[b].in == probs.data + size_t(b) * vocab;
        if (first.len != (boundary ? 129 : vocab) || first.k != k || last.out_cnt > k || !pointers_intact) {
            std::fprintf(stderr, "pass0 len=%d expected=%d k=%d out_cnt=%d pointers_intact=%d\n",
                first.len, boundary ? 129 : vocab, first.k, last.out_cnt, pointers_intact);
            return 1;
        }
        std::sort(selected.begin(), selected.end());
        for (int i = 0; i < k; ++i) {
            int expected = boundary && i > 0 ? 2047 + i : i;
            if (selected[i] != expected || selected_values[i] != (boundary ? 1.0f / 256 : 1.0f / vocab)) {
                std::fprintf(stderr, "incorrect top-k result at %d: index=%d expected=%d\n", i, selected[i], expected);
                return 1;
            }
        }
    }
    if (executable) check_cuda(cudaGraphExecDestroy(executable));
    if (graph) check_cuda(cudaGraphDestroy(graph));
    check_cuda(cudaStreamDestroy(stream));
    return 0;
}

int main(int argc, char** argv) {
    if (argc != 3) return 2;
    int count = 0;
    auto status = cudaGetDeviceCount(&count);
    if (status == cudaErrorNoDevice || (status == cudaSuccess && count == 0)) return 77;
    try {
        check_cuda(status);
        return run(std::string(argv[1]) == "boundary", std::string(argv[2]) == "graph");
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 2;
    }
}

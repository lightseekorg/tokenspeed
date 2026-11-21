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

// Run the fused Top-K/Top-P pipeline behind a producer that signals PDL launch
// completion before it stores the probabilities, as the softmax kernels may.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include "fused_topk_topp.cu"

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

// Dependents may launch as soon as every CTA has signalled; the stores follow.
__global__ void early_signal_producer(float* probs, const float* fresh, size_t count,
                                      unsigned long long delay_cycles) {
    cudaTriggerProgrammaticLaunchCompletion();
    auto start = clock64();
    while (clock64() - start < delay_cycles) {
        __nanosleep(1000);
    }
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += stride) {
        probs[i] = fresh[i];
    }
}

int run(bool graph_mode, unsigned long long delay_cycles) {
    constexpr int batch = 8, vocab = 131072, highs = 160;
    constexpr int no_top_k = 1 << 30;
    // Row 0 takes the Top-P radix; the last row's Top-K indices sit next to its counter.
    const std::vector<int> host_ks = {no_top_k, 1, 20, 64, 20, 50, no_top_k, 128};
    const std::vector<float> host_ps = {0.5f, 1.0f, 1.0f, 1.0f, 0.5f, 1.0f, 0.5f, 1.0f};

    // Every row: `highs` distinct large values at the end of the row, descending.
    std::vector<double> high(highs);
    double weight_sum = 0.0;
    for (int j = 0; j < highs; ++j) weight_sum += std::pow(0.97, j);
    for (int j = 0; j < highs; ++j) high[j] = 0.9 * std::pow(0.97, j) / weight_sum;
    std::vector<float> row(vocab, static_cast<float>(0.1 / (vocab - highs)));
    for (int j = 0; j < highs; ++j) row[vocab - highs + j] = static_cast<float>(high[j]);
    std::vector<float> host_fresh(size_t(batch) * vocab);
    for (int b = 0; b < batch; ++b) {
        std::copy(row.begin(), row.end(), host_fresh.begin() + size_t(b) * vocab);
    }

    DeviceBuffer<unsigned char> workspace(fused_topk_topp::getWorkspaceSize(batch, vocab));
    DeviceBuffer<float> probs(size_t(batch) * vocab), fresh(size_t(batch) * vocab);
    DeviceBuffer<float> out(size_t(batch) * vocab), ps(batch);
    DeviceBuffer<int> ks(batch);
    check_cuda(cudaMemcpy(fresh.data, host_fresh.data(), host_fresh.size() * sizeof(float),
        cudaMemcpyHostToDevice));
    check_cuda(cudaMemcpy(ks.data, host_ks.data(), batch * sizeof(int), cudaMemcpyHostToDevice));
    check_cuda(cudaMemcpy(ps.data, host_ps.data(), batch * sizeof(float), cudaMemcpyHostToDevice));

    auto parts = nv::calc_aligned_pointers(workspace.data, {
        fused_topk_topp::airTopKWorkspaceBytes(batch, vocab),
        sizeof(float) * batch * fused_topk_topp::K_TOPK_MAX,
        sizeof(int) * batch * fused_topk_topp::K_TOPK_MAX,
        air_top_p::getWorkspaceBytes<float>(batch, vocab)});
    auto topp_counters = static_cast<TopPCounter*>(parts[3]);

    cudaStream_t stream, side_stream;
    check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    check_cuda(cudaStreamCreateWithFlags(&side_stream, cudaStreamNonBlocking));
    auto launch = [&] {
        // Stale contents the pipeline must not observe.
        check_cuda(cudaMemsetAsync(probs.data, 0, size_t(batch) * vocab * sizeof(float), stream));
        // A small grid: the sleeping producer must leave the SMs free for the pipeline.
        early_signal_producer<<<64, 256, 0, stream>>>(probs.data, fresh.data,
            size_t(batch) * vocab, delay_cycles);
        fused_topk_topp::invokeFusedTopKTopP(probs.data, ks.data, ps.data, out.data,
            workspace.data, batch, vocab, stream, side_stream, /*enablePdl=*/true,
            /*inPlaceSkip=*/false);
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
    std::vector<float> host_out(size_t(batch) * vocab);
    std::vector<TopPCounter> topp(batch);
    for (int iteration = 0; iteration < 4; ++iteration) {
        if (graph_mode) check_cuda(cudaGraphLaunch(executable, stream));
        else launch();
        check_cuda(cudaStreamSynchronize(stream));
        check_cuda(cudaStreamSynchronize(side_stream));
        check_cuda(cudaMemcpy(host_out.data(), out.data, host_out.size() * sizeof(float),
            cudaMemcpyDeviceToHost));
        check_cuda(cudaMemcpy(topp.data(), topp_counters, sizeof(TopPCounter) * batch,
            cudaMemcpyDeviceToHost));
        for (int b = 0; b < batch; ++b) {
            if (topp[b].in != probs.data + size_t(b) * vocab) {
                std::fprintf(stderr, "iteration %d row %d: Top-P input pointer %p, expected %p\n",
                    iteration, b, static_cast<const void*>(topp[b].in),
                    static_cast<const void*>(probs.data + size_t(b) * vocab));
                return 1;
            }
            // Top-K keeps the K largest; Top-P then keeps the shortest prefix
            // reaching P of the mass Top-K left.
            const bool top_k_active = host_ks[b] <= fused_topk_topp::K_TOPK_MAX;
            int kept = top_k_active ? host_ks[b] : highs;
            double top_k_mass = top_k_active ? 0.0 : 1.0;
            for (int j = 0; top_k_active && j < kept; ++j) top_k_mass += row[vocab - highs + j];
            double mass = 0.0;
            for (int j = 0; j < kept; ++j) {
                mass += row[vocab - highs + j];
                if (mass >= host_ps[b] * top_k_mass) {
                    kept = j + 1;
                    break;
                }
            }
            const float* actual = host_out.data() + size_t(b) * vocab;
            for (int i = 0; i < vocab; ++i) {
                const int j = i - (vocab - highs);
                const double expected = (j >= 0 && j < kept) ? row[i] / mass : 0.0;
                // Negated so that a NaN fails the comparison.
                if (!(std::abs(actual[i] - expected) <= 1e-5 * expected)) {
                    std::fprintf(stderr, "iteration %d row %d token %d: %g, expected %g (kept %d)\n",
                        iteration, b, i, actual[i], expected, kept);
                    return 1;
                }
            }
        }
    }
    if (executable) check_cuda(cudaGraphExecDestroy(executable));
    if (graph) check_cuda(cudaGraphDestroy(graph));
    check_cuda(cudaStreamDestroy(side_stream));
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
        return run(std::string(argv[1]) == "graph", std::strtoull(argv[2], nullptr, 10));
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 2;
    }
}

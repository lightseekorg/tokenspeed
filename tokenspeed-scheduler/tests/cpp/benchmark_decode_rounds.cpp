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

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

#include "benchmark_report.h"
#include "decode_round_benchmark.h"

namespace tokenspeed::benchmark {
namespace {

void Run(const DecodeRoundConfig& config, int warmup_rounds, int rounds) {
    Require(rounds > 0, "rounds must be positive");
    Require(warmup_rounds + rounds <= config.num_rounds, "the requests must have a round left for every sample");
    std::cerr << "Prefilling " << config.num_requests << " requests of " << config.context_tokens
              << " tokens and decoding through their admission reserve...\n";
    DecodeRoundWorkload workload(config);
    std::vector<DecodeRoundSample> samples;
    for (int i = 0; i < warmup_rounds + rounds; ++i) {
        const auto sample = workload.RunRound();
        if (i >= warmup_rounds) samples.push_back(sample);
    }
    std::cout << std::setprecision(9) << "{\"benchmark\":\"decode_rounds\",\"page_size\":" << config.page_size
              << ",\"context_tokens\":" << config.context_tokens << ",\"num_requests\":" << config.num_requests
              << ",\"decode_tokens\":" << config.decode_tokens << ",\"warmup_rounds\":" << warmup_rounds
              << ",\"rounds\":" << rounds << ",\"active_lcm_blocks\":" << workload.ActiveLcmBlocks();
    PrintTiming("plan", samples, &DecodeRoundSample::plan);
    PrintTiming("advance", samples, &DecodeRoundSample::advance);
    std::cout << "}\n" << std::flush;
}

}  // namespace
}  // namespace tokenspeed::benchmark

int main(int argc, char** argv) {
    try {
        namespace bench = tokenspeed::benchmark;
        bench::Require(argc == 7,
                       "usage: benchmark PAGE_SIZE CONTEXT_TOKENS REQUESTS DECODE_TOKENS WARMUP_ROUNDS ROUNDS");
        const int warmup_rounds = bench::ParseNonnegative(argv[5]);
        const int rounds = bench::ParseNonnegative(argv[6]);
        bench::Run(bench::DecodeRoundConfig{.page_size = bench::ParseNonnegative(argv[1]),
                                            .context_tokens = bench::ParseNonnegative(argv[2]),
                                            .num_requests = bench::ParseNonnegative(argv[3]),
                                            .decode_tokens = bench::ParseNonnegative(argv[4]),
                                            .num_rounds = warmup_rounds + rounds},
                   warmup_rounds, rounds);
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "decode_rounds benchmark: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}

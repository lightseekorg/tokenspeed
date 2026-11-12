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

// Profile spans: the scheduler's phases on a profile's timeline, under the
// trace category `scheduler`. While a profile runs, the host installs a sink
// that receives every finished span (ts-host's `/start_profile`,
// `ts_scheduler::profile`); without one a span costs one atomic load. Spans
// go where the scheduler does work: a phase of planning or of an admission,
// not every request it passes over.

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>

namespace tokenspeed {

// Receives one finished span: its name, which has static storage duration,
// and its begin and end in nanoseconds of the system clock (Unix time), the
// clock the torch profiler stamps its CPU events with. Called on the thread
// that ran the span.
using ProfileSpanSink = void (*)(const char* name, std::int64_t begin_ns, std::int64_t end_ns);

inline std::atomic<ProfileSpanSink> profile_span_sink{nullptr};

// Installs `sink`, or with nullptr stops recording. A span open across the
// change reports to the sink it began with, so a sink must stay callable
// after it is replaced.
inline void SetProfileSpanSink(ProfileSpanSink sink) {
    profile_span_sink.store(sink, std::memory_order_relaxed);
}

// Records its own lifetime as the span `name`, a string literal.
class ProfileSpan {
public:
    explicit ProfileSpan(const char* name) : name_{name}, sink_{profile_span_sink.load(std::memory_order_relaxed)} {
        if (sink_ != nullptr) {
            begin_ns_ = Now();
        }
    }
    ~ProfileSpan() {
        if (sink_ != nullptr) {
            sink_(name_, begin_ns_, Now());
        }
    }
    ProfileSpan(const ProfileSpan&) = delete;
    ProfileSpan& operator=(const ProfileSpan&) = delete;

private:
    static std::int64_t Now() {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch())
            .count();
    }

    const char* name_;
    ProfileSpanSink sink_;
    std::int64_t begin_ns_{0};
};

}  // namespace tokenspeed

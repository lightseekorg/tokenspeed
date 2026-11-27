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

//! Typed, role-local timing returned once with a terminal request output.

wire_struct! {
    /// The clock shared by all timestamps in one request timing record.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct RequestClock: map {
        /// Reporting host; does not imply synchronized wall clocks.
        pub host_id: String,
        /// Process instance whose origin defines the monotonic timestamps.
        pub process_instance_id: String,
        /// Clock synchronization status, currently `"unknown"`.
        pub clock_sync: String,
        /// Estimated wall-clock uncertainty in nanoseconds, when known.
        pub clock_uncertainty_ns: Option<String>,
    }
}

wire_struct! {
    /// One observed boundary, using the record's shared clock.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct RequestTimestamp: map {
        /// Wall-clock nanoseconds since the Unix epoch, as a decimal string.
        pub ts_unix_ns: String,
        /// Nanoseconds since the process origin, or absent if it predates that origin.
        pub ts_mono_ns: Option<String>,
    }
}

wire_struct! {
    /// Local engine boundaries; unobserved acceptance/readiness remains absent.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct RequestTimingEvents: map {
        /// Output-processor registration, absent for requests rejected before registration.
        pub accepted: Option<RequestTimestamp>,
        /// First completed plain prefill or agreed decode KV/bootstrap readiness.
        pub kv_ready: Option<RequestTimestamp>,
        /// Local output-processor completion, before transport and resource retirement.
        pub finished: RequestTimestamp,
    }
}

wire_struct! {
    /// One local engine request's timing, independent of logs and trace sampling.
    ///
    /// Request id, completion token count and finish reason belong to the
    /// containing output. Prefill and decode roles report separate lifetimes.
    #[derive(Debug, Clone, PartialEq)]
    pub struct RequestTiming: map {
        /// Local engine role: `"plain"`, `"prefill"`, or `"decode"`.
        pub role: String,
        /// Host-configured data-parallel rank, when supplied.
        pub dp_rank: Option<i64>,
        /// Host-configured attention tensor-parallel rank, when supplied.
        pub tp_rank: Option<i64>,
        /// Shared host/process identity and clock synchronization status.
        pub clock: RequestClock,
        /// Registration, first readiness and local completion timestamps.
        pub events: RequestTimingEvents,
        /// Committed output tokens at first readiness; absent without that boundary.
        pub kv_output_tokens: Option<i64>,
        /// Native backend latency in seconds from frontend `created_time`, clamped at zero.
        pub e2e_latency_s: Option<f64>,
    }
}

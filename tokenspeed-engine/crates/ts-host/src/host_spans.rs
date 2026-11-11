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

//! The host's span recording (`ts_scheduler::profile`) as the data plane's
//! profiler (`engine/profiler.py`) drives it: started and stopped with a
//! torch profile, and merged into the trace that exports.

use pyo3::prelude::*;
use ts_scheduler::profile;

/// A span as Python receives it: `(name, tid, begin_ns, end_ns)`.
pub(crate) type PySpan = (String, u64, i64, i64);

/// Starts and stops the host's span recording.
#[pyclass(frozen, name = "HostSpans")]
pub struct HostSpans;

#[pymethods]
impl HostSpans {
    /// Start a recording, dropping what an earlier one left.
    fn start(&self) {
        profile::start();
    }

    /// Stop the recording. Returns its spans, as `(name, tid, begin_ns,
    /// end_ns)` in nanoseconds of Unix time, and the threads that recorded,
    /// as `(tid, name)`.
    fn stop(&self) -> (Vec<PySpan>, Vec<(u64, String)>) {
        let recording = profile::stop();
        let spans = recording
            .spans
            .into_iter()
            .map(|span| (span.name.into_owned(), span.tid, span.begin_ns, span.end_ns))
            .collect();
        (spans, recording.threads)
    }
}

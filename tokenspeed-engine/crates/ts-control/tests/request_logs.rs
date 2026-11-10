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

//! Request lifecycle events from registration through the terminal output.

use std::collections::BTreeSet;
use ts_control::{
    output::{FinishLog, OutputProcessor, Role},
    request::RequestState,
    request_stats::wall_seconds,
};

#[test]
fn request_events_pair_admission_with_completion_and_transfer_failure() {
    use std::{io::Read, os::unix::net::UnixStream};

    let (writer, mut reader) = UnixStream::pair().unwrap();
    let subscriber = tracing_subscriber::fmt()
        .without_time()
        .with_ansi(false)
        .with_writer(move || writer.try_clone().unwrap())
        .finish();
    tracing::subscriber::with_default(subscriber, || {
        let mut p = OutputProcessor::new(
            None,
            1,
            None,
            Role::Decode,
            FinishLog {
                enabled: true,
                request_stats: false,
                ttft_percentage: 0,
            },
            false,
            None,
        );
        let mut state = RequestState::new(vec![5, 6], 1, BTreeSet::new(), true, None, false);
        state.created_time = wall_seconds() - 1.0;
        p.register("ok".to_owned(), state);
        p.remote_prefill_done("ok", 7, 1, None);
        p.finish_remote_prefill_only("ok");
        p.register(
            "failed".to_owned(),
            RequestState::new(vec![8, 9], 3, BTreeSet::new(), true, None, false),
        );
        p.fail_transfer("failed");
        p.fail_transfer("failed");
    });
    let mut log = String::new();
    reader.read_to_string(&mut log).unwrap();
    let lines: Vec<_> = log
        .lines()
        .filter(|line| line.contains("event=ts_request_"))
        .collect();
    assert_eq!(lines.len(), 4, "{log}");
    assert!(lines[0].contains("event=ts_request_accepted rid=\"ok\""));
    assert!(lines[0].contains("input_tokens=2 max_new_tokens=1"));
    assert!(lines[1].contains("event=ts_request_finished rid=\"ok\""));
    assert!(lines[1].contains("completion_tokens=1 finish_reason=\"length\" cached_tokens=1"));
    assert!(lines[1].contains("e2e_latency_s="));
    assert!(lines[2].contains("event=ts_request_accepted rid=\"failed\""));
    assert!(lines[3].contains("event=ts_request_finished rid=\"failed\""));
    assert!(lines[3].contains("completion_tokens=0 finish_reason=\"abort\""));
    assert!(!lines[3].contains("e2e_latency_s="));
}

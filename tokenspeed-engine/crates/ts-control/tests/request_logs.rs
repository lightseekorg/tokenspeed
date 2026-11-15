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
    for request_stats in [false, true] {
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
                    request_stats,
                    ttft_percentage: 0,
                },
                false,
                None,
            );
            let mut state = RequestState::new(vec![5, 6], 1, BTreeSet::new(), true, None, false);
            state.created_time = wall_seconds() - 1.0;
            p.assign_admission(&mut state);
            p.register("ok".to_owned(), state);
            p.remote_prefill_done("ok", 7, 1, None);
            p.finish_remote_prefill_only("ok");
            let mut failed = RequestState::new(vec![8, 9], 3, BTreeSet::new(), true, None, false);
            p.assign_admission(&mut failed);
            p.register("failed".to_owned(), failed);
            p.fail_transfer("failed");
            p.fail_transfer("failed");
            let mut duplicate =
                RequestState::new(vec![10, 11], 1, BTreeSet::new(), true, None, false);
            p.assign_admission(&mut duplicate);
            p.register("duplicate".to_owned(), duplicate);
            let mut refused = RequestState::new(vec![12], 1, BTreeSet::new(), true, None, false);
            refused.abort("duplicate request ID", true, Some(409));
            assert_eq!(
                p.publish_finished_at_admission("duplicate", refused).len(),
                1
            );
            assert!(p.state("duplicate").unwrap().finished_reason.is_none());
            p.remote_prefill_done("duplicate", 12, 0, None);
            assert_eq!(
                p.finish_remote_prefill_only("duplicate").released,
                ["duplicate"]
            );
        });
        let mut log = String::new();
        reader.read_to_string(&mut log).unwrap();
        let lines: Vec<_> = log
            .lines()
            .filter(|line| line.contains("event=ts_request_"))
            .collect();
        assert_eq!(lines.len(), 7, "{log}");
        // Stats belong to the finish event, with no extra log record.
        assert_eq!(log.lines().count(), 7, "{log}");
        let expected: &[&[&str]] = &[
            &[
                "event=ts_request_accepted",
                "rid=\"ok\"",
                "input_tokens=2",
                "max_new_tokens=1",
            ],
            &[
                "event=ts_request_finished",
                "rid=\"ok\"",
                "completion_tokens=1",
                "finish_reason=\"length\"",
                "cached_tokens=1",
                "e2e_latency_s=",
            ],
            &["event=ts_request_accepted", "rid=\"failed\""],
            &[
                "event=ts_request_finished",
                "rid=\"failed\"",
                "completion_tokens=0",
                "finish_reason=\"abort\"",
            ],
            &["event=ts_request_accepted", "rid=\"duplicate\""],
            &[
                "event=ts_request_finished",
                "rid=\"duplicate\"",
                "finish_reason=\"abort\"",
                "abort_reason=\"duplicate request ID\"",
            ],
            &[
                "event=ts_request_finished",
                "rid=\"duplicate\"",
                "finish_reason=\"length\"",
                "completion_tokens=1",
            ],
        ];
        for (line, fields) in lines.iter().zip(expected) {
            for field in *fields {
                assert!(line.contains(field), "missing {field}: {line}");
            }
        }
        assert!(!lines[3].contains("e2e_latency_s="));
        assert_eq!(
            log.matches("RequestStats(").count(),
            if request_stats { 3 } else { 0 }
        );
        for (index, rid) in [(1, "ok"), (3, "failed"), (6, "duplicate")] {
            assert_eq!(lines[index].contains("request_stats="), request_stats);
            if request_stats {
                for field in [
                    format!("Req: {rid} Finish! RequestStats("),
                    "queue_ms=".into(),
                    "ttft_ms=".into(),
                    "total_ms=".into(),
                ] {
                    assert!(lines[index].contains(&field), "{log}");
                }
            }
        }
    }
}

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

//! Request logs from a CPU transfer through the real PD peers.

use std::time::Duration;
use ts_pd::{BootstrapInfo, TransferPeer};

#[path = "../src/tests_support.rs"]
mod native;
mod support;
use support::*;

#[test]
fn transfer_events_identify_the_request_and_room() {
    use std::{io::Read, os::unix::net::UnixStream};

    let library = native::NativeLibrary::new();
    let source = pattern();
    let destination = vec![0; BYTES];
    let mut pair = pair(
        library.path().to_owned(),
        &source,
        &destination,
        Duration::from_secs(60),
    );
    let (mut mesh, group) = single_rank();
    let (writer, mut reader) = UnixStream::pair().unwrap();
    let subscriber = tracing_subscriber::fmt()
        .without_time()
        .with_ansi(false)
        .with_writer(move || writer.try_clone().unwrap())
        .finish();
    tracing::subscriber::with_default(subscriber, || {
        transfer_one(&mut pair, "ok", 40, &mut mesh, &group);
        let bootstrap = BootstrapInfo {
            room: 41,
            ..pair.bootstrap.clone()
        };
        pair.decode.register("failed", &bootstrap).unwrap();
        pair.decode.cancel("failed");
        expect_cancelled(&mut pair.decode, "failed", &mut mesh, &group);
    });
    let mut log = String::new();
    reader.read_to_string(&mut log).unwrap();
    let events: Vec<_> = log
        .lines()
        .filter(|line| line.contains("event=ts_"))
        .collect();
    assert_eq!(events.len(), 3, "{log}");
    let expected: &[&[&str]] = &[
        &[
            "event=ts_prefill_kv_transfer_queued",
            "rid=\"ok\"",
            "bootstrap_room=40",
            "start_idx=0",
            "end_idx=48",
            "pages=3",
        ],
        &[
            "event=ts_decode_kv_transfer_success",
            "rid=\"ok\"",
            "bootstrap_room=40",
            "cached_tokens=0",
            "output_tokens=1",
        ],
        &[
            "WARN",
            "event=ts_decode_kv_transfer_failed",
            "rid=\"failed\"",
            " room=41",
            "bootstrap_room=41",
            "rank=0",
        ],
    ];
    for (line, fields) in events.iter().zip(expected) {
        for field in *fields {
            assert!(line.contains(field), "missing {field}: {line}");
        }
    }
}

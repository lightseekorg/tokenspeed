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

//! Decode cache registrations lost with a failed control connection, through
//! the real peers, sockets and bootstrap service. Only the network is faked:
//! a TCP relay in front of the Prefill control socket drops the connections
//! a test chooses and forwards the others.

use std::{
    io::{Read, Write},
    net::{Shutdown, TcpListener, TcpStream},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread,
    time::Duration,
};

use serde_json::json;
use ts_pd::TransferPeer;

#[path = "../src/tests_support.rs"]
mod native;
mod support;
use support::*;

/// Relay connections to `target`, closing at once each one `drop` picks by
/// its index. Returns the relay port.
fn relay(target: u16, drop: impl Fn(usize) -> bool + Send + 'static) -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    thread::spawn(move || {
        for (index, client) in listener.incoming().enumerate() {
            let Ok(client) = client else { return };
            if drop(index) {
                let _ = client.shutdown(Shutdown::Both);
                continue;
            }
            let server = TcpStream::connect(("127.0.0.1", target)).unwrap();
            for (mut from, mut to) in [
                (client.try_clone().unwrap(), server.try_clone().unwrap()),
                (server, client),
            ] {
                thread::spawn(move || {
                    let mut buffer = [0u8; 16384];
                    while let Ok(read) = from.read(&mut buffer) {
                        if read == 0 || to.write_all(&buffer[..read]).is_err() {
                            break;
                        }
                    }
                    let _ = to.shutdown(Shutdown::Write);
                });
            }
        }
    });
    port
}

/// Point the Prefill rank's bootstrap entry at `port`.
fn advertise(pair: &Pair, port: u16) {
    let put =
        ureq::put(&format!("http://127.0.0.1:{}/route", pair.bootstrap.port)).send_json(json!({
            "role": "Prefill",
            "world_size": 1,
            "dp_size": 1,
            "pp_size": 1,
            "rank_ip": "127.0.0.1",
            "rank_port": port,
            "engine_rank": 0,
            "cache_layout": layout(None),
            "cache_fields_by_stage": [FIELDS],
        }));
    assert!(put.is_ok(), "{put:?}");
}

#[test]
fn a_registration_lost_with_its_connection_is_resent() {
    let library = native::NativeLibrary::new();
    let source = pattern();
    let mut destination = vec![0u8; BYTES];
    let mut pair = pair(
        library.path().to_path_buf(),
        &source,
        &destination,
        Duration::from_secs(120),
    );
    let (mut mesh, group) = single_rank();
    // The first control connection fails after the registration was queued
    // on it, taking the registration with its backlog.
    advertise(&pair, relay(pair.prefill.port(), |index| index == 0));
    transfer_one(&mut pair, "first", 4242, &mut mesh, &group);
    assert!(landed(&destination, &source));
    // The route stays usable for the next request.
    destination.fill(0);
    transfer_one(&mut pair, "second", 4343, &mut mesh, &group);
    assert!(landed(&destination, &source));
    pair.decode_outbox.release_registration().unwrap();
}

#[test]
fn an_unacknowledged_registration_fails_only_its_room() {
    let library = native::NativeLibrary::new();
    let source = pattern();
    let mut destination = vec![0u8; BYTES];
    let mut pair = pair(
        library.path().to_path_buf(),
        &source,
        &destination,
        Duration::from_secs(120),
    );
    let (mut mesh, group) = single_rank();
    let unreachable = Arc::new(AtomicBool::new(true));
    let dropping = unreachable.clone();
    advertise(
        &pair,
        relay(pair.prefill.port(), move |_| {
            dropping.load(Ordering::Relaxed)
        }),
    );
    // No manifest went out, so the room retires without the Prefill rank.
    pair.decode
        .register("unacknowledged", &pair.bootstrap)
        .unwrap();
    expect_cancelled(&mut pair.decode, "unacknowledged", &mut mesh, &group);
    // The route was not recorded as registered: the next request registers
    // again once the Prefill rank is reachable.
    unreachable.store(false, Ordering::Relaxed);
    destination.fill(0);
    transfer_one(&mut pair, "reachable", 4343, &mut mesh, &group);
    assert!(landed(&destination, &source));
    pair.decode_outbox.release_registration().unwrap();
}

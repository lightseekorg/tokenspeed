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

//! The KV event publisher: the frame the scheduler encodes (the payload of
//! `tests/fixtures/kv_events.hex`, the Python schema's bytes) goes out
//! unchanged with its sequence number on the PUB and replay protocol a
//! subscriber speaks.

use std::time::Duration;

use bytes::Bytes;
use ts_control::kv_events::{KvEventPublisher, KvEventsConfig};
use zeromq::{DealerSocket, Socket, SocketRecv, SocketSend, SubSocket, ZmqMessage};

/// The Python schema's encoding of a three-event batch, as the scheduler's
/// frame drain writes it.
fn fixture_frame() -> Vec<u8> {
    let path = format!(
        "{}/tests/fixtures/kv_events.hex",
        env!("CARGO_MANIFEST_DIR")
    );
    let hex = std::fs::read_to_string(path).unwrap();
    let hex = hex.trim();
    (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
        .collect()
}

fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

#[test]
fn subscribers_receive_frames_and_replay_them() {
    let (port, replay_port) = (free_port(), free_port());
    let publisher = KvEventPublisher::start(
        &KvEventsConfig {
            endpoint: format!("tcp://*:{port}"),
            replay_endpoint: Some(format!("tcp://*:{replay_port}")),
            buffer_steps: 8,
            max_queue_size: 64,
            topic: "kv".to_owned(),
        },
        0,
    )
    .unwrap();
    assert_eq!(publisher.dp_rank(), 0);
    let frame = fixture_frame();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let mut subscriber = SubSocket::new();
    runtime.block_on(async {
        subscriber
            .connect(&format!("tcp://127.0.0.1:{port}"))
            .await
            .unwrap();
        subscriber.subscribe("").await.unwrap();
    });
    // A PUB socket drops batches until the subscription arrives: publish
    // until one lands.
    let received = loop {
        // From the control thread, outside any runtime, as the loop does.
        publisher.publish(&frame);
        let received = runtime.block_on(async {
            tokio::time::timeout(Duration::from_millis(200), subscriber.recv()).await
        });
        if let Ok(Ok(message)) = received {
            break message.into_vec();
        }
    };
    assert_eq!(received.len(), 3);
    assert_eq!(received[0], Bytes::from_static(b"kv"));
    let seq = u64::from_be_bytes(received[1].as_ref().try_into().unwrap());
    assert_eq!(
        received[2].as_ref(),
        frame.as_slice(),
        "the frame goes out as the scheduler encoded it"
    );

    // Replay from that batch: it and every later one, then the end marker.
    let replayed = runtime.block_on(async {
        let mut client = DealerSocket::new();
        client
            .connect(&format!("tcp://127.0.0.1:{replay_port}"))
            .await
            .unwrap();
        let mut request = ZmqMessage::from(Bytes::new());
        request.push_back(Bytes::copy_from_slice(&seq.to_be_bytes()));
        client.send(request).await.unwrap();
        let mut seqs = Vec::new();
        loop {
            let frames = client.recv().await.unwrap().into_vec();
            assert_eq!(frames.len(), 3, "[delimiter, seq, payload]");
            let replayed = i64::from_be_bytes(frames[1].as_ref().try_into().unwrap());
            if replayed == -1 {
                return seqs;
            }
            assert_eq!(frames[2].as_ref(), frame.as_slice());
            seqs.push(replayed as u64);
        }
    });
    assert_eq!(replayed.first(), Some(&seq));
    assert!(replayed.windows(2).all(|pair| pair[1] == pair[0] + 1));
}

#[test]
fn an_empty_frame_is_not_published() {
    let port = free_port();
    let publisher = KvEventPublisher::start(
        &KvEventsConfig {
            endpoint: format!("tcp://*:{port}"),
            replay_endpoint: None,
            buffer_steps: 2,
            max_queue_size: 2,
            topic: "kv".to_owned(),
        },
        3,
    )
    .unwrap();
    assert_eq!(publisher.dp_rank(), 3);
    // Nothing to send: the call returns at once and the thread stays idle;
    // dropping the publisher then returns within its shutdown bound.
    publisher.publish(&[]);
    drop(publisher);
}

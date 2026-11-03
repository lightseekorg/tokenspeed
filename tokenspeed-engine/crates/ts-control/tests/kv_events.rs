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

//! The KV event publisher against the Python one: the payload bytes
//! (`tests/fixtures/kv_events.hex`, from `pd/kv_events.py`) and the PUB and
//! replay protocol a subscriber speaks.

use std::time::Duration;

use bytes::Bytes;
use ts_control::kv_events::{encode_batch, KvEventPublisher, KvEventsConfig};
use ts_scheduler::KvEventView;
use zeromq::{DealerSocket, Socket, SocketRecv, SocketSend, SubSocket, ZmqMessage};

fn fixture_batch() -> Vec<KvEventView> {
    vec![
        KvEventView {
            stored: true,
            block_hashes: vec![11, (1 << 63) + 5],
            has_parent: true,
            parent_block_hash: 7,
            token_ids: (1..=8).collect(),
            block_size: 4,
        },
        KvEventView {
            stored: true,
            block_hashes: vec![u64::MAX],
            has_parent: false,
            parent_block_hash: 0,
            token_ids: vec![9, 10, 11, 12],
            block_size: 4,
        },
        KvEventView {
            stored: false,
            block_hashes: vec![11, 3],
            has_parent: false,
            parent_block_hash: 0,
            token_ids: Vec::new(),
            block_size: 0,
        },
    ]
}

#[test]
fn a_batch_encodes_as_the_python_publisher_s() {
    let path = format!(
        "{}/tests/fixtures/kv_events.hex",
        env!("CARGO_MANIFEST_DIR")
    );
    let expected = std::fs::read_to_string(path).unwrap();
    let encoded: String = encode_batch(1_700_000_000.25, &fixture_batch(), 1)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    assert_eq!(encoded, expected.trim());
}

fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

#[test]
fn subscribers_receive_batches_and_replay_them() {
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
        publisher.publish(&fixture_batch()[2..]);
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
            seqs.push(replayed as u64);
        }
    });
    assert_eq!(replayed.first(), Some(&seq));
    assert!(replayed.windows(2).all(|pair| pair[1] == pair[0] + 1));
}

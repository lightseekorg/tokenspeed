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

//! The engine's side of the wire against the frontend client, over real
//! sockets: what a scheduler and an EPD encode server register as, and the
//! encode server's intake.

use std::{
    sync::atomic::{AtomicU64, Ordering},
    thread,
    time::{Duration, Instant},
};

use ts_client::{ClientConfig, ClientListener, EngineClient, LoadBalance, Routing};
use ts_control::{
    encode::Intake,
    transport::{EncodeIntake, EngineTransport},
};
use ts_wire::{
    frames::{REQUEST_CONTROL, REQUEST_ENCODE},
    EncodeRequest, EngineCoreReadyResponse,
};

fn ipc(name: &str) -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!(
        "ts-control-wire-{}-{name}-{n}.sock",
        std::process::id()
    ));
    format!("ipc://{}", path.display())
}

async fn listener() -> ClientListener {
    ClientListener::bind(ClientConfig {
        handshake_address: ipc("handshake"),
        input_address: ipc("input"),
        output_address: ipc("output"),
        first_engine_index: 0,
        num_engines: 1,
        routing: Routing::Aggregated(LoadBalance::RoundRobin),
        handshake_timeout: Duration::from_secs(20),
        send_timeout: Duration::from_secs(5),
    })
    .await
    .unwrap()
}

fn ready() -> EngineCoreReadyResponse {
    EngineCoreReadyResponse {
        max_model_len: 4096,
        disaggregation_mode: Some("encode".to_owned()),
        bootstrap_host: Some("10.0.0.9".to_owned()),
        bootstrap_port: Some(8998),
        ..Default::default()
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_scheduler_registers_its_request_types_and_layouts() {
    let listener = listener().await;
    let handshake = listener.handshake_address().to_owned();
    let engine = thread::spawn(move || {
        let mut ready = ready();
        ready.declare_scheduler();
        EngineTransport::connect(&handshake, 0, ready, Duration::from_secs(20)).unwrap()
    });
    let client: EngineClient = listener.accept().await.unwrap();
    let transport = engine.join().unwrap();
    let registered = &client.engines()[0].ready;
    assert!(registered.takes(REQUEST_CONTROL));
    assert!(!registered.takes(REQUEST_ENCODE));
    let layouts: Vec<&str> = registered
        .positional_layouts
        .0
        .iter()
        .map(|(name, _)| name.as_str())
        .collect();
    assert!(layouts.contains(&"BatchTokenIDOutSlim"), "{layouts:?}");
    assert!(layouts.contains(&"ControlReplySlim"), "{layouts:?}");
    assert_eq!(registered.bootstrap_port, Some(8998));
    // A transport closes its sockets on its own runtime, off this one.
    thread::spawn(move || drop(transport)).join().unwrap();
}

/// Each rank of an attention-DP group says the group's size in its HELLO,
/// so a frontend that was not told how many engines dial it awaits them all.
#[tokio::test(flavor = "multi_thread")]
async fn a_ranks_hello_says_how_many_engines_dial_with_it() {
    use zeromq::{Socket, SocketRecv};

    let handshake_address = ipc("hello");
    let mut handshake = zeromq::RouterSocket::new();
    handshake.bind(&handshake_address).await.unwrap();
    let engine = thread::spawn(move || {
        let mut ready = ready();
        ready.data_parallel_size = 4;
        ready.declare_scheduler();
        // No INIT comes back: the transport gives up once the HELLO is out.
        EngineTransport::connect(&handshake_address, 1, ready, Duration::from_millis(500))
    });
    let frames = handshake.recv().await.unwrap().into_vec();
    let hello: ts_wire::ReadyMessage = ts_wire::decode_frames(&frames[1..]).unwrap();
    assert_eq!(
        hello.status.as_deref(),
        Some(ts_wire::handshake::STATUS_HELLO)
    );
    assert_eq!(hello.engine_count, Some(4));
    assert!(engine.join().unwrap().is_err());
}

#[tokio::test(flavor = "multi_thread")]
async fn an_encode_server_dials_its_frontend_and_takes_its_encode_requests() {
    let listener = listener().await;
    let handshake = listener.handshake_address().to_owned();
    let engine = thread::spawn(move || {
        let mut ready = ready();
        ready.declare_encoder();
        EncodeIntake::connect(&handshake, 0, ready, Duration::from_secs(20)).unwrap()
    });
    let client = listener.accept().await.unwrap();
    let mut intake = engine.join().unwrap();
    assert!(client.engines()[0].ready.takes(REQUEST_ENCODE));

    let request = EncodeRequest {
        rid: None,
        http_worker_ipc: None,
        request_id: "item-0".to_owned(),
        bootstrap_host: "10.0.0.9".to_owned(),
        bootstrap_port: 8998,
        bootstrap_room: 41,
        items: Vec::new(),
    };
    client.encode(request.clone()).await.unwrap();
    let deadline = Instant::now() + Duration::from_secs(10);
    let received = loop {
        let received = intake.receive(Duration::from_millis(100));
        if !received.is_empty() || Instant::now() > deadline {
            break received;
        }
    };
    let [frames] = received.as_slice() else {
        panic!("one encode request, got {}", received.len());
    };
    let decoded: EncodeRequest = ts_wire::decode_frames(frames).unwrap();
    assert_eq!(decoded, request);

    // The server going away ends the client's wait on it.
    thread::spawn(move || drop(intake)).join().unwrap();
    let deadline = Instant::now() + Duration::from_secs(10);
    while client.encode(request.clone()).await.is_ok() {
        assert!(
            Instant::now() < deadline,
            "the client noticed the server is gone"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

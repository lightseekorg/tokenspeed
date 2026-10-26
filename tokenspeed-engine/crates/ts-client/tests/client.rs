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

//! The client against fake engines over real ZMQ sockets.

use std::time::Duration;

use bytes::Bytes;
use ts_client::{
    fake::{self, ipc_address, FakeEncodeIntake, FakeEngine, Row, ServingFake},
    ClientConfig, ClientListener, EncodeClient, EncodeConfig, EngineClient, EngineLoad, Error,
    LoadBalance, OutputChunk, RequestStream, Routing, StreamError,
};
use ts_wire::{
    control::{FlushCacheReqInput, IsSchedulerPausedReqInput},
    ControlOutput, ControlRequest, Dict, EncodeRequest, EngineOutput, EngineRequest, Modality,
    MultimodalDataItem, SamplingParams, Tensor, TokenizedGenerateReqInput,
};

fn config(handshake_address: &str, num_engines: u16, routing: Routing) -> ClientConfig {
    ClientConfig {
        handshake_address: handshake_address.to_owned(),
        input_address: ipc_address("input"),
        output_address: ipc_address("output"),
        first_engine_index: 0,
        num_engines,
        routing,
        handshake_timeout: Duration::from_secs(10),
        send_timeout: Duration::from_secs(5),
    }
}

/// A serving fake engine `index` of `num_engines`, dialing before the
/// client's handshake completes.
fn start(handshake_address: &str, index: u16, dp_rank: i64, dp_size: i64) -> ServingFake {
    fake::spawn_serving(
        handshake_address,
        index,
        FakeEngine::ready_response(dp_rank, dp_size),
    )
}

async fn connect(num_engines: u16, routing: Routing) -> (EngineClient, Vec<ServingFake>) {
    let handshake_address = ipc_address("handshake");
    let client = tokio::spawn(EngineClient::connect(config(
        &handshake_address,
        num_engines,
        routing,
    )));
    let engines = (0..num_engines)
        .map(|index| {
            start(
                &handshake_address,
                index,
                i64::from(index),
                i64::from(num_engines),
            )
        })
        .collect();
    (client.await.unwrap().unwrap(), engines)
}

/// A request; `max_new_tokens = None` keeps it running until aborted.
fn request(rid: &str, max_new_tokens: Option<i64>) -> TokenizedGenerateReqInput {
    let sampling_params = SamplingParams {
        max_new_tokens,
        ..Default::default()
    };
    TokenizedGenerateReqInput {
        input_ids: Some(vec![1, 2, 3]),
        stream: true,
        ..TokenizedGenerateReqInput::new(rid, sampling_params)
    }
}

async fn next(stream: &mut RequestStream) -> Option<Result<OutputChunk, StreamError>> {
    tokio::time::timeout(Duration::from_secs(5), stream.next())
        .await
        .expect("output in time")
}

async fn collect(mut stream: RequestStream) -> Vec<OutputChunk> {
    let mut chunks = Vec::new();
    while let Some(chunk) = next(&mut stream).await {
        chunks.push(chunk.unwrap());
    }
    chunks
}

fn tokens(chunks: &[OutputChunk]) -> Vec<i32> {
    chunks.iter().flat_map(|c| c.output_ids.clone()).collect()
}

#[tokio::test]
async fn streams_tokens_until_finished() {
    let (client, _engines) = connect(1, Routing::Aggregated(LoadBalance::RoundRobin)).await;
    assert_eq!(client.engines()[0].ready.max_model_len, 4096);

    let chunks = collect(client.generate(request("r1", Some(3))).await.unwrap()).await;
    assert_eq!(tokens(&chunks), [1, 2, 3]);
    let last = chunks.last().unwrap();
    assert_eq!(last.finish_reason.as_deref(), Some("length"));
    assert_eq!((last.prompt_tokens, last.completion_tokens), (3, 3));
    assert!(chunks[..2].iter().all(|c| c.finish_reason.is_none()));
    assert_eq!(client.in_flight(), [0]);
    assert_eq!(client.loads()[0].kv_total_pages, 100);
}

#[tokio::test]
async fn pinned_requests_go_to_their_rank() {
    let (client, _engines) = connect(2, Routing::Aggregated(LoadBalance::RoundRobin)).await;
    for _ in 0..2 {
        let pinned = TokenizedGenerateReqInput {
            data_parallel_rank: Some(1),
            ..request(&format!("pinned-{}", client.in_flight()[1]), Some(1))
        };
        let chunks = collect(client.generate(pinned).await.unwrap()).await;
        assert_eq!(tokens(&chunks), [1001]);
    }
    let out_of_range = TokenizedGenerateReqInput {
        data_parallel_rank: Some(2),
        ..request("bad-pin", Some(1))
    };
    assert!(matches!(
        client.generate(out_of_range).await,
        Err(Error::InvalidRequest(_))
    ));
}

#[tokio::test]
async fn round_robin_cycles_through_engines() {
    let (client, _engines) = connect(2, Routing::Aggregated(LoadBalance::RoundRobin)).await;
    let mut first_tokens = Vec::new();
    for i in 0..4 {
        let chunks = collect(
            client
                .generate(request(&format!("r{i}"), Some(1)))
                .await
                .unwrap(),
        )
        .await;
        first_tokens.push(tokens(&chunks)[0]);
    }
    assert_eq!(first_tokens, [1, 1001, 1, 1001]);
}

#[tokio::test]
async fn least_in_flight_spreads_running_requests() {
    let (client, _engines) = connect(2, Routing::Aggregated(LoadBalance::LeastInFlight)).await;
    let mut a = client.generate(request("a", None)).await.unwrap();
    let mut b = client.generate(request("b", None)).await.unwrap();
    let mut served = vec![
        next(&mut a).await.unwrap().unwrap().output_ids[0],
        next(&mut b).await.unwrap().unwrap().output_ids[0],
    ];
    served.sort();
    assert_eq!(served, [1, 1001]);
    assert_eq!(client.in_flight(), [1, 1]);
}

#[tokio::test]
async fn bootstrap_room_selects_the_rank() {
    let (client, _engines) = connect(2, Routing::BootstrapRoom).await;
    let with_room = TokenizedGenerateReqInput {
        bootstrap_room: Some(7),
        ..request("room", Some(1))
    };
    let chunks = collect(client.generate(with_room).await.unwrap()).await;
    assert_eq!(tokens(&chunks), [1001]);
    assert!(matches!(
        client.generate(request("no-room", Some(1))).await,
        Err(Error::InvalidRequest(_))
    ));
}

#[tokio::test]
async fn a_pinned_decode_request_goes_to_its_rank_not_the_rooms() {
    let (client, _engines) = connect(2, Routing::BootstrapRoom).await;
    // Room 7 names rank 1; the pin wins.
    let pinned = TokenizedGenerateReqInput {
        bootstrap_room: Some(7),
        data_parallel_rank: Some(0),
        ..request("pinned", Some(1))
    };
    let chunks = collect(client.generate(pinned).await.unwrap()).await;
    assert_eq!(tokens(&chunks), [1]);
    let out_of_range = TokenizedGenerateReqInput {
        bootstrap_room: Some(7),
        data_parallel_rank: Some(2),
        ..request("bad-pin", Some(1))
    };
    assert!(matches!(
        client.generate(out_of_range).await,
        Err(Error::InvalidRequest(_))
    ));
    // A pin does not replace the room the engines bootstrap on.
    let no_room = TokenizedGenerateReqInput {
        data_parallel_rank: Some(0),
        ..request("pin-no-room", Some(1))
    };
    assert!(matches!(
        client.generate(no_room).await,
        Err(Error::InvalidRequest(_))
    ));
}

#[tokio::test]
async fn abort_finishes_the_stream() {
    let (client, _engines) = connect(1, Routing::Aggregated(LoadBalance::RoundRobin)).await;
    let mut stream = client.generate(request("r1", None)).await.unwrap();
    assert_eq!(next(&mut stream).await.unwrap().unwrap().output_ids, [1]);
    assert!(client.abort("r1").await.unwrap());
    let last = next(&mut stream).await.unwrap().unwrap();
    assert_eq!(last.finish_reason.as_deref(), Some("abort"));
    assert!(next(&mut stream).await.is_none());
    assert!(!client.abort("r1").await.unwrap());
    assert_eq!(client.in_flight(), [0]);
}

#[tokio::test]
async fn dropping_a_stream_aborts_the_request() {
    let (client, mut engines) = connect(1, Routing::Aggregated(LoadBalance::RoundRobin)).await;
    let mut stream = client.generate(request("r1", None)).await.unwrap();
    next(&mut stream).await.unwrap().unwrap();
    drop(stream);
    assert!(matches!(
        engines[0].next_received().await.unwrap(),
        EngineRequest::Add(_)
    ));
    assert_eq!(
        engines[0].next_received().await.unwrap(),
        EngineRequest::Abort(vec!["r1".to_owned()])
    );
    assert_eq!(client.in_flight(), [0]);
    // The id is free again.
    collect(client.generate(request("r1", Some(1))).await.unwrap()).await;
}

#[tokio::test]
async fn engine_shutdown_ends_every_stream() {
    let (client, engines) = connect(2, Routing::Aggregated(LoadBalance::RoundRobin)).await;
    let mut a = client.generate(request("a", None)).await.unwrap();
    let mut b = client.generate(request("b", None)).await.unwrap();
    next(&mut a).await.unwrap().unwrap();
    next(&mut b).await.unwrap().unwrap();

    engines[0].shut_down();
    assert_eq!(next(&mut a).await, Some(Err(StreamError::EngineDead)));
    assert_eq!(next(&mut b).await, Some(Err(StreamError::EngineDead)));
    assert!(next(&mut a).await.is_none());
    assert!(matches!(
        client.generate(request("c", Some(1))).await,
        Err(Error::Unavailable(StreamError::EngineDead))
    ));
    assert_eq!(client.in_flight(), [0, 0]);
}

#[tokio::test]
async fn duplicate_request_ids_are_rejected() {
    let (client, _engines) = connect(1, Routing::Aggregated(LoadBalance::RoundRobin)).await;
    let _running = client.generate(request("r1", None)).await.unwrap();
    assert!(matches!(
        client.generate(request("r1", None)).await,
        Err(Error::DuplicateRequestId(rid)) if rid == "r1"
    ));
}

#[tokio::test]
async fn dropping_the_client_ends_streams() {
    let (client, _engines) = connect(1, Routing::Aggregated(LoadBalance::RoundRobin)).await;
    let mut stream = client.generate(request("r1", None)).await.unwrap();
    next(&mut stream).await.unwrap().unwrap();
    drop(client);
    assert_eq!(
        next(&mut stream).await,
        Some(Err(StreamError::ClientClosed))
    );
}

#[tokio::test]
async fn handshake_times_out_naming_missing_engines() {
    let handshake_address = ipc_address("handshake");
    let mut config = config(
        &handshake_address,
        2,
        Routing::Aggregated(LoadBalance::RoundRobin),
    );
    config.handshake_timeout = Duration::from_millis(500);
    let client = tokio::spawn(EngineClient::connect(config));
    let _only_one = start(&handshake_address, 0, 0, 2);
    match client.await.unwrap() {
        Err(Error::HandshakeTimeout { stage, pending, .. }) => {
            assert_eq!((stage, pending), ("READY", vec![1]));
        }
        other => panic!("expected a handshake timeout, got {:?}", other.err()),
    }
}

#[tokio::test]
async fn handshake_rejects_a_mismatched_dp_layout() {
    let handshake_address = ipc_address("handshake");
    let client = tokio::spawn(EngineClient::connect(config(
        &handshake_address,
        1,
        Routing::Aggregated(LoadBalance::RoundRobin),
    )));
    // The engine believes it is rank 0 of 2; the client expects one engine.
    let _engine = start(&handshake_address, 0, 0, 2);
    assert!(matches!(client.await.unwrap(), Err(Error::Handshake(_))));
}

#[tokio::test]
async fn binding_port_zero_reports_the_chosen_port() {
    let listener = ClientListener::bind(ClientConfig {
        handshake_address: "tcp://127.0.0.1:0".to_owned(),
        input_address: "tcp://127.0.0.1:0".to_owned(),
        output_address: "tcp://127.0.0.1:0".to_owned(),
        ..config("unused", 1, Routing::Aggregated(LoadBalance::RoundRobin))
    })
    .await
    .unwrap();
    let address = listener.handshake_address().to_owned();
    let port: u16 = address.rsplit(':').next().unwrap().parse().unwrap();
    assert_ne!(port, 0, "{address}");

    // An engine started after binding finds the client at that address.
    let _engine = start(&address, 0, 0, 1);
    let client = listener.accept().await.unwrap();
    let chunks = collect(client.generate(request("r1", Some(2))).await.unwrap()).await;
    assert_eq!(tokens(&chunks), [1, 2]);
}

#[tokio::test]
async fn a_control_operation_gathers_every_engines_reply() {
    let (client, mut engines) = connect(2, Routing::Aggregated(LoadBalance::RoundRobin)).await;
    let flush = ControlRequest::FlushCache(FlushCacheReqInput {
        rid: None,
        http_worker_ipc: None,
    });
    let replies = client.control(flush.clone()).await.unwrap();
    let mut successes: Vec<bool> = replies
        .iter()
        .map(|reply| match reply {
            ControlOutput::FlushCache(output) => output.success,
            other => panic!("unexpected {}", other.tag()),
        })
        .collect();
    successes.sort();
    assert_eq!(successes, [false, true]);
    // Every engine got the operation.
    for engine in &mut engines {
        assert_eq!(
            engine.next_received().await.unwrap(),
            EngineRequest::Control(Box::new(flush.clone()))
        );
    }

    // Operations of different kinds may be out together.
    let paused = ControlRequest::IsSchedulerPaused(IsSchedulerPausedReqInput {
        rid: None,
        http_worker_ipc: None,
    });
    let (a, b) = tokio::join!(client.control(paused.clone()), client.control(flush));
    assert_eq!(a.unwrap().len(), 2);
    assert_eq!(b.unwrap().len(), 2);
}

#[tokio::test]
async fn a_control_operation_fails_when_an_engine_dies() {
    let (client, engines) = connect(2, Routing::Aggregated(LoadBalance::RoundRobin)).await;
    engines[1].shut_down();
    // Let the client see the shutdown sentinel first.
    tokio::time::sleep(Duration::from_millis(200)).await;
    let flush = ControlRequest::FlushCache(FlushCacheReqInput {
        rid: None,
        http_worker_ipc: None,
    });
    assert!(matches!(
        client.control(flush).await,
        Err(Error::Unavailable(StreamError::EngineDead))
    ));
}

#[tokio::test]
async fn a_scripted_fake_answers_what_the_test_sends() {
    let handshake_address = ipc_address("handshake");
    let client = tokio::spawn(EngineClient::connect(config(
        &handshake_address,
        1,
        Routing::Aggregated(LoadBalance::RoundRobin),
    )));
    let mut engine = FakeEngine::connect(&handshake_address, 0, FakeEngine::ready_response(0, 1))
        .await
        .unwrap();
    let client = client.await.unwrap().unwrap();
    assert_eq!(engine.index(), 0);
    assert_eq!(engine.init().addresses.inputs.len(), 1);

    let mut stream = client.generate(request("r1", Some(2))).await.unwrap();
    let EngineRequest::Add(received) = engine.recv().await.unwrap() else {
        panic!("expected the ADD");
    };
    assert_eq!(received.input_ids, Some(vec![1, 2, 3]));
    let load = EngineLoad {
        num_running: 1,
        num_waiting: 0,
        kv_active_pages: 3,
        kv_total_pages: 8,
    };
    let rows = vec![Row {
        rid: "r1",
        output_ids: vec![7, 8],
        finish_reason: Some("stop"),
        prompt_tokens: 3,
        completion_tokens: 2,
    }];
    engine.send_batch(fake::batch(0, rows, load)).await.unwrap();
    let chunk = next(&mut stream).await.unwrap().unwrap();
    assert_eq!(chunk.output_ids, [7, 8]);
    assert_eq!(chunk.finish_reason.as_deref(), Some("stop"));
    assert_eq!(client.loads(), [load]);

    // A scripted control reply: the operation completes once the test
    // answers it; then the shutdown sentinel.
    let mut flush = std::pin::pin!(client.control(ControlRequest::FlushCache(
        FlushCacheReqInput {
            rid: None,
            http_worker_ipc: None,
        }
    )));
    let received = tokio::select! {
        received = engine.recv() => received.unwrap(),
        _ = &mut flush => panic!("a control operation completed before its reply"),
    };
    assert!(matches!(received, EngineRequest::Control(_)));
    engine
        .send(&EngineOutput::Control(Box::new(ControlOutput::FlushCache(
            ts_wire::control::FlushCacheReqOutput {
                rid: None,
                http_worker_ipc: None,
                success: true,
            },
        ))))
        .await
        .unwrap();
    let replies = flush.await.unwrap();
    assert!(matches!(
        replies.as_slice(),
        [ControlOutput::FlushCache(reply)] if reply.success
    ));

    // The sentinel ends a waiting stream, and the client takes nothing more.
    let mut waiting = client.generate(request("r2", None)).await.unwrap();
    assert!(matches!(
        engine.recv().await.unwrap(),
        EngineRequest::Add(_)
    ));
    engine.shut_down().await.unwrap();
    assert_eq!(next(&mut waiting).await, Some(Err(StreamError::EngineDead)));
    assert!(matches!(
        client.generate(request("r3", Some(1))).await,
        Err(Error::Unavailable(StreamError::EngineDead))
    ));
}

fn encode_request(request_id: &str, feature_bytes: usize) -> EncodeRequest {
    EncodeRequest {
        rid: None,
        http_worker_ipc: None,
        request_id: request_id.to_owned(),
        bootstrap_host: "10.0.0.5".to_owned(),
        bootstrap_port: 9001,
        bootstrap_room: 42,
        items: vec![MultimodalDataItem {
            modality: Modality::Image,
            hash: Some(7),
            pad_value: Some(151700),
            offsets: Some(vec![(3, 66)]),
            feature: Some(Tensor {
                dtype: "uint8".to_owned(),
                shape: vec![feature_bytes as i64],
                data: Bytes::from(vec![9u8; feature_bytes]),
            }),
            feature_shm: None,
            model_specific_data: Dict::default(),
            encoded: None,
            encoded_deepstack: None,
            encode_handshake: None,
        }],
    }
}

#[tokio::test]
async fn encode_requests_reach_the_intake() {
    let mut intake = FakeEncodeIntake::bind("tcp://127.0.0.1:0").await.unwrap();
    let client = EncodeClient::connect(EncodeConfig {
        intake_address: intake.endpoint().to_owned(),
        send_timeout: Duration::from_secs(5),
    })
    .await
    .unwrap();
    assert_eq!(client.intake_address(), intake.endpoint());

    // A small feature rides inline, a large one as its own frame; the intake
    // decodes both to the request as submitted.
    for (request_id, feature_bytes) in [("e-inline", 16), ("e-aux", 4096)] {
        let request = encode_request(request_id, feature_bytes);
        client.submit(request.clone()).await.unwrap();
        assert_eq!(intake.recv().await.unwrap(), request);
    }
}

#[tokio::test]
async fn an_encode_request_carries_one_item() {
    let intake = FakeEncodeIntake::bind("tcp://127.0.0.1:0").await.unwrap();
    let client = EncodeClient::connect(EncodeConfig {
        intake_address: intake.endpoint().to_owned(),
        send_timeout: Duration::from_secs(5),
    })
    .await
    .unwrap();
    let mut two_items = encode_request("e-2", 16);
    two_items.items.push(two_items.items[0].clone());
    assert!(matches!(
        client.submit(two_items).await,
        Err(Error::InvalidRequest(message)) if message.contains("one item")
    ));
    let mut no_item = encode_request("e-0", 16);
    no_item.items.clear();
    assert!(matches!(
        client.submit(no_item).await,
        Err(Error::InvalidRequest(_))
    ));
}

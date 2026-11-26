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

mod common;

use std::time::Duration;

use common::{ipc_address, start, FakeEngine, Role};
use ts_client::{
    ClientConfig, ClientListener, EngineClient, Error, LoadBalance, OutputChunk, RequestStream,
    Routing, StreamError,
};
use ts_wire::{
    control::{FlushCacheReqInput, IsSchedulerPausedReqInput},
    ControlOutput, ControlRequest, EncodeRequest, EngineRequest, SamplingParams,
    TokenizedGenerateReqInput,
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

async fn connect(num_engines: u16, routing: Routing) -> (EngineClient, Vec<FakeEngine>) {
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

async fn connect_silent(num_engines: u16) -> (EngineClient, Vec<FakeEngine>) {
    let handshake = ipc_address("silent-handshake");
    let listener = ClientListener::bind(config(
        &handshake,
        num_engines,
        Routing::Aggregated(LoadBalance::RoundRobin),
    ))
    .await
    .unwrap();
    let engines = (0..num_engines)
        .map(|index| {
            common::start_as(
                &handshake,
                index,
                i64::from(index),
                i64::from(num_engines),
                Role::SilentScheduler,
            )
        })
        .collect();
    (listener.accept().await.unwrap(), engines)
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
    assert!(client.fresh_loads(Duration::from_secs(3)).is_empty());

    let chunks = collect(client.generate(request("r1", Some(3))).await.unwrap()).await;
    assert_eq!(tokens(&chunks), [1, 2, 3]);
    let last = chunks.last().unwrap();
    assert_eq!(last.finish_reason.as_deref(), Some("length"));
    assert_eq!((last.prompt_tokens, last.completion_tokens), (3, 3));
    assert_eq!(
        chunks
            .iter()
            .map(|chunk| chunk.reasoning_tokens)
            .collect::<Vec<_>>(),
        [10, 20, 30]
    );
    assert!(chunks[..2].iter().all(|c| c.finish_reason.is_none()));
    assert!(chunks
        .iter()
        .all(|c| c.output_token_cumulative_probs.is_empty()));
    assert_eq!(client.in_flight(), [0]);
    assert_eq!(client.loads()[0].kv_total_pages, 100);
    let snapshots = client.fresh_loads(Duration::from_secs(3));
    assert_eq!(snapshots.len(), 1);
    assert_eq!(snapshots[0].0, 0);
    assert_eq!(snapshots[0].1.kv_used_pages, 7);
    assert_eq!(snapshots[0].1.kv_active_pages, 0);
    assert_eq!(snapshots[0].1.num_resident_tokens, Some(0));
    assert_eq!(snapshots[0].1.num_waiting_uncached_tokens, Some(0));
    assert!(client.fresh_loads(Duration::ZERO).is_empty());
}

#[tokio::test]
async fn selected_token_probabilities_survive_the_wire_and_owned_output_chunks() {
    let (client, _engines) = connect(1, Routing::Aggregated(LoadBalance::RoundRobin)).await;
    let chunks = collect(
        client
            .generate(TokenizedGenerateReqInput {
                return_logprob: true,
                ..request("scored", Some(3))
            })
            .await
            .unwrap(),
    )
    .await;
    assert_eq!(tokens(&chunks), [1, 2, 3]);
    assert_eq!(
        chunks
            .iter()
            .flat_map(|c| c.output_token_logprobs.iter().copied())
            .collect::<Vec<_>>(),
        [-0.1, -0.2, -0.3],
    );
    assert_eq!(
        chunks
            .iter()
            .flat_map(|c| c.output_token_cumulative_probs.iter().copied())
            .collect::<Vec<_>>(),
        [0.1, 0.2, 0.3],
    );
    assert_eq!(
        chunks.last().unwrap().finish_reason.as_deref(),
        Some("length")
    );
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
        assert!(client.fresh_loads(Duration::from_secs(3)).is_empty());
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
async fn abort_preserves_a_queued_engine_finish() {
    let (client, _engines) = connect(1, Routing::Aggregated(LoadBalance::RoundRobin)).await;
    let stream = client.generate(request("r1", Some(1))).await.unwrap();
    collect(client.generate(request("barrier", Some(1))).await.unwrap()).await;
    assert!(!client.abort("r1").await.unwrap());
    let chunks = collect(stream).await;
    assert_eq!(tokens(&chunks), [1]);
    assert_eq!(chunks[0].finish_reason.as_deref(), Some("length"));
    assert_eq!(client.in_flight(), [0]);
}

#[tokio::test]
async fn cancel_closes_a_silent_abort_stream_and_preserves_survivors_and_controls() {
    let (client, mut engines) = connect_silent(2).await;
    let mut cancelled = client.generate(request("cancelled", None)).await.unwrap();
    let mut survivor = client.generate(request("survivor", None)).await.unwrap();
    assert_eq!(next(&mut cancelled).await.unwrap().unwrap().output_ids, [1]);
    assert_eq!(
        next(&mut survivor).await.unwrap().unwrap().output_ids,
        [1001]
    );
    assert!(client.cancel("cancelled").await.unwrap());
    assert_eq!(
        next(&mut cancelled).await,
        Some(Err(StreamError::Cancelled))
    );
    assert_eq!(next(&mut cancelled).await, None);
    assert_eq!(client.in_flight(), [0, 1]);
    assert!(matches!(
        engines[0].next_received().await,
        EngineRequest::Add(_)
    ));
    assert_eq!(
        engines[0].next_received().await,
        EngineRequest::Abort(vec!["cancelled".into()])
    );
    assert!(!client.cancel("cancelled").await.unwrap());
    assert!(!client.cancel("unknown").await.unwrap());

    // Outputs already in transit may arrive after local closure.
    engines[0].output("cancelled", 2, None).await;
    engines[0].output("cancelled", 3, Some("length")).await;
    engines[1].output("survivor", 2, None).await;
    assert_eq!(
        next(&mut survivor).await.unwrap().unwrap().output_ids,
        [1002]
    );
    assert_eq!(client.in_flight(), [0, 1]);
    let replies = client
        .control(ControlRequest::IsSchedulerPaused(
            IsSchedulerPausedReqInput {
                rid: None,
                http_worker_ipc: None,
            },
        ))
        .await
        .unwrap();
    assert_eq!(replies.len(), 2);
    engines[1].output("survivor", 3, Some("length")).await;
    let last = next(&mut survivor).await.unwrap().unwrap();
    assert_eq!(last.output_ids, [1003]);
    assert_eq!(last.finish_reason.as_deref(), Some("length"));
    assert_eq!(next(&mut survivor).await, None);
    assert_eq!(client.in_flight(), [0, 0]);
}

#[tokio::test]
async fn abort_closes_a_silent_stream_after_queued_outputs() {
    let (client, mut engines) = connect_silent(1).await;
    let mut stream = client
        .generate(TokenizedGenerateReqInput {
            return_logprob: true,
            ..request("r", None)
        })
        .await
        .unwrap();
    collect(client.generate(request("barrier", Some(1))).await.unwrap()).await;
    assert_eq!(client.in_flight(), [1]);
    assert!(client.abort("r").await.unwrap());
    assert_eq!(client.in_flight(), [0]);
    assert!(!client.abort("r").await.unwrap());
    for _ in 0..2 {
        assert!(matches!(
            engines[0].next_received().await,
            EngineRequest::Add(_)
        ));
    }
    assert_eq!(
        engines[0].next_received().await,
        EngineRequest::Abort(vec!["r".into()])
    );
    let first = next(&mut stream).await.unwrap().unwrap();
    assert_eq!(first.output_ids, [1]);
    assert_eq!(first.finish_reason, None);
    assert_eq!(first.output_token_logprobs, [-0.1]);
    let terminal = next(&mut stream).await.unwrap().unwrap();
    assert_eq!(terminal.finish_reason.as_deref(), Some("abort"));
    assert_eq!(terminal.prompt_tokens, first.prompt_tokens);
    assert_eq!(terminal.completion_tokens, first.completion_tokens);
    assert_eq!(terminal.cached_tokens, first.cached_tokens);
    assert_eq!(terminal.reasoning_tokens, first.reasoning_tokens);
    assert!(terminal.output_ids.is_empty());
    assert!(terminal.output_token_logprobs.is_empty());
    assert!(terminal.output_token_cumulative_probs.is_empty());
    assert_eq!(next(&mut stream).await, None);
}

#[tokio::test]
async fn abort_before_the_first_output_reports_the_submitted_prompt_length() {
    let (client, _engines) = connect_silent(1).await;
    let stream = client.generate(request("r", Some(0))).await.unwrap();
    assert!(client.abort("r").await.unwrap());
    let chunks = collect(stream).await;
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].finish_reason.as_deref(), Some("abort"));
    assert_eq!(chunks[0].prompt_tokens, 3);
    assert_eq!(chunks[0].completion_tokens, 0);
    assert!(chunks[0].output_ids.is_empty());
    assert_eq!(client.in_flight(), [0]);
}

#[tokio::test]
async fn concurrent_cancels_produce_one_result_after_queued_outputs() {
    let (client, _engines) = connect_silent(1).await;
    let mut stream = client.generate(request("r", None)).await.unwrap();
    // A real output is queued before cancellation, but need not be consumed.
    collect(client.generate(request("barrier", Some(1))).await.unwrap()).await;
    let (a, b) = tokio::join!(client.cancel("r"), client.cancel("r"));
    assert_ne!(a.unwrap(), b.unwrap());
    assert_eq!(next(&mut stream).await.unwrap().unwrap().output_ids, [1]);
    let mut cancellations = 0;
    while let Some(item) = next(&mut stream).await {
        match item {
            Ok(chunk) => assert_eq!(chunk.finish_reason, None),
            Err(StreamError::Cancelled) => cancellations += 1,
            other => panic!("unexpected {other:?}"),
        }
    }
    assert_eq!(cancellations, 1);
    assert_eq!(client.in_flight(), [0]);
}

#[tokio::test]
async fn dropping_send_futures_preserves_their_respective_owners() {
    use std::{
        future::{poll_fn, Future},
        task::Poll,
    };

    let (client, engines) = connect_silent(2).await;
    let mut stream = client.generate(request("r", None)).await.unwrap();
    next(&mut stream).await.unwrap().unwrap();
    engines[0].pause_input(true).await;

    // Hold the actual socket's send queue with a message larger than its
    // buffer. No fake send result or scheduling sleep determines the race.
    let large = TokenizedGenerateReqInput {
        data_parallel_rank: Some(0),
        input_ids: Some(vec![i32::MAX; 1_000_000]),
        ..request("interrupted", None)
    };
    let mut submission = Box::pin(client.generate(large));
    assert!(poll_fn(|cx| Poll::Ready(submission.as_mut().poll(cx)))
        .await
        .is_pending());
    assert_eq!(client.in_flight(), [2, 0]);
    let mut cancel = Box::pin(client.cancel("r"));
    assert!(poll_fn(|cx| Poll::Ready(cancel.as_mut().poll(cx)))
        .await
        .is_pending());
    drop(cancel);
    assert_eq!(client.in_flight(), [2, 0]);
    drop(submission);
    assert_eq!(client.in_flight(), [1, 0]);
    engines[0].pause_input(false).await;

    // A cancelled send may have partially reached the engine. Its stream
    // remains attached, and a genuine terminal output still belongs to it.
    engines[0].output("r", 2, Some("length")).await;
    assert_eq!(
        collect(stream).await[0].finish_reason.as_deref(),
        Some("length")
    );
    let survivor = TokenizedGenerateReqInput {
        data_parallel_rank: Some(1),
        ..request("survivor", Some(2))
    };
    assert_eq!(
        tokens(&collect(client.generate(survivor).await.unwrap()).await),
        [1001, 1002]
    );
    assert_eq!(client.in_flight(), [0, 0]);
}

#[tokio::test]
async fn a_finish_during_abort_send_preserves_a_later_local_owner() {
    use std::{
        future::{poll_fn, Future},
        task::Poll,
    };

    let (client, engines) = connect_silent(2).await;
    let mut old = client.generate(request("r", None)).await.unwrap();
    next(&mut old).await.unwrap().unwrap();
    engines[0].pause_input(true).await;
    let mut blocker = Box::pin(client.generate(TokenizedGenerateReqInput {
        data_parallel_rank: Some(0),
        input_ids: Some(vec![i32::MAX; 1_000_000]),
        ..request("blocker", None)
    }));
    assert!(poll_fn(|cx| Poll::Ready(blocker.as_mut().poll(cx)))
        .await
        .is_pending());
    let mut abort = Box::pin(client.abort("r"));
    assert!(poll_fn(|cx| Poll::Ready(abort.as_mut().poll(cx)))
        .await
        .is_pending());

    // Output completion wins while the abort send waits.
    engines[0].output("r", 2, Some("length")).await;
    tokio::time::timeout(Duration::from_secs(5), async {
        while client.in_flight()[0] != 1 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    // Deliberate reuse tests local cleanup only. Put the replacement on a
    // different engine so the old wire abort cannot name its remote owner.
    let mut replacement = Box::pin(client.generate(TokenizedGenerateReqInput {
        data_parallel_rank: Some(1),
        ..request("r", None)
    }));
    let first_poll = poll_fn(|cx| Poll::Ready(replacement.as_mut().poll(cx))).await;
    assert_eq!(client.in_flight(), [1, 1]);
    drop(old); // The abort still keeps the old sender alive.
    assert_eq!(client.in_flight(), [1, 1]);
    drop(blocker);
    engines[0].pause_input(false).await;
    assert!(abort.await.unwrap());
    let mut later = match first_poll {
        Poll::Ready(result) => result.unwrap(),
        Poll::Pending => replacement.await.unwrap(),
    };
    assert_eq!(next(&mut later).await.unwrap().unwrap().output_ids, [1001]);
    assert_eq!(client.in_flight(), [0, 1]);
    engines[1].output("r", 2, Some("length")).await;
    assert_eq!(collect(later).await[0].output_ids, [1002]);
    assert_eq!(client.in_flight(), [0, 0]);
}

#[tokio::test]
async fn finish_and_cancel_have_exactly_one_terminal_result() {
    let (client, engines) = connect_silent(1).await;
    for i in 0..20 {
        let rid = format!("race-{i}");
        let mut stream = client.generate(request(&rid, None)).await.unwrap();
        next(&mut stream).await.unwrap().unwrap();
        let (cancelled, ()) = tokio::join!(
            client.cancel(&rid),
            engines[0].output(&rid, 2, Some("length"))
        );
        let terminal = next(&mut stream).await.unwrap();
        if cancelled.unwrap() {
            assert_eq!(terminal, Err(StreamError::Cancelled));
        } else {
            assert_eq!(terminal.unwrap().finish_reason.as_deref(), Some("length"));
        }
        assert_eq!(next(&mut stream).await, None);
        assert!(!client.cancel(&rid).await.unwrap());
    }
    assert_eq!(client.in_flight(), [0]);
}

#[tokio::test]
async fn old_stream_cleanup_cannot_remove_a_later_admission() {
    let (client, mut engines) = connect_silent(1).await;
    let old = client.generate(request("r", None)).await.unwrap();
    assert!(client.cancel("r").await.unwrap());
    assert!(matches!(
        engines[0].next_received().await,
        EngineRequest::Add(_)
    ));
    assert_eq!(
        engines[0].next_received().await,
        EngineRequest::Abort(vec!["r".into()])
    );
    // The test engine has retired the old request. Deliberately reuse its id
    // to test local ownership; real callers must use fresh wire ids.
    let mut later = client.generate(request("r", None)).await.unwrap();
    next(&mut later).await.unwrap().unwrap();
    drop(old); // Its queued cancellation has not been consumed.
    assert_eq!(client.in_flight(), [1]);
    engines[0].output("r", 2, Some("length")).await;
    assert_eq!(
        next(&mut later)
            .await
            .unwrap()
            .unwrap()
            .finish_reason
            .as_deref(),
        Some("length")
    );
    assert_eq!(next(&mut later).await, None);

    // A finish received by dispatch also precedes the consumer seeing it.
    let finished = client.generate(request("finished", Some(1))).await.unwrap();
    let barrier = client.generate(request("barrier", Some(1))).await.unwrap();
    collect(barrier).await;
    assert!(!client.cancel("finished").await.unwrap());
    let mut replacement = client.generate(request("finished", None)).await.unwrap();
    next(&mut replacement).await.unwrap().unwrap();
    drop(finished);
    assert_eq!(client.in_flight(), [1]);
    engines[0].output("finished", 2, Some("length")).await;
    assert_eq!(collect(replacement).await[0].output_ids, [2]);
}

#[tokio::test]
async fn send_failure_retains_cancel_ownership_and_cleans_failed_submissions() {
    let (client, engines) = connect_silent(2).await;
    let mut stream = client.generate(request("r", None)).await.unwrap();
    let mut survivor = client.generate(request("survivor", None)).await.unwrap();
    next(&mut stream).await.unwrap().unwrap();
    next(&mut survivor).await.unwrap().unwrap();
    engines[0].disconnect_input().await;
    // Wait for ZMQ's peer bookkeeping, using actual failed sends as evidence.
    tokio::time::timeout(Duration::from_secs(5), async {
        for i in 0.. {
            let probe = TokenizedGenerateReqInput {
                data_parallel_rank: Some(0),
                ..request(&format!("probe-{i}"), None)
            };
            if client.generate(probe).await.is_err() {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    assert!(matches!(
        client.cancel("r").await,
        Err(Error::Zmq(_)) | Err(Error::SendTimeout { .. })
    ));
    assert!(matches!(
        client.abort("r").await,
        Err(Error::Zmq(_)) | Err(Error::SendTimeout { .. })
    ));
    assert_eq!(client.in_flight(), [1, 1]);
    assert!(matches!(
        client
            .generate(TokenizedGenerateReqInput {
                data_parallel_rank: Some(1),
                ..request("r", None)
            })
            .await,
        Err(Error::DuplicateRequestId(_))
    ));
    // The failed cancel did not replace a stream result or lose its owner.
    engines[0].output("r", 2, Some("length")).await;
    assert_eq!(
        next(&mut stream)
            .await
            .unwrap()
            .unwrap()
            .finish_reason
            .as_deref(),
        Some("length")
    );
    assert_eq!(next(&mut stream).await, None);
    engines[1].output("survivor", 2, Some("length")).await;
    assert_eq!(collect(survivor).await[0].output_ids, [1002]);
    assert_eq!(client.in_flight(), [0, 0]);
}

#[tokio::test]
async fn dropping_a_stream_aborts_the_request() {
    let (client, mut engines) = connect(1, Routing::Aggregated(LoadBalance::RoundRobin)).await;
    let mut stream = client.generate(request("r1", None)).await.unwrap();
    next(&mut stream).await.unwrap().unwrap();
    drop(stream);
    assert!(matches!(
        engines[0].next_received().await,
        EngineRequest::Add(_)
    ));
    assert_eq!(
        engines[0].next_received().await,
        EngineRequest::Abort(vec!["r1".to_owned()])
    );
    assert_eq!(client.in_flight(), [0]);
    // A later submission uses a fresh wire id.
    collect(client.generate(request("r2", Some(1))).await.unwrap()).await;
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
            engine.next_received().await,
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

fn encode_request(request_id: &str) -> EncodeRequest {
    EncodeRequest {
        rid: None,
        http_worker_ipc: None,
        request_id: request_id.to_owned(),
        bootstrap_host: "10.0.0.9".to_owned(),
        bootstrap_port: 8998,
        bootstrap_room: 41,
        items: Vec::new(),
    }
}

#[tokio::test]
async fn an_encode_server_takes_encode_requests_and_nothing_else() {
    let handshake = ipc_address("encode-hs");
    let listener = ClientListener::bind(config(
        &handshake,
        1,
        Routing::Aggregated(LoadBalance::RoundRobin),
    ))
    .await
    .unwrap();
    let mut engine = common::start_as(&handshake, 0, 0, 1, Role::Encoder);
    let client = listener.accept().await.unwrap();

    client.encode(encode_request("item-0")).await.unwrap();
    assert_eq!(
        engine.next_received().await,
        EngineRequest::Encode(Box::new(encode_request("item-0")))
    );
    let Err(error) = client.generate(request("r0", Some(1))).await else {
        panic!("an encode server takes no generate request");
    };
    assert!(matches!(error, Error::InvalidRequest(_)), "{error}");
    // Nor a control operation, which it would never answer: the call is
    // refused at once instead of waiting for a reply.
    let control = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        client.control(ControlRequest::FlushCache(FlushCacheReqInput {
            rid: None,
            http_worker_ipc: None,
        })),
    )
    .await
    .expect("refused before any wait");
    assert!(
        matches!(control, Err(Error::InvalidRequest(_))),
        "{control:?}"
    );
}

#[tokio::test]
async fn a_scheduler_takes_no_encode_requests() {
    let (client, _engines) = connect(1, Routing::Aggregated(LoadBalance::RoundRobin)).await;
    let error = client.encode(encode_request("item-0")).await.unwrap_err();
    assert!(matches!(error, Error::InvalidRequest(_)), "{error}");
}

#[test]
fn consuming_output_rows_preserves_request_timing_and_legacy_missing_values() {
    use ts_wire::BatchTokenIDOutSlim;
    let hex = include_str!("../../ts-wire/tests/fixtures/output_slim_timing.hex");
    let frames = hex
        .lines()
        .map(|line| {
            bytes::Bytes::from(
                (0..line.len())
                    .step_by(2)
                    .map(|i| u8::from_str_radix(&line[i..i + 2], 16).unwrap())
                    .collect::<Vec<u8>>(),
            )
        })
        .collect::<Vec<_>>();
    let mut batch: BatchTokenIDOutSlim = ts_wire::decode_frames(&frames).unwrap();
    let timing = batch.request_timing[0].take().unwrap();
    batch.request_timing.clear();
    let legacy = OutputChunk::from_batch(batch.clone())
        .unwrap()
        .collect::<Vec<_>>();
    assert!(legacy
        .iter()
        .all(|(_, chunk)| chunk.request_timing.is_none()));
    batch.request_timing = vec![None, Some(timing.clone())];
    let borrowed = batch
        .requests()
        .unwrap()
        .map(|row| (row.rid.to_owned(), OutputChunk::from(&row)))
        .collect::<Vec<_>>();
    let consumed = OutputChunk::from_batch(batch).unwrap().collect::<Vec<_>>();
    assert_eq!(consumed, borrowed);
    assert_eq!(consumed[1].1.request_timing, Some(timing));
    assert_eq!(consumed[0].1.request_timing, None);
}

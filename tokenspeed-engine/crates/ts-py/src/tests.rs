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

//! Exercise the Python awaitables over the real client and ZMQ sockets.

#[path = "../../ts-client/tests/common/mod.rs"]
mod common;

use std::ffi::CString;

use pyo3::{prelude::*, types::PyDict};
use ts_wire::{EngineRequest, SamplingParams, TokenizedGenerateReqInput};

use crate::{runtime, FrontendListener};

#[test]
fn python_abort_and_cancel_end_streams_with_a_silent_engine() {
    Python::initialize();
    Python::attach(|py| {
        let listener = FrontendListener::bind(
            py,
            common::ipc_address("py-handshake"),
            common::ipc_address("py-input"),
            common::ipc_address("py-output"),
            0,
            1,
            "round_robin",
            5.0,
            5.0,
        )
        .unwrap();
        let pending = listener.accept().unwrap();
        let engine = runtime().block_on(async {
            common::start_as(
                listener.handshake_address(),
                0,
                0,
                1,
                common::Role::SilentScheduler,
            )
        });
        let frontend = pending.wait(py).unwrap();
        let peer = runtime().spawn(async move {
            let mut engine = engine;
            for _ in 0..3 {
                assert!(matches!(
                    engine.next_received().await,
                    EngineRequest::Add(_)
                ));
            }
            assert_eq!(
                engine.next_received().await,
                EngineRequest::Abort(vec!["cancelled".into()])
            );
            assert_eq!(
                engine.next_received().await,
                EngineRequest::Abort(vec!["aborted".into()])
            );
            engine.output("survivor", 2, Some("length")).await;
            engine
        });
        let frames = |rid| {
            EngineRequest::Add(Box::new(TokenizedGenerateReqInput {
                input_ids: Some(vec![1, 2, 3]),
                stream: true,
                ..TokenizedGenerateReqInput::new(rid, SamplingParams::default())
            }))
            .to_frames()
            .into_iter()
            .skip(1)
            .map(|frame| frame.to_vec())
            .collect::<Vec<_>>()
        };
        let locals = PyDict::new(py);
        let native = PyModule::new(py, "tokenspeed_engine._native").unwrap();
        super::_native(&native).unwrap();
        locals.set_item("native", native).unwrap();
        locals.set_item("frontend", frontend).unwrap();
        locals
            .set_item("cancelled_frames", frames("cancelled"))
            .unwrap();
        locals
            .set_item("aborted_frames", frames("aborted"))
            .unwrap();
        locals
            .set_item("survivor_frames", frames("survivor"))
            .unwrap();
        let code = CString::new(
            r#"
import asyncio

async def exercise():
    await frontend.generate(cancelled_frames)
    await frontend.generate(aborted_frames)
    await frontend.generate(survivor_frames)
    outputs = {"cancelled": [], "aborted": [], "survivor": []}
    while not all(outputs.values()):
        for rid, chunk in await frontend.next_outputs():
            outputs[rid].append(chunk)
    assert await frontend.cancel("cancelled") is True
    assert await frontend.cancel("cancelled") is False
    assert await frontend.cancel("unknown") is False
    assert await frontend.abort("aborted") is True
    assert await frontend.abort("aborted") is False
    while not (isinstance(outputs["cancelled"][-1], native.RequestCancelled)
               and outputs["aborted"][-1].get("finish_reason") == "abort"
               and outputs["survivor"][-1].get("finish_reason") == "length"):
        for rid, chunk in await frontend.next_outputs():
            outputs[rid].append(chunk)
    cancelled = outputs["cancelled"]
    assert len(cancelled) == 2, cancelled
    assert cancelled[0]["output_ids"] == [1]
    assert isinstance(cancelled[1], native.EngineError)
    assert isinstance(cancelled[1], native.RequestCancelled)
    aborted = outputs["aborted"]
    assert len(aborted) == 2, aborted
    assert aborted[0]["output_ids"] == [1]
    assert aborted[1]["output_ids"] == []
    assert aborted[1]["prompt_tokens"] == 3
    assert aborted[1]["completion_tokens"] == 1
    survivor = outputs["survivor"]
    assert [token for chunk in survivor for token in chunk["output_ids"]] == [1, 2]
    assert survivor[-1]["completion_tokens"] == 2
    assert frontend.in_flight() == [0]
    assert await frontend.cancel("survivor") is False
    try:
        await asyncio.wait_for(frontend.next_outputs(), 0.05)
    except asyncio.TimeoutError:
        pass
    else:
        raise AssertionError("unexpected output after stream completion")

asyncio.run(asyncio.wait_for(exercise(), 5))
"#,
        )
        .unwrap();
        py.run(&code, Some(&locals), None).unwrap();
        // Keep the peer alive through the Python assertions.
        let _engine = py.detach(|| runtime().block_on(peer)).unwrap();
    });
}

#[test]
fn terminal_request_timing_preserves_typed_fields_in_python() {
    let frames = include_str!("../../ts-wire/tests/fixtures/output_slim_timing.hex")
        .lines()
        .map(|line| {
            bytes::Bytes::from(
                (0..line.len())
                    .step_by(2)
                    .map(|i| u8::from_str_radix(&line[i..i + 2], 16).unwrap())
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<Vec<_>>();
    let batch: ts_wire::BatchTokenIDOutSlim = ts_wire::decode_frames(&frames).unwrap();
    Python::initialize();
    Python::attach(|py| {
        let chunks = ts_client::OutputChunk::from_batch(batch)
            .unwrap()
            .map(|(_, chunk)| super::chunk_dict(py, chunk).unwrap())
            .collect::<Vec<_>>();
        let locals = PyDict::new(py);
        locals.set_item("chunks", chunks).unwrap();
        let code = CString::new(
            r#"
assert chunks[0]["completion_tokens"] == 1
assert chunks[0]["finish_reason"] == "stop"
assert chunks[0]["request_timing"] == {
    "role": "decode", "dp_rank": 2, "tp_rank": 3,
    "clock": {
        "host_id": "worker-a", "process_instance_id": "process-a",
        "clock_sync": "unknown", "clock_uncertainty_ns": "17",
    },
    "events": {
        "accepted": {"ts_unix_ns": "1700000000000000001", "ts_mono_ns": "9007199254740993"},
        "kv_ready": {"ts_unix_ns": "1700000000200000002", "ts_mono_ns": "9007199454740994"},
        "finished": {"ts_unix_ns": "1700000000500000003", "ts_mono_ns": "9007199754740995"},
    },
    "kv_output_tokens": 1, "e2e_latency_s": 0.75,
}
assert chunks[1]["request_timing"] is None
"#,
        )
        .unwrap();
        py.run(&code, Some(&locals), None).unwrap();
    });
}

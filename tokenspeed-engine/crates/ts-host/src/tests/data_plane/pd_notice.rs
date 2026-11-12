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

//! Independent decode allocation's one link between the scheduler and the
//! transfer peers: a Prefill scheduler's plans go through the data plane to
//! a Prefill peer, and a Decode peer hears of a request when the plan
//! carrying its prompt's last chunk is handed to the device.

use std::{
    thread,
    time::{Duration, Instant},
};

use ts_pd::{BootstrapInfo, PdEvent, TransferPeer};
use ts_scheduler::{ExecutionPlan, ForwardEvent, ForwardEventKind, RequestSpec};

use super::*;
use crate::pd::Streams;

#[path = "../../../../ts-pd/src/tests_support.rs"]
mod native;
#[path = "../../../../ts-pd/tests/support/mod.rs"]
mod support;

/// A Prefill scheduler under independent decode allocation: 16-token pages
/// and 64 tokens a round.
const PREFILL_SCHEDULER: &str = r#"
import tokenspeed_scheduler as ts

cfg = ts.SchedulerConfig()
cfg.role = ts.SchedulerConfig.Role.P
cfg.decode_allocation_policy = ts.SchedulerConfig.DecodeAllocationPolicy.Independent
cfg.prefix_granularity = 16
cfg.max_scheduled_tokens = 64
cfg.max_batch_size = 4
cfg.num_device_pages = 64
cfg.num_snapshot_pages = 1
cfg.disable_l2_cache = True
cfg.cache_groups = [
    ts.CacheGroupConfig(
        group_id="history",
        block_granularity=16,
        total_pages=64,
        transfer_policy=ts.CacheTransferPolicy.FullSuffix,
        attn_kind=ts.AttnKind.Full,
    )
]
result = ts.Scheduler(cfg)
"#;

#[test]
fn a_decode_peer_hears_of_a_prompt_when_the_plan_with_its_last_chunk_reaches_the_device() {
    interpreter();
    let library = native::NativeLibrary::new();
    let (source, destination) = (support::pattern(), vec![0; support::BYTES]);
    let mut pair = support::pair_on_port(
        library.path().to_owned(),
        &source,
        &destination,
        Duration::from_secs(60),
        None,
        0,
        "independent",
    );
    let (mut mesh, group) = support::single_rank();
    let streams = Streams::Prefill(pair.prefill_outbox.clone());
    let (plane, _) = plane();
    let mut data_plane = PythonDataPlane::start(plane, Some(streams.clone())).unwrap();
    let mut scheduler = Python::attach(|py| {
        let globals = PyDict::new(py);
        py.run(
            &CString::new(PREFILL_SCHEDULER).unwrap(),
            Some(&globals),
            None,
        )
        .unwrap_or_else(|error| panic!("{}", describe(py, &error)));
        Scheduler::from_python(&globals.get_item("result").unwrap().unwrap()).unwrap()
    });
    let event = |kind, rid: &str, tokens| ForwardEvent {
        kind,
        request_id: rid.to_owned(),
        tokens,
        spec_candidate_ids: Vec::new(),
        reserve_num_tokens: 0,
        bootstrap_token: -1,
    };
    // The plan's rows, with the tokens each feeds.
    let rows = |plan: &ExecutionPlan| {
        let batch = plan.forward_batches().remove(0);
        batch
            .request_ids
            .into_iter()
            .zip(batch.input_lengths)
            .collect::<Vec<_>>()
    };
    // The loop hands a plan with a batch to the device and commits it.
    let mut launch = |plan: ExecutionPlan| {
        let batch = plan.forward_batches().remove(0);
        let dispatch = Dispatch {
            forward: Some(&batch),
            ..Dispatch::default()
        };
        let forward = data_plane.execute(plan, &dispatch).unwrap().unwrap();
        data_plane.complete(forward).unwrap();
    };

    // A one-page prompt and a ten-page one arrive on both nodes: the Decode
    // peer announces each and the Prefill peer admits it.
    for (rid, room, tokens) in [("short", 1, 16), ("long", 2, 160)] {
        let bootstrap = BootstrapInfo {
            room,
            ..pair.bootstrap.clone()
        };
        pair.decode.register(rid, &bootstrap).unwrap();
        pair.prefill.register(rid, &bootstrap).unwrap();
        pair.prefill.occupy(rid, tokens / 16, 64);
        scheduler
            .submit_requests(&[RequestSpec {
                session_id: String::new(),
                cache_group_id: String::new(),
                shared_prefix_tokens: Vec::new(),
                request_id: rid.to_owned(),
                tokens: (0..tokens as i32).map(|t| 1000 * room as i32 + t).collect(),
                max_new_tokens: 4,
                max_cached_prefix_tokens: i32::MAX,
            }])
            .unwrap();
        assert_eq!(
            support::next_event(&mut pair.prefill, &mut mesh, &group),
            PdEvent::Bootstrapped(rid.to_owned())
        );
        scheduler
            .advance(&[event(ForwardEventKind::PdBootstrapped, rid, vec![])])
            .unwrap();
    }
    // What the Decode peer reports first, within `wait`.
    let mut reports = |wait: Duration| {
        let deadline = Instant::now() + wait;
        loop {
            let events = pair.decode.poll(&mut mesh, &group).unwrap();
            if !events.is_empty() || Instant::now() >= deadline {
                return events;
            }
            thread::sleep(Duration::from_millis(5));
        }
    };
    let (quiet, soon) = (Duration::from_millis(200), Duration::from_secs(30));
    assert_eq!(reports(quiet), []);

    // The whole short prompt beside the long one's first chunk: only the
    // short one holds all its pages.
    let plan = scheduler.next_execution_plan(true, false);
    assert_eq!(
        rows(&plan),
        [("short".to_owned(), 16), ("long".to_owned(), 48)]
    );
    launch(plan);
    assert_eq!(reports(soon), [PdEvent::Bootstrapped("short".to_owned())]);
    assert_eq!(reports(quiet), []);
    // A middle chunk.
    scheduler
        .advance(&[event(ForwardEventKind::ExtendResult, "long", vec![])])
        .unwrap();
    let plan = scheduler.next_execution_plan(true, false);
    assert_eq!(rows(&plan), [("long".to_owned(), 64)]);
    launch(plan);
    assert_eq!(reports(quiet), []);
    // The last chunk: a round that launches no forward tells no one, the
    // launch does.
    scheduler
        .advance(&[event(ForwardEventKind::ExtendResult, "long", vec![])])
        .unwrap();
    let plan = scheduler.next_execution_plan(true, false);
    assert_eq!(rows(&plan), [("long".to_owned(), 48)]);
    streams.before_execute(&plan, false).unwrap();
    assert_eq!(reports(quiet), []);
    launch(plan);
    assert_eq!(reports(soon), [PdEvent::Bootstrapped("long".to_owned())]);
    assert_eq!(reports(quiet), []);
}

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

//! Behavioral checks of the CPU workload against the real engine components.
#[path = "../benches/support/mod.rs"]
#[allow(dead_code)] // Shared with the standalone benchmark's CLI and reporting.
mod support;

use support::{admit_one, batch, processor, request, result, Config, Logprobs};
use ts_client::OutputChunk;
use ts_control::{
    event_loop::Load,
    ingress::{self, Admitted},
    transport::output_batch,
};
use ts_wire::EngineOutput;

fn config(suite: &str, transport: &str, steps: usize, logprobs: Logprobs, top_k: usize) -> Config {
    Config {
        suite: suite.into(),
        transport: transport.into(),
        batch: 3,
        prompt: 17,
        steps,
        logprobs,
        top_k,
        period_us: 0,
        consumer_delay_us: 0,
        client_threads: if suite == "pipeline" { 2 } else { 0 },
        warmups: 0,
        repeats: 1,
        phase_timers: true,
    }
}

#[test]
fn complete_prompt_values_and_ids_cross_the_wire_once() {
    let c = config("stages", "none", 3, Logprobs::Prompt, 0);
    let mut p = processor(&c);
    let expected: Vec<_> = (0..c.batch)
        .map(|row| request(&c, row).input_ids.unwrap())
        .collect();
    let specs: Vec<_> = (0..c.batch)
        .map(|row| admit_one(request(&c, row), &c, &mut p))
        .collect();
    let rids: Vec<_> = specs.iter().map(|s| s.request_id.clone()).collect();
    for step in 0..c.steps {
        let view = batch(&c, &rids, step == 0);
        let plan = p.logprob_plan(&view);
        if step == 0 {
            assert_eq!(plan.input.as_ref().unwrap().counts, vec![16; 3]);
        } else {
            assert!(plan.input.is_none());
        }
        let committed = p
            .commit(&view, plan.input.as_ref(), &result(&c, step))
            .unwrap();
        assert_eq!(committed.released.len(), if step == 2 { 3 } else { 0 });
        let bytes = EngineOutput::Batch(Box::new(output_batch(
            committed.outputs,
            0,
            Load::default(),
        )))
        .to_frames();
        let EngineOutput::Batch(wire) = EngineOutput::from_frames(&bytes).unwrap() else {
            panic!("batch")
        };
        for (i, row) in wire.requests().unwrap().enumerate() {
            let chunk = OutputChunk::from(&row);
            assert_eq!(chunk.output_token_logprobs, [-0.5]);
            if step == 0 {
                let (values, ids) = chunk.input_token_logprobs.unwrap();
                assert_eq!(ids, expected[i]);
                assert_eq!(
                    values,
                    std::iter::once(None)
                        .chain(std::iter::repeat_n(Some(-1.25), 16))
                        .collect::<Vec<_>>()
                );
            } else {
                assert!(chunk.input_token_logprobs.is_none());
            }
        }
    }
    assert_eq!(p.num_running(), 0);
}

#[test]
fn top_alternatives_have_every_prompt_position_and_the_sampled_token() {
    let c = config("stages", "none", 1, Logprobs::Prompt, 5);
    let mut p = processor(&c);
    let spec = admit_one(request(&c, 0), &c, &mut p);
    let mut one = c.clone();
    one.batch = 1;
    let view = batch(&one, &[spec.request_id], true);
    let plan = p.logprob_plan(&view);
    let output = p
        .commit(&view, plan.input.as_ref(), &result(&one, 0))
        .unwrap()
        .outputs
        .remove(0);
    let input = output.input_logprobs.unwrap();
    let (values, ids) = input.top.unwrap();
    assert_eq!(values.len(), 17);
    assert_eq!(ids.len(), 17);
    assert!(values[0].is_none());
    assert!(ids[0].is_none());
    for row in 1..17 {
        assert_eq!(
            values[row].as_ref().unwrap(),
            &[-0.5, -1.5, -2.5, -3.5, -4.5]
        );
        assert_eq!(ids[row].as_ref().unwrap(), &[10, 11, 12, 13, 14]);
    }
    let (values, ids) = output.output_top_logprobs.unwrap();
    assert_eq!(values, vec![vec![-0.5, -1.5, -2.5, -3.5, -4.5]]);
    assert_eq!(ids, vec![vec![10, 11, 12, 13, 14]]);
    assert_eq!(p.num_running(), 0);
}

#[test]
fn unsupported_streaming_top_k_is_refused_by_the_real_ingress() {
    let c = config("pipeline", "ipc", 3, Logprobs::Prompt, 5);
    assert!(c.validate().is_err());
    assert!(matches!(
        ingress::admit(request(&c, 0), &support::limits(&c)),
        Some(Admitted::Refused { .. })
    ));
}

#[test]
fn all_stage_work_is_consumed_and_verified() {
    for mode in [Logprobs::None, Logprobs::Output, Logprobs::Prompt] {
        let c = config("stages", "none", 3, mode, 0);
        let report = support::run_stages(&c);
        assert_eq!(report["checked_chunks"], 9);
        assert_eq!(report["output_bytes"].as_array().unwrap().len(), 3);
    }
}

#[test]
fn real_ipc_streams_finish_and_release_client_registrations() {
    let c = config("pipeline", "ipc", 4, Logprobs::Prompt, 0);
    let report = support::pipeline::run(&c);
    assert_eq!(report["checked_chunks"], 12);
    for row in report["rows"].as_array().unwrap() {
        let arrival = row["arrival_ms"].as_array().unwrap();
        assert_eq!(arrival.len(), 4);
        assert!(arrival
            .windows(2)
            .all(|v| v[0].as_f64().unwrap() <= v[1].as_f64().unwrap()));
    }
}

#[test]
fn slow_tcp_consumers_receive_every_chunk_in_order() {
    let mut c = config("pipeline", "tcp", 8, Logprobs::Output, 0);
    c.consumer_delay_us = 2000;
    c.period_us = 100;
    let report = support::pipeline::run(&c);
    assert_eq!(report["checked_chunks"], 24);
    assert_eq!(report["rows"].as_array().unwrap().len(), 3);
}

#[test]
fn disabling_phase_timers_preserves_complete_delivery() {
    let mut c = config("pipeline", "ipc", 3, Logprobs::Prompt, 0);
    c.phase_timers = false;
    let report = support::pipeline::run(&c);
    assert_eq!(report["checked_chunks"], 9);
    assert!(report["engine"]["phases"].as_object().unwrap().is_empty());
}

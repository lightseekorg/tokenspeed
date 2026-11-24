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

use std::sync::{Arc, Mutex};

use pyo3::prelude::*;
use serde_json::Value;
use ts_control::{
    event_loop::Load,
    execution_observation::{Context, Sink},
    output::{FinishLog, OutputProcessor, Role},
    request::RequestState,
    transport::output_batch,
};
use ts_scheduler::ForwardBatchView;
use ts_wire::{decode_frames, encode_frames, BatchTokenIDOutSlim};

#[derive(Debug, Default)]
struct RecordedEvents(Mutex<Vec<Value>>);

impl Sink for RecordedEvents {
    fn emit(&self, event: Value) {
        self.0.lock().unwrap().push(event);
    }

    fn context(&self) -> Context {
        Context {
            instance_id: "completion-isolation-test".into(),
            role: "fused".into(),
            dp_rank: 0,
            physical_rank: 0,
        }
    }

    fn valid(&self) -> bool {
        !self
            .0
            .lock()
            .unwrap()
            .iter()
            .any(|e| e["event"] == "observation_loss")
    }
}

#[test]
fn invalid_optional_diagnostics_commit_tokens_but_invalidate_closed_wire_evidence() {
    super::interpreter();
    Python::attach(|py| {
        let scope = super::completion_diagnostic_fixture(py);
        let plane = scope.get_item("plane").unwrap().unwrap();
        let landed = scope.get_item("landed").unwrap().unwrap().unbind();
        for case in scope
            .get_item("malformed_results")
            .unwrap()
            .unwrap()
            .try_iter()
            .unwrap()
        {
            let (name, value): (String, Bound<'_, PyAny>) = case.unwrap().extract().unwrap();
            let sink = Arc::new(RecordedEvents::default());
            let mut processor = OutputProcessor::new(
                Some(8),
                1,
                Some(4096),
                Role::Plain,
                FinishLog {
                    enabled: false,
                    request_stats: false,
                    ttft_percentage: 0,
                },
                false,
                None,
            );
            processor.observe_execution(sink.clone());
            let mut state = RequestState::new(vec![9], 2, Default::default(), true, None, true);
            processor.assign_admission(&mut state);
            let admission = state.admission;
            processor.register("measured".into(), state);
            let batch = ForwardBatchView {
                request_ids: vec!["measured".into()],
                request_pool_indices: vec![1],
                input_lengths: vec![8],
                prefill_lengths: vec![1],
                extend_prefix_lens: vec![],
                extend_replay_lens: vec![],
                num_extends: 0,
            };
            let lease = processor.observe_dispatch(&batch);
            plane.setattr("result", value).unwrap();
            let result = super::super::complete(&plane, &landed).unwrap();
            let committed = processor
                .commit_observed(&batch, None, &result, lease)
                .unwrap();
            let wire = output_batch(committed.outputs, 0, Load::default());
            let decoded: BatchTokenIDOutSlim = decode_frames(&encode_frames(&wire)).unwrap();
            assert_eq!(decoded.output_ids, [vec![3, 4]], "{name}");
            assert_eq!(decoded.finished_reasons, ["length"], "{name}");
            assert_eq!(
                decoded.output_token_logprobs_val,
                [vec![-0.5, -0.5]],
                "{name}"
            );
            assert_eq!(
                decoded.output_token_cumulative_probs_val,
                [vec![0.5, 0.5]],
                "{name}"
            );
            let snapshot = decoded.execution_observations[0].as_ref().unwrap();
            assert_eq!(snapshot.admission_id, admission, "{name}");
            assert!(!snapshot.evidence_valid, "{name}");
            assert_eq!(snapshot.assigned_rounds, 1, "{name}");
            assert_eq!(snapshot.completed_rounds, 0, "{name}");
            assert!(
                snapshot.bins.is_empty() && snapshot.pre_stop_bins.is_empty(),
                "{name}"
            );
            drop(processor);
            let events = sink.0.lock().unwrap();
            let closed = events.last().unwrap();
            assert_eq!(closed["event"], "request_closed", "{name}");
            assert_eq!(closed["admission_id"], admission, "{name}");
            assert_eq!(closed["missing_rounds"], 1, "{name}");
            assert_eq!(closed["evidence_valid"], false, "{name}");
            assert!(
                !events
                    .iter()
                    .any(|e| e["event"] == "target_round_completed"),
                "{name}"
            );
        }
    });
}

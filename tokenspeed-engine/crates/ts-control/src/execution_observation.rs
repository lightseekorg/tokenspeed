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

//! Fenced execution evidence owned by native admissions and pending forwards.

use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    fmt::Debug,
    sync::{Arc, Mutex},
};
use ts_wire::output::{ExecutionBin, ExecutionSnapshot, PreStopBin, MAX_EXECUTION_SNAPSHOT_BYTES};

/// Closed native trace protocol carried by the shared cache-event publisher.
pub const PROTOCOL: &str = "tokenspeed-execution-observations-v1";
const MAX_BINS: usize = 4096;

#[derive(Debug, Clone)]
/// Immutable publisher identity shared by all admissions on this rank.
pub struct Context {
    /// Caller-supplied engine instance, shared with cache observations.
    pub instance_id: String,
    /// Engine role: fused, prefill, or decode.
    pub role: String,
    /// Logical data-parallel rank.
    pub dp_rank: i64,
    /// Physical rank that executed the target.
    pub physical_rank: u64,
}

/// A bounded, nonblocking native publisher; no Python or device access.
pub trait Sink: Debug + Send + Sync {
    /// Publish an event or account for its loss without blocking serving.
    fn emit(&self, event: Value);
    /// Identity attached to this publisher's records and snapshots.
    fn context(&self) -> Context;
    /// Whether all publications so far remain usable as evidence.
    fn valid(&self) -> bool;
}

/// CPU metadata returned with a landed forward. Width is the consumed layout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionResult {
    /// Native admission identities in live request-row order.
    pub admission_ids: Vec<u64>,
    /// Actual target call or replay route, excluding draft execution.
    pub path: String,
    /// Candidate columns consumed by the ordinary verifier, including anchor.
    pub verify_width: u32,
    /// Real request rows, excluding capture and padding rows.
    pub live_requests: usize,
    /// Request capacity of the executed target route.
    pub request_capacity: usize,
    /// Target query-token capacity, including graph padding.
    pub query_capacity: usize,
    /// Captured segments actually replayed; eager breaks are excluded.
    pub graph_segments: usize,
    /// Landed [accepted_before_stop, applied_force_single] in native row order.
    pub verification: Vec<[i32; 2]>,
}

#[derive(Debug, Default)]
struct Counts {
    assigned: u64,
    completed: u64,
    through: u64,
    missing: u64,
    committed: u64,
    invalid: bool,
    terminal: Option<String>,
    histograms_omitted: bool,
    bins: BTreeMap<(String, String, u32, u32), u64>,
    pre_stop_bins: BTreeMap<(String, u32, i32, bool), u64>,
}

/// This Arc outlives request output state while pending forwards retain it.
#[derive(Debug)]
pub struct RequestObservation {
    sink: Arc<dyn Sink>,
    rid: String,
    admission: u64,
    origin: String,
    counts: Mutex<Counts>,
}

impl RequestObservation {
    /// Start one immutable native admission; later slot changes retain this owner.
    pub fn new(sink: Arc<dyn Sink>, rid: String, admission: u64, startup: bool) -> Arc<Self> {
        let owner = Arc::new(Self {
            sink,
            rid,
            admission,
            origin: if startup { "startup" } else { "serving" }.into(),
            counts: Mutex::new(Counts::default()),
        });
        if owner.rid.len() > 256 || admission == 0 {
            owner.invalidate("invalid_admission_identity");
        } else {
            owner.publish(json!({"event": "request_started"}));
        }
        owner
    }

    fn publish(&self, mut event: Value) {
        if self.rid.len() > 256 {
            return;
        }
        event["request_id"] = self.rid.clone().into();
        event["admission_id"] = self.admission.into();
        event["origin"] = self.origin.clone().into();
        self.sink.emit(event);
    }

    fn invalidate(&self, reason: &str) {
        self.counts
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .invalid = true;
        self.sink
            .emit(json!({"event": "observation_loss", "reason": reason, "dropped_count": 1}));
    }

    /// Remember termination; pending leases still delay execution closure.
    pub fn terminal(&self, reason: String) {
        self.counts
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .terminal = Some(reason);
    }

    pub(crate) fn snapshot(&self) -> ExecutionSnapshot {
        let c = self.counts.lock().unwrap_or_else(|e| e.into_inner());
        let ctx = self.sink.context();
        let mut snapshot = ExecutionSnapshot {
            version: 1,
            instance_id: ctx.instance_id,
            role: ctx.role,
            dp_rank: ctx.dp_rank,
            physical_rank: ctx.physical_rank,
            request_id: self.rid.clone(),
            admission_id: self.admission,
            origin: self.origin.clone(),
            assigned_rounds: c.assigned,
            completed_rounds: c.completed,
            through_round: c.through,
            missing_rounds: c.missing,
            committed_tokens: c.committed,
            evidence_valid: !c.invalid && self.sink.valid(),
            histograms_omitted: c.histograms_omitted,
            pre_stop_bins: c
                .pre_stop_bins
                .iter()
                .map(|((path, width, accepted, force), count)| PreStopBin {
                    path: path.clone(),
                    verify_width: u64::from(*width),
                    accepted_before_stop: *accepted as u64,
                    force_single: *force,
                    count: *count,
                })
                .collect(),
            bins: c
                .bins
                .iter()
                .map(|((phase, path, width, kept), count)| ExecutionBin {
                    phase: phase.clone(),
                    path: path.clone(),
                    verify_width: u64::from(*width),
                    kept_prefix: u64::from(*kept),
                    count: *count,
                })
                .collect(),
        };
        drop(c);
        if ts_wire::encode_frames(&snapshot)
            .iter()
            .map(|frame| frame.len())
            .sum::<usize>()
            > MAX_EXECUTION_SNAPSHOT_BYTES
        {
            {
                let mut c = self.counts.lock().unwrap_or_else(|e| e.into_inner());
                c.histograms_omitted = true;
                c.bins.clear();
                c.pre_stop_bins.clear();
            }
            self.invalidate("snapshot_byte_budget");
            snapshot.evidence_valid = false;
            snapshot.histograms_omitted = true;
            snapshot.bins.clear();
            snapshot.pre_stop_bins.clear();
        }
        snapshot
    }

    /// Invalid identities are unavailable rather than truncated onto the wire.
    pub(crate) fn wire_snapshot(&self) -> Option<ExecutionSnapshot> {
        (self.rid.len() <= 256 && self.admission != 0).then(|| self.snapshot())
    }
}

impl Drop for RequestObservation {
    fn drop(&mut self) {
        let c = self.counts.lock().unwrap_or_else(|e| e.into_inner());
        let event = json!({"event": "request_closed", "assigned_rounds": c.assigned,
            "completed_rounds": c.completed, "missing_rounds": c.missing,
            "terminal": c.terminal, "evidence_valid": !c.invalid && c.terminal.is_some()
                && c.assigned == c.completed && c.missing == 0 && self.sink.valid()});
        drop(c);
        self.publish(event);
    }
}

#[derive(Debug)]
struct Row {
    owner: Arc<RequestObservation>,
    round: u64,
    slot: i32,
    phase: &'static str,
    input_length: i32,
    done: bool,
}

/// A native identity snapshot, moved into the pending-forward queue.
#[derive(Debug, Default)]
pub struct ForwardLease {
    rows: Vec<Row>,
    num_extends: usize,
}

impl ForwardLease {
    /// Freeze a dispatched row and reserve its next admission-local ordinal.
    pub fn push(
        &mut self,
        owner: Arc<RequestObservation>,
        slot: i32,
        decode: bool,
        input_length: i32,
    ) {
        let mut c = owner.counts.lock().unwrap_or_else(|e| e.into_inner());
        let Some(round) = c.assigned.checked_add(1) else {
            drop(c);
            owner.invalidate("round_counter_overflow");
            return;
        };
        c.assigned = round;
        drop(c);
        let phase = if decode { "decode" } else { "prefill" };
        self.num_extends += usize::from(!decode);
        owner.publish(json!({"event": "target_round_dispatched", "round": round,
            "slot": slot, "phase": phase, "input_length": input_length}));
        self.rows.push(Row {
            owner,
            round,
            slot,
            phase,
            input_length,
            done: false,
        });
    }

    /// Whether a current output row still belongs to this frozen admission.
    pub fn matches(&self, index: usize, admission: u64) -> bool {
        self.rows
            .get(index)
            .is_some_and(|r| r.owner.admission == admission)
    }

    /// Observe a fenced row after native commit, including discarded late output.
    pub fn complete(
        &mut self,
        index: usize,
        result: Option<&ExecutionResult>,
        kept: i32,
        committed: usize,
        numerical_error: bool,
        discarded: bool,
    ) {
        let expected_rows = self.rows.len();
        let num_extends = self.num_extends;
        let Some(row) = self.rows.get_mut(index) else {
            return;
        };
        if row.done {
            row.owner.invalidate("duplicate_round");
            return;
        }
        let Some(result) = result else {
            row.owner.invalidate("missing_execution_result");
            return;
        };
        let decode = row.phase == "decode";
        let width = if decode { result.verify_width } else { 0 };
        let path = if row.input_length == 0 {
            "none"
        } else {
            result.path.as_str()
        };
        let Some(&[accepted_before_stop, force_single]) = result.verification.get(index) else {
            row.owner.invalidate("missing_verification_observation");
            return;
        };
        let valid = expected_rows == result.live_requests
            && result.verification.len() == expected_rows
            && row.input_length >= 0
            && row.slot > 0
            && matches!(force_single, 0 | 1)
            && (if decode {
                accepted_before_stop >= 1
                    && accepted_before_stop as u32 <= width
                    && kept <= accepted_before_stop
                    && (force_single == 0 || accepted_before_stop == 1)
            } else {
                accepted_before_stop == -1 && force_single == 0
            })
            && result.admission_ids.len() == result.live_requests
            && result.admission_ids.get(index) == Some(&row.owner.admission)
            && result.live_requests <= result.request_capacity
            && matches!(
                result.path.as_str(),
                "none" | "eager" | "decode_graph" | "prefill_breakable" | "prefill_inline"
            )
            && (result.path != "decode_graph" || decode)
            && (result.path != "none" || row.input_length == 0)
            && (!result.path.contains("graph") && !result.path.starts_with("prefill_")
                || result.graph_segments > 0)
            && kept >= 0
            && committed <= kept as usize
            && (!decode || width > 0 && width == row.input_length as u32 && kept as u32 <= width);
        if !valid {
            row.owner.invalidate("invalid_execution_result");
            return;
        }
        row.done = true;
        let mut c = row.owner.counts.lock().unwrap_or_else(|e| e.into_inner());
        c.completed += 1; // bounded by the checked dispatch counter
        c.through = row.round;
        c.committed = match c.committed.checked_add(committed as u64) {
            Some(value) => value,
            None => {
                c.invalid = true;
                c.committed
            }
        };
        let key = (row.phase.into(), path.to_owned(), width, kept as u32);
        let pre_stop_key = (
            path.to_owned(),
            width,
            accepted_before_stop,
            force_single != 0,
        );
        let new_bins = usize::from(!c.bins.contains_key(&key))
            + usize::from(decode && !c.pre_stop_bins.contains_key(&pre_stop_key));
        let overflow =
            !c.histograms_omitted && c.bins.len() + c.pre_stop_bins.len() + new_bins > MAX_BINS;
        if !overflow && !c.histograms_omitted {
            *c.bins.entry(key).or_default() += 1;
            if decode {
                *c.pre_stop_bins.entry(pre_stop_key).or_default() += 1;
            }
        }
        drop(c);
        if overflow {
            row.owner.invalidate("histogram_capacity");
        }
        row.owner.publish(
            json!({"event": "target_round_completed", "round": row.round,
            "slot": row.slot, "phase": row.phase, "path": path,
            "verify_width": width, "kept_prefix": kept, "committed_tokens": committed,
            "accepted_before_stop": accepted_before_stop, "force_single": force_single != 0,
            "input_length": row.input_length, "request_capacity": result.request_capacity,
            "live_requests": expected_rows, "num_extends": num_extends, "row_index": index,
            "query_capacity": result.query_capacity, "graph_segments": result.graph_segments,
            "discarded": discarded, "numerical_error": numerical_error}),
        );
    }
}

impl Drop for ForwardLease {
    fn drop(&mut self) {
        for row in &self.rows {
            if !row.done {
                row.owner
                    .counts
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .missing += 1;
                row.owner.invalidate("uncompleted_round");
            }
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    #[derive(Debug, Default)]
    pub(crate) struct MemorySink(pub Mutex<Vec<Value>>);
    impl Sink for MemorySink {
        fn emit(&self, value: Value) {
            self.0.lock().unwrap().push(value);
        }
        fn context(&self) -> Context {
            Context {
                instance_id: "test".into(),
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
                .any(|v| v["event"] == "observation_loss")
        }
    }

    fn result(admission: u64, width: u32) -> ExecutionResult {
        ExecutionResult {
            admission_ids: vec![admission],
            path: "decode_graph".into(),
            verify_width: width,
            live_requests: 1,
            request_capacity: 4,
            query_capacity: 4 * width as usize,
            graph_segments: 1,
            verification: vec![[width as i32, 0]],
        }
    }

    #[test]
    fn rounds_keep_width_and_prefix_separate_and_survive_slot_changes() {
        let sink = Arc::new(MemorySink::default());
        let owner = RequestObservation::new(sink.clone(), "request".into(), 41, false);
        for (slot, kept) in [(3, 1), (9, 5), (9, 0)] {
            let mut lease = ForwardLease::default();
            lease.push(owner.clone(), slot, true, 8);
            lease.complete(0, Some(&result(41, 8)), kept, kept as usize, false, false);
        }
        let snapshot = owner.snapshot();
        assert_eq!(snapshot.completed_rounds, 3);
        assert_eq!(snapshot.committed_tokens, 6);
        assert_eq!(
            snapshot
                .bins
                .iter()
                .map(|b| (b.verify_width, b.kept_prefix, b.count))
                .collect::<Vec<_>>(),
            [(8, 0, 1), (8, 1, 1), (8, 5, 1)]
        );
        owner.terminal("length".into());
        drop(owner);
        let records = sink.0.lock().unwrap();
        assert_eq!(records.last().unwrap()["event"], "request_closed");
        assert_eq!(records.last().unwrap()["evidence_valid"], true);
    }

    #[test]
    fn a_pending_lease_keeps_the_old_admission_after_output_and_rid_reuse() {
        let sink = Arc::new(MemorySink::default());
        let old = RequestObservation::new(sink.clone(), "same".into(), 1, false);
        let mut late = ForwardLease::default();
        late.push(old.clone(), 4, true, 1);
        old.terminal("length".into());
        drop(old);
        let new = RequestObservation::new(sink.clone(), "same".into(), 2, false);
        assert!(!late.matches(0, 2));
        assert_eq!(new.snapshot().completed_rounds, 0);
        assert!(!sink
            .0
            .lock()
            .unwrap()
            .iter()
            .any(|v| v["event"] == "request_closed"));
        late.complete(0, Some(&result(1, 1)), 0, 0, false, true);
        drop(late);
        let records = sink.0.lock().unwrap();
        assert_eq!(records.last().unwrap()["admission_id"], 1);
        assert_eq!(records.last().unwrap()["completed_rounds"], 1);
        assert_eq!(records[records.len() - 2]["discarded"], true);
    }

    #[test]
    fn histogram_growth_is_bounded_and_overflow_invalidates_evidence() {
        let sink = Arc::new(MemorySink::default());
        let owner = RequestObservation::new(sink.clone(), "r".into(), 1, false);
        for width in 1..=MAX_BINS as u32 + 1 {
            let mut lease = ForwardLease::default();
            lease.push(owner.clone(), 1, true, width as i32);
            lease.complete(0, Some(&result(1, width)), 1, 1, false, false);
        }
        {
            let counts = owner.counts.lock().unwrap();
            assert_eq!(counts.bins.len() + counts.pre_stop_bins.len(), MAX_BINS);
        }
        let snapshot = owner.snapshot();
        assert!(
            ts_wire::encode_frames(&snapshot)
                .iter()
                .map(|f| f.len())
                .sum::<usize>()
                <= MAX_EXECUTION_SNAPSHOT_BYTES
        );
        assert_eq!(snapshot.completed_rounds, MAX_BINS as u64 + 1);
        assert!(!snapshot.evidence_valid);
        assert!(sink
            .0
            .lock()
            .unwrap()
            .iter()
            .any(|r| r["reason"] == "histogram_capacity"));
    }

    #[test]
    fn acceptance_before_stop_and_force_single_are_separate_from_kept_tokens() {
        let sink = Arc::new(MemorySink::default());
        let owner = RequestObservation::new(sink.clone(), "r".into(), 1, false);
        for (pre_stop, forced, kept) in [(7, 0, 2), (1, 1, 1), (6, 0, 0)] {
            let mut value = result(1, 8);
            value.verification = vec![[pre_stop, forced]];
            let mut lease = ForwardLease::default();
            lease.push(owner.clone(), 1, true, 8);
            lease.complete(0, Some(&value), kept, kept as usize, false, kept == 0);
        }
        let snapshot = owner.snapshot();
        assert_eq!(
            snapshot
                .pre_stop_bins
                .iter()
                .map(|b| (b.accepted_before_stop, b.force_single, b.count))
                .collect::<Vec<_>>(),
            [(1, true, 1), (6, false, 1), (7, false, 1)]
        );
        assert_eq!(
            snapshot
                .bins
                .iter()
                .map(|b| b.kept_prefix)
                .collect::<Vec<_>>(),
            [0, 1, 2]
        );
        let mut bad = result(1, 8);
        bad.verification = vec![[7, 1]];
        let mut lease = ForwardLease::default();
        lease.push(owner.clone(), 1, true, 8);
        lease.complete(0, Some(&bad), 1, 1, false, false);
        drop(lease);
        assert!(!owner.snapshot().evidence_valid);
    }

    #[test]
    fn wire_snapshot_byte_budget_is_enforced_without_truncating_identity() {
        let sink = Arc::new(MemorySink::default());
        let prefill = RequestObservation::new(sink.clone(), "prefill".into(), 99, false);
        let owner = RequestObservation::new(sink.clone(), "decode".into(), 1, false);
        for accepted in 1..=2048 {
            let value = ExecutionResult {
                admission_ids: vec![99, 1],
                path: "prefill_breakable".into(),
                verify_width: 2048,
                live_requests: 2,
                request_capacity: 2,
                query_capacity: 4096,
                graph_segments: 1,
                verification: vec![[-1, 0], [accepted, 0]],
            };
            let mut lease = ForwardLease::default();
            lease.push(prefill.clone(), 2, false, 1);
            lease.push(owner.clone(), 1, true, 2048);
            lease.complete(0, Some(&value), 1, 1, false, false);
            lease.complete(1, Some(&value), accepted, accepted as usize, false, false);
            if accepted == 1500 {
                let below = owner.snapshot();
                assert!(below.evidence_valid);
                assert!(!below.histograms_omitted);
                let size = ts_wire::encode_frames(&below)
                    .iter()
                    .map(|f| f.len())
                    .sum::<usize>();
                assert!(
                    size > 200_000 && size <= MAX_EXECUTION_SNAPSHOT_BYTES,
                    "{size}"
                );
            }
        }
        let bounded = owner.wire_snapshot().unwrap();
        assert!(bounded.histograms_omitted);
        assert!(!bounded.evidence_valid);
        assert_eq!(bounded.request_id, "decode");
        assert_eq!(bounded.admission_id, 1);
        assert_eq!(bounded.completed_rounds, 2048);
        assert!(bounded.bins.is_empty() && bounded.pre_stop_bins.is_empty());
        assert!(
            ts_wire::encode_frames(&bounded)
                .iter()
                .map(|f| f.len())
                .sum::<usize>()
                <= MAX_EXECUTION_SNAPSHOT_BYTES
        );
        assert!(sink
            .0
            .lock()
            .unwrap()
            .iter()
            .any(|r| r["reason"] == "snapshot_byte_budget"));
        assert!(owner.snapshot().histograms_omitted);
        let invalid_identity = RequestObservation::new(
            sink.clone(),
            "x".repeat(MAX_EXECUTION_SNAPSHOT_BYTES + 1),
            100,
            false,
        );
        assert!(invalid_identity.wire_snapshot().is_none());
    }

    #[test]
    fn abandoned_or_wrong_identity_results_cannot_close_as_evidence() {
        for supplied in [None, Some(result(2, 8))] {
            let sink = Arc::new(MemorySink::default());
            let owner = RequestObservation::new(sink.clone(), "r".into(), 1, false);
            let mut lease = ForwardLease::default();
            lease.push(owner.clone(), 1, true, 8);
            if let Some(value) = supplied.as_ref() {
                lease.complete(0, Some(value), 1, 1, false, false);
            }
            drop(lease);
            assert_eq!(owner.snapshot().completed_rounds, 0);
            assert_eq!(owner.snapshot().missing_rounds, 1);
            owner.terminal("abort".into());
            drop(owner);
            assert_eq!(
                sink.0.lock().unwrap().last().unwrap()["evidence_valid"],
                false
            );
        }
    }
}

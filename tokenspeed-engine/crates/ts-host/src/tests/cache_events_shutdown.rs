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

//! Native host shutdown over the real wire/control loop, with CPU device
//! results replacing model startup and GPU execution. The child calls
//! run_host: its normal process::exit must flush a genuinely final trace.

use pyo3::types::{PyBytes, PyModule};
use std::thread;
use ts_wire::{
    handshake::{HandshakeAddresses, HandshakeInitMessage},
    EngineCoreReadyResponse, EngineRequest, SamplingParams, TokenizedGenerateReqInput,
};
use zeromq::{Socket, SocketRecv, SocketSend};

use super::*;

const TEST: &str =
    "cache_events::tests::shutdown::normal_host_exit_finishes_after_native_slot_release";
const RETAINED_TEST: &str =
    "cache_events::tests::shutdown::a_retained_native_owner_cannot_claim_a_finished_trace";

fn frontend() -> String {
    let (started, address) = mpsc::sync_channel(1);
    thread::spawn(move || {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(async {
                let mut handshake = zeromq::RouterSocket::new();
                let mut input = zeromq::RouterSocket::new();
                let mut output = zeromq::PullSocket::new();
                let handshake_address = handshake
                    .bind("tcp://127.0.0.1:0")
                    .await
                    .unwrap()
                    .to_string();
                let input_address = input.bind("tcp://127.0.0.1:0").await.unwrap().to_string();
                let output_address = output.bind("tcp://127.0.0.1:0").await.unwrap().to_string();
                started.send(handshake_address).unwrap();
                let hello = handshake.recv().await.unwrap().into_vec();
                let init = HandshakeInitMessage {
                    addresses: HandshakeAddresses {
                        inputs: vec![input_address],
                        outputs: vec![output_address],
                        ..Default::default()
                    },
                    parallel_config: Default::default(),
                };
                let mut frames = vec![hello[0].clone()];
                frames.extend(ts_wire::encode_frames(&init));
                handshake.send(frames.try_into().unwrap()).await.unwrap();
                handshake.recv().await.unwrap();
                let registered = input.recv().await.unwrap().into_vec();
                let request = TokenizedGenerateReqInput {
                    input_ids: Some(vec![1, 2]),
                    ..TokenizedGenerateReqInput::new(
                        "shutdown-request".to_owned(),
                        SamplingParams {
                            max_new_tokens: Some(64),
                            ignore_eos: true,
                            top_k: 1,
                            ..Default::default()
                        },
                    )
                };
                let mut frames = vec![registered[0].clone()];
                frames.extend(EngineRequest::Add(Box::new(request)).to_frames());
                input.send(frames.try_into().unwrap()).await.unwrap();
                // Keep the peer alive throughout the host's normal SIGTERM path.
                while output.recv().await.is_ok() {}
            });
    });
    address.recv_timeout(Duration::from_secs(5)).unwrap()
}

const CPU_BOOTSTRAP: &str = r#"
from contextlib import nullcontext
from dataclasses import dataclass
from tokenspeed.runtime.engine.gc_hooks import freeze_startup_heap
import os
import signal
import time

@dataclass(frozen=True)
class LogprobConfig:
    enable_output_logprobs: bool = False
    supports_prompt_logprobs: bool = False
    logprob_order: str = 'torch'
    pipeline_parallel: bool = False
    disaggregated: bool = False

@dataclass(frozen=True)
class CpuConfig:
    handshake_address: str
    ready_response: bytes
    logprob_servability: LogprobConfig = LogprobConfig()
    engine_index: int = 0
    rank: int = 0
    replica_ranks: tuple = (0,)
    pp_ranks: object = None
    dp_ranks: object = None
    collective_bind_host: str = '127.0.0.1'
    in_flight_depth: int = 1
    vocab_size: int = 128
    eos_token_ids: tuple = ()
    top_k_exclusive_max: int = 128
    max_model_len: int = 256
    max_req_input_len: int = 128
    multimodal: bool = False
    lm_head_tp_under_dp: bool = False
    stream_interval: int = 1
    spec_num_tokens: object = None
    ngram_context_len: int = 0
    request_token_history: bool = False
    request_token_history_lookback: object = None
    physical_context_len: int = 256
    kv_total_pages: int = 64
    kv_token_capacity: int = 1024
    startup_requests: bool = False
    memory_saver: bool = False
    memory_regions: tuple = ()
    backup_device_kv: bool = False
    allow_reload_forward: bool = False
    l3: object = None
    disk: object = None
    reasoning: object = None
    grammar: object = None
    eplb: object = None
    epd_receive_enabled: bool = False
    eager_grammar: bool = False
    prefill_delay: object = None
    narrowing_decoder_window: object = None
    metrics_prometheus_dir: object = None
    metrics_relay_endpoint: object = None
    engine_metrics: bool = False
    transfer_metrics: bool = False
    l3_metrics: bool = False
    disk_metrics: bool = False
    metrics_model_name: str = 'cpu-boundary'
    metrics_app_key: str = ''
    dp_rank: int = 0
    attn_tp_rank: int = 0
    engine_role: str = 'null'
    batch_log: bool = False
    decode_log_interval: int = 1
    spec_num_steps: int = 0
    cache_state_group_ids: tuple = ()
    log_spec_accept_lengths: bool = False
    finish_log: bool = False
    log_request_stats: bool = False
    ttft_trace_sampling_percentage: int = 0
    otlp_traces_endpoint: object = None
    kv_events: bool = False
    kv_events_publisher: object = None

class CpuDevice:
    def __init__(self):
        self.stopped = False
    def control_thread_guard(self): return nullcontext()
    def enter_forward_thread(self): pass
    def exit_forward_thread(self): pass
    def admit(self, rows): pass
    def execute(self, plan, dp, ngram, grammar, logprobs, history):
        rows = sum(len(batch.request_ids) for batch in plan.forward)
        if not self.stopped:
            self.stopped = True
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(0.02)
        return rows
    def complete(self, rows):
        return ([3] * rows, [1] * rows, None, None, None, None, None, None, None, None)
    def note_committed(self, result): pass
    def release(self, rids): pass
    def replay_retirement_enabled(self): return False
    def poll_cache_events(self): return []
    def poll_cache_observations(self): return ([], 0)

class CpuBootstrap:
    mode = 'serve'
    def __init__(self):
        self.config = CpuConfig(handshake_address, ready_response)
        self.data_plane = CpuDevice()
        self.peer_sharing = None
        self.transfer_peer = None
        self.epd_receive = None
        cfg = native.SchedulerConfig()
        cfg.prefix_granularity = 16
        cfg.max_scheduled_tokens = 64
        cfg.max_batch_size = 4
        cfg.max_resident_requests = 4
        cfg.num_device_pages = 64
        cfg.num_snapshot_pages = 1
        cfg.cache_groups = [native.CacheGroupConfig(group_id='full_attention', block_granularity=16, total_pages=64, attn_kind=native.AttnKind.Full)]
        self.scheduler = native.Scheduler(cfg)
        self.retained = self if retain_owner else None
    def collective_addresses(self, port): return [f'127.0.0.1:{port}']
    def report_ready(self): pass
    def enter_serving(self):
        # Match the production startup GC policy, including frozen objects.
        freeze_startup_heap()
    def close(self):
        print('CPU cache closed', flush=True)

retained_schedulers = []

def prepare_bootstrap(fd):
    return False, start

def start(outbox, protocol, spans, peer, shared):
    result = CpuBootstrap()
    if retain_owner:
        # Keep an explicit native owner as well as the frozen Python cycle:
        # even a future collector must not make this a completed trace.
        retained_schedulers.append(result.scheduler)
    return result
"#;

fn shutdown_child(test: &str, retain_owner: bool) -> Vec<Value> {
    if std::env::var("TS_HOST_SIGNAL_TEST_CHILD").as_deref() == Ok(test) {
        interpreter();
        std::env::set_var("TOKENSPEED_CACHE_EVENT_LOG", "1");
        std::env::set_var("TOKENSPEED_CACHE_EVENT_INSTANCE", "shutdown-test");
        // A failed fixture should report to captured stderr, not signal the
        // parent test process as though it were the production launcher.
        let _ = crate::LAUNCHER_FAILURE.set(crate::LauncherFailure {
            notification: Default::default(),
            logged: Default::default(),
            parent: None,
        });
        let address = frontend();
        Python::attach(|py| {
            py.import("sys")
                .unwrap()
                .getattr("path")
                .unwrap()
                .call_method1(
                    "insert",
                    (0, concat!(env!("CARGO_MANIFEST_DIR"), "/../../../python")),
                )
                .unwrap();
            py.import("tokenspeed.runtime.engine").unwrap();
            let module = PyModule::new(py, "tokenspeed.runtime.engine.rust_host").unwrap();
            let modules = py.import("sys").unwrap().getattr("modules").unwrap();
            modules
                .set_item("tokenspeed.runtime.engine.rust_host", &module)
                .unwrap();
            module.add("handshake_address", address).unwrap();
            module.add("retain_owner", retain_owner).unwrap();
            let ready = ts_wire::encode_frames(&EngineCoreReadyResponse::default());
            assert_eq!(ready.len(), 1);
            module
                .add("ready_response", PyBytes::new(py, &ready[0]))
                .unwrap();
            module
                .add("native", ts_scheduler::python_module(py).unwrap())
                .unwrap();
            py.run(
                &CString::new(CPU_BOOTSTRAP).unwrap(),
                Some(&module.dict()),
                None,
            )
            .unwrap();
            // This never returns: use the production logger guard, shutdown
            // path and process::exit, with no test-side GC or finalization.
            crate::run_host(py, -1, true).unwrap();
        });
        unreachable!();
    }
    let output = crate::tests::captured_child(test);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "{stdout}\n{stderr}");
    assert!(stdout.contains("CPU cache closed"));
    let records: Vec<Value> = stderr
        .lines()
        .filter_map(|line| line.strip_prefix("[cache-event] "))
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(records.first().unwrap()["event"], "trace_started");
    assert!(records
        .iter()
        .any(|r| r["event"] == "slot_assigned" && r["request_id"] == "shutdown-request"));
    records
}

#[test]
fn normal_host_exit_finishes_after_native_slot_release() {
    let records = shutdown_child(TEST, false);
    let released = records
        .iter()
        .position(|r| r["event"] == "slot_released" && r["request_id"] == "shutdown-request")
        .unwrap();
    assert_eq!(released + 2, records.len(), "{records:?}");
    assert_eq!(
        records.last().unwrap()["event"],
        "trace_finished",
        "{records:?}"
    );
    assert_eq!(records.last().unwrap()["evidence_valid"], true);
    for (sequence, record) in records.iter().enumerate() {
        assert_eq!(record["record_sequence"], sequence);
        assert_eq!(record["log_dropped_lines"], 0);
    }
}

#[test]
fn a_retained_native_owner_cannot_claim_a_finished_trace() {
    // Deliberately retain the bootstrap in a frozen Python cycle. A successful
    // process exit alone cannot stand in for the native owner's last release.
    let records = shutdown_child(RETAINED_TEST, true);
    assert!(!records
        .iter()
        .any(|r| r["event"] == "slot_released" || r["event"] == "trace_finished"));
}

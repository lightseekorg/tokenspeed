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

use std::path::PathBuf;
use std::sync::{
    atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    Arc, OnceLock,
};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use rustix::time::ClockId;
use serde_json::{json, Value};
use ts_client::{ClientConfig, ClientListener, EngineClient, LoadBalance, Routing};
use ts_control::{
    event_loop::{Frontend, Load},
    transport::EngineTransport,
};
use ts_wire::{EngineCoreReadyResponse, EngineRequest};

use super::{
    admit_one, batch, check_chunk, cpu_ms, output_token, phases_json, processor, request, result,
    timed, Config, Phases,
};

static SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// Own the IPC namespace through socket shutdown, including error/unwind paths.
struct Endpoints {
    directory: Option<PathBuf>,
    addresses: [String; 3],
}
impl Endpoints {
    fn new(transport: &str) -> Self {
        if transport == "tcp" {
            return Self {
                directory: None,
                addresses: std::array::from_fn(|_| "tcp://127.0.0.1:0".into()),
            };
        }
        let path = std::env::temp_dir().join(format!(
            "ts-engine-cpu-{}-{}",
            std::process::id(),
            SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).expect("unique IPC directory");
        let addresses =
            std::array::from_fn(|i| format!("ipc://{}", path.join(format!("{i}.sock")).display()));
        Self {
            directory: Some(path),
            addresses,
        }
    }
}
impl Drop for Endpoints {
    fn drop(&mut self) {
        if let Some(path) = &self.directory {
            for i in 0..3 {
                let _ = std::fs::remove_file(path.join(format!("{i}.sock")));
            }
            let _ = std::fs::remove_dir(path);
        }
    }
}

struct Shared {
    epoch: OnceLock<Instant>,
    ready_ns: Vec<AtomicU64>,
    intended_ns: Vec<AtomicU64>,
    produced: AtomicUsize,
    received: AtomicUsize,
    last_received_ns: AtomicU64,
    max_pending: AtomicUsize,
    cancel: AtomicBool,
    drained: AtomicBool,
}
impl Shared {
    fn new(steps: usize) -> Self {
        Self {
            epoch: OnceLock::new(),
            ready_ns: (0..steps).map(|_| AtomicU64::new(0)).collect(),
            intended_ns: (0..steps).map(|_| AtomicU64::new(0)).collect(),
            produced: AtomicUsize::new(0),
            received: AtomicUsize::new(0),
            last_received_ns: AtomicU64::new(0),
            max_pending: AtomicUsize::new(0),
            cancel: AtomicBool::new(false),
            drained: AtomicBool::new(false),
        }
    }
    fn ns(&self) -> u64 {
        self.epoch
            .get()
            .expect("requests were released")
            .elapsed()
            .as_nanos() as u64
    }
}

struct ClientObservation {
    started_ms: f64,
    submitted_ms: f64,
    ready_latency: Vec<f64>,
    intended_latency: Vec<f64>,
    arrival: Vec<f64>,
}

impl ClientObservation {
    fn new(steps: usize) -> Self {
        Self {
            started_ms: 0.0,
            submitted_ms: 0.0,
            ready_latency: Vec::with_capacity(steps),
            intended_latency: Vec::with_capacity(steps),
            arrival: Vec::with_capacity(steps),
        }
    }

    fn into_json(self) -> Value {
        json!({"task_started_ms":self.started_ms,"generate_return_ms":self.submitted_ms,
            "ready_to_chunk_ms":self.ready_latency,"intended_to_chunk_ms":self.intended_latency,
            "arrival_ms":self.arrival})
    }
}

/// Cancellation and joining belong to the run, not a collection of cleanup calls.
struct RunningEngine {
    shared: Arc<Shared>,
    thread: Option<JoinHandle<Value>>,
}
impl RunningEngine {
    fn finish(mut self) -> Value {
        self.shared.drained.store(true, Ordering::Release);
        self.thread
            .take()
            .unwrap()
            .join()
            .expect("engine thread succeeds")
    }
}
impl Drop for RunningEngine {
    fn drop(&mut self) {
        self.shared.cancel.store(true, Ordering::Release);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

fn engine(
    handshake: String,
    config: Config,
    shared: Arc<Shared>,
    first_result: super::ForwardResult,
) -> Value {
    super::set_phase_timers(config.phase_timers);
    let mut ready = EngineCoreReadyResponse {
        max_model_len: (config.prompt + config.steps + 1) as i64,
        ..Default::default()
    };
    ready.declare_scheduler();
    let mut transport = EngineTransport::connect(&handshake, 0, ready, Duration::from_secs(30))
        .expect("engine handshake");
    let mut p = processor(&config);
    let mut rids = Vec::with_capacity(config.batch);
    let mut phases = Phases::new();
    let deadline = Instant::now() + Duration::from_secs(300);
    let mut received_at = vec![0.0; config.batch];
    let mut admitted_at = vec![0.0; config.batch];
    while rids.len() < config.batch && !shared.cancel.load(Ordering::Acquire) {
        assert!(Instant::now() < deadline, "request ingress timed out");
        for incoming in transport.receive(Some(Duration::from_millis(20))) {
            let EngineRequest::Add(request) = incoming else {
                panic!("unexpected request during ingress")
            };
            let arrived = shared.ns();
            let spec = timed(&mut phases, "ingress_and_register", || {
                admit_one(*request, &config, &mut p)
            });
            let admitted = shared.ns();
            let row = spec
                .request_id
                .strip_prefix("bench-")
                .unwrap()
                .parse::<usize>()
                .unwrap();
            admitted_at[row] = admitted as f64 / 1e6;
            received_at[row] = arrived as f64 / 1e6;
            rids.push(spec.request_id);
        }
    }
    if shared.cancel.load(Ordering::Acquire) {
        return json!({"cancelled":true});
    }
    let all_admitted_ns = shared.ns();
    let prefill = batch(&config, &rids, true);
    let decode = batch(&config, &rids, false);
    let pacing_start = Instant::now();
    let mut first_result = Some(first_result);
    let mut producer_lateness = Vec::with_capacity(config.steps);
    for step in 0..config.steps {
        if shared.cancel.load(Ordering::Acquire) {
            return json!({"cancelled":true});
        }
        let due =
            pacing_start + Duration::from_micros(config.period_us.saturating_mul(step as u64));
        let model = if step == 0 {
            first_result.take().unwrap()
        } else {
            result(&config, step)
        };
        if let Some(wait) = due.checked_duration_since(Instant::now()) {
            thread::sleep(wait);
        }
        let intended = due.duration_since(*shared.epoch.get().unwrap()).as_nanos() as u64;
        let ready = shared.ns();
        shared.intended_ns[step].store(intended, Ordering::Release);
        shared.ready_ns[step].store(ready, Ordering::Release);
        producer_lateness.push(ready.saturating_sub(intended) as f64 / 1e6);
        let view = if step == 0 { &prefill } else { &decode };
        let name = if step == 0 { "prefill" } else { "decode" };
        let plan = timed(&mut phases, &format!("{name}_logprob_plan"), || {
            p.logprob_plan(view)
        });
        let committed = timed(&mut phases, &format!("{name}_commit"), || {
            p.commit(view, plan.input.as_ref(), &model).unwrap()
        });
        assert_eq!(committed.outputs.len(), config.batch);
        let load = Load {
            num_running: p.num_running() as i64,
            ..Load::default()
        };
        timed(&mut phases, &format!("{name}_transport_enqueue"), || {
            transport.send(committed.outputs, load)
        });
        let produced = shared.produced.fetch_add(config.batch, Ordering::AcqRel) + config.batch;
        let pending = produced.saturating_sub(shared.received.load(Ordering::Acquire));
        shared.max_pending.fetch_max(pending, Ordering::Relaxed);
        timed(&mut phases, &format!("{name}_result_drop"), || drop(model));
    }
    assert_eq!(p.num_running(), 0);
    let producer_done_ns = shared.ns();
    while !shared.drained.load(Ordering::Acquire) && !shared.cancel.load(Ordering::Acquire) {
        assert!(
            Instant::now() < deadline,
            "output consumers did not drain in time"
        );
        thread::sleep(Duration::from_millis(1));
    }
    json!({"phases":phases_json(&phases),"all_admitted_ms":all_admitted_ns as f64/1e6,
        "request_received_ms":received_at,"request_admitted_ms":admitted_at,
        "producer_done_ms":producer_done_ns as f64/1e6,"producer_lateness_ms":producer_lateness})
}

/// Genuine client/engine IPC; only scheduling and the model's output are
/// supplied by the fixture. The producer never waits for each frame's consumer.
pub fn run(config: &Config) -> Value {
    let endpoints = Endpoints::new(&config.transport);
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(config.client_threads)
        .thread_name("bench-client")
        .enable_all()
        .build()
        .unwrap();
    // Do not charge synthetic GPU-result construction or request fixture
    // generation to the engine. Real ingress copies/validation are timed.
    let model = result(config, 0);
    let requests: Vec<_> = (0..config.batch).map(|row| request(config, row)).collect();
    rt.block_on(async {
        let listener=ClientListener::bind(ClientConfig {
            handshake_address:endpoints.addresses[0].clone(),input_address:endpoints.addresses[1].clone(),
            output_address:endpoints.addresses[2].clone(),first_engine_index:0,num_engines:1,
            routing:Routing::Aggregated(LoadBalance::RoundRobin),handshake_timeout:Duration::from_secs(30),
            send_timeout:Duration::from_secs(180),
        }).await.unwrap();
        let handshake=listener.handshake_address().to_owned();
        let shared=Arc::new(Shared::new(config.steps));
        let engine_config=config.clone();let engine_shared=shared.clone();
        let guard=RunningEngine {shared:shared.clone(),thread:Some(thread::Builder::new().name("bench-control".into()).spawn(move ||engine(handshake,engine_config,engine_shared,model)).unwrap())};
        let client:Arc<EngineClient>=Arc::new(listener.accept().await.unwrap());
        let barrier=Arc::new(tokio::sync::Barrier::new(config.batch+1));
        let mut consumers=tokio::task::JoinSet::new();
        for (row,request) in requests.into_iter().enumerate() {
            let config=config.clone();let shared=shared.clone();let client=client.clone();let barrier=barrier.clone();
            let mut observation=ClientObservation::new(config.steps);
            consumers.spawn(async move {
                barrier.wait().await;
                let start=shared.ns();
                let mut stream=client.generate(request).await.unwrap();
                let submitted=shared.ns();
                observation.started_ms=start as f64/1e6;
                observation.submitted_ms=submitted as f64/1e6;
                for step in 0..config.steps {
                    let chunk=tokio::time::timeout(Duration::from_secs(180),stream.next()).await.expect("stream makes progress").expect("one chunk each step").expect("valid output");
                    let now=shared.ns();
                    observation.ready_latency.push(now.saturating_sub(shared.ready_ns[step].load(Ordering::Acquire)) as f64/1e6);
                    observation.intended_latency.push(now.saturating_sub(shared.intended_ns[step].load(Ordering::Acquire)) as f64/1e6);
                    observation.arrival.push(now as f64/1e6);
                    shared.last_received_ns.fetch_max(now,Ordering::Relaxed);
                    shared.received.fetch_add(1,Ordering::AcqRel);
                    // Constant-sized checks in the timing run. Behavioral tests
                    // exercise full values and protocol structure on small data.
                    check_chunk(&chunk,&config,row,step);
                    assert_eq!(chunk.output_ids,[output_token(step)]);
                    drop(chunk);
                    if config.consumer_delay_us>0 && step+1<config.steps {
                        tokio::time::sleep(Duration::from_micros(config.consumer_delay_us)).await;
                    }
                }
                assert!(stream.next().await.is_none());
                (row,observation)
            });
        }
        let mut rows:Vec<Option<ClientObservation>>=(0..config.batch).map(|_|None).collect();
        let cpu_start=cpu_ms(ClockId::ProcessCPUTime);
        shared.epoch.set(Instant::now()).unwrap();
        barrier.wait().await;
        while let Some(result)=consumers.join_next().await {
            let (row,data)=result.expect("consumer succeeds");rows[row]=Some(data);
        }
        let drained_ns=shared.ns();
        let process_cpu=cpu_ms(ClockId::ProcessCPUTime)-cpu_start;
        assert_eq!(shared.received.load(Ordering::Acquire),config.batch*config.steps);
        assert_eq!(client.in_flight(),[0]);
        let pending=shared.max_pending.load(Ordering::Relaxed);
        let engine=guard.finish();
        // Report construction and histogram serialization do not run in the
        // measured client tasks or block receipt of another request's chunk.
        let rows:Vec<_>=rows.into_iter().map(|r|r.unwrap().into_json()).collect();
        json!({"engine":engine,"rows":rows,"all_received_ms":shared.last_received_ns.load(Ordering::Relaxed) as f64/1e6,
            "consumer_completed_ms":drained_ns as f64/1e6,
            "process_cpu_ms":process_cpu,"max_pending_output_rows":pending,
            "checked_chunks":config.batch*config.steps})
    })
}

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

//! CPU `EventLoop` regression tests for peer prefix sharing scheduler-lock and
//! hard-pin invariants.

use std::{
    collections::{HashMap, VecDeque},
    ffi::CString,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        mpsc, Arc, Condvar, Mutex as StdMutex, Once,
    },
    thread,
    time::{Duration, Instant},
};

use parking_lot::Mutex;
use pyo3::{prelude::*, types::PyDict};
use ts_collective::{Group, Listener, Mesh};
use ts_control::{
    batch_log::BatchLogConfig,
    control::ControlConfig,
    eplb::SlotMoves,
    event_loop::{
        DataPlane, DataPlaneError, Dispatch, DpRound, EventLoop, Frontend, Load, LoopConfig, Ranks,
        Subsystems,
    },
    ingress::{Admission, IngressLimits, LogprobServability},
    output::{FinishLog, ForwardResult, OutputProcessor, Role, StreamedOutput},
    peer::{
        geometry::{DeviceField, RankGeometry},
        landing::Timeouts,
        net::{Inbound, PeerNet},
        push::{PagePusher, PushItem},
        read::Arena,
        servicer::empty_event,
        wire::{Prompt, Reply, Request},
        Peer, PeerListener, PeerProgress, PeerSettings, PeerTransfers,
    },
};
use ts_pd::LandedRequest;
use ts_scheduler::{
    CacheEvent, ExecutionPlan, ForwardEvent, ForwardEventKind, PeerEventKind, PeerEventView,
    PeerLeaseView, RequestSpec, Scheduler,
};
use ts_wire::{
    control::{
        PauseSchedulerReqInput, ProfileReq, ReleaseMemoryOccupationReqInput,
        ResumeSchedulerReqInput,
    },
    ControlOutput, ControlRequest, EngineRequest, SamplingParams, TokenizedGenerateReqInput,
};

const PAGE: i32 = 16;
const PAGE_BYTES: u64 = 64;
const PAGES: usize = 65;

fn interpreter() {
    static START: Once = Once::new();
    START.call_once(|| {
        std::env::set_var("PYTHONHOME", env!("TS_CONTROL_TEST_PYTHONHOME"));
        Python::initialize();
        Python::attach(|py| {
            let m = ts_scheduler::python_module(py).unwrap();
            let sys = py.import("sys").unwrap();
            sys.getattr("modules")
                .unwrap()
                .set_item("tokenspeed_scheduler.tokenspeed_scheduler_ext", m)
                .unwrap();
            let pkg = concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../../tokenspeed-scheduler/python"
            );
            sys.getattr("path")
                .unwrap()
                .call_method1("insert", (0, pkg))
                .unwrap();
        });
    });
}

fn peer_scheduler(name: &str, soft_pin_us: i64) -> Scheduler {
    interpreter();
    Python::attach(|py| {
        let code = format!(
            "import tokenspeed_scheduler as ts\n\
             cfg = ts.SchedulerConfig()\n\
             cfg.role = ts.SchedulerConfig.Role.Fused\n\
             cfg.prefix_granularity = {PAGE}; cfg.max_scheduled_tokens = 256\n\
             cfg.max_batch_size = 8; cfg.max_resident_requests = 8\n\
             cfg.num_device_pages = {PAGES}; cfg.num_snapshot_pages = 1; cfg.disable_l2_cache = True\n\
             cfg.cache_groups = [ts.CacheGroupConfig(group_id='full_attention', block_granularity={PAGE}, \
               total_pages={PAGES}, attn_kind=ts.AttnKind.Full, transfer_policy=ts.CacheTransferPolicy.FullSuffix)]\n\
             cfg.peer_config = ts.PeerConfig(name='{name}', incarnation=1, max_leases=16, \
               soft_pin_us={soft_pin_us}, hard_pin_us=5_000_000, lookup_timeout_us=2_000_000, \
               min_fetch_tokens={PAGE}, landing_tier=ts.PeerLandingTier.Device, \
               max_host_landing_blocks=0, host_ready_timeout_us=0, versioned_shards=False)\n\
             result = ts.Scheduler(cfg)\n"
        );
        let g = PyDict::new(py);
        py.run(&CString::new(code).unwrap(), Some(&g), None)
            .unwrap();
        Scheduler::from_python(&g.get_item("result").unwrap().unwrap()).unwrap()
    })
}

struct NoPushes;
impl PagePusher for NoPushes {
    fn push(&self, _: &str, _: &str, _: Duration, _: Vec<PushItem>) -> Result<(), String> {
        Err("no pushes".into())
    }
}

fn rank_geometry(base: u64) -> RankGeometry {
    RankGeometry {
        session: "local".into(),
        device: vec![DeviceField {
            group_id: 0,
            field_id: "layer.0.kv".into(),
            address: base,
            page_stride: PAGE_BYTES,
            bytes: PAGE_BYTES,
        }],
        host: None,
        push: None,
    }
}

fn peer_settings(name: &str) -> PeerSettings {
    PeerSettings {
        name: name.to_owned(),
        incarnation: 1,
        deployment_id: "deployment".into(),
        weight_version: "v1".into(),
        timeouts: Timeouts {
            answer_us: 1_000_000,
            quarantine_us: 5_000_000,
            push_us: 10_000_000,
        },
        record_lifetime: Duration::from_secs(10),
    }
}

fn loop_config(in_flight_depth: usize) -> LoopConfig {
    let kv = (PAGES - 1) as i64;
    LoopConfig {
        in_flight_depth,
        limits: IngressLimits {
            vocab_size: 100_000,
            eos_token_ids: Vec::new(),
            top_k_exclusive_max: Some(ts_control::sampling::TOP_K_FUSED_MAX),
            max_model_len: 4096,
            max_req_input_len: 4000,
            enable_output_logprobs: false,
            requires_bootstrap: false,
            multimodal: false,
            grammar: false,
            reasoning: None,
            reasoning_usage: None,
            logprobs: LogprobServability {
                supports_prompt_logprobs: true,
                logprob_order: "torch".into(),
                pipeline_parallel: false,
                disaggregated: false,
                lm_head_tp_under_dp: false,
                computes_prompt_logprobs: true,
            },
        },
        kv_total_pages: kv,
        kv_token_capacity: kv * i64::from(PAGE),
        ngram_context_len: 0,
        request_token_history: false,
        request_token_history_lookback: None,
        controls: ControlConfig {
            memory_saver: true,
            memory_regions: vec!["kv_cache".into(), "weights".into()],
            shared_weight_version: None,
            allow_reload_forward: true,
        },
        batch_log: BatchLogConfig {
            enabled: false,
            decode_log_interval: 40,
            num_total_pages: kv,
            spec_num_steps: 0,
            dp_rank: 0,
            disaggregated: false,
            cache_state_group_ids: Vec::new(),
            log_spec_accept_lengths: false,
        },
        kv_events: false,
        epd_receive_enabled: false,
        eager_grammar: false,
        prefill_delay: None,
    }
}

#[derive(Default)]
struct PauseHook {
    state: StdMutex<(bool, bool)>,
    cv: Condvar,
}
impl PauseHook {
    fn arm(&self) {
        *self.state.lock().unwrap() = (true, false);
    }
    fn wait_entered(&self) {
        let (_g, r) = self
            .cv
            .wait_timeout_while(self.state.lock().unwrap(), Duration::from_secs(5), |s| !s.1)
            .unwrap();
        assert!(!r.timed_out(), "timed out waiting for PauseHook");
    }
    fn release(&self) {
        *self.state.lock().unwrap() = (false, false);
        self.cv.notify_all();
    }
    fn pause_if_armed(&self) {
        let mut s = self.state.lock().unwrap();
        if !s.0 {
            return;
        }
        s.1 = true;
        self.cv.notify_all();
        drop(self.cv.wait_while(s, |st| st.0).unwrap());
    }
}

#[derive(Default)]
struct FrontendState {
    inbox: StdMutex<VecDeque<EngineRequest>>,
    finished: StdMutex<Vec<String>>,
    replies: StdMutex<Vec<ControlOutput>>,
    receive_hook: PauseHook,
    receive_count: AtomicUsize,
}
impl FrontendState {
    fn push(&self, req: EngineRequest) {
        self.inbox.lock().unwrap().push_back(req);
    }
    fn wait_receive_gt(&self, prev: usize) {
        let t = Instant::now();
        while t.elapsed() < Duration::from_secs(5) {
            if self.receive_count.load(Ordering::SeqCst) > prev {
                return;
            }
            thread::sleep(Duration::from_millis(1));
        }
        panic!("timed out waiting for frontend receive count > {prev}");
    }
    fn wait_finished(&self, rid: &str) {
        let t = Instant::now();
        while t.elapsed() < Duration::from_secs(5) {
            if self.finished.lock().unwrap().iter().any(|r| r == rid) {
                return;
            }
            thread::sleep(Duration::from_millis(2));
        }
        panic!("timed out waiting for {rid}");
    }
    fn wait_replies(&self, count: usize) -> Vec<ControlOutput> {
        let t = Instant::now();
        while t.elapsed() < Duration::from_secs(5) {
            let r = self.replies.lock().unwrap();
            if r.len() >= count {
                return r.clone();
            }
            drop(r);
            thread::sleep(Duration::from_millis(2));
        }
        panic!("timed out waiting for {count} replies");
    }
}

struct ChannelFrontend {
    state: Arc<FrontendState>,
}
impl Frontend for ChannelFrontend {
    fn receive(&mut self, wait: Option<Duration>) -> Vec<EngineRequest> {
        self.state.receive_hook.pause_if_armed();
        self.state.receive_count.fetch_add(1, Ordering::SeqCst);
        let mut items: Vec<_> = self.state.inbox.lock().unwrap().drain(..).collect();
        if items.is_empty() {
            if let Some(w) = wait {
                thread::sleep(w.min(Duration::from_millis(2)));
                items.extend(self.state.inbox.lock().unwrap().drain(..));
            }
        }
        items
    }
    fn closed(&self) -> bool {
        false
    }
    fn send(&mut self, outputs: Vec<StreamedOutput>, _: Load) {
        let mut fin = self.state.finished.lock().unwrap();
        for o in outputs {
            if o.finish_reason.is_some() {
                fin.push(o.rid);
            }
        }
    }
    fn reply(&mut self, output: ControlOutput) {
        self.state.replies.lock().unwrap().push(output);
    }
}

#[derive(Default)]
struct DeviceHooks {
    idle: PauseHook,
    complete: PauseHook,
    execute: PauseHook,
    ops: StdMutex<Vec<String>>,
    copy_only_plans: AtomicUsize,
    idle_execute_calls: AtomicUsize,
}
struct HookedDevice {
    gen: HashMap<String, i32>,
    hooks: Arc<DeviceHooks>,
}
impl DataPlane for HookedDevice {
    type Pending = ForwardResult;
    fn admit(&mut self, reqs: &[Admission]) -> Result<(), DataPlaneError> {
        for r in reqs {
            self.gen.insert(r.rid.clone(), 0);
        }
        Ok(())
    }
    fn execute(
        &mut self,
        plan: ExecutionPlan,
        d: &Dispatch<'_>,
    ) -> Result<Option<ForwardResult>, DataPlaneError> {
        if plan.has_pages_to_copy() && d.forward.is_none() {
            self.hooks.copy_only_plans.fetch_add(1, Ordering::SeqCst);
        }
        if d.dp.is_some_and(|dp| dp.idle_forward) {
            self.hooks.idle_execute_calls.fetch_add(1, Ordering::SeqCst);
        }
        self.hooks.execute.pause_if_armed();
        let Some(b) = d.forward else {
            return Ok(None);
        };
        let mut res = ForwardResult::default();
        for rid in &b.request_ids {
            let g = self.gen.entry(rid.clone()).or_default();
            *g += 1;
            res.output_tokens.push(1000 + *g - 1);
            res.output_lengths.push(1);
        }
        res.output_logprobs = Some(vec![-1.0; res.output_tokens.len()]);
        Ok(Some(res))
    }
    fn complete(&mut self, p: ForwardResult) -> Result<ForwardResult, DataPlaneError> {
        self.hooks.complete.pause_if_armed();
        Ok(p)
    }
    fn poll_cache_events(&mut self) -> Result<Vec<CacheEvent>, DataPlaneError> {
        Ok(Vec::new())
    }
    fn replay_retirement_enabled(&self) -> bool {
        false
    }
    fn poll_replay_retirement(&mut self) -> Result<u64, DataPlaneError> {
        Ok(0)
    }
    fn release(&mut self, rids: &[String], _: Option<u64>) -> Result<(), DataPlaneError> {
        for r in rids {
            self.gen.remove(r);
        }
        Ok(())
    }
    fn land(&mut self, _: Option<(i32, Vec<i32>)>, _: Option<i32>) -> Result<(), DataPlaneError> {
        Ok(())
    }
    fn idle_forward(&mut self, _: &DpRound) -> Result<(), DataPlaneError> {
        self.hooks.idle.pause_if_armed();
        Ok(())
    }
    fn check_weight_update(
        &mut self,
        _: &ControlRequest,
    ) -> Result<(bool, String), DataPlaneError> {
        Ok((true, "ok".into()))
    }
    fn update_weights(&mut self, _: &ControlRequest) -> Result<(bool, String), DataPlaneError> {
        Ok((true, "ok".into()))
    }
    fn commit_weight_update(&mut self, _: bool) -> Result<(), DataPlaneError> {
        Ok(())
    }
    fn profile(&mut self, _: &ProfileReq) -> Result<(bool, String), DataPlaneError> {
        Ok((true, "ok".into()))
    }
    fn release_memory(&mut self, tags: &[String]) -> Result<(), DataPlaneError> {
        self.hooks
            .ops
            .lock()
            .unwrap()
            .push(format!("release:{}", tags.join(",")));
        Ok(())
    }
    fn reload_forward(&mut self, _: &ControlRequest) -> Result<(bool, String), DataPlaneError> {
        Ok((true, "ok".into()))
    }
    fn resume_memory(&mut self, _: &[String]) -> Result<(), DataPlaneError> {
        Ok(())
    }
    fn finish_memory_resume(&mut self) -> Result<(), DataPlaneError> {
        Ok(())
    }
    fn attach_grammars(&mut self, _: &[(String, u64, Vec<i32>)]) -> Result<(), DataPlaneError> {
        Ok(())
    }
    fn epd_publish(&mut self, _: &[LandedRequest]) -> Result<(), DataPlaneError> {
        Ok(())
    }
    fn epd_released(&mut self) -> Result<Vec<usize>, DataPlaneError> {
        Ok(Vec::new())
    }
    fn snapshot_expert_load(&mut self) -> Result<(Vec<Vec<i64>>, Vec<Vec<i32>>), DataPlaneError> {
        Ok((Vec::new(), Vec::new()))
    }
    fn apply_expert_placement(
        &mut self,
        _: &[usize],
        _: &[Vec<i32>],
        _: &[SlotMoves],
    ) -> Result<(), DataPlaneError> {
        Ok(())
    }
}

fn add_req(rid: &str, tokens: Vec<i32>, max_new: i64) -> EngineRequest {
    EngineRequest::Add(Box::new(TokenizedGenerateReqInput {
        input_ids: Some(tokens),
        stream: true,
        ..TokenizedGenerateReqInput::new(
            rid,
            SamplingParams {
                max_new_tokens: Some(max_new),
                ..Default::default()
            },
        )
    }))
}

fn make_lookup_prompt(prefix: &[i32]) -> Prompt {
    let mut h = peer_scheduler("prompt-builder", 5_000_000);
    let mut full = prefix.to_vec();
    full.extend(99_000..99_000 + PAGE);
    h.submit_requests(&[RequestSpec {
        admission_id: 0,
        request_id: "p".into(),
        session_id: String::new(),
        cache_group_id: String::new(),
        shared_prefix_tokens: Vec::new(),
        peer_holder: "holder".into(),
        tokens: full,
        max_new_tokens: 1,
        max_cached_prefix_tokens: i32::MAX,
    }])
    .unwrap();
    let _ = h.next_execution_plan(true, false);
    let l = h.take_peer_lookups().pop().unwrap();
    Prompt {
        prefix_hashes: l.prefix_hashes,
        tokens: l.tokens,
        max_tokens: l.max_tokens,
    }
}

#[allow(clippy::too_many_arguments)]
fn spawn_rank(
    rank: usize,
    listener: Listener,
    addrs: Vec<String>,
    rep_ranks: Vec<usize>,
    dp_ranks: Option<Vec<usize>>,
    sched: Arc<Mutex<Scheduler>>,
    fe: Option<Arc<FrontendState>>,
    hooks: Arc<DeviceHooks>,
    peer: Option<Peer>,
    depth: usize,
    stop: Arc<AtomicBool>,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        let mesh = Mesh::connect(rank, listener, &addrs, Duration::from_secs(10)).unwrap();
        let outputs = OutputProcessor::new(
            None,
            1,
            None,
            Role::Plain,
            false,
            FinishLog {
                enabled: false,
                request_stats: false,
                ttft_percentage: 0,
            },
            false,
            None,
        );
        let mut ev = EventLoop::new(
            sched,
            HookedDevice {
                gen: HashMap::new(),
                hooks,
            },
            fe.map(|state| ChannelFrontend { state }),
            outputs,
            loop_config(depth),
            Ranks {
                mesh,
                replica: Group::new(1, rep_ranks, rank).unwrap(),
                pp: None,
                dp: dp_ranks.map(|r| Group::new(0, r, rank).unwrap()),
                ep: None,
            },
            Subsystems {
                transfer: None,
                embeddings: None,
                metrics: None,
                kv_events: None,
                eplb: None,
                grammars: None,
                l3_metrics: None,
                l3: None,
                disk: None,
                disk_metrics: None,
                shared_l2: None,
                peer,
            },
        );
        ev.run(&stop).unwrap();
    })
}

#[test]
fn unlocked_gate_idle_forward_and_overlap_complete_answer_peer() {
    interpreter();
    let ls = [(); 2].map(|_| Listener::bind("127.0.0.1").unwrap());
    let addrs: Vec<_> = ls
        .iter()
        .map(|l| format!("127.0.0.1:{}", l.port().unwrap()))
        .collect();
    let pl = PeerListener::bind("127.0.0.1", 0).unwrap();
    let holder_name = format!("127.0.0.1:{}", pl.port());
    let (s0, s1) = (
        Arc::new(Mutex::new(peer_scheduler(&holder_name, 5_000_000))),
        Arc::new(Mutex::new(peer_scheduler("dp1", 5_000_000))),
    );
    let p0 = Peer::start(
        peer_settings(&holder_name),
        PeerProgress::Coordinated,
        0,
        vec![rank_geometry(0)],
        Some(pl),
        s0.clone(),
        PeerTransfers {
            reader: Arc::new(Arena::new(200_000)),
            pusher: Arc::new(NoPushes),
        },
    )
    .unwrap();
    let (fe0, fe1, h0, h1, stop) = (
        Arc::new(FrontendState::default()),
        Arc::new(FrontendState::default()),
        Arc::new(DeviceHooks::default()),
        Arc::new(DeviceHooks::default()),
        Arc::new(AtomicBool::new(false)),
    );
    let [l0, l1] = ls;
    let t0 = spawn_rank(
        0,
        l0,
        addrs.clone(),
        vec![0],
        Some(vec![0, 1]),
        s0.clone(),
        Some(fe0.clone()),
        h0.clone(),
        Some(p0),
        2,
        stop.clone(),
    );
    let t1 = spawn_rank(
        1,
        l1,
        addrs,
        vec![1],
        Some(vec![0, 1]),
        s1,
        Some(fe1.clone()),
        h1.clone(),
        None,
        2,
        stop.clone(),
    );

    let seed: Vec<i32> = (1..=3 * PAGE).collect();
    fe0.push(add_req("seed", seed.clone(), 1));
    fe0.wait_finished("seed");
    let (tx, rx) = mpsc::channel();
    let client = PeerNet::start("127.0.0.1", 0, tx).unwrap();
    let caddr = format!("127.0.0.1:{}", client.port());
    let prompt = make_lookup_prompt(&seed);
    let ask_pin = |id: u64| -> (i32, bool) {
        let lease = PeerLeaseView {
            reader: caddr.clone(),
            id,
        };
        client
            .request(
                &holder_name,
                &Request::Lookup {
                    lease: lease.clone(),
                    prompt: prompt.clone(),
                },
            )
            .unwrap();
        let Inbound::Reply(Reply::Answer { answer: la, .. }) =
            Inbound::decode(&rx.recv_timeout(Duration::from_millis(500)).unwrap()).unwrap()
        else {
            panic!()
        };
        client
            .request(
                &holder_name,
                &Request::Pin {
                    lease: lease.clone(),
                },
            )
            .unwrap();
        let Inbound::Reply(Reply::Answer { answer: pa, .. }) =
            Inbound::decode(&rx.recv_timeout(Duration::from_millis(500)).unwrap()).unwrap()
        else {
            panic!()
        };
        client
            .request(
                &holder_name,
                &Request::Release {
                    lease,
                    demote: false,
                },
            )
            .unwrap();
        let _ = rx.recv_timeout(Duration::from_millis(500)).unwrap();
        (la.num_tokens, pa.held)
    };

    // 1a. gate() keeps scheduler LOCKED while in_flight is non-empty so a peer
    //     reader cannot be handed Device pages from an uncompleted batch.
    h0.execute.arm();
    h1.execute.arm();
    fe0.push(add_req("r0-gate", (400..400 + 2 * PAGE).collect(), 2));
    h0.execute.wait_entered();
    h1.execute.wait_entered();
    fe1.receive_hook.arm();
    h0.execute.release();
    h1.execute.release();
    fe1.receive_hook.wait_entered();
    thread::sleep(Duration::from_millis(15));
    client
        .request(
            &holder_name,
            &Request::Lookup {
                lease: PeerLeaseView {
                    reader: caddr.clone(),
                    id: 100,
                },
                prompt: prompt.clone(),
            },
        )
        .unwrap();
    assert!(rx.recv_timeout(Duration::from_millis(80)).is_err());
    fe1.receive_hook.release();
    let Inbound::Reply(Reply::Answer { answer: la, .. }) =
        Inbound::decode(&rx.recv_timeout(Duration::from_secs(2)).unwrap()).unwrap()
    else {
        panic!()
    };
    assert_eq!(la.num_tokens, 3 * PAGE);
    fe0.wait_finished("r0-gate");

    // 1b. gate() unlocks scheduler when in_flight is empty and no copies are pending.
    h1.execute.arm();
    fe1.push(add_req("r1-gate", (450..450 + 2 * PAGE).collect(), 2));
    h1.execute.wait_entered();
    fe1.receive_hook.arm();
    h1.execute.release();
    fe1.receive_hook.wait_entered();
    thread::sleep(Duration::from_millis(15));
    assert_eq!(ask_pin(101), (3 * PAGE, true));
    fe1.receive_hook.release();
    fe1.wait_finished("r1-gate");

    // 2. Unlocked idle_forward answers Lookup+Pin immediately.
    fe0.push(EngineRequest::Control(Box::new(
        ControlRequest::PauseScheduler(PauseSchedulerReqInput {
            rid: None,
            http_worker_ipc: None,
            mode: "keep".into(),
        }),
    )));
    fe0.wait_replies(1);
    h0.idle.arm();
    fe1.push(add_req("r1-only", (500..500 + 2 * PAGE).collect(), 2));
    h0.idle.wait_entered();
    assert_eq!(ask_pin(102), (3 * PAGE, true));
    h0.idle.release();
    fe1.wait_finished("r1-only");
    fe0.push(EngineRequest::Control(Box::new(
        ControlRequest::ResumeScheduler(ResumeSchedulerReqInput {
            rid: None,
            http_worker_ipc: None,
        }),
    )));
    fe0.wait_replies(2);

    // 3. commit_down_to keeps scheduler unlocked under overlap scheduling so
    //    ts-peer-servicer still answers peer RPCs during long prefills.
    h0.complete.arm();
    fe0.push(add_req("r0-overlap", (800..800 + 2 * PAGE).collect(), 4));
    h0.complete.wait_entered();
    assert_eq!(ask_pin(103), (3 * PAGE, true));
    h0.complete.release();
    fe0.wait_finished("r0-overlap");
    stop.store(true, Ordering::Relaxed);
    t0.join().unwrap();
    t1.join().unwrap();
}

#[test]
fn sync_peer_holder_changes_preserves_hard_pins_on_tp_and_straddling_pin_blocks_clear() {
    interpreter();
    let arena = Arena::new(200_000);
    let ls = [(); 2].map(|_| Listener::bind("127.0.0.1").unwrap());
    let addrs: Vec<_> = ls
        .iter()
        .map(|l| format!("127.0.0.1:{}", l.port().unwrap()))
        .collect();
    let pl = PeerListener::bind("127.0.0.1", 0).unwrap();
    let holder_name = format!("127.0.0.1:{}", pl.port());
    let (s0, s1) = (
        Arc::new(Mutex::new(peer_scheduler(&holder_name, 5_000_000))),
        Arc::new(Mutex::new(peer_scheduler(&holder_name, 5_000_000))),
    );
    let geoms = vec![rank_geometry(0), rank_geometry(50_000)];
    let p0 = Peer::start(
        peer_settings(&holder_name),
        PeerProgress::Coordinated,
        0,
        geoms.clone(),
        Some(pl),
        s0.clone(),
        PeerTransfers {
            reader: Arc::new(arena.clone()),
            pusher: Arc::new(NoPushes),
        },
    )
    .unwrap();
    let p1 = Peer::start(
        peer_settings(&holder_name),
        PeerProgress::Coordinated,
        1,
        geoms,
        None,
        s1.clone(),
        PeerTransfers {
            reader: Arc::new(arena),
            pusher: Arc::new(NoPushes),
        },
    )
    .unwrap();
    let (fe0, h0, h1, stop) = (
        Arc::new(FrontendState::default()),
        Arc::new(DeviceHooks::default()),
        Arc::new(DeviceHooks::default()),
        Arc::new(AtomicBool::new(false)),
    );
    let [l0, l1] = ls;
    let t0 = spawn_rank(
        0,
        l0,
        addrs.clone(),
        vec![0, 1],
        None,
        s0.clone(),
        Some(fe0.clone()),
        h0.clone(),
        Some(p0),
        1,
        stop.clone(),
    );
    let t1 = spawn_rank(
        1,
        l1,
        addrs,
        vec![0, 1],
        None,
        s1.clone(),
        None,
        h1.clone(),
        Some(p1),
        1,
        stop.clone(),
    );

    let seed: Vec<i32> = (1..=3 * PAGE).collect();
    fe0.push(add_req("seed-tp", seed.clone(), 1));
    fe0.wait_finished("seed-tp");
    let (tx, rx) = mpsc::channel();
    let client = PeerNet::start("127.0.0.1", 0, tx).unwrap();
    let lease = PeerLeaseView {
        reader: format!("127.0.0.1:{}", client.port()),
        id: 777,
    };
    client
        .request(
            &holder_name,
            &Request::Lookup {
                lease: lease.clone(),
                prompt: make_lookup_prompt(&seed),
            },
        )
        .unwrap();
    let _ = rx.recv_timeout(Duration::from_secs(5)).unwrap();
    client
        .request(
            &holder_name,
            &Request::Pin {
                lease: lease.clone(),
            },
        )
        .unwrap();
    let Inbound::Reply(Reply::Answer { answer, .. }) =
        Inbound::decode(&rx.recv_timeout(Duration::from_secs(5)).unwrap()).unwrap()
    else {
        panic!()
    };
    assert!(answer.held);

    fe0.push(EngineRequest::Control(Box::new(
        ControlRequest::ReleaseMemoryOccupation(ReleaseMemoryOccupationReqInput {
            rid: None,
            http_worker_ipc: None,
            tags: Some(vec!["kv_cache".into()]),
        }),
    )));
    thread::sleep(Duration::from_millis(60));
    assert!(
        fe0.replies.lock().unwrap().is_empty()
            && h0.ops.lock().unwrap().is_empty()
            && h1.ops.lock().unwrap().is_empty()
    );
    assert!(s0.lock().peer_stats().pinned_blocks > 0 && s1.lock().peer_stats().pinned_blocks > 0);
    client
        .request(
            &holder_name,
            &Request::Release {
                lease,
                demote: false,
            },
        )
        .unwrap();
    let _ = rx.recv_timeout(Duration::from_secs(5)).unwrap();
    assert!(
        matches!(&fe0.wait_replies(1)[0], ControlOutput::ReleaseMemoryOccupation(o) if o.success)
    );
    assert_eq!(*h0.ops.lock().unwrap(), vec!["release:kv_cache"]);
    assert_eq!(*h1.ops.lock().unwrap(), vec!["release:kv_cache"]);
    stop.store(true, Ordering::Relaxed);
    t0.join().unwrap();
    t1.join().unwrap();

    // A Pin straddling soft_pin_us promotes to Phase::kHard before Expire runs.
    let mut sched = peer_scheduler("holder", 40_000);
    let tokens: Vec<i32> = (1..=4 * PAGE).collect();
    sched
        .submit_requests(&[RequestSpec {
            admission_id: 0,
            request_id: "warm".into(),
            session_id: String::new(),
            cache_group_id: String::new(),
            shared_prefix_tokens: Vec::new(),
            peer_holder: String::new(),
            tokens: tokens.clone(),
            max_new_tokens: 1,
            max_cached_prefix_tokens: i32::MAX,
        }])
        .unwrap();
    let _ = sched.next_execution_plan(true, false);
    sched
        .advance(&[
            ForwardEvent {
                kind: ForwardEventKind::ExtendResult,
                request_id: "warm".into(),
                tokens: vec![7],
                spec_candidate_ids: Vec::new(),
                reserve_num_tokens: 0,
                bootstrap_token: -1,
            },
            ForwardEvent {
                kind: ForwardEventKind::Finish,
                request_id: "warm".into(),
                tokens: Vec::new(),
                spec_candidate_ids: Vec::new(),
                reserve_num_tokens: 0,
                bootstrap_token: -1,
            },
        ])
        .unwrap();
    let _ = sched.next_execution_plan(true, false);
    let p = make_lookup_prompt(&tokens[..(3 * PAGE) as usize]);
    let lease = PeerLeaseView {
        reader: "127.0.0.1:9999".into(),
        id: 101,
    };
    let lev = PeerEventView {
        kind: PeerEventKind::Lookup,
        lease: lease.clone(),
        prefix_hashes: p.prefix_hashes,
        tokens: p.tokens,
        max_tokens: p.max_tokens,
        ..empty_event()
    };
    let ans = sched.answer_peer_lookup_bounded(&lev, 0);
    assert_eq!(ans.num_tokens, 3 * PAGE);
    let t0 = 2_000_000_000_000_000_i64;
    sched.advance_peer(t0, &[lev]).unwrap();
    let pev = PeerEventView {
        kind: PeerEventKind::Pin,
        lease: lease.clone(),
        num_tokens: ans.num_tokens,
        pages: ans.pages,
        ..empty_event()
    };
    sched.advance_peer(t0 + 100_000, &[pev]).unwrap();
    assert!(sched.peer_lease_held(&lease) && !sched.clear_cache());
    let rev = PeerEventView {
        kind: PeerEventKind::Release,
        lease: lease.clone(),
        ..empty_event()
    };
    sched.advance_peer(t0 + 150_000, &[rev]).unwrap();
    assert!(!sched.peer_lease_held(&lease) && sched.clear_cache());
}

fn compaction_peer_scheduler(name: &str, soft_pin_us: i64) -> Scheduler {
    interpreter();
    Python::attach(|py| {
        let code = format!(
            "import tokenspeed_scheduler as ts\n\
             cfg = ts.SchedulerConfig()\n\
             cfg.role = ts.SchedulerConfig.Role.Fused\n\
             cfg.prefix_granularity = 8; cfg.max_scheduled_tokens = 64\n\
             cfg.max_batch_size = 64; cfg.max_resident_requests = 64\n\
             cfg.max_retracted_requests = 0; cfg.decode_input_tokens = 1\n\
             cfg.num_device_pages = 29; cfg.num_snapshot_pages = 1; cfg.disable_l2_cache = True\n\
             cfg.cache_groups = [\n\
               ts.CacheGroupConfig(group_id='full_attention', block_granularity=8,\n\
                 total_pages=58, attn_kind=ts.AttnKind.Full,\n\
                 cache_blocks_per_lcm_block=2, transfer_policy=ts.CacheTransferPolicy.FullSuffix),\n\
               ts.CacheGroupConfig(group_id='sliding', block_granularity=8,\n\
                 total_pages=29, attn_kind=ts.AttnKind.SlidingWindow,\n\
                 sliding_window_tokens=64, cache_blocks_per_lcm_block=1,\n\
                 transfer_policy=ts.CacheTransferPolicy.FullSuffix),\n\
             ]\n\
             cfg.peer_config = ts.PeerConfig(name='{name}', incarnation=1, max_leases=32,\n\
               soft_pin_us={soft_pin_us}, hard_pin_us=30_000_000, lookup_timeout_us=2_000_000,\n\
               min_fetch_tokens=8, landing_tier=ts.PeerLandingTier.Device,\n\
               max_host_landing_blocks=0, host_ready_timeout_us=0, versioned_shards=False)\n\
             result = ts.Scheduler(cfg)\n"
        );
        let g = PyDict::new(py);
        py.run(&CString::new(code).unwrap(), Some(&g), None)
            .unwrap();
        Scheduler::from_python(&g.get_item("result").unwrap().unwrap()).unwrap()
    })
}

fn make_compaction_lookup_prompt(prefix: &[i32]) -> Prompt {
    let mut h = compaction_peer_scheduler("prompt-builder-compact", 30_000_000);
    let mut full = prefix.to_vec();
    full.extend(99_000..99_008);
    h.submit_requests(&[RequestSpec {
        admission_id: 0,
        request_id: "p".into(),
        session_id: String::new(),
        cache_group_id: String::new(),
        shared_prefix_tokens: Vec::new(),
        peer_holder: "holder".into(),
        tokens: full,
        max_new_tokens: 1,
        max_cached_prefix_tokens: i32::MAX,
    }])
    .unwrap();
    let _ = h.next_execution_plan(true, false);
    let l = h.take_peer_lookups().pop().unwrap();
    Prompt {
        prefix_hashes: l.prefix_hashes,
        tokens: l.tokens,
        max_tokens: l.max_tokens,
    }
}

fn compaction_rank_geometry(base: u64) -> RankGeometry {
    RankGeometry {
        session: "local".into(),
        device: vec![
            DeviceField {
                group_id: 0,
                field_id: "layer.0.full".into(),
                address: base,
                page_stride: PAGE_BYTES,
                bytes: PAGE_BYTES,
            },
            DeviceField {
                group_id: 1,
                field_id: "layer.0.sliding".into(),
                address: base + 10_000,
                page_stride: PAGE_BYTES,
                bytes: PAGE_BYTES,
            },
        ],
        host: None,
        push: None,
    }
}

#[test]
fn copies_confirmed_at_holds_gate_and_idle_forward_locked_until_confirming_forward_completes() {
    interpreter();
    let ls = [(); 2].map(|_| Listener::bind("127.0.0.1").unwrap());
    let addrs: Vec<_> = ls
        .iter()
        .map(|l| format!("127.0.0.1:{}", l.port().unwrap()))
        .collect();
    let pl = PeerListener::bind("127.0.0.1", 0).unwrap();
    let holder_name = format!("127.0.0.1:{}", pl.port());
    let (s0, s1) = (
        Arc::new(Mutex::new(compaction_peer_scheduler(
            &holder_name,
            30_000_000,
        ))),
        Arc::new(Mutex::new(peer_scheduler("dp1-compact", 30_000_000))),
    );
    let p0 = Peer::start(
        peer_settings(&holder_name),
        PeerProgress::Coordinated,
        0,
        vec![compaction_rank_geometry(0)],
        Some(pl),
        s0.clone(),
        PeerTransfers {
            reader: Arc::new(Arena::new(200_000)),
            pusher: Arc::new(NoPushes),
        },
    )
    .unwrap();
    let (fe0, fe1, h0, h1, stop) = (
        Arc::new(FrontendState::default()),
        Arc::new(FrontendState::default()),
        Arc::new(DeviceHooks::default()),
        Arc::new(DeviceHooks::default()),
        Arc::new(AtomicBool::new(false)),
    );
    let [l0, l1] = ls;
    let t0 = spawn_rank(
        0,
        l0,
        addrs.clone(),
        vec![0],
        Some(vec![0, 1]),
        s0.clone(),
        Some(fe0.clone()),
        h0.clone(),
        Some(p0),
        1,
        stop.clone(),
    );
    let t1 = spawn_rank(
        1,
        l1,
        addrs,
        vec![1],
        Some(vec![0, 1]),
        s1,
        Some(fe1.clone()),
        h1.clone(),
        None,
        1,
        stop.clone(),
    );

    // 1. Fragment the pool with eight 1-page prompts submitted in one round so
    //    each full_attention parent holds [cached page | hole] alongside eight
    //    cached sliding parents.
    let a_tokens: Vec<Vec<i32>> = (0..8)
        .map(|k| (100 + 50 * k..100 + 50 * k + 8).collect())
        .collect();
    fe0.receive_hook.arm();
    fe0.receive_hook.wait_entered();
    for (k, toks) in a_tokens.iter().enumerate() {
        fe0.push(add_req(&format!("a{k}"), toks.clone(), 1));
    }
    fe0.receive_hook.release();
    for k in 0..8 {
        fe0.wait_finished(&format!("a{k}"));
    }

    let (tx, rx) = mpsc::channel();
    let client = PeerNet::start("127.0.0.1", 0, tx).unwrap();
    let caddr = format!("127.0.0.1:{}", client.port());
    let send_lookup = |id: u64, prompt: &Prompt| {
        client
            .request(
                &holder_name,
                &Request::Lookup {
                    lease: PeerLeaseView {
                        reader: caddr.clone(),
                        id,
                    },
                    prompt: prompt.clone(),
                },
            )
            .unwrap();
    };
    let recv_answer = || -> (i32, bool) {
        let Inbound::Reply(Reply::Answer { answer, .. }) =
            Inbound::decode(&rx.recv_timeout(Duration::from_secs(5)).unwrap()).unwrap()
        else {
            panic!("expected peer Answer reply")
        };
        (answer.num_tokens, answer.held)
    };
    let pin_prompt = |id: u64, prompt: &Prompt| -> PeerLeaseView {
        let lease = PeerLeaseView {
            reader: caddr.clone(),
            id,
        };
        send_lookup(id, prompt);
        let (tokens, _) = recv_answer();
        assert_eq!(tokens, 8);
        client
            .request(
                &holder_name,
                &Request::Pin {
                    lease: lease.clone(),
                },
            )
            .unwrap();
        let (_, held) = recv_answer();
        assert!(held);
        lease
    };
    let release_lease = |lease: PeerLeaseView| {
        client
            .request(
                &holder_name,
                &Request::Release {
                    lease,
                    demote: false,
                },
            )
            .unwrap();
        let _ = recv_answer();
    };

    // 2. Hard-pin a1..a7 (leaving a0 unpinned so chunk 1 of R can take a0's
    //    sliding parent and fill the eight full_attention holes).
    let a_prompts: Vec<Prompt> = a_tokens
        .iter()
        .map(|toks| make_compaction_lookup_prompt(toks))
        .collect();
    let mut leases: Vec<Option<PeerLeaseView>> = (0..8)
        .map(|k| (k >= 1).then(|| pin_prompt(10 + k as u64, &a_prompts[k])))
        .collect();

    // 3. Submit the 126-token lone request R. Chunk 1 (64 tokens) runs and
    //    completes; chunk 2 blocks on a1..a7, so lone-request recovery
    //    publishes chunk 1's eight full pages first without forwarding.
    let r_tokens: Vec<i32> = (3000..3126).collect();
    let r_page0_prompt = make_compaction_lookup_prompt(&r_tokens[..8]);
    fe0.push(add_req("R", r_tokens.clone(), 1));
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut probe_id = 100u64;
    loop {
        assert!(
            Instant::now() < deadline,
            "timed out waiting for R chunk 1 pages to publish"
        );
        probe_id += 1;
        send_lookup(probe_id, &r_page0_prompt);
        let (tokens, _) = recv_answer();
        if tokens >= 8 {
            break;
        }
        thread::sleep(Duration::from_millis(5));
    }

    // 4. Arm h0.execute and release one kept parent (a1) plus one vacatable
    //    parent (a7), keeping a2..a6 pinned. Lone-request compaction copies
    //    R's published full_attention block out of a7's parent into a1's
    //    cleared slot (emitting pages_to_copy), while the chunk-2 retry still
    //    fails for lack of sliding parents (forward == None, so in_flight
    //    stays empty while copies_confirmed_at is set).
    assert_eq!(h0.copy_only_plans.load(Ordering::SeqCst), 0);
    h0.execute.arm();
    for k in [1usize, 7] {
        release_lease(leases[k].take().unwrap());
    }
    h0.execute.wait_entered();
    assert_eq!(
        h0.copy_only_plans.load(Ordering::SeqCst),
        1,
        "compaction must emit a plan with pages_to_copy and no forward"
    );

    // 5. While rank 0 is paused inside the copy-only execute, arm dp1's
    //    receive_hook so rank 0 enters gate() on the next round with
    //    in_flight.is_empty() == true and copies_confirmed_at == Some(...).
    //    Wait for rank 0's receive_count to advance so we know it exited the
    //    copy-only round and entered the next round's gate().
    let r0_rounds = fe0.receive_count.load(Ordering::SeqCst);
    fe1.receive_hook.arm();
    h0.execute.release();
    fe1.receive_hook.wait_entered();
    fe0.wait_receive_gt(r0_rounds);
    thread::sleep(Duration::from_millis(15));
    send_lookup(201, &a_prompts[4]);
    assert!(
        rx.recv_timeout(Duration::from_millis(80)).is_err(),
        "gate() must keep scheduler locked while copies_confirmed_at is pending"
    );

    // 6. Trigger an idle_forward on rank 0 while copies_confirmed_at is still
    //    set (dp1 runs a local request). plan_and_dispatch must keep
    //    self.scheduler locked during idle_forward, and idle_forward must not
    //    clear copies_confirmed_at.
    let idle_before = h0.idle_execute_calls.load(Ordering::SeqCst);
    h0.execute.arm();
    fe1.push(add_req(
        "r1-while-copies",
        (600..600 + 2 * PAGE).collect(),
        1,
    ));
    fe1.receive_hook.release();
    h0.execute.wait_entered();
    assert!(
        h0.idle_execute_calls.load(Ordering::SeqCst) > idle_before,
        "rank 0 should be executing an idle_forward"
    );
    assert!(
        rx.recv_timeout(Duration::from_millis(80)).is_err(),
        "idle_forward must keep scheduler locked while copies_confirmed_at is pending"
    );
    let r0_rounds = fe0.receive_count.load(Ordering::SeqCst);
    fe1.receive_hook.arm();
    h0.execute.release();
    assert_eq!(recv_answer().0, 8);
    fe1.receive_hook.wait_entered();
    fe0.wait_receive_gt(r0_rounds);
    thread::sleep(Duration::from_millis(15));
    send_lookup(202, &a_prompts[4]);
    assert!(
        rx.recv_timeout(Duration::from_millis(80)).is_err(),
        "idle_forward must not clear copies_confirmed_at before a real forward completes"
    );
    fe1.receive_hook.release();
    assert_eq!(recv_answer().0, 8);
    fe1.wait_finished("r1-while-copies");

    // 7. Release a2..a6 so R dispatches and completes chunk 2's forward,
    //    clearing copies_confirmed_at. Afterwards gate() unlocks again.
    for lease in leases[2..=6].iter_mut() {
        release_lease(lease.take().unwrap());
    }
    fe0.wait_finished("R");

    h1.execute.arm();
    fe1.push(add_req(
        "r1-after-copies",
        (700..700 + 2 * PAGE).collect(),
        2,
    ));
    h1.execute.wait_entered();
    let r0_rounds = fe0.receive_count.load(Ordering::SeqCst);
    fe1.receive_hook.arm();
    h1.execute.release();
    fe1.receive_hook.wait_entered();
    fe0.wait_receive_gt(r0_rounds);
    thread::sleep(Duration::from_millis(15));
    let post_lease = pin_prompt(204, &r_page0_prompt);
    release_lease(post_lease);
    fe1.receive_hook.release();
    fe1.wait_finished("r1-after-copies");

    stop.store(true, Ordering::Relaxed);
    t0.join().unwrap();
    t1.join().unwrap();
}

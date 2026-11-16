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

//! Real EventLoop completion bursts, with a CPU result producer and the real
//! KV publisher. Request admission, scheduler feedback, queues and codecs run.
use crate::*;
use rustix::time::{clock_gettime, ClockId};
use serde_json::{json, Value};
use std::collections::VecDeque;
use ts_control::kv_events::{KvEventProgress, KvEventPublisher, KvEventsConfig};

struct BurstFrontend {
    requests: VecDeque<Vec<EngineRequest>>,
    steps: usize,
    issued: usize,
    finished: usize,
    last_load: Load,
    encoder: mpsc::Sender<(Vec<StreamedOutput>, Load)>,
}
impl Frontend for BurstFrontend {
    fn receive(&mut self, _: Option<Duration>) -> Vec<EngineRequest> {
        if self.issued != self.finished {
            return Vec::new();
        }
        let requests = self.requests.pop_front().unwrap_or_default();
        self.issued += requests.len();
        requests
    }
    fn closed(&self) -> bool {
        false
    }
    fn send(&mut self, outputs: Vec<StreamedOutput>, load: Load) {
        for output in &outputs {
            if output.finish_reason.is_some() {
                assert_eq!(output.finish_reason, Some("length"));
                assert_eq!(output.completion_tokens, self.steps as i64);
            }
            assert_eq!(output.output_ids, [7]);
        }
        self.finished += outputs
            .iter()
            .filter(|output| output.finish_reason.is_some())
            .count();
        self.last_load = load;
        if !outputs.is_empty() {
            self.encoder.send((outputs, load)).unwrap();
        }
    }
    fn reply(&mut self, _: ControlOutput) {
        unreachable!("no control requests")
    }
}
fn cpu_ms(clock: ClockId) -> f64 {
    let t = clock_gettime(clock);
    t.tv_sec as f64 * 1000.0 + t.tv_nsec as f64 / 1e6
}
fn progress_json(progress: Option<&KvEventProgress>) -> Value {
    progress.map_or(Value::Null, |p| {
        let s = p.snapshot();
        json!({"pending_batches":s.pending_batches,"pending_bytes":s.pending_bytes,
            "peak_pending_batches":s.peak_pending_batches,"peak_pending_bytes":s.peak_pending_bytes,
            "completed_batches":s.completed_batches,"enqueue_ms":s.enqueue_ns as f64/1e6})
    })
}
fn requests(
    batch: usize,
    prompt: usize,
    steps: usize,
    waves: usize,
) -> VecDeque<Vec<EngineRequest>> {
    (0..waves)
        .map(|wave| {
            (0..batch)
                .map(|row| {
                    let index = wave * batch + row;
                    let mut req = TokenizedGenerateReqInput::new(
                        format!("{index:08x}-9f09-89be-5c9d-5d65f086d356"),
                        SamplingParams {
                            max_new_tokens: Some(steps as i64),
                            ..SamplingParams::default()
                        },
                    );
                    let mut ids: Vec<i32> =
                        (0..prompt).map(|i| (i % 100000 + 100) as i32).collect();
                    ids[0] = 110000 + index as i32;
                    req.input_ids = Some(ids);
                    req.stream = true;
                    EngineRequest::Add(Box::new(req))
                })
                .collect()
        })
        .collect()
}
fn resident_kib(field: &str) -> usize {
    std::fs::read_to_string("/proc/self/status")
        .unwrap()
        .lines()
        .find_map(|line| line.strip_prefix(field))
        .unwrap()
        .split_whitespace()
        .next()
        .unwrap()
        .parse()
        .unwrap()
}

struct ControlAffinity {
    tid: String,
    previous: String,
}
impl ControlAffinity {
    fn pin(cpu: usize) -> Self {
        let thread = std::fs::read_link("/proc/thread-self").unwrap();
        let tid = thread.file_name().unwrap().to_str().unwrap().to_owned();
        let status = std::fs::read_to_string("/proc/thread-self/status").unwrap();
        let previous = status
            .lines()
            .find_map(|line| line.strip_prefix("Cpus_allowed_list:"))
            .unwrap()
            .trim()
            .to_owned();
        assert!(std::process::Command::new("taskset")
            .args(["-pc", &cpu.to_string(), &tid])
            .stdout(std::process::Stdio::null())
            .status()
            .unwrap()
            .success());
        Self { tid, previous }
    }
}
impl Drop for ControlAffinity {
    fn drop(&mut self) {
        // Restoration is outside timing, including unwinding after a failed check.
        let _ = std::process::Command::new("taskset")
            .args(["-pc", &self.previous, &self.tid])
            .stdout(std::process::Stdio::null())
            .status();
    }
}

fn trial(c: &Value) -> Value {
    let n = |key: &str| c[key].as_u64().unwrap_or_else(|| panic!("missing {key}")) as usize;
    let enabled = c["events"].as_bool().expect("events required");
    let (rows, prompt, steps, waves, page, depth) = (
        n("batch"),
        n("prompt"),
        n("steps"),
        n("waves"),
        n("page"),
        n("overlap"),
    );
    assert!(rows > 0 && prompt >= 2 && steps > 0 && waves > 0 && page > 0 && depth <= 1);
    assert!(rows.checked_mul(waves).unwrap() < 21000);
    let extent = prompt as u128 + steps as u128 + 2 * page as u128;
    assert!(
        rows as u128 * extent * 2 + 64 < i32::MAX as u128,
        "synthetic token and page counts must fit the scheduler bounds"
    );
    let shape = RoundShape {
        rows,
        prompt,
        page,
        generate: steps,
        verify_width: 1,
        accepted: 1,
        warmup_rounds: 0,
        rounds: 0,
    };
    let pages = rows * ((prompt + steps + 2 * page).div_ceil(page)) * 2 + 64;
    let submitted = requests(rows, prompt, steps, waves);
    let listener = Listener::bind("127.0.0.1").unwrap();
    let address = format!("127.0.0.1:{}", listener.port().unwrap());
    let ranks = Ranks {
        mesh: Mesh::connect(0, listener, &[address], Duration::from_secs(5)).unwrap(),
        replica: Group::new(1, vec![0], 0).unwrap(),
        pp: None,
        dp: None,
        ep: None,
    };
    // Configure the same native scheduler through its embedded module, including
    // the serving overlap depth and event generation used by this trial.
    interpreter();
    let scheduler = Python::attach(|py| {
        let g = PyDict::new(py);
        let code = format!(
            r#"
import tokenspeed_scheduler as ts
cfg=ts.SchedulerConfig()
cfg.role=ts.SchedulerConfig.Role.Fused
cfg.prefix_granularity={page}
cfg.max_scheduled_tokens={tokens}
cfg.max_batch_size={rows}
cfg.max_resident_requests={rows}
cfg.num_device_pages={pages}+1
cfg.num_snapshot_pages=1
cfg.disable_l2_cache=True
cfg.enable_l3_storage=False
cfg.disable_prefix_cache=False
cfg.prefix_replay_tokens=0
cfg.overlap_schedule_depth={depth}
cfg.decode_input_tokens=1
cfg.enable_kv_cache_events={events}
cfg.cache_groups=[ts.CacheGroupConfig(group_id="full_attention",block_granularity={page},total_pages={pages}+1,attn_kind=ts.AttnKind.Full)]
result=ts.Scheduler(cfg)
"#,
            tokens = rows * prompt,
            events = if enabled { "True" } else { "False" }
        );
        py.run(&CString::new(code).unwrap(), Some(&g), None)
            .unwrap();
        Scheduler::from_python(&g.get_item("result").unwrap().unwrap()).unwrap()
    });
    let path = std::env::temp_dir().join(format!(
        "ts-finish-{}-{}.ipc",
        std::process::id(),
        profile_now()
    ));
    let publisher = enabled.then(|| {
        KvEventPublisher::start(
            &KvEventsConfig {
                endpoint: format!("ipc://{}", path.display()),
                replay_endpoint: None,
                buffer_steps: 1,
                max_queue_size: n("queue_capacity"),
                topic: "kv".to_owned(),
            },
            0,
        )
        .unwrap()
    });
    let progress = publisher.as_ref().map(KvEventPublisher::progress);
    let (plans, receiver) = mpsc::channel();
    let freeing = std::thread::spawn(move || free_plans(receiver));
    let (encoder, batches) = mpsc::channel::<(Vec<StreamedOutput>, Load)>();
    let (encoded_tx, encoded_rx) = mpsc::channel();
    let encoding = std::thread::spawn(move || {
        for (outputs, load) in batches {
            if outputs.is_empty() {
                encoded_tx.send(()).unwrap();
                continue;
            }
            let batch = ts_control::transport::output_batch(outputs, 0, load);
            std::hint::black_box(EngineOutput::Batch(Box::new(batch)).to_frames());
        }
    });
    let mut config = round_loop_config(&shape, pages);
    config.in_flight_depth = depth;
    config.kv_events = enabled;
    let mut event_loop = EventLoop::new(
        scheduler,
        InstantDevice {
            plans: plans.clone(),
            verify_width: 1,
            accepted: 1,
        },
        BurstFrontend {
            requests: submitted,
            steps,
            issued: 0,
            finished: 0,
            last_load: Load::default(),
            encoder: encoder.clone(),
        },
        OutputProcessor::new(
            None,
            1,
            Some(1 << 24),
            Role::Plain,
            NO_FINISH_LOG,
            false,
            None,
        ),
        config,
        ranks,
        Subsystems {
            kv_events: publisher,
            ..Subsystems::default()
        },
    );
    // Workers inherit the process CPU set; pin only the control thread after
    // they exist, so moving encode work does not serialize it on that CPU.
    assert!(
        c.get("control_cpu").is_some(),
        "control_cpu is required (null leaves affinity unchanged)"
    );
    let _affinity = if c["control_cpu"].is_null() {
        None
    } else {
        Some(ControlAffinity::pin(
            c["control_cpu"]
                .as_u64()
                .expect("control_cpu must be an integer or null") as usize,
        ))
    };
    let initial_rss = resident_kib("VmRSS:");
    let start = Instant::now();
    let cpu = cpu_ms(ClockId::ProcessCPUTime);
    let mut records = Vec::new();
    let mut drained = 0;
    for i in 0..waves * (steps + 6) + 10 {
        let before = event_loop.frontend_mut().finished;
        let thread_cpu = cpu_ms(ClockId::ThreadCPUTime);
        let wall = Instant::now();
        assert!(!event_loop.round(false).unwrap());
        let wall_ms = wall.elapsed().as_secs_f64() * 1000.0;
        let thread_cpu_ms = cpu_ms(ClockId::ThreadCPUTime) - thread_cpu;
        let after = event_loop.frontend_mut().finished;
        records.push(json!({"round":i,"wall_ms":wall_ms,"thread_cpu_ms":thread_cpu_ms,
            "new_finishes":after-before,"finished":after,"publisher":progress_json(progress.as_ref())}));
        if before == rows * waves {
            drained += 1;
        }
        if drained == 3 {
            break;
        }
    }
    assert_eq!(event_loop.frontend_mut().finished, rows * waves);
    assert_eq!(event_loop.frontend_mut().last_load.num_running, 0);
    assert_eq!(event_loop.frontend_mut().last_load.num_waiting, 0);
    let control_done_ms = start.elapsed().as_secs_f64() * 1000.0;
    let (tx, rx) = mpsc::channel();
    plans.send(PlanJob::Fence(tx)).unwrap();
    rx.recv_timeout(Duration::from_secs(30)).unwrap();
    encoder.send((Vec::new(), Load::default())).unwrap();
    encoded_rx.recv_timeout(Duration::from_secs(30)).unwrap();
    let wait = Instant::now();
    if let Some(p) = &progress {
        while p.snapshot().pending_batches != 0 {
            assert!(
                wait.elapsed() < Duration::from_secs(120),
                "publisher did not drain"
            );
            std::thread::sleep(Duration::from_millis(1));
        }
        assert!(p.snapshot().completed_batches > 0);
    }
    let total_wall_ms = start.elapsed().as_secs_f64() * 1000.0;
    let process_cpu_ms = cpu_ms(ClockId::ProcessCPUTime) - cpu;
    let report = json!({"rounds":records,"control_done_ms":control_done_ms,
        "total_wall_ms":total_wall_ms,"process_cpu_ms":process_cpu_ms,
        "publisher_wait_ms":wait.elapsed().as_secs_f64()*1000.0,
        "publisher":progress_json(progress.as_ref()),"initial_rss_kib":initial_rss,
        "final_rss_kib":resident_kib("VmRSS:"),"peak_rss_kib":resident_kib("VmHWM:")});
    // Teardown of the persistent cache is not a request's completion cost.
    drop(event_loop);
    drop(plans);
    drop(encoder);
    freeing.join().unwrap();
    encoding.join().unwrap();
    if enabled {
        let _ = std::fs::remove_file(path);
    }
    report
}
fn profile_now() -> i64 {
    ts_scheduler::profile::now_ns()
}

#[test]
fn bursts_finish_through_the_real_loop_and_drain_the_publisher() {
    for events in [false, true] {
        for steps in [1, 2] {
            for overlap in [0, 1] {
                trial(
                    &json!({"batch":3,"prompt":32,"steps":steps,"waves":2,"page":8,
                    "overlap":overlap,"events":events,"queue_capacity":1,"control_cpu":null}),
                );
            }
        }
    }
}
#[test]
#[ignore = "explicit CPU EventLoop/publisher workload, no model execution"]
fn benchmark_finish_event_loop() {
    let c: Value =
        serde_json::from_str(&std::env::var("TS_LOOP_CONFIG").expect("TS_LOOP_CONFIG required"))
            .unwrap();
    let warmups = c["warmups"].as_u64().expect("warmups required");
    let repeats = c["repeats"].as_u64().expect("repeats required");
    assert!(repeats > 0);
    let mut samples = Vec::new();
    for i in 0..warmups + repeats {
        let sample = trial(&c);
        if i >= warmups {
            samples.push(sample);
        }
    }
    std::fs::write(
        std::env::var("TS_LOOP_OUTPUT").expect("TS_LOOP_OUTPUT required"),
        json!({"benchmark":"finish_event_loop","config":c,"samples":samples}).to_string(),
    )
    .unwrap();
}

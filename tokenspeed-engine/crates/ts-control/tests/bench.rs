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

//! Sustained microbenchmarks of the control plane, opt-in with
//! `cargo test --release -p tokenspeed-control --test bench -- --ignored
//! --nocapture`. Numbers are wall-clock on one machine: A/B two checkouts,
//! several runs each, and compare within a machine.

use std::collections::BTreeSet;
use std::ffi::CString;
use std::sync::{mpsc, Once};
use std::time::{Duration, Instant};

use pyo3::{prelude::*, types::PyDict};
use ts_collective::{Group, Listener, Mesh};
use ts_control::batch_log::BatchLogConfig;
use ts_control::control::ControlConfig;
use ts_control::event_loop::{
    DataPlane, DataPlaneError, Dispatch, DpRound, EventLoop, Frontend, Load, LoopConfig, Ranks,
    Subsystems,
};
use ts_control::ingress::{Admission, IngressLimits, LogprobServability};
use ts_control::output::{FinishLog, ForwardResult, OutputProcessor, Role, StreamedOutput};
use ts_control::request::RequestState;
use ts_scheduler::{CacheEvent, ExecutionPlan, ForwardBatchView, Scheduler};
use ts_wire::{
    ControlOutput, ControlRequest, EngineOutput, EngineRequest, SamplingParams,
    TokenizedGenerateReqInput,
};

/// The benchmarks allocate as `ts-host` does.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

const NO_FINISH_LOG: FinishLog = FinishLog {
    enabled: false,
    request_stats: false,
};

/// (rid, input_length, extend_prefix_len, prefill_length)
fn batch(rows: &[(&str, i32, i32, i32)], num_extends: usize) -> ForwardBatchView {
    ForwardBatchView {
        request_ids: rows.iter().map(|r| r.0.to_owned()).collect(),
        request_pool_indices: (1..=rows.len() as i32).collect(),
        input_lengths: rows.iter().map(|r| r.1).collect(),
        prefill_lengths: rows.iter().map(|r| r.3).collect(),
        extend_prefix_lens: rows[..num_extends].iter().map(|r| r.2).collect(),
        extend_replay_lens: vec![0; num_extends],
        num_extends,
    }
}

fn result(tokens: &[i32], lengths: &[i32]) -> ForwardResult {
    ForwardResult {
        output_tokens: tokens.to_vec(),
        output_lengths: lengths.to_vec(),
        output_logprobs: None,
        output_token_cumulative_probs: None,
        nan_flags: None,
        next_input_ids: None,
        serving_compiles: None,
        grammar_terminated_at: None,
        input_token_logprobs: None,
        input_top_logprobs: None,
        output_top_logprobs: None,
    }
}

/// `OutputProcessor::commit`'s steady decode cost: one prebuilt batch of
/// never-finishing requests, every token streamed, so the wall time is
/// commit's own work. The batch and result are built once outside the
/// loop; it prints the microseconds per commit.
#[test]
#[ignore = "a sustained load that A/Bs the commit path"]
fn bench_commit_decode() {
    let bs = 256;
    let run = Duration::from_secs(2);
    let mut p = OutputProcessor::new(
        None,
        1,
        Some(1 << 24),
        Role::Plain,
        NO_FINISH_LOG,
        false,
        None,
    );
    let mut rids: Vec<String> = Vec::with_capacity(bs);
    for i in 0..bs {
        let rid = format!("r{i}");
        let state = RequestState::new(vec![0; 16], 1 << 30, BTreeSet::new(), true, None, false);
        p.register(rid.clone(), state);
        let _ = p
            .commit(
                &batch(&[(rid.as_str(), 16, 0, 16)], 1),
                None,
                &result(&[7], &[1]),
            )
            .unwrap();
        rids.push(rid);
    }
    let rows: Vec<(&str, i32, i32, i32)> =
        rids.iter().map(|rid| (rid.as_str(), 1, 0, 16)).collect();
    let one = batch(&rows, 0);
    let res = result(&vec![9; bs], &vec![1; bs]);
    let start = Instant::now();
    let mut commits = 0usize;
    while start.elapsed() < run {
        let _ = p.commit(&one, None, &res).unwrap();
        commits += 1;
    }
    println!(
        "commit: {:.0} us/commit over {commits} commits",
        start.elapsed().as_secs_f64() * 1e6 / commits as f64
    );
}

/// A whole scheduling round's shape: `rows` requests of `prompt` tokens on
/// `page`-token KV pages, each generating `generate` tokens and replaced by
/// a new request once it finishes (0: it decodes for the whole run). With
/// speculative decoding, a decode row verifies `verify_width` tokens a round
/// and keeps `accepted` of them; without, both are 1.
struct RoundShape {
    rows: usize,
    prompt: usize,
    page: usize,
    generate: usize,
    verify_width: usize,
    accepted: usize,
    warmup_rounds: usize,
    rounds: usize,
}

fn interpreter() {
    static START: Once = Once::new();
    START.call_once(|| {
        std::env::set_var("PYTHONHOME", env!("TS_CONTROL_TEST_PYTHONHOME"));
        ts_scheduler::register_builtin_module();
        Python::initialize();
        Python::attach(|py| {
            ts_scheduler::alias_builtin_module(py).unwrap();
            let package = concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../../tokenspeed-scheduler/python"
            );
            py.import("sys")
                .unwrap()
                .getattr("path")
                .unwrap()
                .call_method1("insert", (0, package))
                .unwrap();
        });
    });
}

/// A scheduler with room for `pages` KV pages, admitting every row at once.
fn round_scheduler(shape: &RoundShape, pages: usize) -> Scheduler {
    interpreter();
    let code = format!(
        r#"
import tokenspeed_scheduler as ts
cfg = ts.SchedulerConfig()
cfg.prefix_granularity = {page}
cfg.max_scheduled_tokens = {tokens}
cfg.max_batch_size = {rows}
cfg.num_device_pages = {pages} + 1
cfg.num_snapshot_pages = 1
cfg.cache_groups = [ts.CacheGroupConfig(group_id="full_attention", block_granularity={page},
                                        total_pages={pages} + 1, attn_kind=ts.AttnKind.Full)]
cfg.decode_input_tokens = {verify_width}
cfg.overlap_schedule_depth = 1
result = ts.Scheduler(cfg)
"#,
        page = shape.page,
        rows = shape.rows,
        tokens = shape.rows * (shape.prompt + 16),
        verify_width = shape.verify_width,
    );
    Python::attach(|py| {
        let globals = PyDict::new(py);
        py.run(&CString::new(code).unwrap(), Some(&globals), None)
            .unwrap();
        Scheduler::from_python(&globals.get_item("result").unwrap().unwrap()).unwrap()
    })
}

/// A device whose forwards land at once: a prefill row keeps one token, a
/// decode row stores its verify window and keeps `accepted` tokens of it.
/// Plans go to another thread, which frees them, as the host's forward
/// thread does.
struct InstantDevice {
    plans: mpsc::Sender<ExecutionPlan>,
    verify_width: usize,
    accepted: i32,
}

impl DataPlane for InstantDevice {
    type Pending = ForwardResult;

    fn admit(&mut self, _: &[Admission]) -> Result<(), DataPlaneError> {
        Ok(())
    }

    fn execute(
        &mut self,
        plan: ExecutionPlan,
        dispatch: &Dispatch<'_>,
    ) -> Result<Option<ForwardResult>, DataPlaneError> {
        let _ = self.plans.send(plan);
        Ok(dispatch.forward.map(|batch| {
            let decodes = batch.request_ids.len() - batch.num_extends;
            let mut output_lengths = vec![1; batch.num_extends];
            output_lengths.extend(std::iter::repeat_n(self.accepted, decodes));
            ForwardResult {
                output_tokens: vec![7; batch.num_extends + decodes * self.verify_width],
                output_lengths,
                ..ForwardResult::default()
            }
        }))
    }

    fn complete(&mut self, pending: ForwardResult) -> Result<ForwardResult, DataPlaneError> {
        Ok(pending)
    }

    fn poll_cache_events(&mut self) -> Result<Vec<CacheEvent>, DataPlaneError> {
        Ok(Vec::new())
    }

    fn release(&mut self, _: &[String]) -> Result<(), DataPlaneError> {
        Ok(())
    }

    fn land(&mut self, _: Option<(i32, Vec<i32>)>, _: Option<i32>) -> Result<(), DataPlaneError> {
        unreachable!("no PD in the benchmark")
    }

    fn idle_forward(&mut self, _: &DpRound) -> Result<(), DataPlaneError> {
        unreachable!("no attention DP in the benchmark")
    }

    fn check_weight_update(
        &mut self,
        _: &ControlRequest,
    ) -> Result<(bool, String), DataPlaneError> {
        unreachable!("no control operation in the benchmark")
    }

    fn update_weights(&mut self, _: &ControlRequest) -> Result<(bool, String), DataPlaneError> {
        unreachable!("no control operation in the benchmark")
    }

    fn commit_weight_update(&mut self, _: bool) -> Result<(), DataPlaneError> {
        unreachable!("no control operation in the benchmark")
    }

    fn profile(
        &mut self,
        _: &ts_wire::control::ProfileReq,
    ) -> Result<(bool, String), DataPlaneError> {
        unreachable!("no control operation in the benchmark")
    }

    fn release_memory(&mut self, _: &[String]) -> Result<(), DataPlaneError> {
        unreachable!("no control operation in the benchmark")
    }

    fn reload_forward(&mut self, _: &ControlRequest) -> Result<(bool, String), DataPlaneError> {
        unreachable!("no control operation in the benchmark")
    }

    fn resume_memory(&mut self, _: &[String]) -> Result<(), DataPlaneError> {
        unreachable!("no control operation in the benchmark")
    }

    #[allow(clippy::type_complexity)]
    fn snapshot_expert_load(&mut self) -> Result<(Vec<Vec<i64>>, Vec<Vec<i32>>), DataPlaneError> {
        unreachable!("no expert rebalance in the benchmark")
    }

    fn attach_grammars(&mut self, _: &[(String, u64, Vec<i32>)]) -> Result<(), DataPlaneError> {
        unreachable!("no grammar in the benchmark")
    }

    fn epd_publish(&mut self, _: &[ts_pd::LandedRequest]) -> Result<(), DataPlaneError> {
        unreachable!("no EPD in the benchmark")
    }

    fn epd_released(&mut self) -> Result<Vec<usize>, DataPlaneError> {
        unreachable!("no EPD in the benchmark")
    }

    fn apply_expert_placement(
        &mut self,
        _: &[usize],
        _: &[Vec<i32>],
        _: &[ts_control::eplb::SlotMoves],
    ) -> Result<(), DataPlaneError> {
        unreachable!("no expert rebalance in the benchmark")
    }
}

/// A frontend that hands each round's outputs to another thread to encode,
/// as the transport does, and submits a new request for every one that
/// finished.
struct ChurningFrontend {
    prompt: usize,
    max_new_tokens: i64,
    /// Requests to submit next round.
    due: usize,
    submitted: usize,
    encoder: mpsc::Sender<(Vec<StreamedOutput>, Load)>,
}

impl Frontend for ChurningFrontend {
    fn receive(&mut self, _: Option<Duration>) -> Vec<EngineRequest> {
        let due = std::mem::take(&mut self.due);
        (0..due)
            .map(|_| {
                // As long as the UUIDs frontends name requests with.
                let rid = format!("{:08x}-9f09-89be-5c9d-5d65f086d356", self.submitted);
                let mut request = TokenizedGenerateReqInput::new(
                    rid,
                    SamplingParams {
                        max_new_tokens: Some(self.max_new_tokens),
                        ..SamplingParams::default()
                    },
                );
                // A distinct first token gives every request its own prefix
                // hash chain: no row shares a published block with another,
                // as with real traffic.
                let mut input_ids: Vec<i32> = (1..=self.prompt as i32).collect();
                input_ids[0] = 1_000_000 + self.submitted as i32;
                request.input_ids = Some(input_ids);
                request.stream = true;
                self.submitted += 1;
                EngineRequest::Add(Box::new(request))
            })
            .collect()
    }

    fn closed(&self) -> bool {
        false
    }

    fn send(&mut self, outputs: Vec<StreamedOutput>, load: Load) {
        self.due += outputs
            .iter()
            .filter(|output| output.finish_reason.is_some())
            .count();
        let _ = self.encoder.send((outputs, load));
    }

    fn reply(&mut self, _: ControlOutput) {}
}

/// Run `shape`'s rounds on one rank and print the microseconds per round.
fn bench_rounds(name: &str, shape: RoundShape) {
    let generated = if shape.generate > 0 {
        shape.generate
    } else {
        (shape.warmup_rounds + shape.rounds + 1) * shape.accepted
    };
    // Every row may hold its whole request, and a verify window past it.
    let pages = shape.rows * (shape.prompt + generated + shape.verify_width + 2 * shape.page)
        / shape.page
        + 64;
    let listener = Listener::bind("127.0.0.1").unwrap();
    let address = format!("127.0.0.1:{}", listener.port().unwrap());
    let ranks = Ranks {
        mesh: Mesh::connect(0, listener, &[address], Duration::from_secs(5)).unwrap(),
        replica: Group::new(1, vec![0], 0).unwrap(),
        pp: None,
        dp: None,
        ep: None,
    };
    let (plans, freed) = mpsc::channel::<ExecutionPlan>();
    let freeing = std::thread::spawn(move || freed.into_iter().for_each(drop));
    let (encoder, batches) = mpsc::channel::<(Vec<StreamedOutput>, Load)>();
    let encoding = std::thread::spawn(move || {
        for (outputs, load) in batches {
            let batch = ts_control::transport::output_batch(outputs, 0, load);
            std::hint::black_box(EngineOutput::Batch(Box::new(batch)).to_frames());
        }
    });
    let config = LoopConfig {
        in_flight_depth: 1,
        limits: IngressLimits {
            vocab_size: 1 << 17,
            eos_token_ids: vec![2],
            top_k_exclusive_max: Some(ts_control::sampling::TOP_K_FUSED_MAX),
            max_model_len: 1 << 24,
            max_req_input_len: 1 << 20,
            enable_output_logprobs: false,
            requires_bootstrap: false,
            multimodal: false,
            grammar: false,
            reasoning: None,
            reasoning_usage: None,
            logprobs: LogprobServability {
                supports_prompt_logprobs: true,
                logprob_order: "torch".to_owned(),
                pipeline_parallel: false,
                disaggregated: false,
                lm_head_tp_under_dp: false,
                computes_prompt_logprobs: true,
            },
        },
        kv_total_pages: pages as i64,
        kv_token_capacity: (pages * shape.page) as i64,
        ngram_context_len: 0,
        request_token_history: false,
        request_token_history_lookback: None,
        controls: ControlConfig {
            memory_saver: false,
            memory_regions: Vec::new(),
            l3_weight_version: None,
            allow_reload_forward: false,
        },
        batch_log: BatchLogConfig {
            enabled: false,
            decode_log_interval: 40,
            num_total_pages: pages as i64,
            spec_num_steps: 0,
            dp_rank: 0,
            disaggregated: false,
            cache_state_group_ids: Vec::new(),
            log_spec_accept_lengths: false,
        },
        kv_events: false,
        epd_receive_enabled: false,
        eager_grammar: false,
    };
    let mut event_loop = EventLoop::new(
        round_scheduler(&shape, pages),
        InstantDevice {
            plans,
            verify_width: shape.verify_width,
            accepted: shape.accepted as i32,
        },
        ChurningFrontend {
            prompt: shape.prompt,
            max_new_tokens: generated as i64,
            due: shape.rows,
            submitted: 0,
            encoder,
        },
        OutputProcessor::new(
            (shape.verify_width > 1).then_some(shape.verify_width),
            1,
            Some(1 << 24),
            Role::Plain,
            NO_FINISH_LOG,
            false,
            None,
        ),
        config,
        ranks,
        Subsystems::default(),
    );
    for _ in 0..shape.warmup_rounds {
        event_loop.round(false).unwrap();
    }
    let start = Instant::now();
    for _ in 0..shape.rounds {
        event_loop.round(false).unwrap();
    }
    println!(
        "{name}: {:.0} us/round over {} rounds",
        start.elapsed().as_secs_f64() * 1e6 / shape.rounds as f64,
        shape.rounds
    );
    drop(event_loop);
    freeing.join().unwrap();
    encoding.join().unwrap();
}

/// A whole `EventLoop::round` in steady decode: 256 rows of 2K-token
/// prompts on 64-token pages, every token streamed, over the real
/// scheduler, with a device and a frontend that cost nothing beyond the
/// output encoding, so the wall time is the control thread's own.
#[test]
#[ignore = "a sustained load that A/Bs a whole round"]
fn bench_round_decode() {
    bench_rounds(
        "round, decode",
        RoundShape {
            rows: 256,
            prompt: 2048,
            page: 64,
            generate: 0,
            verify_width: 1,
            accepted: 1,
            warmup_rounds: 200,
            rounds: 1000,
        },
    );
}

/// [`bench_round_decode`] under churn: each request generates 64 tokens and
/// a new one takes its row, so every round admits and finishes four
/// requests besides decoding the rest.
#[test]
#[ignore = "a sustained load that A/Bs a whole round"]
fn bench_round_churn() {
    bench_rounds(
        "round, churn",
        RoundShape {
            rows: 256,
            prompt: 512,
            page: 64,
            generate: 64,
            verify_width: 1,
            accepted: 1,
            warmup_rounds: 200,
            rounds: 2000,
        },
    );
}

/// [`bench_round_churn`]'s steady twin: the same 256 rows of 512-token
/// prompts, no request finishing, so `churn` minus this is the cost of the
/// four admissions and finishes a round.
#[test]
#[ignore = "a sustained load that A/Bs a whole round"]
fn bench_round_decode_short() {
    bench_rounds(
        "round, decode, 512",
        RoundShape {
            rows: 256,
            prompt: 512,
            page: 64,
            generate: 0,
            verify_width: 1,
            accepted: 1,
            warmup_rounds: 200,
            rounds: 1000,
        },
    );
}

/// [`bench_round_churn`] under a burst of short requests: each request
/// generates four tokens, so a quarter of the 256 rows finishes every round
/// and 64 new requests are admitted beside decoding the rest. The admission
/// rate, not the context, is what the round's cost tracks here.
#[test]
#[ignore = "a sustained load that A/Bs a whole round"]
fn bench_round_admit_burst() {
    bench_rounds(
        "round, admit burst",
        RoundShape {
            rows: 256,
            prompt: 512,
            page: 64,
            generate: 4,
            verify_width: 1,
            accepted: 1,
            warmup_rounds: 200,
            rounds: 2000,
        },
    );
}

/// [`bench_round_admit_burst`] with long prompts: 8K tokens per request and
/// eight admissions a round, so the part of admission that walks the prompt
/// dominates and cannot hide behind the per-request fixed cost.
#[test]
#[ignore = "a sustained load that A/Bs a whole round"]
fn bench_round_admit_long() {
    bench_rounds(
        "round, admit long",
        RoundShape {
            rows: 64,
            prompt: 8192,
            page: 64,
            generate: 8,
            verify_width: 1,
            accepted: 1,
            warmup_rounds: 100,
            rounds: 1000,
        },
    );
}

/// [`bench_round_admit_long`]'s steady twin: the same 64 rows of 8K-token
/// prompts, no request finishing, so `admit_long` minus this is the cost of
/// the eight admissions a round.
#[test]
#[ignore = "a sustained load that A/Bs a whole round"]
fn bench_round_decode_mid() {
    bench_rounds(
        "round, decode, 8K",
        RoundShape {
            rows: 64,
            prompt: 8192,
            page: 64,
            generate: 0,
            verify_width: 1,
            accepted: 1,
            warmup_rounds: 100,
            rounds: 1000,
        },
    );
}

/// [`bench_round_admit_long`] with one 8K admission a round instead of eight:
/// the same 8K prefill tokens a round, spread over one request, so the pair
/// separates a per-request admission cost from a per-token one.
#[test]
#[ignore = "a sustained load that A/Bs a whole round"]
fn bench_round_prefill_long() {
    bench_rounds(
        "round, prefill 8K x1",
        RoundShape {
            rows: 64,
            prompt: 8192,
            page: 64,
            generate: 64,
            verify_width: 1,
            accepted: 1,
            warmup_rounds: 100,
            rounds: 1000,
        },
    );
}

/// [`bench_round_prefill_long`]'s twin with 1K prompts: eight admissions a
/// round, the same 8K prefill tokens a round, eight times the requests.
#[test]
#[ignore = "a sustained load that A/Bs a whole round"]
fn bench_round_prefill_short() {
    bench_rounds(
        "round, prefill 1K x8",
        RoundShape {
            rows: 64,
            prompt: 1024,
            page: 64,
            generate: 8,
            verify_width: 1,
            accepted: 1,
            warmup_rounds: 100,
            rounds: 1000,
        },
    );
}

/// [`bench_round_prefill_short`]'s steady twin: 1K-token prompts, nothing
/// finishing.
#[test]
#[ignore = "a sustained load that A/Bs a whole round"]
fn bench_round_decode_1k() {
    bench_rounds(
        "round, decode, 1K",
        RoundShape {
            rows: 64,
            prompt: 1024,
            page: 64,
            generate: 0,
            verify_width: 1,
            accepted: 1,
            warmup_rounds: 100,
            rounds: 1000,
        },
    );
}

/// [`bench_round_decode`] at a long context: 64 rows of 128K-token prompts,
/// where the round's work that grows with the context dominates.
#[test]
#[ignore = "a sustained load that A/Bs a whole round"]
fn bench_round_decode_long() {
    bench_rounds(
        "round, decode, 128K",
        RoundShape {
            rows: 64,
            prompt: 128 * 1024,
            page: 64,
            generate: 0,
            verify_width: 1,
            accepted: 1,
            warmup_rounds: 20,
            rounds: 300,
        },
    );
}

/// [`bench_round_decode_long`] with speculative decoding: each row verifies
/// six tokens a round and keeps five, so it crosses a page boundary five
/// times as often.
#[test]
#[ignore = "a sustained load that A/Bs a whole round"]
fn bench_round_decode_long_spec() {
    bench_rounds(
        "round, decode, 128K, speculative",
        RoundShape {
            rows: 64,
            prompt: 128 * 1024,
            page: 64,
            generate: 0,
            verify_width: 6,
            accepted: 5,
            warmup_rounds: 20,
            rounds: 300,
        },
    );
}

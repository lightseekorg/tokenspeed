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

use std::cell::Cell;
use std::collections::BTreeMap;
use std::time::Instant;

use rustix::time::{clock_gettime, ClockId};
use serde_json::{json, Value};
use ts_client::OutputChunk;
use ts_control::{
    event_loop::Load,
    ingress::{self, Admitted, IngressLimits, LogprobServability},
    output::{FinishLog, ForwardResult, OutputProcessor, Role},
    transport::output_batch,
};
use ts_scheduler::{ForwardBatchView, RequestSpec};
use ts_wire::{EngineOutput, EngineRequest, SamplingParams, TokenizedGenerateReqInput};

pub mod pipeline;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Logprobs {
    None,
    Output,
    Prompt,
}

#[derive(Clone, Debug)]
pub struct Config {
    pub suite: String,
    pub transport: String,
    pub batch: usize,
    pub prompt: usize,
    pub steps: usize,
    pub logprobs: Logprobs,
    pub top_k: usize,
    pub period_us: u64,
    pub consumer_delay_us: u64,
    pub client_threads: usize,
    pub warmups: usize,
    pub repeats: usize,
    pub phase_timers: bool,
}

impl Config {
    pub fn validate(&self) -> Result<(), String> {
        if self.batch == 0
            || self.batch > i32::MAX as usize
            || self.prompt < 2
            || self.steps == 0
            || self.repeats == 0
        {
            return Err("batch, steps, repeats must be positive; prompt must be >= 2".into());
        }
        if self
            .prompt
            .checked_add(self.steps)
            .is_none_or(|n| n > i32::MAX as usize)
        {
            return Err("prompt plus output must fit the real scheduler's i32 token counts".into());
        }
        if self.top_k > 20
            || (self.top_k > 0 && (self.steps != 1 || self.logprobs == Logprobs::None))
        {
            return Err(
                "the real ingress permits top-k only with logprobs and max_new_tokens=1".into(),
            );
        }
        match self.suite.as_str() {
            "stages"
                if self.transport == "none"
                    && self.period_us == 0
                    && self.consumer_delay_us == 0
                    && self.client_threads == 0 =>
            {
                Ok(())
            }
            "pipeline"
                if matches!(self.transport.as_str(), "ipc" | "tcp") && self.client_threads > 0 =>
            {
                Ok(())
            }
            _ => Err("stages requires transport=none and timing/thread knobs=0; pipeline requires ipc/tcp and client_threads>0".into()),
        }
    }
    pub fn json(&self) -> Value {
        json!({"suite":self.suite,"transport":self.transport,"batch":self.batch,"prompt_tokens":self.prompt,
            "steps":self.steps,"logprobs":format!("{:?}",self.logprobs).to_lowercase(),"top_k":self.top_k,
            "period_us":self.period_us,"consumer_delay_us":self.consumer_delay_us,"client_threads":self.client_threads,
            "warmups":self.warmups,"repeats":self.repeats,"phase_timers":self.phase_timers})
    }
}

pub fn cpu_ms(clock: ClockId) -> f64 {
    let t = clock_gettime(clock);
    t.tv_sec as f64 * 1000.0 + t.tv_nsec as f64 / 1e6
}

#[derive(Debug, Clone, Copy)]
pub struct Timing {
    pub wall_ms: f64,
    pub thread_cpu_ms: f64,
}
impl Timing {
    pub fn json(self) -> Value {
        json!({"wall_ms":self.wall_ms,"thread_cpu_ms":self.thread_cpu_ms})
    }
}

pub type Phases = BTreeMap<String, Vec<Timing>>;

thread_local! { static PHASE_TIMERS: Cell<bool> = const { Cell::new(false) }; }

pub fn set_phase_timers(enabled: bool) {
    PHASE_TIMERS.with(|flag| flag.set(enabled));
}

pub fn timed<T>(phases: &mut Phases, name: &str, f: impl FnOnce() -> T) -> T {
    if !PHASE_TIMERS.with(Cell::get) {
        return f();
    }
    let cpu = cpu_ms(ClockId::ThreadCPUTime);
    let wall = Instant::now();
    let result = f();
    let timing = Timing {
        wall_ms: wall.elapsed().as_secs_f64() * 1000.0,
        thread_cpu_ms: cpu_ms(ClockId::ThreadCPUTime) - cpu,
    };
    phases.entry(name.to_owned()).or_default().push(timing);
    result
}

pub fn phases_json(phases: &Phases) -> Value {
    Value::Object(
        phases
            .iter()
            .map(|(key, values)| {
                (
                    key.clone(),
                    json!(values.iter().map(|v| v.json()).collect::<Vec<_>>()),
                )
            })
            .collect(),
    )
}

pub fn prompt_token(row: usize, position: usize) -> i32 {
    ((row * 31 + position * 17) % 100_000 + 100) as i32
}
pub fn output_token(step: usize) -> i32 {
    120_000 + (step % 10_000) as i32
}

pub fn request(config: &Config, row: usize) -> TokenizedGenerateReqInput {
    TokenizedGenerateReqInput {
        input_ids: Some((0..config.prompt).map(|i| prompt_token(row, i)).collect()),
        stream: true,
        return_logprob: config.logprobs != Logprobs::None,
        logprob_start_len: if config.logprobs == Logprobs::Prompt {
            0
        } else {
            -1
        },
        top_logprobs_num: config.top_k as i64,
        ..TokenizedGenerateReqInput::new(
            format!("bench-{row}"),
            SamplingParams {
                max_new_tokens: Some(config.steps as i64),
                ignore_eos: true,
                stream_interval: Some(1),
                ..SamplingParams::default()
            },
        )
    }
}

pub fn limits(config: &Config) -> IngressLimits {
    IngressLimits {
        vocab_size: 131_072,
        eos_token_ids: vec![],
        top_k_exclusive_max: None,
        max_model_len: (config.prompt + config.steps + 1) as i64,
        max_req_input_len: config.prompt as i64,
        enable_output_logprobs: true,
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
    }
}

pub fn processor(config: &Config) -> OutputProcessor {
    OutputProcessor::new(
        None,
        1,
        Some((config.prompt + config.steps + 1) as i64),
        Role::Plain,
        FinishLog {
            enabled: false,
            request_stats: false,
            ttft_percentage: 0,
        },
        false,
        None,
    )
}

pub fn admit_one(
    request: TokenizedGenerateReqInput,
    config: &Config,
    processor: &mut OutputProcessor,
) -> RequestSpec {
    match ingress::admit(request, &limits(config)).expect("request has a rid") {
        Admitted::Run(mut admission) => {
            processor.assign_admission(&mut admission.state);
            processor.register(admission.rid, admission.state);
            admission.spec
        }
        Admitted::Refused { state, .. } => panic!(
            "benchmark request must be served: {:?}",
            state.finished_reason
        ),
    }
}

pub fn batch(config: &Config, rids: &[String], prefill: bool) -> ForwardBatchView {
    ForwardBatchView {
        request_ids: rids.to_vec(),
        request_pool_indices: (1..=config.batch as i32).collect(),
        input_lengths: vec![if prefill { config.prompt as i32 } else { 1 }; config.batch],
        prefill_lengths: vec![config.prompt as i32; config.batch],
        extend_prefix_lens: if prefill {
            vec![0; config.batch]
        } else {
            vec![]
        },
        extend_replay_lens: if prefill {
            vec![0; config.batch]
        } else {
            vec![]
        },
        num_extends: if prefill { config.batch } else { 0 },
    }
}

/// Model work is deliberately outside the timed boundary. These arrays have
/// the exact production layout that OutputProcessor::logprob_plan requests.
pub fn result(config: &Config, step: usize) -> ForwardResult {
    let prompt_rows = if step == 0 && config.logprobs == Logprobs::Prompt {
        config.batch * (config.prompt - 1)
    } else {
        0
    };
    let alternatives = |n| {
        (
            (0..n)
                .map(|_| (0..config.top_k).map(|k| -0.5 - k as f64).collect())
                .collect(),
            (0..n)
                .map(|_| (0..config.top_k).map(|k| k as i32 + 10).collect())
                .collect(),
        )
    };
    ForwardResult {
        output_tokens: vec![output_token(step); config.batch],
        output_lengths: vec![1; config.batch],
        output_logprobs: (config.logprobs != Logprobs::None).then(|| vec![-0.5; config.batch]),
        input_token_logprobs: (prompt_rows > 0).then(|| vec![-1.25; prompt_rows]),
        input_top_logprobs: (prompt_rows > 0 && config.top_k > 0)
            .then(|| alternatives(prompt_rows)),
        output_top_logprobs: (step == 0 && config.top_k > 0).then(|| alternatives(config.batch)),
        ..ForwardResult::default()
    }
}

pub fn check_chunk(chunk: &OutputChunk, config: &Config, row: usize, step: usize) {
    assert_eq!(chunk.output_ids, [output_token(step)]);
    assert_eq!(chunk.completion_tokens, step as i64 + 1);
    assert_eq!(chunk.prompt_tokens, config.prompt as i64);
    assert_eq!(chunk.finish_reason.is_some(), step + 1 == config.steps);
    if config.logprobs == Logprobs::None {
        assert!(chunk.output_token_logprobs.is_empty());
    } else {
        assert_eq!(chunk.output_token_logprobs, [-0.5]);
    }
    if step == 0 && config.logprobs != Logprobs::None {
        let (values, ids) = chunk
            .input_token_logprobs
            .as_ref()
            .expect("first chunk owns prompt logprobs");
        let start = if config.logprobs == Logprobs::Prompt {
            0
        } else {
            config.prompt - 1
        };
        assert_eq!(ids.len(), config.prompt - start);
        assert_eq!(values.len(), ids.len());
        assert_eq!(values[0], None);
        if values.len() > 1 {
            assert_eq!(values[1], Some(-1.25));
            assert_eq!(values.last(), Some(&Some(-1.25)));
        }
        assert_eq!(ids[0], prompt_token(row, start));
        assert_eq!(*ids.last().unwrap(), prompt_token(row, config.prompt - 1));
        if config.top_k > 0 {
            let (values, ids) = chunk
                .input_top_logprobs
                .as_ref()
                .expect("prompt alternatives");
            assert_eq!(values.len(), config.prompt - start);
            assert_eq!(ids.len(), values.len());
            assert!(values[0].is_none());
            if values.len() > 1 {
                assert_eq!(values[1].as_ref().unwrap().len(), config.top_k);
            }
        }
    } else {
        assert!(
            chunk.input_token_logprobs.is_none(),
            "prompt data must be sent once"
        );
    }
    if config.top_k > 0 {
        let (values, ids) = chunk
            .output_top_logprobs
            .as_ref()
            .expect("sampled alternatives");
        assert_eq!(values.len(), 1);
        assert_eq!(values[0].len(), config.top_k);
        assert_eq!(ids[0].len(), config.top_k);
    } else {
        assert!(chunk.output_top_logprobs.is_none());
    }
}

pub fn run_stages(config: &Config) -> Value {
    set_phase_timers(config.phase_timers);
    let requests: Vec<_> = (0..config.batch)
        .map(|row| EngineRequest::Add(Box::new(request(config, row))))
        .collect();
    let mut phases = Phases::new();
    let frames = timed(&mut phases, "request_encode", || {
        requests
            .into_iter()
            .map(|r| r.to_frames())
            .collect::<Vec<_>>()
    });
    let request_bytes: usize = frames.iter().flat_map(|f| f.iter()).map(|b| b.len()).sum();
    let decoded = timed(&mut phases, "request_decode", || {
        frames
            .iter()
            .map(|f| EngineRequest::from_frames(f).unwrap())
            .collect::<Vec<_>>()
    });
    drop(frames);
    let mut p = processor(config);
    let specs = timed(&mut phases, "ingress_and_register", || {
        decoded
            .into_iter()
            .map(|r| match r {
                EngineRequest::Add(request) => admit_one(*request, config, &mut p),
                _ => panic!("generate request"),
            })
            .collect::<Vec<_>>()
    });
    let rids: Vec<_> = specs.iter().map(|s| s.request_id.clone()).collect();
    drop(specs);
    let prefill = batch(config, &rids, true);
    let decode = batch(config, &rids, false);
    let mut output_bytes = Vec::new();
    for step in 0..config.steps {
        let batch = if step == 0 { &prefill } else { &decode };
        let model = result(config, step);
        let phase = if step == 0 { "prefill" } else { "decode" };
        let plan = timed(&mut phases, &format!("{phase}_logprob_plan"), || {
            p.logprob_plan(batch)
        });
        let committed = timed(&mut phases, &format!("{phase}_commit"), || {
            p.commit(batch, plan.input.as_ref(), &model).unwrap()
        });
        assert_eq!(committed.outputs.len(), config.batch);
        let wire = timed(&mut phases, &format!("{phase}_output_batch"), || {
            output_batch(committed.outputs, 0, Load::default())
        });
        let encoded = timed(&mut phases, &format!("{phase}_encode"), || {
            EngineOutput::Batch(Box::new(wire)).to_frames()
        });
        output_bytes.push(encoded.iter().map(|b| b.len()).sum::<usize>());
        let decoded = timed(&mut phases, &format!("{phase}_wire_decode"), || {
            EngineOutput::from_frames(&encoded).unwrap()
        });
        let EngineOutput::Batch(wire) = decoded else {
            panic!("output batch")
        };
        let chunks = timed(&mut phases, &format!("{phase}_client_materialize"), || {
            OutputChunk::from_batch(*wire)
                .unwrap()
                .map(|(_, chunk)| chunk)
                .collect::<Vec<_>>()
        });
        for (row, chunk) in chunks.iter().enumerate() {
            check_chunk(chunk, config, row, step);
        }
        timed(&mut phases, &format!("{phase}_client_drop"), || {
            drop(chunks)
        });
    }
    assert_eq!(p.num_running(), 0, "finished states release ownership");
    json!({"phases":phases_json(&phases),"request_bytes":request_bytes,"output_bytes":output_bytes,"checked_chunks":config.batch*config.steps})
}

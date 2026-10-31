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

//! `ts-host`: one TokenSpeed engine rank with its control plane in Rust.
//!
//! Every launch path starts one per rank (`engine/rust_host.py` is its Python
//! half; `TOKENSPEED_HOST_BINARY` overrides the installed binary):
//!
//! ```text
//! ts-host --python <the launcher's sys.executable> --control-fd <fd>
//! ```
//!
//! The rank's arguments arrive on `--control-fd`, and readiness goes back
//! on it. The embedded interpreter builds the data plane and the scheduler;
//! the ranks then connect their collective mesh, the replica's first member
//! connects to the frontend, and the main thread runs the scheduling loop
//! while the forward thread runs the data plane; the main thread takes the
//! GIL only for the data plane's completion queries (`data_plane`).

mod data_plane;
mod encode;
mod interpreter;
mod pd;

use std::{
    process::ExitCode,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread,
    time::Duration,
};

use pyo3::{prelude::*, types::PyBytes};
use tracing_subscriber::EnvFilter;
use ts_collective::{Group, Listener, Mesh};
use ts_control::{
    batch_log::BatchLogConfig,
    control::ControlConfig,
    eplb::{Algorithm, Controller, EplbConfig, EplbSpecs, COMMIT_DELAY_FORWARDS},
    event_loop::{EventLoop, LoopConfig, LoopError, Ranks, Subsystems},
    grammar::{GrammarConfig, Grammars, ReasoningGrammar},
    ingress::{IngressLimits, LogprobServability},
    kv_events::{KvEventPublisher, KvEventsConfig},
    output::{FinishLog, OutputProcessor, Role},
    transport::{EngineTransport, TransportError},
};
use ts_metrics::{
    EngineLabels, EngineMetrics, Exporter, KvTransferMetrics, L3Labels, L3Metrics, MetricsConfig,
};
use ts_scheduler::Scheduler;
use ts_wire::EngineCoreReadyResponse;

use crate::data_plane::PythonDataPlane;

/// How long the host waits for the frontend's INIT and for the other ranks
/// to connect.
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(600);

/// Collective group ids. Replicas are disjoint, and so are pipelines, so each
/// kind shares one id.
const DP_GROUP: u32 = 0;
const REPLICA_GROUP: u32 = 1;
const PP_GROUP: u32 = 2;
const EP_GROUP: u32 = 3;

#[derive(Debug, thiserror::Error)]
enum HostError {
    #[error("usage: ts-host --python <executable> --control-fd <fd>: {0}")]
    Usage(String),
    #[error("{0}")]
    Interpreter(String),
    #[error("python:\n{0}")]
    Python(String),
    #[error("data plane: {0}")]
    DataPlane(String),
    #[error("the ready response does not decode: {0}")]
    Ready(#[from] ts_wire::Error),
    #[error("frontend: {0}")]
    Transport(#[from] TransportError),
    #[error("rank mesh: {0}")]
    Collective(#[from] ts_collective::Error),
    #[error(transparent)]
    Metrics(#[from] ts_metrics::MetricsError),
    #[error(transparent)]
    KvEvents(#[from] ts_control::kv_events::KvEventsError),
    #[error("the L3 store: {0}")]
    Storage(#[from] ts_mooncake::Error),
    #[error(transparent)]
    Loop(#[from] LoopError),
    #[error("the EPD embedding sender: {0}")]
    Sender(#[from] ts_pd::StartError),
    #[error("the encode loop: {0}")]
    Encode(#[from] ts_control::encode::EncodeError),
    #[error("{0}")]
    Io(#[from] std::io::Error),
}

fn python_error(py: Python<'_>, error: PyErr) -> HostError {
    HostError::Python(data_plane::describe(py, &error))
}

struct Args {
    python: String,
    control_fd: i32,
}

fn parse_args() -> Result<Args, HostError> {
    let mut python = None;
    let mut control_fd = None;
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        let value = args
            .next()
            .ok_or_else(|| HostError::Usage(format!("{flag} needs a value")))?;
        match flag.as_str() {
            "--python" => python = Some(value),
            "--control-fd" => {
                control_fd = Some(
                    value
                        .parse()
                        .map_err(|_| HostError::Usage(format!("bad --control-fd {value}")))?,
                )
            }
            _ => return Err(HostError::Usage(format!("unknown argument {flag}"))),
        }
    }
    Ok(Args {
        python: python.ok_or_else(|| HostError::Usage("--python is required".to_owned()))?,
        control_fd: control_fd
            .ok_or_else(|| HostError::Usage("--control-fd is required".to_owned()))?,
    })
}

/// What `rust_host.HostConfig` carries.
struct HostConfig {
    handshake_address: String,
    engine_index: u16,
    ready_response: Vec<u8>,
    in_flight_depth: usize,
    limits: IngressLimits,
    stream_interval: i64,
    spec_num_tokens: Option<usize>,
    ngram_context_len: usize,
    request_token_history: bool,
    physical_context_len: i64,
    kv_total_pages: i64,
    rank: usize,
    replica_ranks: Vec<usize>,
    pp_ranks: Option<Vec<usize>>,
    dp_ranks: Option<Vec<usize>>,
    collective_bind_host: String,
    controls: ControlConfig,
    metrics: Metrics,
    batch_log: BatchLogConfig,
    finish_log: FinishLog,
    /// The scheduler records KV cache events.
    kv_events: bool,
    /// Where this rank publishes them, if it does.
    kv_events_publisher: Option<KvEventsConfig>,
    /// With the L3 storage tier, the executor's settings.
    l3: Option<ts_l3::Settings>,
    /// `--enable-eplb`: the rebalance's settings.
    eplb: Option<Eplb>,
    /// An EPD prefill node: it receives image embeddings from encode
    /// workers.
    epd_receive_enabled: bool,
    /// Grammar-constrained decoding's settings, if it is on.
    grammar: Option<GrammarConfig>,
    /// The data plane fills bitmasks from the matchers' committed state.
    eager_grammar: bool,
}

/// `rust_host.HostConfig.eplb`.
struct Eplb {
    config: EplbConfig,
    /// The EP group in EP-rank order, when it has more than one rank.
    ep_ranks: Option<Vec<usize>>,
}

impl Eplb {
    fn from_python(eplb: &Bound<'_, PyAny>) -> PyResult<Option<Self>> {
        if eplb.is_none() {
            return Ok(None);
        }
        let specs = eplb.getattr("specs")?;
        let specs = EplbSpecs {
            num_layers: specs.getattr("num_layers")?.extract()?,
            num_logical_experts: specs.getattr("num_logical_experts")?.extract()?,
            num_physical_experts: specs.getattr("num_physical_experts")?.extract()?,
            ep_size: specs.getattr("ep_size")?.extract()?,
            ep_rank: specs.getattr("ep_rank")?.extract()?,
            ep_rank_nodes: specs.getattr("ep_rank_nodes")?.extract()?,
            all_to_all_ep: specs.getattr("all_to_all_ep")?.extract()?,
            num_groups: specs.getattr("num_groups")?.extract()?,
            num_nodes: specs.getattr("num_nodes")?.extract()?,
        };
        let algorithm: String = eplb.getattr("algorithm")?.extract()?;
        let algorithm = Algorithm::resolve(
            &algorithm,
            specs.num_groups.map(|groups| groups as i64),
            specs.num_nodes as i64,
        )
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
        let ep_ranks: Vec<usize> = eplb.getattr("ep_ranks")?.extract()?;
        Ok(Some(Self {
            config: EplbConfig {
                specs,
                rebalance_num_iterations: eplb.getattr("rebalance_num_iterations")?.extract()?,
                layers_per_chunk: eplb.getattr("layers_per_chunk")?.extract()?,
                algorithm,
                commit_delay_forwards: COMMIT_DELAY_FORWARDS,
            },
            ep_ranks: (ep_ranks.len() > 1).then_some(ep_ranks),
        }))
    }
}

/// The grammar settings' `reasoning` (`rust_host.GrammarSettings`), if
/// grammars are on and it is set.
fn reasoning_grammar(grammar: &Bound<'_, PyAny>) -> PyResult<Option<ReasoningGrammar>> {
    if grammar.is_none() {
        return Ok(None);
    }
    let reasoning = grammar.getattr("reasoning")?;
    if reasoning.is_none() {
        return Ok(None);
    }
    Ok(Some(ReasoningGrammar {
        continues: reasoning.getattr("continues")?.extract()?,
        may_open: reasoning.getattr("may_open")?.extract()?,
        schema_slot: reasoning.getattr("schema_slot")?.extract()?,
        markers: reasoning.getattr("markers")?.extract()?,
    }))
}

/// Which metrics this rank exports, and where (`rust_host.HostConfig`'s
/// `metrics_*` fields).
struct Metrics {
    outputs: MetricsConfig,
    /// The scheduler loop's metrics, on the attention-TP group's first rank.
    engine: bool,
    /// The PD transfer metrics, on every rank of a disaggregated engine.
    transfer: bool,
    /// The L3 tier's metrics, on every rank with the tier.
    l3: bool,
    labels: EngineLabels,
    /// The L3 metrics' extra labels: the attention-TP rank and the role.
    tp_rank: i64,
    engine_role: String,
}

impl Metrics {
    fn from_python(config: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            outputs: MetricsConfig {
                prometheus_dir: config
                    .getattr("metrics_prometheus_dir")?
                    .extract::<Option<String>>()?
                    .map(Into::into),
                relay_endpoint: config.getattr("metrics_relay_endpoint")?.extract()?,
            },
            engine: config.getattr("engine_metrics")?.extract()?,
            transfer: config.getattr("transfer_metrics")?.extract()?,
            l3: config.getattr("l3_metrics")?.extract()?,
            labels: EngineLabels {
                model_name: config.getattr("metrics_model_name")?.extract()?,
                app_key: config.getattr("metrics_app_key")?.extract()?,
                dp_rank: config.getattr("dp_rank")?.extract()?,
            },
            tp_rank: config.getattr("attn_tp_rank")?.extract()?,
            engine_role: config.getattr("engine_role")?.extract()?,
        })
    }

    /// The L3 tier's metrics, on a rank that reports them.
    fn l3_metrics(&self, exporter: Option<&Arc<Exporter>>) -> Option<L3Metrics> {
        exporter.filter(|_| self.l3).map(|exporter| {
            L3Metrics::new(
                exporter,
                &L3Labels {
                    model_name: self.labels.model_name.clone(),
                    app_key: self.labels.app_key.clone(),
                    dp_rank: self.labels.dp_rank,
                    tp_rank: self.tp_rank,
                    engine_role: self.engine_role.clone(),
                },
            )
        })
    }

    /// The exporter, when any metric of this rank is on.
    fn exporter(&self) -> Result<Option<Arc<Exporter>>, ts_metrics::MetricsError> {
        let any_output =
            self.outputs.prometheus_dir.is_some() || self.outputs.relay_endpoint.is_some();
        if !any_output || !(self.engine || self.transfer || self.l3) {
            return Ok(None);
        }
        Exporter::new(&self.outputs).map(Some)
    }
}

impl HostConfig {
    /// `requires_bootstrap`: the engine is disaggregated, so it serves only
    /// requests that name their peer; `role` is its PD role.
    fn from_python(
        config: &Bound<'_, PyAny>,
        requires_bootstrap: bool,
        role: Role,
    ) -> PyResult<Self> {
        let l3 = {
            let l3 = config.getattr("l3")?;
            if l3.is_none() {
                None
            } else {
                Some(ts_l3::Settings::from_python(&l3)?)
            }
        };
        Ok(Self {
            handshake_address: config.getattr("handshake_address")?.extract()?,
            engine_index: config.getattr("engine_index")?.extract()?,
            ready_response: config
                .getattr("ready_response")?
                .cast_into::<PyBytes>()?
                .as_bytes()
                .to_vec(),
            in_flight_depth: config.getattr("in_flight_depth")?.extract()?,
            limits: IngressLimits {
                vocab_size: config.getattr("vocab_size")?.extract()?,
                eos_token_ids: config.getattr("eos_token_ids")?.extract()?,
                max_model_len: config.getattr("max_model_len")?.extract()?,
                max_req_input_len: config.getattr("max_req_input_len")?.extract()?,
                enable_output_logprobs: config
                    .getattr("logprob_servability")?
                    .getattr("enable_output_logprobs")?
                    .extract()?,
                requires_bootstrap,
                multimodal: config.getattr("multimodal")?.extract()?,
                grammar: !config.getattr("grammar")?.is_none(),
                reasoning: reasoning_grammar(&config.getattr("grammar")?)?,
                logprobs: {
                    let servability = config.getattr("logprob_servability")?;
                    LogprobServability {
                        supports_prompt_logprobs: servability
                            .getattr("supports_prompt_logprobs")?
                            .extract()?,
                        logprob_order: servability.getattr("logprob_order")?.extract()?,
                        pipeline_parallel: servability.getattr("pipeline_parallel")?.extract()?,
                        disaggregated: servability.getattr("disaggregated")?.extract()?,
                        lm_head_tp_under_dp: config.getattr("lm_head_tp_under_dp")?.extract()?,
                        // The PD decode role's prefill node returns them.
                        computes_prompt_logprobs: role != Role::Decode,
                    }
                },
            },
            stream_interval: config.getattr("stream_interval")?.extract()?,
            spec_num_tokens: config.getattr("spec_num_tokens")?.extract()?,
            ngram_context_len: config.getattr("ngram_context_len")?.extract()?,
            request_token_history: config.getattr("request_token_history")?.extract()?,
            physical_context_len: config.getattr("physical_context_len")?.extract()?,
            kv_total_pages: config.getattr("kv_total_pages")?.extract()?,
            rank: config.getattr("rank")?.extract()?,
            replica_ranks: config.getattr("replica_ranks")?.extract()?,
            pp_ranks: config.getattr("pp_ranks")?.extract()?,
            dp_ranks: config.getattr("dp_ranks")?.extract()?,
            collective_bind_host: config.getattr("collective_bind_host")?.extract()?,
            controls: ControlConfig {
                memory_saver: config.getattr("memory_saver")?.extract()?,
                memory_regions: config.getattr("memory_regions")?.extract()?,
                // With L3, the namespace's weight version.
                l3_weight_version: l3.as_ref().map(|l3| l3.weight_version.clone()),
            },
            l3,
            eplb: Eplb::from_python(&config.getattr("eplb")?)?,
            epd_receive_enabled: config.getattr("epd_receive_enabled")?.extract()?,
            grammar: {
                let grammar = config.getattr("grammar")?;
                if grammar.is_none() {
                    None
                } else {
                    Some(GrammarConfig {
                        tokenizer_info: grammar.getattr("tokenizer_info")?.extract()?,
                        any_whitespace: grammar.getattr("any_whitespace")?.extract()?,
                        compile_timeout: Duration::from_secs_f64(
                            grammar.getattr("compile_timeout_s")?.extract()?,
                        ),
                        max_retries: grammar.getattr("max_retries")?.extract()?,
                    })
                }
            },
            eager_grammar: config.getattr("eager_grammar")?.extract()?,
            metrics: Metrics::from_python(config)?,
            batch_log: BatchLogConfig {
                enabled: config.getattr("batch_log")?.extract()?,
                decode_log_interval: config.getattr("decode_log_interval")?.extract()?,
                num_total_pages: config.getattr("kv_total_pages")?.extract()?,
                spec_num_steps: config.getattr("spec_num_steps")?.extract()?,
                dp_rank: config.getattr("dp_rank")?.extract()?,
                disaggregated: requires_bootstrap,
                cache_state_group_ids: config.getattr("cache_state_group_ids")?.extract()?,
                log_spec_accept_lengths: config.getattr("log_spec_accept_lengths")?.extract()?,
            },
            finish_log: FinishLog {
                enabled: config.getattr("finish_log")?.extract()?,
                request_stats: config.getattr("log_request_stats")?.extract()?,
            },
            kv_events: config.getattr("kv_events")?.extract()?,
            kv_events_publisher: {
                let publisher = config.getattr("kv_events_publisher")?;
                if publisher.is_none() {
                    None
                } else {
                    Some(KvEventsConfig {
                        endpoint: publisher.getattr("endpoint")?.extract()?,
                        replay_endpoint: publisher.getattr("replay_endpoint")?.extract()?,
                        buffer_steps: publisher.getattr("buffer_steps")?.extract()?,
                        max_queue_size: publisher.getattr("max_queue_size")?.extract()?,
                        topic: publisher.getattr("topic")?.extract()?,
                    })
                }
            },
        })
    }
}

/// Set `stop` on SIGTERM or SIGINT, so the loop finishes its round and the
/// host shuts down in order.
fn watch_signals(stop: Arc<AtomicBool>) -> std::io::Result<()> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_io()
        .build()?;
    let (mut terminate, mut interrupt) = {
        let _entered = runtime.enter();
        use tokio::signal::unix::{signal, SignalKind};
        (
            signal(SignalKind::terminate())?,
            signal(SignalKind::interrupt())?,
        )
    };
    thread::Builder::new()
        .name("ts-signals".to_owned())
        .spawn(move || {
            runtime.block_on(async {
                tokio::select! {
                    _ = terminate.recv() => tracing::info!("SIGTERM: stopping"),
                    _ = interrupt.recv() => tracing::info!("SIGINT: stopping"),
                }
            });
            stop.store(true, Ordering::Relaxed);
        })?;
    Ok(())
}

fn serve(args: &Args) -> Result<(), HostError> {
    let stop = Arc::new(AtomicBool::new(false));
    watch_signals(stop.clone())?;

    let (bootstrap, outbox, mode) = Python::attach(|py| {
        let built = (|| {
            // Before anything imports the scheduler package.
            ts_scheduler::alias_builtin_module(py)?;
            let outbox = Bound::new(py, pd::Outbox::new())?;
            let bootstrap = py
                .import("tokenspeed.runtime.engine.rust_host")?
                .call_method1("bootstrap", (args.control_fd, outbox.clone(), 2_u32))?;
            let mode: String = bootstrap.getattr("mode")?.extract()?;
            Ok((bootstrap.unbind(), outbox.unbind(), mode))
        })();
        built.map_err(|error| python_error(py, error))
    })?;
    if mode == "encode" {
        return encode::serve(bootstrap, &stop);
    }

    let (bootstrap, scheduler, config, data_plane, role, peers, exporter) = Python::attach(|py| {
        let built = (|| {
            let bootstrap = bootstrap.into_bound(py);
            let outbox = outbox.into_bound(py);
            let scheduler = Scheduler::from_python(&bootstrap.getattr("scheduler")?)?;
            let python_config = bootstrap.getattr("config")?;
            let metrics = Metrics::from_python(&python_config)?;
            let exporter = metrics
                .exporter()
                .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))?;
            let transfer_metrics = exporter
                .as_ref()
                .filter(|_| metrics.transfer)
                .map(|exporter| {
                    Arc::new(KvTransferMetrics::new(
                        exporter,
                        &metrics.labels.model_name,
                        &metrics.labels.app_key,
                    ))
                });
            let peers = pd::start(
                py,
                bootstrap.getattr("transfer_peer")?,
                bootstrap.getattr("epd_receive")?,
                outbox.get(),
                transfer_metrics,
            )?;
            let role = peers.role;
            let config = HostConfig::from_python(&python_config, peers.transfer.is_some(), role)?;
            let data_plane = bootstrap.getattr("data_plane")?.unbind();
            Ok((
                bootstrap.unbind(),
                scheduler,
                config,
                data_plane,
                role,
                peers,
                exporter,
            ))
        })();
        built.map_err(|error| python_error(py, error))
    })?;

    // The ranks find each other through the data plane's rendezvous, then
    // connect: every rank is past startup once the mesh is up.
    let listener = Listener::bind(&config.collective_bind_host)?;
    let port = listener.port()?;
    let addresses: Vec<String> = Python::attach(|py| {
        bootstrap
            .call_method1(py, "collective_addresses", (port,))
            .and_then(|addresses| addresses.extract(py))
            .map_err(|error| python_error(py, error))
    })?;
    let mesh = Mesh::connect(config.rank, listener, &addresses, HANDSHAKE_TIMEOUT)?;
    let ranks = Ranks {
        mesh,
        replica: Group::new(REPLICA_GROUP, config.replica_ranks.clone(), config.rank)?,
        pp: config
            .pp_ranks
            .clone()
            .map(|ranks| Group::new(PP_GROUP, ranks, config.rank))
            .transpose()?,
        dp: config
            .dp_ranks
            .clone()
            .map(|ranks| Group::new(DP_GROUP, ranks, config.rank))
            .transpose()?,
        ep: config
            .eplb
            .as_ref()
            .and_then(|eplb| eplb.ep_ranks.clone())
            .map(|ranks| Group::new(EP_GROUP, ranks, config.rank))
            .transpose()?,
    };

    // The replica's first member speaks for it.
    let transport = if ranks.replica.index() == 0 {
        let mut ready: EngineCoreReadyResponse = ts_wire::decode_bytes(&config.ready_response)?;
        ready.declare_scheduler();
        Some(EngineTransport::connect(
            &config.handshake_address,
            config.engine_index,
            ready,
            HANDSHAKE_TIMEOUT,
        )?)
    } else {
        None
    };
    let call = |method: &str| {
        Python::attach(|py| {
            bootstrap
                .call_method0(py, method)
                .map(drop)
                .map_err(|error| python_error(py, error))
        })
    };
    call("report_ready")?;
    call("enter_serving")?;

    if config.grammar.is_some() {
        // The data plane drives the matchers through these.
        Python::attach(|py| {
            let entries: Vec<(&str, usize)> = ts_grammar::table::entries();
            data_plane
                .call_method1(py, "set_grammar_entries", (entries,))
                .map(drop)
                .map_err(|error| python_error(py, error))
        })?;
    }
    let data_plane =
        PythonDataPlane::start(data_plane, peers.streams).map_err(HostError::DataPlane)?;
    let outputs = OutputProcessor::new(
        config.spec_num_tokens,
        config.stream_interval,
        Some(config.physical_context_len),
        role,
        config.finish_log,
    );
    let loop_config = LoopConfig {
        in_flight_depth: config.in_flight_depth,
        limits: config.limits,
        kv_total_pages: config.kv_total_pages,
        ngram_context_len: config.ngram_context_len,
        request_token_history: config.request_token_history,
        controls: config.controls.clone(),
        batch_log: config.batch_log.clone(),
        kv_events: config.kv_events,
        epd_receive_enabled: config.epd_receive_enabled,
        eager_grammar: config.eager_grammar,
    };
    tracing::info!(
        rank = config.rank,
        engine_index = config.engine_index,
        replica_ranks = ?config.replica_ranks,
        pp_ranks = ?config.pp_ranks,
        dp_ranks = ?config.dp_ranks,
        in_flight_depth = config.in_flight_depth,
        "serving"
    );
    let metrics = exporter
        .as_ref()
        .filter(|_| config.metrics.engine)
        .map(|exporter| EngineMetrics::new(exporter, &config.metrics.labels));
    // The L3 tier's executor connects its store before serving starts.
    let l3 = config.l3.as_ref().map(ts_l3::Settings::start).transpose()?;
    let grammars = config
        .grammar
        .clone()
        .map(Grammars::new)
        .transpose()
        .map_err(HostError::Interpreter)?;
    let eplb = config
        .eplb
        .as_ref()
        .map(|eplb| Controller::new(eplb.config.clone()))
        .transpose()
        .map_err(HostError::Interpreter)?;
    let kv_events = config
        .kv_events_publisher
        .as_ref()
        .map(|publisher| KvEventPublisher::start(publisher, config.metrics.labels.dp_rank))
        .transpose()?;
    let mut event_loop = EventLoop::new(
        scheduler,
        data_plane,
        transport,
        outputs,
        loop_config,
        ranks,
        Subsystems {
            transfer: peers.transfer,
            embeddings: peers.embeddings,
            metrics,
            kv_events,
            eplb,
            grammars,
            l3_metrics: config.metrics.l3_metrics(exporter.as_ref()),
            l3,
        },
    );
    let served = event_loop.run(&stop);
    // Stops the forward thread and tells the frontend the engine is gone.
    drop(event_loop);
    call("close")?;
    Ok(served?)
}

fn main() -> ExitCode {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_env("TS_HOST_LOG").unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .init();
    let served = parse_args().and_then(|args| {
        interpreter::start(&args.python).map_err(HostError::Interpreter)?;
        let served = serve(&args);
        if served.is_err() {
            // As `run_event_loop` does: the launcher then stops the other
            // ranks.
            let notified = Python::attach(|py| {
                py.import("tokenspeed.runtime.engine.rust_host")?
                    .call_method0("notify_launcher_of_failure")
                    .map(drop)
            });
            if let Err(error) = notified {
                tracing::error!(%error, "could not notify the launcher");
            }
        }
        served
    });
    match served {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            tracing::error!("{error}");
            ExitCode::FAILURE
        }
    }
}

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

//! The scheduling loop for one rank.
//!
//! Each round admits new requests, advances the cache ops acknowledged since
//! the last round, plans, dispatches the round's work to the data plane
//! without waiting on the device, commits the forwards that are more than
//! `in_flight_depth` rounds old, and feeds the results back to the
//! scheduler in one `advance`. Depth 0 commits a forward in the round that
//! dispatched it; depth 1 is the overlap schedule. Those two advances are
//! the scheduler's only feedback.
//!
//! Every rank of a replica -- an attention-TP group, across every pipeline
//! stage -- runs the same scheduler. The replica's first member owns the
//! frontend and relays each round's messages to the others, so all of them
//! admit the same requests in the same round; only it sends outputs. Only a
//! pipeline's last stage samples, so it shares each forward's results with
//! the other stages before they commit. With attention DP, the replicas'
//! schedulers differ but their forwards run in step: every round, every
//! rank shares the size of its batch, and a rank with none runs an idle
//! forward when another has one. A stop request rides those collectives, so
//! every rank stops after the same round. One rank is the same path with
//! groups of one.
//!
//! Control operations (`control`) arrive with the requests and are relayed
//! like them. A pause withholds new requests from the scheduler or, frozen,
//! forbids forwards while the ranks stay in step: the scheduler learns the
//! permission from the round's storage tick, so a frozen round plans only
//! cache and storage work. Weight operations and cache flushes complete
//! through a same-round gate, so every attention-DP rank runs the same one
//! in the same round.

use std::{
    collections::{HashSet, VecDeque},
    sync::atomic::{AtomicBool, Ordering},
    time::{Duration, Instant},
};

use ts_collective::{Group, Mesh, Reduce};
use ts_metrics::{EngineMetrics, L3Metrics};
use ts_pd::{EmbeddingItem, EmbeddingReceiver, LandedRequest, PdError, PdEvent, TransferPeer};
use ts_scheduler::{
    CacheEvent, CacheOpKind, ExecutionPlan, ForwardBatchView, ForwardEvent, ForwardEventKind,
    Scheduler, StorageRound, StorageTickView,
};
use ts_wire::{
    control::{FlushCacheReqOutput, ProfileReqOutput},
    ControlOutput, ControlRequest, EngineRequest,
};

use crate::{
    batch_log::{BatchLogConfig, BatchLogger, RoundStats},
    cache_ops::{CacheOpError, CacheOps},
    codec::{Reader, Writer},
    control::{self, ControlConfig, Controls, Drain, Effect},
    eplb::{self, InternalOp, SlotMoves},
    grammar::{self as grammar_cache, GrammarKey, Grammars, InvalidGrammar, Lookup},
    ingress::{self, Admission, Admitted, IngressLimits},
    kv_events::KvEventPublisher,
    l3::{L3Error, Tier},
    output::{
        CommitStats, Committed, ForwardResult, HistoryRows, HistorySeeds, InputLogprobPlan,
        LogprobPlan, NgramWindows, OutputProcessor, StreamedOutput,
    },
    relay::{self, RelayError},
    request::RequestGrammar,
};

/// Fold what an output-processor call produced into the round's results.
fn absorb(
    committed: Committed,
    events: &mut Vec<ForwardEvent>,
    outputs: &mut Vec<StreamedOutput>,
    released: &mut Vec<String>,
) {
    events.extend(committed.events);
    outputs.extend(committed.outputs);
    released.extend(committed.released);
}

fn pd_event(kind: ForwardEventKind, rid: &str, bootstrap_token: i32) -> ForwardEvent {
    ForwardEvent {
        kind,
        request_id: rid.to_owned(),
        tokens: Vec::new(),
        spec_candidate_ids: Vec::new(),
        reserve_num_tokens: 0,
        bootstrap_token,
    }
}

/// A 64-bit FNV-1a digest of `values`' little-endian bytes, for comparing
/// counters across ranks.
fn fnv1a<'a>(values: impl Iterator<Item = &'a i64>) -> i64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for value in values {
        for byte in value.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0100_0000_01b3);
        }
    }
    // Keeps `-digest` representable for the MAX agreement.
    (hash >> 1) as i64
}

/// A placement `[layers, physical]` as bytes: the row width, then the values.
fn encode_rows(rows: &[Vec<i32>]) -> Vec<u8> {
    let mut out = Writer::new();
    out.count(rows.first().map_or(0, Vec::len));
    out.i32s(&rows.concat());
    out.finish()
}

fn decode_rows(bytes: &[u8]) -> Option<Vec<Vec<i32>>> {
    let mut reader = Reader::new(bytes);
    let width = reader.count()?;
    let values = reader.i32s()?;
    if width == 0 || values.len() % width != 0 {
        return None;
    }
    reader.finish(values.chunks(width).map(<[i32]>::to_vec).collect())
}

/// How long an idle loop waits for a request before checking `stop` again.
const IDLE_WAIT: Duration = Duration::from_millis(10);

/// An admitted request on its way to the scheduler.
struct Submission {
    rid: String,
    spec: ts_scheduler::RequestSpec,
    bootstrap: Option<ts_pd::BootstrapInfo>,
    /// EPD: the items whose embeddings it waits for.
    embeddings: Vec<EmbeddingItem>,
}

/// How long a frozen round sleeps, so a `keep` pause does not spin a core.
const PAUSED_IDLE_SLEEP: Duration = Duration::from_millis(1);

/// A failure of a data-plane operation; serving cannot continue.
#[derive(Debug, thiserror::Error)]
#[error("data plane: {0}")]
pub struct DataPlaneError(pub String);

/// The device side: what the loop asks of it.
pub trait DataPlane {
    /// A submitted forward, resolved by [`DataPlane::complete`].
    type Pending;

    /// Register requests before their first forward.
    fn admit(&mut self, requests: &[Admission]) -> Result<(), DataPlaneError>;

    /// Run a round's plan and, with a batch, its forward; never waits on the
    /// device. Returns the forward to commit later, if one ran.
    fn execute(
        &mut self,
        plan: ExecutionPlan,
        dispatch: &Dispatch<'_>,
    ) -> Result<Option<Self::Pending>, DataPlaneError>;

    /// Wait for a forward's results.
    fn complete(&mut self, pending: Self::Pending) -> Result<ForwardResult, DataPlaneError>;

    /// The cache ops acknowledged since the last poll, in the order they
    /// landed; never waits on the device.
    fn poll_cache_events(&mut self) -> Result<Vec<CacheEvent>, DataPlaneError>;

    /// Forget requests that ended.
    fn release(&mut self, rids: &[String]) -> Result<(), DataPlaneError>;

    /// D role: apply a landed remote prefill's device-side effects before
    /// the request's first decode: the prefill node's drafter candidates
    /// into its slot, then the slot's readiness for decode.
    fn land(
        &mut self,
        spec_candidates: Option<(i32, Vec<i32>)>,
        cache_slot: Option<i32>,
    ) -> Result<(), DataPlaneError>;

    /// A frozen round's idle forward, when another attention-DP rank runs a
    /// batch; returns once it ran.
    fn idle_forward(&mut self, dp: &DpRound) -> Result<(), DataPlaneError>;

    /// Check a weight load's local prerequisites without touching weights
    /// or their validity. Returns whether this rank can load and why not.
    /// Every rank must agree before any rank enters a loader.
    fn check_weight_update(
        &mut self,
        op: &ControlRequest,
    ) -> Result<(bool, String), DataPlaneError>;

    /// Apply one weight operation (a load, a group's init or teardown),
    /// behind the forwards dispatched before it. Returns whether it worked
    /// and a message.
    fn update_weights(&mut self, op: &ControlRequest) -> Result<(bool, String), DataPlaneError>;

    /// Publish or invalidate a weight load every rank started, after every
    /// rank agreed its result and the new cache namespace. Local load
    /// success alone must never permit a forward or a KV wake.
    fn commit_weight_update(&mut self, success: bool) -> Result<(), DataPlaneError>;

    /// Start or stop the profiler (`ProfileReq`). Returns the reply.
    fn profile(
        &mut self,
        request: &ts_wire::control::ProfileReq,
    ) -> Result<(bool, String), DataPlaneError>;

    /// Free these memory regions, in order, once the scheduler drained.
    fn release_memory(&mut self, tags: &[String]) -> Result<(), DataPlaneError>;

    /// Restore these memory regions, in order.
    fn resume_memory(&mut self, tags: &[String]) -> Result<(), DataPlaneError>;

    /// EPLB: this rank's route counters since the previous snapshot,
    /// `[layers, physical]`, with the placement they were counted under;
    /// starts a new window behind the forwards dispatched before it.
    #[allow(clippy::type_complexity)]
    fn snapshot_expert_load(&mut self) -> Result<(Vec<Vec<i64>>, Vec<Vec<i32>>), DataPlaneError>;

    /// Grammar: these requests' matchers, by handle, with the tokens that
    /// end each grammar, for the forwards they run in from now on.
    fn attach_grammars(
        &mut self,
        grammars: &[(String, u64, Vec<i32>)],
    ) -> Result<(), DataPlaneError>;

    /// EPD: put the landed embeddings onto the requests' items, before any
    /// forward of theirs; with row sharding, the replica's ranks rebuild
    /// each image's other rows in the same order.
    fn epd_publish(&mut self, landed: &[LandedRequest]) -> Result<(), DataPlaneError>;

    /// EPD: the receive slots whose rows `epd_publish` finished copying out.
    fn epd_released(&mut self) -> Result<Vec<usize>, DataPlaneError>;

    /// EPLB: move one chunk of layers' experts into `rows`, this rank's part
    /// being `moves` (one per layer), and switch their routing tables.
    fn apply_expert_placement(
        &mut self,
        layer_ids: &[usize],
        rows: &[Vec<i32>],
        moves: &[SlotMoves],
    ) -> Result<(), DataPlaneError>;
}

/// What a round hands the data plane besides its plan.
#[derive(Debug, Clone, Copy, Default)]
pub struct Dispatch<'a> {
    /// The model batch, if one runs.
    pub forward: Option<&'a ForwardBatchView>,
    /// With attention DP, every rank's batch this round.
    pub dp: Option<&'a DpRound>,
    /// Engram's input windows for the batch, for a model that reads them.
    pub ngram: Option<&'a NgramWindows>,
    /// With a grammar row in the batch, whether each row's sampled token
    /// advances its matcher (`advance_mask`): not on a prefill chunk that
    /// does not complete its prompt.
    pub grammar_advance: Option<&'a [bool]>,
    /// The logprobs the batch computes beyond its sampled tokens'.
    pub logprobs: Option<&'a LogprobPlan>,
    /// For a model that reads request token history, the committed
    /// prefixes the batch seeds; `None` when no row needs one.
    pub history: Option<&'a HistorySeeds>,
}

/// Every attention-DP rank's batch this round, in rank order
/// (`DpForwardMetadata`), for the model's cross-rank collectives.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DpRound {
    /// Tokens each rank forwards.
    pub num_tokens: Vec<i64>,
    /// Requests in each rank's batch.
    pub batch_sizes: Vec<i64>,
    /// Prefill rows in each rank's batch.
    pub num_extends: Vec<i64>,
    /// This rank has no batch while another has: it runs an idle forward so
    /// the model's collectives see every rank.
    pub idle_forward: bool,
}

/// The ranks a loop agrees with every round.
pub struct Ranks {
    /// Connections to every other rank.
    pub mesh: Mesh,
    /// The replica: one scheduler, mirrored on every member (attention TP
    /// times pipeline stages). Its first member owns the frontend.
    pub replica: Group,
    /// With pipeline parallelism, this rank's pipeline: the ranks of its
    /// attention-TP position, one per stage, in stage order. The last
    /// samples.
    pub pp: Option<Group>,
    /// With attention DP, every rank: their forwards run in step.
    pub dp: Option<Group>,
    /// With expert parallelism over several ranks, this rank's EP group in
    /// EP-rank order: the expert rebalance agrees over it.
    pub ep: Option<Group>,
}

/// The scheduler's load as reported with each output batch.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Load {
    /// Requests admitted and not ended.
    pub num_running: i64,
    /// Requests awaiting admission.
    pub num_waiting: i64,
    /// Cache pages held by running requests.
    pub kv_active_pages: i64,
    /// Usable cache pages.
    pub kv_total_pages: i64,
    /// Resident cache blocks, including reclaimable prefix-cache entries.
    pub kv_used_pages: i64,
}

/// The frontend: requests in, outputs out.
pub trait Frontend {
    /// Messages received since the last call. Waits up to `wait` for the
    /// first when given one; otherwise returns at once.
    fn receive(&mut self, wait: Option<Duration>) -> Vec<EngineRequest>;

    /// The frontend is gone: no message will arrive again.
    fn closed(&self) -> bool;

    /// Send one round's outputs.
    fn send(&mut self, outputs: Vec<StreamedOutput>, load: Load);

    /// Send a control operation's reply.
    fn reply(&mut self, output: ControlOutput);
}

/// A rank that does not own a frontend: the replica's members other than
/// the first.
impl<F: Frontend> Frontend for Option<F> {
    fn receive(&mut self, wait: Option<Duration>) -> Vec<EngineRequest> {
        self.as_mut()
            .map_or_else(Vec::new, |frontend| frontend.receive(wait))
    }

    fn closed(&self) -> bool {
        self.as_ref().is_some_and(Frontend::closed)
    }

    fn send(&mut self, outputs: Vec<StreamedOutput>, load: Load) {
        if let Some(frontend) = self {
            frontend.send(outputs, load);
        }
    }

    fn reply(&mut self, output: ControlOutput) {
        if let Some(frontend) = self {
            frontend.reply(output);
        }
    }
}

/// A loop failure.
#[derive(Debug, thiserror::Error)]
pub enum LoopError {
    /// The data plane failed.
    #[error(transparent)]
    DataPlane(#[from] DataPlaneError),
    /// A forward result broke an engine invariant.
    #[error(transparent)]
    Commit(#[from] crate::output::CommitError),
    /// The cache acknowledgments disagree with the dispatched plans.
    #[error(transparent)]
    Cache(#[from] CacheOpError),
    /// The ranks could not agree.
    #[error(transparent)]
    Collective(#[from] ts_collective::Error),
    /// The relayed frontend messages did not decode.
    #[error(transparent)]
    Relay(#[from] RelayError),
    /// The PD transfer peer failed.
    #[error(transparent)]
    Transfer(#[from] PdError),
    /// The replica could not agree on its L3 storage reports.
    #[error(transparent)]
    Storage(#[from] L3Error),
    /// The L3 executor refused what a plan or a control operation asked.
    #[error(transparent)]
    StorageExecutor(#[from] ts_l3::ExecutorError),
    /// The ranks disagree on a control operation.
    #[error("control: {0}")]
    Control(String),
    /// The scheduler refused an event.
    #[error("scheduler: {0}")]
    Scheduler(String),
}

/// Fixed settings of the loop.
#[derive(Debug, Clone)]
pub struct LoopConfig {
    /// Dispatched forwards that may await commit.
    pub in_flight_depth: usize,
    /// Checks every admission runs.
    pub limits: IngressLimits,
    /// Usable cache pages, reported with the load.
    pub kv_total_pages: i64,
    /// Predecessors Engram reads per token; 0 for a model without Engram.
    pub ngram_context_len: usize,
    /// The model reads each request's committed token history, which the
    /// forwards seed into the history rows.
    pub request_token_history: bool,
    /// The control operations' settings.
    pub controls: ControlConfig,
    /// The batch log's settings.
    pub batch_log: BatchLogConfig,
    /// The scheduler records KV cache events: every rank drains them each
    /// round, and the rank with a publisher sends them.
    pub kv_events: bool,
    /// An EPD prefill node: a request whose images an encode node embeds
    /// waits for the embeddings before it is scheduled. Its first stage
    /// receives them ([`Subsystems::embeddings`]); a later stage follows the
    /// first's decisions.
    pub epd_receive_enabled: bool,
    /// The data plane fills grammar bitmasks before a forward from the
    /// matchers' state, which only a commit advances: a grammar batch
    /// cannot overlap a pending forward (`uses_eager_grammar`).
    pub eager_grammar: bool,
}

/// The optional subsystems a loop coordinates; each is `None` on an engine
/// that runs without it.
#[derive(Default)]
pub struct Subsystems {
    /// The PD transfer peer of a disaggregated engine.
    pub transfer: Option<Box<dyn TransferPeer + Send>>,
    /// EPD: the embedding receiver of a multimodal prefill node's first
    /// stage.
    pub embeddings: Option<Box<dyn EmbeddingReceiver + Send>>,
    /// The engine metrics, on the rank that reports them.
    pub metrics: Option<EngineMetrics>,
    /// The KV cache event publisher, on the rank that publishes them.
    pub kv_events: Option<KvEventPublisher>,
    /// The online expert rebalance controller (`--enable-eplb`).
    pub eplb: Option<eplb::Controller>,
    /// Grammar compilation and its cache (`--grammar-backend xgrammar`).
    pub grammars: Option<Grammars>,
    /// The L3 tier's metrics, with L3 and `--enable-metrics`.
    pub l3_metrics: Option<L3Metrics>,
    /// With the L3 storage tier, this rank's executor: the replica agrees
    /// on the executors' reports every round, and flushes delete the tier's
    /// namespace. `None` without the tier.
    pub l3: Option<ts_l3::Executor>,
}

/// One rank's scheduling loop.
pub struct EventLoop<D: DataPlane, F: Frontend> {
    scheduler: Scheduler,
    data_plane: D,
    frontend: F,
    outputs: OutputProcessor,
    config: LoopConfig,
    ranks: Ranks,
    transfer: Option<Box<dyn TransferPeer + Send>>,
    embeddings: Option<Box<dyn EmbeddingReceiver + Send>>,
    metrics: Option<EngineMetrics>,
    kv_events: Option<KvEventPublisher>,
    eplb: Option<eplb::Controller>,
    grammars: Option<Grammars>,
    /// The L3 tier: this rank's executor and the replica's agreement.
    l3: Option<Tier>,
    l3_metrics: Option<L3Metrics>,
    /// Admissions waiting for their grammars to compile, in arrival order.
    grammar_queue: Vec<(Submission, GrammarKey, Instant)>,
    batch_logger: BatchLogger,
    /// The token-history rows' owners, for a model that reads them.
    history: Option<HistoryRows>,
    /// Dispatched forwards, oldest first, with the prompt rows each gathers.
    in_flight: VecDeque<(ForwardBatchView, Option<InputLogprobPlan>, D::Pending)>,
    cache_ops: CacheOps,
    controls: Controls,
    /// Admissions withheld from the scheduler while paused, in arrival order.
    withheld: Vec<Submission>,
    /// EPD: admissions waiting for their images' embeddings, in staging
    /// order (rank-identical, the order the receiver reports them in).
    staged: Vec<Submission>,
    /// No attention-DP rank had a batch last round, so an idle rank may wait
    /// for requests without holding the others' collectives back.
    dp_idle: bool,
    /// Without L3, the forward permission the scheduler last received; it
    /// starts allowed. With L3 every round's tick carries it.
    forward_allowed: bool,
    /// Last load report on the output wire, including idle heartbeats.
    last_load_report: Option<Instant>,
    /// Last load sent to the frontend, including rounds with no token output.
    reported_load: Option<Load>,
}

/// What a round starts from, the same on every replica member.
struct Relayed {
    /// The replica's first member is stopping.
    stop: bool,
    /// Its frontend messages.
    messages: Vec<EngineRequest>,
    /// With L3, every member's storage report, in member order; empty
    /// without.
    l3_reports: Vec<Vec<u8>>,
}

impl<D: DataPlane, F: Frontend> EventLoop<D, F> {
    /// A loop over `scheduler`, with `outputs` configured for the engine's
    /// speculation and streaming. `frontend` is used only on the replica's
    /// first member; with a PD transfer peer in `subsystems`, `outputs`
    /// follows its role.
    pub fn new(
        scheduler: Scheduler,
        data_plane: D,
        frontend: F,
        outputs: OutputProcessor,
        config: LoopConfig,
        ranks: Ranks,
        subsystems: Subsystems,
    ) -> Self {
        Self {
            controls: Controls::new(config.controls.clone()),
            batch_logger: BatchLogger::new(config.batch_log.clone()),
            history: config.request_token_history.then(HistoryRows::default),
            l3: subsystems.l3.map(Tier::new),
            l3_metrics: subsystems.l3_metrics,
            scheduler,
            data_plane,
            frontend,
            outputs,
            config,
            ranks,
            transfer: subsystems.transfer,
            embeddings: subsystems.embeddings,
            metrics: subsystems.metrics,
            kv_events: subsystems.kv_events,
            eplb: subsystems.eplb,
            grammars: subsystems.grammars,
            grammar_queue: Vec::new(),
            in_flight: VecDeque::new(),
            cache_ops: CacheOps::new(),
            withheld: Vec::new(),
            staged: Vec::new(),
            dp_idle: true,
            forward_allowed: true,
            last_load_report: None,
            reported_load: None,
        }
    }

    /// Run rounds until a stop is requested on the rank that decides for the
    /// others, or something fails.
    pub fn run(&mut self, stop: &AtomicBool) -> Result<(), LoopError> {
        while !self.round(stop.load(Ordering::Relaxed))? {}
        Ok(())
    }

    fn owns_frontend(&self) -> bool {
        self.ranks.replica.index() == 0
    }

    fn idle(&self) -> bool {
        let sizes = self.scheduler.sizes();
        self.in_flight.is_empty()
            && self.cache_ops.open() == 0
            && self.outputs.num_running() == 0
            && sizes.waiting == 0
            && sizes.prefilling == 0
            && sizes.decoding == 0
            && sizes.retracted == 0
            && !self.scheduler.storage_pending()
    }

    /// One round, with this rank's stop request. Returns whether the ranks
    /// agreed to stop after it.
    pub fn round(&mut self, stop_requested: bool) -> Result<bool, LoopError> {
        let mut outputs: Vec<StreamedOutput> = Vec::new();
        let mut released: Vec<String> = Vec::new();

        let received = if self.owns_frontend() {
            let wait = (self.idle() && self.dp_idle).then_some(IDLE_WAIT);
            self.frontend.receive(wait)
        } else {
            Vec::new()
        };
        // Nothing reaches a rank whose frontend is gone: it stops as on a
        // stop request, so every rank stops after this round.
        let stop_requested = stop_requested || self.frontend.closed();
        let report = self.l3.as_ref().map(Tier::report);
        let Relayed {
            mut stop,
            messages,
            l3_reports,
        } = self.relay(stop_requested, received, report)?;
        self.admit(messages, &mut outputs, &mut released)?;
        self.apply_pause_transitions(&mut outputs, &mut released)?;
        self.drain_grammar_queue(&mut outputs, &mut released)?;
        self.drain_staged(&mut outputs, &mut released)?;
        // Every round on every rank, so the attention-DP ranks agree on the
        // flushes and weight operations they run together.
        self.gate()?;

        // The storage tick and the cache-op acknowledgments advance at the
        // head of the round, so this round's plan sees them.
        let storage = self.storage_round(&l3_reports)?;
        let ready = if self.cache_ops.open() > 0 {
            let polled = self.data_plane.poll_cache_events()?;
            self.cache_ops
                .acknowledged(polled, &mut self.ranks.mesh, &self.ranks.replica)?
        } else {
            Vec::new()
        };
        if storage.tick.is_some() || !ready.is_empty() {
            if !ready.is_empty() {
                tracing::debug!(
                    ?ready,
                    open = self.cache_ops.open(),
                    "cache ops acknowledged"
                );
            }
            self.scheduler
                .advance_head(&storage, &ready)
                .map_err(|error| LoopError::Scheduler(error.what().to_owned()))?;
        }
        self.record_storage_metrics()?;

        let mut events = Vec::new();
        if self.controls.forward_blocked() {
            // Frozen: what is dispatched commits, no forward is planned, and
            // the attention-DP ranks stay in step. With memory released not
            // even an idle forward may run: the weights are unmapped.
            self.commit_down_to(0, &mut events, &mut outputs, &mut released)?;
            // Cache and storage work still progresses: the tick forbade
            // forwards, so the plan carries nothing else.
            let plan = self.scheduler.next_execution_plan();
            if let Some(l3) = &self.l3 {
                l3.submit(&plan)?;
            }
            self.cache_ops.dispatched(&plan.cache_ops());
            if plan.needs_device() {
                self.data_plane.execute(plan, &Dispatch::default())?;
            }
            let mut forwarded = false;
            if let Some(dp) = self.dp_round(None, &mut stop)? {
                if dp.idle_forward && !self.controls.memory_released() {
                    self.data_plane.idle_forward(&dp)?;
                    forwarded = true;
                }
            }
            self.note_forward(forwarded);
            std::thread::sleep(PAUSED_IDLE_SLEEP);
        } else {
            self.plan_and_dispatch(&mut stop, &mut events, &mut outputs, &mut released)?;
            self.poll_transfers(&mut events, &mut outputs, &mut released)?;
        }
        // The forward-result feedback point: everything committed this round
        // reaches the scheduler before the next round plans.
        if !events.is_empty() {
            self.scheduler
                .advance(&events)
                .map_err(|error| LoopError::Scheduler(error.what().to_owned()))?;
        }
        // One drain per round, after its last scheduler mutation.
        if self.config.kv_events {
            let drained = self.scheduler.drain_kv_events();
            if let Some(publisher) = &self.kv_events {
                publisher.publish(&drained);
            }
        }
        if !released.is_empty() {
            self.data_plane.release(&released)?;
        }
        if self.owns_frontend() {
            let sizes = self.scheduler.sizes();
            let load = Load {
                num_running: self.outputs.num_running() as i64,
                num_waiting: sizes.waiting as i64,
                kv_active_pages: i64::from(sizes.active_lcm_blocks),
                kv_total_pages: self.config.kv_total_pages,
                kv_used_pages: self.config.kv_total_pages - i64::from(sizes.empty_lcm_blocks),
            };
            // Silent aborts must refresh load immediately; periodic heartbeats
            // also keep an idle engine's load snapshot fresh at its frontend.
            if !outputs.is_empty()
                || self.reported_load != Some(load)
                || self
                    .last_load_report
                    .is_none_or(|last| last.elapsed() >= Duration::from_secs(1))
            {
                self.frontend.send(outputs, load);
                self.reported_load = Some(load);
                self.last_load_report = Some(Instant::now());
            }
        }
        // A deferred pause or release resolves once the scheduler drained.
        self.finish_drain()?;
        for reply in self.controls.take_replies() {
            self.frontend.reply(reply);
        }
        Ok(stop)
    }

    /// The round's plan, dispatched, and the forwards older than the
    /// in-flight depth committed.
    fn plan_and_dispatch(
        &mut self,
        stop: &mut bool,
        events: &mut Vec<ForwardEvent>,
        outputs: &mut Vec<StreamedOutput>,
        released: &mut Vec<String>,
    ) -> Result<(), LoopError> {
        let plan = self.scheduler.next_execution_plan();
        if tracing::enabled!(tracing::Level::DEBUG) {
            if let Some(batch) = plan.remote_prefill() {
                tracing::debug!(rids = ?batch.request_ids, "remote prefill planned");
            }
            if let Some(batch) = plan.remote_decode() {
                tracing::debug!(rids = ?batch.request_ids, "remote decode planned");
            }
        }
        // The storage operations start on the L3 executor at once, apart
        // from the device work.
        if let Some(l3) = &self.l3 {
            l3.submit(&plan)?;
        }
        // Counted here because `execute` below always runs this plan.
        let cache_ops = plan.cache_ops();
        self.cache_ops.dispatched(&cache_ops);
        let aborted = self.outputs.finish_scheduler_aborts(&plan.aborts());
        outputs.extend(aborted.outputs);
        released.extend(aborted.released);
        let mut forward = plan
            .forward_batches()
            .into_iter()
            .next()
            .filter(|batch| !batch.request_ids.is_empty());
        if let Some(metrics) = &self.metrics {
            // A retraction victim's image store is the retraction signal.
            let retracted = cache_ops
                .iter()
                .filter(|op| op.kind == CacheOpKind::SnapshotStore)
                .count();
            metrics.record_preemptions(retracted as i64);
            for _ in 0..aborted.stats.capacity_aborts {
                metrics.record_capacity_abort();
            }
            // A pre-dispatch snapshot: the running count still holds what
            // this round's commits end.
            let sizes = self.scheduler.sizes();
            let iteration_tokens: i64 = forward.as_ref().map_or(0, |batch| {
                batch.input_lengths.iter().map(|&n| i64::from(n)).sum()
            });
            metrics.record_scheduler_iteration(
                self.outputs.num_running() as i64,
                sizes.waiting as i64,
                i64::from(sizes.active_lcm_blocks),
                self.config.kv_total_pages,
                iteration_tokens,
            );
        }

        let dp = self.dp_round(forward.as_ref(), stop)?;
        if dp.as_ref().is_some_and(|dp| dp.idle_forward) {
            // The idle forward replaces this rank's batch; what is dispatched
            // commits first, as nothing would follow it.
            forward = None;
            self.commit_down_to(0, events, outputs, released)?;
        }
        // Gathered before this round's commits: a commit may end requests
        // the batch still names.
        let ngram = match &forward {
            Some(batch) if self.config.ngram_context_len > 0 => Some(
                self.outputs
                    .ngram_windows(batch, self.config.ngram_context_len)?,
            ),
            _ => None,
        };
        let logprobs = forward
            .as_ref()
            .map(|batch| self.outputs.logprob_plan(batch));
        let history = match (&forward, self.history.as_mut()) {
            (Some(batch), Some(rows)) => rows.seeds(batch, &self.outputs)?,
            _ => None,
        };
        let grammar_advance = forward
            .as_ref()
            .and_then(|batch| self.grammar_advance(batch));
        if grammar_advance.is_some() && self.config.eager_grammar && !self.in_flight.is_empty() {
            // The bitmask fill reads matchers only a pending commit advances.
            self.commit_down_to(0, events, outputs, released)?;
        }
        let dispatch = Dispatch {
            forward: forward.as_ref(),
            dp: dp.as_ref(),
            ngram: ngram.as_ref(),
            grammar_advance: grammar_advance.as_deref(),
            logprobs: logprobs.as_ref(),
            history: history.as_ref(),
        };
        // A round with nothing for the device stays off the data plane: a
        // loop waiting on a transfer would otherwise hand the interpreter an
        // empty plan per round and starve the device's own threads.
        let idle_forward = dp.as_ref().is_some_and(|dp| dp.idle_forward);
        if let Some(batch) = &forward {
            self.log_dispatch(batch);
            self.outputs.mark_scheduled(&batch.request_ids);
        }
        if let (Some(metrics), Some(batch)) = (&self.metrics, &forward) {
            // Each request's wait, observed once, at its first dispatch.
            for rid in &batch.request_ids {
                if let Some(at) = self.outputs.take_submitted_at(rid) {
                    metrics.observe_request_queue_time(at.elapsed().as_secs_f64());
                }
            }
        }
        let mut forwarded = idle_forward;
        if plan.needs_device() || idle_forward {
            if let Some(pending) = self.data_plane.execute(plan, &dispatch)? {
                let batch = forward.clone().expect("a pending forward has a batch");
                let input_logprobs = logprobs.and_then(|plan| plan.input);
                self.in_flight.push_back((batch, input_logprobs, pending));
                forwarded = true;
            }
        }
        self.note_forward(forwarded);

        // Commit from the head once the queue exceeds the depth; a round with
        // no new work drains fully so results never wait on future traffic.
        let depth = if forward.is_some() {
            self.config.in_flight_depth
        } else {
            0
        };
        self.commit_down_to(depth, events, outputs, released)
    }

    /// Count this round's forward, real or idle, for the expert rebalance;
    /// the steps it makes due complete through the gate. Every round on
    /// every rank, frozen or not: the attention-DP ranks forward in step,
    /// so their counts, and the steps that follow, stay rank-identical.
    fn note_forward(&mut self, forwarded: bool) {
        if let Some(controller) = self.eplb.as_mut() {
            let due = controller.note_round(forwarded);
            self.controls.enqueue_internal(due);
        }
    }

    /// The round's storage input to the scheduler. With L3, the replica's
    /// agreement on the members' reports, every round. Without, the forward
    /// permission when a freeze starts or ends: the tick is the scheduler's
    /// only input that carries it, and its clock, which times only storage
    /// deadlines, stays 0.
    fn storage_round(&mut self, l3_reports: &[Vec<u8>]) -> Result<StorageRound, LoopError> {
        let allowed = !self.controls.forward_blocked();
        if let Some(l3) = self.l3.as_mut() {
            return Ok(l3.agree(l3_reports, allowed, allowed)?);
        }
        if allowed == self.forward_allowed {
            return Ok(StorageRound::default());
        }
        self.forward_allowed = allowed;
        Ok(StorageRound {
            tick: Some(StorageTickView {
                now_us: 0,
                allow_forward: allowed,
                allow_storage: allowed,
                queued_operations: 0,
                queued_bytes: 0,
                oldest_operation_us: 0,
            }),
            ..StorageRound::default()
        })
    }

    /// One-shot pause edges: an `abort` pause aborts every request,
    /// notifying its client; once unpaused, the withheld admissions reach
    /// the scheduler, but for those aborted while withheld, which are only
    /// reaped.
    fn apply_pause_transitions(
        &mut self,
        outputs: &mut Vec<StreamedOutput>,
        released: &mut Vec<String>,
    ) -> Result<(), LoopError> {
        if self.controls.take_abort_all() {
            for rid in self.outputs.rids() {
                self.outputs.mark_abort(&rid, true);
            }
        }
        if self.controls.take_cancel_unscheduled() {
            // Requests still compiling their grammars or receiving their
            // images' embeddings produced nothing yet: cancelled, they end
            // toward their clients (the grammar queue's next pass, or their
            // receive settling) instead of being withheld past the drain.
            let unscheduled = self
                .grammar_queue
                .iter()
                .map(|(submission, _, _)| submission)
                .chain(&self.staged);
            for submission in unscheduled {
                self.outputs.mark_abort(&submission.rid, true);
            }
        }
        if self.controls.admit_blocked() || self.withheld.is_empty() {
            return Ok(());
        }
        let mut admitted = Vec::new();
        for submission in std::mem::take(&mut self.withheld) {
            match self.outputs.state(&submission.rid) {
                None => {}
                Some(state) if state.finished() => {
                    let reaped = self.outputs.reap_unscheduled(&submission.rid);
                    outputs.extend(reaped.outputs);
                    released.extend(reaped.released);
                }
                Some(_) => admitted.push(submission),
            }
        }
        self.submit(admitted)
    }

    /// Give a request its grammar's matcher, or fail it if the grammar
    /// cannot be used; whether it proceeds to the scheduler.
    fn start_grammar(
        &mut self,
        rid: &str,
        key: &GrammarKey,
        lookup: Lookup,
        outputs: &mut Vec<StreamedOutput>,
        released: &mut Vec<String>,
    ) -> Result<bool, LoopError> {
        let grammar = match lookup {
            Lookup::Ready(grammar) => grammar,
            Lookup::Invalid(invalid) => {
                let failed = self.outputs.fail_unscheduled(
                    rid,
                    &format!(
                        "Failed to compile {} grammar: {}",
                        key.kind, invalid.message
                    ),
                    invalid.permanent.then_some(400),
                );
                outputs.extend(failed.outputs);
                released.extend(failed.released);
                return Ok(false);
            }
            Lookup::Compiling => unreachable!("only a compiled or invalid grammar starts"),
        };
        let matcher = Grammars::matcher(&grammar).map_err(LoopError::Control)?;
        let stop_token_ids = matcher.stop_token_ids();
        let (handle, matcher) = ts_grammar::table::register(matcher);
        self.outputs
            .attach_grammar(rid, RequestGrammar { handle, matcher });
        self.data_plane
            .attach_grammars(&[(rid.to_owned(), handle, stop_token_ids)])?;
        Ok(true)
    }

    /// Whether each row of `batch` advances its grammar, if any row has
    /// one: a prefill chunk that does not complete the
    /// prompt samples a token the commit discards.
    fn grammar_advance(&self, batch: &ForwardBatchView) -> Option<Vec<bool>> {
        let any = batch.request_ids.iter().any(|rid| {
            self.outputs
                .state(rid)
                .is_some_and(|state| state.grammar.is_some())
        });
        if !any {
            return None;
        }
        Some(
            (0..batch.request_ids.len())
                .map(|i| {
                    i >= batch.num_extends
                        || batch.extend_prefix_lens[i] + batch.input_lengths[i]
                            >= batch.prefill_lengths[i]
                })
                .collect(),
        )
    }

    /// Admit the queued requests whose grammars compiled on every replica
    /// rank, and fail those that timed out on any or do not compile; every
    /// rank admits the same ones in the same round.
    fn drain_grammar_queue(
        &mut self,
        outputs: &mut Vec<StreamedOutput>,
        released: &mut Vec<String>,
    ) -> Result<(), LoopError> {
        let Some(grammars) = self.grammars.as_mut() else {
            return Ok(());
        };
        if self.grammar_queue.is_empty() {
            if grammars.has_compiled() {
                grammars.retain(&[], &HashSet::new());
            }
            return Ok(());
        }
        let now = Instant::now();
        let timeout = grammars.compile_timeout();
        let mut local = Vec::with_capacity(3 * self.grammar_queue.len());
        let (mut ready, mut failed, mut invalid) = (Vec::new(), Vec::new(), Vec::new());
        let mut invalid_here = Vec::with_capacity(self.grammar_queue.len());
        for (submission, key, queued_at) in &self.grammar_queue {
            let finished = self
                .outputs
                .state(&submission.rid)
                .is_none_or(|state| state.finished());
            let status = grammars.status(key);
            let is_ready = finished || !matches!(status, Lookup::Compiling);
            let since = grammars.started_at(key).unwrap_or(*queued_at);
            ready.push(i64::from(is_ready));
            failed.push(i64::from(!is_ready && now.duration_since(since) >= timeout));
            invalid.push(i64::from(matches!(status, Lookup::Invalid(_))));
            invalid_here.push(match status {
                Lookup::Invalid(invalid) => Some(invalid),
                _ => None,
            });
        }
        // One MIN: ready on every rank; failed or invalid on any.
        local.extend(ready.iter().copied());
        local.extend(failed.iter().map(|&f| -f));
        local.extend(invalid.iter().map(|&v| -v));
        let agreed = self.replica_min(local)?;
        let n = self.grammar_queue.len();
        let (ready, failed, invalid) = (&agreed[..n], &agreed[n..2 * n], &agreed[2 * n..]);
        if ready.iter().all(|&r| r == 0) && failed.iter().all(|&f| f == 0) {
            return Ok(());
        }
        // A grammar unusable on any rank fails its request the same way on
        // every rank: with the first member's reason, as its frontend sees.
        let agreed_invalid = if invalid.contains(&-1) {
            self.agree_invalid_grammars(invalid_here)?
        } else {
            vec![None; n]
        };
        let queue = std::mem::take(&mut self.grammar_queue);
        let mut promoted = Vec::new();
        let mut timed_out: Vec<(Submission, GrammarKey)> = Vec::new();
        let mut waiting = Vec::new();
        for (index, (submission, key, queued_at)) in queue.into_iter().enumerate() {
            if ready[index] == 1 {
                promoted.push((submission, key, agreed_invalid[index].clone()));
            } else if failed[index] == -1 {
                timed_out.push((submission, key));
            } else {
                waiting.push((submission, key, queued_at));
            }
        }
        let mut admitted = Vec::new();
        let mut submissions = Vec::new();
        for (submission, key, agreed) in promoted {
            let finished = self
                .outputs
                .state(&submission.rid)
                .is_none_or(|state| state.finished());
            if finished {
                let reaped = self.outputs.reap_unscheduled(&submission.rid);
                outputs.extend(reaped.outputs);
                released.extend(reaped.released);
                continue;
            }
            let grammars = self.grammars.as_ref().expect("checked above");
            let lookup = match agreed {
                Some(invalid) => Lookup::Invalid(invalid),
                None => grammars.status(&key),
            };
            if let Lookup::Ready(grammar) = &lookup {
                admitted.push((key.clone(), grammar.clone()));
            }
            if self.start_grammar(&submission.rid, &key, lookup, outputs, released)? {
                submissions.push(submission);
            }
        }
        let mut reported = HashSet::new();
        for (submission, key) in timed_out {
            let grammars = self.grammars.as_mut().expect("checked above");
            let status = grammars.status(&key);
            if self.ranks.replica.size() == 1 {
                // It finished right at the deadline: admit it after all.
                if let Lookup::Ready(grammar) = &status {
                    admitted.push((key.clone(), grammar.clone()));
                    if self.start_grammar(
                        &submission.rid,
                        &key,
                        status.clone(),
                        outputs,
                        released,
                    )? {
                        submissions.push(submission);
                    }
                    continue;
                }
            }
            let message = format!(
                "Grammar compilation timed out after {:.1}s",
                grammars.compile_timeout().as_secs_f64()
            );
            let ready_here = !matches!(status, Lookup::Compiling);
            if !ready_here && reported.insert(key.clone()) {
                grammars.record_timeout(&key, &message);
            }
            let failed = self.outputs.fail_unscheduled(
                &submission.rid,
                &format!("{message}: key=('{}', {:?})", key.kind, key.source),
                None,
            );
            outputs.extend(failed.outputs);
            released.extend(failed.released);
        }
        let waiting_keys: HashSet<GrammarKey> =
            waiting.iter().map(|(_, key, _)| key.clone()).collect();
        self.grammar_queue = waiting;
        if let Some(grammars) = self.grammars.as_mut() {
            grammars.retain(&admitted, &waiting_keys);
        }
        self.route(submissions, outputs, released)
    }

    /// Send admitted requests on: EPD ones wait for their embeddings, the
    /// rest reach the scheduler, or are withheld while paused.
    fn route(
        &mut self,
        submissions: Vec<Submission>,
        outputs: &mut Vec<StreamedOutput>,
        released: &mut Vec<String>,
    ) -> Result<(), LoopError> {
        let submissions = if self.transfer.is_some() && self.controls.releasing_or_released() {
            // Registering its transfer now would name the KV memory a
            // release is about to unmap (and a Decode rank registers it
            // under a new transfer session on wake), so the request could
            // never complete: refuse it instead of withholding it.
            for submission in submissions {
                let failed = self.outputs.fail_unscheduled(
                    &submission.rid,
                    "PD engine is releasing or has released its GPU memory; retry after \
                     resume_memory_occupation",
                    None,
                );
                outputs.extend(failed.outputs);
                released.extend(failed.released);
            }
            Vec::new()
        } else {
            submissions
        };
        let (staged, submissions): (Vec<_>, Vec<_>) =
            submissions.into_iter().partition(|submission| {
                self.config.epd_receive_enabled && !submission.embeddings.is_empty()
            });
        if !staged.is_empty() {
            // Its transfer registers only once it reaches the scheduler: a
            // transfer event for a request the scheduler does not know is a
            // failure.
            if let Some(receiver) = self.embeddings.as_mut() {
                for submission in &staged {
                    receiver.stage(&submission.rid, &submission.embeddings);
                }
            }
            self.staged.extend(staged);
        }
        if self.controls.admit_blocked() {
            self.withheld.extend(submissions);
            return Ok(());
        }
        self.submit(submissions)
    }

    /// EPD: admit the staged requests whose embeddings landed on every rank
    /// of the replica, and fail those whose receive failed or timed out on
    /// any. The receives settle while paused too, so a drain's cancelled
    /// requests end and a frozen one's do not time out: a landed request is
    /// withheld then, like any admission.
    fn drain_staged(
        &mut self,
        outputs: &mut Vec<StreamedOutput>,
        released: &mut Vec<String>,
    ) -> Result<(), LoopError> {
        if self
            .embeddings
            .as_ref()
            .is_some_and(|receiver| receiver.awaiting_release())
        {
            let slots = self.data_plane.epd_released()?;
            if let Some(receiver) = self.embeddings.as_mut() {
                receiver.release(&slots);
            }
        }
        if self.staged.is_empty() {
            return Ok(());
        }
        // A later pipeline stage receives nothing: it agrees with whatever
        // the first decides (landed, the MIN's neutral value).
        let codes = match self.embeddings.as_mut() {
            Some(receiver) => receiver.poll(),
            None => vec![2; self.staged.len()],
        };
        if codes.len() != self.staged.len() {
            return Err(LoopError::DataPlane(DataPlaneError(format!(
                "{} EPD receive states for {} staged requests",
                codes.len(),
                self.staged.len()
            ))));
        }
        let codes = self.replica_min(codes)?;
        if let Some(receiver) = self.embeddings.as_mut() {
            let landed = receiver.settle(&codes);
            if !landed.is_empty() {
                self.data_plane.epd_publish(&landed)?;
            }
        }
        let mut admitted = Vec::new();
        let mut leftover = Vec::new();
        for (submission, code) in std::mem::take(&mut self.staged).into_iter().zip(codes) {
            match code {
                2 => {
                    // Aborted while receiving: finish it instead of running it.
                    if self
                        .outputs
                        .state(&submission.rid)
                        .is_some_and(|state| state.finished())
                    {
                        let reaped = self.outputs.reap_unscheduled(&submission.rid);
                        outputs.extend(reaped.outputs);
                        released.extend(reaped.released);
                    } else {
                        admitted.push(submission);
                    }
                }
                0 => {
                    if let (Some(transfer), Some(bootstrap)) =
                        (self.transfer.as_mut(), &submission.bootstrap)
                    {
                        transfer.abort_room(&submission.rid, bootstrap.room);
                    }
                    let failed = self.outputs.fail_unscheduled(
                        &submission.rid,
                        "EPD embedding receive failed or timed out",
                        None,
                    );
                    outputs.extend(failed.outputs);
                    released.extend(failed.released);
                }
                _ => leftover.push(submission),
            }
        }
        self.staged = leftover;
        if self.controls.admit_blocked() {
            self.withheld.extend(admitted);
            return Ok(());
        }
        self.submit(admitted)
    }

    /// Hand admitted requests to the scheduler: a disaggregated request's
    /// transfer starts only now, as it must not finish before the scheduler
    /// knows the request.
    fn submit(&mut self, submissions: Vec<Submission>) -> Result<(), LoopError> {
        if submissions.is_empty() {
            return Ok(());
        }
        let mut specs = Vec::with_capacity(submissions.len());
        let now = std::time::Instant::now();
        for submission in submissions {
            self.outputs.mark_submitted(&submission.rid, now);
            if let (Some(transfer), Some(bootstrap)) =
                (self.transfer.as_mut(), &submission.bootstrap)
            {
                transfer.register(&submission.rid, bootstrap)?;
            }
            specs.push(submission.spec);
        }
        self.scheduler.submit_requests(&specs);
        Ok(())
    }

    /// Sample the L3 tier's metrics: at most once a second, and whenever
    /// storage turns pending or idle. Observations never feed the agreement
    /// or the scheduler.
    fn record_storage_metrics(&mut self) -> Result<(), LoopError> {
        let (Some(l3), Some(metrics)) = (self.l3.as_mut(), self.l3_metrics.as_mut()) else {
            return Ok(());
        };
        if !l3.agreement.metrics_due(self.scheduler.storage_pending()) {
            return Ok(());
        }
        let snapshot = l3.executor.snapshot();
        metrics.record(
            &snapshot,
            &crate::l3::scheduler_stats(self.scheduler.storage_stats()),
        );
        Ok(())
    }

    /// MIN of `values` over the replica: what every mirrored scheduler can
    /// agree on. Attention-DP ranks hold other requests and stay out.
    fn replica_min(&mut self, values: Vec<i64>) -> Result<Vec<i64>, LoopError> {
        if self.ranks.replica.size() == 1 || values.is_empty() {
            return Ok(values);
        }
        Ok(self
            .ranks
            .mesh
            .all_reduce(&self.ranks.replica, &values, Reduce::Min)?)
    }

    /// Per queued grammar request, the first replica member's reason its
    /// grammar is unusable, if any member's is.
    fn agree_invalid_grammars(
        &mut self,
        local: Vec<Option<InvalidGrammar>>,
    ) -> Result<Vec<Option<InvalidGrammar>>, LoopError> {
        if self.ranks.replica.size() == 1 {
            return Ok(local);
        }
        let count = local.len();
        let gathered = self
            .ranks
            .mesh
            .all_gather(&self.ranks.replica, &grammar_cache::encode_invalid(&local))?;
        let mut agreed = vec![None; count];
        for (member, bytes) in gathered.iter().enumerate() {
            let entries = grammar_cache::decode_invalid(bytes, count).ok_or_else(|| {
                LoopError::Control(format!(
                    "replica member {member} sent malformed grammar failures"
                ))
            })?;
            for (slot, entry) in agreed.iter_mut().zip(entries) {
                if slot.is_none() {
                    *slot = entry;
                }
            }
        }
        Ok(agreed)
    }

    /// The same-round gate: standalone cache
    /// flushes, and the weight operation at the queue's head once every
    /// attention-DP rank holds one of the same kind. One blocking weight
    /// operation per round.
    fn gate(&mut self) -> Result<(), LoopError> {
        let flushes = self.controls.take_flushes();
        let head = self.controls.weight_head().and_then(control::gate_code);
        let internal = self.controls.internal_head().map(InternalOp::gate_code);
        let mut want_flush = flushes > 0;
        let mut all_have_head = head.is_some();
        let mut all_have_internal = internal.is_some();
        if let Some(world) = &self.ranks.dp {
            let code = head.unwrap_or(0);
            let internal_code = internal.unwrap_or(0);
            let row = [
                i64::from(want_flush),
                if head.is_some() { -1 } else { 0 },
                code,
                -code,
                if internal.is_some() { -1 } else { 0 },
                internal_code,
                -internal_code,
            ];
            let reduced = self.ranks.mesh.all_reduce(world, &row, Reduce::Max)?;
            want_flush = reduced[0] > 0;
            all_have_head = reduced[1] == -1;
            if all_have_head && reduced[2] != -reduced[3] {
                return Err(LoopError::Control(
                    "attention-DP ranks hold different weight operations at their queue \
                     heads; the frontend must send every weight operation to every DP \
                     engine in the same order"
                        .to_owned(),
                ));
            }
            all_have_internal = reduced[4] == -1;
            if all_have_internal && reduced[5] != -reduced[6] {
                return Err(LoopError::Control(
                    "attention-DP ranks hold different internal operations at their queue \
                     heads; the rebalance controller must queue the same operations in the \
                     same rounds on every rank"
                        .to_owned(),
                ));
            }
        }
        let ready = if all_have_head {
            self.controls.pop_weight_op()
        } else {
            None
        };
        // A weight load's prerequisites are agreed before its flush: a load
        // that no rank may run neither drops the caches nor reaches a device.
        let ready = match ready {
            Some(op) if control::is_weight_load(&op) => match self.refuse_weight_load(&op)? {
                Some(refusal) => {
                    tracing::info!(op = op.tag(), message = refusal, "weight load refused");
                    self.controls
                        .reply(control::gated_reply(&op, false, refusal));
                    None
                }
                None => Some(op),
            },
            ready => ready,
        };
        // One blocking device operation per round: a frontend one first.
        let ready_internal = if all_have_internal && ready.is_none() {
            self.controls.pop_internal()
        } else {
            None
        };
        if let Some(op) = &ready {
            want_flush |= control::wants_flush(op);
        }
        let flushed = if want_flush {
            self.clear_cache()?
        } else {
            true
        };
        for _ in 0..flushes {
            self.controls
                .reply(ControlOutput::FlushCache(FlushCacheReqOutput {
                    rid: None,
                    http_worker_ipc: None,
                    success: flushed,
                }));
        }
        if let Some(op) = ready {
            self.complete_weight_op(op, flushed)?;
        } else if let Some(op) = ready_internal {
            self.complete_internal_op(op)?;
        }
        Ok(())
    }

    /// Whether every rank sharing this decision says yes: MIN over the
    /// replica, then over the attention-DP ranks.
    fn converge(&mut self, local: bool) -> Result<bool, LoopError> {
        let mut agreed = i64::from(local);
        if self.ranks.replica.size() > 1 {
            agreed = self
                .ranks
                .mesh
                .all_reduce(&self.ranks.replica, &[agreed], Reduce::Min)?[0];
        }
        if let Some(world) = &self.ranks.dp {
            agreed = self.ranks.mesh.all_reduce(world, &[agreed], Reduce::Min)?[0];
        }
        Ok(agreed != 0)
    }

    /// Drop the prefix caches on every rank or on none: the ranks agree the
    /// flush can happen before any of them clears.
    fn clear_cache(&mut self) -> Result<bool, LoopError> {
        let can_clear = self.scheduler.can_clear_cache();
        if !self.converge(can_clear)? {
            return Ok(false);
        }
        // The L3 objects go first, in a phase of its own: a store that
        // fails to delete must leave every rank's indexes untouched.
        if let Some(l3) = &self.l3 {
            let deleted = l3.executor.delete_namespace();
            if !self.converge(deleted)? {
                return Ok(false);
            }
        }
        Ok(self.scheduler.clear_cache())
    }

    /// Finish a gated weight operation: its result is agreed across the
    /// ranks before the reply. A weight load reaching here passed the
    /// all-rank admission in [`Self::gate`].
    fn complete_weight_op(&mut self, op: ControlRequest, flushed: bool) -> Result<(), LoopError> {
        let flush_refused = control::wants_flush(&op) && !flushed;
        let (success, message) = match &op {
            ControlRequest::RebalanceExperts(_) => {
                // Not a weight operation: the frontend's trigger of the
                // rebalance, whose snapshot runs in this gated round.
                let (local, message) = self.begin_manual_rebalance()?;
                let agreed = self.converge(local)?;
                let message = if local && !agreed {
                    format!("rebalance refused on another rank in the replica ({message})")
                } else {
                    message
                };
                (agreed, message)
            }
            _ if flush_refused => (
                false,
                "cache flush failed; retry the update after in-flight Host cache ops drain \
                 and no request is suspended with a retraction image (retracted or restoring)"
                    .to_owned(),
            ),
            _ => {
                let (local, message) = self.data_plane.update_weights(&op)?;
                let agreed = self.converge(local)?;
                let message = if local && !agreed {
                    format!("weight update failed on another rank in the replica ({message})")
                } else {
                    message
                };
                if agreed && control::is_weight_load(&op) {
                    // New pages publish under the new checkpoint only once
                    // every rank holds its weights.
                    match self.controls.commit_l3_weight_version(&op) {
                        Err(refusal) => (false, refusal),
                        Ok(Some(version)) => {
                            if let Some(l3) = &self.l3 {
                                l3.executor.set_weight_version(&version)?;
                            }
                            (true, message)
                        }
                        Ok(None) => (true, message),
                    }
                } else {
                    (agreed, message)
                }
            }
        };
        let success = if control::is_weight_load(&op) && !flush_refused {
            // Every rank started this load: a refused admission never gets
            // here, and a refused flush loads and commits nothing. Namespace
            // publication can fail after the load agreement, so no rank
            // commits usability until that final result also converged.
            let agreed = self.converge(success)?;
            self.data_plane.commit_weight_update(agreed)?;
            agreed
        } else {
            success
        };
        tracing::info!(op = op.tag(), success, message, "weight operation done");
        self.controls
            .reply(control::gated_reply(&op, success, message));
        Ok(())
    }

    /// Agree a weight load's device prerequisites before its flush and
    /// before any rank loads. A rank that would refuse must not strand a
    /// ready peer inside a loading collective; a refusal leaves every rank's
    /// weights and caches untouched, so serving goes on and the load may be
    /// retried. Returns the refusal, or `None` when every rank is ready.
    fn refuse_weight_load(&mut self, op: &ControlRequest) -> Result<Option<String>, LoopError> {
        let (ready, message) = self.data_plane.check_weight_update(op)?;
        if self.converge(ready)? {
            return Ok(None);
        }
        Ok(Some(if ready {
            "weight load refused on another rank in the replica".to_owned()
        } else {
            message
        }))
    }

    /// `RebalanceExpertsReqInput`: take the snapshot now, if no rebalance runs.
    fn begin_manual_rebalance(&mut self) -> Result<(bool, String), LoopError> {
        let Some(controller) = self.eplb.as_mut() else {
            return Ok((
                false,
                "expert rebalancing needs the server to start with --enable-eplb".to_owned(),
            ));
        };
        let Some(op) = controller.request_snapshot() else {
            return Ok((
                false,
                format!(
                    "an expert rebalance is already in progress ({})",
                    controller.phase().name()
                ),
            ));
        };
        let delay = controller.commit_delay_forwards();
        self.complete_internal_op(op)?;
        Ok((
            true,
            format!("expert load snapshot taken; the placement commits after {delay} forwards"),
        ))
    }

    /// Run a gate-agreed rebalance step.
    fn complete_internal_op(&mut self, op: InternalOp) -> Result<(), LoopError> {
        let eplb_error = |error: String| LoopError::Control(format!("expert rebalance: {error}"));
        match op {
            InternalOp::Snapshot => {
                let (mut physical, old_map) = self.data_plane.snapshot_expert_load()?;
                let all_to_all = self
                    .eplb
                    .as_ref()
                    .is_some_and(|controller| controller.specs().all_to_all_ep);
                if let Some(ep) = &self.ranks.ep {
                    if all_to_all {
                        // Each rank counted its own tokens: the load is the sum.
                        let flat: Vec<i64> = physical.iter().flatten().copied().collect();
                        let summed = self.ranks.mesh.all_reduce(ep, &flat, Reduce::Sum)?;
                        let width = physical.first().map_or(0, Vec::len);
                        for (row, chunk) in physical.iter_mut().zip(summed.chunks(width.max(1))) {
                            row.copy_from_slice(chunk);
                        }
                    } else {
                        // Every rank routed every token through the same
                        // tables: the counters already are the group's load,
                        // and must agree.
                        let digest = fnv1a(physical.iter().flatten());
                        let reduced =
                            self.ranks
                                .mesh
                                .all_reduce(ep, &[digest, -digest], Reduce::Max)?;
                        if reduced[0] != -reduced[1] {
                            return Err(LoopError::Control(
                                "expert load counters differ across EP ranks under \
                                 replicated-input EP: the ranks routed tokens differently \
                                 within a forward, so the MoE output is wrong; this is a \
                                 routing bug"
                                    .to_owned(),
                            ));
                        }
                    }
                }
                self.eplb_controller()?
                    .on_snapshot(physical, old_map)
                    .map_err(eplb_error)?;
            }
            InternalOp::Commit => {
                let placement = self
                    .eplb_controller()?
                    .placement_for_commit()
                    .map_err(eplb_error)?;
                let placement = match &self.ranks.ep {
                    Some(ep) => {
                        let payload = placement.as_deref().map(encode_rows).unwrap_or_default();
                        decode_rows(&self.ranks.mesh.broadcast(ep, 0, payload)?)
                            .ok_or(LoopError::Relay(RelayError::Truncated))?
                    }
                    None => placement.ok_or_else(|| {
                        LoopError::Control("EP rank 0 derived no placement".to_owned())
                    })?,
                };
                self.eplb_controller()?
                    .on_commit(placement)
                    .map_err(eplb_error)?;
            }
            InternalOp::ApplyChunk(layer_ids) => {
                let (rows, moves) = self
                    .eplb_controller()?
                    .chunk_payload(&layer_ids)
                    .map_err(eplb_error)?;
                self.data_plane
                    .apply_expert_placement(&layer_ids, &rows, &moves)?;
                self.eplb_controller()?
                    .on_chunk_applied(&layer_ids)
                    .map_err(eplb_error)?;
            }
        }
        Ok(())
    }

    fn eplb_controller(&mut self) -> Result<&mut eplb::Controller, LoopError> {
        self.eplb
            .as_mut()
            .ok_or_else(|| LoopError::Control("expert rebalancing is not enabled".to_owned()))
    }

    /// No PD transfer can still read or write the cache.
    fn transfers_idle(&self) -> bool {
        let sizes = self.scheduler.sizes();
        sizes.bootstrapping == 0
            && sizes.pd_transfer == 0
            && self
                .transfer
                .as_ref()
                .is_none_or(|transfer| !transfer.has_in_flight())
    }

    /// Resolve a deferred pause or memory release once the scheduler drained
    /// and nothing it dispatched is in flight.
    fn finish_drain(&mut self) -> Result<(), LoopError> {
        let Some(drain) = self.controls.drain() else {
            return Ok(());
        };
        let sizes = self.scheduler.sizes();
        if sizes.waiting != 0 || sizes.decoding != 0 || sizes.prefilling != 0 {
            return Ok(());
        }
        if !self.in_flight.is_empty() {
            return Ok(());
        }
        // EPD: a request the drain cancelled (or its client aborted) ends
        // toward its client only once its embedding receive settles.
        let settling = self.staged.iter().any(|submission| {
            self.outputs
                .state(&submission.rid)
                .is_some_and(|state| state.finished())
        });
        if settling {
            return Ok(());
        }
        if let Drain::Release(tags) = drain {
            // A gated operation still writes the weights, a transfer still
            // touches the cache, or the cache index still pins pages.
            if self.controls.weights_busy() || !self.transfers_idle() {
                return Ok(());
            }
            if tags.iter().any(|tag| tag == "kv_cache") && !self.scheduler.clear_l1_cache() {
                return Ok(());
            }
        }
        match self.controls.take_drain().expect("checked above") {
            Drain::Pause => self.controls.finish_pause_drain(),
            Drain::Release(tags) => {
                self.data_plane.release_memory(&tags)?;
                tracing::info!(?tags, "memory released");
                self.controls.finish_release(tags);
            }
        }
        Ok(())
    }

    /// Act on a control operation as it arrives.
    fn control(&mut self, request: ControlRequest) -> Result<(), LoopError> {
        // A disaggregated release cannot drain while a pause withholds
        // requests that would still need their transfers, nor a release run
        // while a rebalance's steps would land on unmapped weights.
        let refusal = (self.transfer.is_some() && !self.withheld.is_empty())
            .then(|| {
                "PD requests are buffered by a scheduler pause; resume_scheduler and drain \
                 them before releasing memory"
                    .to_owned()
            })
            .or_else(|| {
                self.eplb
                    .as_ref()
                    .filter(|controller| controller.phase() != eplb::Phase::Idle)
                    .map(|controller| {
                        format!(
                            "an expert rebalance is in progress ({}); retry the release once \
                             it has been applied",
                            controller.phase().name()
                        )
                    })
            });
        let profile = match &request {
            ControlRequest::Profile(profile) => Some(profile.clone()),
            _ => None,
        };
        match self.controls.handle(request, refusal) {
            Effect::None => {}
            Effect::Profile => {
                let profile = profile.expect("a profile effect comes from a profile request");
                let (success, message) = self.data_plane.profile(&profile)?;
                self.controls
                    .reply(ControlOutput::Profile(ProfileReqOutput {
                        rid: None,
                        http_worker_ipc: None,
                        success,
                        message,
                    }));
            }
            Effect::ResumeMemory(tags) => {
                self.data_plane.resume_memory(&tags)?;
                tracing::info!(?tags, "memory resumed");
                self.controls.finish_resume(&tags);
            }
        }
        Ok(())
    }

    /// The round's frontend messages, from the replica's first member, and
    /// its stop request; with L3, every member's storage report, which this
    /// rank's `l3_report` joins.
    fn relay(
        &mut self,
        stop_requested: bool,
        received: Vec<EngineRequest>,
        l3_report: Option<Vec<u8>>,
    ) -> Result<Relayed, LoopError> {
        let replica = &self.ranks.replica;
        if replica.size() == 1 {
            return Ok(Relayed {
                stop: stop_requested,
                messages: received,
                l3_reports: l3_report.into_iter().collect(),
            });
        }
        let owner = self.owns_frontend();
        let payload = if owner {
            relay::encode(stop_requested, &received)
        } else {
            Vec::new()
        };
        let (relayed, l3_reports) = match l3_report {
            None => (self.ranks.mesh.broadcast(replica, 0, payload)?, Vec::new()),
            Some(report) => {
                // Every member's storage report rides the relay: one
                // exchange a round for both.
                let gathered = self
                    .ranks
                    .mesh
                    .all_gather(replica, &relay::join(&payload, &report))?;
                let mut relayed = Vec::new();
                let mut reports = Vec::with_capacity(gathered.len());
                for (member, part) in gathered.iter().enumerate() {
                    let (relay, report) = relay::split(part)?;
                    if member == 0 {
                        relayed = relay.to_vec();
                    }
                    reports.push(report.to_vec());
                }
                (relayed, reports)
            }
        };
        let (stop, messages) = if owner {
            (stop_requested, received)
        } else {
            relay::decode(&relayed)?
        };
        Ok(Relayed {
            stop,
            messages,
            l3_reports,
        })
    }

    /// With attention DP, every rank's batch this round, and whether any
    /// rank is stopping.
    fn dp_round(
        &mut self,
        forward: Option<&ForwardBatchView>,
        stop: &mut bool,
    ) -> Result<Option<DpRound>, LoopError> {
        let Some(world) = &self.ranks.dp else {
            return Ok(None);
        };
        let (num_tokens, batch_size, num_extends) = match forward {
            Some(batch) => {
                let tokens: i64 = batch.input_lengths.iter().map(|&n| i64::from(n)).sum();
                if tokens > 0 {
                    (
                        tokens,
                        batch.request_ids.len() as i64,
                        batch.num_extends as i64,
                    )
                } else {
                    (0, 0, 0)
                }
            }
            None => (0, 0, 0),
        };
        let row = [num_tokens, batch_size, num_extends, i64::from(*stop)];
        let rows = self.ranks.mesh.all_gather_i64(world, &row)?;
        *stop = rows.iter().any(|row| row[3] != 0);
        let any_work = rows.iter().any(|row| row[0] > 0);
        self.dp_idle = !any_work;
        Ok(Some(DpRound {
            num_tokens: rows.iter().map(|row| row[0]).collect(),
            batch_sizes: rows.iter().map(|row| row[1]).collect(),
            num_extends: rows.iter().map(|row| row[2]).collect(),
            idle_forward: num_tokens == 0 && any_work,
        }))
    }

    /// The pipeline's last stage's results of a forward: only it sampled;
    /// the others adopt its tokens, and with them its finish and abort
    /// decisions, and its prompt logprobs.
    fn adopt_last_stage(&mut self, result: ForwardResult) -> Result<ForwardResult, LoopError> {
        let Some(pp) = &self.ranks.pp else {
            return Ok(result);
        };
        let last = pp.size() - 1;
        let payload = if pp.index() == last {
            result.to_bytes()
        } else {
            Vec::new()
        };
        let shared = self.ranks.mesh.broadcast(pp, last, payload)?;
        if pp.index() == last {
            return Ok(result);
        }
        ForwardResult::from_bytes(&shared).ok_or(LoopError::Relay(RelayError::Truncated))
    }

    /// The transfers the replica agreed changed state, acted on: their
    /// scheduler events join the round's feedback.
    fn poll_transfers(
        &mut self,
        events: &mut Vec<ForwardEvent>,
        outputs: &mut Vec<StreamedOutput>,
        released: &mut Vec<String>,
    ) -> Result<(), LoopError> {
        let Some(transfer) = self.transfer.as_mut() else {
            return Ok(());
        };
        let polled = transfer.poll(&mut self.ranks.mesh, &self.ranks.replica)?;
        for event in polled {
            tracing::debug!(?event, "transfer event");
            match event {
                PdEvent::Bootstrapped(rid) => {
                    events.push(pd_event(ForwardEventKind::PdBootstrapped, &rid, -1));
                }
                PdEvent::Succeeded(rid) => {
                    events.push(pd_event(ForwardEventKind::PdSucceeded, &rid, -1));
                    absorb(self.outputs.finish_prefill(&rid), events, outputs, released);
                }
                PdEvent::Failed(rid) => {
                    // The scheduler releases the request on the event itself.
                    events.push(pd_event(ForwardEventKind::PdFailed, &rid, -1));
                    absorb(self.outputs.fail_transfer(&rid), events, outputs, released);
                }
                PdEvent::RemotePrefillDone(done) => {
                    let rid = done.rid.clone();
                    events.push(pd_event(
                        ForwardEventKind::PdRemotePrefillDone,
                        &rid,
                        done.bootstrap_token,
                    ));
                    if self
                        .outputs
                        .state(&rid)
                        .is_some_and(|state| !state.to_abort)
                    {
                        self.outputs.remote_prefill_done(
                            &rid,
                            done.bootstrap_token,
                            done.cached_tokens,
                            done.bootstrap_logprob,
                        );
                    }
                    absorb(
                        self.outputs.finish_remote_prefill_only(&rid),
                        events,
                        outputs,
                        released,
                    );
                    let still_decoding = self
                        .outputs
                        .state(&rid)
                        .is_some_and(|state| !state.to_abort && !state.finished());
                    let candidates = done
                        .spec_candidate_ids
                        .map(|ids| (done.request_pool_index, ids));
                    let slot = still_decoding.then_some(done.request_pool_index);
                    if candidates.is_some() || slot.is_some() {
                        // Both are device writes the request's first decode
                        // reads, so they ride the FIFO ahead of it.
                        self.data_plane.land(candidates, slot)?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Commit dispatched forwards from the oldest until at most `depth`
    /// remain.
    fn commit_down_to(
        &mut self,
        depth: usize,
        events: &mut Vec<ForwardEvent>,
        outputs: &mut Vec<StreamedOutput>,
        released: &mut Vec<String>,
    ) -> Result<(), LoopError> {
        while self.in_flight.len() > depth {
            let (batch, input_logprobs, pending) = self.in_flight.pop_front().expect("non-empty");
            let result = self.data_plane.complete(pending)?;
            let serving_compiles = result.serving_compiles;
            let result = self.adopt_last_stage(result)?;
            let Committed {
                events: committed_events,
                outputs: committed_outputs,
                released: committed_released,
                stats,
            } = self
                .outputs
                .commit(&batch, input_logprobs.as_ref(), &result)?;
            self.record_commit(stats, serving_compiles);
            if batch.num_extends == 0 {
                self.batch_logger.record_decode(&result.output_lengths);
            }
            if let Some(transfer) = self.transfer.as_mut() {
                // P role: the usage a completed prompt's transfer reports.
                for rid in &batch.request_ids {
                    if let Some(state) = self.outputs.state(rid) {
                        transfer.record_usage(
                            rid,
                            state.cached_tokens,
                            state.output_logprobs.first().copied(),
                        );
                    }
                }
            }
            events.extend(committed_events);
            outputs.extend(committed_outputs);
            released.extend(committed_released);
        }
        Ok(())
    }

    /// The batch line of a dispatched forward.
    fn log_dispatch(&mut self, batch: &ForwardBatchView) {
        let sizes = self.scheduler.sizes();
        let active = i64::from(sizes.active_lcm_blocks);
        let stats = RoundStats {
            queued: sizes.waiting,
            active_pages: active,
            cached_pages: self.config.kv_total_pages - i64::from(sizes.empty_lcm_blocks) - active,
            pd: [
                sizes.bootstrapping,
                sizes.prefilling,
                sizes.remote_prefilling,
                sizes.decoding,
                sizes.pd_transfer,
            ],
        };
        let (outputs, scheduler) = (&self.outputs, &self.scheduler);
        self.batch_logger.log_dispatch(
            batch,
            &stats,
            |rid| {
                outputs
                    .state(rid)
                    .map_or(0, |state| state.prompt_len + state.output_ids.len() as i64)
            },
            |group| {
                let pages = scheduler.cache_group_pages(group);
                (pages.total, pages.available)
            },
        );
    }

    /// A commit's counts, into the metrics.
    fn record_commit(&mut self, stats: CommitStats, serving_compiles: Option<(u64, f64)>) {
        let Some(metrics) = self.metrics.as_mut() else {
            return;
        };
        for _ in 0..stats.nan_aborts {
            metrics.record_nan_abort();
        }
        if let Some(width) = self.outputs.spec_num_tokens() {
            metrics.record_spec_decode_step(
                stats.spec_decode_slots,
                stats.spec_accepted_tokens,
                width as i64 - 1,
            );
        }
        if let Some((compiles, seconds)) = serving_compiles {
            metrics.record_jit_compiles(compiles, seconds);
        }
    }

    /// Take a round's messages in arrival order. A control operation acts
    /// after the requests that arrived before it are admitted, so a pause
    /// withholds only what came after it.
    fn admit(
        &mut self,
        received: Vec<EngineRequest>,
        outputs: &mut Vec<StreamedOutput>,
        released: &mut Vec<String>,
    ) -> Result<(), LoopError> {
        let mut admissions = Vec::new();
        for message in received {
            match message {
                EngineRequest::Abort(rids) => {
                    for rid in rids {
                        self.outputs.mark_abort(&rid, false);
                        // On the storage path the abort detaches the request
                        // from its storage work at the next round's head; one
                        // that holds no pages yet finishes there, and no
                        // forward will reap it.
                        let reaped = match self.l3.as_mut() {
                            Some(l3) => l3.agreement.cancel(&self.scheduler, &rid),
                            None => false,
                        };
                        if reaped {
                            let committed = self.outputs.reap_unscheduled(&rid);
                            outputs.extend(committed.outputs);
                            released.extend(committed.released);
                        }
                    }
                }
                EngineRequest::Add(request) => {
                    match ingress::admit(*request, &self.config.limits) {
                        Some(Admitted::Run(admission)) => admissions.push(*admission),
                        Some(Admitted::Refused { rid, state }) => {
                            outputs
                                .extend(self.outputs.publish_finished_at_admission(&rid, *state));
                        }
                        None => {}
                    }
                }
                EngineRequest::Control(request) => {
                    self.register(std::mem::take(&mut admissions), outputs, released)?;
                    self.control(*request)?;
                }
                // The transport turns calls into operations and drops encode
                // requests; a frontend that passes them on has nothing here.
                EngineRequest::Call(_) | EngineRequest::Encode(_) => {
                    tracing::warn!("dropping a message the scheduler does not take")
                }
            }
        }
        self.outputs.sweep_pending_aborts();
        self.register(admissions, outputs, released)
    }

    /// Register admitted requests with the data plane and the output
    /// processor, then hand them to the scheduler, or withhold them while
    /// paused (registered, so an abort still reaches them).
    fn register(
        &mut self,
        admissions: Vec<Admission>,
        outputs: &mut Vec<StreamedOutput>,
        released: &mut Vec<String>,
    ) -> Result<(), LoopError> {
        if admissions.is_empty() {
            return Ok(());
        }
        tracing::debug!(admitted = admissions.len(), "requests admitted");
        // The data plane learns a request before the scheduler can plan it.
        self.data_plane.admit(&admissions)?;
        let mut submissions = Vec::with_capacity(admissions.len());
        for admission in admissions {
            self.outputs
                .register(admission.rid.clone(), admission.state);
            let submission = Submission {
                rid: admission.rid,
                spec: admission.spec,
                bootstrap: admission.bootstrap,
                embeddings: admission.embeddings,
            };
            if let Some(key) = admission.grammar {
                // A cache hit admits at once, unless the replica must first
                // agree it is ready on every rank.
                let grammars = self
                    .grammars
                    .as_mut()
                    .expect("the ingress admits grammars only with a backend");
                let (lookup, hit) = grammars.lookup(&key);
                if !hit || self.ranks.replica.size() > 1 {
                    self.grammar_queue.push((submission, key, Instant::now()));
                    continue;
                }
                if !self.start_grammar(&submission.rid, &key, lookup, outputs, released)? {
                    continue;
                }
            }
            submissions.push(submission);
        }
        self.route(submissions, outputs, released)
    }
}

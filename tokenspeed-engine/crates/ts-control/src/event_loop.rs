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
    io::Read,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

use parking_lot::{lock_api::ArcMutexGuard, Mutex, RawMutex};
use ts_collective::{Group, Mesh, Reduce};
use ts_metrics::{DiskMetrics, EngineMetrics, L3Metrics};
use ts_pd::{EmbeddingItem, EmbeddingReceiver, LandedRequest, PdError, PdEvent, TransferPeer};
use ts_scheduler::{
    profile, CacheEvent, CacheOpKind, ExecutionPlan, ForwardBatchView, ForwardEvent,
    ForwardEventKind, Scheduler, SchedulerSizes, StorageRound, StorageTickView,
};
use ts_wire::{
    control::{FlushCacheReqOutput, ProfileReqOutput},
    ControlOutput, ControlRequest, EngineRequest, StrOrList,
};

use crate::{
    batch_log::{BatchLogConfig, BatchLogger, RoundStats},
    cache_ops::{CacheOpError, CacheOps},
    codec::{Reader, Writer},
    control::{self, ControlConfig, Controls, Drain, Effect},
    disk::{DiskError, DiskTier},
    eplb::{self, InternalOp, SlotMoves},
    grammar::{self as grammar_cache, GrammarKey, Grammars, InvalidGrammar, Lookup},
    ingress::{self, Admission, Admitted, IngressLimits},
    kv_events::KvEventPublisher,
    l3::{L3Error, Tier},
    output::{
        CommitStats, Committed, ForwardResult, HistoryRows, HistorySeeds, InputLogprobPlan,
        LogprobPlan, NgramWindows, OutputProcessor, StreamedOutput,
    },
    peer::{Peer, PeerError},
    relay::{self, RelayError},
    request::RequestGrammar,
    shared_l2::{Pool, PoolClient, PoolError},
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

/// How long the scheduler's active page count is reused before it is taken
/// again.
const ACTIVE_PAGES_REFRESH: Duration = Duration::from_millis(100);

/// The scheduler's active page count, as the load report, the metrics and
/// the batch line read it. Reading it is O(1) (the scheduler's block tables
/// keep the count), and it is taken at most once per
/// [`ACTIVE_PAGES_REFRESH`] -- the frontend considers a load report fresh
/// for seconds -- and whenever the loop runs no request, so an engine that
/// went idle reports none active at once. A count taken while idle is not kept:
/// the first round after requests arrive takes a new one, so an engine that
/// just took a burst does not report none active for the next 100 ms.
#[derive(Debug, Default)]
struct ActivePages {
    /// The last count taken while requests ran, and when.
    taken: Option<(Instant, i32)>,
}

impl ActivePages {
    /// The count at `now`: the last one while it is fresh and requests run,
    /// else a new one from `take`.
    fn read(&mut self, now: Instant, idle: bool, take: impl FnOnce() -> i32) -> i32 {
        match self.taken {
            Some((at, pages)) if !idle && now.duration_since(at) < ACTIVE_PAGES_REFRESH => pages,
            _ => {
                let pages = take();
                self.taken = (!idle).then_some((now, pages));
                pages
            }
        }
    }
}

/// Rooms reserved by admitted PD requests that have not yet reached
/// [`TransferPeer::register`] (same-round admissions, `grammar_queue`, EPD
/// `staged`, `withheld`). Dropping a [`RoomLease`] releases its room
/// automatically on any early failure, cancellation, or handoff.
#[derive(Clone, Default)]
struct PendingRooms(Arc<Mutex<HashSet<u64>>>);

struct RoomLease {
    rooms: Arc<Mutex<HashSet<u64>>>,
    room: u64,
}

impl Drop for RoomLease {
    fn drop(&mut self) {
        self.rooms.lock().remove(&self.room);
    }
}

impl PendingRooms {
    fn try_reserve(&self, room: u64) -> Option<RoomLease> {
        self.0.lock().insert(room).then(|| RoomLease {
            rooms: Arc::clone(&self.0),
            room,
        })
    }
}

/// An admitted request on its way to the scheduler.
struct Submission {
    rid: String,
    spec: ts_scheduler::RequestSpec,
    bootstrap: Option<ts_pd::BootstrapInfo>,
    /// EPD: the items whose embeddings it waits for.
    embeddings: Vec<EmbeddingItem>,
    /// Holds the request's PD bootstrap room until `TransferPeer::register`
    /// takes ownership (or the submission ends before scheduling).
    _room_lease: Option<RoomLease>,
}

/// `control::gate_code` of a forward reload.
const RELOAD_GATE_CODE: i64 = 7;

/// How long a frozen round sleeps, so a `keep` pause does not spin a core.
const PAUSED_IDLE_SLEEP: Duration = Duration::from_millis(1);

/// Without L3, how long a cache clear waits for the Host and Disk writes
/// still in flight -- the Disk tier's operations, peer landings into Host --
/// to retire. Each takes milliseconds; one still in flight past this fails
/// the clear, since none is retained across one
/// (`Scheduler::CanClearCacheRetainingStorage`).
const LOCAL_CLEAR_DRAIN_TIMEOUT: Duration = Duration::from_secs(30);

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

    /// Whether this data plane defers replay across forwards.
    fn replay_retirement_enabled(&self) -> bool;

    /// Latest completed replay fence in host queue order. Never waits.
    fn poll_replay_retirement(&mut self) -> Result<u64, DataPlaneError>;

    /// Forget requests that ended.
    fn release(
        &mut self,
        rids: &[String],
        retirement_id: Option<u64>,
    ) -> Result<(), DataPlaneError>;

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

    /// Development: reload edited forward code and recapture the graphs on a drained, flushed engine; returns success and a report.
    fn reload_forward(&mut self, op: &ControlRequest) -> Result<(bool, String), DataPlaneError>;

    /// Restore these memory regions, in order.
    fn resume_memory(&mut self, tags: &[String]) -> Result<(), DataPlaneError>;

    /// Restore resources that require every rank's memory to be resident.
    fn finish_memory_resume(&mut self) -> Result<(), DataPlaneError>;

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
    /// Completion fence for replay handled before this plan's cache work.
    pub retirement_id: Option<u64>,
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
    /// Prompt plus committed output tokens of every request held by the rank.
    pub num_resident_tokens: i64,
    /// Waiting prompts less observed cache hits, plus unscheduled prefill tokens.
    pub num_waiting_uncached_tokens: i64,
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
    /// The Disk tier refused a plan's operation, or its replica could not
    /// agree on its reports.
    #[error(transparent)]
    Disk(#[from] DiskError),
    /// The shared L2 pool's client or its owner failed.
    #[error(transparent)]
    SharedL2(#[from] PoolError),
    /// The replica could not agree on its peer sharing reports.
    #[error(transparent)]
    Peer(#[from] PeerError),
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
    /// Effective token capacity of the usable KV pool.
    pub kv_token_capacity: i64,
    /// Predecessors Engram reads per token; 0 for a model without Engram.
    pub ngram_context_len: usize,
    /// The model reads each request's committed token history, which the
    /// forwards seed into the history rows.
    pub request_token_history: bool,
    /// Committed tokens behind the frontier the model reads, which bounds
    /// the seeds; None reads the whole prefix.
    pub request_token_history_lookback: Option<usize>,
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
    /// The fused role's prefill-delay gate (`--prefill-attempt-interval` and
    /// `--prefill-any-attempt-interval`); `None` plans prefill every round.
    pub prefill_delay: Option<PrefillDelay>,
}

/// The prefill-delay gate's two vote intervals, in rounds since the last
/// round that scheduled prefill work. Every attention-DP rank runs the same
/// counter and the same vote, so the intervals count the same rounds
/// everywhere. See `docs/design/prefill-delay.md`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefillDelay {
    /// Rounds between ALL attempt points: the gate opens when every rank
    /// that would decode also has prefill work to batch into the round, so
    /// one rank's prefill does not interrupt the others' decodes alone.
    pub attempt_interval: u64,
    /// Rounds between ANY attempt points: one rank's backlog opens the gate,
    /// which bounds how long a prefill can be held. At least
    /// `attempt_interval`, otherwise the ALL points could never be reached
    /// consecutively.
    pub any_interval: u64,
    /// `--prefill-single-round`: one prefill round per window. The round
    /// after a prefill round is not an ALL point while any rank would
    /// decode, so a backlog does not drain in consecutive prefill rounds;
    /// the next ALL point is `attempt_interval` rounds later. An idle engine
    /// still admits at once.
    pub single_round: bool,
}

impl PrefillDelay {
    /// Whether round `rounds` since the last prefill round is an ANY point
    /// and whether it is an ALL point; `decoding` says whether any rank
    /// would decode this round.
    fn attempt_points(&self, rounds: u64, decoding: bool) -> (bool, bool) {
        let any_point = rounds != 0 && rounds % self.any_interval == 0;
        let all_point =
            rounds % self.attempt_interval == 0 && !(self.single_round && rounds == 0 && decoding);
        (any_point, all_point)
    }
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
    /// With the Disk tier (`docs/rfcs/0006-disk-tier.md`), this rank's
    /// executor and file: the replica agrees on what the executors finished
    /// every round. `None` without the tier, which excludes L3.
    pub disk: Option<DiskTier>,
    /// The Disk tier's metrics, on a rank that reports them.
    pub disk_metrics: Option<DiskMetrics>,
    /// With a shared L2 pool as the Host tier, this rank's client of the
    /// pool's owner. `None` without one.
    pub shared_l2: Option<Box<dyn PoolClient + Send>>,
    /// With peer prefix sharing, this rank's part of it: started over the
    /// loop's scheduler, which its endpoint serves between rounds. `None`
    /// without.
    pub peer: Option<Peer>,
}

/// Fences issued through the forward queue, and the prefix every replica
/// member has completed. IDs start at one.
#[derive(Default)]
struct ReplayRetirements {
    issued: u64,
    completed: u64,
}

/// One rank's scheduling loop.
pub struct EventLoop<D: DataPlane, F: Frontend> {
    /// Held for the whole round. A peer holder's endpoint answers from the
    /// scheduler between rounds, so the loop lets go of it only where it
    /// waits on something else: a forward's result, an idle frontend, a
    /// paused round's sleep (when no device copies were just queued), or an
    /// attention-DP collective / idle forward when no forward or device copy
    /// batch is still queued on the device.
    scheduler: ArcMutexGuard<RawMutex, Scheduler>,
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
    /// The Disk tier: this rank's executor and the replica's agreement.
    disk: Option<DiskTier>,
    disk_metrics: Option<DiskMetrics>,
    /// The shared L2 pool: this rank's client and the writes in flight there.
    shared_l2: Option<Pool>,
    peer: Option<Peer>,
    /// The peer clock the replica agreed on this round.
    peer_now_us: i64,
    /// Admissions waiting for their grammars to compile, in arrival order.
    grammar_queue: Vec<(Submission, GrammarKey, Instant)>,
    /// The last cumulative prefix-hit counters reported, for deltas.
    last_prefix_hits: ts_scheduler::PrefixHitView,
    batch_logger: BatchLogger,
    /// The token-history rows' owners, for a model that reads them.
    history: Option<HistoryRows>,
    /// Dispatched forwards, oldest first, with the prompt rows each gathers.
    in_flight: VecDeque<(ForwardBatchView, Option<InputLogprobPlan>, D::Pending)>,
    /// Forwards queued on the device and forwards whose results came back.
    forwards_dispatched: u64,
    forwards_completed: u64,
    /// Set while a dispatched plan's `pages_to_copy` may still be queued on the
    /// device: the `forwards_dispatched` count whose completion proves, by
    /// device order, that the copies ran. A later plan never clears it.
    copies_confirmed_at: Option<u64>,
    cache_ops: CacheOps,
    replay_retirements: Option<ReplayRetirements>,
    controls: Controls,
    /// Accepted cache clear waiting for native completions between rounds.
    cache_clear_started: Option<Instant>,
    /// Warm-up drives its clear directly between serving rounds, without a
    /// queued frontend control request. Keep that demand alive while it waits.
    startup_cache_clear: bool,
    /// Standalone flush callers whose replies wait for that clear.
    deferred_flushes: usize,
    /// Admissions withheld from the scheduler while paused, in arrival order.
    withheld: Vec<Submission>,
    /// EPD: admissions waiting for their images' embeddings, in staging
    /// order (rank-identical, the order the receiver reports them in).
    staged: Vec<Submission>,
    /// PD bootstrap rooms reserved by admitted requests not yet registered
    /// with `transfer`.
    pending_rooms: PendingRooms,
    /// No attention-DP rank had a batch last round, so an idle rank may wait
    /// for requests without holding the others' collectives back.
    dp_idle: bool,
    /// Without L3, the forward permission the scheduler last received; it
    /// starts allowed. With L3 every round's tick carries it.
    forward_allowed: bool,
    /// A collective NCCL release may have discarded projection workspaces and graphs.
    collective_resume_pending: bool,
    /// Last load report on the output wire, including idle heartbeats.
    last_load_report: Option<Instant>,
    /// Last load sent to the frontend, including rounds with no token output.
    reported_load: Option<Load>,
    /// The scheduler's active page count, reused across rounds.
    active_pages: ActivePages,
    /// Rounds since the last round in which any rank scheduled prefill work:
    /// the prefill-delay gate's counter. Reset by an all-idle round too, so
    /// a prompt arriving at an idle engine sees an ALL attempt point.
    rounds_since_prefill: u64,
    /// The last gathered prefill-delay probe, one entry per attention-DP
    /// rank (one local entry without attention DP): whether the rank has
    /// prefill work to schedule, and whether it would decode this round.
    dp_wants_prefill: Vec<bool>,
    dp_has_decode_work: Vec<bool>,
}

/// What a round starts from, the same on every replica member.
struct Relayed {
    /// The replica's first member is stopping.
    stop: bool,
    /// Its frontend messages.
    messages: Vec<EngineRequest>,
    /// With L3 or the Disk tier, every member's storage report, in member
    /// order; empty without.
    storage_reports: Vec<Vec<u8>>,
    /// With peer sharing, every member's peer report, in member order;
    /// empty without.
    peer_reports: Vec<Vec<u8>>,
    /// Most recent replay fence every replica member has completed.
    replay_completed: Option<u64>,
}

impl<D: DataPlane, F: Frontend> EventLoop<D, F> {
    /// A loop over `scheduler`, with `outputs` configured for the engine's
    /// speculation and streaming. `frontend` is used only on the replica's
    /// first member; with a PD transfer peer in `subsystems`, `outputs`
    /// follows its role. A peer in `subsystems` was started over
    /// `scheduler`.
    pub fn new(
        scheduler: Arc<Mutex<Scheduler>>,
        data_plane: D,
        frontend: F,
        outputs: OutputProcessor,
        config: LoopConfig,
        ranks: Ranks,
        subsystems: Subsystems,
    ) -> Self {
        if let Some(delay) = &config.prefill_delay {
            tracing::info!(?delay, "prefill delay gate on");
        }
        let replay_retirements = data_plane
            .replay_retirement_enabled()
            .then(ReplayRetirements::default);
        Self {
            controls: Controls::new(config.controls.clone()),
            batch_logger: BatchLogger::new(config.batch_log.clone()),
            history: config
                .request_token_history
                .then(|| HistoryRows::new(config.request_token_history_lookback)),
            l3: subsystems.l3.map(Tier::new),
            l3_metrics: subsystems.l3_metrics,
            disk: subsystems.disk,
            disk_metrics: subsystems.disk_metrics,
            shared_l2: subsystems.shared_l2.map(Pool::new),
            peer: subsystems.peer,
            peer_now_us: 0,
            scheduler: scheduler.lock_arc(),
            data_plane,
            replay_retirements,
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
            last_prefix_hits: Default::default(),
            in_flight: VecDeque::new(),
            forwards_dispatched: 0,
            forwards_completed: 0,
            copies_confirmed_at: None,
            cache_ops: CacheOps::new(),
            cache_clear_started: None,
            startup_cache_clear: false,
            deferred_flushes: 0,
            withheld: Vec::new(),
            staged: Vec::new(),
            pending_rooms: PendingRooms::default(),
            dp_idle: true,
            forward_allowed: true,
            collective_resume_pending: false,
            last_load_report: None,
            reported_load: None,
            active_pages: ActivePages::default(),
            rounds_since_prefill: 0,
            dp_wants_prefill: Vec::new(),
            dp_has_decode_work: Vec::new(),
        }
    }

    /// Run rounds until a stop is requested on the rank that decides for the
    /// others, or something fails.
    pub fn run(&mut self, stop: &AtomicBool) -> Result<(), LoopError> {
        while !self.round(stop.load(Ordering::Relaxed))? {}
        Ok(())
    }

    /// Serve `phases` of the engine's own requests before a frontend is
    /// attached: each phase on its own, until every rank is idle, then the
    /// prefix caches it filled are dropped. The requests take the serving
    /// path -- admission, planning, the data plane and the commits -- so the
    /// first requests a client sends find that path prepared. Every rank
    /// calls this with the same phases; the replica's first member submits
    /// them, as it does a frontend's requests. Their outputs go to the
    /// frontend, which drops them while none is attached.
    ///
    /// Returns whether a stop was agreed meanwhile; the loop must not
    /// [`Self::run`] then.
    pub fn warm_up(
        &mut self,
        phases: Vec<Vec<EngineRequest>>,
        stop: &AtomicBool,
    ) -> Result<bool, LoopError> {
        for requests in phases {
            let mut submitted = if self.owns_frontend() {
                requests
            } else {
                Vec::new()
            };
            loop {
                let received = std::mem::take(&mut submitted);
                if self.round_with(stop.load(Ordering::Relaxed), received)? {
                    return Ok(true);
                }
                // Background backups need not finish before entering clear:
                // the clear itself owns their bounded drain/isolation policy.
                if self.converge(self.requests_idle() && self.dp_idle)? {
                    break;
                }
            }
            self.startup_cache_clear = true;
            loop {
                self.sync_peer_holder_changes()?;
                if self.clear_cache()? == Some(true) {
                    break;
                }
                if self.round(stop.load(Ordering::Relaxed))? {
                    self.startup_cache_clear = false;
                    return Ok(true);
                }
            }
            self.startup_cache_clear = false;
        }
        Ok(false)
    }

    /// The frontend, for attaching one after [`Self::warm_up`].
    pub fn frontend_mut(&mut self) -> &mut F {
        &mut self.frontend
    }

    fn owns_frontend(&self) -> bool {
        self.ranks.replica.index() == 0
    }

    /// The scheduler's active page count, at most [`ACTIVE_PAGES_REFRESH`]
    /// old (see [`ActivePages`]).
    fn active_pages(&mut self) -> i32 {
        let (scheduler, idle) = (&self.scheduler, self.outputs.num_running() == 0);
        self.active_pages
            .read(Instant::now(), idle, || scheduler.active_lcm_blocks())
    }

    fn idle(&self) -> bool {
        self.cache_clear_started.is_none()
            && self.requests_idle()
            && !self.scheduler.storage_progress_pending()
    }

    /// The model requests and Device copies drained. Storage I/O has its own
    /// lifetime; warm-up must reach clear even if a background PUT is stuck.
    fn requests_idle(&self) -> bool {
        // The scheduler's counts are a pass over its requests per state: ask
        // for them only once the loop's own state says nothing runs.
        if !self.in_flight.is_empty()
            || self.cache_ops.open() != 0
            || self.replay_retirements_pending()
            || self.outputs.num_running() != 0
            || self
                .shared_l2
                .as_ref()
                .is_some_and(|shared_l2| shared_l2.open_writes() != 0)
        {
            return false;
        }
        let sizes = self.scheduler.sizes();
        sizes.waiting == 0 && sizes.prefilling == 0 && sizes.decoding == 0 && sizes.retracted == 0
    }

    fn replay_retirements_pending(&self) -> bool {
        self.replay_retirements
            .as_ref()
            .is_some_and(|fences| fences.issued != fences.completed)
    }

    /// Set the gate while the scheduler mutex still hides the batch's new
    /// publications from the peer servicer.
    fn issue_replay_retirement(&mut self) -> Option<u64> {
        let fences = self.replay_retirements.as_mut()?;
        if fences.issued == fences.completed {
            if let Some(peer) = self.peer.as_ref() {
                peer.block_device_pins(true);
            }
        }
        fences.issued = fences
            .issued
            .checked_add(1)
            .expect("replay retirement ID overflow");
        Some(fences.issued)
    }

    fn poll_replay_retirements(&mut self) -> Result<Option<u64>, LoopError> {
        let Some(fences) = self
            .replay_retirements
            .as_ref()
            .filter(|fences| fences.issued != fences.completed)
        else {
            return Ok(None);
        };
        let local = self.data_plane.poll_replay_retirement()?;
        if local < fences.completed || local > fences.issued {
            return Err(LoopError::Control(format!(
                "replay retirement {local} is outside {}..={}",
                fences.completed, fences.issued
            )));
        }
        Ok(Some(local))
    }

    fn agree_replay_retirements(&mut self, completed: Option<u64>) -> bool {
        let Some((fences, completed)) = self.replay_retirements.as_mut().zip(completed) else {
            return false;
        };
        if completed == fences.completed {
            return false;
        }
        fences.completed = completed;
        if completed == fences.issued {
            if let Some(peer) = self.peer.as_ref() {
                peer.block_device_pins(false);
            }
        }
        true
    }

    /// One round, with this rank's stop request. Returns whether the ranks
    /// agreed to stop after it.
    pub fn round(&mut self, stop_requested: bool) -> Result<bool, LoopError> {
        self.round_with(stop_requested, Vec::new())
    }

    /// A round whose replica-first member also receives `submitted`, the
    /// engine's own requests, ahead of the frontend's.
    fn round_with(
        &mut self,
        stop_requested: bool,
        submitted: Vec<EngineRequest>,
    ) -> Result<bool, LoopError> {
        // Hand off to a waiting peer servicer even in rounds with no device,
        // transfer, collective, or frontend wait. With no waiter this costs
        // one atomic read.
        ArcMutexGuard::bump(&mut self.scheduler);
        // A profile keeps the spans of a round that did something: a loop
        // waiting on a transfer spins through empty rounds.
        let held_spans = profile::hold();
        let round_span = profile::span("EventLoop::round");
        let mut outputs: Vec<StreamedOutput> = Vec::new();
        let mut released: Vec<String> = Vec::new();

        let received = if self.owns_frontend() {
            let wait = (submitted.is_empty() && self.idle() && self.dp_idle).then_some(IDLE_WAIT);
            let _span = profile::span("Frontend::receive");
            let mut received = submitted;
            let frontend = &mut self.frontend;
            received.extend(match wait {
                Some(_) => ArcMutexGuard::unlocked(&mut self.scheduler, || frontend.receive(wait)),
                None => frontend.receive(wait),
            });
            received
        } else {
            Vec::new()
        };
        // Nothing reaches a rank whose frontend is gone: it stops as on a
        // stop request, so every rank stops after this round.
        let stop_requested = stop_requested || self.frontend.closed();
        let storage_report = self
            .l3
            .as_ref()
            .map(Tier::report)
            .or_else(|| self.disk.as_ref().map(DiskTier::report));
        let peer_report = self.peer.as_mut().map(Peer::report);
        let replay_completed = self.poll_replay_retirements()?;
        let Relayed {
            mut stop,
            messages,
            storage_reports,
            peer_reports,
            replay_completed,
        } = self.relay(
            stop_requested,
            received,
            storage_report,
            peer_report,
            replay_completed,
        )?;
        let mut busy = self.agree_replay_retirements(replay_completed) || !messages.is_empty();
        // First at the head: the holder's changes since the last round, as
        // its replica agreed on them, then the landings' progress.
        busy |= self.peer_round(&peer_reports)?;
        self.admit(messages, &mut outputs, &mut released)?;
        self.apply_pause_transitions(&mut outputs, &mut released)?;
        self.drain_grammar_queue(&mut outputs, &mut released)?;
        self.drain_staged(&mut outputs, &mut released)?;
        // A reload past its timeout expires on every replica member in the same round.
        if self.controls.reload_pending() {
            let timed_out = self.controls.reload_timed_out(Instant::now());
            if self.replica_any(timed_out)? {
                self.controls.expire_reload();
            }
        }
        // Every round on every rank, so the attention-DP ranks agree on the
        // flushes and weight operations they run together.
        let (memory_resident, release_pending, nccl_release_pending) = self.gate()?;
        if memory_resident && self.collective_resume_pending {
            self.data_plane.finish_memory_resume()?;
            self.collective_resume_pending = false;
        }
        let forward_allowed = memory_resident
            && !self.controls.forward_blocked()
            && self.cache_clear_started.is_none();

        // The storage tick and the cache-op acknowledgments advance at the
        // head of the round, so this round's plan sees them.
        let storage = self.storage_round(&storage_reports, forward_allowed)?;
        let mut ready = if self.cache_ops.open() > 0 {
            let polled = self.data_plane.poll_cache_events()?;
            self.cache_ops
                .acknowledged(polled, &mut self.ranks.mesh, &self.ranks.replica)?
        } else {
            Vec::new()
        };
        // A shared L2 pool's finished writes are acknowledgments too: polled
        // in every round, frozen or not.
        if let Some(shared_l2) = self.shared_l2.as_mut() {
            ready.extend(shared_l2.poll_writes()?);
        }
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
        if !forward_allowed {
            // Frozen: what is dispatched commits, no forward is planned, and
            // the attention-DP ranks stay in step. With memory released not
            // even an idle forward may run: the weights are unmapped.
            self.commit_down_to(0, &mut events, &mut outputs, &mut released)?;
            // Cache and storage work still progresses: the tick forbade
            // forwards, so the plan carries nothing else.
            let plan = self
                .scheduler
                .next_execution_plan(false, self.config.prefill_delay.is_some());
            if let Some(l3) = &self.l3 {
                busy |= l3.submit(&plan)?;
            }
            if let Some(disk) = &self.disk {
                busy |= disk.submit(&plan)?;
            }
            if let Some(shared_l2) = self.shared_l2.as_mut() {
                busy |= shared_l2.dispatch(&plan, self.in_flight.len())?;
            }
            let cache_ops = plan.cache_ops();
            self.cache_ops.dispatched(&cache_ops);
            let retirement_id = if self.replay_retirements.is_some()
                && cache_ops
                    .iter()
                    .any(|op| op.kind == CacheOpKind::SnapshotStore)
            {
                self.issue_replay_retirement()
            } else {
                None
            };
            self.dispatch_peer(&plan);
            // The storage tick disabled forward planning: scheduler aborts arise
            // only in that build, and frozen rounds reset the no-progress streak.
            self.record_scheduler_metrics(None, &cache_ops, 0);
            let frozen_has_copies = plan.has_pages_to_copy();
            let device_work = plan.needs_device() || retirement_id.is_some();
            busy |= device_work;
            if device_work {
                self.data_plane.execute(
                    plan,
                    &Dispatch {
                        retirement_id,
                        ..Dispatch::default()
                    },
                )?;
            }
            if device_work && frozen_has_copies {
                // No forward of its own: the next one dispatched is behind it.
                self.copies_confirmed_at = Some(self.forwards_dispatched + 1);
            }
            let copies_clear = self.copies_confirmed_at.is_none();
            let mut forwarded = false;
            if let Some(dp) = self.dp_round(None, &mut stop, copies_clear)? {
                if dp.idle_forward && memory_resident {
                    if !copies_clear {
                        self.data_plane.idle_forward(&dp)?;
                    } else {
                        let data_plane = &mut self.data_plane;
                        ArcMutexGuard::unlocked(&mut self.scheduler, || {
                            data_plane.idle_forward(&dp)
                        })?;
                    }
                    forwarded = true;
                }
            }
            busy |= forwarded;
            self.note_forward(forwarded);
            if !copies_clear {
                std::thread::sleep(PAUSED_IDLE_SLEEP);
            } else {
                ArcMutexGuard::unlocked(&mut self.scheduler, || {
                    std::thread::sleep(PAUSED_IDLE_SLEEP);
                });
            }
        } else {
            busy |= self.shared_l2_round()?;
            let allow_prefill = self.prefill_delay_allows_prefill();
            busy |= self.plan_and_dispatch(
                &mut stop,
                &mut events,
                &mut outputs,
                &mut released,
                allow_prefill,
            )?;
            self.poll_transfers(&mut events, &mut outputs, &mut released)?;
        }
        // The forward-result feedback point: everything committed this round
        // reaches the scheduler before the next round plans.
        if !events.is_empty() {
            // An ended request's transfer stops; its pages wait for that
            // (PdEvent::Stopped) unless it succeeds first.
            if let Some(transfer) = self.transfer.as_mut() {
                for event in &events {
                    if event.kind == ForwardEventKind::Abort {
                        transfer.cancel(&event.request_id);
                    }
                }
            }
            self.scheduler
                .advance(&events)
                .map_err(|error| LoopError::Scheduler(error.what().to_owned()))?;
        }
        // One drain per round, after its last scheduler mutation.
        if self.config.kv_events {
            let drained = self.scheduler.drain_kv_events();
            if let Some(publisher) = &self.kv_events {
                publisher.publish(drained);
            }
        }
        // A storage operation's outcome is an acknowledgment too.
        busy |= !ready.is_empty()
            || !storage.done.is_empty()
            || !storage.disk_done.is_empty()
            || !events.is_empty()
            || !outputs.is_empty()
            || !released.is_empty();
        if !released.is_empty() {
            let retirement_id = self.issue_replay_retirement();
            self.data_plane.release(&released, retirement_id)?;
        }
        if self.owns_frontend() {
            let sizes = self.scheduler.sizes();
            let load = Load {
                num_running: self.outputs.num_running() as i64,
                num_waiting: sizes.waiting as i64,
                kv_active_pages: i64::from(self.active_pages()),
                kv_total_pages: self.config.kv_total_pages,
                kv_used_pages: self.config.kv_total_pages - i64::from(sizes.empty_lcm_blocks),
                num_resident_tokens: self.outputs.num_resident_tokens(),
                num_waiting_uncached_tokens: sizes.pending_prefill_tokens,
            };
            // Silent aborts must refresh load immediately; periodic heartbeats
            // also keep an idle engine's load snapshot fresh at its frontend.
            if !outputs.is_empty()
                || self.reported_load != Some(load)
                || self
                    .last_load_report
                    .is_none_or(|last| last.elapsed() >= Duration::from_secs(1))
            {
                let _span = profile::span("Frontend::send");
                self.frontend.send(outputs, load);
                self.reported_load = Some(load);
                self.last_load_report = Some(Instant::now());
            }
        }
        // A deferred pause or release resolves once the scheduler drained.
        self.finish_drain(release_pending, nccl_release_pending)?;
        let replies = self.controls.take_replies();
        busy |= !replies.is_empty();
        for reply in replies {
            self.frontend.reply(reply);
        }
        drop(round_span);
        if busy {
            held_spans.keep();
        }
        Ok(stop)
    }

    /// The round's plan, dispatched, and the forwards older than the
    /// in-flight depth committed. Returns whether the data plane, the L3
    /// executor or a shared L2 pool's client got work.
    fn plan_and_dispatch(
        &mut self,
        stop: &mut bool,
        events: &mut Vec<ForwardEvent>,
        outputs: &mut Vec<StreamedOutput>,
        released: &mut Vec<String>,
        allow_prefill: bool,
    ) -> Result<bool, LoopError> {
        let _span = profile::span("EventLoop::plan_and_dispatch");
        let plan = self
            .scheduler
            .next_execution_plan(allow_prefill, self.config.prefill_delay.is_some());
        let has_device_copies = plan.has_pages_to_copy();
        let sample_ttft = self.outputs.ttft_sampling_enabled();
        let debug = tracing::enabled!(tracing::Level::DEBUG);
        // Reading the batch crosses FFI and copies its arrays.
        if sample_ttft || debug {
            if let Some(batch) = plan.remote_prefill() {
                if sample_ttft {
                    self.outputs.mark_planned(&batch.request_ids, true);
                }
                if debug {
                    tracing::debug!(rids = ?batch.request_ids, "remote prefill planned");
                }
            }
        }
        if debug {
            if let Some(batch) = plan.remote_decode() {
                tracing::debug!(rids = ?batch.request_ids, "remote decode planned");
            }
        }
        // The storage operations start on the L3 executor at once, apart
        // from the device work.
        let storage_work = match (&self.l3, &self.disk) {
            (Some(l3), _) => l3.submit(&plan)?,
            (None, Some(disk)) => disk.submit(&plan)?,
            (None, None) => false,
        };
        // So do a plan's shared L2 ops, on the pool's client; its writes once
        // the forwards in flight landed.
        let shared_l2_work = match self.shared_l2.as_mut() {
            Some(shared_l2) => shared_l2.dispatch(&plan, self.in_flight.len())?,
            None => false,
        };
        // Counted here because `execute` below always runs this plan.
        let cache_ops = plan.cache_ops();
        self.cache_ops.dispatched(&cache_ops);
        self.dispatch_peer(&plan);
        let aborts = plan.aborts();
        let retirement_id = if self.replay_retirements.is_some()
            && (!aborts.is_empty()
                || cache_ops
                    .iter()
                    .any(|op| op.kind == CacheOpKind::SnapshotStore))
        {
            self.issue_replay_retirement()
        } else {
            None
        };
        let aborted = self.outputs.finish_scheduler_aborts(&aborts);
        outputs.extend(aborted.outputs);
        released.extend(aborted.released);
        let mut forward = plan
            .forward_batches()
            .into_iter()
            .next()
            .filter(|batch| !batch.request_ids.is_empty());
        if sample_ttft {
            if let Some(batch) = &forward {
                self.outputs.mark_planned(&batch.request_ids, false);
            }
        }
        // Keep `self.scheduler` locked for coordinated peers and whenever
        // `pages_to_copy` may still be pending on the device; an autonomous
        // single-member peer can safely use the scheduler during the DP
        // collective when no device copy batch is unconfirmed.
        let dp_unlock = self.peer.as_ref().is_some_and(Peer::autonomous)
            && !has_device_copies
            && self.copies_confirmed_at.is_none();
        let dp = self.dp_round(forward.as_ref(), stop, dp_unlock)?;
        if dp.as_ref().is_some_and(|dp| dp.idle_forward) {
            // Padding for a peer is not a local request batch.
            forward = None;
        }
        let planned_sizes = self.record_scheduler_metrics(
            forward.as_ref(),
            &cache_ops,
            aborted.stats.capacity_aborts,
        );

        if dp.as_ref().is_some_and(|dp| dp.idle_forward) {
            // What was dispatched commits first, as no local batch follows.
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
            retirement_id,
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
            self.log_dispatch(batch, planned_sizes.as_ref());
            self.outputs.mark_scheduled(&batch.request_ids);
        }
        if let (Some(metrics), Some(batch)) = (&self.metrics, &forward) {
            // Each request's wait, observed once, at its first dispatch.
            for rid in &batch.request_ids {
                if let Some(at) = self.outputs.take_submitted_at(rid) {
                    metrics.observe_request_queue_time(at.elapsed().as_secs_f64());
                }
                if let Some(at) = self.outputs.take_decode_ready_at(rid) {
                    metrics.observe_decode_ready_wait(at.elapsed().as_secs_f64());
                }
            }
        }
        let has_forward = forward.is_some();
        let mut forwarded = idle_forward;
        let mut queued = false;
        let device_work = plan.needs_device() || idle_forward || retirement_id.is_some();
        if device_work {
            let copies_clear = !has_device_copies && self.copies_confirmed_at.is_none();
            let executed = if copies_clear
                && (idle_forward || self.peer.as_ref().is_some_and(Peer::autonomous))
            {
                let data_plane = &mut self.data_plane;
                ArcMutexGuard::unlocked(&mut self.scheduler, || {
                    data_plane.execute(plan, &dispatch)
                })?
            } else {
                self.data_plane.execute(plan, &dispatch)?
            };
            if let Some(pending) = executed {
                // The batch itself waits for its commit: nothing below reads it.
                let batch = forward.take().expect("a pending forward has a batch");
                let input_logprobs = logprobs.and_then(|plan| plan.input);
                self.in_flight.push_back((batch, input_logprobs, pending));
                self.forwards_dispatched += 1;
                forwarded = true;
                queued = true;
            }
        }
        if device_work && has_device_copies {
            // Its own forward runs behind the copies; without one, the next
            // forward dispatched does.
            self.copies_confirmed_at = Some(self.forwards_dispatched + u64::from(!queued));
        }
        self.note_forward(forwarded);

        // Commit from the head once the queue exceeds the depth; a round with
        // no new work drains fully so results never wait on future traffic.
        let depth = if has_forward {
            self.config.in_flight_depth
        } else {
            0
        };
        self.commit_down_to(depth, events, outputs, released)?;
        Ok(device_work || storage_work || shared_l2_work)
    }

    /// Export a round's snapshot on both serving and paused paths. The
    /// returned scheduler sizes can also be reused by the batch log.
    fn record_scheduler_metrics(
        &mut self,
        forward: Option<&ForwardBatchView>,
        cache_ops: &[ts_scheduler::CacheOpView],
        capacity_aborts: i64,
    ) -> Option<SchedulerSizes> {
        let mut planned_sizes: Option<SchedulerSizes> = None;
        if let Some(metrics) = &self.metrics {
            let _span = profile::span("EventLoop::record_metrics");
            // A retraction victim's image store is the retraction signal.
            let retracted = cache_ops
                .iter()
                .filter(|op| op.kind == CacheOpKind::SnapshotStore)
                .count();
            metrics.record_preemptions(retracted as i64);
            for _ in 0..capacity_aborts {
                metrics.record_capacity_abort();
            }
            // State occupancy is not forward batch size: PrefillDone can
            // wait for PD retirement or a decode permit, and an overlapped
            // round can still hold rows from earlier forwards.
            let sizes = planned_sizes.insert(self.scheduler.sizes());
            let iteration_tokens: i64 = forward.map_or(0, |batch| {
                batch.input_lengths.iter().map(|&n| i64::from(n)).sum()
            });
            let (scheduler, idle) = (&self.scheduler, self.outputs.num_running() == 0);
            let active_pages = self
                .active_pages
                .read(Instant::now(), idle, || scheduler.active_lcm_blocks());
            metrics.record_scheduler_iteration(
                forward.map_or(0, |batch| batch.request_ids.len() as i64),
                self.outputs.num_running() as i64,
                sizes.waiting as i64 + sizes.ready_waiting as i64 + self.withheld.len() as i64,
                i64::from(active_pages),
                self.config.kv_total_pages,
                iteration_tokens,
            );
            metrics.record_decode_slot_usage(
                sizes.allocated_request_slots as i64,
                sizes.active_decode_slots as i64,
                sizes.ready_waiting as i64,
            );
            let total_pages = self.config.kv_total_pages;
            metrics.record_cache_occupancy(
                total_pages - i64::from(sizes.empty_lcm_blocks),
                i64::from(active_pages),
                total_pages,
                self.config.kv_token_capacity,
            );
            match self.outputs.role() {
                crate::output::Role::Prefill => metrics.record_prefill_queues(
                    sizes.bootstrapping as i64,
                    sizes.prefill_inflight as i64,
                    sizes.retracted as i64,
                ),
                crate::output::Role::Decode => metrics.record_decode_queues(
                    sizes.bootstrapping as i64,     // prealloc: waiting for the peer
                    sizes.remote_prefilling as i64, // kv_wait: waiting for its KV
                    sizes.remote_prefilling as i64, // transfer queue: alias of kv_wait
                    sizes.retracted as i64,
                ),
                crate::output::Role::Plain => {}
            }
            metrics.record_generation_tokens(self.outputs.take_generated_tokens());
            let (prompt_tokens, cache_hit_tokens) = self.outputs.take_request_tokens();
            metrics.record_request_tokens(prompt_tokens, cache_hit_tokens);
            let request = self.outputs.take_request_metrics();
            for (rid, lookup, admission, aborted) in request.peer_stops {
                let outcome = if aborted { "cancelled" } else { "no_gain" };
                if let Some(start) = admission {
                    metrics
                        .peer
                        .stage("admission", outcome, start.elapsed().as_secs_f64());
                }
                if let Some(start) = lookup {
                    metrics
                        .peer
                        .stage("lookup_to_ready", outcome, start.elapsed().as_secs_f64());
                }
                if let Some(peer) = &mut self.peer {
                    peer.end_lookup(&rid, outcome);
                }
            }
            for reason in request.peer_local_reasons {
                metrics.peer.decision(&reason);
            }
            for (expected, actual, outcome) in request.peer_usage {
                metrics.peer.usage(expected, actual, outcome);
            }
            for seconds in request.ttft_seconds {
                metrics.observe_time_to_first_token(seconds);
            }
            for seconds in request.request_seconds {
                metrics.observe_request_time(seconds);
            }
            for seconds in request.decode_seconds {
                metrics.observe_decode_time(seconds);
            }
            for seconds in request.tpot_seconds {
                metrics.observe_time_per_output_token(seconds);
            }
            for seconds in request.e2e_tpot_seconds {
                metrics.observe_e2e_time_per_output_token(seconds);
            }
            metrics.record_completion_tokens(request.completion_tokens);
            for count in request.completion_token_samples {
                metrics.observe_request_completion_tokens(count);
            }
            for accept_length in request.spec_decode_accept_length_samples {
                metrics.observe_request_spec_decode_accept_length(accept_length);
            }
            metrics.record_grammar_queue(self.grammar_queue.len() as i64);
            let hits = self.scheduler.prefix_hit_stats();
            let delta = prefix_hit_delta(&mut self.last_prefix_hits, hits);
            if self.outputs.role() == crate::output::Role::Decode {
                metrics.record_decode_prefix_hits(
                    delta.prompt_tokens,
                    delta.device_hit_tokens + delta.host_hit_tokens,
                );
            }
            metrics.record_prefix_hits_with_peer(
                delta.prompt_tokens,
                delta.device_hit_tokens,
                delta.host_hit_tokens,
                delta.l3_hit_tokens,
                delta.peer_hit_tokens,
            );
        }

        planned_sizes
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

    /// With a shared L2 pool, the waiting prompts not looked up yet ask the
    /// pool in one request, and the round waits for its answers: a prompt is
    /// in no plan before its answer, so that a prefix the pool holds is not
    /// recomputed. An answer is no event of the head's batch: the proxy
    /// blocks it creates go back to the pool's client. Returns whether any
    /// prompt was asked about.
    fn shared_l2_round(&mut self) -> Result<bool, LoopError> {
        let Some(shared_l2) = self.shared_l2.as_mut() else {
            return Ok(false);
        };
        let lookups = self.scheduler.start_shared_l2_lookups();
        if lookups.is_empty() {
            return Ok(false);
        }
        let _span = profile::span("EventLoop::shared_l2_round");
        let answers = shared_l2.lookup(&lookups)?;
        let proxy_blocks = self
            .scheduler
            .apply_shared_l2_answers(&answers)
            .map_err(|error| LoopError::Scheduler(error.what().to_owned()))?;
        shared_l2.map_proxy_blocks(&proxy_blocks);
        Ok(true)
    }

    /// The round's storage input to the scheduler. With L3 or the Disk
    /// tier, the replica's agreement on the members' reports, every round.
    /// Without, the forward permission when a freeze starts or ends: the
    /// tick is the scheduler's only input that carries it, and its clock,
    /// which times only storage deadlines, stays 0.
    fn storage_round(
        &mut self,
        storage_reports: &[Vec<u8>],
        allowed: bool,
    ) -> Result<StorageRound, LoopError> {
        if let Some(l3) = self.l3.as_mut() {
            let _span = profile::span("Tier::agree");
            return Ok(l3.agree(storage_reports, allowed, allowed)?);
        }
        if let Some(disk) = self.disk.as_mut() {
            let _span = profile::span("DiskTier::agree");
            return Ok(disk.agree(storage_reports, allowed, allowed)?);
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
                if let Some(transfer) = self.transfer.as_mut() {
                    transfer.cancel(&rid);
                }
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
        self.submit(admitted, outputs, released)
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
        // Only an engine with a grammar backend gives requests grammars.
        self.grammars.as_ref()?;
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
        self.submit(submissions, outputs, released)
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
        self.submit(admitted, outputs, released)
    }

    /// Hand admitted requests to the scheduler: a disaggregated request's
    /// transfer starts only now, as it must not finish before the scheduler
    /// knows the request.
    fn submit(
        &mut self,
        submissions: Vec<Submission>,
        outputs: &mut Vec<StreamedOutput>,
        released: &mut Vec<String>,
    ) -> Result<(), LoopError> {
        if submissions.is_empty() {
            return Ok(());
        }
        let (specs, requests): (Vec<_>, Vec<_>) = submissions
            .into_iter()
            .map(|submission| {
                (
                    submission.spec,
                    (submission.rid, submission.bootstrap, submission._room_lease),
                )
            })
            .unzip();
        if let Err(error) = self.scheduler.submit_requests(&specs) {
            // SubmitRequests validates the whole batch before inserting it.
            // No transfer may start for requests the scheduler rejected.
            let reason = format!("Scheduler submission failed: {}", error.what());
            for (rid, bootstrap, _room_lease) in requests {
                if let (Some(transfer), Some(bootstrap)) = (self.transfer.as_mut(), bootstrap) {
                    transfer.abort_room(&rid, bootstrap.room);
                }
                let failed = self.outputs.fail_unscheduled(&rid, &reason, Some(400));
                outputs.extend(failed.outputs);
                released.extend(failed.released);
            }
            return Ok(());
        }
        let now = std::time::Instant::now();
        for (rid, bootstrap, _room_lease) in requests {
            self.outputs.mark_submitted(&rid, now);
            if let (Some(transfer), Some(bootstrap)) = (self.transfer.as_mut(), &bootstrap) {
                transfer.register(&rid, bootstrap)?;
                // An abort may precede this round's batched submission.
                if self.outputs.state(&rid).is_some_and(|state| state.to_abort) {
                    transfer.cancel(&rid);
                }
            }
        }
        Ok(())
    }

    /// Sample L3 or Disk metrics once a second while stable, plus each
    /// transition between progress-pending and idle. Observations never feed
    /// the agreement or the scheduler.
    fn record_storage_metrics(&mut self) -> Result<(), LoopError> {
        if let (Some(disk), Some(metrics)) = (self.disk.as_mut(), self.disk_metrics.as_mut()) {
            if disk.metrics_due(self.scheduler.storage_progress_pending()) {
                metrics.record(
                    &crate::disk::scheduler_stats(self.scheduler.disk_stats()),
                    &disk.executor_stats(),
                );
            }
            return Ok(());
        }
        let (Some(l3), Some(metrics)) = (self.l3.as_mut(), self.l3_metrics.as_mut()) else {
            return Ok(());
        };
        if !l3
            .agreement
            .metrics_due(self.scheduler.storage_progress_pending())
        {
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
    /// operation per round. Returns whether every rank's memory is resident
    /// and whether any rank has a pending memory release or NCCL release.
    fn gate(&mut self) -> Result<(bool, bool, bool), LoopError> {
        let _span = profile::span("EventLoop::gate");
        let flushes = self
            .deferred_flushes
            .checked_add(self.controls.take_flushes())
            .ok_or_else(|| LoopError::Control("cache flush request count overflow".to_owned()))?;
        self.deferred_flushes = 0;
        let head = self.controls.weight_head().and_then(control::gate_code);
        let internal = self.controls.internal_head().map(InternalOp::gate_code);
        // A reload heads the queue from its arrival; it runs once drained, with the same id, everywhere.
        let reload_key = match self.controls.weight_head() {
            Some(ControlRequest::ReloadForward(op)) => Some(control::reload_key(op)),
            _ => None,
        };
        let reload_ready = reload_key.is_none() || self.controls.reload_ready();
        let mut want_flush = flushes > 0;
        let mut all_have_head = head.is_some() && reload_ready;
        let mut all_have_internal = internal.is_some();
        let mut memory_resident = !self.controls.memory_released();
        let mut release_pending = matches!(self.controls.drain(), Some(Drain::Release(_)));
        let mut nccl_release_pending = matches!(
            self.controls.drain(),
            Some(Drain::Release(tags)) if tags.iter().any(|tag| tag == "nccl")
        );
        if let Some(world) = &self.ranks.dp {
            let code = head.unwrap_or(0);
            let internal_code = internal.unwrap_or(0);
            let key = reload_key.unwrap_or(0);
            let row = [
                i64::from(want_flush),
                if head.is_some() { -1 } else { 0 },
                code,
                -code,
                if internal.is_some() { -1 } else { 0 },
                internal_code,
                -internal_code,
                i64::from(!memory_resident),
                // MAX propagates NCCL intent even while a peer is already asleep.
                if nccl_release_pending {
                    2
                } else {
                    i64::from(release_pending)
                },
                key,
                -key,
                if reload_ready { -1 } else { 0 },
            ];
            let reduced = if self.ranks.replica.size() == 1
                && self.in_flight.is_empty()
                && self.copies_confirmed_at.is_none()
            {
                let reduced = {
                    let mesh = &mut self.ranks.mesh;
                    ArcMutexGuard::unlocked(&mut self.scheduler, || {
                        mesh.all_reduce(world, &row, Reduce::Max)
                    })?
                };
                // Any peer Lookup/Pin/Release answered while waiting on sibling DP
                // ranks in `all_reduce` must be applied to `self.scheduler` before
                // `clear_cache` or `next_execution_plan` runs in this round.
                // Limited to single-member replicas so TP/PP adds no per-round broadcast,
                // and only when no forward is in flight and every dispatched copy batch
                // is behind a forward that completed. The round-head `bump`, the idle
                // receive and `complete()` hand the scheduler over regardless, as on main.
                self.sync_peer_holder_changes()?;
                reduced
            } else {
                self.ranks.mesh.all_reduce(world, &row, Reduce::Max)?
            };
            // A resume reaches DP engines independently. None may forward
            // until every peer can join its device collectives again.
            memory_resident = reduced[7] == 0;
            release_pending = reduced[8] != 0;
            nccl_release_pending = reduced[8] == 2;
            want_flush = reduced[0] > 0;
            all_have_head = reduced[1] == -1;
            let reload_heads = reduced[2] == RELOAD_GATE_CODE || -reduced[3] == RELOAD_GATE_CODE;
            if reload_heads {
                // A reload some rank was not sent, cancelled, or not drained yet waits; it is never fatal.
                all_have_head = all_have_head
                    && reduced[2] == -reduced[3]
                    && reduced[9] == -reduced[10]
                    && reduced[11] == -1;
            } else if all_have_head && reduced[2] != -reduced[3] {
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
        if let Some(op) = &ready {
            want_flush |= control::wants_flush(op);
        }
        if !want_flush && !self.startup_cache_clear {
            // A deferred reload can expire or be cancelled, and a load can
            // lose admission while storage drains. With no remaining flush
            // caller, stop withholding forwards for that abandoned clear.
            self.cache_clear_started = None;
        }
        let flushed = if want_flush {
            match self.clear_cache()? {
                Some(flushed) => flushed,
                None => {
                    // No frontend admission runs inside this gate. Restore the
                    // original head before yielding so weights_busy() and the
                    // next round still see this operation, without cloning its
                    // potentially large serialized weight payload each round.
                    if let Some(op) = ready {
                        self.controls.defer_weight_op(op);
                    }
                    self.deferred_flushes = flushes;
                    return Ok((memory_resident, release_pending, nccl_release_pending));
                }
            }
        } else {
            true
        };
        // One blocking device operation per round: a frontend one first.
        let ready_internal =
            if all_have_internal && ready.is_none() && self.cache_clear_started.is_none() {
                self.controls.pop_internal()
            } else {
                None
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
        Ok((memory_resident, release_pending, nccl_release_pending))
    }

    /// Whether any member of this rank's replica says yes.
    fn replica_any(&mut self, local: bool) -> Result<bool, LoopError> {
        if self.ranks.replica.size() == 1 {
            return Ok(local);
        }
        let reduced =
            self.ranks
                .mesh
                .all_reduce(&self.ranks.replica, &[i64::from(local)], Reduce::Max)?;
        Ok(reduced[0] != 0)
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

    /// One clear attempt: None yields to the normal round so native I/O can
    /// retire. Explicit unsafe access or the drain deadline selects recovery.
    fn clear_cache(&mut self) -> Result<Option<bool>, LoopError> {
        let result = self.advance_cache_clear()?;
        if result.is_some() {
            self.cache_clear_started = None;
        }
        Ok(result)
    }

    fn advance_cache_clear(&mut self) -> Result<Option<bool>, LoopError> {
        let has_storage_owners = self.scheduler.storage_pending();
        let storage_path = !self.converge(!has_storage_owners)?;
        // Ordinary clear keeps the original indexed-pin semantics, including
        // deployments without L3. Storage draining only discounts its own pins;
        // the stronger all-owner check belongs to actual retained recovery.
        let can_clear = !self.replay_retirements_pending()
            && if storage_path {
                self.scheduler.can_prepare_storage_clear()
            } else {
                self.scheduler.can_clear_cache()
            };
        // Refusals log at debug, and only on the refusing rank: warm-up retries
        // its clear every round, and the flush reply already reports failure.
        if !self.converge(can_clear)? {
            if !can_clear {
                tracing::debug!("cache flush refused: non-storage resources remain held");
            }
            return Ok(Some(false));
        }
        let unsafe_current = self.scheduler.storage_blocking_quarantined() > 0
            || self
                .l3
                .as_ref()
                .is_some_and(|l3| l3.executor.current_namespace_needs_isolation());
        // Historical unsafe flags/tickets protect their own addresses. They
        // do not force every later clear into another private namespace.
        if !self.converge(!unsafe_current)? {
            return self.clear_cache_retaining_storage().map(Some);
        }
        if self.cache_clear_started.is_none() {
            if storage_path {
                self.scheduler.cancel_storage_for_clear();
            }
            self.cache_clear_started = Some(Instant::now());
        }
        let pending = self.scheduler.storage_current_pending();
        if !self.converge(!pending)? {
            let budget = match &self.l3 {
                Some(l3) => l3.executor.config().clear_cache_drain_timeout,
                None => LOCAL_CLEAR_DRAIN_TIMEOUT,
            };
            let expired = self.cache_clear_started.expect("clear started").elapsed() >= budget;
            if !self.converge(!expired)? {
                return self.clear_cache_retaining_storage().map(Some);
            }
            return Ok(None);
        }
        // Current-namespace work has drained. Old retained tickets may still
        // occupy slots, but cannot prevent purging this safe active prefix.
        let retaining = self.scheduler.storage_has_retained_operations();
        let can_finish = if retaining {
            self.scheduler.can_clear_cache_retaining_storage()
        } else {
            self.scheduler.can_clear_cache()
        };
        if !self.converge(can_finish)? {
            if !can_finish {
                tracing::debug!("cache clear refused after storage drain: references remain held");
            }
            return Ok(Some(false));
        }
        if let Some(l3) = &self.l3 {
            // A failed backend purge logs its own error.
            let deleted = l3.executor.delete_namespace();
            if !self.converge(deleted)? {
                if !deleted {
                    tracing::debug!("cache flush refused at L3 namespace deletion");
                }
                return Ok(Some(false));
            }
        }
        let cleared = if retaining {
            self.scheduler.clear_cache_retaining_storage()
        } else {
            self.scheduler.clear_cache()
        };
        self.converge(cleared).map(Some)
    }

    /// Isolate old native I/O, then allow L3 in a fresh namespace using only
    /// the remaining Host capacity. Old jobs keep their keys and addresses.
    fn clear_cache_retaining_storage(&mut self) -> Result<bool, LoopError> {
        let can_clear = self.scheduler.can_clear_cache_retaining_storage();
        if !self.converge(can_clear)? {
            if !can_clear {
                tracing::debug!("cache recovery refused: non-storage resources remain held");
            }
            return Ok(false);
        }
        let Some(generation) = self.fresh_cache_generation()? else {
            return Ok(false);
        };
        if let Some(l3) = &self.l3 {
            // Fails only on a malformed or reused agreed nonce or an exhausted
            // epoch counter: invariant violations, not refusals to converge.
            // The scheduler clear below repeats the pure preflight agreed above
            // with no scheduler mutation in between, so it cannot refuse either.
            l3.executor.begin_cache_generation(&generation)?;
        }
        let cleared = self.scheduler.clear_cache_retaining_storage();
        let cleared = self.converge(cleared)?;
        tracing::warn!(
            cleared,
            "cache indexes cleared with storage references retained; L3 uses a fresh generation; remote purge deferred"
        );
        Ok(cleared)
    }

    fn fresh_cache_generation(&mut self) -> Result<Option<String>, LoopError> {
        let mut nonce = [0u8; 16];
        let generated = std::fs::File::open("/dev/urandom")
            .and_then(|mut random| random.read_exact(&mut nonce))
            .is_ok();
        if !self.converge(generated)? {
            if !generated {
                tracing::warn!("cache recovery could not allocate a fresh namespace generation");
            }
            return Ok(None);
        }
        let group = self.ranks.dp.as_ref().unwrap_or(&self.ranks.replica);
        let nonce = self.ranks.mesh.broadcast(group, 0, nonce.to_vec())?;
        let generation: String = nonce.iter().map(|byte| format!("{byte:02x}")).collect();
        Ok(Some(generation))
    }

    /// Finish a gated weight operation: its result is agreed across the
    /// ranks before the reply. A weight load reaching here passed the
    /// all-rank admission in [`Self::gate`].
    fn complete_weight_op(&mut self, op: ControlRequest, flushed: bool) -> Result<(), LoopError> {
        if let ControlRequest::ResumeMemoryOccupation(resume) = &op {
            let tags = resume
                .tags
                .as_deref()
                .expect("queued resume has normalized tags");
            self.data_plane.resume_memory(tags)?;
            tracing::info!(?tags, "memory resumed");
            self.controls.finish_resume(tags);
            return Ok(());
        }
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
            ControlRequest::ReloadForward(_) => self.complete_reload(&op, flushed)?,
            _ if flush_refused => (
                false,
                "cache flush failed; retry the update after in-flight Host cache ops and State retirements drain \
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
                    self.commit_weight_namespace(&op, message)?
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
        if matches!(op, ControlRequest::ReloadForward(_)) {
            self.controls.finish_reload();
        }
        Ok(())
    }

    /// Every member of the agreed load executes the same collective sequence,
    /// even if its local version stays unchanged or namespace publication fails.
    fn commit_weight_namespace(
        &mut self,
        op: &ControlRequest,
        message: String,
    ) -> Result<(bool, String), LoopError> {
        // Model loading is already agreed, but no namespace metadata becomes
        // visible until entropy and every local namespace plan are ready.
        let controls = self.controls.prepare_shared_weight_version(op);
        let isolation = match &controls {
            Err(refusal) => Err(refusal.clone()),
            Ok(prepared) => match (prepared.changed_version(), &self.l3) {
                (Some(version), Some(l3)) => l3
                    .executor
                    .weight_version_needs_isolation(version)
                    .map_err(|error| error.to_string()),
                _ => Ok(false),
            },
        };
        if !self.converge(isolation.is_ok())? {
            return Ok((
                false,
                isolation.err().unwrap_or_else(|| {
                    "weight namespace preparation failed on another rank".to_owned()
                }),
            ));
        }
        // A -> B -> A must not re-enter a prefix touched by retained old A I/O.
        // Unchanged-version ranks participate too; local change is not a gate
        // for collective membership.
        let generation = if !self.converge(!isolation.expect("namespace inspection agreed"))? {
            let Some(generation) = self.fresh_cache_generation()? else {
                return Ok((
                    false,
                    "could not isolate the weight namespace from old I/O".to_owned(),
                ));
            };
            Some(generation)
        } else {
            None
        };
        let controls = controls.expect("control metadata preparation agreed");
        // Allocate the peer's identity before publication too. An unchanged
        // member still applies an agreed nonce if another owner needs it.
        let peer_version = controls.changed_version().map(str::to_owned);
        let storage = match &self.l3 {
            Some(l3) if peer_version.is_some() || generation.is_some() => {
                let version = controls
                    .target_version()
                    .map(str::to_owned)
                    .unwrap_or_else(|| l3.executor.weight_version());
                l3.executor
                    .prepare_weight_version(&version, generation.as_deref())
                    .map(Some)
                    .map_err(|error| error.to_string())
            }
            _ => Ok(None),
        };
        if !self.converge(storage.is_ok())? {
            return Ok((
                false,
                storage.err().unwrap_or_else(|| {
                    "weight namespace preparation failed on another rank".to_owned()
                }),
            ));
        }
        // The exclusive control gate has submitted no new I/O or namespace
        // changes since preparation. Publication now moves checked values;
        // there are no backend/entropy calls or expected failures between the
        // executor, Controls and Peer updates.
        if let Some(prepared) = storage.expect("storage namespace preparation agreed") {
            self.l3
                .as_ref()
                .expect("a storage plan has its executor")
                .executor
                .commit_prepared_weight_version(prepared);
        }
        self.controls
            .commit_prepared_shared_weight_version(controls);
        if let Some(version) = peer_version {
            if let Some(peer) = &mut self.peer {
                peer.set_weight_version(&version);
            }
        }
        Ok((true, message))
    }

    /// Run a drained, gate-agreed forward reload; its result is agreed across the ranks.
    fn complete_reload(
        &mut self,
        op: &ControlRequest,
        flushed: bool,
    ) -> Result<(bool, String), LoopError> {
        if !flushed {
            return Ok((
                false,
                "the prefix caches could not be flushed, so the forward code was not reloaded; \
                 retry once in-flight Host cache operations drain"
                    .to_owned(),
            ));
        }
        let (local, message) = self.data_plane.reload_forward(op)?;
        let agreed = self.converge(local)?;
        Ok(if local && !agreed {
            (
                false,
                format!("forward reload failed on another rank in the replica ({message})"),
            )
        } else {
            (agreed, message)
        })
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

    /// The local scheduler drained, including parked PD transfers and State
    /// retirements, and no forward or cancelled embedding receive can still
    /// finish a request.
    fn drained(&self) -> bool {
        let sizes = self.scheduler.sizes();
        if sizes.waiting != 0
            || sizes.decoding != 0
            || sizes.prefilling != 0
            || sizes.retracted != 0
            || sizes.pd_transfer != 0
        {
            return false;
        }
        if !self.in_flight.is_empty() || self.replay_retirements_pending() {
            return false;
        }
        // EPD: a request the drain cancelled (or its client aborted) ends
        // toward its client only once its embedding receive settles.
        !self.staged.iter().any(|submission| {
            self.outputs
                .state(&submission.rid)
                .is_some_and(|state| state.finished())
        })
    }

    /// Resolve a local pause once drained; release memory only once every
    /// rank has stopped admissions and drained its work; NCCL also needs intent.
    fn finish_drain(
        &mut self,
        release_pending: bool,
        nccl_release_pending: bool,
    ) -> Result<(), LoopError> {
        let drained = self.controls.admit_blocked() && self.drained();
        let nccl_ready = !nccl_release_pending
            || matches!(
                self.controls.drain(),
                Some(Drain::Release(tags)) if tags.iter().any(|tag| tag == "nccl")
            );
        // Every rank joins, even without a local drain. An idle but unpaused
        // peer may admit work next round and still needs our idle forwards.
        let all_paused = release_pending
            && self.converge(
                nccl_ready
                    && drained
                    && !self.controls.weights_busy()
                    && self.transfers_idle()
                    && self.cache_ops.open() == 0,
            )?;
        // Sync any peer Lookup/Pin/Release answered while `self.scheduler` was
        // unlocked earlier in this round before `storage_non_storage_clearable`
        // or `clear_l1_cache` inspects hard pins. Only runs when a Reload or
        // Release drain is actually active, adding zero per-round collectives.
        if self.controls.reload_pending() && self.peer.is_some() && self.replica_any(drained)? {
            self.sync_peer_holder_changes()?;
        }
        let Some(drain) = self.controls.drain() else {
            return Ok(());
        };
        if !drained {
            return Ok(());
        }
        if matches!(drain, Drain::Reload)
            && (!self.transfers_idle() || !self.scheduler.storage_non_storage_clearable())
        {
            // Capture writes KV pages a transfer or a Host write-back could still touch.
            return Ok(());
        }
        if let Drain::Release(tags) = drain {
            if !all_paused {
                return Ok(());
            }
            if tags.iter().any(|tag| tag == "kv_cache") {
                self.sync_peer_holder_changes()?;
                if !self.scheduler.clear_l1_cache() {
                    return Ok(());
                }
            }
        }
        match self.controls.take_drain().expect("checked above") {
            Drain::Pause => self.controls.finish_pause_drain(),
            Drain::Release(tags) => {
                self.data_plane.release_memory(&tags)?;
                self.collective_resume_pending |= tags.iter().any(|tag| tag == "nccl");
                tracing::info!(?tags, "memory released");
                self.controls.finish_release(tags);
            }
            Drain::Reload => {
                tracing::info!("drained for a forward reload");
                self.controls.mark_reload_ready();
            }
        }
        Ok(())
    }

    /// Act on a control operation as it arrives.
    fn control(&mut self, request: ControlRequest) -> Result<(), LoopError> {
        // A shared L2 pool has no weight version: its pages would be read
        // back under weights or a forward that did not compute them. Such an
        // operation is refused here, before it drains or flushes anything.
        if self.shared_l2.is_some()
            && (control::is_weight_load(&request)
                || matches!(request, ControlRequest::ReloadForward(_)))
        {
            let refusal = "a shared L2 pool holds KV of the weights and forward the server \
                           started with; restart the ranks and clear the pool instead"
                .to_owned();
            tracing::info!(
                op = request.tag(),
                message = refusal,
                "refused with a shared L2 pool"
            );
            self.controls
                .reply(control::gated_reply(&request, false, refusal));
            return Ok(());
        }
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
            Effect::DemoteSessions(ids, ended) => {
                let changed = self.scheduler.demote_sessions(&ids, ended);
                self.controls.reply(ControlOutput::DemoteSessions(
                    ts_wire::control::DemoteSessionsReqOutput {
                        rid: None,
                        http_worker_ipc: None,
                        success: true,
                        message: format!("demoted {changed} exclusive cached blocks"),
                    },
                ));
            }
            Effect::DemoteCacheGroups(ids, ended) => {
                let changed = self.scheduler.demote_cache_groups(&ids, ended);
                self.controls.reply(ControlOutput::DemoteCacheGroups(
                    ts_wire::control::DemoteCacheGroupsReqOutput {
                        rid: None,
                        http_worker_ipc: None,
                        success: true,
                        message: format!("demoted {changed} exclusive cached blocks"),
                    },
                ));
            }
            Effect::SharedPrefixPublication(query_id, input_ids, max_prefix_tokens) => {
                let result = self
                    .scheduler
                    .shared_prefix_publication(&input_ids, max_prefix_tokens);
                let (success, message, status, confirmed_prefix_tokens, cache_epoch) = match result
                {
                    Ok(value) => (
                        true,
                        String::new(),
                        value.status,
                        i64::from(value.confirmed_prefix_tokens),
                        value.cache_epoch,
                    ),
                    Err(error) => (false, error.to_string(), "unsupported".to_owned(), 0, 0),
                };
                self.controls
                    .reply(ControlOutput::GetSharedPrefixPublication(
                        ts_wire::control::GetSharedPrefixPublicationReqOutput {
                            rid: None,
                            http_worker_ipc: None,
                            success,
                            message,
                            query_id,
                            dp_rank: -1,
                            status,
                            confirmed_prefix_tokens,
                            storage_namespace: self
                                .l3
                                .as_ref()
                                .map(|tier| tier.executor.key_prefix())
                                .unwrap_or_default(),
                            cache_epoch,
                        },
                    ));
            }
            Effect::AdmissionSnapshot => {
                // Version 1 rounds demand up to 1024-token buckets. Bump the
                // contract version if the curve's bucketing semantics change;
                // see docs/design/cache-concepts.md.
                const ADMISSION_CONTRACT_VERSION: i64 = 1;
                const ADMISSION_TOKEN_STEP: i32 = 1024;
                let snapshot = self.scheduler.admission_snapshot(ADMISSION_TOKEN_STEP);
                self.controls.reply(ControlOutput::GetAdmission(
                    ts_wire::control::GetAdmissionReqOutput {
                        rid: None,
                        http_worker_ipc: None,
                        success: true,
                        contract_version: ADMISSION_CONTRACT_VERSION,
                        dp_rank: -1,
                        free_slots: snapshot.free_slots.into(),
                        total_slots: snapshot.total_slots.into(),
                        available_blocks: snapshot.available_blocks.into(),
                        total_blocks: snapshot.total_blocks.into(),
                        host_free_blocks: snapshot.host_free_blocks.into(),
                        host_total_blocks: snapshot.host_total_blocks.into(),
                        retracted: snapshot.retracted as i64,
                        waiting: snapshot.waiting as i64,
                        max_tokens: snapshot.max_tokens.into(),
                        capacity_token_step: snapshot.capacity_token_step.into(),
                        capacity_blocks: snapshot.capacity_blocks,
                        cache_group_contract: 1,
                        shared_prefix_publication_contract: 1,
                        shared_prefix_storage_enabled: self.l3.is_some()
                            && self.scheduler.shared_prefix_storage_enabled(),
                    },
                ));
            }
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
    /// its stop request; with L3 or the Disk tier, every member's storage
    /// report. Peer reports and outstanding replay fence completion also
    /// ride this exchange.
    fn relay(
        &mut self,
        stop_requested: bool,
        received: Vec<EngineRequest>,
        storage_report: Option<Vec<u8>>,
        peer_report: Option<Vec<u8>>,
        mut replay_completed: Option<u64>,
    ) -> Result<Relayed, LoopError> {
        let replica = &self.ranks.replica;
        if replica.size() == 1 {
            return Ok(Relayed {
                stop: stop_requested,
                messages: received,
                storage_reports: storage_report.into_iter().collect(),
                peer_reports: peer_report.into_iter().collect(),
                replay_completed,
            });
        }
        let _span = profile::span("EventLoop::relay");
        let owner = self.owns_frontend();
        let payload = if owner {
            relay::encode(stop_requested, &received)
        } else {
            Vec::new()
        };
        // Every member's reports ride the relay: one exchange a round for
        // all of them, in the same order on every member.
        let (has_storage, has_peer) = (storage_report.is_some(), peer_report.is_some());
        let reports: Vec<Vec<u8>> = storage_report
            .into_iter()
            .chain(peer_report)
            .chain(replay_completed.map(|id| id.to_le_bytes().to_vec()))
            .collect();
        let (mut storage_reports, mut peer_reports) = (Vec::new(), Vec::new());
        let relayed = if reports.is_empty() {
            self.ranks.mesh.broadcast(replica, 0, payload)?
        } else {
            let gathered = self
                .ranks
                .mesh
                .all_gather(replica, &relay::join(&payload, &reports))?;
            let mut relayed = Vec::new();
            for (member, part) in gathered.iter().enumerate() {
                let (relay, parts) = relay::split(part)?;
                if parts.len() != reports.len() {
                    return Err(RelayError::Truncated.into());
                }
                if member == 0 {
                    relayed = relay.to_vec();
                }
                let mut parts = parts.into_iter();
                if has_storage {
                    storage_reports.extend(parts.next().map(<[u8]>::to_vec));
                }
                if has_peer {
                    peer_reports.extend(parts.next().map(<[u8]>::to_vec));
                }
                if let Some(completed) = replay_completed.as_mut() {
                    let id = parts
                        .next()
                        .and_then(|bytes| bytes.try_into().ok())
                        .map(u64::from_le_bytes)
                        .ok_or(RelayError::Truncated)?;
                    *completed = (*completed).min(id);
                }
            }
            relayed
        };
        let (stop, messages) = if owner {
            (stop_requested, received)
        } else {
            relay::decode(&relayed)?
        };
        Ok(Relayed {
            stop,
            messages,
            storage_reports,
            peer_reports,
            replay_completed,
        })
    }

    /// The round's peer agreement, fed to the scheduler: the holder's
    /// changes and the reader's answers and landings, then the clock its
    /// deadlines run on. Returns whether anything happened.
    fn peer_round(&mut self, reports: &[Vec<u8>]) -> Result<bool, LoopError> {
        let Some(peer) = self.peer.as_mut() else {
            return Ok(false);
        };
        let round = peer.agree_with_scheduler(reports, &mut self.scheduler)?;
        self.peer_now_us = round.now_us;
        if let Some(metrics) = &self.metrics {
            for (rid, result) in round.decisions {
                if let Some(usage) = self.outputs.peer_usage(&rid) {
                    usage.disposition = result;
                    if result == "admission" {
                        usage.admission_at = Some(Instant::now());
                    } else if result == "landing" {
                        usage.admission_at = None;
                    } else if let Some(start) = usage.lookup_at.take() {
                        metrics.peer.stage(
                            "lookup_to_ready",
                            "fallback",
                            start.elapsed().as_secs_f64(),
                        );
                    }
                }
            }
            for (rid, begin, end) in round.landed {
                if let Some(usage) = self.outputs.peer_usage(&rid) {
                    usage.window = Some((begin, end));
                    usage.disposition = "unused_after_landing";
                    if let Some(start) = usage.lookup_at.take() {
                        metrics
                            .peer
                            .stage("lookup_to_ready", "ok", start.elapsed().as_secs_f64());
                    }
                }
            }
        }
        peer.sample_scheduler(&mut self.scheduler, round.now_us);
        self.scheduler
            .advance_peer(round.now_us, &round.events)
            .map_err(|error| LoopError::Scheduler(error.what().to_owned()))?;
        Ok(!round.events.is_empty())
    }

    /// A plan's peer work goes out: its landings start (each read waits for
    /// this plan's device work on its member), and the lookups it queued are
    /// sent.
    fn dispatch_peer(&mut self, plan: &ExecutionPlan) {
        if let Some(peer) = self.peer.as_mut() {
            let fetches = plan.peer_fetches();
            let lookups = self.scheduler.take_peer_lookups();
            if let Some(metrics) = &self.metrics {
                for choice in plan.peer_local_choices() {
                    if let Some(usage) = self.outputs.peer_usage(&choice.request_id) {
                        usage.local_reason = Some(choice.reason);
                    }
                }
                let now = Instant::now();
                for lookup in &lookups {
                    if let Some(usage) = self.outputs.peer_usage(&lookup.request_id) {
                        usage.lookup_at = Some(now);
                        usage.disposition = "lookup";
                    }
                }
                for fetch in &fetches {
                    if let Some(usage) = self.outputs.peer_usage(&fetch.request_id) {
                        if let Some(start) = usage.admission_at.take() {
                            metrics.peer.stage(
                                "admission",
                                "ok",
                                now.duration_since(start).as_secs_f64(),
                            );
                        }
                        usage.disposition = "landing";
                    }
                }
            }
            peer.dispatch(fetches, lookups, self.peer_now_us);
        }
    }

    /// Drain any holder changes answered on `ts-peer-servicer` while
    /// `self.scheduler` was unlocked, share them across the replica so every
    /// TP/PP member applies the exact same sequence, and advance `self.scheduler`.
    fn sync_peer_holder_changes(&mut self) -> Result<(), LoopError> {
        let Some(peer) = &self.peer else {
            return Ok(());
        };
        let changes = if self.ranks.replica.size() == 1 {
            peer.take_holder_changes()
        } else {
            let payload = if self.owns_frontend() {
                Peer::encode_changes(&peer.take_holder_changes())
            } else {
                Vec::new()
            };
            let shared = self.ranks.mesh.broadcast(&self.ranks.replica, 0, payload)?;
            Peer::decode_changes(&shared)?
        };
        if !changes.is_empty() {
            self.scheduler
                .advance_peer(self.peer_now_us, &changes)
                .map_err(|error| LoopError::Scheduler(error.what().to_owned()))?;
        }
        Ok(())
    }

    /// With attention DP, every rank's batch this round, and whether any
    /// rank is stopping. When `allow_unlock` is true and no forward is in
    /// flight, releases `self.scheduler` across the DP `all_gather` so
    /// `ts-peer-servicer` can answer while waiting on sibling DP ranks.
    fn dp_round(
        &mut self,
        forward: Option<&ForwardBatchView>,
        stop: &mut bool,
        allow_unlock: bool,
    ) -> Result<Option<DpRound>, LoopError> {
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
        // What this rank publishes about the delay gate: whether it has
        // prefill work the next round could take, and whether it would decode
        // instead. Gathered with the batch counts so every rank votes on the
        // same observations without a second collective. An engine without
        // the gate publishes zeros and never pays the probe's scan.
        let (wants_prefill, has_decode_work) = if self.config.prefill_delay.is_some() {
            let probe = self.scheduler.probe_prefill_gate();
            (probe.wants_prefill, probe.has_decode_work)
        } else {
            (false, false)
        };
        let Some(world) = &self.ranks.dp else {
            self.dp_wants_prefill = vec![wants_prefill];
            self.dp_has_decode_work = vec![has_decode_work];
            self.note_prefill_round(num_extends, batch_size);
            return Ok(None);
        };
        let _span = profile::span("EventLoop::dp_round");
        let row = [
            num_tokens,
            batch_size,
            num_extends,
            i64::from(*stop),
            i64::from(wants_prefill),
            i64::from(has_decode_work),
        ];
        let rows = if allow_unlock
            && (self.in_flight.is_empty() || self.peer.as_ref().is_some_and(Peer::autonomous))
        {
            let mesh = &mut self.ranks.mesh;
            ArcMutexGuard::unlocked(&mut self.scheduler, || mesh.all_gather_i64(world, &row))?
        } else {
            self.ranks.mesh.all_gather_i64(world, &row)?
        };
        *stop = rows.iter().any(|row| row[3] != 0);
        let any_work = rows.iter().any(|row| row[0] > 0);
        self.dp_idle = !any_work;
        self.dp_wants_prefill = rows.iter().map(|row| row[4] != 0).collect();
        self.dp_has_decode_work = rows.iter().map(|row| row[5] != 0).collect();
        // The counter is rank-identical: it reads only the gathered rows.
        self.rounds_since_prefill =
            if rows.iter().any(|row| row[2] > 0) || !rows.iter().any(|row| row[1] > 0) {
                0
            } else {
                self.rounds_since_prefill.saturating_add(1)
            };
        Ok(Some(DpRound {
            num_tokens: rows.iter().map(|row| row[0]).collect(),
            batch_sizes: rows.iter().map(|row| row[1]).collect(),
            num_extends: rows.iter().map(|row| row[2]).collect(),
            idle_forward: num_tokens == 0 && any_work,
        }))
    }

    /// Without attention DP the counter advances on this rank's own plan.
    fn note_prefill_round(&mut self, num_extends: i64, batch_size: i64) {
        self.rounds_since_prefill = if num_extends > 0 || batch_size == 0 {
            0
        } else {
            self.rounds_since_prefill.saturating_add(1)
        };
    }

    /// Whether this round may schedule prefill work: the control plane's half
    /// of the prefill-delay gate (`docs/design/prefill-delay.md`).
    ///
    /// Every attention-DP rank takes this decision from the same gathered
    /// probes and the same counter, so the whole group opens its gate on the
    /// same round: the prefills of all ranks batch into one window instead of
    /// each rank interrupting the group's decode rounds on its own schedule.
    ///
    /// This is the group's permission. Whether *this* rank plans prefill is
    /// [`Self::prefill_delay_allows_prefill`].
    fn prefill_allowed(&self) -> bool {
        let Some(delay) = self.config.prefill_delay.as_ref() else {
            return true;
        };
        if self.dp_wants_prefill.is_empty() {
            // Nothing gathered yet: the loop's first round plans what it
            // would have planned without the gate.
            return true;
        }
        let rounds = self.rounds_since_prefill;
        let (any_point, all_point) =
            delay.attempt_points(rounds, self.dp_has_decode_work.iter().any(|d| *d));
        if !any_point && !all_point {
            return false;
        }
        let open = if any_point {
            // Anti-starvation: one rank's backlog opens the group's gate, so
            // eligible prefill waits at most `any_interval` rounds. New
            // prompts above the watermark must also wait for capacity.
            self.dp_wants_prefill.iter().any(|want| *want)
        } else {
            // ALL: a rank with nothing to decode protects nothing, so only a
            // rank that would decode is a blocker. (A strict conjunction over
            // `want` would hold a prompt in an idle engine for a whole
            // `any_interval`, and would stall every rank at the tail of a
            // batch whose other ranks already finished.)
            self.dp_wants_prefill
                .iter()
                .zip(&self.dp_has_decode_work)
                .all(|(want, decodes)| *want || !*decodes)
        };
        tracing::debug!(
            rounds,
            any_point,
            all_point,
            open,
            wants = ?self.dp_wants_prefill,
            decodes = ?self.dp_has_decode_work,
            "prefill gate"
        );
        open
    }

    /// Whether this rank plans prefill this round: the group's permission
    /// and this rank's own probe, read now rather than from the last round's
    /// publication.
    ///
    /// A window its peers open must not drag a rank into an admission it
    /// does not want: a resident chunk, or a new prompt at or below the
    /// watermark, is the same condition the vote publishes. A resident
    /// chunk must be able to consume its own reservation above the watermark;
    /// the scheduler independently checks pressure before any new admission
    /// in that window, including after the resident's final chunk.
    fn prefill_delay_allows_prefill(&self) -> bool {
        if self.config.prefill_delay.is_none() {
            return true;
        }
        self.prefill_allowed() && self.scheduler.probe_prefill_gate().wants_prefill
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
        let _span = profile::span("EventLoop::poll_transfers");
        // Decode may be waiting for KV from a Prefill that needs a lookup
        // from this same Decode. Only the transport participates in this poll.
        let polled = ArcMutexGuard::unlocked(&mut self.scheduler, || {
            transfer.poll(&mut self.ranks.mesh, &self.ranks.replica)
        })?;
        for event in polled {
            tracing::debug!(?event, "transfer event");
            match event {
                PdEvent::Bootstrapped(rid) => {
                    self.outputs.bootstrapped(&rid);
                    events.push(pd_event(ForwardEventKind::PdBootstrapped, &rid, -1));
                }
                PdEvent::DestinationReady(rid) => {
                    events.push(pd_event(ForwardEventKind::PdDestinationReady, &rid, -1));
                }
                PdEvent::Succeeded(rid) => {
                    events.push(pd_event(ForwardEventKind::PdSucceeded, &rid, -1));
                    absorb(self.outputs.finish_prefill(&rid), events, outputs, released);
                }
                PdEvent::Failed(rid) => {
                    // The scheduler ends the request on the event; its pages
                    // wait for the transfer to stop.
                    events.push(pd_event(ForwardEventKind::PdFailed, &rid, -1));
                    absorb(self.outputs.fail_transfer(&rid), events, outputs, released);
                }
                PdEvent::Stopped(rid) => {
                    events.push(pd_event(ForwardEventKind::PdTransferStopped, &rid, -1));
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
                            done.kv_ready_at,
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
            let _span = profile::span("EventLoop::commit");
            let (batch, input_logprobs, pending) = self.in_flight.pop_front().expect("non-empty");
            let data_plane = &mut self.data_plane;
            let result =
                ArcMutexGuard::unlocked(&mut self.scheduler, || data_plane.complete(pending))?;
            self.forwards_completed += 1;
            if self
                .copies_confirmed_at
                .is_some_and(|at| self.forwards_completed >= at)
            {
                self.copies_confirmed_at = None;
            }
            if let Some(shared_l2) = self.shared_l2.as_mut() {
                shared_l2.forward_landed()?;
            }
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

    /// The batch line of a dispatched forward, from the scheduler's counts
    /// as the round planned (`planned_sizes`, if the round took them).
    fn log_dispatch(&mut self, batch: &ForwardBatchView, planned_sizes: Option<&SchedulerSizes>) {
        let (outputs, scheduler) = (&self.outputs, &self.scheduler);
        let idle = outputs.num_running() == 0;
        let active_pages = &mut self.active_pages;
        let kv_total_pages = self.config.kv_total_pages;
        self.batch_logger.log_dispatch(
            batch,
            || {
                let sizes = planned_sizes.cloned().unwrap_or_else(|| scheduler.sizes());
                let active = i64::from(
                    active_pages.read(Instant::now(), idle, || scheduler.active_lcm_blocks()),
                );
                // The active count may be a moment older than the empty one.
                let resident = kv_total_pages - i64::from(sizes.empty_lcm_blocks);
                RoundStats {
                    queued: sizes.waiting,
                    allocated_request_slots: sizes.allocated_request_slots,
                    active_decode_slots: sizes.active_decode_slots,
                    ready_waiting: sizes.ready_waiting,
                    active_pages: active,
                    cached_pages: (resident - active).max(0),
                    pd: [
                        sizes.bootstrapping,
                        sizes.prefilling,
                        sizes.remote_prefilling,
                        sizes.decoding,
                        sizes.pd_transfer,
                    ],
                }
            },
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

    /// Reserve `bootstrap.room` while a disaggregated request moves from
    /// admission to `TransferPeer::register`, refusing any room already held
    /// by a pending or tracked (including draining) request.
    fn reserve_bootstrap_room(
        &self,
        bootstrap: Option<&ts_pd::BootstrapInfo>,
    ) -> Result<Option<RoomLease>, u64> {
        let (Some(transfer), Some(bootstrap)) = (self.transfer.as_ref(), bootstrap) else {
            return Ok(None);
        };
        if transfer.has_room(bootstrap.room) {
            return Err(bootstrap.room);
        }
        self.pending_rooms
            .try_reserve(bootstrap.room)
            .map(Some)
            .ok_or(bootstrap.room)
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
        let _span = profile::span("EventLoop::admit");
        let mut admissions = Vec::new();
        let mut room_leases = Vec::new();
        let mut seen = HashSet::new();
        for message in received {
            match message {
                EngineRequest::Abort(rids) => {
                    for rid in rids {
                        self.outputs.mark_abort(&rid, false);
                        if let Some(transfer) = self.transfer.as_mut() {
                            transfer.cancel(&rid);
                        }
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
                    if let Some(StrOrList::One(rid)) = &request.rid {
                        // The wire routes outputs by rid alone: an error for a
                        // duplicate would terminate the original stream too.
                        // Check before ingress, which can itself emit errors.
                        if !seen.insert(rid.clone()) || self.outputs.state(rid).is_some() {
                            tracing::warn!(
                                rid,
                                "dropping duplicate request id; original request retained"
                            );
                            continue;
                        }
                    }
                    match ingress::admit(*request, &self.config.limits) {
                        Some(Admitted::Run(mut admission)) => {
                            if self.scheduler.request_token_size(&admission.rid) >= 0 {
                                // Final output precedes scheduler retirement.
                                // No old output state remains, so this refusal
                                // can safely answer the new request.
                                admission.state.abort(
                                    "Request id is still retiring; retry with a new request id",
                                    true,
                                    Some(409),
                                );
                                outputs.extend(self.outputs.publish_finished_at_admission(
                                    &admission.rid,
                                    admission.state,
                                ));
                            } else {
                                match self.reserve_bootstrap_room(admission.bootstrap.as_ref()) {
                                    Ok(room_lease) => {
                                        if self.owns_frontend() {
                                            self.outputs.sample_ttft(
                                                &mut admission.state,
                                                admission.bootstrap.as_ref(),
                                            );
                                        }
                                        admissions.push(*admission);
                                        room_leases.push(room_lease);
                                    }
                                    Err(room) => {
                                        tracing::warn!(
                                            rid = admission.rid,
                                            room,
                                            "refusing request with an active bootstrap room"
                                        );
                                        admission.state.abort(
                                            &format!(
                                                "Bootstrap room {room} is already active; retry \
                                                 with a new bootstrap_room"
                                            ),
                                            true,
                                            Some(409),
                                        );
                                        outputs.extend(self.outputs.publish_finished_at_admission(
                                            &admission.rid,
                                            admission.state,
                                        ));
                                    }
                                }
                            }
                        }
                        Some(Admitted::Refused { rid, state }) => {
                            outputs
                                .extend(self.outputs.publish_finished_at_admission(&rid, *state));
                        }
                        None => {}
                    }
                }
                EngineRequest::Control(request) => {
                    self.register(
                        std::mem::take(&mut admissions),
                        std::mem::take(&mut room_leases),
                        outputs,
                        released,
                    )?;
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
        self.register(admissions, room_leases, outputs, released)
    }

    /// Register admitted requests with the data plane and the output
    /// processor, then hand them to the scheduler, or withhold them while
    /// paused (registered, so an abort still reaches them).
    fn register(
        &mut self,
        mut admissions: Vec<Admission>,
        room_leases: Vec<Option<RoomLease>>,
        outputs: &mut Vec<StreamedOutput>,
        released: &mut Vec<String>,
    ) -> Result<(), LoopError> {
        if admissions.is_empty() {
            return Ok(());
        }
        let _span = profile::span("EventLoop::register");
        tracing::debug!(admitted = admissions.len(), "requests admitted");
        for admission in &mut admissions {
            self.outputs.assign_admission(&mut admission.state);
            admission.spec.admission_id = admission.state.admission;
        }
        // The data plane learns a request before the scheduler can plan it.
        self.data_plane.admit(&admissions)?;
        let mut submissions = Vec::with_capacity(admissions.len());
        for (admission, room_lease) in admissions.into_iter().zip(room_leases) {
            self.outputs
                .register(admission.rid.clone(), admission.state);
            let submission = Submission {
                rid: admission.rid,
                spec: admission.spec,
                bootstrap: admission.bootstrap,
                embeddings: admission.embeddings,
                _room_lease: room_lease,
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

fn prefix_hit_delta(
    previous: &mut ts_scheduler::PrefixHitView,
    hits: ts_scheduler::PrefixHitView,
) -> ts_scheduler::PrefixHitView {
    let last = std::mem::replace(previous, hits);
    ts_scheduler::PrefixHitView {
        prompt_tokens: hits.prompt_tokens.saturating_sub(last.prompt_tokens),
        peer_hit_tokens: hits.peer_hit_tokens.saturating_sub(last.peer_hit_tokens),
        device_hit_tokens: hits
            .device_hit_tokens
            .saturating_sub(last.device_hit_tokens),
        host_hit_tokens: hits.host_hit_tokens.saturating_sub(last.host_hit_tokens),
        l3_hit_tokens: hits.l3_hit_tokens.saturating_sub(last.l3_hit_tokens),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_prefix_sample_includes_admissions_and_unchanged_counters_add_nothing() {
        let mut previous = ts_scheduler::PrefixHitView::default();
        let hits = ts_scheduler::PrefixHitView {
            prompt_tokens: 48,
            device_hit_tokens: 16,
            host_hit_tokens: 8,
            peer_hit_tokens: 12,
            l3_hit_tokens: 4,
        };
        assert_eq!(prefix_hit_delta(&mut previous, hits), hits);
        assert_eq!(prefix_hit_delta(&mut previous, hits), Default::default());
    }

    #[test]
    fn the_active_page_count_is_reused_while_fresh_and_requests_run() {
        let mut active = ActivePages::default();
        let start = Instant::now();
        let mut walks = 0;
        let mut read = |active: &mut ActivePages, after: Duration, idle: bool, live: i32| {
            active.read(start + after, idle, || {
                walks += 1;
                live
            })
        };
        assert_eq!(read(&mut active, Duration::ZERO, false, 7), 7);
        // Fresh: the last count, though the scheduler's changed.
        assert_eq!(read(&mut active, Duration::from_millis(99), false, 9), 7);
        // Stale: taken again.
        assert_eq!(read(&mut active, ACTIVE_PAGES_REFRESH, false, 9), 9);
        // No request runs: taken at once, so an idle engine reports none.
        assert_eq!(
            read(
                &mut active,
                ACTIVE_PAGES_REFRESH + Duration::from_millis(1),
                true,
                0
            ),
            0
        );
        // Requests arrive: the idle count is not reused, though it is fresh.
        assert_eq!(
            read(
                &mut active,
                ACTIVE_PAGES_REFRESH + Duration::from_millis(2),
                false,
                5
            ),
            5
        );
        assert_eq!(walks, 4);
    }

    #[test]
    fn single_round_windows_close_the_round_after_a_prefill_while_ranks_decode() {
        let delay = |single_round| PrefillDelay {
            attempt_interval: 3,
            any_interval: 6,
            single_round,
        };
        let all_points = |single_round, decoding| -> Vec<u64> {
            (0..7)
                .filter(|&r| delay(single_round).attempt_points(r, decoding).1)
                .collect()
        };
        // Draining: the round after a prefill round is an ALL point.
        assert_eq!(all_points(false, true), [0, 3, 6]);
        // One round per window while anything decodes.
        assert_eq!(all_points(true, true), [3, 6]);
        // An idle engine admits at once.
        assert_eq!(all_points(true, false), [0, 3, 6]);
        // ANY points are unchanged.
        let any_points: Vec<u64> = (0..13)
            .filter(|&r| delay(true).attempt_points(r, true).0)
            .collect();
        assert_eq!(any_points, [6, 12]);
    }
}

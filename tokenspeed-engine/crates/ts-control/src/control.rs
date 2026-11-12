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

//! The control operations' state on the scheduling loop: pause and resume,
//! memory release and resume, and the queue of weight operations that complete
//! through the same-round gate.
//!
//! Every replica member runs the same operations in the same rounds (the
//! relay carries them), so the state here is rank-identical; only the
//! frontend's owner sends the replies.
//!
//! Pause modes: `abort` aborts what runs, `wait` lets it finish; both stop
//! admissions and answer once the scheduler drained. `keep` freezes
//! everything and answers at once. A memory release drains like `wait`,
//! then frees its regions and freezes until they are resumed; it is refused
//! under a `keep` pause, as a pause is under a release.
//! A forward reload (`--allow-reload-forward`) drains, flushes and runs at the gate, then resumes unless a pause preceded it.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use ts_wire::{
    control::{
        DestroyWeightsUpdateGroupReqOutput, InitWeightsUpdateGroupReqOutput,
        IsSchedulerPausedReqOutput, IsSleepingReqOutput, PauseSchedulerReqOutput,
        RebalanceExpertsReqOutput, ReleaseMemoryOccupationReqOutput, ReloadForwardReqInput,
        ReloadForwardReqOutput, ResumeMemoryOccupationReqOutput, ResumeSchedulerReqOutput,
        UpdateWeightFromDiskReqOutput, UpdateWeightsFromDistributedReqOutput,
        UpdateWeightsFromMooncakeReqOutput,
    },
    ControlOutput, ControlRequest,
};

/// Every memory region, in release order: the KV cache first (a PD engine
/// deregisters it from the transfer engine before it is unmapped), NCCL last
/// (nothing may issue a collective after it is suspended). Resume runs the
/// reverse order.
pub const MEMORY_REGIONS: [&str; 4] = ["kv_cache", "weights", "cuda_graph", "nccl"];

/// How far a pause reaches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PauseState {
    /// Normal operation.
    Unpaused,
    /// No new request is admitted; running ones keep stepping.
    PausedNew,
    /// Nothing is scheduled.
    PausedAll,
}

/// What a drain finishes with.
#[derive(Debug, Clone, PartialEq)]
pub enum Drain {
    /// An `abort` or `wait` pause: reply.
    Pause,
    /// A memory release: free these regions, freeze and reply.
    Release(Vec<String>),
    /// A forward reload, queued on the gate since it arrived: mark it ready.
    Reload,
}

/// Settings of the control operations.
#[derive(Debug, Clone)]
pub struct ControlConfig {
    /// `--enable-memory-saver`: memory can be released.
    pub memory_saver: bool,
    /// The regions this engine can release (`nccl` needs an NCCL that can
    /// suspend its communicators).
    pub memory_regions: Vec<String>,
    /// With the L3 storage tier, the checkpoint identity its pages publish
    /// under (`--weight-version`); `None` without the tier.
    pub l3_weight_version: Option<String>,
    /// `--allow-reload-forward`: `reload_forward` is served; refused on arrival otherwise.
    pub allow_reload_forward: bool,
}

/// `L3_FLUSH_REQUIRES_WEIGHT_VERSION`.
const L3_FLUSH_REQUIRES_WEIGHT_VERSION: &str = "L3 flushed updates require weight_version so \
     independent replicas cannot restore another checkpoint's objects";

/// The L3 namespace a weight load asks to publish: its explicit
/// `weight_version`; a flushed Mooncake load without one takes the committed
/// version's own identity (`mooncake_load_weight_version`).
fn requested_weight_version(op: &ControlRequest) -> Option<String> {
    match op {
        ControlRequest::UpdateWeightsFromMooncake(op) => match &op.weight_version {
            Some(version) => Some(version.clone()),
            None if op.flush_cache => Some(op.version.to_string()),
            None => None,
        },
        ControlRequest::UpdateWeightsFromDistributed(op) => op.weight_version.clone(),
        ControlRequest::UpdateWeightFromDisk(op) => op.weight_version.clone(),
        _ => None,
    }
}

/// Whether a weight operation rewrites the parameters, and so may flush the
/// cache first.
pub fn is_weight_load(op: &ControlRequest) -> bool {
    matches!(
        op,
        ControlRequest::UpdateWeightsFromDistributed(_)
            | ControlRequest::UpdateWeightsFromMooncake(_)
            | ControlRequest::UpdateWeightFromDisk(_)
    )
}

/// Whether a weight operation asks for a cache flush first.
pub fn wants_flush(op: &ControlRequest) -> bool {
    match op {
        ControlRequest::UpdateWeightsFromDistributed(op) => op.flush_cache,
        ControlRequest::UpdateWeightsFromMooncake(op) => op.flush_cache,
        ControlRequest::UpdateWeightFromDisk(op) => op.flush_cache,
        // Capture writes dummy KV, and cached KV was computed by the old code.
        ControlRequest::ReloadForward(_) => true,
        _ => false,
    }
}

/// A weight operation's code on the same-round gate;
/// `None` for an operation that does not go through the gate.
pub fn gate_code(op: &ControlRequest) -> Option<i64> {
    Some(match op {
        ControlRequest::InitWeightsUpdateGroup(_) => 1,
        ControlRequest::UpdateWeightsFromDistributed(_) => 2,
        ControlRequest::DestroyWeightsUpdateGroup(_) => 3,
        ControlRequest::UpdateWeightsFromMooncake(_) => 4,
        ControlRequest::RebalanceExperts(_) => 5,
        ControlRequest::UpdateWeightFromDisk(_) => 6,
        ControlRequest::ReloadForward(_) => 7,
        _ => return None,
    })
}

/// A reload's identity on the gate: the caller's id, or the request's content without one.
pub fn reload_key(op: &ReloadForwardReqInput) -> i64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    match &op.id {
        Some(id) => id.hash(&mut hasher),
        None => (&op.mode, &op.modules).hash(&mut hasher),
    }
    // Non-negative, so the gate's MAX of key and -key compares it across ranks.
    (hasher.finish() >> 2) as i64
}

/// A gated operation's reply.
pub fn gated_reply(op: &ControlRequest, success: bool, message: String) -> ControlOutput {
    match op {
        ControlRequest::InitWeightsUpdateGroup(_) => {
            ControlOutput::InitWeightsUpdateGroup(InitWeightsUpdateGroupReqOutput {
                rid: None,
                http_worker_ipc: None,
                success,
                message,
            })
        }
        ControlRequest::UpdateWeightsFromDistributed(_) => {
            ControlOutput::UpdateWeightsFromDistributed(UpdateWeightsFromDistributedReqOutput {
                rid: None,
                http_worker_ipc: None,
                success,
                message,
            })
        }
        ControlRequest::DestroyWeightsUpdateGroup(_) => {
            ControlOutput::DestroyWeightsUpdateGroup(DestroyWeightsUpdateGroupReqOutput {
                rid: None,
                http_worker_ipc: None,
                success,
                message,
            })
        }
        ControlRequest::UpdateWeightsFromMooncake(_) => {
            ControlOutput::UpdateWeightsFromMooncake(UpdateWeightsFromMooncakeReqOutput {
                rid: None,
                http_worker_ipc: None,
                success,
                message,
            })
        }
        ControlRequest::RebalanceExperts(_) => {
            ControlOutput::RebalanceExperts(RebalanceExpertsReqOutput {
                rid: None,
                http_worker_ipc: None,
                success,
                message,
            })
        }
        ControlRequest::UpdateWeightFromDisk(_) => {
            ControlOutput::UpdateWeightFromDisk(UpdateWeightFromDiskReqOutput {
                rid: None,
                http_worker_ipc: None,
                success,
                message,
                num_paused_requests: Some(0),
            })
        }
        ControlRequest::ReloadForward(_) => reload_reply(success, &message),
        other => unreachable!("{} is not gated", other.tag()),
    }
}

fn pause_reply(success: bool, message: &str) -> ControlOutput {
    ControlOutput::PauseScheduler(PauseSchedulerReqOutput {
        rid: None,
        http_worker_ipc: None,
        success,
        message: message.to_owned(),
    })
}

fn resume_reply(success: bool, message: &str) -> ControlOutput {
    ControlOutput::ResumeScheduler(ResumeSchedulerReqOutput {
        rid: None,
        http_worker_ipc: None,
        success,
        message: message.to_owned(),
    })
}

fn release_reply(success: bool, message: &str) -> ControlOutput {
    ControlOutput::ReleaseMemoryOccupation(ReleaseMemoryOccupationReqOutput {
        rid: None,
        http_worker_ipc: None,
        success,
        message: message.to_owned(),
    })
}

fn reload_reply(success: bool, message: &str) -> ControlOutput {
    ControlOutput::ReloadForward(ReloadForwardReqOutput {
        rid: None,
        http_worker_ipc: None,
        success,
        message: message.to_owned(),
    })
}

fn resume_memory_reply(success: bool, message: &str) -> ControlOutput {
    ControlOutput::ResumeMemoryOccupation(ResumeMemoryOccupationReqOutput {
        rid: None,
        http_worker_ipc: None,
        success,
        message: message.to_owned(),
    })
}

/// Pause, memory and weight-operation state of one loop.
#[derive(Debug)]
pub struct Controls {
    config: ControlConfig,
    state: PauseState,
    drain: Option<Drain>,
    /// Memory regions released.
    released: Vec<String>,
    /// A release was accepted and has not freed its regions or failed.
    release_pending: bool,
    /// A forward reload was accepted and has not been answered.
    reload_pending: bool,
    /// The pending reload found the scheduler unpaused, so it resumes it.
    resume_after_reload: bool,
    /// The pending reload drained: the gate may run it once every rank's is.
    reload_ready: bool,
    /// When the pending reload gives up waiting, if it has a timeout.
    reload_deadline: Option<Instant>,
    /// An `abort` pause still has to abort what runs.
    abort_all: bool,
    /// An `abort` or `wait` pause, or a release, still has to cancel the
    /// admitted requests not in the scheduler yet, whose grammars compile
    /// or whose images' embeddings are on their way: the scheduler's drain
    /// does not see them, and once ready they would be withheld until a
    /// resume.
    cancel_unscheduled: bool,
    /// Gated weight operations, in arrival order.
    weight_ops: VecDeque<ControlRequest>,
    /// Standalone cache flushes received this round.
    flushes: usize,
    /// Replies to send.
    replies: Vec<ControlOutput>,
    /// The L3 namespace pages publish under, with the L3 tier.
    l3_weight_version: Option<String>,
    /// The engine's own gated operations (the expert rebalance's steps), in
    /// their own queue: frontend operations reach the attention-DP ranks in
    /// different rounds, internal ones in the same round everywhere.
    internal_ops: VecDeque<crate::eplb::InternalOp>,
}

/// What a control operation needs of the data plane right away.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Effect {
    /// Poll native slot and cache capacity on the control thread.
    AdmissionSnapshot,
    /// Read-only query of successful shared-store object confirmations.
    SharedPrefixPublication(String, Vec<i32>, i32),
    /// Cache metadata changes, executed by the control thread on every replica rank.
    DemoteSessions(Vec<String>, bool),
    /// Group ownership is independent of each request session.
    DemoteCacheGroups(Vec<String>, bool),
    /// Nothing beyond the state change and the reply.
    None,
    /// Start or stop the profiler (the data plane answers).
    Profile,
    /// Resume these memory regions now, in order, then reply.
    ResumeMemory(Vec<String>),
}

impl Controls {
    /// No pause, nothing released.
    pub fn new(config: ControlConfig) -> Self {
        Self {
            state: PauseState::Unpaused,
            drain: None,
            released: Vec::new(),
            release_pending: false,
            reload_pending: false,
            resume_after_reload: false,
            reload_ready: false,
            reload_deadline: None,
            abort_all: false,
            cancel_unscheduled: false,
            weight_ops: VecDeque::new(),
            flushes: 0,
            replies: Vec::new(),
            l3_weight_version: config.l3_weight_version.clone(),
            internal_ops: VecDeque::new(),
            config,
        }
    }

    /// Why a weight load would break the L3 namespace (the request
    /// handler's `_require_*` checks): a flushed load must name its
    /// checkpoint, and the namespace may change only with a flush, or later
    /// copies of the old weights' KV would land under the new one.
    fn l3_version_refusal(&self, op: &ControlRequest) -> Option<String> {
        let current = self.l3_weight_version.as_ref()?;
        let requested = requested_weight_version(op);
        if wants_flush(op) {
            return requested
                .is_none()
                .then(|| L3_FLUSH_REQUIRES_WEIGHT_VERSION.to_owned());
        }
        match requested {
            Some(version) if version != *current => Some(
                "L3 weight_version cannot change without flush_cache; retry the update with \
                 flush_cache=True so Device/Host KV and in-flight writebacks cannot land in the \
                 new namespace"
                    .to_owned(),
            ),
            _ => None,
        }
    }

    /// After a weight load succeeded on every rank, the L3 namespace new
    /// pages publish under: `Ok(None)` keeps the current one, `Err` refuses
    /// the load. Re-sending the current version keeps it too: the namespace
    /// already names it, and the storage executor refuses a namespace change
    /// while it has unretired operations.
    pub fn commit_l3_weight_version(
        &mut self,
        op: &ControlRequest,
    ) -> Result<Option<String>, String> {
        if self.l3_weight_version.is_none() {
            return Ok(None);
        }
        if let Some(refusal) = self.l3_version_refusal(op) {
            return Err(refusal);
        }
        match requested_weight_version(op) {
            Some(version) if self.l3_weight_version.as_ref() != Some(&version) => {
                self.l3_weight_version = Some(version.clone());
                Ok(Some(version))
            }
            _ => Ok(None),
        }
    }

    /// New requests are withheld from the scheduler.
    pub fn admit_blocked(&self) -> bool {
        self.state != PauseState::Unpaused
    }

    /// Nothing is scheduled this round.
    pub fn forward_blocked(&self) -> bool {
        self.state == PauseState::PausedAll
    }

    /// Memory is released: no forward may run, not even an idle one.
    pub fn memory_released(&self) -> bool {
        !self.released.is_empty()
    }

    /// A release is draining or memory is released; a disaggregated engine
    /// refuses new requests then, as their transfers would name memory that
    /// is about to go away.
    pub fn releasing_or_released(&self) -> bool {
        self.release_pending || !self.released.is_empty()
    }

    /// Take (once) whether an `abort` pause has to abort what runs.
    pub fn take_abort_all(&mut self) -> bool {
        std::mem::take(&mut self.abort_all)
    }

    /// Whether to cancel the admitted requests not in the scheduler yet,
    /// once per drain.
    pub fn take_cancel_unscheduled(&mut self) -> bool {
        std::mem::take(&mut self.cancel_unscheduled)
    }

    /// Take the replies produced so far.
    pub fn take_replies(&mut self) -> Vec<ControlOutput> {
        std::mem::take(&mut self.replies)
    }

    /// Queue a reply.
    pub fn reply(&mut self, output: ControlOutput) {
        self.replies.push(output);
    }

    /// Take this round's standalone flush count.
    pub fn take_flushes(&mut self) -> usize {
        std::mem::take(&mut self.flushes)
    }

    /// The gated operation at the head of the queue.
    pub fn weight_head(&self) -> Option<&ControlRequest> {
        self.weight_ops.front()
    }

    /// Pop the head once every rank holds one.
    pub fn pop_weight_op(&mut self) -> Option<ControlRequest> {
        self.weight_ops.pop_front()
    }

    /// Gated operations still queued, frontend or internal: either may
    /// still touch model memory.
    pub fn weights_busy(&self) -> bool {
        !self.weight_ops.is_empty() || !self.internal_ops.is_empty()
    }

    /// Queue the engine's own gated operations, at a rank-identical round.
    pub fn enqueue_internal(&mut self, ops: impl IntoIterator<Item = crate::eplb::InternalOp>) {
        self.internal_ops.extend(ops);
    }

    /// The internal operation at the head of its queue.
    pub fn internal_head(&self) -> Option<&crate::eplb::InternalOp> {
        self.internal_ops.front()
    }

    /// Pop the internal head once every rank holds one.
    pub fn pop_internal(&mut self) -> Option<crate::eplb::InternalOp> {
        self.internal_ops.pop_front()
    }

    /// The drain awaiting an empty scheduler, if any.
    pub fn drain(&self) -> Option<&Drain> {
        self.drain.as_ref()
    }

    /// Start a drain: stop admitting and arm `action` (`request_drain`). An
    /// engine with memory released stays frozen: no forward may run before
    /// its memory is back.
    fn request_drain(&mut self, abort_inflight: bool, action: Drain) {
        if self.released.is_empty() {
            self.state = PauseState::PausedNew;
        }
        self.drain = Some(action);
        self.cancel_unscheduled = true;
        if abort_inflight {
            self.abort_all = true;
        }
    }

    /// The drained scheduler's action, cleared before it runs.
    pub fn take_drain(&mut self) -> Option<Drain> {
        self.drain.take()
    }

    /// The pending reload drained here; the gate runs it once every attention-DP rank's has.
    pub fn mark_reload_ready(&mut self) {
        self.reload_ready = true;
    }

    /// Whether a reload was accepted and not answered; the same on every replica member.
    pub fn reload_pending(&self) -> bool {
        self.reload_pending
    }

    /// Whether the pending reload drained here.
    pub fn reload_ready(&self) -> bool {
        self.reload_ready
    }

    /// Whether a pending reload's timeout passed by `now`; the replica agrees before it expires.
    pub fn reload_timed_out(&self, now: Instant) -> bool {
        self.reload_pending && self.reload_deadline.is_some_and(|deadline| now >= deadline)
    }

    /// Give up the pending reload whose timeout passed on some replica member, and resume if it paused.
    pub fn expire_reload(&mut self) {
        if !self.reload_pending {
            return;
        }
        let resume =
            self.cancel_reload("timed out before it drained and every attention-DP engine held it");
        if resume {
            self.state = PauseState::Unpaused;
            self.abort_all = false;
            self.cancel_unscheduled = false;
        }
    }

    /// The reload was answered: schedule again unless a pause preceded it.
    pub fn finish_reload(&mut self) {
        self.reload_pending = false;
        self.reload_ready = false;
        self.reload_deadline = None;
        if std::mem::take(&mut self.resume_after_reload) {
            self.state = PauseState::Unpaused;
        }
    }

    /// The pause reply of a finished `abort`/`wait` drain.
    pub fn finish_pause_drain(&mut self) {
        self.reply(pause_reply(true, ""));
    }

    /// Memory regions were freed: freeze and reply.
    pub fn finish_release(&mut self, tags: Vec<String>) {
        self.release_pending = false;
        for tag in tags {
            if !self.released.contains(&tag) {
                self.released.push(tag);
            }
        }
        self.state = PauseState::PausedAll;
        self.reply(release_reply(true, ""));
    }

    /// Memory regions were resumed.
    pub fn finish_resume(&mut self, tags: &[String]) {
        self.released.retain(|tag| !tags.contains(tag));
        if self.released.is_empty() {
            // Fully awake: schedule again.
            self.state = PauseState::Unpaused;
        }
        self.reply(resume_memory_reply(true, ""));
    }

    /// Why a weight load cannot run now: it writes the weights and may issue
    /// collectives, so both regions must be resident.
    fn refuse_weight_load(&self) -> Option<String> {
        let released: Vec<&str> = ["weights", "nccl"]
            .into_iter()
            .filter(|tag| self.released.iter().any(|released| released == tag))
            .collect();
        (!released.is_empty()).then(|| {
            format!(
                "memory regions {released:?} are released; resume them with \
                 resume_memory_occupation before updating weights"
            )
        })
    }

    /// Order `tags` (`None`: every releasable region) by `order`; an
    /// unknown tag or one this engine cannot release is an error.
    fn normalize_tags(
        &self,
        tags: Option<&[String]>,
        order: &[&str],
    ) -> Result<Vec<String>, String> {
        let available = &self.config.memory_regions;
        let Some(tags) = tags else {
            return Ok(order
                .iter()
                .filter(|tag| available.iter().any(|a| a == *tag))
                .map(|tag| (*tag).to_owned())
                .collect());
        };
        let bad: Vec<&String> = tags
            .iter()
            .filter(|tag| !MEMORY_REGIONS.contains(&tag.as_str()))
            .collect();
        if !bad.is_empty() {
            return Err(format!("invalid tags: {bad:?}; valid: {MEMORY_REGIONS:?}"));
        }
        let unavailable: Vec<&String> =
            tags.iter().filter(|tag| !available.contains(tag)).collect();
        if !unavailable.is_empty() {
            let releasable: Vec<&str> = MEMORY_REGIONS
                .into_iter()
                .filter(|tag| available.iter().any(|a| a == tag))
                .collect();
            return Err(format!(
                "tags not releasable on this engine: {unavailable:?}; available: {releasable:?}"
            ));
        }
        Ok(order
            .iter()
            .filter(|tag| tags.iter().any(|t| t == *tag))
            .map(|tag| (*tag).to_owned())
            .collect())
    }

    /// Take one operation as it arrives. `release_refusal` says why a
    /// release cannot drain (buffered PD requests), if it cannot.
    pub fn handle(&mut self, request: ControlRequest, release_refusal: Option<String>) -> Effect {
        match request {
            ControlRequest::DemoteSessions(request) => {
                return Effect::DemoteSessions(request.session_ids, request.ended);
            }
            ControlRequest::DemoteCacheGroups(request) => {
                return Effect::DemoteCacheGroups(request.cache_group_ids, request.ended);
            }
            ControlRequest::GetAdmission(_) => return Effect::AdmissionSnapshot,
            ControlRequest::GetSharedPrefixPublication(request) => {
                return Effect::SharedPrefixPublication(
                    request.query_id,
                    request.input_ids,
                    request.max_prefix_tokens,
                );
            }
            ControlRequest::FlushCache(_) => self.flushes += 1,
            ControlRequest::PauseScheduler(pause) => self.pause(&pause.mode),
            ControlRequest::ResumeScheduler(_) => self.resume(),
            ControlRequest::IsSchedulerPaused(_) => {
                let is_paused = self.state != PauseState::Unpaused;
                self.reply(ControlOutput::IsSchedulerPaused(
                    IsSchedulerPausedReqOutput {
                        rid: None,
                        http_worker_ipc: None,
                        is_paused,
                    },
                ));
            }
            ControlRequest::ReleaseMemoryOccupation(release) => {
                self.release(release.tags.as_deref(), release_refusal);
            }
            ControlRequest::ResumeMemoryOccupation(resume) => {
                return self.resume_memory(resume.tags.as_deref());
            }
            ControlRequest::IsSleeping(_) => {
                let is_sleeping = !self.released.is_empty();
                self.reply(ControlOutput::IsSleeping(IsSleepingReqOutput {
                    rid: None,
                    http_worker_ipc: None,
                    is_sleeping,
                }));
            }
            ControlRequest::GetInternalState(_) => {
                self.reply(ControlOutput::GetInternalState(
                    ts_wire::control::GetInternalStateReqOutput {
                        rid: None,
                        http_worker_ipc: None,
                        internal_state: ts_wire::Dict(Vec::new()),
                    },
                ));
            }
            ControlRequest::SetInternalState(_) => {
                self.reply(ControlOutput::SetInternalState(
                    ts_wire::control::SetInternalStateReqOutput {
                        rid: None,
                        http_worker_ipc: None,
                        updated: false,
                        server_args: ts_wire::Dict(Vec::new()),
                    },
                ));
            }
            ControlRequest::UpdateWeightsFromTensor(_) => {
                self.reply(ControlOutput::UpdateWeightsFromTensor(
                    ts_wire::control::UpdateWeightsFromTensorReqOutput {
                        rid: None,
                        http_worker_ipc: None,
                        success: false,
                        message: "update_weights_from_tensor is not supported on this engine"
                            .to_owned(),
                    },
                ));
            }
            ControlRequest::Profile(_) => return Effect::Profile,
            ControlRequest::ReloadForward(op) => self.reload(op),
            op @ (ControlRequest::InitWeightsUpdateGroup(_)
            | ControlRequest::DestroyWeightsUpdateGroup(_)
            | ControlRequest::RebalanceExperts(_)) => self.weight_ops.push_back(op),
            op @ (ControlRequest::UpdateWeightsFromDistributed(_)
            | ControlRequest::UpdateWeightsFromMooncake(_)
            | ControlRequest::UpdateWeightFromDisk(_)) => match self
                .refuse_weight_load()
                .or_else(|| self.l3_version_refusal(&op))
            {
                Some(message) => {
                    let reply = gated_reply(&op, false, message);
                    self.reply(reply);
                }
                None => self.weight_ops.push_back(op),
            },
        }
        Effect::None
    }

    fn pause(&mut self, mode: &str) {
        if !matches!(mode, "abort" | "wait" | "keep") {
            self.reply(pause_reply(false, &format!("invalid pause mode: {mode:?}")));
            return;
        }
        if !self.released.is_empty() {
            // A wait or abort pause would let forwards run before the memory
            // is back, and the wake would silently undo a keep pause.
            self.reply(pause_reply(
                false,
                "memory is unavailable; use resume_memory_occupation",
            ));
            return;
        }
        if self.drain.is_some() {
            self.reply(pause_reply(false, "a pause is already in progress"));
            return;
        }
        if self.reload_pending {
            self.reply(pause_reply(false, "a forward reload is in progress"));
            return;
        }
        if mode == "keep" {
            self.state = PauseState::PausedAll;
            self.reply(pause_reply(true, ""));
            return;
        }
        self.request_drain(mode == "abort", Drain::Pause);
    }

    fn resume(&mut self) {
        if self.reload_pending {
            // The way out of a reload some attention-DP rank never received: it has not run yet.
            self.cancel_reload("cancelled by continue_generation before it ran");
        }
        if !self.released.is_empty() {
            self.reply(resume_reply(
                false,
                "memory is released; call resume_memory_occupation to wake before \
                 resuming the scheduler",
            ));
            return;
        }
        // A pause still waiting for its drain has not drained: it fails, so a
        // caller about to swap weights does not proceed under old requests.
        match self.drain.take() {
            Some(Drain::Pause) => self.reply(pause_reply(false, "resumed before pause drained")),
            Some(Drain::Release(_)) => {
                self.release_pending = false;
                self.reply(release_reply(false, "resumed before release drained"));
            }
            Some(Drain::Reload) => unreachable!("cancel_reload took it"),
            None => {}
        }
        self.state = PauseState::Unpaused;
        self.abort_all = false;
        self.cancel_unscheduled = false;
        self.reply(resume_reply(true, ""));
    }

    /// Withdraw a reload that is still draining or waiting at the gate and answer it as failed; returns whether it would have resumed.
    fn cancel_reload(&mut self, why: &str) -> bool {
        if matches!(self.drain, Some(Drain::Reload)) {
            self.drain = None;
        }
        self.weight_ops
            .retain(|op| !matches!(op, ControlRequest::ReloadForward(_)));
        self.reload_pending = false;
        self.reload_ready = false;
        self.reload_deadline = None;
        self.reply(reload_reply(false, why));
        std::mem::take(&mut self.resume_after_reload)
    }

    fn release(&mut self, tags: Option<&[String]>, refusal: Option<String>) {
        let fail = |controls: &mut Self, message: String| {
            controls.release_pending = false;
            controls.reply(release_reply(false, &message));
        };
        if !self.config.memory_saver {
            fail(
                self,
                "memory saver not enabled (--enable-memory-saver)".to_owned(),
            );
            return;
        }
        let tags = match self.normalize_tags(tags, &MEMORY_REGIONS) {
            Ok(tags) => tags,
            Err(error) => return fail(self, error),
        };
        let already: Vec<&String> = tags
            .iter()
            .filter(|tag| self.released.contains(tag))
            .collect();
        if !already.is_empty() {
            let message = format!("tags already released: {already:?}");
            return fail(self, message);
        }
        if self.drain.is_some() || self.reload_pending {
            return fail(
                self,
                "a pause, release or reload is already in progress".to_owned(),
            );
        }
        if self.state == PauseState::PausedAll && self.released.is_empty() {
            // Its drain would let the frozen requests run, and the wake
            // would silently end the keep pause.
            return fail(
                self,
                "a keep pause froze the scheduler; resume_scheduler before releasing memory"
                    .to_owned(),
            );
        }
        if let Some(refusal) = refusal {
            return fail(self, format!("cannot release memory: {refusal}"));
        }
        self.release_pending = true;
        self.request_drain(false, Drain::Release(tags));
    }

    fn reload(&mut self, op: ReloadForwardReqInput) {
        let refusal = if !self.config.allow_reload_forward {
            Some(
                "forward reload is off: it is a development operation; start the server with \
                 --allow-reload-forward"
                    .to_owned(),
            )
        } else if op.id.is_none() {
            Some("reload_forward needs an id, the same on every engine it is sent to".to_owned())
        } else if op
            .timeout_s
            .is_some_and(|timeout| !timeout.is_finite() || timeout < 0.0)
        {
            Some("reload_forward timeout_s must be a finite, non-negative number".to_owned())
        } else if !matches!(op.mode.as_str(), "wait" | "abort") {
            Some(format!(
                "invalid reload mode: {:?}; use wait or abort",
                op.mode
            ))
        } else if !self.released.is_empty() {
            Some("memory is released; resume_memory_occupation before reloading".to_owned())
        } else if self.drain.is_some() || self.reload_pending {
            Some("a pause, release or reload is already in progress".to_owned())
        } else if self.state == PauseState::PausedAll {
            Some("a keep pause froze the scheduler; resume_scheduler before reloading".to_owned())
        } else {
            None
        };
        if let Some(message) = refusal {
            self.reply(reload_reply(false, &message));
            return;
        }
        self.resume_after_reload = self.state == PauseState::Unpaused;
        self.reload_pending = true;
        self.reload_ready = false;
        self.reload_deadline = op
            .timeout_s
            .map(|timeout| Instant::now() + Duration::from_secs_f64(timeout));
        let abort = op.mode == "abort";
        // Queued on arrival, so it keeps the frontend's order with the weight operations on every rank.
        self.weight_ops.push_back(ControlRequest::ReloadForward(op));
        self.request_drain(abort, Drain::Reload);
    }

    fn resume_memory(&mut self, tags: Option<&[String]>) -> Effect {
        let resume_order: Vec<&str> = MEMORY_REGIONS.iter().rev().copied().collect();
        // `None` wakes exactly what sleeps, so a partial release round-trips.
        let tags = match tags {
            None => resume_order
                .iter()
                .filter(|tag| self.released.iter().any(|r| r == *tag))
                .map(|tag| (*tag).to_owned())
                .collect(),
            Some(tags) => match self.normalize_tags(Some(tags), &resume_order) {
                Ok(tags) => tags,
                Err(error) => {
                    self.reply(resume_memory_reply(false, &error));
                    return Effect::None;
                }
            },
        };
        let not_released: Vec<&String> = tags
            .iter()
            .filter(|tag| !self.released.contains(tag))
            .collect();
        if !not_released.is_empty() {
            let message = format!("tags not released: {not_released:?}");
            self.reply(resume_memory_reply(false, &message));
            return Effect::None;
        }
        if tags.is_empty() {
            self.reply(resume_memory_reply(true, ""));
            return Effect::None;
        }
        Effect::ResumeMemory(tags)
    }
}

#[cfg(test)]
mod tests {
    use ts_wire::control::{
        PauseSchedulerReqInput, ReleaseMemoryOccupationReqInput, ReloadForwardReqInput,
        ResumeSchedulerReqInput,
    };

    use super::*;

    fn pause(mode: &str) -> ControlRequest {
        ControlRequest::PauseScheduler(PauseSchedulerReqInput {
            rid: None,
            http_worker_ipc: None,
            mode: mode.to_owned(),
        })
    }

    fn resume() -> ControlRequest {
        ControlRequest::ResumeScheduler(ResumeSchedulerReqInput {
            rid: None,
            http_worker_ipc: None,
        })
    }

    fn release(tags: Option<Vec<&str>>) -> ControlRequest {
        ControlRequest::ReleaseMemoryOccupation(ReleaseMemoryOccupationReqInput {
            rid: None,
            http_worker_ipc: None,
            tags: tags.map(|tags| tags.into_iter().map(str::to_owned).collect()),
        })
    }

    fn rebalance() -> ControlRequest {
        ControlRequest::RebalanceExperts(ts_wire::control::RebalanceExpertsReqInput {
            rid: None,
            http_worker_ipc: None,
        })
    }

    fn reload(mode: &str) -> ControlRequest {
        ControlRequest::ReloadForward(ReloadForwardReqInput {
            rid: None,
            http_worker_ipc: None,
            mode: mode.to_owned(),
            modules: None,
            id: Some("test".to_owned()),
            timeout_s: None,
        })
    }

    #[test]
    fn a_reload_without_an_id_is_refused_and_one_past_its_timeout_expires() {
        let mut controls = controls(false);
        let ControlRequest::ReloadForward(mut op) = reload("wait") else {
            unreachable!();
        };
        op.id = None;
        controls.handle(ControlRequest::ReloadForward(op.clone()), None);
        assert!(!controls.admit_blocked());
        op.id = Some("late".to_owned());
        op.timeout_s = Some(0.0);
        controls.handle(ControlRequest::ReloadForward(op), None);
        assert!(controls.admit_blocked());
        assert!(controls.reload_timed_out(Instant::now()));
        controls.expire_reload();
        assert!(!controls.admit_blocked() && controls.weight_head().is_none());
        assert_eq!(
            successes(&controls.take_replies()),
            [
                ("ReloadForwardReqOutput", false),
                ("ReloadForwardReqOutput", false)
            ]
        );
    }

    #[test]
    fn a_reload_key_names_the_callers_id_or_else_the_request() {
        let keyed = |id: Option<&str>, mode: &str| {
            reload_key(&ReloadForwardReqInput {
                rid: None,
                http_worker_ipc: None,
                mode: mode.to_owned(),
                modules: None,
                id: id.map(str::to_owned),
                timeout_s: None,
            })
        };
        assert_eq!(keyed(Some("a"), "wait"), keyed(Some("a"), "abort"));
        assert_ne!(keyed(Some("a"), "wait"), keyed(Some("b"), "wait"));
        assert_ne!(keyed(None, "wait"), keyed(None, "abort"));
        assert!(keyed(Some("a"), "wait") >= 0);
    }

    fn successes(replies: &[ControlOutput]) -> Vec<(&'static str, bool)> {
        replies
            .iter()
            .map(|reply| {
                let success = match reply {
                    ControlOutput::PauseScheduler(r) => r.success,
                    ControlOutput::ResumeScheduler(r) => r.success,
                    ControlOutput::ReleaseMemoryOccupation(r) => r.success,
                    ControlOutput::ResumeMemoryOccupation(r) => r.success,
                    ControlOutput::UpdateWeightFromDisk(r) => r.success,
                    ControlOutput::UpdateWeightsFromMooncake(r) => r.success,
                    ControlOutput::ReloadForward(r) => r.success,
                    ControlOutput::RebalanceExperts(r) => r.success,
                    _ => true,
                };
                (reply.tag(), success)
            })
            .collect()
    }

    fn controls(memory_saver: bool) -> Controls {
        Controls::new(ControlConfig {
            memory_saver,
            memory_regions: vec!["kv_cache".to_owned(), "weights".to_owned()],
            l3_weight_version: None,
            allow_reload_forward: true,
        })
    }

    #[test]
    fn a_wait_pause_answers_after_its_drain_and_a_resume_cancels_it() {
        let mut controls = controls(false);
        controls.handle(pause("wait"), None);
        assert!(controls.admit_blocked() && !controls.forward_blocked());
        assert!(controls.take_replies().is_empty());
        // A second pause while draining is refused.
        controls.handle(pause("keep"), None);
        assert_eq!(
            successes(&controls.take_replies()),
            [("PauseSchedulerReqOutput", false)]
        );
        // A resume before the drain fails the pause and resumes.
        controls.handle(resume(), None);
        assert_eq!(
            successes(&controls.take_replies()),
            [
                ("PauseSchedulerReqOutput", false),
                ("ResumeSchedulerReqOutput", true)
            ]
        );
        assert!(!controls.admit_blocked());

        controls.handle(pause("abort"), None);
        assert!(controls.take_abort_all());
        assert!(!controls.take_abort_all());
        assert!(controls.take_cancel_unscheduled());
        assert!(!controls.take_cancel_unscheduled());
        assert_eq!(controls.take_drain(), Some(Drain::Pause));
        controls.finish_pause_drain();
        assert_eq!(
            successes(&controls.take_replies()),
            [("PauseSchedulerReqOutput", true)]
        );
    }

    /// Run the gate's part of a drained reload as the loop does, answering `success`.
    fn complete_reload(controls: &mut Controls, success: bool) {
        assert_eq!(controls.take_drain(), Some(Drain::Reload));
        controls.mark_reload_ready();
        let head = controls.weight_head().expect("the reload is queued");
        assert_eq!(gate_code(head), Some(7));
        assert!(wants_flush(head));
        let op = controls.pop_weight_op().unwrap();
        controls.reply(gated_reply(&op, success, String::new()));
        controls.finish_reload();
    }

    #[test]
    fn a_forward_reload_drains_waits_at_the_gate_and_resumes() {
        let mut controls = controls(false);
        controls.handle(reload("abort"), None);
        assert!(controls.admit_blocked() && !controls.forward_blocked());
        assert!(controls.take_abort_all() && controls.take_cancel_unscheduled());
        assert!(controls.take_replies().is_empty());
        // Nothing pauses, releases or reloads again while it is pending.
        controls.handle(pause("wait"), None);
        controls.handle(release(None), None);
        controls.handle(reload("wait"), None);
        assert_eq!(
            successes(&controls.take_replies()),
            [
                ("PauseSchedulerReqOutput", false),
                ("ReleaseMemoryOccupationReqOutput", false),
                ("ReloadForwardReqOutput", false)
            ]
        );
        // A gated operation queues behind it, in arrival order on every rank.
        controls.handle(rebalance(), None);
        assert!(controls.take_replies().is_empty());
        assert!(controls.admit_blocked() && !controls.reload_ready());
        complete_reload(&mut controls, false);
        assert_eq!(
            successes(&controls.take_replies()),
            [("ReloadForwardReqOutput", false)]
        );
        // A failed reload still resumes: the old code serves on.
        assert!(!controls.admit_blocked());
        assert!(matches!(
            controls.weight_head(),
            Some(ControlRequest::RebalanceExperts(_))
        ));
    }

    #[test]
    fn a_resume_cancels_a_reload_still_draining_or_waiting_at_the_gate() {
        let mut controls = controls(false);
        controls.handle(reload("wait"), None);
        controls.handle(resume(), None);
        assert!(!controls.admit_blocked() && controls.drain().is_none());
        controls.handle(reload("wait"), None);
        assert_eq!(controls.take_drain(), Some(Drain::Reload));
        controls.mark_reload_ready();
        controls.handle(resume(), None);
        assert!(!controls.admit_blocked() && controls.weight_head().is_none());
        assert_eq!(
            successes(&controls.take_replies()),
            [
                ("ReloadForwardReqOutput", false),
                ("ResumeSchedulerReqOutput", true),
                ("ReloadForwardReqOutput", false),
                ("ResumeSchedulerReqOutput", true)
            ]
        );
    }

    #[test]
    fn a_forward_reload_after_a_pause_leaves_the_scheduler_paused() {
        let mut controls = controls(false);
        controls.handle(pause("wait"), None);
        assert_eq!(controls.take_drain(), Some(Drain::Pause));
        controls.finish_pause_drain();
        controls.handle(reload("wait"), None);
        complete_reload(&mut controls, true);
        assert_eq!(
            successes(&controls.take_replies()),
            [
                ("PauseSchedulerReqOutput", true),
                ("ReloadForwardReqOutput", true)
            ]
        );
        assert!(controls.admit_blocked());
        controls.handle(resume(), None);
        assert!(!controls.admit_blocked());
    }

    #[test]
    fn a_forward_reload_is_refused_on_arrival_without_its_flag_or_under_a_freeze() {
        let mut off = Controls::new(ControlConfig {
            memory_saver: false,
            memory_regions: Vec::new(),
            l3_weight_version: None,
            allow_reload_forward: false,
        });
        off.handle(reload("wait"), None);
        // Refused before any drain: admission never stops.
        assert!(!off.admit_blocked() && off.drain().is_none());
        let mut on = controls(false);
        on.handle(reload("now"), None);
        on.handle(pause("keep"), None);
        on.handle(reload("wait"), None);
        assert!(on.drain().is_none());
        assert_eq!(
            successes(&off.take_replies())
                .into_iter()
                .chain(successes(&on.take_replies()))
                .collect::<Vec<_>>(),
            [
                ("ReloadForwardReqOutput", false),
                ("ReloadForwardReqOutput", false),
                ("PauseSchedulerReqOutput", true),
                ("ReloadForwardReqOutput", false)
            ]
        );
    }

    fn mooncake_load(flush_cache: bool, weight_version: Option<&str>) -> ControlRequest {
        ControlRequest::UpdateWeightsFromMooncake(
            ts_wire::control::UpdateWeightsFromMooncakeReqInput {
                rid: None,
                http_worker_ipc: None,
                version: 7,
                flush_cache,
                weight_version: weight_version.map(str::to_owned),
                snapshot: None,
            },
        )
    }

    #[test]
    fn l3_loads_keep_the_namespace_tied_to_the_checkpoint() {
        let mut l3 = Controls::new(ControlConfig {
            memory_saver: false,
            memory_regions: Vec::new(),
            l3_weight_version: Some("v1".to_owned()),
            allow_reload_forward: false,
        });
        // A flushed load without a version publishes the committed one's.
        assert_eq!(
            l3.commit_l3_weight_version(&mooncake_load(true, None)),
            Ok(Some("7".to_owned()))
        );
        // Switching the namespace needs a flush; keeping it does not.
        l3.handle(mooncake_load(false, Some("v2")), None);
        assert_eq!(
            successes(&l3.take_replies()),
            [("UpdateWeightsFromMooncakeReqOutput", false)]
        );
        // Re-sending the current version changes nothing.
        assert_eq!(
            l3.commit_l3_weight_version(&mooncake_load(false, Some("7"))),
            Ok(None)
        );
        assert_eq!(
            l3.commit_l3_weight_version(&mooncake_load(false, None)),
            Ok(None)
        );
        // Without the tier nothing is checked.
        let mut plain = controls(false);
        assert_eq!(
            plain.commit_l3_weight_version(&mooncake_load(false, Some("x"))),
            Ok(None)
        );
    }

    #[test]
    fn a_release_neither_ends_a_keep_pause_nor_thaws_a_released_engine() {
        let mut controls = controls(true);
        controls.handle(pause("keep"), None);
        controls.take_replies();
        // Its drain would run the frozen requests, and the wake would end
        // the pause.
        controls.handle(release(None), None);
        assert_eq!(
            successes(&controls.take_replies()),
            [("ReleaseMemoryOccupationReqOutput", false)]
        );
        assert!(controls.forward_blocked() && controls.drain().is_none());
        controls.handle(resume(), None);
        controls.take_replies();

        // A second release, while the first one's region is released, drains
        // frozen: no forward may run before the memory is back.
        controls.handle(release(Some(vec!["kv_cache"])), None);
        let Some(Drain::Release(tags)) = controls.take_drain() else {
            panic!("a release drains");
        };
        controls.finish_release(tags);
        controls.take_replies();
        controls.handle(release(Some(vec!["weights"])), None);
        assert!(controls.forward_blocked());
        assert_eq!(
            controls.drain(),
            Some(&Drain::Release(vec!["weights".to_owned()]))
        );
    }

    #[test]
    fn a_keep_pause_freezes_at_once() {
        let mut controls = controls(false);
        controls.handle(pause("keep"), None);
        assert!(controls.forward_blocked());
        assert_eq!(
            successes(&controls.take_replies()),
            [("PauseSchedulerReqOutput", true)]
        );
        controls.handle(pause("bogus"), None);
        assert_eq!(
            successes(&controls.take_replies()),
            [("PauseSchedulerReqOutput", false)]
        );
    }

    #[test]
    fn a_release_drains_frees_and_blocks_weight_loads_until_resumed() {
        let mut off = controls(false);
        off.handle(release(None), None);
        assert_eq!(
            successes(&off.take_replies()),
            [("ReleaseMemoryOccupationReqOutput", false)]
        );

        let mut controls = controls(true);
        controls.handle(release(Some(vec!["nccl"])), None);
        assert_eq!(
            successes(&controls.take_replies()),
            [("ReleaseMemoryOccupationReqOutput", false)]
        );
        controls.handle(release(None), Some("buffered PD requests".to_owned()));
        assert_eq!(
            successes(&controls.take_replies()),
            [("ReleaseMemoryOccupationReqOutput", false)]
        );
        controls.handle(release(None), None);
        assert!(controls.releasing_or_released());
        let Some(Drain::Release(tags)) = controls.take_drain() else {
            panic!("a release drains");
        };
        assert_eq!(tags, ["kv_cache", "weights"]);
        controls.finish_release(tags);
        assert!(controls.forward_blocked() && controls.memory_released());

        // A load cannot run while the weights are released.
        let load =
            ControlRequest::UpdateWeightFromDisk(ts_wire::control::UpdateWeightFromDiskReqInput {
                rid: None,
                http_worker_ipc: None,
                model_path: "/ckpt".to_owned(),
                load_format: None,
                weight_version: None,
                flush_cache: true,
            });
        controls.take_replies();
        controls.handle(load, None);
        assert!(!controls.weights_busy());
        assert_eq!(
            successes(&controls.take_replies()),
            [("UpdateWeightFromDiskReqOutput", false)]
        );

        let resume_memory = ControlRequest::ResumeMemoryOccupation(
            ts_wire::control::ResumeMemoryOccupationReqInput {
                rid: None,
                http_worker_ipc: None,
                tags: None,
            },
        );
        let Effect::ResumeMemory(tags) = controls.handle(resume_memory, None) else {
            panic!("a resume wakes the released regions");
        };
        // Reverse of the release order.
        assert_eq!(tags, ["weights", "kv_cache"]);
        controls.finish_resume(&tags);
        assert!(!controls.admit_blocked() && !controls.memory_released());
    }
}

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

use std::collections::VecDeque;

use ts_wire::{
    control::{
        DestroyWeightsUpdateGroupReqOutput, InitWeightsUpdateGroupReqOutput,
        IsSchedulerPausedReqOutput, IsSleepingReqOutput, PauseSchedulerReqOutput,
        RebalanceExpertsReqOutput, ReleaseMemoryOccupationReqOutput,
        ResumeMemoryOccupationReqOutput, ResumeSchedulerReqOutput, UpdateWeightFromDiskReqOutput,
        UpdateWeightsFromDistributedReqOutput, UpdateWeightsFromMooncakeReqOutput,
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
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Drain {
    /// An `abort` or `wait` pause: reply.
    Pause,
    /// A memory release: free these regions, freeze and reply.
    Release(Vec<String>),
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
        _ => return None,
    })
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
        if mode == "keep" {
            self.state = PauseState::PausedAll;
            self.reply(pause_reply(true, ""));
            return;
        }
        self.request_drain(mode == "abort", Drain::Pause);
    }

    fn resume(&mut self) {
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
            None => {}
        }
        self.state = PauseState::Unpaused;
        self.abort_all = false;
        self.cancel_unscheduled = false;
        self.reply(resume_reply(true, ""));
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
        if self.drain.is_some() {
            return fail(self, "a pause or release is already in progress".to_owned());
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
        PauseSchedulerReqInput, ReleaseMemoryOccupationReqInput, ResumeSchedulerReqInput,
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

    fn mooncake_load(flush_cache: bool, weight_version: Option<&str>) -> ControlRequest {
        ControlRequest::UpdateWeightsFromMooncake(
            ts_wire::control::UpdateWeightsFromMooncakeReqInput {
                rid: None,
                http_worker_ipc: None,
                version: 7,
                flush_cache,
                weight_version: weight_version.map(str::to_owned),
            },
        )
    }

    #[test]
    fn l3_loads_keep_the_namespace_tied_to_the_checkpoint() {
        let mut l3 = Controls::new(ControlConfig {
            memory_saver: false,
            memory_regions: Vec::new(),
            l3_weight_version: Some("v1".to_owned()),
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

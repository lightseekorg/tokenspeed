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

//! The Rust control plane end to end without a GPU: a `ts-client` frontend,
//! the engine-side wire, the real scheduler and a stand-in for the device.
//!
//! The stand-in samples a request's `k`-th generated token as `1000 + k`
//! (from 0), so outputs say exactly which tokens each request was given.
//! Several ranks run as threads with their own schedulers and devices,
//! connected by a real collective mesh.

use std::{
    collections::{HashMap, VecDeque},
    ffi::CString,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex, Once,
    },
    thread,
    time::Duration,
};

use pyo3::{prelude::*, types::PyDict};
use ts_client::{
    ClientConfig, ClientListener, EngineClient, LoadBalance, OutputChunk, RequestStream, Routing,
};
use ts_collective::{Group, Listener, Mesh, Reduce};
use ts_control::{
    batch_log::BatchLogConfig,
    control::ControlConfig,
    eplb::{Algorithm, Controller, EplbConfig, EplbSpecs, SlotMoves},
    event_loop::{
        DataPlane, DataPlaneError, Dispatch, DpRound, EventLoop, Frontend, Load, LoopConfig,
        LoopError, Ranks, Subsystems,
    },
    grammar::{GrammarConfig, GrammarKey, Grammars},
    ingress::{Admission, IngressLimits, LogprobServability},
    output::{FinishLog, ForwardResult, LogprobPlan, OutputProcessor, Role, StreamedOutput},
    transport::EngineTransport,
};
use ts_l3::{
    monotonic_clock, Executor, ExecutorConfig, GroupLayout, HostBuffer, HostLayout, IoResult,
    Namespace, Range, Store, StoreError,
};
use ts_pd::{
    BootstrapInfo, EmbeddingItem, EmbeddingReceiver, LandedItem, LandedRequest, PdError, PdEvent,
    RemotePrefill, TransferPeer, TransferPoll,
};
use ts_scheduler::{
    CacheEvent, CacheEventKind, CacheOpKind, CacheOpView, ExecutionPlan, Scheduler,
};
use ts_wire::{
    codec::Dict,
    control::{
        FlushCacheReqInput, IsSchedulerPausedReqInput, IsSleepingReqInput, PauseSchedulerReqInput,
        ProfileReq, ReleaseMemoryOccupationReqInput, ResumeMemoryOccupationReqInput,
        ResumeSchedulerReqInput, UpdateWeightFromDiskReqInput, PROFILE_START,
    },
    multimodal::{Modality, MultimodalDataItem, MultimodalInputs},
    ControlOutput, ControlRequest, EngineCoreReadyResponse, EngineRequest, SamplingParams,
    TokenizedGenerateReqInput, Value,
};

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

const MAX_SCHEDULED_TOKENS: usize = 256;

/// The cache the scheduler manages besides the device pages.
#[derive(Clone, Copy)]
enum Tiers {
    /// None: no op ever needs an acknowledgment.
    DeviceOnly,
    /// A retraction snapshot pool, with a retraction forced every few plans.
    ForcedRetraction,
    /// A Host L2 tier that published prefixes are written back to.
    HostL2,
    /// A Host L2 tier that what the Device evicts is written back to.
    HostL2WriteBack,
    /// `HostL2WriteBack` beside `ForcedRetraction`'s snapshot pool and
    /// retractions, on 64 usable Full blocks, few enough that restores evict
    /// cache.
    HostL2WriteBackForcedRetraction,
    /// A Host L2 tier backed by an L3 store, which holds every prefix page
    /// a query asks for.
    L3,
}

/// The scheduler over `tiers` for `role`, its KV cache split over
/// `kv_shard_count` shards: an LCM block then packs one Full block per
/// shard, so the Device keeps 512 usable Full blocks (64 on
/// `HostL2WriteBackForcedRetraction`) and a tier with Host pages 1024, the
/// counts the tests' numbers rely on.
fn new_scheduler(tiers: Tiers, role: Role, kv_shard_count: usize) -> Scheduler {
    interpreter();
    let role = match role {
        Role::Plain => "Fused",
        Role::Prefill => "P",
        Role::Decode => "D",
    };
    let tiers = match tiers {
        Tiers::DeviceOnly => "",
        Tiers::ForcedRetraction => {
            "cfg.num_snapshot_pages = 65\n\
             cfg.max_retracted_requests = 8\n\
             cfg.debug_force_retraction_interval = 3"
        }
        Tiers::HostL2 => {
            "cfg.disable_l2_cache = False\n\
             cfg.num_host_pages = 1025\n\
             cfg.host_write_policy = ts.SchedulerConfig.HostWritePolicy.WriteThrough"
        }
        Tiers::HostL2WriteBack => {
            "cfg.disable_l2_cache = False\n\
             cfg.num_host_pages = 1025\n\
             cfg.host_write_policy = ts.SchedulerConfig.HostWritePolicy.WriteBack"
        }
        Tiers::HostL2WriteBackForcedRetraction => {
            "cfg.num_device_pages = 64 // cfg.cache_groups[0].shard_count + 1\n\
             cfg.disable_l2_cache = False\n\
             cfg.num_host_pages = 1025\n\
             cfg.host_write_policy = ts.SchedulerConfig.HostWritePolicy.WriteBack\n\
             cfg.num_snapshot_pages = 65\n\
             cfg.max_retracted_requests = 8\n\
             cfg.debug_force_retraction_interval = 3"
        }
        Tiers::L3 => {
            "cfg.disable_l2_cache = False\n\
             cfg.num_host_pages = 1025\n\
             cfg.host_write_policy = ts.SchedulerConfig.HostWritePolicy.WriteThrough\n\
             cfg.enable_l3_storage = True\n\
             cfg.storage_config = ts.StorageConfig(\n\
                 8, 8, 64, 64, 1024, 1, 10**9, 10**9,\n\
                 ts.StoragePressure(64, 2**40, 10**9),\n\
                 ts.StoragePressure(32, 2**39, 5 * 10**8))"
        }
    };
    let code = format!(
        r#"
import tokenspeed_scheduler as ts
cfg = ts.SchedulerConfig()
cfg.prefix_granularity = 16
cfg.max_scheduled_tokens = {MAX_SCHEDULED_TOKENS}
cfg.max_batch_size = 8
cfg.num_device_pages = 512 // {kv_shard_count} + 1
cfg.num_snapshot_pages = 1
cfg.cache_groups = [ts.CacheGroupConfig(group_id="full_attention", block_granularity=16,
                                        total_pages=513, attn_kind=ts.AttnKind.Full,
                                        cache_blocks_per_lcm_block={kv_shard_count},
                                        shard_count={kv_shard_count})]
cfg.role = ts.SchedulerConfig.Role.{role}
if cfg.role != ts.SchedulerConfig.Role.Fused:
    cfg.cache_groups[0].transfer_policy = ts.CacheTransferPolicy.FullSuffix
{tiers}
if cfg.num_host_pages > 1:
    cfg.num_host_pages = 1024 // {kv_shard_count} + 1
result = ts.Scheduler(cfg)
"#
    );
    Python::attach(|py| {
        let globals = PyDict::new(py);
        py.run(&CString::new(code).unwrap(), Some(&globals), None)
            .unwrap();
        Scheduler::from_python(&globals.get_item("result").unwrap().unwrap()).unwrap()
    })
}

/// The prompt logprob the fake device reports for the token after prompt
/// position `p`, and the alternatives of a row whose own logprob is
/// `logprob`: itself as `id`, then lower ones with ids 900, 901, ...
fn fake_prompt_logprob(p: i64) -> f64 {
    -(p as f64 + 1.0) / 10.0
}

fn fake_alternatives(logprob: f64, id: i32, k: usize) -> (Vec<f64>, Vec<i32>) {
    (0..k)
        .map(|r| {
            if r == 0 {
                (logprob, id)
            } else {
                (logprob - r as f64, 900 + r as i32)
            }
        })
        .unzip()
}

/// What the device computes for a forward's logprob plan: the planned prompt
/// rows (the prompt is `1..=n`, so the token after position `p` is `p + 2`)
/// and the prefill rows' sampled-token alternatives.
fn fake_logprobs(plan: &LogprobPlan, result: &mut ForwardResult) {
    if let Some(input) = &plan.input {
        let mut values = Vec::new();
        let mut tops = (Vec::new(), Vec::new());
        for (&count, &start) in input.counts.iter().zip(&input.position_starts) {
            for p in start..start + count {
                values.push(fake_prompt_logprob(p));
                let (top_values, top_ids) =
                    fake_alternatives(fake_prompt_logprob(p), p as i32 + 2, input.top_k);
                tops.0.push(top_values);
                tops.1.push(top_ids);
            }
        }
        result.input_token_logprobs = Some(values);
        if input.top_k > 0 {
            result.input_top_logprobs = Some(tops);
        }
    }
    if plan.sampled_top_k > 0 {
        let logprobs = result.output_logprobs.as_ref().expect("set above");
        let rows: Vec<_> = result
            .output_tokens
            .iter()
            .zip(logprobs)
            .map(|(&token, &logprob)| fake_alternatives(logprob, token, plan.sampled_top_k))
            .collect();
        result.output_top_logprobs = Some(rows.into_iter().unzip());
    }
}

/// One token-history seed as the device saw it: (slot, prefix).
type HistorySeed = (i32, Vec<i32>);

/// One row's n-gram window as the device saw it: (position, window).
type NgramRow = (i64, Vec<i32>);

/// How long a fake forward takes; long enough that a request with a large
/// budget is still running when the test acts on it.
const FORWARD_TIME: Duration = Duration::from_millis(1);

/// One plan as a rank ran it: the cache-event polls before it, its
/// forward's request ids and its cache ops.
type ExecutedPlan = (usize, Vec<String>, Vec<CacheOpView>);

/// Stands in for the device: results are ready when the forward returns,
/// and a cache op is acknowledged once, on the first poll after it ran or
/// `ack_polls` polls later, and never while `ack_gate` is closed.
#[derive(Default)]
struct FakeDevice {
    /// This rank; PD names it as a sender of a request's KV.
    rank: usize,
    /// Tokens generated per request so far.
    generated: HashMap<String, i32>,
    /// Each released request with the tokens it was given.
    released: Arc<Mutex<Vec<(String, i32)>>>,
    /// Every plan run, in order.
    plans: Arc<Mutex<Vec<ExecutedPlan>>>,
    /// Cache ops run and not acknowledged yet, in the order they ran, each
    /// with the polls it still waits.
    unacknowledged: Vec<(usize, CacheOpView)>,
    /// The polls a cache op waits after the first one before it is
    /// acknowledged.
    ack_polls: usize,
    /// Cache-event polls so far.
    polls: usize,
    /// While closed, a poll acknowledges nothing.
    ack_gate: Arc<Gate>,
    /// Every acknowledgment sent, in order: the poll that sent it, its kind
    /// and op id.
    acknowledged: Arc<Mutex<Vec<(usize, CacheEventKind, u32)>>>,
    /// Idle forwards run for attention-DP peers.
    idle_forwards: Arc<Mutex<usize>>,
    /// A pipeline stage before the last: it samples nothing meaningful.
    placeholder_stage: bool,
    /// Every forward's n-gram windows: (position, window) per row.
    ngram: Arc<Mutex<Vec<NgramRow>>>,
    /// Every token-history seed: (slot, prefix).
    history: Arc<Mutex<Vec<HistorySeed>>>,
    /// PD: the transfers this engine's peer node sees.
    wire: Option<Arc<Mutex<Wire>>>,
    /// PD: each admitted request's room.
    rooms: HashMap<String, u64>,
    /// PD, D role: every landing, in order.
    landed: Arc<Mutex<Vec<Landing>>>,
    /// Control operations the device ran, in order.
    device_ops: Arc<Mutex<Vec<String>>>,
    /// Results published after the control plane's all-rank agreement.
    weight_commits: Arc<Mutex<Vec<bool>>>,
    /// Inject a local loading failure while other ranks succeed.
    fail_weight_update: bool,
    /// EPLB: this rank's EP rank, which skews the load it counts.
    ep_rank: usize,
    /// EPLB: the placement the experts sit in, as the routing tables say.
    placement: Arc<Mutex<Vec<Vec<i32>>>>,
    /// EPLB: every chunk applied, with its layers' rows.
    applied: Applied,
    /// EPD: requests whose landed embeddings were published, in order.
    epd_admitted: Arc<Mutex<Vec<String>>>,
    /// EPD: receive slots published and not yet reported copied out.
    epd_copying: Vec<usize>,
    /// EPD: slots reported copied out.
    epd_released: Arc<Mutex<Vec<usize>>>,
    /// Grammar: each constrained request's matcher handle.
    grammars: HashMap<String, u64>,
}

/// EPLB chunks applied: their layers and those layers' new rows.
type Applied = Arc<Mutex<Vec<(Vec<usize>, Vec<Vec<i32>>)>>>;

/// The toy vocabulary's grammar settings (`ts-grammar`'s fixture: a, b, {, },
/// ", x, :, ",", and the stop token 8).
/// The grammar `Setup::grammar_timeouts` times out.
const TIMED_OUT_GRAMMAR: &str = "a(b|x)";

fn toy_grammars() -> Grammars {
    Grammars::new(GrammarConfig {
        tokenizer_info: include_str!("../../ts-grammar/tests/fixtures/toy_tokenizer_info.json")
            .to_owned(),
        any_whitespace: true,
        compile_timeout: Duration::from_secs(30),
        max_retries: 2,
    })
    .unwrap()
}

/// The lowest token the matcher behind `handle` allows next; a finished
/// grammar is left alone, as the data plane's fill does, and its row samples
/// anything.
#[allow(unsafe_code)]
fn lowest_allowed(handle: u64) -> i32 {
    let entries: HashMap<&str, usize> = ts_grammar::table::entries().into_iter().collect();
    // SAFETY: the entries' own addresses, with their signatures.
    let (is_terminated, fill) = unsafe {
        (
            std::mem::transmute::<usize, extern "C" fn(u64) -> i32>(entries["is_terminated"]),
            std::mem::transmute::<usize, extern "C" fn(u64, usize, i64, i64, i32) -> i32>(
                entries["fill_bitmask"],
            ),
        )
    };
    if is_terminated(handle) == 1 {
        return 0;
    }
    let mut bitmask = [0i32; 1];
    assert_eq!(fill(handle, bitmask.as_mut_ptr() as usize, 1, 1, 0), 0);
    (0..9)
        .find(|token| bitmask[0] >> token & 1 == 1)
        .expect("some token is allowed")
}

/// A D-role landing: the drafter candidates and the slot made ready.
type Landing = (Option<(i32, Vec<i32>)>, Option<i32>);

/// What two PD engines' stand-in transfer peers share, by room.
#[derive(Default)]
struct Wire {
    rooms: HashMap<u64, WireRoom>,
    /// How many attention-TP ranks the prefill engine has; each sends its
    /// part of a request's KV.
    prefill_ranks: usize,
}

#[derive(Default)]
struct WireRoom {
    /// The decode engine connected to the room.
    decode_registered: bool,
    /// The decode engine admitted the pages, in this slot.
    preallocated: Option<i32>,
    /// The prefill engine sent the KV and this first token.
    sent: Option<i32>,
    /// The prefill ranks that sent their part of the KV.
    sent_by: Vec<usize>,
    /// The transfer breaks.
    fails: bool,
    /// The transfer breaks on this prefill rank alone.
    fails_on_prefill_rank: Option<usize>,
}

/// A stand-in transfer peer over a [`Wire`]: a transfer advances as soon as
/// both engines' devices did their part, as every rank of the replica sees
/// it.
struct FakePeer {
    role: Role,
    rank: usize,
    wire: Arc<Mutex<Wire>>,
    /// Tracked requests: rid, room, bootstrapped.
    tracked: Vec<(String, u64, bool)>,
}

impl TransferPeer for FakePeer {
    fn register(&mut self, rid: &str, bootstrap: &BootstrapInfo) -> Result<(), PdError> {
        if self.role == Role::Decode {
            self.wire
                .lock()
                .unwrap()
                .rooms
                .entry(bootstrap.room)
                .or_default()
                .decode_registered = true;
        }
        self.tracked.push((rid.to_owned(), bootstrap.room, false));
        Ok(())
    }

    /// As `ts-pd`'s peers poll: each rank reads its view of every tracked
    /// transfer, the replica agrees on the least advanced one, and the
    /// agreed state is acted on. The wire is unlocked while the replica
    /// agrees: the other ranks lock it to read their own views first.
    fn poll(&mut self, mesh: &mut Mesh, replica: &Group) -> Result<Vec<PdEvent>, PdError> {
        if self.tracked.is_empty() {
            return Ok(Vec::new());
        }
        let (role, rank) = (self.role, self.rank);
        let polls: Vec<i64> = {
            let mut wire = self.wire.lock().unwrap();
            let prefill_ranks = wire.prefill_ranks;
            self.tracked
                .iter()
                .map(|(_, room, bootstrapped)| {
                    let room = wire.rooms.entry(*room).or_default();
                    let poll = if room.fails
                        || (role == Role::Prefill && room.fails_on_prefill_rank == Some(rank))
                    {
                        TransferPoll::Failed
                    } else if !*bootstrapped {
                        let connected = match role {
                            Role::Prefill => room.preallocated.is_some(),
                            _ => room.decode_registered,
                        };
                        if connected {
                            TransferPoll::Bootstrapped
                        } else {
                            TransferPoll::Bootstrapping
                        }
                    } else {
                        // The KV left this prefill rank, or arrived from
                        // every one.
                        let sent = match role {
                            Role::Prefill => room.sent_by.contains(&rank),
                            _ => room.sent_by.len() == prefill_ranks,
                        };
                        if sent {
                            TransferPoll::Success
                        } else {
                            TransferPoll::Bootstrapped
                        }
                    };
                    poll as i64
                })
                .collect()
        };
        let agreed = if replica.size() > 1 {
            mesh.all_reduce(replica, &polls, Reduce::Min)?
        } else {
            polls
        };

        let mut wire = self.wire.lock().unwrap();
        let mut events = Vec::new();
        let mut agreed = agreed.into_iter();
        self.tracked.retain_mut(|(rid, room, bootstrapped)| {
            let poll = agreed.next().expect("one state per tracked request");
            let room = wire.rooms.entry(*room).or_default();
            match TransferPoll::from_wire(poll).unwrap_or(TransferPoll::Failed) {
                TransferPoll::Failed => {
                    // The peer engine sees the failure only on the wire.
                    room.fails = true;
                    events.push(PdEvent::Failed(rid.clone()));
                    false
                }
                TransferPoll::Bootstrapped if !*bootstrapped => {
                    *bootstrapped = true;
                    events.push(PdEvent::Bootstrapped(rid.clone()));
                    true
                }
                TransferPoll::Success => {
                    events.push(match role {
                        Role::Prefill => PdEvent::Succeeded(rid.clone()),
                        _ => PdEvent::RemotePrefillDone(RemotePrefill {
                            rid: rid.clone(),
                            bootstrap_token: room.sent.expect("sent with the KV"),
                            cached_tokens: 0,
                            bootstrap_logprob: None,
                            spec_candidate_ids: None,
                            request_pool_index: room
                                .preallocated
                                .expect("admitted before the KV moved"),
                        }),
                    });
                    false
                }
                _ => true,
            }
        });
        Ok(events)
    }

    fn record_usage(&mut self, _rid: &str, _cached_tokens: i64, _bootstrap_logprob: Option<f64>) {}

    fn has_in_flight(&self) -> bool {
        !self.tracked.is_empty()
    }

    fn abort_room(&mut self, _: &str, _: u64) {}
}

/// EPD: a `lost-*` request's receive fails at its second poll, a `slow-*`
/// one's lands after a hundred; the others' land after three polls, one more
/// on each later rank. Each landed item takes one slot of eight.
struct FakeReceiver {
    rank: usize,
    staged: Vec<(String, Vec<EmbeddingItem>, usize)>,
    free: Vec<usize>,
    awaiting: Vec<usize>,
}

impl EmbeddingReceiver for FakeReceiver {
    fn stage(&mut self, rid: &str, items: &[EmbeddingItem]) {
        assert!(!items.is_empty());
        self.staged.push((rid.to_owned(), items.to_vec(), 0));
    }

    fn poll(&mut self) -> Vec<i64> {
        let rank = self.rank;
        self.staged
            .iter_mut()
            .map(|(rid, _, polls)| {
                *polls += 1;
                let landed_at = if rid.starts_with("lost") {
                    return i64::from(*polls < 2);
                } else if rid.starts_with("slow") {
                    100
                } else {
                    3 + rank
                };
                if *polls >= landed_at {
                    2
                } else {
                    1
                }
            })
            .collect()
    }

    fn settle(&mut self, codes: &[i64]) -> Vec<LandedRequest> {
        assert_eq!(codes.len(), self.staged.len());
        let mut landed = Vec::new();
        let mut kept = Vec::new();
        for ((rid, items, polls), &code) in self.staged.drain(..).zip(codes) {
            match code {
                2 => landed.push(LandedRequest {
                    rid,
                    items: items
                        .into_iter()
                        .map(|item| {
                            let slot = self.free.pop().expect("a free slot");
                            self.awaiting.push(slot);
                            LandedItem {
                                index: item.index,
                                slot: Some(slot),
                                row_start: 0,
                                row_count: item.rows,
                                rows: item.rows,
                                sharded: false,
                            }
                        })
                        .collect(),
                }),
                1 => kept.push((rid, items, polls)),
                _ => {}
            }
        }
        self.staged = kept;
        landed
    }

    fn release(&mut self, slots: &[usize]) {
        for slot in slots {
            let position = self.awaiting.iter().position(|s| s == slot).unwrap();
            self.free.push(self.awaiting.remove(position));
        }
    }

    fn awaiting_release(&self) -> bool {
        !self.awaiting.is_empty()
    }
}

impl DataPlane for FakeDevice {
    type Pending = ForwardResult;

    fn admit(&mut self, requests: &[Admission]) -> Result<(), DataPlaneError> {
        for request in requests {
            self.generated.insert(request.rid.clone(), 0);
            if let Some(bootstrap) = &request.bootstrap {
                self.rooms.insert(request.rid.clone(), bootstrap.room);
            }
        }
        Ok(())
    }

    fn execute(
        &mut self,
        plan: ExecutionPlan,
        dispatch: &Dispatch<'_>,
    ) -> Result<Option<ForwardResult>, DataPlaneError> {
        let forward = dispatch.forward;
        if let Some(windows) = dispatch.ngram {
            let mut seen = self.ngram.lock().unwrap();
            seen.extend(
                windows
                    .positions
                    .iter()
                    .copied()
                    .zip(windows.tokens.iter().cloned()),
            );
        }
        if let Some(seeds) = dispatch.history {
            self.history.lock().unwrap().extend(
                seeds
                    .slots
                    .iter()
                    .copied()
                    .zip(seeds.tokens.iter().cloned()),
            );
        }
        if dispatch.dp.is_some_and(|dp| dp.idle_forward) {
            assert!(forward.is_none(), "an idle forward replaces the batch");
            *self.idle_forwards.lock().unwrap() += 1;
        }
        let ops = plan.cache_ops();
        self.plans.lock().unwrap().push((
            self.polls,
            forward.map_or_else(Vec::new, |batch| batch.request_ids.clone()),
            ops.clone(),
        ));
        self.unacknowledged
            .extend(ops.into_iter().map(|op| (self.ack_polls, op)));
        if let Some(wire) = &self.wire {
            let mut wire = wire.lock().unwrap();
            if let Some(admitted) = plan.remote_prefill() {
                for (rid, &slot) in admitted
                    .request_ids
                    .iter()
                    .zip(&admitted.request_pool_indices)
                {
                    wire.rooms.entry(self.rooms[rid]).or_default().preallocated = Some(slot);
                    // The prefill node's token comes first.
                    self.generated.insert(rid.clone(), 1);
                }
            }
            if let Some(prompts) = plan.remote_decode() {
                for rid in &prompts.request_ids {
                    let first = 1000 + self.generated[rid] - 1;
                    let room = wire.rooms.entry(self.rooms[rid]).or_default();
                    room.sent = Some(first);
                    room.sent_by.push(self.rank);
                }
            }
        }
        let Some(batch) = forward else {
            return Ok(None);
        };
        thread::sleep(FORWARD_TIME);
        let mut result = ForwardResult::default();
        for (i, rid) in batch.request_ids.iter().enumerate() {
            let mid_chunk = i < batch.num_extends
                && batch.extend_prefix_lens[i] + batch.input_lengths[i] < batch.prefill_lengths[i];
            let token = if self.placeholder_stage {
                // What a stage without the LM head reports.
                -7
            } else if mid_chunk {
                0
            } else if let Some(&handle) = self.grammars.get(rid) {
                // A constrained request samples the lowest token its grammar
                // allows, read through the entries the data plane uses.
                lowest_allowed(handle)
            } else {
                let generated = self.generated.entry(rid.clone()).or_default();
                *generated += 1;
                1000 + *generated - 1
            };
            result.output_tokens.push(token);
            result.output_lengths.push(1);
        }
        // Like the sampler, the device always reports each token's logprob:
        // here -token/1000.
        result.output_logprobs = Some(
            result
                .output_tokens
                .iter()
                .map(|&token| -f64::from(token) / 1000.0)
                .collect(),
        );
        if let Some(plan) = dispatch.logprobs {
            fake_logprobs(plan, &mut result);
        }
        Ok(Some(result))
    }

    fn complete(&mut self, pending: ForwardResult) -> Result<ForwardResult, DataPlaneError> {
        Ok(pending)
    }

    fn poll_cache_events(&mut self) -> Result<Vec<CacheEvent>, DataPlaneError> {
        self.polls += 1;
        let mut events = Vec::new();
        if !*self.ack_gate.open.lock().unwrap() {
            return Ok(events);
        }
        for (polls_left, op) in std::mem::take(&mut self.unacknowledged) {
            if polls_left > 0 {
                self.unacknowledged.push((polls_left - 1, op));
                continue;
            }
            let kind = match op.kind {
                CacheOpKind::WriteBack => CacheEventKind::WriteBackDone,
                CacheOpKind::LoadBack => CacheEventKind::LoadBackDone,
                CacheOpKind::SnapshotStore => CacheEventKind::SnapshotDone,
                CacheOpKind::SnapshotRestore => CacheEventKind::RestoreDone,
                kind => return Err(DataPlaneError(format!("no fake for {kind:?}"))),
            };
            self.acknowledged
                .lock()
                .unwrap()
                .push((self.polls, kind, op.op_id));
            events.push(CacheEvent {
                kind,
                op_id: op.op_id,
            });
        }
        Ok(events)
    }

    fn release(&mut self, rids: &[String]) -> Result<(), DataPlaneError> {
        let mut released = self.released.lock().unwrap();
        for rid in rids {
            let generated = self
                .generated
                .remove(rid)
                .expect("released once, after admission");
            released.push((rid.clone(), generated));
        }
        Ok(())
    }

    fn land(
        &mut self,
        spec_candidates: Option<(i32, Vec<i32>)>,
        cache_slot: Option<i32>,
    ) -> Result<(), DataPlaneError> {
        self.landed
            .lock()
            .unwrap()
            .push((spec_candidates, cache_slot));
        Ok(())
    }

    fn idle_forward(&mut self, _dp: &DpRound) -> Result<(), DataPlaneError> {
        *self.idle_forwards.lock().unwrap() += 1;
        Ok(())
    }

    fn update_weights(&mut self, op: &ControlRequest) -> Result<(bool, String), DataPlaneError> {
        self.device_ops.lock().unwrap().push(op.tag().to_owned());
        Ok((!self.fail_weight_update, "applied".to_owned()))
    }

    fn commit_weight_update(&mut self, success: bool) -> Result<(), DataPlaneError> {
        self.weight_commits.lock().unwrap().push(success);
        Ok(())
    }

    fn profile(&mut self, request: &ProfileReq) -> Result<(bool, String), DataPlaneError> {
        self.device_ops
            .lock()
            .unwrap()
            .push(format!("profile {}", request.kind));
        Ok((true, "Succeeded".to_owned()))
    }

    fn release_memory(&mut self, tags: &[String]) -> Result<(), DataPlaneError> {
        self.device_ops
            .lock()
            .unwrap()
            .push(format!("release {}", tags.join(",")));
        Ok(())
    }

    fn resume_memory(&mut self, tags: &[String]) -> Result<(), DataPlaneError> {
        self.device_ops
            .lock()
            .unwrap()
            .push(format!("resume {}", tags.join(",")));
        Ok(())
    }

    fn attach_grammars(
        &mut self,
        grammars: &[(String, u64, Vec<i32>)],
    ) -> Result<(), DataPlaneError> {
        for (rid, handle, _) in grammars {
            self.grammars.insert(rid.clone(), *handle);
        }
        Ok(())
    }

    fn epd_publish(&mut self, landed: &[LandedRequest]) -> Result<(), DataPlaneError> {
        for request in landed {
            self.epd_admitted.lock().unwrap().push(request.rid.clone());
            self.epd_copying
                .extend(request.items.iter().filter_map(|item| item.slot));
        }
        Ok(())
    }

    /// The published rows are copied out by the next query.
    fn epd_released(&mut self) -> Result<Vec<usize>, DataPlaneError> {
        let slots = std::mem::take(&mut self.epd_copying);
        self.epd_released.lock().unwrap().extend(&slots);
        Ok(slots)
    }

    #[allow(clippy::type_complexity)]
    fn snapshot_expert_load(&mut self) -> Result<(Vec<Vec<i64>>, Vec<Vec<i32>>), DataPlaneError> {
        let placement = self.placement.lock().unwrap().clone();
        // Expert 0 is hot, more so on rank 0's tokens.
        let counts = placement
            .iter()
            .map(|row| {
                row.iter()
                    .map(|&expert| {
                        if expert == 0 {
                            100 - 10 * self.ep_rank as i64
                        } else {
                            1
                        }
                    })
                    .collect()
            })
            .collect();
        Ok((counts, placement))
    }

    fn apply_expert_placement(
        &mut self,
        layer_ids: &[usize],
        rows: &[Vec<i32>],
        _: &[SlotMoves],
    ) -> Result<(), DataPlaneError> {
        let mut placement = self.placement.lock().unwrap();
        for (&layer, row) in layer_ids.iter().zip(rows) {
            placement[layer] = row.clone();
        }
        self.applied
            .lock()
            .unwrap()
            .push((layer_ids.to_vec(), rows.to_vec()));
        Ok(())
    }
}

fn ipc(name: &str) -> String {
    use std::sync::atomic::AtomicU64;
    static N: AtomicU64 = AtomicU64::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    format!(
        "ipc://{}",
        std::env::temp_dir()
            .join(format!("ts-control-{}-{name}-{n}.sock", std::process::id()))
            .display()
    )
}

/// A latch a store's queries wait on, or a device's cache acknowledgments.
#[derive(Default)]
struct Gate {
    open: Mutex<bool>,
    opened: std::sync::Condvar,
}

impl Gate {
    fn closed() -> Arc<Self> {
        Arc::new(Self::default())
    }

    fn opened() -> Arc<Self> {
        let gate = Self::default();
        *gate.open.lock().unwrap() = true;
        Arc::new(gate)
    }

    fn open(&self) {
        *self.open.lock().unwrap() = true;
        self.opened.notify_all();
    }

    fn pass(&self) {
        let open = self.open.lock().unwrap();
        drop(self.opened.wait_while(open, |open| !*open).unwrap());
    }
}

/// The L3 store behind one rank's executor (`Tiers::L3`): it holds every key
/// a query asks for, once `gate` lets the query through; a get lands its
/// leading `landed` keys (every key when `None`) and a put stores
/// everything. It counts the queries and gets it ran.
struct ScriptedStore {
    landed: Option<usize>,
    gate: Arc<Gate>,
    queries: Arc<Mutex<usize>>,
    gets: Arc<Mutex<usize>>,
}

impl Store for ScriptedStore {
    fn failures_retire(&self) -> bool {
        true
    }

    fn batch_exists(&self, keys: &[String]) -> Result<Vec<bool>, StoreError> {
        *self.queries.lock().unwrap() += 1;
        self.gate.pass();
        Ok(vec![true; keys.len()])
    }

    fn batch_get_into(
        &self,
        keys: &[String],
        _: HostBuffer,
        _: &[Range],
    ) -> Result<IoResult, StoreError> {
        *self.gets.lock().unwrap() += 1;
        let landed = self.landed.unwrap_or(keys.len());
        Ok(IoResult {
            success: (0..keys.len()).map(|key| key < landed).collect(),
            retired: true,
            object_failures: vec![false; keys.len()],
        })
    }

    fn batch_put_from(
        &self,
        keys: &[String],
        _: HostBuffer,
        _: &[Range],
    ) -> Result<IoResult, StoreError> {
        Ok(IoResult {
            success: vec![true; keys.len()],
            retired: true,
            object_failures: vec![false; keys.len()],
        })
    }

    fn remove_by_prefix(&self, _: &str) -> bool {
        true
    }

    fn close(&self) {}
}

/// `Tiers::L3`'s Host pool: the scheduler's 1024 Host pages, one 8-byte
/// block each, owned by every rank.
fn l3_layout(rank: usize) -> HostLayout {
    HostLayout {
        groups: vec![GroupLayout {
            packing: 1,
            block_bytes: 8,
            shard_count: 1,
        }],
        lcm_block_bytes: 8,
        num_lcm_blocks: 1024,
        owner_rank: 0,
        tp_rank: rank,
    }
}

/// One rank's L3 executor over `store` and a pool of its own, which lives
/// for the process.
fn l3_executor(store: ScriptedStore, rank: usize) -> Executor {
    let layout = l3_layout(rank);
    let pool: &'static mut [u8] = vec![0_u8; layout.bytes()].leak();
    Executor::start(
        Box::new(store),
        HostBuffer {
            base: pool.as_mut_ptr() as usize,
            len: pool.len(),
        },
        layout,
        Namespace {
            deployment_id: "serving".to_owned(),
            has_draft: false,
        },
        "v1",
        ExecutorConfig {
            max_operations: 8,
            max_keys_per_operation: 8,
            query_workers: 1,
            read_workers: 1,
            write_workers: 1,
            failure_threshold: 1,
            recovery_interval: Duration::from_secs(30),
        },
        monotonic_clock(),
    )
}

/// What a test reads back from one rank.
struct Rank {
    thread: Option<thread::JoinHandle<()>>,
    released: Arc<Mutex<Vec<(String, i32)>>>,
    plans: Arc<Mutex<Vec<ExecutedPlan>>>,
    /// What the rank's cache acknowledgments wait on.
    ack_gate: Arc<Gate>,
    /// The rank's cache acknowledgments: poll, kind, op id.
    acknowledged: Arc<Mutex<Vec<(usize, CacheEventKind, u32)>>>,
    idle_forwards: Arc<Mutex<usize>>,
    ngram: Arc<Mutex<Vec<NgramRow>>>,
    history: Arc<Mutex<Vec<HistorySeed>>>,
    /// L3: the queries and gets the rank's store ran.
    storage_queries: Arc<Mutex<usize>>,
    storage_gets: Arc<Mutex<usize>>,
    /// L3: what the rank's store queries wait on.
    storage_gate: Arc<Gate>,
    landed: Arc<Mutex<Vec<Landing>>>,
    device_ops: Arc<Mutex<Vec<String>>>,
    weight_commits: Arc<Mutex<Vec<bool>>>,
    applied: Applied,
    epd_admitted: Arc<Mutex<Vec<String>>>,
    epd_released: Arc<Mutex<Vec<usize>>>,
}

/// How the ranks split into schedulers.
#[derive(Debug, Clone, Copy)]
enum Layout {
    /// One rank.
    Single,
    /// One scheduler mirrored on this many attention-TP ranks.
    Tp(usize),
    /// One scheduler mirrored on this many attention-TP ranks (the first
    /// count), its KV cache split over this many shards (the second).
    TpKvp(usize, usize),
    /// This many attention-DP ranks, one scheduler and frontend each.
    Dp(usize),
    /// One scheduler mirrored on this many pipeline stages; the last samples.
    Pp(usize),
}

struct Engine {
    client: EngineClient,
    stop: Arc<AtomicBool>,
    ranks: Vec<Rank>,
}

impl Engine {
    /// The queries rank 0's store ran.
    fn storage_queries(&self) -> usize {
        *self.ranks[0].storage_queries.lock().unwrap()
    }

    /// The gets rank 0's store ran.
    fn storage_gets(&self) -> usize {
        *self.ranks[0].storage_gets.lock().unwrap()
    }

    fn cache_ops(&self, kind: CacheOpKind) -> usize {
        self.ranks[0]
            .plans
            .lock()
            .unwrap()
            .iter()
            .flat_map(|(_, _, ops)| ops)
            .filter(|op| op.kind == kind)
            .count()
    }

    fn released(&self) -> Vec<(String, i32)> {
        self.released_on(0)
    }

    fn released_on(&self, rank: usize) -> Vec<(String, i32)> {
        let mut released = self.ranks[rank].released.lock().unwrap().clone();
        released.sort();
        released
    }

    fn idle_forwards(&self, rank: usize) -> usize {
        *self.ranks[rank].idle_forwards.lock().unwrap()
    }

    /// Wait until every rank of a mirrored replica (TP, PP) released the
    /// requests the frontend's rank did: they end each request in the same
    /// round, but their threads may get there after the client saw it end.
    async fn settle(&self) {
        let rids = |rank: usize| -> Vec<String> {
            self.released_on(rank)
                .into_iter()
                .map(|(rid, _)| rid)
                .collect()
        };
        for _ in 0..500 {
            let first = rids(0);
            if (1..self.ranks.len()).all(|rank| rids(rank) == first) {
                return;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        // A store query still waiting would hold its executor's close.
        for rank in &self.ranks {
            rank.storage_gate.open();
        }
        for rank in &mut self.ranks {
            if let Some(thread) = rank.thread.take() {
                thread.join().unwrap();
            }
        }
    }
}

async fn start_engine(in_flight_depth: usize) -> Engine {
    start_engine_with(in_flight_depth, Tiers::DeviceOnly).await
}

async fn start_engine_with(in_flight_depth: usize, tiers: Tiers) -> Engine {
    start_ranks(Layout::Single, in_flight_depth, tiers).await
}

async fn start_ranks(layout: Layout, in_flight_depth: usize, tiers: Tiers) -> Engine {
    start_ranks_with(layout, in_flight_depth, tiers, Setup::default()).await
}

/// The rebalance the EPLB tests run: two MoE layers of four experts on six
/// slots, a snapshot every few forwards, one layer moved per round.
fn eplb_config(ep_size: usize, ep_rank: usize) -> EplbConfig {
    EplbConfig {
        specs: EplbSpecs {
            num_layers: 2,
            num_logical_experts: 4,
            num_physical_experts: 6,
            ep_size,
            ep_rank,
            ep_rank_nodes: vec![0; ep_size],
            all_to_all_ep: true,
            num_groups: None,
            num_nodes: 1,
        },
        rebalance_num_iterations: 4,
        layers_per_chunk: 1,
        algorithm: Algorithm::Deepseek,
        commit_delay_forwards: 3,
    }
}

/// A PD engine's half: its role and the wire to its peer engine.
#[derive(Clone)]
struct Pd {
    role: Role,
    wire: Arc<Mutex<Wire>>,
}

/// What an engine runs with beyond its layout, depth and cache tiers.
#[derive(Default, Clone)]
struct Setup {
    /// Predecessors Engram reads per token.
    ngram_context_len: usize,
    /// PD: the engine's role and its peer's wire.
    pd: Option<Pd>,
    /// The expert rebalance's settings per (ranks, rank).
    eplb: Option<fn(usize, usize) -> EplbConfig>,
    /// An EPD prefill node.
    epd_receive_enabled: bool,
    /// Grammar-constrained decoding over the toy vocabulary.
    grammar: bool,
    /// Grammar (`Setup::grammar`): the compile timeouts each rank recorded
    /// for [`TIMED_OUT_GRAMMAR`] before serving.
    grammar_timeouts: Option<fn(usize) -> u32>,
    /// `--enable-output-logprobs`.
    logprobs: bool,
    /// The model reads request token history.
    request_token_history: bool,
    /// L3 (`Tiers::L3`): the leading keys of a get that land on each rank;
    /// all of them when unset.
    l3_landed_keys: Option<fn(usize) -> usize>,
    /// L3: the stores' queries wait until the test opens their gates.
    l3_held_queries: bool,
    /// The polls each rank's cache ops wait after the first before they are
    /// acknowledged; none when unset.
    cache_ack_polls: Option<fn(usize) -> usize>,
    /// The devices acknowledge no cache op until the test opens their gates.
    held_cache_acks: bool,
    /// A rank whose device refuses a weight load.
    weight_update_failure_rank: Option<usize>,
}

impl Layout {
    /// Ranks, attention-TP ranks per replica, and engines.
    fn sizes(self) -> (usize, usize, usize) {
        match self {
            Layout::Single => (1, 1, 1),
            Layout::Tp(size) | Layout::TpKvp(size, _) | Layout::Pp(size) => (size, size, 1),
            Layout::Dp(size) => (size, 1, size),
        }
    }
}

/// The loop's settings for `setup`.
fn loop_config(in_flight_depth: usize, pipeline: bool, setup: &Setup) -> LoopConfig {
    let pd = setup.pd.as_ref();
    LoopConfig {
        in_flight_depth,
        limits: IngressLimits {
            vocab_size: 100_000,
            eos_token_ids: Vec::new(),
            max_model_len: 4096,
            max_req_input_len: 4000,
            enable_output_logprobs: setup.logprobs,
            requires_bootstrap: pd.is_some(),
            multimodal: setup.epd_receive_enabled,
            grammar: setup.grammar,
            reasoning: None,
            logprobs: LogprobServability {
                supports_prompt_logprobs: true,
                logprob_order: "torch".to_owned(),
                pipeline_parallel: pipeline,
                disaggregated: pd.is_some(),
                lm_head_tp_under_dp: false,
                computes_prompt_logprobs: pd.is_none_or(|pd| pd.role != Role::Decode),
            },
        },
        kv_total_pages: 512,
        ngram_context_len: setup.ngram_context_len,
        request_token_history: setup.request_token_history,
        controls: ControlConfig {
            memory_saver: true,
            memory_regions: vec!["kv_cache".to_owned(), "weights".to_owned()],
            l3_weight_version: None,
        },
        batch_log: BatchLogConfig {
            enabled: true,
            decode_log_interval: 40,
            num_total_pages: 512,
            spec_num_steps: 0,
            dp_rank: 0,
            disaggregated: pd.is_some(),
            cache_state_group_ids: Vec::new(),
            log_spec_accept_lengths: false,
        },
        kv_events: false,
        epd_receive_enabled: setup.epd_receive_enabled,
        // The fake device fills bitmasks from the committed matcher state,
        // as the eager path does.
        eager_grammar: setup.grammar,
    }
}

/// What one rank's loop is built from besides its device, store and
/// frontend.
struct RankSpec {
    rank: usize,
    layout: Layout,
    tiers: Tiers,
    config: LoopConfig,
    setup: Setup,
    mesh_listener: Listener,
    addresses: Vec<String>,
}

/// Connect `spec`'s rank to the others, build its loop over `device`,
/// `store` and the frontend `frontend` makes, and run it until the ranks
/// stop.
fn run_rank<F: Frontend>(
    spec: RankSpec,
    device: FakeDevice,
    store: Option<ScriptedStore>,
    frontend: impl FnOnce() -> F,
    stop: &AtomicBool,
) -> Result<(), LoopError> {
    let RankSpec {
        rank,
        layout,
        tiers,
        config,
        setup,
        mesh_listener,
        addresses,
    } = spec;
    let (world, tp_size, num_engines) = layout.sizes();
    let mesh = Mesh::connect(rank, mesh_listener, &addresses, Duration::from_secs(20)).unwrap();
    let tp_first = rank / tp_size * tp_size;
    let replica = Group::new(1, (tp_first..tp_first + tp_size).collect(), rank).unwrap();
    let dp = (num_engines > 1).then(|| Group::new(0, (0..world).collect(), rank).unwrap());
    let frontend = frontend();
    let Setup {
        pd,
        eplb,
        grammar,
        grammar_timeouts,
        epd_receive_enabled,
        ..
    } = setup;
    let role = pd.as_ref().map_or(Role::Plain, |pd| pd.role);
    let outputs = OutputProcessor::new(
        None,
        1,
        None,
        role,
        FinishLog {
            enabled: true,
            request_stats: true,
        },
    );
    let transfer = pd.map(|pd| {
        Box::new(FakePeer {
            role: pd.role,
            rank,
            wire: pd.wire,
            tracked: Vec::new(),
        }) as Box<dyn TransferPeer + Send>
    });
    let pp =
        matches!(layout, Layout::Pp(_)).then(|| Group::new(2, (0..world).collect(), rank).unwrap());
    // Expert parallelism over every rank.
    let ep = eplb
        .filter(|_| world > 1)
        .map(|_| Group::new(3, (0..world).collect(), rank).unwrap());
    let ranks = Ranks {
        mesh,
        replica,
        pp,
        dp,
        ep,
    };
    let kv_shard_count = match layout {
        Layout::TpKvp(_, shards) => shards,
        _ => 1,
    };
    let mut event_loop = EventLoop::new(
        new_scheduler(tiers, role, kv_shard_count),
        device,
        frontend,
        outputs,
        config,
        ranks,
        Subsystems {
            transfer,
            embeddings: epd_receive_enabled.then(|| {
                Box::new(FakeReceiver {
                    rank,
                    staged: Vec::new(),
                    free: (0..8).collect(),
                    awaiting: Vec::new(),
                }) as Box<dyn EmbeddingReceiver + Send>
            }),
            metrics: None,
            kv_events: None,
            eplb: eplb.map(|config| Controller::new(config(world, rank)).unwrap()),
            grammars: grammar.then(|| {
                let mut grammars = toy_grammars();
                let key = GrammarKey {
                    kind: "regex",
                    source: TIMED_OUT_GRAMMAR.to_owned(),
                };
                for _ in 0..grammar_timeouts.map_or(0, |timeouts| timeouts(rank)) {
                    grammars.record_timeout(&key, "timed out");
                }
                grammars
            }),
            l3_metrics: None,
            l3: store.map(|store| l3_executor(store, rank)),
        },
    );
    event_loop.run(stop)
}

/// A rank's device as the tests start it.
fn fake_device(layout: Layout, rank: usize, setup: &Setup) -> FakeDevice {
    let (world, _, _) = layout.sizes();
    FakeDevice {
        rank,
        ack_polls: setup.cache_ack_polls.map_or(0, |polls| polls(rank)),
        ack_gate: if setup.held_cache_acks {
            Gate::closed()
        } else {
            Gate::opened()
        },
        placeholder_stage: matches!(layout, Layout::Pp(_)) && rank + 1 < world,
        wire: setup.pd.as_ref().map(|pd| pd.wire.clone()),
        ep_rank: rank,
        fail_weight_update: setup.weight_update_failure_rank == Some(rank),
        // The trivial initial placement: slot p holds expert p % 4.
        placement: Arc::new(Mutex::new(vec![vec![0, 1, 2, 3, 0, 1]; 2])),
        ..Default::default()
    }
}

/// Listeners for `world` ranks' mesh and the addresses they dial.
fn mesh_listeners(world: usize) -> (Vec<Listener>, Vec<String>) {
    let listeners: Vec<Listener> = (0..world)
        .map(|_| Listener::bind("127.0.0.1").unwrap())
        .collect();
    let addresses = listeners
        .iter()
        .map(|listener| format!("127.0.0.1:{}", listener.port().unwrap()))
        .collect();
    (listeners, addresses)
}

async fn start_ranks_with(
    layout: Layout,
    in_flight_depth: usize,
    tiers: Tiers,
    setup: Setup,
) -> Engine {
    let (world, tp_size, num_engines) = layout.sizes();
    let pipeline = matches!(layout, Layout::Pp(_));
    let config = loop_config(in_flight_depth, pipeline, &setup);
    let listener = ClientListener::bind(ClientConfig {
        handshake_address: ipc("handshake"),
        input_address: ipc("input"),
        output_address: ipc("output"),
        first_engine_index: 0,
        num_engines: num_engines as u16,
        routing: Routing::Aggregated(LoadBalance::RoundRobin),
        handshake_timeout: Duration::from_secs(20),
        send_timeout: Duration::from_secs(5),
    })
    .await
    .unwrap();
    let handshake = listener.handshake_address().to_owned();
    let stop = Arc::new(AtomicBool::new(false));
    let (mesh_listeners, addresses) = mesh_listeners(world);
    let ranks = mesh_listeners
        .into_iter()
        .enumerate()
        .map(|(rank, mesh_listener)| {
            let device = fake_device(layout, rank, &setup);
            let storage_gate = if setup.l3_held_queries {
                Gate::closed()
            } else {
                Gate::opened()
            };
            let store = matches!(tiers, Tiers::L3).then(|| ScriptedStore {
                landed: setup.l3_landed_keys.map(|landed| landed(rank)),
                gate: storage_gate.clone(),
                queries: Arc::new(Mutex::new(0)),
                gets: Arc::new(Mutex::new(0)),
            });
            let handle = Rank {
                thread: None,
                released: device.released.clone(),
                plans: device.plans.clone(),
                ack_gate: device.ack_gate.clone(),
                acknowledged: device.acknowledged.clone(),
                idle_forwards: device.idle_forwards.clone(),
                ngram: device.ngram.clone(),
                history: device.history.clone(),
                storage_queries: store
                    .as_ref()
                    .map_or_else(Default::default, |store| store.queries.clone()),
                storage_gets: store
                    .as_ref()
                    .map_or_else(Default::default, |store| store.gets.clone()),
                storage_gate,
                landed: device.landed.clone(),
                device_ops: device.device_ops.clone(),
                weight_commits: device.weight_commits.clone(),
                applied: device.applied.clone(),
                epd_admitted: device.epd_admitted.clone(),
                epd_released: device.epd_released.clone(),
            };
            let spec = RankSpec {
                rank,
                layout,
                tiers,
                config: config.clone(),
                setup: setup.clone(),
                mesh_listener,
                addresses: addresses.clone(),
            };
            let (handshake, stop) = (handshake.clone(), stop.clone());
            let thread = thread::spawn(move || {
                // The TP group's first member owns the frontend; each
                // attention-DP rank is its own engine.
                let frontend = || {
                    (rank % tp_size == 0).then(|| {
                        let engine_index = (rank / tp_size) as u16;
                        let mut ready = EngineCoreReadyResponse {
                            max_model_len: 4096,
                            data_parallel_size: num_engines as i64,
                            data_parallel_rank: i64::from(engine_index),
                            ..Default::default()
                        };
                        // As `ts-host` registers a scheduler.
                        ready.declare_scheduler();
                        EngineTransport::connect(
                            &handshake,
                            engine_index,
                            ready,
                            Duration::from_secs(20),
                        )
                        .unwrap()
                    })
                };
                run_rank(spec, device, store, frontend, &stop).unwrap();
            });
            Rank {
                thread: Some(thread),
                ..handle
            }
        })
        .collect();
    Engine {
        client: listener.accept().await.unwrap(),
        stop,
        ranks,
    }
}

/// What a scripted frontend hands its engine: one batch of messages per
/// round.
type Script = Vec<Vec<EngineRequest>>;

/// One engine's frontend in a scripted run: each round it hands the engine
/// the next batch of its script, after the messages `react` made of the
/// previous round's outputs. Once its script ran out it is gone, so the
/// engine stops.
struct ScriptedFrontend {
    script: VecDeque<Vec<EngineRequest>>,
    react: fn(&[StreamedOutput]) -> Vec<EngineRequest>,
    reacted: Vec<EngineRequest>,
    replies: Arc<Mutex<Vec<ControlOutput>>>,
    /// The requests that finished, with their reasons.
    finished: Arc<Mutex<Vec<(String, String)>>>,
}

impl Frontend for ScriptedFrontend {
    fn receive(&mut self, _wait: Option<Duration>) -> Vec<EngineRequest> {
        let mut messages = std::mem::take(&mut self.reacted);
        messages.extend(self.script.pop_front().unwrap_or_default());
        messages
    }

    fn closed(&self) -> bool {
        self.script.is_empty() && self.reacted.is_empty()
    }

    fn send(&mut self, outputs: Vec<StreamedOutput>, _load: Load) {
        self.reacted.extend((self.react)(&outputs));
        self.finished
            .lock()
            .unwrap()
            .extend(outputs.iter().filter_map(|output| {
                output
                    .finish_reason
                    .map(|reason| (output.rid.clone(), reason.to_owned()))
            }));
    }

    fn reply(&mut self, output: ControlOutput) {
        self.replies.lock().unwrap().push(output);
    }
}

/// What a scripted run reads back from one rank.
struct ScriptedRank {
    outcome: Result<(), String>,
    /// Its engine's control replies, on the engine's first rank.
    replies: Vec<ControlOutput>,
    finished: Vec<(String, String)>,
    cache_ops: Vec<CacheOpKind>,
    idle_forwards: usize,
}

/// Run `layout`'s ranks, one script per engine (its first rank's frontend
/// plays it, reacting to outputs with `react`), until a script runs out and
/// the ranks stop by themselves.
fn run_scripted(
    layout: Layout,
    tiers: Tiers,
    setup: Setup,
    scripts: Vec<Script>,
    react: fn(&[StreamedOutput]) -> Vec<EngineRequest>,
) -> Vec<ScriptedRank> {
    let (world, tp_size, num_engines) = layout.sizes();
    assert_eq!(scripts.len(), num_engines);
    let config = loop_config(0, matches!(layout, Layout::Pp(_)), &setup);
    let stop = Arc::new(AtomicBool::new(false));
    let (mesh_listeners, addresses) = mesh_listeners(world);
    let mut scripts = scripts.into_iter();
    let (done, outcomes) = std::sync::mpsc::channel();
    let ranks: Vec<_> = mesh_listeners
        .into_iter()
        .enumerate()
        .map(|(rank, mesh_listener)| {
            let device = fake_device(layout, rank, &setup);
            let (plans, idle_forwards) = (device.plans.clone(), device.idle_forwards.clone());
            let replies = Arc::new(Mutex::new(Vec::new()));
            let finished = Arc::new(Mutex::new(Vec::new()));
            let frontend = (rank % tp_size == 0).then(|| ScriptedFrontend {
                script: scripts.next().unwrap().into(),
                react,
                reacted: Vec::new(),
                replies: replies.clone(),
                finished: finished.clone(),
            });
            let spec = RankSpec {
                rank,
                layout,
                tiers,
                config: config.clone(),
                setup: setup.clone(),
                mesh_listener,
                addresses: addresses.clone(),
            };
            let (stop, done) = (stop.clone(), done.clone());
            let thread = thread::spawn(move || {
                let outcome = run_rank(spec, device, None, || frontend, &stop)
                    .map_err(|error| error.to_string());
                done.send(rank).unwrap();
                outcome
            });
            (thread, replies, finished, plans, idle_forwards)
        })
        .collect();
    for _ in 0..world {
        if outcomes.recv_timeout(Duration::from_secs(60)).is_err() {
            stop.store(true, Ordering::Relaxed);
            panic!("the scripted ranks did not stop by themselves");
        }
    }
    ranks
        .into_iter()
        .map(
            |(thread, replies, finished, plans, idle_forwards)| ScriptedRank {
                outcome: thread.join().unwrap(),
                replies: replies.lock().unwrap().clone(),
                finished: finished.lock().unwrap().clone(),
                cache_ops: plans
                    .lock()
                    .unwrap()
                    .iter()
                    .flat_map(|(_, _, ops)| ops.iter().map(|op| op.kind))
                    .collect(),
                idle_forwards: *idle_forwards.lock().unwrap(),
            },
        )
        .collect()
}

/// A scripted frontend that does not react to outputs.
fn no_reaction(_: &[StreamedOutput]) -> Vec<EngineRequest> {
    Vec::new()
}

fn request(rid: &str, prompt_len: usize, params: SamplingParams) -> TokenizedGenerateReqInput {
    TokenizedGenerateReqInput {
        input_ids: Some((1..=prompt_len as i32).collect()),
        stream: true,
        ..TokenizedGenerateReqInput::new(rid, params)
    }
}

fn budget(max_new_tokens: i64) -> SamplingParams {
    SamplingParams {
        max_new_tokens: Some(max_new_tokens),
        ..Default::default()
    }
}

async fn collect(mut stream: RequestStream) -> Vec<OutputChunk> {
    let mut chunks = Vec::new();
    while let Some(chunk) = tokio::time::timeout(Duration::from_secs(10), stream.next())
        .await
        .unwrap()
    {
        chunks.push(chunk.unwrap());
    }
    chunks
}

async fn run(client: &EngineClient, request: TokenizedGenerateReqInput) -> Vec<OutputChunk> {
    collect(client.generate(request).await.unwrap()).await
}

fn tokens(chunks: &[OutputChunk]) -> Vec<i32> {
    chunks.iter().flat_map(|c| c.output_ids.clone()).collect()
}

#[tokio::test(flavor = "multi_thread")]
async fn idle_load_reports_preserve_resident_cache_capacity() {
    let engine = start_engine(0).await;
    let snapshots = tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            let snapshots = engine.client.fresh_loads(Duration::from_secs(3));
            if !snapshots.is_empty() {
                break snapshots;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    assert_eq!(snapshots.len(), 1);
    assert_eq!(snapshots[0].0, 0);
    assert_eq!(snapshots[0].1.kv_total_pages, 512);
    assert_eq!(snapshots[0].1.kv_used_pages, 0);

    run(&engine.client, request("cached", 32, budget(1))).await;
    tokio::time::sleep(Duration::from_millis(1100)).await;
    let snapshots = engine.client.fresh_loads(Duration::from_millis(500));
    assert_eq!(snapshots.len(), 1, "idle reports keep capacity fresh");
    let load = snapshots[0].1;
    assert_eq!((load.num_running, load.num_waiting), (0, 0));
    assert_eq!(load.kv_active_pages, 0);
    assert!(load.kv_used_pages > 0, "reclaimable prefixes stay resident");
    assert_eq!(load.kv_total_pages, 512);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_request_generates_its_budget() {
    for depth in [0, 1] {
        let engine = start_engine(depth).await;
        let chunks = run(&engine.client, request("r1", 5, budget(3))).await;
        assert_eq!(tokens(&chunks), [1000, 1001, 1002], "depth {depth}");
        let last = chunks.last().unwrap();
        assert_eq!(last.finish_reason.as_deref(), Some("length"));
        assert_eq!((last.prompt_tokens, last.completion_tokens), (5, 3));
        // Release precedes the final output. The overlap schedule ran one
        // more decode before the finish committed.
        assert_eq!(
            engine.released(),
            [("r1".to_owned(), 3 + depth as i32)],
            "depth {depth}"
        );
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_stop_token_ends_a_request() {
    let engine = start_engine(1).await;
    let params = SamplingParams {
        stop_token_ids: Some(vec![1002]),
        ..budget(10)
    };
    let chunks = run(&engine.client, request("r1", 3, params)).await;
    assert_eq!(tokens(&chunks), [1000, 1001, 1002]);
    assert_eq!(
        chunks.last().unwrap().finish_reason.as_deref(),
        Some("stop")
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn a_long_prompt_is_prefilled_in_chunks() {
    let engine = start_engine(1).await;
    let chunks = run(
        &engine.client,
        request("long", MAX_SCHEDULED_TOKENS * 2 + 10, budget(2)),
    )
    .await;
    // The garbage tokens of the incomplete chunks never reach the client.
    assert_eq!(tokens(&chunks), [1000, 1001]);
}

#[tokio::test(flavor = "multi_thread")]
async fn concurrent_requests_each_get_their_own_tokens() {
    let engine = start_engine(1).await;
    // Submit all before reading any, so they share forwards.
    let mut streams = Vec::new();
    for i in 0..8 {
        let request = request(&format!("r{i}"), 4 + i, budget(2 + i as i64));
        streams.push(engine.client.generate(request).await.unwrap());
    }
    for (i, stream) in streams.into_iter().enumerate() {
        let expected: Vec<i32> = (1000..1000 + 2 + i as i32).collect();
        assert_eq!(tokens(&collect(stream).await), expected, "r{i}");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn an_invalid_request_finishes_with_an_abort() {
    let engine = start_engine(1).await;
    let params = SamplingParams {
        top_p: 0.0,
        ..budget(3)
    };
    let chunks = run(&engine.client, request("bad", 3, params)).await;
    assert!(tokens(&chunks).is_empty());
    assert_eq!(
        chunks.last().unwrap().finish_reason.as_deref(),
        Some("abort")
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn dropping_a_stream_aborts_the_request() {
    let engine = start_engine(1).await;
    let mut stream = engine
        .client
        .generate(request("r1", 3, budget(100_000)))
        .await
        .unwrap();
    let first = tokio::time::timeout(Duration::from_secs(10), stream.next())
        .await
        .unwrap();
    assert!(first.unwrap().is_ok());
    drop(stream);
    // The engine finishes the request at its next commit and releases it,
    // long before its budget runs out.
    for _ in 0..200 {
        if let [(rid, generated)] = engine.released().as_slice() {
            assert_eq!(rid, "r1");
            assert!(
                *generated < 100,
                "generated {generated} tokens after the abort"
            );
            // Cancellation produces no final output for the dropped stream,
            // but the frontend must still learn that the engine is idle.
            if engine
                .client
                .loads()
                .iter()
                .all(|load| load.num_running == 0 && load.num_waiting == 0)
            {
                return;
            }
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("the aborted request was not released and reported idle");
}

/// Submit every request, then read each stream to its end.
async fn run_all(
    client: &EngineClient,
    requests: Vec<TokenizedGenerateReqInput>,
) -> Vec<Vec<OutputChunk>> {
    let mut streams = Vec::new();
    for request in requests {
        streams.push(client.generate(request).await.unwrap());
    }
    let mut results = Vec::new();
    for stream in streams {
        results.push(collect(stream).await);
    }
    results
}

#[tokio::test(flavor = "multi_thread")]
async fn prompt_logprobs_and_alternatives_reach_the_frontend() {
    for (layout, depth) in [(Layout::Single, 0), (Layout::Single, 1), (Layout::Tp(2), 1)] {
        let engine = start_ranks_with(
            layout,
            depth,
            Tiers::DeviceOnly,
            Setup {
                logprobs: true,
                ..Setup::default()
            },
        )
        .await;
        // A prompt over three chunks, scored from position 100 on, its one
        // sampled token with two alternatives.
        let prompt_len = MAX_SCHEDULED_TOKENS * 2 + 88;
        let scored = |rid: &str, top_logprobs_num: i64| TokenizedGenerateReqInput {
            return_logprob: true,
            logprob_start_len: 100,
            top_logprobs_num,
            ..request(rid, prompt_len, budget(1))
        };
        // The default start: the last position, no prompt rows to compute.
        let sampled = TokenizedGenerateReqInput {
            return_logprob: true,
            ..request("sampled", 5, budget(3))
        };
        let results = run_all(
            &engine.client,
            vec![scored("scored", 2), sampled, scored("again", 0)],
        )
        .await;

        let chunks = &results[0];
        let with_prompt: Vec<_> = chunks
            .iter()
            .filter(|chunk| chunk.input_token_logprobs.is_some())
            .collect();
        assert_eq!(
            with_prompt.len(),
            1,
            "{layout:?} depth {depth}: shipped once"
        );
        let (values, ids) = with_prompt[0].input_token_logprobs.clone().unwrap();
        let expected: Vec<Option<f64>> = std::iter::once(None)
            .chain((100..prompt_len as i64 - 1).map(|p| Some(fake_prompt_logprob(p))))
            .collect();
        assert_eq!(values, expected);
        assert_eq!(ids, (101..=prompt_len as i32).collect::<Vec<_>>());
        let (top_values, top_ids) = with_prompt[0].input_top_logprobs.clone().unwrap();
        assert_eq!((top_values[0].clone(), top_ids[0].clone()), (None, None));
        assert_eq!(
            (top_values[1].clone(), top_ids[1].clone()),
            (Some(vec![-10.1, -11.1]), Some(vec![102, 901]))
        );
        let last = chunks.last().unwrap();
        assert_eq!(last.output_ids, [1000]);
        assert_eq!(last.output_token_logprobs, [-1.0]);
        assert_eq!(
            last.output_top_logprobs,
            Some((vec![vec![-1.0, -2.0]], vec![vec![1000, 901]]))
        );
        let finish = last.finish_details.as_ref().unwrap();
        assert_eq!(finish.0[0], ("type".to_owned(), "length".into()));

        // The default start ships `(None, last token)` and computes nothing.
        let chunks = &results[1];
        assert_eq!(tokens(chunks), [1000, 1001, 1002]);
        let (values, ids) = chunks
            .iter()
            .find_map(|chunk| chunk.input_token_logprobs.clone())
            .unwrap();
        assert_eq!((values, ids), (vec![None], vec![5]));
        assert!(chunks
            .iter()
            .all(|chunk| chunk.output_top_logprobs.is_none()));
        let logprobs: Vec<f64> = chunks
            .iter()
            .flat_map(|chunk| chunk.output_token_logprobs.clone())
            .collect();
        assert_eq!(logprobs, [-1.0, -1.001, -1.002]);

        // The prefix cache serves at most the positions before the start.
        let again = &results[2];
        let (values, _) = again
            .iter()
            .find_map(|chunk| chunk.input_token_logprobs.clone())
            .unwrap();
        assert_eq!(values, expected);
        let cached = again.last().unwrap().cached_tokens;
        assert!(
            cached <= 100,
            "{layout:?} depth {depth}: {cached} cached tokens"
        );
        drop(engine);
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn prompt_logprobs_reach_the_frontend_from_the_last_stage() {
    let engine = start_ranks_with(
        Layout::Pp(2),
        2,
        Tiers::DeviceOnly,
        Setup {
            logprobs: true,
            ..Setup::default()
        },
    )
    .await;
    // A prompt over three chunks, scored from position 100 on.
    let prompt_len = MAX_SCHEDULED_TOKENS * 2 + 88;
    let results = run_all(
        &engine.client,
        vec![TokenizedGenerateReqInput {
            return_logprob: true,
            logprob_start_len: 100,
            ..request("scored", prompt_len, budget(2))
        }],
    )
    .await;
    let chunks = &results[0];
    assert_eq!(tokens(chunks), [1000, 1001]);
    let (values, ids) = chunks
        .iter()
        .find_map(|chunk| chunk.input_token_logprobs.clone())
        .unwrap();
    let expected: Vec<Option<f64>> = std::iter::once(None)
        .chain((100..prompt_len as i64 - 1).map(|p| Some(fake_prompt_logprob(p))))
        .collect();
    assert_eq!(values, expected);
    assert_eq!(ids, (101..=prompt_len as i32).collect::<Vec<_>>());
}

#[tokio::test(flavor = "multi_thread")]
async fn token_histories_are_seeded_where_the_row_lost_them() {
    for depth in [0, 1] {
        let engine = start_ranks_with(
            Layout::Single,
            depth,
            Tiers::DeviceOnly,
            Setup {
                request_token_history: true,
                ..Setup::default()
            },
        )
        .await;
        // A fresh request starts its row at position 0, and its decodes
        // keep extending it: nothing to seed.
        run(&engine.client, request("first", 40, budget(3))).await;
        assert!(engine.ranks[0].history.lock().unwrap().is_empty());
        // The same prompt again hits the prefix cache: its prefill starts
        // past position 0, so it seeds the cached prefix.
        run(&engine.client, request("second", 40, budget(3))).await;
        let seeds = engine.ranks[0].history.lock().unwrap().clone();
        assert_eq!(seeds.len(), 1, "depth {depth}: {seeds:?}");
        assert_eq!(seeds[0].1, (1..=32).collect::<Vec<i32>>());
        drop(engine);
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn retracted_requests_resume_where_they_stopped() {
    // Also on mirrored ranks, the second of which acknowledges every cache
    // op three polls after the first, and with a Host tier written back to.
    for tiers in [
        Tiers::ForcedRetraction,
        Tiers::HostL2WriteBackForcedRetraction,
    ] {
        let write_back = matches!(tiers, Tiers::HostL2WriteBackForcedRetraction);
        for layout in [Layout::Single, Layout::Tp(2), Layout::TpKvp(2, 2)] {
            for depth in [0, 1] {
                let engine = start_ranks_with(
                    layout,
                    depth,
                    tiers,
                    Setup {
                        cache_ack_polls: Some(|rank| 3 * rank),
                        ..Setup::default()
                    },
                )
                .await;
                let case = format!("{layout:?} at depth {depth}, write-back {write_back}");
                // Under write-back, finished prompts of 2 full pages and a
                // tail each first fill the Device's 64 blocks with cache (the
                // last one evicts the first), so the restores below evict
                // some of it.
                let warm_prompts = if write_back { 22 } else { 0 };
                let warm: Vec<_> = (1..=warm_prompts)
                    .map(|k| TokenizedGenerateReqInput {
                        input_ids: Some((1000 * k + 1..=1000 * k + 40).collect()),
                        ..request(&format!("w{k}"), 1, budget(1))
                    })
                    .collect();
                for prompt in &warm {
                    let chunks = run(&engine.client, prompt.clone()).await;
                    assert_eq!(tokens(&chunks), [1000], "{case}");
                }
                let requests = (0..4)
                    .map(|i| request(&format!("r{i}"), 20 + i, budget(12 + i as i64)))
                    .collect();
                let results = run_all(&engine.client, requests).await;
                for (i, chunks) in results.iter().enumerate() {
                    let expected: Vec<i32> = (1000..1012 + i as i32).collect();
                    assert_eq!(tokens(chunks), expected, "r{i} on {case}");
                }
                let stores = engine.cache_ops(CacheOpKind::SnapshotStore);
                assert!(stores > 0, "nothing was retracted on {case}");
                assert_eq!(
                    engine.cache_ops(CacheOpKind::SnapshotRestore),
                    stores,
                    "{case}"
                );
                if !write_back {
                    continue;
                }
                engine.settle().await;
                // Under write-back a plan's WriteBack ops are one per image it
                // stores to the Host and at most one for the cache it evicts:
                // more of them than SnapshotStore ops write back evicted
                // cache. Some plan restores and writes back what it evicted,
                // on every rank.
                for (rank, member) in engine.ranks.iter().enumerate() {
                    let plans = member.plans.lock().unwrap();
                    assert!(
                        plans.iter().any(|(_, _, ops)| {
                            let count = |kind| ops.iter().filter(|op| op.kind == kind).count();
                            count(CacheOpKind::SnapshotRestore) > 0
                                && count(CacheOpKind::WriteBack) > count(CacheOpKind::SnapshotStore)
                        }),
                        "rank {rank}, {case}: no restore wrote back what it evicted"
                    );
                }
                // What the restores evicted is on the Host: every warm prompt
                // sent again hits its 2 full pages, on the Device or loaded
                // back. Each one waits for the acknowledgments of the cache
                // ops before it, so nothing it hits is still in flight.
                let mut cached = Vec::new();
                for prompt in warm {
                    let dispatched = engine.ranks[0]
                        .plans
                        .lock()
                        .unwrap()
                        .iter()
                        .map(|(_, _, ops)| ops.len())
                        .sum::<usize>();
                    for _ in 0..500 {
                        if engine
                            .ranks
                            .iter()
                            .all(|rank| rank.acknowledged.lock().unwrap().len() == dispatched)
                        {
                            break;
                        }
                        tokio::time::sleep(Duration::from_millis(10)).await;
                    }
                    let chunks = run(&engine.client, prompt).await;
                    assert_eq!(tokens(&chunks), [1000], "{case}");
                    cached.push(chunks.last().unwrap().cached_tokens);
                }
                assert!(cached.iter().all(|&hit| hit == 32), "{case}: {cached:?}");
                assert!(engine.cache_ops(CacheOpKind::LoadBack) >= 1, "{case}");
            }
        }
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn written_back_prefixes_are_acknowledged() {
    // Also on KV-sharded attention-TP ranks, the second of which
    // acknowledges three polls after the first.
    for layout in [Layout::Single, Layout::TpKvp(2, 2)] {
        let engine = start_ranks_with(
            layout,
            1,
            Tiers::HostL2,
            Setup {
                cache_ack_polls: Some(|rank| 3 * rank),
                ..Setup::default()
            },
        )
        .await;
        let shared: Vec<i32> = (1..=64).collect();
        let requests = (0..3)
            .map(|i| TokenizedGenerateReqInput {
                input_ids: Some(shared.iter().copied().chain([100 + i]).collect()),
                ..request(&format!("r{i}"), 1, budget(4))
            })
            .collect();
        for chunks in run_all(&engine.client, requests).await {
            assert_eq!(tokens(&chunks), [1000, 1001, 1002, 1003], "{layout:?}");
        }
        assert!(engine.cache_ops(CacheOpKind::WriteBack) > 0, "{layout:?}");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_fused_engine_writes_back_what_it_evicts_and_loads_it_back() {
    // The prefill engine's case on a fused engine, which decodes its prompts:
    // publishing a prompt's end at its first decode, a decode page once full
    // and the rest at the finish stores nothing to the Host; what an
    // admission evicts is written back in the round that evicts it. Also on
    // mirrored ranks, acknowledging at the next poll or the odd ones three
    // polls late.
    let odd_ranks_three_polls_late: fn(usize) -> usize = |rank| 3 * (rank % 2);
    for (layout, cache_ack_polls) in [
        (Layout::Single, None),
        (Layout::Tp(2), None),
        (Layout::Tp(2), Some(odd_ranks_three_polls_late)),
        (Layout::TpKvp(2, 2), None),
        (Layout::TpKvp(2, 2), Some(odd_ranks_three_polls_late)),
    ] {
        for depth in [0, 1] {
            let engine = start_ranks_with(
                layout,
                depth,
                Tiers::HostL2WriteBack,
                Setup {
                    cache_ack_polls,
                    ..Setup::default()
                },
            )
            .await;
            let case = format!(
                "{layout:?} at depth {depth}, delayed {}",
                cache_ack_polls.is_some()
            );
            // Three 3000-token prompts, 187 full pages and an 8-token tail
            // each, on 512 Device pages; each decodes 24 tokens, filling
            // its tail's page. The third one's admission evicts the first
            // one's last pages. Then the first one is sent again.
            let mut cached = Vec::new();
            for (rid, first) in [("a", 0), ("b", 10_000), ("c", 20_000), ("a-again", 0)] {
                if rid == "c" {
                    // Caching the first two stored nothing to the Host.
                    assert_eq!(engine.cache_ops(CacheOpKind::WriteBack), 0, "{case}");
                }
                let chunks = run(
                    &engine.client,
                    TokenizedGenerateReqInput {
                        input_ids: Some((first + 1..=first + 3000).collect()),
                        ..request(rid, 1, budget(24))
                    },
                )
                .await;
                assert_eq!(
                    tokens(&chunks),
                    (1000..1024).collect::<Vec<i32>>(),
                    "{rid} on {case}"
                );
                cached.push(chunks.last().unwrap().cached_tokens);
            }
            engine.settle().await;
            // Every rank ran the same plans and acknowledged each of their
            // cache ops once.
            let plans = engine.ranks[0].plans.lock().unwrap().clone();
            let mut dispatched: Vec<u32> = plans
                .iter()
                .flat_map(|(_, _, ops)| ops)
                .map(|op| op.op_id)
                .collect();
            dispatched.sort_unstable();
            for _ in 0..500 {
                if engine
                    .ranks
                    .iter()
                    .all(|rank| rank.acknowledged.lock().unwrap().len() == dispatched.len())
                {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
            for (rank, member) in engine.ranks.iter().enumerate() {
                assert_eq!(*member.plans.lock().unwrap(), plans, "rank {rank}, {case}");
                let mut op_ids: Vec<u32> = member
                    .acknowledged
                    .lock()
                    .unwrap()
                    .iter()
                    .map(|&(_, _, op_id)| op_id)
                    .collect();
                op_ids.sort_unstable();
                assert_eq!(op_ids, dispatched, "rank {rank}, {case}");
            }
            // The eviction's write-backs were acknowledged: the first
            // prompt hits all its full pages again, loading the
            // written-back ones from the Host.
            assert!(engine.cache_ops(CacheOpKind::WriteBack) > 0, "{case}");
            assert_eq!(engine.cache_ops(CacheOpKind::LoadBack), 1, "{case}");
            assert_eq!(cached, [0, 0, 0, 187 * 16], "{case}");
        }
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn an_abort_while_a_write_back_or_restore_is_in_flight_ends_cleanly() {
    // The devices acknowledge nothing until the test opens their gates; then
    // the odd ranks of a mirrored replica acknowledge three polls late.
    let held = Setup {
        cache_ack_polls: Some(|rank| 3 * (rank % 2)),
        held_cache_acks: true,
        ..Setup::default()
    };
    for layout in [Layout::Single, Layout::Tp(2), Layout::TpKvp(2, 2)] {
        // A prompt aborted while the write-back of what its admission evicted
        // is in flight: the write-back still lands, and the evicted prompt
        // sent again loads it back.
        let engine = start_ranks_with(layout, 1, Tiers::HostL2WriteBack, held.clone()).await;
        let prompt = |rid: &str, first: i32, max_new_tokens: i64| TokenizedGenerateReqInput {
            input_ids: Some((first + 1..=first + 3000).collect()),
            ..request(rid, 1, budget(max_new_tokens))
        };
        for (rid, first) in [("a", 0), ("b", 10_000)] {
            let chunks = run(&engine.client, prompt(rid, first, 8)).await;
            assert_eq!(
                tokens(&chunks),
                (1000..1008).collect::<Vec<i32>>(),
                "{rid} on {layout:?}"
            );
        }
        // The third prompt's admission evicts the first one's last pages; it
        // decodes long enough to be aborted while it runs.
        let doomed = engine
            .client
            .generate(prompt("c", 20_000, 1000))
            .await
            .unwrap();
        for _ in 0..5000 {
            if engine.cache_ops(CacheOpKind::WriteBack) > 0 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
        assert!(engine.cache_ops(CacheOpKind::WriteBack) > 0, "{layout:?}");
        // The engine streams nothing more for a client's own abort. Every
        // rank releases the prompt before its budget runs out, with its
        // write-back still unacknowledged.
        assert!(engine.client.abort("c").await.unwrap());
        drop(doomed);
        for _ in 0..500 {
            if (0..engine.ranks.len()).all(|rank| engine.released_on(rank).len() == 3) {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        for (rank, member) in engine.ranks.iter().enumerate() {
            let released = engine.released_on(rank);
            assert_eq!(released.len(), 3, "rank {rank} on {layout:?}");
            let (rid, generated) = &released[2];
            assert_eq!(rid, "c", "rank {rank} on {layout:?}");
            assert!(
                *generated < 1000,
                "rank {rank} on {layout:?}: generated {generated} tokens after the abort"
            );
            assert!(member.acknowledged.lock().unwrap().is_empty(), "{layout:?}");
            member.ack_gate.open();
        }
        let written_back = engine.cache_ops(CacheOpKind::WriteBack);
        for _ in 0..500 {
            if engine
                .ranks
                .iter()
                .all(|rank| rank.acknowledged.lock().unwrap().len() == written_back)
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        let chunks = run(&engine.client, prompt("a-again", 0, 8)).await;
        assert_eq!(
            tokens(&chunks),
            (1000..1008).collect::<Vec<i32>>(),
            "{layout:?}"
        );
        assert!(engine.cache_ops(CacheOpKind::LoadBack) >= 1, "{layout:?}");
        assert_eq!(chunks.last().unwrap().cached_tokens, 187 * 16, "{layout:?}");
        engine.settle().await;
        assert_eq!(engine.released_on(0).len(), 4, "{layout:?}");
        for rank in 1..engine.ranks.len() {
            assert_eq!(
                engine.released_on(rank),
                engine.released_on(0),
                "{layout:?}"
            );
        }
        drop(engine);

        // A request retracted while its image is in flight, so its restore
        // cannot have run, is aborted. The engine ends a client's abort at the
        // request's next forward: once its image lands it is restored, ends
        // short of its budget and is released once, on every rank, and the
        // requests after it run to the end.
        let engine = start_ranks_with(
            layout,
            1,
            Tiers::HostL2WriteBackForcedRetraction,
            held.clone(),
        )
        .await;
        let doomed = engine
            .client
            .generate(request("r0", 20, budget(40)))
            .await
            .unwrap();
        for _ in 0..5000 {
            if engine.cache_ops(CacheOpKind::SnapshotStore) > 0 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
        assert!(
            engine.cache_ops(CacheOpKind::SnapshotStore) > 0,
            "{layout:?}"
        );
        assert!(engine.client.abort("r0").await.unwrap());
        drop(doomed);
        // A control request acts after the messages that arrived before it:
        // its reply means the engine took the abort.
        let paused = ControlRequest::IsSchedulerPaused(IsSchedulerPausedReqInput {
            rid: None,
            http_worker_ipc: None,
        });
        assert_eq!(
            control(&engine, paused).await,
            [("IsSchedulerPausedReqOutput", false)],
            "{layout:?}"
        );
        for rank in &engine.ranks {
            assert!(rank.acknowledged.lock().unwrap().is_empty(), "{layout:?}");
            rank.ack_gate.open();
        }
        let requests = (1..3)
            .map(|i| request(&format!("r{i}"), 20 + i, budget(12)))
            .collect();
        for (i, chunks) in run_all(&engine.client, requests).await.iter().enumerate() {
            assert_eq!(
                tokens(chunks),
                (1000..1012).collect::<Vec<i32>>(),
                "r{} on {layout:?}",
                i + 1
            );
        }
        engine.settle().await;
        // Every rank ran the same plans and acknowledged each of their cache
        // ops once, the aborted request's image included.
        let plans = engine.ranks[0].plans.lock().unwrap().clone();
        let mut dispatched: Vec<u32> = plans
            .iter()
            .flat_map(|(_, _, ops)| ops)
            .map(|op| op.op_id)
            .collect();
        dispatched.sort_unstable();
        for _ in 0..500 {
            if engine
                .ranks
                .iter()
                .all(|rank| rank.acknowledged.lock().unwrap().len() == dispatched.len())
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        for (rank, member) in engine.ranks.iter().enumerate() {
            assert_eq!(
                *member.plans.lock().unwrap(),
                plans,
                "rank {rank} on {layout:?}"
            );
            let mut op_ids: Vec<u32> = member
                .acknowledged
                .lock()
                .unwrap()
                .iter()
                .map(|&(_, _, op_id)| op_id)
                .collect();
            op_ids.sort_unstable();
            assert_eq!(op_ids, dispatched, "rank {rank} on {layout:?}");
            let released = engine.released_on(rank);
            assert_eq!(
                released
                    .iter()
                    .map(|(rid, _)| rid.as_str())
                    .collect::<Vec<_>>(),
                ["r0", "r1", "r2"],
                "rank {rank} on {layout:?}"
            );
            let (_, generated) = &released[0];
            assert!(
                *generated < 40,
                "rank {rank} on {layout:?}: generated {generated} tokens after the abort"
            );
        }
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn an_l3_get_admits_the_prefix_every_owner_landed() {
    // The store holds the whole 4-page prefix; rank 0 lands every page of
    // the get, rank 1 only the first 2: the owners agree on 2 pages, and
    // both load back the same rows.
    let engine = start_ranks_with(
        Layout::Tp(2),
        1,
        Tiers::L3,
        Setup {
            l3_landed_keys: Some(|rank| if rank == 0 { 4 } else { 2 }),
            ..Setup::default()
        },
    )
    .await;
    let prompt: Vec<i32> = (1..=65).collect();
    let chunks = run_all(
        &engine.client,
        vec![TokenizedGenerateReqInput {
            input_ids: Some(prompt),
            ..request("r0", 1, budget(4))
        }],
    )
    .await;
    assert_eq!(tokens(&chunks[0]), [1000, 1001, 1002, 1003]);
    assert_eq!(chunks[0].last().unwrap().cached_tokens, 32);
    assert_eq!(engine.storage_queries(), 1);
    assert_eq!(engine.storage_gets(), 1);
    assert!(engine.cache_ops(CacheOpKind::LoadBack) > 0);
    engine.settle().await;
    assert_eq!(engine.released_on(1), engine.released_on(0));
}

#[tokio::test(flavor = "multi_thread")]
async fn an_abort_while_storage_prepares_finishes_the_request() {
    // The store holds its queries; the client gives up while the request
    // still waits on its storage query. Every rank finishes it at the next
    // round's head and releases it without a forward.
    let engine = start_ranks_with(
        Layout::Tp(2),
        1,
        Tiers::L3,
        Setup {
            l3_held_queries: true,
            ..Setup::default()
        },
    )
    .await;
    let prompt: Vec<i32> = (1..=65).collect();
    let stream = engine
        .client
        .generate(TokenizedGenerateReqInput {
            input_ids: Some(prompt),
            ..request("r0", 1, budget(4))
        })
        .await
        .unwrap();
    // The query went out, so the request waits in storage preparation.
    while engine.storage_queries() == 0 {
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    // The engine streams nothing more for a client's own abort.
    assert!(engine.client.abort("r0").await.unwrap());
    drop(stream);
    for _ in 0..500 {
        if engine.released_on(0).len() == 1 && engine.released_on(1).len() == 1 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert_eq!(engine.released_on(0), [("r0".to_owned(), 0)]);
    assert_eq!(engine.released_on(1), engine.released_on(0));
    assert_eq!(engine.storage_gets(), 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn tensor_parallel_ranks_mirror_one_scheduler() {
    // The follower acknowledges each cache op three polls after the first
    // member, so for a while the members hold different acknowledgments.
    // Also with the KV cache sharded over the members, and with a Host tier
    // the images are also written back to.
    for layout in [Layout::Tp(2), Layout::TpKvp(2, 2)] {
        for tiers in [
            Tiers::ForcedRetraction,
            Tiers::HostL2WriteBackForcedRetraction,
        ] {
            for depth in [0, 1] {
                let engine = start_ranks_with(
                    layout,
                    depth,
                    tiers,
                    Setup {
                        cache_ack_polls: Some(|rank| 3 * rank),
                        ..Setup::default()
                    },
                )
                .await;
                let case = format!(
                    "{layout:?} at depth {depth}, write-back {}",
                    matches!(tiers, Tiers::HostL2WriteBackForcedRetraction)
                );
                let requests = (0..4)
                    .map(|i| request(&format!("r{i}"), 20 + i, budget(8 + i as i64)))
                    .collect();
                let results = run_all(&engine.client, requests).await;
                for (i, chunks) in results.iter().enumerate() {
                    let expected: Vec<i32> = (1000..1008 + i as i32).collect();
                    assert_eq!(tokens(chunks), expected, "r{i} on {case}");
                }
                // The follower admitted, scheduled and released the same
                // requests, and agreed on every cache acknowledgment with the
                // first member.
                engine.settle().await;
                assert_eq!(engine.released_on(1), engine.released_on(0), "{case}");
                assert_eq!(engine.released_on(0).len(), 4, "{case}");
                let stores = engine.cache_ops(CacheOpKind::SnapshotStore);
                assert!(stores > 0, "{case}");
                assert_eq!(
                    engine.cache_ops(CacheOpKind::SnapshotRestore),
                    stores,
                    "{case}"
                );
                let plans = engine.ranks[0].plans.lock().unwrap().clone();
                assert_eq!(*engine.ranks[1].plans.lock().unwrap(), plans, "{case}");
                // Each member acknowledged every op once, the follower at
                // least three polls after the first member.
                let mut dispatched: Vec<u32> = plans
                    .iter()
                    .flat_map(|(_, _, ops)| ops)
                    .map(|op| op.op_id)
                    .collect();
                dispatched.sort_unstable();
                let [first, follower] = [0, 1].map(|rank| {
                    let mut acknowledged = engine.ranks[rank].acknowledged.lock().unwrap().clone();
                    acknowledged.sort_by_key(|&(_, _, op_id)| op_id);
                    acknowledged
                });
                let op_ids: Vec<u32> = first.iter().map(|&(_, _, op_id)| op_id).collect();
                assert_eq!(op_ids, dispatched, "{case}");
                assert_eq!(follower.len(), first.len(), "{case}");
                for (&(early, kind, op_id), &(late, follower_kind, follower_op_id)) in
                    first.iter().zip(&follower)
                {
                    assert_eq!((follower_kind, follower_op_id), (kind, op_id), "{case}");
                    assert!(
                        late >= early + 3,
                        "op {op_id} on {case}: polls {early} and {late}"
                    );
                }
                // A restored request forwards again only from the poll that
                // brought the follower a restore's acknowledgment: in order,
                // the k-th request to resume does so no earlier than the k-th
                // such poll. A forced retraction can take a restored request
                // back before it forwards, so fewer requests may resume than
                // were restored; at depth 1 that happens often, and the check
                // is looser there. A prefill runs alone outside mixed mode, so
                // a decoding request also sits out the plan that prefills a
                // later prompt: it resumes only where a plan decoded others
                // since its last forward. Every prompt here prefills in one
                // forward, so a plan that forwards a request again decodes.
                let mut restored: Vec<usize> = follower
                    .iter()
                    .filter(|(_, kind, _)| *kind == CacheEventKind::RestoreDone)
                    .map(|&(poll, _, _)| poll)
                    .collect();
                restored.sort_unstable();
                let mut last_forward: HashMap<&str, usize> = HashMap::new();
                let mut decoded = Vec::with_capacity(plans.len());
                let mut resumed = Vec::new();
                for (index, (polls, rids, _)) in plans.iter().enumerate() {
                    decoded.push(
                        rids.iter()
                            .any(|rid| last_forward.contains_key(rid.as_str())),
                    );
                    for rid in rids {
                        // A plan decoded others since the request's last
                        // forward.
                        let returned = last_forward
                            .insert(rid, index)
                            .is_some_and(|last| decoded[last + 1..index].contains(&true));
                        if returned {
                            resumed.push(*polls);
                        }
                    }
                }
                assert!(!resumed.is_empty(), "{case}");
                assert!(resumed.len() <= restored.len(), "{case}");
                for (resumed, restored) in resumed.iter().zip(&restored) {
                    assert!(
                        resumed >= restored,
                        "{case}: resumed at poll {resumed}, restored at poll {restored}"
                    );
                }
            }
        }
    }
}

/// A request whose image an encode worker embeds (EPD).
fn epd_request(rid: &str) -> TokenizedGenerateReqInput {
    TokenizedGenerateReqInput {
        multimodal_inputs: Some(MultimodalInputs {
            mm_items: vec![MultimodalDataItem {
                modality: Modality::Image,
                hash: None,
                pad_value: None,
                offsets: Some(vec![(0, 2)]),
                feature: None,
                feature_shm: None,
                model_specific_data: Dict::default(),
                encoded: None,
                encoded_deepstack: None,
                encode_handshake: Some(Dict(vec![
                    (Value::from("bootstrap_room"), Value::from(7)),
                    (Value::from("bootstrap_host"), Value::from("127.0.0.1")),
                    (Value::from("bootstrap_port"), Value::from(8998)),
                ])),
            }],
            im_token_id: None,
            video_token_id: None,
            mrope_positions: None,
            mrope_position_delta: None,
            mrope_position_delta_scalar: None,
            mrope_position_delta_repeated_cache: None,
        }),
        ..request(rid, 8, budget(3))
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn constrained_requests_follow_their_grammars_on_every_rank() {
    for depth in [0, 1] {
        let engine = start_ranks_with(
            Layout::Tp(2),
            depth,
            Tiers::DeviceOnly,
            Setup {
                grammar: true,
                ..Setup::default()
            },
        )
        .await;
        let constrained = |rid: &str, regex: &str| {
            let mut request = request(rid, 8, budget(6));
            request.sampling_params.regex = Some(regex.to_owned());
            request
        };
        let outputs = run_all(
            &engine.client,
            vec![
                constrained("g-0", "a(b|x)"),
                constrained("bad-0", "([a-z"),
                // A cache hit, queued all the same: the ranks agree first.
                constrained("g-1", "a(b|x)"),
                request("plain-0", 8, budget(3)),
            ],
        )
        .await;
        // a, then the lower of b and x, then the stop token ends the grammar.
        assert_eq!(tokens(&outputs[0]), [0, 1, 8], "depth {depth}");
        assert_eq!(
            outputs[0].last().unwrap().finish_reason.as_deref(),
            Some("stop")
        );
        let bad = outputs[1].last().unwrap();
        assert_eq!(bad.finish_reason.as_deref(), Some("abort"));
        // The request's own error: the frontend answers it as a client one.
        let details = &bad.finish_details.as_ref().unwrap().0;
        let field = |key: &str| details.iter().find(|(k, _)| k == key).map(|(_, v)| v);
        assert_eq!(field("status_code"), Some(&Value::from(400i64)));
        assert!(matches!(
            field("message"),
            Some(Value::Str(message)) if message.starts_with("Failed to compile regex grammar")
        ));
        assert_eq!(tokens(&outputs[2]), [0, 1, 8]);
        assert_eq!(tokens(&outputs[3]), [1000, 1001, 1002]);
        engine.settle().await;
        assert_eq!(engine.released_on(1), engine.released_on(0));
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_grammar_unusable_on_one_rank_fails_alike_on_every_rank() {
    // The compiled grammar on one rank, a timeout marker on the other: an
    // expiring one (one timeout) or a given-up one (past the two retries).
    for (timeouts, status, gave_up) in [(1, None, false), (3, Some(400i64), true)] {
        for marked in [0, 1] {
            let per_rank: fn(usize) -> u32 = match (timeouts, marked) {
                (1, 0) => |rank| u32::from(rank == 0),
                (1, _) => |rank| u32::from(rank == 1),
                (_, 0) => |rank| if rank == 0 { 3 } else { 0 },
                _ => |rank| if rank == 1 { 3 } else { 0 },
            };
            let engine = start_ranks_with(
                Layout::Tp(2),
                0,
                Tiers::DeviceOnly,
                Setup {
                    grammar: true,
                    grammar_timeouts: Some(per_rank),
                    ..Setup::default()
                },
            )
            .await;
            let mut request = request("g-0", 8, budget(6));
            request.sampling_params.regex = Some(TIMED_OUT_GRAMMAR.to_owned());
            let chunks = run(&engine.client, request).await;
            let last = chunks.last().unwrap();
            assert_eq!(last.finish_reason.as_deref(), Some("abort"));
            let details = &last.finish_details.as_ref().unwrap().0;
            let field = |key: &str| details.iter().find(|(k, _)| k == key).map(|(_, v)| v);
            assert_eq!(
                field("status_code"),
                status.map(Value::from).as_ref(),
                "{timeouts} timeouts on rank {marked}"
            );
            let Some(Value::Str(message)) = field("message") else {
                panic!("no message: {details:?}");
            };
            assert!(message.contains("timed out"), "{message}");
            assert_eq!(message.contains("gave up"), gave_up, "{message}");
            // Both ranks ended it.
            engine.settle().await;
            assert_eq!(engine.released_on(1), engine.released_on(0));
            assert_eq!(engine.released_on(0).len(), 1);
        }
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn epd_requests_wait_for_their_embeddings_on_every_rank() {
    let engine = start_ranks_with(
        Layout::Tp(2),
        0,
        Tiers::DeviceOnly,
        Setup {
            epd_receive_enabled: true,
            ..Setup::default()
        },
    )
    .await;
    let outputs = run_all(
        &engine.client,
        vec![
            epd_request("img-0"),
            epd_request("lost-0"),
            request("text-0", 8, budget(3)),
        ],
    )
    .await;
    // The text request runs at once; the embedded one only once both ranks
    // received its image (rank 1 a poll later); the lost one fails.
    assert_eq!(tokens(&outputs[0]), [1000, 1001, 1002]);
    assert_eq!(
        outputs[1].last().unwrap().finish_reason.as_deref(),
        Some("abort")
    );
    assert_eq!(tokens(&outputs[2]), [1000, 1001, 1002]);
    engine.settle().await;
    for rank in 0..2 {
        assert_eq!(*engine.ranks[rank].epd_admitted.lock().unwrap(), ["img-0"]);
        // Its slot went back once the data plane copied its rows out.
        assert_eq!(*engine.ranks[rank].epd_released.lock().unwrap(), [7]);
        let released: Vec<String> = engine
            .released_on(rank)
            .into_iter()
            .map(|(rid, _)| rid)
            .collect();
        // The failed one too: the data plane forgets every admitted request.
        assert_eq!(released, ["img-0", "lost-0", "text-0"]);
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn expert_parallel_ranks_rebalance_to_the_same_placement() {
    let engine = start_ranks_with(
        Layout::Dp(2),
        1,
        Tiers::DeviceOnly,
        Setup {
            eplb: Some(eplb_config),
            ..Setup::default()
        },
    )
    .await;
    let requests = (0..4)
        .map(|i| TokenizedGenerateReqInput {
            data_parallel_rank: Some(i % 2),
            ..request(&format!("r{i}"), 20, budget(12))
        })
        .collect();
    run_all(&engine.client, requests).await;
    // Snapshot, commit and two one-layer chunks complete over the next
    // rounds, which run on even without requests. How many forwards the
    // requests took depends on how they were batched, so a later rebalance
    // may follow; the first one's load is the one known here.
    let applied: Vec<_> = engine
        .ranks
        .iter()
        .map(|rank| rank.applied.clone())
        .collect();
    let chunks = |rank: usize| applied[rank].lock().unwrap().len();
    for _ in 0..200 {
        if chunks(0) >= 2 && chunks(1) >= 2 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    // The ranks stop after the same round, so each applied the same chunks.
    drop(engine);
    let (rank0, rank1) = (
        applied[0].lock().unwrap().clone(),
        applied[1].lock().unwrap().clone(),
    );
    assert_eq!(
        rank0, rank1,
        "every EP rank applies the broadcast placement"
    );
    assert!(rank0.len() >= 2, "{rank0:?}");
    // One layer per round, in order, rebalance after rebalance.
    for (index, (layers, _)) in rank0.iter().enumerate() {
        assert_eq!(*layers, [index % 2], "{rank0:?}");
    }
    let rank0 = &rank0[..2];
    // The group load: expert 0 counted 100 + 90 per slot it held, twice,
    // the others 2 per slot; the solver replicates the hot expert.
    let load: Vec<Vec<f64>> = vec![vec![380.0, 4.0, 2.0, 2.0]; 2];
    let expected =
        ts_control::eplb::rebalance_experts(&load, 6, None, 1, 2, Algorithm::Deepseek).unwrap();
    for (layers, rows) in rank0 {
        let row: Vec<usize> = rows[0].iter().map(|&e| e as usize).collect();
        assert_eq!(row, expected[layers[0]]);
    }
    assert!(rows_hold_expert_zero_thrice(rank0));
}

fn rows_hold_expert_zero_thrice(applied: &[(Vec<usize>, Vec<Vec<i32>>)]) -> bool {
    applied
        .iter()
        .all(|(_, rows)| rows[0].iter().filter(|&&expert| expert == 0).count() == 3)
}

#[tokio::test(flavor = "multi_thread")]
async fn an_idle_data_parallel_rank_forwards_with_the_busy_one() {
    let engine = start_ranks(Layout::Dp(2), 1, Tiers::DeviceOnly).await;
    let requests = (0..3)
        .map(|i| TokenizedGenerateReqInput {
            data_parallel_rank: Some(0),
            ..request(&format!("r{i}"), 30, budget(6))
        })
        .collect();
    for chunks in run_all(&engine.client, requests).await {
        assert_eq!(tokens(&chunks), (1000..1006).collect::<Vec<_>>());
    }
    assert_eq!(engine.released_on(0).len(), 3);
    assert!(engine.released_on(1).is_empty());
    assert!(engine.idle_forwards(1) > 0);
    assert_eq!(engine.idle_forwards(0), 0);

    // The other rank serves its own requests.
    let pinned = TokenizedGenerateReqInput {
        data_parallel_rank: Some(1),
        ..request("s0", 10, budget(3))
    };
    assert_eq!(
        tokens(&run(&engine.client, pinned).await),
        [1000, 1001, 1002]
    );
    assert_eq!(engine.released_on(1), [("s0".to_owned(), 3 + 1)]);
}

#[tokio::test(flavor = "multi_thread")]
async fn pipeline_stages_commit_the_last_stages_tokens() {
    // The chunk pipeline keeps one forward per stage in flight.
    let engine = start_ranks(Layout::Pp(2), 2, Tiers::DeviceOnly).await;
    let requests = (0..4)
        .map(|i| {
            request(
                &format!("r{i}"),
                MAX_SCHEDULED_TOKENS + 20 * i,
                budget(3 + i as i64),
            )
        })
        .collect();
    let results = run_all(&engine.client, requests).await;
    for (i, chunks) in results.iter().enumerate() {
        let expected: Vec<i32> = (1000..1003 + i as i32).collect();
        assert_eq!(tokens(chunks), expected, "r{i}");
    }
    // The first stage finished each request exactly when the last did.
    engine.settle().await;
    assert_eq!(engine.released_on(0).len(), 4);
    let last: Vec<String> = engine
        .released_on(1)
        .into_iter()
        .map(|(rid, _)| rid)
        .collect();
    let first: Vec<String> = engine
        .released_on(0)
        .into_iter()
        .map(|(rid, _)| rid)
        .collect();
    assert_eq!(first, last);
}

#[tokio::test(flavor = "multi_thread")]
async fn engram_windows_follow_each_requests_tokens() {
    for depth in [0, 1] {
        let engine = start_ranks_with(
            Layout::Single,
            depth,
            Tiers::DeviceOnly,
            Setup {
                ngram_context_len: 2,
                ..Setup::default()
            },
        )
        .await;
        let chunks = run(&engine.client, request("r", 3, budget(3))).await;
        assert_eq!(tokens(&chunks), [1000, 1001, 1002]);
        let seen = engine.ranks[0].ngram.lock().unwrap().clone();
        // The prompt 1, 2, 3 prefills from position 0; each decode then sits
        // on the newest committed token. With the overlap schedule a decode
        // is dispatched before the previous forward commits, so it sits one
        // token behind; the in-flight token stays on the device.
        let expected = if depth == 0 {
            [
                (0, vec![1, -1, -1]),
                (3, vec![1000, 3, 2]),
                (4, vec![1001, 1000, 3]),
            ]
        } else {
            [
                (0, vec![1, -1, -1]),
                (2, vec![3, 2, 1]),
                (3, vec![1000, 3, 2]),
            ]
        };
        assert_eq!(seen[..3], expected, "depth {depth}");
    }
}

/// A prefill engine over `prefill_layout` and `prefill_tiers`, with
/// `prefill_setup` besides its PD half, and a single-rank decode engine,
/// joined by one wire.
async fn start_pd_pair(
    in_flight_depth: usize,
    prefill_layout: Layout,
    prefill_tiers: Tiers,
    prefill_setup: Setup,
) -> (Engine, Engine, Arc<Mutex<Wire>>) {
    let wire = Arc::new(Mutex::new(Wire {
        prefill_ranks: prefill_layout.sizes().1,
        ..Wire::default()
    }));
    let half = |role| {
        Some(Pd {
            role,
            wire: wire.clone(),
        })
    };
    let prefill = start_ranks_with(
        prefill_layout,
        in_flight_depth,
        prefill_tiers,
        Setup {
            pd: half(Role::Prefill),
            ..prefill_setup
        },
    )
    .await;
    let decode = start_ranks_with(
        Layout::Single,
        in_flight_depth,
        Tiers::DeviceOnly,
        Setup {
            pd: half(Role::Decode),
            ..Setup::default()
        },
    )
    .await;
    (prefill, decode, wire)
}

fn disaggregated(
    rid: &str,
    room: u64,
    prompt_len: usize,
    max_new_tokens: i64,
) -> TokenizedGenerateReqInput {
    TokenizedGenerateReqInput {
        bootstrap_host: Some("127.0.0.1".to_owned()),
        bootstrap_port: Some(8998),
        bootstrap_room: Some(room),
        ..request(rid, prompt_len, budget(max_new_tokens))
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_prompt_prefills_on_one_engine_and_decodes_on_the_other() {
    for (prefill_layout, depth) in [
        (Layout::Single, 0),
        (Layout::Single, 1),
        (Layout::Tp(2), 0),
        (Layout::Tp(2), 1),
    ] {
        let (prefill, decode, _wire) =
            start_pd_pair(depth, prefill_layout, Tiers::DeviceOnly, Setup::default()).await;
        // As a gateway does: the same request to both nodes at once.
        let requests: Vec<_> = (0..3)
            .map(|i| disaggregated(&format!("r{i}"), 70 + i, 20 + 300 * i as usize, 4))
            .collect();
        let (prefilled, decoded) = tokio::join!(
            run_all(&prefill.client, requests.clone()),
            run_all(&decode.client, requests)
        );
        for (i, chunks) in prefilled.iter().enumerate() {
            // The prefill node streams only its first token, once the KV
            // reached the decode node.
            assert_eq!(
                tokens(chunks),
                [1000],
                "r{i} at depth {depth} on {prefill_layout:?}"
            );
            assert_eq!(
                chunks.last().unwrap().finish_reason.as_deref(),
                Some("length")
            );
        }
        for (i, chunks) in decoded.iter().enumerate() {
            // The decode node's output starts with the prefill node's token.
            assert_eq!(
                tokens(chunks),
                [1000, 1001, 1002, 1003],
                "r{i} at depth {depth} on {prefill_layout:?}"
            );
        }
        assert_eq!(prefill.released().len(), 3);
        prefill.settle().await;
        assert_eq!(
            prefill.released_on(prefill.ranks.len() - 1),
            prefill.released()
        );
        assert_eq!(decode.released().len(), 3);
        // Each request's slot was made ready for its first decode.
        let landed = decode.ranks[0].landed.lock().unwrap().clone();
        assert_eq!(landed.len(), 3, "depth {depth}");
        assert!(landed
            .iter()
            .all(|(candidates, slot)| candidates.is_none() && slot.is_some()));
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_prefill_engine_writes_back_what_it_evicts_and_loads_it_back() {
    // Under Host write-back the prefill engine stores nothing to the Host as
    // it caches a prompt; what an admission evicts is written back in the
    // round that evicts it and published on the Host at its acknowledgment.
    // Each prompt takes a dozen rounds or more, so acknowledgments three
    // polls late land before the next admission: the numbers stay the same.
    let three_polls_late: fn(usize) -> usize = |_| 3;
    for (depth, cache_ack_polls) in [
        (0, None),
        (1, None),
        (0, Some(three_polls_late)),
        (1, Some(three_polls_late)),
    ] {
        let (prefill, decode, _wire) = start_pd_pair(
            depth,
            Layout::Single,
            Tiers::HostL2WriteBack,
            Setup {
                cache_ack_polls,
                ..Setup::default()
            },
        )
        .await;
        let case = format!("depth {depth}, delayed {}", cache_ack_polls.is_some());
        // Three 3000-token prompts, 187 full pages and a tail each, on 512
        // Device pages: the third one's admission evicts the first one's
        // last pages. Then the first one is sent again.
        let mut cached = Vec::new();
        let prompts = [("a", 0), ("b", 10_000), ("c", 20_000), ("a-again", 0)];
        for (room, (rid, first)) in (70_u64..).zip(prompts) {
            if rid == "c" {
                // Caching the first two stored nothing to the Host.
                assert_eq!(prefill.cache_ops(CacheOpKind::WriteBack), 0, "{case}");
            }
            let request = TokenizedGenerateReqInput {
                input_ids: Some((first + 1..=first + 3000).collect()),
                ..disaggregated(rid, room, 1, 2)
            };
            let (prefilled, decoded) = tokio::join!(
                run(&prefill.client, request.clone()),
                run(&decode.client, request)
            );
            assert_eq!(tokens(&prefilled), [1000], "{rid} at {case}");
            assert_eq!(tokens(&decoded), [1000, 1001], "{rid} at {case}");
            cached.push(prefilled.last().unwrap().cached_tokens);
        }
        // The eviction's write-backs were acknowledged: the first prompt hits
        // all its full pages again, loading the written-back ones from the
        // Host. Unacknowledged, the Host would hold none of them.
        assert!(prefill.cache_ops(CacheOpKind::WriteBack) > 0, "{case}");
        assert_eq!(prefill.cache_ops(CacheOpKind::LoadBack), 1, "{case}");
        assert_eq!(cached, [0, 0, 0, 187 * 16], "{case}");
    }

    // The device acknowledges nothing until the test opens its gate. While
    // the third prompt's write-backs are unacknowledged, none of the pages
    // they carry is reused from the Host: the first prompt sent again
    // computes them anew, evicting the second prompt's last pages. Once the
    // gate opens, the second prompt sent again loads those back.
    let (prefill, decode, _wire) = start_pd_pair(
        1,
        Layout::Single,
        Tiers::HostL2WriteBack,
        Setup {
            held_cache_acks: true,
            ..Setup::default()
        },
    )
    .await;
    let mut cached = Vec::new();
    let prompts = [
        ("a", 0),
        ("b", 10_000),
        ("c", 20_000),
        ("a-again", 0),
        ("b-again", 10_000),
    ];
    for (room, (rid, first)) in (70_u64..).zip(prompts) {
        if rid == "b-again" {
            assert_eq!(prefill.cache_ops(CacheOpKind::LoadBack), 0);
            assert!(prefill.ranks[0].acknowledged.lock().unwrap().is_empty());
            prefill.ranks[0].ack_gate.open();
            let written_back = prefill.cache_ops(CacheOpKind::WriteBack);
            for _ in 0..500 {
                if prefill.ranks[0].acknowledged.lock().unwrap().len() == written_back {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
            assert_eq!(
                prefill.ranks[0].acknowledged.lock().unwrap().len(),
                written_back
            );
        }
        let request = TokenizedGenerateReqInput {
            input_ids: Some((first + 1..=first + 3000).collect()),
            ..disaggregated(rid, room, 1, 2)
        };
        let (prefilled, decoded) = tokio::join!(
            run(&prefill.client, request.clone()),
            run(&decode.client, request)
        );
        assert_eq!(tokens(&prefilled), [1000], "{rid} held");
        assert_eq!(tokens(&decoded), [1000, 1001], "{rid} held");
        cached.push(prefilled.last().unwrap().cached_tokens);
    }
    // The first prompt sent again hits only the full pages the Device kept;
    // the second hits all of its own, loading the written-back ones.
    assert_eq!(prefill.cache_ops(CacheOpKind::LoadBack), 1);
    assert_eq!(cached, [0, 0, 0, 136 * 16, 187 * 16]);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_prefill_replica_writes_back_what_it_evicts_on_every_rank() {
    // The test above on mirrored prefill ranks, the odd ones acknowledging
    // every cache op three polls after the even ones. Every rank copies its
    // part of what the replica evicts, and a written-back segment publishes
    // only once every rank holds its acknowledgment.
    for (layout, depth) in [
        (Layout::Tp(2), 0),
        (Layout::TpKvp(2, 2), 0),
        (Layout::TpKvp(2, 2), 1),
        (Layout::TpKvp(4, 2), 0),
    ] {
        let (prefill, decode, _wire) = start_pd_pair(
            depth,
            layout,
            Tiers::HostL2WriteBack,
            Setup {
                cache_ack_polls: Some(|rank| 3 * (rank % 2)),
                ..Setup::default()
            },
        )
        .await;
        let case = format!("{layout:?} at depth {depth}");
        let mut cached = Vec::new();
        let prompts = [("a", 0), ("b", 10_000), ("c", 20_000), ("a-again", 0)];
        for (room, (rid, first)) in (70_u64..).zip(prompts) {
            let request = TokenizedGenerateReqInput {
                input_ids: Some((first + 1..=first + 3000).collect()),
                ..disaggregated(rid, room, 1, 2)
            };
            let (prefilled, decoded) = tokio::join!(
                run(&prefill.client, request.clone()),
                run(&decode.client, request)
            );
            assert_eq!(tokens(&prefilled), [1000], "{rid} on {case}");
            assert_eq!(tokens(&decoded), [1000, 1001], "{rid} on {case}");
            cached.push(prefilled.last().unwrap().cached_tokens);
        }
        prefill.settle().await;
        // A request's release does not wait for its last plan's
        // acknowledgments, which the odd ranks send three polls late.
        let dispatched_ops: usize = [CacheOpKind::WriteBack, CacheOpKind::LoadBack]
            .into_iter()
            .map(|kind| prefill.cache_ops(kind))
            .sum();
        for _ in 0..500 {
            if prefill
                .ranks
                .iter()
                .all(|rank| rank.acknowledged.lock().unwrap().len() == dispatched_ops)
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        let plans: Vec<Vec<ExecutedPlan>> = prefill
            .ranks
            .iter()
            .map(|rank| rank.plans.lock().unwrap().clone())
            .collect();
        let acknowledged: Vec<Vec<(usize, CacheEventKind, u32)>> = prefill
            .ranks
            .iter()
            .map(|rank| rank.acknowledged.lock().unwrap().clone())
            .collect();
        // The ranks poll in the same rounds, so their poll counts are one
        // timeline: on every rank, the first prompt's load-back comes no
        // earlier than the poll at which the slowest rank acknowledged the
        // write-backs dispatched before it.
        for (rank, rank_plans) in plans.iter().enumerate() {
            let load = rank_plans
                .iter()
                .position(|(_, _, ops)| ops.iter().any(|op| op.kind == CacheOpKind::LoadBack))
                .unwrap_or_else(|| panic!("no load-back on rank {rank}, {case}"));
            let written_back: Vec<u32> = rank_plans[..load]
                .iter()
                .flat_map(|(_, _, ops)| ops)
                .filter(|op| op.kind == CacheOpKind::WriteBack)
                .map(|op| op.op_id)
                .collect();
            let landed = acknowledged
                .iter()
                .flatten()
                .filter(|&&(_, kind, op_id)| {
                    kind == CacheEventKind::WriteBackDone && written_back.contains(&op_id)
                })
                .map(|&(poll, _, _)| poll)
                .max()
                .unwrap_or_else(|| panic!("no write-back before the load on rank {rank}, {case}"));
            assert!(
                rank_plans[load].0 >= landed,
                "rank {rank}, {case}: loaded back at poll {}, written back at poll {landed}",
                rank_plans[load].0
            );
        }
        // Every rank ran the same plans and acknowledged each of their cache
        // ops once.
        let mut dispatched: Vec<u32> = plans[0]
            .iter()
            .flat_map(|(_, _, ops)| ops)
            .map(|op| op.op_id)
            .collect();
        dispatched.sort_unstable();
        for (rank, (rank_plans, rank_acks)) in plans.iter().zip(&acknowledged).enumerate() {
            assert_eq!(rank_plans, &plans[0], "rank {rank}, {case}");
            let mut op_ids: Vec<u32> = rank_acks.iter().map(|&(_, _, op_id)| op_id).collect();
            op_ids.sort_unstable();
            assert_eq!(op_ids, dispatched, "rank {rank}, {case}");
        }
        assert!(prefill.cache_ops(CacheOpKind::WriteBack) > 0, "{case}");
        assert_eq!(prefill.cache_ops(CacheOpKind::LoadBack), 1, "{case}");
        assert_eq!(cached, [0, 0, 0, 187 * 16], "{case}");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_failed_transfer_aborts_the_request_on_both_engines() {
    let (prefill, decode, wire) =
        start_pd_pair(1, Layout::Single, Tiers::DeviceOnly, Setup::default()).await;
    wire.lock().unwrap().rooms.entry(5).or_default().fails = true;
    let doomed = disaggregated("doomed", 5, 30, 4);
    let (prefilled, decoded) = tokio::join!(
        run(&prefill.client, doomed.clone()),
        run(&decode.client, doomed)
    );
    for chunks in [prefilled, decoded] {
        assert!(tokens(&chunks).is_empty());
        assert_eq!(
            chunks.last().unwrap().finish_reason.as_deref(),
            Some("abort")
        );
    }
    // A disaggregated engine refuses a request that names no peer.
    let chunks = run(&decode.client, request("lost", 3, budget(2))).await;
    assert_eq!(
        chunks.last().unwrap().finish_reason.as_deref(),
        Some("abort")
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn a_transfer_failing_on_one_prefill_rank_fails_on_every_rank() {
    // Only the second prefill rank sees the transfer break. The decode
    // engine connects first, so on its own the first rank would go on to
    // prefill; the replica agrees on the failure instead, and the decode
    // engine learns of it from the wire.
    let (prefill, decode, wire) =
        start_pd_pair(1, Layout::Tp(2), Tiers::DeviceOnly, Setup::default()).await;
    wire.lock()
        .unwrap()
        .rooms
        .entry(5)
        .or_default()
        .fails_on_prefill_rank = Some(1);
    let doomed = disaggregated("doomed", 5, 30, 4);
    let decoding = decode.client.generate(doomed.clone()).await.unwrap();
    for _ in 0..500 {
        if wire.lock().unwrap().rooms[&5].preallocated.is_some() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert!(wire.lock().unwrap().rooms[&5].preallocated.is_some());
    let (prefilled, decoded) = tokio::join!(run(&prefill.client, doomed), collect(decoding));
    for chunks in [prefilled, decoded] {
        assert!(tokens(&chunks).is_empty());
        assert_eq!(
            chunks.last().unwrap().finish_reason.as_deref(),
            Some("abort")
        );
    }
    // Both prefill ranks released it without a forward.
    prefill.settle().await;
    assert_eq!(prefill.released_on(0), [("doomed".to_owned(), 0)]);
    assert_eq!(prefill.released_on(1), prefill.released_on(0));

    // Under Host write-back, with the second rank holding back every cache
    // acknowledgment, the transfer breaks on that rank once the doomed
    // prompt's admission has written back what it evicted. Both ranks
    // release the prompt alike, and once the acknowledgments land nothing is
    // pinned or in flight: the cache flushes and serving goes on.
    for layout in [Layout::Tp(2), Layout::TpKvp(2, 2)] {
        let (prefill, decode, wire) = start_pd_pair(
            1,
            layout,
            Tiers::HostL2WriteBack,
            Setup {
                held_cache_acks: true,
                ..Setup::default()
            },
        )
        .await;
        prefill.ranks[0].ack_gate.open();
        // Three 3000-token prompts fill the Device; the third evicts the
        // first one's last pages.
        let prompts = [("a", 0), ("b", 10_000), ("c", 20_000)];
        for (room, (rid, first)) in (70_u64..).zip(prompts) {
            let request = TokenizedGenerateReqInput {
                input_ids: Some((first + 1..=first + 3000).collect()),
                ..disaggregated(rid, room, 1, 2)
            };
            let (prefilled, decoded) = tokio::join!(
                run(&prefill.client, request.clone()),
                run(&decode.client, request)
            );
            assert_eq!(tokens(&prefilled), [1000], "{rid} on {layout:?}");
            assert_eq!(tokens(&decoded), [1000, 1001], "{rid} on {layout:?}");
        }
        let written_back = prefill.cache_ops(CacheOpKind::WriteBack);
        assert!(written_back > 0, "{layout:?}");
        // The doomed prompt's first chunk evicts more. Polling once per
        // forward (FORWARD_TIME) sees that with a dozen chunks to go, and the
        // transfer breaks then.
        let doomed = TokenizedGenerateReqInput {
            input_ids: Some((30_001..=33_500).collect()),
            ..disaggregated("doomed", 90, 1, 2)
        };
        let decoding = decode.client.generate(doomed.clone()).await.unwrap();
        let prefilling = prefill.client.generate(doomed).await.unwrap();
        for _ in 0..5000 {
            if prefill.cache_ops(CacheOpKind::WriteBack) > written_back {
                break;
            }
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
        assert!(
            prefill.cache_ops(CacheOpKind::WriteBack) > written_back,
            "{layout:?}"
        );
        wire.lock()
            .unwrap()
            .rooms
            .entry(90)
            .or_default()
            .fails_on_prefill_rank = Some(1);
        let (prefilled, decoded) = tokio::join!(collect(prefilling), collect(decoding));
        for chunks in [prefilled, decoded] {
            assert!(tokens(&chunks).is_empty(), "{layout:?}");
            assert_eq!(
                chunks.last().unwrap().finish_reason.as_deref(),
                Some("abort"),
                "{layout:?}"
            );
        }
        prefill.settle().await;
        assert!(
            prefill.released_on(0).contains(&("doomed".to_owned(), 0)),
            "{layout:?}"
        );
        assert_eq!(prefill.released_on(1), prefill.released_on(0), "{layout:?}");
        assert!(
            prefill.ranks[1].acknowledged.lock().unwrap().is_empty(),
            "{layout:?}"
        );
        prefill.ranks[1].ack_gate.open();
        let dispatched: usize = [CacheOpKind::WriteBack, CacheOpKind::LoadBack]
            .into_iter()
            .map(|kind| prefill.cache_ops(kind))
            .sum();
        for _ in 0..500 {
            if prefill.ranks[1].acknowledged.lock().unwrap().len() == dispatched {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        // Both ranks ran the same plans.
        let plans = prefill.ranks[0].plans.lock().unwrap().clone();
        for (rank, member) in prefill.ranks.iter().enumerate() {
            assert_eq!(
                *member.plans.lock().unwrap(),
                plans,
                "rank {rank} on {layout:?}"
            );
            assert_eq!(
                member.acknowledged.lock().unwrap().len(),
                dispatched,
                "rank {rank} on {layout:?}"
            );
        }
        let flush = ControlRequest::FlushCache(FlushCacheReqInput {
            rid: None,
            http_worker_ipc: None,
        });
        assert_eq!(
            control(&prefill, flush).await,
            [("FlushCacheReqOutput", true)],
            "{layout:?}"
        );
        let after = disaggregated("after", 91, 30, 2);
        let (prefilled, decoded) = tokio::join!(
            run(&prefill.client, after.clone()),
            run(&decode.client, after)
        );
        assert_eq!(tokens(&prefilled), [1000], "{layout:?}");
        assert_eq!(tokens(&decoded), [1000, 1001], "{layout:?}");
    }
}

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

/// Each engine's reply: its tag and whether it reports success (or the
/// queried flag).
fn outcome(replies: Vec<ControlOutput>) -> Vec<(&'static str, bool)> {
    replies
        .into_iter()
        .map(|reply| {
            let flag = match &reply {
                ControlOutput::PauseScheduler(r) => r.success,
                ControlOutput::ResumeScheduler(r) => r.success,
                ControlOutput::IsSchedulerPaused(r) => r.is_paused,
                ControlOutput::FlushCache(r) => r.success,
                ControlOutput::UpdateWeightFromDisk(r) => r.success,
                ControlOutput::ReleaseMemoryOccupation(r) => r.success,
                ControlOutput::ResumeMemoryOccupation(r) => r.success,
                ControlOutput::IsSleeping(r) => r.is_sleeping,
                ControlOutput::Profile(r) => r.success,
                other => panic!("unexpected {}", other.tag()),
            };
            (reply.tag(), flag)
        })
        .collect()
}

async fn control(engine: &Engine, request: ControlRequest) -> Vec<(&'static str, bool)> {
    let tag = request.tag();
    outcome(
        tokio::time::timeout(Duration::from_secs(10), engine.client.control(request))
            .await
            .unwrap_or_else(|_| panic!("no reply to {tag}"))
            .unwrap(),
    )
}

#[tokio::test(flavor = "multi_thread")]
async fn a_wait_pause_answers_once_drained_and_withholds_new_requests() {
    let engine = start_engine(1).await;
    let running = engine
        .client
        .generate(request("r1", 3, budget(40)))
        .await
        .unwrap();
    // The pause answers only after r1 finished.
    let (paused, chunks) = tokio::join!(control(&engine, pause("wait")), collect(running));
    assert_eq!(paused, [("PauseSchedulerReqOutput", true)]);
    assert_eq!(tokens(&chunks).len(), 40);
    assert_eq!(
        control(
            &engine,
            ControlRequest::IsSchedulerPaused(IsSchedulerPausedReqInput {
                rid: None,
                http_worker_ipc: None,
            })
        )
        .await,
        [("IsSchedulerPausedReqOutput", true)]
    );
    // A request arriving while paused waits for the resume.
    let mut withheld = engine
        .client
        .generate(request("r2", 3, budget(3)))
        .await
        .unwrap();
    assert!(
        tokio::time::timeout(Duration::from_millis(200), withheld.next())
            .await
            .is_err()
    );
    assert_eq!(
        control(&engine, resume()).await,
        [("ResumeSchedulerReqOutput", true)]
    );
    let mut chunks = Vec::new();
    while let Some(chunk) = tokio::time::timeout(Duration::from_secs(10), withheld.next())
        .await
        .unwrap()
    {
        chunks.push(chunk.unwrap());
    }
    assert_eq!(tokens(&chunks), [1000, 1001, 1002]);
}

/// A JSON schema the toy vocabulary takes most of a second to compile.
fn slow_schema() -> String {
    let names: Vec<String> = (0..3000).map(|i| format!("p{i}")).collect();
    let properties: serde_json::Map<String, serde_json::Value> = names
        .iter()
        .map(|name| {
            (
                name.clone(),
                serde_json::json!({"type": "string", "pattern": "[ab]{3,9}x"}),
            )
        })
        .collect();
    serde_json::json!({"type": "object", "properties": properties, "required": names}).to_string()
}

#[tokio::test(flavor = "multi_thread")]
async fn a_drain_cancels_the_requests_still_compiling_their_grammars() {
    let release = ControlRequest::ReleaseMemoryOccupation(ReleaseMemoryOccupationReqInput {
        rid: None,
        http_worker_ipc: None,
        tags: None,
    });
    for (drain, undo) in [
        (pause("wait"), resume()),
        (
            release,
            ControlRequest::ResumeMemoryOccupation(ResumeMemoryOccupationReqInput {
                rid: None,
                http_worker_ipc: None,
                tags: None,
            }),
        ),
    ] {
        let engine = start_ranks_with(
            Layout::Single,
            1,
            Tiers::DeviceOnly,
            Setup {
                grammar: true,
                ..Setup::default()
            },
        )
        .await;
        let mut cold = request("cold", 8, budget(6));
        cold.sampling_params.json_schema = Some(slow_schema());
        let stream = engine.client.generate(cold).await.unwrap();
        // The drain arrives while the grammar compiles: the request is not
        // in the scheduler, so the drain must end it rather than answer
        // with it withheld until the resume.
        let tag = drain.tag();
        let drained = control(&engine, drain).await;
        assert_eq!(drained.len(), 1);
        assert!(drained[0].1, "{tag}");
        let chunks = tokio::time::timeout(Duration::from_secs(1), collect(stream))
            .await
            .unwrap_or_else(|_| panic!("{tag} answered while the request still waits"));
        let last = chunks.last().unwrap();
        assert_eq!(last.finish_reason.as_deref(), Some("abort"), "{tag}");
        let details = &last.finish_details.as_ref().unwrap().0;
        assert!(
            details
                .iter()
                .any(|(k, v)| k == "message" && *v == Value::from("Aborted by pause")),
            "{tag}: {details:?}"
        );
        assert!(tokens(&chunks).is_empty());
        assert!(control(&engine, undo).await[0].1);
        // The engine serves on.
        let mut warm = request("warm", 8, budget(2));
        warm.sampling_params.regex = Some("a(b|x)".to_owned());
        assert_eq!(tokens(&run(&engine.client, warm).await), [0, 1]);
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn an_abort_pause_aborts_what_runs_and_a_keep_pause_freezes_it() {
    let engine = start_engine(1).await;
    let running = engine
        .client
        .generate(request("r1", 3, budget(100_000)))
        .await
        .unwrap();
    let (paused, chunks) = tokio::join!(control(&engine, pause("abort")), collect(running));
    assert_eq!(paused, [("PauseSchedulerReqOutput", true)]);
    assert_eq!(
        chunks.last().unwrap().finish_reason.as_deref(),
        Some("abort")
    );
    assert_eq!(
        control(&engine, resume()).await,
        [("ResumeSchedulerReqOutput", true)]
    );

    // keep: the request stops where it is and continues after the resume.
    let mut frozen = engine
        .client
        .generate(request("r2", 3, budget(30)))
        .await
        .unwrap();
    let first = tokio::time::timeout(Duration::from_secs(10), frozen.next())
        .await
        .unwrap();
    assert!(first.unwrap().is_ok());
    assert_eq!(
        control(&engine, pause("keep")).await,
        [("PauseSchedulerReqOutput", true)]
    );
    // Drain what was dispatched before the freeze, then nothing moves.
    tokio::time::sleep(Duration::from_millis(100)).await;
    let mut before = 1;
    while let Ok(Some(chunk)) = tokio::time::timeout(Duration::from_millis(50), frozen.next()).await
    {
        chunk.unwrap();
        before += 1;
    }
    assert!(before < 30, "the request kept running while frozen");
    assert_eq!(
        control(&engine, resume()).await,
        [("ResumeSchedulerReqOutput", true)]
    );
    let mut rest = Vec::new();
    while let Some(chunk) = tokio::time::timeout(Duration::from_secs(10), frozen.next())
        .await
        .unwrap()
    {
        rest.push(chunk.unwrap());
    }
    assert_eq!(
        rest.last().unwrap().finish_reason.as_deref(),
        Some("length")
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn flushes_and_weight_updates_run_on_every_data_parallel_rank() {
    let engine = start_ranks(Layout::Dp(2), 1, Tiers::DeviceOnly).await;
    let flush = ControlRequest::FlushCache(FlushCacheReqInput {
        rid: None,
        http_worker_ipc: None,
    });
    assert_eq!(
        control(&engine, flush).await,
        [("FlushCacheReqOutput", true), ("FlushCacheReqOutput", true)]
    );
    let load = ControlRequest::UpdateWeightFromDisk(UpdateWeightFromDiskReqInput {
        rid: None,
        http_worker_ipc: None,
        model_path: "/checkpoint".to_owned(),
        load_format: None,
        weight_version: None,
        flush_cache: true,
    });
    assert_eq!(
        control(&engine, load).await,
        [
            ("UpdateWeightFromDiskReqOutput", true),
            ("UpdateWeightFromDiskReqOutput", true)
        ]
    );
    for rank in 0..2 {
        assert_eq!(
            *engine.ranks[rank].device_ops.lock().unwrap(),
            ["UpdateWeightFromDiskReqInput"]
        );
        assert_eq!(*engine.ranks[rank].weight_commits.lock().unwrap(), [true]);
    }
    // Serving goes on.
    let chunks = run(&engine.client, request("r", 3, budget(2))).await;
    assert_eq!(tokens(&chunks), [1000, 1001]);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_failed_weight_load_is_committed_as_invalid_on_every_rank() {
    for layout in [Layout::Dp(2), Layout::Tp(2), Layout::Pp(2)] {
        let engine = start_ranks_with(
            layout,
            1,
            Tiers::DeviceOnly,
            Setup {
                weight_update_failure_rank: Some(1),
                ..Setup::default()
            },
        )
        .await;
        let load = ControlRequest::UpdateWeightFromDisk(UpdateWeightFromDiskReqInput {
            rid: None,
            http_worker_ipc: None,
            model_path: "/checkpoint".to_owned(),
            load_format: None,
            weight_version: None,
            flush_cache: true,
        });
        let replies = control(&engine, load).await;
        assert_eq!(replies.len(), layout.sizes().2);
        assert!(replies
            .iter()
            .all(|(tag, success)| { *tag == "UpdateWeightFromDiskReqOutput" && !success }));
        for rank in &engine.ranks {
            assert_eq!(*rank.weight_commits.lock().unwrap(), [false]);
        }
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_released_pd_engine_refuses_requests_until_it_wakes() {
    let (prefill, decode, _wire) =
        start_pd_pair(1, Layout::Single, Tiers::DeviceOnly, Setup::default()).await;
    let release = ControlRequest::ReleaseMemoryOccupation(ReleaseMemoryOccupationReqInput {
        rid: None,
        http_worker_ipc: None,
        tags: None,
    });
    assert_eq!(
        control(&decode, release).await,
        [("ReleaseMemoryOccupationReqOutput", true)]
    );
    // Its transfer would name memory that is gone: an abort, not a wait.
    let refused = run(&decode.client, disaggregated("asleep", 90, 20, 4)).await;
    let last = refused.last().unwrap();
    assert_eq!(last.finish_reason.as_deref(), Some("abort"));
    let message = last.finish_details.as_ref().unwrap().0[1].1.clone();
    assert_eq!(
        message,
        "PD engine is releasing or has released its GPU memory; retry after \
         resume_memory_occupation"
            .into()
    );
    let wake = ControlRequest::ResumeMemoryOccupation(ResumeMemoryOccupationReqInput {
        rid: None,
        http_worker_ipc: None,
        tags: None,
    });
    assert_eq!(
        control(&decode, wake).await,
        [("ResumeMemoryOccupationReqOutput", true)]
    );
    let request = disaggregated("awake", 91, 20, 4);
    let (prefilled, decoded) = tokio::join!(
        run(&prefill.client, request.clone()),
        run(&decode.client, request)
    );
    assert_eq!(
        prefilled.last().unwrap().finish_reason.as_deref(),
        Some("length")
    );
    assert_eq!(tokens(&decoded).len(), 4);
}

#[tokio::test(flavor = "multi_thread")]
async fn memory_is_released_after_the_drain_and_resumed_in_reverse_order() {
    let engine = start_engine(1).await;
    let release = ControlRequest::ReleaseMemoryOccupation(ReleaseMemoryOccupationReqInput {
        rid: None,
        http_worker_ipc: None,
        tags: None,
    });
    let running = engine
        .client
        .generate(request("r1", 3, budget(20)))
        .await
        .unwrap();
    let (released, chunks) = tokio::join!(control(&engine, release), collect(running));
    assert_eq!(released, [("ReleaseMemoryOccupationReqOutput", true)]);
    assert_eq!(tokens(&chunks).len(), 20);
    let sleeping = ControlRequest::IsSleeping(IsSleepingReqInput {
        rid: None,
        http_worker_ipc: None,
    });
    assert_eq!(
        control(&engine, sleeping.clone()).await,
        [("IsSleepingReqOutput", true)]
    );
    // A pause cannot undo a release.
    assert_eq!(
        control(&engine, resume()).await,
        [("ResumeSchedulerReqOutput", false)]
    );
    let wake = ControlRequest::ResumeMemoryOccupation(ResumeMemoryOccupationReqInput {
        rid: None,
        http_worker_ipc: None,
        tags: None,
    });
    assert_eq!(
        control(&engine, wake).await,
        [("ResumeMemoryOccupationReqOutput", true)]
    );
    assert_eq!(
        control(&engine, sleeping).await,
        [("IsSleepingReqOutput", false)]
    );
    assert_eq!(
        *engine.ranks[0].device_ops.lock().unwrap(),
        ["release kv_cache,weights", "resume weights,kv_cache"]
    );
    let profile = ControlRequest::Profile(ProfileReq {
        rid: None,
        http_worker_ipc: None,
        kind: PROFILE_START,
        output_dir: None,
        start_step: None,
        num_steps: Some(3),
        activities: None,
        profile_by_stage: false,
        with_stack: None,
        record_shapes: None,
        profile_id: None,
    });
    assert_eq!(
        control(&engine, profile).await,
        [("ProfileReqOutput", true)]
    );
    let chunks = run(&engine.client, request("r2", 3, budget(2))).await;
    assert_eq!(tokens(&chunks), [1000, 1001]);
}

#[tokio::test(flavor = "multi_thread")]
async fn memory_release_waits_for_the_host_stores_in_flight() {
    // The devices hold every cache acknowledgment until the test opens their
    // gates; then the odd ranks of a mirrored replica acknowledge three polls
    // late. Under write-through a finished request's prefix is stored to the
    // Host; under write-back, what the third of three 3000-token prompts
    // evicts of the first one. Either way the KV cache cannot be cleared, nor
    // released, until the stores land.
    for layout in [Layout::Single, Layout::Tp(2), Layout::TpKvp(2, 2)] {
        for (tiers, prompts) in [
            (Tiers::HostL2, &[("r1", 0, 65)][..]),
            (
                Tiers::HostL2WriteBack,
                &[("a", 0, 3000), ("b", 10_000, 3000), ("c", 20_000, 3000)][..],
            ),
        ] {
            let write_back = matches!(tiers, Tiers::HostL2WriteBack);
            let case = format!("{layout:?}, write-back {write_back}");
            let engine = start_ranks_with(
                layout,
                1,
                tiers,
                Setup {
                    cache_ack_polls: Some(|rank| 3 * (rank % 2)),
                    held_cache_acks: true,
                    ..Setup::default()
                },
            )
            .await;
            for &(rid, first, len) in prompts {
                let chunks = run(
                    &engine.client,
                    TokenizedGenerateReqInput {
                        input_ids: Some((first + 1..=first + len).collect()),
                        ..request(rid, 1, budget(2))
                    },
                )
                .await;
                assert_eq!(tokens(&chunks), [1000, 1001], "{rid} on {case}");
            }
            for _ in 0..500 {
                if engine.cache_ops(CacheOpKind::WriteBack) > 0 {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
            assert!(engine.cache_ops(CacheOpKind::WriteBack) > 0, "{case}");
            for rank in &engine.ranks {
                assert!(rank.acknowledged.lock().unwrap().is_empty(), "{case}");
            }
            let release =
                ControlRequest::ReleaseMemoryOccupation(ReleaseMemoryOccupationReqInput {
                    rid: None,
                    http_worker_ipc: None,
                    tags: Some(vec!["kv_cache".to_owned()]),
                });
            let mut release = std::pin::pin!(engine.client.control(release));
            assert!(
                tokio::time::timeout(Duration::from_millis(200), release.as_mut())
                    .await
                    .is_err(),
                "released with Host stores in flight on {case}"
            );
            for rank in &engine.ranks {
                assert!(rank.device_ops.lock().unwrap().is_empty(), "{case}");
                rank.ack_gate.open();
            }
            let released = tokio::time::timeout(Duration::from_secs(10), release)
                .await
                .expect("no reply to the release once the stores landed")
                .unwrap();
            assert_eq!(
                outcome(released),
                [("ReleaseMemoryOccupationReqOutput", true)],
                "{case}"
            );
            let wake = ControlRequest::ResumeMemoryOccupation(ResumeMemoryOccupationReqInput {
                rid: None,
                http_worker_ipc: None,
                tags: Some(vec!["kv_cache".to_owned()]),
            });
            assert_eq!(
                control(&engine, wake).await,
                [("ResumeMemoryOccupationReqOutput", true)],
                "{case}"
            );
            // The first prompt is served exactly after the wake. Under
            // write-through it hits its full pages on the Host. Under
            // write-back the Host holds only the first prompt's last pages,
            // which continued a prefix the cleared Device alone held: nothing
            // reaches them, and nothing hits.
            let (_, first, len) = prompts[0];
            let chunks = run(
                &engine.client,
                TokenizedGenerateReqInput {
                    input_ids: Some((first + 1..=first + len).collect()),
                    ..request("again", 1, budget(2))
                },
            )
            .await;
            assert_eq!(tokens(&chunks), [1000, 1001], "{case}");
            assert_eq!(
                chunks.last().unwrap().cached_tokens,
                if write_back { 0 } else { 64 },
                "{case}"
            );
            // Each rank released and resumed the KV cache once; a follower may
            // get there after the first member replied.
            engine.settle().await;
            for (rank, member) in engine.ranks.iter().enumerate() {
                assert_eq!(
                    *member.device_ops.lock().unwrap(),
                    ["release kv_cache", "resume kv_cache"],
                    "rank {rank}, {case}"
                );
            }
        }
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_drain_cancels_the_requests_still_receiving_their_embeddings() {
    let release = ControlRequest::ReleaseMemoryOccupation(ReleaseMemoryOccupationReqInput {
        rid: None,
        http_worker_ipc: None,
        tags: None,
    });
    let wake = ControlRequest::ResumeMemoryOccupation(ResumeMemoryOccupationReqInput {
        rid: None,
        http_worker_ipc: None,
        tags: None,
    });
    for (drain, undo) in [(pause("wait"), resume()), (release, wake)] {
        let engine = start_ranks_with(
            Layout::Single,
            1,
            Tiers::DeviceOnly,
            Setup {
                epd_receive_enabled: true,
                ..Setup::default()
            },
        )
        .await;
        // The drain arrives while the request waits for its image's
        // embeddings: it is not in the scheduler, so the drain must end it
        // rather than answer with it staged until the resume.
        let stream = engine.client.generate(epd_request("slow-0")).await.unwrap();
        let tag = drain.tag();
        let drained = control(&engine, drain).await;
        assert_eq!(drained.len(), 1);
        assert!(drained[0].1, "{tag}");
        let chunks = tokio::time::timeout(Duration::from_secs(1), collect(stream))
            .await
            .unwrap_or_else(|_| panic!("{tag} answered while the request still waits"));
        let last = chunks.last().unwrap();
        assert_eq!(last.finish_reason.as_deref(), Some("abort"), "{tag}");
        let details = &last.finish_details.as_ref().unwrap().0;
        assert!(
            details
                .iter()
                .any(|(k, v)| k == "message" && *v == Value::from("Aborted by pause")),
            "{tag}: {details:?}"
        );
        assert!(control(&engine, undo).await[0].1);
        // The engine serves on.
        assert_eq!(
            tokens(&run(&engine.client, epd_request("img-1")).await),
            [1000, 1001, 1002]
        );
    }
}

#[test]
fn an_engine_stops_once_its_frontend_is_gone() {
    // The frontend's last message is a request, then it is gone: every rank
    // of the replica stops after that round, without a stop request.
    let script = vec![vec![EngineRequest::Add(Box::new(request(
        "r1",
        5,
        budget(3),
    )))]];
    for (rank, outcome) in run_scripted(
        Layout::Tp(2),
        Tiers::DeviceOnly,
        Setup::default(),
        vec![script],
        no_reaction,
    )
    .into_iter()
    .enumerate()
    {
        assert_eq!(outcome.outcome, Ok(()), "rank {rank}");
    }
}

#[test]
fn a_frozen_data_parallel_rank_keeps_its_rebalance_in_step() {
    // Rank 1 is frozen while rank 0 forwards its request, then both get a
    // rebalance trigger in the same round. The frozen rank's idle forwards
    // count like the busy rank's real ones, so both rebalances sit in the
    // same phase and the trigger runs the same collectives on both ranks.
    let rounds = 80;
    let mut first: Script = vec![Vec::new(); rounds];
    let mut second: Script = vec![Vec::new(); rounds];
    first[0].push(EngineRequest::Add(Box::new(request("r0", 20, budget(12)))));
    second[0].push(EngineRequest::Control(Box::new(pause("keep"))));
    for script in [&mut first, &mut second] {
        script[40].push(EngineRequest::Control(Box::new(
            ControlRequest::RebalanceExperts(ts_wire::control::RebalanceExpertsReqInput {
                rid: None,
                http_worker_ipc: None,
            }),
        )));
    }
    second[60].push(EngineRequest::Control(Box::new(resume())));
    let ranks = run_scripted(
        Layout::Dp(2),
        Tiers::DeviceOnly,
        Setup {
            eplb: Some(eplb_config),
            ..Setup::default()
        },
        vec![first, second],
        no_reaction,
    );
    for (index, rank) in ranks.iter().enumerate() {
        assert_eq!(rank.outcome, Ok(()), "rank {index}");
    }
    assert_eq!(ranks[0].finished, [("r0".to_owned(), "length".to_owned())]);
    assert!(ranks[1].idle_forwards > 0);
    let rebalanced = |rank: &ScriptedRank| -> Vec<bool> {
        rank.replies
            .iter()
            .filter_map(|reply| match reply {
                ControlOutput::RebalanceExperts(reply) => Some(reply.success),
                _ => None,
            })
            .collect()
    };
    assert_eq!(rebalanced(&ranks[0]).len(), 1);
    assert_eq!(rebalanced(&ranks[0]), rebalanced(&ranks[1]));
}

#[test]
fn a_flush_under_a_keep_pause_runs_once_the_cache_work_did() {
    // A finished request's prefix pages are written back to Host in the
    // round after it ends; the keep pause its finish triggers freezes the
    // engine first. Frozen rounds still plan cache work (the scheduler only
    // forbids forwards), so the write-back runs, and a flush under the
    // pause finds no Host copy in flight.
    fn freeze_on_finish(outputs: &[StreamedOutput]) -> Vec<EngineRequest> {
        if outputs.iter().any(|output| output.finish_reason.is_some()) {
            vec![EngineRequest::Control(Box::new(pause("keep")))]
        } else {
            Vec::new()
        }
    }
    let mut script: Script = vec![Vec::new(); 40];
    script[0].push(EngineRequest::Add(Box::new(request("r1", 65, budget(2)))));
    script[30].push(EngineRequest::Control(Box::new(
        ControlRequest::FlushCache(FlushCacheReqInput {
            rid: None,
            http_worker_ipc: None,
        }),
    )));
    let ranks = run_scripted(
        Layout::Single,
        Tiers::HostL2,
        Setup::default(),
        vec![script],
        freeze_on_finish,
    );
    let rank = &ranks[0];
    assert_eq!(rank.outcome, Ok(()));
    assert_eq!(rank.finished, [("r1".to_owned(), "length".to_owned())]);
    assert_eq!(
        outcome(rank.replies.clone()),
        [
            ("PauseSchedulerReqOutput", true),
            ("FlushCacheReqOutput", true)
        ]
    );
    assert!(rank.cache_ops.contains(&CacheOpKind::WriteBack));
}

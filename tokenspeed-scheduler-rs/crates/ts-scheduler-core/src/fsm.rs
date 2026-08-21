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

//! Request lifecycle state machine.
//!
//! Ported from `tokenspeed-scheduler/csrc/fsm/*`. The C++ `std::variant` state
//! and event-struct visitor pattern map directly to a Rust `State` enum with a
//! `FsmEvent` enum; transitions that free request tables receive the
//! coordinator explicitly (C++ stored a raw `CacheCoordinator*` in each event).
//! `TokenContainer` is shared through `Rc<RefCell<..>>` (C++ used a raw
//! pointer into the owning `Request`).

use std::cell::RefCell;
use std::rc::Rc;

use crate::block_table::BlockTable;
use crate::cache_coordinator::CacheCoordinator;
use crate::cache_ops::free_request;
use crate::cache_types::BlockTransfer;
use crate::req_pool_allocator::{self, ReqPoolIndex};
use crate::token_container::{TokenContainer, Window};

/// Whether a prefill runs on this rank (local) or is split across PD.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefillSource {
    Local,
    Remote,
}

/// One source of truth for both the next hash-chain seed and the cumulative
/// history needed to publish a resumable boundary across chunk edges.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CacheProgress {
    pub prefix_hashes: Vec<String>,
    pub access_epoch: u64,
    /// Pending closed-prefix boundary; zero once published or when absent.
    pub promotion_boundary_tokens: i32,
}

/// Shared fields of the forward states (C++ `ForwardState` base).
#[derive(Debug)]
pub struct ForwardState {
    pub token_container: Rc<RefCell<TokenContainer>>,
    pub prefix_granularity: i32,
    pub req_pool_index: Option<ReqPoolIndex>,
    pub block_tables: Vec<BlockTable>,
    pub cache_progress: CacheProgress,
}

impl ForwardState {
    pub fn token_container(&self) -> &Rc<RefCell<TokenContainer>> {
        &self.token_container
    }
    pub fn prefix_granularity(&self) -> i32 {
        self.prefix_granularity
    }
    pub fn request_pool_index(&self) -> i32 {
        self.req_pool_index.as_ref().map_or(-1, |idx| idx.slot())
    }
    pub fn take_request_pool_index(&mut self) -> Option<ReqPoolIndex> {
        self.req_pool_index.take()
    }
    pub fn block_tables(&self) -> &[BlockTable] {
        &self.block_tables
    }
    pub fn block_tables_mut(&mut self) -> &mut Vec<BlockTable> {
        &mut self.block_tables
    }
    pub fn take_block_tables(&mut self) -> Vec<BlockTable> {
        std::mem::take(&mut self.block_tables)
    }
    pub fn cache_progress(&self) -> &CacheProgress {
        &self.cache_progress
    }
    pub fn take_cache_progress(&mut self) -> CacheProgress {
        std::mem::take(&mut self.cache_progress)
    }
}

/// Ready to schedule its first prefill chunk.
#[derive(Debug)]
pub struct Submitted {
    pub token_container: Rc<RefCell<TokenContainer>>,
    pub prefix_granularity: i32,
}

/// PD decode-side: waiting for remote bootstrap.
#[derive(Debug)]
pub struct Bootstrapping {
    pub token_container: Rc<RefCell<TokenContainer>>,
    pub prefix_granularity: i32,
}

/// Mid-prefill: `window` covers the chunk already dispatched.
#[derive(Debug)]
pub struct Prefilling {
    pub forward: ForwardState,
    pub window: Window,
    pub reserve_num_tokens_in_next_schedule_event: i32,
    pub source: PrefillSource,
}

/// Prefill complete; decode may start.
#[derive(Debug)]
pub struct PrefillDone {
    pub forward: ForwardState,
    pub window: Window,
    pub reserve_num_tokens_in_next_schedule_event: i32,
}

/// Decoding (one step per forward; generated tokens accumulate in the
/// container).
#[derive(Debug)]
pub struct Decoding {
    pub forward: ForwardState,
    pub reserve_num_tokens_in_next_schedule_event: i32,
}

/// Decode retracted: all accepted tokens folded back into the prefill window.
#[derive(Debug)]
pub struct Retracted {
    pub token_container: Rc<RefCell<TokenContainer>>,
    pub prefix_granularity: i32,
}

/// Terminal.
#[derive(Debug, Default)]
pub struct Finished;

/// The request state variant.
#[derive(Debug)]
pub enum State {
    Bootstrapping(Bootstrapping),
    Submitted(Submitted),
    Prefilling(Prefilling),
    PrefillDone(PrefillDone),
    Decoding(Decoding),
    Retracted(Retracted),
    Finished(Finished),
}

impl State {
    pub fn state_name(&self) -> &'static str {
        match self {
            State::Bootstrapping(_) => "Bootstrapping",
            State::Submitted(_) => "Submitted",
            State::Prefilling(_) => "Prefilling",
            State::PrefillDone(_) => "PrefillDone",
            State::Decoding(_) => "Decoding",
            State::Retracted(_) => "Retracted",
            State::Finished(_) => "Finished",
        }
    }

    /// Shared forward fields, when the state is a forward state.
    pub fn forward(&self) -> Option<&ForwardState> {
        match self {
            State::Prefilling(s) => Some(&s.forward),
            State::PrefillDone(s) => Some(&s.forward),
            State::Decoding(s) => Some(&s.forward),
            _ => None,
        }
    }

    pub fn forward_mut(&mut self) -> Option<&mut ForwardState> {
        match self {
            State::Prefilling(s) => Some(&mut s.forward),
            State::PrefillDone(s) => Some(&mut s.forward),
            State::Decoding(s) => Some(&mut s.forward),
            _ => None,
        }
    }
}

/// Shifted input ids: the prefill window shifted one token right, padded with
/// `-1` to the window length.
pub fn compute_shifted_input_ids(
    token_container: &Rc<RefCell<TokenContainer>>,
    window: Window,
) -> Vec<i32> {
    let tc = token_container.borrow();
    let shifted_start = window.begin + 1;
    let shifted_end = tc.prefill_size().min(shifted_start + window.size);
    let shifted_size = (shifted_end - shifted_start).max(0);
    let mut shifted = Vec::with_capacity(window.size as usize);
    if shifted_size > 0 {
        shifted.extend_from_slice(tc.token_slice(Window {
            begin: shifted_start,
            size: shifted_size,
        }));
    }
    shifted.resize(window.size as usize, -1);
    shifted
}

/// Data for the first-chunk scheduling event.
pub struct SchedulePrefillFirstChunk {
    pub tokens_this_round: i32,
    pub reserve_num_tokens_in_next_schedule_event: i32,
    pub req_pool_allocator: Rc<RefCell<crate::req_pool_allocator::ReqPoolAllocator>>,
    pub source: PrefillSource,
    pub block_tables: Vec<BlockTable>,
    pub hit_tokens: i32,
    pub cache_progress: CacheProgress,
    pub load_pairs: Vec<BlockTransfer>,
}

/// Data for a subsequent-chunk scheduling event.
pub struct SchedulePrefill {
    pub tokens_this_round: i32,
    pub reserve_num_tokens_in_next_schedule_event: i32,
    pub cache_progress: CacheProgress,
}

/// Data for the decode scheduling event.
pub struct ScheduleDecode {
    pub decode_input_tokens: i32,
    pub cache_progress: CacheProgress,
}

/// The full set of lifecycle events, mirroring C++ `fsm::*Event`.
pub enum FsmEvent {
    SchedulePrefillFirstChunk(SchedulePrefillFirstChunk),
    SchedulePrefill(SchedulePrefill),
    ScheduleDecode(ScheduleDecode),
    Finish,
    Abort,
    Retraction,
    Retract,
    UpdateReserveNumTokens(i32),
    ExtendResult(Vec<i32>),
    Bootstrapped,
    Succeeded,
    RemotePrefillDone(i32),
}

impl FsmEvent {
    /// Apply the event to `state`. `coordinator` is required only by events
    /// that release request tables (`Finish`/`Abort`/`Retraction`/`Retract`);
    /// the C++ events carried a raw coordinator pointer.
    pub fn apply(self, state: State, coordinator: &mut CacheCoordinator) -> State {
        match self {
            FsmEvent::SchedulePrefillFirstChunk(ev) => match state {
                State::Submitted(s) => {
                    ev.schedule_first_chunk(s.token_container, s.prefix_granularity, coordinator)
                }
                State::Retracted(s) => {
                    ev.schedule_first_chunk(s.token_container, s.prefix_granularity, coordinator)
                }
                other => invalid_transition("SchedulePrefillFirstChunkEvent", other.state_name()),
            },
            FsmEvent::SchedulePrefill(ev) => match state {
                State::Prefilling(s) => ev.schedule_prefill(s),
                other => invalid_transition("SchedulePrefillEvent", other.state_name()),
            },
            FsmEvent::ScheduleDecode(ev) => match state {
                State::PrefillDone(s) => State::Decoding(ev.decode(s)),
                State::Decoding(s) => State::Decoding(ev.decode_decoding(s)),
                other => invalid_transition("ScheduleDecodeEvent", other.state_name()),
            },
            FsmEvent::Finish => match state {
                State::PrefillDone(s) => finish(s.forward, coordinator),
                State::Decoding(s) => finish(s.forward, coordinator),
                State::Retracted(_) => State::Finished(Finished),
                other => invalid_transition("FinishEvent", other.state_name()),
            },
            FsmEvent::Abort => match state {
                State::Bootstrapping(_) | State::Submitted(_) | State::Retracted(_) => {
                    State::Finished(Finished)
                }
                State::Prefilling(s) => finish(s.forward, coordinator),
                State::PrefillDone(s) => finish(s.forward, coordinator),
                State::Decoding(s) => finish(s.forward, coordinator),
                State::Finished(_) => State::Finished(Finished),
            },
            FsmEvent::Retraction => match state {
                State::Decoding(mut s) => {
                    s.forward.token_container.borrow_mut().rebase_prefill();
                    let token_container = s.forward.token_container.clone();
                    let prefix_granularity = s.forward.prefix_granularity;
                    let tables = s.forward.take_block_tables();
                    free_request(coordinator, &mut tables.into_iter().collect::<Vec<_>>());
                    State::Retracted(Retracted {
                        token_container,
                        prefix_granularity,
                    })
                }
                other => invalid_transition("RetractionEvent", other.state_name()),
            },
            FsmEvent::Retract => match state {
                State::PrefillDone(s) => retract(s.forward, coordinator),
                State::Decoding(s) => retract(s.forward, coordinator),
                other => invalid_transition("RetractEvent", other.state_name()),
            },
            FsmEvent::UpdateReserveNumTokens(value) => match state {
                State::Decoding(mut s) => {
                    s.reserve_num_tokens_in_next_schedule_event = value;
                    State::Decoding(s)
                }
                State::Finished(_) => State::Finished(Finished),
                other => invalid_transition("UpdateReserveNumTokensEvent", other.state_name()),
            },
            FsmEvent::ExtendResult(tokens) => match state {
                State::PrefillDone(s) => {
                    s.forward.token_container.borrow_mut().extend(&tokens);
                    State::PrefillDone(s)
                }
                State::Decoding(s) => {
                    s.forward.token_container.borrow_mut().extend(&tokens);
                    State::Decoding(s)
                }
                State::Finished(_) => State::Finished(Finished),
                other => invalid_transition("ExtendResultEvent", other.state_name()),
            },
            FsmEvent::Bootstrapped => match state {
                State::Bootstrapping(s) => State::Submitted(Submitted {
                    token_container: s.token_container,
                    prefix_granularity: s.prefix_granularity,
                }),
                other => invalid_transition("BootstrappedEvent", other.state_name()),
            },
            FsmEvent::Succeeded => match state {
                State::Decoding(_) => State::Finished(Finished),
                other => invalid_transition("SucceededEvent", other.state_name()),
            },
            FsmEvent::RemotePrefillDone(bootstrap_token) => match state {
                State::Prefilling(s) => {
                    let Prefilling {
                        forward,
                        window,
                        reserve_num_tokens_in_next_schedule_event,
                        ..
                    } = s;
                    let done = PrefillDone {
                        forward,
                        window,
                        reserve_num_tokens_in_next_schedule_event,
                    };
                    done.forward
                        .token_container
                        .borrow_mut()
                        .extend(&[bootstrap_token]);
                    State::PrefillDone(done)
                }
                other => invalid_transition("RemotePrefillDoneEvent", other.state_name()),
            },
        }
    }
}

impl SchedulePrefillFirstChunk {
    fn schedule_first_chunk(
        self,
        token_container: Rc<RefCell<TokenContainer>>,
        prefix_granularity: i32,
        coordinator: &CacheCoordinator,
    ) -> State {
        assert!(
            self.block_tables.len() == coordinator.num_groups() as usize,
            "SchedulePrefillFirstChunkEvent requires one admitted table per cache group"
        );
        let req_pool_index = req_pool_allocator::allocate(&self.req_pool_allocator);
        let window = Window {
            begin: self.hit_tokens,
            size: self.tokens_this_round,
        };
        let is_last_chunk = window.begin + window.size == token_container.borrow().prefill_size();
        let forward = ForwardState {
            token_container,
            prefix_granularity,
            req_pool_index: Some(req_pool_index),
            block_tables: self.block_tables,
            cache_progress: self.cache_progress,
        };
        if is_last_chunk && self.source == PrefillSource::Local {
            State::PrefillDone(PrefillDone {
                forward,
                window,
                reserve_num_tokens_in_next_schedule_event: self
                    .reserve_num_tokens_in_next_schedule_event,
            })
        } else {
            State::Prefilling(Prefilling {
                forward,
                window,
                reserve_num_tokens_in_next_schedule_event: self
                    .reserve_num_tokens_in_next_schedule_event,
                source: self.source,
            })
        }
    }
}

impl SchedulePrefill {
    fn schedule_prefill(self, state: Prefilling) -> State {
        let Prefilling {
            forward,
            window,
            reserve_num_tokens_in_next_schedule_event: _,
            source,
        } = state;
        let new_window = Window {
            begin: window.begin + window.size,
            size: self.tokens_this_round,
        };
        let is_last_chunk =
            new_window.begin + new_window.size == forward.token_container.borrow().prefill_size();
        if is_last_chunk {
            State::PrefillDone(PrefillDone {
                forward,
                window: new_window,
                reserve_num_tokens_in_next_schedule_event: self
                    .reserve_num_tokens_in_next_schedule_event,
            })
        } else {
            State::Prefilling(Prefilling {
                forward,
                window: new_window,
                reserve_num_tokens_in_next_schedule_event: self
                    .reserve_num_tokens_in_next_schedule_event,
                source,
            })
        }
    }
}

impl ScheduleDecode {
    fn decode(self, state: PrefillDone) -> Decoding {
        let PrefillDone { forward, .. } = state;
        Decoding {
            forward,
            reserve_num_tokens_in_next_schedule_event: self.decode_input_tokens,
        }
    }

    fn decode_decoding(self, state: Decoding) -> Decoding {
        let Decoding { forward, .. } = state;
        Decoding {
            forward,
            reserve_num_tokens_in_next_schedule_event: self.decode_input_tokens,
        }
    }
}

fn finish(mut forward: ForwardState, coordinator: &mut CacheCoordinator) -> State {
    let tables = forward.take_block_tables();
    free_request(coordinator, &mut tables.into_iter().collect::<Vec<_>>());
    State::Finished(Finished)
}

fn retract(mut forward: ForwardState, coordinator: &mut CacheCoordinator) -> State {
    forward.token_container.borrow_mut().rebase_prefill();
    let token_container = forward.token_container.clone();
    let prefix_granularity = forward.prefix_granularity;
    let tables = forward.take_block_tables();
    free_request(coordinator, &mut tables.into_iter().collect::<Vec<_>>());
    State::Submitted(Submitted {
        token_container,
        prefix_granularity,
    })
}

fn invalid_transition(event: &str, state: &str) -> ! {
    // The C++ `InvalidTransitionHandler` reports `detail::TypeName<>`, which
    // includes the `tokenspeed::fsm::` namespace prefix; tests match on it.
    panic!("FSM transition invalid: event=tokenspeed::fsm::{event}; state=tokenspeed::fsm::{state}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block_pool::BlockPool;

    use crate::cache_coordinator::make_coordinator;
    use crate::cache_types::CacheGroupSpec;
    use crate::req_pool_allocator::ReqPoolAllocator;

    fn token_container(tokens: &[i32]) -> Rc<RefCell<TokenContainer>> {
        Rc::new(RefCell::new(TokenContainer::new(tokens.to_vec())))
    }

    fn coordinator() -> CacheCoordinator {
        let spec = CacheGroupSpec {
            kind: crate::cache_types::AttnKind::Full,
            cache_blocks_per_lcm_block: 1,
            block_granularity: 4,
            ..CacheGroupSpec::default()
        };
        make_coordinator(
            &[spec],
            4,
            Rc::new(RefCell::new(BlockPool::new(8))),
            None,
            false,
        )
    }

    fn one_table() -> Vec<BlockTable> {
        vec![BlockTable::default()]
    }

    fn first_chunk_event(
        tokens_this_round: i32,
        hit_tokens: i32,
        tables: Vec<BlockTable>,
    ) -> SchedulePrefillFirstChunk {
        SchedulePrefillFirstChunk {
            tokens_this_round,
            reserve_num_tokens_in_next_schedule_event: 1,
            req_pool_allocator: Rc::new(RefCell::new(ReqPoolAllocator::new(4))),
            source: PrefillSource::Local,
            block_tables: tables,
            hit_tokens,
            cache_progress: CacheProgress {
                prefix_hashes: Vec::new(),
                access_epoch: 1,
                promotion_boundary_tokens: 0,
            },
            load_pairs: Vec::new(),
        }
    }

    #[test]
    fn first_chunk_last_chunk_becomes_prefill_done() {
        let tc = token_container(&[1, 2, 3, 4, 5, 6, 7, 8]);
        let state = State::Submitted(Submitted {
            token_container: tc,
            prefix_granularity: 4,
        });
        let ev = FsmEvent::SchedulePrefillFirstChunk(first_chunk_event(8, 0, one_table()));
        let mut coord = coordinator();
        let next = ev.apply(state, &mut coord);
        assert!(matches!(next, State::PrefillDone(_)));
        assert_eq!(next.state_name(), "PrefillDone");
    }

    #[test]
    fn first_chunk_partial_becomes_prefilling_and_chains() {
        let tc = token_container(&[1, 2, 3, 4, 5, 6, 7, 8]);
        let state = State::Submitted(Submitted {
            token_container: tc,
            prefix_granularity: 4,
        });
        let ev = FsmEvent::SchedulePrefillFirstChunk(first_chunk_event(4, 0, one_table()));
        let mut coord = coordinator();
        let next = ev.apply(state, &mut coord);
        assert!(matches!(next, State::Prefilling(_)));
        // Continue with the second chunk -> PrefillDone.
        let ev2 = FsmEvent::SchedulePrefill(SchedulePrefill {
            tokens_this_round: 4,
            reserve_num_tokens_in_next_schedule_event: 1,
            cache_progress: CacheProgress {
                prefix_hashes: Vec::new(),
                access_epoch: 1,
                promotion_boundary_tokens: 0,
            },
        });
        let final_state = ev2.apply(next, &mut coord);
        assert!(matches!(final_state, State::PrefillDone(_)));
    }

    #[test]
    fn prefill_done_then_schedule_decode() {
        let tc = token_container(&[1, 2, 3, 4]);
        let state = State::Submitted(Submitted {
            token_container: tc,
            prefix_granularity: 4,
        });
        let ev = FsmEvent::SchedulePrefillFirstChunk(first_chunk_event(4, 0, one_table()));
        let mut coord = coordinator();
        let done = ev.apply(state, &mut coord);
        let ev2 = FsmEvent::ScheduleDecode(ScheduleDecode {
            decode_input_tokens: 1,
            cache_progress: CacheProgress {
                prefix_hashes: Vec::new(),
                access_epoch: 1,
                promotion_boundary_tokens: 0,
            },
        });
        let decoding = ev2.apply(done, &mut coord);
        assert!(matches!(decoding, State::Decoding(_)));
    }

    #[test]
    fn extend_result_appends_tokens() {
        let tc = token_container(&[1, 2, 3, 4]);
        let state = State::Submitted(Submitted {
            token_container: tc.clone(),
            prefix_granularity: 4,
        });
        let mut coord = coordinator();
        let done = FsmEvent::SchedulePrefillFirstChunk(first_chunk_event(4, 0, one_table()))
            .apply(state, &mut coord);
        let decoding = FsmEvent::ScheduleDecode(ScheduleDecode {
            decode_input_tokens: 1,
            cache_progress: CacheProgress {
                prefix_hashes: Vec::new(),
                access_epoch: 1,
                promotion_boundary_tokens: 0,
            },
        })
        .apply(done, &mut coord);
        let _ = FsmEvent::ExtendResult(vec![9, 10]).apply(decoding, &mut coord);
        assert_eq!(tc.borrow().size(), 6);
    }

    #[test]
    fn finish_releases_tables() {
        let p = Rc::new(RefCell::new(BlockPool::new(4)));
        let blocks: Vec<_> = p.borrow_mut().acquire_blocks(&p, 0, 1, 2);
        let tables = vec![crate::block_table::BlockTable::from_blocks(blocks, 0)];
        let tc = token_container(&[1, 2, 3, 4]);
        let state = State::Submitted(Submitted {
            token_container: tc,
            prefix_granularity: 4,
        });
        let mut coord = make_coordinator(
            &[CacheGroupSpec {
                kind: crate::cache_types::AttnKind::Full,
                cache_blocks_per_lcm_block: 1,
                block_granularity: 4,
                ..CacheGroupSpec::default()
            }],
            4,
            p.clone(),
            None,
            false,
        );
        let mut ev = first_chunk_event(4, 0, tables);
        // The event takes the tables; apply Finish on the resulting state.
        let done = FsmEvent::SchedulePrefillFirstChunk(std::mem::replace(
            &mut ev,
            first_chunk_event(0, 0, Vec::new()),
        ))
        .apply(state, &mut coord);
        assert!(matches!(done, State::PrefillDone(_)));
        assert_eq!(p.borrow().num_occupied_slots(), 2);
        let finished = FsmEvent::Finish.apply(done, &mut coord);
        assert!(matches!(finished, State::Finished(_)));
        assert_eq!(p.borrow().num_occupied_slots(), 0);
    }

    #[test]
    #[should_panic(
        expected = "FSM transition invalid: event=tokenspeed::fsm::ScheduleDecodeEvent; state=tokenspeed::fsm::Submitted"
    )]
    fn invalid_transition_panics() {
        let tc = token_container(&[1, 2, 3, 4]);
        let state = State::Submitted(Submitted {
            token_container: tc,
            prefix_granularity: 4,
        });
        let mut coord = coordinator();
        let _ = FsmEvent::ScheduleDecode(ScheduleDecode {
            decode_input_tokens: 1,
            cache_progress: CacheProgress {
                prefix_hashes: Vec::new(),
                access_epoch: 0,
                promotion_boundary_tokens: 0,
            },
        })
        .apply(state, &mut coord);
    }

    #[test]
    fn remote_prefill_done_builds_prefill_done_with_bootstrap_token() {
        let tc = token_container(&[1, 2, 3, 4]);
        let state = State::Submitted(Submitted {
            token_container: tc,
            prefix_granularity: 4,
        });
        let mut coord = coordinator();
        // Partial first chunk keeps the request in Prefilling.
        let prefilling = FsmEvent::SchedulePrefillFirstChunk(first_chunk_event(2, 0, one_table()))
            .apply(state, &mut coord);
        assert!(matches!(prefilling, State::Prefilling(_)));
        let done = FsmEvent::RemotePrefillDone(42).apply(prefilling, &mut coord);
        assert!(matches!(done, State::PrefillDone(_)));
        // The bootstrap token extends the container (8 + 42 appended? no: 4 prompt + bootstrap 42).
        let tc_size = match &done {
            State::PrefillDone(s) => s.forward.token_container.borrow().size(),
            _ => 0,
        };
        assert_eq!(tc_size, 5);
    }
}

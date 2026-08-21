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

//! One in-flight request: token container + lifecycle state.
//!
//! Ported from `tokenspeed-scheduler/csrc/scheduler/request.{h,cpp}`. The
//! `TokenContainer` is shared with the FSM states through `Rc<RefCell<..>>`
//! (C++ used a raw pointer into the owning `Request`); slice-returning
//! accessors return owned vectors instead of spans.

use std::cell::RefCell;
use std::rc::Rc;

use crate::block_table::BlockTable;
use crate::cache_coordinator::CacheCoordinator;
use crate::fsm::{
    Bootstrapping, CacheProgress, Decoding, Finished, FsmEvent, PrefillDone, PrefillSource,
    Prefilling, Retracted, State, Submitted,
};
use crate::request_spec::{PrefillInfo, RequestSpec};
use crate::token_container::{TokenContainer, Window};
use crate::types::Role;

/// One in-flight request.
pub struct Request {
    id: String,
    token_container: Rc<RefCell<TokenContainer>>,
    prefix_granularity: i32,
    state: State,
}

impl Request {
    /// Fused roles start in `Submitted`; PD roles start in `Bootstrapping`.
    pub fn new(spec: &RequestSpec, prefix_granularity: i32, role: Role) -> Self {
        let token_container = Rc::new(RefCell::new(TokenContainer::new(spec.tokens.clone())));
        let state = if role == Role::Fused {
            State::Submitted(Submitted {
                token_container: token_container.clone(),
                prefix_granularity,
            })
        } else {
            State::Bootstrapping(Bootstrapping {
                token_container: token_container.clone(),
                prefix_granularity,
            })
        };
        Self {
            id: spec.request_id.clone(),
            token_container,
            prefix_granularity,
            state,
        }
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    /// Apply a lifecycle event. `coordinator` is needed by events that release
    /// request tables.
    pub fn apply(&mut self, event: FsmEvent, coordinator: &mut CacheCoordinator) {
        let state = std::mem::take(&mut self.state);
        self.state = event.apply(state, coordinator);
    }

    /// Whether the request is in a specific state.
    pub fn is<Tag>(&self) -> bool
    where
        Tag: StateTag,
    {
        Tag::matches(&self.state)
    }

    /// Full prefix pages of `prefix_granularity` tokens (owned copies; the C++
    /// accessor returned spans into the container).
    pub fn full_prefix_pages(&self, except_last: bool) -> Vec<Vec<i32>> {
        self.token_container
            .borrow()
            .full_prefix_pages(self.prefix_granularity, except_last)
            .into_iter()
            .map(|page| page.to_vec())
            .collect()
    }

    pub fn token_size(&self) -> i32 {
        self.token_container.borrow().size()
    }

    pub fn last_token(&self) -> i32 {
        self.token_container.borrow().last_token()
    }

    pub fn prefill_size(&self) -> i32 {
        self.token_container.borrow().prefill_size()
    }

    pub fn current_prefill_info(&self) -> PrefillInfo {
        match &self.state {
            State::Prefilling(s) => prefill_info(&s.forward.token_container, s.window),
            State::PrefillDone(s) => prefill_info(&s.forward.token_container, s.window),
            other => panic!(
                "Request::CurrentPrefillInfo: expected Prefilling or PrefillDone; got {}",
                other.state_name()
            ),
        }
    }

    pub fn unscheduled_prefill_size(&self) -> i32 {
        match &self.state {
            State::Submitted(_) => -1,
            State::Prefilling(s) => self.prefill_size() - (s.window.begin + s.window.size),
            _ => 0,
        }
    }

    pub fn request_pool_index(&self) -> i32 {
        self.state
            .forward()
            .map_or(-1, |forward| forward.request_pool_index())
    }

    pub fn block_tables_ref(&self) -> &[BlockTable] {
        match self.state.forward() {
            Some(forward) => forward.block_tables(),
            None => &[],
        }
    }

    pub fn block_tables_ref_mut(&mut self) -> &mut Vec<BlockTable> {
        match self.state.forward_mut() {
            Some(forward) => forward.block_tables_mut(),
            None => panic!("Request::BlockTablesRef: expected a forward state"),
        }
    }

    pub fn cache_progress(&self) -> CacheProgress {
        match self.state.forward() {
            Some(forward) => forward.cache_progress().clone(),
            None => panic!("Request::CacheProgress: expected a forward state"),
        }
    }

    pub fn prefill_source(&self) -> PrefillSource {
        match &self.state {
            State::Prefilling(s) => s.source,
            other => panic!(
                "Request::PrefillSource: expected Prefilling; got {}",
                other.state_name()
            ),
        }
    }

    pub fn reserve_num_tokens_in_next_schedule_event(&self) -> i32 {
        match &self.state {
            State::PrefillDone(s) => s.reserve_num_tokens_in_next_schedule_event,
            State::Decoding(s) => s.reserve_num_tokens_in_next_schedule_event,
            other => panic!(
                "Request::ReserveNumTokensInNextScheduleEvent: expected PrefillDone or Decoding; got {}",
                other.state_name()
            ),
        }
    }

    pub fn state_name(&self) -> &'static str {
        self.state.state_name()
    }
}

impl Default for State {
    fn default() -> Self {
        State::Finished(Finished)
    }
}

fn prefill_info(token_container: &Rc<RefCell<TokenContainer>>, window: Window) -> PrefillInfo {
    let input_ids = token_container.borrow().token_slice(window).to_vec();
    let shifted_input_ids = crate::fsm::compute_shifted_input_ids(token_container, window);
    PrefillInfo {
        input_ids,
        shifted_input_ids,
        already_scheduled_len: window.begin,
        extend_len: window.size,
    }
}

/// Compile-time state tag used by `Request::is<Tag>()`.
pub trait StateTag {
    fn matches(state: &State) -> bool;
}

macro_rules! impl_state_tag {
    ($tag:ident, $variant:ident) => {
        impl StateTag for $tag {
            fn matches(state: &State) -> bool {
                matches!(state, State::$variant(_))
            }
        }
    };
}

impl_state_tag!(Bootstrapping, Bootstrapping);
impl_state_tag!(Submitted, Submitted);
impl_state_tag!(Prefilling, Prefilling);
impl_state_tag!(PrefillDone, PrefillDone);
impl_state_tag!(Decoding, Decoding);
impl_state_tag!(Retracted, Retracted);
impl_state_tag!(Finished, Finished);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block_pool::BlockPool;
    use crate::cache_coordinator::make_coordinator;
    use crate::cache_types::{AttnKind, CacheGroupSpec};
    use crate::fsm::{CacheProgress, ScheduleDecode, SchedulePrefillFirstChunk};
    use crate::req_pool_allocator::ReqPoolAllocator;
    use std::cell::RefCell;
    use std::rc::Rc;

    fn coord() -> CacheCoordinator {
        let spec = CacheGroupSpec {
            kind: AttnKind::Full,
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

    #[test]
    fn fused_request_starts_submitted() {
        let spec = RequestSpec {
            request_id: "r1".into(),
            tokens: vec![1, 2, 3, 4],
            max_new_tokens: 0,
        };
        let req = Request::new(&spec, 4, Role::Fused);
        assert!(req.is::<Submitted>());
        assert_eq!(req.state_name(), "Submitted");
        assert_eq!(req.prefill_size(), 4);
        assert_eq!(req.last_token(), 4);
        assert_eq!(req.unscheduled_prefill_size(), -1);
    }

    #[test]
    fn pd_request_starts_bootstrapping() {
        let spec = RequestSpec {
            request_id: "r1".into(),
            tokens: vec![1, 2, 3, 4],
            max_new_tokens: 0,
        };
        let req = Request::new(&spec, 4, Role::D);
        assert!(req.is::<Bootstrapping>());
    }

    #[test]
    fn schedule_and_decode_via_apply() {
        let spec = RequestSpec {
            request_id: "r1".into(),
            tokens: vec![1, 2, 3, 4],
            max_new_tokens: 0,
        };
        let mut req = Request::new(&spec, 4, Role::Fused);
        let mut coord = coord();
        let alloc = Rc::new(RefCell::new(ReqPoolAllocator::new(4)));
        let ev = FsmEvent::SchedulePrefillFirstChunk(SchedulePrefillFirstChunk {
            tokens_this_round: 4,
            reserve_num_tokens_in_next_schedule_event: 1,
            req_pool_allocator: alloc,
            source: PrefillSource::Local,
            block_tables: vec![BlockTable::default()],
            hit_tokens: 0,
            cache_progress: CacheProgress {
                prefix_hashes: Vec::new(),
                access_epoch: 1,
                promotion_boundary_tokens: 0,
            },
            load_pairs: Vec::new(),
        });
        req.apply(ev, &mut coord);
        assert!(req.is::<PrefillDone>());
        assert_eq!(req.request_pool_index(), 1);
        assert_eq!(req.current_prefill_info().extend_len, 4);

        let ev2 = FsmEvent::ScheduleDecode(ScheduleDecode {
            decode_input_tokens: 1,
            cache_progress: CacheProgress {
                prefix_hashes: Vec::new(),
                access_epoch: 1,
                promotion_boundary_tokens: 0,
            },
        });
        req.apply(ev2, &mut coord);
        assert!(req.is::<Decoding>());
        assert_eq!(req.reserve_num_tokens_in_next_schedule_event(), 1);
    }

    #[test]
    fn full_prefix_pages_returns_owned_copies() {
        let spec = RequestSpec {
            request_id: "r1".into(),
            tokens: vec![1, 2, 3, 4, 5, 6, 7, 8],
            max_new_tokens: 0,
        };
        let req = Request::new(&spec, 4, Role::Fused);
        let pages = req.full_prefix_pages(false);
        assert_eq!(pages, vec![vec![1, 2, 3, 4], vec![5, 6, 7, 8]]);
    }
}

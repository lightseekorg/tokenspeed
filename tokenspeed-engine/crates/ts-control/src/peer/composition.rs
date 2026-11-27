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

//! Join disjoint storage and versioned Peer reads under one scheduler admission.
use super::{layout::covers_interval, servicer::empty_event, PeerError, PeerFence, PeerRound};
use std::{
    collections::{HashMap, HashSet, VecDeque},
    sync::mpsc,
    time::Instant,
};
use ts_pd::layout::contract::Contract;
use ts_scheduler::{PeerEventKind, PeerEventView, PeerFetchView, PeerLeaseView};

/// One signal belongs to one plan, even when only one protocol has work in it.
pub(super) struct Fences {
    ready: mpsc::Receiver<Instant>,
    routes: VecDeque<Vec<PeerFence>>,
}
impl Fences {
    pub fn new() -> (Self, PeerFence) {
        let (tx, ready) = mpsc::channel();
        (
            Self {
                ready,
                routes: VecDeque::new(),
            },
            PeerFence(tx),
        )
    }
    pub fn plan(&mut self, routes: Vec<PeerFence>) {
        self.routes.push_back(routes);
    }
    pub fn advance(&mut self) {
        while !self.routes.is_empty() {
            let Ok(at) = self.ready.try_recv() else {
                break;
            };
            for fence in self.routes.pop_front().unwrap() {
                let _ = fence.0.send(at);
            }
        }
    }
}

#[derive(Clone, Copy)]
enum State {
    Waiting,
    Frozen,
    Retired(bool),
}
struct Part {
    begin: i32,
    end: i32,
    state: State,
}
struct Landing {
    request: String,
    begin: i32,
    end: i32,
    parts: HashMap<PeerLeaseView, Part>,
    frozen: bool,
}
#[derive(Default)]
pub(super) struct Landings {
    groups: HashMap<PeerLeaseView, Landing>,
    owners: HashMap<PeerLeaseView, PeerLeaseView>,
}
impl Landings {
    pub fn len(&self) -> usize {
        self.groups.len()
    }
    /// The L3 lease owns the scheduler's final target hold, including hot pages.
    pub fn register(
        &mut self,
        fetches: &[PeerFetchView],
        layout: &Contract,
    ) -> Result<HashSet<u64>, PeerError> {
        let mut requests = HashMap::<&str, Vec<&PeerFetchView>>::new();
        for fetch in fetches {
            requests.entry(&fetch.request_id).or_default().push(fetch);
        }
        let mut partials = HashSet::new();
        for (request, parts) in requests {
            let storage: Vec<_> = parts
                .iter()
                .filter(|f| f.holder.starts_with("l3@"))
                .collect();
            if storage.is_empty() {
                continue;
            }
            let fail = || PeerError::Plan(format!("invalid composed landing for {request}"));
            if storage.len() != 1 {
                return Err(fail());
            }
            let primary = storage[0];
            let destinations: Vec<_> = parts
                .iter()
                .flat_map(|f| f.destinations.iter().copied())
                .collect();
            if !covers_interval(
                layout,
                primary.begin_tokens,
                primary.end_tokens,
                &destinations,
            ) || parts.iter().any(|f| {
                f.begin_tokens != primary.begin_tokens
                    || f.end_tokens > primary.end_tokens
                    || f.lease.reader != primary.lease.reader
                    || self.owners.contains_key(&f.lease)
            }) {
                return Err(fail());
            }
            let mut states = HashMap::new();
            for part in &parts {
                if states
                    .insert(
                        part.lease.clone(),
                        Part {
                            begin: part.begin_tokens,
                            end: part.end_tokens,
                            state: State::Waiting,
                        },
                    )
                    .is_some()
                {
                    return Err(fail());
                }
                if !part.holder.starts_with("l3@") {
                    partials.insert(part.lease.id);
                }
            }
            for lease in states.keys() {
                self.owners.insert(lease.clone(), primary.lease.clone());
            }
            self.groups.insert(
                primary.lease.clone(),
                Landing {
                    request: request.into(),
                    begin: primary.begin_tokens,
                    end: primary.end_tokens,
                    parts: states,
                    frozen: false,
                },
            );
        }
        Ok(partials)
    }
    /// Unknown v1 retirement freezes the entire grant. Never publish a partial
    /// hit or retire the final target hold while either protocol can still write.
    pub fn advance(&mut self, round: &mut PeerRound) -> Result<(), PeerError> {
        let mut joined = Vec::new();
        for event in std::mem::take(&mut round.events) {
            if !matches!(
                event.kind,
                PeerEventKind::FetchDone
                    | PeerEventKind::FetchFrozen
                    | PeerEventKind::TargetRetired
            ) {
                round.events.push(event);
                continue;
            }
            let Some(primary) = self.owners.get(&event.lease).cloned() else {
                round.events.push(event);
                continue;
            };
            let group = self.groups.get_mut(&primary).unwrap();
            if event.request_id != group.request {
                return Err(PeerError::Plan(
                    "composed completion request mismatch".into(),
                ));
            }
            let part = group.parts.get_mut(&event.lease).unwrap();
            round.landed.retain(|(rid, begin, end)| {
                rid != &group.request || *begin != part.begin || *end != part.end
            });
            match event.kind {
                PeerEventKind::FetchFrozen => {
                    if matches!(part.state, State::Retired(_)) {
                        continue;
                    }
                    part.state = State::Frozen;
                    if !group.frozen {
                        group.frozen = true;
                        joined.push(PeerEventView {
                            kind: PeerEventKind::FetchFrozen,
                            request_id: group.request.clone(),
                            lease: primary.clone(),
                            ..empty_event()
                        });
                    }
                }
                PeerEventKind::TargetRetired => {
                    part.state = State::Retired(false);
                }
                PeerEventKind::FetchDone => {
                    if !matches!(part.state, State::Retired(_)) {
                        part.state = State::Retired(event.num_tokens == part.end);
                    }
                }
                _ => unreachable!(),
            }
            if group
                .parts
                .values()
                .all(|p| matches!(p.state, State::Retired(_)))
            {
                let group = self.groups.remove(&primary).unwrap();
                for lease in group.parts.keys() {
                    self.owners.remove(lease);
                }
                let ok = !group.frozen
                    && group
                        .parts
                        .values()
                        .all(|p| matches!(p.state, State::Retired(true)));
                if ok {
                    round
                        .landed
                        .push((group.request.clone(), group.begin, group.end));
                }
                joined.push(PeerEventView {
                    kind: if group.frozen {
                        PeerEventKind::TargetRetired
                    } else {
                        PeerEventKind::FetchDone
                    },
                    request_id: group.request,
                    lease: primary,
                    num_tokens: if ok { group.end } else { 0 },
                    ..empty_event()
                });
            }
        }
        round.events.extend(joined);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lease(id: u64) -> PeerLeaseView {
        PeerLeaseView {
            reader: "reader".into(),
            id,
        }
    }
    fn landing() -> Landings {
        let mut joined = Landings::default();
        joined.groups.insert(
            lease(2),
            Landing {
                request: "r".into(),
                begin: 0,
                end: 32,
                frozen: false,
                parts: [
                    (
                        lease(1),
                        Part {
                            begin: 0,
                            end: 16,
                            state: State::Waiting,
                        },
                    ),
                    (
                        lease(2),
                        Part {
                            begin: 0,
                            end: 32,
                            state: State::Waiting,
                        },
                    ),
                ]
                .into(),
            },
        );
        joined.owners = [(lease(1), lease(2)), (lease(2), lease(2))].into();
        joined
    }
    fn completion(id: u64, kind: PeerEventKind, tokens: i32) -> PeerEventView {
        PeerEventView {
            kind,
            request_id: "r".into(),
            lease: lease(id),
            num_tokens: tokens,
            ..empty_event()
        }
    }
    fn advance(joined: &mut Landings, events: Vec<PeerEventView>) -> PeerRound {
        let landed = events
            .iter()
            .filter(|e| e.kind == PeerEventKind::FetchDone && e.num_tokens > 0)
            .map(|e| (e.request_id.clone(), 0, e.num_tokens))
            .collect();
        let mut round = PeerRound {
            events,
            landed,
            ..Default::default()
        };
        joined.advance(&mut round).unwrap();
        round
    }
    #[test]
    fn completes_once_in_either_order_or_the_same_round() {
        for order in [[1, 2], [2, 1]] {
            for same_round in [false, true] {
                let mut joined = landing();
                let events: Vec<_> = order
                    .into_iter()
                    .map(|id| {
                        completion(id, PeerEventKind::FetchDone, if id == 1 { 16 } else { 32 })
                    })
                    .collect();
                let round = if same_round {
                    advance(&mut joined, events)
                } else {
                    let first = advance(&mut joined, vec![events[0].clone()]);
                    assert!(first.events.is_empty());
                    assert!(first.landed.is_empty());
                    assert_eq!(joined.len(), 1);
                    advance(&mut joined, vec![events[1].clone()])
                };
                assert_eq!(round.events.len(), 1);
                assert_eq!(round.events[0].lease, lease(2));
                assert_eq!(round.events[0].num_tokens, 32);
                assert_eq!(round.landed, vec![("r".into(), 0, 32)]);
                assert_eq!(joined.len(), 0);
                assert!(joined.owners.is_empty());
            }
        }
    }
    #[test]
    fn retired_failure_waits_for_the_other_writer() {
        let mut joined = landing();
        assert!(advance(
            &mut joined,
            vec![completion(2, PeerEventKind::FetchDone, 0)]
        )
        .events
        .is_empty());
        let final_round = advance(
            &mut joined,
            vec![completion(1, PeerEventKind::FetchDone, 16)],
        );
        assert_eq!(final_round.events.len(), 1);
        assert_eq!(final_round.events[0].num_tokens, 0);
        assert!(final_round.landed.is_empty());
        assert_eq!(joined.len(), 0);
    }
    #[test]
    fn freeze_holds_the_whole_destination_until_both_retire() {
        for hot_retires_first in [true, false] {
            let mut joined = landing();
            let frozen = advance(
                &mut joined,
                vec![completion(1, PeerEventKind::FetchFrozen, 0)],
            );
            assert_eq!(frozen.events[0].kind, PeerEventKind::FetchFrozen);
            assert_eq!(frozen.events[0].lease, lease(2));
            assert!(advance(
                &mut joined,
                vec![completion(1, PeerEventKind::FetchFrozen, 0)]
            )
            .events
            .is_empty());
            let hot = completion(1, PeerEventKind::TargetRetired, 0);
            let storage = completion(2, PeerEventKind::FetchDone, 32);
            let order = if hot_retires_first {
                [hot, storage]
            } else {
                [storage, hot]
            };
            assert!(advance(&mut joined, vec![order[0].clone()])
                .events
                .is_empty());
            assert_eq!(joined.len(), 1);
            let final_round = advance(&mut joined, vec![order[1].clone()]);
            assert_eq!(final_round.events.len(), 1);
            assert_eq!(final_round.events[0].kind, PeerEventKind::TargetRetired);
            assert_eq!(final_round.events[0].lease, lease(2));
            assert!(final_round.landed.is_empty());
            assert_eq!(joined.len(), 0);
        }
    }
    #[test]
    fn plan_fences_do_not_leak_to_the_other_protocol() {
        let (mut routes, public) = Fences::new();
        let (tx1, rx1) = mpsc::channel();
        let (tx2, rx2) = mpsc::channel();
        let hot = PeerFence(tx1);
        let storage = PeerFence(tx2);
        routes.plan(vec![hot.clone()]);
        routes.plan(vec![storage.clone()]);
        routes.plan(vec![hot, storage]);
        public.ready();
        routes.advance();
        assert!(rx1.try_recv().is_ok());
        assert!(rx2.try_recv().is_err());
        public.ready();
        routes.advance();
        assert!(rx1.try_recv().is_err());
        assert!(rx2.try_recv().is_ok());
        routes.advance();
        assert!(rx1.try_recv().is_err());
        assert!(rx2.try_recv().is_err());
        public.ready();
        routes.advance();
        assert!(rx1.try_recv().is_ok());
        assert!(rx2.try_recv().is_ok());
    }
}

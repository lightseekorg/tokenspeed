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

//! The rank's pool client over real ZMQ: against an owner that answers what
//! each test scripts, and once against the Python owner service
//! (`tests/fixtures/shared_l2_owner.py`). Test memory stands in for the
//! rank's Device cache and the provider's bytes, and a copy within it for
//! RDMA.

use std::{
    io::{BufRead, BufReader, Write as _},
    path::{Path, PathBuf},
    process::{Child, ChildStdin, ChildStdout, Command, Stdio},
    sync::atomic::{AtomicBool, Ordering},
};

use parking_lot::Condvar;
use ts_scheduler::SharedL2CheckpointView;
use ts_wire::Value;

use super::*;
use crate::shared_l2::{
    geometry::{PageField, PageGroup, Provider},
    load::LoadRow,
    testing::{
        hit_value, key_value, lookup, miss_value,
        owner::{field, ints, ok, op, refused, uint, TestOwner},
        prompt_keys, wait_until, PlainSignals, PROMPT,
    },
};

const KV: u32 = 0;
const STATE: u32 = 1;
const PROVIDER: u64 = 10_000;
const STATE_DEVICE: u64 = 2_000;
const TIMEOUT: Duration = Duration::from_secs(5);
/// No ping within a test that does not wait for one.
const NO_KEEPALIVE: Duration = Duration::from_secs(600);

/// The pool of the Python service tests (`shared_l2_test_utils.GEOMETRY`):
/// LCM blocks of 64 bytes, 8 of them on `node0`; two KV pages fit in one, a
/// State page fills one. A Device KV block is 32 bytes (layer 0), a State
/// block 64 (layer 1).
fn geometry() -> PoolGeometry {
    PoolGeometry {
        lcm_block_bytes: 64,
        providers: vec![Provider {
            session: "node0".to_owned(),
            base_address: PROVIDER,
            num_lcm_blocks: 8,
            first_lcm_block: 0,
        }],
        groups: vec![
            PageGroup {
                packing: 2,
                page_bytes: 32,
                fields: vec![PageField {
                    device_address: 0,
                    device_stride: 32,
                    bytes: 32,
                    offset_in_page: 0,
                    layer: 0,
                }],
            },
            PageGroup {
                packing: 1,
                page_bytes: 64,
                fields: vec![PageField {
                    device_address: STATE_DEVICE,
                    device_stride: 64,
                    bytes: 64,
                    offset_in_page: 0,
                    layer: 1,
                }],
            },
        ],
        num_layers: 2,
    }
}

/// Where a Device block is, and its size.
fn device(group: u32, block: i32) -> (u64, usize) {
    if group == KV {
        (block as u64 * 32, 32)
    } else {
        (STATE_DEVICE + block as u64 * 64, 64)
    }
}

/// Where a pool page is.
fn pool(group: u32, page: i32) -> u64 {
    geometry().page_address(group, 0, page).unwrap().1
}

/// Test memory behind a transfer that records its writes and can fail its
/// copies or hold its writes.
struct Transfer {
    arena: Arena,
    writes: Mutex<Vec<(String, Spans)>>,
    failing: AtomicBool,
    holding: Mutex<bool>,
    resumed: Condvar,
    held: AtomicBool,
}

impl PageWriter for Transfer {
    fn write(&self, session: &str, spans: &Spans) -> Result<(), String> {
        let mut holding = self.holding.lock();
        while *holding {
            self.held.store(true, Ordering::Relaxed);
            self.resumed.wait(&mut holding);
        }
        drop(holding);
        if self.failing.load(Ordering::Relaxed) {
            return Err("down".to_owned());
        }
        self.writes.lock().push((session.to_owned(), spans.clone()));
        PageWriter::write(&self.arena, session, spans)
    }
}

impl PageReader for Transfer {
    fn read(&self, session: &str, spans: &Spans) -> Result<(), String> {
        if self.failing.load(Ordering::Relaxed) {
            return Err("down".to_owned());
        }
        self.arena.read(session, spans)
    }
}

/// A rank that joined the pool of the owner at `address` as `client_id`.
struct Rank {
    client: RankClient,
    threads: RankThreads,
    transfer: Arc<Transfer>,
    signal_words: Arc<PlainSignals>,
}

impl Rank {
    fn join(address: &str, client_id: u64, request_timeout: Duration, keepalive: Duration) -> Self {
        let transfer = Arc::new(Transfer {
            arena: Arena::new(12_000),
            writes: Mutex::default(),
            failing: AtomicBool::new(false),
            holding: Mutex::new(false),
            resumed: Condvar::new(),
            held: AtomicBool::new(false),
        });
        // Every Device block holds bytes of its own.
        for (group, blocks) in [(KV, 16), (STATE, 8)] {
            for block in 1..blocks {
                let (address, bytes) = device(group, block);
                let seed = 40 * group as u8 + 3 * block as u8;
                let payload: Vec<u8> = (0..bytes as u8).map(|byte| seed + byte).collect();
                transfer.arena.write(address, &payload);
            }
        }
        // Two sets of a start word and a ready word per layer.
        let signal_words = Arc::new(PlainSignals::new(6));
        let (client, threads) = start(
            ClientSettings {
                owner_address: address.to_owned(),
                client_id,
                request_timeout,
                keepalive,
                read_workers: NonZeroUsize::new(2).unwrap(),
                write_workers: NonZeroUsize::new(2).unwrap(),
                geometry: geometry(),
            },
            transfer.clone(),
            transfer.clone(),
            signal_words.clone(),
        )
        .unwrap();
        Self {
            client,
            threads,
            transfer,
            signal_words,
        }
    }

    fn of(owner: &TestOwner) -> Self {
        Self::join(&owner.address, 7, TIMEOUT, NO_KEEPALIVE)
    }

    /// The writes finished, once there are `count` of them.
    fn finished(&mut self, count: usize) -> Vec<u32> {
        let mut finished = Vec::new();
        wait_until(|| {
            finished.extend(self.client.poll_writes().unwrap());
            finished.len() >= count
        });
        finished.sort();
        finished
    }

    /// What failed, once the client's polls fail.
    fn failure(&mut self) -> String {
        let mut failure = None;
        wait_until(|| {
            failure = self.client.poll_writes().err();
            failure.is_some()
        });
        failure.unwrap().0
    }

    fn block(&self, group: u32, block: i32) -> Vec<u8> {
        let (address, bytes) = device(group, block);
        self.transfer.arena.read_bytes(address, bytes)
    }

    fn page(&self, group: u32, page: i32) -> Vec<u8> {
        self.transfer
            .arena
            .read_bytes(pool(group, page), device(group, 0).1)
    }
}

/// The write that stores `prompt` from Device blocks 1, 2, ... of each
/// group, with the checkpoint at its end.
fn write(op_id: u32, prompt: &[i32]) -> SharedL2OpView {
    let (prefix_hashes, tail, keys) = prompt_keys(prompt);
    let (tail_tokens, tail_hash) = tail.unwrap_or_default();
    let pages = keys.len() as i32 - 1;
    SharedL2OpView {
        kind: SharedL2OpKind::Write,
        op_id,
        ticket: 0,
        buckets: vec![0; keys.len()],
        // KV blocks 1.. and State block 1.
        device_blocks: (1..=pages).chain([1]).collect(),
        keys,
        checkpoints: vec![SharedL2CheckpointView {
            tokens: prompt.to_vec(),
            prefix_hashes,
            tail_tokens,
            tail_hash,
            access_epoch: 7,
        }],
    }
}

fn notice(kind: SharedL2OpKind, ticket: u64, keys: Vec<StorageKeyView>) -> SharedL2OpView {
    SharedL2OpView {
        kind,
        op_id: 0,
        ticket,
        keys,
        buckets: Vec::new(),
        device_blocks: Vec::new(),
        checkpoints: Vec::new(),
    }
}

fn hits(hits: Vec<Value>) -> Option<Value> {
    Some(ok(vec![("hits", Value::Array(hits))]))
}

#[test]
fn one_request_looks_up_every_prompt_and_answers_them_in_order() {
    let owner = TestOwner::serve(|_| {
        hits(vec![
            hit_value(41, [3, 4, 8]),
            miss_value(),
            hit_value(42, [3, 4, 8]),
        ])
    });
    let mut rank = Rank::of(&owner);

    let answers = rank
        .client
        .lookup(&[
            lookup("hit", &[1, 2, 3, 4, 5, 6, 7]),
            lookup("cold", &[9, 9, 9, 9]),
            lookup("same", &PROMPT),
        ])
        .unwrap();

    let (_, _, keys) = prompt_keys(&PROMPT);
    assert_eq!(
        answers[0],
        SharedL2AnswerView {
            request_id: "hit".to_owned(),
            ticket: 41,
            num_tokens: 5,
            kv_tokens: 5,
            tail_tokens: 1,
            tail_hash: "1-2-3-4-5".to_owned(),
            keys,
            buckets: vec![0, 0, 0],
            pool_pages: vec![3, 4, 8],
        }
    );
    // A prompt the pool holds no prefix of is answered like the others: a
    // miss.
    assert_eq!(
        answers[1],
        SharedL2AnswerView {
            request_id: "cold".to_owned(),
            ticket: 0,
            num_tokens: 0,
            kv_tokens: 0,
            tail_tokens: 0,
            tail_hash: String::new(),
            keys: Vec::new(),
            buckets: Vec::new(),
            pool_pages: Vec::new(),
        }
    );
    assert_eq!(
        (answers[2].request_id.as_str(), answers[2].ticket),
        ("same", 42)
    );
    // One request asked for all of them.
    assert_eq!(owner.ops(), ["lookup"]);
    assert_eq!(rank.client.poll_writes().unwrap(), Vec::<u32>::new());
}

#[test]
fn a_lookup_the_owner_does_not_answer_in_time_fails() {
    let owner = TestOwner::serve(|_| None);
    let mut rank = Rank::join(&owner.address, 7, Duration::from_millis(100), NO_KEEPALIVE);

    let started = Instant::now();
    let error = rank
        .client
        .lookup(&[lookup("request", &PROMPT)])
        .unwrap_err();
    assert!(error.0.contains("did not answer 'lookup'"), "{error}");
    assert!(started.elapsed() >= Duration::from_millis(100));
    // It was asked once.
    assert_eq!(owner.ops(), ["lookup"]);
}

#[test]
fn a_lookup_the_owner_answers_with_an_error_fails() {
    // The pool failed; the owner does not know the rank.
    for reason in [
        "RuntimeError: lookup failed",
        "client 7 has not joined the pool",
    ] {
        let owner = TestOwner::serve(move |_| Some(refused(reason)));
        let error = Rank::of(&owner)
            .client
            .lookup(&[lookup("request", &PROMPT)])
            .unwrap_err();
        assert!(error.0.contains("rejected 'lookup'"), "{error}");
        assert!(error.0.contains(reason), "{error}");
    }
}

#[test]
fn a_hit_names_pages_the_pool_has() {
    // The pool's 8 LCM blocks end at KV page 16 and State page 8.
    let owner = TestOwner::serve(|_| hits(vec![hit_value(41, [16, 16, 8])]));
    let answers = Rank::of(&owner)
        .client
        .lookup(&[lookup("request", &PROMPT)])
        .unwrap();
    assert_eq!(answers[0].pool_pages, [16, 16, 8]);

    for pages in [[0, 4, 8], [-3, 4, 8], [17, 4, 8], [3, 4, 9], [3, 4, 0]] {
        let owner =
            TestOwner::serve(move |_| hits(vec![hit_value(41, [3, 4, 8]), hit_value(42, pages)]));
        let error = Rank::of(&owner)
            .client
            .lookup(&[lookup("hit", &PROMPT), lookup("spoiled", &PROMPT)])
            .unwrap_err();
        assert!(error.0.contains("is out of range"), "{pages:?}: {error}");
    }

    // A group the pool does not have, and a shard it does not have.
    let (_, _, keys) = prompt_keys(&PROMPT);
    for (group_id, bucket, message) in [
        (2, 0, "the shared L2 pool has no cache group 2"),
        (KV, 1, "a shared L2 pool holds bucket 0 only"),
    ] {
        let Value::Array(mut hit) = hit_value(41, [3, 4, 8]) else {
            unreachable!()
        };
        hit[4] = Value::Array(vec![Value::Array(vec![
            key_value(&StorageKeyView {
                group_id,
                ..keys[0].clone()
            }),
            Value::from(bucket),
            Value::from(3),
        ])]);
        let owner = TestOwner::serve(move |_| hits(vec![Value::Array(hit.clone())]));
        let error = Rank::of(&owner)
            .client
            .lookup(&[lookup("request", &PROMPT)])
            .unwrap_err();
        assert_eq!(error.0, message);
    }
}

#[test]
fn a_write_copies_only_the_blocks_the_owner_placed() {
    let mut reservations = 0;
    let owner = TestOwner::serve(move |request| {
        Some(match op(request) {
            "reserve" => {
                reservations += 1;
                if reservations == 1 {
                    // The second KV block is in the pool already.
                    ok(vec![
                        ("ticket", Value::from(12)),
                        ("pages", ints(&[3, -1, 8])),
                    ])
                } else {
                    // Every block is: nothing to write.
                    ok(vec![
                        ("ticket", Value::from(0)),
                        ("pages", ints(&[-1, -1, -1])),
                    ])
                }
            }
            _ => ok(Vec::new()),
        })
    });
    let mut rank = Rank::of(&owner);

    rank.client.submit(&[write(1, &PROMPT)]).unwrap();
    assert_eq!(rank.finished(1), [1]);

    // KV block 1 and State block 1 are on their pages, in one copy to the
    // provider; KV block 2 was not copied.
    assert_eq!(rank.page(KV, 3), rank.block(KV, 1));
    assert_eq!(rank.page(STATE, 8), rank.block(STATE, 1));
    {
        let writes = rank.transfer.writes.lock();
        let (session, spans) = &writes[0];
        assert_eq!((writes.len(), session.as_str()), (1, "node0"));
        assert_eq!(spans.sources, [device(KV, 1).0, device(STATE, 1).0]);
        assert_eq!(spans.destinations, [pool(KV, 3), pool(STATE, 8)]);
    }
    let requests = owner.requests();
    assert_eq!(owner.ops(), ["reserve", "commit"]);
    let (_, _, keys) = prompt_keys(&PROMPT);
    assert_eq!(
        *field(&requests[0], "keys"),
        Value::Array(keys.iter().map(key_value).collect())
    );
    assert_eq!(uint(field(&requests[1], "ticket")), 12);
    let Value::Array(checkpoints) = field(&requests[1], "checkpoints") else {
        panic!("checkpoints are an array");
    };
    assert_eq!(checkpoints.len(), 1);

    // The pool holds every block of the same prompt: no copy, no commit.
    rank.client.submit(&[write(2, &PROMPT)]).unwrap();
    assert_eq!(rank.finished(1), [2]);
    assert_eq!(rank.transfer.writes.lock().len(), 1);
    assert_eq!(owner.ops(), ["reserve", "commit", "reserve"]);
}

#[test]
fn a_write_that_fails_at_any_step_fails_the_polls_from_then_on() {
    let placed = || {
        ok(vec![
            ("ticket", Value::from(12)),
            ("pages", ints(&[3, 4, 8])),
        ])
    };
    let cases: Vec<(Option<Value>, Option<Value>, bool, &str)> = vec![
        (
            Some(refused("RuntimeError: reserve failed")),
            None,
            false,
            "reserve failed",
        ),
        (None, None, true, "shared L2 write to node0 failed: down"),
        (
            None,
            Some(refused("RuntimeError: commit failed")),
            false,
            "commit failed",
        ),
        // Not a page for every key.
        (
            Some(ok(vec![
                ("ticket", Value::from(12)),
                ("pages", ints(&[3, 4])),
            ])),
            None,
            false,
            "a reservation's page list has 2 elements, not 3",
        ),
        // Page 0: pages start at 1, and a key without one is negative.
        (
            Some(ok(vec![
                ("ticket", Value::from(12)),
                ("pages", ints(&[3, 0, 8])),
            ])),
            None,
            false,
            "shared L2 page 0 of group 0 is out of range",
        ),
    ];
    for (reserve, commit, failing_copy, message) in cases {
        let owner = TestOwner::serve(move |request| {
            Some(match op(request) {
                "reserve" => reserve.clone().unwrap_or_else(placed),
                "commit" => commit.clone().unwrap_or_else(|| ok(Vec::new())),
                _ => ok(Vec::new()),
            })
        });
        let mut rank = Rank::of(&owner);
        rank.transfer.failing.store(failing_copy, Ordering::Relaxed);

        rank.client.submit(&[write(5, &PROMPT)]).unwrap();

        let failure = rank.failure();
        assert!(failure.contains(message), "{message}: {failure}");
        // The write is not reported as finished, and the failure stays, for
        // the poll at the round's head and the check at a forward's commit.
        assert_eq!(rank.client.poll_writes().unwrap_err().0, failure);
        assert_eq!(rank.client.check_failure().unwrap_err().0, failure);
        assert_eq!(rank.threads.loads().poll().unwrap_err().0, failure);
    }
}

#[test]
fn a_notice_the_owner_does_not_answer_fails_the_polls_and_is_sent_once() {
    let owner = TestOwner::serve(|request| (op(request) != "release").then(|| ok(Vec::new())));
    let mut rank = Rank::join(&owner.address, 7, Duration::from_millis(100), NO_KEEPALIVE);
    rank.client.check_failure().unwrap();

    rank.client
        .submit(&[notice(SharedL2OpKind::Release, 41, Vec::new())])
        .unwrap();

    let failure = rank.failure();
    assert!(failure.contains("did not answer 'release'"), "{failure}");
    // Well past the request timeout: the release was not sent again.
    thread::sleep(Duration::from_millis(300));
    assert_eq!(owner.ops(), ["release"]);
    assert_eq!(uint(field(&owner.requests()[0], "ticket")), 41);
}

#[test]
fn an_idle_rank_pings_the_owner() {
    let owner = TestOwner::serve(|_| Some(ok(Vec::new())));
    let mut rank = Rank::join(&owner.address, 7, TIMEOUT, Duration::from_millis(50));

    // Neither a lookup, nor a write, nor a notice.
    wait_until(|| owner.ops().len() >= 4);

    assert!(owner.ops().iter().all(|op| op == "ping"));
    assert_eq!(uint(field(&owner.requests()[0], "client")), 7);
    assert_eq!(rank.client.poll_writes().unwrap(), Vec::<u32>::new());
    // A notice goes out between the pings.
    rank.client
        .submit(&[notice(SharedL2OpKind::Release, 41, Vec::new())])
        .unwrap();
    wait_until(|| owner.ops().iter().any(|op| op == "release"));
    rank.client.check_failure().unwrap();
}

#[test]
fn a_plans_ops_reach_the_owner_each_kind_in_plan_order() {
    let owner = TestOwner::serve(|request| {
        Some(match op(request) {
            // Nothing to copy: no key gets a page.
            "reserve" => {
                let Value::Array(keys) = field(request, "keys") else {
                    panic!("keys are an array");
                };
                ok(vec![
                    ("ticket", Value::from(0)),
                    ("pages", ints(&vec![-1; keys.len()])),
                ])
            }
            _ => ok(Vec::new()),
        })
    });
    let mut rank = Rank::of(&owner);
    let (_, _, keys) = prompt_keys(&PROMPT);

    rank.client
        .submit(&[
            notice(SharedL2OpKind::Touch, 0, keys[..2].to_vec()),
            write(31, &PROMPT),
            notice(SharedL2OpKind::Release, 41, Vec::new()),
            write(32, &[1, 2, 3, 4, 5, 6]),
            notice(SharedL2OpKind::Release, 42, Vec::new()),
            notice(SharedL2OpKind::Touch, 0, keys[2..].to_vec()),
        ])
        .unwrap();

    // Each write is queued as a write of its own.
    assert_eq!(rank.finished(2), [31, 32]);
    // The ops of one kind reach the owner in plan order.
    wait_until(|| owner.ops().len() == 6);
    let of = |kind: &str| -> Vec<Value> {
        owner
            .requests()
            .into_iter()
            .filter(|request| op(request) == kind)
            .collect()
    };
    let releases: Vec<u64> = of("release")
        .iter()
        .map(|release| uint(field(release, "ticket")))
        .collect();
    assert_eq!(releases, [41, 42]);
    let touches = of("touch");
    assert_eq!(
        *field(&touches[0], "keys"),
        Value::Array(keys[..2].iter().map(key_value).collect())
    );
    assert_eq!(
        *field(&touches[1], "keys"),
        Value::Array(keys[2..].iter().map(key_value).collect())
    );

    // An op of no known kind is refused.
    let error = rank
        .client
        .submit(&[notice(SharedL2OpKind { repr: 9 }, 0, Vec::new())])
        .unwrap_err();
    assert!(
        error.0.contains("no client call for the shared L2 op"),
        "{error}"
    );
}

#[test]
fn a_client_does_not_start_from_settings_its_copies_cannot_use() {
    // A group with no page in an LCM block; a field of a layer past the
    // rank's.
    let mut unpacked = geometry();
    unpacked.groups[STATE as usize].packing = 0;
    let mut layerless = geometry();
    layerless.num_layers = 1;
    for (geometry, message) in [
        (
            unpacked,
            "shared L2 cache group 1 has no page in an LCM block",
        ),
        (
            layerless,
            "a field of shared L2 cache group 1 is read by layer 1, and the rank has 1 layers",
        ),
    ] {
        let memory = Arc::new(Arena::new(0));
        // Refused before any connection: nothing listens at the address.
        let error = start(
            ClientSettings {
                owner_address: "tcp://127.0.0.1:1".to_owned(),
                client_id: 7,
                request_timeout: TIMEOUT,
                keepalive: NO_KEEPALIVE,
                read_workers: NonZeroUsize::new(2).unwrap(),
                write_workers: NonZeroUsize::new(2).unwrap(),
                geometry,
            },
            memory.clone(),
            memory,
            Arc::new(PlainSignals::new(6)),
        )
        .err()
        .unwrap();
        assert_eq!(error.0, message);
    }
}

#[test]
fn closing_ends_the_threads_once_what_was_queued_ran() {
    let owner = TestOwner::serve(|request| {
        Some(match op(request) {
            "reserve" => ok(vec![
                ("ticket", Value::from(12)),
                ("pages", ints(&[3, 4, 8])),
            ]),
            _ => ok(Vec::new()),
        })
    });
    let mut rank = Rank::of(&owner);
    rank.client
        .submit(&[
            write(5, &PROMPT),
            notice(SharedL2OpKind::Release, 41, Vec::new()),
        ])
        .unwrap();

    assert!(rank.threads.close(TIMEOUT));

    // The write and the notice queued before the close went out.
    assert_eq!(rank.client.poll_writes().unwrap(), [5]);
    let mut ops = owner.ops();
    ops.sort();
    assert_eq!(ops, ["commit", "release", "reserve"]);
    // Nothing runs any more: an op and a load are refused.
    let error = rank.client.submit(&[write(6, &PROMPT)]).unwrap_err();
    assert!(error.0.contains("threads have stopped"), "{error}");
    let error = rank.threads.loads().submit(vec![9], &[], 0, 1).unwrap_err();
    assert!(error.0.contains("loads have stopped"), "{error}");
    // A second close finds nothing to stop.
    assert!(rank.threads.close(TIMEOUT));
}

#[test]
fn closing_leaves_what_a_thread_that_still_runs_uses() {
    let owner = TestOwner::serve(|request| {
        Some(match op(request) {
            "reserve" => ok(vec![
                ("ticket", Value::from(12)),
                ("pages", ints(&[3, 4, 8])),
            ]),
            _ => ok(Vec::new()),
        })
    });
    let mut rank = Rank::of(&owner);
    *rank.transfer.holding.lock() = true;
    rank.client.submit(&[write(5, &PROMPT)]).unwrap();
    wait_until(|| rank.transfer.held.load(Ordering::Relaxed));

    // The copy outlasts the close, which waits no longer than it was told.
    let started = Instant::now();
    assert!(!rank.threads.close(Duration::from_millis(200)));
    assert!(started.elapsed() < Duration::from_secs(5));

    // The memory it writes stays usable, and the socket its commit goes out
    // on stays open.
    *rank.transfer.holding.lock() = false;
    rank.transfer.resumed.notify_all();
    assert_eq!(rank.finished(1), [5]);
    assert_eq!(rank.page(KV, 3), rank.block(KV, 1));
    assert_eq!(owner.ops(), ["reserve", "commit"]);
}

/// Test memory behind a transfer that records, when it is dropped, on
/// which thread and after which of its writes that was.
struct Recorded {
    arena: Arena,
    holding: Mutex<bool>,
    resumed: Condvar,
    written: Arc<Mutex<Vec<Instant>>>,
    dropped: Arc<Mutex<Vec<(thread::ThreadId, Instant)>>>,
}

impl PageWriter for Recorded {
    fn write(&self, session: &str, spans: &Spans) -> Result<(), String> {
        let mut holding = self.holding.lock();
        while *holding {
            self.resumed.wait(&mut holding);
        }
        drop(holding);
        // The copy takes a while: the close waits for it.
        thread::sleep(Duration::from_millis(50));
        PageWriter::write(&self.arena, session, spans)?;
        self.written.lock().push(Instant::now());
        Ok(())
    }
}

impl PageReader for Recorded {
    fn read(&self, session: &str, spans: &Spans) -> Result<(), String> {
        self.arena.read(session, spans)
    }
}

impl Drop for Recorded {
    fn drop(&mut self) {
        self.dropped
            .lock()
            .push((thread::current().id(), Instant::now()));
    }
}

#[test]
fn closing_releases_the_transfer_last_and_on_the_closing_thread() {
    let owner = TestOwner::serve(|request| {
        Some(match op(request) {
            "reserve" => ok(vec![
                ("ticket", Value::from(12)),
                ("pages", ints(&[3, 4, 8])),
            ]),
            _ => ok(Vec::new()),
        })
    });
    let (written, dropped) = (Arc::default(), Arc::default());
    let transfer = Arc::new(Recorded {
        arena: Arena::new(12_000),
        holding: Mutex::new(true),
        resumed: Condvar::new(),
        written: Arc::clone(&written),
        dropped: Arc::clone(&dropped),
    });
    let (mut client, threads) = start(
        ClientSettings {
            owner_address: owner.address.clone(),
            client_id: 7,
            request_timeout: TIMEOUT,
            keepalive: NO_KEEPALIVE,
            read_workers: NonZeroUsize::new(2).unwrap(),
            write_workers: NonZeroUsize::new(2).unwrap(),
            geometry: geometry(),
        },
        transfer.clone(),
        transfer.clone(),
        Arc::new(PlainSignals::new(6)),
    )
    .unwrap();
    // The client's threads and the close hold the transfer from here.
    let resume = Arc::downgrade(&transfer);
    drop(transfer);
    client.submit(&[write(5, &PROMPT)]).unwrap();

    // A write thread is still in its copy when the wait is over: nothing is
    // released, the transfer stays for the thread and for the process's
    // exit.
    assert!(!threads.close(Duration::from_millis(100)));
    assert!(dropped.lock().is_empty());

    // Once the thread has ended, a close releases the transfer itself:
    // after the copy, and on the thread that closes, not on a worker.
    let held = resume.upgrade().unwrap();
    *held.holding.lock() = false;
    held.resumed.notify_all();
    drop(held);
    assert!(threads.close(TIMEOUT));
    assert_eq!(client.poll_writes().unwrap(), [5]);
    let (written, dropped) = (written.lock(), dropped.lock());
    assert_eq!(dropped.len(), 1);
    assert_eq!(dropped[0].0, thread::current().id());
    assert!(dropped[0].1 >= written[0]);
}

#[test]
fn a_failed_load_fails_the_clients_check_at_a_forwards_commit() {
    let owner = TestOwner::serve(|_| Some(ok(Vec::new())));
    let mut rank = Rank::of(&owner);
    rank.client.map_proxy_blocks(&[SharedL2ProxyBlockView {
        group_id: KV,
        host_block: 100,
        bucket: 0,
        pool_page: 3,
    }]);
    rank.client.check_failure().unwrap();
    rank.transfer.failing.store(true, Ordering::Relaxed);

    rank.signal_words.words[0].store(1, Ordering::Relaxed);
    rank.threads
        .loads()
        .submit(
            vec![9],
            &[LoadRow {
                group_id: KV,
                host_block: 100,
                device_block: 5,
            }],
            0,
            1,
        )
        .unwrap();

    // The forward is let go with the failure already there for the loop.
    wait_until(|| rank.signal_words.get(1) == 1);
    let failure = rank.client.check_failure().unwrap_err().0;
    assert!(
        failure.contains("shared L2 read from node0 failed"),
        "{failure}"
    );
    assert_eq!(rank.client.poll_writes().unwrap_err().0, failure);
}

/// The Python owner service, run as a child.
struct PythonOwner {
    child: Child,
    commands: ChildStdin,
    answers: BufReader<ChildStdout>,
    errors: PathBuf,
    address: String,
}

impl PythonOwner {
    /// Start the service; `None`, after saying so, without a Python that
    /// has the owner's modules: the one `TS_TEST_PYTHON` or `PYO3_PYTHON`
    /// names, else `python3`.
    fn start(settings: serde_json::Value) -> Option<Self> {
        let python = std::env::var("TS_TEST_PYTHON")
            .or_else(|_| std::env::var("PYO3_PYTHON"))
            .unwrap_or_else(|_| "python3".to_owned());
        let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
        let repository = manifest.join("../../..");
        let mut modules = vec![
            repository.join("python"),
            repository.join("tokenspeed-kernel/python"),
        ];
        modules.extend(std::env::split_paths(
            &std::env::var_os("PYTHONPATH").unwrap_or_default(),
        ));
        let errors = std::env::temp_dir().join(format!(
            "ts-shared-l2-owner-{}-{:?}.stderr",
            std::process::id(),
            thread::current().id()
        ));
        let mut child = Command::new(&python)
            .arg(manifest.join("tests/fixtures/shared_l2_owner.py"))
            .arg(settings.to_string())
            .env("PYTHONPATH", std::env::join_paths(modules).unwrap())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(std::fs::File::create(&errors).unwrap())
            .spawn()
            .expect("running Python");
        let mut owner = Self {
            commands: child.stdin.take().unwrap(),
            answers: BufReader::new(child.stdout.take().unwrap()),
            child,
            errors,
            address: String::new(),
        };
        let mut ready = String::new();
        owner.answers.read_line(&mut ready).unwrap();
        if ready.is_empty() {
            let _ = owner.child.wait();
            let stderr = std::fs::read_to_string(&owner.errors).unwrap_or_default();
            if stderr.contains("No module named") {
                eprintln!("skipped: {python} cannot import the shared L2 owner");
                return None;
            }
            panic!("the Python owner did not start:\n{stderr}");
        }
        let ready: serde_json::Value = serde_json::from_str(&ready).unwrap();
        owner.address = ready["address"].as_str().unwrap().to_owned();
        Some(owner)
    }

    /// Run one command of the fixture and return its answer.
    fn run(&mut self, command: serde_json::Value) -> serde_json::Value {
        writeln!(self.commands, "{command}").unwrap();
        let mut answer = String::new();
        self.answers.read_line(&mut answer).unwrap();
        assert!(
            !answer.is_empty(),
            "the Python owner ended:\n{}",
            std::fs::read_to_string(&self.errors).unwrap_or_default()
        );
        serde_json::from_str(&answer).unwrap()
    }

    fn books(&mut self) -> serde_json::Value {
        self.run(serde_json::json!({"do": "books"}))
    }
}

impl Drop for PythonOwner {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        let _ = std::fs::remove_file(&self.errors);
    }
}

#[test]
fn the_client_speaks_the_python_owners_wire() {
    // The owner drops a client it has not heard for half a second: the
    // rank's pings keep it there.
    let Some(mut owner) = PythonOwner::start(serde_json::json!({
        "client_timeout_s": 0.5,
        "base": PROVIDER,
        "blocks": 8,
        "clients": [7],
    })) else {
        return;
    };
    let address = owner.address.clone();
    let mut rank = Rank::join(
        &address,
        7,
        Duration::from_secs(2),
        Duration::from_millis(100),
    );
    let (_, tail, keys) = prompt_keys(&PROMPT);

    // An empty pool answers a miss.
    let answers = rank.client.lookup(&[lookup("cold", &PROMPT)]).unwrap();
    assert_eq!(
        (answers[0].request_id.as_str(), answers[0].ticket),
        ("cold", 0)
    );
    assert!(answers[0].keys.is_empty());

    // A write reserves pages, copies its blocks there and commits them with
    // its checkpoint.
    rank.client.submit(&[write(1, &PROMPT)]).unwrap();
    assert_eq!(rank.finished(1), [1]);
    let books = owner.books();
    assert_eq!(books["stats"]["cached_blocks"], 3);
    assert_eq!(books["stats"]["checkpoints"], 1);

    // A longer prompt hits the checkpoint; its blocks are pinned.
    let answers = rank
        .client
        .lookup(&[
            lookup("hit", &[1, 2, 3, 4, 5, 6, 7]),
            lookup("cold", &[9, 9, 9, 9]),
        ])
        .unwrap();
    let hit = &answers[0];
    assert!(hit.ticket > 0);
    assert_eq!((hit.num_tokens, hit.kv_tokens), (5, 5));
    assert_eq!((hit.tail_tokens, hit.tail_hash.clone()), tail.unwrap());
    assert_eq!((&hit.keys, &hit.buckets), (&keys, &vec![0, 0, 0]));
    assert_eq!(
        (answers[1].request_id.as_str(), answers[1].ticket),
        ("cold", 0)
    );
    assert_eq!(owner.books()["stats"]["pinned_blocks"], 3);

    // The hit's pages hold the bytes the write copied: a load reads them
    // back into other Device blocks.
    let pages: Vec<_> = (0..3)
        .map(|index| SharedL2ProxyBlockView {
            group_id: keys[index].group_id,
            host_block: 100 + index as i32,
            bucket: hit.buckets[index],
            pool_page: hit.pool_pages[index],
        })
        .collect();
    rank.client.map_proxy_blocks(&pages);
    rank.signal_words.words[0].store(1, Ordering::Relaxed);
    rank.threads
        .loads()
        .submit(
            vec![9],
            &[(KV, 100, 11), (KV, 101, 12), (STATE, 102, 5)].map(
                |(group_id, host_block, device_block)| LoadRow {
                    group_id,
                    host_block,
                    device_block,
                },
            ),
            0,
            1,
        )
        .unwrap();
    let mut loaded = Vec::new();
    wait_until(|| {
        loaded.extend(rank.threads.loads().poll().unwrap());
        !loaded.is_empty()
    });
    assert_eq!(loaded, [9]);
    assert_eq!(rank.block(KV, 11), rank.block(KV, 1));
    assert_eq!(rank.block(KV, 12), rank.block(KV, 2));
    assert_eq!(rank.block(STATE, 5), rank.block(STATE, 1));

    // A release unpins the hit; a touch names its keys.
    rank.client
        .submit(&[
            notice(SharedL2OpKind::Release, hit.ticket, Vec::new()),
            notice(SharedL2OpKind::Touch, 0, keys[..2].to_vec()),
        ])
        .unwrap();
    wait_until(|| {
        let books = owner.books();
        books["stats"]["pinned_blocks"] == 0 && books["touched"].as_array().unwrap().len() == 2
    });
    assert_eq!(
        owner.books()["touched"],
        serde_json::json!([[0, 0, "1-2", 0], [0, 0, "1-2-3-4", 0]])
    );

    // More than twice the owner's client timeout without a lookup, a write
    // or a notice: the pings kept the rank there.
    thread::sleep(Duration::from_millis(1_200));
    assert_eq!(owner.books()["disconnected"], serde_json::json!([]));
    rank.client.check_failure().unwrap();

    // An error of the pool is the lookup's.
    owner.run(serde_json::json!({"do": "fail", "method": "lookup"}));
    let error = rank
        .client
        .lookup(&[lookup("request", &PROMPT)])
        .unwrap_err();
    assert!(error.0.contains("rejected 'lookup'"), "{error}");
    assert!(error.0.contains("lookup failed"), "{error}");
    owner.run(serde_json::json!({"do": "heal", "method": "lookup"}));

    // A rank the owner does not know is told so.
    let mut stranger = Rank::join(&address, 8, Duration::from_secs(2), NO_KEEPALIVE);
    let error = stranger
        .client
        .lookup(&[lookup("request", &PROMPT)])
        .unwrap_err();
    assert!(
        error.0.contains("client 8 has not joined the pool"),
        "{error}"
    );

    // An owner that does not answer in time fails the lookup.
    let mut hasty = Rank::join(&address, 7, Duration::from_millis(200), NO_KEEPALIVE);
    owner.run(serde_json::json!({"do": "stall"}));
    let error = hasty
        .client
        .lookup(&[lookup("request", &PROMPT)])
        .unwrap_err();
    assert!(error.0.contains("did not answer 'lookup'"), "{error}");
    owner.run(serde_json::json!({"do": "resume"}));
}

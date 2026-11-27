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

//! The pool's owner over the scheduler's pool, which the embedded
//! interpreter builds as the owner's launcher does: its requests handled one
//! by one at a time the test names, and the owner served over real ZMQ to a
//! rank's connection.

use std::{
    sync::{mpsc, Arc},
    thread,
};

use parking_lot::Mutex;
use ts_scheduler::{SharedL2AnswerView, SharedL2OpKind, SharedL2OpView};
use ts_wire::decode_bytes;
use zeromq::DealerSocket;

use super::*;
use crate::shared_l2::{
    testing::{
        build_pool, key_value, lookup, miss_value,
        owner::{field, ints, ok, refused, uint},
        prompt_keys, request, wait_until, ServedOwner,
    },
    wire::OwnerConnection,
};

/// Bytes of a rank's LCM block.
const LCM_BYTES: i64 = 64;
/// Longer than any test: no client is dropped unless the test says when.
const CLIENT_TIMEOUT: Duration = Duration::from_secs(600);
/// A prompt that ends on a page: the checkpoint at its end has no tail.
const PAGED: [i32; 4] = [1, 2, 3, 4];

/// A rank's cache geometry with LCM blocks of `lcm_block_bytes`. The owner
/// reads the block size and compares the rest.
fn geometry(lcm_block_bytes: i64) -> Value {
    Value::Map(vec![
        (Value::from("prefix_granularity"), Value::from(2)),
        (Value::from("lcm_block_bytes"), Value::from(lcm_block_bytes)),
        (
            Value::from("groups"),
            Value::Array(vec![Value::from("kv"), Value::from("state")]),
        ),
    ])
}

fn add_provider(session: &str, base_address: i64, bytes: i64) -> Value {
    request(
        "add_provider",
        vec![
            ("session", Value::from(session)),
            ("base_address", Value::from(base_address)),
            ("bytes", Value::from(bytes)),
        ],
    )
}

fn hello(client: i64, geometry: Value, deployment_id: &str) -> Value {
    request(
        "hello",
        vec![
            ("client", Value::from(client)),
            ("geometry", geometry),
            ("deployment_id", Value::from(deployment_id)),
        ],
    )
}

/// A provider's share of the pool as a hello's reply names it.
fn provider_value(session: &str, base_address: i64, num_lcm_blocks: i64, first: i64) -> Value {
    Value::Map(vec![
        (Value::from("session"), Value::from(session)),
        (Value::from("base_address"), Value::from(base_address)),
        (Value::from("num_lcm_blocks"), Value::from(num_lcm_blocks)),
        (Value::from("first_lcm_block"), Value::from(first)),
    ])
}

/// The owner's reply to `request`, heard at `now`.
fn reply(owner: &mut Owner, now: Instant, request: Value) -> Value {
    decode_bytes(&owner.handle(encode_frames(&request).remove(0), now)).unwrap()
}

/// One of the pool's counters.
fn stat(owner: &mut Owner, now: Instant, name: &str) -> u64 {
    let stats = reply(owner, now, request("stats", Vec::new()));
    uint(field(field(&stats, "stats"), name))
}

/// What the pool's builder was called with: the geometry's bytes and the
/// pool's LCM blocks.
type Built = Arc<Mutex<Vec<(Vec<u8>, i32)>>>;

/// An owner that waits for `num_providers` providers and builds the tests'
/// pool, with the record of what its builder was called with.
fn owner(num_providers: usize) -> (Owner, Built) {
    let built = Built::default();
    let record = built.clone();
    let owner = Owner::new(
        num_providers,
        CLIENT_TIMEOUT,
        move |geometry, num_lcm_blocks| {
            record.lock().push((geometry.to_vec(), num_lcm_blocks));
            build_pool(num_lcm_blocks)
        },
    );
    (owner, built)
}

/// An owner whose provider `node0` holds `num_lcm_blocks` LCM blocks, with
/// `clients` joined at `now`.
fn joined_owner(num_lcm_blocks: i64, clients: &[i64], now: Instant) -> Owner {
    let (mut owner, _) = owner(1);
    let joined = add_provider("node0", 1 << 40, num_lcm_blocks * LCM_BYTES);
    assert_eq!(reply(&mut owner, now, joined), ok(Vec::new()));
    for client in clients {
        let joined = reply(
            &mut owner,
            now,
            hello(*client, geometry(LCM_BYTES), "deployment"),
        );
        assert_eq!(*field(&joined, "ready"), Value::from(true));
    }
    owner
}

fn tokens_value(tokens: &[i32]) -> Value {
    Value::Bin(Bytes::from(
        tokens
            .iter()
            .flat_map(|token| token.to_ne_bytes())
            .collect::<Vec<_>>(),
    ))
}

fn hashes_value(hashes: &[String]) -> Value {
    Value::Array(
        hashes
            .iter()
            .map(|hash| Value::from(hash.as_str()))
            .collect(),
    )
}

/// `client`'s lookup of each of `prompts`, in one request.
fn lookup_request(client: i64, prompts: &[&[i32]]) -> Value {
    let lookups = prompts
        .iter()
        .map(|prompt| {
            Value::Array(vec![
                hashes_value(&prompt_keys(prompt).0),
                tokens_value(prompt),
                Value::from(prompt.len() as i64),
            ])
        })
        .collect();
    request(
        "lookup",
        vec![
            ("client", Value::from(client)),
            ("lookups", Value::Array(lookups)),
        ],
    )
}

/// `client`'s reservation of `prompt`'s blocks.
fn reserve_request(client: i64, prompt: &[i32]) -> Value {
    let (_, _, keys) = prompt_keys(prompt);
    request(
        "reserve",
        vec![
            ("client", Value::from(client)),
            ("keys", Value::Array(keys.iter().map(key_value).collect())),
            ("buckets", ints(&vec![0; keys.len()])),
        ],
    )
}

/// `client`'s commit of what it reserved under `ticket`, with the
/// checkpoint at the end of `prompt`, which ends on a page.
fn commit_request(client: i64, ticket: &Value, prompt: &[i32]) -> Value {
    let checkpoint = Value::Array(vec![
        tokens_value(prompt),
        hashes_value(&prompt_keys(prompt).0),
        Value::Nil,
        Value::from(7),
    ]);
    request(
        "commit",
        vec![
            ("client", Value::from(client)),
            ("ticket", ticket.clone()),
            ("checkpoints", Value::Array(vec![checkpoint])),
        ],
    )
}

/// Reserve and commit `prompt`'s blocks and checkpoint for `client`; the
/// pages they got.
fn store(owner: &mut Owner, now: Instant, client: i64, prompt: &[i32]) -> Value {
    let reserved = reply(owner, now, reserve_request(client, prompt));
    let commit = commit_request(client, field(&reserved, "ticket"), prompt);
    assert_eq!(reply(owner, now, commit), ok(Vec::new()));
    field(&reserved, "pages").clone()
}

fn ping(client: i64) -> Value {
    request("ping", vec![("client", Value::from(client))])
}

fn release(client: i64, ticket: &Value) -> Value {
    request(
        "release",
        vec![("client", Value::from(client)), ("ticket", ticket.clone())],
    )
}

/// The hits of a lookup's reply.
fn hits(reply: &Value) -> &[Value] {
    match field(reply, "hits") {
        Value::Array(hits) => hits,
        other => panic!("hits are an array: {other:?}"),
    }
}

/// The ticket of a hit.
fn ticket(hit: &Value) -> &Value {
    match hit {
        Value::Array(hit) => &hit[0],
        other => panic!("a hit is an array: {other:?}"),
    }
}

#[test]
fn a_hello_is_not_ready_until_every_provider_joined() {
    let now = Instant::now();
    let (mut owner, built) = owner(2);
    let joined = add_provider("node0", 1 << 40, 8 * LCM_BYTES);
    assert_eq!(reply(&mut owner, now, joined), ok(Vec::new()));

    assert_eq!(
        reply(&mut owner, now, hello(1, geometry(LCM_BYTES), "deployment")),
        ok(vec![("ready", Value::from(false))])
    );
    // It has not joined either: nothing is kept for it, and no pool is built.
    assert_eq!(
        reply(&mut owner, now, ping(1)),
        refused("client 1 has not joined the pool")
    );
    assert!(built.lock().is_empty());

    let joined = add_provider("node1", 1 << 41, 4 * LCM_BYTES);
    assert_eq!(reply(&mut owner, now, joined), ok(Vec::new()));
    // A provider's LCM blocks follow those of the providers before it.
    assert_eq!(
        reply(&mut owner, now, hello(1, geometry(LCM_BYTES), "deployment")),
        ok(vec![
            ("ready", Value::from(true)),
            (
                "providers",
                Value::Array(vec![
                    provider_value("node0", 1 << 40, 8, 0),
                    provider_value("node1", 1 << 41, 4, 8),
                ])
            ),
            ("lcm_block_bytes", Value::from(LCM_BYTES)),
        ])
    );
    // The pool is built from that client's geometry, as the bytes it
    // arrived as, over the blocks of both providers.
    assert_eq!(
        *built.lock(),
        [(encode_frames(&geometry(LCM_BYTES)).remove(0).to_vec(), 12)]
    );
    assert_eq!(stat(&mut owner, now, "num_lcm_blocks"), 12);
    assert_eq!(reply(&mut owner, now, ping(1)), ok(Vec::new()));
}

#[test]
fn the_first_client_sets_the_geometry_and_deployment_later_ones_match() {
    let now = Instant::now();
    let (mut owner, built) = owner(1);
    let joined = add_provider("node0", 1 << 40, 8 * LCM_BYTES);
    assert_eq!(reply(&mut owner, now, joined), ok(Vec::new()));
    for client in [1, 2, 1] {
        let joined = reply(
            &mut owner,
            now,
            hello(client, geometry(LCM_BYTES), "deployment"),
        );
        assert_eq!(*field(&joined, "ready"), Value::from(true));
    }
    assert_eq!(built.lock().len(), 1);

    let differs = refused("the client's cache geometry differs from the pool's");
    // Another LCM block; and the same fields in another order: the geometry
    // is compared as the value it arrived as.
    let Value::Map(mut reordered) = geometry(LCM_BYTES) else {
        unreachable!()
    };
    reordered.swap(0, 1);
    for other in [geometry(2 * LCM_BYTES), Value::Map(reordered)] {
        assert_eq!(
            reply(&mut owner, now, hello(3, other, "deployment")),
            differs
        );
    }
    // The same layout holding another checkpoint's KV.
    assert_eq!(
        reply(&mut owner, now, hello(3, geometry(LCM_BYTES), "other")),
        refused("the client's deployment differs from the pool's")
    );
    assert_eq!(
        reply(&mut owner, now, ping(3)),
        refused("client 3 has not joined the pool")
    );
    let stats = reply(&mut owner, now, request("stats", Vec::new()));
    assert_eq!(uint(field(&stats, "num_clients")), 2);
}

#[test]
fn providers_join_before_the_pool_runs() {
    let now = Instant::now();
    let (mut owner, _) = owner(2);
    for session in ["node0", "node1"] {
        let joined = add_provider(session, 1 << 40, 8 * LCM_BYTES);
        assert_eq!(reply(&mut owner, now, joined), ok(Vec::new()));
    }
    let late = refused("every expected provider has already joined");
    let third = add_provider("node2", 1 << 40, 8 * LCM_BYTES);
    assert_eq!(reply(&mut owner, now, third.clone()), late);
    let joined = reply(&mut owner, now, hello(1, geometry(LCM_BYTES), "deployment"));
    assert_eq!(*field(&joined, "ready"), Value::from(true));
    assert_eq!(reply(&mut owner, now, third), late);
}

#[test]
fn the_first_client_sets_the_providers_shares() {
    let now = Instant::now();
    let (mut owner, built) = owner(2);
    // Providers join with their bytes, whatever a block will hold.
    for joined in [
        add_provider("node0", 4096, 200),
        add_provider("node1", 8192, LCM_BYTES),
    ] {
        assert_eq!(reply(&mut owner, now, joined), ok(Vec::new()));
    }

    let joined = reply(&mut owner, now, hello(1, geometry(LCM_BYTES), "deployment"));

    // The pool is cut into the first client's LCM blocks; a provider's bytes
    // short of a whole block are left out.
    let shares = Value::Array(vec![
        provider_value("node0", 4096, 3, 0),
        provider_value("node1", 8192, 1, 3),
    ]);
    assert_eq!(*field(&joined, "providers"), shares);
    assert_eq!(*field(&joined, "lcm_block_bytes"), Value::from(LCM_BYTES));
    assert_eq!(built.lock()[0].1, 4);
    let stats = reply(&mut owner, now, request("stats", Vec::new()));
    assert_eq!(*field(&stats, "providers"), shares);
    // From then on that block size is the pool's.
    assert_eq!(
        reply(&mut owner, now, hello(2, geometry(128), "deployment")),
        refused("the client's cache geometry differs from the pool's")
    );
    let joined = reply(&mut owner, now, hello(2, geometry(LCM_BYTES), "deployment"));
    assert_eq!(*field(&joined, "providers"), shares);
}

#[test]
fn a_provider_smaller_than_a_block_stops_the_pool_from_being_built() {
    let now = Instant::now();
    let (mut owner, built) = owner(2);
    for joined in [
        add_provider("node0", 4096, 10 * LCM_BYTES),
        add_provider("small", 8192, LCM_BYTES - 1),
    ] {
        assert_eq!(reply(&mut owner, now, joined), ok(Vec::new()));
    }

    // Every hello says which provider is too small, and no pool is built.
    for _ in 0..2 {
        assert_eq!(
            reply(&mut owner, now, hello(1, geometry(LCM_BYTES), "deployment")),
            refused("provider small holds 63 bytes, less than one LCM block of 64 bytes")
        );
    }
    // So does a geometry whose LCM block holds nothing.
    assert_eq!(
        reply(&mut owner, now, hello(1, geometry(0), "deployment")),
        refused("provider node0 holds 640 bytes, less than one LCM block of 0 bytes")
    );
    assert!(built.lock().is_empty());
    assert_eq!(
        reply(&mut owner, now, request("stats", Vec::new())),
        refused("the shared L2 pool is not ready")
    );
    assert_eq!(
        reply(&mut owner, now, ping(1)),
        refused("client 1 has not joined the pool")
    );
}

#[test]
fn the_providers_blocks_are_held_to_the_pools_int32_count() {
    let now = Instant::now();
    // A geometry of one byte to an LCM block: a provider's bytes are its
    // blocks.
    let attempts = Built::default();
    let record = attempts.clone();
    let mut at_the_bound = Owner::new(2, CLIENT_TIMEOUT, move |geometry, num_lcm_blocks| {
        record.lock().push((geometry.to_vec(), num_lcm_blocks));
        Err("no memory for the index".to_owned())
    });
    for joined in [
        add_provider("node0", 4096, i64::from(i32::MAX) - 1),
        add_provider("node1", 8192, 1),
    ] {
        assert_eq!(reply(&mut at_the_bound, now, joined), ok(Vec::new()));
    }
    // The builder's failure is the hello's, and the pool stays unbuilt.
    for _ in 0..2 {
        assert_eq!(
            reply(&mut at_the_bound, now, hello(1, geometry(1), "deployment")),
            refused("no memory for the index")
        );
    }
    assert_eq!(
        attempts
            .lock()
            .iter()
            .map(|(_, num_lcm_blocks)| *num_lcm_blocks)
            .collect::<Vec<_>>(),
        [i32::MAX; 2]
    );
    assert_eq!(
        reply(&mut at_the_bound, now, request("stats", Vec::new())),
        refused("the shared L2 pool is not ready")
    );

    // One block more is refused before the builder is asked.
    let (mut past_the_bound, built) = owner(2);
    for joined in [
        add_provider("node0", 4096, i64::from(i32::MAX)),
        add_provider("node1", 8192, 1),
    ] {
        assert_eq!(reply(&mut past_the_bound, now, joined), ok(Vec::new()));
    }
    assert_eq!(
        reply(
            &mut past_the_bound,
            now,
            hello(1, geometry(1), "deployment")
        ),
        refused(
            "the providers hold 2147483648 LCM blocks of 1 bytes, more than a pool's 2147483647"
        )
    );
    assert!(built.lock().is_empty());
}

#[test]
fn one_lookup_request_answers_every_prompt_in_order_and_pins_each_hit() {
    let now = Instant::now();
    let mut owner = joined_owner(8, &[1, 2], now);
    let Value::Array(pages) = store(&mut owner, now, 1, &PAGED) else {
        panic!("pages are an array");
    };

    let answered = reply(
        &mut owner,
        now,
        lookup_request(2, &[&[1, 2, 3, 4, 5, 6, 7], &[9, 9, 9]]),
    );

    // The longer prompt hits the checkpoint, which has no tail: every page
    // is a key with its bucket and the pool page the write got.
    let (_, _, keys) = prompt_keys(&PAGED);
    let Value::Array(hit) = &hits(&answered)[0] else {
        panic!("a hit is an array");
    };
    assert_ne!(hit[0], Value::from(0));
    assert_eq!(hit[1..4], [Value::from(4), Value::from(4), Value::Nil]);
    assert_eq!(
        hit[4],
        Value::Array(
            keys.iter()
                .zip(&pages)
                .map(|(key, page)| Value::Array(vec![key_value(key), Value::from(0), page.clone()]))
                .collect()
        )
    );
    assert_eq!(hits(&answered)[1], miss_value());
    assert_eq!(stat(&mut owner, now, "pinned_blocks"), 3);
    assert_eq!(stat(&mut owner, now, "open_lookups"), 1);
}

#[test]
fn a_write_is_reserved_then_committed() {
    let now = Instant::now();
    // Two LCM blocks: the prompt's two KV pages fill one, its State page
    // the other.
    let mut owner = joined_owner(2, &[1, 2], now);

    let reserved = reply(&mut owner, now, reserve_request(1, &PAGED));
    assert_ne!(*field(&reserved, "ticket"), Value::from(0));
    let Value::Array(pages) = field(&reserved, "pages") else {
        panic!("pages are an array");
    };
    assert!(pages.iter().all(|page| uint(page) >= 1), "{pages:?}");
    assert_eq!(stat(&mut owner, now, "reserved_blocks"), 3);
    // Another client gets no page for a block that is being written, and
    // none for a new block while every page is reserved.
    assert_eq!(
        reply(&mut owner, now, reserve_request(2, &PAGED)),
        ok(vec![
            ("ticket", Value::from(0)),
            ("pages", ints(&[-3, -3, -3]))
        ])
    );
    assert_eq!(
        reply(&mut owner, now, reserve_request(2, &[8, 8])),
        ok(vec![("ticket", Value::from(0)), ("pages", ints(&[-2, -2]))])
    );

    let commit = commit_request(1, field(&reserved, "ticket"), &PAGED);
    assert_eq!(reply(&mut owner, now, commit.clone()), ok(Vec::new()));
    assert_eq!(stat(&mut owner, now, "reserved_blocks"), 0);
    assert_eq!(stat(&mut owner, now, "cached_blocks"), 3);
    assert_eq!(stat(&mut owner, now, "checkpoints"), 1);
    // A ticket that reserves nothing is refused, and the pool holds the
    // blocks now.
    assert_eq!(
        reply(&mut owner, now, commit),
        refused("a shared L2 commit names a ticket under which its client reserves nothing")
    );
    assert_eq!(
        reply(&mut owner, now, reserve_request(2, &PAGED)),
        ok(vec![
            ("ticket", Value::from(0)),
            ("pages", ints(&[-1, -1, -1]))
        ])
    );
}

#[test]
fn a_release_unpins_what_its_clients_lookup_hit() {
    let now = Instant::now();
    let mut owner = joined_owner(8, &[1, 2], now);
    store(&mut owner, now, 1, &PAGED);

    let first = reply(&mut owner, now, lookup_request(2, &[&PAGED]));
    let second = reply(&mut owner, now, lookup_request(2, &[&PAGED]));
    let (first, second) = (ticket(&hits(&first)[0]), ticket(&hits(&second)[0]));
    assert_ne!(first, second);
    assert_eq!(stat(&mut owner, now, "open_lookups"), 2);
    assert_eq!(stat(&mut owner, now, "pinned_blocks"), 3);

    assert_eq!(reply(&mut owner, now, release(2, first)), ok(Vec::new()));
    assert_eq!(stat(&mut owner, now, "open_lookups"), 1);
    // A ticket is its client's: another client's release is refused.
    assert_eq!(
        reply(&mut owner, now, release(1, second)),
        refused("a shared L2 release names a ticket no lookup of its client holds")
    );
    assert_eq!(stat(&mut owner, now, "pinned_blocks"), 3);
    assert_eq!(reply(&mut owner, now, release(2, second)), ok(Vec::new()));
    assert_eq!(stat(&mut owner, now, "pinned_blocks"), 0);
}

#[test]
fn a_silent_client_loses_what_it_held_and_is_told_so() {
    let joined = Instant::now();
    let mut owner = joined_owner(8, &[1, 2], joined);
    store(&mut owner, joined, 2, &PAGED);
    let hit = reply(&mut owner, joined, lookup_request(1, &[&PAGED]));
    let held = ticket(&hits(&hit)[0]).clone();
    reply(&mut owner, joined, reserve_request(1, &[8, 8]));
    assert_eq!(stat(&mut owner, joined, "pinned_blocks"), 3);
    assert_eq!(stat(&mut owner, joined, "reserved_blocks"), 2);

    // Client 2 goes on talking. A client is kept while it was heard within
    // the timeout, to the moment.
    let later = joined + CLIENT_TIMEOUT;
    assert_eq!(reply(&mut owner, later, ping(2)), ok(Vec::new()));
    owner.disconnect_timed_out_clients(later);
    assert_eq!(stat(&mut owner, later, "pinned_blocks"), 3);
    let later = later + Duration::from_millis(1);
    owner.disconnect_timed_out_clients(later);

    assert_eq!(stat(&mut owner, later, "pinned_blocks"), 0);
    assert_eq!(stat(&mut owner, later, "reserved_blocks"), 0);
    // Its next request tells it; the client that kept talking is still in.
    assert_eq!(
        reply(&mut owner, later, release(1, &held)),
        refused("client 1 has not joined the pool")
    );
    assert_eq!(reply(&mut owner, later, ping(2)), ok(Vec::new()));
    let stats = reply(&mut owner, later, request("stats", Vec::new()));
    assert_eq!(uint(field(&stats, "num_clients")), 1);
}

#[test]
fn touch_clear_and_stats() {
    let now = Instant::now();
    let mut owner = joined_owner(8, &[1], now);
    store(&mut owner, now, 1, &PAGED);
    let (_, _, keys) = prompt_keys(&PAGED);

    let touch = request(
        "touch",
        vec![
            ("client", Value::from(1)),
            (
                "keys",
                Value::Array(keys[..2].iter().map(key_value).collect()),
            ),
        ],
    );
    assert_eq!(reply(&mut owner, now, touch), ok(Vec::new()));

    // The pool is not emptied while a client holds pins.
    let hit = reply(&mut owner, now, lookup_request(1, &[&PAGED]));
    let clear = request("clear", Vec::new());
    assert_eq!(
        reply(&mut owner, now, clear.clone()),
        ok(vec![("cleared", Value::from(false))])
    );
    let held = ticket(&hits(&hit)[0]);
    assert_eq!(reply(&mut owner, now, release(1, held)), ok(Vec::new()));
    // Two of the pool's eight LCM blocks hold the prompt's pages.
    let counters = |cached_blocks, checkpoints, empty_lcm_blocks| {
        Value::Map(
            [
                ("cached_blocks", cached_blocks),
                ("checkpoints", checkpoints),
                ("empty_lcm_blocks", empty_lcm_blocks),
                ("evicted_blocks", 0),
                ("num_lcm_blocks", 8),
                ("open_lookups", 0),
                ("pinned_blocks", 0),
                ("reserved_blocks", 0),
            ]
            .into_iter()
            .map(|(name, value)| (Value::from(name), Value::from(value)))
            .collect(),
        )
    };
    let stats = |counters| {
        ok(vec![
            ("stats", counters),
            (
                "providers",
                Value::Array(vec![provider_value("node0", 1 << 40, 8, 0)]),
            ),
            ("num_clients", Value::from(1)),
        ])
    };
    assert_eq!(
        reply(&mut owner, now, request("stats", Vec::new())),
        stats(counters(3, 1, 6))
    );
    assert_eq!(
        reply(&mut owner, now, clear),
        ok(vec![("cleared", Value::from(true))])
    );
    assert_eq!(
        reply(&mut owner, now, request("stats", Vec::new())),
        stats(counters(0, 0, 8))
    );
}

#[test]
fn a_failed_request_gets_an_error_and_the_owner_keeps_serving() {
    let now = Instant::now();
    let (mut owner, _) = owner(1);
    assert_eq!(
        reply(&mut owner, now, request("stats", Vec::new())),
        refused("the shared L2 pool is not ready")
    );
    let joined = add_provider("node0", 1 << 40, 8 * LCM_BYTES);
    assert_eq!(reply(&mut owner, now, joined), ok(Vec::new()));
    let joined = reply(&mut owner, now, hello(1, geometry(LCM_BYTES), "deployment"));
    assert_eq!(*field(&joined, "ready"), Value::from(true));

    assert_eq!(
        reply(
            &mut owner,
            now,
            request("steal", vec![("client", Value::from(1))])
        ),
        refused("unknown shared L2 request 'steal'")
    );
    // A request that does not decode: no operation, no map, a field of
    // another type, tokens that are not int32s.
    for (spoiled, reason) in [
        (
            Value::Map(vec![(Value::from("client"), Value::from(1))]),
            "missing required field `op`",
        ),
        (Value::from("ping"), "expected map"),
        (
            request("release", vec![("client", Value::from(1))]),
            "missing required field `ticket`",
        ),
        (
            request("ping", vec![("client", Value::from("one"))]),
            "expected int",
        ),
        (
            request(
                "lookup",
                vec![
                    ("client", Value::from(1)),
                    (
                        "lookups",
                        Value::Array(vec![Value::Array(vec![
                            Value::Array(Vec::new()),
                            Value::Bin(Bytes::from_static(b"abc")),
                            Value::from(0),
                        ])]),
                    ),
                ],
            ),
            "shared L2 tokens must be int32 bytes",
        ),
    ] {
        let failed = reply(&mut owner, now, spoiled.clone());
        assert_eq!(*field(&failed, "ok"), Value::from(false), "{spoiled:?}");
        let Value::Str(error) = field(&failed, "error") else {
            panic!("an error is a string: {failed:?}");
        };
        assert!(error.contains(reason), "{error}");
    }
    // A pool error fails the request, not the owner: a bucket the pool does
    // not have.
    let mut misplaced = reserve_request(1, &PAGED);
    let Value::Map(fields) = &mut misplaced else {
        unreachable!()
    };
    fields[3].1 = ints(&[0, 0, 1]);
    assert_eq!(
        reply(&mut owner, now, misplaced),
        refused("a shared L2 bucket lies outside its group's shard count")
    );
    assert_eq!(stat(&mut owner, now, "reserved_blocks"), 0);
    assert_eq!(reply(&mut owner, now, ping(1)), ok(Vec::new()));
}

/// An owner served at `address`, its provider `node0` joined with
/// `num_lcm_blocks` LCM blocks and `clients` joined.
fn served_owner(owner: Owner, address: &str, num_lcm_blocks: i64, clients: &[i64]) -> ServedOwner {
    let owner = ServedOwner::start(owner, address);
    let joined = add_provider("node0", 10_000, num_lcm_blocks * LCM_BYTES);
    assert_eq!(owner.ask(&joined), ok(Vec::new()));
    for client in clients {
        let joined = owner.ask(&hello(*client, geometry(LCM_BYTES), "deployment"));
        assert_eq!(*field(&joined, "ready"), Value::from(true));
    }
    owner
}

#[test]
fn the_owner_serves_a_ranks_connection_over_the_pool() {
    let owner = served_owner(owner(1).0, "tcp://127.0.0.1:0", 8, &[7]);
    let timeout = Duration::from_secs(5);
    let mut rank = OwnerConnection::connect(&owner.address, 7, timeout).unwrap();
    let (hashes, _, keys) = prompt_keys(&PAGED);

    // An empty pool answers a miss.
    let longer = lookup("longer", &[1, 2, 3, 4, 5, 6, 7]);
    let answers = rank.lookup(std::slice::from_ref(&longer)).unwrap();
    assert_eq!(
        (answers[0].request_id.as_str(), answers[0].ticket),
        ("longer", 0)
    );

    // A write reserves pages and commits them with its checkpoint.
    let write = SharedL2OpView {
        kind: SharedL2OpKind::Write,
        op_id: 1,
        ticket: 0,
        keys: keys.clone(),
        buckets: vec![0; 3],
        device_blocks: vec![1, 2, 1],
        checkpoints: vec![ts_scheduler::SharedL2CheckpointView {
            tokens: PAGED.to_vec(),
            prefix_hashes: hashes,
            tail_tokens: 0,
            tail_hash: String::new(),
            access_epoch: 7,
        }],
    };
    let (reservation, pages) = rank.reserve(&write).unwrap().unwrap();
    rank.commit(reservation, &write.checkpoints).unwrap();
    assert_eq!(rank.reserve(&write).unwrap(), None);

    // The longer prompt hits the checkpoint at the pages the write got.
    let answers = rank.lookup(&[longer, lookup("cold", &[9, 9])]).unwrap();
    let hit = &answers[0];
    assert!(hit.ticket != 0);
    assert_eq!(
        *hit,
        SharedL2AnswerView {
            request_id: "longer".to_owned(),
            ticket: hit.ticket,
            num_tokens: 4,
            kv_tokens: 4,
            tail_tokens: 0,
            tail_hash: String::new(),
            keys: keys.clone(),
            buckets: vec![0; 3],
            pool_pages: pages,
        }
    );
    assert_eq!(
        (answers[1].request_id.as_str(), answers[1].ticket),
        ("cold", 0)
    );
    let stats = owner.ask(&request("stats", Vec::new()));
    assert_eq!(uint(field(field(&stats, "stats"), "pinned_blocks")), 3);

    rank.touch(&keys[..2]).unwrap();
    rank.release(hit.ticket).unwrap();
    rank.ping().unwrap();
    let stats = owner.ask(&request("stats", Vec::new()));
    assert_eq!(uint(field(field(&stats, "stats"), "pinned_blocks")), 0);

    // An error of the pool and of the owner are the request's.
    assert_eq!(
        rank.release(hit.ticket).unwrap_err().0,
        "the shared L2 owner rejected 'release': a shared L2 release names a ticket no \
         lookup of its client holds"
    );
    let mut stranger = OwnerConnection::connect(&owner.address, 8, timeout).unwrap();
    assert_eq!(
        stranger.ping().unwrap_err().0,
        "the shared L2 owner rejected 'ping': client 8 has not joined the pool"
    );
}

#[test]
fn every_frame_before_the_payload_goes_back_unchanged() {
    let owner = served_owner(owner(1).0, "tcp://127.0.0.1:0", 8, &[7]);
    let payload = encode_frames(&ping(7)).remove(0);
    // A REQ socket that correlates its requests sends a request id and an
    // empty delimiter before the payload; a rank sends a correlation frame.
    for envelope in [
        vec![Bytes::from_static(b"\x00\x00\x00\x2a"), Bytes::new()],
        vec![Bytes::from_static(b"\x00\x00\x00\x00\x00\x00\x00\x01")],
        Vec::new(),
    ] {
        let mut frames = envelope.clone();
        frames.push(payload.clone());
        let mut answered = owner.exchange(frames);
        let reply: Value = decode_bytes(&answered.pop().unwrap()).unwrap();
        assert_eq!(reply, ok(Vec::new()));
        assert_eq!(answered, envelope);
    }
}

#[test]
fn a_client_that_left_before_its_replies_does_not_stop_the_owner() {
    // The pool's builder waits for the test: the owner is busy with the
    // first hello until then.
    let (building, entered) = mpsc::channel();
    let (go_on, resumed) = mpsc::channel::<()>();
    let owner = Owner::new(1, CLIENT_TIMEOUT, move |_, num_lcm_blocks| {
        building.send(()).unwrap();
        resumed.recv().unwrap();
        build_pool(num_lcm_blocks)
    });
    let owner = served_owner(owner, "tcp://127.0.0.1:0", 8, &[]);
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let mut leaving = runtime.block_on(async {
        let mut socket = DealerSocket::new();
        socket.connect(&owner.address).await.unwrap();
        socket
    });

    thread::scope(|scope| {
        let joining = scope.spawn(|| owner.ask(&hello(1, geometry(LCM_BYTES), "deployment")));
        entered.recv().unwrap();
        // While the owner is busy a client sends its requests and leaves.
        let payload = encode_frames(&ping(9)).remove(0);
        runtime.block_on(async {
            for _ in 0..32 {
                leaving
                    .send(ZmqMessage::from(payload.clone()))
                    .await
                    .unwrap();
            }
        });
        drop(leaving);
        go_on.send(()).unwrap();
        assert_eq!(*field(&joining.join().unwrap(), "ready"), Value::from(true));
    });

    // The owner read the requests of the client that left, and serves on.
    assert_eq!(owner.ask(&ping(1)), ok(Vec::new()));
    let stats = owner.ask(&request("stats", Vec::new()));
    assert_eq!(uint(field(&stats, "num_clients")), 1);
}

#[test]
fn the_owner_binds_every_interface_or_a_named_host() {
    for address in ["tcp://*:0", "tcp://localhost:0"] {
        let owner = ServedOwner::start(owner(1).0, address);
        assert!(!owner.address.ends_with(":0"), "{}", owner.address);
        assert_eq!(
            owner.ask(&request("stats", Vec::new())),
            refused("the shared L2 pool is not ready")
        );
    }
}

#[test]
fn a_served_client_that_goes_silent_is_dropped_within_its_timeout() {
    let client_timeout = Duration::from_millis(500);
    let owner = Owner::new(1, client_timeout, |_, num_lcm_blocks| {
        build_pool(num_lcm_blocks)
    });
    let joined = Instant::now();
    let owner = served_owner(owner, "tcp://127.0.0.1:0", 8, &[1, 2]);
    let num_clients = || {
        uint(field(
            &owner.ask(&request("stats", Vec::new())),
            "num_clients",
        ))
    };
    assert_eq!(num_clients(), 2);

    // Client 2 keeps talking; client 1 is dropped once it has been silent
    // for the timeout, at the owner's next sweep, a quarter of it later at
    // most.
    wait_until(|| {
        assert_eq!(owner.ask(&ping(2)), ok(Vec::new()));
        num_clients() == 1
    });
    assert!(joined.elapsed() > client_timeout);
    assert_eq!(
        owner.ask(&ping(1)),
        refused("client 1 has not joined the pool")
    );
}

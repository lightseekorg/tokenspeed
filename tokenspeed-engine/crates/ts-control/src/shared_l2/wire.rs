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

//! A rank's requests to the pool's owner (`cache/shared_l2/owner.py`).
//!
//! A request is a msgpack map, `op`, `client` and the operation's fields; a
//! reply is `{"ok": true, ...}` or `{"ok": false, "error": ...}`. Cache keys
//! travel as `[namespace_id, group_id, content_hash, page_offset]`, a key the
//! owner places with its bucket, and a prompt's tokens as int32 bytes in
//! native byte order (`cache/shared_l2/wire.py`).
//!
//! The owner answers on a ROUTER socket and returns every frame before the
//! payload unchanged. A connection is a DEALER socket of one thread, and
//! each request carries a frame that tells its reply from the late reply of
//! an earlier request.
//!
//! No request is sent twice, and every failure is the rank's: no reply
//! within the connection's timeout (the owner may still handle the
//! request), an error of the owner (one that does not know the rank
//! answers so), an answer that is not a reply of the wire or lacks a field
//! of the operation's reply, a socket that failed.

use std::time::Duration;

use bytes::Bytes;
use tokio::runtime::Runtime;
use ts_scheduler::{
    SharedL2AnswerView, SharedL2CheckpointView, SharedL2LookupView, SharedL2OpView, StorageKeyView,
};
use ts_wire::{
    msgpack::{Decoder, Encoder},
    Wire,
};
use zeromq::{DealerSocket, Socket, SocketOptions, SocketRecv, SocketSend, ZmqMessage};

use crate::shared_l2::PoolError;

/// The namespace of every pool key. The scheduler's keys do not carry it: a
/// pool holds the default namespace alone.
const DEFAULT_CACHE_NAMESPACE_ID: u32 = 0;

/// Read an array's header off `decoder` and check that the array, `what`,
/// has `len` elements.
fn expect_array_len(decoder: &mut Decoder<'_>, len: usize, what: &str) -> ts_wire::Result<()> {
    let found = decoder.array_len()?;
    if found != len {
        return Err(ts_wire::Error::Invalid(format!(
            "{what} has {found} elements, not {len}"
        )));
    }
    Ok(())
}

/// A lookup's hit as the owner answers it, `[ticket, tokens, KV tokens,
/// tail, [[key, bucket, page]]]`, for the request that asked. A miss has
/// ticket 0 and no key.
fn decode_answer(
    decoder: &mut Decoder<'_>,
    request_id: &str,
) -> ts_wire::Result<SharedL2AnswerView> {
    expect_array_len(decoder, 5, "a hit")?;
    let ticket = u64::decode(decoder)?;
    let num_tokens = i32::decode(decoder)?;
    let kv_tokens = i32::decode(decoder)?;
    // The partial last page: its tokens and its hash.
    let (tail_tokens, tail_hash) = Option::<(i32, String)>::decode(decoder)?.unwrap_or_default();
    let (mut keys, mut buckets, mut pool_pages) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..decoder.array_len()? {
        expect_array_len(decoder, 3, "a page")?;
        expect_array_len(decoder, 4, "a key")?;
        let namespace_id = u32::decode(decoder)?;
        if namespace_id != DEFAULT_CACHE_NAMESPACE_ID {
            return Err(ts_wire::Error::Invalid(format!(
                "a key of namespace {namespace_id}, not {DEFAULT_CACHE_NAMESPACE_ID}"
            )));
        }
        keys.push(StorageKeyView {
            group_id: u32::decode(decoder)?,
            content_hash: String::decode(decoder)?,
            page_offset: i32::decode(decoder)?,
        });
        buckets.push(i32::decode(decoder)?);
        pool_pages.push(i32::decode(decoder)?);
    }
    Ok(SharedL2AnswerView {
        request_id: request_id.to_owned(),
        ticket,
        num_tokens,
        kv_tokens,
        tail_tokens,
        tail_hash,
        keys,
        buckets,
        pool_pages,
    })
}

/// Tokens as the pool carries them: int32, native byte order.
fn token_bytes(tokens: &[i32]) -> Vec<u8> {
    tokens
        .iter()
        .flat_map(|token| token.to_ne_bytes())
        .collect()
}

fn encode_keys(encoder: &mut Encoder, keys: &[StorageKeyView]) {
    encoder.array_len(keys.len());
    for key in keys {
        encoder.array_len(4);
        DEFAULT_CACHE_NAMESPACE_ID.encode(encoder);
        key.group_id.encode(encoder);
        key.content_hash.encode(encoder);
        key.page_offset.encode(encoder);
    }
}

/// The owner's reply to one request: a map that says `ok`.
struct Reply {
    op: &'static str,
    /// The msgpack map alone.
    payload: [Bytes; 1],
}

impl Reply {
    /// Read the reply's field `name` with `decode`.
    fn read<T>(
        &self,
        name: &'static str,
        decode: impl FnOnce(&mut Decoder<'_>) -> ts_wire::Result<T>,
    ) -> Result<T, PoolError> {
        let mut decoder = Decoder::new(&self.payload);
        let field = (|| {
            for _ in 0..decoder.map_len()? {
                if decoder.str()? == name {
                    return decode(&mut decoder);
                }
                decoder.skip()?;
            }
            Err(ts_wire::Error::MissingField(name))
        })();
        field.map_err(|error| {
            PoolError(format!(
                "the shared L2 owner's reply to '{}' is not a reply: {error}",
                self.op
            ))
        })
    }
}

/// One thread's connection to the owner, as the owner's client `client_id`.
pub struct OwnerConnection {
    /// Drives the socket during a request and at no other time.
    runtime: Runtime,
    socket: DealerSocket,
    client_id: u64,
    timeout: Duration,
    /// Requests sent so far: the next one's correlation frame.
    sent: u64,
}

impl OwnerConnection {
    /// Connect to the owner at `address`, waiting up to `timeout`; every
    /// request then waits up to `timeout` for its reply.
    pub fn connect(address: &str, client_id: u64, timeout: Duration) -> Result<Self, PoolError> {
        let failed = |error: &dyn std::fmt::Display| {
            PoolError(format!(
                "the connection to the shared L2 owner failed: {error}"
            ))
        };
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|error| failed(&error))?;
        let socket = runtime
            .block_on(async {
                let mut options = SocketOptions::default();
                options.connect_timeout(timeout);
                let mut socket = DealerSocket::with_options(options);
                socket.connect(address).await.map(|()| socket)
            })
            .map_err(|error| failed(&error))?;
        Ok(Self {
            runtime,
            socket,
            client_id,
            timeout,
            sent: 0,
        })
    }

    /// Send the request `op` with the `num_fields` fields `fields` writes,
    /// and wait for its reply.
    fn request(
        &mut self,
        op: &'static str,
        num_fields: usize,
        fields: impl FnOnce(&mut Encoder),
    ) -> Result<Reply, PoolError> {
        let mut encoder = Encoder::new();
        encoder.map_len(2 + num_fields);
        encoder.str("op");
        encoder.str(op);
        encoder.str("client");
        self.client_id.encode(&mut encoder);
        fields(&mut encoder);
        self.sent += 1;
        let correlation = Bytes::copy_from_slice(&self.sent.to_be_bytes());
        let message = ZmqMessage::try_from(vec![
            correlation.clone(),
            Bytes::from(encoder.into_single_frame()),
        ])
        .expect("a message has frames");
        let socket = &mut self.socket;
        let answered = self.runtime.block_on(async {
            tokio::time::timeout(self.timeout, async {
                socket.send(message).await?;
                loop {
                    let frames = socket.recv().await?.into_vec();
                    // A reply to an earlier request, which timed out, is not
                    // this one's.
                    if frames.first() == Some(&correlation) {
                        return Ok::<_, zeromq::ZmqError>(frames);
                    }
                }
            })
            .await
        });
        let frames = match answered {
            Err(_) => {
                return Err(PoolError(format!(
                    "the shared L2 owner did not answer '{op}'"
                )))
            }
            Ok(Err(error)) => {
                return Err(PoolError(format!(
                    "the connection to the shared L2 owner failed: {error}"
                )))
            }
            Ok(Ok(frames)) => frames,
        };
        // The correlation frame, then the reply's map.
        let Ok([_, payload]) = <[Bytes; 2]>::try_from(frames) else {
            return Err(PoolError(format!(
                "the shared L2 owner's reply to '{op}' is not a reply: it is not one payload"
            )));
        };
        let reply = Reply {
            op,
            payload: [payload],
        };
        if reply.read("ok", bool::decode)? {
            return Ok(reply);
        }
        Err(PoolError(format!(
            "the shared L2 owner rejected '{op}': {}",
            reply.read("error", String::decode)?
        )))
    }

    /// Ask for the pool's hit of every prompt, in one request, and return
    /// one answer for each lookup, in order. The owner pins each hit's
    /// pages under the hit's ticket.
    pub fn lookup(
        &mut self,
        lookups: &[SharedL2LookupView],
    ) -> Result<Vec<SharedL2AnswerView>, PoolError> {
        let reply = self.request("lookup", 1, |encoder| {
            encoder.str("lookups");
            encoder.array_len(lookups.len());
            for lookup in lookups {
                encoder.array_len(3);
                lookup.prefix_hashes.encode(encoder);
                encoder.bin(&token_bytes(&lookup.tokens));
                lookup.max_tokens.encode(encoder);
            }
        })?;
        reply.read("hits", |decoder| {
            let num_hits = decoder.array_len()?;
            if num_hits != lookups.len() {
                return Err(ts_wire::Error::Invalid(format!(
                    "{num_hits} hits for {} lookups",
                    lookups.len()
                )));
            }
            lookups
                .iter()
                .map(|lookup| decode_answer(decoder, &lookup.request_id))
                .collect()
        })
    }

    /// Reserve pool pages for the keys of `write`. Returns the
    /// reservation's ticket and one page for each key; a key without one
    /// (negative) is in the pool, is being written by another rank, or has
    /// no room. `None`: no key got a page (the owner's ticket 0).
    pub fn reserve(
        &mut self,
        write: &SharedL2OpView,
    ) -> Result<Option<(u64, Vec<i32>)>, PoolError> {
        let reply = self.request("reserve", 2, |encoder| {
            encoder.str("keys");
            encode_keys(encoder, &write.keys);
            encoder.str("buckets");
            write.buckets.encode(encoder);
        })?;
        let pages = reply.read("pages", |decoder| {
            expect_array_len(decoder, write.keys.len(), "a reservation's page list")?;
            write.keys.iter().map(|_| i32::decode(decoder)).collect()
        })?;
        let ticket = reply.read("ticket", u64::decode)?;
        Ok((ticket != 0).then_some((ticket, pages)))
    }

    /// Commit the pages reserved under `ticket`, with the checkpoints among
    /// them. The owner rejects a ticket under which the rank reserves
    /// nothing.
    pub fn commit(
        &mut self,
        ticket: u64,
        checkpoints: &[SharedL2CheckpointView],
    ) -> Result<(), PoolError> {
        self.request("commit", 2, |encoder| {
            encoder.str("ticket");
            ticket.encode(encoder);
            encoder.str("checkpoints");
            encoder.array_len(checkpoints.len());
            for checkpoint in checkpoints {
                encoder.array_len(4);
                encoder.bin(&token_bytes(&checkpoint.tokens));
                checkpoint.prefix_hashes.encode(encoder);
                // A checkpoint that ends on a page has no tail.
                (checkpoint.tail_tokens != 0)
                    .then(|| (checkpoint.tail_tokens, checkpoint.tail_hash.clone()))
                    .encode(encoder);
                checkpoint.access_epoch.encode(encoder);
            }
        })
        .map(drop)
    }

    /// Unpin what a lookup hit under `ticket`.
    pub fn release(&mut self, ticket: u64) -> Result<(), PoolError> {
        self.request("release", 1, |encoder| {
            encoder.str("ticket");
            ticket.encode(encoder);
        })
        .map(drop)
    }

    /// Tell the owner that the rank used its own copies of `keys`.
    pub fn touch(&mut self, keys: &[StorageKeyView]) -> Result<(), PoolError> {
        self.request("touch", 1, |encoder| {
            encoder.str("keys");
            encode_keys(encoder, keys);
        })
        .map(drop)
    }

    /// Tell the owner the rank is there: it drops a client it has not heard
    /// for its client timeout.
    pub fn ping(&mut self) -> Result<(), PoolError> {
        self.request("ping", 0, |_| ()).map(drop)
    }
}

#[cfg(test)]
mod tests {
    use ts_scheduler::SharedL2OpKind;
    use ts_wire::Value;

    use super::*;
    use crate::shared_l2::testing::{
        connect, hit_value, key_value, lookup, miss_value,
        owner::{field, ints, ok, op, refused, uint, TestOwner},
        prompt_keys, PROMPT,
    };

    const TIMEOUT: Duration = Duration::from_secs(5);

    /// The write of `PROMPT`'s blocks from Device blocks 1, 2 and 1.
    fn write() -> SharedL2OpView {
        let (prefix_hashes, tail, keys) = prompt_keys(&PROMPT);
        let (tail_tokens, tail_hash) = tail.unwrap();
        SharedL2OpView {
            kind: SharedL2OpKind::Write,
            op_id: 31,
            ticket: 0,
            keys,
            buckets: vec![0, 0, 0],
            device_blocks: vec![1, 2, 1],
            checkpoints: vec![
                SharedL2CheckpointView {
                    tokens: PROMPT.to_vec(),
                    prefix_hashes: prefix_hashes.clone(),
                    tail_tokens,
                    tail_hash,
                    access_epoch: 7,
                },
                // A checkpoint that ends on a page.
                SharedL2CheckpointView {
                    tokens: PROMPT[..4].to_vec(),
                    prefix_hashes,
                    tail_tokens: 0,
                    tail_hash: String::new(),
                    access_epoch: u64::MAX,
                },
            ],
        }
    }

    #[test]
    fn a_lookup_asks_once_and_reads_an_answer_for_each_prompt() {
        let owner = TestOwner::serve(|_| {
            Some(ok(vec![(
                "hits",
                Value::Array(vec![hit_value(41, [3, 4, 8]), miss_value()]),
            )]))
        });
        let mut connection = connect(&owner, TIMEOUT);

        let answers = connection
            .lookup(&[
                lookup("hit", &[1, 2, 3, 4, 5, 6, 7]),
                lookup("cold", &[9, 9]),
            ])
            .unwrap();

        let (_, _, keys) = prompt_keys(&PROMPT);
        assert_eq!(
            answers,
            [
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
                },
                // A prompt the pool holds no prefix of is answered like the
                // others: a miss.
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
                },
            ]
        );
        // One request carried both prompts, as the Python client sent them:
        // hashes, the tokens as int32 bytes, the bound.
        let requests = owner.requests();
        assert_eq!(requests.len(), 1);
        let request = &requests[0];
        assert_eq!(op(request), "lookup");
        assert_eq!(uint(field(request, "client")), 7);
        let prompt = |hashes: &[&str], tokens: &[i32]| {
            Value::Array(vec![
                Value::Array(hashes.iter().map(|hash| Value::from(*hash)).collect()),
                Value::Bin(Bytes::from(token_bytes(tokens))),
                Value::from(tokens.len() as i64),
            ])
        };
        assert_eq!(
            *field(request, "lookups"),
            Value::Array(vec![
                prompt(&["1-2", "1-2-3-4", "1-2-3-4-5-6"], &[1, 2, 3, 4, 5, 6, 7]),
                prompt(&["9-9"], &[9, 9]),
            ])
        );
    }

    #[test]
    fn a_write_reserves_and_commits_with_its_checkpoints() {
        let owner = TestOwner::serve(|request| {
            Some(match op(request) {
                "reserve" => ok(vec![
                    ("ticket", Value::from(12)),
                    ("pages", ints(&[4, -1, 8])),
                ]),
                _ => ok(Vec::new()),
            })
        });
        let mut connection = connect(&owner, TIMEOUT);
        let write = write();

        assert_eq!(
            connection.reserve(&write).unwrap(),
            Some((12, vec![4, -1, 8]))
        );
        connection.commit(12, &write.checkpoints).unwrap();
        connection.release(41).unwrap();
        connection.touch(&write.keys[..2]).unwrap();
        connection.ping().unwrap();

        assert_eq!(
            owner.ops(),
            ["reserve", "commit", "release", "touch", "ping"]
        );
        let requests = owner.requests();
        let keys = |keys: &[StorageKeyView]| Value::Array(keys.iter().map(key_value).collect());
        assert_eq!(*field(&requests[0], "keys"), keys(&write.keys));
        assert_eq!(*field(&requests[0], "buckets"), ints(&[0, 0, 0]));
        assert_eq!(uint(field(&requests[1], "ticket")), 12);
        let Value::Array(sent) = field(&requests[1], "checkpoints") else {
            panic!("checkpoints are an array");
        };
        // The tokens as int32 bytes, the hashes, the tail, the access epoch. A
        // checkpoint that ends on a page has no tail.
        let (Value::Array(tailed), Value::Array(paged)) = (&sent[0], &sent[1]) else {
            panic!("a checkpoint is an array");
        };
        assert_eq!(tailed[0], Value::Bin(Bytes::from(token_bytes(&PROMPT))));
        assert_eq!(
            tailed[2],
            Value::Array(vec![Value::from(1), Value::from("1-2-3-4-5")])
        );
        assert_eq!(uint(&tailed[3]), 7);
        assert_eq!((&paged[2], uint(&paged[3])), (&Value::Nil, u64::MAX));
        assert_eq!(uint(field(&requests[2], "ticket")), 41);
        assert_eq!(*field(&requests[3], "keys"), keys(&write.keys[..2]));
    }

    #[test]
    fn a_request_the_owner_does_not_answer_in_time_fails_and_its_late_reply_is_skipped() {
        // The ping's reply leaves the owner with the reservation's, ahead of
        // it.
        let owner = TestOwner::serve_first_reply_late(|request| {
            Some(match op(request) {
                "reserve" => ok(vec![
                    ("ticket", Value::from(12)),
                    ("pages", ints(&[4, 5, 8])),
                ]),
                _ => ok(Vec::new()),
            })
        });
        let mut connection = connect(&owner, Duration::from_millis(100));

        let started = std::time::Instant::now();
        let error = connection.ping().unwrap_err();
        assert_eq!(error.0, "the shared L2 owner did not answer 'ping'");
        assert!(started.elapsed() >= Duration::from_millis(100));
        // The next request takes its own reply, not the ping's.
        assert_eq!(
            connection.reserve(&write()).unwrap(),
            Some((12, vec![4, 5, 8]))
        );
        // Nothing was sent twice.
        assert_eq!(owner.ops(), ["ping", "reserve"]);
    }

    #[test]
    fn an_error_of_the_owner_is_the_requests() {
        let owner = TestOwner::serve(|_| Some(refused("client 7 has not joined the pool")));
        let error = connect(&owner, TIMEOUT)
            .lookup(&[lookup("request", &PROMPT)])
            .unwrap_err();
        assert_eq!(
            error.0,
            "the shared L2 owner rejected 'lookup': client 7 has not joined the pool"
        );
    }

    #[test]
    fn an_answer_that_is_not_a_reply_of_the_wire_fails() {
        // No `ok`, not a map, no field of the operation's reply, an error
        // without its words, not one hit for each lookup.
        for (reply, reason) in [
            (
                Value::Map(vec![(Value::from("hits"), Value::Array(Vec::new()))]),
                "missing required field `ok`",
            ),
            (Value::Array(vec![Value::from("ok")]), "expected map"),
            (ok(Vec::new()), "missing required field `hits`"),
            (
                Value::Map(vec![(Value::from("ok"), Value::from(false))]),
                "missing required field `error`",
            ),
            (
                ok(vec![(
                    "hits",
                    Value::Array(vec![miss_value(), miss_value()]),
                )]),
                "2 hits for 1 lookups",
            ),
        ] {
            let owner = TestOwner::serve(move |_| Some(reply.clone()));
            let error = connect(&owner, TIMEOUT)
                .lookup(&[lookup("request", &PROMPT)])
                .unwrap_err();
            assert!(
                error
                    .0
                    .starts_with("the shared L2 owner's reply to 'lookup' is not a reply"),
                "{error}"
            );
            assert!(error.0.contains(reason), "{error}");
        }
    }

    /// Hits that are not of the wire's shape.
    fn spoiled_hits() -> Vec<Value> {
        let Value::Array(hit) = hit_value(41, [3, 4, 8]) else {
            unreachable!()
        };
        let (_, _, keys) = prompt_keys(&PROMPT);
        let with = |index: usize, value: Value| {
            let mut hit = hit.clone();
            hit[index] = value;
            Value::Array(hit)
        };
        let hit_page = |key: Value, page: i64| {
            with(
                4,
                Value::Array(vec![Value::Array(vec![
                    key,
                    Value::from(0),
                    Value::from(page),
                ])]),
            )
        };
        let key = |index: usize, value: Value| {
            let Value::Array(mut key) = key_value(&keys[0]) else {
                unreachable!()
            };
            key[index] = value;
            Value::Array(key)
        };
        vec![
            // One element short.
            Value::Array(hit[..4].to_vec()),
            with(1, Value::from("5")),
            with(0, Value::from(true)),
            with(0, Value::from(-1)),
            with(2, Value::from(4.0)),
            with(
                3,
                Value::Array(vec![
                    Value::from(1),
                    Value::Bin(Bytes::from_static(b"tail")),
                ]),
            ),
            // A key without its page offset, a page past int32, a negative
            // group, another namespace.
            hit_page(
                Value::Array(vec![Value::from(0), Value::from(0), Value::from("1-2")]),
                3,
            ),
            hit_page(key_value(&keys[0]), 1 << 31),
            hit_page(key(1, Value::from(-1)), 3),
            hit_page(key(0, Value::from(1)), 3),
            Value::Nil,
        ]
    }

    #[test]
    fn a_hit_that_is_not_of_the_wires_shape_fails() {
        for spoiled in spoiled_hits() {
            // The second of the two hits is spoiled.
            let hits = Value::Array(vec![hit_value(41, [3, 4, 8]), spoiled.clone()]);
            let owner = TestOwner::serve(move |_| Some(ok(vec![("hits", hits.clone())])));
            let error = connect(&owner, TIMEOUT)
                .lookup(&[lookup("hit", &PROMPT), lookup("spoiled", &PROMPT)])
                .unwrap_err();
            assert!(
                error
                    .0
                    .starts_with("the shared L2 owner's reply to 'lookup' is not a reply"),
                "{spoiled:?}: {error}"
            );
        }
        // A key of another namespace is named as such.
        let other = spoiled_hits().remove(9);
        let owner =
            TestOwner::serve(move |_| Some(ok(vec![("hits", Value::Array(vec![other.clone()]))])));
        let error = connect(&owner, TIMEOUT)
            .lookup(&[lookup("request", &PROMPT)])
            .unwrap_err();
        assert!(error.0.contains("a key of namespace 1, not 0"), "{error}");
    }

    #[test]
    fn a_connection_to_no_owner_fails_within_the_timeout() {
        // A port nothing listens on.
        let free = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let address = format!("tcp://{}", free.local_addr().unwrap());
        drop(free);
        let started = std::time::Instant::now();
        let error = OwnerConnection::connect(&address, 7, Duration::from_millis(200))
            .err()
            .unwrap();
        assert!(
            error
                .0
                .starts_with("the connection to the shared L2 owner failed"),
            "{error}"
        );
        assert!(started.elapsed() < Duration::from_secs(5));
    }
}

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

//! Fixtures of the shared L2 pool's tests.

use std::{
    sync::{
        atomic::{AtomicBool, AtomicU32, Ordering},
        Arc,
    },
    thread,
    time::{Duration, Instant},
};

use bytes::Bytes;
use pyo3::{prelude::*, types::PyDict};
use ts_scheduler::{SharedL2LookupView, SharedL2Pool, StorageKeyView};
use ts_wire::{decode_bytes, encode_frames, Value};
use zeromq::{DealerSocket, Socket, SocketRecv, SocketSend, ZmqMessage};

use crate::shared_l2::{
    load::SignalWords,
    owner::{Owner, OwnerSocket},
    wire::OwnerConnection,
};

pub(crate) mod owner;

/// A pool of `num_lcm_blocks` LCM blocks as the owner's launcher builds
/// one, by the embedded interpreter: pages of two tokens, two blocks of
/// group 0 (full attention) or one of group 1 (State) to an LCM block
/// (`shared_l2_test_utils.GEOMETRY`).
pub(crate) fn build_pool(num_lcm_blocks: i32) -> Result<SharedL2Pool, String> {
    crate::init_test_scheduler();
    Python::attach(|py| {
        let scope = PyDict::new(py);
        scope.set_item("ts", ts_scheduler::python_module(py)?)?;
        scope.set_item("num_lcm_blocks", num_lcm_blocks)?;
        py.run(
            cr#"
kv = ts.CacheGroupConfig()
kv.group_id, kv.block_granularity, kv.attn_kind = "kv", 2, ts.AttnKind.Full
kv.cache_blocks_per_lcm_block = 2
state = ts.CacheGroupConfig()
state.group_id, state.block_granularity, state.attn_kind = "state", 0, ts.AttnKind.State
state.cache_blocks_per_lcm_block = 1
config = ts.SchedulerConfig()
config.prefix_granularity = 2
config.cache_groups = [kv, state]
config.host_group_packing = [2, 1]
config.checkpoint_retention_requests = 0
pool = ts.SharedL2Pool(config, num_lcm_blocks, 0)
"#,
            Some(&scope),
            None,
        )?;
        SharedL2Pool::from_python(&scope.get_item("pool")?.expect("the code sets `pool`"))
    })
    .map_err(|error: PyErr| error.to_string())
}

/// A request of the wire.
pub(crate) fn request(op: &str, fields: Vec<(&str, Value)>) -> Value {
    let mut entries = vec![(Value::from("op"), Value::from(op))];
    entries.extend(
        fields
            .into_iter()
            .map(|(name, value)| (Value::from(name), value)),
    );
    Value::Map(entries)
}

/// An owner that serves on a thread of its own until dropped.
pub(crate) struct ServedOwner {
    /// Where it listens.
    pub(crate) address: String,
    stop: Arc<AtomicBool>,
    thread: Option<thread::JoinHandle<()>>,
}

impl ServedOwner {
    /// Serve `owner` at `address`.
    pub(crate) fn start(mut owner: Owner, address: &str) -> Self {
        let mut socket = OwnerSocket::bind(address).unwrap();
        let address = socket.address().to_owned();
        let stop = Arc::new(AtomicBool::new(false));
        let stopped = stop.clone();
        let thread = thread::spawn(move || owner.serve(&mut socket, &stopped));
        Self {
            address,
            stop,
            thread: Some(thread),
        }
    }

    /// Send `frames` over a connection of its own and return the frames of
    /// the reply.
    pub(crate) fn exchange(&self, frames: Vec<Bytes>) -> Vec<Bytes> {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        runtime.block_on(async {
            let mut socket = DealerSocket::new();
            socket.connect(&self.address).await.unwrap();
            socket
                .send(ZmqMessage::try_from(frames).unwrap())
                .await
                .unwrap();
            tokio::time::timeout(Duration::from_secs(10), socket.recv())
                .await
                .expect("the owner answers")
                .unwrap()
                .into_vec()
        })
    }

    /// The owner's reply to `request`, as a client that is no rank asks it:
    /// a provider, a rank's hello, an operator.
    pub(crate) fn ask(&self, request: &Value) -> Value {
        decode_bytes(&self.exchange(encode_frames(request))[0]).unwrap()
    }
}

impl Drop for ServedOwner {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(thread) = self.thread.take() {
            // A thread that panicked has ended too.
            let _ = thread.join();
        }
    }
}

/// Return once `reached` holds; panics when it does not within 10 seconds.
pub(crate) fn wait_until(mut reached: impl FnMut() -> bool) {
    let start = Instant::now();
    while !reached() {
        assert!(
            start.elapsed() < Duration::from_secs(10),
            "the awaited state never came"
        );
        thread::sleep(Duration::from_millis(1));
    }
}

/// A rank's signal words in plain memory.
pub(crate) struct PlainSignals {
    /// The words.
    pub(crate) words: Vec<AtomicU32>,
    /// Called before each word is set.
    before_set: Box<dyn Fn() + Send + Sync>,
}

impl PlainSignals {
    /// `count` words, all zero.
    pub(crate) fn new(count: usize) -> Self {
        Self::observed(count, || ())
    }

    /// `count` words that call `before_set` before one is set.
    pub(crate) fn observed(count: usize, before_set: impl Fn() + Send + Sync + 'static) -> Self {
        Self {
            words: (0..count).map(|_| AtomicU32::new(0)).collect(),
            before_set: Box::new(before_set),
        }
    }
}

impl SignalWords for PlainSignals {
    fn num_words(&self) -> usize {
        self.words.len()
    }

    fn get(&self, word: usize) -> u32 {
        self.words[word].load(Ordering::Relaxed)
    }

    fn set(&self, word: usize, value: u32) {
        (self.before_set)();
        self.words[word].store(value, Ordering::Relaxed);
    }
}

/// A connection to a test owner as client 7.
pub(crate) fn connect(owner: &owner::TestOwner, timeout: Duration) -> OwnerConnection {
    OwnerConnection::connect(&owner.address, 7, timeout).unwrap()
}

/// The prompt the tests store and look up.
pub(crate) const PROMPT: [i32; 5] = [1, 2, 3, 4, 5];

/// A key of a cache group.
pub(crate) fn key(group_id: u32, content_hash: &str) -> StorageKeyView {
    StorageKeyView {
        group_id,
        content_hash: content_hash.to_owned(),
        page_offset: 0,
    }
}

/// A key as it travels: `[namespace_id, group_id, content_hash, page_offset]`.
pub(crate) fn key_value(key: &StorageKeyView) -> Value {
    Value::Array(vec![
        Value::from(0),
        Value::from(i64::from(key.group_id)),
        Value::from(key.content_hash.as_str()),
        Value::from(i64::from(key.page_offset)),
    ])
}

/// A prompt's page hashes (two tokens a page), its tail, and the keys of
/// its blocks: a block of group 0 per page and the block of group 1 at its
/// end (`shared_l2_test_utils.prompt_keys`).
pub(crate) fn prompt_keys(
    prompt: &[i32],
) -> (Vec<String>, Option<(i32, String)>, Vec<StorageKeyView>) {
    let joined = |end: usize| {
        prompt[..end]
            .iter()
            .map(i32::to_string)
            .collect::<Vec<_>>()
            .join("-")
    };
    let hashes: Vec<String> = (2..=prompt.len()).step_by(2).map(joined).collect();
    let tail = (prompt.len() % 2 == 1).then(|| (1, joined(prompt.len())));
    let end = tail
        .as_ref()
        .map_or_else(|| hashes.last().unwrap().clone(), |(_, hash)| hash.clone());
    let mut keys: Vec<_> = hashes.iter().map(|hash| key(0, hash)).collect();
    keys.push(key(1, &end));
    (hashes, tail, keys)
}

/// The lookup of `prompt` for the request `request_id`.
pub(crate) fn lookup(request_id: &str, prompt: &[i32]) -> SharedL2LookupView {
    SharedL2LookupView {
        request_id: request_id.to_owned(),
        prefix_hashes: prompt_keys(prompt).0,
        tokens: prompt.to_vec(),
        max_tokens: prompt.len() as i32,
    }
}

/// A hit on `PROMPT`'s blocks at `pages`, as the owner answers it.
pub(crate) fn hit_value(ticket: i64, pages: [i64; 3]) -> Value {
    let (_, tail, keys) = prompt_keys(&PROMPT);
    let (tail_tokens, tail_hash) = tail.unwrap();
    Value::Array(vec![
        Value::from(ticket),
        Value::from(5),
        Value::from(5),
        Value::Array(vec![
            Value::from(i64::from(tail_tokens)),
            Value::from(tail_hash.as_str()),
        ]),
        Value::Array(
            keys.iter()
                .zip(pages)
                .map(|(key, page)| {
                    Value::Array(vec![key_value(key), Value::from(0), Value::from(page)])
                })
                .collect(),
        ),
    ])
}

/// A miss, as the owner answers it.
pub(crate) fn miss_value() -> Value {
    Value::Array(vec![
        Value::from(0),
        Value::from(0),
        Value::from(0),
        Value::Nil,
        Value::Array(Vec::new()),
    ])
}

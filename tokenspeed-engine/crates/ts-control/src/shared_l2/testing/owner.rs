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

//! An owner for tests: a ROUTER socket that answers every request with what
//! the test's closure returns for it, as the Python owner's socket does.
//!
//! The file names nothing of its crate: the host crate's tests include it by
//! path.

use std::sync::Arc;

use parking_lot::Mutex;
use tokio::runtime::Runtime;
use ts_wire::{decode_bytes, encode_frames, msgpack::Integer, Value};
use zeromq::{RouterSocket, Socket, SocketRecv, SocketSend, ZmqMessage};

/// A reply that says `ok` with `fields`.
pub(crate) fn ok(fields: Vec<(&str, Value)>) -> Value {
    let mut entries = vec![(Value::from("ok"), Value::from(true))];
    entries.extend(
        fields
            .into_iter()
            .map(|(name, value)| (Value::from(name), value)),
    );
    Value::Map(entries)
}

/// The reply of an owner that refuses a request.
pub(crate) fn refused(reason: &str) -> Value {
    Value::Map(vec![
        (Value::from("ok"), Value::from(false)),
        (Value::from("error"), Value::from(reason)),
    ])
}

/// An array of integers.
pub(crate) fn ints(values: &[i64]) -> Value {
    Value::Array(values.iter().map(|value| Value::from(*value)).collect())
}

/// The field `name` of a request.
pub(crate) fn field<'v>(request: &'v Value, name: &str) -> &'v Value {
    let Value::Map(entries) = request else {
        panic!("a request is a map: {request:?}");
    };
    entries
        .iter()
        .find(|(key, _)| *key == Value::from(name))
        .map(|(_, value)| value)
        .unwrap_or_else(|| panic!("the request has no {name}: {request:?}"))
}

/// The operation of a request.
pub(crate) fn op(request: &Value) -> &str {
    match field(request, "op") {
        Value::Str(op) => op,
        other => panic!("an operation is a string: {other:?}"),
    }
}

/// An unsigned integer of a request.
pub(crate) fn uint(value: &Value) -> u64 {
    match value {
        Value::Int(Integer::Unsigned(value)) => *value,
        other => panic!("not an unsigned integer: {other:?}"),
    }
}

/// The owner, serving until dropped.
pub(crate) struct TestOwner {
    /// Where it listens.
    pub(crate) address: String,
    requests: Arc<Mutex<Vec<Value>>>,
    _runtime: Runtime,
}

impl TestOwner {
    /// Serve `answer`: the reply to each request, or `None` to leave it
    /// unanswered.
    pub(crate) fn serve(answer: impl FnMut(&Value) -> Option<Value> + Send + 'static) -> Self {
        Self::serve_from(answer, 0)
    }

    /// Serve `answer`, but send the reply to the first request only when
    /// the second request has come, right before that one's reply.
    pub(crate) fn serve_first_reply_late(
        answer: impl FnMut(&Value) -> Option<Value> + Send + 'static,
    ) -> Self {
        Self::serve_from(answer, 1)
    }

    /// Serve `answer`, holding back the replies to the first `late`
    /// requests until the next request has come.
    fn serve_from(
        mut answer: impl FnMut(&Value) -> Option<Value> + Send + 'static,
        mut late: usize,
    ) -> Self {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .enable_all()
            .build()
            .unwrap();
        let requests = Arc::new(Mutex::new(Vec::new()));
        let received = requests.clone();
        let (mut socket, address) = runtime.block_on(async {
            let mut socket = RouterSocket::new();
            let endpoint = socket.bind("tcp://127.0.0.1:0").await.unwrap();
            (socket, endpoint.to_string())
        });
        runtime.spawn(async move {
            let mut held = Vec::new();
            while let Ok(message) = socket.recv().await {
                // The identity and the correlation frame go back
                // unchanged, as from the Python owner.
                let mut frames = message.into_vec();
                let payload = frames.pop().unwrap();
                let request: Value = decode_bytes(&payload).unwrap();
                received.lock().push(request.clone());
                let Some(reply) = answer(&request) else {
                    continue;
                };
                frames.extend(encode_frames(&reply));
                if late > 0 {
                    late -= 1;
                    held.push(frames);
                    continue;
                }
                for frames in held.drain(..).chain([frames]) {
                    socket
                        .send(ZmqMessage::try_from(frames).unwrap())
                        .await
                        .unwrap();
                }
            }
        });
        Self {
            address,
            requests,
            _runtime: runtime,
        }
    }

    /// Every request received so far.
    pub(crate) fn requests(&self) -> Vec<Value> {
        self.requests.lock().clone()
    }

    /// The operations of the requests received so far.
    pub(crate) fn ops(&self) -> Vec<String> {
        self.requests()
            .iter()
            .map(|request| op(request).to_owned())
            .collect()
    }
}

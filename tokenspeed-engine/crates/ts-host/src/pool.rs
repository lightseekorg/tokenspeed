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

//! A shared L2 pool's client as the loop drives it: the rank's Python client
//! of the pool's owner (`SharedL2Client` in `cache/shared_l2/client.py`),
//! called by its own method names.
//!
//! The control thread makes every call, taking the GIL for the call. The
//! client's `lookup` sends one request to the owner and waits for the reply,
//! without the GIL; its other calls hand work to the client's threads and
//! read what they finished. A call that raises is the client's failure.

use pyo3::{prelude::*, types::PyBytes};
use ts_control::shared_l2::{PoolClient, PoolError};
use ts_scheduler::{
    profile, SharedL2AnswerView, SharedL2LookupView, SharedL2OpKind, SharedL2OpView,
    SharedL2PageView, StorageKeyView,
};

use crate::data_plane::describe;

/// A pool key as the owner's wire carries it: namespace, group, content
/// hash, page offset.
type PyKey = (u32, u32, String, i32);

/// The namespace of every pool key. The scheduler's keys do not carry it: a
/// pool holds the default namespace alone.
const NAMESPACE: u32 = 0;

/// A partial last page: its tokens and its hash.
type PyTail = (i32, String);

/// A lookup's answer as the client returns it: the request it answers, the
/// ticket, the hit's tokens, the pool's KV reach, the tail, and each key
/// with its bucket and pool page.
type PyAnswer = (
    String,
    u64,
    i32,
    i32,
    Option<PyTail>,
    Vec<(PyKey, i32, i32)>,
);

/// Tokens as the pool carries them: int32, native byte order. Written in one
/// pass into the bytes object, which only exists under the GIL: a prompt's
/// tokens cost one copy on the control thread.
fn token_bytes<'py>(py: Python<'py>, tokens: &[i32]) -> PyResult<Bound<'py, PyBytes>> {
    PyBytes::new_with(py, std::mem::size_of_val(tokens), |bytes| {
        for (bytes, token) in bytes.chunks_exact_mut(4).zip(tokens) {
            bytes.copy_from_slice(&token.to_ne_bytes());
        }
        Ok(())
    })
}

/// The Python client, called from the control thread.
pub struct PythonPoolClient {
    client: Py<PyAny>,
}

impl PythonPoolClient {
    /// The client behind a Python `SharedL2Client`.
    pub fn new(client: Py<PyAny>) -> Self {
        Self { client }
    }

    /// Run one client call under the GIL; its failure is the client's.
    fn call<T>(&self, call: impl FnOnce(&Bound<'_, PyAny>) -> PyResult<T>) -> Result<T, PoolError> {
        // The wait for the GIL, which the forward thread's Python holds.
        let acquiring = profile::span("Pool::acquire_gil");
        Python::attach(|py| {
            drop(acquiring);
            call(self.client.bind(py)).map_err(|error| PoolError(describe(py, &error)))
        })
    }
}

impl PoolClient for PythonPoolClient {
    fn lookup(
        &mut self,
        lookups: &[SharedL2LookupView],
    ) -> Result<Vec<SharedL2AnswerView>, PoolError> {
        let answers: Vec<PyAnswer> = self.call(|client| {
            let py = client.py();
            let lookups = lookups
                .iter()
                .map(|lookup| {
                    Ok((
                        &lookup.request_id,
                        &lookup.prefix_hashes,
                        token_bytes(py, &lookup.tokens)?,
                        lookup.max_tokens,
                    ))
                })
                .collect::<PyResult<Vec<_>>>()?;
            client.call_method1("lookup", (lookups,))?.extract()
        })?;
        answers
            .into_iter()
            .map(|(request_id, ticket, num_tokens, kv_tokens, tail, slots)| {
                let (tail_tokens, tail_hash) = tail.unwrap_or_default();
                Ok(SharedL2AnswerView {
                    request_id,
                    ticket,
                    num_tokens,
                    kv_tokens,
                    tail_tokens,
                    tail_hash,
                    buckets: slots.iter().map(|(_, bucket, _)| *bucket).collect(),
                    pages: slots.iter().map(|(_, _, page)| *page).collect(),
                    keys: slots
                        .into_iter()
                        .map(|((namespace, group_id, content_hash, page_offset), _, _)| {
                            if namespace != NAMESPACE {
                                return Err(PoolError(format!(
                                    "the pool answered a key of namespace {namespace}, not \
                                     {NAMESPACE}"
                                )));
                            }
                            Ok(StorageKeyView {
                                group_id,
                                content_hash,
                                page_offset,
                            })
                        })
                        .collect::<Result<_, _>>()?,
                })
            })
            .collect()
    }

    fn poll_writes(&mut self) -> Result<Vec<u32>, PoolError> {
        self.call(|client| client.call_method0("poll_writes")?.extract())
    }

    fn map_proxy_pages(&mut self, pages: &[SharedL2PageView]) -> Result<(), PoolError> {
        self.call(|client| {
            let pages: Vec<_> = pages
                .iter()
                .map(|page| (page.group_id, page.block, page.bucket, page.page))
                .collect();
            client.call_method1("map_proxy_pages", (pages,)).map(drop)
        })
    }

    fn submit(&mut self, commands: &[SharedL2OpView]) -> Result<(), PoolError> {
        self.call(|client| {
            let py = client.py();
            let (mut writes, mut releases, mut touches) = (Vec::new(), Vec::new(), Vec::new());
            for command in commands {
                let keys = || -> Vec<PyKey> {
                    command
                        .keys
                        .iter()
                        .map(|key| {
                            (
                                NAMESPACE,
                                key.group_id,
                                key.content_hash.clone(),
                                key.page_offset,
                            )
                        })
                        .collect()
                };
                match command.kind {
                    SharedL2OpKind::Write => {
                        let publications = command
                            .publications
                            .iter()
                            .map(|publication| {
                                Ok((
                                    token_bytes(py, &publication.tokens)?,
                                    &publication.prefix_hashes,
                                    (publication.tail_tokens != 0).then(|| -> PyTail {
                                        (publication.tail_tokens, publication.tail_hash.clone())
                                    }),
                                    publication.publisher,
                                ))
                            })
                            .collect::<PyResult<Vec<_>>>()?;
                        writes.push((
                            command.id,
                            keys(),
                            &command.buckets,
                            &command.blocks,
                            publications,
                        ));
                    }
                    SharedL2OpKind::Release => releases.push(command.id),
                    SharedL2OpKind::Touch => touches.push(keys()),
                    kind => {
                        return Err(pyo3::exceptions::PyValueError::new_err(format!(
                            "no client call for the pool command {kind:?}"
                        )))
                    }
                }
            }
            client
                .call_method1("submit", (writes, releases, touches))
                .map(drop)
        })
    }

    fn check_failure(&mut self) -> Result<(), PoolError> {
        self.call(|client| client.call_method0("check_failure").map(drop))
    }
}

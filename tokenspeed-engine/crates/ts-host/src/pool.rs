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

//! A shared L2 pool's client as the loop drives it: the Python adapter
//! (`HostSharedL2` in `engine/rust_host.py`) over the rank's client of the
//! pool's owner.
//!
//! The control thread makes every call, taking the GIL for the call. The
//! adapter only hands work to the client's threads and reads what they
//! finished; with `awaited` lookups its poll waits, without the GIL.

use std::time::Duration;

use pyo3::{prelude::*, types::PyBytes};
use ts_control::shared_l2::{ClientStats, Polled, PoolClient, PoolError};
use ts_scheduler::{
    profile, SharedL2AnswerView, SharedL2KeyView, SharedL2LookupView, SharedL2OpKind,
    SharedL2OpView, SharedL2PageView, SharedL2WriteDoneView,
};

use crate::data_plane::describe;

/// A pool key as the owner's wire carries it: namespace, group, content
/// hash, page offset.
type PyKey = (u32, u32, String, i32);

/// A partial last page: its tokens and its hash.
type PyTail = (i32, String);

/// A lookup's answer as the adapter returns it: the lookup, the ticket, the
/// hit's tokens, the pool's KV reach, the tail, and each key with its
/// bucket and pool page.
type PyAnswer = (u64, u64, i32, i32, Option<PyTail>, Vec<(PyKey, i32, i32)>);

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

/// The Python adapter, called from the control thread.
pub struct PythonPoolClient {
    adapter: Py<PyAny>,
}

impl PythonPoolClient {
    /// The client behind a Python `HostSharedL2`.
    pub fn new(adapter: Py<PyAny>) -> Self {
        Self { adapter }
    }

    /// Run one adapter call under the GIL; its failure is the client's.
    fn call<T>(&self, call: impl FnOnce(&Bound<'_, PyAny>) -> PyResult<T>) -> Result<T, PoolError> {
        // The wait for the GIL, which the forward thread's Python holds.
        let acquiring = profile::span("Pool::acquire_gil");
        Python::attach(|py| {
            drop(acquiring);
            call(self.adapter.bind(py)).map_err(|error| PoolError(describe(py, &error)))
        })
    }
}

impl PoolClient for PythonPoolClient {
    fn submit_lookups(&mut self, lookups: &[SharedL2LookupView]) -> Result<(), PoolError> {
        self.call(|adapter| {
            let py = adapter.py();
            let lookups = lookups
                .iter()
                .map(|lookup| {
                    Ok((
                        lookup.lookup_id,
                        &lookup.request_id,
                        &lookup.prefix_hashes,
                        token_bytes(py, &lookup.tokens)?,
                        lookup.max_tokens,
                    ))
                })
                .collect::<PyResult<Vec<_>>>()?;
            adapter.call_method1("submit_lookups", (lookups,)).map(drop)
        })
    }

    fn poll(
        &mut self,
        awaited: &[u64],
        wait: Duration,
        tickets: &[u64],
    ) -> Result<Polled, PoolError> {
        let (answers, writes, lease_lost): (Vec<PyAnswer>, Vec<(u64, bool)>, bool) =
            self.call(|adapter| {
                adapter
                    .call_method1(
                        "poll",
                        (awaited.to_vec(), wait.as_secs_f64(), tickets.to_vec()),
                    )?
                    .extract()
            })?;
        Ok(Polled {
            answers: answers
                .into_iter()
                .map(|(lookup_id, ticket, num_tokens, kv_tokens, tail, slots)| {
                    let (tail_tokens, tail_hash) = tail.unwrap_or_default();
                    SharedL2AnswerView {
                        lookup_id,
                        ticket,
                        num_tokens,
                        kv_tokens,
                        tail_tokens,
                        tail_hash,
                        buckets: slots.iter().map(|(_, bucket, _)| *bucket).collect(),
                        pages: slots.iter().map(|(_, _, page)| *page).collect(),
                        keys: slots
                            .into_iter()
                            .map(
                                |((namespace_id, group_id, content_hash, page_offset), _, _)| {
                                    SharedL2KeyView {
                                        namespace_id,
                                        group_id,
                                        content_hash,
                                        page_offset,
                                    }
                                },
                            )
                            .collect(),
                    }
                })
                .collect(),
            writes: writes
                .into_iter()
                .map(|(write_id, success)| SharedL2WriteDoneView { write_id, success })
                .collect(),
            lease_lost,
        })
    }

    fn map_pages(&mut self, ticket: u64, pages: &[SharedL2PageView]) -> Result<(), PoolError> {
        self.call(|adapter| {
            let pages: Vec<_> = pages
                .iter()
                .map(|page| (page.group_id, page.block, page.bucket, page.page))
                .collect();
            adapter.call_method1("map_pages", (ticket, pages)).map(drop)
        })
    }

    fn submit(&mut self, commands: &[SharedL2OpView]) -> Result<(), PoolError> {
        self.call(|adapter| {
            let py = adapter.py();
            let (mut writes, mut releases, mut touches, mut abandons) =
                (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            for command in commands {
                let keys = || -> Vec<PyKey> {
                    command
                        .keys
                        .iter()
                        .map(|key| {
                            (
                                key.namespace_id,
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
                    SharedL2OpKind::Abandon => abandons.push(command.id),
                    kind => {
                        return Err(pyo3::exceptions::PyValueError::new_err(format!(
                            "no adapter call for the pool command {kind:?}"
                        )))
                    }
                }
            }
            adapter
                .call_method1("submit", (writes, releases, touches, abandons))
                .map(drop)
        })
    }

    fn stats(&mut self) -> Result<ClientStats, PoolError> {
        let (lookup_timeouts, dropped_write_blocks, no_pool) =
            self.call(|adapter| adapter.call_method0("stats")?.extract())?;
        Ok(ClientStats {
            lookup_timeouts,
            dropped_write_blocks,
            no_pool,
        })
    }
}

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

//! The L3 storage tier's Host-only executor, owned by the engine host
//! (`docs/design/async-l3-cache.md`).
//!
//! The C++ scheduler plans each request's storage operations and the
//! control plane agrees on their outcomes across the replica
//! (`ts-control`'s `l3`); this crate runs the operations on this rank:
//! bounded queues, worker threads, the backend's health, and the stores —
//! the Mooncake store client, and an in-memory one for tests. Workers read
//! and write the Host pool the data plane allocated, by address; no thread
//! here takes the GIL, calls the scheduler or touches the GPU.

pub mod executor;
pub mod keys;
pub mod layout;
pub mod mooncake;
pub mod settings;
pub mod store;

pub use executor::{
    monotonic_clock, Clock, Executor, ExecutorConfig, ExecutorError, Namespace, Operation,
    StorageCompletion, StorageKey, StoragePoll, StoragePressure,
};
pub use layout::{GroupLayout, HostLayout, LayoutError, Range};
pub use settings::{Settings, StoreSettings};
pub use store::{process_memory_store, HostBuffer, IoResult, MemoryStore, Store, StoreError};

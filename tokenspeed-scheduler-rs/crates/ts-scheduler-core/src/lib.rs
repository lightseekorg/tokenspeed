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

//! TokenSpeed scheduler core — Rust port of `tokenspeed-scheduler/csrc`.
//!
//! The scheduler is a single-threaded control plane; every module in this
//! crate is unsafe-free by construction (`#![forbid(unsafe_code)]`). Shared
//! ownership between the scheduler and long-lived handles uses
//! `Rc<RefCell<..>>` / `Arc` (see `docs/design/rust-port.md` §5).

#![forbid(unsafe_code)]

pub mod acquire_plan;
pub mod block_pool;
pub mod block_table;
pub mod cache_block_ref;
pub mod cache_config;
pub mod cache_coordinator;
pub mod cache_group;
pub mod cache_ops;
pub mod cache_types;
pub mod events;
pub mod fsm;
pub mod group_allocator;
pub mod group_geometry;
pub mod kv_events;
pub mod prefix_hasher;
pub mod prefix_index;
pub mod prefix_matcher;
pub mod req_pool_allocator;
pub mod request;
pub mod request_spec;
pub mod scheduler;
pub mod tier;
pub mod token_container;
pub mod types;

pub use block_pool::BlockPool;
pub use block_table::{block_table_lcm_block_ids, BlockTable};
pub use cache_block_ref::{CacheBlock, CacheBlockLocation, CacheBlockRef};
pub use cache_config::{CacheGroupConfig, CacheGroupFamily, CacheTransferPolicy, Retention};
pub use cache_coordinator::{
    make_coordinator, AdmissionResult, CacheCoordinator, CacheMutation, CacheTier, PrefixProbe,
    StoreCandidate,
};
pub use cache_ops::{
    align_prefill_chunk, build_block_tables, free_request, make_specs_from_config,
};
pub use cache_types::{
    AttnKind, CacheBoundaryKind, CacheGroupSpec, CacheKey, ContentHash, GroupDemand,
    GroupPrefixProbe, PrefixMatch, DEFAULT_CACHE_NAMESPACE_ID,
};
pub use group_allocator::GroupAllocator;
pub use group_geometry::GroupGeometry;
pub use prefix_index::PrefixCacheIndex;
pub use prefix_matcher::{FullAttnMatcher, PrefixMatcher, SwaMatcher};
pub use req_pool_allocator::{allocate as allocate_req_pool_slot, ReqPoolAllocator, ReqPoolIndex};
pub use request::Request;
pub use scheduler::Scheduler;
pub use token_container::{TokenContainer, Window};
pub use types::{AllocatorConfig, Role, SchedulerConfig};

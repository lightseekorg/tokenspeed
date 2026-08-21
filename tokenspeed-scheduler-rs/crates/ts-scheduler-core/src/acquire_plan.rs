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

//! Token-free placement order for `GroupAllocator`.
//!
//! Ported from `tokenspeed-scheduler/csrc/cache/core/acquire_plan.h`. All
//! token -> page arithmetic happens on the logical side (`GroupGeometry`); the
//! allocator only executes block counts and bookkeeping values.

/// A token-free placement order for the group allocator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AcquirePlan {
    /// Fresh blocks to acquire from the pool.
    pub num_blocks: i32,
    /// Dense append when < 0. Otherwise the logical slot the sparse suffix
    /// starts at; slots below it stay null holes.
    pub suffix_start: i32,
    /// Sparse only: the table's logical block count after placement.
    pub table_blocks_after: i32,
    /// Unconsumed tail capacity after placement, in tokens. The allocator
    /// stores it verbatim; it never derives it.
    pub available_tokens_after: i32,
}

impl Default for AcquirePlan {
    fn default() -> Self {
        Self {
            num_blocks: 0,
            suffix_start: -1,
            table_blocks_after: 0,
            available_tokens_after: 0,
        }
    }
}

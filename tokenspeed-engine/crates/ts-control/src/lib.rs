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

//! The engine's control plane for one rank: admitting requests, the
//! scheduling loop, committing forward results, the control operations and
//! the disaggregated transfers' bookkeeping; and an EPD encode rank's loop
//! ([`encode`]).

#![forbid(unsafe_code)]

pub mod batch_log;
pub mod cache_ops;
mod codec;
pub mod control;
pub mod encode;
pub mod eplb;
pub mod event_loop;
pub mod grammar;
pub mod ingress;
pub mod kv_events;
pub mod l3;
pub mod output;
pub mod relay;
pub mod request;
pub mod request_stats;
pub mod sampling;
pub mod transport;

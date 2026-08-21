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

//! Public request specification types.
//!
//! Ported from `tokenspeed-scheduler/csrc/scheduler/request_spec.h`.

/// Inbound request as submitted by the runtime.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RequestSpec {
    pub request_id: String,
    pub tokens: Vec<i32>,
    pub max_new_tokens: i32,
}

/// The portion of a prefill scheduled for the next forward step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefillInfo {
    pub input_ids: Vec<i32>,
    pub shifted_input_ids: Vec<i32>,
    pub already_scheduled_len: i32,
    pub extend_len: i32,
}

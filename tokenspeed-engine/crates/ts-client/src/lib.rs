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

//! A Rust frontend's connection to TokenSpeed engines.
//!
//! The client binds the handshake, input and output sockets; the engines
//! (one per attention-DP rank) connect in and register. The client then routes
//! tokenized requests to engines and streams each request's outputs back.
//! It does not tokenize or detokenize: the frontend owns both, and so also
//! matches stop strings itself. The engine refuses requests that carry stop
//! strings on this wire; send stop token ids.
//!
//! ```no_run
//! # async fn run() -> ts_client::Result<()> {
//! use std::time::Duration;
//! use ts_client::{ClientConfig, EngineClient, LoadBalance, Routing};
//! use ts_wire::{SamplingParams, TokenizedGenerateReqInput};
//!
//! let client = EngineClient::connect(ClientConfig {
//!     handshake_address: "tcp://127.0.0.1:30500".into(),
//!     input_address: "tcp://127.0.0.1:0".into(),
//!     output_address: "tcp://127.0.0.1:0".into(),
//!     first_engine_index: 0,
//!     num_engines: 1,
//!     routing: Routing::Aggregated(LoadBalance::LeastInFlight),
//!     handshake_timeout: Duration::from_secs(600),
//!     send_timeout: Duration::from_secs(5),
//! })
//! .await?;
//! let mut stream = client
//!     .generate(TokenizedGenerateReqInput {
//!         input_ids: Some(vec![1, 2, 3]),
//!         stream: true,
//!         ..TokenizedGenerateReqInput::new("r1", SamplingParams::default())
//!     })
//!     .await?;
//! while let Some(chunk) = stream.next().await {
//!     println!("{:?}", chunk.expect("engine alive").output_ids);
//! }
//! # Ok(())
//! # }
//! ```

#![forbid(unsafe_code)]

mod client;
mod config;
mod error;
mod handshake;

pub use client::{ClientListener, EngineClient, EngineLoad, OutputChunk, RequestStream};
pub use config::{ClientConfig, LoadBalance, Routing};
pub use error::{Error, Result, StreamError};
pub use handshake::EngineInfo;

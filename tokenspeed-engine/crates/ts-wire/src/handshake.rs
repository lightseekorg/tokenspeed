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

//! The startup handshake (`engine/zmq_wire.py`).
//!
//! The frontend binds a ROUTER handshake socket, a ROUTER input socket and a
//! PULL output socket. Each engine (one per attention-DP rank) connects with
//! its two-byte little-endian engine index as its identity and:
//!
//! 1. sends HELLO ([`ReadyMessage`]) on the handshake socket;
//! 2. receives INIT ([`HandshakeInitMessage`]) with the input and output
//!    addresses;
//! 3. sends READY on the handshake socket;
//! 4. connects its input DEALER and registers on it with
//!    [`EngineCoreReadyResponse`];
//! 5. connects its output PUSH.
//!
//! These structs use the map layout, matching the frontend's named codec.

use crate::{codec::Dict, value::Value};

/// HELLO status.
pub const STATUS_HELLO: &str = "HELLO";
/// READY status.
pub const STATUS_READY: &str = "READY";

wire_struct! {
    /// Engine to frontend handshake status (HELLO or READY).
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct ReadyMessage: map + Default {
        /// `HELLO` or `READY`.
        pub status: Option<String> = None,
        /// The engine runs on the frontend's host.
        pub local: Option<bool> = None,
        /// The engine runs without an in-process frontend.
        pub headless: Option<bool> = None,
        /// Hash of the engine's parallel configuration.
        pub parallel_config_hash: Option<String> = None,
    }
}

wire_struct! {
    /// Frontend-owned socket addresses sent in INIT.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct HandshakeAddresses: map + Default {
        /// Input ROUTER addresses; engines use the first.
        pub inputs: Vec<String> = Vec::new(),
        /// Output PULL addresses; engines use the first.
        pub outputs: Vec<String> = Vec::new(),
        /// DP coordinator input, unused by TokenSpeed.
        pub coordinator_input: Option<String> = None,
        /// DP coordinator output, unused by TokenSpeed.
        pub coordinator_output: Option<String> = None,
        /// Frontend stats publisher, unused by TokenSpeed.
        pub frontend_stats_publish_address: Option<String> = None,
    }
}

wire_struct! {
    /// Frontend to engine INIT, the reply to HELLO.
    #[derive(Debug, Clone, PartialEq)]
    pub struct HandshakeInitMessage: map {
        /// Where to connect.
        pub addresses: HandshakeAddresses,
        /// Opaque to the engine.
        pub parallel_config: Dict<Value, Value> = Dict::default(),
    }
}

wire_struct! {
    /// The engine's configuration, sent when it registers on the input socket.
    ///
    /// Field names, not order, are the contract; some names are the
    /// frontend's vocabulary rather than TokenSpeed's.
    #[derive(Debug, Clone, PartialEq)]
    pub struct EngineCoreReadyResponse: map + Default {
        /// Longest supported sequence.
        pub max_model_len: i64 = 0,
        /// KV pages on the device.
        pub num_gpu_blocks: i64 = 0,
        /// Prefix-cache matching granularity in tokens.
        pub prefix_granularity: i64 = 0,
        /// DP stats address, unused by TokenSpeed.
        pub dp_stats_address: Option<String> = None,
        /// Model dtype: `bfloat16`, `float16` or `float32`.
        pub dtype: String = "bfloat16".to_owned(),
        /// Multimodal encoder dtype, if any.
        pub multimodal_encoder_dtype: Option<String> = None,
        /// `tokenspeed-<version>`; the frontend's field name.
        pub vllm_version: String = String::new(),
        /// Ranks in the engine.
        pub world_size: i64 = 1,
        /// Attention-DP ranks.
        pub data_parallel_size: i64 = 1,
        /// Attention-TP ranks.
        pub tensor_parallel_size: i64 = 1,
        /// Pipeline stages.
        pub pipeline_parallel_size: i64 = 1,
        /// KV parallelism (`--kv-parallel-size`); the frontend's field name.
        pub decode_context_parallel_size: i64 = 1,
        /// This engine's attention-DP rank.
        pub data_parallel_rank: i64 = 0,
        /// Running-request limit.
        pub max_num_seqs: i64 = 0,
        /// Tokens per step limit; `0` is no cap.
        pub max_num_batched_tokens: i64 = 0,
        /// Served model name.
        pub instance_id: String = String::new(),
        /// KV cache capacity in tokens.
        pub kv_cache_size_tokens: Option<i64> = None,
        /// KV cache capacity in maximum-length requests.
        pub kv_cache_max_concurrency: Option<f64> = None,
        /// KV event publisher configuration.
        pub kv_events_config: Option<Dict<Value, Value>> = None,
    }
}

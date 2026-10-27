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
//!
//! The registration also declares what the engine's wire speaks
//! ([`WireCapabilities`]): a frontend reads that one key set at start, for
//! the whole connection, instead of inferring a revision from the frames it
//! later sees. An engine that sends no `capabilities` predates the
//! declaration.

use crate::{
    codec::Dict,
    frames::REQUEST_TYPES,
    output::{BatchTokenIDOutSlim, FINISH_DETAIL_KEYS},
    value::Value,
};

/// HELLO status.
pub const STATUS_HELLO: &str = "HELLO";
/// READY status.
pub const STATUS_READY: &str = "READY";
/// The revision of the wire this crate speaks, as [`WireCapabilities`]
/// declares it. Revision 1: tagged control operations on the CONTROL
/// message, the slim output with its finish-details column and its prompt,
/// top-k and speculation columns, and the encode node's request intake. A
/// field appended with a default keeps the revision; a reordered or removed
/// field moves it.
pub const WIRE_REVISION: i64 = 1;

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
        /// What the engine's wire speaks; `None` from an engine that
        /// predates the declaration.
        pub capabilities: Option<WireCapabilities> = None,
    }
}

wire_struct! {
    /// What an engine's wire speaks, declared once in its registration.
    ///
    /// The engine side of the wire fills it from the schema it was built
    /// with ([`WireCapabilities::declared`]), so the frontend compares the
    /// declaration with its own schema at start and names a skew before the
    /// first request.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct WireCapabilities: map + Default {
        /// [`WIRE_REVISION`] of the engine's wire; `0` is undeclared.
        pub wire_revision: i64 = 0,
        /// The input message types the engine takes, by name
        /// ([`REQUEST_TYPES`]).
        pub request_types: Vec<String> = Vec::new(),
        /// The slim output's columns in wire order
        /// ([`BatchTokenIDOutSlim::FIELDS`]).
        pub output_columns: Vec<String> = Vec::new(),
        /// The keys a finish-details map may carry ([`FINISH_DETAIL_KEYS`]).
        pub finish_detail_keys: Vec<String> = Vec::new(),
        /// Most top-k alternatives a request may ask for per position; `0`
        /// when the engine serves none.
        pub max_top_logprobs: i64 = 0,
    }
}

impl WireCapabilities {
    /// The capabilities of the wire this crate speaks, with the engine's
    /// own top-k limit.
    pub fn declared(max_top_logprobs: i64) -> Self {
        let names = |names: &[&str]| names.iter().map(|name| (*name).to_owned()).collect();
        Self {
            wire_revision: WIRE_REVISION,
            request_types: names(REQUEST_TYPES),
            output_columns: names(BatchTokenIDOutSlim::FIELDS),
            finish_detail_keys: names(FINISH_DETAIL_KEYS),
            max_top_logprobs,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::{decode_bytes, encode_frames};

    #[test]
    fn the_declaration_names_this_crates_schema() {
        let declared = WireCapabilities::declared(20);
        assert_eq!(declared.wire_revision, 1);
        assert_eq!(declared.request_types, ["add", "abort", "control"]);
        assert_eq!(declared.output_columns.len(), 22);
        assert_eq!(declared.output_columns[..2], ["rids", "output_ids"]);
        assert_eq!(declared.output_columns[13], "finish_reason_details");
        assert_eq!(
            declared.output_columns[20..],
            ["output_top_logprobs_val", "output_top_logprobs_idx"]
        );
        assert_eq!(declared.finish_detail_keys.len(), 6);
        assert_eq!(declared.max_top_logprobs, 20);
    }

    #[test]
    fn a_registration_without_the_declaration_reads_as_undeclared() {
        // An older engine's registration carries no `capabilities` key.
        let mut encoder = crate::msgpack::Encoder::new();
        encoder.map_len(1);
        encoder.str("max_model_len");
        encoder.int(7);
        let older: EngineCoreReadyResponse = decode_bytes(&encoder.into_single_frame()).unwrap();
        assert_eq!(older.max_model_len, 7);
        assert_eq!(older.capabilities, None);

        let declared = EngineCoreReadyResponse {
            capabilities: Some(WireCapabilities::declared(20)),
            ..Default::default()
        };
        let frames = encode_frames(&declared);
        assert_eq!(
            decode_bytes::<EngineCoreReadyResponse>(&frames[0]).unwrap(),
            declared
        );
    }
}

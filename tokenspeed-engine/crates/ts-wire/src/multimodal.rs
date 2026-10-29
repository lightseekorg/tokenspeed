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

//! Multimodal payloads (`multimodal/inputs.py`, `multimodal/shm_transport.py`),
//! and the EPD encode request that carries an item to an encode node
//! (`epd/encode_worker.py`).

use crate::{
    codec::{Dict, StrOrList, Wire},
    error::{Error, Result},
    msgpack::{Decoder, Encoder},
    tensor::Tensor,
    value::Value,
};

/// The media kind of an item. Encoded as the Python enum's value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Modality {
    /// Images.
    Image = 1,
    /// Video.
    Video = 2,
    /// Audio.
    Audio = 3,
}

impl Wire for Modality {
    fn encode(&self, encoder: &mut Encoder) {
        encoder.uint(*self as u64);
    }

    fn decode(decoder: &mut Decoder<'_>) -> Result<Self> {
        match decoder.integer()?.narrow::<i64>("Modality")? {
            1 => Ok(Modality::Image),
            2 => Ok(Modality::Video),
            3 => Ok(Modality::Audio),
            other => Err(Error::Invalid(format!("unknown modality {other}"))),
        }
    }
}

wire_struct! {
    /// A CPU tensor published in a POSIX shared-memory segment on the same
    /// host. A map-layout struct.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct ShmTensorHandle: map {
        /// Segment name.
        pub shm_name: String,
        /// Tensor dimensions.
        pub shape: Vec<i64>,
        /// torch dtype name.
        pub dtype: String,
    }
}

wire_struct! {
    /// One image, video or audio item of a request.
    #[derive(Debug, Clone, PartialEq)]
    pub struct MultimodalDataItem: array {
        /// Media kind.
        pub modality: Modality,
        /// Content id used for prefix matching; up to 64 bits.
        pub hash: Option<u64> = None,
        /// The placeholder token id substituted into the prompt.
        pub pad_value: Option<i64> = None,
        /// `(start, end)` token ranges of the item in the prompt.
        pub offsets: Option<Vec<(i64, i64)>> = None,
        /// Inline feature tensor. Exactly one of `feature` and `feature_shm`
        /// is set in flight.
        pub feature: Option<Tensor> = None,
        /// Feature tensor in shared memory.
        pub feature_shm: Option<ShmTensorHandle> = None,
        /// Model-specific tensors, such as image grid sizes.
        pub model_specific_data: Dict<String, Tensor> = Dict::default(),
        /// Engine-local encoder output; always `None` on the wire.
        pub encoded: Option<Tensor> = None,
        /// Engine-local deepstack encoder output; always `None` on the wire.
        pub encoded_deepstack: Option<Tensor> = None,
        /// EPD: the encode worker's rendezvous for this item's embedding.
        pub encode_handshake: Option<Dict<Value, Value>> = None,
    }
}

wire_struct! {
    /// The multimodal part of a request.
    #[derive(Debug, Clone, PartialEq)]
    pub struct MultimodalInputs: array {
        /// The media items, in prompt order.
        pub mm_items: Vec<MultimodalDataItem>,
        /// Image placeholder token id.
        pub im_token_id: Option<i64> = None,
        /// Video placeholder token id.
        pub video_token_id: Option<i64> = None,
        /// M-RoPE position ids.
        pub mrope_positions: Option<Tensor> = None,
        /// M-RoPE position delta.
        pub mrope_position_delta: Option<Tensor> = None,
        /// M-RoPE position delta as a scalar.
        pub mrope_position_delta_scalar: Option<i64> = None,
        /// Retired slot kept for arity; always `None`.
        pub mrope_position_delta_repeated_cache: Option<Tensor> = None,
    }
}

wire_struct! {
    /// EPD: one item for an encode node to embed and ship to the prefill
    /// ranks that wait for it under `bootstrap_room`. The first two fields
    /// come from the Python `BaseReq` base class.
    #[derive(Debug, Clone, PartialEq)]
    pub struct EncodeRequest: array("EncodeRequest") {
        /// Request id; unused, `request_id` names the request.
        pub rid: Option<StrOrList> = None,
        /// In-process HTTP worker address; unused across processes.
        pub http_worker_ipc: Option<String> = None,
        /// The gateway's id of the request.
        pub request_id: String,
        /// The encode node's bootstrap server, as the gateway gave it.
        pub bootstrap_host: String,
        /// Its port.
        pub bootstrap_port: i64,
        /// The room the prefill ranks receive the embedding under.
        pub bootstrap_room: u64,
        /// The items to embed; the gateway sends one per request.
        pub items: Vec<MultimodalDataItem>,
    }
}

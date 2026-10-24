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

//! TokenSpeed's engine IPC schema and its msgpack codec.
//!
//! The messages a frontend and the engine exchange over ZMQ, encoded exactly
//! as the Python side encodes them (`msgspec` structs in
//! `python/tokenspeed/runtime/engine/io_struct.py` and `zmq_wire.py`). Golden
//! fixtures written by the Python encoder keep the two sides in step: every
//! fixture decodes here and re-encodes byte for byte (`tests/conformance.rs`).
//!
//! Field order of the array-layout structs is the wire contract. To add a
//! field, append it on both sides with a default and regenerate the fixtures.

#![forbid(unsafe_code)]

#[macro_use]
pub mod codec;
pub mod control;
pub mod error;
pub mod frames;
pub mod handshake;
pub mod msgpack;
pub mod multimodal;
pub mod output;
pub mod request;
pub mod sampling;
pub mod tensor;
pub mod value;

pub use codec::{decode_bytes, decode_frames, encode_frames, Dict, StrOrList, Wire};
pub use control::{ControlOutput, ControlRequest};
pub use error::{Error, Result};
pub use frames::{engine_identity, EngineOutput, EngineRequest, ENGINE_CORE_DEAD};
pub use handshake::{
    EngineCoreReadyResponse, HandshakeAddresses, HandshakeInitMessage, ReadyMessage,
};
pub use multimodal::{
    EncodeRequest, Modality, MultimodalDataItem, MultimodalInputs, ShmTensorHandle,
};
pub use output::{BatchTokenIDOutSlim, RequestOutput};
pub use request::{SessionParams, TokenizedGenerateReqInput};
pub use sampling::SamplingParams;
pub use tensor::Tensor;
pub use value::Value;

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

//! ZMQ message framing on the frontend–engine sockets.
//!
//! * Input (frontend ROUTER to engine DEALER): `[type, payload, aux...]`.
//!   `type` is one byte so the engine dispatches without decoding: ADD
//!   carries a [`TokenizedGenerateReqInput`] whose tensors may reference the
//!   `aux` frames; ABORT carries a msgpack list of request ids; CONTROL
//!   carries a [`ControlRequest`].
//! * Output (engine PUSH to frontend PULL): a [`BatchTokenIDOutSlim`]
//!   message, a [`ControlOutput`] (told apart by its tag), or the single
//!   frame [`ENGINE_CORE_DEAD`] when the engine exits.

use bytes::Bytes;

use crate::{
    codec::{decode_frames, encode_frames},
    control::{ControlOutput, ControlRequest},
    error::{Error, Result},
    msgpack::Decoder,
    output::BatchTokenIDOutSlim,
    request::TokenizedGenerateReqInput,
};

/// Type byte of an ADD message.
pub const REQ_TYPE_ADD: u8 = 0x00;
/// Type byte of an ABORT message.
pub const REQ_TYPE_ABORT: u8 = 0x01;
/// Type byte of a CONTROL message.
pub const REQ_TYPE_CONTROL: u8 = 0x02;
/// The frame an engine sends on its output socket when it shuts down; the
/// frontend then treats the engine as dead.
pub const ENGINE_CORE_DEAD: &[u8] = b"ENGINE_CORE_DEAD";

/// The ZMQ routing identity of engine `engine_index`: its index as two
/// little-endian bytes.
pub fn engine_identity(engine_index: u16) -> [u8; 2] {
    engine_index.to_le_bytes()
}

/// A message on the engine's input socket.
#[derive(Debug, Clone, PartialEq)]
pub enum EngineRequest {
    /// Start a request.
    Add(Box<TokenizedGenerateReqInput>),
    /// Abort requests by id.
    Abort(Vec<String>),
    /// A control operation.
    Control(Box<ControlRequest>),
}

impl EngineRequest {
    /// The message's frames.
    pub fn to_frames(&self) -> Vec<Bytes> {
        let (type_frame, payload): (&'static [u8], _) = match self {
            EngineRequest::Add(request) => (&[REQ_TYPE_ADD], encode_frames(request.as_ref())),
            EngineRequest::Abort(rids) => (&[REQ_TYPE_ABORT], encode_frames(rids)),
            EngineRequest::Control(request) => {
                (&[REQ_TYPE_CONTROL], encode_frames(request.as_ref()))
            }
        };
        let mut frames = Vec::with_capacity(1 + payload.len());
        frames.push(Bytes::from_static(type_frame));
        frames.extend(payload);
        frames
    }

    /// Decode a message from its frames.
    pub fn from_frames(frames: &[Bytes]) -> Result<Self> {
        let [type_frame, payload @ ..] = frames else {
            return Err(Error::Invalid("an input message has no frames".to_owned()));
        };
        if payload.is_empty() {
            return Err(Error::Invalid(
                "an input message needs a type frame and a payload".to_owned(),
            ));
        }
        match type_frame.as_ref() {
            [REQ_TYPE_ADD] => Ok(EngineRequest::Add(Box::new(decode_frames(payload)?))),
            [REQ_TYPE_ABORT] => Ok(EngineRequest::Abort(decode_frames(payload)?)),
            [REQ_TYPE_CONTROL] => Ok(EngineRequest::Control(Box::new(decode_frames(payload)?))),
            other => Err(Error::Invalid(format!(
                "unknown request type frame {other:02x?}"
            ))),
        }
    }
}

/// A message on the engine's output socket.
#[derive(Debug, Clone, PartialEq)]
pub enum EngineOutput {
    /// One step's outputs.
    Batch(Box<BatchTokenIDOutSlim>),
    /// A control operation's reply.
    Control(Box<ControlOutput>),
    /// The engine shut down.
    Dead,
}

impl EngineOutput {
    /// The message's frames.
    pub fn to_frames(&self) -> Vec<Bytes> {
        match self {
            EngineOutput::Batch(batch) => encode_frames(batch.as_ref()),
            EngineOutput::Control(output) => encode_frames(output.as_ref()),
            EngineOutput::Dead => vec![Bytes::from_static(ENGINE_CORE_DEAD)],
        }
    }

    /// Decode a message from its frames.
    pub fn from_frames(frames: &[Bytes]) -> Result<Self> {
        match frames {
            [] => Err(Error::Invalid("an output message has no frames".to_owned())),
            [only] if only.as_ref() == ENGINE_CORE_DEAD => Ok(EngineOutput::Dead),
            _ if ControlOutput::knows(Decoder::new(frames).peek_tag()?) => {
                Ok(EngineOutput::Control(Box::new(decode_frames(frames)?)))
            }
            _ => Ok(EngineOutput::Batch(Box::new(decode_frames(frames)?))),
        }
    }
}

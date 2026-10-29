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
//!   carries a [`ControlRequest`] or a [`ControlCall`]; ENCODE carries an
//!   [`EncodeRequest`], to an EPD encode server. An engine lists the types
//!   it takes in its ready response (`request_types`).
//! * Output (engine PUSH to frontend PULL): a [`BatchTokenIDOutSlim`]
//!   message, a [`ControlOutput`] or a [`ControlReplySlim`] (told apart by
//!   their tags), or the single frame [`ENGINE_CORE_DEAD`] when the engine
//!   exits.

use bytes::Bytes;

use crate::{
    codec::{decode_frames, encode_frames},
    control::{ControlCall, ControlOutput, ControlReplySlim, ControlRequest},
    error::{Error, Result},
    msgpack::{Decoder, Kind},
    multimodal::EncodeRequest,
    output::BatchTokenIDOutSlim,
    request::TokenizedGenerateReqInput,
};

/// Type byte of an ADD message.
pub const REQ_TYPE_ADD: u8 = 0x00;
/// Type byte of an ABORT message.
pub const REQ_TYPE_ABORT: u8 = 0x01;
/// Type byte of a CONTROL message.
pub const REQ_TYPE_CONTROL: u8 = 0x02;
/// Type byte of an ENCODE message.
pub const REQ_TYPE_ENCODE: u8 = 0x03;

/// `request_types` names: what an engine lists in its ready response for
/// the request types it takes.
pub const REQUEST_ADD: &str = "add";
/// See [`REQUEST_ADD`].
pub const REQUEST_ABORT: &str = "abort";
/// See [`REQUEST_ADD`].
pub const REQUEST_CONTROL: &str = "control";
/// See [`REQUEST_ADD`].
pub const REQUEST_ENCODE: &str = "encode";
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
    /// A control operation named by method.
    Call(Box<ControlCall>),
    /// An item to embed, on an EPD encode server.
    Encode(Box<EncodeRequest>),
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
            EngineRequest::Call(call) => (&[REQ_TYPE_CONTROL], encode_frames(call.as_ref())),
            EngineRequest::Encode(request) => (&[REQ_TYPE_ENCODE], encode_frames(request.as_ref())),
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
            [REQ_TYPE_CONTROL] if is_call(payload)? => {
                Ok(EngineRequest::Call(Box::new(decode_frames(payload)?)))
            }
            [REQ_TYPE_CONTROL] => Ok(EngineRequest::Control(Box::new(decode_frames(payload)?))),
            [REQ_TYPE_ENCODE] => Ok(EngineRequest::Encode(Box::new(decode_frames(payload)?))),
            other => Err(Error::Invalid(format!(
                "unknown request type frame {other:02x?}"
            ))),
        }
    }
}

/// Whether a CONTROL payload is a [`ControlCall`]: its first element is the
/// call id, where a [`ControlRequest`]'s is its tag.
fn is_call(payload: &[Bytes]) -> Result<bool> {
    let mut probe = Decoder::new(payload);
    Ok(probe.array_len()? > 0 && probe.peek_kind()? == Kind::Int)
}

/// A message on the engine's output socket.
#[derive(Debug, Clone, PartialEq)]
pub enum EngineOutput {
    /// One step's outputs.
    Batch(Box<BatchTokenIDOutSlim>),
    /// A control operation's reply.
    Control(Box<ControlOutput>),
    /// A control call's reply.
    ControlReply(ControlReplySlim),
    /// The engine shut down.
    Dead,
}

impl EngineOutput {
    /// The message's frames.
    pub fn to_frames(&self) -> Vec<Bytes> {
        match self {
            EngineOutput::Batch(batch) => encode_frames(batch.as_ref()),
            EngineOutput::Control(output) => encode_frames(output.as_ref()),
            EngineOutput::ControlReply(reply) => encode_frames(reply),
            EngineOutput::Dead => vec![Bytes::from_static(ENGINE_CORE_DEAD)],
        }
    }

    /// Decode a message from its frames.
    pub fn from_frames(frames: &[Bytes]) -> Result<Self> {
        match frames {
            [] => Err(Error::Invalid("an output message has no frames".to_owned())),
            [only] if only.as_ref() == ENGINE_CORE_DEAD => Ok(EngineOutput::Dead),
            _ => match Decoder::new(frames).peek_tag()? {
                tag if ControlOutput::knows(tag) => {
                    Ok(EngineOutput::Control(Box::new(decode_frames(frames)?)))
                }
                "ControlReplySlim" => Ok(EngineOutput::ControlReply(decode_frames(frames)?)),
                _ => Ok(EngineOutput::Batch(Box::new(decode_frames(frames)?))),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::control::FlushCacheReqInput;

    fn round_trip(request: EngineRequest) -> EngineRequest {
        EngineRequest::from_frames(&request.to_frames()).unwrap()
    }

    #[test]
    fn a_control_message_carries_either_a_tagged_operation_or_a_call() {
        let tagged =
            EngineRequest::Control(Box::new(ControlRequest::FlushCache(FlushCacheReqInput {
                rid: None,
                http_worker_ipc: None,
            })));
        assert_eq!(round_trip(tagged.clone()), tagged);
        let call = EngineRequest::Call(Box::new(ControlCall {
            call_id: 3,
            method: "stop_profile".to_owned(),
            args: Vec::new(),
        }));
        assert_eq!(call.to_frames()[0].as_ref(), [REQ_TYPE_CONTROL]);
        assert_eq!(round_trip(call.clone()), call);
    }

    #[test]
    fn an_encode_message_rides_its_own_type() {
        let request = EngineRequest::Encode(Box::new(EncodeRequest {
            rid: None,
            http_worker_ipc: None,
            request_id: "item-0".to_owned(),
            bootstrap_host: "10.0.0.9".to_owned(),
            bootstrap_port: 8998,
            bootstrap_room: 41,
            items: Vec::new(),
        }));
        assert_eq!(request.to_frames()[0].as_ref(), [REQ_TYPE_ENCODE]);
        assert_eq!(round_trip(request.clone()), request);
    }

    #[test]
    fn a_call_reply_is_told_apart_from_batches_and_control_outputs() {
        let reply = ControlReplySlim {
            call_id: 3,
            success: false,
            message: Some("no profiler running".to_owned()),
            engine_index: 1,
        };
        let frames = EngineOutput::ControlReply(reply.clone()).to_frames();
        assert_eq!(
            EngineOutput::from_frames(&frames).unwrap(),
            EngineOutput::ControlReply(reply)
        );
    }
}

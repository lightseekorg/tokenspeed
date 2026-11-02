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

//! One round's frontend messages as the attention-TP group's first member
//! relays them to the others: the messages' own wire frames, and whether
//! the first member is stopping. With the L3 tier, every member's storage
//! report rides the same exchange ([`join`]), so a round agrees on both in
//! one collective.

use bytes::Bytes;
use ts_wire::EngineRequest;

use crate::codec::{Reader, Writer};

/// A relay that does not decode.
#[derive(Debug, thiserror::Error)]
pub enum RelayError {
    /// The payload is cut short.
    #[error("the relayed messages are cut short")]
    Truncated,
    /// A relayed message did not decode.
    #[error("a relayed message: {0}")]
    Wire(#[from] ts_wire::Error),
}

/// The relay payload.
pub fn encode(stop: bool, messages: &[EngineRequest]) -> Vec<u8> {
    let mut out = Writer::new();
    out.flag(stop);
    out.count(messages.len());
    for message in messages {
        let frames = message.to_frames();
        out.count(frames.len());
        for frame in &frames {
            out.bytes(frame);
        }
    }
    out.finish()
}

/// The first member's stop request and messages.
pub fn decode(payload: &[u8]) -> Result<(bool, Vec<EngineRequest>), RelayError> {
    let mut reader = Reader::new(payload);
    let stop = reader.flag().ok_or(RelayError::Truncated)?;
    let count = reader.count().ok_or(RelayError::Truncated)?;
    let mut messages = Vec::with_capacity(count.min(1024));
    for _ in 0..count {
        let num_frames = reader.count().ok_or(RelayError::Truncated)?;
        let frames = (0..num_frames)
            .map(|_| reader.bytes().map(Bytes::copy_from_slice))
            .collect::<Option<Vec<_>>>()
            .ok_or(RelayError::Truncated)?;
        messages.push(EngineRequest::from_frames(&frames)?);
    }
    reader.finish(()).ok_or(RelayError::Truncated)?;
    Ok((stop, messages))
}

/// One member's part of a round's exchange with the L3 tier: the relay
/// payload (the first member's; the others send none) and its storage
/// report.
pub fn join(relay: &[u8], report: &[u8]) -> Vec<u8> {
    let mut out = Writer::new();
    out.bytes(relay);
    out.bytes(report);
    out.finish()
}

/// What [`join`] put together.
pub fn split(payload: &[u8]) -> Result<(&[u8], &[u8]), RelayError> {
    let mut reader = Reader::new(payload);
    let relay = reader.bytes();
    let report = reader.bytes();
    relay
        .zip(report)
        .and_then(|parts| reader.finish(parts))
        .ok_or(RelayError::Truncated)
}

#[cfg(test)]
mod tests {
    use ts_wire::{SamplingParams, TokenizedGenerateReqInput};

    use super::*;

    #[test]
    fn messages_survive_the_relay() {
        let add = EngineRequest::Add(Box::new(TokenizedGenerateReqInput {
            input_ids: Some(vec![1, 2, 3]),
            ..TokenizedGenerateReqInput::new(
                "r1",
                SamplingParams {
                    max_new_tokens: Some(4),
                    ..Default::default()
                },
            )
        }));
        let abort = EngineRequest::Abort(vec!["r0".to_owned()]);
        let messages = vec![add, abort];
        let (stop, relayed) = decode(&encode(true, &messages)).unwrap();
        assert!(stop);
        assert_eq!(relayed, messages);
        assert_eq!(decode(&encode(false, &[])).unwrap(), (false, Vec::new()));
    }

    #[test]
    fn a_cut_payload_is_an_error() {
        let payload = encode(false, &[EngineRequest::Abort(vec!["r0".to_owned()])]);
        assert!(matches!(
            decode(&payload[..payload.len() - 1]),
            Err(RelayError::Truncated)
        ));
    }

    #[test]
    fn a_storage_report_rides_with_the_relay() {
        let relay = encode(true, &[EngineRequest::Abort(vec!["r0".to_owned()])]);
        let joined = join(&relay, b"report");
        assert_eq!(split(&joined).unwrap(), (&relay[..], &b"report"[..]));
        // A member other than the first relays nothing.
        assert_eq!(split(&join(&[], b"")).unwrap(), (&[][..], &[][..]));
        assert!(matches!(
            split(&joined[..joined.len() - 1]),
            Err(RelayError::Truncated)
        ));
    }
}

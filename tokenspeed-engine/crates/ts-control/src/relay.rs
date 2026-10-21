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
//! the first member is stopping.

use bytes::Bytes;
use ts_wire::EngineRequest;

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
    let mut out = vec![u8::from(stop)];
    out.extend_from_slice(&(messages.len() as u64).to_le_bytes());
    for message in messages {
        let frames = message.to_frames();
        out.extend_from_slice(&(frames.len() as u64).to_le_bytes());
        for frame in frames {
            out.extend_from_slice(&(frame.len() as u64).to_le_bytes());
            out.extend_from_slice(&frame);
        }
    }
    out
}

struct Reader<'a>(&'a [u8]);

impl<'a> Reader<'a> {
    fn take(&mut self, len: usize) -> Result<&'a [u8], RelayError> {
        if self.0.len() < len {
            return Err(RelayError::Truncated);
        }
        let (taken, rest) = self.0.split_at(len);
        self.0 = rest;
        Ok(taken)
    }

    fn len(&mut self) -> Result<usize, RelayError> {
        let bytes = self.take(8)?.try_into().expect("eight bytes");
        usize::try_from(u64::from_le_bytes(bytes)).map_err(|_| RelayError::Truncated)
    }
}

/// The first member's stop request and messages.
pub fn decode(payload: &[u8]) -> Result<(bool, Vec<EngineRequest>), RelayError> {
    let mut reader = Reader(payload);
    let stop = reader.take(1)?[0] != 0;
    let count = reader.len()?;
    let mut messages = Vec::with_capacity(count.min(1024));
    for _ in 0..count {
        let num_frames = reader.len()?;
        let mut frames = Vec::with_capacity(num_frames.min(64));
        for _ in 0..num_frames {
            let len = reader.len()?;
            frames.push(Bytes::copy_from_slice(reader.take(len)?));
        }
        messages.push(EngineRequest::from_frames(&frames)?);
    }
    if !reader.0.is_empty() {
        return Err(RelayError::Truncated);
    }
    Ok((stop, messages))
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
}

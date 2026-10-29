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

//! The P/D control messages. The frames are those the Python peers sent
//! before this crate replaced them, byte for byte, so nodes of either
//! version interoperate.

use bytes::Bytes;

use crate::{
    net::{ascii, parse_uint},
    status::TransferPoll,
};

/// The largest room a pre-allocation may name.
const MAX_ROOM: u64 = (1 << 63) - 1;

/// A malformed control message.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct WireError(pub String);

fn error(message: &str) -> WireError {
    WireError(message.to_owned())
}

fn text(frame: &[u8], name: &str) -> Result<String, WireError> {
    if !frame.is_ascii() {
        return Err(WireError(format!("{name} must be ASCII")));
    }
    Ok(String::from_utf8_lossy(frame).into_owned())
}

/// A Decode rank's registration with a Prefill rank: where to reach it,
/// its transfer session, and its cache.
#[allow(clippy::too_many_arguments)]
pub fn registration_frames(
    host: &str,
    port: u16,
    session_id: &str,
    arena_base: u64,
    layout: &[u8],
    tp_size: i64,
    tp_rank: i64,
) -> Vec<Bytes> {
    vec![
        Bytes::from_static(b"None"),
        ascii(host),
        ascii(port),
        ascii(session_id),
        // struct.pack("Q"): native order.
        Bytes::copy_from_slice(&arena_base.to_ne_bytes()),
        Bytes::copy_from_slice(layout),
        ascii(tp_size),
        ascii(tp_rank),
    ]
}

/// Whether `frames` is a registration rather than a pre-allocation.
pub fn is_registration(frames: &[Bytes]) -> bool {
    frames
        .first()
        .is_some_and(|header| header.as_ref() == b"None")
}

/// A registration's reply route, readable even when its layout is rejected
/// (`KVArgsRegisterInfo.route_header_from_zmq`): endpoint, port, session.
pub fn registration_route(frames: &[Bytes]) -> Result<(String, u16, String), WireError> {
    if frames.len() != 8 || !is_registration(frames) {
        return Err(error("CachePD slab registration requires exactly 8 frames"));
    }
    let endpoint = text(&frames[1], "CachePD registration endpoint")?;
    let session_id = text(&frames[3], "CachePD registration session")?;
    if endpoint.is_empty() || endpoint.len() > 255 {
        return Err(error("CachePD registration endpoint is invalid"));
    }
    if session_id.is_empty() || session_id.len() > 512 {
        return Err(error("CachePD registration session is invalid"));
    }
    let port = parse_uint(&frames[2], 65535)
        .filter(|&port| port >= 1)
        .ok_or_else(|| error("CachePD registration port is invalid"))?;
    Ok((endpoint, port as u16, session_id))
}

/// A Decode rank's pre-allocation of one request: its destination block
/// manifest, empty for a rank that only joins the status agreement.
pub fn preallocation_frames(room: u64, session_id: &str, manifest: Option<&[u8]>) -> Vec<Bytes> {
    vec![
        ascii(room),
        ascii(session_id),
        manifest.map_or_else(Bytes::new, Bytes::copy_from_slice),
    ]
}

/// A pre-allocation's room and session (`TransferInfo.route_header_from_zmq`).
pub fn preallocation_route(frames: &[Bytes]) -> Result<(u64, String), WireError> {
    if frames.len() != 3 {
        return Err(error("CachePD pre-allocation requires exactly 3 frames"));
    }
    let room = parse_uint(&frames[0], MAX_ROOM)
        .ok_or_else(|| error("CachePD room must be canonical unsigned ASCII"))?;
    let session_id = text(&frames[1], "CachePD session")?;
    if session_id.is_empty() || session_id.len() > 512 {
        return Err(error("CachePD request requires a Decode session"));
    }
    Ok((room, session_id))
}

/// A Prefill rank's report on one room.
#[derive(Debug, Clone, PartialEq)]
pub struct StatusMessage {
    /// The room.
    pub room: u64,
    /// Its state on the sender.
    pub status: TransferPoll,
    /// The sender in Decode's completion-tracking rank space.
    pub prefill_rank: i64,
    /// The first output token, -1 if none.
    pub bootstrap_token: i64,
    /// The drafter's candidates for the first decode, if any.
    pub spec_candidate_ids: Option<Vec<i32>>,
    /// Prompt tokens the sender served from its cache.
    pub cached_tokens: i64,
    /// The first token's logprob, if the request returns logprobs.
    pub bootstrap_logprob: Option<f64>,
}

impl StatusMessage {
    /// A bare status: no first token and no usage.
    pub fn bare(room: u64, status: TransferPoll, prefill_rank: i64) -> Self {
        Self {
            room,
            status,
            prefill_rank,
            bootstrap_token: -1,
            spec_candidate_ids: None,
            cached_tokens: 0,
            bootstrap_logprob: None,
        }
    }

    /// The message's frames.
    pub fn to_frames(&self) -> Vec<Bytes> {
        let candidates = self
            .spec_candidate_ids
            .as_ref()
            .map_or_else(Bytes::new, |ids| {
                ids.iter()
                    .flat_map(|id| id.to_le_bytes())
                    .collect::<Vec<u8>>()
                    .into()
            });
        let logprob = self.bootstrap_logprob.map_or_else(Bytes::new, |value| {
            Bytes::copy_from_slice(&value.to_le_bytes())
        });
        vec![
            ascii(self.room),
            ascii(self.status as i64),
            ascii(self.prefill_rank),
            ascii(self.bootstrap_token),
            candidates,
            ascii(self.cached_tokens),
            logprob,
        ]
    }

    /// Parse a status message; the last
    /// three frames are optional, as older senders omit them.
    pub fn from_frames(frames: &[Bytes]) -> Result<Self, WireError> {
        fn int(frame: &[u8], name: &str) -> Result<i64, WireError> {
            std::str::from_utf8(frame)
                .ok()
                .and_then(|value| value.parse().ok())
                .ok_or_else(|| WireError(format!("{name} is not an integer")))
        }
        if frames.len() < 3 {
            return Err(error("a status message has at least 3 frames"));
        }
        let room =
            u64::try_from(int(&frames[0], "room")?).map_err(|_| error("room is negative"))?;
        let status = TransferPoll::from_wire(int(&frames[1], "status")?)
            .ok_or_else(|| error("unknown status"))?;
        let prefill_rank = int(&frames[2], "prefill rank")?;
        let bootstrap_token = match frames.get(3) {
            Some(frame) => int(frame, "bootstrap token")?,
            None => -1,
        };
        let spec_candidate_ids = match frames.get(4) {
            Some(frame) if !frame.is_empty() => {
                if frame.len() % 4 != 0 {
                    return Err(error("candidate ids are not int32"));
                }
                Some(
                    frame
                        .chunks_exact(4)
                        .map(|chunk| i32::from_le_bytes(chunk.try_into().expect("4 bytes")))
                        .collect(),
                )
            }
            _ => None,
        };
        let cached_tokens = match frames.get(5) {
            Some(frame) => int(frame, "cached tokens")?,
            None => 0,
        };
        let bootstrap_logprob = match frames.get(6) {
            Some(frame) if !frame.is_empty() => Some(f64::from_le_bytes(
                frame
                    .as_ref()
                    .try_into()
                    .map_err(|_| error("the logprob is not a double"))?,
            )),
            _ => None,
        };
        Ok(Self {
            room,
            status,
            prefill_rank,
            bootstrap_token,
            spec_candidate_ids,
            cached_tokens,
            bootstrap_logprob,
        })
    }
}

/// The port a session id (`host:port`) ends in, which shards a room's
/// transfers across workers.
pub fn session_port(session_id: &str) -> Option<u64> {
    session_id.rsplit(':').next()?.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_status_message_round_trips() {
        let message = StatusMessage {
            room: 77,
            status: TransferPoll::Success,
            prefill_rank: 3,
            bootstrap_token: 1234,
            spec_candidate_ids: Some(vec![5, -1, 7]),
            cached_tokens: 64,
            bootstrap_logprob: Some(-0.25),
        };
        assert_eq!(
            StatusMessage::from_frames(&message.to_frames()),
            Ok(message)
        );
        let bare = StatusMessage::bare(9, TransferPoll::Failed, 0);
        assert_eq!(StatusMessage::from_frames(&bare.to_frames()), Ok(bare));
        // Older senders stop after the rank.
        let short = [ascii(9), ascii(5), ascii(1)];
        let parsed = StatusMessage::from_frames(&short).unwrap();
        assert_eq!(parsed.bootstrap_token, -1);
        assert_eq!(parsed.cached_tokens, 0);
    }

    #[test]
    fn routes_parse_from_well_formed_frames_only() {
        let frames = registration_frames("10.0.0.1", 4000, "10.0.0.1:5000", 0x1000, b"L", 4, 2);
        assert_eq!(
            registration_route(&frames),
            Ok(("10.0.0.1".to_owned(), 4000, "10.0.0.1:5000".to_owned()))
        );
        assert_eq!(frames[4].as_ref(), 0x1000u64.to_ne_bytes());
        assert!(registration_route(&frames[..7]).is_err());

        let frames = preallocation_frames(42, "10.0.0.1:5000", None);
        assert_eq!(
            preallocation_route(&frames),
            Ok((42, "10.0.0.1:5000".to_owned()))
        );
        assert!(frames[2].is_empty());
        let bad = [ascii("042"), ascii("s"), Bytes::new()];
        assert!(preallocation_route(&bad).is_err());
        assert_eq!(session_port("[::1]:5000"), Some(5000));
    }
}

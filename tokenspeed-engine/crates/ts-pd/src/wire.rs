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

//! The P/D control messages. Cache registrations carry an incarnation
//! separate from the transport address: every P/D peer must support this
//! nine-frame protocol before serving together. A tenth frame asks the
//! Prefill rank to acknowledge the registration once installed
//! ([`registration_ack_frames`]), or to say why it rejected it
//! ([`registration_reject_frames`]); a Decode rank waits for that reply, so
//! it needs Prefill ranks that send it.

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
/// its transfer session, the identity of its current cache, and the id the
/// Prefill rank acknowledges it by.
#[allow(clippy::too_many_arguments)]
pub fn registration_frames(
    host: &str,
    port: u16,
    session_id: &str,
    cache_id: &str,
    arena_base: u64,
    layout: &[u8],
    tp_size: i64,
    tp_rank: i64,
    registration_id: u64,
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
        ascii(cache_id),
        ascii(registration_id),
    ]
}

/// Whether `frames` is a registration rather than a pre-allocation.
pub fn is_registration(frames: &[Bytes]) -> bool {
    frames
        .first()
        .is_some_and(|header| header.as_ref() == b"None")
}

/// A registration's reply route, readable even when its layout is rejected
/// (`KVArgsRegisterInfo.route_header_from_zmq`): endpoint, port, cache identity.
pub fn registration_route(frames: &[Bytes]) -> Result<(String, u16, String), WireError> {
    if !matches!(frames.len(), 9 | 10) || !is_registration(frames) {
        return Err(error("CachePD slab registration requires 9 or 10 frames"));
    }
    let endpoint = text(&frames[1], "CachePD registration endpoint")?;
    let cache_id = text(&frames[8], "CachePD registration cache identity")?;
    if endpoint.is_empty() || endpoint.len() > 255 {
        return Err(error("CachePD registration endpoint is invalid"));
    }
    if cache_id.is_empty() || cache_id.len() > 512 {
        return Err(error("CachePD registration cache identity is invalid"));
    }
    let port = parse_uint(&frames[2], 65535)
        .filter(|&port| port >= 1)
        .ok_or_else(|| error("CachePD registration port is invalid"))?;
    Ok((endpoint, port as u16, cache_id))
}

/// The id a registration is acknowledged by; `None` for the nine frames of
/// a Decode rank that predates acknowledgments, which is not acknowledged.
pub fn registration_id(frames: &[Bytes]) -> Result<Option<u64>, WireError> {
    frames
        .get(9)
        .map(|frame| {
            parse_uint(frame, u64::MAX)
                .ok_or_else(|| error("CachePD registration id must be canonical unsigned ASCII"))
        })
        .transpose()
}

/// A Prefill rank installed registration `registration_id` of the Decode
/// cache `cache_id`: pre-allocations that follow it find the cache.
pub fn registration_ack_frames(registration_id: u64, cache_id: &str, rank: i64) -> Vec<Bytes> {
    vec![
        Bytes::from_static(b"RegisterAck"),
        ascii(registration_id),
        ascii(cache_id),
        ascii(rank),
    ]
}

pub fn is_registration_ack(frames: &[Bytes]) -> bool {
    frames
        .first()
        .is_some_and(|frame| frame.as_ref() == b"RegisterAck")
}

/// A registration acknowledgment's id, cache identity and Prefill rank.
pub fn registration_ack(frames: &[Bytes]) -> Result<(u64, String, i64), WireError> {
    if frames.len() != 4 || !is_registration_ack(frames) {
        return Err(error(
            "CachePD registration acknowledgment requires exactly 4 frames",
        ));
    }
    registration_reply(frames)
}

/// A Prefill rank rejected registration `registration_id` of the Decode
/// cache `cache_id`, for `reason`.
pub fn registration_reject_frames(
    registration_id: u64,
    cache_id: &str,
    rank: i64,
    reason: &str,
) -> Vec<Bytes> {
    vec![
        Bytes::from_static(b"RegisterReject"),
        ascii(registration_id),
        ascii(cache_id),
        ascii(rank),
        Bytes::copy_from_slice(reason.as_bytes()),
    ]
}

pub fn is_registration_reject(frames: &[Bytes]) -> bool {
    frames
        .first()
        .is_some_and(|frame| frame.as_ref() == b"RegisterReject")
}

/// A registration rejection's id, cache identity, Prefill rank and reason.
pub fn registration_reject(frames: &[Bytes]) -> Result<(u64, String, i64, String), WireError> {
    if frames.len() != 5 || !is_registration_reject(frames) {
        return Err(error(
            "CachePD registration rejection requires exactly 5 frames",
        ));
    }
    let (registration_id, cache_id, rank) = registration_reply(frames)?;
    let reason = String::from_utf8_lossy(&frames[4]).into_owned();
    Ok((registration_id, cache_id, rank, reason))
}

/// The id, cache identity and Prefill rank after a registration reply's header.
fn registration_reply(frames: &[Bytes]) -> Result<(u64, String, i64), WireError> {
    let registration_id = parse_uint(&frames[1], u64::MAX)
        .ok_or_else(|| error("CachePD registration id must be canonical unsigned ASCII"))?;
    let cache_id = text(&frames[2], "CachePD registration cache identity")?;
    if cache_id.is_empty() || cache_id.len() > 512 {
        return Err(error("CachePD registration cache identity is invalid"));
    }
    let rank = parse_uint(&frames[3], i64::MAX as u64)
        .ok_or_else(|| error("CachePD registration reply rank is invalid"))?;
    Ok((registration_id, cache_id, rank as i64))
}

/// A Decode rank's pre-allocation of one request: its destination block
/// manifest, empty for a rank that only joins the status agreement.
pub fn preallocation_frames(room: u64, cache_id: &str, manifest: Option<&[u8]>) -> Vec<Bytes> {
    vec![
        ascii(room),
        ascii(cache_id),
        manifest.map_or_else(Bytes::new, Bytes::copy_from_slice),
    ]
}

/// A pre-allocation's room and cache identity (`TransferInfo.route_header_from_zmq`).
pub fn preallocation_route(frames: &[Bytes]) -> Result<(u64, String), WireError> {
    if frames.len() != 3 {
        return Err(error("CachePD pre-allocation requires exactly 3 frames"));
    }
    let room = parse_uint(&frames[0], MAX_ROOM)
        .ok_or_else(|| error("CachePD room must be canonical unsigned ASCII"))?;
    let cache_id = text(&frames[1], "CachePD cache identity")?;
    if cache_id.is_empty() || cache_id.len() > 512 {
        return Err(error("CachePD request requires a Decode cache identity"));
    }
    Ok((room, cache_id))
}

/// A Decode rank holds `room`'s request: its cache is the request's
/// destination. Sent to a Prefill node under deferred decode allocation, which
/// answers with the room's status (`Bootstrapped`, or `Failed`) once the
/// request holds every page of its prompt there. A Prefill rank that does
/// not know the message drops it as a malformed pre-allocation.
pub fn announce_frames(room: u64, cache_id: &str) -> Vec<Bytes> {
    vec![
        Bytes::from_static(b"Announce"),
        ascii(room),
        ascii(cache_id),
    ]
}

pub fn is_announce(frames: &[Bytes]) -> bool {
    frames
        .first()
        .is_some_and(|frame| frame.as_ref() == b"Announce")
}

/// An announcement's room and cache identity, framed as a cancellation's.
pub fn announce_route(frames: &[Bytes]) -> Result<(u64, String), WireError> {
    if frames.len() != 3 || !is_announce(frames) {
        return Err(error("CachePD announcement requires exactly 3 frames"));
    }
    let mut request = frames.to_vec();
    request[0] = Bytes::from_static(b"Cancel");
    cancellation_route(&request)
}

/// Explicit cancellation is distinct from a failure report: only its ACK
/// proves that this Prefill rank can no longer write the room's pages.
pub fn cancellation_frames(room: u64, cache_id: &str) -> Vec<Bytes> {
    vec![Bytes::from_static(b"Cancel"), ascii(room), ascii(cache_id)]
}

pub fn is_cancellation(frames: &[Bytes]) -> bool {
    frames
        .first()
        .is_some_and(|frame| frame.as_ref() == b"Cancel")
}

pub fn cancellation_route(frames: &[Bytes]) -> Result<(u64, String), WireError> {
    if frames.len() != 3 || !is_cancellation(frames) {
        return Err(error("CachePD cancellation requires exactly 3 frames"));
    }
    let room = parse_uint(&frames[1], MAX_ROOM)
        .ok_or_else(|| error("CachePD room must be canonical unsigned ASCII"))?;
    let cache_id = text(&frames[2], "CachePD cache identity")?;
    if cache_id.is_empty() || cache_id.len() > 512 {
        return Err(error(
            "CachePD cancellation requires a Decode cache identity",
        ));
    }
    Ok((room, cache_id))
}

pub fn cancellation_ack_frames(room: u64, cache_id: &str, rank: i64) -> Vec<Bytes> {
    vec![
        Bytes::from_static(b"CancelAck"),
        ascii(room),
        ascii(cache_id),
        ascii(rank),
    ]
}

pub fn is_cancellation_ack(frames: &[Bytes]) -> bool {
    frames
        .first()
        .is_some_and(|frame| frame.as_ref() == b"CancelAck")
}

pub fn cancellation_ack(frames: &[Bytes]) -> Result<(u64, String, i64), WireError> {
    if frames.len() != 4 || !is_cancellation_ack(frames) {
        return Err(error(
            "CachePD cancellation acknowledgment requires exactly 4 frames",
        ));
    }
    let mut request = frames[..3].to_vec();
    request[0] = Bytes::from_static(b"Cancel");
    let (room, cache_id) = cancellation_route(&request)?;
    let rank = parse_uint(&frames[3], i64::MAX as u64)
        .ok_or_else(|| error("CachePD cancellation rank is invalid"))?;
    Ok((room, cache_id, rank as i64))
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
    /// The first token's packed full-vocabulary Top-K row, copied before reuse.
    pub bootstrap_topk_row: Option<Vec<u8>>,
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
            bootstrap_topk_row: None,
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
            self.bootstrap_topk_row
                .as_deref()
                .map_or_else(Bytes::new, Bytes::copy_from_slice),
        ]
    }

    /// Parse a status message; frames after the rank are optional because
    /// older senders omit bootstrap metadata and score payloads.
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
        let bootstrap_topk_row = match frames.get(7) {
            Some(frame) if !frame.is_empty() => {
                // Packed int32 rows hold a sampled ID, tail bits and K ID/logq pairs.
                if frame.len() < 16 || frame.len() % 8 != 0 {
                    return Err(error("the full-vocabulary Top-K row has invalid byte length"));
                }
                Some(frame.to_vec())
            }
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
            bootstrap_topk_row,
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
    fn cancellation_and_its_retirement_ack_have_distinct_strict_formats() {
        let request = cancellation_frames(7, "cache-a");
        assert_eq!(
            cancellation_route(&request).unwrap(),
            (7, "cache-a".to_owned())
        );
        let ack = cancellation_ack_frames(7, "cache-a", 2);
        assert_eq!(
            cancellation_ack(&ack).unwrap(),
            (7, "cache-a".to_owned(), 2)
        );
        assert!(cancellation_route(&ack).is_err());
        assert!(cancellation_ack(&request).is_err());
        assert!(cancellation_route(&cancellation_frames(7, "")).is_err());
        assert!(cancellation_ack(&cancellation_ack_frames(7, "cache-a", -1)).is_err());
        let mut invalid = request.clone();
        invalid[1] = Bytes::from_static(b"07");
        assert!(cancellation_route(&invalid).is_err());
    }

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
            bootstrap_topk_row: Some(vec![0; 16]),
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
        assert_eq!(parsed.bootstrap_topk_row, None);
    }

    #[test]
    fn routes_parse_from_well_formed_frames_only() {
        let frames = registration_frames(
            "10.0.0.1",
            4000,
            "10.0.0.1:5000",
            "cache-a",
            0x1000,
            b"L",
            4,
            2,
            7,
        );
        assert_eq!(
            registration_route(&frames),
            Ok(("10.0.0.1".to_owned(), 4000, "cache-a".to_owned()))
        );
        assert_eq!(frames[4].as_ref(), 0x1000u64.to_ne_bytes());
        assert!(registration_route(&frames[..8]).is_err());

        let frames = preallocation_frames(42, "cache-a", None);
        assert_eq!(preallocation_route(&frames), Ok((42, "cache-a".to_owned())));
        assert!(frames[2].is_empty());
        let bad = [ascii("042"), ascii("s"), Bytes::new()];
        assert!(preallocation_route(&bad).is_err());
        assert_eq!(session_port("[::1]:5000"), Some(5000));
    }

    #[test]
    fn a_registration_routes_by_cache_identity_even_when_its_transport_is_invalid() {
        let mut frames = registration_frames(
            "10.0.0.1",
            4000,
            "10.0.0.1:5000",
            "cache-b",
            0x1000,
            b"L",
            1,
            0,
            7,
        );
        frames[3] = Bytes::from_static(b"\xff");
        assert_eq!(
            registration_route(&frames),
            Ok(("10.0.0.1".to_owned(), 4000, "cache-b".to_owned()))
        );
        for identity in [
            Bytes::new(),
            Bytes::from_static(b"\xff"),
            ascii("x".repeat(513)),
        ] {
            frames[8] = identity.clone();
            assert!(registration_route(&frames).is_err());
            assert!(preallocation_route(&[ascii(1), identity, Bytes::new()]).is_err());
        }
    }

    #[test]
    fn a_registration_keeps_its_cache_identity_separate_from_its_transport() {
        use crate::layout::registration::{Preallocation, RegistrationInfo};

        let layout = crate::layout::contract::tests::wire();
        let mut frames = registration_frames(
            "10.0.0.1",
            4000,
            "10.0.0.1:5000",
            "cache-c",
            0x1000,
            layout.as_bytes(),
            1,
            0,
            7,
        );
        let registration = RegistrationInfo::from_frames(&frames).unwrap();
        assert_eq!(registration.session_id, "10.0.0.1:5000");
        assert_eq!(registration.cache_id, "cache-c");
        let allocation =
            Preallocation::from_frames(&preallocation_frames(42, "cache-c", None)).unwrap();
        assert_eq!(allocation.cache_id, registration.cache_id);
        for session in [
            Bytes::new(),
            Bytes::from_static(b"\xff"),
            ascii("x".repeat(513)),
        ] {
            frames[3] = session;
            assert_eq!(
                RegistrationInfo::from_frames(&frames).unwrap_err().0,
                "CachePD registration session is invalid"
            );
        }
    }
    #[test]
    fn a_cancellation_round_trips_and_is_never_a_preallocation() {
        let frames = cancellation_frames(42, "10.0.0.1:5000");
        assert!(is_cancellation(&frames));
        assert!(!is_registration(&frames));
        assert_eq!(
            cancellation_route(&frames),
            Ok((42, "10.0.0.1:5000".to_owned()))
        );
        // A pre-allocation starts with its room, a registration with "None":
        // neither is taken for a cancellation.
        let preallocation = preallocation_frames(42, "10.0.0.1:5000", None);
        assert!(!is_cancellation(&preallocation));
        assert!(cancellation_route(&preallocation).is_err());
        let registration = registration_frames(
            "10.0.0.1",
            4000,
            "10.0.0.1:5000",
            "cache",
            0x1000,
            b"L",
            4,
            2,
            7,
        );
        assert!(!is_cancellation(&registration));
        assert!(cancellation_route(&registration).is_err());
        assert!(!is_cancellation(&[]));
        assert!(cancellation_route(&[]).is_err());
    }

    #[test]
    fn a_malformed_cancellation_is_rejected() {
        let short = cancellation_frames(1, "s");
        assert!(cancellation_route(&short[..2]).is_err());
        let mut long = cancellation_frames(1, "s");
        long.push(Bytes::new());
        assert!(cancellation_route(&long).is_err());
        // The room is a canonical unsigned integer at most `MAX_ROOM`.
        let room = |frame: &[u8]| {
            cancellation_route(&[
                Bytes::from_static(b"Cancel"),
                Bytes::copy_from_slice(frame),
                ascii("s"),
            ])
        };
        assert!(room(b"01").is_err());
        assert!(room(b"-1").is_err());
        assert!(room(b"").is_err());
        assert!(room(b"9223372036854775808").is_err());
        assert_eq!(room(b"9223372036854775807"), Ok((MAX_ROOM, "s".to_owned())));
        // The Decode session must be a non-empty ASCII text frame.
        let session = |frame: &[u8]| {
            cancellation_route(&[
                Bytes::from_static(b"Cancel"),
                ascii(3),
                Bytes::copy_from_slice(frame),
            ])
        };
        assert!(session(b"").is_err());
        assert!(session(&[0xff]).is_err());
        assert!(session(&[b's'; 513]).is_err());
        assert_eq!(session(b"s"), Ok((3, "s".to_owned())));
        // Another header is a pre-allocation, not a cancellation.
        assert!(
            cancellation_route(&[Bytes::from_static(b"Cancel "), ascii(1), ascii("s")]).is_err()
        );
    }

    #[test]
    fn a_registration_carries_the_id_it_is_acknowledged_by() {
        let frames = registration_frames(
            "10.0.0.1",
            4000,
            "10.0.0.1:5000",
            "cache-a",
            0x1000,
            b"L",
            4,
            2,
            u64::MAX,
        );
        assert_eq!(frames.len(), 10);
        assert_eq!(registration_id(&frames), Ok(Some(u64::MAX)));
        // The nine frames of an older Decode rank still route, unacknowledged.
        assert_eq!(
            registration_route(&frames[..9]),
            Ok(("10.0.0.1".to_owned(), 4000, "cache-a".to_owned()))
        );
        assert_eq!(registration_id(&frames[..9]), Ok(None));
        let mut bad = frames.clone();
        for id in [b"07".as_slice(), b"", b"x", b"18446744073709551616"] {
            bad[9] = Bytes::copy_from_slice(id);
            assert!(registration_id(&bad).is_err());
        }
        let mut long = frames;
        long.push(Bytes::new());
        assert!(registration_route(&long).is_err());
    }

    #[test]
    fn a_registration_acknowledgment_round_trips() {
        let frames = registration_ack_frames(9, "cache-a", 3);
        assert!(is_registration_ack(&frames));
        assert!(!is_cancellation_ack(&frames) && !is_registration(&frames));
        assert!(!is_registration_reject(&frames));
        assert_eq!(registration_ack(&frames), Ok((9, "cache-a".to_owned(), 3)));
        let rejection = registration_reject_frames(9, "cache-a", 3, "bad layout");
        assert!(is_registration_reject(&rejection) && !is_registration_ack(&rejection));
        assert_eq!(
            registration_reject(&rejection),
            Ok((9, "cache-a".to_owned(), 3, "bad layout".to_owned()))
        );
        assert!(registration_reject(&rejection[..4]).is_err());
        assert!(registration_ack(&rejection[..4]).is_err());
        assert!(registration_ack(&frames[..3]).is_err());
        assert!(!is_registration_ack(&cancellation_ack_frames(
            9, "cache-a", 3
        )));
        for (index, frame) in [(1, b"09".as_slice()), (2, b""), (3, b"-1")] {
            let mut bad = frames.clone();
            bad[index] = Bytes::copy_from_slice(frame);
            assert!(registration_ack(&bad).is_err());
        }
    }
}

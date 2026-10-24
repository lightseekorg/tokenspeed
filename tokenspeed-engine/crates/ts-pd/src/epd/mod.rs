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

//! EPD: an encode node runs the vision tower and writes each multimodal
//! item's embedding into the prefill node's receive buffers over Mooncake.
//!
//! One item is one room. A prefill rank finds the encode rank it pairs with
//! through the encode node's bootstrap server (contiguous blocks of prefill
//! ranks share one encode rank), registers with it once, and per item tells
//! it where to write and how much ([`TransferInfo`]); the encode rank writes
//! once every prefill rank of its block asked, and pushes each a status.
//! With row sharding each prefill rank receives only its share of an
//! image's rows ([`shard_rows`]) and the ranks rebuild the rest among
//! themselves.
//!
//! The frames are those of the Python peers this module replaced.

#[cfg(test)]
mod live_tests;
pub mod pool;
pub mod receiver;
pub mod sender;

use bytes::Bytes;

use crate::{net::ascii, status::TransferPoll, wire::WireError};

/// The room a registration names in place of an item's.
const REGISTER_ROOM: &[u8] = b"None";

/// A balanced contiguous row shard of `span` rows over `size` ranks: this
/// rank's `(row_start, row_count)` (`shard_rows`). The first `span % size`
/// ranks get one extra row; the shards tile `[0, span)` in rank order. Both
/// ends of a transfer derive their geometry from this one function. A shard
/// may be empty when `span < size`.
pub fn shard_rows(span: i64, rank: i64, size: i64) -> (i64, i64) {
    let (base, rem) = (span / size, span % size);
    let start = rank * base + rank.min(rem);
    let count = base + i64::from(rank < rem);
    (start, count)
}

fn pointer(value: u64) -> Bytes {
    // struct.pack("Q"): native order.
    Bytes::copy_from_slice(&value.to_ne_bytes())
}

/// A prefill rank's one registration with an encode rank
/// (`EmbeddingArgsRegisterInfo`): where to push its statuses.
pub fn registration_frames(
    endpoint: &str,
    port: u16,
    session_id: &str,
    embedding_base: u64,
    deepstack_base: u64,
) -> Vec<Bytes> {
    vec![
        Bytes::from_static(REGISTER_ROOM),
        ascii(endpoint),
        ascii(port),
        ascii(session_id),
        pointer(embedding_base),
        pointer(deepstack_base),
    ]
}

/// Whether `frames` is a registration rather than a pre-allocation.
pub fn is_registration(frames: &[Bytes]) -> bool {
    frames
        .first()
        .is_some_and(|room| room.as_ref() == REGISTER_ROOM)
}

/// A prefill rank's request for one item's rows (`EmbeddingTransferInfo`):
/// where to write them and how many.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransferInfo {
    /// The item's room.
    pub room: u64,
    /// Where the prefill rank's status socket listens.
    pub endpoint: String,
    /// Its port.
    pub port: u16,
    /// Its transfer session.
    pub session_id: String,
    /// Where its rows land.
    pub embedding_ptr: u64,
    /// Where its deepstack rows land; 0 without deepstack.
    pub deepstack_ptr: u64,
    /// The rows it receives (its shard's).
    pub rows: i64,
    /// Elements per row.
    pub hidden: i64,
    /// The rows' dtype (`str(torch.dtype)`).
    pub dtype: String,
    /// It receives deepstack rows too.
    pub has_deepstack: bool,
    /// The prefill ranks the encode rank waits for before it writes.
    pub fanout: i64,
    /// Its shard's first row within the image.
    pub row_start: i64,
    /// The image's rows: the token-count check of both ends.
    pub span: i64,
}

impl TransferInfo {
    /// The request's frames.
    pub fn to_frames(&self) -> Vec<Bytes> {
        vec![
            ascii(self.room),
            ascii(&self.endpoint),
            ascii(self.port),
            ascii(&self.session_id),
            pointer(self.embedding_ptr),
            pointer(self.deepstack_ptr),
            ascii(self.rows),
            ascii(self.hidden),
            ascii(&self.dtype),
            Bytes::from_static(if self.has_deepstack { b"1" } else { b"0" }),
            ascii(self.fanout),
            ascii(self.row_start),
            ascii(self.span),
        ]
    }

    /// Parse a request (`EmbeddingTransferInfo.from_zmq`).
    pub fn from_frames(frames: &[Bytes]) -> Result<Self, WireError> {
        if frames.len() < 13 {
            return Err(WireError(format!(
                "malformed EmbeddingTransferInfo frame: {} parts < 13",
                frames.len()
            )));
        }
        let text = |index: usize, name: &str| {
            std::str::from_utf8(&frames[index])
                .ok()
                .filter(|text| text.is_ascii())
                .map(str::to_owned)
                .ok_or_else(|| WireError(format!("{name} must be ASCII")))
        };
        let int = |index: usize, name: &str| -> Result<i64, WireError> {
            text(index, name)?
                .parse()
                .map_err(|_| WireError(format!("{name} is not an integer")))
        };
        let ptr = |index: usize, name: &str| -> Result<u64, WireError> {
            let bytes: [u8; 8] = frames[index]
                .as_ref()
                .try_into()
                .map_err(|_| WireError(format!("{name} must be 8 bytes")))?;
            Ok(u64::from_ne_bytes(bytes))
        };
        Ok(Self {
            room: u64::try_from(int(0, "room")?)
                .map_err(|_| WireError("negative room".to_owned()))?,
            endpoint: text(1, "endpoint")?,
            port: u16::try_from(int(2, "port")?)
                .map_err(|_| WireError("invalid port".to_owned()))?,
            session_id: text(3, "session")?,
            embedding_ptr: ptr(4, "embedding pointer")?,
            deepstack_ptr: ptr(5, "deepstack pointer")?,
            rows: int(6, "rows")?,
            hidden: int(7, "hidden")?,
            dtype: text(8, "dtype")?,
            has_deepstack: frames[9].as_ref() == b"1",
            fanout: int(10, "fanout")?,
            row_start: int(11, "row_start")?,
            span: int(12, "span")?,
        })
    }
}

/// An encode rank's status of one room for one prefill rank: room, status,
/// the encode rank.
pub fn status_frames(room: u64, status: TransferPoll, rank: i64) -> Vec<Bytes> {
    vec![ascii(room), ascii(status as i64), ascii(rank)]
}

/// Parse a status: room, status, rank.
pub fn parse_status(frames: &[Bytes]) -> Result<(u64, TransferPoll, i64), WireError> {
    let [room, status, rank, ..] = frames else {
        return Err(WireError("an EPD status has 3 frames".to_owned()));
    };
    let int = |frame: &Bytes, name: &str| -> Result<i64, WireError> {
        std::str::from_utf8(frame)
            .ok()
            .and_then(|text| text.parse().ok())
            .ok_or_else(|| WireError(format!("{name} is not an integer")))
    };
    let room =
        u64::try_from(int(room, "room")?).map_err(|_| WireError("negative room".to_owned()))?;
    let status = TransferPoll::from_wire(int(status, "status")?)
        .ok_or_else(|| WireError("unknown status".to_owned()))?;
    Ok((room, status, int(rank, "rank")?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shards_tile_the_span_in_rank_order() {
        for (span, size) in [(10, 3), (2, 4), (0, 2), (7, 1), (16, 4)] {
            let mut next = 0;
            for rank in 0..size {
                let (start, count) = shard_rows(span, rank, size);
                assert_eq!(start, next, "span {span} size {size} rank {rank}");
                next += count;
            }
            assert_eq!(next, span);
        }
        assert_eq!(shard_rows(10, 0, 3), (0, 4));
        assert_eq!(shard_rows(10, 2, 3), (7, 3));
        assert_eq!(shard_rows(2, 3, 4), (2, 0));
    }

    #[test]
    fn a_transfer_request_round_trips() {
        let info = TransferInfo {
            room: 42,
            endpoint: "10.0.0.2".to_owned(),
            port: 7001,
            session_id: "10.0.0.2:15000".to_owned(),
            embedding_ptr: 0x7f00_0000_1000,
            deepstack_ptr: 0,
            rows: 3,
            hidden: 1024,
            dtype: "torch.bfloat16".to_owned(),
            has_deepstack: false,
            fanout: 2,
            row_start: 4,
            span: 7,
        };
        let frames = info.to_frames();
        assert_eq!(frames.len(), 13);
        assert!(!is_registration(&frames));
        assert_eq!(TransferInfo::from_frames(&frames), Ok(info));
        assert!(TransferInfo::from_frames(&frames[..12]).is_err());
        let registration = registration_frames("10.0.0.2", 7001, "s", 0, 0);
        assert!(is_registration(&registration));
        assert_eq!(registration.len(), 6);
        let status = status_frames(42, TransferPoll::Success, 1);
        assert_eq!(parse_status(&status), Ok((42, TransferPoll::Success, 1)));
    }
}

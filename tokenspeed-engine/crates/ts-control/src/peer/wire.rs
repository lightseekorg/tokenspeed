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

//! The messages between replicas (`docs/rfcs/0004-peer-prefix-sharing.md`):
//! a reader's requests to a holder, and the holder's replies, which carry
//! its incarnation and namespace identity so a reader can tell a restarted or
//! foreign holder from the one it read from. Replies are pushed back to the
//! reader a request names (a lease's reader is the reader's endpoint).

use ts_scheduler::{
    PeerAnswerKind, PeerAnswerView, PeerEventKind, PeerEventView, PeerLeaseView, PeerPageView,
};

use crate::{
    codec::{Reader, Writer},
    peer::geometry::{DeviceField, HostGroup, HostPool, RankGeometry},
};

/// A lookup's prompt: its full-page chain hashes and tokens up to its bound.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Prompt {
    /// The full pages' chain hashes below `max_tokens`.
    pub prefix_hashes: Vec<String>,
    /// The tokens up to `max_tokens`.
    pub tokens: Vec<i32>,
    /// The bound.
    pub max_tokens: i32,
}

/// A reader's request to a holder.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Request {
    /// The holder's per-rank geometry, pushed to `reply_to`.
    Layout {
        /// The reader's endpoint.
        reply_to: String,
    },
    /// A lookup opening `lease`.
    Lookup {
        /// The lease.
        lease: PeerLeaseView,
        /// The prompt.
        prompt: Prompt,
    },
    /// A pin of what the lookup answered.
    Pin {
        /// The lease.
        lease: PeerLeaseView,
    },
    /// The release of a lease, asking to demote after a complete read.
    Release {
        /// The lease.
        lease: PeerLeaseView,
        /// The reader read and kept the whole prefix.
        demote: bool,
    },
}

/// Who answered: the holder's name, incarnation and namespace identity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Stamp {
    /// The holder's name (its endpoint).
    pub name: String,
    /// Changes with every restart.
    pub incarnation: u64,
    /// The cache namespace (deployment identity and weight version).
    pub identity: String,
}

/// A holder's reply.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Reply {
    /// The holder's geometry, one entry per rank of its replica.
    Layout {
        /// Who answered.
        stamp: Stamp,
        /// Rank `i`'s geometry at index `i`.
        ranks: Vec<RankGeometry>,
    },
    /// An answer to a lookup, a pin or a release.
    Answer {
        /// Who answered.
        stamp: Stamp,
        /// The answer.
        answer: PeerAnswerView,
    },
}

/// A message that does not decode.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("a malformed peer message")]
pub struct Malformed;

const LAYOUT: u8 = 0;
const LOOKUP: u8 = 1;
const PIN: u8 = 2;
const RELEASE: u8 = 3;
const ANSWER: u8 = 4;

fn write_str(out: &mut Writer, value: &str) {
    out.bytes(value.as_bytes());
}

fn read_str(reader: &mut Reader<'_>) -> Option<String> {
    String::from_utf8(reader.bytes()?.to_vec()).ok()
}

fn write_lease(out: &mut Writer, lease: &PeerLeaseView) {
    write_str(out, &lease.reader);
    out.u64(lease.id);
}

fn read_lease(reader: &mut Reader<'_>) -> Option<PeerLeaseView> {
    Some(PeerLeaseView {
        reader: read_str(reader)?,
        id: reader.u64()?,
    })
}

fn write_pages(out: &mut Writer, pages: &[PeerPageView]) {
    out.count(pages.len());
    for page in pages {
        out.u32(page.group_id);
        out.i64(i64::from(page.slot));
        out.flag(page.host);
        out.i64(i64::from(page.block_id));
    }
}

fn read_i32(reader: &mut Reader<'_>) -> Option<i32> {
    i32::try_from(reader.i64()?).ok()
}

fn read_pages(reader: &mut Reader<'_>) -> Option<Vec<PeerPageView>> {
    let count = reader.count()?;
    let mut pages = Vec::with_capacity(count.min(1 << 16));
    for _ in 0..count {
        pages.push(PeerPageView {
            group_id: reader.u32()?,
            slot: read_i32(reader)?,
            host: reader.flag()?,
            block_id: read_i32(reader)?,
        });
    }
    Some(pages)
}

fn write_stamp(out: &mut Writer, stamp: &Stamp) {
    write_str(out, &stamp.name);
    out.u64(stamp.incarnation);
    write_str(out, &stamp.identity);
}

fn read_stamp(reader: &mut Reader<'_>) -> Option<Stamp> {
    Some(Stamp {
        name: read_str(reader)?,
        incarnation: reader.u64()?,
        identity: read_str(reader)?,
    })
}

impl RankGeometry {
    /// The geometry as a replica's members exchange it at startup.
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Writer::new();
        write_geometry(&mut out, self);
        out.finish()
    }

    /// What [`RankGeometry::encode`] wrote; `None` if it does not decode.
    pub fn decode(bytes: &[u8]) -> Option<Self> {
        let mut reader = Reader::new(bytes);
        let geometry = read_geometry(&mut reader)?;
        reader.finish(geometry)
    }
}

fn write_geometry(out: &mut Writer, geometry: &RankGeometry) {
    write_str(out, &geometry.session);
    out.count(geometry.device.len());
    for field in &geometry.device {
        out.u32(field.group_id);
        write_str(out, &field.field_id);
        out.u64(field.address);
        out.u64(field.page_stride);
        out.u64(field.bytes);
    }
    out.option(geometry.host.as_ref(), |out, pool| {
        out.u64(pool.address);
        out.count(pool.groups.len());
        for group in &pool.groups {
            out.u64(group.parent_bytes);
            out.u64(group.packing);
            out.u64(group.block_bytes);
            out.count(group.field_offsets.len());
            for (field, offset) in &group.field_offsets {
                write_str(out, field);
                out.u64(*offset);
            }
        }
    });
}

fn read_geometry(reader: &mut Reader<'_>) -> Option<RankGeometry> {
    let session = read_str(reader)?;
    let fields = reader.count()?;
    let mut device = Vec::with_capacity(fields.min(1 << 16));
    for _ in 0..fields {
        device.push(DeviceField {
            group_id: reader.u32()?,
            field_id: read_str(reader)?,
            address: reader.u64()?,
            page_stride: reader.u64()?,
            bytes: reader.u64()?,
        });
    }
    let host = reader.option(|reader| {
        let address = reader.u64()?;
        let count = reader.count()?;
        let mut groups = Vec::with_capacity(count.min(1 << 10));
        for _ in 0..count {
            let parent_bytes = reader.u64()?;
            let packing = reader.u64()?;
            let block_bytes = reader.u64()?;
            let num_fields = reader.count()?;
            let mut field_offsets = Vec::with_capacity(num_fields.min(1 << 16));
            for _ in 0..num_fields {
                field_offsets.push((read_str(reader)?, reader.u64()?));
            }
            groups.push(HostGroup {
                parent_bytes,
                packing,
                block_bytes,
                field_offsets,
            });
        }
        Some(HostPool { address, groups })
    })?;
    Some(RankGeometry {
        session,
        device,
        host,
    })
}

impl Request {
    /// The request's bytes.
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Writer::new();
        match self {
            Request::Layout { reply_to } => {
                out.u8(LAYOUT);
                write_str(&mut out, reply_to);
            }
            Request::Lookup { lease, prompt } => {
                out.u8(LOOKUP);
                write_lease(&mut out, lease);
                out.count(prompt.prefix_hashes.len());
                for hash in &prompt.prefix_hashes {
                    write_str(&mut out, hash);
                }
                out.i32s(&prompt.tokens);
                out.i64(i64::from(prompt.max_tokens));
            }
            Request::Pin { lease } => {
                out.u8(PIN);
                write_lease(&mut out, lease);
            }
            Request::Release { lease, demote } => {
                out.u8(RELEASE);
                write_lease(&mut out, lease);
                out.flag(*demote);
            }
        }
        out.finish()
    }

    /// A request off the wire.
    pub fn decode(bytes: &[u8]) -> Result<Self, Malformed> {
        let mut reader = Reader::new(bytes);
        let request = match reader.u8().ok_or(Malformed)? {
            LAYOUT => Request::Layout {
                reply_to: read_str(&mut reader).ok_or(Malformed)?,
            },
            LOOKUP => {
                let lease = read_lease(&mut reader).ok_or(Malformed)?;
                let count = reader.count().ok_or(Malformed)?;
                let mut prefix_hashes = Vec::with_capacity(count.min(1 << 20));
                for _ in 0..count {
                    prefix_hashes.push(read_str(&mut reader).ok_or(Malformed)?);
                }
                Request::Lookup {
                    lease,
                    prompt: Prompt {
                        prefix_hashes,
                        tokens: reader.i32s().ok_or(Malformed)?,
                        max_tokens: read_i32(&mut reader).ok_or(Malformed)?,
                    },
                }
            }
            PIN => Request::Pin {
                lease: read_lease(&mut reader).ok_or(Malformed)?,
            },
            RELEASE => Request::Release {
                lease: read_lease(&mut reader).ok_or(Malformed)?,
                demote: reader.flag().ok_or(Malformed)?,
            },
            _ => return Err(Malformed),
        };
        reader.finish(request).ok_or(Malformed)
    }
}

fn answer_kind(kind: PeerAnswerKind) -> u8 {
    match kind {
        PeerAnswerKind::Lookup => 0,
        PeerAnswerKind::Pin => 1,
        _ => 2,
    }
}

fn read_answer_kind(value: u8) -> Option<PeerAnswerKind> {
    Some(match value {
        0 => PeerAnswerKind::Lookup,
        1 => PeerAnswerKind::Pin,
        2 => PeerAnswerKind::Release,
        _ => return None,
    })
}

impl Reply {
    /// The reply's bytes.
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Writer::new();
        match self {
            Reply::Layout { stamp, ranks } => {
                out.u8(LAYOUT);
                write_stamp(&mut out, stamp);
                out.count(ranks.len());
                for rank in ranks {
                    write_geometry(&mut out, rank);
                }
            }
            Reply::Answer { stamp, answer } => {
                out.u8(ANSWER);
                write_stamp(&mut out, stamp);
                write_lease(&mut out, &answer.lease);
                out.u8(answer_kind(answer.kind));
                out.i64(i64::from(answer.num_tokens));
                write_pages(&mut out, &answer.pages);
                out.flag(answer.held);
            }
        }
        out.finish()
    }

    /// A reply off the wire.
    pub fn decode(bytes: &[u8]) -> Result<Self, Malformed> {
        let mut reader = Reader::new(bytes);
        let reply = match reader.u8().ok_or(Malformed)? {
            LAYOUT => {
                let stamp = read_stamp(&mut reader).ok_or(Malformed)?;
                let count = reader.count().ok_or(Malformed)?;
                let mut ranks = Vec::with_capacity(count.min(1 << 10));
                for _ in 0..count {
                    ranks.push(read_geometry(&mut reader).ok_or(Malformed)?);
                }
                Reply::Layout { stamp, ranks }
            }
            ANSWER => {
                let stamp = read_stamp(&mut reader).ok_or(Malformed)?;
                let lease = read_lease(&mut reader).ok_or(Malformed)?;
                let kind = reader.u8().and_then(read_answer_kind).ok_or(Malformed)?;
                Reply::Answer {
                    stamp,
                    answer: PeerAnswerView {
                        lease,
                        kind,
                        num_tokens: read_i32(&mut reader).ok_or(Malformed)?,
                        pages: read_pages(&mut reader).ok_or(Malformed)?,
                        held: reader.flag().ok_or(Malformed)?,
                    },
                }
            }
            _ => return Err(Malformed),
        };
        reader.finish(reply).ok_or(Malformed)
    }
}

const EVENT_LOOKUP: u8 = 0;
const EVENT_PIN: u8 = 1;
const EVENT_RELEASE: u8 = 2;

/// A holder's change (a lookup, pin or release event) as a report carries it.
pub(crate) fn write_change(out: &mut Writer, event: &PeerEventView) {
    out.u8(match event.kind {
        PeerEventKind::Lookup => EVENT_LOOKUP,
        PeerEventKind::Pin => EVENT_PIN,
        _ => EVENT_RELEASE,
    });
    write_lease(out, &event.lease);
    out.count(event.prefix_hashes.len());
    for hash in &event.prefix_hashes {
        write_str(out, hash);
    }
    out.i32s(&event.tokens);
    out.i64(i64::from(event.max_tokens));
    write_pages(out, &event.pages);
    out.flag(event.demote);
    out.i64(i64::from(event.num_tokens));
}

/// What [`write_change`] wrote.
pub(crate) fn read_change(reader: &mut Reader<'_>) -> Option<PeerEventView> {
    let kind = match reader.u8()? {
        EVENT_LOOKUP => PeerEventKind::Lookup,
        EVENT_PIN => PeerEventKind::Pin,
        EVENT_RELEASE => PeerEventKind::Release,
        _ => return None,
    };
    let lease = read_lease(reader)?;
    let count = reader.count()?;
    let mut prefix_hashes = Vec::with_capacity(count.min(1 << 20));
    for _ in 0..count {
        prefix_hashes.push(read_str(reader)?);
    }
    Some(PeerEventView {
        kind,
        lease,
        prefix_hashes,
        tokens: reader.i32s()?,
        max_tokens: read_i32(reader)?,
        pages: read_pages(reader)?,
        demote: reader.flag()?,
        num_tokens: read_i32(reader)?,
        request_id: String::new(),
    })
}

impl Prompt {
    /// The scheduler's lookup event for `lease` over this prompt.
    pub fn lookup_event(&self, lease: &PeerLeaseView) -> PeerEventView {
        PeerEventView {
            kind: PeerEventKind::Lookup,
            lease: lease.clone(),
            prefix_hashes: self.prefix_hashes.clone(),
            tokens: self.tokens.clone(),
            max_tokens: self.max_tokens,
            pages: Vec::new(),
            demote: false,
            num_tokens: 0,
            request_id: String::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lease() -> PeerLeaseView {
        PeerLeaseView {
            reader: "10.0.0.3:7100".into(),
            id: (7 << 32) + 1,
        }
    }

    #[test]
    fn every_request_survives_the_wire() {
        for request in [
            Request::Layout {
                reply_to: "10.0.0.3:7100".into(),
            },
            Request::Lookup {
                lease: lease(),
                prompt: Prompt {
                    prefix_hashes: vec!["ab".into(), "cd".into()],
                    tokens: vec![1, 2, 3, 4, 5],
                    max_tokens: 5,
                },
            },
            Request::Pin { lease: lease() },
            Request::Release {
                lease: lease(),
                demote: true,
            },
        ] {
            assert_eq!(Request::decode(&request.encode()), Ok(request));
        }
    }

    #[test]
    fn every_reply_survives_the_wire() {
        let stamp = Stamp {
            name: "10.0.0.2:7100".into(),
            incarnation: 42,
            identity: "ns-1".into(),
        };
        let geometry = RankGeometry {
            session: "10.0.0.2:12345".into(),
            device: vec![DeviceField {
                group_id: 1,
                field_id: "layer.3.k".into(),
                address: 1 << 40,
                page_stride: 4096,
                bytes: 2048,
            }],
            host: Some(HostPool {
                address: 1 << 41,
                groups: vec![HostGroup {
                    parent_bytes: 8192,
                    packing: 2,
                    block_bytes: 4096,
                    field_offsets: vec![("layer.3.k".into(), 0)],
                }],
            }),
        };
        for reply in [
            Reply::Layout {
                stamp: stamp.clone(),
                ranks: vec![
                    geometry.clone(),
                    RankGeometry {
                        host: None,
                        ..geometry.clone()
                    },
                ],
            },
            Reply::Answer {
                stamp,
                answer: PeerAnswerView {
                    lease: lease(),
                    kind: PeerAnswerKind::Lookup,
                    num_tokens: 128,
                    pages: vec![PeerPageView {
                        group_id: 0,
                        slot: 3,
                        host: true,
                        block_id: 99,
                    }],
                    held: false,
                },
            },
        ] {
            assert_eq!(Reply::decode(&reply.encode()), Ok(reply));
        }
        // As a replica's members exchange it at startup.
        let bytes = geometry.encode();
        assert_eq!(RankGeometry::decode(&bytes), Some(geometry));
        assert_eq!(RankGeometry::decode(&bytes[..bytes.len() - 1]), None);
    }

    #[test]
    fn a_cut_or_unknown_message_is_malformed() {
        let bytes = Request::Pin { lease: lease() }.encode();
        assert_eq!(Request::decode(&bytes[..bytes.len() - 1]), Err(Malformed));
        assert_eq!(Request::decode(&[9]), Err(Malformed));
        assert_eq!(Reply::decode(&[]), Err(Malformed));
    }
}

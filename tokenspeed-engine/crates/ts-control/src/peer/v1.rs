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

//! Versioned messages for checkpoint-bound peer reads.
use super::wire::Prompt;
use crate::codec::{Reader, Writer};
use std::collections::BTreeSet;
use ts_scheduler::{PeerPageView, PeerTier};

/// SHA-256 of the canonical representation agreed by the endpoints.
pub type Digest = [u8; 32];
// Earlier readers may submit an unbounded time after PIN. Never grant them
// an offer which the holder can reclaim on a deadline.
const VERSION: u8 = 2;
pub(super) const SUBMIT_WINDOW: std::time::Duration = std::time::Duration::from_secs(30);
const MAX_FRAME: usize = 16 << 20;
const MAX_TOKENS: usize = 1 << 20;
const MAX_PAGES: usize = 1 << 16;

/// A specific generation of a replica. The name includes its DP rank.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct Generation {
    /// Public reply address.
    pub name: String,
    /// Process generation.
    pub incarnation: u64,
    /// Registered layout generation within this process.
    pub epoch: u64,
}

/// The identity of one attempt. Retries use the same key.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct Key {
    /// Replica that owns the destination.
    pub reader: Generation,
    /// Replica that owns the source.
    pub holder: Generation,
    /// Monotonic reader lease id.
    pub lease: u64,
}

/// A holder-issued offer with its exact checkpoint and canonical page digest.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Offer {
    /// Never reused within a holder incarnation.
    pub id: u64,
    /// Cacheable tokens at the offered checkpoint.
    pub end: i32,
    /// Hash of the immutable source manifest.
    pub manifest: Digest,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Sticky release disposition.
pub enum Disposition {
    /// All readers retired successfully.
    Complete,
    /// Reader abandoned this offer.
    Abort,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Result of a close request at the holder.
pub enum Closure {
    /// All references removed with this disposition.
    Closed,
    /// Issued and retired beyond the ACK replay window.
    Gone,
    /// No evidence of closure.
    Unknown,
    /// Holder generation changed.
    Stale,
    /// Original disposition or offer differs.
    Conflict,
}

/// One message after DP dispatch. Layout describes registered members; role and
/// capability bits are carried inside the immutable versioned descriptor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Message {
    /// Probe for a one- or eight-member source usable by a TP8 P reader.
    LayoutReq(u64, String),
    /// Query id, holder generation, semantic digest, layout digest, descriptor.
    LayoutReply(u64, Generation, Digest, Digest, Vec<u8>),
    /// Query bound to semantics, reader layout and holder layout.
    Lookup(Key, Digest, Digest, Digest, Prompt),
    /// None is a miss; a hit binds the prefix identity and full source manifest.
    Offered(Key, Option<(Offer, Digest, Vec<PeerPageView>)>),
    /// Acquire exactly this offer.
    Pin(Key, Offer),
    /// A grant requires every source member to have pinned.
    Grant(Key, Offer, bool),
    /// Terminal reader disposition.
    Release(Key, u64, Disposition),
    /// Closure evidence from the original holder.
    Ack(Key, u64, Disposition, Closure),
}

fn string(r: &mut Reader<'_>, max: usize) -> Option<String> {
    let b = r.bytes()?;
    if b.is_empty() || b.len() > max {
        return None;
    }
    Some(std::str::from_utf8(b).ok()?.to_owned())
}
fn nonzero(r: &mut Reader<'_>) -> Option<u64> {
    r.u64().filter(|v| *v != 0)
}
impl Generation {
    fn put(&self, w: &mut Writer) {
        w.bytes(self.name.as_bytes());
        w.u64(self.incarnation);
        w.u64(self.epoch);
    }
    fn get(r: &mut Reader<'_>) -> Option<Self> {
        Some(Self {
            name: string(r, 512)?,
            incarnation: nonzero(r)?,
            epoch: nonzero(r)?,
        })
    }
}
impl Key {
    pub(super) fn put(&self, w: &mut Writer) {
        self.reader.put(w);
        w.u64(self.lease);
        self.holder.put(w);
    }
    pub(super) fn get(r: &mut Reader<'_>) -> Option<Self> {
        Some(Self {
            reader: Generation::get(r)?,
            lease: nonzero(r)?,
            holder: Generation::get(r)?,
        })
    }
}
impl Offer {
    fn put(&self, w: &mut Writer) {
        w.u64(self.id);
        w.i64(i64::from(self.end));
        w.fixed(&self.manifest);
    }
    fn get(r: &mut Reader<'_>) -> Option<Self> {
        let id = nonzero(r)?;
        let end = i32::try_from(r.i64()?).ok()?;
        if !(1..=MAX_TOKENS as i32).contains(&end) {
            return None;
        }
        Some(Self {
            id,
            end,
            manifest: r.array()?,
        })
    }
}

pub(super) fn put_pages(w: &mut Writer, pages: &[PeerPageView]) {
    w.count(pages.len());
    for p in pages {
        w.u32(p.group_id);
        w.i64(i64::from(p.slot));
        w.u8(if p.tier == PeerTier::Device {
            0
        } else if p.tier == PeerTier::Host {
            1
        } else {
            2
        });
        w.i64(i64::from(p.block_id));
    }
}
fn pages(r: &mut Reader<'_>) -> Option<Vec<PeerPageView>> {
    let n = r.count()?;
    if n == 0 || n > MAX_PAGES {
        return None;
    }
    let mut seen = BTreeSet::new();
    let mut result = Vec::new();
    for _ in 0..n {
        let group_id = r.u32()?;
        let slot = i32::try_from(r.i64()?).ok()?;
        let tier = match r.u8()? {
            0 => PeerTier::Device,
            1 => PeerTier::Host,
            _ => return None,
        };
        let block_id = i32::try_from(r.i64()?).ok()?;
        if slot < 0 || block_id <= 0 || !seen.insert((group_id, slot)) {
            return None;
        }
        result.push(PeerPageView {
            group_id,
            slot,
            tier,
            block_id,
        });
    }
    Some(result)
}
fn disposition(r: &mut Reader<'_>) -> Option<Disposition> {
    match r.u8()? {
        0 => Some(Disposition::Complete),
        1 => Some(Disposition::Abort),
        _ => None,
    }
}
impl Message {
    /// The existing two-frame DP transport uses r for replies and q for requests.
    pub fn reply(&self) -> bool {
        matches!(
            self,
            Self::LayoutReply(..) | Self::Offered(..) | Self::Grant(..) | Self::Ack(..)
        )
    }
    /// Canonical little-endian encoding, beginning with the tag and version.
    pub fn encode(&self) -> Vec<u8> {
        use Message::*;
        let mut w = Writer::new();
        let tag = match self {
            LayoutReq(..) => 0x10,
            LayoutReply(..) => 0x11,
            Lookup(..) => 0x12,
            Offered(..) => 0x13,
            Pin(..) => 0x14,
            Grant(..) => 0x15,
            Release(..) => 0x17,
            Ack(..) => 0x18,
        };
        w.u8(tag);
        w.u8(VERSION);
        match self {
            LayoutReq(query, name) => {
                w.u64(*query);
                w.bytes(name.as_bytes());
                w.u8(8);
            }
            LayoutReply(query, g, s, l, descriptor) => {
                w.u64(*query);
                g.put(&mut w);
                w.fixed(s);
                w.fixed(l);
                w.bytes(descriptor);
            }
            Lookup(k, s, rl, hl, p) => {
                k.put(&mut w);
                w.fixed(s);
                w.fixed(rl);
                w.fixed(hl);
                w.u8(8);
                w.i64(i64::from(p.max_tokens));
                w.count(p.prefix_hashes.len());
                for h in &p.prefix_hashes {
                    w.bytes(h.as_bytes());
                }
                w.i32s(&p.tokens);
            }
            Offered(k, o) => {
                k.put(&mut w);
                w.option(o.as_ref(), |w, (o, prefix, pages)| {
                    o.put(w);
                    w.fixed(prefix);
                    put_pages(w, pages);
                });
            }
            Pin(k, o) => {
                k.put(&mut w);
                o.put(&mut w);
            }
            Grant(k, o, ok) => {
                k.put(&mut w);
                o.put(&mut w);
                w.flag(*ok);
            }
            Release(k, o, d) => {
                k.put(&mut w);
                w.u64(*o);
                w.u8(*d as u8);
            }
            Ack(k, o, d, c) => {
                k.put(&mut w);
                w.u64(*o);
                w.u8(*d as u8);
                w.u8(*c as u8);
            }
        }
        w.finish()
    }
    /// Decode bounded fields and require a complete frame, including its direction.
    pub fn decode(payload: &[u8], reply: bool) -> Option<Self> {
        if payload.len() > MAX_FRAME {
            return None;
        }
        let mut r = Reader::new(payload);
        let tag = r.u8()?;
        if r.u8()? != VERSION {
            return None;
        }
        let msg = match tag {
            0x10 => {
                let id = nonzero(&mut r)?;
                let name = string(&mut r, 512)?;
                if r.u8()? != 8 {
                    return None;
                }
                Self::LayoutReq(id, name)
            }
            0x11 => {
                let id = nonzero(&mut r)?;
                let gen = Generation::get(&mut r)?;
                let s = r.array()?;
                let l = r.array()?;
                let d = r.bytes()?;
                if d.is_empty() {
                    return None;
                }
                Self::LayoutReply(id, gen, s, l, d.to_vec())
            }
            0x12 => {
                let k = Key::get(&mut r)?;
                let s = r.array()?;
                let rl = r.array()?;
                let hl = r.array()?;
                if r.u8()? != 8 {
                    return None;
                }
                let max_tokens = i32::try_from(r.i64()?).ok()?;
                if !(1..=MAX_TOKENS as i32).contains(&max_tokens) {
                    return None;
                }
                let n = r.count()?;
                if n > max_tokens as usize {
                    return None;
                }
                let mut prefix_hashes = Vec::new();
                for _ in 0..n {
                    prefix_hashes.push(string(&mut r, 128)?);
                }
                let tokens = r.i32s()?;
                if tokens.len() != max_tokens as usize {
                    return None;
                }
                Self::Lookup(
                    k,
                    s,
                    rl,
                    hl,
                    Prompt {
                        prefix_hashes,
                        tokens,
                        max_tokens,
                    },
                )
            }
            0x13 => Self::Offered(
                Key::get(&mut r)?,
                r.option(|r| Some((Offer::get(r)?, r.array()?, pages(r)?)))?,
            ),
            0x14 => Self::Pin(Key::get(&mut r)?, Offer::get(&mut r)?),
            0x15 => Self::Grant(Key::get(&mut r)?, Offer::get(&mut r)?, r.flag()?),
            0x17 => Self::Release(Key::get(&mut r)?, nonzero(&mut r)?, disposition(&mut r)?),
            0x18 => Self::Ack(
                Key::get(&mut r)?,
                nonzero(&mut r)?,
                disposition(&mut r)?,
                match r.u8()? {
                    0 => Closure::Closed,
                    1 => Closure::Gone,
                    2 => Closure::Unknown,
                    3 => Closure::Stale,
                    4 => Closure::Conflict,
                    _ => return None,
                },
            ),
            _ => return None,
        };
        if msg.reply() != reply {
            return None;
        }
        r.finish(msg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generation(name: &str) -> Generation {
        Generation {
            name: name.to_owned(),
            incarnation: 2,
            epoch: 1,
        }
    }

    #[test]
    fn all_tags_reject_truncation_tail_wrong_version_and_wrong_direction() {
        let key = Key {
            reader: generation("r:12"),
            holder: generation("h:34"),
            lease: 5,
        };
        let offer = Offer {
            id: 6,
            end: 1,
            manifest: [8; 32],
        };
        let page = PeerPageView {
            group_id: 0,
            slot: 0,
            tier: PeerTier::Host,
            block_id: 3,
        };
        let frames = [
            Message::LayoutReq(1, "r:12".into()),
            Message::LayoutReply(1, key.holder.clone(), [2; 32], [3; 32], vec![0]),
            Message::Lookup(
                key.clone(),
                [1; 32],
                [2; 32],
                [3; 32],
                Prompt {
                    prefix_hashes: vec![],
                    tokens: vec![7],
                    max_tokens: 1,
                },
            ),
            Message::Offered(key.clone(), Some((offer.clone(), [4; 32], vec![page]))),
            Message::Pin(key.clone(), offer.clone()),
            Message::Grant(key.clone(), offer, true),
            Message::Release(key.clone(), 6, Disposition::Abort),
            Message::Ack(key, 6, Disposition::Abort, Closure::Closed),
        ];
        for (tag, m) in [0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x17, 0x18]
            .into_iter()
            .zip(&frames)
        {
            let mut bytes = m.encode();
            assert_eq!(bytes[..2], [tag, 2]);
            assert_eq!(Message::decode(&bytes, m.reply()).as_ref(), Some(m));
            assert!(Message::decode(&bytes, !m.reply()).is_none());
            for end in 0..bytes.len() {
                assert!(Message::decode(&bytes[..end], m.reply()).is_none());
            }
            bytes.push(0);
            assert!(Message::decode(&bytes, m.reply()).is_none());
            bytes.pop();
            bytes[1] = 1;
            assert!(Message::decode(&bytes, m.reply()).is_none());
        }
    }
    #[test]
    fn discovery_bytes_are_stable_and_never_decode_as_a_legacy_request() {
        let query = Message::LayoutReq(1, "a:1".into());
        let golden = [
            0x10, 2, 1, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, b'a', b':', b'1', 8,
        ];
        assert_eq!(query.encode(), golden);
        assert!(super::super::wire::Request::decode(&golden).is_err());
    }
}

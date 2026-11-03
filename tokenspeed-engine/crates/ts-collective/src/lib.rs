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

//! CPU collectives among the ranks of an engine: what the control plane's
//! gloo groups do today, without the GIL.
//!
//! Every pair of ranks shares one TCP connection. A collective blocks until
//! it completes, and every member of a group must run the same collectives
//! on it in the same order: the rank-identical ordering the control plane
//! keeps anyway (`docs/design/event-loop.md`). A reader thread per
//! connection drains the socket into a queue, so a send never waits on the
//! peer reading; a collective sends this rank's part to the members that
//! need it, then takes each part it needs from its sender's queue.
//!
//! Each frame names its collective's kind, group and sequence number on that
//! group, so a member that took another path fails loudly instead of reading
//! another collective's bytes. A peer that exits fails every collective
//! waiting on it.

use std::{
    collections::HashMap,
    io::{self, Read, Write},
    net::{Shutdown, TcpListener, TcpStream, ToSocketAddrs},
    sync::mpsc,
    thread,
    time::{Duration, Instant},
};

/// What a mesh connection says first: who is calling.
const HELLO_MAGIC: u32 = 0x5453_434d; // "TSCM"

/// A collective failure; the ranks can no longer agree.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A socket operation failed.
    #[error("collective I/O: {0}")]
    Io(#[from] io::Error),
    /// The peers did not all connect in time.
    #[error("the mesh did not connect within {0:?}")]
    Timeout(Duration),
    /// A connection did not introduce itself as a rank of this mesh.
    #[error("bad mesh handshake: {0}")]
    Handshake(String),
    /// The ranks do not form a valid group.
    #[error("invalid group: {0}")]
    InvalidGroup(String),
    /// A peer's connection closed.
    #[error("rank {0} left the mesh")]
    PeerClosed(usize),
    /// A peer sent a frame of another collective.
    #[error("rank {rank} is out of step: expected {expected:?}, received {received:?}")]
    Desync {
        /// The peer.
        rank: usize,
        /// The collective this rank is in.
        expected: Tag,
        /// The collective the peer sent.
        received: Tag,
    },
    /// The members passed reductions of different lengths.
    #[error("rank {rank} reduced {found} values, this rank {expected}")]
    LengthMismatch {
        /// The peer.
        rank: usize,
        /// Values this rank passed.
        expected: usize,
        /// Values the peer passed.
        found: usize,
    },
}

/// Which collective a frame belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tag {
    /// The collective's kind.
    pub kind: u8,
    /// Its group.
    pub group: u32,
    /// Its position among the group's collectives.
    pub sequence: u64,
}

const BROADCAST: u8 = 1;
const ALL_GATHER: u8 = 2;

/// An element-wise reduction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reduce {
    /// The largest value.
    Max,
    /// The smallest value.
    Min,
    /// The sum.
    Sum,
}

/// A set of ranks that run collectives together, as one rank sees it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Group {
    id: u32,
    ranks: Vec<usize>,
    index: usize,
}

impl Group {
    /// The group `id` of `ranks` (global ranks, in member order), from
    /// `rank`'s side. Every member must use the same id for it, and no two
    /// groups the same id.
    pub fn new(id: u32, ranks: Vec<usize>, rank: usize) -> Result<Self, Error> {
        let mut sorted = ranks.clone();
        sorted.sort_unstable();
        sorted.dedup();
        if sorted.len() != ranks.len() {
            return Err(Error::InvalidGroup(format!("{ranks:?} repeats a rank")));
        }
        let index = ranks
            .iter()
            .position(|member| *member == rank)
            .ok_or_else(|| Error::InvalidGroup(format!("rank {rank} is not in {ranks:?}")))?;
        Ok(Self { id, ranks, index })
    }

    /// The group's members, as global ranks.
    pub fn ranks(&self) -> &[usize] {
        &self.ranks
    }

    /// Members.
    pub fn size(&self) -> usize {
        self.ranks.len()
    }

    /// This rank's position among the members.
    pub fn index(&self) -> usize {
        self.index
    }
}

/// This rank's listening socket, bound before the ranks exchange addresses.
pub struct Listener(TcpListener);

impl Listener {
    /// Bind an ephemeral port on `host`.
    pub fn bind(host: &str) -> io::Result<Self> {
        TcpListener::bind((host, 0)).map(Self)
    }

    /// The bound port, for the address the other ranks dial.
    pub fn port(&self) -> io::Result<u16> {
        Ok(self.0.local_addr()?.port())
    }
}

type Frame = (Tag, Vec<u8>);

struct Peer {
    stream: TcpStream,
    frames: mpsc::Receiver<io::Result<Frame>>,
    reader: Option<thread::JoinHandle<()>>,
}

/// One rank's connections to every other rank.
pub struct Mesh {
    rank: usize,
    peers: Vec<Option<Peer>>,
    sequences: HashMap<u32, u64>,
}

fn write_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn read_u32(stream: &mut impl Read) -> io::Result<u32> {
    let mut bytes = [0; 4];
    stream.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_u64(stream: &mut impl Read) -> io::Result<u64> {
    let mut bytes = [0; 8];
    stream.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

fn read_frame(stream: &mut impl Read) -> io::Result<Frame> {
    let mut kind = [0; 1];
    stream.read_exact(&mut kind)?;
    let group = read_u32(stream)?;
    let sequence = read_u64(stream)?;
    let len = usize::try_from(read_u64(stream)?)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "frame too long"))?;
    let mut payload = vec![0; len];
    stream.read_exact(&mut payload)?;
    let tag = Tag {
        kind: kind[0],
        group,
        sequence,
    };
    Ok((tag, payload))
}

fn frame_header(tag: Tag, len: usize) -> Vec<u8> {
    let mut header = Vec::with_capacity(21);
    header.push(tag.kind);
    write_u32(&mut header, tag.group);
    header.extend_from_slice(&tag.sequence.to_le_bytes());
    header.extend_from_slice(&(len as u64).to_le_bytes());
    header
}

fn hello(rank: usize, world: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(12);
    write_u32(&mut out, HELLO_MAGIC);
    write_u32(&mut out, rank as u32);
    write_u32(&mut out, world as u32);
    out
}

fn remaining(deadline: Instant, timeout: Duration) -> Result<Duration, Error> {
    deadline
        .checked_duration_since(Instant::now())
        .filter(|left| !left.is_zero())
        .ok_or(Error::Timeout(timeout))
}

impl Mesh {
    /// Connect `rank` to every other rank of `addresses` (`host:port`, one
    /// per rank, in rank order). Every rank calls this; it returns once
    /// every connection is up, so it is also a barrier.
    pub fn connect(
        rank: usize,
        listener: Listener,
        addresses: &[String],
        timeout: Duration,
    ) -> Result<Self, Error> {
        let world = addresses.len();
        if rank >= world {
            return Err(Error::InvalidGroup(format!(
                "rank {rank} of a {world}-rank mesh"
            )));
        }
        let deadline = Instant::now() + timeout;
        let mut streams: Vec<Option<TcpStream>> = (0..world).map(|_| None).collect();

        // Each rank dials the lower ranks and accepts the higher ones.
        for (peer, address) in addresses.iter().enumerate().take(rank) {
            let stream = Self::dial(address, deadline, timeout)?;
            (&stream).write_all(&hello(rank, world))?;
            streams[peer] = Some(stream);
        }
        let listener = listener.0;
        listener.set_nonblocking(true)?;
        let mut expected = world - rank - 1;
        while expected > 0 {
            match listener.accept() {
                Ok((stream, _)) => {
                    stream.set_nonblocking(false)?;
                    stream.set_read_timeout(Some(remaining(deadline, timeout)?))?;
                    let mut reader = &stream;
                    let (magic, peer, peer_world) = (
                        read_u32(&mut reader)?,
                        read_u32(&mut reader)? as usize,
                        read_u32(&mut reader)? as usize,
                    );
                    if magic != HELLO_MAGIC || peer_world != world || peer <= rank || peer >= world
                    {
                        return Err(Error::Handshake(format!(
                            "magic {magic:#x} rank {peer} of {peer_world} at rank {rank} of {world}"
                        )));
                    }
                    if streams[peer].is_some() {
                        return Err(Error::Handshake(format!("rank {peer} connected twice")));
                    }
                    stream.set_read_timeout(None)?;
                    streams[peer] = Some(stream);
                    expected -= 1;
                }
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                    remaining(deadline, timeout)?;
                    thread::sleep(Duration::from_millis(5));
                }
                Err(error) => return Err(error.into()),
            }
        }

        let mut peers = Vec::with_capacity(world);
        for (peer, stream) in streams.into_iter().enumerate() {
            let Some(stream) = stream else {
                peers.push(None);
                continue;
            };
            stream.set_nodelay(true)?;
            let mut reading = stream.try_clone()?;
            let (sender, frames) = mpsc::channel();
            let reader = thread::Builder::new()
                .name(format!("ts-mesh-{peer}"))
                .spawn(move || loop {
                    let frame = read_frame(&mut reading);
                    let failed = frame.is_err();
                    if sender.send(frame).is_err() || failed {
                        return;
                    }
                })?;
            peers.push(Some(Peer {
                stream,
                frames,
                reader: Some(reader),
            }));
        }
        tracing::info!(rank, world, "collective mesh connected");
        Ok(Self {
            rank,
            peers,
            sequences: HashMap::new(),
        })
    }

    fn dial(address: &str, deadline: Instant, timeout: Duration) -> Result<TcpStream, Error> {
        let resolved: Vec<_> = address.to_socket_addrs()?.collect();
        let target = *resolved
            .first()
            .ok_or_else(|| Error::Handshake(format!("{address} resolves to nothing")))?;
        loop {
            match TcpStream::connect_timeout(&target, remaining(deadline, timeout)?) {
                Ok(stream) => return Ok(stream),
                // The peer bound its listener before publishing the address,
                // so a refusal is transient only while it starts up.
                Err(error) if error.kind() == io::ErrorKind::ConnectionRefused => {
                    thread::sleep(Duration::from_millis(20));
                }
                Err(error) => return Err(error.into()),
            }
        }
    }

    /// This rank.
    pub fn rank(&self) -> usize {
        self.rank
    }

    fn next_tag(&mut self, kind: u8, group: &Group) -> Tag {
        let sequence = self.sequences.entry(group.id).or_insert(0);
        *sequence += 1;
        Tag {
            kind,
            group: group.id,
            sequence: *sequence,
        }
    }

    fn peer(&mut self, rank: usize) -> &mut Peer {
        self.peers[rank]
            .as_mut()
            .expect("a member other than this rank")
    }

    fn send(&mut self, rank: usize, tag: Tag, payload: &[u8]) -> Result<(), Error> {
        let header = frame_header(tag, payload.len());
        let stream = &mut self.peer(rank).stream;
        stream
            .write_all(&header)
            .and_then(|()| stream.write_all(payload))
            .map_err(|error| match error.kind() {
                io::ErrorKind::BrokenPipe | io::ErrorKind::ConnectionReset => {
                    Error::PeerClosed(rank)
                }
                _ => Error::Io(error),
            })
    }

    fn receive(&mut self, rank: usize, expected: Tag) -> Result<Vec<u8>, Error> {
        let frame = self
            .peer(rank)
            .frames
            .recv()
            .map_err(|_| Error::PeerClosed(rank))?;
        let (received, payload) = frame.map_err(|error| match error.kind() {
            io::ErrorKind::UnexpectedEof | io::ErrorKind::ConnectionReset => {
                Error::PeerClosed(rank)
            }
            _ => Error::Io(error),
        })?;
        if received != expected {
            return Err(Error::Desync {
                rank,
                expected,
                received,
            });
        }
        Ok(payload)
    }

    /// The payload of the member at `root` (an index into the group), on
    /// every member. Members other than the root pass any payload; it is
    /// ignored.
    pub fn broadcast(
        &mut self,
        group: &Group,
        root: usize,
        payload: Vec<u8>,
    ) -> Result<Vec<u8>, Error> {
        if group.size() == 1 {
            return Ok(payload);
        }
        let tag = self.next_tag(BROADCAST, group);
        if group.index() == root {
            for (index, &member) in group.ranks().iter().enumerate() {
                if index != root {
                    self.send(member, tag, &payload)?;
                }
            }
            Ok(payload)
        } else {
            self.receive(group.ranks()[root], tag)
        }
    }

    /// Every member's payload, in member order.
    pub fn all_gather(&mut self, group: &Group, payload: &[u8]) -> Result<Vec<Vec<u8>>, Error> {
        if group.size() == 1 {
            return Ok(vec![payload.to_vec()]);
        }
        let tag = self.next_tag(ALL_GATHER, group);
        for (index, &member) in group.ranks().iter().enumerate() {
            if index != group.index() {
                self.send(member, tag, payload)?;
            }
        }
        let mut gathered = Vec::with_capacity(group.size());
        for (index, &member) in group.ranks().iter().enumerate() {
            if index == group.index() {
                gathered.push(payload.to_vec());
            } else {
                gathered.push(self.receive(member, tag)?);
            }
        }
        Ok(gathered)
    }

    /// `values` reduced element-wise across the members.
    pub fn all_reduce(
        &mut self,
        group: &Group,
        values: &[i64],
        op: Reduce,
    ) -> Result<Vec<i64>, Error> {
        let mut rows = self.all_gather_i64(group, values)?.into_iter();
        let mut reduced = rows.next().expect("a group has a member");
        for row in rows {
            for (out, value) in reduced.iter_mut().zip(row) {
                *out = match op {
                    Reduce::Max => (*out).max(value),
                    Reduce::Min => (*out).min(value),
                    Reduce::Sum => *out + value,
                };
            }
        }
        Ok(reduced)
    }

    /// Every member's `values`, in member order; all of the same length.
    pub fn all_gather_i64(
        &mut self,
        group: &Group,
        values: &[i64],
    ) -> Result<Vec<Vec<i64>>, Error> {
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let gathered = self.all_gather(group, &bytes)?;
        let mut rows = Vec::with_capacity(gathered.len());
        for (index, row) in gathered.into_iter().enumerate() {
            if row.len() != bytes.len() {
                return Err(Error::LengthMismatch {
                    rank: group.ranks()[index],
                    expected: values.len(),
                    found: row.len() / 8,
                });
            }
            rows.push(
                row.chunks_exact(8)
                    .map(|chunk| i64::from_le_bytes(chunk.try_into().expect("eight bytes")))
                    .collect(),
            );
        }
        Ok(rows)
    }
}

impl Drop for Mesh {
    fn drop(&mut self) {
        for peer in self.peers.iter_mut().flatten() {
            let _ = peer.stream.shutdown(Shutdown::Both);
            if let Some(reader) = peer.reader.take() {
                let _ = reader.join();
            }
        }
    }
}

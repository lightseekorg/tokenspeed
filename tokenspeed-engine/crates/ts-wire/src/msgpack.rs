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

//! msgpack primitives that make msgspec's encoding choices.
//!
//! The Python side encodes with `msgspec`, which always picks the smallest
//! integer, string, binary, array, map and ext format and writes every float
//! as float64. [`Encoder`] makes the same choices, so a message decoded and
//! re-encoded here is byte-identical to the Python encoding. [`Decoder`]
//! accepts every msgpack format, as msgspec's decoder does.
//!
//! A message is a list of ZMQ frames: frame 0 is the msgpack buffer, and
//! large tensors ride in further frames that the buffer references by index
//! (see [`crate::tensor`]).

use bytes::Bytes;

use crate::error::{Error, Result};

/// Tensors smaller than this many bytes are inlined in the msgpack buffer;
/// larger ones become their own frame. Matches `MSGPACK_ZERO_COPY_THRESHOLD`.
pub const INLINE_THRESHOLD: usize = 256;

/// Writes one message: the msgpack buffer plus out-of-band frames.
#[derive(Debug)]
pub struct Encoder {
    buf: Vec<u8>,
    frames: Vec<Bytes>,
    inline_threshold: usize,
}

impl Default for Encoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Encoder {
    /// An encoder with the Python side's inline threshold.
    pub fn new() -> Self {
        Self::with_inline_threshold(INLINE_THRESHOLD)
    }

    /// An encoder that frames out buffers of at least `inline_threshold` bytes.
    pub fn with_inline_threshold(inline_threshold: usize) -> Self {
        Self {
            buf: Vec::new(),
            frames: Vec::new(),
            inline_threshold,
        }
    }

    /// Finish the message: the msgpack buffer, then the out-of-band frames.
    pub fn into_frames(self) -> Vec<Bytes> {
        let mut frames = Vec::with_capacity(1 + self.frames.len());
        frames.push(Bytes::from(self.buf));
        frames.extend(self.frames);
        frames
    }

    /// Finish a message that has no out-of-band frames.
    ///
    /// # Panics
    ///
    /// If a buffer was framed out: the caller would drop it from the wire.
    pub fn into_single_frame(self) -> Vec<u8> {
        assert!(
            self.frames.is_empty(),
            "message has {} out-of-band frames; use into_frames",
            self.frames.len()
        );
        self.buf
    }

    /// Write `nil`.
    pub fn nil(&mut self) {
        self.buf.push(0xc0);
    }

    /// Write a boolean.
    pub fn bool(&mut self, value: bool) {
        self.buf.push(if value { 0xc3 } else { 0xc2 });
    }

    /// Write a signed integer: non-negative values use the unsigned formats.
    pub fn int(&mut self, value: i64) {
        if value >= 0 {
            self.uint(value as u64);
        } else if value >= -32 {
            self.buf.push(value as u8);
        } else if value >= i64::from(i8::MIN) {
            self.buf.push(0xd0);
            self.buf.push(value as i8 as u8);
        } else if value >= i64::from(i16::MIN) {
            self.buf.push(0xd1);
            self.buf.extend_from_slice(&(value as i16).to_be_bytes());
        } else if value >= i64::from(i32::MIN) {
            self.buf.push(0xd2);
            self.buf.extend_from_slice(&(value as i32).to_be_bytes());
        } else {
            self.buf.push(0xd3);
            self.buf.extend_from_slice(&value.to_be_bytes());
        }
    }

    /// Write an unsigned integer.
    pub fn uint(&mut self, value: u64) {
        if value < 0x80 {
            self.buf.push(value as u8);
        } else if value <= u64::from(u8::MAX) {
            self.buf.push(0xcc);
            self.buf.push(value as u8);
        } else if value <= u64::from(u16::MAX) {
            self.buf.push(0xcd);
            self.buf.extend_from_slice(&(value as u16).to_be_bytes());
        } else if value <= u64::from(u32::MAX) {
            self.buf.push(0xce);
            self.buf.extend_from_slice(&(value as u32).to_be_bytes());
        } else {
            self.buf.push(0xcf);
            self.buf.extend_from_slice(&value.to_be_bytes());
        }
    }

    /// Write a float, always as float64.
    pub fn float(&mut self, value: f64) {
        self.buf.push(0xcb);
        self.buf.extend_from_slice(&value.to_be_bytes());
    }

    /// Write a string.
    pub fn str(&mut self, value: &str) {
        let len = value.len();
        if len < 32 {
            self.buf.push(0xa0 | len as u8);
        } else {
            self.len_header(len, [0xd9, 0xda, 0xdb]);
        }
        self.buf.extend_from_slice(value.as_bytes());
    }

    /// Write a binary value.
    pub fn bin(&mut self, value: &[u8]) {
        self.len_header(value.len(), [0xc4, 0xc5, 0xc6]);
        self.buf.extend_from_slice(value);
    }

    /// Write an array header for `len` elements.
    pub fn array_len(&mut self, len: usize) {
        if len < 16 {
            self.buf.push(0x90 | len as u8);
        } else {
            self.len_header16(len, [0xdc, 0xdd]);
        }
    }

    /// Write a map header for `len` entries.
    pub fn map_len(&mut self, len: usize) {
        if len < 16 {
            self.buf.push(0x80 | len as u8);
        } else {
            self.len_header16(len, [0xde, 0xdf]);
        }
    }

    /// Write an extension value.
    pub fn ext(&mut self, code: i8, data: &[u8]) {
        let fixed = match data.len() {
            1 => Some(0xd4),
            2 => Some(0xd5),
            4 => Some(0xd6),
            8 => Some(0xd7),
            16 => Some(0xd8),
            _ => None,
        };
        match fixed {
            Some(marker) => self.buf.push(marker),
            None => self.len_header(data.len(), [0xc7, 0xc8, 0xc9]),
        }
        self.buf.push(code as u8);
        self.buf.extend_from_slice(data);
    }

    /// Write `data` either inline as an ext value or as a reference to a new
    /// out-of-band frame, by size.
    pub(crate) fn buffer(&mut self, ext_code: i8, data: &Bytes) {
        if data.len() < self.inline_threshold {
            self.ext(ext_code, data);
        } else {
            self.frames.push(data.clone());
            // Frame 0 is the msgpack buffer, so the first frame is index 1.
            self.uint(self.frames.len() as u64);
        }
    }

    fn len_header(&mut self, len: usize, markers: [u8; 3]) {
        if len <= usize::from(u8::MAX) {
            self.buf.push(markers[0]);
            self.buf.push(len as u8);
        } else {
            self.len_header16(len, [markers[1], markers[2]]);
        }
    }

    fn len_header16(&mut self, len: usize, markers: [u8; 2]) {
        if len <= usize::from(u16::MAX) {
            self.buf.push(markers[0]);
            self.buf.extend_from_slice(&(len as u16).to_be_bytes());
        } else {
            let len = u32::try_from(len).expect("msgpack lengths are at most u32::MAX");
            self.buf.push(markers[1]);
            self.buf.extend_from_slice(&len.to_be_bytes());
        }
    }
}

/// The msgpack type of the next value, for decoding unions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// `nil`
    Nil,
    /// `true` / `false`
    Bool,
    /// Any integer format.
    Int,
    /// float32 or float64.
    Float,
    /// A string.
    Str,
    /// A binary value.
    Bin,
    /// An array.
    Array,
    /// A map.
    Map,
    /// An extension value.
    Ext,
}

impl Kind {
    fn of(marker: u8) -> Option<Self> {
        Some(match marker {
            0x00..=0x7f | 0xcc..=0xd3 | 0xe0..=0xff => Kind::Int,
            0x80..=0x8f | 0xde | 0xdf => Kind::Map,
            0x90..=0x9f | 0xdc | 0xdd => Kind::Array,
            0xa0..=0xbf | 0xd9..=0xdb => Kind::Str,
            0xc0 => Kind::Nil,
            0xc2 | 0xc3 => Kind::Bool,
            0xc4..=0xc6 => Kind::Bin,
            0xc7..=0xc9 | 0xd4..=0xd8 => Kind::Ext,
            0xca | 0xcb => Kind::Float,
            0xc1 => return None,
        })
    }

    fn name(self) -> &'static str {
        match self {
            Kind::Nil => "nil",
            Kind::Bool => "bool",
            Kind::Int => "int",
            Kind::Float => "float",
            Kind::Str => "str",
            Kind::Bin => "bin",
            Kind::Array => "array",
            Kind::Map => "map",
            Kind::Ext => "ext",
        }
    }
}

/// An integer as read from the wire, before narrowing to a schema type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Integer {
    /// A value from a signed format, or any value below zero.
    Signed(i64),
    /// A value from an unsigned format.
    Unsigned(u64),
}

impl Integer {
    fn wide(self) -> i128 {
        match self {
            Integer::Signed(value) => i128::from(value),
            Integer::Unsigned(value) => i128::from(value),
        }
    }

    /// Narrow to any integer type, failing loudly on overflow.
    pub fn narrow<T: TryFrom<i128>>(self, target: &'static str) -> Result<T> {
        let value = self.wide();
        T::try_from(value).map_err(|_| Error::IntRange { value, target })
    }
}

/// Reads one message from its frames.
#[derive(Debug, Clone)]
pub struct Decoder<'f> {
    frames: &'f [Bytes],
    pos: usize,
}

impl<'f> Decoder<'f> {
    /// A decoder over a message's frames (frame 0 is the msgpack buffer).
    ///
    /// # Panics
    ///
    /// If `frames` is empty.
    pub fn new(frames: &'f [Bytes]) -> Self {
        assert!(!frames.is_empty(), "a message has at least one frame");
        Self { frames, pos: 0 }
    }

    fn buf(&self) -> &'f Bytes {
        &self.frames[0]
    }

    /// Fail unless the whole msgpack buffer was consumed.
    pub fn finish(&self) -> Result<()> {
        let left = self.buf().len() - self.pos;
        if left == 0 {
            Ok(())
        } else {
            Err(Error::TrailingBytes(left))
        }
    }

    /// The out-of-band frame a buffer reference names.
    pub(crate) fn frame(&self, index: Integer) -> Result<Bytes> {
        let index: usize = index.narrow("frame index")?;
        if index == 0 || index >= self.frames.len() {
            return Err(Error::Invalid(format!(
                "frame index {index} out of range: the message has {} out-of-band frames",
                self.frames.len() - 1
            )));
        }
        Ok(self.frames[index].clone())
    }

    /// The tag of the tagged array struct that comes next, without
    /// consuming it.
    pub fn peek_tag(&self) -> Result<&'f str> {
        let mut probe = self.clone();
        if probe.array_len()? == 0 {
            return Err(Error::MissingField("_tag"));
        }
        probe.str()
    }

    /// The type of the next value, without consuming it.
    pub fn peek_kind(&self) -> Result<Kind> {
        let marker = self.peek_marker()?;
        Kind::of(marker)
            .ok_or_else(|| Error::Invalid(format!("reserved marker 0xc1 at offset {}", self.pos)))
    }

    /// Consume a `nil` if the next value is one.
    pub fn take_nil(&mut self) -> Result<bool> {
        if self.peek_marker()? == 0xc0 {
            self.pos += 1;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Read a boolean.
    pub fn bool(&mut self) -> Result<bool> {
        match self.peek_marker()? {
            0xc2 => {
                self.pos += 1;
                Ok(false)
            }
            0xc3 => {
                self.pos += 1;
                Ok(true)
            }
            _ => Err(self.type_error("bool")),
        }
    }

    /// Read an integer in any format.
    pub fn integer(&mut self) -> Result<Integer> {
        let marker = self.peek_marker()?;
        let start = self.pos;
        let value = match marker {
            0x00..=0x7f => {
                self.pos += 1;
                Integer::Unsigned(u64::from(marker))
            }
            0xe0..=0xff => {
                self.pos += 1;
                Integer::Signed(i64::from(marker as i8))
            }
            0xcc => Integer::Unsigned(u64::from(self.be::<1>(start)?[0])),
            0xcd => Integer::Unsigned(u64::from(u16::from_be_bytes(self.be(start)?))),
            0xce => Integer::Unsigned(u64::from(u32::from_be_bytes(self.be(start)?))),
            0xcf => Integer::Unsigned(u64::from_be_bytes(self.be(start)?)),
            0xd0 => Integer::Signed(i64::from(self.be::<1>(start)?[0] as i8)),
            0xd1 => Integer::Signed(i64::from(i16::from_be_bytes(self.be(start)?))),
            0xd2 => Integer::Signed(i64::from(i32::from_be_bytes(self.be(start)?))),
            0xd3 => Integer::Signed(i64::from_be_bytes(self.be(start)?)),
            _ => return Err(self.type_error("int")),
        };
        Ok(value)
    }

    /// Read a float. Integers are accepted, as msgspec does for float fields.
    pub fn float(&mut self) -> Result<f64> {
        let start = self.pos;
        match self.peek_marker()? {
            0xca => Ok(f64::from(f32::from_be_bytes(self.be(start)?))),
            0xcb => Ok(f64::from_be_bytes(self.be(start)?)),
            marker if Kind::of(marker) == Some(Kind::Int) => Ok(self.integer()?.wide() as f64),
            _ => Err(self.type_error("float")),
        }
    }

    /// Read a string, borrowed from the msgpack buffer.
    pub fn str(&mut self) -> Result<&'f str> {
        let marker = self.peek_marker()?;
        let start = self.pos;
        let len = match marker {
            0xa0..=0xbf => {
                self.pos += 1;
                usize::from(marker & 0x1f)
            }
            0xd9 => usize::from(self.be::<1>(start)?[0]),
            0xda => usize::from(u16::from_be_bytes(self.be(start)?)),
            0xdb => u32::from_be_bytes(self.be(start)?) as usize,
            _ => return Err(self.type_error("str")),
        };
        let bytes = self.take(start, len)?;
        std::str::from_utf8(bytes).map_err(|_| Error::Utf8 { offset: start })
    }

    /// Read a binary value, sharing the msgpack buffer.
    pub fn bin(&mut self) -> Result<Bytes> {
        let marker = self.peek_marker()?;
        let start = self.pos;
        let len = match marker {
            0xc4 => usize::from(self.be::<1>(start)?[0]),
            0xc5 => usize::from(u16::from_be_bytes(self.be(start)?)),
            0xc6 => u32::from_be_bytes(self.be(start)?) as usize,
            _ => return Err(self.type_error("bin")),
        };
        self.take_shared(start, len)
    }

    /// Read an array header and return its length.
    pub fn array_len(&mut self) -> Result<usize> {
        let marker = self.peek_marker()?;
        let start = self.pos;
        match marker {
            0x90..=0x9f => {
                self.pos += 1;
                Ok(usize::from(marker & 0x0f))
            }
            0xdc => Ok(usize::from(u16::from_be_bytes(self.be(start)?))),
            0xdd => Ok(u32::from_be_bytes(self.be(start)?) as usize),
            _ => Err(self.type_error("array")),
        }
    }

    /// Read a map header and return its number of entries.
    pub fn map_len(&mut self) -> Result<usize> {
        let marker = self.peek_marker()?;
        let start = self.pos;
        match marker {
            0x80..=0x8f => {
                self.pos += 1;
                Ok(usize::from(marker & 0x0f))
            }
            0xde => Ok(usize::from(u16::from_be_bytes(self.be(start)?))),
            0xdf => Ok(u32::from_be_bytes(self.be(start)?) as usize),
            _ => Err(self.type_error("map")),
        }
    }

    /// Read an extension value, sharing the msgpack buffer.
    pub fn ext(&mut self) -> Result<(i8, Bytes)> {
        let marker = self.peek_marker()?;
        let start = self.pos;
        let len = match marker {
            0xd4 => 1,
            0xd5 => 2,
            0xd6 => 4,
            0xd7 => 8,
            0xd8 => 16,
            0xc7 => usize::from(self.be::<1>(start)?[0]),
            0xc8 => usize::from(u16::from_be_bytes(self.be(start)?)),
            0xc9 => u32::from_be_bytes(self.be(start)?) as usize,
            _ => return Err(self.type_error("ext")),
        };
        if (0xd4..=0xd8).contains(&marker) {
            self.pos += 1;
        }
        let code = self.take(start, 1)?[0] as i8;
        Ok((code, self.take_shared(start, len)?))
    }

    /// Skip one value of any type.
    pub fn skip(&mut self) -> Result<()> {
        let mut pending: usize = 1;
        while pending > 0 {
            pending -= 1;
            match self.peek_kind()? {
                Kind::Nil | Kind::Bool => self.pos += 1,
                Kind::Int => {
                    self.integer()?;
                }
                Kind::Float => {
                    self.float()?;
                }
                Kind::Str => {
                    self.str()?;
                }
                Kind::Bin => {
                    self.bin()?;
                }
                Kind::Ext => {
                    self.ext()?;
                }
                Kind::Array => pending += self.array_len()?,
                Kind::Map => pending += 2 * self.map_len()?,
            }
        }
        Ok(())
    }

    fn peek_marker(&self) -> Result<u8> {
        self.buf().get(self.pos).copied().ok_or(Error::Truncated {
            offset: self.pos,
            needed: 1,
        })
    }

    /// Consume the marker at `start` and the `N` big-endian bytes after it.
    fn be<const N: usize>(&mut self, start: usize) -> Result<[u8; N]> {
        self.pos = start + 1;
        let bytes = self.take(start, N)?;
        Ok(bytes.try_into().expect("take returned N bytes"))
    }

    fn take(&mut self, start: usize, len: usize) -> Result<&'f [u8]> {
        let buf: &'f [u8] = self.buf();
        let end = self
            .pos
            .checked_add(len)
            .filter(|end| *end <= buf.len())
            .ok_or(Error::Truncated {
                offset: start,
                needed: (self.pos + len).saturating_sub(buf.len()),
            })?;
        let bytes = &buf[self.pos..end];
        self.pos = end;
        Ok(bytes)
    }

    fn take_shared(&mut self, start: usize, len: usize) -> Result<Bytes> {
        let from = self.pos;
        self.take(start, len)?;
        Ok(self.buf().slice(from..self.pos))
    }

    fn type_error(&self, expected: &'static str) -> Error {
        let found = self
            .buf()
            .get(self.pos)
            .and_then(|marker| Kind::of(*marker))
            .map_or("reserved marker", Kind::name);
        Error::Type {
            expected,
            found,
            offset: self.pos,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encoded(write: impl FnOnce(&mut Encoder)) -> Vec<u8> {
        let mut encoder = Encoder::new();
        write(&mut encoder);
        encoder.into_single_frame()
    }

    /// The format boundaries msgspec uses, as observed from the Python encoder.
    #[test]
    fn integers_use_the_smallest_format() {
        let cases: &[(i64, &[u8])] = &[
            (127, &[0x7f]),
            (128, &[0xcc, 0x80]),
            (255, &[0xcc, 0xff]),
            (256, &[0xcd, 0x01, 0x00]),
            (65535, &[0xcd, 0xff, 0xff]),
            (65536, &[0xce, 0x00, 0x01, 0x00, 0x00]),
            (-1, &[0xff]),
            (-32, &[0xe0]),
            (-33, &[0xd0, 0xdf]),
            (-128, &[0xd0, 0x80]),
            (-129, &[0xd1, 0xff, 0x7f]),
            (1 << 32, &[0xcf, 0, 0, 0, 1, 0, 0, 0, 0]),
        ];
        for (value, expected) in cases {
            assert_eq!(encoded(|e| e.int(*value)), *expected, "{value}");
            let frames = [Bytes::copy_from_slice(expected)];
            let mut decoder = Decoder::new(&frames);
            let decoded: i64 = decoder.integer().unwrap().narrow("i64").unwrap();
            assert_eq!(decoded, *value);
            decoder.finish().unwrap();
        }
    }

    #[test]
    fn strings_switch_to_str8_at_32_bytes() {
        assert_eq!(encoded(|e| e.str(&"a".repeat(31)))[0], 0xbf);
        assert_eq!(encoded(|e| e.str(&"a".repeat(32)))[..2], [0xd9, 32]);
    }

    #[test]
    fn ext_uses_fixext_for_power_of_two_sizes() {
        assert_eq!(encoded(|e| e.ext(3, b"abcd")), b"\xd6\x03abcd");
        assert_eq!(encoded(|e| e.ext(3, b"abc")), b"\xc7\x03\x03abc");
    }

    #[test]
    fn float_fields_accept_integers() {
        let frames = [Bytes::from_static(&[0x03])];
        assert_eq!(Decoder::new(&frames).float().unwrap(), 3.0);
    }

    #[test]
    fn truncated_and_trailing_input_fail() {
        let frames = [Bytes::from_static(&[0xcd, 0x01])];
        assert!(matches!(
            Decoder::new(&frames).integer(),
            Err(Error::Truncated { .. })
        ));
        let frames = [Bytes::from_static(&[0x01, 0x02])];
        let mut decoder = Decoder::new(&frames);
        decoder.integer().unwrap();
        assert!(matches!(decoder.finish(), Err(Error::TrailingBytes(1))));
    }

    #[test]
    fn narrowing_overflow_fails() {
        let frames = [Bytes::from_static(&[0xcf, 0x80, 0, 0, 0, 0, 0, 0, 0])];
        let value = Decoder::new(&frames).integer().unwrap();
        assert!(matches!(
            value.narrow::<i64>("i64"),
            Err(Error::IntRange { .. })
        ));
        assert_eq!(value.narrow::<u64>("u64").unwrap(), 1 << 63);
    }

    #[test]
    fn skip_walks_nested_values() {
        let mut encoder = Encoder::new();
        encoder.array_len(2);
        encoder.map_len(1);
        encoder.str("k");
        encoder.array_len(2);
        encoder.float(1.5);
        encoder.nil();
        encoder.bin(b"xy");
        encoder.int(7);
        let frames = [Bytes::from(encoder.into_single_frame())];
        let mut decoder = Decoder::new(&frames);
        decoder.skip().unwrap();
        assert_eq!(decoder.integer().unwrap(), Integer::Unsigned(7));
        decoder.finish().unwrap();
    }
}

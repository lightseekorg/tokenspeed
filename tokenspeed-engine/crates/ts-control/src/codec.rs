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

//! The byte layout of what the ranks exchange over the collective mesh: the
//! relayed frontend messages, a pipeline's shared forward results, the L3
//! reports, grammar failures, cache-op acknowledgments and expert
//! placements. Every rank runs the same build, so the layout is private to
//! it: fixed-width little-endian values, flags as one byte, and sequences
//! after their length as a `u64`.

/// Builds a payload.
#[derive(Debug, Default)]
pub(crate) struct Writer(Vec<u8>);

impl Writer {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn u8(&mut self, value: u8) {
        self.0.push(value);
    }

    pub(crate) fn flag(&mut self, value: bool) {
        self.0.push(u8::from(value));
    }

    pub(crate) fn u32(&mut self, value: u32) {
        self.0.extend_from_slice(&value.to_le_bytes());
    }

    pub(crate) fn u64(&mut self, value: u64) {
        self.0.extend_from_slice(&value.to_le_bytes());
    }

    pub(crate) fn i64(&mut self, value: i64) {
        self.0.extend_from_slice(&value.to_le_bytes());
    }

    /// A sequence's length.
    pub(crate) fn count(&mut self, count: usize) {
        self.u64(count as u64);
    }

    /// Bytes, after their length.
    pub(crate) fn bytes(&mut self, bytes: &[u8]) {
        self.count(bytes.len());
        self.0.extend_from_slice(bytes);
    }

    pub(crate) fn i32s(&mut self, values: &[i32]) {
        self.count(values.len());
        for value in values {
            self.0.extend_from_slice(&value.to_le_bytes());
        }
    }

    pub(crate) fn f64s(&mut self, values: &[f64]) {
        self.count(values.len());
        for value in values {
            self.0.extend_from_slice(&value.to_le_bytes());
        }
    }

    pub(crate) fn flags(&mut self, values: &[bool]) {
        self.count(values.len());
        self.0.extend(values.iter().map(|&value| u8::from(value)));
    }

    /// `value` behind a presence flag.
    pub(crate) fn option<T>(&mut self, value: Option<T>, write: impl FnOnce(&mut Self, T)) {
        self.flag(value.is_some());
        if let Some(value) = value {
            write(self, value);
        }
    }

    pub(crate) fn finish(self) -> Vec<u8> {
        self.0
    }
}

/// Reads what a [`Writer`] built. A read past the end, or of a value no
/// writer produces, is `None`.
pub(crate) struct Reader<'a>(&'a [u8]);

impl<'a> Reader<'a> {
    pub(crate) fn new(bytes: &'a [u8]) -> Self {
        Self(bytes)
    }

    fn take(&mut self, len: usize) -> Option<&'a [u8]> {
        let (taken, rest) = self.0.split_at_checked(len)?;
        self.0 = rest;
        Some(taken)
    }

    fn array<const N: usize>(&mut self) -> Option<[u8; N]> {
        self.take(N)?.try_into().ok()
    }

    pub(crate) fn u8(&mut self) -> Option<u8> {
        Some(self.array::<1>()?[0])
    }

    pub(crate) fn flag(&mut self) -> Option<bool> {
        match self.u8()? {
            0 => Some(false),
            1 => Some(true),
            _ => None,
        }
    }

    pub(crate) fn u32(&mut self) -> Option<u32> {
        self.array().map(u32::from_le_bytes)
    }

    pub(crate) fn u64(&mut self) -> Option<u64> {
        self.array().map(u64::from_le_bytes)
    }

    pub(crate) fn i64(&mut self) -> Option<i64> {
        self.array().map(i64::from_le_bytes)
    }

    /// A sequence's length.
    pub(crate) fn count(&mut self) -> Option<usize> {
        usize::try_from(self.u64()?).ok()
    }

    /// Bytes after their length.
    pub(crate) fn bytes(&mut self) -> Option<&'a [u8]> {
        let len = self.count()?;
        self.take(len)
    }

    pub(crate) fn i32s(&mut self) -> Option<Vec<i32>> {
        let len = self.count()?;
        let raw = self.take(len.checked_mul(4)?)?;
        Some(
            raw.chunks_exact(4)
                .map(|chunk| i32::from_le_bytes(chunk.try_into().expect("four bytes")))
                .collect(),
        )
    }

    pub(crate) fn f64s(&mut self) -> Option<Vec<f64>> {
        let len = self.count()?;
        let raw = self.take(len.checked_mul(8)?)?;
        Some(
            raw.chunks_exact(8)
                .map(|chunk| f64::from_le_bytes(chunk.try_into().expect("eight bytes")))
                .collect(),
        )
    }

    pub(crate) fn flags(&mut self) -> Option<Vec<bool>> {
        let len = self.count()?;
        self.take(len)?
            .iter()
            .map(|&byte| match byte {
                0 => Some(false),
                1 => Some(true),
                _ => None,
            })
            .collect()
    }

    /// A value behind a presence flag.
    pub(crate) fn option<T>(
        &mut self,
        read: impl FnOnce(&mut Self) -> Option<T>,
    ) -> Option<Option<T>> {
        match self.flag()? {
            false => Some(None),
            true => read(self).map(Some),
        }
    }

    /// `value`, if nothing is left to read.
    pub(crate) fn finish<T>(self, value: T) -> Option<T> {
        self.0.is_empty().then_some(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn values_round_trip_and_a_cut_payload_does_not_read() {
        let mut out = Writer::new();
        out.u8(7);
        out.flag(true);
        out.u32(u32::MAX);
        out.u64(1 << 40);
        out.i64(-5);
        out.bytes(b"abc");
        out.i32s(&[-1, 2]);
        out.f64s(&[0.5]);
        out.flags(&[true, false]);
        out.option(Some(3_i64), Writer::i64);
        out.option(None::<i64>, Writer::i64);
        let bytes = out.finish();

        let read = |bytes: &[u8]| {
            let mut reader = Reader::new(bytes);
            let values = (
                reader.u8()?,
                reader.flag()?,
                reader.u32()?,
                reader.u64()?,
                reader.i64()?,
                reader.bytes()?.to_vec(),
                reader.i32s()?,
                reader.f64s()?,
                reader.flags()?,
                reader.option(Reader::i64)?,
                reader.option(Reader::i64)?,
            );
            reader.finish(values)
        };
        assert_eq!(
            read(&bytes),
            Some((
                7,
                true,
                u32::MAX,
                1 << 40,
                -5,
                b"abc".to_vec(),
                vec![-1, 2],
                vec![0.5],
                vec![true, false],
                Some(3),
                None
            ))
        );
        assert_eq!(read(&bytes[..bytes.len() - 1]), None);
        // Trailing bytes are no payload either.
        let mut longer = bytes.clone();
        longer.push(0);
        assert_eq!(read(&longer), None);
    }

    #[test]
    fn a_length_past_the_payload_or_a_bad_flag_does_not_read() {
        let mut out = Writer::new();
        out.count(usize::MAX);
        assert_eq!(Reader::new(&out.finish()).i32s(), None);
        assert_eq!(Reader::new(&[2]).flag(), None);
        let mut out = Writer::new();
        out.count(1);
        out.u8(2);
        assert_eq!(Reader::new(&out.finish()).flags(), None);
    }
}

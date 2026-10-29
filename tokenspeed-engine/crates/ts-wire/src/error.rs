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

//! Decode errors.

/// A message that does not match the schema, or is not valid msgpack.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The input ended inside a value.
    #[error("input truncated at offset {offset}: {needed} more bytes needed")]
    Truncated {
        /// Byte offset in the primary frame where the value started.
        offset: usize,
        /// Bytes missing past the end of the frame.
        needed: usize,
    },
    /// A value of a different msgpack type than the schema allows.
    #[error("expected {expected}, found {found} at offset {offset}")]
    Type {
        /// What the schema allows here.
        expected: &'static str,
        /// The msgpack type found.
        found: &'static str,
        /// Byte offset in the primary frame.
        offset: usize,
    },
    /// An integer outside the range of the schema's type.
    #[error("integer {value} out of range for {target}")]
    IntRange {
        /// The decoded integer.
        value: i128,
        /// The schema type.
        target: &'static str,
    },
    /// A string that is not valid UTF-8.
    #[error("invalid UTF-8 string at offset {offset}")]
    Utf8 {
        /// Byte offset in the primary frame.
        offset: usize,
    },
    /// A tagged struct whose tag names a different message type.
    #[error("wrong message tag: expected `{expected}`, found `{found}`")]
    Tag {
        /// The tag of the type being decoded.
        expected: &'static str,
        /// The tag on the wire.
        found: String,
    },
    /// A struct without a value for a field that has no default.
    #[error("missing required field `{0}`")]
    MissingField(&'static str),
    /// Bytes left in the primary frame after the top-level value.
    #[error("{0} trailing bytes after the message")]
    TrailingBytes(usize),
    /// A value the schema rejects for a reason other than its type.
    #[error("{0}")]
    Invalid(String),
    /// An error inside a named field, kept with the field name for context.
    #[error("field `{field}`: {source}")]
    Field {
        /// The field being decoded.
        field: &'static str,
        /// The underlying error.
        #[source]
        source: Box<Error>,
    },
}

impl Error {
    /// Attach the name of the field being decoded.
    pub fn in_field(self, field: &'static str) -> Self {
        Error::Field {
            field,
            source: Box::new(self),
        }
    }
}

/// Result alias for decoding.
pub type Result<T> = std::result::Result<T, Error>;

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

//! The [`Wire`] trait and the struct layouts msgspec uses.
//!
//! msgspec encodes a `Struct` in one of two layouts, both mirrored here by
//! [`wire_struct!`]:
//!
//! * **array** (`array_like=True`): `[tag?, field, ...]` in declaration
//!   order. Declaration order is the wire contract, so fields are only ever
//!   appended. A shorter array takes the remaining fields' defaults; a
//!   longer one has its extra trailing elements ignored. A tagged struct
//!   (`tag=True`) starts with its class name.
//! * **map**: `{name: value, ...}` with every field present, in declaration
//!   order. Unknown keys are ignored and missing keys take their defaults.
//!
//! Both layouts fail when a field without a default is absent.

use bytes::Bytes;

use crate::{
    error::{Error, Result},
    msgpack::{Decoder, Encoder, Kind},
};

/// A type with a msgpack encoding shared with the Python side.
pub trait Wire: Sized {
    /// Append this value to a message.
    fn encode(&self, encoder: &mut Encoder);

    /// Read one value of this type.
    fn decode(decoder: &mut Decoder<'_>) -> Result<Self>;
}

/// Encode a value as a message: the msgpack buffer, then its out-of-band
/// frames.
pub fn encode_frames<T: Wire>(value: &T) -> Vec<Bytes> {
    let mut encoder = Encoder::new();
    value.encode(&mut encoder);
    encoder.into_frames()
}

/// Decode a message from its frames, failing on bytes after the value.
pub fn decode_frames<T: Wire>(frames: &[Bytes]) -> Result<T> {
    let mut decoder = Decoder::new(frames);
    let value = T::decode(&mut decoder)?;
    decoder.finish()?;
    Ok(value)
}

/// Decode a message that is a single msgpack buffer.
pub fn decode_bytes<T: Wire>(bytes: &[u8]) -> Result<T> {
    decode_frames(&[Bytes::copy_from_slice(bytes)])
}

impl Wire for bool {
    fn encode(&self, encoder: &mut Encoder) {
        encoder.bool(*self);
    }

    fn decode(decoder: &mut Decoder<'_>) -> Result<Self> {
        decoder.bool()
    }
}

macro_rules! wire_int {
    ($($ty:ty => $write:ident),* $(,)?) => {$(
        impl Wire for $ty {
            fn encode(&self, encoder: &mut Encoder) {
                encoder.$write((*self).into());
            }

            fn decode(decoder: &mut Decoder<'_>) -> Result<Self> {
                decoder.integer()?.narrow(stringify!($ty))
            }
        }
    )*};
}

wire_int!(i32 => int, i64 => int, u16 => uint, u32 => uint, u64 => uint);

impl Wire for f64 {
    fn encode(&self, encoder: &mut Encoder) {
        encoder.float(*self);
    }

    fn decode(decoder: &mut Decoder<'_>) -> Result<Self> {
        decoder.float()
    }
}

impl Wire for String {
    fn encode(&self, encoder: &mut Encoder) {
        encoder.str(self);
    }

    fn decode(decoder: &mut Decoder<'_>) -> Result<Self> {
        Ok(decoder.str()?.to_owned())
    }
}

/// `None` is `nil` on the wire.
impl<T: Wire> Wire for Option<T> {
    fn encode(&self, encoder: &mut Encoder) {
        match self {
            Some(value) => value.encode(encoder),
            None => encoder.nil(),
        }
    }

    fn decode(decoder: &mut Decoder<'_>) -> Result<Self> {
        if decoder.take_nil()? {
            Ok(None)
        } else {
            T::decode(decoder).map(Some)
        }
    }
}

/// Lists, tuples of any length and sets are all arrays on the wire.
impl<T: Wire> Wire for Vec<T> {
    fn encode(&self, encoder: &mut Encoder) {
        encoder.array_len(self.len());
        for item in self {
            item.encode(encoder);
        }
    }

    fn decode(decoder: &mut Decoder<'_>) -> Result<Self> {
        let len = decoder.array_len()?;
        // Bound the up-front reservation: the length is untrusted input.
        let mut items = Vec::with_capacity(len.min(1 << 16));
        for _ in 0..len {
            items.push(T::decode(decoder)?);
        }
        Ok(items)
    }
}

/// A two-element tuple, an array of length two on the wire.
impl<A: Wire, B: Wire> Wire for (A, B) {
    fn encode(&self, encoder: &mut Encoder) {
        encoder.array_len(2);
        self.0.encode(encoder);
        self.1.encode(encoder);
    }

    fn decode(decoder: &mut Decoder<'_>) -> Result<Self> {
        let len = decoder.array_len()?;
        if len != 2 {
            return Err(Error::Invalid(format!(
                "expected a 2-element array, found {len} elements"
            )));
        }
        Ok((A::decode(decoder)?, B::decode(decoder)?))
    }
}

/// A Python `dict`: a msgpack map whose entries keep their wire order.
#[derive(Debug, Clone, PartialEq)]
pub struct Dict<K, V>(pub Vec<(K, V)>);

impl<K, V> Default for Dict<K, V> {
    fn default() -> Self {
        Dict(Vec::new())
    }
}

impl<K: PartialEq, V> Dict<K, V> {
    /// The value for `key`, if present.
    pub fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: PartialEq<Q>,
        Q: ?Sized,
    {
        self.0.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }
}

impl<K: Wire, V: Wire> Wire for Dict<K, V> {
    fn encode(&self, encoder: &mut Encoder) {
        encoder.map_len(self.0.len());
        for (key, value) in &self.0 {
            key.encode(encoder);
            value.encode(encoder);
        }
    }

    fn decode(decoder: &mut Decoder<'_>) -> Result<Self> {
        let len = decoder.map_len()?;
        let mut entries = Vec::with_capacity(len.min(1 << 16));
        for _ in 0..len {
            let key = K::decode(decoder)?;
            entries.push((key, V::decode(decoder)?));
        }
        Ok(Dict(entries))
    }
}

/// A Python `str | list[str]` field.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StrOrList {
    /// A single string.
    One(String),
    /// A list of strings.
    Many(Vec<String>),
}

impl Wire for StrOrList {
    fn encode(&self, encoder: &mut Encoder) {
        match self {
            StrOrList::One(value) => value.encode(encoder),
            StrOrList::Many(values) => values.encode(encoder),
        }
    }

    fn decode(decoder: &mut Decoder<'_>) -> Result<Self> {
        match decoder.peek_kind()? {
            Kind::Str => Ok(StrOrList::One(String::decode(decoder)?)),
            Kind::Array => Ok(StrOrList::Many(Vec::decode(decoder)?)),
            _ => Err(Error::Invalid(
                "expected a string or a list of strings".to_owned(),
            )),
        }
    }
}

/// Reads the fields of an array-layout struct in declaration order.
#[doc(hidden)]
pub struct ArrayFields<'d, 'f> {
    decoder: &'d mut Decoder<'f>,
    remaining: usize,
}

impl<'d, 'f> ArrayFields<'d, 'f> {
    /// Read the array header.
    pub fn begin(decoder: &'d mut Decoder<'f>) -> Result<Self> {
        let remaining = decoder.array_len()?;
        Ok(Self { decoder, remaining })
    }

    /// Check the tag a tagged struct starts with.
    pub fn tag(&mut self, expected: &'static str) -> Result<()> {
        if self.remaining == 0 {
            return Err(Error::MissingField("_tag"));
        }
        self.remaining -= 1;
        let found = self.decoder.str()?;
        if found != expected {
            return Err(Error::Tag {
                expected,
                found: found.to_owned(),
            });
        }
        Ok(())
    }

    /// The next field, or `None` once the array is exhausted.
    pub fn next<T: Wire>(&mut self, name: &'static str) -> Result<Option<T>> {
        if self.remaining == 0 {
            return Ok(None);
        }
        self.remaining -= 1;
        T::decode(self.decoder)
            .map(Some)
            .map_err(|error| error.in_field(name))
    }

    /// Skip trailing elements a newer peer appended.
    pub fn finish(self) -> Result<()> {
        for _ in 0..self.remaining {
            self.decoder.skip()?;
        }
        Ok(())
    }
}

/// Define a struct and its [`Wire`] codec in one of msgspec's layouts.
///
/// ```ignore
/// wire_struct! {
///     /// Doc.
///     pub struct Example: array("Example") {
///         /// A required field.
///         pub name: String,
///         /// A field with a default, used when an older peer omits it.
///         pub count: i64 = 0,
///     }
/// }
/// ```
///
/// The layout is `array` (optionally with a tag) or `map`. Writing
/// `+ Default` after the layout also derives `Default` from the field
/// defaults, which then must be given for every field.
macro_rules! wire_struct {
    (
        $(#[$meta:meta])*
        $vis:vis struct $name:ident: $layout:ident $(($tag:expr))? + Default {
            $( $(#[$fmeta:meta])* $fvis:vis $field:ident: $fty:ty = $fdefault:expr ),* $(,)?
        }
    ) => {
        wire_struct! {
            $(#[$meta])*
            $vis struct $name: $layout $(($tag))? {
                $( $(#[$fmeta])* $fvis $field: $fty = $fdefault ),*
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self { $( $field: $fdefault ),* }
            }
        }
    };
    (
        $(#[$meta:meta])*
        $vis:vis struct $name:ident: $layout:ident $(($tag:expr))? {
            $( $(#[$fmeta:meta])* $fvis:vis $field:ident: $fty:ty $(= $fdefault:expr)? ),* $(,)?
        }
    ) => {
        $(#[$meta])*
        $vis struct $name {
            $( $(#[$fmeta])* $fvis $field: $fty ),*
        }

        wire_struct!(@$layout $name $(($tag))? { $( $field: $fty $(= $fdefault)? ),* });
    };
    (@array $name:ident $(($tag:expr))? { $( $field:ident: $fty:ty $(= $fdefault:expr)? ),* }) => {
        impl $crate::codec::Wire for $name {
            fn encode(&self, encoder: &mut $crate::msgpack::Encoder) {
                let tag_len: usize = 0 $( + { let _ = $tag; 1 } )?;
                encoder.array_len(tag_len $( + { let _ = stringify!($field); 1 } )*);
                $( encoder.str($tag); )?
                $( $crate::codec::Wire::encode(&self.$field, encoder); )*
            }

            fn decode(
                decoder: &mut $crate::msgpack::Decoder<'_>,
            ) -> $crate::error::Result<Self> {
                let mut fields = $crate::codec::ArrayFields::begin(decoder)?;
                $( fields.tag($tag)?; )?
                $(
                    let $field: $fty = match fields.next(stringify!($field))? {
                        Some(value) => value,
                        None => wire_struct!(@default $field $($fdefault)?),
                    };
                )*
                fields.finish()?;
                Ok(Self { $( $field ),* })
            }
        }
    };
    (@map $name:ident { $( $field:ident: $fty:ty $(= $fdefault:expr)? ),* }) => {
        impl $crate::codec::Wire for $name {
            fn encode(&self, encoder: &mut $crate::msgpack::Encoder) {
                encoder.map_len(0 $( + { let _ = stringify!($field); 1 } )*);
                $(
                    encoder.str(stringify!($field));
                    $crate::codec::Wire::encode(&self.$field, encoder);
                )*
            }

            fn decode(
                decoder: &mut $crate::msgpack::Decoder<'_>,
            ) -> $crate::error::Result<Self> {
                $( let mut $field: Option<$fty> = None; )*
                for _ in 0..decoder.map_len()? {
                    match decoder.str()? {
                        $(
                            stringify!($field) => {
                                $field = Some(
                                    $crate::codec::Wire::decode(decoder)
                                        .map_err(|error| error.in_field(stringify!($field)))?,
                                );
                            }
                        )*
                        _ => decoder.skip()?,
                    }
                }
                Ok(Self {
                    $(
                        $field: match $field {
                            Some(value) => value,
                            None => wire_struct!(@default $field $($fdefault)?),
                        },
                    )*
                })
            }
        }
    };
    (@default $field:ident) => {
        return Err($crate::error::Error::MissingField(stringify!($field)))
    };
    (@default $field:ident $default:expr) => {
        $default
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    wire_struct! {
        /// A tagged array struct with an appended defaulted field.
        #[derive(Debug, PartialEq)]
        struct Tagged: array("Tagged") {
            name: String,
            count: i64 = 5,
        }
    }

    wire_struct! {
        /// A map struct.
        #[derive(Debug, PartialEq)]
        struct Named: map + Default {
            x: i64 = 1,
            y: Option<String> = None,
        }
    }

    fn encode_value(write: impl FnOnce(&mut Encoder)) -> Vec<u8> {
        let mut encoder = Encoder::new();
        write(&mut encoder);
        encoder.into_single_frame()
    }

    #[test]
    fn short_arrays_take_defaults_and_long_arrays_drop_extras() {
        let short = encode_value(|e| {
            e.array_len(2);
            e.str("Tagged");
            e.str("a");
        });
        let decoded: Tagged = decode_bytes(&short).unwrap();
        assert_eq!(
            decoded,
            Tagged {
                name: "a".into(),
                count: 5
            }
        );

        let long = encode_value(|e| {
            e.array_len(4);
            e.str("Tagged");
            e.str("a");
            e.int(9);
            e.str("from a newer peer");
        });
        let decoded: Tagged = decode_bytes(&long).unwrap();
        assert_eq!(
            decoded,
            Tagged {
                name: "a".into(),
                count: 9
            }
        );
    }

    #[test]
    fn required_fields_and_tags_are_enforced() {
        let missing = encode_value(|e| {
            e.array_len(1);
            e.str("Tagged");
        });
        assert!(matches!(
            decode_bytes::<Tagged>(&missing),
            Err(Error::MissingField("name"))
        ));

        let wrong_tag = encode_value(|e| {
            e.array_len(2);
            e.str("Other");
            e.str("a");
        });
        assert!(matches!(
            decode_bytes::<Tagged>(&wrong_tag),
            Err(Error::Tag { .. })
        ));
    }

    #[test]
    fn map_structs_ignore_unknown_keys_and_default_missing_ones() {
        let bytes = encode_value(|e| {
            e.map_len(2);
            e.str("unknown");
            e.array_len(1);
            e.nil();
            e.str("y");
            e.str("v");
        });
        let decoded: Named = decode_bytes(&bytes).unwrap();
        assert_eq!(
            decoded,
            Named {
                x: 1,
                y: Some("v".into())
            }
        );

        let encoded = encode_frames(&Named::default());
        assert_eq!(&encoded[0][..], b"\x82\xa1x\x01\xa1y\xc0");
    }

    #[test]
    fn errors_name_the_field() {
        let bytes = encode_value(|e| {
            e.array_len(3);
            e.str("Tagged");
            e.str("a");
            e.str("not an int");
        });
        let error = decode_bytes::<Tagged>(&bytes).unwrap_err();
        assert!(error.to_string().starts_with("field `count`:"), "{error}");
    }
}

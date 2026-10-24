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

//! Untyped values, for fields the Python schema types as `dict` or `Any`.

use bytes::Bytes;

use crate::{
    codec::Wire,
    error::Result,
    msgpack::{Decoder, Encoder, Integer, Kind},
};

/// Any msgpack value.
///
/// Maps keep their wire order, so a decoded value re-encodes to the same
/// bytes. Integers keep the signedness of their format for the same reason.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// `nil`
    Nil,
    /// A boolean.
    Bool(bool),
    /// An integer.
    Int(Integer),
    /// A float.
    Float(f64),
    /// A string.
    Str(String),
    /// A binary value.
    Bin(Bytes),
    /// An array.
    Array(Vec<Value>),
    /// A map, in wire order.
    Map(Vec<(Value, Value)>),
    /// An extension value.
    Ext(i8, Bytes),
}

impl From<&str> for Value {
    fn from(value: &str) -> Self {
        Value::Str(value.to_owned())
    }
}

impl From<i64> for Value {
    fn from(value: i64) -> Self {
        Value::Int(if value < 0 {
            Integer::Signed(value)
        } else {
            Integer::Unsigned(value as u64)
        })
    }
}

impl From<f64> for Value {
    fn from(value: f64) -> Self {
        Value::Float(value)
    }
}

impl From<bool> for Value {
    fn from(value: bool) -> Self {
        Value::Bool(value)
    }
}

impl Wire for Value {
    fn encode(&self, encoder: &mut Encoder) {
        match self {
            Value::Nil => encoder.nil(),
            Value::Bool(value) => encoder.bool(*value),
            Value::Int(Integer::Signed(value)) => encoder.int(*value),
            Value::Int(Integer::Unsigned(value)) => encoder.uint(*value),
            Value::Float(value) => encoder.float(*value),
            Value::Str(value) => encoder.str(value),
            Value::Bin(value) => encoder.bin(value),
            Value::Array(items) => {
                encoder.array_len(items.len());
                for item in items {
                    item.encode(encoder);
                }
            }
            Value::Map(entries) => {
                encoder.map_len(entries.len());
                for (key, value) in entries {
                    key.encode(encoder);
                    value.encode(encoder);
                }
            }
            Value::Ext(code, data) => encoder.ext(*code, data),
        }
    }

    fn decode(decoder: &mut Decoder<'_>) -> Result<Self> {
        Ok(match decoder.peek_kind()? {
            Kind::Nil => {
                decoder.take_nil()?;
                Value::Nil
            }
            Kind::Bool => Value::Bool(decoder.bool()?),
            Kind::Int => Value::Int(decoder.integer()?),
            Kind::Float => Value::Float(decoder.float()?),
            Kind::Str => Value::Str(decoder.str()?.to_owned()),
            Kind::Bin => Value::Bin(decoder.bin()?),
            Kind::Array => {
                let len = decoder.array_len()?;
                let mut items = Vec::with_capacity(len.min(4096));
                for _ in 0..len {
                    items.push(Value::decode(decoder)?);
                }
                Value::Array(items)
            }
            Kind::Map => {
                let len = decoder.map_len()?;
                let mut entries = Vec::with_capacity(len.min(4096));
                for _ in 0..len {
                    let key = Value::decode(decoder)?;
                    entries.push((key, Value::decode(decoder)?));
                }
                Value::Map(entries)
            }
            Kind::Ext => {
                let (code, data) = decoder.ext()?;
                Value::Ext(code, data)
            }
        })
    }
}

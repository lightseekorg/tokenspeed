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

//! Tensors as the Python codec carries them (`MsgpackEncoder` in
//! `io_struct.py`).
//!
//! A tensor is the array `[dtype, shape, data]`. `dtype` is a torch dtype
//! name (`"bfloat16"`) or a numpy type string (`"<f4"`); `data` is the
//! contiguous little-endian bytes, either inline as an ext value with code
//! [`RAW_VIEW_EXT`] or, from [`crate::msgpack::INLINE_THRESHOLD`] bytes up,
//! the index of the out-of-band frame that holds them.

use bytes::Bytes;

use crate::{
    codec::Wire,
    error::{Error, Result},
    msgpack::{Decoder, Encoder, Kind},
};

/// Ext code of an inline tensor buffer (`CUSTOM_TYPE_RAW_VIEW`). Codes 1 and
/// 2 are reserved for pickled payloads, which this protocol never carries.
pub const RAW_VIEW_EXT: i8 = 3;

/// A CPU tensor's metadata and bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tensor {
    /// torch dtype name or numpy type string.
    pub dtype: String,
    /// Dimensions, outermost first.
    pub shape: Vec<i64>,
    /// Contiguous element bytes. Shares the received frame; no copy.
    pub data: Bytes,
}

impl Wire for Tensor {
    fn encode(&self, encoder: &mut Encoder) {
        encoder.array_len(3);
        encoder.str(&self.dtype);
        self.shape.encode(encoder);
        encoder.buffer(RAW_VIEW_EXT, &self.data);
    }

    fn decode(decoder: &mut Decoder<'_>) -> Result<Self> {
        let len = decoder.array_len()?;
        if len != 3 {
            return Err(Error::Invalid(format!(
                "a tensor is [dtype, shape, data], found {len} elements"
            )));
        }
        let dtype = String::decode(decoder)?;
        let shape = Vec::decode(decoder)?;
        let data = match decoder.peek_kind()? {
            Kind::Ext => {
                let (code, data) = decoder.ext()?;
                if code != RAW_VIEW_EXT {
                    return Err(Error::Invalid(format!(
                        "tensor data has ext code {code}, expected {RAW_VIEW_EXT}"
                    )));
                }
                data
            }
            Kind::Int => {
                let index = decoder.integer()?;
                decoder.frame(index)?
            }
            _ => {
                return Err(Error::Invalid(
                    "tensor data must be an inline ext buffer or a frame index".to_owned(),
                ))
            }
        };
        Ok(Self { dtype, shape, data })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::{decode_frames, encode_frames};

    fn tensor(len: usize) -> Tensor {
        Tensor {
            dtype: "uint8".to_owned(),
            shape: vec![len as i64],
            data: Bytes::from(vec![7u8; len]),
        }
    }

    #[test]
    fn small_tensors_inline_and_large_ones_get_a_frame() {
        let small = encode_frames(&tensor(255));
        assert_eq!(small.len(), 1);
        assert_eq!(decode_frames::<Tensor>(&small).unwrap(), tensor(255));

        let large = encode_frames(&tensor(256));
        assert_eq!(large.len(), 2);
        assert_eq!(large[1].len(), 256);
        assert_eq!(decode_frames::<Tensor>(&large).unwrap(), tensor(256));
    }

    #[test]
    fn a_frame_index_past_the_message_fails() {
        let mut frames = encode_frames(&tensor(300));
        frames.truncate(1);
        assert!(decode_frames::<Tensor>(&frames).is_err());
    }
}

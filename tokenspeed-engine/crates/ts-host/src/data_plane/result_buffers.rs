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

//! Copy landed logprob buffers while the buffer exports retain their owners.
//!
//! No Python pointer escapes this query. PyUntypedBuffer releases its export on
//! every return/unwind path, and the returned vectors use Rust's allocator.

use pyo3::{
    buffer::{Element, PyUntypedBuffer},
    exceptions::{PyBufferError, PyOverflowError, PyValueError},
    prelude::*,
};
use ts_control::output::TopLogprobs;

fn require_rank(buffer: &PyUntypedBuffer, rank: usize) -> PyResult<()> {
    if buffer.dimensions() != rank {
        return Err(PyValueError::new_err(format!(
            "expected a rank-{rank} result buffer, got rank {}",
            buffer.dimensions()
        )));
    }
    let count = buffer
        .shape()
        .iter()
        .try_fold(1usize, |n, &d| n.checked_mul(d));
    if buffer.item_size() == 0 || count != Some(buffer.item_count()) {
        return Err(PyValueError::new_err(
            "result buffer shape does not match its size",
        ));
    }
    Ok(())
}

fn has_format<T: Element>(buffer: &PyUntypedBuffer) -> bool {
    buffer.item_size() == std::mem::size_of::<T>() && T::is_compatible_format(buffer.format())
}

fn copy<T: Element>(buffer: &PyUntypedBuffer, py: Python<'_>) -> PyResult<Vec<T>> {
    if !has_format::<T>(buffer) {
        return Err(PyBufferError::new_err(format!(
            "result buffer must contain native {} values",
            std::any::type_name::<T>()
        )));
    }
    // An empty exporter may use an unaligned sentinel address. There is no
    // memory to read, but its element format must still satisfy the contract.
    if buffer.item_count() == 0 {
        return Ok(Vec::new());
    }
    buffer.as_typed::<T>()?.to_vec(py)
}

fn float_values(buffer: &PyUntypedBuffer, py: Python<'_>) -> PyResult<Vec<f64>> {
    if has_format::<f32>(buffer) {
        Ok(copy::<f32>(buffer, py)?
            .into_iter()
            .map(f64::from)
            .collect())
    } else {
        copy::<f64>(buffer, py)
    }
}

fn token_ids(buffer: &PyUntypedBuffer, py: Python<'_>) -> PyResult<Vec<i32>> {
    if has_format::<i32>(buffer) {
        copy::<i32>(buffer, py)
    } else {
        copy::<i64>(buffer, py)?
            .into_iter()
            .map(|id| {
                i32::try_from(id).map_err(|_| PyOverflowError::new_err("token ID does not fit i32"))
            })
            .collect()
    }
}

pub(super) fn float_vector(value: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    let buffer = PyUntypedBuffer::get(value)?;
    require_rank(&buffer, 1)?;
    float_values(&buffer, value.py())
}

pub(super) fn top_logprobs(
    values: &Bound<'_, PyAny>,
    ids: &Bound<'_, PyAny>,
) -> PyResult<TopLogprobs> {
    let values_buffer = PyUntypedBuffer::get(values)?;
    let ids_buffer = PyUntypedBuffer::get(ids)?;
    require_rank(&values_buffer, 2)?;
    require_rank(&ids_buffer, 2)?;
    if values_buffer.shape() != ids_buffer.shape() {
        return Err(PyValueError::new_err(
            "top-k values and token IDs have different shapes",
        ));
    }
    let (rows, columns) = (values_buffer.shape()[0], values_buffer.shape()[1]);
    fn split<T>(values: Vec<T>, rows: usize, columns: usize) -> Vec<Vec<T>> {
        let mut values = values.into_iter();
        (0..rows)
            .map(|_| values.by_ref().take(columns).collect())
            .collect()
    }
    Ok((
        split(float_values(&values_buffer, values.py())?, rows, columns),
        split(token_ids(&ids_buffer, ids.py())?, rows, columns),
    ))
}

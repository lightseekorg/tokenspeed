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

use super::interpreter;
use crate::data_plane::result_buffers::{float_vector, top_logprobs};
use pyo3::{prelude::*, types::PyDict};

#[test]
fn numeric_buffers_preserve_precision_strides_and_empty_vectors() {
    interpreter();
    Python::attach(|py| {
        let scope = PyDict::new(py);
        py.run(c"from array import array", Some(&scope), None)
            .unwrap();
        let a = py
            .eval(
                c"memoryview(array('f', [1.25, 2.0, -3.5, 4.0]))[::2]",
                Some(&scope),
                None,
            )
            .unwrap();
        assert_eq!(float_vector(&a).unwrap(), [1.25, -3.5]);
        let b = py
            .eval(
                c"array('d', [1.00000000001, -0.0, float('inf'), float('nan')])",
                Some(&scope),
                None,
            )
            .unwrap();
        let values = float_vector(&b).unwrap();
        assert_eq!(values[0], 1.00000000001);
        assert!(values[1].is_sign_negative());
        assert_eq!(values[2], f64::INFINITY);
        assert!(values[3].is_nan());
        let empty = py.eval(c"array('f')", Some(&scope), None).unwrap();
        assert!(float_vector(&empty).unwrap().is_empty());
    });
}

#[test]
fn top_k_buffers_validate_shapes_and_checked_id_conversion() {
    interpreter();
    Python::attach(|py| {
        let scope = PyDict::new(py);
        py.run(c"from array import array", Some(&scope), None)
            .unwrap();
        let values = py
            .eval(
                c"memoryview(array('f', [-0.25, -1.0, -0.5, -2.0])).cast('B').cast('f', [2, 2])",
                Some(&scope),
                None,
            )
            .unwrap();
        let ids = py
            .eval(
                c"memoryview(array('q', [100000, 3, 100001, 4])).cast('B').cast('q', [2, 2])",
                Some(&scope),
                None,
            )
            .unwrap();
        assert_eq!(
            top_logprobs(&values, &ids).unwrap(),
            (
                vec![vec![-0.25, -1.0], vec![-0.5, -2.0]],
                vec![vec![100000, 3], vec![100001, 4]]
            )
        );
        assert!(float_vector(&values).is_err());
        let flat = py
            .eval(c"array('q', [1, 2, 3, 4])", Some(&scope), None)
            .unwrap();
        assert!(top_logprobs(&values, &flat).is_err());
        let wrong_shape = py
            .eval(
                c"memoryview(array('q', [1, 2, 3, 4])).cast('B').cast('q', [1, 4])",
                Some(&scope),
                None,
            )
            .unwrap();
        assert!(top_logprobs(&values, &wrong_shape).is_err());
        let too_large = py
            .eval(
                c"memoryview(array('q', [1 << 40, 2, 3, 4])).cast('B').cast('q', [2, 2])",
                Some(&scope),
                None,
            )
            .unwrap();
        assert!(top_logprobs(&values, &too_large)
            .unwrap_err()
            .is_instance_of::<pyo3::exceptions::PyOverflowError>(py));
    });
}

#[test]
fn copying_or_rejecting_a_buffer_releases_its_export() {
    interpreter();
    Python::attach(|py| {
        let array = py.import("array").unwrap();
        let weakref = py.import("weakref").unwrap();
        for kind in ["f", "h"] {
            let owner = array
                .getattr("array")
                .unwrap()
                .call1((kind, vec![1, 2, 3]))
                .unwrap();
            let weak = weakref.call_method1("ref", (&owner,)).unwrap();
            let copied = float_vector(&owner);
            assert_eq!(copied.is_ok(), kind == "f");
            drop(owner);
            assert!(weak.call0().unwrap().is_none());
            if let Ok(copied) = copied {
                assert_eq!(copied, [1.0, 2.0, 3.0]);
            }
        }
    });
}

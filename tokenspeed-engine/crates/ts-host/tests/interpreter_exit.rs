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

//! An isolated process: retaining the exit GIL must not lock the test harness.

#![allow(unsafe_code)]

#[path = "../src/interpreter.rs"]
mod interpreter;

use pyo3::{ffi, prelude::*};

fn main() {
    // A failed initialization path must also be safe to exit.
    interpreter::hold_gil_for_process_exit();
    let executable = std::env::var("PYO3_PYTHON").expect("set PYO3_PYTHON for the exit test");
    interpreter::start(&executable).unwrap();
    Python::attach(|py| {
        // Import the real extension types whose native statics outlive serving.
        py.import("torch").unwrap();
    });
    // SAFETY: Python is initialized, and PyGILState_Check reads thread state.
    assert_eq!(unsafe { ffi::PyGILState_Check() }, 0);
    interpreter::hold_gil_for_process_exit();
    assert_eq!(unsafe { ffi::PyGILState_Check() }, 1);
    // Process exit runs extension destructors while the host's GIL is held.
}

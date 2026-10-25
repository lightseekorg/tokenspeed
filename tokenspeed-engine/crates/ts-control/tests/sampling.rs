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

//! Sampling-parameter resolution against cases the Python engine resolved
//! (`tests/fixtures/generate.py`).

use ts_control::sampling;
use ts_wire::{decode_bytes, SamplingParams};

fn unhex(text: &str) -> Vec<u8> {
    (0..text.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&text[i..i + 2], 16).unwrap())
        .collect()
}

/// The same parameters, compared as values: the Python encoder writes an
/// integral float field it was given as an int, so bytes can differ.
fn assert_same(rid: &str, rust: &SamplingParams, python: &SamplingParams) {
    assert_eq!(rust, python, "{rid}");
}

#[test]
fn rust_resolves_parameters_as_python_does() {
    let path = format!("{}/tests/fixtures/sampling.hex", env!("CARGO_MANIFEST_DIR"));
    let text = std::fs::read_to_string(path).unwrap();
    let mut cases = 0;
    for line in text.lines() {
        let [raw, rid, expected, verdict] = line.split(' ').collect::<Vec<_>>()[..] else {
            panic!("malformed line {line:?}");
        };
        let rid = String::from_utf8(unhex(rid)).unwrap();
        let mut params: SamplingParams = decode_bytes(&unhex(raw)).unwrap();
        sampling::resolve(&mut params, &rid);
        let python: SamplingParams = decode_bytes(&unhex(expected)).unwrap();
        assert_same(&rid, &params, &python);
        let valid = sampling::verify(&params, 1000);
        assert_eq!(valid.is_ok(), verdict == "ok", "{rid}: {valid:?}");
        cases += 1;
    }
    assert!(cases >= 20, "only {cases} cases");
}

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

use super::parse_args;

#[test]
fn restoring_the_gil_for_exit_is_opt_in() {
    let required = [
        "--python",
        "python3",
        "--control-fd",
        "3",
        "--enable-async-logging",
        "false",
    ];
    let parse =
        |extra: &[&str]| parse_args(required.iter().chain(extra).map(|arg| (*arg).to_owned()));
    assert!(!parse(&[]).unwrap().restore_gil_for_exit);
    assert!(
        parse(&["--restore-gil-for-exit", "true"])
            .unwrap()
            .restore_gil_for_exit
    );
    assert!(
        !parse(&["--restore-gil-for-exit", "false"])
            .unwrap()
            .restore_gil_for_exit
    );
    assert!(parse(&["--restore-gil-for-exit", "yes"]).is_err());
    assert!(parse(&["--restore-gil-for-exit"]).is_err());
}

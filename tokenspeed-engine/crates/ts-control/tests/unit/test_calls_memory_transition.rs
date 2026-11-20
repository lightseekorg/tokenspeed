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

#[test]
fn memory_operations_preserve_transition_ids_and_allow_legacy_calls() {
    for method in ["release_memory_occupation", "resume_memory_occupation"] {
        for supplied in [false, true] {
            let args = if supplied {
                vec![body(vec![
                    ("tags", Value::Array(vec!["kv_cache".into()])),
                    ("transition_id", "training-step-42".into()),
                ])]
            } else {
                Vec::new()
            };
            let (tags, rid) = match rl(method, args).unwrap() {
                ControlRequest::ReleaseMemoryOccupation(request) => (request.tags, request.rid),
                ControlRequest::ResumeMemoryOccupation(request) => (request.tags, request.rid),
                _ => panic!("a memory operation"),
            };
            assert_eq!(tags, supplied.then(|| vec!["kv_cache".to_owned()]));
            assert_eq!(
                rid,
                supplied.then(|| StrOrList::One("training-step-42".to_owned()))
            );
        }
    }
}

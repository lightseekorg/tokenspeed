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

//! The HTTP calls to the bootstrap server ([`crate::bootstrap`]): source
//! ranks register at `PUT /route`; the ranks that read from them look them
//! up at `GET /route` and watch `GET /health`. Plain HTTP; never through a
//! proxy.

use std::time::Duration;

fn agent(connect: Duration, read: Duration) -> ureq::Agent {
    ureq::AgentBuilder::new()
        .timeout_connect(connect)
        .timeout_read(read)
        .build()
}

fn describe(error: ureq::Error) -> String {
    match error {
        ureq::Error::Status(code, response) => {
            format!("{code}, {}", response.into_string().unwrap_or_default())
        }
        ureq::Error::Transport(transport) => transport.to_string(),
    }
}

/// `PUT url` with a JSON body.
pub fn put_json(url: &str, body: &serde_json::Value, timeout: Duration) -> Result<(), String> {
    agent(timeout, timeout)
        .put(url)
        .send_json(body)
        .map(|_| ())
        .map_err(describe)
}

/// `GET url`'s body.
pub fn get_text(url: &str, timeout: Duration) -> Result<String, String> {
    agent(timeout, timeout)
        .get(url)
        .call()
        .map_err(describe)?
        .into_string()
        .map_err(|error| error.to_string())
}

/// Whether `GET url` answers 200 in time.
pub fn healthy(url: &str, connect: Duration, read: Duration) -> bool {
    agent(connect, read)
        .get(url)
        .call()
        .is_ok_and(|response| response.status() == 200)
}

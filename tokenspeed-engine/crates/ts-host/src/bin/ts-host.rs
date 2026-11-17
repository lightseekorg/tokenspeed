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

//! Keep the rank launch protocol independent of the Python ABI.

use std::{ffi::OsString, os::unix::process::CommandExt, process::Command};

fn command() -> Result<Command, String> {
    let mut args = std::env::args_os();
    let title = args.next().unwrap_or_default();
    let mut python = None;
    let mut forwarded: Vec<OsString> = Vec::new();
    while let Some(flag) = args.next() {
        let value = args
            .next()
            .ok_or_else(|| format!("{flag:?} needs a value"))?;
        if flag == "--python" {
            python = Some(value);
        } else {
            forwarded.extend([flag, value]);
        }
    }
    let mut command = Command::new(python.ok_or("--python is required")?);
    command
        .args(["-m", "tokenspeed_engine.host", "--process-title"])
        .arg(title)
        .args(forwarded);
    Ok(command)
}

#[allow(unsafe_code)]
fn main() {
    // SAFETY: getppid has no preconditions. Preserve the original launcher.
    let parent = unsafe { libc::getppid() };
    let error = match command() {
        Ok(mut command) => command.exec().to_string(),
        Err(error) => error,
    };
    // SAFETY: notify only the original positive parent, never a process group.
    if parent > 0 && unsafe { libc::getppid() == parent } {
        unsafe { libc::kill(parent, libc::SIGUSR1) };
    }
    eprintln!("ts-host: {error}");
    std::process::exit(1);
}

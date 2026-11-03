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

//! Prometheus multiprocess files, in the format `prometheus_client` reads
//! (`mmap_dict.py`): the engine's `/metrics` endpoint aggregates every
//! process's files in `PROMETHEUS_MULTIPROC_DIR`, so the host's series join
//! the same scrape as the Python processes' under the same names.
//!
//! A file is a 4-byte used length and 4 bytes of padding, then entries: a
//! 4-byte key length, the UTF-8 key padded with spaces to 8-byte alignment,
//! and two doubles, the value and a timestamp. Integers and doubles are in
//! native byte order. Readers take the used length first and read only up
//! to it, so an entry is written before the length that publishes it.

use std::{
    collections::HashMap,
    fs::{File, OpenOptions},
    io,
    os::unix::fs::FileExt,
    path::{Path, PathBuf},
};

/// The size `prometheus_client` gives a new file.
const INITIAL_SIZE: u64 = 1 << 16;

/// One `{kind}_{pid}.db` file.
#[derive(Debug)]
pub(crate) struct PromFile {
    file: File,
    used: u64,
    /// Where each key's value lives.
    positions: HashMap<String, u64>,
}

impl PromFile {
    /// A new, empty file at `path`, replacing one a previous process with
    /// this pid left.
    pub(crate) fn create(path: &Path) -> io::Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;
        file.set_len(INITIAL_SIZE)?;
        let used = 8;
        file.write_all_at(&(used as i32).to_ne_bytes(), 0)?;
        Ok(Self {
            file,
            used,
            positions: HashMap::new(),
        })
    }

    /// Store `value` under `key`, adding the key on first use.
    pub(crate) fn write(&mut self, key: &str, value: f64, timestamp: f64) -> io::Result<()> {
        let position = match self.positions.get(key) {
            Some(&position) => position,
            None => self.append(key)?,
        };
        let mut doubles = [0u8; 16];
        doubles[..8].copy_from_slice(&value.to_ne_bytes());
        doubles[8..].copy_from_slice(&timestamp.to_ne_bytes());
        self.file.write_all_at(&doubles, position)
    }

    fn append(&mut self, key: &str) -> io::Result<u64> {
        let encoded = key.as_bytes();
        let padding = 8 - (encoded.len() + 4) % 8;
        let mut entry = Vec::with_capacity(4 + encoded.len() + padding + 16);
        entry.extend_from_slice(&(encoded.len() as i32).to_ne_bytes());
        entry.extend_from_slice(encoded);
        entry.resize(entry.len() + padding, b' ');
        entry.extend_from_slice(&[0u8; 16]);
        self.file.write_all_at(&entry, self.used)?;
        self.used += entry.len() as u64;
        self.file
            .write_all_at(&(self.used as i32).to_ne_bytes(), 0)?;
        let position = self.used - 16;
        self.positions.insert(key.to_owned(), position);
        Ok(position)
    }
}

/// The files of one process, opened on first use.
#[derive(Debug)]
pub(crate) struct PromFiles {
    directory: PathBuf,
    pid: u32,
    files: HashMap<&'static str, PromFile>,
}

impl PromFiles {
    pub(crate) fn new(directory: PathBuf) -> Self {
        Self {
            directory,
            pid: std::process::id(),
            files: HashMap::new(),
        }
    }

    /// Store `value` under `key` in the file for `prefix` (`counter`,
    /// `gauge_livemax`, `histogram`).
    pub(crate) fn write(&mut self, prefix: &'static str, key: &str, value: f64) -> io::Result<()> {
        let file = match self.files.get_mut(prefix) {
            Some(file) => file,
            None => {
                let path = self.directory.join(format!("{prefix}_{}.db", self.pid));
                self.files.entry(prefix).or_insert(PromFile::create(&path)?)
            }
        };
        file.write(key, value, 0.0)
    }
}

/// The key `prometheus_client` stores a sample under (`mmap_key`):
/// `json.dumps([metric_name, name, labels, help_text], sort_keys=True)`,
/// byte for byte, so a series another process writes merges with it.
pub(crate) fn sample_key(
    metric_name: &str,
    name: &str,
    labels: &[(&str, &str)],
    help_text: &str,
) -> String {
    let mut sorted: Vec<&(&str, &str)> = labels.iter().collect();
    sorted.sort_by(|a, b| a.0.cmp(b.0));
    let mut key = String::from("[");
    push_json_str(&mut key, metric_name);
    key.push_str(", ");
    push_json_str(&mut key, name);
    key.push_str(", {");
    for (index, (label, value)) in sorted.iter().enumerate() {
        if index > 0 {
            key.push_str(", ");
        }
        push_json_str(&mut key, label);
        key.push_str(": ");
        push_json_str(&mut key, value);
    }
    key.push_str("}, ");
    push_json_str(&mut key, help_text);
    key.push(']');
    key
}

/// A JSON string as Python's `json.dumps` writes it (`ensure_ascii`).
fn push_json_str(out: &mut String, value: &str) {
    out.push('"');
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            ' '..='~' => out.push(ch),
            _ => {
                let mut units = [0u16; 2];
                for unit in ch.encode_utf16(&mut units) {
                    out.push_str(&format!("\\u{unit:04x}"));
                }
            }
        }
    }
    out.push('"');
}

/// A histogram bound as its `le` label (`floatToGoString`): Python's float
/// repr, with Go's earlier switch to an exponent.
pub(crate) fn le_label(bound: f64) -> String {
    if bound == f64::INFINITY {
        return "+Inf".to_owned();
    }
    let repr = python_float_repr(bound);
    if let Some(dot) = repr.find('.') {
        if bound > 0.0 && dot > 6 {
            let digits = format!("{}{}", &repr[1..dot], &repr[dot + 1..]);
            let mantissa = format!("{}.{digits}", &repr[..1]);
            let mantissa = mantissa.trim_end_matches(['0', '.']);
            return format!("{mantissa}e+{:02}", dot - 1);
        }
    }
    repr
}

/// `repr(float)`: the shortest round-trip digits, fixed between 1e-4 and
/// 1e16, else an exponent of at least two digits.
fn python_float_repr(value: f64) -> String {
    let shortest = format!("{value:e}");
    let (mantissa, exponent) = shortest.split_once('e').expect("`{:e}` has an exponent");
    let exponent: i32 = exponent.parse().expect("an integer exponent");
    if (-4..16).contains(&exponent) {
        let fixed = format!("{value}");
        if fixed.contains('.') {
            fixed
        } else {
            format!("{fixed}.0")
        }
    } else {
        let sign = if exponent < 0 { '-' } else { '+' };
        format!("{mantissa}e{sign}{:02}", exponent.abs())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_are_python_json() {
        assert_eq!(
            sample_key(
                "tokenspeed:x",
                "tokenspeed:x_total",
                &[("dp_rank", "0"), ("app_key", "")],
                "Help \"text\".",
            ),
            r#"["tokenspeed:x", "tokenspeed:x_total", {"app_key": "", "dp_rank": "0"}, "Help \"text\"."]"#
        );
        assert_eq!(
            sample_key("m", "m", &[], "é"),
            r#"["m", "m", {}, "\u00e9"]"#
        );
    }

    #[test]
    fn bounds_format_like_float_to_go_string() {
        let cases = [
            (0.001, "0.001"),
            (0.3, "0.3"),
            (1.0, "1.0"),
            (7680.0, "7680.0"),
            (100000.0, "100000.0"),
            (1000000.0, "1e+06"),
            (2000000.0, "2e+06"),
            (1234567.0, "1.234567e+06"),
            (f64::INFINITY, "+Inf"),
        ];
        for (bound, expected) in cases {
            assert_eq!(le_label(bound), expected, "{bound}");
        }
    }
}

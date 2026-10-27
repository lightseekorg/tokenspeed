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

//! Two engines in one process write host memory to each other over RDMA.
//!
//! Needs an RDMA device and `TS_MOONCAKE_LIBRARY` naming
//! `libtokenspeed_transfer_engine.so`; without the variable the test returns
//! at once, as the library and the devices are not on every machine.

use std::{path::PathBuf, sync::Arc};

use ts_mooncake::{Engine, Library, PageGrid};

fn library() -> Option<Arc<Library>> {
    let path = PathBuf::from(std::env::var_os("TS_MOONCAKE_LIBRARY")?);
    Some(Library::load(&path).unwrap())
}

#[test]
fn spans_and_page_grids_land_on_the_peer() {
    let Some(library) = library() else {
        eprintln!("TS_MOONCAKE_LIBRARY is not set; skipping");
        return;
    };
    let sender = Engine::start(library.clone(), "127.0.0.1", "rdma", "").unwrap();
    let receiver = Engine::start(library, "127.0.0.1", "rdma", "").unwrap();
    assert_ne!(sender.session_id(), receiver.session_id());

    const PAGE: usize = 64;
    let source: Vec<u8> = (0..PAGE * 8).map(|i| (i % 251) as u8).collect();
    let mut destination = vec![0u8; PAGE * 8];
    let (src, dst) = (source.as_ptr() as usize, destination.as_mut_ptr() as usize);
    sender.register(src, source.len()).unwrap();
    receiver.register(dst, destination.len()).unwrap();

    // One span: the first page lands in the last.
    sender
        .write(
            receiver.session_id(),
            &[src as u64],
            &[(dst + 7 * PAGE) as u64],
            &[PAGE as u64],
        )
        .unwrap();
    assert_eq!(destination[7 * PAGE..], source[..PAGE]);

    // A grid of two fields (each half a page) over pages 1 and 2 to pages 3
    // and 5, written one span per batch.
    let half = (PAGE / 2) as i64;
    let grid = PageGrid {
        src_pages: vec![1, 2],
        dst_pages: vec![3, 5],
        fields: vec![
            [src as i64, PAGE as i64, dst as i64, PAGE as i64, half],
            [
                src as i64 + half,
                PAGE as i64,
                dst as i64 + half,
                PAGE as i64,
                half,
            ],
        ],
    };
    sender.write_pages(receiver.session_id(), &grid, 1).unwrap();
    assert_eq!(destination[3 * PAGE..4 * PAGE], source[PAGE..2 * PAGE]);
    assert_eq!(destination[5 * PAGE..6 * PAGE], source[2 * PAGE..3 * PAGE]);
    assert!(destination[4 * PAGE..5 * PAGE]
        .iter()
        .all(|&byte| byte == 0));

    sender.unregister(src).unwrap();
    receiver.unregister(dst).unwrap();
}

#[test]
fn a_missing_library_is_a_load_error() {
    let missing = PathBuf::from("/nonexistent/libtokenspeed_transfer_engine.so");
    assert!(matches!(
        Library::load(&missing),
        Err(ts_mooncake::Error::Load { .. })
    ));
}

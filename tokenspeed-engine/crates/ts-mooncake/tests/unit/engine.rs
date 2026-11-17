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

//! The real engine wrapper with only its native network calls replaced.

use std::ffi::CStr;

use super::*;

unsafe extern "C" fn create(
    host: *const c_char,
    metadata: *const c_char,
    protocol: *const c_char,
    devices: *const c_char,
) -> *mut c_void {
    // SAFETY: Engine::start holds each C string for the call.
    let arguments = [host, metadata, protocol, devices]
        .map(|value| unsafe { CStr::from_ptr(value).to_str().unwrap().to_owned() });
    Box::into_raw(Box::new(arguments)).cast()
}

unsafe extern "C" fn destroy(raw: *mut c_void) {
    // SAFETY: create allocated this box; Engine::drop destroys it once.
    drop(unsafe { Box::from_raw(raw.cast::<[String; 4]>()) });
}

unsafe extern "C" fn rpc_port(_: *mut c_void) -> c_int {
    4321
}

unsafe extern "C" fn register(_: *mut c_void, _: *mut c_void, _: usize) -> c_int {
    0
}

unsafe extern "C" fn unregister(_: *mut c_void, _: *mut c_void) -> c_int {
    0
}

unsafe extern "C" fn write(
    _: *mut c_void,
    _: *const c_char,
    _: *const u64,
    _: *const u64,
    _: *const u64,
    _: usize,
) -> c_int {
    0
}

unsafe extern "C" fn write_pages(
    _: *mut c_void,
    _: *const c_char,
    _: *const i64,
    _: *const i64,
    _: usize,
    _: *const i64,
    _: usize,
    _: usize,
) -> c_int {
    0
}

/// The last span READ: (target, sources, destinations, lengths), as the C
/// entry point names them: `sources` on the peer, `destinations` local.
type SpanRead = (String, Vec<u64>, Vec<u64>, Vec<u64>);
static LAST_SPAN_READ: std::sync::Mutex<Option<SpanRead>> = std::sync::Mutex::new(None);

unsafe extern "C" fn read(
    _: *mut c_void,
    target: *const c_char,
    sources: *const u64,
    destinations: *const u64,
    lengths: *const u64,
    count: usize,
) -> c_int {
    // SAFETY: Engine::read passes columns of `count` entries.
    let read = unsafe {
        (
            CStr::from_ptr(target).to_str().unwrap().to_owned(),
            std::slice::from_raw_parts(sources, count).to_vec(),
            std::slice::from_raw_parts(destinations, count).to_vec(),
            std::slice::from_raw_parts(lengths, count).to_vec(),
        )
    };
    *LAST_SPAN_READ.lock().unwrap() = Some(read);
    0
}

/// The last page READ: (target, source pages, destination pages, fields,
/// batch size).
type PageRead = (String, Vec<i64>, Vec<i64>, Vec<i64>, usize);
static LAST_PAGE_READ: std::sync::Mutex<Option<PageRead>> = std::sync::Mutex::new(None);

unsafe extern "C" fn read_pages(
    _: *mut c_void,
    target: *const c_char,
    src_pages: *const i64,
    dst_pages: *const i64,
    num_pages: usize,
    fields: *const i64,
    num_fields: usize,
    max_batch_size: usize,
) -> c_int {
    // SAFETY: Engine::read_pages passes columns of the counts it states.
    let read = unsafe {
        (
            CStr::from_ptr(target).to_str().unwrap().to_owned(),
            std::slice::from_raw_parts(src_pages, num_pages).to_vec(),
            std::slice::from_raw_parts(dst_pages, num_pages).to_vec(),
            std::slice::from_raw_parts(fields, num_fields * 5).to_vec(),
            max_batch_size,
        )
    };
    *LAST_PAGE_READ.lock().unwrap() = Some(read);
    0
}

/// A library of ABI version 1 from before the READ entries.
fn library() -> Arc<Library> {
    Arc::new(Library {
        path: PathBuf::from("test-process"),
        create,
        destroy,
        rpc_port,
        register,
        unregister,
        write,
        write_pages,
        read: None,
        read_pages: None,
        // All function pointers above live in this test process.
        _library: libloading::os::unix::Library::this().into(),
    })
}

/// A library with the READ entries.
fn library_with_reads() -> Arc<Library> {
    Arc::new(Library {
        path: PathBuf::from("test-process"),
        create,
        destroy,
        rpc_port,
        register,
        unregister,
        write,
        write_pages,
        read: Some(read),
        read_pages: Some(read_pages),
        _library: libloading::os::unix::Library::this().into(),
    })
}

#[test]
fn a_library_without_reads_refuses_a_read_and_says_so() {
    let library = library();
    assert!(!library.reads());
    let engine = Engine::start(library, "127.0.0.1", "rdma", "").unwrap();
    assert!(matches!(
        engine.read_pages("peer:1", &PageGrid::default(), 0),
        Err(Error::NoReads(_))
    ));
    assert!(matches!(
        engine.read("peer:1", &[], &[], &[]),
        Err(Error::NoReads(_))
    ));
}

#[test]
fn a_span_read_takes_the_peers_addresses_as_its_sources() {
    let engine = Engine::start(library_with_reads(), "127.0.0.1", "rdma", "").unwrap();
    engine
        .read("peer:1", &[0x1000, 0x2000], &[0x9000, 0xa000], &[64, 32])
        .unwrap();
    assert_eq!(
        LAST_SPAN_READ.lock().unwrap().clone(),
        Some((
            "peer:1".to_owned(),
            vec![0x1000, 0x2000],
            vec![0x9000, 0xa000],
            vec![64, 32]
        )),
        "the C entry point takes the peer's sources first and the local destinations second"
    );
}

#[test]
fn a_page_read_takes_the_peers_pages_as_its_source() {
    let library = library_with_reads();
    assert!(library.reads());
    let engine = Engine::start(library, "127.0.0.1", "rdma", "").unwrap();
    let grid = PageGrid {
        src_pages: vec![7, 9],
        dst_pages: vec![1, 2],
        fields: vec![[1000, 64, 2000, 32, 16]],
    };
    engine.read_pages("peer:1", &grid, 8).unwrap();
    assert_eq!(
        LAST_PAGE_READ.lock().unwrap().clone(),
        Some((
            "peer:1".to_owned(),
            vec![7, 9],
            vec![1, 2],
            vec![1000, 64, 2000, 32, 16],
            8
        ))
    );
}

#[test]
fn selected_protocol_reaches_the_native_engine() {
    for protocol in ["rdma", "ascend_direct"] {
        let engine = Engine::start(library(), "127.0.0.1", protocol, "device0").unwrap();
        // SAFETY: the fake native engine owns these arguments until drop.
        let arguments = unsafe { engine.raw.cast::<[String; 4]>().as_ref() };
        assert_eq!(
            arguments,
            &["127.0.0.1", "P2PHANDSHAKE", protocol, "device0"]
        );
        assert_eq!(engine.session_id(), "127.0.0.1:4321");
    }
}

#[test]
fn protocol_with_a_nul_is_rejected_before_native_initialization() {
    assert!(matches!(
        Engine::start(library(), "127.0.0.1", "rdma\0ascend_direct", ""),
        Err(Error::Nul("protocol"))
    ));
}

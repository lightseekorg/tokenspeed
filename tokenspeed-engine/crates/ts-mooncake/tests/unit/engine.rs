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
        // All function pointers above live in this test process.
        _library: libloading::os::unix::Library::this().into(),
    })
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

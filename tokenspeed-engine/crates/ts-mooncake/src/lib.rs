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

//! The Mooncake transfer engine and store client, owned by the engine host.
//!
//! `tokenspeed-mooncake` ships `libtokenspeed_transfer_engine.so` beside its
//! Python module: TokenSpeed's C entry points
//! (`mooncake/include/transfer_engine_tokenspeed.h`), which behave as the
//! Python `TransferEngine` methods TokenSpeed calls. The store client comes
//! from the same wheel's `libmooncake_store.so` ([`store`]). The libraries
//! are loaded at run time, so building the host needs no Mooncake, an engine
//! without PD never loads the first and one without the L3 store never loads
//! the second.

// The crate is the FFI boundary to the library.
#![allow(unsafe_code)]

use std::{
    ffi::{c_char, c_int, c_void, CString},
    path::{Path, PathBuf},
    ptr::NonNull,
    sync::Arc,
};

pub mod store;

#[cfg(test)]
#[path = "../tests/unit/engine.rs"]
mod tests;

/// The ABI of the entry points this crate calls.
const ABI_VERSION: c_int = 1;

/// The library's file name inside the `mooncake` package.
pub const LIBRARY_NAME: &str = "libtokenspeed_transfer_engine.so";

/// A failure of the transfer engine.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The library did not load or lacks an entry point.
    #[error("loading {path}: {source}")]
    Load {
        /// Where it was looked for.
        path: PathBuf,
        /// Why it failed.
        source: libloading::Error,
    },
    /// The library implements another ABI.
    #[error("{path} implements ABI {found}, this host needs {ABI_VERSION}")]
    Abi {
        /// The library.
        path: PathBuf,
        /// Its ABI.
        found: c_int,
    },
    /// The engine did not start.
    #[error("the transfer engine did not start on {0}")]
    Create(String),
    /// An argument holds a NUL byte.
    #[error("{0} holds a NUL byte")]
    Nul(&'static str),
    /// A memory registration failed.
    #[error("registering {length} bytes at {addr:#x} failed ({status})")]
    Register {
        /// The region.
        addr: usize,
        /// Its length.
        length: usize,
        /// The engine's status.
        status: c_int,
    },
    /// A deregistration failed.
    #[error("deregistering {addr:#x} failed ({status})")]
    Unregister {
        /// The region.
        addr: usize,
        /// The engine's status.
        status: c_int,
    },
    /// A transfer failed or timed out.
    #[error("transfer to {target} failed ({status})")]
    Transfer {
        /// The peer session.
        target: String,
        /// The engine's status.
        status: c_int,
    },
    /// The columns of a transfer disagree.
    #[error("transfer columns disagree: {0}")]
    Shape(&'static str),
    /// The store client did not connect.
    #[error("the store client did not connect to {master} ({status})")]
    Setup {
        /// The master's address.
        master: String,
        /// The client's status.
        status: c_int,
    },
    /// A store call failed as a whole.
    #[error("{call} failed ({status})")]
    Call {
        /// The call.
        call: &'static str,
        /// Its status.
        status: i64,
    },
}

type Create =
    unsafe extern "C" fn(*const c_char, *const c_char, *const c_char, *const c_char) -> *mut c_void;
type Destroy = unsafe extern "C" fn(*mut c_void);
type RpcPort = unsafe extern "C" fn(*mut c_void) -> c_int;
type Register = unsafe extern "C" fn(*mut c_void, *mut c_void, usize) -> c_int;
type Unregister = unsafe extern "C" fn(*mut c_void, *mut c_void) -> c_int;
type Write = unsafe extern "C" fn(
    *mut c_void,
    *const c_char,
    *const u64,
    *const u64,
    *const u64,
    usize,
) -> c_int;
type WritePages = unsafe extern "C" fn(
    *mut c_void,
    *const c_char,
    *const i64,
    *const i64,
    usize,
    *const i64,
    usize,
    usize,
) -> c_int;

/// The loaded library.
pub struct Library {
    path: PathBuf,
    create: Create,
    destroy: Destroy,
    rpc_port: RpcPort,
    register: Register,
    unregister: Unregister,
    write: Write,
    write_pages: WritePages,
    /// Keeps the functions above mapped.
    _library: libloading::Library,
}

impl Library {
    /// Load the library at `path`.
    pub fn load(path: &Path) -> Result<Arc<Self>, Error> {
        let load_error = |source| Error::Load {
            path: path.to_owned(),
            source,
        };
        // SAFETY: loading runs the library's initializers, which only set up
        // the engine's statics.
        let library = unsafe { libloading::Library::new(path) }.map_err(load_error)?;
        // SAFETY: each symbol has the C signature of
        // transfer_engine_tokenspeed.h at ABI_VERSION, checked first.
        unsafe {
            let abi: libloading::Symbol<unsafe extern "C" fn() -> c_int> = library
                .get(b"ts_mooncake_abi_version")
                .map_err(load_error)?;
            let found = abi();
            if found != ABI_VERSION {
                return Err(Error::Abi {
                    path: path.to_owned(),
                    found,
                });
            }
            Ok(Arc::new(Self {
                path: path.to_owned(),
                create: *library.get(b"ts_mooncake_create").map_err(load_error)?,
                destroy: *library.get(b"ts_mooncake_destroy").map_err(load_error)?,
                rpc_port: *library.get(b"ts_mooncake_rpc_port").map_err(load_error)?,
                register: *library.get(b"ts_mooncake_register").map_err(load_error)?,
                unregister: *library.get(b"ts_mooncake_unregister").map_err(load_error)?,
                write: *library.get(b"ts_mooncake_write").map_err(load_error)?,
                write_pages: *library
                    .get(b"ts_mooncake_write_pages")
                    .map_err(load_error)?,
                _library: library,
            }))
        }
    }

    /// Where the library was loaded from.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

fn c_string(value: &str, name: &'static str) -> Result<CString, Error> {
    CString::new(value).map_err(|_| Error::Nul(name))
}

/// A started transfer engine.
///
/// Every operation may be called from several threads at once, as the
/// Python binding's are.
pub struct Engine {
    library: Arc<Library>,
    raw: NonNull<c_void>,
    session_id: String,
}

// SAFETY: the engine's entry points lock what they share (the segment-handle
// cache) and the engine itself is thread-safe, as the Python binding relies
// on from its transfer worker threads.
unsafe impl Send for Engine {}
// SAFETY: as above.
unsafe impl Sync for Engine {}

impl Engine {
    /// Start an engine on `hostname`, through peer-to-peer handshakes over
    /// `protocol`, on the devices `device_filter` names (comma-separated; empty
    /// for all). Its session id is `hostname:rpc_port`, with an IPv6 host in
    /// brackets.
    pub fn start(
        library: Arc<Library>,
        hostname: &str,
        protocol: &str,
        device_filter: &str,
    ) -> Result<Self, Error> {
        let host = c_string(hostname, "hostname")?;
        let metadata = c_string("P2PHANDSHAKE", "metadata server")?;
        let protocol = c_string(protocol, "protocol")?;
        let devices = c_string(device_filter, "device filter")?;
        // SAFETY: the strings outlive the call.
        let raw = unsafe {
            (library.create)(
                host.as_ptr(),
                metadata.as_ptr(),
                protocol.as_ptr(),
                devices.as_ptr(),
            )
        };
        let raw = NonNull::new(raw).ok_or_else(|| Error::Create(hostname.to_owned()))?;
        // SAFETY: `raw` is a started engine.
        let port = unsafe { (library.rpc_port)(raw.as_ptr()) };
        let session_id = if hostname.contains(':') && !hostname.starts_with('[') {
            format!("[{hostname}]:{port}")
        } else {
            format!("{hostname}:{port}")
        };
        Ok(Self {
            library,
            raw,
            session_id,
        })
    }

    /// The id peers address this engine's memory by.
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// Register `length` bytes at `addr` (device or host memory) as a
    /// transfer source or destination.
    pub fn register(&self, addr: usize, length: usize) -> Result<(), Error> {
        // SAFETY: the engine only records the range; the caller keeps it
        // mapped until it is deregistered.
        let status =
            unsafe { (self.library.register)(self.raw.as_ptr(), addr as *mut c_void, length) };
        if status != 0 {
            return Err(Error::Register {
                addr,
                length,
                status,
            });
        }
        Ok(())
    }

    /// Undo [`Engine::register`].
    pub fn unregister(&self, addr: usize) -> Result<(), Error> {
        // SAFETY: as in `register`.
        let status = unsafe { (self.library.unregister)(self.raw.as_ptr(), addr as *mut c_void) };
        if status != 0 {
            return Err(Error::Unregister { addr, status });
        }
        Ok(())
    }

    /// WRITE spans to the peer session `target` and wait: span i copies
    /// `lengths[i]` bytes from local `sources[i]` to the peer's
    /// `destinations[i]`.
    pub fn write(
        &self,
        target: &str,
        sources: &[u64],
        destinations: &[u64],
        lengths: &[u64],
    ) -> Result<(), Error> {
        if sources.len() != destinations.len() || sources.len() != lengths.len() {
            return Err(Error::Shape("sources, destinations and lengths"));
        }
        let target_c = c_string(target, "target")?;
        // SAFETY: the columns hold `sources.len()` entries each and outlive
        // the call; the addresses are registered by the two engines.
        let status = unsafe {
            (self.library.write)(
                self.raw.as_ptr(),
                target_c.as_ptr(),
                sources.as_ptr(),
                destinations.as_ptr(),
                lengths.as_ptr(),
                sources.len(),
            )
        };
        if status != 0 {
            return Err(Error::Transfer {
                target: target.to_owned(),
                status,
            });
        }
        Ok(())
    }

    /// WRITE a pages x fields grid to `target` and wait; see
    /// [`PageGrid`].
    pub fn write_pages(
        &self,
        target: &str,
        grid: &PageGrid,
        max_batch_size: usize,
    ) -> Result<(), Error> {
        if grid.src_pages.len() != grid.dst_pages.len() {
            return Err(Error::Shape("source and destination pages"));
        }
        let target_c = c_string(target, "target")?;
        let fields: Vec<i64> = grid.fields.iter().flatten().copied().collect();
        // SAFETY: as in `write`; `fields` holds five values per field.
        let status = unsafe {
            (self.library.write_pages)(
                self.raw.as_ptr(),
                target_c.as_ptr(),
                grid.src_pages.as_ptr(),
                grid.dst_pages.as_ptr(),
                grid.src_pages.len(),
                fields.as_ptr(),
                grid.fields.len(),
                max_batch_size,
            )
        };
        if status != 0 {
            return Err(Error::Transfer {
                target: target.to_owned(),
                status,
            });
        }
        Ok(())
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        // SAFETY: `raw` is this engine's, destroyed once.
        unsafe { (self.library.destroy)(self.raw.as_ptr()) };
    }
}

/// One page-gathered WRITE: span (f, i) copies `fields[f][4]` bytes from
/// `fields[f][0] + src_pages[i] * fields[f][1]` to
/// `fields[f][2] + dst_pages[i] * fields[f][3]`.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PageGrid {
    /// Local pages.
    pub src_pages: Vec<i64>,
    /// The peer's pages, one per local page.
    pub dst_pages: Vec<i64>,
    /// Per field: source base, source page stride, destination base,
    /// destination page stride, length.
    pub fields: Vec<[i64; 5]>,
}

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

//! The Mooncake store client, owned by the engine host.
//!
//! `tokenspeed-mooncake` ships `libmooncake_store.so` beside its Python
//! module: upstream's store C ABI (`mooncake/include/store_c.h`), built so
//! it exports only the `mooncake_store_*` functions, plus TokenSpeed's
//! `mooncake_store_setup_with_tenant`. An upstream SDK with only the original
//! setup entry point supports the default tenant explicitly. The calls behave as the Python
//! `MooncakeDistributedStore` methods of the same names. The library is
//! loaded at run time, so an engine without the L3 store never loads it.

use std::{
    ffi::{c_char, c_int, c_void, CString},
    path::{Path, PathBuf},
    ptr::NonNull,
    sync::Arc,
};

use crate::Error;

/// The library's file name inside the `mooncake` package.
pub const STORE_LIBRARY_NAME: &str = "libmooncake_store.so";

type Create = unsafe extern "C" fn() -> *mut c_void;
type Destroy = unsafe extern "C" fn(*mut c_void);
type SetupWithTenant = unsafe extern "C" fn(
    *mut c_void,
    *const c_char,
    *const c_char,
    u64,
    u64,
    *const c_char,
    *const c_char,
    *const c_char,
    *const c_char,
) -> c_int;
type SetupDefaultTenant = unsafe extern "C" fn(
    *mut c_void,
    *const c_char,
    *const c_char,
    u64,
    u64,
    *const c_char,
    *const c_char,
    *const c_char,
) -> c_int;

#[derive(Clone, Copy)]
enum Setup {
    WithTenant(SetupWithTenant),
    DefaultTenant(SetupDefaultTenant),
}
type RegisterBuffer = unsafe extern "C" fn(*mut c_void, *mut c_void, usize) -> c_int;
type UnregisterBuffer = unsafe extern "C" fn(*mut c_void, *mut c_void) -> c_int;
type BatchIsExist =
    unsafe extern "C" fn(*mut c_void, *const *const c_char, usize, *mut c_int) -> c_int;
type BatchGetInto = unsafe extern "C" fn(
    *mut c_void,
    *const *const c_char,
    *const *mut c_void,
    *const usize,
    usize,
    *mut i64,
) -> c_int;
type BatchPutFrom = unsafe extern "C" fn(
    *mut c_void,
    *const *const c_char,
    *const *mut c_void,
    *const usize,
    usize,
    *const c_void,
    *mut c_int,
) -> c_int;
type RemoveByRegex = unsafe extern "C" fn(*mut c_void, *const c_char, c_int) -> i64;
type BatchPutFromWithRetirement = unsafe extern "C" fn(
    *mut c_void,
    *const *const c_char,
    *const *mut c_void,
    *const usize,
    usize,
    *const c_void,
    *mut c_int,
    *mut u8,
) -> c_int;

/// Publication results and the native source-buffer lifetime, independently.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PutOutcome {
    /// Per key, zero for a successful End, or the SDK's negative status.
    /// Legacy SDKs may also collapse a duplicate rejection into zero.
    pub results: Vec<i32>,
    /// None for an SDK without the retirement extension; false means unknown.
    pub source_retired: Option<bool>,
}

/// The loaded store library.
pub struct StoreLibrary {
    path: PathBuf,
    create: Create,
    destroy: Destroy,
    setup: Setup,
    register_buffer: RegisterBuffer,
    unregister_buffer: UnregisterBuffer,
    batch_is_exist: BatchIsExist,
    batch_get_into: BatchGetInto,
    batch_put_from: BatchPutFrom,
    batch_put_from_with_retirement: Option<BatchPutFromWithRetirement>,
    remove_by_regex: RemoveByRegex,
    /// Keeps the functions above mapped.
    _library: libloading::Library,
}

impl StoreLibrary {
    /// Load the library at `path`.
    pub fn load(path: &Path) -> Result<Arc<Self>, Error> {
        let load_error = |source| Error::Load {
            path: path.to_owned(),
            source,
        };
        // SAFETY: loading runs the library's initializers, which only set up
        // the client's statics.
        let library = unsafe { libloading::Library::new(path) }.map_err(load_error)?;
        // SAFETY: each symbol has the C signature store_c.h declares.
        unsafe {
            Ok(Arc::new(Self {
                path: path.to_owned(),
                create: *library.get(b"mooncake_store_create").map_err(load_error)?,
                destroy: *library.get(b"mooncake_store_destroy").map_err(load_error)?,
                setup: match library.get::<SetupWithTenant>(b"mooncake_store_setup_with_tenant") {
                    Ok(symbol) => Setup::WithTenant(*symbol),
                    Err(_) => Setup::DefaultTenant(
                        *library.get(b"mooncake_store_setup").map_err(load_error)?,
                    ),
                },
                register_buffer: *library
                    .get(b"mooncake_store_register_buffer")
                    .map_err(load_error)?,
                unregister_buffer: *library
                    .get(b"mooncake_store_unregister_buffer")
                    .map_err(load_error)?,
                batch_is_exist: *library
                    .get(b"mooncake_store_batch_is_exist")
                    .map_err(load_error)?,
                batch_get_into: *library
                    .get(b"mooncake_store_batch_get_into")
                    .map_err(load_error)?,
                batch_put_from: *library
                    .get(b"mooncake_store_batch_put_from")
                    .map_err(load_error)?,
                batch_put_from_with_retirement: library
                    .get::<BatchPutFromWithRetirement>(
                        b"mooncake_store_batch_put_from_with_retirement_v1",
                    )
                    .ok()
                    .map(|symbol| *symbol),
                remove_by_regex: *library
                    .get(b"mooncake_store_remove_by_regex")
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

/// What `MooncakeDistributedStore.setup` takes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoreSetup {
    /// This client's host name.
    pub local_hostname: String,
    /// The transfer engine's metadata server (`P2PHANDSHAKE` for none).
    pub metadata_server: String,
    /// The bytes this client mounts into the store; 0 mounts none.
    pub global_segment_size: u64,
    /// The client's local transfer buffer.
    pub local_buffer_size: u64,
    /// `tcp` or `rdma`.
    pub protocol: String,
    /// The RDMA devices, comma-separated; empty for all.
    pub device_name: String,
    /// The master's address.
    pub master_server_address: String,
    /// The tenant whose objects this client sees; `default` is shared.
    pub tenant_id: String,
}

/// A store client.
///
/// Every operation may be called from several threads at once, as the
/// Python binding's are from the L3 executor's workers.
pub struct Store {
    library: Arc<StoreLibrary>,
    raw: NonNull<c_void>,
}

// SAFETY: the client is thread-safe; the Python binding relies on it from
// several worker threads at once.
unsafe impl Send for Store {}
// SAFETY: as above.
unsafe impl Sync for Store {}

/// Keys as NUL-terminated strings, and the pointer array C takes.
struct CKeys {
    _owned: Vec<CString>,
    pointers: Vec<*const c_char>,
}

impl CKeys {
    fn new(keys: &[String]) -> Result<Self, Error> {
        let owned = keys
            .iter()
            .map(|key| CString::new(key.as_str()).map_err(|_| Error::Nul("key")))
            .collect::<Result<Vec<_>, _>>()?;
        let pointers = owned.iter().map(|key| key.as_ptr()).collect();
        Ok(Self {
            _owned: owned,
            pointers,
        })
    }
}

impl Store {
    /// Create a client and connect it to the store.
    pub fn connect(library: Arc<StoreLibrary>, setup: &StoreSetup) -> Result<Self, Error> {
        if matches!(library.setup, Setup::DefaultTenant(_)) && setup.tenant_id != "default" {
            // The upstream ABI has no tenant argument. Never silently connect a
            // named tenant to the shared default namespace.
            return Err(Error::Setup {
                master: setup.master_server_address.clone(),
                status: -1012, // NOT_SUPPORTED
            });
        }
        // SAFETY: plain constructor.
        let raw = unsafe { (library.create)() };
        let raw =
            NonNull::new(raw).ok_or_else(|| Error::Create(setup.master_server_address.clone()))?;
        let store = Self { library, raw };
        let hostname = CString::new(setup.local_hostname.as_str())
            .map_err(|_| Error::Nul("local hostname"))?;
        let metadata = CString::new(setup.metadata_server.as_str())
            .map_err(|_| Error::Nul("metadata server"))?;
        let protocol = CString::new(setup.protocol.as_str()).map_err(|_| Error::Nul("protocol"))?;
        let device =
            CString::new(setup.device_name.as_str()).map_err(|_| Error::Nul("device name"))?;
        let master = CString::new(setup.master_server_address.as_str())
            .map_err(|_| Error::Nul("master address"))?;
        let tenant = CString::new(setup.tenant_id.as_str()).map_err(|_| Error::Nul("tenant"))?;
        // SAFETY: the strings outlive the call; the client copies them.
        let status = unsafe {
            match store.library.setup {
                Setup::WithTenant(call) => call(
                    store.raw.as_ptr(),
                    hostname.as_ptr(),
                    metadata.as_ptr(),
                    setup.global_segment_size,
                    setup.local_buffer_size,
                    protocol.as_ptr(),
                    device.as_ptr(),
                    master.as_ptr(),
                    tenant.as_ptr(),
                ),
                Setup::DefaultTenant(call) => call(
                    store.raw.as_ptr(),
                    hostname.as_ptr(),
                    metadata.as_ptr(),
                    setup.global_segment_size,
                    setup.local_buffer_size,
                    protocol.as_ptr(),
                    device.as_ptr(),
                    master.as_ptr(),
                ),
            }
        };
        if status != 0 {
            // Dropping `store` destroys the half-set-up client.
            return Err(Error::Setup {
                master: setup.master_server_address.clone(),
                status,
            });
        }
        Ok(store)
    }

    /// Register `length` bytes at `addr` for zero-copy gets and puts.
    pub fn register_buffer(&self, addr: usize, length: usize) -> Result<(), Error> {
        // SAFETY: the client only records the range; the caller keeps it
        // mapped until it is deregistered.
        let status = unsafe {
            (self.library.register_buffer)(self.raw.as_ptr(), addr as *mut c_void, length)
        };
        if status != 0 {
            return Err(Error::Register {
                addr,
                length,
                status,
            });
        }
        Ok(())
    }

    /// Undo [`Store::register_buffer`].
    pub fn unregister_buffer(&self, addr: usize) -> Result<(), Error> {
        // SAFETY: as in `register_buffer`.
        let status =
            unsafe { (self.library.unregister_buffer)(self.raw.as_ptr(), addr as *mut c_void) };
        if status != 0 {
            return Err(Error::Unregister { addr, status });
        }
        Ok(())
    }

    /// Per key: 1 present, 0 absent, negative an error
    /// (`batch_is_exist`).
    pub fn batch_is_exist(&self, keys: &[String]) -> Result<Vec<i32>, Error> {
        let keys_c = CKeys::new(keys)?;
        let mut results = vec![-1 as c_int; keys.len()];
        // SAFETY: the key array and the result buffer hold `keys.len()`
        // entries and outlive the call.
        let status = unsafe {
            (self.library.batch_is_exist)(
                self.raw.as_ptr(),
                keys_c.pointers.as_ptr(),
                keys.len(),
                results.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(Error::Call {
                call: "batch_is_exist",
                status: i64::from(status),
            });
        }
        Ok(results)
    }

    /// Read object `keys[i]` into the registered `sizes[i]` bytes at
    /// `addrs[i]`; per key, the bytes read or a negative error
    /// (`batch_get_into`).
    pub fn batch_get_into(
        &self,
        keys: &[String],
        addrs: &[usize],
        sizes: &[usize],
    ) -> Result<Vec<i64>, Error> {
        if keys.len() != addrs.len() || keys.len() != sizes.len() {
            return Err(Error::Shape("keys, addresses and sizes"));
        }
        let keys_c = CKeys::new(keys)?;
        let buffers: Vec<*mut c_void> = addrs.iter().map(|&addr| addr as *mut c_void).collect();
        let mut results = vec![0_i64; keys.len()];
        // SAFETY: every array holds `keys.len()` entries and outlives the
        // call; the ranges lie in a registered buffer the caller keeps
        // mapped until the call returns.
        let status = unsafe {
            (self.library.batch_get_into)(
                self.raw.as_ptr(),
                keys_c.pointers.as_ptr(),
                buffers.as_ptr(),
                sizes.as_ptr(),
                keys.len(),
                results.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(Error::Call {
                call: "batch_get_into",
                status: i64::from(status),
            });
        }
        Ok(results)
    }

    /// Store the registered `sizes[i]` bytes at `addrs[i]` as object
    /// `keys[i]`, with the default replication; per key, 0 or a negative
    /// error (`batch_put_from`).
    pub fn batch_put_from(
        &self,
        keys: &[String],
        addrs: &[usize],
        sizes: &[usize],
    ) -> Result<Vec<i32>, Error> {
        if keys.len() != addrs.len() || keys.len() != sizes.len() {
            return Err(Error::Shape("keys, addresses and sizes"));
        }
        let keys_c = CKeys::new(keys)?;
        let buffers: Vec<*mut c_void> = addrs.iter().map(|&addr| addr as *mut c_void).collect();
        let mut results = vec![0 as c_int; keys.len()];
        // SAFETY: as in `batch_get_into`; a null config is the default
        // replication.
        let status = unsafe {
            (self.library.batch_put_from)(
                self.raw.as_ptr(),
                keys_c.pointers.as_ptr(),
                buffers.as_ptr(),
                sizes.as_ptr(),
                keys.len(),
                std::ptr::null(),
                results.as_mut_ptr(),
            )
        };
        if status != 0 {
            return Err(Error::Call {
                call: "batch_put_from",
                status: i64::from(status),
            });
        }
        Ok(results)
    }

    /// Store bytes, using an explicit lifetime result when the SDK supports it.
    /// A legacy SDK preserves its existing conservative return-code contract.
    pub fn batch_put_from_with_retirement(
        &self,
        keys: &[String],
        addrs: &[usize],
        sizes: &[usize],
    ) -> Result<PutOutcome, Error> {
        let Some(call) = self.library.batch_put_from_with_retirement else {
            return self
                .batch_put_from(keys, addrs, sizes)
                .map(|results| PutOutcome {
                    results,
                    source_retired: None,
                });
        };
        if keys.len() != addrs.len() || keys.len() != sizes.len() {
            return Err(Error::Shape("keys, addresses and sizes"));
        }
        let keys_c = CKeys::new(keys)?;
        let buffers: Vec<*mut c_void> = addrs.iter().map(|&addr| addr as *mut c_void).collect();
        let mut results = vec![-1 as c_int; keys.len()];
        let mut retired = 0_u8;
        // SAFETY: the v1 extension has the declared C signature; all arrays and
        // the lifetime output remain allocated for the duration of the call.
        let status = unsafe {
            call(
                self.raw.as_ptr(),
                keys_c.pointers.as_ptr(),
                buffers.as_ptr(),
                sizes.as_ptr(),
                keys.len(),
                std::ptr::null(),
                results.as_mut_ptr(),
                &mut retired,
            )
        };
        if status != 0 || retired > 1 {
            return Err(Error::Call {
                call: "batch_put_from_with_retirement_v1",
                status: if status != 0 { i64::from(status) } else { -1 },
            });
        }
        Ok(PutOutcome {
            results,
            source_retired: Some(retired == 1),
        })
    }

    /// Remove every object whose key matches `pattern`; the count removed,
    /// or a negative error (`remove_by_regex`).
    pub fn remove_by_regex(&self, pattern: &str, force: bool) -> Result<i64, Error> {
        let pattern_c = CString::new(pattern).map_err(|_| Error::Nul("pattern"))?;
        // SAFETY: the pattern outlives the call.
        Ok(unsafe {
            (self.library.remove_by_regex)(
                self.raw.as_ptr(),
                pattern_c.as_ptr(),
                c_int::from(force),
            )
        })
    }
}

impl Drop for Store {
    fn drop(&mut self) {
        // SAFETY: `raw` is this client's, destroyed once; destroying tears
        // the client down first.
        unsafe { (self.library.destroy)(self.raw.as_ptr()) };
    }
}

/// `pattern` as a regular expression matching itself, as Python's
/// `re.escape` writes it.
pub fn regex_escape(pattern: &str) -> String {
    let mut escaped = String::with_capacity(pattern.len());
    for character in pattern.chars() {
        if "()[]{}?*+-|^$\\.&~# \t\n\r\x0b\x0c".contains(character) {
            escaped.push('\\');
        }
        escaped.push(character);
    }
    escaped
}

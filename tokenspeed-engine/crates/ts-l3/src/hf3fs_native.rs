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

//! Optional 3FS C ABI. Only registered staging memory is exposed to native I/O.
#![allow(unsafe_code)]

use crate::{HostBuffer, Range, StoreError};
use libloading::Library;
use std::{
    ffi::{c_char, c_int, c_void, CStr, CString},
    fs::File,
    os::{fd::AsRawFd, unix::ffi::OsStrExt},
    path::Path,
    sync::Arc,
};

// C ABI: deepseek-ai/3FS src/lib/api/hf3fs_usrbio.h at
// 6c856acb4baad524e6def1d3381aeded4e491b5a.
const HF3FS_IOR_FORBID_READ_HOLES: u64 = 2;
const IO_DEPTH_ASAP: c_int = 0;
const RING_TIMEOUT_UNUSED: c_int = 0; // Only used with positive io_depth.
const NUMA_ANY: c_int = -1;

#[repr(C)]
struct Iov {
    base: *mut u8,
    handle: *mut c_void,
    id: [u8; 16],
    mount: [c_char; 256],
    size: usize,
    block_size: usize,
    numa: c_int,
}
#[repr(C)]
struct Ior {
    iov: Iov,
    handle: *mut c_void,
    mount: [c_char; 256],
    read: bool,
    depth: c_int,
    priority: c_int,
    timeout: c_int,
    flags: u64,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct Completion {
    index: i32,
    reserved: i32,
    result: i64,
    userdata: *const c_void,
}

type Extract = unsafe extern "C" fn(*mut c_char, c_int, *const c_char) -> c_int;
type IovCreate = unsafe extern "C" fn(*mut Iov, *const c_char, usize, usize, c_int) -> c_int;
type IovDestroy = unsafe extern "C" fn(*mut Iov);
type IorCreate =
    unsafe extern "C" fn(*mut Ior, *const c_char, c_int, bool, c_int, c_int, c_int, u64) -> c_int;
type IorDestroy = unsafe extern "C" fn(*mut Ior);
type Register = unsafe extern "C" fn(c_int, u64) -> c_int;
type Deregister = unsafe extern "C" fn(c_int);
type Prep = unsafe extern "C" fn(
    *const Ior,
    *const Iov,
    bool,
    *mut c_void,
    c_int,
    usize,
    u64,
    *const c_void,
) -> c_int;
type Submit = unsafe extern "C" fn(*const Ior) -> c_int;
type Wait =
    unsafe extern "C" fn(*const Ior, *mut Completion, c_int, c_int, *const libc::timespec) -> c_int;

pub(crate) struct Api {
    _library: Library,
    extract: Extract,
    iov_create: IovCreate,
    iov_unlink: IovDestroy,
    iov_destroy: IovDestroy,
    ior_create: IorCreate,
    ior_destroy: IorDestroy,
    register: Register,
    deregister: Deregister,
    prep: Prep,
    submit: Submit,
    wait: Wait,
}
fn error(operation: &str, code: i32) -> StoreError {
    StoreError(format!(
        "3FS USRBIO {operation}: {} ({code})",
        std::io::Error::from_raw_os_error(code.abs())
    ))
}
impl Api {
    pub(crate) fn load(path: &Path) -> Result<Arc<Self>, StoreError> {
        unsafe {
            let lib =
                Library::new(path).map_err(|e| StoreError(format!("3FS USRBIO library: {e}")))?;
            macro_rules! symbol {
                ($name:literal) => {
                    *lib.get(concat!($name, "\0").as_bytes())
                        .map_err(|e| StoreError(e.to_string()))?
                };
            }
            Ok(Arc::new(Self {
                extract: symbol!("hf3fs_extract_mount_point"),
                iov_create: symbol!("hf3fs_iovcreate"),
                iov_unlink: symbol!("hf3fs_iovunlink"),
                iov_destroy: symbol!("hf3fs_iovdestroy"),
                ior_create: symbol!("hf3fs_iorcreate4"),
                ior_destroy: symbol!("hf3fs_iordestroy"),
                register: symbol!("hf3fs_reg_fd"),
                deregister: symbol!("hf3fs_dereg_fd"),
                prep: symbol!("hf3fs_prep_io"),
                submit: symbol!("hf3fs_submit_ios"),
                wait: symbol!("hf3fs_wait_for_ios"),
                _library: lib,
            }))
        }
    }
    pub(crate) fn mount(&self, path: &Path) -> Result<CString, StoreError> {
        let path =
            CString::new(path.as_os_str().as_bytes()).map_err(|e| StoreError(e.to_string()))?;
        let mut mount = [0 as c_char; 256];
        let n = unsafe { (self.extract)(mount.as_mut_ptr(), 256, path.as_ptr()) };
        if n <= 0 || n > 256 {
            return Err(StoreError(
                "3FS USRBIO mount unavailable: expose the matching FUSE root/3fs-virt and /dev/shm"
                    .into(),
            ));
        }
        Ok(unsafe { CStr::from_ptr(mount.as_ptr()) }.to_owned())
    }
    pub(crate) fn file(self: &Arc<Self>, file: File) -> Result<Arc<RegisteredFile>, StoreError> {
        let size = file
            .metadata()
            .map_err(|e| StoreError(format!("3FS segment metadata: {e}")))?
            .len();
        let rc = unsafe { (self.register)(file.as_raw_fd(), 0) };
        // The 3FS API returns <=0 for a usable fd, positive errno on failure.
        if rc > 0 {
            return Err(error("register fd", rc));
        }
        Ok(Arc::new(RegisteredFile {
            file,
            size,
            api: self.clone(),
        }))
    }
}
pub(crate) struct RegisteredFile {
    pub(crate) file: File,
    // Segments are pre-sized and never resized after registration.
    pub(crate) size: u64,
    api: Arc<Api>,
}
impl Drop for RegisteredFile {
    fn drop(&mut self) {
        unsafe { (self.api.deregister)(self.file.as_raw_fd()) };
    }
}

pub(crate) struct Request {
    pub(crate) file: Arc<RegisteredFile>,
    pub(crate) offset: usize,
    pub(crate) range: Range,
}

fn batch_len(
    sizes: impl Iterator<Item = usize>,
    entries: usize,
    capacity: usize,
) -> Result<usize, StoreError> {
    let mut count = 0;
    let mut bytes = 0;
    for size in sizes.take(entries) {
        if size > capacity {
            return Err(StoreError(
                "3FS object exceeds configured USRBIO buffer_bytes".into(),
            ));
        }
        if size > capacity - bytes {
            break;
        }
        bytes += size;
        count += 1;
    }
    Ok(count)
}

fn completion_success(read: bool, result: i64, size: usize) -> Result<bool, StoreError> {
    if result < 0 && !read {
        return Err(error("completion", result as i32));
    }
    // All completions have retired before this is called. Failed reads carry
    // object evidence to the executor without discarding healthy batch peers.
    Ok(result >= 0 && result as u64 == size as u64)
}

/// One serialized native ring, owned by a Store worker lane.
pub(crate) struct Ring {
    api: Arc<Api>,
    iov: Box<Iov>,
    ior: Box<Ior>,
    iov_live: bool,
    ior_live: bool,
    read: bool,
    entries: usize,
    bytes: usize,
    timeout_ms: u64,
    poisoned: bool,
    // Hold every referenced fd until all native completions have arrived.
    pending: Vec<Arc<RegisteredFile>>,
}
// Native pointers are accessed only while the Store's lane mutex is held.
unsafe impl Send for Ring {}
impl Ring {
    pub(crate) fn new(
        api: Arc<Api>,
        mount: &CStr,
        read: bool,
        entries: usize,
        bytes: usize,
        timeout_ms: u64,
    ) -> Result<Self, StoreError> {
        let mut ring = Self {
            api,
            iov: Box::new(unsafe { std::mem::zeroed() }),
            ior: Box::new(unsafe { std::mem::zeroed() }),
            iov_live: false,
            ior_live: false,
            read,
            entries,
            bytes,
            timeout_ms,
            poisoned: false,
            pending: Vec::new(),
        };
        let rc =
            unsafe { (ring.api.iov_create)(&mut *ring.iov, mount.as_ptr(), bytes, 0, NUMA_ANY) };
        if rc < 0 {
            return Err(error("iovcreate", rc));
        }
        ring.iov_live = true;
        unsafe { (ring.api.iov_unlink)(&mut *ring.iov) };
        let rc = unsafe {
            (ring.api.ior_create)(
                &mut *ring.ior,
                mount.as_ptr(),
                entries as i32,
                read,
                IO_DEPTH_ASAP,
                RING_TIMEOUT_UNUSED,
                NUMA_ANY,
                // Request hole errors from 3FS; this is not a payload checksum.
                if read { HF3FS_IOR_FORBID_READ_HOLES } else { 0 },
            )
        };
        if rc < 0 {
            return Err(error("iorcreate", rc));
        }
        ring.ior_live = true;
        Ok(ring)
    }
    pub(crate) fn run(
        &mut self,
        host: HostBuffer,
        requests: &[Request],
    ) -> Result<Vec<bool>, StoreError> {
        if self.poisoned {
            return Err(StoreError(
                "3FS USRBIO ring quarantined after unresolved I/O; restart client".into(),
            ));
        }
        let mut output = Vec::with_capacity(requests.len());
        let mut start = 0;
        while start < requests.len() {
            let count = batch_len(
                requests[start..].iter().map(|r| r.range.size),
                self.entries,
                self.bytes,
            )?;
            let end = start + count;
            output.extend(self.batch(host, &requests[start..end])?);
            start = end;
        }
        Ok(output)
    }
    fn batch(&mut self, host: HostBuffer, requests: &[Request]) -> Result<Vec<bool>, StoreError> {
        let mut deadline: libc::timespec = unsafe { std::mem::zeroed() };
        if unsafe { libc::clock_gettime(libc::CLOCK_REALTIME, &mut deadline) } != 0 {
            return Err(StoreError("3FS USRBIO clock_gettime failed".into()));
        }
        deadline.tv_sec += (self.timeout_ms / 1000) as libc::time_t;
        deadline.tv_nsec += ((self.timeout_ms % 1000) * 1_000_000) as libc::c_long;
        if deadline.tv_nsec >= 1_000_000_000 {
            deadline.tv_nsec -= 1_000_000_000;
            deadline.tv_sec += 1;
        }
        let mut offsets = Vec::with_capacity(requests.len());
        let mut cursor = 0;
        for request in requests {
            let address = host.address(request.range);
            offsets.push(cursor);
            if !self.read {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        address as *const u8,
                        self.iov.base.add(cursor),
                        request.range.size,
                    )
                };
            }
            cursor += request.range.size;
        }
        self.pending = requests.iter().map(|r| r.file.clone()).collect();
        // prep_io may already expose work to FUSE before submit. Any early
        // return from this point retains the staging allocation and fds.
        self.poisoned = true;
        let mut indexes = Vec::with_capacity(requests.len());
        for (i, request) in requests.iter().enumerate() {
            let rc = unsafe {
                (self.api.prep)(
                    &*self.ior,
                    &*self.iov,
                    self.read,
                    self.iov.base.add(offsets[i]).cast(),
                    request.file.file.as_raw_fd(),
                    request.offset,
                    request.range.size as u64,
                    (i + 1) as *const c_void,
                )
            };
            if rc < 0 {
                // A rejected first prep has exposed no I/O. Once any prep
                // succeeds, FUSE may already be using the staging memory.
                if indexes.is_empty() {
                    self.poisoned = false;
                    self.pending.clear();
                }
                return Err(error("prep_io", rc));
            }
            indexes.push(rc);
        }
        let rc = unsafe { (self.api.submit)(&*self.ior) };
        if rc < 0 {
            return Err(error("submit_ios", rc));
        }
        let mut cqes = vec![
            Completion {
                index: 0,
                reserved: 0,
                result: 0,
                userdata: std::ptr::null()
            };
            requests.len()
        ];
        let mut completed = vec![None; requests.len()];
        let mut count = 0;
        while count < requests.len() {
            let n = unsafe {
                (self.api.wait)(
                    &*self.ior,
                    cqes.as_mut_ptr(),
                    (requests.len() - count) as i32,
                    1,
                    &deadline,
                )
            };
            if n == -libc::EINTR {
                continue;
            }
            if n < 0 {
                return Err(error("wait_for_ios", n));
            }
            if n == 0 {
                return Err(StoreError(
                    "3FS USRBIO timeout; staging quarantined until process exit".into(),
                ));
            }
            if n as usize > requests.len() - count {
                return Err(StoreError("3FS invalid completion count".into()));
            }
            for cqe in &cqes[..n as usize] {
                let index = (cqe.userdata as usize)
                    .checked_sub(1)
                    .filter(|&i| i < requests.len())
                    .ok_or_else(|| StoreError("3FS invalid completion token".into()))?;
                if completed[index].is_some() || cqe.index != indexes[index] {
                    return Err(StoreError("3FS invalid completion index".into()));
                }
                completed[index] = Some(cqe.result);
                count += 1;
            }
        }
        self.poisoned = false;
        self.pending.clear();
        let mut success = Vec::with_capacity(requests.len());
        for (i, result) in completed.into_iter().enumerate() {
            let result = result.expect("all native completions received");
            let ok = completion_success(self.read, result, requests[i].range.size)?;
            if !ok {
                tracing::warn!(
                    read = self.read,
                    result,
                    "3FS object I/O failed after retirement"
                );
            }
            if ok && self.read {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        self.iov.base.add(offsets[i]),
                        host.address(requests[i].range) as *mut u8,
                        requests[i].range.size,
                    )
                };
            }
            success.push(ok);
        }
        tracing::debug!(
            direction = if self.read { "read" } else { "write" },
            objects = requests.len(),
            bytes = cursor,
            "3FS USRBIO native batch completed"
        );
        Ok(success)
    }
}
impl Drop for Ring {
    fn drop(&mut self) {
        if self.poisoned {
            // A timeout does not cancel DMA. Keep native structures, library,
            // shared memory and file registrations alive until process exit.
            std::mem::forget(self.api.clone());
            std::mem::forget(std::mem::take(&mut self.pending));
            let iov = std::mem::replace(&mut self.iov, Box::new(unsafe { std::mem::zeroed() }));
            let ior = std::mem::replace(&mut self.ior, Box::new(unsafe { std::mem::zeroed() }));
            std::mem::forget(iov);
            std::mem::forget(ior);
            return;
        }
        unsafe {
            if self.ior_live {
                (self.api.ior_destroy)(&mut *self.ior);
            }
            if self.iov_live {
                (self.api.iov_destroy)(&mut *self.iov);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batches_respect_entry_and_staging_limits() {
        assert_eq!(batch_len([4, 4, 4].into_iter(), 2, 16).unwrap(), 2);
        assert_eq!(batch_len([4, 4, 4].into_iter(), 8, 8).unwrap(), 2);
        assert_eq!(batch_len([8].into_iter(), 1, 8).unwrap(), 1);
        assert_eq!(batch_len([].into_iter(), 1, 8).unwrap(), 0);
        assert!(batch_len([9].into_iter(), 1, 8).is_err());
    }

    #[test]
    fn retired_read_errors_do_not_fail_healthy_peers() {
        let success = [64, -(libc::EIO as i64), 32, 64]
            .into_iter()
            .map(|result| completion_success(true, result, 64))
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(success, [true, false, false, true]);
        assert!(completion_success(false, -(libc::EIO as i64), 64).is_err());
        assert!(!completion_success(false, 32, 64).unwrap());
    }
}

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

//! Starting the embedded interpreter.

// The C API's configuration calls have no safe wrapper.
#![allow(unsafe_code)]

use std::{
    ffi::{CStr, CString},
    mem::MaybeUninit,
    ptr::addr_of_mut,
};

use pyo3::ffi;

/// Start the interpreter as the `executable` interpreter would start, with
/// the scheduler's module built in, and release the GIL.
///
/// `executable` is the launcher's `sys.executable`: the interpreter derives
/// its prefix from it, so a virtual environment's site-packages and `.pth`
/// files are the launcher's. Python installs no signal handlers; the host
/// handles signals itself.
pub fn start(executable: &str) -> Result<(), String> {
    let executable =
        CString::new(executable).map_err(|_| "the Python executable path has a NUL".to_owned())?;
    ts_scheduler::register_builtin_module();
    let mut config = MaybeUninit::<ffi::PyConfig>::uninit();
    let config = config.as_mut_ptr();
    // SAFETY: `PyConfig_InitPythonConfig` initializes every field before the
    // others read them; the strings it copies outlive the calls; the
    // interpreter is not running yet, which `register_builtin_module`
    // asserted.
    unsafe {
        ffi::PyConfig_InitPythonConfig(config);
        (*config).install_signal_handlers = 0;
        let mut status = ffi::PyConfig_SetBytesString(
            config,
            addr_of_mut!((*config).program_name),
            executable.as_ptr(),
        );
        if ffi::PyStatus_Exception(status) == 0 {
            status = ffi::PyConfig_SetBytesString(
                config,
                addr_of_mut!((*config).executable),
                executable.as_ptr(),
            );
        }
        if ffi::PyStatus_Exception(status) == 0 {
            status = ffi::Py_InitializeFromConfig(config);
        }
        ffi::PyConfig_Clear(config);
        if ffi::PyStatus_Exception(status) != 0 {
            let message = if status.err_msg.is_null() {
                "unknown error".to_owned()
            } else {
                CStr::from_ptr(status.err_msg)
                    .to_string_lossy()
                    .into_owned()
            };
            return Err(format!("starting the interpreter: {message}"));
        }
        // The thread that starts the interpreter holds the GIL; every thread,
        // this one included, attaches when it needs it.
        ffi::PyEval_SaveThread();
    }
    Ok(())
}

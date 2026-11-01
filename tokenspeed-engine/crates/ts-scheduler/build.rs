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

//! Builds the scheduler core, its nanobind module and nanobind itself with
//! the scheduler's CMake project, then the cxx facade over them.
//!
//! The Python interpreter is the one PyO3 links (`PYO3_PYTHON`); it must have
//! `nanobind` and `tokenspeed-spdlog` installed, as for the scheduler wheel.

use std::{
    path::{Path, PathBuf},
    process::Command,
};

fn python_output(python: &str, code: &str) -> String {
    let output = Command::new(python)
        .args(["-c", code])
        .output()
        .unwrap_or_else(|e| panic!("running {python}: {e}"));
    assert!(
        output.status.success(),
        "{python} -c {code:?} failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).unwrap().trim().to_owned()
}

fn main() {
    cxx_build::CFG.include_prefix = "ts-scheduler";
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let scheduler_dir = manifest
        .join("scheduler")
        .canonicalize()
        .expect("bundled scheduler source");
    let python = pyo3_build_config::get()
        .executable()
        .expect("PyO3 found no Python interpreter; set PYO3_PYTHON")
        .to_owned();

    let built = cmake::Config::new(&manifest)
        .define("TOKENSPEED_SCHEDULER_DIR", &scheduler_dir)
        .define("Python_EXECUTABLE", &python)
        .build_target("ts_scheduler_embed")
        .build()
        .join("build");

    let spdlog_include = python_output(
        &python,
        "import pathlib, tokenspeed_spdlog; \
         print(pathlib.Path(tokenspeed_spdlog.__file__).parent / 'include')",
    );
    let csrc = scheduler_dir.join("csrc");
    cxx_build::bridge("src/lib.rs")
        .file("cxx/facade.cc")
        .include(&csrc)
        .include(&spdlog_include)
        .define("OPENSSL_SUPPRESS_DEPRECATED", None)
        .std("c++20")
        .compile("ts_scheduler_facade");

    // Dependents after their users: the facade and the module need the core;
    // the module needs nanobind.
    for dir in [built.clone(), built.join("scheduler")] {
        println!("cargo:rustc-link-search=native={}", dir.display());
    }
    for lib in [
        "ts_scheduler_embed",
        "tokenspeed_scheduler_core",
        "nanobind-static",
    ] {
        println!("cargo:rustc-link-lib=static={lib}");
    }
    println!("cargo:rustc-link-lib=dylib=crypto");
    println!("cargo:rustc-link-lib=dylib=stdc++");

    // This package's tests start the interpreter: find libpython at run time,
    // and the standard library under the interpreter's base prefix.
    if let Some(lib_dir) = pyo3_build_config::get().lib_dir() {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{lib_dir}");
    }
    let home = python_output(&python, "import sys; print(sys.base_prefix)");
    println!("cargo:rustc-env=TS_SCHEDULER_TEST_PYTHONHOME={home}");
    // For dependents' build scripts (`DEP_TS_SCHEDULER_*`): binaries that
    // embed the interpreter need the same two facts.
    println!("cargo:python_home={home}");
    if let Some(lib_dir) = pyo3_build_config::get().lib_dir() {
        println!("cargo:python_lib_dir={lib_dir}");
    }

    for path in [
        Path::new("CMakeLists.txt"),
        Path::new("cxx"),
        Path::new("src/lib.rs"),
    ] {
        println!("cargo:rerun-if-changed={}", path.display());
    }
    for path in ["csrc", "bindings", "CMakeLists.txt"] {
        println!(
            "cargo:rerun-if-changed={}",
            scheduler_dir.join(path).display()
        );
    }
}

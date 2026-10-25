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

//! Builds xgrammar's core with CMake (`CMakeLists.txt` fetches its source
//! release), then the cxx facade over it.

use std::path::PathBuf;

fn main() {
    cxx_build::CFG.include_prefix = "ts-grammar";
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let mut config = cmake::Config::new(&manifest);
    if let Ok(source) = std::env::var("FETCHCONTENT_SOURCE_DIR_XGRAMMAR") {
        config.define("FETCHCONTENT_SOURCE_DIR_XGRAMMAR", source);
    }
    let build = config
        .define("CMAKE_BUILD_TYPE", "Release")
        .build_target("xgrammar")
        .build()
        .join("build");
    let source = PathBuf::from(
        std::fs::read_to_string(build.join("xgrammar_source_dir.txt"))
            .expect("the CMake step names the xgrammar source"),
    );

    cxx_build::bridge("src/lib.rs")
        .file("cxx/facade.cc")
        .include(source.join("include"))
        .include(source.join("3rdparty/dlpack/include"))
        .include(source.join("3rdparty/picojson"))
        .std("c++17")
        .compile("ts_grammar_facade");

    // The library sits in xgrammar's own build directory.
    println!(
        "cargo:rustc-link-search=native={}",
        build.join("_deps/xgrammar-build").display()
    );
    println!("cargo:rustc-link-lib=static=xgrammar");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    for path in ["CMakeLists.txt", "cxx", "src/lib.rs"] {
        println!("cargo:rerun-if-changed={path}");
    }
    println!("cargo:rerun-if-env-changed=FETCHCONTENT_SOURCE_DIR_XGRAMMAR");
}

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

// Moving scheduler objects between the binary and the embedded interpreter.
// Both functions run with the GIL held and need the built-in
// `tokenspeed_scheduler_ext` module imported, which registers the types.

#include <nanobind/nanobind.h>

#include <memory>

#include "scheduler/execution_plan.h"
#include "scheduler/scheduler.h"

namespace nb = nanobind;

// The C++ scheduler behind a Python `tokenspeed_scheduler.Scheduler`, which
// keeps owning it. Null with a Python error set if `object` is something else.
extern "C" tokenspeed::Scheduler* ts_scheduler_from_python(PyObject* object) {
    nb::handle handle(object);
    if (!nb::isinstance<tokenspeed::Scheduler>(handle)) {
        PyErr_SetString(PyExc_TypeError, "expected a tokenspeed_scheduler.Scheduler");
        return nullptr;
    }
    return nb::inst_ptr<tokenspeed::Scheduler>(handle);
}

// A new reference to a Python `ExecutionPlan` that takes over `plan`, which
// is consumed either way. Null with a Python error set on failure.
extern "C" PyObject* ts_scheduler_plan_into_python(tokenspeed::ExecutionPlan* plan) {
    std::unique_ptr<tokenspeed::ExecutionPlan> owned(plan);
    try {
        return nb::cast(std::move(*owned)).release().ptr();
    } catch (const std::exception& error) {
        PyErr_SetString(PyExc_RuntimeError, error.what());
        return nullptr;
    }
}

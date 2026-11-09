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

#include <gtest/gtest.h>
#include <spdlog/sinks/stdout_sinks.h>

#include <stdexcept>
#include <string>

#include "utils.h"

namespace tokenspeed::test {
namespace {

// The checks fail out of line; a failure must still name the caller's
// location, not the helper's.

TEST(ChecksTest, AFailedAssertThrowsWithTheCallersLocation) {
    const auto line = __LINE__ + 2;
    try {
        _assert(false, "the table is too short");
        FAIL() << "a failed check throws";
    } catch (const std::runtime_error& error) {
        const std::string what = error.what();
        const std::string prefix =
            std::string("the table is too short at ") + __FILE__ + ":" + std::to_string(line) + " in ";
        EXPECT_EQ(what.rfind(prefix, 0), 0U) << what;
        EXPECT_NE(what.find("TestBody"), std::string::npos) << what;
    }
}

TEST(ChecksDeathTest, AFailedFatalCheckAbortsWithTheCallersLocation) {
    EXPECT_DEATH(
        {
            // The default logger writes to stdout; the death test reads stderr.
            spdlog::set_default_logger(spdlog::stderr_logger_mt("fatal-check-test"));
            FatalCheck(false, "the pool is corrupt");
        },
        "Fatal check failed: the pool is corrupt at .*test_utils\\.cpp:[0-9]+ in .*TestBody");
}

}  // namespace
}  // namespace tokenspeed::test

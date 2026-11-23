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

#include <cstdint>
#include <limits>
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

TEST(CeilDivTest, StaysExactUpToInt32Max) {
    constexpr std::int32_t kInt32Max = std::numeric_limits<std::int32_t>::max();
    EXPECT_EQ(CeilDiv(0, 64), 0);
    EXPECT_EQ(CeilDiv(1, 64), 1);
    EXPECT_EQ(CeilDiv(64, 64), 1);
    EXPECT_EQ(CeilDiv(65, 64), 2);
    EXPECT_EQ(CeilDiv(kInt32Max, 1024), 2097152);
    EXPECT_EQ(CeilDiv(kInt32Max - 1022, 1024), 2097152);
    EXPECT_EQ(CeilDiv(kInt32Max - 1023, 1024), 2097151);
    EXPECT_EQ(CeilDiv(kInt32Max, kInt32Max), 1);
    EXPECT_EQ(CeilDiv(kInt32Max, 1), kInt32Max);
}

TEST(CheckedInt32Test, NarrowsInRangeAndThrowsPastIt) {
    EXPECT_EQ(CheckedInt32(std::int64_t{std::numeric_limits<std::int32_t>::max()}),
              std::numeric_limits<std::int32_t>::max());
    EXPECT_EQ(CheckedInt32(std::int64_t{std::numeric_limits<std::int32_t>::min()}),
              std::numeric_limits<std::int32_t>::min());
    EXPECT_THROW((void)CheckedInt32(std::int64_t{1} << 31), std::runtime_error);
    EXPECT_THROW((void)CheckedInt32(-(std::int64_t{1} << 31) - 1), std::runtime_error);
}

}  // namespace
}  // namespace tokenspeed::test

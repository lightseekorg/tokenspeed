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

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

#include "core/profile_span.h"

namespace tokenspeed::test {
namespace {

struct Recorded {
    std::string name;
    std::int64_t begin_ns;
    std::int64_t end_ns;
};

std::vector<Recorded> recorded;

void Record(const char* name, std::int64_t begin_ns, std::int64_t end_ns) {
    recorded.push_back({name, begin_ns, end_ns});
}

std::int64_t UnixNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch())
        .count();
}

class ProfileSpanTest : public ::testing::Test {
protected:
    void SetUp() override { recorded.clear(); }
    void TearDown() override { SetProfileSpanSink(nullptr); }
};

TEST_F(ProfileSpanTest, ASpanReportsItsLifetimeInUnixTimeToTheSink) {
    SetProfileSpanSink(&Record);
    const std::int64_t before = UnixNs();
    {
        ProfileSpan outer{"outer"};
        { ProfileSpan inner{"inner"}; }
    }
    const std::int64_t after = UnixNs();
    ASSERT_EQ(recorded.size(), 2U);
    EXPECT_EQ(recorded[0].name, "inner");
    EXPECT_EQ(recorded[1].name, "outer");
    EXPECT_LE(before, recorded[1].begin_ns);
    EXPECT_LE(recorded[1].begin_ns, recorded[0].begin_ns);
    EXPECT_LE(recorded[0].end_ns, recorded[1].end_ns);
    EXPECT_LE(recorded[1].end_ns, after);
}

TEST_F(ProfileSpanTest, NothingIsReportedWithoutASink) {
    { ProfileSpan span{"unrecorded"}; }
    EXPECT_TRUE(recorded.empty());
}

TEST_F(ProfileSpanTest, ASpanOpenAcrossTheSinksRemovalReportsToItsOwn) {
    SetProfileSpanSink(&Record);
    {
        ProfileSpan open{"open"};
        SetProfileSpanSink(nullptr);
        { ProfileSpan after_removal{"after_removal"}; }
    }
    ASSERT_EQ(recorded.size(), 1U);
    EXPECT_EQ(recorded[0].name, "open");
}

}  // namespace
}  // namespace tokenspeed::test

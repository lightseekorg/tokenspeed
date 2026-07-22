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

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include <spdlog/spdlog.h>

#if defined(__has_include)
#if __has_include(<source_location>)
#include <source_location>
#define LLM_HAS_STD_SOURCE_LOCATION 1
#else
#define LLM_HAS_STD_SOURCE_LOCATION 0
#endif
#else
#define LLM_HAS_STD_SOURCE_LOCATION 0
#endif

namespace tokenspeed {

struct SourceLocation {
#if LLM_HAS_STD_SOURCE_LOCATION
    std::source_location loc;

    static constexpr SourceLocation current(std::source_location loc = std::source_location::current()) noexcept {
        return SourceLocation{loc};
    }

    constexpr const char* file_name() const noexcept { return loc.file_name(); }
    constexpr std::uint_least32_t line() const noexcept { return loc.line(); }
    constexpr const char* function_name() const noexcept { return loc.function_name(); }
#else
    const char* file_name_;
    std::uint_least32_t line_;
    const char* function_name_;

    static constexpr SourceLocation current(const char* file_name = __builtin_FILE(),
                                            std::uint_least32_t line = __builtin_LINE(),
                                            const char* function_name = __builtin_FUNCTION()) noexcept {
        return SourceLocation{file_name, line, function_name};
    }

    constexpr const char* file_name() const noexcept { return file_name_; }
    constexpr std::uint_least32_t line() const noexcept { return line_; }
    constexpr const char* function_name() const noexcept { return function_name_; }
#endif
};

namespace detail {

template <typename T>
constexpr std::string_view TypeName() {
#if defined(__clang__)
    std::string_view name = __PRETTY_FUNCTION__;
    std::string_view prefix = "std::string_view tokenspeed::detail::TypeName() [T = ";
    std::string_view suffix = "]";
#elif defined(__GNUC__)
    std::string_view name = __PRETTY_FUNCTION__;
    std::string_view prefix = "constexpr std::string_view tokenspeed::detail::TypeName() [with T = ";
    std::string_view suffix = "; std::string_view = std::basic_string_view<char>]";
#elif defined(_MSC_VER)
    std::string_view name = __FUNCSIG__;
    std::string_view prefix =
        "class std::basic_string_view<char,struct std::char_traits<char> > __cdecl tokenspeed::detail::TypeName<";
    std::string_view suffix = ">(void)";
#else
    return "unknown";
#endif

    name.remove_prefix(prefix.size());
    name.remove_suffix(suffix.size());
    return name;
}

}  // namespace detail

template <typename... Ts>
struct Overloaded : Ts... {
    using Ts::operator()...;
};

template <typename... Ts>
Overloaded(Ts...) -> Overloaded<Ts...>;

inline void _assert(bool condition, const char* message = "Assertion failed",
                    SourceLocation loc = SourceLocation::current()) {
    if (!condition) [[unlikely]] {
        std::string msg = message;
        msg = msg + " at " + loc.file_name() + ":" + std::to_string(loc.line()) + " in " + loc.function_name();
        throw std::runtime_error(msg);
    }
}

inline void FatalCheck(bool condition, const char* message = "Fatal check failed",
                       SourceLocation loc = SourceLocation::current()) noexcept {
    if (!condition) [[unlikely]] {
        bool logged = false;
        try {
            if (auto logger = spdlog::default_logger()) {
                logger->critical("Fatal check failed: {} at {}:{} in {}", message, loc.file_name(), loc.line(),
                                 loc.function_name());
                logger->flush();
                logged = true;
            }
        } catch (...) {
        }
        if (!logged) {
            std::fprintf(stderr, "Fatal check failed: %s at %s:%lu in %s\n", message, loc.file_name(),
                         static_cast<unsigned long>(loc.line()), loc.function_name());
            std::fflush(stderr);
        }
        std::abort();
    }
}

template <typename T, typename Variant>
concept IsAlternativeOf = requires(Variant v) { std::get<T>(v); };

template <typename T>
concept CanExtendTokenContainer =
    requires(T& state, std::vector<std::int32_t> tokens) { state.ExtendResultTokens(tokens); };

enum class Role {
    kP,
    kD,
    kFused,
};

}  // namespace tokenspeed

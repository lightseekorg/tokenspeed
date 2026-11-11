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

#include <array>
#include <compare>
#include <cstdint>
#include <cstring>
#include <functional>
#include <ostream>
#include <string>
#include <string_view>
#include <variant>

namespace tokenspeed {

// Complete internal identity. Canonical SHA-256 text has an allocation-free
// representation; opaque public keys retain their exact bytes, including
// uppercase hex, embedded nulls and the empty prior.
class ContentHash {
public:
    using Digest = std::array<std::uint8_t, 32>;

    ContentHash() : value_{std::string{}} {}
    ContentHash(const char* text) : ContentHash(std::string_view{text}) {}
    ContentHash(const std::string& text) : ContentHash(std::string_view{text}) {}
    ContentHash(std::string_view text) : value_{std::string{}} {
        Digest digest;
        if (decode(text, digest))
            value_ = digest;
        else
            value_ = std::string(text);
    }
    ContentHash(std::string&& text) : value_{std::string{}} {
        Digest digest;
        if (decode(text, digest))
            value_ = digest;
        else
            value_ = std::move(text);
    }
    explicit ContentHash(Digest digest) : value_{digest} {}

    bool empty() const noexcept {
        const auto* text = std::get_if<std::string>(&value_);
        return text && text->empty();
    }
    std::string ToString() const {
        if (const auto* text = std::get_if<std::string>(&value_)) return *text;
        static constexpr char hex[] = "0123456789abcdef";
        std::string text(64, '0');
        const auto& digest = std::get<Digest>(value_);
        for (std::size_t i = 0; i < digest.size(); ++i) {
            text[2 * i] = hex[digest[i] >> 4];
            text[2 * i + 1] = hex[digest[i] & 15];
        }
        return text;
    }
    operator std::string() const { return ToString(); }

    bool operator==(const ContentHash&) const noexcept = default;
    std::strong_ordering operator<=>(const ContentHash& other) const {
        if (const auto* digest = std::get_if<Digest>(&value_)) {
            if (const auto* rhs = std::get_if<Digest>(&other.value_)) {
                const int order = std::memcmp(digest->data(), rhs->data(), digest->size());
                return order < 0   ? std::strong_ordering::less
                       : order > 0 ? std::strong_ordering::greater
                                   : std::strong_ordering::equal;
            }
        }
        const int order = ToString().compare(other.ToString());
        return order < 0   ? std::strong_ordering::less
               : order > 0 ? std::strong_ordering::greater
                           : std::strong_ordering::equal;
    }
    std::size_t Hash() const {
        if (const auto* digest = std::get_if<Digest>(&value_)) {
            // SHA's bits are already distributed. This is a table hash only:
            // equality always compares all 32 bytes, including on collisions.
            std::uint64_t word;
            std::memcpy(&word, digest->data(), sizeof(word));
            return static_cast<std::size_t>(word);
        }
        return std::hash<std::string>{}(std::get<std::string>(value_));
    }
    friend std::ostream& operator<<(std::ostream& out, const ContentHash& hash) { return out << hash.ToString(); }

private:
    static bool decode(std::string_view text, Digest& digest) noexcept {
        if (text.size() != 64) return false;
        // Random hex digits make per-nibble branches unpredictable. Decode
        // through one table and validate once, retaining opaque bytes exactly.
        static constexpr auto nibbles = [] {
            std::array<std::uint8_t, 256> table{};
            table.fill(255);
            for (unsigned i = 0; i < 10; ++i) table['0' + i] = i;
            for (unsigned i = 0; i < 6; ++i) table['a' + i] = 10 + i;
            return table;
        }();
        unsigned invalid = 0;
        for (std::size_t i = 0; i < digest.size(); ++i) {
            const auto hi = nibbles[static_cast<unsigned char>(text[2 * i])];
            const auto lo = nibbles[static_cast<unsigned char>(text[2 * i + 1])];
            invalid |= hi | lo;
            digest[i] = static_cast<std::uint8_t>(hi * 16 + lo);
        }
        return invalid < 16;
    }

    std::variant<Digest, std::string> value_;
};

}  // namespace tokenspeed

template <>
struct std::hash<tokenspeed::ContentHash> {
    std::size_t operator()(const tokenspeed::ContentHash& value) const { return value.Hash(); }
};

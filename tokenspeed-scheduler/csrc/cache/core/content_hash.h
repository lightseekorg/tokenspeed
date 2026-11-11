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

#include "cache/core/hex_codec.h"

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
    ContentHash(std::string_view text) : value_{parse(text)} {}
    ContentHash(std::string&& text) : value_{parse(std::move(text))} {}
    explicit ContentHash(Digest digest) : value_{digest} {}

    bool empty() const noexcept {
        const auto* text = std::get_if<std::string>(&value_);
        return text && text->empty();
    }
    std::string ToString() const {
        if (const auto* text = std::get_if<std::string>(&value_)) return *text;
        std::string text(64, '0');
        WriteLowercaseHex(std::get<Digest>(value_), text.data());
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
    using Value = std::variant<Digest, std::string>;

    // Built in place: a key is constructed for every prompt page.
    static Value parse(std::string_view text) {
        Digest digest;
        if (ReadLowercaseHex(text, digest)) return digest;
        return std::string(text);
    }
    static Value parse(std::string&& text) {
        Digest digest;
        if (ReadLowercaseHex(text, digest)) return digest;
        return std::move(text);
    }

    Value value_;
};

}  // namespace tokenspeed

template <>
struct std::hash<tokenspeed::ContentHash> {
    std::size_t operator()(const tokenspeed::ContentHash& value) const { return value.Hash(); }
};

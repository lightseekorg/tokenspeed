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

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <string_view>
#include <vector>

namespace tokenspeed {

// Lowercase hexadecimal text of prefix-page digests. Every prompt page is
// encoded once and decoded into its cache key once, so both directions convert
// eight characters (four bytes) per 64-bit word instead of branching or
// looking up a table for each random digit.
namespace hex_codec_detail {

inline constexpr std::uint64_t kEachByte = 0x0101010101010101ULL;

// The lane arithmetic below takes a word's first byte as its lowest. Words
// are loaded and stored in host byte order, which is little-endian on every
// supported platform.
static_assert(std::endian::native == std::endian::little, "the hex codec assumes a little-endian host");

template <typename Word>
inline Word LoadWord(const void* source) noexcept {
    Word word;
    std::memcpy(&word, source, sizeof(word));
    return word;
}

template <typename Word>
inline void StoreWord(void* target, Word word) noexcept {
    std::memcpy(target, &word, sizeof(word));
}

// Four bytes to their eight characters.
inline void EncodeWord(const std::uint8_t* bytes, char* out) noexcept {
    // One byte per 16-bit lane, its high nibble moved to the lane's first
    // (lower) character.
    std::uint64_t lanes = LoadWord<std::uint32_t>(bytes);
    lanes = (lanes | lanes << 16) & 0x0000ffff0000ffffULL;
    lanes = (lanes | lanes << 8) & 0x00ff00ff00ff00ffULL;
    const std::uint64_t nibbles = (lanes >> 4 & 0x000f000f000f000fULL) | (lanes & 0x000f000f000f000fULL) << 8;
    // '0' + n, plus 'a' - '0' - 10 where n >= 10: adding 6 carries exactly
    // those nibbles into bit 4. No byte sum reaches the next byte.
    const std::uint64_t letters = (nibbles + 6 * kEachByte) >> 4 & kEachByte;
    StoreWord<std::uint64_t>(out, nibbles + '0' * kEachByte + letters * ('a' - '0' - 10));
}

// Eight characters to their four bytes. Returns zero exactly when every
// character is in [0-9a-f]; the bytes are meaningful only then.
inline std::uint64_t DecodeWord(const char* text, std::uint8_t* bytes) noexcept {
    const std::uint64_t chars = LoadWord<std::uint64_t>(text);
    // Adding 0x80 - k to a byte below 0x80 sets its bit 7 exactly when the
    // byte is at least k, and never carries into the next byte. A byte with
    // bit 7 set is no digit.
    const std::uint64_t low = chars & 0x7f * kEachByte;
    const auto at_least = [low](std::uint64_t k) { return low + (0x80 - k) * kEachByte; };
    const std::uint64_t digits = (at_least('0') & ~at_least('9' + 1)) | (at_least('a') & ~at_least('f' + 1));
    // '0'-'9' keep their low nibble; 'a'-'f' (0x61-0x66, bit 6 set) add 9.
    const std::uint64_t nibbles = (chars & 0x0f * kEachByte) + (chars >> 6 & kEachByte) * 9;
    // Join each 16-bit lane's (high, low) pair into its low byte, then pack
    // the four lane bytes.
    std::uint64_t packed = (nibbles << 4 | nibbles >> 8) & 0x00ff00ff00ff00ffULL;
    packed = (packed | packed >> 8) & 0x0000ffff0000ffffULL;
    StoreWord<std::uint32_t>(bytes, static_cast<std::uint32_t>(packed | packed >> 16));
    return (chars | ~digits) & 0x80 * kEachByte;
}

}  // namespace hex_codec_detail

// Writes the 2 * bytes.size() lowercase hex characters of bytes to out.
inline void WriteLowercaseHex(std::span<const std::uint8_t> bytes, char* out) noexcept {
    std::size_t i = 0;
    for (; i + 4 <= bytes.size(); i += 4) {
        hex_codec_detail::EncodeWord(bytes.data() + i, out + 2 * i);
    }
    if (i < bytes.size()) {
        // A short last word goes through zero-padded scratch.
        std::array<std::uint8_t, 4> word{};
        std::array<char, 8> text;
        std::copy(bytes.begin() + static_cast<std::ptrdiff_t>(i), bytes.end(), word.begin());
        hex_codec_detail::EncodeWord(word.data(), text.data());
        std::copy_n(text.begin(), 2 * (bytes.size() - i), out + 2 * i);
    }
}

// Decodes text, exactly 2 * out.size() characters of [0-9a-f], into out.
// Returns false, leaving out unspecified, for any other length or character
// (uppercase digits included).
inline bool ReadLowercaseHex(std::string_view text, std::span<std::uint8_t> out) noexcept {
    if (text.size() != 2 * out.size()) {
        return false;
    }
    // Digests are random and valid: check every word once at the end rather
    // than branching per word.
    std::uint64_t invalid = 0;
    std::size_t i = 0;
    for (; i + 4 <= out.size(); i += 4) {
        invalid |= hex_codec_detail::DecodeWord(text.data() + 2 * i, out.data() + i);
    }
    if (i < out.size()) {
        // A short last word goes through scratch padded with valid digits.
        std::array<char, 8> word;
        word.fill('0');
        std::array<std::uint8_t, 4> bytes;
        std::copy(text.begin() + static_cast<std::ptrdiff_t>(2 * i), text.end(), word.begin());
        invalid |= hex_codec_detail::DecodeWord(word.data(), bytes.data());
        std::copy_n(bytes.begin(), out.size() - i, out.begin() + static_cast<std::ptrdiff_t>(i));
    }
    return invalid == 0;
}

// Decodes any text two characters per byte, the way a prefix-hash prior has
// always been read: canonical lowercase text through ReadLowercaseHex, other
// text (an opaque public key used as a prior) character by character with
// uppercase digits accepted and other characters taken by their offset from
// '0'. An odd last character is ignored. Chained hashes depend on these exact
// bytes.
inline std::vector<std::uint8_t> HexToBytes(std::string_view hex) {
    std::vector<std::uint8_t> bytes(hex.size() / 2);
    if (ReadLowercaseHex(hex.substr(0, 2 * bytes.size()), bytes)) {
        return bytes;
    }
    const auto nibble = [](char c) {
        return (c >= 'a')   ? static_cast<std::uint8_t>(c - 'a' + 10)
               : (c >= 'A') ? static_cast<std::uint8_t>(c - 'A' + 10)
                            : static_cast<std::uint8_t>(c - '0');
    };
    for (std::size_t i = 0; i < bytes.size(); ++i) {
        bytes[i] = static_cast<std::uint8_t>((nibble(hex[2 * i]) << 4) | nibble(hex[2 * i + 1]));
    }
    return bytes;
}

}  // namespace tokenspeed

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

#include <cstdint>
#include <string>
#include <variant>
#include <vector>

namespace tokenspeed {
namespace pd {
// The transfer peer lets the request be admitted. On the D role the prefill
// peer answered. On the P role under early decode allocation every Decode
// rank allocated the destination; under independent decode allocation the
// destination has its own event, DestinationReadyEvent.
struct BootstrappedEvent {
    std::string request_id;
    explicit BootstrappedEvent(std::string id) : request_id(std::move(id)) {}
};

// P role, independent decode allocation: every Decode rank allocated the
// request's destination, so its KV may be sent once the prompt is computed.
struct DestinationReadyEvent {
    std::string request_id;
    explicit DestinationReadyEvent(std::string id) : request_id(std::move(id)) {}
};

// The transfer failed. It may still be stopping: the pages return with
// TransferStoppedEvent.
struct FailedEvent {
    std::string request_id;
    explicit FailedEvent(std::string id) : request_id(std::move(id)) {}
};

// A transfer that failed or was cancelled stopped: no reader or writer on
// any rank touches the request's pages any more.
struct TransferStoppedEvent {
    std::string request_id;
    explicit TransferStoppedEvent(std::string id) : request_id(std::move(id)) {}
};

struct SucceededEvent {
    std::string request_id;
    explicit SucceededEvent(std::string id) : request_id(std::move(id)) {}
};

struct RemotePrefillDoneEvent {
    std::string request_id;
    std::int32_t bootstrap_token{-1};
    explicit RemotePrefillDoneEvent(std::string id, std::int32_t token)
        : request_id(std::move(id)), bootstrap_token(token) {}
};

}  // namespace pd

using PDEvent = std::variant<pd::BootstrappedEvent, pd::DestinationReadyEvent, pd::FailedEvent,
                             pd::TransferStoppedEvent, pd::SucceededEvent, pd::RemotePrefillDoneEvent>;

}  // namespace tokenspeed

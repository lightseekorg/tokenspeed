#pragma once
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#include <span>
#include <functional>
#include <chrono>
namespace mooncake {
using BatchID = uint64_t;
using SegmentHandle = uint64_t;
inline constexpr BatchID INVALID_BATCH_ID = UINT64_MAX;
enum class TransferStatusEnum { WAITING, COMPLETED, FAILED, TIMEOUT };
struct TransferStatus {
    TransferStatusEnum s;
};
struct TransferRequest {
    enum OpCode { READ, WRITE } opcode;
    void* source;
    uint64_t target_offset, length;
    SegmentHandle target_id;
};
struct Status {
    bool success = true;
    bool ok() const { return success; }
    bool IsBatchCleanupDeferred() const { return !success; }
    std::string ToString() const { return "fixture status"; }
};
inline int submitted = 0, released = 0, opened = 0, registered = 0;
inline int fail_mode = 0;
inline int blocking_releases = 0;
inline std::string last_location;
inline std::vector<TransferRequest> last_requests;
inline std::vector<size_t> batch_sizes;
class TransferEngine {
   public:
    struct ScatterTransferRange {
        TransferRequest::OpCode opcode;
        std::string remote_segment;
        uint64_t remote_base_offset;
        size_t remote_size;
        void* local_buffer;
        size_t local_capacity;
        std::span<const size_t> local_offsets, remote_offsets, lengths;
        std::function<void(size_t, const Status&)> on_fragment_complete;
    };
    struct ScatterTransferOperation {
        bool pending = true;
        ~ScatterTransferOperation() {
            // A wait timeout/failure leaves work outstanding until this owner
            // retires. Returning from the adapter must run this destructor.
            if (pending) ++blocking_releases;
        }
        Status waitFor(std::chrono::nanoseconds) { return {fail_mode == 0}; }
    };
    ScatterTransferOperation submitScatter(const std::vector<ScatterTransferRange>& ranges) {
        last_requests.clear();
        for (const auto& range : ranges)
            for (size_t i = 0; i < range.lengths.size(); ++i)
                last_requests.push_back({range.opcode,
                    static_cast<char*>(range.local_buffer) + range.local_offsets[i],
                    range.remote_base_offset + range.remote_offsets[i], range.lengths[i], 1});
        return {};
    }
    explicit TransferEngine(bool) {}
    void setLocalProtocolHint(const char*) {}
    int init(const char*, const char*, const char*, int, const char*) {
        return 0;
    }
    int getRpcPort() { return 1234; }
    int registerLocalMemory(void*, size_t, const std::string& location) {
        last_location = location;
        ++registered;
        return 0;
    }
    int unregisterLocalMemory(void*) { return 0; }
    SegmentHandle openSegment(const char*) {
        ++opened;
        return 1;
    }
    BatchID allocateBatchID(size_t size) {
        batch_sizes.push_back(size);
        return 1;
    }
    Status freeBatchID(BatchID) {
        ++released;
        return {fail_mode == 0};
    }
    Status freeBatchID(BatchID, bool blocking) {
        blocking_releases += blocking;
        ++released;
        return {};
    }
    Status submitTransfer(BatchID, const std::vector<TransferRequest>& spans) {
        last_requests = spans;
        ++submitted;
        if (fail_mode == 2)
            throw std::runtime_error("submit fixture exception");
        return {fail_mode != 1};
    }
    Status getBatchTransferStatus(BatchID, TransferStatus& status) {
        status.s = fail_mode == 3   ? TransferStatusEnum::FAILED
                   : fail_mode == 4 ? TransferStatusEnum::TIMEOUT
                                    : TransferStatusEnum::COMPLETED;
        return {fail_mode != 5};
    }
};
}  // namespace mooncake

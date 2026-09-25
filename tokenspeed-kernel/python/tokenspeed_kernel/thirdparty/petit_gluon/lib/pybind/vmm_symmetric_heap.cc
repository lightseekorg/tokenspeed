#include "pybind.h"

#include <c10/core/DeviceGuard.h>
#include <hip/hip_runtime_api.h>

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <cstddef>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace causalflow::petit::pybind {
namespace {

constexpr std::uint32_t kPageBytes = 4096;

std::uint32_t AlignUp(std::uint32_t value, std::uint32_t alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
}

void HipCheck(hipError_t status, const char *what) {
    TORCH_CHECK(status == hipSuccess, what, ": ", hipGetErrorString(status));
}

void SysCheck(bool ok, const char *what) {
    TORCH_CHECK(ok, what, ": ", std::strerror(errno));
}

void Close(int &fd) {
    if (fd >= 0) {
        (void)::close(fd);
        fd = -1;
    }
}

socklen_t SocketAddress(const std::string &name, sockaddr_un *address) {
    TORCH_CHECK(!name.empty() && name.size() + 1 < sizeof(address->sun_path),
                "invalid VMM symmetric heap socket name");
    std::memset(address, 0, sizeof(*address));
    address->sun_family = AF_UNIX;
    address->sun_path[0] = '\0';
    std::memcpy(address->sun_path + 1, name.data(), name.size());
    return static_cast<socklen_t>(offsetof(sockaddr_un, sun_path) + 1 +
                                  name.size());
}

void SendFd(int socket, int rank, int fd) {
    std::array<char, CMSG_SPACE(sizeof(int))> control{};
    iovec iov{&rank, sizeof(rank)};
    msghdr message{};
    message.msg_iov = &iov;
    message.msg_iovlen = 1;
    message.msg_control = control.data();
    message.msg_controllen = control.size();
    auto *cmsg = CMSG_FIRSTHDR(&message);
    cmsg->cmsg_level = SOL_SOCKET;
    cmsg->cmsg_type = SCM_RIGHTS;
    cmsg->cmsg_len = CMSG_LEN(sizeof(int));
    std::memcpy(CMSG_DATA(cmsg), &fd, sizeof(fd));
    SysCheck(sendmsg(socket, &message, 0) == sizeof(rank),
             "sendmsg(VMM symmetric heap)");
}

std::pair<int, int> ReceiveFd(int socket) {
    std::array<char, CMSG_SPACE(sizeof(int))> control{};
    int rank = -1;
    iovec iov{&rank, sizeof(rank)};
    msghdr message{};
    message.msg_iov = &iov;
    message.msg_iovlen = 1;
    message.msg_control = control.data();
    message.msg_controllen = control.size();
    SysCheck(recvmsg(socket, &message, 0) == sizeof(rank),
             "recvmsg(VMM symmetric heap)");
    for (cmsghdr *cmsg = CMSG_FIRSTHDR(&message); cmsg != nullptr;
         cmsg = CMSG_NXTHDR(&message, cmsg)) {
        if (cmsg->cmsg_level == SOL_SOCKET && cmsg->cmsg_type == SCM_RIGHTS) {
            int fd = -1;
            std::memcpy(&fd, CMSG_DATA(cmsg), sizeof(fd));
            TORCH_CHECK(fd >= 0, "received invalid VMM symmetric heap fd");
            return {rank, fd};
        }
    }
    TORCH_CHECK(false, "missing VMM symmetric heap fd");
}

} // namespace

struct VmmSymmetricHeap::Impl {
    struct Allocation {
        std::uint32_t bytes = 0;
        hipMemGenericAllocationHandle_t local = nullptr;
        int export_fd = -1;
        std::vector<hipMemGenericAllocationHandle_t> peers;
    };

    int world_size = 0;
    int rank = 0;
    int device = 0;
    Layout layout{};
    void *base = nullptr;
    std::uint32_t reservation_bytes = 0;
    Allocation barriers;
    Allocation slots;
    Allocation padding;
    Allocation local;
    bool mapped = false;

    std::vector<int> Exchange(int export_fd, unsigned region) const {
        if (world_size == 1)
            return std::vector<int>(1, -1);
        namespace py = pybind11;
        py::gil_scoped_acquire gil;
        auto dist = py::module_::import("torch.distributed");
        const std::string name = "petit-vmm-symmetric-heap-" +
                                 std::to_string(::getpid()) + "-" +
                                 std::to_string(region) + "-" +
                                 std::to_string(rank);
        int listener = -1;
        std::vector<int> fds(static_cast<size_t>(world_size), -1);
        try {
            listener = socket(AF_UNIX, SOCK_STREAM, 0);
            SysCheck(listener >= 0, "socket(VMM symmetric heap)");
            sockaddr_un address{};
            const auto address_len = SocketAddress(name, &address);
            SysCheck(bind(listener, reinterpret_cast<sockaddr *>(&address),
                          address_len) == 0,
                     "bind(VMM symmetric heap)");
            SysCheck(listen(listener, world_size) == 0,
                     "listen(VMM symmetric heap)");
            py::dict local_info;
            local_info["rank"] = rank;
            local_info["socket"] = name;
            py::list infos;
            for (int i = 0; i < world_size; ++i) infos.append(py::none());
            dist.attr("all_gather_object")(infos, local_info);
            std::vector<std::string> names(static_cast<size_t>(world_size));
            for (py::handle item : infos) {
                const auto info = py::reinterpret_borrow<py::dict>(item);
                const int peer = info["rank"].cast<int>();
                TORCH_CHECK(peer >= 0 && peer < world_size,
                            "invalid VMM symmetric heap peer rank");
                names[peer] = info["socket"].cast<std::string>();
            }
            for (int accepted = 0; accepted < rank; ++accepted) {
                const int connection = accept(listener, nullptr, nullptr);
                SysCheck(connection >= 0, "accept(VMM symmetric heap)");
                const auto [received_rank, fd] = ReceiveFd(connection);
                SendFd(connection, rank, export_fd);
                (void)::close(connection);
                TORCH_CHECK(received_rank >= 0 && received_rank < rank &&
                                fds[received_rank] < 0,
                            "invalid VMM symmetric heap peer fd");
                fds[received_rank] = fd;
            }
            for (int peer = rank + 1; peer < world_size; ++peer) {
                const int connection = socket(AF_UNIX, SOCK_STREAM, 0);
                SysCheck(connection >= 0, "socket(VMM symmetric heap peer)");
                sockaddr_un address{};
                const auto address_len = SocketAddress(names[peer], &address);
                SysCheck(connect(connection, reinterpret_cast<sockaddr *>(&address),
                                 address_len) == 0,
                         "connect(VMM symmetric heap peer)");
                SendFd(connection, rank, export_fd);
                const auto [received_rank, fd] = ReceiveFd(connection);
                (void)::close(connection);
                TORCH_CHECK(received_rank == peer,
                            "VMM symmetric heap peer rank mismatch");
                fds[peer] = fd;
            }
            dist.attr("barrier")();
            Close(listener);
            return fds;
        } catch (...) {
            for (int &fd : fds) Close(fd);
            Close(listener);
            throw;
        }
    }

    void MapSymmetric(Allocation &allocation, std::uint32_t start,
                      unsigned region) {
        hipMemAllocationProp prop{};
        prop.type = hipMemAllocationTypePinned;
        prop.requestedHandleTypes = hipMemHandleTypePosixFileDescriptor;
        prop.location.type = hipMemLocationTypeDevice;
        prop.location.id = device;
        HipCheck(hipMemCreate(&allocation.local, allocation.bytes, &prop, 0),
                 "hipMemCreate(VMM symmetric heap)");
        HipCheck(hipMemExportToShareableHandle(&allocation.export_fd,
                                               allocation.local,
                                               hipMemHandleTypePosixFileDescriptor,
                                               0),
                 "hipMemExportToShareableHandle(VMM symmetric heap)");
        auto fds = Exchange(allocation.export_fd, region);
        allocation.peers.assign(static_cast<size_t>(world_size), nullptr);
        for (int peer = 0; peer < world_size; ++peer) {
            auto handle = allocation.local;
            if (peer != rank) {
                HipCheck(hipMemImportFromShareableHandle(
                             &handle,
                             reinterpret_cast<void *>(static_cast<intptr_t>(fds[peer])),
                             hipMemHandleTypePosixFileDescriptor),
                         "hipMemImportFromShareableHandle(VMM symmetric heap)");
                Close(fds[peer]);
                allocation.peers[peer] = handle;
            }
            HipCheck(hipMemMap(static_cast<char *>(base) + start +
                                   static_cast<size_t>(peer) * allocation.bytes,
                               allocation.bytes, 0, handle, 0),
                     "hipMemMap(VMM symmetric heap)");
        }
    }

    void MapPrivate(Allocation &allocation, std::uint32_t start) {
        hipMemAllocationProp prop{};
        prop.type = hipMemAllocationTypePinned;
        prop.location.type = hipMemLocationTypeDevice;
        prop.location.id = device;
        HipCheck(hipMemCreate(&allocation.local, allocation.bytes, &prop, 0),
                 "hipMemCreate(VMM symmetric heap private)");
        HipCheck(hipMemMap(static_cast<char *>(base) + start, allocation.bytes,
                           0, allocation.local, 0),
                 "hipMemMap(VMM symmetric heap private)");
    }

    void Release(Allocation &allocation, std::uint32_t start, bool symmetric) {
        if (allocation.bytes == 0) return;
        if (base) {
            const int count = symmetric ? world_size : 1;
            for (int rank_idx = 0; rank_idx < count; ++rank_idx)
                (void)hipMemUnmap(static_cast<char *>(base) + start +
                                      static_cast<size_t>(rank_idx) * allocation.bytes,
                                  allocation.bytes);
        }
        for (auto peer : allocation.peers)
            if (peer) (void)hipMemRelease(peer);
        if (allocation.local) (void)hipMemRelease(allocation.local);
        Close(allocation.export_fd);
        allocation = {};
    }
};

VmmSymmetricHeap::VmmSymmetricHeap(int world_size)
    : impl_(std::make_unique<Impl>()) {
    TORCH_CHECK(world_size > 0 && world_size <= 8,
                "world_size must be in [1, 8]");
    namespace py = pybind11;
    py::gil_scoped_acquire gil;
    auto dist = py::module_::import("torch.distributed");
    TORCH_CHECK(dist.attr("is_initialized")().cast<bool>(),
                "torch.distributed must be initialized");
    TORCH_CHECK(dist.attr("get_world_size")().cast<int>() == world_size,
                "world_size must match torch.distributed");
    impl_->world_size = world_size;
    impl_->rank = dist.attr("get_rank")().cast<int>();
    HipCheck(hipGetDevice(&impl_->device), "hipGetDevice");
}

VmmSymmetricHeap::~VmmSymmetricHeap() { Cleanup(); }

bool VmmSymmetricHeap::Allocate(const Layout &layout) {
    auto &heap = *impl_;
    const auto barrier_bytes = AlignUp(layout.barrier_record_bytes, kPageBytes);
    const auto slot_bytes = AlignUp(layout.rank_slot_bytes, kPageBytes);
    const auto local_bytes = AlignUp(layout.local_bytes, kPageBytes);
    TORCH_CHECK(layout.rank_sym_buffer_base ==
                    static_cast<std::uint32_t>(heap.world_size) * barrier_bytes &&
                    layout.local_offset >= layout.rank_sym_buffer_base +
                        static_cast<std::uint32_t>(heap.world_size) * slot_bytes &&
                    layout.local_offset % kPageBytes == 0,
                "invalid VMM symmetric heap layout");
    const std::uint64_t reservation =
        static_cast<std::uint64_t>(layout.local_offset) + local_bytes;
    TORCH_CHECK(reservation < (1ull << 32),
                "VMM symmetric heap exceeds 32-bit offsets");
    if (heap.mapped) {
        TORCH_CHECK(std::memcmp(&heap.layout, &layout, sizeof(Layout)) == 0,
                    "VMM symmetric heap was allocated for a different layout");
        return false;
    }
    heap.layout = layout;
    heap.reservation_bytes = static_cast<std::uint32_t>(reservation);
    heap.barriers.bytes = barrier_bytes;
    heap.slots.bytes = slot_bytes;
    heap.local.bytes = local_bytes;
    c10::DeviceGuard guard(c10::Device(
        c10::DeviceType::CUDA,
        static_cast<c10::DeviceIndex>(heap.device)));
    HipCheck(hipMemAddressReserve(&heap.base, heap.reservation_bytes, kPageBytes,
                                  nullptr, 0),
             "hipMemAddressReserve(VMM symmetric heap)");
    heap.MapSymmetric(heap.barriers, 0, 0);
    heap.MapSymmetric(heap.slots, layout.rank_sym_buffer_base, 1);
    const auto shared_end = layout.rank_sym_buffer_base +
                            static_cast<std::uint32_t>(heap.world_size) * slot_bytes;
    if (layout.local_offset > shared_end) {
        heap.padding.bytes = layout.local_offset - shared_end;
        heap.MapPrivate(heap.padding, shared_end);
    }
    heap.MapPrivate(heap.local, layout.local_offset);
    hipMemAccessDesc access{};
    access.location.type = hipMemLocationTypeDevice;
    access.location.id = heap.device;
    access.flags = hipMemAccessFlagsProtReadWrite;
    HipCheck(hipMemSetAccess(heap.base, heap.reservation_bytes, &access, 1),
             "hipMemSetAccess(VMM symmetric heap)");

    // Each rank initializes only the physical allocations it owns.  Once all
    // ranks reach the barrier, every symmetric counter and private workspace
    // is ready for its first kernel launch.
    HipCheck(hipMemset(static_cast<char *>(heap.base) +
                           static_cast<std::size_t>(heap.rank) *
                               heap.barriers.bytes,
                       0, heap.barriers.bytes),
             "hipMemset(VMM symmetric heap barrier)");
    HipCheck(hipMemset(static_cast<char *>(heap.base) +
                           layout.rank_sym_buffer_base +
                           static_cast<std::size_t>(heap.rank) *
                               heap.slots.bytes,
                       0, heap.slots.bytes),
             "hipMemset(VMM symmetric heap rank slot)");
    if (heap.padding.bytes != 0) {
        HipCheck(hipMemset(static_cast<char *>(heap.base) + shared_end, 0,
                           heap.padding.bytes),
                 "hipMemset(VMM symmetric heap padding)");
    }
    HipCheck(hipMemset(static_cast<char *>(heap.base) + layout.local_offset, 0,
                       heap.local.bytes),
             "hipMemset(VMM symmetric heap local workspace)");
    HipCheck(hipDeviceSynchronize(),
             "hipDeviceSynchronize(VMM symmetric heap initialization)");
    {
        namespace py = pybind11;
        py::gil_scoped_acquire gil;
        py::module_::import("torch.distributed").attr("barrier")();
    }
    heap.mapped = true;
    return true;
}

torch::Tensor VmmSymmetricHeap::LocalTensor() const {
    const auto &heap = *impl_;
    TORCH_CHECK(heap.mapped, "VMM symmetric heap must be mapped before use");
    return torch::from_blob(
        heap.base,
        {static_cast<int64_t>(heap.reservation_bytes / sizeof(std::int32_t))},
        torch::TensorOptions().device(torch::Device(torch::kCUDA, heap.device))
            .dtype(torch::kInt32));
}

int VmmSymmetricHeap::world_size() const { return impl_->world_size; }
int VmmSymmetricHeap::rank() const { return impl_->rank; }
int VmmSymmetricHeap::device_index() const { return impl_->device; }

void VmmSymmetricHeap::Cleanup() {
    if (!impl_) return;
    auto &heap = *impl_;
    c10::DeviceGuard guard(c10::Device(
        c10::DeviceType::CUDA,
        static_cast<c10::DeviceIndex>(heap.device)));
    const auto slot_bytes = heap.slots.bytes;
    const auto shared_end = heap.layout.rank_sym_buffer_base +
                            static_cast<std::uint32_t>(heap.world_size) * slot_bytes;
    heap.Release(heap.barriers, 0, true);
    heap.Release(heap.slots, heap.layout.rank_sym_buffer_base, true);
    heap.Release(heap.padding, shared_end, false);
    heap.Release(heap.local, heap.layout.local_offset, false);
    if (heap.base) {
        (void)hipMemAddressFree(heap.base, heap.reservation_bytes);
        heap.base = nullptr;
    }
    heap.mapped = false;
}

} // namespace causalflow::petit::pybind

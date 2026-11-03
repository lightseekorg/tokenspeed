# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""The per-forward grammar work, in C++.

It reaches the matchers the engine host owns through their C entries
(``HostGrammarEntries``), so it never takes the GIL; taking it in a CUDA
host node, even for a callback that does nothing, measured ~3% of a decode
step. A ``Matcher`` holds its handle (the ``retain`` entry) until the last
batch that refers to it is done, so a request's release does not pull the
matcher from under a queued forward.

A ``Filler`` writes a forward's masks into a host bitmask. A ``Host`` runs
it as a CUDA host node (``cudaLaunchHostFunc``) on the grammar side stream:
each forward adds one ``Batch`` and each execution of the node pops the
next, advances the previous one by its sampled tokens, then fills the new
one. A ``Batch`` is also the forward's completion: ``wait`` blocks until its
matchers have advanced, and ``advance_on_host`` advances them from the
forward's results when no next forward comes. Exactly one of the two
advances. Without a host node the forward fills its masks before it runs
(``Filler.fill``), and the commit advances the matchers.

The C++ stub is JIT-compiled on first use via
``torch.utils.cpp_extension.load_inline``.
"""

from __future__ import annotations

import threading
from typing import Any

_ext = None
_ext_lock = threading.Lock()


_CPP_SRC = r"""
#ifdef TOKENSPEED_GRAMMAR_HOST_NODE
#include <cuda_runtime.h>
#endif
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

// Logs from the host node, which cannot raise and holds no GIL; allocates nothing.
void LogError(const char* what, const char* message) noexcept {
    std::fprintf(stderr, "[grammar host] %s failed: %s\n", what, message);
}

// The engine host's matcher entries (ts_grammar::table::entries), by
// handle; a negative status is a failure.
struct Entries {
    // 1 terminated, 0 not.
    using IsTerminatedFn = int32_t (*)(uint64_t handle);
    // 1 accepted, 0 rejected (the matcher unchanged).
    using AcceptTokenFn = int32_t (*)(uint64_t handle, int32_t token);
    // Row `row` of the [rows, width] int32 bitmask at `data`.
    using FillBitmaskFn =
        int32_t (*)(uint64_t handle, size_t data, int64_t rows, int64_t width, int32_t row);
    // One row per draft-tree node, from num_nodes-long int64 columns.
    using TraverseDraftTreeFn = int32_t (*)(
        uint64_t handle, size_t next_node, size_t next_sibling, size_t draft_tokens,
        int64_t num_nodes, size_t bitmask, int64_t width);
    using RetainFn = int32_t (*)(uint64_t handle);
    using FreeFn = void (*)(uint64_t handle);

    explicit Entries(const std::map<std::string, uintptr_t>& addresses)
        : is_terminated(Lookup<IsTerminatedFn>(addresses, "is_terminated")),
          accept_token(Lookup<AcceptTokenFn>(addresses, "accept_token")),
          fill_bitmask(Lookup<FillBitmaskFn>(addresses, "fill_bitmask")),
          traverse_draft_tree(Lookup<TraverseDraftTreeFn>(addresses, "traverse_draft_tree")),
          retain(Lookup<RetainFn>(addresses, "retain")),
          free(Lookup<FreeFn>(addresses, "free")) {}

    const IsTerminatedFn is_terminated;
    const AcceptTokenFn accept_token;
    const FillBitmaskFn fill_bitmask;
    const TraverseDraftTreeFn traverse_draft_tree;
    const RetainFn retain;
    const FreeFn free;

  private:
    template <typename Entry>
    static Entry Lookup(const std::map<std::string, uintptr_t>& addresses, const char* name) {
        auto found = addresses.find(name);
        if (found == addresses.end() || found->second == 0) {
            throw py::value_error(std::string("no grammar entry ") + name);
        }
        return reinterpret_cast<Entry>(found->second);
    }
};

// One request's matcher: a hold on its handle and the tokens that end its grammar.
struct Matcher {
    Matcher(
        std::shared_ptr<Entries> entries, uint64_t handle,
        const std::vector<int64_t>& stop_token_ids)
        : entries(std::move(entries)),
          handle(handle),
          stop_token_ids(stop_token_ids.begin(), stop_token_ids.end()) {
        if (this->entries->retain(handle) != 0) {
            throw py::value_error("unknown grammar handle " + std::to_string(handle));
        }
    }

    ~Matcher() { entries->free(handle); }

    Matcher(const Matcher&) = delete;
    Matcher& operator=(const Matcher&) = delete;

    const std::shared_ptr<const Entries> entries;
    const uint64_t handle;
    const std::unordered_set<int64_t> stop_token_ids;
};

bool IsTerminated(const Matcher& matcher) {
    const int32_t status = matcher.entries->is_terminated(matcher.handle);
    if (status < 0) {
        // A matcher that cannot answer gets no mask and no tokens.
        LogError("is_terminated", "the matcher did not answer");
        return true;
    }
    return status != 0;
}

bool AcceptToken(const Matcher& matcher, int32_t token) {
    return matcher.entries->accept_token(matcher.handle, token) == 1;
}

// One forward's grammar work, and its completion.
class Batch {
  public:
    Batch(
        std::vector<std::shared_ptr<Matcher>> matchers,
        std::vector<bool> advance,
        bool has_candidates,
        int64_t num_extends,
        int64_t num_prefill_outputs,
        int64_t num_decodes,
        int64_t decode_width)
        : matchers_(std::move(matchers)),
          advance_(std::move(advance)),
          has_candidates_(has_candidates),
          num_extends_(num_extends),
          num_prefill_outputs_(num_prefill_outputs),
          decode_width_(decode_width),
          terminated_at_(matchers_.size(), -1) {
        const int64_t bs = static_cast<int64_t>(matchers_.size());
        if (static_cast<int64_t>(advance_.size()) != bs) {
            throw py::value_error("advance has a different length than matchers");
        }
        bool live = false;
        for (int64_t i = 0; i < bs; ++i) {
            if (matchers_[i] == nullptr) {
                continue;
            }
            if (i >= num_extends + num_decodes) {
                throw py::index_error("a grammar's request is outside the output layout");
            }
            live = true;
        }
        // Nothing to advance: the batch completes as it is made.
        state_ = live ? kPending : kDone;
    }

    int64_t bs() const { return static_cast<int64_t>(matchers_.size()); }
    bool has_candidates() const { return has_candidates_; }
    const std::shared_ptr<Matcher>& matcher(int64_t i) const { return matchers_[i]; }

    // ForwardOutputLayout.output_width
    int64_t OutputWidth(int64_t request) const {
        if (request < num_extends_) {
            return request < num_prefill_outputs_ ? 1 : 0;
        }
        return decode_width_;
    }

    // ForwardOutputLayout.token_offset
    int64_t TokenOffset(int64_t request) const {
        if (request < num_extends_) {
            return std::min(request, num_prefill_outputs_);
        }
        return num_prefill_outputs_ + (request - num_extends_) * decode_width_;
    }

    // From the host node: advances the batch, or waits for the thread that took that on first.
    void Advance(const int32_t* tokens, int64_t num_tokens, const int32_t* accept_lengths) {
        if (!TryAdvance(tokens, num_tokens, accept_lengths)) {
            WaitDone();
        }
    }

    // From post_process, with the GIL: it keeps the GIL to advance (microseconds) and
    // releases it only to wait for the host node.
    void AdvanceOnHost(const std::vector<int32_t>& tokens, const std::vector<int32_t>& accept_lengths) {
        if (static_cast<int64_t>(accept_lengths.size()) < bs()) {
            throw py::value_error("accept_lengths is shorter than the batch");
        }
        if (TryAdvance(tokens.data(), static_cast<int64_t>(tokens.size()), accept_lengths.data()) ||
            Done()) {
            return;
        }
        py::gil_scoped_release release;
        WaitDone();
    }

    // Whether the batch has advanced within timeout_s; the GIL is released only to wait.
    bool Wait(double timeout_s) {
        if (Done() || timeout_s <= 0) {
            return Done();
        }
        py::gil_scoped_release release;
        std::unique_lock<std::mutex> lock(mutex_);
        return done_.wait_for(
            lock, std::chrono::duration<double>(timeout_s), [this] { return state_ == kDone; });
    }

    std::vector<int32_t> TerminatedAt() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (state_ != kDone) {
            throw std::runtime_error("the batch has not advanced yet");
        }
        return terminated_at_;
    }

  private:
    enum State { kPending, kAdvancing, kDone };

    // Advances each matcher by its accepted tokens and completes the batch, unless
    // another thread took that on first.
    bool TryAdvance(const int32_t* tokens, int64_t num_tokens, const int32_t* accept_lengths) {
        // Allocated before the claim: past it nothing throws, so the batch always completes.
        std::vector<int32_t> terminated_at(matchers_.size(), -1);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (state_ != kPending) {
                return false;
            }
            state_ = kAdvancing;
        }
        for (int64_t i = 0; i < bs(); ++i) {
            const Matcher* matcher = matchers_[i].get();
            if (matcher == nullptr || !advance_[i] || IsTerminated(*matcher)) {
                continue;
            }
            const int64_t offset = TokenOffset(i);
            for (int64_t j = 0; j < accept_lengths[i] && offset + j < num_tokens; ++j) {
                if (!AcceptToken(*matcher, tokens[offset + j])) {
                    break;
                }
                if (IsTerminated(*matcher)) {
                    terminated_at[i] = static_cast<int32_t>(j);
                    break;
                }
            }
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            terminated_at_ = std::move(terminated_at);
            state_ = kDone;
        }
        done_.notify_all();
        return true;
    }

    bool Done() {
        std::lock_guard<std::mutex> lock(mutex_);
        return state_ == kDone;
    }

    void WaitDone() {
        std::unique_lock<std::mutex> lock(mutex_);
        done_.wait(lock, [this] { return state_ == kDone; });
    }

    const std::vector<std::shared_ptr<Matcher>> matchers_;
    const std::vector<bool> advance_;
    const bool has_candidates_;
    const int64_t num_extends_;
    const int64_t num_prefill_outputs_;
    const int64_t decode_width_;

    std::mutex mutex_;
    std::condition_variable done_;
    State state_;
    std::vector<int32_t> terminated_at_;
};

// Writes each forward's masks into a host bitmask; every other row reads -1 (allow all).
class Filler {
  public:
    Filler(
        uintptr_t bitmask,
        int64_t max_bs,
        int64_t tokens_per_request,
        int64_t vocab_size,
        uintptr_t candidates)
        : bitmask_(reinterpret_cast<int32_t*>(bitmask)),
          max_bs_(max_bs),
          n_(tokens_per_request),
          rows_(max_bs * tokens_per_request),
          width_((vocab_size + 31) / 32),
          vocab_size_(vocab_size),
          candidates_(reinterpret_cast<const int32_t*>(candidates)),
          chain_next_(n_),
          chain_sibling_(n_, -1),
          draft_tokens_(n_),
          filled_epoch_(rows_, 0) {
        // The verify drafts form a chain: node k's child is node k + 1.
        for (int64_t k = 0; k < n_; ++k) {
            chain_next_[k] = k + 1 < n_ ? k + 1 : -1;
        }
    }

    int64_t rows() const { return rows_; }

    std::shared_ptr<Batch> MakeBatch(
        std::vector<std::shared_ptr<Matcher>> matchers,
        std::vector<bool> advance,
        bool has_candidates,
        int64_t num_extends,
        int64_t num_prefill_outputs,
        int64_t num_decodes,
        int64_t decode_width) const {
        if (static_cast<int64_t>(matchers.size()) > max_bs_) {
            throw py::value_error("the batch is larger than max_bs");
        }
        return std::make_shared<Batch>(
            std::move(matchers), std::move(advance), has_candidates, num_extends,
            num_prefill_outputs, num_decodes, decode_width);
    }

    // Without a host node: fills a forward's masks before it runs; the commit advances the matchers.
    void FillNow(
        std::vector<std::shared_ptr<Matcher>> matchers,
        bool has_candidates,
        int64_t num_extends,
        int64_t num_prefill_outputs,
        int64_t num_decodes,
        int64_t decode_width) {
        std::vector<bool> advance(matchers.size(), false);
        if (!Fill(*MakeBatch(
                std::move(matchers), std::move(advance), has_candidates, num_extends,
                num_prefill_outputs, num_decodes, decode_width))) {
            throw std::runtime_error("a grammar matcher did not fill its mask");
        }
    }

    // Writes a forward's masks, resetting rows written before; false if a matcher failed (rows allow all).
    bool Fill(const Batch& batch) {
        bool ok = true;
        const int64_t per_request_rows = batch.has_candidates() ? n_ : 1;
        ++epoch_;
        filled_rows_.clear();
        auto filled = [this](int64_t row) {
            filled_epoch_[row] = epoch_;
            filled_rows_.push_back(row);
        };
        for (int64_t i = 0; i < batch.bs(); ++i) {
            const Matcher* matcher = batch.matcher(i).get();
            if (matcher == nullptr) {
                continue;
            }
            const int32_t terminated = matcher->entries->is_terminated(matcher->handle);
            if (terminated < 0) {
                LogError("is_terminated", "the matcher did not answer");
                ok = false;
            }
            if (terminated != 0) {
                continue;
            }
            const int64_t row = i * per_request_rows;
            const int64_t positions = batch.OutputWidth(i);
            // Rows are marked dirty before the matcher writes them, so a failed write is reset.
            if (positions == 1 || (positions && !batch.has_candidates())) {
                dirty_rows_.push_back(row);
                if (FillRow(*matcher, row)) {
                    filled(row);
                } else {
                    ok = false;
                }
            } else if (positions) {
                // Draft 0 was accepted by an earlier advance; a later advance
                // applies whatever the sampler accepts.
                const int32_t* drafts = candidates_ + i * n_;
                std::copy(drafts, drafts + n_, draft_tokens_.begin());
                for (int64_t k = 0; k < n_; ++k) {
                    dirty_rows_.push_back(row + k);
                }
                if (FillAlongDrafts(*matcher, row)) {
                    const int64_t masked = MaskedPositions(*matcher, row, drafts, positions);
                    for (int64_t k = 0; k < masked; ++k) {
                        filled(row + k);
                    }
                } else {
                    ok = false;
                }
            }
        }
        for (int64_t row : dirty_rows_) {
            if (filled_epoch_[row] != epoch_) {
                std::memset(bitmask_ + row * width_, 0xff, width_ * sizeof(int32_t));
            }
        }
        dirty_rows_.swap(filled_rows_);
        return ok;
    }

  private:
    bool FillRow(const Matcher& matcher, int64_t row) {
        if (matcher.entries->fill_bitmask(
                matcher.handle, reinterpret_cast<size_t>(bitmask_), rows_, width_,
                static_cast<int32_t>(row)) != 0) {
            LogError("fill_bitmask", "the matcher did not fill the row");
            return false;
        }
        return true;
    }

    // Fills rows row..row+n-1 with the masks after drafts 1..k; the matcher is left as it was.
    bool FillAlongDrafts(const Matcher& matcher, int64_t row) {
        if (matcher.entries->traverse_draft_tree(
                matcher.handle, reinterpret_cast<size_t>(chain_next_.data()),
                reinterpret_cast<size_t>(chain_sibling_.data()),
                reinterpret_cast<size_t>(draft_tokens_.data()), n_,
                reinterpret_cast<size_t>(bitmask_ + row * width_), width_) != 0) {
            LogError("traverse_draft_tree", "the matcher did not walk the drafts");
            return false;
        }
        return true;
    }

    // How many leading positions of a draft walk got a mask: position k does
    // when draft k is in the vocabulary, position k-1's mask allows it and it
    // does not end the grammar. The vocabulary check comes first: a mask can
    // set the bits past the vocabulary in its last word.
    int64_t MaskedPositions(
        const Matcher& matcher, int64_t row, const int32_t* drafts, int64_t positions) const {
        int64_t k = 1;
        for (; k < positions; ++k) {
            const int64_t token = drafts[k];
            if (matcher.stop_token_ids.count(token) || token < 0 || token >= vocab_size_) {
                break;
            }
            const auto word =
                static_cast<uint32_t>(bitmask_[(row + k - 1) * width_ + (token >> 5)]);
            if (!((word >> (token & 31)) & 1u)) {
                break;
            }
        }
        return k;
    }

    int32_t* const bitmask_;
    const int64_t max_bs_;
    const int64_t n_;
    const int64_t rows_;
    const int64_t width_;
    const int64_t vocab_size_;
    const int32_t* const candidates_;

    std::vector<int64_t> chain_next_;
    std::vector<int64_t> chain_sibling_;
    std::vector<int64_t> draft_tokens_;

    // Rows that may hold anything but -1, and the fill that last wrote each row.
    std::vector<int64_t> dirty_rows_;
    std::vector<int64_t> filled_rows_;
    std::vector<uint64_t> filled_epoch_;
    uint64_t epoch_ = 0;
};

#ifdef TOKENSPEED_GRAMMAR_HOST_NODE
// Runs each forward's grammar work as a CUDA host node.
class Host {
  public:
    Host(std::shared_ptr<Filler> filler, uintptr_t output_tokens, uintptr_t accept_lengths)
        : filler_(std::move(filler)),
          output_tokens_(reinterpret_cast<const int32_t*>(output_tokens)),
          accept_lengths_(reinterpret_cast<const int32_t*>(accept_lengths)) {}

    std::shared_ptr<Batch> Add(
        std::vector<std::shared_ptr<Matcher>> matchers,
        std::vector<bool> advance,
        bool has_candidates,
        int64_t num_extends,
        int64_t num_prefill_outputs,
        int64_t num_decodes,
        int64_t decode_width) {
        auto batch = filler_->MakeBatch(
            std::move(matchers), std::move(advance), has_candidates, num_extends,
            num_prefill_outputs, num_decodes, decode_width);
        std::vector<std::shared_ptr<Batch>> retired;
        std::lock_guard<std::mutex> lock(queue_mutex_);
        retired.swap(retired_);
        // Room for every batch the running and queued executions retire, so retiring never allocates.
        retired_.reserve(running_ + queue_.size() + 1);
        queue_.push_back(batch);
        return batch;
    }

    // Drops the batches no execution will pop, and the one the last execution left as current.
    void Clear() {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        queue_.clear();
        current_.reset();
        retired_.clear();
    }

    size_t Queued() {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        return queue_.size();
    }

    void Launch(uintptr_t stream) {
        cudaError_t error =
            cudaLaunchHostFunc(reinterpret_cast<cudaStream_t>(stream), &Host::Execute, this);
        if (error != cudaSuccess) {
            throw std::runtime_error(
                std::string("cudaLaunchHostFunc failed: ") + cudaGetErrorString(error));
        }
    }

  private:
    static void CUDART_CB Execute(void* self) {
        try {
            static_cast<Host*>(self)->Run();
        } catch (const std::exception& error) {
            LogError("host node", error.what());
        } catch (...) {
            LogError("host node", "unknown exception");
        }
    }

    // Pops this forward's batch, advances the previous one, then fills the bitmask.
    void Run() {
        std::shared_ptr<Batch> batch;
        std::shared_ptr<Batch> previous;
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            if (queue_.empty()) {
                LogError("host node", "no batch was added for this forward");
                return;
            }
            batch = std::move(queue_.front());
            queue_.pop_front();
            previous = std::exchange(current_, batch);
            ++running_;
        }
        try {
            if (previous != nullptr) {
                previous->Advance(output_tokens_, filler_->rows(), accept_lengths_);
            }
            // The node cannot raise: a failed fill is logged and its rows allow every token.
            filler_->Fill(*batch);
        } catch (...) {
            Retire(std::move(previous));
            throw;
        }
        Retire(std::move(previous));
    }

    // It may hold a matcher's last reference: the next add releases it, on a Python thread.
    void Retire(std::shared_ptr<Batch> batch) {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        retired_.push_back(std::move(batch));
        --running_;
    }

    const std::shared_ptr<Filler> filler_;
    const int32_t* const output_tokens_;
    const int32_t* const accept_lengths_;

    std::mutex queue_mutex_;
    std::deque<std::shared_ptr<Batch>> queue_;
    std::shared_ptr<Batch> current_;
    std::vector<std::shared_ptr<Batch>> retired_;
    size_t running_ = 0;
};
#endif

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<Entries, std::shared_ptr<Entries>>(m, "Entries")
        .def(py::init<const std::map<std::string, uintptr_t>&>());
    py::class_<Matcher, std::shared_ptr<Matcher>>(m, "Matcher")
        .def(py::init<std::shared_ptr<Entries>, uint64_t, const std::vector<int64_t>&>());
    py::class_<Batch, std::shared_ptr<Batch>>(m, "Batch")
        .def("wait", &Batch::Wait)
        .def("advance_on_host", &Batch::AdvanceOnHost)
        .def("terminated_at", &Batch::TerminatedAt);
    py::class_<Filler, std::shared_ptr<Filler>>(m, "Filler")
        .def(py::init<uintptr_t, int64_t, int64_t, int64_t, uintptr_t>())
        // Keeps the GIL, as the matcher calls from Python do: each takes microseconds.
        .def("fill", &Filler::FillNow);
#ifdef TOKENSPEED_GRAMMAR_HOST_NODE
    py::class_<Host, std::shared_ptr<Host>>(m, "Host")
        .def(py::init<std::shared_ptr<Filler>, uintptr_t, uintptr_t>())
        .def("add", &Host::Add)
        .def("clear", &Host::Clear)
        .def("__len__", &Host::Queued)
        .def("launch", &Host::Launch);
#endif
}
"""


def load_grammar_host_ext() -> Any:
    """The JIT-compiled extension: ``Filler``, ``Batch``, ``Matcher`` and
    ``Entries``, and on CUDA ``Host``, the host node. One build per process,
    so its types mix."""
    global _ext
    if _ext is not None:
        return _ext
    with _ext_lock:
        if _ext is not None:
            return _ext
        import glob
        import os
        import sys

        import torch
        from torch.utils.cpp_extension import CUDA_HOME, load_inline

        # The host node needs the CUDA runtime; elsewhere the build needs no GPU runtime.
        host_node = torch.version.cuda is not None

        include_dirs = []
        library_dirs = []
        cflags = ["-O2"]
        ldflags = []
        if host_node:
            cuda_home = CUDA_HOME or os.environ.get("CUDA_HOME")
            if cuda_home:
                inc = os.path.join(cuda_home, "include")
                lib = os.path.join(cuda_home, "lib64")
                if os.path.isdir(inc):
                    include_dirs.append(inc)
                if os.path.isdir(lib):
                    library_dirs.append(lib)
            # conda-forge keeps the toolkit under <prefix>/targets/<platform>; base_prefix covers a venv.
            for root in dict.fromkeys(
                r for r in (cuda_home, sys.prefix, sys.base_prefix) if r
            ):
                include_dirs += sorted(
                    glob.glob(os.path.join(root, "targets", "*", "include"))
                )
                library_dirs += sorted(
                    glob.glob(os.path.join(root, "targets", "*", "lib"))
                )
            cflags.append("-DTOKENSPEED_GRAMMAR_HOST_NODE")
            ldflags.append("-lcudart")

        _ext = load_inline(
            name="tokenspeed_grammar_host_ext",
            cpp_sources=[_CPP_SRC],
            extra_cflags=cflags + [f"-I{d}" for d in include_dirs],
            extra_ldflags=[f"-L{d}" for d in library_dirs] + ldflags,
            verbose=False,
        )
        return _ext

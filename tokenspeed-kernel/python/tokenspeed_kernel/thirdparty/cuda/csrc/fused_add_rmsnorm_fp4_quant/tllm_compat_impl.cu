// Minimal TRT-LLM compatibility stubs so fused_add_rmsnorm_fp4_quant.so links
// standalone without the full TRT-LLM runtime or flashinfer nv_internal .cpp
// files.  Only provides symbols actually referenced through the header chain:
//   cudaUtils.h -> tllmException.h  (TllmException ctor/dtor/demangle)
//   cudaUtils.h -> stringUtils.h    (fmtstr_)
//   cudaUtils.h -> assert.h         (uses both of the above)

#include "tensorrt_llm/common/tllmException.h"
#include "tensorrt_llm/common/stringUtils.h"

#include <cstdarg>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <unordered_set>
#if !defined(_MSC_VER)
#include <cxxabi.h>
#include <execinfo.h>
#endif

namespace tensorrt_llm::common {

// ─── fmtstr_ (from stringUtils.cpp) ────────────────────────────────────────

void fmtstr_(char const* format, fmtstr_allocator alloc, void* target, va_list args) {
  va_list args0;
  va_copy(args0, args);

  size_t constexpr init_size = 2048;
  char fixed_buffer[init_size];
  auto const size = std::vsnprintf(fixed_buffer, init_size, format, args0);
  va_end(args0);
  if (size <= 0) {
    return;
  }

  auto* memory = alloc(target, size);
  if (static_cast<size_t>(size) < init_size) {
    std::memcpy(memory, fixed_buffer, size + 1);
  } else {
    std::vsnprintf(memory, size + 1, format, args);
  }
}

std::unordered_set<std::string> str2set(std::string const& input, char delimiter) {
  std::unordered_set<std::string> values;
  if (!input.empty()) {
    std::stringstream valStream(input);
    std::string val;
    while (std::getline(valStream, val, delimiter)) {
      if (!val.empty()) {
        values.insert(val);
      }
    }
  }
  return values;
}

// ─── TllmException (from tllmException.cpp) ─────────────────────────────────

#if !defined(_MSC_VER)
TllmException::TllmException(char const* file, std::size_t line, char const* msg)
    : std::runtime_error{""} {
  mNbFrames = backtrace(mCallstack.data(), MAX_FRAMES);
  auto const trace = getTrace();
  std::runtime_error::operator=(
      std::runtime_error{fmtstr("%s (%s:%zu)\n%s", msg, file, line, trace.c_str())});
}
#else
TllmException::TllmException(char const* file, std::size_t line, char const* msg)
    : mNbFrames{}, std::runtime_error{fmtstr("%s (%s:%zu)", msg, file, line)} {}
#endif

TllmException::~TllmException() noexcept = default;

std::string TllmException::getTrace() const {
#if defined(_MSC_VER)
  return "";
#else
  auto const trace = std::unique_ptr<char const*, void (*)(char const**)>(
      const_cast<char const**>(backtrace_symbols(mCallstack.data(), mNbFrames)),
      [](char const** p) { std::free(p); });
  if (trace == nullptr) {
    return "[backtrace unavailable]";
  }
  std::ostringstream buf;
  for (auto i = 1; i < mNbFrames; ++i) {
    buf << trace.get()[i];
    if (i < mNbFrames - 1) buf << '\n';
  }
  if (mNbFrames == MAX_FRAMES) buf << "\n[truncated]";
  return buf.str();
#endif
}

std::string TllmException::demangle(char const* name) {
#if defined(_MSC_VER)
  return name;
#else
  std::string clearName{name};
  auto status = -1;
  auto const demangled = abi::__cxa_demangle(name, nullptr, nullptr, &status);
  if (status == 0) {
    clearName = demangled;
    std::free(demangled);
  }
  return clearName;
#endif
}

// ─── Logger (minimal stub) ──────────────────────────────────────────────────

}  // namespace tensorrt_llm::common

#include "tensorrt_llm/common/logger.h"

namespace tensorrt_llm::common {

Logger::Logger() {}

Logger* Logger::getLogger() {
  thread_local Logger instance;
  return &instance;
}

void Logger::log(std::exception const& ex, Logger::Level level) {
  log(level, "%s: %s", TllmException::demangle(typeid(ex).name()).c_str(), ex.what());
}

}  // namespace tensorrt_llm::common

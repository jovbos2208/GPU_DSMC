#pragma once

// Lightweight phase profiler (M5.2). When enabled, every phase is bracketed
// by Kokkos::fence() so that the measured wall time belongs to the kernels of
// that phase on any backend (CPU, CUDA, HIP). Disabled (default) it costs
// nothing: no fences are inserted and the solver runs asynchronously.
// Enabled with the command-line option --profile or VLEO_PROFILE=1.
//
// Independently of that, every scope is also a Kokkos profiling region
// (pushRegion/popRegion) whenever a Kokkos tool is loaded (rocprofv3
// --kokkos-trace, KOKKOS_TOOLS_LIBS=..., rocprof-sys): the phases then appear
// as named host ranges in external timelines, without fences, so the
// asynchronous execution is not disturbed. Without a tool this is one branch.

#include <chrono>
#include <map>
#include <string>

#include <Kokkos_Core.hpp>

namespace vleo {

// Named Kokkos profiling region for external tools (no-op without a loaded tool).
class Region {
 public:
  explicit Region(const char* name) : active_(Kokkos::Profiling::profileLibraryLoaded()) {
    if (active_) Kokkos::Profiling::pushRegion(name);
  }
  ~Region() {
    if (active_) Kokkos::Profiling::popRegion();
  }
  Region(const Region&) = delete;
  Region& operator=(const Region&) = delete;

 private:
  bool active_;
};

class Profiler {
 public:
  explicit Profiler(bool enabled = false) : enabled_(enabled) {}

  bool enabled() const { return enabled_; }

  class Scope {
   public:
    Scope(Profiler& p, const char* name) : p_(p), name_(name), region_(name) {
      if (p_.enabled_) {
        Kokkos::fence();
        t0_ = std::chrono::steady_clock::now();
      }
    }
    ~Scope() {
      if (p_.enabled_) {
        Kokkos::fence();
        auto& e = p_.phases_[name_];
        e.seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_).count();
        ++e.calls;
      }
    }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;

   private:
    Profiler& p_;
    const char* name_;
    Region region_;
    std::chrono::steady_clock::time_point t0_;
  };

  struct Entry {
    double seconds = 0.0;
    long calls = 0;
  };

  const std::map<std::string, Entry>& phases() const { return phases_; }

  // Adds an externally timed phase (no-op when disabled).
  void add(const std::string& name, double seconds) {
    if (!enabled_) return;
    auto& e = phases_[name];
    e.seconds += seconds;
    ++e.calls;
  }

 private:
  bool enabled_;
  std::map<std::string, Entry> phases_;
};

// True if profiling was requested through the environment (VLEO_PROFILE=1).
bool profiling_requested_by_environment();

}  // namespace vleo

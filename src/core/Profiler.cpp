#include "vleo/core/Profiler.hpp"

#include <cstdlib>
#include <string_view>

namespace vleo {

bool profiling_requested_by_environment() {
  const char* v = std::getenv("VLEO_PROFILE");
  return v != nullptr && std::string_view(v) != "" && std::string_view(v) != "0";
}

}  // namespace vleo

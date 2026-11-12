#include "vleo/io/PlatformConfig.hpp"

#include <fstream>
#include <limits>
#include <sstream>

#include <Kokkos_Core.hpp>

#include "YamlHelpers.hpp"

namespace vleo {

using namespace io_detail;

std::string_view to_string(Backend b) {
  switch (b) {
    case Backend::CPU: return "cpu";
    case Backend::CUDA: return "cuda";
    case Backend::HIP: return "hip";
  }
  return "?";
}

Backend compiled_backend() {
#if defined(KOKKOS_ENABLE_CUDA)
  return Backend::CUDA;
#elif defined(KOKKOS_ENABLE_HIP)
  return Backend::HIP;
#else
  return Backend::CPU;
#endif
}

PlatformConfig parse_platform_config(std::string_view yaml_text) {
  YAML::Node root;
  try {
    root = YAML::Load(std::string(yaml_text));
  } catch (const YAML::Exception& e) {
    throw ConfigError({"YAML syntax error at line " + std::to_string(e.mark.line + 1) + ": " + e.msg},
                      "platform configuration");
  }
  Issues is;
  PlatformConfig cfg;
  if (!root.IsDefined() || root.IsNull()) {
    is.add("", "platform configuration is empty");
    is.throw_if_any("platform configuration");
  }
  if (check_map(root, "", {"platform"}, is)) {
    if (auto n = child(root, "platform", "", is, Need::Required)) {
      const std::string p = "platform";
      if (check_map(*n, p, {"backend", "device_id", "num_threads", "description"}, is)) {
        static constexpr std::array<EnumEntry<Backend>, 3> kBackends{
            {{"cpu", Backend::CPU}, {"cuda", Backend::CUDA}, {"hip", Backend::HIP}}};
        if (auto b = get_enum(*n, "backend", p, is, Need::Required, kBackends)) cfg.backend = *b;
        constexpr auto kMaxInt = static_cast<std::uint64_t>(std::numeric_limits<int>::max());
        if (auto v = get_uint64(*n, "device_id", p, is, Need::Optional)) {
          if (*v > kMaxInt)
            is.add(p + ".device_id", "too large");
          else
            cfg.device_id = static_cast<int>(*v);
        }
        if (auto v = get_uint64(*n, "num_threads", p, is, Need::Optional)) {
          if (*v > kMaxInt)
            is.add(p + ".num_threads", "too large");
          else
            cfg.num_threads = static_cast<int>(*v);
        }
        if (auto v = get_string(*n, "description", p, is, Need::Optional, true)) cfg.description = *v;
      }
    }
  }
  is.throw_if_any("platform configuration");
  return cfg;
}

PlatformConfig load_platform_config(const std::filesystem::path& file) {
  const auto abs = std::filesystem::absolute(file).lexically_normal();
  std::ifstream in(abs, std::ios::binary);
  if (!in) throw ConfigError({"cannot open platform file '" + abs.string() + "'"}, "platform configuration");
  std::ostringstream text;
  text << in.rdbuf();
  PlatformConfig cfg = parse_platform_config(text.str());
  cfg.source_file = abs;
  return cfg;
}

void check_platform_matches_build(const PlatformConfig& p) {
  if (p.backend != compiled_backend()) {
    throw ConfigError({"platform.backend: '" + std::string(to_string(p.backend)) +
                       "' requested, but this binary was built for '" + std::string(to_string(compiled_backend())) +
                       "'. Use the binary from the matching build (see docs/building.md)."},
                      "platform configuration");
  }
}

}  // namespace vleo

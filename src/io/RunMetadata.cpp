#include "vleo/io/RunMetadata.hpp"

#include <chrono>
#include <ctime>
#include <stdexcept>

#include "vleo/core/BuildInfo.hpp"
#include "vleo/io/ConfigWriter.hpp"
#include "vleo/io/JsonWriter.hpp"

namespace vleo {

std::string utc_timestamp_now() {
  const std::time_t t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  std::tm tm{};
#if defined(_WIN32)
  gmtime_s(&tm, &t);
#else
  gmtime_r(&t, &tm);
#endif
  char buf[32];
  std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm);
  return buf;
}

std::string make_run_metadata_json(const RunMetadataInput& in) {
  if (!in.config) throw std::invalid_argument("make_run_metadata_json: config is required");
  const SimulationConfig& cfg = *in.config;
  const BuildInfo& b = build_info();

  JsonWriter w;
  w.begin_object();
  w.field("schema", kRunMetadataSchema);
  w.field("status", in.status);

  w.key("simulation").begin_object();
  w.field("name", cfg.simulation.name);
  w.field("mode", to_string(cfg.simulation.mode));
  w.field("backend", b.configure.backend);
  w.field("gsi_model", to_string(cfg.surface.model));
  w.field("random_seed", cfg.simulation.random_seed);
  w.end_object();

  w.key("build").begin_object();
  w.field("solver_version", b.configure.solver_version);
  w.key("git").begin_object();
  w.field("available", b.git.available);
  w.field("commit", b.git.commit);
  w.field("describe", b.git.describe);
  w.field("dirty", b.git.dirty);
  w.end_object();
  w.field("build_type", b.configure.build_type);
  w.key("compiler").begin_object();
  w.field("id", b.configure.compiler_id);
  w.field("version", b.configure.compiler_version);
  w.field("detail", b.compiler_detail);
  w.field("path", b.configure.compiler_path);
  w.field("flags", b.configure.cxx_flags);
  w.field("cxx_standard", static_cast<std::int64_t>(b.cxx_standard));
  w.end_object();
  w.key("kokkos").begin_object();
  w.field("version", b.kokkos_version);
  w.key("enabled_backends").begin_array();
  for (const auto& s : b.enabled_kokkos_backends) w.value(s);
  w.end_array();
  w.field("default_execution_space", b.default_execution_space);
  w.field("default_host_execution_space", b.default_host_execution_space);
  w.field("arch", b.configure.kokkos_arch);
  w.end_object();
  w.field("backend", b.configure.backend);
  w.field("cmake_version", b.configure.cmake_version);
  w.field("system", b.configure.system);
  w.end_object();

  w.key("runtime").begin_object();
  w.field("timestamp_utc", in.runtime.timestamp_utc);
  w.key("command_line").begin_array();
  for (const auto& a : in.runtime.command_line) w.value(a);
  w.end_array();
  w.field("working_directory", in.runtime.working_directory);
  w.field("execution_space", in.runtime.execution_space);
  w.field("concurrency", in.runtime.concurrency);
  w.field("device_id", in.runtime.device_id);
  w.end_object();

  w.key("platform");
  if (in.platform) {
    w.begin_object();
    w.field("file", in.platform->source_file.string());
    w.field("backend", to_string(in.platform->backend));
    w.field("device_id", in.platform->device_id);
    w.field("num_threads", in.platform->num_threads);
    w.field("description", in.platform->description);
    w.end_object();
  } else {
    w.null();
  }

  w.key("input").begin_object();
  w.field("config_file", cfg.source_file.string());
  w.end_object();

  w.key("resolved_config");
  write_resolved_json(w, cfg);
  w.key("derived");
  write_derived_json(w, cfg);

  w.key("output_files").begin_array();
  for (const auto& f : in.output_files) w.value(f);
  w.end_array();
  w.key("notes").begin_array();
  for (const auto& n : in.notes) w.value(n);
  w.end_array();

  w.end_object();
  return w.str();
}

}  // namespace vleo

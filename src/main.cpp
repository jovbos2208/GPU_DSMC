// vleo_solver - command-line entry point.
//
// Current state (milestone M1): reads and validates the simulation (and
// optional platform) configuration and the geometry (STL import, topology and
// orientation checks, BVH), sets up the simulation domain and the free-stream
// inflow, initialises Kokkos on the compiled backend, runs a backend self-test
// and archives the resolved configuration plus complete run metadata. The
// physical surface models and force/heat tallies follow in M2.

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

#include "vleo/core/BuildInfo.hpp"
#include "vleo/core/SelfTest.hpp"
#include "vleo/geometry/Geometry.hpp"
#include "vleo/io/ConfigParser.hpp"
#include "vleo/io/ConfigWriter.hpp"
#include "vleo/io/PlatformConfig.hpp"
#include "vleo/io/RunMetadata.hpp"
#include "vleo/particles/Inflow.hpp"

namespace fs = std::filesystem;

namespace {

constexpr int kExitOk = 0;
constexpr int kExitRuntimeError = 1;
constexpr int kExitConfigError = 2;
constexpr int kExitUsage = 64;

struct Options {
  std::optional<fs::path> config;
  std::optional<fs::path> platform;
  std::optional<fs::path> output_dir;
  bool check_only = false;
  bool print_config = false;
  bool version = false;
  bool help = false;
};

void print_usage(std::ostream& os) {
  os << "Usage: vleo_solver --config <simulation.yaml> [options]\n"
        "\n"
        "Options:\n"
        "  -c, --config <file>      Simulation configuration (required)\n"
        "  -p, --platform <file>    Platform configuration; its backend must match this build\n"
        "  -o, --output-dir <dir>   Override output.directory from the configuration\n"
        "      --check              Validate the configuration(s) and exit (no Kokkos initialisation)\n"
        "      --print-config       Print the resolved configuration (YAML) to stdout\n"
        "      --version            Print build information and exit\n"
        "  -h, --help               Show this help\n"
        "\n"
        "Exit codes: 0 success, 1 runtime error, 2 invalid configuration, 64 usage error.\n";
}

std::optional<Options> parse_args(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto need_value = [&](const char* name) -> std::optional<std::string> {
      if (i + 1 >= argc) {
        std::cerr << "error: " << name << " requires a value\n";
        return std::nullopt;
      }
      return std::string(argv[++i]);
    };
    if (a == "-c" || a == "--config") {
      auto v = need_value("--config");
      if (!v) return std::nullopt;
      o.config = *v;
    } else if (a == "-p" || a == "--platform") {
      auto v = need_value("--platform");
      if (!v) return std::nullopt;
      o.platform = *v;
    } else if (a == "-o" || a == "--output-dir") {
      auto v = need_value("--output-dir");
      if (!v) return std::nullopt;
      o.output_dir = *v;
    } else if (a == "--check") {
      o.check_only = true;
    } else if (a == "--print-config") {
      o.print_config = true;
    } else if (a == "--version") {
      o.version = true;
    } else if (a == "-h" || a == "--help") {
      o.help = true;
    } else {
      std::cerr << "error: unknown argument '" << a << "'\n";
      return std::nullopt;
    }
  }
  return o;
}

void write_text_file(const fs::path& path, const std::string& content) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) throw std::runtime_error("cannot write '" + path.string() + "'");
  out << content;
  if (!out) throw std::runtime_error("error while writing '" + path.string() + "'");
}

std::string read_text_file(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot read '" + path.string() + "'");
  std::ostringstream s;
  s << in.rdbuf();
  return s.str();
}

void write_geometry_json(vleo::JsonWriter& w, const vleo::Geometry& g, const vleo::SimulationConfig& cfg) {
  const vleo::MeshReport& r = g.report;
  w.begin_object();
  w.field("file", cfg.geometry.resolved_file.string());
  w.field("input_triangles", static_cast<std::uint64_t>(r.input_triangles));
  w.field("degenerate_removed", static_cast<std::uint64_t>(r.degenerate_removed));
  w.field("triangles", static_cast<std::uint64_t>(r.triangles));
  w.field("vertices", static_cast<std::uint64_t>(r.vertices));
  w.field("edges", static_cast<std::uint64_t>(r.edges));
  w.field("boundary_edges", static_cast<std::uint64_t>(r.boundary_edges));
  w.field("nonmanifold_edges", static_cast<std::uint64_t>(r.nonmanifold_edges));
  w.field("inconsistent_edges", static_cast<std::uint64_t>(r.inconsistent_edges));
  w.field("stored_normal_disagreements", static_cast<std::uint64_t>(r.stored_normal_disagreements));
  w.field("closed", r.closed);
  w.field("total_area_m2", r.total_area_m2);
  w.field("bbox_min_m", r.bbox.lo);
  w.field("bbox_max_m", r.bbox.hi);
  w.field("surface_offset_m", g.device.surface_offset_m);
  w.key("materials").begin_array();
  for (const auto& m : g.host.material_names) w.value(m);
  w.end_array();
  w.key("components").begin_array();
  for (const auto& c : r.components) {
    w.begin_object();
    w.field("triangles", static_cast<std::uint64_t>(c.triangles));
    w.field("closed", c.closed);
    w.field("flipped", c.flipped);
    w.field("area_m2", c.area_m2);
    w.field("volume_m3", c.signed_volume_m3);
    w.end_object();
  }
  w.end_array();
  w.key("bvh").begin_object();
  w.field("nodes", static_cast<std::uint64_t>(g.bvh.stats.nodes));
  w.field("leaves", static_cast<std::uint64_t>(g.bvh.stats.leaves));
  w.field("max_depth", g.bvh.stats.max_depth);
  w.field("max_leaf_triangles", g.bvh.stats.max_leaf_triangles);
  w.field("sah_cost", g.bvh.stats.sah_cost);
  w.end_object();
  w.key("warnings").begin_array();
  for (const auto& m : r.warnings) w.value(m);
  w.end_array();
  w.end_object();
}

void write_inflow_json(vleo::JsonWriter& w, const vleo::InflowModel& inflow, const vleo::SimulationConfig& cfg) {
  static const char* const face_names[6] = {"-x", "+x", "-y", "+y", "-z", "+z"};
  w.begin_object();
  w.field("domain_min_m", inflow.domain.box.lo);
  w.field("domain_max_m", inflow.domain.box.hi);
  w.field("total_rate_per_s", inflow.total_rate_per_s);
  w.field("tpmc_event_rate_factor_per_s",
          inflow.total_rate_per_s / static_cast<double>(cfg.tpmc.particle_histories));
  w.key("channels").begin_array();
  for (const auto& c : inflow.channels) {
    w.begin_object();
    w.field("face", face_names[c.face]);
    w.field("species", cfg.gas.species[c.species].name);
    w.field("speed_ratio", c.s);
    w.field("rate_per_s", c.rate_per_s);
    w.end_object();
  }
  w.end_array();
  w.end_object();
}

// Releases device memory before Kokkos::finalize (declared after the ScopeGuard).
struct DeviceGeometryReset {
  vleo::Geometry& g;
  ~DeviceGeometryReset() { g.device = vleo::DeviceGeometry{}; }
};

int run(const Options& opt, int argc, char** argv) {
  // --- Configuration (before touching any device) ---------------------------
  std::optional<vleo::PlatformConfig> platform;
  vleo::SimulationConfig cfg;
  try {
    if (opt.platform) {
      platform = vleo::load_platform_config(*opt.platform);
      vleo::check_platform_matches_build(*platform);
    }
    cfg = vleo::load_simulation_config(*opt.config);
  } catch (const vleo::ConfigError& e) {
    std::cerr << "error: " << e.what() << "\n";
    return kExitConfigError;
  }

  // --- Geometry, domain and inflow (host side) --------------------------------
  std::vector<std::string> notes;
  vleo::Geometry geometry;
  vleo::InflowModel inflow;
  try {
    vleo::MeshOptions mopt;
    mopt.require_closed = cfg.geometry.require_closed;
    geometry = vleo::make_host_geometry(
        vleo::read_stl(cfg.geometry.resolved_file, cfg.derived.geometry_scale_to_m), mopt);
    for (const auto& w : geometry.report.warnings) {
      notes.push_back("geometry: " + w);
      std::cerr << "warning: geometry: " << w << "\n";
    }
    const vleo::Domain domain =
        vleo::make_domain(cfg.domain, geometry.host.bbox, 1.0e3 * geometry.device.surface_offset_m);
    inflow = vleo::InflowModel::create(domain, cfg.gas);
  } catch (const std::exception& e) {  // StlError, GeometryError
    std::cerr << "error: " << e.what() << "\n";
    return kExitConfigError;
  }

  if (opt.print_config) std::cout << vleo::to_resolved_yaml(cfg);
  if (opt.check_only) {
    std::cerr << "configuration valid: " << cfg.source_file.string() << " (" << geometry.report.triangles
              << " triangles, " << geometry.report.components.size() << " component(s))\n";
    return kExitOk;
  }

  // --- Kokkos ----------------------------------------------------------------
  Kokkos::InitializationSettings settings;
  if (platform) {
    if (platform->backend != vleo::Backend::CPU) settings.set_device_id(platform->device_id);
    if (platform->num_threads > 0) settings.set_num_threads(platform->num_threads);
  }
  Kokkos::ScopeGuard guard(settings);
  vleo::upload_geometry(geometry);
  const DeviceGeometryReset release_device_geometry{geometry};

  vleo::RuntimeInfo rt;
  rt.timestamp_utc = vleo::utc_timestamp_now();
  for (int i = 0; i < argc; ++i) rt.command_line.emplace_back(argv[i]);
  rt.working_directory = fs::current_path().string();
  rt.execution_space = Kokkos::DefaultExecutionSpace::name();
  rt.concurrency = Kokkos::DefaultExecutionSpace().concurrency();
  rt.device_id = vleo::compiled_backend() == vleo::Backend::CPU ? -1 : Kokkos::device_id();

  const vleo::SelfTestResult self_test = vleo::backend_self_test(cfg.simulation.random_seed);
  const bool self_test_ok = self_test.passed;
  notes.push_back(self_test.detail);
  if (!self_test_ok) std::cerr << "error: " << self_test.detail << "\n";

  notes.push_back("milestone M1: geometry, domain and inflow set up; no surface model is available yet, so "
                  "no particle simulation was performed");

  // --- Outputs -----------------------------------------------------------------
  const fs::path out_dir = opt.output_dir ? *opt.output_dir : fs::path(cfg.output.directory);
  try {
    fs::create_directories(out_dir);

    const fs::path f_resolved = out_dir / "resolved_config.yaml";
    const fs::path f_input = out_dir / "input_config.yaml";
    const fs::path f_kokkos = out_dir / "kokkos_configuration.txt";
    const fs::path f_meta = out_dir / "run_metadata.json";

    write_text_file(f_resolved, vleo::to_resolved_yaml(cfg));
    write_text_file(f_input, read_text_file(cfg.source_file));
    std::ostringstream kc;
    Kokkos::print_configuration(kc, true);
    write_text_file(f_kokkos, kc.str());

    vleo::RunMetadataInput meta;
    meta.config = &cfg;
    meta.platform = platform ? &*platform : nullptr;
    meta.runtime = rt;
    meta.status = self_test_ok ? "configured" : "self_test_failed";
    meta.notes = notes;
    meta.sections.emplace_back("geometry", [&](vleo::JsonWriter& w) { write_geometry_json(w, geometry, cfg); });
    meta.sections.emplace_back("inflow", [&](vleo::JsonWriter& w) { write_inflow_json(w, inflow, cfg); });
    meta.output_files = {f_meta.filename().string(), f_resolved.filename().string(), f_input.filename().string(),
                         f_kokkos.filename().string()};
    write_text_file(f_meta, vleo::make_run_metadata_json(meta));

    std::cerr << "vleo_solver: backend " << vleo::to_string(vleo::compiled_backend()) << " ("
              << rt.execution_space << "), wrote " << fs::absolute(f_meta).lexically_normal().string() << "\n";
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return kExitRuntimeError;
  }
  return self_test_ok ? kExitOk : kExitRuntimeError;
}

}  // namespace

int main(int argc, char** argv) {
  const auto opt = parse_args(argc, argv);
  if (!opt) {
    print_usage(std::cerr);
    return kExitUsage;
  }
  if (opt->help) {
    print_usage(std::cout);
    return kExitOk;
  }
  if (opt->version) {
    std::cout << vleo::build_info_string();
    return kExitOk;
  }
  if (!opt->config) {
    std::cerr << "error: --config is required\n";
    print_usage(std::cerr);
    return kExitUsage;
  }
  try {
    return run(*opt, argc, argv);
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return kExitRuntimeError;
  }
}

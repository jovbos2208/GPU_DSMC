// vleo_solver - command-line entry point.
//
// Milestone M0: reads and validates the simulation (and optional platform)
// configuration, initialises Kokkos on the compiled backend, runs a backend
// self-test and archives the resolved configuration plus complete run
// metadata. The TPMC/DSMC engines follow in M1-M3.

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
#include "vleo/io/ConfigParser.hpp"
#include "vleo/io/ConfigWriter.hpp"
#include "vleo/io/PlatformConfig.hpp"
#include "vleo/io/RunMetadata.hpp"

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

  std::vector<std::string> notes;
  if (!fs::exists(cfg.geometry.resolved_file)) {
    notes.push_back("geometry file '" + cfg.geometry.resolved_file.string() +
                    "' does not exist; it is required once geometry import is available (M1)");
    std::cerr << "warning: " << notes.back() << "\n";
  }

  if (opt.print_config) std::cout << vleo::to_resolved_yaml(cfg);
  if (opt.check_only) {
    std::cerr << "configuration valid: " << cfg.source_file.string() << "\n";
    return kExitOk;
  }

  // --- Kokkos ----------------------------------------------------------------
  Kokkos::InitializationSettings settings;
  if (platform) {
    if (platform->backend != vleo::Backend::CPU) settings.set_device_id(platform->device_id);
    if (platform->num_threads > 0) settings.set_num_threads(platform->num_threads);
  }
  Kokkos::ScopeGuard guard(settings);

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

  notes.push_back("milestone M0: configuration and provenance only; no particle simulation was performed");

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

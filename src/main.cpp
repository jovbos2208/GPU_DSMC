// vleo_solver - command-line entry point.
//
// Current state (milestone M2): reads and validates the simulation (and
// optional platform) configuration and the geometry, sets up the domain and
// the free-stream inflow, initialises Kokkos on the compiled backend, runs a
// backend self-test and the TPMC solver (Maxwell or CLL), and writes the
// results with uncertainties plus the resolved configuration and complete run
// metadata. With simulation.mode: dsmc the DSMC solver (single monatomic
// species, VHS collisions) is run instead.
//
// MPI builds (VLEO_ENABLE_MPI, M6.1): start with one rank per GPU, e.g.
// `mpirun -n 2 vleo_solver --config ...`. All ranks run the solver (TPMC:
// histories split by id; DSMC: domain decomposition into cell planes); only
// rank 0 prints progress and writes the output files (docs/parallel.md).

#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

#include "vleo/core/BuildInfo.hpp"
#include "vleo/core/Comm.hpp"
#include "vleo/core/SelfTest.hpp"
#include "vleo/geometry/Geometry.hpp"
#include "vleo/io/ConfigParser.hpp"
#include "vleo/io/ConfigWriter.hpp"
#include "vleo/io/JsonWriter.hpp"
#include "vleo/io/PlatformConfig.hpp"
#include "vleo/io/RunMetadata.hpp"
#include "vleo/io/VtkHdf.hpp"
#include "vleo/particles/Inflow.hpp"
#include "vleo/dsmc/DsmcResults.hpp"
#include "vleo/tpmc/Results.hpp"
#include "vleo/tpmc/Tpmc.hpp"
#include "RunLog.hpp"

namespace fs = std::filesystem;

namespace {

constexpr int kExitOk = 0;
constexpr int kExitRuntimeError = 1;
constexpr int kExitConfigError = 2;
constexpr int kExitUsage = 64;

struct Options {
  std::optional<fs::path> config;
  std::optional<fs::path> batch;   // batch file: several cases in one process
  std::optional<fs::path> platform;
  std::optional<fs::path> output_dir;
  std::uint64_t histories = 0;  // 0: use tpmc.particle_histories
  std::optional<std::uint64_t> sample_steps, warmup_steps;
  std::optional<std::pair<double, double>> attitude_deg;  // (alpha, beta), M6.8
  bool check_only = false;
  bool skip_existing = false;  // batch/single: skip cases whose results.json exists
  bool quiet = false;
  bool profile = false;
  bool print_config = false;
  bool version = false;
  bool help = false;
};

void print_usage(std::ostream& os) {
  os << "Usage: vleo_solver --config <simulation.yaml> [options]\n"
        "       vleo_solver --batch <cases.txt> [options]\n"
        "\n"
        "Options:\n"
        "  -c, --config <file>      Simulation configuration\n"
        "      --batch <file>       Run several cases in one process (Kokkos, self-test and geometries set up once).\n"
        "                           One case per line: <config.yaml> [<output dir>]; '#' comments; paths relative\n"
        "                           to the batch file. Overrides below apply to every case.\n"
        "      --skip-existing      Skip cases whose output directory already contains results.json\n"
        "  -p, --platform <file>    Platform configuration; its backend must match this build\n"
        "  -o, --output-dir <dir>   Override output.directory from the configuration\n"
        "  -n, --histories <N>      Override tpmc.particle_histories (recorded in the resolved config)\n"
        "      --sample-steps <N>   Override dsmc.sample_steps\n"
        "      --warmup-steps <N>   Override dsmc.warmup_steps\n"
        "      --attitude <A> <B>   Flight direction (cos A cos B, sin B, sin A cos B) in the body frame [deg];\n"
        "                           replaces the direction of gas.bulk_velocity_body_mps (speed kept)\n"
        "  -q, --quiet              No progress output\n"
        "      --profile            Record phase timings (fenced) in results.json\n"
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
    } else if (a == "--batch") {
      auto v = need_value("--batch");
      if (!v) return std::nullopt;
      o.batch = *v;
    } else if (a == "--skip-existing") {
      o.skip_existing = true;
    } else if (a == "-p" || a == "--platform") {
      auto v = need_value("--platform");
      if (!v) return std::nullopt;
      o.platform = *v;
    } else if (a == "-o" || a == "--output-dir") {
      auto v = need_value("--output-dir");
      if (!v) return std::nullopt;
      o.output_dir = *v;
    } else if (a == "-n" || a == "--histories") {
      auto v = need_value("--histories");
      if (!v) return std::nullopt;
      std::uint64_t n = 0;
      const auto [p, ec] = std::from_chars(v->data(), v->data() + v->size(), n);
      if (ec != std::errc() || p != v->data() + v->size() || n == 0) {
        std::cerr << "error: --histories expects a positive integer\n";
        return std::nullopt;
      }
      o.histories = n;
    } else if (a == "--sample-steps" || a == "--warmup-steps") {
      auto v = need_value(a.c_str());
      if (!v) return std::nullopt;
      std::uint64_t n = 0;
      const auto [p, ec] = std::from_chars(v->data(), v->data() + v->size(), n);
      if (ec != std::errc() || p != v->data() + v->size()) {
        std::cerr << "error: " << a << " expects a non-negative integer\n";
        return std::nullopt;
      }
      (a == "--sample-steps" ? o.sample_steps : o.warmup_steps) = n;
    } else if (a == "--attitude") {
      if (i + 2 >= argc) {
        std::cerr << "error: --attitude requires two angles (alpha beta, degrees)\n";
        return std::nullopt;
      }
      double ang[2];
      for (double& x : ang) {
        const std::string v = argv[++i];
        std::size_t used = 0;
        try {
          x = std::stod(v, &used);
        } catch (const std::exception&) {
          used = 0;
        }
        if (used != v.size() || !std::isfinite(x)) {
          std::cerr << "error: --attitude expects two numbers (degrees)\n";
          return std::nullopt;
        }
      }
      o.attitude_deg = std::make_pair(ang[0], ang[1]);
    } else if (a == "--profile") {
      o.profile = true;
    } else if (a == "-q" || a == "--quiet") {
      o.quiet = true;
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

// Runtime error on one rank: the other ranks may be waiting in a collective
// operation, so with several ranks the whole job is aborted.
std::streambuf* g_stderr = nullptr;  // real stderr (std::cerr is silenced on ranks > 0)
vleo::RunLog* g_log = nullptr;       // simulation.out (rank 0, once the run has started)

int runtime_failure(const std::string& message) {
  if (g_log) {
    g_log->message("error", message);
    g_log->finish("failed", {});
  }
  std::ostream err(g_stderr ? g_stderr : std::cerr.rdbuf());
  err << "error";
  if (vleo::comm::size() > 1) err << " (rank " << vleo::comm::rank() << ")";
  err << ": " << message << std::endl;
  if (vleo::comm::size() > 1) vleo::comm::abort(kExitRuntimeError);
  return kExitRuntimeError;
}

// One case of a run: a configuration and its output directory. Without
// --batch there is exactly one case (--config / --output-dir).
struct CaseInput {
  fs::path config;
  std::optional<fs::path> output_dir;
};

// Geometries are read and their BVH built once per (file, scale, require_closed)
// and uploaded once; cases of a batch share them.
struct GeometryCache {
  struct Entry {
    std::string key;
    std::unique_ptr<vleo::Geometry> geometry;
    std::vector<std::string> warnings;
  };
  std::vector<Entry> entries;

  // Throws StlError / GeometryError.
  const Entry& get(const vleo::SimulationConfig& cfg) {
    std::ostringstream k;
    k << cfg.geometry.resolved_file.string() << '|' << std::setprecision(17) << cfg.derived.geometry_scale_to_m << '|'
      << cfg.geometry.require_closed;
    for (const Entry& e : entries)
      if (e.key == k.str()) return e;
    vleo::MeshOptions mopt;
    mopt.require_closed = cfg.geometry.require_closed;
    Entry e;
    e.key = k.str();
    e.geometry = std::make_unique<vleo::Geometry>(vleo::make_host_geometry(
        vleo::read_stl(cfg.geometry.resolved_file, cfg.derived.geometry_scale_to_m), mopt));
    for (const auto& w : e.geometry->report.warnings) {
      e.warnings.push_back("geometry: " + w);
      std::cerr << "warning: geometry: " << w << "\n";
    }
    entries.push_back(std::move(e));
    return entries.back();
  }
  void upload() {
    for (Entry& e : entries) vleo::upload_geometry(*e.geometry);
  }
  // Releases device memory before Kokkos::finalize.
  void release_device() {
    for (Entry& e : entries) e.geometry->device = vleo::DeviceGeometry{};
  }
};

struct DeviceGeometryRelease {  // declared after the ScopeGuard
  GeometryCache& cache;
  ~DeviceGeometryRelease() { cache.release_device(); }
};

// A case prepared on the host (configuration, geometry, inflow, DSMC settings).
struct PreparedCase {
  vleo::SimulationConfig cfg;
  const vleo::Geometry* geometry = nullptr;
  vleo::InflowModel inflow;
  vleo::DsmcSettings dsmc_settings;
  std::vector<std::string> notes;
  fs::path out_dir;
};

// Reads a batch file: one case per line, "<config.yaml> [<output dir>]";
// blank lines and lines starting with '#' are ignored. Relative paths are
// resolved against the directory of the batch file.
std::vector<CaseInput> read_batch_file(const fs::path& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot read batch file '" + path.string() + "'");
  const fs::path base = fs::absolute(path).parent_path();
  std::vector<CaseInput> cases;
  std::string line;
  int n = 0;
  while (std::getline(in, line)) {
    ++n;
    std::istringstream ls(line);
    std::string a, b, extra;
    if (!(ls >> a) || a[0] == '#') continue;
    ls >> b >> extra;
    if (!extra.empty())
      throw std::runtime_error(path.string() + ":" + std::to_string(n) + ": expected '<config> [<output dir>]'");
    CaseInput c;
    c.config = fs::path(a).is_absolute() ? fs::path(a) : base / a;
    if (!b.empty()) c.output_dir = fs::path(b).is_absolute() ? fs::path(b) : base / b;
    cases.push_back(std::move(c));
  }
  if (cases.empty()) throw std::runtime_error("batch file '" + path.string() + "' lists no cases");
  return cases;
}

// Host-side preparation of one case. Returns an exit code (kExitOk on success).
int prepare_case(const Options& opt, const CaseInput& in, GeometryCache& cache, PreparedCase& pc,
                 const std::string& ctx) {
  vleo::SimulationConfig& cfg = pc.cfg;
  try {
    cfg = vleo::load_simulation_config(in.config);
    if (opt.attitude_deg) vleo::set_attitude(cfg, opt.attitude_deg->first, opt.attitude_deg->second);
  } catch (const vleo::ConfigError& e) {
    std::cerr << "error: " << ctx << e.what() << "\n";
    return kExitConfigError;
  }
  try {
    const GeometryCache::Entry& g = cache.get(cfg);
    pc.geometry = g.geometry.get();
    pc.notes = g.warnings;
    const vleo::Domain domain =
        vleo::make_domain(cfg.domain, pc.geometry->host.bbox, 1.0e3 * pc.geometry->device.surface_offset_m);
    pc.inflow = vleo::InflowModel::create(domain, cfg.gas);
  } catch (const std::exception& e) {  // StlError, GeometryError
    std::cerr << "error: " << ctx << e.what() << "\n";
    return kExitConfigError;
  }
  if (cfg.output.format == vleo::OutputFormat::HDF5) {
    std::cerr << "error: " << ctx << "output.format: hdf5 is not available in this build (use json or csv)\n";
    return kExitConfigError;
  }
  const bool write_states = cfg.simulation.mode == vleo::SolverMode::DSMC && cfg.dsmc.dt_DSMCState_out.has_value();
  if (write_states && !vleo::vtkhdf_available()) {
    std::cerr << "error: dsmc.dt_DSMCState_out: HDF5 output is not available in this build "
                 "(install the HDF5 C library and configure with -DVLEO_HDF5=ON)\n";
    return kExitConfigError;
  }
  if (opt.histories > 0) cfg.tpmc.particle_histories = opt.histories;
  if (opt.sample_steps) {
    if (*opt.sample_steps < 2 || *opt.sample_steps < cfg.dsmc.blocks) {
      std::cerr << "error: " << ctx << "--sample-steps must be at least 2 and at least dsmc.blocks\n";
      return kExitUsage;
    }
    cfg.dsmc.sample_steps = *opt.sample_steps;
  }
  if (opt.warmup_steps) cfg.dsmc.warmup_steps = *opt.warmup_steps;
  if (cfg.simulation.mode == vleo::SolverMode::DSMC) {
    try {
      pc.dsmc_settings = vleo::resolve_dsmc_settings(cfg, pc.inflow.domain);
    } catch (const std::exception& e) {
      std::cerr << "error: " << ctx << "DSMC setup: " << e.what() << "\n";
      return kExitConfigError;
    }
  }
  pc.out_dir = in.output_dir ? *in.output_dir : (opt.output_dir ? *opt.output_dir : fs::path(cfg.output.directory));
  return kExitOk;
}

// Runs one prepared case on the initialised device and writes its outputs.
int run_case(const Options& opt, PreparedCase& pc, const std::optional<vleo::PlatformConfig>& platform,
             vleo::RuntimeInfo rt, const vleo::SelfTestResult& self_test, bool self_test_ok) {
  const vleo::SimulationConfig& cfg = pc.cfg;
  const vleo::Geometry& geometry = *pc.geometry;
  const vleo::InflowModel& inflow = pc.inflow;
  vleo::DsmcSettings& dsmc_settings = pc.dsmc_settings;
  std::vector<std::string>& notes = pc.notes;
  const fs::path& out_dir = pc.out_dir;
  const int nranks = vleo::comm::size();
  rt.timestamp_utc = vleo::utc_timestamp_now();
  notes.push_back(self_test.detail);

  // --- Run log simulation.out (rank 0), written while the run progresses ------------
  const fs::path f_log = out_dir / "simulation.out";
  std::optional<vleo::RunLog> run_log;
  if (vleo::comm::root()) {
    try {
      fs::create_directories(out_dir);
      run_log.emplace(f_log);
    } catch (const std::exception& e) {
      return runtime_failure(e.what());
    }
    g_log = &*run_log;
    run_log->header(cfg, geometry, inflow, rt, out_dir);
    for (const auto& n : notes) run_log->message("note", n);
  }
  struct LogReset {
    ~LogReset() { g_log = nullptr; }
  } const log_reset;

  // --- Solver ----------------------------------------------------------------
  const bool dsmc = cfg.simulation.mode == vleo::SolverMode::DSMC;
  const bool write_states = dsmc && cfg.dsmc.dt_DSMCState_out.has_value();
  const fs::path state_dir = out_dir / "dsmc_state";
  const fs::path state_series = state_dir / "dsmc_state.vtkhdf.series";
  std::vector<std::pair<std::string, double>> state_files;  // (file name, physical time)
  vleo::TpmcResult result;
  vleo::DsmcResult dresult;
  if (self_test_ok && dsmc) {
    vleo::CellSizePilotReport pilot;
    if (dsmc_settings.cell_size_automatic && cfg.dsmc.cell_size_pilot) {
      if (!opt.quiet)
        std::cerr << "  dsmc: pilot runs for the cell size (free-stream rule: " << dsmc_settings.cell_size_m
                  << " m)\n";
      if (g_log)
        g_log->message("note", "DSMC pilot runs for the cell size (free-stream rule: " +
                                   std::to_string(dsmc_settings.cell_size_m) + " m), started " +
                                   vleo::utc_timestamp_now());
      try {
        dsmc_settings = vleo::refine_cell_size_with_pilot(cfg, geometry, inflow, dsmc_settings, pilot);
      } catch (const std::exception& e) {
        return runtime_failure(std::string("DSMC pilot run failed: ") + e.what());
      }
      if (!opt.quiet)
        std::cerr << "  dsmc: " << pilot.pilots << " pilot run(s), " << pilot.wall_time_s
                  << " s: min local lambda " << pilot.min_local_lambda_m << " m -> cell size " << pilot.cell_size_m
                  << " m" << (pilot.limited_by_max_cells ? " (limited by the maximum cell count)" : "") << "\n";
    }
    if (g_log) g_log->dsmc_parameters(cfg, inflow, dsmc_settings, pilot);
    vleo::DsmcProgressCallback progress;
    if (vleo::comm::root()) {
      // run_dsmc reports every 1 %; the terminal shows every 5 %
      progress = [quiet = opt.quiet, dt = dsmc_settings.timestep_s,
                  shown = std::uint64_t{0}](const vleo::DsmcProgress& p) mutable {
        if (g_log) g_log->progress(p, dt);
        const std::uint64_t twentieth = 20 * p.step / std::max<std::uint64_t>(1, p.total_steps);
        if (quiet || (twentieth == shown && p.step != p.total_steps)) return;
        shown = twentieth;
        std::cerr << "  dsmc: step " << p.step << " / " << p.total_steps << (p.sampling ? " (sampling)" : " (warm-up)")
                  << ", " << p.particles << " particles, " << p.elapsed_s << " s\n";
      };
    }
    if (!opt.quiet) {
      std::cerr << "  dsmc: dt = " << dsmc_settings.timestep_s << " s, cell = " << dsmc_settings.cell_size_m
                << " m, W = " << dsmc_settings.particle_weight << ", lambda_inf = " << dsmc_settings.lambda_inf_m
                << " m, warm-up " << dsmc_settings.warmup_steps << " + sampling " << dsmc_settings.sample_steps
                << " steps\n";
    }
    vleo::DsmcStateCallback on_state;
    if (write_states) {
      if (*cfg.dsmc.dt_DSMCState_out < dsmc_settings.timestep_s) {
        const std::string w = "dsmc.dt_DSMCState_out is shorter than the time step; a cell-state file is written "
                              "every step";
        notes.push_back("dsmc: " + w);
        std::cerr << "warning: dsmc: " << w << "\n";
      }
      on_state = [&](const vleo::DsmcCellState& st) {
        char name[64];
        std::snprintf(name, sizeof(name), "dsmc_state_%06llu.vtkhdf", static_cast<unsigned long long>(st.index));
        fs::create_directories(state_dir);
        vleo::write_dsmc_cell_state(state_dir / name, st);
        state_files.emplace_back(name, st.time_s);
        // ParaView file series (physical times); rewritten after every file so
        // that a running or aborted simulation can be inspected.
        vleo::JsonWriter w;
        w.begin_object();
        w.key("file-series-version").value("1.0");
        w.key("files").begin_array();
        for (const auto& [f, t] : state_files) {
          w.begin_object();
          w.key("name").value(f);
          w.key("time").value(t);
          w.end_object();
        }
        w.end_array();
        w.end_object();
        write_text_file(state_series, w.str());
        if (g_log)
          g_log->message("note", std::string("cell state ") + name + " written (t = " + std::to_string(st.time_s) +
                                     " s, " + std::to_string(st.steps) + " steps averaged)");
        if (!opt.quiet)
          std::cerr << "  dsmc: cell state " << name << " (t = " << st.time_s << " s, averaged over " << st.steps
                    << " steps)\n";
      };
    }
    try {
      dsmc_settings.profile = opt.profile;
      dresult = vleo::run_dsmc(cfg, geometry, inflow, dsmc_settings, progress, on_state);
      dresult.pilot = pilot;
      if (pilot.limited_by_max_cells)
        dresult.warnings.push_back("automatic cell size limited by the maximum cell count; the grid does not resolve "
                                   "the local mean free path");
    } catch (const std::exception& e) {
      return runtime_failure(std::string("DSMC run failed: ") + e.what());
    }
    for (const auto& w : dresult.warnings) {
      notes.push_back("dsmc: " + w);
      std::cerr << "warning: dsmc: " << w << "\n";
      if (g_log) g_log->message("warning", "dsmc: " + w);
    }
    if (g_log) g_log->dsmc_summary(dresult);
  }
  if (self_test_ok && !dsmc) {
    if (g_log) g_log->tpmc_parameters(cfg, inflow);
    vleo::TpmcProgressCallback progress;
    if (vleo::comm::root()) {
      const std::string monitored =
          cfg.tpmc.convergence ? std::string(vleo::to_string(cfg.tpmc.convergence->quantity)) : "C_D";
      progress = [quiet = opt.quiet, monitored](const vleo::TpmcProgress& p) {
        if (g_log) g_log->progress(p, monitored);
        if (!quiet)
          std::cerr << "  tpmc: " << p.histories << " / " << p.target << " histories, " << p.elapsed_s << " s, "
                    << "monitored = " << p.monitored.value << " +- " << p.monitored.std_error << "\n";
      };
    }
    try {
      result = vleo::run_tpmc(cfg, geometry, inflow, 0, progress, opt.profile);
    } catch (const std::exception& e) {
      return runtime_failure(std::string("TPMC run failed: ") + e.what());
    }
    for (const auto& w : result.warnings) {
      notes.push_back("tpmc: " + w);
      std::cerr << "warning: tpmc: " << w << "\n";
      if (g_log) g_log->message("warning", "tpmc: " + w);
    }
    if (g_log) g_log->tpmc_summary(result);
  }

  // --- Outputs (rank 0 only) ------------------------------------------------------
  if (!vleo::comm::root()) return self_test_ok ? kExitOk : kExitRuntimeError;
  try {
    const fs::path f_resolved = out_dir / "resolved_config.yaml";
    const fs::path f_input = out_dir / "input_config.yaml";
    const fs::path f_kokkos = out_dir / "kokkos_configuration.txt";
    const fs::path f_meta = out_dir / "run_metadata.json";
    const fs::path f_results = out_dir / "results.json";
    const fs::path f_csv = out_dir / "results.csv";
    const fs::path f_surface = out_dir / "surface_fields.csv";
    std::vector<std::string> files = {f_meta.filename().string(), f_resolved.filename().string(),
                                      f_input.filename().string(), f_kokkos.filename().string(),
                                      f_log.filename().string()};
    const vleo::TpmcEstimates& est = dsmc ? dresult.estimates : result.estimates;
    const std::optional<vleo::SurfaceFields>& surface = dsmc ? dresult.surface : result.surface;
    write_text_file(f_resolved, vleo::to_resolved_yaml(cfg));
    write_text_file(f_input, read_text_file(cfg.source_file));
    std::ostringstream kc;
    Kokkos::print_configuration(kc, true);
    write_text_file(f_kokkos, kc.str());
    if (self_test_ok) {
      if (cfg.output.format == vleo::OutputFormat::CSV) {
        write_text_file(f_csv, vleo::estimates_csv(est));
        files.push_back(f_csv.filename().string());
      }
      if (surface) {
        write_text_file(f_surface, vleo::surface_fields_csv(*surface, geometry));
        files.push_back(f_surface.filename().string());
      }
      files.push_back(f_results.filename().string());
    }
    if (!state_files.empty()) {
      files.push_back((fs::path("dsmc_state") / state_series.filename()).string());
      for (const auto& [f, t] : state_files) files.push_back((fs::path("dsmc_state") / f).string());
    }

    vleo::RunMetadataInput meta;
    meta.config = &cfg;
    meta.platform = platform ? &*platform : nullptr;
    meta.runtime = rt;
    const bool clean = dsmc ? dresult.counters.lost_inside_body + dresult.counters.invalid == 0
                            : result.counters.lost_inside_body + result.counters.invalid == 0;
    meta.status = !self_test_ok ? "self_test_failed" : (clean ? "completed" : "completed_with_errors");
    meta.notes = notes;
    meta.sections.emplace_back("geometry", [&](vleo::JsonWriter& w) { write_geometry_json(w, geometry, cfg); });
    meta.sections.emplace_back("inflow", [&](vleo::JsonWriter& w) { write_inflow_json(w, inflow, cfg); });
    meta.output_files = files;
    write_text_file(f_meta, vleo::make_run_metadata_json(meta));
    // results.json last: its presence marks a finished case (--skip-existing)
    if (self_test_ok)
      write_text_file(f_results, dsmc ? vleo::dsmc_results_json(dresult, cfg, rt.timestamp_utc)
                                      : vleo::results_json(result, cfg, rt.timestamp_utc));
    if (!self_test_ok) run_log->message("error", self_test.detail);
    run_log->finish(meta.status, files);

    std::cerr << "vleo_solver: backend " << vleo::to_string(vleo::compiled_backend()) << " ("
              << rt.execution_space << (nranks > 1 ? ", " + std::to_string(nranks) + " MPI ranks" : std::string())
              << "), "
              << (dsmc ? std::to_string(dresult.counters.steps) + " steps in " + std::to_string(dresult.wall_time_s)
                       : std::to_string(result.histories) + " histories in " + std::to_string(result.wall_time_s))
              << " s, wrote " << fs::absolute(out_dir).lexically_normal().string() << "\n";
    if (self_test_ok && !opt.quiet) {
      const auto& e = est;
      std::cerr << "  C_D = " << e.C_D.value << " +- " << e.C_D.std_error << ", C_L = " << e.C_L.value << " +- "
                << e.C_L.std_error << ", Q = " << e.heat_rate_W.value << " +- " << e.heat_rate_W.std_error
                << " W\n";
    }
  } catch (const std::exception& e) {
    run_log->message("error", e.what());
    run_log->finish("failed", {});
    std::cerr << "error: " << e.what() << "\n";
    return kExitRuntimeError;
  }
  return self_test_ok ? kExitOk : kExitRuntimeError;
}

int run(const Options& opt, int argc, char** argv) {
  // --- Configuration (before touching any device) ---------------------------
  std::optional<vleo::PlatformConfig> platform;
  std::vector<CaseInput> inputs;
  try {
    if (opt.platform) {
      platform = vleo::load_platform_config(*opt.platform);
      vleo::check_platform_matches_build(*platform);
    }
  } catch (const vleo::ConfigError& e) {
    std::cerr << "error: " << e.what() << "\n";
    return kExitConfigError;
  }
  const bool batch = opt.batch.has_value();
  if (batch) {
    try {
      inputs = read_batch_file(*opt.batch);
    } catch (const std::exception& e) {
      std::cerr << "error: " << e.what() << "\n";
      return kExitConfigError;
    }
  } else {
    inputs.push_back({*opt.config, opt.output_dir});
  }

  // --- Geometry, domain and inflow of every case (host side) ------------------
  GeometryCache cache;
  std::vector<PreparedCase> cases(inputs.size());
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    const std::string ctx = batch ? inputs[i].config.string() + ": " : std::string();
    const int rc = prepare_case(opt, inputs[i], cache, cases[i], ctx);
    if (rc != kExitOk) return rc;
    if (opt.print_config) std::cout << vleo::to_resolved_yaml(cases[i].cfg);
  }
  if (opt.check_only) {
    for (const PreparedCase& pc : cases)
      std::cerr << "configuration valid: " << pc.cfg.source_file.string() << " (" << pc.geometry->report.triangles
                << " triangles, " << pc.geometry->report.components.size() << " component(s))\n";
    return kExitOk;
  }
  std::vector<bool> skip(cases.size(), false);
  std::size_t to_run = cases.size();
  if (opt.skip_existing)
    for (std::size_t i = 0; i < cases.size(); ++i)
      if (fs::exists(cases[i].out_dir / "results.json")) {
        skip[i] = true;
        --to_run;
      }
  if (batch)
    std::cerr << "vleo_solver: batch of " << cases.size() << " case(s), " << cache.entries.size() << " geometr"
              << (cache.entries.size() == 1 ? "y" : "ies") << ", " << cases.size() - to_run
              << " already finished\n";
  if (to_run == 0) return kExitOk;

  // --- Kokkos ----------------------------------------------------------------
  Kokkos::InitializationSettings settings;
  const int nranks = vleo::comm::size();
  if (platform) {
    // Several ranks: Kokkos assigns GPU (local rank mod visible devices) from
    // the launcher's local-rank variable; the platform's device_id is ignored.
    if (platform->backend != vleo::Backend::CPU && nranks == 1) settings.set_device_id(platform->device_id);
    if (platform->num_threads > 0) settings.set_num_threads(platform->num_threads);
  }
  if (nranks > 1 && vleo::compiled_backend() != vleo::Backend::CPU) settings.set_map_device_id_by("mpi_rank");
  Kokkos::ScopeGuard guard(settings);
  cache.upload();
  const DeviceGeometryRelease release_device_geometry{cache};

  vleo::RuntimeInfo rt;
  for (int i = 0; i < argc; ++i) rt.command_line.emplace_back(argv[i]);
  rt.working_directory = fs::current_path().string();
  rt.execution_space = Kokkos::DefaultExecutionSpace::name();
  rt.concurrency = Kokkos::DefaultExecutionSpace().concurrency();
  rt.device_id = vleo::compiled_backend() == vleo::Backend::CPU ? -1 : Kokkos::device_id();
  rt.mpi_ranks = nranks;
  rt.mpi_library = vleo::comm::library_version();
  rt.rank_device_ids = vleo::comm::allgather(rt.device_id);
  std::vector<std::string> run_notes;
  if (nranks > 1 && vleo::compiled_backend() != vleo::Backend::CPU) {
    for (int a = 0; a < nranks; ++a)
      for (int b = 0; b < a; ++b)
        if (rt.rank_device_ids[static_cast<std::size_t>(a)] == rt.rank_device_ids[static_cast<std::size_t>(b)]) {
          const std::string w = "several MPI ranks share one GPU (device " +
                                std::to_string(rt.rank_device_ids[static_cast<std::size_t>(a)]) +
                                "); start one rank per GPU";
          run_notes.push_back(w);
          if (vleo::comm::root()) std::cerr << "warning: " << w << "\n";
          a = nranks;
          break;
        }
  }

  const vleo::SelfTestResult self_test = vleo::backend_self_test(cases.front().cfg.simulation.random_seed);
  const bool self_test_ok = vleo::comm::all(self_test.passed);
  if (!self_test_ok) std::cerr << "error: " << self_test.detail << "\n";

  // --- Cases -------------------------------------------------------------------
  int rc = kExitOk;
  std::size_t done = 0, failed = 0;
  for (std::size_t i = 0; i < cases.size(); ++i) {
    if (skip[i]) continue;
    PreparedCase& pc = cases[i];
    pc.notes.insert(pc.notes.end(), run_notes.begin(), run_notes.end());
    if (batch && !opt.quiet)
      std::cerr << "[" << done + failed + 1 << "/" << to_run << "] " << pc.cfg.source_file.string() << "\n";
    const int case_rc = run_case(opt, pc, platform, rt, self_test, self_test_ok);
    if (case_rc == kExitOk) {
      ++done;
    } else {
      ++failed;
      rc = case_rc;
      if (!self_test_ok) break;  // every case would fail
    }
  }
  if (batch)
    std::cerr << "vleo_solver: batch finished: " << done << " completed, " << failed << " failed, "
              << cases.size() - to_run << " skipped\n";
  return rc;
}

}  // namespace

int main(int argc, char** argv) {
  vleo::comm::Session mpi(&argc, &argv);
  auto opt = parse_args(argc, argv);
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
  if (opt->config && opt->batch) {
    std::cerr << "error: use either --config or --batch\n";
    return kExitUsage;
  }
  if (opt->batch && opt->output_dir) {
    std::cerr << "error: --output-dir cannot be combined with --batch (give the directory per case)\n";
    return kExitUsage;
  }
  if (!opt->config && !opt->batch) {
    std::cerr << "error: --config or --batch is required\n";
    print_usage(std::cerr);
    return kExitUsage;
  }
  // Configuration errors are detected identically on every rank; only rank 0
  // reports them and the progress (the other ranks print runtime errors only).
  std::ofstream null_stream;
  std::streambuf* cerr_buf = std::cerr.rdbuf();
  g_stderr = cerr_buf;
  if (!vleo::comm::root()) {
    opt->quiet = true;
    std::cerr.rdbuf(null_stream.rdbuf());
  }
  int rc = kExitRuntimeError;
  try {
    rc = run(*opt, argc, argv);
  } catch (const std::exception& e) {
    std::cerr.rdbuf(cerr_buf);
    rc = runtime_failure(e.what());
  }
  std::cerr.rdbuf(cerr_buf);
  return rc;
}

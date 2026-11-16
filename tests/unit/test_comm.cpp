// MPI layer (M6.1) and multi-rank DSMC building blocks (M6.3, M6.5). Run with
// any number of ranks (ctest: 1 and 3 in MPI builds).
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <vector>

#include "vleo/core/Comm.hpp"
#include "vleo/dsmc/Dsmc.hpp"
#include "vleo/geometry/Primitives.hpp"
#include "vleo/geometry/Stl.hpp"
#include "vleo/io/ConfigParser.hpp"

using namespace vleo;
using Catch::Matchers::WithinRel;
namespace fs = std::filesystem;

TEST_CASE("comm: rank queries are consistent") {
  REQUIRE(comm::size() >= 1);
  REQUIRE(comm::rank() >= 0);
  REQUIRE(comm::rank() < comm::size());
  REQUIRE(comm::local_rank() >= 0);
  REQUIRE(comm::root() == (comm::rank() == 0));
  REQUIRE(comm::enabled() == !comm::library_version().empty());
  if (!comm::enabled()) REQUIRE(comm::size() == 1);
}

TEST_CASE("comm: reductions") {
  const int p = comm::size(), r = comm::rank();
  std::vector<double> d = {1.0, static_cast<double>(r), 0.5};
  comm::sum(d.data(), d.size());
  REQUIRE(d[0] == p);
  REQUIRE(d[1] == p * (p - 1) / 2.0);
  REQUIRE(d[2] == 0.5 * p);
  std::uint64_t u[2] = {std::uint64_t{1} << 40, static_cast<std::uint64_t>(r)};
  comm::sum(u, 2);
  REQUIRE(u[0] == (std::uint64_t{1} << 40) * static_cast<std::uint64_t>(p));
  REQUIRE(u[1] == static_cast<std::uint64_t>(p * (p - 1) / 2));
  REQUIRE(comm::max(static_cast<double>(r)) == p - 1);
  REQUIRE(comm::min(static_cast<double>(r)) == 0.0);
  REQUIRE(comm::any(r == p - 1));
  REQUIRE_FALSE(comm::any(false));
  REQUIRE(comm::all(true));
  REQUIRE(comm::all(r >= 0));
}

TEST_CASE("comm: allgather, broadcast and alltoallv") {
  const int p = comm::size(), r = comm::rank();
  const std::vector<int> all = comm::allgather(10 * r + 1);
  REQUIRE(static_cast<int>(all.size()) == p);
  for (int k = 0; k < p; ++k) REQUIRE(all[static_cast<std::size_t>(k)] == 10 * k + 1);

  double v[2] = {r == 0 ? 3.25 : -1.0, r == 0 ? 7.0 : -1.0};
  comm::broadcast(v, sizeof(v));
  REQUIRE(v[0] == 3.25);
  REQUIRE(v[1] == 7.0);

  // rank r sends (r + 1) * k + 1 ints with value 1000 r + k to rank k (empty blocks included)
  std::vector<std::vector<char>> send(static_cast<std::size_t>(p));
  for (int k = 0; k < p; ++k) {
    const std::size_t n = static_cast<std::size_t>(((r + 1) * k) % 3);
    std::vector<int> block(n, 1000 * r + k);
    send[static_cast<std::size_t>(k)].resize(n * sizeof(int));
    if (n > 0) std::memcpy(send[static_cast<std::size_t>(k)].data(), block.data(), n * sizeof(int));
  }
  const auto recv = comm::alltoallv(send);
  REQUIRE(static_cast<int>(recv.size()) == p);
  for (int k = 0; k < p; ++k) {
    const std::size_t n = static_cast<std::size_t>(((k + 1) * r) % 3);
    REQUIRE(recv[static_cast<std::size_t>(k)].size() == n * sizeof(int));
    for (std::size_t i = 0; i < n; ++i) {
      int x;
      std::memcpy(&x, recv[static_cast<std::size_t>(k)].data() + i * sizeof(int), sizeof(int));
      REQUIRE(x == 1000 * k + r);
    }
  }
}

TEST_CASE("DSMC: load balancing and migration do not change the results", "[dsmc][mpi]") {
  // Body off-centre in z, so the initial split by gas volume is unbalanced in
  // particles and the planes move at the end of the warm-up.
  const TriangleSoup soup = primitives::box({-0.3, -0.3, -0.5}, {0.3, 0.3, 0.1});
  const fs::path dir = fs::temp_directory_path() / ("vleo_test_comm_" + std::to_string(comm::rank()));
  fs::create_directories(dir);
  const fs::path stl = dir / "body.stl";
  write_stl_ascii(stl, soup);
  const std::string body =
      "geometry: {file: " + stl.string() + "}\n"
      "simulation: {name: lb, mode: dsmc, random_seed: 3}\n"
      "domain: {margin_m: 0.3}\n"
      "reference: {area_m2: 1, length_m: 1, moment_center_m: [0,0,0], lift_axis_body: [0,0,1]}\n"
      "gas: {number_density_m3: 1.0e20, translational_temperature_K: 300, "
      "bulk_velocity_body_mps: [-2000, 0, -1500], species: {Ar: 0.7, N2: 0.3}}\n"
      "surface: {model: cll, temperature_K: 300, cll: {alpha_n: 0.7, sigma_t: 0.6}}\n"
      "dsmc: {cell_size_mode: fixed, cell_size_m: 0.075, particles_per_cell_target: 8, warmup_steps: 15, "
      "sample_steps: 20, blocks: 4}\n"
      "output: {save_surface_fields: true}\n";
  const SimulationConfig cfg = parse_simulation_config(body, dir);
  const Geometry geometry = make_geometry(soup);
  const Domain dom = make_domain(cfg.domain, geometry.host.bbox, 1e3 * geometry.device.surface_offset_m);
  const InflowModel inflow = InflowModel::create(dom, cfg.gas);
  DsmcSettings s = resolve_dsmc_settings(cfg, inflow.domain);
  s.load_balance = false;
  const DsmcResult a = run_dsmc(cfg, geometry, inflow, s);
  s.load_balance = true;
  const DsmcResult b = run_dsmc(cfg, geometry, inflow, s);

  REQUIRE(a.decomposition.ranks == comm::size());
  REQUIRE(b.decomposition.rebalanced == (comm::size() > 1));
  CHECK(a.counters.collisions > 1000);
  CHECK(a.counters.collisions == b.counters.collisions);
  CHECK(a.counters.collision_candidates == b.counters.collision_candidates);
  CHECK(a.counters.sigma_g_max_updates == b.counters.sigma_g_max_updates);
  CHECK(a.counters.injected == b.counters.injected);
  CHECK(a.counters.exited == b.counters.exited);
  CHECK(a.counters.surface_interactions == b.counters.surface_interactions);
  CHECK(a.species[0].collisions_with == b.species[0].collisions_with);
  CHECK(a.diagnostics.particles_mean == b.diagnostics.particles_mean);
  CHECK(a.diagnostics.min_local_lambda_m == b.diagnostics.min_local_lambda_m);
  CHECK_THAT(b.estimates.C_D.value, WithinRel(a.estimates.C_D.value, 1e-11));
  CHECK_THAT(b.estimates.heat_rate_W.value, WithinRel(a.estimates.heat_rate_W.value, 1e-11));
  if (comm::size() > 1) {
    CHECK(b.decomposition.imbalance_after <= b.decomposition.imbalance_before);
    CHECK(b.decomposition.migrated > 0);
  }
  // all ranks hold the same (global) result
  const double cd = b.estimates.C_D.value;
  CHECK(comm::max(cd) == comm::min(cd));
}

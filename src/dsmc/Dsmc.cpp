#include "vleo/dsmc/Dsmc.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <sstream>
#include <stdexcept>

#include "vleo/core/PhysicalConstants.hpp"

namespace vleo {

namespace {

using policy = Kokkos::RangePolicy<Kokkos::IndexType<std::size_t>>;
using IntPolicy = Kokkos::RangePolicy<>;

// Per-particle step tally: the 9 TPMC history tallies plus the number of
// surface interactions in the step.
constexpr int kStepInteractions = kHistoryTallies;
constexpr int kStepTallies = kHistoryTallies + 1;

constexpr int kMaxChannels = 64;
constexpr int kMaxDsmcSpecies = 8;

template <class T>
Kokkos::View<const T*> to_device(const std::vector<T>& v, const char* label) {
  Kokkos::View<T*> d(std::string(label), v.size());
  auto h = Kokkos::create_mirror_view(d);
  for (std::size_t i = 0; i < v.size(); ++i) h(i) = v[i];
  Kokkos::deep_copy(d, h);
  return d;
}

// Time until origin + t v leaves the box and the axis of the face reached.
KOKKOS_INLINE_FUNCTION double box_exit_time(const Aabb& box, const Vec3& o, const Vec3& v, int& axis) {
  double t = Kokkos::infinity_v<double>;
  axis = -1;
  for (int k = 0; k < 3; ++k) {
    double tk = Kokkos::infinity_v<double>;
    if (v[k] > 0.0)
      tk = (box.hi[k] - o[k]) / v[k];
    else if (v[k] < 0.0)
      tk = (box.lo[k] - o[k]) / v[k];
    if (tk < t) {
      t = tk;
      axis = k;
    }
  }
  return Kokkos::fmax(t, 0.0);
}

struct MoveKernel {
  ParticleStorage p;
  Kokkos::View<double*> dt_rem;
  Kokkos::View<double**, Kokkos::LayoutLeft> tally;
  Kokkos::View<double**, Kokkos::LayoutLeft> surface;  // empty unless sampling with surface fields
  DeviceGeometry geo;
  Aabb box;
  GsiInteraction gsi;
  Kokkos::View<const double*> mass;
  Kokkos::View<const SpeciesModes*> modes;
  Vec3 r0;
  std::uint64_t seed;
  double dt;
  std::uint32_t max_interactions;
  bool specular_box;

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t i) const {
    for (int k = 0; k < kStepTallies; ++k) tally(i, k) = 0.0;
    if (p.status(i) != static_cast<std::uint8_t>(ParticleStatus::Active)) return;
    Vec3 pos = p.position(i);
    Vec3 vel = p.velocity(i);
    const std::uint64_t id = p.particle_id(i);
    const std::uint32_t sp = p.species(i);
    const double m = mass(sp);
    const SpeciesModes sm = modes(sp);
    double e_rot = p.e_rot(i), e_vib = p.e_vib(i);
    std::uint32_t k = p.interactions(i);
    double t = dt_rem(i);
    int ignore = -1;
    std::uint32_t in_step = 0;
    ParticleStatus status = ParticleStatus::Active;
    Vec3 dp_sum, dl_sum;
    double de_sum = 0.0, de_pos = 0.0, de_neg = 0.0;
    const bool surf = surface.extent(0) > 0;

    while (t > 0.0 && status == ParticleStatus::Active) {
      if (!is_finite(pos) || !is_finite(vel)) {
        status = ParticleStatus::Invalid;
        break;
      }
      if (norm2(vel) == 0.0) break;  // at rest: stays in place for this step
      int axis = -1;
      const double t_exit = box_exit_time(box, pos, vel, axis);
      const double t_lim = Kokkos::fmin(t, t_exit);
      const RayHit hit = geo.closest_hit(pos, vel, t_lim, ignore);
      if (hit.valid()) {
        pos = pos + hit.t * vel;
        t -= hit.t;
        const Vec3 n = geo.mesh.normal(hit.triangle);
        const bool front = dot(vel, n) < 0.0;
        if (!front && geo.mesh.two_sided(hit.triangle) == 0) {
          status = ParticleStatus::LostInsideBody;
          break;
        }
        if (in_step >= max_interactions) {
          status = ParticleStatus::Truncated;
          break;
        }
        const Vec3 n_gas = front ? n : -n;
        const SurfaceEvent ev{pos, vel, n_gas, hit.triangle, sp, m, id, k, e_rot, e_vib, sm};
        RandomStream rng(seed, id, k + 1u, RngStream::SurfaceInteraction);
        const Vec3 v_in = vel;
        const double e_int_in = e_rot + e_vib;
        const InteractionResult res = gsi(ev, rng);
        vel = res.velocity;
        e_rot = res.e_rot;
        e_vib = res.e_vib;
        ++k;
        ++in_step;
        if (!(dot(vel, n_gas) > 0.0) || !is_finite(vel)) {
          status = ParticleStatus::Invalid;
          break;
        }
        const Vec3 vw = gsi.wall_velocity(hit.triangle);
        const Vec3 dp = m * (v_in - vel);
        const double de = 0.5 * m * (norm2(v_in - vw) - norm2(vel - vw)) + (e_int_in - e_rot - e_vib);
        dp_sum += dp;
        dl_sum += cross(pos - r0, dp);
        de_sum += de;
        (de >= 0.0 ? de_pos : de_neg) += de;
        if (surf) {
          const int tr = hit.triangle;
          Kokkos::atomic_add(&surface(tr, kSurfPx), dp.x);
          Kokkos::atomic_add(&surface(tr, kSurfPy), dp.y);
          Kokkos::atomic_add(&surface(tr, kSurfPz), dp.z);
          Kokkos::atomic_add(&surface(tr, kSurfE), de);
          Kokkos::atomic_add(&surface(tr, de >= 0.0 ? kSurfEPos : kSurfENeg), de);
          Kokkos::atomic_add(&surface(tr, kSurfHits), 1.0);
        }
        pos = pos + geo.surface_offset_m * n_gas;
        ignore = hit.triangle;
        continue;
      }
      if (t_exit <= t) {
        pos = pos + t_exit * vel;
        t -= t_exit;
        if (!specular_box) {
          status = ParticleStatus::Exited;
          break;
        }
        if (in_step >= max_interactions) {
          status = ParticleStatus::Truncated;
          break;
        }
        ++in_step;
        vel[axis] = -vel[axis];
        pos[axis] = vel[axis] > 0.0 ? box.lo[axis] : box.hi[axis];
        ignore = -1;
        continue;
      }
      pos = pos + t * vel;
      t = 0.0;
    }
    p.set_position(i, pos);
    p.set_velocity(i, vel);
    p.interactions(i) = k;
    p.e_rot(i) = e_rot;
    p.e_vib(i) = e_vib;
    p.status(i) = static_cast<std::uint8_t>(status);
    dt_rem(i) = dt;
    tally(i, kTallyPx) = dp_sum.x;
    tally(i, kTallyPy) = dp_sum.y;
    tally(i, kTallyPz) = dp_sum.z;
    tally(i, kTallyLx) = dl_sum.x;
    tally(i, kTallyLy) = dl_sum.y;
    tally(i, kTallyLz) = dl_sum.z;
    tally(i, kTallyE) = de_sum;
    tally(i, kTallyEPos) = de_pos;
    tally(i, kTallyENeg) = de_neg;
    tally(i, kStepInteractions) = static_cast<double>(in_step);
  }
};

// Sums of the step tallies and the final status counts.
// Per-species slots after the global ones: particle count, momentum (3), energy.
constexpr int kSpeciesSlots = 5;

struct StepReduce {
  using value_type = double[];
  using size_type = std::size_t;
  static constexpr int kGlobal = kStepTallies + kNumParticleStatus;
  static constexpr int kSlots = kGlobal + kSpeciesSlots * kMaxDsmcSpecies;  // upper bound (host array)
  // Sized for the actual number of species: keeps the shared-memory footprint
  // of the reduction small (55 doubles per thread exceeded the LDS budget of
  // small GPUs).
  size_type value_count = kSlots;
  Kokkos::View<double**, Kokkos::LayoutLeft> tally;
  Kokkos::View<const std::uint8_t*> status;
  Kokkos::View<const std::uint32_t*> species;
  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t i, value_type s) const {
    for (int k = 0; k < kStepTallies; ++k) s[k] += tally(i, k);
    const int st = status(i) < kNumParticleStatus ? status(i) : static_cast<int>(ParticleStatus::Invalid);
    s[kStepTallies + st] += 1.0;
    const int sp = static_cast<int>(species(i));
    double* q = s + kGlobal + kSpeciesSlots * sp;
    q[0] += 1.0;
    q[1] += tally(i, kTallyPx);
    q[2] += tally(i, kTallyPy);
    q[3] += tally(i, kTallyPz);
    q[4] += tally(i, kTallyE);
  }
  KOKKOS_INLINE_FUNCTION void init(value_type s) const {
    for (std::size_t k = 0; k < value_count; ++k) s[k] = 0.0;
  }
  KOKKOS_INLINE_FUNCTION void join(value_type d, const value_type s) const {
    for (std::size_t k = 0; k < value_count; ++k) d[k] += s[k];
  }
};

struct InjectKernel {
  ParticleStorage p;
  Kokkos::View<double*> dt_rem;
  DeviceInflow inflow;
  Kokkos::Array<std::uint64_t, kMaxChannels + 1> offsets;  // prefix of new particles per channel
  int channels;
  std::size_t base;
  std::uint64_t first_id;
  std::uint64_t seed;
  double dt;
  double weight;
  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t j) const {
    int ch = 0;
    while (ch + 1 < channels && j >= offsets[ch + 1]) ++ch;
    const std::size_t i = base + j;
    const std::uint64_t id = first_id + j;
    RandomStream rng(seed, id, 0u, RngStream::Injection);
    const InflowSample s = inflow.sample_channel(ch, rng);
    p.set_position(i, s.position);
    p.set_velocity(i, s.velocity);
    p.species(i) = s.species;
    p.particle_id(i) = id;
    p.weight(i) = weight;
    p.interactions(i) = 0u;
    p.e_rot(i) = s.e_rot;
    p.e_vib(i) = s.e_vib;
    p.status(i) = static_cast<std::uint8_t>(ParticleStatus::Active);
    dt_rem(i) = rng.uniform() * dt;  // enters at a random time within the step
  }
};

struct FillKernel {
  ParticleStorage p;
  Kokkos::View<double*> dt_rem;
  DeviceGeometry geo;
  Aabb box;
  Vec3 u;
  Kokkos::View<const double*> species_cdf;
  Kokkos::View<const double*> sigma;  // sqrt(kT/m) per species
  Kokkos::View<const SpeciesModes*> modes;
  double T;
  std::uint64_t first_id;
  std::uint64_t seed;
  double dt, weight;
  bool has_closed;
  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t i) const {
    const std::uint64_t id = first_id + i;
    RandomStream rng(seed, id, 0u, RngStream::Injection);
    const double x = rng.uniform();
    std::uint32_t sp = 0;
    while (sp + 1 < species_cdf.extent(0) && x >= species_cdf(sp)) ++sp;
    const Vec3 pos(box.lo.x + (box.hi.x - box.lo.x) * rng.uniform(), box.lo.y + (box.hi.y - box.lo.y) * rng.uniform(),
                   box.lo.z + (box.hi.z - box.lo.z) * rng.uniform());
    const double s = sigma(sp);
    const Vec3 vel = u + s * Vec3(rng.normal(), rng.normal(), rng.normal());
    p.set_position(i, pos);
    p.set_velocity(i, vel);
    p.species(i) = sp;
    p.particle_id(i) = id;
    p.weight(i) = weight;
    p.interactions(i) = 0u;
    p.e_rot(i) = sample_rotational_energy(modes(sp).rot_dof, T, rng);
    p.e_vib(i) = sample_vibrational_energy(modes(sp).vib, modes(sp).theta_v, T, rng);
    dt_rem(i) = dt;
    const bool inside = has_closed && inside_body(geo, pos);
    p.status(i) = static_cast<std::uint8_t>(inside ? ParticleStatus::Free : ParticleStatus::Active);
  }
};

struct AliveFlag {
  using value_type = std::size_t;
  Kokkos::View<const std::uint8_t*> status;
  Kokkos::View<std::size_t*> index;
  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t i, std::size_t& update, const bool final) const {
    const bool alive = status(i) == static_cast<std::uint8_t>(ParticleStatus::Active);
    if (final) index(i) = update;
    if (alive) ++update;
  }
};

struct CompactCopy {
  ParticleStorage src, dst;
  Kokkos::View<const double*> dt_src;
  Kokkos::View<double*> dt_dst;
  Kokkos::View<const std::size_t*> index;
  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t i) const {
    if (src.status(i) != static_cast<std::uint8_t>(ParticleStatus::Active)) return;
    const std::size_t j = index(i);
    dst.x(j) = src.x(i);
    dst.y(j) = src.y(i);
    dst.z(j) = src.z(i);
    dst.vx(j) = src.vx(i);
    dst.vy(j) = src.vy(i);
    dst.vz(j) = src.vz(i);
    dst.weight(j) = src.weight(i);
    dst.species(j) = src.species(i);
    dst.particle_id(j) = src.particle_id(i);
    dst.status(j) = src.status(i);
    dst.interactions(j) = src.interactions(i);
    dst.e_rot(j) = src.e_rot(i);
    dst.e_vib(j) = src.e_vib(i);
    dt_dst(j) = dt_src(i);
  }
};

struct CellIndexKernel {
  ParticleStorage p;
  DeviceGrid grid;
  Kokkos::View<int*> cell;
  Kokkos::View<int*> count;
  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t i) const {
    const int c = grid.cell_of(p.position(i));
    cell(i) = c;
    Kokkos::atomic_inc(&count(c));
  }
};

struct OffsetScan {
  using value_type = int;
  Kokkos::View<const int*> count;
  Kokkos::View<int*> offset;  // size n_cells + 1
  int n;
  KOKKOS_INLINE_FUNCTION void operator()(const int c, int& update, const bool final) const {
    if (final) offset(c) = update;
    update += count(c);
    if (final && c == n - 1) offset(n) = update;
  }
};

struct FillCells {
  Kokkos::View<const int*> cell;
  Kokkos::View<const int*> offset;
  Kokkos::View<int*> cursor;
  Kokkos::View<int*> list;
  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t i) const {
    const int c = cell(i);
    const int slot = Kokkos::atomic_fetch_add(&cursor(c), 1);
    list(offset(c) + slot) = static_cast<int>(i);
  }
};

// Insertion sort of each cell by particle id: the order within a cell no
// longer depends on the (non-deterministic) atomic fill order.
struct SortCells {
  Kokkos::View<const int*> offset;
  Kokkos::View<int*> list;
  Kokkos::View<const std::uint64_t*> id;
  KOKKOS_INLINE_FUNCTION void operator()(const int c) const {
    const int b = offset(c), e = offset(c + 1);
    for (int a = b + 1; a < e; ++a) {
      const int v = list(a);
      const std::uint64_t key = id(v);
      int k = a - 1;
      while (k >= b && id(list(k)) > key) {
        list(k + 1) = list(k);
        --k;
      }
      list(k + 1) = v;
    }
  }
};

struct CollideKernel {
  ParticleStorage p;
  Kokkos::View<const int*> offset;
  Kokkos::View<const int*> list;
  Kokkos::View<const double*> volume;
  Kokkos::View<double*> sg_max;
  Kokkos::View<double*> remainder;
  Kokkos::View<const VhsPair*> pairs;
  Kokkos::View<const SpeciesModes*> modes;
  Kokkos::View<std::uint64_t*> cand, acc, upd;
  Kokkos::View<std::uint64_t*> pair_acc;  // nsp * nsp, accepted collisions per (min, max) species pair
  int nsp;
  double weight_dt;  // W dt
  std::uint64_t seed;
  std::uint32_t step;

  KOKKOS_INLINE_FUNCTION void operator()(const int c) const {
    const int b = offset(c);
    const int n = offset(c + 1) - b;
    std::uint64_t n_cand = 0, n_acc = 0, n_upd = 0;
    if (n >= 2) {
      double max_sg = sg_max(c);
      const double expected = 0.5 * n * (n - 1.0) * weight_dt * max_sg / volume(c) + remainder(c);
      n_cand = static_cast<std::uint64_t>(Kokkos::floor(expected));
      remainder(c) = expected - static_cast<double>(n_cand);
      RandomStream rng(seed, static_cast<std::uint64_t>(c), step, RngStream::GasCollision);
      for (std::uint64_t k = 0; k < n_cand; ++k) {
        int a = static_cast<int>(rng.uniform() * n);
        if (a >= n) a = n - 1;
        int q = static_cast<int>(rng.uniform() * (n - 1));
        if (q >= n - 1) q = n - 2;
        if (q >= a) ++q;  // never the same particle
        const int i = list(b + a), j = list(b + q);
        Vec3 vi = p.velocity(i), vj = p.velocity(j);
        const VhsPair& pr = pairs(static_cast<int>(p.species(i)) * nsp + static_cast<int>(p.species(j)));
        const double sg = pr.sigma_g(norm(vi - vj));
        if (sg > max_sg) {  // safe update of the bound (M3.7)
          max_sg = sg;
          ++n_upd;
        }
        if (rng.uniform() * max_sg < sg) {
          const int si = static_cast<int>(p.species(i)), sj = static_cast<int>(p.species(j));
          const SpeciesModes& mi = modes(si);
          const SpeciesModes& mj = modes(sj);
          if (mi.rot_dof > 0 || mi.vib || mj.rot_dof > 0 || mj.vib) {
            // Larsen-Borgnakke (particle selection, serial): vibration, then
            // rotation of each molecule with probability 1 / Z.
            const double mr = pr.m_p * pr.m_q / (pr.m_p + pr.m_q);
            const double omega = 0.5 * (mi.omega + mj.omega);
            double Et = 0.5 * mr * norm2(vi - vj);
            double er_i = p.e_rot(i), ev_i = p.e_vib(i), er_j = p.e_rot(j), ev_j = p.e_vib(j);
            if (mi.vib && rng.uniform() < mi.vib_relax_prob) lb_vibration(Et, ev_i, mi.theta_v, omega, rng);
            if (mj.vib && rng.uniform() < mj.vib_relax_prob) lb_vibration(Et, ev_j, mj.theta_v, omega, rng);
            if (mi.rot_dof == 2 && rng.uniform() < mi.rot_relax_prob) lb_rotation(Et, er_i, omega, rng);
            if (mj.rot_dof == 2 && rng.uniform() < mj.rot_relax_prob) lb_rotation(Et, er_j, omega, rng);
            p.e_rot(i) = er_i;
            p.e_vib(i) = ev_i;
            p.e_rot(j) = er_j;
            p.e_vib(j) = ev_j;
            const double g_new = Kokkos::sqrt(2.0 * Et / mr);
            if (pr.alpha == 1.0)
              vhs_collide_with_speed(vi, vj, pr.m_p, pr.m_q, g_new, rng);
            else
              vss_collide_with_speed(vi, vj, pr.m_p, pr.m_q, g_new, pr.alpha, rng);
          } else if (pr.alpha == 1.0) {
            vhs_collide(vi, vj, pr.m_p, pr.m_q, rng);
          } else {
            vss_collide_with_speed(vi, vj, pr.m_p, pr.m_q, norm(vi - vj), pr.alpha, rng);
          }
          p.set_velocity(i, vi);
          p.set_velocity(j, vj);
          ++n_acc;
          const int a1 = static_cast<int>(p.species(i)), a2 = static_cast<int>(p.species(j));
          Kokkos::atomic_inc(&pair_acc(a1 < a2 ? a1 * nsp + a2 : a2 * nsp + a1));
        }
      }
      sg_max(c) = max_sg;
    }
    cand(c) = n_cand;
    acc(c) = n_acc;
    upd(c) = n_upd;
  }
};

struct SumU64x3 {
  using value_type = std::uint64_t[];
  using size_type = std::size_t;
  size_type value_count = 3;
  Kokkos::View<const std::uint64_t*> a, b, c;
  KOKKOS_INLINE_FUNCTION void operator()(const int i, value_type s) const {
    s[0] += a(i);
    s[1] += b(i);
    s[2] += c(i);
  }
  KOKKOS_INLINE_FUNCTION void init(value_type s) const { s[0] = s[1] = s[2] = 0; }
  KOKKOS_INLINE_FUNCTION void join(value_type d, const value_type s) const {
    d[0] += s[0];
    d[1] += s[1];
    d[2] += s[2];
  }
};

struct AccumulateCounts {
  Kokkos::View<const int*> count;
  Kokkos::View<double*> sum;
  KOKKOS_INLINE_FUNCTION void operator()(const int c) const { sum(c) += count(c); }
};

struct MaxSpeed {
  ParticleStorage p;
  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t i, double& m) const {
    const double s = norm(p.velocity(i));
    if (s > m) m = s;
  }
};

}  // namespace

// ============================================================================

DsmcSettings resolve_dsmc_settings(const SimulationConfig& cfg, const Domain& domain) {
  if (cfg.gas.species.empty() || cfg.gas.species.size() > static_cast<std::size_t>(kMaxDsmcSpecies))
    throw std::invalid_argument("DSMC: 1 to 8 species are supported");
  for (const auto& sp : cfg.gas.species)
    if (!sp.vhs.available) throw std::invalid_argument("DSMC: species without VHS data");
  DsmcSettings s;
  const double T = cfg.gas.translational_temperature_K, n = cfg.gas.number_density_m3;
  double xsum = 0.0;
  for (const auto& sp : cfg.gas.species) xsum += sp.mole_fraction;
  double lambda = 0.0, nu_mean = 0.0, m_min = 1e300, m_mean = 0.0;
  for (const auto& a : cfg.gas.species) {
    const double xa = a.mole_fraction / xsum;
    double nu = 0.0;
    const bool vss = cfg.dsmc.collision_model == CollisionModel::VSS;
    for (const auto& b : cfg.gas.species)
      nu += n * (b.mole_fraction / xsum) *
            vhs_mean_sigma_g(a.mass_kg, for_model(a.vhs, vss), b.mass_kg, for_model(b.vhs, vss), T);
    const double c_mean = std::sqrt(8.0 * constants::k_B * T / (constants::pi * a.mass_kg));
    if (xa > 0.0) {
      lambda += xa * c_mean / nu;
      nu_mean += xa * nu;
    }
    m_min = std::min(m_min, a.mass_kg);
    m_mean += xa * a.mass_kg;
  }
  s.lambda_inf_m = lambda;
  s.tau_inf_s = 1.0 / nu_mean;
  // fastest species for the time-step criterion
  s.c_mp_inf_mps = std::sqrt(2.0 * constants::k_B * T / m_min);
  const double c_mp_mean = std::sqrt(2.0 * constants::k_B * T / m_mean);
  const Vec3 e = domain.box.extent();
  const double l_max = max_abs_component(e);

  s.cell_size_automatic = cfg.dsmc.cell_size_mode == ParameterMode::Automatic;
  s.cell_size_m = s.cell_size_automatic ? std::min(0.5 * s.lambda_inf_m, l_max / 16.0) : *cfg.dsmc.cell_size_m;
  const DeviceGrid g = make_device_grid(domain, s.cell_size_m, s.grid.max_cells);
  const double dx_min = std::min({g.cell_size.x, g.cell_size.y, g.cell_size.z});
  const double v_char = cfg.derived.freestream_speed_mps + 3.0 * s.c_mp_inf_mps;

  s.timestep_automatic = cfg.dsmc.timestep_mode == ParameterMode::Automatic;
  s.timestep_s = s.timestep_automatic ? std::min(0.1 * s.tau_inf_s, dx_min / (4.0 * v_char)) : *cfg.dsmc.timestep_s;
  s.particle_weight = n * g.cell_volume() / cfg.dsmc.particles_per_cell_target;

  s.warmup_automatic = !cfg.dsmc.warmup_steps.has_value();
  if (s.warmup_automatic) {
    const double v_slow = std::max(cfg.derived.freestream_speed_mps, c_mp_mean);
    s.warmup_steps = static_cast<std::uint64_t>(std::ceil(3.0 * norm(e) / v_slow / s.timestep_s));
  } else {
    s.warmup_steps = *cfg.dsmc.warmup_steps;
  }
  s.sample_steps = cfg.dsmc.sample_steps;
  s.blocks = cfg.dsmc.blocks;
  const double expected = n * g.cell_volume() * g.num_cells() / s.particle_weight;
  s.max_particles = cfg.dsmc.max_particles ? *cfg.dsmc.max_particles
                                           : static_cast<std::uint64_t>(std::max(1.0e6, 6.0 * expected));
  return s;
}

// ============================================================================

struct DsmcSimulation::Impl {
  SimulationConfig cfg;
  DsmcSettings set;
  DeviceGeometry geo;
  bool has_closed = false;
  std::size_t num_triangles = 0;
  InflowModel inflow_host;
  DeviceInflow inflow;
  CollisionGrid grid;
  GsiInteraction gsi;
  Kokkos::View<const double*> mass;
  Kokkos::View<const VhsPair*> pairs;
  Kokkos::View<const SpeciesModes*> modes;
  int nsp = 1;

  std::size_t capacity = 0;
  std::size_t n = 0;
  std::uint64_t next_id = 0;
  std::uint64_t step_index = 0;
  ParticleStorage cur, alt;
  Kokkos::View<double*> dt_cur, dt_alt;
  Kokkos::View<double**, Kokkos::LayoutLeft> tally;
  Kokkos::View<std::size_t*> compact_index;
  Kokkos::View<int*> cell_of_particle, list;

  Kokkos::View<int*> count, offset, cursor;
  Kokkos::View<double*> sg_max, remainder, count_sum;
  Kokkos::View<std::uint64_t*> cand, acc, upd;
  Kokkos::View<std::uint64_t*> pair_acc;
  Kokkos::View<double**, Kokkos::LayoutLeft> surface;
  std::vector<double> species_count_sum, species_E_sum;
  std::vector<Vec3> species_P_sum;
  std::vector<std::uint64_t> species_injected;

  DsmcCounters counters;
  std::array<std::vector<double>, kHistoryQuantities> series;
  std::uint64_t sampled = 0;
  double particles_sum = 0.0, particles_min = 1e300, particles_max = 0.0;
  double max_speed = 0.0;
  std::uint64_t sampled_cand = 0, sampled_acc = 0, sampled_particle_steps = 0;

  void allocate(std::size_t cap) {
    if (cap > set.max_particles) {
      std::ostringstream os;
      os << "DSMC needs " << cap << " particles, more than dsmc.max_particles = " << set.max_particles;
      throw std::runtime_error(os.str());
    }
    if (capacity == 0) {
      cur = ParticleStorage(cap, "dsmc_a");
      alt = ParticleStorage(cap, "dsmc_b");
      dt_cur = Kokkos::View<double*>("dt_a", cap);
      dt_alt = Kokkos::View<double*>("dt_b", cap);
    } else {
      for (ParticleStorage* s : {&cur, &alt}) {
        Kokkos::resize(s->x, cap);
        Kokkos::resize(s->y, cap);
        Kokkos::resize(s->z, cap);
        Kokkos::resize(s->vx, cap);
        Kokkos::resize(s->vy, cap);
        Kokkos::resize(s->vz, cap);
        Kokkos::resize(s->weight, cap);
        Kokkos::resize(s->species, cap);
        Kokkos::resize(s->particle_id, cap);
        Kokkos::resize(s->status, cap);
        Kokkos::resize(s->interactions, cap);
        Kokkos::resize(s->e_rot, cap);
        Kokkos::resize(s->e_vib, cap);
      }
      Kokkos::resize(dt_cur, cap);
      Kokkos::resize(dt_alt, cap);
    }
    tally = Kokkos::View<double**, Kokkos::LayoutLeft>("dsmc_step_tally", cap, kStepTallies);
    compact_index = Kokkos::View<std::size_t*>("dsmc_compact_index", cap);
    cell_of_particle = Kokkos::View<int*>("dsmc_cell", cap);
    list = Kokkos::View<int*>("dsmc_cell_list", cap);
    capacity = cap;
  }

  void ensure_capacity(std::size_t need) {
    if (need <= capacity) return;
    allocate(std::max<std::size_t>(need, capacity + capacity / 2 + 1024));
  }

  void compact() {
    std::size_t alive = 0;
    Kokkos::parallel_scan("vleo::dsmc_alive", policy(0, n), AliveFlag{cur.status, compact_index}, alive);
    Kokkos::parallel_for("vleo::dsmc_compact", policy(0, n), CompactCopy{cur, alt, dt_cur, dt_alt, compact_index});
    Kokkos::fence();
    std::swap(cur, alt);
    std::swap(dt_cur, dt_alt);
    n = alive;
  }

  void inject() {
    const int nch = static_cast<int>(inflow_host.channels.size());
    if (nch > kMaxChannels) throw std::runtime_error("too many inflow channels");
    Kokkos::Array<std::uint64_t, kMaxChannels + 1> off{};
    std::uint64_t total = 0;
    for (int c = 0; c < nch; ++c) {
      off[c] = total;
      const double expected = inflow_host.channels[static_cast<std::size_t>(c)].rate_per_s * set.timestep_s /
                              set.particle_weight;
      RandomStream rng(cfg.simulation.random_seed, step_index, static_cast<std::uint32_t>(c),
                       RngStream::InjectionCount);
      const auto k = static_cast<std::uint64_t>(std::floor(expected + rng.uniform()));
      species_injected[inflow_host.channels[static_cast<std::size_t>(c)].species] += k;
      total += k;
    }
    off[nch] = total;
    if (total == 0) return;
    ensure_capacity(n + total);
    Kokkos::parallel_for("vleo::dsmc_inject", policy(0, total),
                         InjectKernel{cur, dt_cur, inflow, off, nch, n, next_id, cfg.simulation.random_seed,
                                      set.timestep_s, set.particle_weight});
    n += total;
    next_id += total;
    counters.injected += total;
  }

  void bin() {
    const int nc = grid.grid.num_cells();
    Kokkos::deep_copy(count, 0);
    Kokkos::deep_copy(cursor, 0);
    Kokkos::parallel_for("vleo::dsmc_cell_index", policy(0, n), CellIndexKernel{cur, grid.grid, cell_of_particle, count});
    int total = 0;
    Kokkos::parallel_scan("vleo::dsmc_offsets", IntPolicy(0, nc), OffsetScan{count, offset, nc}, total);
    Kokkos::parallel_for("vleo::dsmc_fill_cells", policy(0, n), FillCells{cell_of_particle, offset, cursor, list});
    Kokkos::parallel_for("vleo::dsmc_sort_cells", IntPolicy(0, nc), SortCells{offset, list, cur.particle_id});
  }

  void collide() {
    const int nc = grid.grid.num_cells();
    Kokkos::parallel_for("vleo::dsmc_collide", IntPolicy(0, nc),
                         CollideKernel{cur, offset, list, grid.collision_volume, sg_max, remainder, pairs, modes, cand, acc, upd,
                                       pair_acc, nsp, set.particle_weight * set.timestep_s, cfg.simulation.random_seed,
                                       static_cast<std::uint32_t>(step_index)});
    std::uint64_t s[3] = {0, 0, 0};
    Kokkos::parallel_reduce("vleo::dsmc_collision_counts", IntPolicy(0, nc), SumU64x3{3, cand, acc, upd}, s);
    counters.collision_candidates += s[0];
    counters.collisions += s[1];
    counters.sigma_g_max_updates += s[2];
    last_cand = s[0];
    last_acc = s[1];
  }
  std::uint64_t last_cand = 0, last_acc = 0;

  void step(bool sample) {
    if (set.boundary == DsmcBoundary::Open) inject();
    MoveKernel mk{cur,
                  dt_cur,
                  tally,
                  sample ? surface : Kokkos::View<double**, Kokkos::LayoutLeft>(),
                  geo,
                  inflow_host.domain.box,
                  gsi,
                  mass,
                  modes,
                  cfg.reference.moment_center_m,
                  cfg.simulation.random_seed,
                  set.timestep_s,
                  set.max_interactions_per_step,
                  set.boundary == DsmcBoundary::Specular};
    Kokkos::parallel_for("vleo::dsmc_move", policy(0, n), mk);

    double s[StepReduce::kSlots] = {};
    if (n > 0) {
      StepReduce red;
      red.tally = tally;
      red.status = cur.status;
      red.species = cur.species;
      red.value_count = static_cast<std::size_t>(StepReduce::kGlobal + kSpeciesSlots * nsp);
      Kokkos::parallel_reduce("vleo::dsmc_step_reduce", policy(0, n), red, s);
    }
    counters.exited += static_cast<std::uint64_t>(s[kStepTallies + static_cast<int>(ParticleStatus::Exited)]);
    counters.truncated += static_cast<std::uint64_t>(s[kStepTallies + static_cast<int>(ParticleStatus::Truncated)]);
    counters.lost_inside_body +=
        static_cast<std::uint64_t>(s[kStepTallies + static_cast<int>(ParticleStatus::LostInsideBody)]);
    counters.invalid += static_cast<std::uint64_t>(s[kStepTallies + static_cast<int>(ParticleStatus::Invalid)]);
    counters.surface_interactions += static_cast<std::uint64_t>(s[kStepInteractions]);

    if (sample) {
      const double scale = set.particle_weight / set.timestep_s;  // per-step event sums -> rates
      for (int k = 0; k < kHistoryTallies; ++k) series[static_cast<std::size_t>(k)].push_back(scale * s[k]);
      const Vec3 dp(s[kTallyPx], s[kTallyPy], s[kTallyPz]);
      const AeroAxes& ax = cfg.derived.aero_axes;
      series[kQPDrag].push_back(scale * dot(dp, ax.drag));
      series[kQPLift].push_back(scale * dot(dp, ax.lift));
      series[kQPSide].push_back(scale * dot(dp, ax.side));
      for (int q = 0; q < nsp; ++q) {
        const double* v = s + StepReduce::kGlobal + kSpeciesSlots * q;
        species_count_sum[static_cast<std::size_t>(q)] += v[0];
        species_P_sum[static_cast<std::size_t>(q)] += scale * Vec3(v[1], v[2], v[3]);
        species_E_sum[static_cast<std::size_t>(q)] += scale * v[4];
      }
    }

    compact();
    bin();
    collide();

    if (sample) {
      Kokkos::parallel_for("vleo::dsmc_count_sum", IntPolicy(0, grid.grid.num_cells()), AccumulateCounts{count, count_sum});
      double vmax = 0.0;
      if (n > 0)
        Kokkos::parallel_reduce("vleo::dsmc_max_speed", policy(0, n), MaxSpeed{cur}, Kokkos::Max<double>(vmax));
      max_speed = std::max(max_speed, vmax);
      const double np = static_cast<double>(n);
      particles_sum += np;
      particles_min = std::min(particles_min, np);
      particles_max = std::max(particles_max, np);
      sampled_cand += last_cand;
      sampled_acc += last_acc;
      sampled_particle_steps += n;
      ++sampled;
    }
    ++step_index;
    ++counters.steps;
  }
};

DsmcSimulation::DsmcSimulation(const SimulationConfig& cfg, const Geometry& geometry, const InflowModel& inflow,
                               const DsmcSettings& settings)
    : impl_(std::make_unique<Impl>()) {
  Impl& s = *impl_;
  s.cfg = cfg;
  s.set = settings;
  s.geo = geometry.device;
  s.num_triangles = geometry.host.size();
  for (auto t : geometry.host.two_sided) s.has_closed = s.has_closed || t == 0;
  s.inflow_host = inflow;
  s.inflow = upload(inflow);
  s.grid = make_collision_grid(inflow.domain, settings.cell_size_m, geometry, settings.grid);

  std::vector<SurfaceProperties> props(geometry.host.material_names.size(), make_surface_properties(cfg.surface));
  s.gsi.properties = to_device(props, "dsmc_surface_properties");
  s.gsi.material = geometry.device.mesh.material;

  std::vector<double> masses;
  for (const auto& sp : cfg.gas.species) masses.push_back(sp.mass_kg);
  s.mass = to_device(masses, "dsmc_species_mass");
  s.modes = to_device(make_species_modes(cfg.gas), "dsmc_species_modes");
  s.nsp = static_cast<int>(cfg.gas.species.size());
  std::vector<VhsPair> pairs;
  for (const auto& a : cfg.gas.species)
    for (const auto& b : cfg.gas.species)
      pairs.push_back(make_vhs_pair(a.mass_kg, a.vhs, b.mass_kg, b.vhs, cfg.dsmc.collision_model == CollisionModel::VSS));
  s.pairs = to_device(pairs, "dsmc_vhs_pairs");

  const auto nc = static_cast<std::size_t>(s.grid.grid.num_cells());
  s.count = Kokkos::View<int*>("dsmc_count", nc);
  s.cursor = Kokkos::View<int*>("dsmc_cursor", nc);
  s.offset = Kokkos::View<int*>("dsmc_offset", nc + 1);
  s.remainder = Kokkos::View<double*>("dsmc_remainder", nc);
  s.count_sum = Kokkos::View<double*>("dsmc_count_sum", nc);
  s.cand = Kokkos::View<std::uint64_t*>("dsmc_cand", nc);
  s.acc = Kokkos::View<std::uint64_t*>("dsmc_acc", nc);
  s.upd = Kokkos::View<std::uint64_t*>("dsmc_upd", nc);
  s.pair_acc = Kokkos::View<std::uint64_t*>("dsmc_pair_acc", static_cast<std::size_t>(s.nsp * s.nsp));
  s.species_count_sum.assign(static_cast<std::size_t>(s.nsp), 0.0);
  s.species_E_sum.assign(static_cast<std::size_t>(s.nsp), 0.0);
  s.species_P_sum.assign(static_cast<std::size_t>(s.nsp), Vec3());
  s.species_injected.assign(static_cast<std::size_t>(s.nsp), 0);
  s.sg_max = Kokkos::View<double*>("dsmc_sg_max", nc);
  // Initial bound: largest pair sigma g at three most probable thermal speeds of
  // the fastest species (raised when exceeded).
  double sg0 = 0.0;
  for (const VhsPair& pr : pairs) sg0 = std::max(sg0, pr.sigma_g(3.0 * settings.c_mp_inf_mps));
  Kokkos::deep_copy(s.sg_max, sg0);
  if (cfg.output.save_surface_fields)
    s.surface = Kokkos::View<double**, Kokkos::LayoutLeft>("dsmc_surface", geometry.host.size(), kSurfaceTallies);

  s.allocate(std::min<std::size_t>(settings.max_particles, 1u << 16));
}

DsmcSimulation::~DsmcSimulation() = default;

void DsmcSimulation::fill_freestream() {
  Impl& s = *impl_;
  const Aabb box = s.inflow_host.domain.box;
  const Vec3 e = box.extent();
  const double expected = s.cfg.gas.number_density_m3 * e.x * e.y * e.z / s.set.particle_weight;
  RandomStream rng(s.cfg.simulation.random_seed, 0, 0, RngStream::InjectionCount);
  const auto total = static_cast<std::size_t>(std::floor(expected + rng.uniform()));
  s.n = 0;
  s.ensure_capacity(total);
  std::vector<double> cdf, sigma;
  double acc = 0.0, sum = 0.0;
  for (const auto& sp : s.cfg.gas.species) sum += sp.mole_fraction;
  for (const auto& sp : s.cfg.gas.species) {
    acc += sp.mole_fraction / sum;
    cdf.push_back(acc);
    sigma.push_back(std::sqrt(constants::k_B * s.cfg.gas.translational_temperature_K / sp.mass_kg));
  }
  Kokkos::parallel_for("vleo::dsmc_fill", policy(0, total),
                       FillKernel{s.cur, s.dt_cur, s.geo, box, s.cfg.gas.bulk_velocity_body_mps,
                                  to_device(cdf, "fill_cdf"), to_device(sigma, "fill_sigma"), s.modes,
                                  s.cfg.gas.translational_temperature_K, s.next_id,
                                  s.cfg.simulation.random_seed, s.set.timestep_s, s.set.particle_weight, s.has_closed});
  s.n = total;
  s.next_id += total;
  s.compact();
  s.counters.removed_inside_at_fill += total - s.n;
  s.bin();
}

void DsmcSimulation::set_particles(const std::vector<Vec3>& positions, const std::vector<Vec3>& velocities,
                                   const std::vector<std::uint32_t>& species, const std::vector<double>& e_rot,
                                   const std::vector<double>& e_vib) {
  Impl& s = *impl_;
  const std::size_t total = positions.size();
  s.ensure_capacity(total);
  auto hx = Kokkos::create_mirror_view(s.cur.x), hy = Kokkos::create_mirror_view(s.cur.y),
       hz = Kokkos::create_mirror_view(s.cur.z);
  auto hvx = Kokkos::create_mirror_view(s.cur.vx), hvy = Kokkos::create_mirror_view(s.cur.vy),
       hvz = Kokkos::create_mirror_view(s.cur.vz);
  auto hsp = Kokkos::create_mirror_view(s.cur.species);
  auto hid = Kokkos::create_mirror_view(s.cur.particle_id);
  auto hst = Kokkos::create_mirror_view(s.cur.status);
  auto hw = Kokkos::create_mirror_view(s.cur.weight);
  auto hk = Kokkos::create_mirror_view(s.cur.interactions);
  auto hdt = Kokkos::create_mirror_view(s.dt_cur);
  auto her = Kokkos::create_mirror_view(s.cur.e_rot);
  auto hev = Kokkos::create_mirror_view(s.cur.e_vib);
  for (std::size_t i = 0; i < total; ++i) {
    hx(i) = positions[i].x;
    hy(i) = positions[i].y;
    hz(i) = positions[i].z;
    hvx(i) = velocities[i].x;
    hvy(i) = velocities[i].y;
    hvz(i) = velocities[i].z;
    hsp(i) = species[i];
    hid(i) = s.next_id + i;
    hst(i) = static_cast<std::uint8_t>(ParticleStatus::Active);
    hw(i) = s.set.particle_weight;
    hk(i) = 0;
    her(i) = i < e_rot.size() ? e_rot[i] : 0.0;
    hev(i) = i < e_vib.size() ? e_vib[i] : 0.0;
    hdt(i) = s.set.timestep_s;
  }
  Kokkos::deep_copy(s.cur.x, hx);
  Kokkos::deep_copy(s.cur.y, hy);
  Kokkos::deep_copy(s.cur.z, hz);
  Kokkos::deep_copy(s.cur.vx, hvx);
  Kokkos::deep_copy(s.cur.vy, hvy);
  Kokkos::deep_copy(s.cur.vz, hvz);
  Kokkos::deep_copy(s.cur.species, hsp);
  Kokkos::deep_copy(s.cur.particle_id, hid);
  Kokkos::deep_copy(s.cur.status, hst);
  Kokkos::deep_copy(s.cur.weight, hw);
  Kokkos::deep_copy(s.cur.interactions, hk);
  Kokkos::deep_copy(s.dt_cur, hdt);
  Kokkos::deep_copy(s.cur.e_rot, her);
  Kokkos::deep_copy(s.cur.e_vib, hev);
  s.n = total;
  s.next_id += total;
  s.bin();
}

void DsmcSimulation::step(bool sample) { impl_->step(sample); }
std::size_t DsmcSimulation::num_particles() const { return impl_->n; }
const DsmcCounters& DsmcSimulation::counters() const { return impl_->counters; }
const CollisionGrid& DsmcSimulation::grid() const { return impl_->grid; }
const DsmcSettings& DsmcSimulation::settings() const { return impl_->set; }
std::uint64_t DsmcSimulation::sampled_steps() const { return impl_->sampled; }

namespace {
std::vector<Vec3> host_vec3(const Kokkos::View<double*>& x, const Kokkos::View<double*>& y,
                            const Kokkos::View<double*>& z, std::size_t n) {
  auto hx = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), x);
  auto hy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), y);
  auto hz = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), z);
  std::vector<Vec3> v(n);
  for (std::size_t i = 0; i < n; ++i) v[i] = Vec3(hx(i), hy(i), hz(i));
  return v;
}
}  // namespace

std::vector<Vec3> DsmcSimulation::velocities() const {
  return host_vec3(impl_->cur.vx, impl_->cur.vy, impl_->cur.vz, impl_->n);
}
std::vector<Vec3> DsmcSimulation::positions() const {
  return host_vec3(impl_->cur.x, impl_->cur.y, impl_->cur.z, impl_->n);
}
std::vector<double> DsmcSimulation::rotational_energies() const {
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), impl_->cur.e_rot);
  return std::vector<double>(h.data(), h.data() + impl_->n);
}

std::vector<double> DsmcSimulation::vibrational_energies() const {
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), impl_->cur.e_vib);
  return std::vector<double>(h.data(), h.data() + impl_->n);
}

std::vector<std::uint32_t> DsmcSimulation::species() const {
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), impl_->cur.species);
  return std::vector<std::uint32_t>(h.data(), h.data() + impl_->n);
}

std::uint64_t DsmcSimulation::pair_collisions(int p, int q) const {
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), impl_->pair_acc);
  const int a = std::min(p, q), b = std::max(p, q);
  return h(a * impl_->nsp + b);
}

std::vector<std::uint64_t> DsmcSimulation::particle_ids() const {
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), impl_->cur.particle_id);
  return std::vector<std::uint64_t>(h.data(), h.data() + impl_->n);
}

void DsmcSimulation::binning(std::vector<int>& offsets, std::vector<int>& list) const {
  auto ho = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), impl_->offset);
  auto hl = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), impl_->list);
  offsets.assign(ho.data(), ho.data() + ho.extent(0));
  list.assign(hl.data(), hl.data() + impl_->n);
}

DsmcResult DsmcSimulation::result() const {
  const Impl& s = *impl_;
  DsmcResult r;
  r.settings = s.set;
  r.counters = s.counters;
  r.grid = s.grid.report;
  r.critical_value = critical_value(s.cfg.statistics.confidence_level);
  const double z = r.critical_value;

  std::array<Estimate, kHistoryQuantities> base{};
  for (std::size_t q = 0; q < base.size(); ++q) {
    r.series[q] = analyze_series(s.series[q], s.set.blocks);
    base[q].value = r.series[q].mean;
    base[q].std_error = r.series[q].std_error_blocks;
  }
  auto est = [&](int q, double scale) {
    Estimate e;
    e.value = scale * base[static_cast<std::size_t>(q)].value;
    e.std_error = std::fabs(scale) * base[static_cast<std::size_t>(q)].std_error;
    e.ci_low = e.value - z * e.std_error;
    e.ci_high = e.value + z * e.std_error;
    return e;
  };
  TpmcEstimates& e = r.estimates;
  for (int k = 0; k < 3; ++k) {
    e.force_N[static_cast<std::size_t>(k)] = est(kQPx + k, 1.0);
    e.moment_Nm[static_cast<std::size_t>(k)] = est(kQLx + k, 1.0);
  }
  e.heat_rate_W = est(kQE, 1.0);
  e.heat_rate_positive_W = est(kQEPos, 1.0);
  e.heat_rate_negative_W = est(kQENeg, 1.0);
  e.drag_N = est(kQPDrag, 1.0);
  e.lift_N = est(kQPLift, 1.0);
  e.side_N = est(kQPSide, 1.0);
  const double qa = s.cfg.derived.dynamic_pressure_Pa * s.cfg.reference.area_m2;
  const double qal = qa * s.cfg.reference.length_m;
  e.C_D = est(kQPDrag, 1.0 / qa);
  e.C_L = est(kQPLift, 1.0 / qa);
  e.C_Y = est(kQPSide, 1.0 / qa);
  for (int k = 0; k < 3; ++k) e.C_M[static_cast<std::size_t>(k)] = est(kQLx + k, 1.0 / qal);

  // Surface fields: event sums over the sampling phase -> rates.
  if (s.surface.extent(0) > 0 && s.sampled > 0) {
    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), s.surface);
    const double g = s.set.particle_weight / (s.set.timestep_s * static_cast<double>(s.sampled));
    SurfaceFields f;
    for (std::size_t t = 0; t < s.num_triangles; ++t) {
      f.force_N.push_back(g * Vec3(h(t, kSurfPx), h(t, kSurfPy), h(t, kSurfPz)));
      f.heat_rate_W.push_back(g * h(t, kSurfE));
      f.heat_rate_positive_W.push_back(g * h(t, kSurfEPos));
      f.heat_rate_negative_W.push_back(g * h(t, kSurfENeg));
      f.impingement_rate_per_s.push_back(g * h(t, kSurfHits));
    }
    r.surface = std::move(f);
  }

  // Diagnostics.
  DsmcDiagnostics& d = r.diagnostics;
  const DeviceGrid& g = s.grid.grid;
  d.cell_dx_m = g.cell_size.x;
  d.cell_dy_m = g.cell_size.y;
  d.cell_dz_m = g.cell_size.z;
  const double dx_max = max_abs_component(g.cell_size);
  const double dx_min = std::min({g.cell_size.x, g.cell_size.y, g.cell_size.z});
  d.timestep_s = s.set.timestep_s;
  d.lambda_inf_m = s.set.lambda_inf_m;
  d.tau_inf_s = s.set.tau_inf_s;
  d.cell_over_lambda_inf = dx_max / s.set.lambda_inf_m;
  d.dt_over_tau_inf = s.set.timestep_s / s.set.tau_inf_s;
  d.max_cfl = s.max_speed * s.set.timestep_s / dx_min;
  d.free_stream_collisions_per_particle_per_step = s.set.timestep_s / s.set.tau_inf_s;
  if (s.sampled > 0) {
    d.particles_mean = s.particles_sum / static_cast<double>(s.sampled);
    d.particles_min = s.particles_min;
    d.particles_max = s.particles_max;
    d.acceptance_ratio = s.sampled_cand > 0 ? static_cast<double>(s.sampled_acc) / static_cast<double>(s.sampled_cand) : 0.0;
    d.collisions_per_particle_per_step =
        s.sampled_particle_steps > 0 ? 2.0 * static_cast<double>(s.sampled_acc) / static_cast<double>(s.sampled_particle_steps)
                                     : 0.0;
    auto cs = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), s.count_sum);
    auto fr = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), s.grid.gas_fraction);
    std::int64_t gas = 0, below5 = 0, coarse = 0;
    double ppc = 0.0, min_lambda = Kokkos::infinity_v<double>;
    const double n_ref = s.cfg.gas.number_density_m3;
    for (int c = 0; c < g.num_cells(); ++c) {
      if (fr(c) <= 0.0) continue;
      ++gas;
      const double mean_count = cs(c) / static_cast<double>(s.sampled);
      ppc += mean_count;
      if (mean_count < 5.0) ++below5;
      const double n_cell = mean_count * s.set.particle_weight / (fr(c) * g.cell_volume());
      if (n_cell > 0.0) {
        const double lam = s.set.lambda_inf_m * n_ref / n_cell;  // density scaling at free-stream temperature
        min_lambda = std::min(min_lambda, lam);
        if (dx_max > lam) ++coarse;
      }
    }
    d.mean_particles_per_gas_cell = gas > 0 ? ppc / static_cast<double>(gas) : 0.0;
    d.fraction_cells_below_5 = gas > 0 ? static_cast<double>(below5) / static_cast<double>(gas) : 0.0;
    d.fraction_cells_dx_gt_lambda = gas > 0 ? static_cast<double>(coarse) / static_cast<double>(gas) : 0.0;
    d.min_local_lambda_m = std::isfinite(min_lambda) ? min_lambda : s.set.lambda_inf_m;
  }

  // Per-species diagnostics.
  {
    auto pa = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), s.pair_acc);
    const double ns = s.sampled > 0 ? static_cast<double>(s.sampled) : 1.0;
    for (int q = 0; q < s.nsp; ++q) {
      DsmcSpeciesDiagnostics sd;
      sd.name = s.cfg.gas.species[static_cast<std::size_t>(q)].name;
      sd.mean_particles = s.species_count_sum[static_cast<std::size_t>(q)] / ns;
      sd.injected = s.species_injected[static_cast<std::size_t>(q)];
      sd.force_N = s.species_P_sum[static_cast<std::size_t>(q)] / ns;
      sd.heat_rate_W = s.species_E_sum[static_cast<std::size_t>(q)] / ns;
      for (int o = 0; o < s.nsp; ++o) sd.collisions_with.push_back(pa(std::min(q, o) * s.nsp + std::max(q, o)));
      r.species.push_back(sd);
    }
  }

  // Warnings.
  auto warn = [&r](const std::string& w) { r.warnings.push_back(w); };
  const DsmcCounters& c = s.counters;
  if (c.truncated > 0) warn(std::to_string(c.truncated) + " particles exceeded the interaction limit per step");
  if (c.lost_inside_body + c.invalid > 0)
    warn(std::to_string(c.lost_inside_body) + " particles lost inside the body, " + std::to_string(c.invalid) +
         " invalid (numerical failures)");
  if (r.grid.small_cells > 0)
    warn(std::to_string(r.grid.small_cells) + " cut cells have a gas fraction below " +
         std::to_string(s.set.grid.small_fraction) + "; their collision volume is limited");
  if (d.fraction_cells_dx_gt_lambda > 0.0) {
    std::ostringstream os;
    os << "cell size exceeds the local mean free path in " << 100.0 * d.fraction_cells_dx_gt_lambda
       << " % of the gas cells (min local lambda " << d.min_local_lambda_m << " m)";
    warn(os.str());
  }
  if (d.dt_over_tau_inf > 0.2) warn("time step exceeds 0.2 free-stream collision times");
  if (d.max_cfl > 1.0) warn("fastest particles cross more than one cell per time step (CFL > 1)");
  if (d.fraction_cells_below_5 > 0.1) warn("more than 10 % of the gas cells hold fewer than 5 particles");
  for (int q : {static_cast<int>(kQPDrag), static_cast<int>(kQE)})
    if (!r.series[static_cast<std::size_t>(q)].blocks_long_enough) {
      warn("block length shorter than 5 autocorrelation times; the standard error may be underestimated");
      break;
    }
  return r;
}

DsmcResult run_dsmc(const SimulationConfig& cfg, const Geometry& geometry, const InflowModel& inflow,
                    const DsmcSettings& settings, const DsmcProgressCallback& progress) {
  const auto t0 = std::chrono::steady_clock::now();
  DsmcSimulation sim(cfg, geometry, inflow, settings);
  if (settings.initial_fill) sim.fill_freestream();
  const std::uint64_t total = settings.warmup_steps + settings.sample_steps;
  const std::uint64_t report = std::max<std::uint64_t>(1, total / 20);
  for (std::uint64_t k = 0; k < total; ++k) {
    const bool sample = k >= settings.warmup_steps;
    sim.step(sample);
    if (progress && ((k + 1) % report == 0 || k + 1 == total)) {
      const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      progress({k + 1, total, sample, sim.num_particles(), el});
    }
  }
  Kokkos::fence();
  DsmcResult r = sim.result();
  r.wall_time_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  return r;
}

}  // namespace vleo

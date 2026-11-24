#include "vleo/dsmc/Dsmc.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <type_traits>

#include "vleo/core/Comm.hpp"
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

// Particle sent to another rank (M6.3). `slot` is the storage index on the
// sending rank (deterministic send order).
struct MigrantRecord {
  double x, y, z, vx, vy, vz, weight, e_rot, e_vib;
  std::uint64_t id;
  std::uint32_t species, interactions;
  int dest, slot;
};

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
  // Several ranks (M6.7): particles that end the step outside the planes of
  // this rank are recorded here (atomic append; the host restores the
  // storage order). They stay Active until the step reduction has counted them.
  bool find_migrants = false;
  DeviceGrid grid{};
  Kokkos::View<const int*> plane_owner{};
  int me = 0;
  Kokkos::View<MigrantRecord*> migrants{};
  Kokkos::View<int*> migrant_count{};

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t i) const {
    for (int k = 0; k < kStepTallies; ++k) tally(i, k) = 0.0;
    if (p.status(i) != static_cast<std::uint8_t>(ParticleStatus::Active)) {
      // ended in an earlier step and not yet compacted: retire (counted once, M5.3)
      p.status(i) = static_cast<std::uint8_t>(ParticleStatus::Free);
      return;
    }
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
    if (find_migrants && status == ParticleStatus::Active) {
      const int owner = plane_owner(grid.cell_of(pos) / (grid.nx * grid.ny));
      if (owner != me) {
        const int slot = Kokkos::atomic_fetch_add(&migrant_count(0), 1);
        if (slot < static_cast<int>(migrants.extent(0))) {
          MigrantRecord r;
          r.x = pos.x;
          r.y = pos.y;
          r.z = pos.z;
          r.vx = vel.x;
          r.vy = vel.y;
          r.vz = vel.z;
          r.weight = p.weight(i);
          r.e_rot = e_rot;
          r.e_vib = e_vib;
          r.id = id;
          r.species = sp;
          r.interactions = k;
          r.dest = owner;
          r.slot = static_cast<int>(i);
          migrants(slot) = r;
        }
      }
    }
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
    // active particles only (slots of ended particles wait for compaction)
    q[0] += st == static_cast<int>(ParticleStatus::Active) ? 1.0 : 0.0;
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

// Status counts and interactions only (steps outside the sampling phase).
struct StatusReduce {
  using value_type = double[];
  using size_type = std::size_t;
  static constexpr int kSlots = kNumParticleStatus + 1;
  size_type value_count = kSlots;
  Kokkos::View<const std::uint8_t*> status;
  Kokkos::View<double**, Kokkos::LayoutLeft> tally;
  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t i, value_type s) const {
    const int st = status(i) < kNumParticleStatus ? status(i) : static_cast<int>(ParticleStatus::Invalid);
    s[st] += 1.0;
    s[kNumParticleStatus] += tally(i, kStepInteractions);
  }
  KOKKOS_INLINE_FUNCTION void init(value_type s) const {
    for (int k = 0; k < kSlots; ++k) s[k] = 0.0;
  }
  KOKKOS_INLINE_FUNCTION void join(value_type d, const value_type s) const {
    for (int k = 0; k < kSlots; ++k) d[k] += s[k];
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
  std::uint64_t j0;  // first injection index of this rank (M6.3)
  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t jl) const {
    const std::uint64_t j = j0 + jl;
    int ch = 0;
    while (ch + 1 < channels && j >= offsets[ch + 1]) ++ch;
    const std::size_t i = base + jl;
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
  int c_lo, c_hi;               // cells owned by this rank
  Kokkos::View<int*> foreign;   // particles outside the owned cells (must stay 0)
  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t i) const {
    if (p.status(i) != static_cast<std::uint8_t>(ParticleStatus::Active)) {
      cell(i) = -1;  // not yet compacted
      return;
    }
    const int c = grid.cell_of(p.position(i));
    if (c < c_lo || c >= c_hi) {
      cell(i) = -1;
      Kokkos::atomic_inc(&foreign(0));
      return;
    }
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
  // GPU backends with several ranks: the sort key (particle id) is stored
  // next to the list entry, so the rank sort reads it contiguously (M6.7)
  Kokkos::View<const std::uint64_t*> id{};
  Kokkos::View<std::uint64_t*> key{};
  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t i) const {
    const int c = cell(i);
    if (c < 0) return;
    const int slot = Kokkos::atomic_fetch_add(&cursor(c), 1);
    list(offset(c) + slot) = static_cast<int>(i);
    if (key.extent(0) > 0) key(offset(c) + slot) = id(i);
  }
};

// Insertion sort of each cell: the order within a cell no longer depends on
// the (non-deterministic) atomic fill order. Sorting by storage index is the
// same as sorting by particle id, because ids increase with the storage index
// (fill, injection and set_particles assign ids in index order; compaction
// preserves the order) - checked in test_grid. Comparing indices avoids one
// gather per comparison (M5.3).
//
// With several ranks (M6.3) migrated particles are appended to the storage,
// so the lists are sorted by the global particle id instead (`by_id`): every
// cell then holds the same particles in the same order as in a single-rank
// run, and the collisions are identical.
struct SortCells {
  Kokkos::View<const int*> offset;
  Kokkos::View<int*> list;
  Kokkos::View<const std::uint64_t*> id;
  bool by_id;
  KOKKOS_INLINE_FUNCTION std::uint64_t key_of(const int v) const {
    return by_id ? id(v) : static_cast<std::uint64_t>(v);
  }
  KOKKOS_INLINE_FUNCTION void operator()(const int c) const {
    const int b = offset(c), e = offset(c + 1);
    for (int a = b + 1; a < e; ++a) {
      const int v = list(a);
      const std::uint64_t key = key_of(v);
      int k = a - 1;
      while (k >= b && key_of(list(k)) > key) {
        list(k + 1) = list(k);
        --k;
      }
      list(k + 1) = v;
    }
  }
};

// GPU variant (M5, profiled on Radeon PRO W7700, where the serial insertion
// chains of one thread per cell dominated the step): one thread per list
// entry computes its rank within its cell = number of smaller entries (entries
// are unique storage indices) and writes it to its final position. Same sorted
// list as SortCells; all reads of one cell hit the cache.
struct RankSortEntries {
  Kokkos::View<const int*> offset;
  Kokkos::View<const int*> cell_of_particle;
  Kokkos::View<const int*> in;
  Kokkos::View<int*> out;
  Kokkos::View<const std::uint64_t*> key;  // per list entry (by_id only)
  bool by_id;
  KOKKOS_INLINE_FUNCTION void operator()(const int k) const {
    const int v = in(k);
    const int c = cell_of_particle(v);
    const int b = offset(c), e = offset(c + 1);
    int r = 0;
    if (by_id) {
      const std::uint64_t kv = key(k);
      for (int j = b; j < e; ++j) r += key(j) < kv ? 1 : 0;
    } else {
      for (int j = b; j < e; ++j) r += in(j) < v ? 1 : 0;
    }
    out(b + r) = v;
  }
};

inline constexpr bool kHostExecution =
    std::is_same_v<Kokkos::DefaultExecutionSpace, Kokkos::DefaultHostExecutionSpace>;

// NTC collisions (docs/dsmc.md section 4). Every candidate pair k of cell c
// has its own random stream (seed, c + k 2^32, step), and all candidates of a
// step use the bound (sigma g)_max from the start of the step, so the outcome
// of a candidate depends only on the velocities of its two particles. The
// result of processing the candidates in index order is therefore reproduced
// exactly by any order that respects "a candidate runs after all earlier
// candidates sharing one of its particles" - the serial host kernel and the
// wave-parallel GPU kernel below give bitwise identical results.
struct CollisionPairOp {
  ParticleStorage p;
  Kokkos::View<const VhsPair*> pairs;
  Kokkos::View<const SpeciesModes*> modes;
  Kokkos::View<std::uint64_t*> pair_acc;  // nsp * nsp, accepted collisions per (min, max) species pair
  int nsp;
  std::uint64_t seed;
  std::uint32_t step;

  KOKKOS_INLINE_FUNCTION RandomStream stream(const int c, const std::uint64_t k) const {
    return RandomStream(seed, static_cast<std::uint64_t>(static_cast<std::uint32_t>(c)) | (k << 32), step,
                        RngStream::GasCollision);
  }
  // Pair positions (a, q) within the cell list of n >= 2 particles.
  KOKKOS_INLINE_FUNCTION static void select(RandomStream& rng, const int n, int& a, int& q) {
    a = static_cast<int>(rng.uniform() * n);
    if (a >= n) a = n - 1;
    q = static_cast<int>(rng.uniform() * (n - 1));
    if (q >= n - 1) q = n - 2;
    if (q >= a) ++q;  // never the same particle
  }
  // Candidate (i, j) after selection; returns sigma g, `accepted` set if the
  // collision was performed.
  KOKKOS_INLINE_FUNCTION double candidate(const int i, const int j, const double max_sg, RandomStream& rng,
                                          bool& accepted) const {
    Vec3 vi = p.velocity(i), vj = p.velocity(j);
    const VhsPair& pr = pairs(static_cast<int>(p.species(i)) * nsp + static_cast<int>(p.species(j)));
    const double sg = pr.sigma_g(norm(vi - vj));
    accepted = rng.uniform() * max_sg < sg;  // always accepted if sg exceeds the bound
    if (!accepted) return sg;
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
    Kokkos::atomic_inc(&pair_acc(si < sj ? si * nsp + sj : sj * nsp + si));
    return sg;
  }
};

// Candidate number of a cell (integer part; the fraction is carried over).
struct CellCandidates {
  Kokkos::View<const int*> offset;
  Kokkos::View<const double*> volume;
  Kokkos::View<const double*> sg_max;
  Kokkos::View<double*> remainder;
  Kokkos::View<std::uint64_t*> cand;
  double weight_dt;  // W dt
  KOKKOS_INLINE_FUNCTION void operator()(const int c) const {
    const int n = offset(c + 1) - offset(c);
    std::uint64_t k = 0;
    if (n >= 2) {
      const double expected = 0.5 * n * (n - 1.0) * weight_dt * sg_max(c) / volume(c) + remainder(c);
      k = static_cast<std::uint64_t>(Kokkos::floor(expected));
      remainder(c) = expected - static_cast<double>(k);
    }
    cand(c) = k;
  }
};

// Host backends: one thread per cell, candidates in index order.
struct CollideCellsSerial {
  CollisionPairOp m;
  Kokkos::View<const int*> offset;
  Kokkos::View<const int*> list;
  Kokkos::View<double*> sg_max;
  Kokkos::View<const std::uint64_t*> cand;
  Kokkos::View<std::uint64_t*> acc, upd;
  KOKKOS_INLINE_FUNCTION void operator()(const int c) const {
    const int b = offset(c), n = offset(c + 1) - b;
    const std::uint64_t n_cand = cand(c);
    std::uint64_t n_acc = 0, n_upd = 0;
    const double max0 = sg_max(c);
    double max_new = max0;
    for (std::uint64_t k = 0; k < n_cand; ++k) {
      RandomStream rng = m.stream(c, k);
      int a, q;
      CollisionPairOp::select(rng, n, a, q);
      bool accepted;
      const double sg = m.candidate(list(b + a), list(b + q), max0, rng, accepted);
      if (sg > max0) ++n_upd;  // safe update of the bound for the next step (M3.7)
      if (sg > max_new) max_new = sg;
      if (accepted) ++n_acc;
    }
    if (n_cand > 0) sg_max(c) = max_new;
    acc(c) = n_acc;
    upd(c) = n_upd;
  }
};

// GPU backends: one team per cell. The candidates are taken in chunks of
// kCollideChunk; within a chunk they run in waves - a candidate runs once all
// earlier candidates of the chunk that share one of its particles have run.
// Chunks run one after the other. This is the same dependency order as the
// serial kernel, hence the same result.
enum : int { kCollideChunk = 256 };  // enumerator: usable in device code also where bound to a reference (CUDA)
struct CollideCellsWaves {
  using Member = Kokkos::TeamPolicy<>::member_type;
  CollisionPairOp m;
  Kokkos::View<const int*> offset;
  Kokkos::View<const int*> list;
  Kokkos::View<double*> sg_max;
  Kokkos::View<const std::uint64_t*> cand;
  Kokkos::View<std::uint64_t*> acc, upd;
  int c_lo = 0;  // first owned cell (league rank 0)

  static std::size_t scratch_bytes() {
    return kCollideChunk * (sizeof(double) + 2 * sizeof(int) + 2 * sizeof(char)) + 64;
  }

  KOKKOS_INLINE_FUNCTION void operator()(const Member& t) const {
    const int c = c_lo + t.league_rank();
    const std::uint64_t n_cand = cand(c);
    if (n_cand == 0) {
      Kokkos::single(Kokkos::PerTeam(t), [&]() {
        acc(c) = 0;
        upd(c) = 0;
      });
      return;
    }
    const int b = offset(c), n = offset(c + 1) - b;
    const double max0 = sg_max(c);
    auto& sm = t.team_scratch(0);
    double* sgv = static_cast<double*>(sm.get_shmem(kCollideChunk * sizeof(double)));
    int* pi = static_cast<int*>(sm.get_shmem(kCollideChunk * sizeof(int)));
    int* pj = static_cast<int*>(sm.get_shmem(kCollideChunk * sizeof(int)));
    char* state = static_cast<char*>(sm.get_shmem(kCollideChunk));  // 0 pending, 1 accepted, 2 rejected
    char* ready = static_cast<char*>(sm.get_shmem(kCollideChunk));
    std::uint64_t n_acc = 0, n_upd = 0;
    double max_new = max0;
    for (std::uint64_t c0 = 0; c0 < n_cand; c0 += kCollideChunk) {
      const std::uint64_t chunk = static_cast<std::uint64_t>(kCollideChunk);
      const int nk = static_cast<int>(n_cand - c0 < chunk ? n_cand - c0 : chunk);
      Kokkos::parallel_for(Kokkos::TeamThreadRange(t, nk), [&](const int k) {
        RandomStream rng = m.stream(c, c0 + static_cast<std::uint64_t>(k));
        int a, q;
        CollisionPairOp::select(rng, n, a, q);
        pi[k] = list(b + a);
        pj[k] = list(b + q);
        state[k] = 0;
      });
      t.team_barrier();
      int pending = nk;
      while (pending > 0) {
        // a pending candidate is ready if no earlier pending one shares a particle
        Kokkos::parallel_for(Kokkos::TeamThreadRange(t, nk), [&](const int k) {
          bool r = state[k] == 0;
          for (int l = 0; r && l < k; ++l)
            if (state[l] == 0 && (pi[l] == pi[k] || pi[l] == pj[k] || pj[l] == pi[k] || pj[l] == pj[k])) r = false;
          ready[k] = r ? 1 : 0;
        });
        t.team_barrier();
        Kokkos::parallel_for(Kokkos::TeamThreadRange(t, nk), [&](const int k) {
          if (!ready[k]) return;
          RandomStream rng = m.stream(c, c0 + static_cast<std::uint64_t>(k));
          int a, q;
          CollisionPairOp::select(rng, n, a, q);  // advances the stream as in the serial kernel
          bool accepted;
          sgv[k] = m.candidate(pi[k], pj[k], max0, rng, accepted);
          state[k] = accepted ? 1 : 2;
        });
        t.team_barrier();
        int still = 0;
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(t, nk), [&](const int k, int& st) { st += state[k] == 0 ? 1 : 0; }, still);
        pending = still;
      }
      std::uint64_t ca = 0, cu = 0;
      double cm = max0;
      Kokkos::parallel_reduce(
          Kokkos::TeamThreadRange(t, nk), [&](const int k, std::uint64_t& x) { x += state[k] == 1 ? 1 : 0; }, ca);
      Kokkos::parallel_reduce(
          Kokkos::TeamThreadRange(t, nk), [&](const int k, std::uint64_t& x) { x += sgv[k] > max0 ? 1 : 0; }, cu);
      Kokkos::parallel_reduce(
          Kokkos::TeamThreadRange(t, nk), [&](const int k, double& x) { x = sgv[k] > x ? sgv[k] : x; },
          Kokkos::Max<double>(cm));
      n_acc += ca;
      n_upd += cu;
      max_new = cm > max_new ? cm : max_new;
      t.team_barrier();  // scratch is reused by the next chunk
    }
    Kokkos::single(Kokkos::PerTeam(t), [&]() {
      sg_max(c) = max_new;
      acc(c) = n_acc;
      upd(c) = n_upd;
    });
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

// --- Particle migration between ranks (M6.3) -----------------------------------

// Destination rank of every active particle that is not in an owned cell.
struct MigrantScan {
  using value_type = std::size_t;
  ParticleStorage p;
  DeviceGrid grid;
  Kokkos::View<const int*> plane_owner;
  Kokkos::View<int*> dest;
  Kokkos::View<std::size_t*> index;
  int me;
  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t i, std::size_t& update, const bool final) const {
    int d = -1;
    if (p.status(i) == static_cast<std::uint8_t>(ParticleStatus::Active)) {
      const int c = grid.cell_of(p.position(i));
      const int o = plane_owner(c / (grid.nx * grid.ny));
      if (o != me) d = o;
    }
    if (final) {
      dest(i) = d;
      index(i) = update;
    }
    if (d >= 0) ++update;
  }
};

struct MigrantPack {
  ParticleStorage p;
  Kokkos::View<const int*> dest;
  Kokkos::View<const std::size_t*> index;
  Kokkos::View<MigrantRecord*> buf;
  bool drop;  // only remove (set_particles)
  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t i) const {
    const int d = dest(i);
    if (d < 0) return;
    if (!drop) {
      MigrantRecord r;
      r.x = p.x(i);
      r.y = p.y(i);
      r.z = p.z(i);
      r.vx = p.vx(i);
      r.vy = p.vy(i);
      r.vz = p.vz(i);
      r.weight = p.weight(i);
      r.e_rot = p.e_rot(i);
      r.e_vib = p.e_vib(i);
      r.id = p.particle_id(i);
      r.species = p.species(i);
      r.interactions = p.interactions(i);
      r.dest = d;
      r.slot = static_cast<int>(i);
      buf(index(i)) = r;
    }
    p.status(i) = static_cast<std::uint8_t>(ParticleStatus::Free);  // left this rank (not counted)
  }
};

struct MigrantRetire {
  ParticleStorage p;
  Kokkos::View<const MigrantRecord*> buf;
  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t k) const {
    p.status(static_cast<std::size_t>(buf(k).slot)) = static_cast<std::uint8_t>(ParticleStatus::Free);
  }
};

struct MigrantUnpack {
  ParticleStorage p;
  Kokkos::View<double*> dt_rem;
  Kokkos::View<const MigrantRecord*> buf;
  std::size_t base;
  double dt;
  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t k) const {
    const MigrantRecord& r = buf(k);
    const std::size_t i = base + k;
    p.x(i) = r.x;
    p.y(i) = r.y;
    p.z(i) = r.z;
    p.vx(i) = r.vx;
    p.vy(i) = r.vy;
    p.vz(i) = r.vz;
    p.weight(i) = r.weight;
    p.e_rot(i) = r.e_rot;
    p.e_vib(i) = r.e_vib;
    p.particle_id(i) = r.id;
    p.species(i) = r.species;
    p.interactions(i) = r.interactions;
    p.status(i) = static_cast<std::uint8_t>(ParticleStatus::Active);
    dt_rem(i) = dt;
  }
};

// Contiguous split of the planes [0, w.size()) into `parts` ranges of about
// equal weight; returns parts + 1 bounds. Every part gets at least one plane
// if there are enough planes.
std::vector<int> split_planes(const std::vector<double>& w, int parts) {
  const int nz = static_cast<int>(w.size());
  std::vector<double> cum(w.size() + 1, 0.0);
  for (std::size_t k = 0; k < w.size(); ++k) cum[k + 1] = cum[k] + std::max(0.0, w[k]);
  std::vector<int> b(static_cast<std::size_t>(parts) + 1, 0);
  b[static_cast<std::size_t>(parts)] = nz;
  const double total = cum.back();
  for (int r = 1; r < parts; ++r) {
    int k;
    if (total > 0.0) {
      const double target = total * r / parts;
      k = static_cast<int>(std::lower_bound(cum.begin(), cum.end(), target) - cum.begin());
      // the bound closer to the target
      if (k > 0 && target - cum[static_cast<std::size_t>(k - 1)] < cum[static_cast<std::size_t>(k)] - target) --k;
    } else {
      k = static_cast<int>(static_cast<long long>(nz) * r / parts);
    }
    const int lo = b[static_cast<std::size_t>(r - 1)] + (nz >= parts ? 1 : 0);
    const int hi = nz - (nz >= parts ? parts - r : 0);
    b[static_cast<std::size_t>(r)] = std::clamp(k, lo, std::max(lo, hi));
  }
  return b;
}


// Cell-state moments (dsmc.dt_DSMCState_out): one thread per cell sums its
// particles in list order (deterministic, no atomics) and adds the sums to
// the window accumulator. Columns: kCellMom global moments, then per species
// (count, sum e_vib).
inline constexpr int kCellN = 0, kCellM = 1, kCellMV = 2, kCellMV2 = 5, kCellErot = 6, kCellRotDof = 7,
                     kCellColl = 8, kCellMom = 9;
struct CellMoments {
  ParticleStorage p;
  Kokkos::View<const int*> offset, list;
  Kokkos::View<const double*> mass;
  Kokkos::View<const SpeciesModes*> modes;
  Kokkos::View<const std::uint64_t*> acc;
  Kokkos::View<double**, Kokkos::LayoutLeft> sum;
  int nsp;
  KOKKOS_INLINE_FUNCTION void operator()(const int c) const {
    double s[kCellMom + 2 * kMaxDsmcSpecies] = {};
    for (int j = offset(c); j < offset(c + 1); ++j) {
      const int i = list(j);
      const auto sp = static_cast<int>(p.species(i));
      const double m = mass(sp);
      const Vec3 v = p.velocity(i);
      s[kCellN] += 1.0;
      s[kCellM] += m;
      s[kCellMV + 0] += m * v.x;
      s[kCellMV + 1] += m * v.y;
      s[kCellMV + 2] += m * v.z;
      s[kCellMV2] += m * norm2(v);
      s[kCellErot] += p.e_rot(i);
      s[kCellRotDof] += modes(sp).rot_dof;
      s[kCellMom + 2 * sp] += 1.0;
      s[kCellMom + 2 * sp + 1] += p.e_vib(i);
    }
    s[kCellColl] = static_cast<double>(acc(c));
    for (int k = 0; k < kCellMom + 2 * nsp; ++k) sum(c, k) += s[k];
  }
};

struct MaxSpeed {
  ParticleStorage p;
  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t i, double& m) const {
    if (p.status(i) != static_cast<std::uint8_t>(ParticleStatus::Active)) return;
    const double s = norm(p.velocity(i));
    if (s > m) m = s;
  }
};

}  // namespace

// ============================================================================

DsmcSettings resolve_dsmc_settings(const SimulationConfig& cfg, const Domain& domain,
                                   std::optional<double> cell_size_override_m) {
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
  if (cell_size_override_m) s.cell_size_m = *cell_size_override_m;
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
  std::size_t n = 0;         // occupied slots (active + ended, not yet compacted)
  std::size_t n_active = 0;  // active particles
  std::uint64_t next_id = 0;
  std::uint64_t step_index = 0;
  ParticleStorage cur, alt;
  Kokkos::View<double*> dt_cur, dt_alt;
  Kokkos::View<double**, Kokkos::LayoutLeft> tally;
  Kokkos::View<std::size_t*> compact_index;
  Kokkos::View<int*> cell_of_particle, list, list_unsorted;
  Kokkos::View<std::uint64_t*> list_key;  // GPU backends, several ranks  // list_unsorted: GPU backends only
  int offset_total = 0;

  Kokkos::View<int*> count, offset, cursor;
  Kokkos::View<double*> sg_max, remainder, count_sum;
  Kokkos::View<std::uint64_t*> cand, acc, upd;
  Kokkos::View<std::uint64_t*> pair_acc;
  Kokkos::View<double**, Kokkos::LayoutLeft> surface;
  std::vector<double> species_count_sum, species_E_sum;
  std::vector<Vec3> species_P_sum;
  std::vector<std::uint64_t> species_injected;

  // cell-state sampling (dsmc.dt_DSMCState_out)
  bool state_sampling = false, last_sample = false;
  std::uint64_t state_steps = 0, state_first_step = 0, state_index = 0;
  Kokkos::View<double**, Kokkos::LayoutLeft> state_sum;

  DsmcCounters counters;
  std::array<std::vector<double>, kHistoryQuantities> series;
  std::uint64_t sampled = 0;
  std::vector<double> particles_sampled;  // active particles of this rank per sampled step
  double max_speed = 0.0;
  std::uint64_t sampled_cand = 0, sampled_acc = 0, sampled_particle_steps = 0;
  double total_particle_steps = 0.0;
  Profiler prof;

  // --- Domain decomposition (M6.3): rank `me` owns the cell planes
  // [plane_bounds[me], plane_bounds[me + 1]) along z = cells [c_lo, c_hi).
  int nranks = 1, me = 0;
  bool decomposed = false;            // nranks > 1
  std::vector<int> plane_bounds;
  int c_lo = 0, c_hi = 0;
  Kokkos::View<int*> plane_owner;     // per z plane
  Kokkos::View<int*> migrant_dest, foreign;
  Kokkos::View<MigrantRecord*> migrant_buf, recv_dev, move_migrants;
  Kokkos::View<int*> move_migrant_count;
  // pinned host staging (fast device copies; HostSpace on host backends)
  Kokkos::View<MigrantRecord*, Kokkos::SharedHostPinnedSpace> migrant_host, send_host, recv_host;
  std::vector<std::uint32_t> order;
  DsmcDecomposition decomposition;

  int planes() const { return grid.grid.nz; }
  int cells_per_plane() const { return grid.grid.nx * grid.grid.ny; }

  void set_partition(const std::vector<int>& bounds) {
    plane_bounds = bounds;
    c_lo = bounds[static_cast<std::size_t>(me)] * cells_per_plane();
    c_hi = bounds[static_cast<std::size_t>(me) + 1] * cells_per_plane();
    auto h = Kokkos::create_mirror_view(plane_owner);
    for (int r = 0; r < nranks; ++r)
      for (int k = bounds[static_cast<std::size_t>(r)]; k < bounds[static_cast<std::size_t>(r) + 1]; ++k) h(k) = r;
    Kokkos::deep_copy(plane_owner, h);
  }

  // Weight of each plane on all ranks: sum over its cells of a per-cell value
  // (host copy of the owned cells; the sum over ranks is exact since every
  // plane has a single owner).
  template <class View>
  std::vector<double> plane_sums(const View& v, bool owned_only) const {
    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v);
    std::vector<double> w(static_cast<std::size_t>(planes()), 0.0);
    const int cpp = cells_per_plane();
    const int lo = owned_only ? c_lo : 0, hi = owned_only ? c_hi : grid.grid.num_cells();
    for (int c = lo; c < hi; ++c) w[static_cast<std::size_t>(c / cpp)] += static_cast<double>(h(c));
    return w;
  }

  // Bounds computed on rank 0 and broadcast (identical on all ranks).
  std::vector<int> agreed_split(const std::vector<double>& w) const {
    std::vector<int> b = split_planes(w, nranks);
    comm::broadcast(b.data(), b.size() * sizeof(int));
    return b;
  }

  static double imbalance(const std::vector<double>& w, const std::vector<int>& b) {
    const int p = static_cast<int>(b.size()) - 1;
    double total = 0.0, mx = 0.0;
    for (int r = 0; r < p; ++r) {
      double l = 0.0;
      for (int k = b[static_cast<std::size_t>(r)]; k < b[static_cast<std::size_t>(r) + 1]; ++k)
        l += w[static_cast<std::size_t>(k)];
      total += l;
      mx = std::max(mx, l);
    }
    return total > 0.0 ? mx * p / total : 1.0;
  }

  // Sends every active particle that is not in an owned cell to the owner
  // (drop: removes them without sending, set_particles). Full scan over the
  // particles: used after the fill, a change of the partition and as the
  // fallback of migrate_after_move().
  void migrate(bool drop = false) {
    if (!decomposed) return;
    Profiler::Scope ps(prof, "dsmc.migrate");
    if (migrant_dest.extent(0) < capacity) migrant_dest = Kokkos::View<int*>("dsmc_migrant_dest", capacity);
    std::size_t m = 0;
    if (n > 0) {
      Profiler::Scope pq(prof, "dsmc.migrate.scan");
      Kokkos::parallel_scan("vleo::dsmc_migrant_scan", policy(0, n),
                            MigrantScan{cur, grid.grid, plane_owner, migrant_dest, compact_index, me}, m);
    }
    if (m > migrant_buf.extent(0))
      migrant_buf = Kokkos::View<MigrantRecord*>(Kokkos::view_alloc(Kokkos::WithoutInitializing, "dsmc_migrants"),
                                                 m + m / 2 + 64);
    if (m > 0) {
      Profiler::Scope pq(prof, "dsmc.migrate.pack");
      Kokkos::parallel_for("vleo::dsmc_migrant_pack", policy(0, n),
                           MigrantPack{cur, migrant_dest, compact_index, migrant_buf, drop});
    }
    n_active -= m;
    if (drop) {
      Kokkos::fence();
      return;
    }
    exchange(migrant_buf, m);
  }

  // After the move (M6.7): the move kernel has recorded the leaving particles,
  // so no scan over all particles is needed. Falls back to migrate() if the
  // record buffer overflowed (it then grows for the next steps).
  void migrate_after_move() {
    if (!decomposed) return;
    int m = 0;
    Kokkos::deep_copy(m, Kokkos::subview(move_migrant_count, 0));
    Kokkos::deep_copy(move_migrant_count, 0);
    const auto mm = static_cast<std::size_t>(m);
    if (mm > move_migrants.extent(0)) {
      move_migrants = Kokkos::View<MigrantRecord*>(
          Kokkos::view_alloc(Kokkos::WithoutInitializing, "dsmc_move_migrants"), mm + mm / 2 + 1024);
      ++migrate_fallbacks;
      migrate();  // the particles are still Active: the scan finds all of them
      return;
    }
    Profiler::Scope ps(prof, "dsmc.migrate");
    if (mm > 0) {
      Profiler::Scope pq(prof, "dsmc.migrate.pack");
      Kokkos::parallel_for("vleo::dsmc_migrant_retire", policy(0, mm), MigrantRetire{cur, move_migrants});
    }
    n_active -= mm;
    exchange(move_migrants, mm);
  }

  // Exchange of m records (device buffer) through pinned host memory. The
  // records are sent in the order (destination, storage index), so the
  // receivers append them deterministically.
  void exchange(const Kokkos::View<MigrantRecord*>& dev, std::size_t m) {
    std::vector<std::size_t> send_bytes(static_cast<std::size_t>(nranks), 0);
    {
      Profiler::Scope pq(prof, "dsmc.migrate.to_host");
      if (m > migrant_host.extent(0)) {
        const std::size_t cap = m + m / 2 + 1024;
        migrant_host = Kokkos::View<MigrantRecord*, Kokkos::SharedHostPinnedSpace>(
            Kokkos::view_alloc(Kokkos::WithoutInitializing, "dsmc_migrants_host"), cap);
        send_host = Kokkos::View<MigrantRecord*, Kokkos::SharedHostPinnedSpace>(
            Kokkos::view_alloc(Kokkos::WithoutInitializing, "dsmc_send_host"), cap);
      }
      if (m > 0) {
        const auto range = Kokkos::make_pair(std::size_t{0}, m);
        Kokkos::deep_copy(Kokkos::subview(migrant_host, range), Kokkos::subview(dev, range));
        order.resize(m);
        for (std::size_t k = 0; k < m; ++k) order[k] = static_cast<std::uint32_t>(k);
        const MigrantRecord* h = migrant_host.data();
        std::sort(order.begin(), order.end(), [h](std::uint32_t a, std::uint32_t b) {
          return h[a].dest != h[b].dest ? h[a].dest < h[b].dest : h[a].slot < h[b].slot;
        });
        for (std::size_t k = 0; k < m; ++k) {
          send_host(k) = h[order[k]];
          send_bytes[static_cast<std::size_t>(h[order[k]].dest)] += sizeof(MigrantRecord);
        }
      }
    }
    std::size_t total = 0;
    {
      Profiler::Scope pm(prof, "dsmc.mpi");
      const std::vector<std::size_t> recv_bytes = comm::exchange_counts(send_bytes);
      std::size_t bytes = 0;
      for (std::size_t b : recv_bytes) bytes += b;
      total = bytes / sizeof(MigrantRecord);
      if (total > recv_host.extent(0)) {
        const std::size_t cap = total + total / 2 + 1024;
        recv_host = Kokkos::View<MigrantRecord*, Kokkos::SharedHostPinnedSpace>(
            Kokkos::view_alloc(Kokkos::WithoutInitializing, "dsmc_received_host"), cap);
        recv_dev = Kokkos::View<MigrantRecord*>(Kokkos::view_alloc(Kokkos::WithoutInitializing, "dsmc_received"), cap);
      }
      comm::alltoallv(send_host.data(), send_bytes, recv_host.data(), recv_bytes);
    }
    migrated += m;
    if (total == 0) return;
    Profiler::Scope pq(prof, "dsmc.migrate.from_host");
    const auto range = Kokkos::make_pair(std::size_t{0}, total);
    const auto dr = Kokkos::subview(recv_dev, range);
    Kokkos::deep_copy(dr, Kokkos::subview(recv_host, range));
    ensure_capacity(n + total);
    Kokkos::parallel_for("vleo::dsmc_migrant_unpack", policy(0, total),
                         MigrantUnpack{cur, dt_cur, dr, n, set.timestep_s});
    n += total;
    n_active += total;
  }
  std::uint64_t migrated = 0;  // particles sent by this rank
  std::uint64_t migrate_fallbacks = 0;

  void init_partition() {
    nranks = comm::size();
    me = comm::rank();
    decomposed = nranks > 1;
    plane_owner = Kokkos::View<int*>("dsmc_plane_owner", static_cast<std::size_t>(planes()));
    foreign = Kokkos::View<int*>("dsmc_foreign", 1);
    move_migrant_count = Kokkos::View<int*>("dsmc_move_migrant_count", 1);
    if (decomposed)
      move_migrants =
          // small on purpose: the first steps overflow and exercise the fallback
          // (multi-rank tests); the buffer then grows to the actual need
          Kokkos::View<MigrantRecord*>(Kokkos::view_alloc(Kokkos::WithoutInitializing, "dsmc_move_migrants"), 64);
    // initial split by gas volume per plane (no particles yet)
    const std::vector<double> w = plane_sums(grid.gas_fraction, false);
    set_partition(decomposed ? agreed_split(w) : std::vector<int>{0, planes()});
    decomposition.ranks = nranks;
    decomposition.initial_bounds = plane_bounds;
    decomposition.bounds = plane_bounds;
  }

  // Load balancing at the end of the warm-up (M6.5): planes re-split by the
  // particle count of the last binning. The cell state (sigma g max, NTC
  // remainder, sampled populations) moves with the cells and the particles
  // migrate to the new owners, so the run continues exactly as before.
  void rebalance() {
    if (!decomposed) return;
    Profiler::Scope ps(prof, "dsmc.rebalance");
    std::vector<double> w = plane_sums(count, true);
    comm::sum(w.data(), w.size());
    const std::vector<int> b = agreed_split(w);
    decomposition.imbalance_before = imbalance(w, plane_bounds);
    decomposition.imbalance_after = imbalance(w, b);
    decomposition.rebalanced = true;
    if (b == plane_bounds) return;
    auto move_cells = [&](auto& view) {
      auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), view);
      const int nc = grid.grid.num_cells();
      for (int c = 0; c < nc; ++c)
        if (c < c_lo || c >= c_hi) h(c) = 0;
      comm::sum(h.data(), static_cast<std::size_t>(nc));  // one non-zero contribution per cell: exact
      Kokkos::deep_copy(view, h);
    };
    move_cells(sg_max);
    move_cells(remainder);
    move_cells(count_sum);
    set_partition(b);
    decomposition.bounds = b;
    migrate();
    compact();
  }

  // Additive statistics. Each rank accumulates its own contributions during
  // the run (no communication per step, M6.7); totals() sums them over the
  // ranks once (collective). Steps, injection counts and the fill count are
  // the same on every rank and are not summed.
  struct Totals {
    DsmcCounters counters;
    std::array<std::vector<double>, kHistoryQuantities> series;
    std::vector<double> particles;  // per sampled step
    std::vector<double> species_count, species_E;
    std::vector<Vec3> species_P;
    double total_particle_steps = 0.0;
    std::uint64_t sampled_cand = 0, sampled_acc = 0, sampled_particle_steps = 0;
  };
  Totals totals() const {
    Totals t;
    t.counters = counters;
    t.series = series;
    t.particles = particles_sampled;
    t.species_count = species_count_sum;
    t.species_E = species_E_sum;
    t.species_P = species_P_sum;
    t.total_particle_steps = total_particle_steps;
    t.sampled_cand = sampled_cand;
    t.sampled_acc = sampled_acc;
    t.sampled_particle_steps = sampled_particle_steps;
    if (!decomposed) return t;
    DsmcCounters& c = t.counters;
    std::uint64_t u[11] = {c.exited, c.truncated, c.lost_inside_body, c.invalid, c.surface_interactions,
                           c.collision_candidates, c.collisions, c.sigma_g_max_updates, t.sampled_cand, t.sampled_acc,
                           t.sampled_particle_steps};
    comm::sum(u, 11);
    c.exited = u[0];
    c.truncated = u[1];
    c.lost_inside_body = u[2];
    c.invalid = u[3];
    c.surface_interactions = u[4];
    c.collision_candidates = u[5];
    c.collisions = u[6];
    c.sigma_g_max_updates = u[7];
    t.sampled_cand = u[8];
    t.sampled_acc = u[9];
    t.sampled_particle_steps = u[10];
    for (auto& v : t.series) comm::sum(v.data(), v.size());  // same length on all ranks
    comm::sum(t.particles.data(), t.particles.size());
    comm::sum(t.species_count.data(), t.species_count.size());
    comm::sum(t.species_E.data(), t.species_E.size());
    static_assert(sizeof(Vec3) == 3 * sizeof(double));
    comm::sum(reinterpret_cast<double*>(t.species_P.data()), 3 * t.species_P.size());
    t.total_particle_steps = comm::sum(t.total_particle_steps);
    return t;
  }

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
    if constexpr (!kHostExecution) {
      list_unsorted = Kokkos::View<int*>("dsmc_cell_list_unsorted", cap);
      if (comm::size() > 1)
        list_key = Kokkos::View<std::uint64_t*>(Kokkos::view_alloc(Kokkos::WithoutInitializing, "dsmc_cell_list_key"), cap);
    }
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
    n_active = alive;
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
    // several ranks: contiguous share of the injection indices (the particles
    // migrate to the owners of their cells after the move)
    const std::uint64_t j0 = total * static_cast<std::uint64_t>(me) / static_cast<std::uint64_t>(nranks);
    const std::uint64_t local = total * static_cast<std::uint64_t>(me + 1) / static_cast<std::uint64_t>(nranks) - j0;
    if (local > 0) {
      ensure_capacity(n + local);
      Kokkos::parallel_for("vleo::dsmc_inject", policy(0, local),
                           InjectKernel{cur, dt_cur, inflow, off, nch, n, next_id, cfg.simulation.random_seed,
                                        set.timestep_s, set.particle_weight, j0});
    }
    n += local;
    n_active += local;
    next_id += total;
    counters.injected += total;
  }

  void bin() {
    const auto owned = Kokkos::make_pair(static_cast<std::size_t>(c_lo), static_cast<std::size_t>(c_hi));
    {
      Profiler::Scope ps(prof, "dsmc.bin.histogram");
      Kokkos::deep_copy(Kokkos::subview(count, owned), 0);
      Kokkos::deep_copy(Kokkos::subview(cursor, owned), 0);
      Kokkos::parallel_for("vleo::dsmc_cell_index", policy(0, n),
                           CellIndexKernel{cur, grid.grid, cell_of_particle, count, c_lo, c_hi, foreign});
    }
    {
      Profiler::Scope ps(prof, "dsmc.bin.scan");
      offset_total = 0;  // number of binned (active) particles
      if (c_hi > c_lo)
        Kokkos::parallel_scan("vleo::dsmc_offsets", IntPolicy(c_lo, c_hi), OffsetScan{count, offset, c_hi},
                              offset_total);
    }
    {
      Profiler::Scope ps(prof, "dsmc.bin.fill");
      Kokkos::parallel_for("vleo::dsmc_fill_cells", policy(0, n),
                           FillCells{cell_of_particle, offset, cursor, kHostExecution ? list : list_unsorted,
                                     cur.particle_id, list_key});
    }
    {
      Profiler::Scope ps(prof, "dsmc.bin.sort");
      if constexpr (kHostExecution) {
        Kokkos::parallel_for("vleo::dsmc_sort_cells", IntPolicy(c_lo, c_hi),
                             SortCells{offset, list, cur.particle_id, decomposed});
      } else {
        Kokkos::parallel_for("vleo::dsmc_rank_sort", IntPolicy(0, offset_total),
                             RankSortEntries{offset, cell_of_particle, list_unsorted, list, list_key, decomposed});
      }
    }
  }

  void collide() {
    const int nc = c_hi - c_lo;  // owned cells
    if (nc > 0)
      Kokkos::parallel_for("vleo::dsmc_candidates", IntPolicy(c_lo, c_hi),
                         CellCandidates{offset, grid.collision_volume, sg_max, remainder, cand,
                                        set.particle_weight * set.timestep_s});
    const CollisionPairOp model{cur, pairs, modes, pair_acc, nsp, cfg.simulation.random_seed,
                               static_cast<std::uint32_t>(step_index)};
    const bool per_cell = set.collision_kernel == CollisionKernel::PerCell ||
                          (set.collision_kernel == CollisionKernel::Automatic && kHostExecution);
    if (nc == 0) {
      // no owned cells (more ranks than planes)
    } else if (per_cell) {
      Kokkos::parallel_for("vleo::dsmc_collide", IntPolicy(c_lo, c_hi),
                           CollideCellsSerial{model, offset, list, sg_max, cand, acc, upd});
    } else {
      // 32 threads per cell on GPUs (wave32 / warp); host backends (tests) choose themselves
      Kokkos::TeamPolicy<> tp = kHostExecution ? Kokkos::TeamPolicy<>(nc, Kokkos::AUTO) : Kokkos::TeamPolicy<>(nc, 32);
      tp.set_scratch_size(0, Kokkos::PerTeam(static_cast<int>(CollideCellsWaves::scratch_bytes())));
      Kokkos::parallel_for("vleo::dsmc_collide_waves", tp,
                           CollideCellsWaves{model, offset, list, sg_max, cand, acc, upd, c_lo});
    }
    std::uint64_t s[3] = {0, 0, 0};
    if (nc > 0)
      Kokkos::parallel_reduce("vleo::dsmc_collision_counts", IntPolicy(c_lo, c_hi), SumU64x3{3, cand, acc, upd}, s);
    counters.collision_candidates += s[0];
    counters.collisions += s[1];
    counters.sigma_g_max_updates += s[2];
    last_cand = s[0];
    last_acc = s[1];
  }
  std::uint64_t last_cand = 0, last_acc = 0;

  void step(bool sample) {
    if (set.boundary == DsmcBoundary::Open) {
      Profiler::Scope ps(prof, "dsmc.inject");
      inject();
    }
    total_particle_steps += static_cast<double>(n_active);
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
    if (decomposed) {
      mk.find_migrants = true;
      mk.grid = grid.grid;
      mk.plane_owner = plane_owner;
      mk.me = me;
      mk.migrants = move_migrants;
      mk.migrant_count = move_migrant_count;
    }
    {
      Profiler::Scope ps(prof, "dsmc.move");
      Kokkos::parallel_for("vleo::dsmc_move", policy(0, n), mk);
    }

    double s[StepReduce::kSlots] = {};
    if (n > 0 && sample) {
      Profiler::Scope ps(prof, "dsmc.step_reduce");
      StepReduce red;
      red.tally = tally;
      red.status = cur.status;
      red.species = cur.species;
      red.value_count = static_cast<std::size_t>(StepReduce::kGlobal + kSpeciesSlots * nsp);
      Kokkos::parallel_reduce("vleo::dsmc_step_reduce", policy(0, n), red, s);
    } else if (n > 0) {
      Profiler::Scope ps(prof, "dsmc.step_reduce");
      double c[StatusReduce::kSlots] = {};
      StatusReduce red;
      red.status = cur.status;
      red.tally = tally;
      Kokkos::parallel_reduce("vleo::dsmc_status_reduce", policy(0, n), red, c);
      for (int k = 0; k < kNumParticleStatus; ++k) s[kStepTallies + k] = c[k];
      s[kStepInteractions] = c[kNumParticleStatus];
    }
    n_active = static_cast<std::size_t>(s[kStepTallies + static_cast<int>(ParticleStatus::Active)]);
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

    migrate_after_move();
    // Compact only when more than 5 % of the slots are no longer active
    // (deterministic decision; ended particles are skipped meanwhile).
    if (n - n_active > n / 20) {
      Profiler::Scope ps(prof, "dsmc.compact");
      compact();
    }
    bin();
    {
      Profiler::Scope ps(prof, "dsmc.collide");
      collide();
    }

    if (state_sampling) {
      Profiler::Scope ps(prof, "dsmc.cell_state");
      if (c_hi > c_lo)  // owned cells (several ranks: the other cells have no particles here)
        Kokkos::parallel_for("vleo::dsmc_cell_state", IntPolicy(c_lo, c_hi),
                             CellMoments{cur, offset, list, mass, modes, acc, state_sum, nsp});
      ++state_steps;
    }
    last_sample = sample;

    if (sample) {
      Profiler::Scope ps(prof, "dsmc.sample");
      if (c_hi > c_lo)
        Kokkos::parallel_for("vleo::dsmc_count_sum", IntPolicy(c_lo, c_hi), AccumulateCounts{count, count_sum});
      double vmax = 0.0;
      if (n > 0)
        Kokkos::parallel_reduce("vleo::dsmc_max_speed", policy(0, n), MaxSpeed{cur}, Kokkos::Max<double>(vmax));
      max_speed = std::max(max_speed, vmax);
      particles_sampled.push_back(static_cast<double>(n_active));
      sampled_cand += last_cand;
      sampled_acc += last_acc;
      sampled_particle_steps += n_active;
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
  s.prof = Profiler(settings.profile || profiling_requested_by_environment());
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
  s.init_partition();
}

DsmcSimulation::~DsmcSimulation() = default;

void DsmcSimulation::fill_freestream() {
  Impl& s = *impl_;
  const Aabb box = s.inflow_host.domain.box;
  const Vec3 e = box.extent();
  const double expected = s.cfg.gas.number_density_m3 * e.x * e.y * e.z / s.set.particle_weight;
  RandomStream rng(s.cfg.simulation.random_seed, 0, 0, RngStream::InjectionCount);
  const auto total = static_cast<std::size_t>(std::floor(expected + rng.uniform()));
  // several ranks: contiguous share of the ids, then migration to the owners
  const std::size_t lo = total * static_cast<std::size_t>(s.me) / static_cast<std::size_t>(s.nranks);
  const std::size_t local = total * static_cast<std::size_t>(s.me + 1) / static_cast<std::size_t>(s.nranks) - lo;
  s.n = 0;
  s.ensure_capacity(local);
  std::vector<double> cdf, sigma;
  double acc = 0.0, sum = 0.0;
  for (const auto& sp : s.cfg.gas.species) sum += sp.mole_fraction;
  for (const auto& sp : s.cfg.gas.species) {
    acc += sp.mole_fraction / sum;
    cdf.push_back(acc);
    sigma.push_back(std::sqrt(constants::k_B * s.cfg.gas.translational_temperature_K / sp.mass_kg));
  }
  Kokkos::parallel_for("vleo::dsmc_fill", policy(0, local),
                       FillKernel{s.cur, s.dt_cur, s.geo, box, s.cfg.gas.bulk_velocity_body_mps,
                                  to_device(cdf, "fill_cdf"), to_device(sigma, "fill_sigma"), s.modes,
                                  s.cfg.gas.translational_temperature_K, s.next_id + lo,
                                  s.cfg.simulation.random_seed, s.set.timestep_s, s.set.particle_weight, s.has_closed});
  s.n = local;
  s.next_id += total;
  s.compact();
  std::uint64_t removed = local - s.n;
  if (s.decomposed) {
    removed = comm::sum(removed);
    s.migrate();
    s.compact();
  }
  s.counters.removed_inside_at_fill += removed;
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
  s.n_active = total;
  s.next_id += total;
  if (s.decomposed) {  // every rank keeps the particles in its own cells
    s.migrate(true);
    s.compact();
  }
  s.bin();
}

void DsmcSimulation::step(bool sample) { impl_->step(sample); }
std::size_t DsmcSimulation::num_particles() const {
  return impl_->decomposed ? comm::sum(static_cast<std::uint64_t>(impl_->n_active)) : impl_->n_active;
}
std::size_t DsmcSimulation::num_local_particles() const { return impl_->n_active; }
void DsmcSimulation::rebalance() { impl_->rebalance(); }
void DsmcSimulation::record_phase(const std::string& name, double seconds) { impl_->prof.add(name, seconds); }
const DsmcDecomposition& DsmcSimulation::decomposition() const { return impl_->decomposition; }

void DsmcSimulation::compact() {
  impl_->compact();
  impl_->bin();
}
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
  list.assign(hl.data(), hl.data() + ho(ho.extent(0) - 1));
}

DsmcResult DsmcSimulation::result() const {
  const Impl& s = *impl_;
  if (s.decomposed) {  // migration must have left every particle in a cell of its rank
    int lost = 0;
    Kokkos::deep_copy(lost, Kokkos::subview(s.foreign, 0));
    if (comm::any(lost > 0)) throw std::logic_error("DSMC: particles outside the cells of their rank after migration");
  }
  const Impl::Totals tot = s.totals();
  DsmcResult r;
  r.settings = s.set;
  r.counters = tot.counters;
  r.grid = s.grid.report;
  r.critical_value = critical_value(s.cfg.statistics.confidence_level);
  const double z = r.critical_value;

  std::array<Estimate, kHistoryQuantities> base{};
  for (std::size_t q = 0; q < base.size(); ++q) {
    r.series[q] = analyze_series(tot.series[q], s.set.blocks);
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
    // separate host copy: on host backends the mirror would alias the tally
    Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace> h("dsmc_surface_host", s.surface.extent(0),
                                                                   s.surface.extent(1));
    Kokkos::deep_copy(h, s.surface);
    comm::sum(h.data(), h.size());
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
  d.max_cfl = comm::max(s.max_speed) * s.set.timestep_s / dx_min;
  d.free_stream_collisions_per_particle_per_step = s.set.timestep_s / s.set.tau_inf_s;
  if (s.sampled > 0) {
    double psum = 0.0, pmin = 1e300, pmax = 0.0;
    for (double np : tot.particles) {
      psum += np;
      pmin = std::min(pmin, np);
      pmax = std::max(pmax, np);
    }
    d.particles_mean = psum / static_cast<double>(s.sampled);
    d.particles_min = pmin;
    d.particles_max = pmax;
    d.acceptance_ratio = tot.sampled_cand > 0 ? static_cast<double>(tot.sampled_acc) / static_cast<double>(tot.sampled_cand) : 0.0;
    d.collisions_per_particle_per_step =
        tot.sampled_particle_steps > 0 ? 2.0 * static_cast<double>(tot.sampled_acc) / static_cast<double>(tot.sampled_particle_steps)
                                     : 0.0;
    Kokkos::View<double*, Kokkos::HostSpace> cs("dsmc_count_sum_host", s.count_sum.extent(0));
    Kokkos::deep_copy(cs, s.count_sum);
    comm::sum(cs.data(), cs.size());  // one owner per cell: exact
    auto fr = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), s.grid.gas_fraction);
    std::int64_t gas = 0, below5 = 0, coarse = 0;
    double ppc = 0.0, min_lambda = Kokkos::infinity_v<double>, min_lambda_resolved = Kokkos::infinity_v<double>;
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
        if (fr(c) >= 0.25) min_lambda_resolved = std::min(min_lambda_resolved, lam);
        if (dx_max > lam) ++coarse;
      }
    }
    d.mean_particles_per_gas_cell = gas > 0 ? ppc / static_cast<double>(gas) : 0.0;
    d.fraction_cells_below_5 = gas > 0 ? static_cast<double>(below5) / static_cast<double>(gas) : 0.0;
    d.fraction_cells_dx_gt_lambda = gas > 0 ? static_cast<double>(coarse) / static_cast<double>(gas) : 0.0;
    d.min_local_lambda_m = std::isfinite(min_lambda) ? min_lambda : s.set.lambda_inf_m;
    d.min_local_lambda_resolved_m = std::isfinite(min_lambda_resolved) ? min_lambda_resolved : d.min_local_lambda_m;
  }

  r.profile = s.prof.phases();
  r.particle_steps = tot.total_particle_steps;
  r.decomposition = s.decomposition;
  r.decomposition.migrated = s.decomposed ? comm::sum(s.migrated) : 0;
  if (s.decomposed) {  // per-rank load and (with profiling) phase timings (M6.7)
    double local_mean = 0.0;
    for (double np : s.particles_sampled) local_mean += np;
    if (!s.particles_sampled.empty()) local_mean /= static_cast<double>(s.particles_sampled.size());
    const std::vector<char> pm = comm::allgather(&local_mean, sizeof(double));
    r.decomposition.rank_mean_particles.resize(static_cast<std::size_t>(s.nranks));
    std::memcpy(r.decomposition.rank_mean_particles.data(), pm.data(), pm.size());
    if (s.prof.enabled()) {
      std::string own;
      for (const auto& [name, entry] : s.prof.phases()) own += name + "\t" + std::to_string(entry.seconds) + "\n";
      for (const std::string& text : comm::allgather(own)) {
        std::map<std::string, double> m;
        std::size_t a = 0;
        while (a < text.size()) {
          const std::size_t tab = text.find('\t', a), nl = text.find('\n', a);
          if (tab == std::string::npos || nl == std::string::npos) break;
          m[text.substr(a, tab - a)] = std::stod(text.substr(tab + 1, nl - tab - 1));
          a = nl + 1;
        }
        r.decomposition.rank_profiles.push_back(std::move(m));
      }
    }
  }

  // Per-species diagnostics.
  {
    Kokkos::View<std::uint64_t*, Kokkos::HostSpace> pa("dsmc_pair_acc_host", s.pair_acc.extent(0));
    Kokkos::deep_copy(pa, s.pair_acc);
    comm::sum(pa.data(), pa.size());
    const double ns = s.sampled > 0 ? static_cast<double>(s.sampled) : 1.0;
    for (int q = 0; q < s.nsp; ++q) {
      DsmcSpeciesDiagnostics sd;
      sd.name = s.cfg.gas.species[static_cast<std::size_t>(q)].name;
      sd.mean_particles = tot.species_count[static_cast<std::size_t>(q)] / ns;
      sd.injected = s.species_injected[static_cast<std::size_t>(q)];
      sd.force_N = tot.species_P[static_cast<std::size_t>(q)] / ns;
      sd.heat_rate_W = tot.species_E[static_cast<std::size_t>(q)] / ns;
      for (int o = 0; o < s.nsp; ++o) sd.collisions_with.push_back(pa(std::min(q, o) * s.nsp + std::max(q, o)));
      r.species.push_back(sd);
    }
  }

  // Warnings.
  auto warn = [&r](const std::string& w) { r.warnings.push_back(w); };
  const DsmcCounters& c = tot.counters;
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

void DsmcSimulation::enable_cell_state_sampling() {
  Impl& s = *impl_;
  if (s.state_sampling) return;
  s.state_sum = Kokkos::View<double**, Kokkos::LayoutLeft>("dsmc_cell_state", s.grid.grid.num_cells(),
                                                          kCellMom + 2 * s.nsp);
  s.state_sampling = true;
  s.state_steps = 0;
  s.state_first_step = s.step_index;
}

std::uint64_t DsmcSimulation::cell_state_steps() const { return impl_->state_steps; }

DsmcCellState DsmcSimulation::take_cell_state() {
  Impl& s = *impl_;
  if (!s.state_sampling) throw std::logic_error("take_cell_state: cell-state sampling is not enabled");
  Kokkos::fence();
  const auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), s.state_sum);
  // Several ranks: every cell has a single owner (zero elsewhere), so the sum is exact.
  if (s.decomposed) comm::sum(h.data(), h.size());
  const auto f = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), s.grid.gas_fraction);
  const int nc = s.grid.grid.num_cells(), nsp = s.nsp;
  const auto ncs = static_cast<std::size_t>(nc);
  const double steps = static_cast<double>(s.state_steps);
  const double dt = s.set.timestep_s, W = s.set.particle_weight, kB = constants::k_B;
  const double v_cell = s.grid.grid.cell_volume();
  const Vec3 cs = s.grid.grid.cell_size;
  const double edge = std::max({cs.x, cs.y, cs.z});
  const double n_inf = s.cfg.gas.number_density_m3, lambda_inf = s.set.lambda_inf_m;
  const std::vector<SpeciesModes> modes = make_species_modes(s.cfg.gas);
  const double nan = std::numeric_limits<double>::quiet_NaN();

  DsmcCellState st;
  st.index = s.state_index;
  st.steps = s.state_steps;
  st.first_step = s.state_first_step;
  st.last_step = s.step_index > 0 ? s.step_index - 1 : 0;
  st.time_s = static_cast<double>(s.step_index) * dt;
  st.window_s = steps * dt;
  st.sampling = s.last_sample;
  st.grid = s.grid.grid;
  for (const auto& sp : s.cfg.gas.species) st.species.push_back(sp.name);
  auto vec = [&](double init) { return std::vector<double>(ncs, init); };
  st.gas_fraction = vec(0.0);
  st.particles = vec(0.0);
  st.number_density_m3 = vec(0.0);
  st.mass_density_kg_m3 = vec(0.0);
  st.velocity_mps.assign(ncs, Vec3(nan, nan, nan));
  st.temperature_trans_K = vec(nan);
  st.temperature_rot_K = vec(nan);
  st.temperature_vib_K = vec(nan);
  st.temperature_K = vec(nan);
  st.pressure_Pa = vec(0.0);
  st.mach = vec(nan);
  st.mean_free_path_m = vec(nan);
  st.cell_size_over_lambda = vec(0.0);
  st.collision_frequency_per_s = vec(nan);
  st.species_number_density_m3.assign(static_cast<std::size_t>(nsp), vec(0.0));

  for (int c = 0; c < nc; ++c) {
    const auto u = static_cast<std::size_t>(c);
    const double frac = f(c);
    st.gas_fraction[u] = frac;
    const double N = h(c, kCellN);
    if (!(steps > 0.0) || !(frac > 0.0) || !(N > 0.0)) continue;
    const double v_gas = frac * v_cell;
    const double M = h(c, kCellM);
    st.particles[u] = N / steps;
    const double n = N / steps * W / v_gas;
    st.number_density_m3[u] = n;
    st.mass_density_kg_m3[u] = M / steps * W / v_gas;
    for (int q = 0; q < nsp; ++q)
      st.species_number_density_m3[static_cast<std::size_t>(q)][u] = h(c, kCellMom + 2 * q) / steps * W / v_gas;
    const Vec3 mv(h(c, kCellMV), h(c, kCellMV + 1), h(c, kCellMV + 2));
    const Vec3 vel = mv / M;
    st.velocity_mps[u] = vel;
    const double lambda = lambda_inf * n_inf / n;
    st.mean_free_path_m[u] = lambda;
    st.cell_size_over_lambda[u] = edge / lambda;
    st.collision_frequency_per_s[u] = 2.0 * h(c, kCellColl) / (N * dt);
    // Translational temperature from the second moment about the window mean
    // velocity. It needs a velocity spread: with fewer than two samples, or
    // samples of one molecule that did not collide (spread at round-off level),
    // the temperature is undefined (NaN).
    const double mv2 = h(c, kCellMV2);
    const double spread = mv2 - M * norm2(vel);
    if (N < 2.0 || !(spread > 1e-9 * mv2)) {
      st.pressure_Pa[u] = nan;
      continue;
    }
    const double t_tr = spread / (3.0 * kB * N);
    st.temperature_trans_K[u] = t_tr;
    st.pressure_Pa[u] = n * kB * t_tr;
    const double rot_dof = h(c, kCellRotDof);
    const double z_rot = rot_dof / N;
    if (rot_dof > 0.0) st.temperature_rot_K[u] = 2.0 * h(c, kCellErot) / (kB * rot_dof);
    // vibration: harmonic oscillator per species, T_v = theta_v / ln(1 + k theta_v / <e_vib>)
    double nv = 0.0, tv_sum = 0.0, zv_t = 0.0, z_vib = 0.0;
    for (int q = 0; q < nsp; ++q) {
      const SpeciesModes& md = modes[static_cast<std::size_t>(q)];
      const double nq = h(c, kCellMom + 2 * q);
      if (!md.vib || !(nq > 0.0) || !(md.theta_v > 0.0)) continue;
      const double e = h(c, kCellMom + 2 * q + 1) / nq;
      const double tv = e > 0.0 ? md.theta_v / std::log1p(kB * md.theta_v / e) : 0.0;
      const double x = tv > 0.0 ? md.theta_v / tv : 0.0;
      const double zv = x > 0.0 ? 2.0 * x / std::expm1(x) : 0.0;
      nv += nq;
      tv_sum += nq * tv;
      z_vib += nq / N * zv;
      zv_t += nq / N * zv * tv;
    }
    if (nv > 0.0) st.temperature_vib_K[u] = tv_sum / nv;
    const double t_rot = rot_dof > 0.0 ? st.temperature_rot_K[u] : 0.0;
    st.temperature_K[u] = (3.0 * t_tr + z_rot * t_rot + zv_t) / (3.0 + z_rot + z_vib);
    const double m_mean = M / N;
    const double gamma = (5.0 + z_rot) / (3.0 + z_rot);
    if (t_tr > 0.0) st.mach[u] = norm(vel) / std::sqrt(gamma * kB * t_tr / m_mean);
  }

  Kokkos::deep_copy(s.state_sum, 0.0);
  s.state_steps = 0;
  s.state_first_step = s.step_index;
  ++s.state_index;
  return st;
}

DsmcResult run_dsmc(const SimulationConfig& cfg, const Geometry& geometry, const InflowModel& inflow,
                    const DsmcSettings& settings, const DsmcProgressCallback& progress,
                    const DsmcStateCallback& on_state) {
  const auto t0 = std::chrono::steady_clock::now();
  const bool profile = settings.profile || profiling_requested_by_environment();
  auto lap = [profile, last = t0]() mutable {  // fenced interval since the previous lap (profiling only)
    if (!profile) return 0.0;
    Kokkos::fence();
    const auto now = std::chrono::steady_clock::now();
    const double dt = std::chrono::duration<double>(now - last).count();
    last = now;
    return dt;
  };
  DsmcSimulation sim(cfg, geometry, inflow, settings);
  sim.record_phase("run.setup", lap());
  if (settings.initial_fill) sim.fill_freestream();
  sim.record_phase("run.fill", lap());
  const std::uint64_t total = settings.warmup_steps + settings.sample_steps;
  const std::uint64_t report = std::max<std::uint64_t>(1, total / 100);  // progress every 1 % (simulation.out)
  const bool states = on_state && cfg.dsmc.dt_DSMCState_out.has_value();
  const double interval = states ? *cfg.dsmc.dt_DSMCState_out : 0.0;
  std::uint64_t windows_done = 0;
  if (states) sim.enable_cell_state_sampling();
  for (std::uint64_t k = 0; k < total; ++k) {
    const bool sample = k >= settings.warmup_steps;
    sim.step(sample);
    if (k + 1 == settings.warmup_steps && settings.load_balance) sim.rebalance();
    if (states) {
      // window m ends with the first step whose end time reaches (m + 1) * interval
      const double t_end = static_cast<double>(k + 1) * settings.timestep_s;
      const double t_next = static_cast<double>(windows_done + 1) * interval;
      if (t_end >= t_next * (1.0 - 1e-12) || k + 1 == total) {
        DsmcCellState st = sim.take_cell_state();  // collective with several ranks
        if (comm::root()) on_state(st);
        windows_done = static_cast<std::uint64_t>(std::floor(t_end / interval * (1.0 + 1e-12)));
      }
    }
    // report steps are the same on every rank (num_particles() is collective)
    if ((k + 1) % report == 0 || k + 1 == total) {
      const std::size_t np = progress || comm::size() > 1 ? sim.num_particles() : 0;
      const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      if (progress) progress({k + 1, total, sample, np, el});
    }
  }
  Kokkos::fence();
  sim.record_phase("run.steps", lap());
  DsmcResult r = sim.result();
  if (profile) r.profile["run.result"] = Profiler::Entry{lap(), 1};
  r.wall_time_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  return r;
}

DsmcSettings refine_cell_size_with_pilot(const SimulationConfig& cfg, const Geometry& geometry,
                                         const InflowModel& inflow, const DsmcSettings& settings,
                                         CellSizePilotReport& report, const CellSizePilotOptions& options) {
  report = CellSizePilotReport{};
  report.initial_cell_size_m = settings.cell_size_m;
  report.cell_size_m = settings.cell_size_m;
  if (!settings.cell_size_automatic) return settings;
  const auto t0 = std::chrono::steady_clock::now();
  const Domain& domain = inflow.domain;

  // smallest cell size the grid allows (GridOptions::max_cells), with margin
  const Vec3 e = domain.box.extent();
  const double dx_floor =
      1.05 * std::cbrt(e.x * e.y * e.z / static_cast<double>(std::max<std::int64_t>(1, settings.grid.max_cells)));

  SimulationConfig pc = cfg;  // pilot: few particles, automatic (then halved) warm-up, short sampling
  pc.dsmc.particles_per_cell_target = std::min(cfg.dsmc.particles_per_cell_target, options.particles_per_cell);
  pc.dsmc.warmup_steps.reset();
  pc.dsmc.sample_steps = options.sample_steps;
  pc.dsmc.blocks = std::min<std::uint32_t>(cfg.dsmc.blocks, 10);
  pc.output.save_surface_fields = false;
  pc.dsmc.dt_DSMCState_out.reset();

  double dx = settings.cell_size_m;
  double lambda_min = settings.lambda_inf_m;
  for (int k = 0; k < options.max_pilots; ++k) {
    DsmcSettings ps = resolve_dsmc_settings(pc, domain, dx);
    ps.grid = settings.grid;
    ps.warmup_steps = std::max<std::uint64_t>(1, ps.warmup_steps / 2);
    ps.profile = false;
    const DsmcResult r = run_dsmc(pc, geometry, inflow, ps);
    ++report.pilots;
    report.steps += ps.warmup_steps + ps.sample_steps;
    lambda_min = std::min(lambda_min, r.diagnostics.min_local_lambda_resolved_m);
    // the largest edge counts (cells divide the domain exactly)
    const double edge = std::max({r.diagnostics.cell_dx_m, r.diagnostics.cell_dy_m, r.diagnostics.cell_dz_m});
    const double target = options.lambda_fraction * lambda_min;
    if (edge <= target) break;  // already resolved
    double next = std::min(dx, target);
    if (next < dx_floor) {
      next = dx_floor;
      report.limited_by_max_cells = true;
    }
    const bool large_change = next < 0.8 * dx;
    dx = next;
    // the pilot underestimates the peak density on a coarse grid: re-measure
    // once on the refined grid if the cell size changed substantially
    if (!large_change || report.limited_by_max_cells) break;
  }
  report.performed = true;
  report.min_local_lambda_m = lambda_min;
  report.cell_size_m = dx;
  report.wall_time_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  DsmcSettings out = resolve_dsmc_settings(cfg, domain, dx);
  out.grid = settings.grid;
  out.profile = settings.profile;
  out.collision_kernel = settings.collision_kernel;
  out.initial_fill = settings.initial_fill;
  out.boundary = settings.boundary;
  out.max_interactions_per_step = settings.max_interactions_per_step;
  return out;
}

}  // namespace vleo

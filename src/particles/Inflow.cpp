#include "vleo/particles/Inflow.hpp"

#include <cmath>
#include <sstream>

#include "vleo/core/PhysicalConstants.hpp"
#include "vleo/geometry/Mesh.hpp"

namespace vleo {

namespace {
constexpr double kSqrtPi = 1.77245385090551602730;
}

double flux_factor(double s) {
  // 1 + erf(s) = erfc(-s). For s < 0 both terms are ~exp(-s^2) and nearly
  // cancel; erfc keeps full relative accuracy in the tail, so the difference
  // loses only log10(2 s^2) digits (e.g. 2 digits at s = -8).
  return std::exp(-s * s) + kSqrtPi * s * std::erfc(-s);
}

double number_flux(double n, double c, double s) { return n * c / (2.0 * kSqrtPi) * flux_factor(s); }

double flux_normal_cdf(double z, double s) {
  if (z <= 0.0) return 0.0;
  // G(z) = s sqrt(pi)/2 erf(z - s) - exp(-(z - s)^2)/2 is an antiderivative of z exp(-(z - s)^2).
  auto g = [s](double x) { return 0.5 * kSqrtPi * s * std::erf(x - s) - 0.5 * std::exp(-(x - s) * (x - s)); };
  const double total = 0.5 * flux_factor(s);  // G(inf) - G(0)
  return (g(z) - g(0.0)) / total;
}

Domain make_domain(const DomainSection& cfg, const Aabb& mesh_box, double min_margin) {
  Domain d;
  if (cfg.box_min_m && cfg.box_max_m) {
    d.box.lo = *cfg.box_min_m;
    d.box.hi = *cfg.box_max_m;
    for (int k = 0; k < 3; ++k) {
      if (!(d.box.lo[k] <= mesh_box.lo[k] - min_margin && d.box.hi[k] >= mesh_box.hi[k] + min_margin)) {
        std::ostringstream os;
        os << "domain box does not contain the geometry with a margin of at least " << min_margin
           << " m along axis " << "xyz"[k] << " (geometry " << mesh_box.lo[k] << " .. " << mesh_box.hi[k]
           << ", domain " << d.box.lo[k] << " .. " << d.box.hi[k] << ")";
        throw GeometryError(os.str());
      }
    }
    return d;
  }
  const double margin =
      cfg.margin_m ? *cfg.margin_m : kDefaultDomainMarginFraction * max_abs_component(mesh_box.extent());
  if (!(margin >= min_margin)) {
    std::ostringstream os;
    os << "domain margin " << margin << " m is smaller than the minimum " << min_margin << " m";
    throw GeometryError(os.str());
  }
  const Vec3 m(margin, margin, margin);
  d.box.lo = mesh_box.lo - m;
  d.box.hi = mesh_box.hi + m;
  return d;
}

InflowModel InflowModel::create(const Domain& domain, const GasSection& gas) {
  InflowModel model;
  model.domain = domain;
  model.bulk_velocity = gas.bulk_velocity_body_mps;
  double fraction_sum = 0.0;
  for (const auto& sp : gas.species) fraction_sum += sp.mole_fraction;

  for (int face = 0; face < 6; ++face) {
    const int axis = face / 2;
    const double sign = (face % 2 == 0) ? 1.0 : -1.0;
    const double area = domain.area_of_face(face);
    for (std::size_t i = 0; i < gas.species.size(); ++i) {
      const auto& sp = gas.species[i];
      InflowChannel ch;
      ch.face = face;
      ch.species = static_cast<std::uint32_t>(i);
      ch.thermal_speed = std::sqrt(2.0 * constants::k_B * gas.translational_temperature_K / sp.mass_kg);
      ch.s = sign * gas.bulk_velocity_body_mps[axis] / ch.thermal_speed;
      ch.exp_s2 = std::exp(-ch.s * ch.s);
      if (ch.s >= 0.0) {
        const double mass_a = 1.0 - 0.5 * ch.exp_s2;
        const double mass_b = 0.5 * kSqrtPi * ch.s * std::erfc(-ch.s);
        ch.p_a = mass_a / (mass_a + mass_b);
      }
      const double n_s = gas.number_density_m3 * sp.mole_fraction / fraction_sum;
      ch.rate_per_s = number_flux(n_s, ch.thermal_speed, ch.s) * area;
      model.total_rate_per_s += ch.rate_per_s;
      model.channels.push_back(ch);
    }
  }
  if (!(model.total_rate_per_s > 0.0)) throw GeometryError("inflow rate through the domain boundary is zero");
  return model;
}

DeviceInflow upload(const InflowModel& model) {
  const std::size_t n = model.channels.size();
  Kokkos::View<double*> cdf("inflow_cdf", n);
  Kokkos::View<InflowChannel*> ch("inflow_channels", n);
  auto hc = Kokkos::create_mirror_view(cdf);
  auto hch = Kokkos::create_mirror_view(ch);
  double acc = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    acc += model.channels[i].rate_per_s;
    hc(i) = acc;
    hch(i) = model.channels[i];
  }
  Kokkos::deep_copy(cdf, hc);
  Kokkos::deep_copy(ch, hch);
  DeviceInflow d;
  d.cdf = cdf;
  d.channels = ch;
  d.num_channels = static_cast<int>(n);
  d.total_rate = acc;
  d.domain = model.domain;
  d.bulk_velocity = model.bulk_velocity;
  return d;
}

}  // namespace vleo

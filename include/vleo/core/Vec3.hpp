#pragma once

// Three-component double-precision vector usable in host and device code.
// All operations are KOKKOS_INLINE_FUNCTION and free of dynamic allocation,
// exceptions and std:: math calls, so they compile for CUDA, HIP and CPU.

#include <Kokkos_Core.hpp>

namespace vleo {

struct Vec3 {
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;

  KOKKOS_DEFAULTED_FUNCTION constexpr Vec3() = default;
  KOKKOS_INLINE_FUNCTION constexpr Vec3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}

  // Component access by index 0, 1, 2 (no bounds check beyond the ternary).
  KOKKOS_INLINE_FUNCTION constexpr double operator[](int i) const { return i == 0 ? x : (i == 1 ? y : z); }
  KOKKOS_INLINE_FUNCTION constexpr double& operator[](int i) { return i == 0 ? x : (i == 1 ? y : z); }

  KOKKOS_INLINE_FUNCTION constexpr Vec3& operator+=(const Vec3& o) {
    x += o.x;
    y += o.y;
    z += o.z;
    return *this;
  }
  KOKKOS_INLINE_FUNCTION constexpr Vec3& operator-=(const Vec3& o) {
    x -= o.x;
    y -= o.y;
    z -= o.z;
    return *this;
  }
  KOKKOS_INLINE_FUNCTION constexpr Vec3& operator*=(double s) {
    x *= s;
    y *= s;
    z *= s;
    return *this;
  }
  KOKKOS_INLINE_FUNCTION constexpr Vec3& operator/=(double s) {
    x /= s;
    y /= s;
    z /= s;
    return *this;
  }

  KOKKOS_INLINE_FUNCTION static constexpr Vec3 zero() { return {0.0, 0.0, 0.0}; }
  KOKKOS_INLINE_FUNCTION static constexpr Vec3 unit_x() { return {1.0, 0.0, 0.0}; }
  KOKKOS_INLINE_FUNCTION static constexpr Vec3 unit_y() { return {0.0, 1.0, 0.0}; }
  KOKKOS_INLINE_FUNCTION static constexpr Vec3 unit_z() { return {0.0, 0.0, 1.0}; }
};

KOKKOS_INLINE_FUNCTION constexpr Vec3 operator+(const Vec3& a, const Vec3& b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
KOKKOS_INLINE_FUNCTION constexpr Vec3 operator-(const Vec3& a, const Vec3& b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
KOKKOS_INLINE_FUNCTION constexpr Vec3 operator-(const Vec3& a) { return {-a.x, -a.y, -a.z}; }
KOKKOS_INLINE_FUNCTION constexpr Vec3 operator*(const Vec3& a, double s) { return {a.x * s, a.y * s, a.z * s}; }
KOKKOS_INLINE_FUNCTION constexpr Vec3 operator*(double s, const Vec3& a) { return {a.x * s, a.y * s, a.z * s}; }
KOKKOS_INLINE_FUNCTION constexpr Vec3 operator/(const Vec3& a, double s) { return {a.x / s, a.y / s, a.z / s}; }
KOKKOS_INLINE_FUNCTION constexpr bool operator==(const Vec3& a, const Vec3& b) { return a.x == b.x && a.y == b.y && a.z == b.z; }
KOKKOS_INLINE_FUNCTION constexpr bool operator!=(const Vec3& a, const Vec3& b) { return !(a == b); }

KOKKOS_INLINE_FUNCTION constexpr double dot(const Vec3& a, const Vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

KOKKOS_INLINE_FUNCTION constexpr Vec3 cross(const Vec3& a, const Vec3& b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

// Component-wise product.
KOKKOS_INLINE_FUNCTION constexpr Vec3 hadamard(const Vec3& a, const Vec3& b) { return {a.x * b.x, a.y * b.y, a.z * b.z}; }

KOKKOS_INLINE_FUNCTION constexpr double norm2(const Vec3& a) { return dot(a, a); }

KOKKOS_INLINE_FUNCTION double norm(const Vec3& a) { return Kokkos::sqrt(norm2(a)); }

KOKKOS_INLINE_FUNCTION double distance(const Vec3& a, const Vec3& b) { return norm(a - b); }

// Unit vector in the direction of a. The caller guarantees norm(a) > 0;
// a zero vector yields non-finite components.
KOKKOS_INLINE_FUNCTION Vec3 normalized(const Vec3& a) { return a / norm(a); }

// Unit vector in the direction of a, or `fallback` if norm(a) <= min_norm.
KOKKOS_INLINE_FUNCTION Vec3 normalized_or(const Vec3& a, const Vec3& fallback, double min_norm = 0.0) {
  const double n = norm(a);
  return n > min_norm ? a / n : fallback;
}

KOKKOS_INLINE_FUNCTION double max_abs_component(const Vec3& a) {
  return Kokkos::fmax(Kokkos::fabs(a.x), Kokkos::fmax(Kokkos::fabs(a.y), Kokkos::fabs(a.z)));
}

KOKKOS_INLINE_FUNCTION bool is_finite(const Vec3& a) {
  return Kokkos::isfinite(a.x) && Kokkos::isfinite(a.y) && Kokkos::isfinite(a.z);
}

// Component of a along the unit vector n, and the remainder orthogonal to n.
KOKKOS_INLINE_FUNCTION constexpr Vec3 project_onto_unit(const Vec3& a, const Vec3& n) { return dot(a, n) * n; }
KOKKOS_INLINE_FUNCTION constexpr Vec3 reject_from_unit(const Vec3& a, const Vec3& n) { return a - dot(a, n) * n; }

// Specular reflection of a about the plane with unit normal n: a - 2 (a.n) n.
KOKKOS_INLINE_FUNCTION constexpr Vec3 reflect(const Vec3& a, const Vec3& n) { return a - 2.0 * dot(a, n) * n; }

// True if |a - b|_inf <= abs_tol + rel_tol * max(|a|_inf, |b|_inf).
KOKKOS_INLINE_FUNCTION bool approx_equal(const Vec3& a, const Vec3& b, double abs_tol, double rel_tol = 0.0) {
  const double scale = Kokkos::fmax(max_abs_component(a), max_abs_component(b));
  return max_abs_component(a - b) <= abs_tol + rel_tol * scale;
}

}  // namespace vleo

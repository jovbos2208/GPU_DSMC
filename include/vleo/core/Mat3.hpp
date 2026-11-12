#pragma once

// Row-major 3x3 double matrix for coordinate transformations (host and device).

#include "vleo/core/Vec3.hpp"

namespace vleo {

struct Mat3 {
  // m[row][col]
  double m[3][3] = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};

  KOKKOS_INLINE_FUNCTION static constexpr Mat3 identity() {
    Mat3 r;
    r.m[0][0] = r.m[1][1] = r.m[2][2] = 1.0;
    return r;
  }

  KOKKOS_INLINE_FUNCTION static constexpr Mat3 from_rows(const Vec3& r0, const Vec3& r1, const Vec3& r2) {
    Mat3 r;
    for (int j = 0; j < 3; ++j) {
      r.m[0][j] = r0[j];
      r.m[1][j] = r1[j];
      r.m[2][j] = r2[j];
    }
    return r;
  }

  KOKKOS_INLINE_FUNCTION static constexpr Mat3 from_columns(const Vec3& c0, const Vec3& c1, const Vec3& c2) {
    return from_rows(c0, c1, c2).transposed();
  }

  KOKKOS_INLINE_FUNCTION constexpr Vec3 row(int i) const { return {m[i][0], m[i][1], m[i][2]}; }
  KOKKOS_INLINE_FUNCTION constexpr Vec3 col(int j) const { return {m[0][j], m[1][j], m[2][j]}; }

  KOKKOS_INLINE_FUNCTION constexpr Mat3 transposed() const {
    Mat3 r;
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) r.m[i][j] = m[j][i];
    return r;
  }

  KOKKOS_INLINE_FUNCTION constexpr double determinant() const {
    return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
           m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
  }
};

KOKKOS_INLINE_FUNCTION constexpr Vec3 operator*(const Mat3& a, const Vec3& v) {
  return {a.m[0][0] * v.x + a.m[0][1] * v.y + a.m[0][2] * v.z, a.m[1][0] * v.x + a.m[1][1] * v.y + a.m[1][2] * v.z,
          a.m[2][0] * v.x + a.m[2][1] * v.y + a.m[2][2] * v.z};
}

KOKKOS_INLINE_FUNCTION constexpr Mat3 operator*(const Mat3& a, const Mat3& b) {
  Mat3 r;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) r.m[i][j] = a.m[i][0] * b.m[0][j] + a.m[i][1] * b.m[1][j] + a.m[i][2] * b.m[2][j];
  return r;
}

// Maximum absolute entry of a - b.
KOKKOS_INLINE_FUNCTION double max_abs_difference(const Mat3& a, const Mat3& b) {
  double d = 0.0;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) d = Kokkos::fmax(d, Kokkos::fabs(a.m[i][j] - b.m[i][j]));
  return d;
}

// Active rotation by `angle_rad` (right-hand rule) about the unit vector `axis`
// (Rodrigues' formula). R * v rotates v; R^T undoes it.
KOKKOS_INLINE_FUNCTION Mat3 rotation_about_axis(const Vec3& axis, double angle_rad) {
  const double c = Kokkos::cos(angle_rad);
  const double s = Kokkos::sin(angle_rad);
  const double t = 1.0 - c;
  const double x = axis.x, y = axis.y, z = axis.z;
  Mat3 r;
  r.m[0][0] = c + x * x * t;
  r.m[0][1] = x * y * t - z * s;
  r.m[0][2] = x * z * t + y * s;
  r.m[1][0] = y * x * t + z * s;
  r.m[1][1] = c + y * y * t;
  r.m[1][2] = y * z * t - x * s;
  r.m[2][0] = z * x * t - y * s;
  r.m[2][1] = z * y * t + x * s;
  r.m[2][2] = c + z * z * t;
  return r;
}

}  // namespace vleo

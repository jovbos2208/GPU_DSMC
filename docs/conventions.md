# Units, axes, normals and sign conventions (M0.7)

This document is normative: code, tests and configuration files must follow it.
Where the implementation plan and this file disagree, fix one of them explicitly.

## 1. Units

The solver computes exclusively in **SI base and derived units** with `double`
precision. Conversions happen only at the input boundary.

| Quantity              | Unit     | Config key suffix | Example key                     |
| --------------------- | -------- | ----------------- | ------------------------------- |
| Length, position      | m        | `_m`              | `reference.length_m`            |
| Area                  | m²       | `_m2`             | `reference.area_m2`             |
| Number density        | m⁻³      | `_m3`             | `gas.number_density_m3`         |
| Velocity              | m/s      | `_mps`            | `gas.bulk_velocity_body_mps`    |
| Temperature           | K        | `_K`              | `surface.temperature_K`         |
| Time                  | s        | `_s`              | `dsmc.timestep_s`               |
| Mass                  | kg       | `_kg`             | (species database)              |
| Pressure              | Pa       | `_Pa`             | `dynamic_pressure_Pa` (output)  |
| Force                 | N        | `_N`              | `force_N` (output)              |
| Moment                | N·m      | `_Nm`             | `moment_Nm` (output)            |
| Power / heat rate     | W        | `_W`              | `heat_rate_W` (output)          |
| Energy                | J        |                   |                                 |

Every dimensional configuration key carries its unit suffix. The only
exception is `geometry.units` (`m`, `cm`, `mm`): STL files carry no unit, so the
mesh coordinates are multiplied by the corresponding factor to metres during
import. Everything else - including `reference.*` - is always in SI.

Constants (`include/vleo/core/PhysicalConstants.hpp`): k_B = 1.380649e-23 J/K
and N_A = 6.02214076e23 1/mol (exact, SI 2019), u = 1.66053906892e-27 kg
(CODATA 2022). Species masses are standard atomic weights times u
(`src/core/Species.cpp`).

Species fractions in `gas.species` are **number (mole) fractions** and must sum
to 1 within 1e-6. They are stored as given; consumers normalise by their sum.

## 2. Body frame

- The **body frame** is the coordinate system of the input geometry: right-handed,
  Cartesian, origin and orientation as in the STL file (after unit scaling).
- All vector inputs (`moment_center_m`, `lift_axis_body`, `bulk_velocity_body_mps`)
  and all vector outputs (force, moment) are given in body-frame components.
- `gas.bulk_velocity_body_mps` is the free-stream **gas velocity relative to the
  body**. A satellite flying in +x through a resting atmosphere therefore has
  `bulk_velocity_body_mps: [-V, 0, 0]`.

## 3. Surface normals and the local surface frame

- The surface normal **n points from the solid into the gas** (outward normal).
- Triangle vertices (a, b, c) are ordered **counter-clockwise seen from the gas
  side**; then n = (b − a) × (c − a) / |(b − a) × (c − a)| (right-hand rule).
  The STL importer verifies this orientation per closed component and flips
  components with negative signed volume ([geometry.md](geometry.md)).
- Open components (only with `geometry.require_closed: false`) are two-sided:
  the interaction uses the normal on the side the particle comes from.
- Velocities at a wall are taken relative to the wall: **c = v − v_w**.
- An incident particle satisfies **c_i · n < 0**, a re-emitted one c_r · n > 0.
- The local surface frame (t1, t2, n) is right-handed orthonormal with
  t1 × t2 = n, built from n by the branch-free method of Duff et al. (2017),
  `make_basis_from_normal()`. Local components are (c·t1, c·t2, c·n); the
  normal component is the third.

## 4. Aerodynamic frame and coefficients

With V∞ = `gas.bulk_velocity_body_mps` and the user lift axis l = `reference.lift_axis_body`:

- **e_D = V∞ / |V∞|** - drag direction, along the incoming gas flow.
- **e_L** = normalised component of l orthogonal to e_D. If l is not exactly
  orthogonal to V∞ it is projected (reported as `lift_axis_was_projected`);
  if it is (numerically) parallel to V∞ the configuration is rejected.
- **e_Y = e_L × e_D** - side-force direction. (e_D, e_Y, e_L) is right-handed:
  e_D × e_Y = e_L. Example: V∞ = [−7800, 0, 0], l = [0, 0, 1] gives
  e_D = −x, e_L = +z, e_Y = −y.

Reference values: q∞ = ½ ρ∞ |V∞|², ρ∞ = n∞ Σ x_s m_s (x_s normalised).
`reference.area_m2` (A_ref) and `reference.length_m` (L_ref) are explicit inputs,
never derived from the geometry.

| Output | Definition |
| ------ | ---------- |
| C_D    | F · e_D / (q∞ A_ref) |
| C_L    | F · e_L / (q∞ A_ref) |
| C_Y    | F · e_Y / (q∞ A_ref)  (additional output) |
| **C_M** | M / (q∞ A_ref L_ref), three body-frame components |

## 5. Signs of forces, moments and heat transfer

For every gas-surface event e with incident velocity v_i, re-emitted velocity
v_r, position r_e and particle mass m_e:

- Momentum transferred **to the body**: Δp_e = m_e (v_i − v_r).
- Moment about `reference.moment_center_m` (r_0): Δl_e = (r_e − r_0) × Δp_e.
- Translational energy transferred **to the surface** (wall frame):
  ΔE_e = ½ m_e (|c_i|² − |c_r|²); internal energy adds E_int,i − E_int,r (M4).
- With the event rate factor γ_e [1/s]: F = Σ γ_e Δp_e, M = Σ γ_e Δl_e,
  Q̇ = Σ γ_e ΔE_e.

**F and M are the force and moment exerted by the gas on the body. Positive Q̇
means heat flows from the gas into the surface.** Positive and negative
contributions to Q̇ are tallied separately.

## 6. Gas-surface interaction parameters

- Maxwell: `diffuse_fraction` f_d ∈ [0, 1] (fraction of diffuse re-emissions).
- CLL: `alpha_n` ∈ [0, 1] is the **normal energy** accommodation coefficient,
  `sigma_t` ∈ [0, 2] the **tangential momentum** accommodation coefficient.
  The tangential energy accommodation is α_t = σ_t (2 − σ_t).
  Codes parametrised by α_t (e.g. SPARTA's `acc_t`) must receive α_t, not σ_t;
  every comparison case documents the conversion.
- Limits: CLL(α_n = 0, σ_t = 0) is specular, CLL(α_n = 1, σ_t = 1) fully diffuse.

## 7. Random numbers

Philox4x32-10, counter based (`include/vleo/core/Random.hpp`). A stream is a
pure function of (seed, particle id, event id, stream id): key = seed,
counter = (particle id, event id, stream id << 16 | block). One
(particle, event, stream) triple provides 2¹⁶ blocks = 262144 32-bit words;
callers must use a new event id before that (detected by `exhausted()`).
Integer and uniform draws are bitwise identical on all backends; normal
variates may differ in the last ulps between device math libraries.

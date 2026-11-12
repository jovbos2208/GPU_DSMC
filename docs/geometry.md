# Geometry, inflow and particle transport (M1)

This document describes the geometry engine and the collisionless transport
that TPMC (M2) and DSMC (M3) build on. Conventions (units, normals, frames)
are in [conventions.md](conventions.md).

## 1. STL import (`include/vleo/geometry/Stl.hpp`)

- Binary and ASCII STL. Binary is recognised by its size (84 + 50 n bytes),
  even if the 80-byte header starts with `solid`; otherwise the file must
  start with `solid` (ASCII).
- Coordinates are multiplied by the factor of `geometry.units` (`m`, `cm`, `mm`).
  Binary STL stores single precision; the values are converted exactly to double.
- **Material ids:** in ASCII files each `solid … endsolid` block is one material
  (id = block index, name = solid name). Binary files have one material (id 0).
- Facet normals stored in the file are **not** used; the normal follows from the
  vertex order. Disagreeing file normals are counted and reported as a warning.
- Malformed files (unknown keywords, wrong vertex count, non-finite numbers,
  truncated binary data, empty files) are rejected with file name and line
  (ASCII) or the announced/actual size (binary).

## 2. Mesh validation and orientation (`Mesh.hpp`)

| Check | Rule | Consequence |
| ----- | ---- | ----------- |
| Degenerate triangle | area ≤ 1e-12 × (longest edge)² | removed, counted, warning |
| Vertex welding | exact coordinate equality (−0 = +0) | basis for topology |
| Boundary edge | edge used by 1 triangle | component is open |
| Non-manifold edge | edge used by > 2 triangles | component is open |
| Inconsistent edge | 2 triangles traverse the edge in the same direction | component is open |
| Components | connected via shared edges | reported individually |
| Closed component | none of the three edge defects | must enclose a volume (else error) |
| Orientation | signed volume < 0 | component flipped, warning |

With `geometry.require_closed: true` (default) every component must be closed;
otherwise the run is rejected and the message lists the defects. With
`require_closed: false`, open components are accepted and treated as
**two-sided** surfaces (e.g. a flat plate for free-molecular validation):
a hit on either side interacts with the normal that faces the particle.
Closed components are always one-sided: a hit from inside is a numerical
failure (`LostInsideBody`).

Device storage (`DeviceMesh`): structure of arrays `v0, v1, v2, normal, area,
material, two_sided` in the default memory space. Triangles are stored in BVH
leaf order; `HostMesh::source_index` maps back to the input file order.

## 3. Ray-triangle intersection and epsilon strategy (`RayTriangle.hpp`)

Watertight algorithm of Woop, Benthin and Wald (JCGT 2013), double precision.

| Aspect | Strategy |
| ------ | -------- |
| Inside test | no epsilon. Edge functions of a shared edge are computed from the same two sheared vertices, so neighbours see exactly negated values: **a ray entering a closed mesh can never pass between two triangles**. Edge and vertex hits are inclusive. |
| FMA | the products in the edge functions use `__dmul_rn` on GPUs (and a non-contractible product on the host); FMA contraction would break the exact negation. |
| Ties | a ray through a shared edge can hit both neighbours; the closest computed `t` wins, exact ties go to the smaller triangle index (deterministic, same rule for BVH and brute force). |
| Valid range | `0 < t ≤ t_max`; rays parallel to the plane (`det == 0`) never hit. |
| Self-hits | after a surface interaction the new origin is moved by `surface_offset = 1e-9 × L` along the gas-side normal (L = largest absolute coordinate or extent of the mesh bounding box; 1e-8 m for a 10 m satellite) **and** the triangle just hit is excluded from the next search. |
| Grazing silhouette rays | may miss by an ulp because the ray direction itself is rounded. They do not enter the body, so nothing is lost. |

The offset is far above the rounding error of hit points (~1e-16 L) and far
below any geometric feature of interest. The domain margin must be at least
1000 × surface_offset.

## 4. BVH (`Bvh.hpp`)

- Built on the CPU: binned SAH (16 bins, all three axes), leaves of ≤ 4
  triangles where profitable, median split as fallback for primitives that the
  SAH cannot separate, maximum depth 64.
- Flat node array in depth-first order (left child = next node), uploaded once.
- Iterative traversal with a fixed 64-entry stack, nearer child first. It is
  identical on CPU, CUDA and HIP.
- **Conservative box test:** the slab interval is widened by the relative slack
  `kBoxSlack = 1e-9` before it is compared with the current search limit (the
  computed `t` of the best triangle so far). Without it, a ray aimed exactly at
  a vertex lying on a box corner could cull the box that contains an equally
  close triangle. With it, the BVH returns **exactly the same triangle and the
  bitwise same `t`** as a brute-force loop (verified by `test_ray` on random
  meshes, including axis-parallel rays and finite segments).

## 5. Simulation domain and inflow (`Domain.hpp`, `Inflow.hpp`)

- The domain is an axis-aligned box: the mesh bounding box grown by
  `domain.margin_m` (default 5 % of the largest extent) or an explicit
  `domain.box_min_m/box_max_m` that contains the body with at least the minimum
  margin. In collisionless flow a particle leaving the convex box never returns,
  so the box size affects only efficiency, not the result.
- Free-stream molecules enter through all six faces. For species s and a face
  with inward normal e: speed ratio s = u·e / c, c = √(2kT/m), number flux
  Γ = n_s c / (2√π) [e^{−s²} + √π s (1 + erf s)] (evaluated with `erfc` so
  that faces with s ≪ 0 keep full relative accuracy).
- Each (face, species) pair is a channel chosen with probability Γ A / Σ.
  The normal speed follows the flux-weighted density z e^{−(z−s)²}, sampled by
  an **exact rejection method** (acceptance ≥ 0.73 for s ≥ 0); tangential
  components are Maxwellian around u_t; positions are uniform on the face.
- The TPMC event-rate factor (weight per history) is γ = Σ Γ A / N_histories
  [1/s] (`inflow.tpmc_event_rate_factor_per_s` in the metadata).

## 6. Transport and particle balance (`Transport.hpp`)

One thread follows one history. Each step computes the distance to the domain
boundary and the closest surface hit before it.

| Final status | Meaning |
| ------------ | ------- |
| `Exited` | left the domain (normal end) |
| `Truncated` | the next hit would exceed `tpmc.max_reflections`; the history ends **before** that interaction (its contribution is missing - counted so the bias is visible) |
| `LostInsideBody` | hit a closed surface from inside - numerical failure, must be 0 |
| `Invalid` | non-finite state, zero velocity or re-emission into the wall - must be 0 |

Every history ends in exactly one status, so
`injected = exited + truncated + lost_inside_body + invalid` holds by
construction. `count_particles()` reduces the final states; additionally the
counters record processed surface interactions, histories with ≥ 1 hit and
the maximum number of interactions per history.

Random numbers: injection of history `id` uses (seed, id, 0, Injection); its
k-th surface interaction uses (seed, id, k + 1, SurfaceInteraction). A history
is therefore bitwise reproducible independent of batch size, slot and backend.

The surface interaction model is a template parameter. M1 ships
`SpecularInteraction` for testing; the Maxwell and CLL kernels follow in M2.

## 7. Verification (Gate M1)

| Test | Verifies |
| ---- | -------- |
| `test_stl` | ASCII/binary import, unit scaling, multiple solids, round trip, error cases |
| `test_mesh` | cube/sphere topology, area, volume, re-orientation, open/non-manifold/inconsistent/degenerate/zero-volume cases, satellite example, device arrays |
| `test_ray` | analytic hits (interior, edges, vertices, t range, parallel, back face) on host and device; watertightness at every vertex and edge of closed meshes; BVH ≡ brute force on random soups, spheres, many boxes, satellite |
| `test_inflow` | flux factor limits and tail; KS test of the normal-speed sampler for s ∈ [−1.5, 8]; mean normal speed; channel frequencies (χ²); tangential mean and variance; positions on the faces; domain construction |
| `test_transport` | **cube gate case**: 400 000 histories, no lost or double-counted particle, every history exits, hit fraction equals the analytic free-molecular collision rate within 5σ; reproducibility independent of batching; concave gaps with diffuse-like re-emission (no losses); analytic multiple reflections between parallel plates; truncation; failure states |

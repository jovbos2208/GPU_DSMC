# Species database (M4.1)

`src/core/Species.cpp` is the only place where species data are defined.
Every value has a source:

| Key | Source |
| --- | ------ |
| [IUPAC] | IUPAC standard atomic weights (conventional values) for the molar masses |
| [Bird94] | G. A. Bird, *Molecular Gas Dynamics and the Direct Simulation of Gas Flows*, Oxford 1994, Table A1 (VHS and VSS columns, T_ref = 273 K) |
| [SPARTA] | SPARTA DSMC code, release 27Aug2026, `data/air.vss` and `data/air.species` |

## Data

| Species | M [g/mol] | d_VHS [m] | ω | T_ref [K] | d_VSS [m] | α_VSS | collision source | rot. DOF | Z_rot | vib. DOF | θ_v [K] | Z_vib | modes source |
| ------- | --------- | --------- | - | --------- | --------- | ----- | ---------------- | -------- | ----- | -------- | ------- | ----- | ------------ |
| O  | 15.999 | 3.00e-10 | 0.80 | 273.15 | 3.00e-10 | 1.00 | [SPARTA] air.vss | 0 | – | 0 | – | – | – |
| N  | 14.007 | 3.00e-10 | 0.80 | 273.15 | 3.00e-10 | 1.00 | [SPARTA] air.vss | 0 | – | 0 | – | – | – |
| He | 4.002602 | 2.33e-10 | 0.66 | 273 | 2.30e-10 | 1.26 | [Bird94] | 0 | – | 0 | – | – | – |
| Ar | 39.948 | 4.17e-10 | 0.81 | 273 | 4.11e-10 | 1.40 | [Bird94] | 0 | – | 0 | – | – | – |
| N2 | 28.014 | 4.17e-10 | 0.74 | 273 | 4.11e-10 | 1.36 | [Bird94] | 2 | 5 | 2 | 3371 | 52600 | [SPARTA] air.species |
| O2 | 31.998 | 4.07e-10 | 0.77 | 273 | 4.01e-10 | 1.40 | [Bird94] | 2 | 5 | 2 | 2256 | 17900 | [SPARTA] air.species |

Notes:

- Atomic O and N are not part of [Bird94] Table A1; their parameters are those
  of SPARTA's `air.vss` (where the VSS exponent is 1, i.e. VHS scattering).
- [SPARTA] states relaxation *probabilities* (`RotRel` = 0.2, `VibRel` =
  1.90114e-5 for N2 and 5.58659e-5 for O2); the collision numbers above are
  their inverses. All collision numbers are constant (temperature independent).
- [SPARTA] uses rounded molecular masses (e.g. 2.65e-26 kg for O); the SPARTA
  comparison cases (M4.10) write the masses used by this solver into the
  SPARTA input so both codes use identical data.
- Pair parameters for unlike species: d = (d_p + d_q)/2, ω = (ω_p + ω_q)/2,
  T_ref = (T_ref,p + T_ref,q)/2, α = (α_p + α_q)/2, reduced mass m_r.

Adding a species requires all columns and a source; tests check that every
DSMC species has collision data and every molecule has its internal modes.

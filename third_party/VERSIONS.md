# Vendored third-party dependencies

Each directory is an unmodified copy of the upstream source at the listed tag
(the upstream `.git` directory was removed). Do not edit files here; update a
dependency only as described in `docs/dependencies.md`.

| Directory   | Project  | Tag             | Upstream commit                            | Commit date | License        |
| ----------- | -------- | --------------- | ------------------------------------------ | ----------- | -------------- |
| `kokkos/`   | Kokkos   | `5.2.2`         | `8e15454876298832f013ebcaf08d53825b5bda40` | 2026-09-01  | Apache-2.0 WITH LLVM-exception |
| `yaml-cpp/` | yaml-cpp | `yaml-cpp-0.9.0`| `56e3bb550c91fd7005566f19c079cb7a503223cf` | 2026-02-04  | MIT            |
| `catch2/`   | Catch2   | `v3.8.1`        | `2b60af89e23d28eefc081bc930831ee9d45ea58b` | 2025-04-08  | BSL-1.0        |

Upstream repositories:

- https://github.com/kokkos/kokkos
- https://github.com/jbeder/yaml-cpp
- https://github.com/catchorg/Catch2

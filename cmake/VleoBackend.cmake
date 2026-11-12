# Derive the solver backend name from the Kokkos configuration.
# Exactly one device backend may be active per build: cuda, hip, or none (cpu).

if(Kokkos_ENABLE_CUDA AND Kokkos_ENABLE_HIP)
  message(FATAL_ERROR "CUDA and HIP cannot be enabled in the same build.")
endif()

if(Kokkos_ENABLE_CUDA)
  set(VLEO_BACKEND "cuda")
elseif(Kokkos_ENABLE_HIP)
  set(VLEO_BACKEND "hip")
else()
  set(VLEO_BACKEND "cpu")
endif()

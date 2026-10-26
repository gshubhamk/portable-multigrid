cmake_minimum_required(VERSION 3.18)

set(TARGET "program")
set(TARGET_SRC ${TARGET}.cc)

# Build the NVSHMEM variants (needs CUDA and the NVSHMEM module, e.g. on
# Jupiter: module load NVSHMEM/<version matching the CUDA of the build>)
option(WITH_NVSHMEM "Build the NVSHMEM variants of the halo exchange" OFF)

find_package(deal.II 9.8.0
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
)

if(NOT ${deal.II_FOUND})
  message(FATAL_ERROR "Could not locate deal.II.")
endif()

deal_ii_initialize_cached_variables()
project(${TARGET} CXX)

deal_ii_invoke_autopilot()

target_include_directories(${TARGET} PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../../include
  ${CMAKE_CURRENT_SOURCE_DIR}/../common
)

if(WITH_NVSHMEM)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 90)
  endif()
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)

  set(_nvshmem_hints $ENV{NVSHMEM_HOME} $ENV{NVSHMEM_DIR} $ENV{EBROOTNVSHMEM})
  find_path(NVSHMEM_INCLUDE_DIR nvshmem.h HINTS ${_nvshmem_hints}
    PATH_SUFFIXES include REQUIRED)
  find_library(NVSHMEM_HOST_LIBRARY nvshmem_host HINTS ${_nvshmem_hints}
    PATH_SUFFIXES lib lib64 REQUIRED)
  find_library(NVSHMEM_DEVICE_LIBRARY libnvshmem_device.a HINTS ${_nvshmem_hints}
    PATH_SUFFIXES lib lib64 REQUIRED)
  message(STATUS "NVSHMEM include dir:    ${NVSHMEM_INCLUDE_DIR}")
  message(STATUS "NVSHMEM host library:   ${NVSHMEM_HOST_LIBRARY}")
  message(STATUS "NVSHMEM device library: ${NVSHMEM_DEVICE_LIBRARY}")

  # The NVSHMEM code lives in its own shared library with relocatable device
  # code, so that deal.II and Kokkos do not need to be built with it.
  add_library(nvshmem_halo SHARED nvshmem_halo.cu)
  set_target_properties(nvshmem_halo PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_STANDARD 17)
  target_include_directories(nvshmem_halo PRIVATE ${NVSHMEM_INCLUDE_DIR})
  target_link_libraries(nvshmem_halo PRIVATE
    ${NVSHMEM_DEVICE_LIBRARY} ${NVSHMEM_HOST_LIBRARY} CUDA::cudart CUDA::cuda_driver)

  # deal.II links with the plain signature, so do the same here
  target_link_libraries(${TARGET} nvshmem_halo)
  target_compile_definitions(${TARGET} PRIVATE HALO_WITH_NVSHMEM)
endif()

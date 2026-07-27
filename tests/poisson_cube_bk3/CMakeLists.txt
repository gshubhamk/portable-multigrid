cmake_minimum_required(VERSION 3.13.4)

# 1. Set the target name
set(TARGET "program")

# 2. Source files
set(TARGET_SRC ${TARGET}.cc)

# 3. Find deal.II
find_package(deal.II 9.8.0
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
)

if(NOT ${deal.II_FOUND})
  message(FATAL_ERROR "Could not locate deal.II.")
endif()

# 4. Initialize and Project setup
deal_ii_initialize_cached_variables()
project(${TARGET} CXX)

# 5. Define Macros for the Performance Driver
# These are required by your "performance_test_driver.h"
# add_definitions(-DENABLE_MPI)
# add_definitions(-DENABLE_PERFORMANCE_TESTS)
# add_definitions(-DTESTING_ENVIRONMENT=light) # Change to medium/heavy as needed

# 5. Locate NVSHMEM
find_path(NVSHMEM_INCLUDE_DIR nvshmem.h
  HINTS $ENV{NVSHMEM_HOME}/include $ENV{NVSHMEM_DIR}/include /opt/nvidia/hpc_sdk/Linux_x86_64/latest/comm_libs/nvshmem/include
)

find_library(NVSHMEM_LIBRARY nvshmem
  HINTS $ENV{NVSHMEM_HOME}/lib $ENV{NVSHMEM_DIR}/lib /opt/nvidia/hpc_sdk/Linux_x86_64/latest/comm_libs/nvshmem/lib
)

find_package(CUDAToolkit REQUIRED)

if(NVSHMEM_INCLUDE_DIR AND NVSHMEM_LIBRARY)
  message(STATUS "Found NVSHMEM Header: ${NVSHMEM_INCLUDE_DIR}")
  message(STATUS "Found NVSHMEM Library: ${NVSHMEM_LIBRARY}")
  add_compile_definitions(WITH_NVSHMEM)
else()
  message(WARNING "NVSHMEM not found! Set NVSHMEM_HOME environment variable to enable NVSHMEM support.")
endif()

# 6. Autopilot handles linking, flags (-O3), and Kokkos/MPI setup
deal_ii_invoke_autopilot()

# 7. Include Directories
# This points to where your "multigrid/" and "operators/" folders are
target_include_directories(${TARGET} PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../../include
  ${CMAKE_CURRENT_SOURCE_DIR}/../common
)

# 8. Link NVSHMEM and CUDA libraries (Must be placed AFTER deal_ii_invoke_autopilot)
if(NVSHMEM_INCLUDE_DIR AND NVSHMEM_LIBRARY)
  target_include_directories(${TARGET} PRIVATE ${NVSHMEM_INCLUDE_DIR})
  target_link_libraries(${TARGET} PRIVATE ${NVSHMEM_LIBRARY} CUDA::cuda CUDA::cudart)
endif()
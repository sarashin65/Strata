cmake_minimum_required(VERSION 3.24)

# PUBLIC BUILD STATUS: this portable CMake file has not been built or linked.
# The base is the upstream Engine 0.1.17 generation; this project does not claim
# the upstream version number.
project(strata_sycl)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(STRATA_PORT_VERSION "sycl" CACHE STRING "Published port identifier")
add_compile_definitions(STRATA_VERSION="${STRATA_PORT_VERSION}")
option(STRATA_DPCT_PROFILING "Enable DPCT queue profiling properties" ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Applied through CMAKE_CXX_FLAGS to every host compile command. icpx -fsycl
# propagates these options to the device compilation as well.
set(STRATA_CXX_FLAGS "-fsycl -O1 -fp-model=precise -ffp-contract=off -Wno-overriding-option")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${STRATA_CXX_FLAGS}")
add_link_options(-fsycl)

set(STRATA_GGML_DIR "" CACHE PATH "Pinned local llama.cpp checkout")
if(NOT STRATA_GGML_DIR OR NOT EXISTS "${STRATA_GGML_DIR}/ggml/CMakeLists.txt")
    message(FATAL_ERROR "STRATA_GGML_DIR must point to the extracted llama.cpp checkout")
endif()
if(NOT EXISTS "${STRATA_GGML_DIR}/ggml/src/ggml-sycl")
    message(FATAL_ERROR "STRATA_GGML_DIR is missing ggml/src/ggml-sycl")
endif()

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_STATIC ON CACHE BOOL "" FORCE)
set(GGML_CUDA OFF CACHE BOOL "" FORCE)
set(GGML_SYCL OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE ON CACHE BOOL "" FORCE)
add_subdirectory("${STRATA_GGML_DIR}/ggml" "${CMAKE_BINARY_DIR}/ggml" EXCLUDE_FROM_ALL)

find_package(Threads REQUIRED)
set(MKL_LINK dynamic CACHE STRING "" FORCE)
set(MKL_SYCL_LINK dynamic CACHE STRING "" FORCE)
set(MKL_INTERFACE ilp64 CACHE STRING "" FORCE)
set(MKL_SYCL_THREADING sequential CACHE STRING "" FORCE)
# Prefer the reader's MKLROOT; retain the conventional oneAPI location only
# when MKLROOT is absent. The actual oneMKL link has not been verified here.
if(DEFINED ENV{MKLROOT} AND NOT "$ENV{MKLROOT}" STREQUAL "")
    set(STRATA_MKL_CMAKE_DIR "$ENV{MKLROOT}/lib/cmake/mkl")
    set(STRATA_MKL_INCLUDE_DIR "$ENV{MKLROOT}/include")
else()
    set(STRATA_MKL_CMAKE_DIR "/opt/intel/oneapi/mkl/latest/lib/cmake/mkl")
    set(STRATA_MKL_INCLUDE_DIR "/opt/intel/oneapi/mkl/latest/include")
endif()
find_package(MKL CONFIG REQUIRED
    PATHS "${STRATA_MKL_CMAKE_DIR}"
    NO_DEFAULT_PATH)

add_library(strata_sycl_includes INTERFACE)
if(STRATA_DPCT_PROFILING)
    target_compile_definitions(strata_sycl_includes INTERFACE DPCT_PROFILING_ENABLED)
endif()

# CUDA headers are optional here. Whether any converted translation unit needs
# them has not been verified; an empty default is intentional.
set(STRATA_CUDA_INCLUDE_DIR "" CACHE PATH "Optional CUDA headers for converted sources")
if(STRATA_CUDA_INCLUDE_DIR AND NOT IS_DIRECTORY "${STRATA_CUDA_INCLUDE_DIR}")
    message(FATAL_ERROR "STRATA_CUDA_INCLUDE_DIR is not a directory: ${STRATA_CUDA_INCLUDE_DIR}")
endif()
set(STRATA_PROJECT_INCLUDE_DIRS
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml"
    "${CMAKE_CURRENT_SOURCE_DIR}/tools/include"
    "${STRATA_MKL_INCLUDE_DIR}"
    "${STRATA_GGML_DIR}/ggml/include"
    "${STRATA_GGML_DIR}/ggml/src"
    "${STRATA_GGML_DIR}/ggml/src/ggml-cpu"
    "${STRATA_GGML_DIR}/ggml/src/ggml-cuda"
    "${STRATA_GGML_DIR}/tools/mtmd"
    "${STRATA_GGML_DIR}/include"
)
if(STRATA_CUDA_INCLUDE_DIR)
    list(APPEND STRATA_PROJECT_INCLUDE_DIRS "${STRATA_CUDA_INCLUDE_DIR}")
endif()
target_include_directories(strata_sycl_includes INTERFACE ${STRATA_PROJECT_INCLUDE_DIRS})

file(GLOB_RECURSE STRATA_ALL_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
list(SORT STRATA_ALL_SOURCES)
list(LENGTH STRATA_ALL_SOURCES STRATA_SOURCE_COUNT)
message(STATUS "Strata converted C++ translation units: ${STRATA_SOURCE_COUNT}")

set(STRATA_LINK_SOURCES ${STRATA_ALL_SOURCES})
foreach(_main_re IN ITEMS
    ".*/artifact/gguf_reader\\.cpp$"
    ".*/core/(concurrent_main|device_main|load_main|overlap_main)\\.cpp(\\.dp\\.cpp)?$"
    ".*/plan/plan_main\\.cpp$"
    ".*/program/generate\\.cpp(\\.dp)?\\.cpp$"
    ".*/.*_test\\.cpp(\\.dp\\.cpp)?$"
    ".*/.*_parity\\.cpp(\\.dp\\.cpp)?$"
    ".*/.*_stress\\.cpp(\\.dp\\.cpp)?$"
    )
    list(FILTER STRATA_LINK_SOURCES EXCLUDE REGEX "${_main_re}")
endforeach()

set_source_files_properties("${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/cpu/expert.cpp"
    PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512dq;-mavx512vl;-mavx512vbmi;-mavx512vnni")
set_source_files_properties("${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/cpu/q2_avx2.cpp"
    PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
set_source_files_properties("${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/cpu/iq_avx2.cpp"
    PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
set_source_files_properties("${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/cpu/iq_avx512.cpp"
    PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512vl")

set(STRATA_ENTRY_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/program/generate.cpp.dp.cpp")
if(NOT EXISTS "${STRATA_ENTRY_SOURCE}")
    message(FATAL_ERROR "converted generate entry point is missing: ${STRATA_ENTRY_SOURCE}")
endif()
list(REMOVE_ITEM STRATA_LINK_SOURCES "${STRATA_ENTRY_SOURCE}")
set(STRATA_COMPILE_ONLY ${STRATA_ALL_SOURCES})
list(REMOVE_ITEM STRATA_COMPILE_ONLY ${STRATA_LINK_SOURCES} "${STRATA_ENTRY_SOURCE}")

add_library(strata_link_objects OBJECT ${STRATA_LINK_SOURCES})
add_library(strata_compile_only OBJECT ${STRATA_COMPILE_ONLY})
add_library(strata_entry OBJECT "${STRATA_ENTRY_SOURCE}")
foreach(_obj IN ITEMS strata_link_objects strata_compile_only strata_entry)
    target_link_libraries(${_obj} PRIVATE strata_sycl_includes)
endforeach()

add_executable(strata-sycl)
target_sources(strata-sycl PRIVATE
    $<TARGET_OBJECTS:strata_entry>
    $<TARGET_OBJECTS:strata_link_objects>)
add_dependencies(strata-sycl strata_compile_only)
target_link_libraries(strata-sycl PRIVATE
    strata_sycl_includes
    ggml-cpu ggml-base Threads::Threads
    MKL::MKL_SYCL)

target_compile_definitions(strata-sycl PRIVATE STRATA_PORT02N_BUILD=1)
target_compile_definitions(strata_link_objects PRIVATE STRATA_NATIVE_EXPERTS=1 STRATA_PREFILL_MMQ=1)

message(STATUS "STRATA_GGML_DIR=${STRATA_GGML_DIR}")
message(STATUS "GGML_SYCL=OFF (converted Strata SYCL path is compiled directly)")
message(STATUS "STRATA_CXX_FLAGS=${STRATA_CXX_FLAGS}")
message(STATUS "All converted and retained project sources are in the compile graph")

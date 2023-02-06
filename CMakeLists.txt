cmake_minimum_required(VERSION 3.6)
project(voronoi_approximator)

set(CMAKE_TOOLCHAIN_FILE ${CMAKE_SOURCE_DIR}/conda-toolchain.cmake)

option(BUILD_SHARED_LIBS "Build libraries as shared as opposed to static" ON)

if(NOT DEFINED CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug")
endif()

set(CMAKE_CXX_STANDARD 17)

find_package(Eigen3)
include( ${EIGEN3_USE_FILE} )

add_subdirectory(libs)

include_directories(${CMAKE_SOURCE_DIR}/include)

INCLUDE(CheckCXXSourceCompiles)

FIND_PACKAGE(OpenMP)

set (CMAKE_REQUIRED_FLAGS ${OpenMP_CXX_FLAGS})
CHECK_CXX_SOURCE_COMPILES("
#include <omp.h>
int main() {
#if (_OPENMP >= 200805 || _MSC_VER >= 1500)
  return 0;
#else
  breaks_on_purpose
#endif
}
" OPENMP_VERSION)

if(OPENMP_VERSION)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else()
    message("

OpenMP 3.0 not supported by the compiler (${CMAKE_CXX_COMPILER})!
To get optimal performance for the \"chunk\" algorithm, use a C++ compiler with OpenMP 3.0 support (e.g., GCC >=4.4).
To use a different compiler, pass it to cmake in the variable CMAKE_CXX_COMPILER:
 cmake . -DCMAKE_CXX_COMPILER=g++-4.7

")
endif()

find_package(PythonLibs 3 REQUIRED)
find_package(Boost COMPONENTS python numpy REQUIRED)

#find_package(MPFR REQUIRED)
#include_directories(${MPFR_INCLUDES})
#if(NOT MPFR_FOUND)
#    message(FATAL_ERROR "Could not find MPFR!")

#find_package(GMP REQUIRED)
#include_directories(${GMP_INCLUDE_DIR})
#if(NOT (GMP_INCLUDE_DIR AND GMP_LIBRARIES))
#    message(FATAL_ERROR "Could not find GMP!")

# Without this, any build libraries automatically have names "lib{x}.so"
set(CMAKE_SHARED_MODULE_PREFIX "")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -frounding-math -O3 -fPIC")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -frounding-math -O3 -fPIC")


add_subdirectory(vgt)

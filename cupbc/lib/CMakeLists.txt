# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# This file is part of ByteQC.
#
# ByteQC includes code adapted from PySCF
# (https://github.com/pyscf/pyscf), which is licensed under the Apache License
# 2.0. The original copyright:
#     Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
#
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.
#
#     Author: Qiming Sun <osirpt.sun@gmail.com>
#

cmake_minimum_required (VERSION 3.9)
project (cupbc C CUDA)

if(DEFINED CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "${CUDA_ARCHITECTURES}")
else()
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()
message("CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")

find_library(CUTENSOR_LIB NAMES cutensor PATHS "$ENV{CUTENSOR_ROOT}/lib/12" $ENV{LD_LIBRARY_PATH} $ENV{PATH})
find_file(CUTENSOR_HEADER "cutensor.h" PATHS "$ENV{CUTENSOR_ROOT}/include" $ENV{C_INCLUDE_PATH} $ENV{CPLUS_INCLUDE_PATH} $ENV{PATH})

if(NOT CUTENSOR_LIB OR NOT CUTENSOR_HEADER)
  message("Try to find cuTENSOR in conda...")
  execute_process(
    COMMAND bash -c "conda info --base"
    OUTPUT_VARIABLE CONDA_CONDA_BASE)
  string(REGEX REPLACE "\n$" "" CONDA_CONDA_BASE "${CONDA_CONDA_BASE}")
  execute_process(
    COMMAND bash -c "conda list --json | python -c \"
import sys, json
from packaging.version import parse
path = ''
version = parse('0.0.0')
pkg_list = json.load(sys.stdin)
for pkg in pkg_list:
    if 'cutensor' in pkg['name'] and parse(pkg['version']) > version:
        path = pkg['dist_name']
        version = pkg['version']
print(path)
    \""
    OUTPUT_VARIABLE CONDA_CUTENSOR_NAME)
  string(REGEX REPLACE "\n$" "" CONDA_CUTENSOR_NAME "${CONDA_CUTENSOR_NAME}")
  set(CUTENSOR_CONDA_ROOT "${CONDA_CONDA_BASE}/pkgs/${CONDA_CUTENSOR_NAME}")
  find_library(CUTENSOR_LIB NAMES cutensor PATHS "${CUTENSOR_CONDA_ROOT}/lib")
  find_file(CUTENSOR_HEADER "cutensor.h" PATHS "${CUTENSOR_CONDA_ROOT}/include")
endif()

if(NOT CUTENSOR_LIB OR NOT CUTENSOR_HEADER)
  if(NOT CUTENSOR_LIB)
  message("Cannot find cuTENSOR library! Try following ways:\n\t1. Install cutensor via conda.\n2. Make sure one of following path has libcutensor.so: \$CUTENSOR_ROOT/lib/12, \$LD_LIBRARY_PATH, \$PATH.\n")
  endif()
  if(NOT CUTENSOR_HEADER)
  message("Cannot find cuTENSOR header! Try following ways:\n\t1. Install cutensor via conda.\n2. Make sure one of following path has libcutensor.so: \$CUTENSOR_ROOT/include, \$C_INCLUDE_PATH, \$CPLUS_INCLUDE_PATH, \$PATH.\n")
  endif()
  message(FATAL_ERROR "Aborting")
else()
  message(STATUS "Found cuTENSOR lib: ${CUTENSOR_LIB}")
  message(STATUS "Found cuTENSOR header: ${CUTENSOR_HEADER}")
endif()
string(REGEX REPLACE "/cutensor.h$" "" CUTENSOR_INCLUDE ${CUTENSOR_HEADER})

include_directories(${CUTENSOR_INCLUDE})

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RELWITHDEBINFO)
endif()

set(CMAKE_VERBOSE_MAKEFILE OFF)
if (CMAKE_COMPILER_IS_GNUCC) # Does it skip the link flag on old OsX?
  # TODO: If updating to minimum requirement cmake>=3.7, use
  # CMAKE_SHARED_LINKER_FLAGS_INIT to combine LDFLAGS options.
  # https://cmake.org/cmake/help/v3.7/variable/CMAKE_SHARED_LINKER_FLAGS_INIT.html
  # see also issue #661
  if(UNIX AND NOT APPLE AND NOT DEFINED ENV{LDFLAGS})
    set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-as-needed")
  endif()
endif()
set(CMAKE_C_FLAGS "-Wall ${CMAKE_C_FLAGS}")

option(BUILD_MARCH_NATIVE "gcc flag -march=native" off)
if (BUILD_MARCH_NATIVE)
  include(CheckCCompilerFlag)
  CHECK_C_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
  if(COMPILER_SUPPORTS_MARCH_NATIVE)
    if ("${CMAKE_C_COMPILER_ID}" STREQUAL "Intel")
      message("Add CFLAGS -march=native -unroll-aggressive")
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -unroll-aggressive -ipo")
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fno-math-errno")
    else()
      message("Add CFLAGS -march=native -ftree-vectorize")
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native -funroll-loops -ftree-vectorize")
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fno-math-errno")
    endif()
  endif()
else()
  if ("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
    include(CheckCCompilerFlag)
    CHECK_C_COMPILER_FLAG("-msse3" COMPILER_SUPPORTS_SSE3)
    if(COMPILER_SUPPORTS_SSE3)
      # Avoids error "‘SIMDD’ undeclared here (not in a function)" in the qcint two-electron integral interface
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -msse3")
    endif()
  endif()
endif()

include_directories(${PROJECT_SOURCE_DIR})

set(C_LINK_TEMPLATE "<CMAKE_C_COMPILER> <CMAKE_SHARED_LIBRARY_C_FLAGS> <LANGUAGE_COMPILE_FLAGS> <LINK_FLAGS> <CMAKE_SHARED_LIBRARY_CREATE_C_FLAGS> -o <TARGET> <OBJECTS> <LINK_LIBRARIES>")
set(CXX_LINK_TEMPLATE "<CMAKE_CXX_COMPILER> <CMAKE_SHARED_LIBRARY_CXX_FLAGS> <LANGUAGE_COMPILE_FLAGS> <LINK_FLAGS> <CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS> -o <TARGET> <OBJECTS> <LINK_LIBRARIES>")

add_subdirectory(gpuaft)
add_subdirectory(pbc_gpu)
add_subdirectory(gint)

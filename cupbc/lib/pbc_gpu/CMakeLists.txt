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

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++17")

add_library(gpbc SHARED fill_ints_sr.c fill_ints_sr.cu util.cu fill_int2e.cu get_Rcuts.cu)
# add_dependencies(gpbc cgto cvhf)

set_target_properties(gpbc PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
  CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(gpbc -lcublas -std=c++11 ${BLAS_LIBRARIES} ${OPENMP_C_PROPERTIES} ${CUTENSOR_LIB})
